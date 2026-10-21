// GPU test: KV rows split between device memory and a host-memory tail (KvPool with device_rows) give byte-identical
// cache contents and attention outputs to an all-device cache, through the real KV-format kernels (prefill chunks and
// verify blocks that cross the device/host boundary).
#include "kv_format.hpp"
#include "kv_store.hpp"
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace tqllm;
using namespace tqllm_cuda;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

int main() {
    const int heads = 24, kvh = 4, hd = 256, nver = 12, nbuf = 512;
    const int64_t ctx = 65536, dev_rows = 30000;  // the device head rounds up to a 2 MiB granule (row 32768 for 448 B rows)
    cudaStream_t st;
    st = cudaStreamPerThread;
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    auto mem = make_cuda_kv_memory(st);
    if (!mem->granule()) {
        std::printf("kv_host: SKIP (no virtual memory management on this device)\n");
        return 0;
    }
    const size_t part_floats = std::max({size_t(nver) * attention_partial_floats(heads, int(ctx), hd),
                                         size_t(attention_mq_partial_floats(heads, nver, 256)),
                                         size_t(attention_mq_partial_floats(heads, nbuf, 1))});
    float * part;
    __nv_bfloat16 *k0, *q, *o1, *o2;
    int * dpos;
    const int ts = 2 * kvh * hd;
    CK(cudaMalloc(&part, part_floats * 4));
    CK(cudaMalloc(&k0, size_t(nbuf) * ts * 2));
    CK(cudaMalloc(&q, size_t(nbuf) * heads * hd * 2));
    CK(cudaMalloc(&o1, size_t(nbuf) * heads * hd * 2));
    CK(cudaMalloc(&o2, size_t(nbuf) * heads * hd * 2));
    CK(cudaMalloc(&dpos, 8));
    std::mt19937 g(5);
    std::normal_distribution<float> nd;
    std::vector<__nv_bfloat16> hk(size_t(nbuf) * ts), hq(size_t(nbuf) * heads * hd), ha(hq.size()), hb(hq.size());

    for (const char * name : {"tq3", "tq3k-tq2v", "bf16"}) {
        const KvFormat * f = kv_format(name);
        const size_t krow = f->key_bytes(1, kvh, hd), vrow = f->value_bytes(1, kvh, hd);
        void *fk, *fv;
        CK(cudaMalloc(&fk, krow * ctx));
        CK(cudaMalloc(&fv, vrow * ctx));
        KvGeometry geo{1, krow, vrow, ctx, dev_rows};
        KvPool pool(*mem, geo, 1, 0);
        auto s = kv_store_policy("flat")->make(pool, ctx);
        CHECK(s->reserve(ctx), "%s: split store reserves the context", name);
        void * sk = s->base(0, true), * sv = s->base(0, false);
        const int64_t boundary = int64_t(pool.bytes(KvTier::device) / 2 / krow);  // approx: first host row (K side)
        CHECK(pool.bytes(KvTier::host) > 0, "%s: part of the KV is in host memory", name);
        // fill to just before the boundary, then cross it with a prefill chunk and with verify blocks
        const int64_t start = std::max<int64_t>(0, boundary - 3 * nbuf);
        std::vector<std::pair<int64_t, int>> steps;
        for (int64_t p = 0; p < start; p += nbuf) steps.push_back({p, int(std::min<int64_t>(nbuf, start - p))});
        for (int64_t p = start; p < boundary + 2 * nbuf; p += nbuf) steps.push_back({p, nbuf});
        for (int64_t p = boundary + 2 * nbuf; p < boundary + 2 * nbuf + 5 * nver; p += nver) steps.push_back({p, nver});
        int bad_o = 0, checked = 0;
        for (const auto & [p, n] : steps) {
            for (auto & v : hk) v = __float2bfloat16(nd(g));
            for (auto & v : hq) v = __float2bfloat16(nd(g));
            CK(cudaMemcpyAsync(k0, hk.data(), size_t(n) * ts * 2, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(q, hq.data(), size_t(n) * heads * hd * 2, cudaMemcpyHostToDevice, st));
            const int hp[2] = {int(p), int(p + 1)};
            CK(cudaMemcpyAsync(dpos, hp, 8, cudaMemcpyHostToDevice, st));
            const int split = std::clamp(f->split_factor() * sms / kvh, 1, 256);
            const bool check_o = p + n > start;
            f->write_and_attend({k0, ts, n, fk, fv, dpos, heads, kvh, hd, nver, split, ctx, q, o1, part, 1.f / 16.f, st});
            f->write_and_attend({k0, ts, n, sk, sv, dpos, heads, kvh, hd, nver, split, ctx, q, o2, part, 1.f / 16.f, st});
            if (!check_o) continue;
            CK(cudaMemcpyAsync(ha.data(), o1, size_t(n) * heads * hd * 2, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(hb.data(), o2, size_t(n) * heads * hd * 2, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            bad_o += std::memcmp(ha.data(), hb.data(), size_t(n) * heads * hd * 2) != 0;
            ++checked;
        }
        CK(cudaStreamSynchronize(st));
        const int64_t used = steps.back().first + steps.back().second;
        std::vector<char> a(krow * used), b(krow * used);
        CK(cudaMemcpy(a.data(), fk, a.size(), cudaMemcpyDefault));
        CK(cudaMemcpy(b.data(), sk, b.size(), cudaMemcpyDefault));
        const bool keys_same = a == b;
        a.resize(vrow * used), b.resize(vrow * used);
        CK(cudaMemcpy(a.data(), fv, a.size(), cudaMemcpyDefault));
        CK(cudaMemcpy(b.data(), sv, b.size(), cudaMemcpyDefault));
        CHECK(keys_same && a == b, "%s: cache bytes differ between all-device and split", name);
        CHECK(bad_o == 0, "%s: %d of %d attention outputs differ", name, bad_o, checked);
        std::printf("kv_host %-10s positions %lld, boundary ~%lld, %d outputs compared, host %.1f MB\n", name, (long long)used,
                    (long long)boundary, checked, double(pool.bytes(KvTier::host)) / 1e6);
        s.reset();
        cudaFree(fk);
        cudaFree(fv);
    }
    // staged: old positions evicted to host and staged back before every call (the engine's prefetch), real kernels
    for (const char * name : {"tq3", "tq3k-tq2v"}) {
        const KvFormat * f = kv_format(name);
        const size_t krow = f->key_bytes(1, kvh, hd), vrow = f->value_bytes(1, kvh, hd);
        void *fk, *fv;
        CK(cudaMalloc(&fk, krow * ctx));
        CK(cudaMalloc(&fv, vrow * ctx));
        KvGeometry geo{1, krow, vrow, ctx, 12000, true};
        KvPool pool(*mem, geo, 1, 0);
        auto s = kv_store_policy("flat")->make(pool, ctx);
        s->reserve(ctx);
        void * sk = s->base(0, true), * sv = s->base(0, false);
        int64_t * nb;
        CK(cudaMalloc(&nb, 16));
        std::vector<std::pair<int64_t, int>> steps;
        for (int64_t p = 0; p < 40000; p += nbuf) steps.push_back({p, nbuf});
        for (int64_t p = 40000; p < 40000 + 6 * nver; p += nver) steps.push_back({p, nver});
        steps.push_back({20000, nbuf});  // a rewind below the evicted boundary (request carryover)
        int bad_o = 0, checked = 0;
        for (const auto & [p, n] : steps) {
            CHECK(pool.stage_ensure(p, p + n), "%s: window holds the block at %lld", name, (long long)p);
            const int64_t b[2] = {int64_t(pool.staged_bytes(true)), int64_t(pool.staged_bytes(false))};
            CK(cudaMemcpyAsync(nb, b, 16, cudaMemcpyHostToDevice, st));
            kv_stage_copy(pool.host_copy(0, true), sk, nb, 4 * sms, st);
            kv_stage_copy(pool.host_copy(0, false), sv, nb + 1, 4 * sms, st);
            for (auto & v : hk) v = __float2bfloat16(nd(g));
            for (auto & v : hq) v = __float2bfloat16(nd(g));
            CK(cudaMemcpyAsync(k0, hk.data(), size_t(n) * ts * 2, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(q, hq.data(), size_t(n) * heads * hd * 2, cudaMemcpyHostToDevice, st));
            const int hp[2] = {int(p), int(p + 1)};
            CK(cudaMemcpyAsync(dpos, hp, 8, cudaMemcpyHostToDevice, st));
            const int split = std::clamp(f->split_factor() * sms / kvh, 1, 256);
            f->write_and_attend({k0, ts, n, fk, fv, dpos, heads, kvh, hd, nver, split, ctx, q, o1, part, 1.f / 16.f, st});
            f->write_and_attend({k0, ts, n, sk, sv, dpos, heads, kvh, hd, nver, split, ctx, q, o2, part, 1.f / 16.f, st});
            if (pool.staged_bytes(true) == 0) continue;
            CK(cudaMemcpyAsync(ha.data(), o1, size_t(n) * heads * hd * 2, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(hb.data(), o2, size_t(n) * heads * hd * 2, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            bad_o += std::memcmp(ha.data(), hb.data(), size_t(n) * heads * hd * 2) != 0;
            ++checked;
        }
        CHECK(bad_o == 0 && checked > 10, "%s staged: %d of %d attention outputs differ", name, bad_o, checked);
        std::printf("kv_host staged %-10s %lld evictions, %.1f MB staged per layer at the end, %d outputs compared\n", name,
                    (long long)pool.evictions(), double(pool.staged_bytes(true) + pool.staged_bytes(false)) / 1e6, checked);
        s.reset();
        cudaFree(nb), cudaFree(fk), cudaFree(fv);
    }
    {  // teardown order of an engine: its stream is destroyed before the pool unmaps (regression: WSL lost the context)
        cudaStream_t s2;
        CK(cudaStreamCreate(&s2));
        auto m2 = make_cuda_kv_memory(s2);
        KvGeometry geo{2, 448, 448, ctx, dev_rows};
        auto pool = std::make_unique<KvPool>(*m2, geo, 1, 0);
        auto st2 = kv_store_policy("flat")->make(*pool, ctx);
        st2->reserve(ctx);
        CK(cudaMemsetAsync(st2->base(0, true), 1, size_t(ctx) * 448, s2));
        CK(cudaStreamSynchronize(s2));
        CK(cudaStreamDestroy(s2));
        st2.reset();
        pool.reset();
        void * probe = nullptr;
        const cudaError_t e = cudaMalloc(&probe, 1 << 20);
        CHECK(e == cudaSuccess && cudaMemset(probe, 0, 1 << 20) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
              "CUDA still usable after a pool is released past its stream (%s)", cudaGetErrorString(e));
        cudaFree(probe);
    }
    if (fails) std::printf("kv_host: %d failure(s)\n", fails);
    else std::printf("kv_host: ok\n");
    return fails ? 1 : 0;
}
