// Host-tail cost model: bulk copy out of a KvPool host chunk (memcpy and a coalesced kernel) vs attention reading the
// host tail in place, tq3 KV, decode (1 query), verify (12) and a 512-token prefill chunk. Decides whether staging the host
// tail into device memory before each attention layer can pay.
// usage: test-kv-host-bench [context 131072] [host positions 65536] [iters 10]
#include "kv_format.hpp"
#include "kv_store.hpp"
#include "qwen2_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace tqllm;
using namespace tqllm_cuda;

__global__ void copy_k(const uint4 * src, uint4 * dst, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) dst[i] = src[i];
}

int main(int argc, char ** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 131072, host = argc > 2 ? std::atoll(argv[2]) : 65536;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 10;
    const int heads = 24, kvh = 4, hd = 256, nver = 12, nbuf = 512;
    cudaStream_t st;
    st = cudaStreamPerThread;
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const KvFormat * f = kv_format("tq3");
    const size_t krow = f->key_bytes(1, kvh, hd), vrow = f->value_bytes(1, kvh, hd);
    auto mem = make_cuda_kv_memory(st);
    KvGeometry gs{1, krow, vrow, ctx, ctx - host}, gd{1, krow, vrow, ctx, -1};
    KvPool split(*mem, gs, 1, 0), dev(*mem, gd, 1, 0);
    auto ss = kv_store_policy("flat")->make(split, ctx), sd = kv_store_policy("flat")->make(dev, ctx);
    ss->reserve(ctx), sd->reserve(ctx);
    char * sk = static_cast<char *>(ss->base(0, true)), * sv = static_cast<char *>(ss->base(0, false));
    char * dk = static_cast<char *>(sd->base(0, true)), * dv = static_cast<char *>(sd->base(0, false));
    cudaMemsetAsync(sk, 0x11, krow * ctx, st), cudaMemsetAsync(sv, 0x22, vrow * ctx, st);
    cudaMemsetAsync(dk, 0x11, krow * ctx, st), cudaMemsetAsync(dv, 0x22, vrow * ctx, st);
    const size_t host_b = split.bytes(KvTier::host);  // K + V tail bytes
    const int64_t first_host = ctx - host;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    auto timed = [&](auto && fn) {
        fn();
        cudaEventRecord(e0, st);
        for (int i = 0; i < iters; ++i) fn();
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / iters;
    };
    // 1. bulk copies of the host tail (K and V) into device memory
    const size_t tk = krow * host, tv = vrow * host;
    char * stage;
    cudaMalloc(&stage, tk + tv);
    const float m_cpy = timed([&] {
        cudaMemcpyAsync(stage, sk + krow * first_host, tk, cudaMemcpyDefault, st);
        cudaMemcpyAsync(stage + tk, sv + vrow * first_host, tv, cudaMemcpyDefault, st);
    });
    const float m_ker = timed([&] {
        copy_k<<<sms * 8, 256, 0, st>>>(reinterpret_cast<const uint4 *>(sk + krow * first_host), reinterpret_cast<uint4 *>(stage), tk / 16);
        copy_k<<<sms * 8, 256, 0, st>>>(reinterpret_cast<const uint4 *>(sv + vrow * first_host), reinterpret_cast<uint4 *>(stage + tk), tv / 16);
    });
    std::printf("host tail %lld positions, %.1f MB (K+V, one layer): memcpy %.3f ms = %.1f GB/s | copy kernel %.3f ms = %.1f GB/s\n",
                (long long)host, double(tk + tv) / 1e6, m_cpy, double(tk + tv) / m_cpy / 1e6, m_ker, double(tk + tv) / m_ker / 1e6);
    // 2. attention over the whole context with the tail in host memory vs all on the device
    const size_t part_floats = std::max({size_t(attention_mq_partial_floats(heads, nver, 256)), size_t(attention_mq_partial_floats(heads, nbuf, 1))});
    float * part;
    __nv_bfloat16 *k0, *q, *o;
    int * dpos;
    const int ts = 2 * kvh * hd;
    cudaMalloc(&part, part_floats * 4), cudaMalloc(&k0, size_t(nbuf) * ts * 2), cudaMalloc(&q, size_t(nbuf) * heads * hd * 2);
    cudaMalloc(&o, size_t(nbuf) * heads * hd * 2), cudaMalloc(&dpos, 8);
    cudaMemsetAsync(k0, 0, size_t(nbuf) * ts * 2, st), cudaMemsetAsync(q, 0, size_t(nbuf) * heads * hd * 2, st);
    const int split_n = std::clamp(f->split_factor() * sms / kvh, 1, 256);
    for (int n : {1, nver, nbuf}) {
        const int p = int(ctx - nbuf - 1);  // attend over ~the whole context; writes land in the last (host) rows
        const int hp[2] = {p, p + 1};
        cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice);
        auto run = [&](char * kc, char * vc) {
            return timed([&] { f->write_and_attend({k0, ts, n, kc, vc, dpos, heads, kvh, hd, nver, split_n, ctx, q, o, part, 1.f / 16.f, st}); });
        };
        const float t_dev = run(dk, dv), t_split = run(sk, sv);
        const double extra = t_split - t_dev;
        std::printf("attention n=%3d: device %.3f ms, host tail %.3f ms -> +%.3f ms = %.1f GB/s effective over the host tail\n", n,
                    t_dev, t_split, extra, extra > 0 ? double(host_b) / extra / 1e6 : 0.0);
    }
    return 0;
}
