// Turbo KV formats with a wide GQA group (MiniCPM-V-4.7 / Qwen3.5-35B-A3B: 16 query heads per 2 KV heads, group 8):
// an n-token step (verify or prefill chunk) must work and match n 1-token steps (the CTV kernel takes at most 48 query rows per launch).
#include "src/formats/kv_format.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace tqllm;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s\n", cudaGetErrorString(e_)); return 1; } } while (0)

static int run(const char * name, int heads, int kvh, int n) {
    const int hd = 256, nver = 8, ctx = 64, split = 4, ts = 2 * kvh * hd;
    const KvFormat * f = kv_format(name);
    std::mt19937 gen(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> kv(size_t(n) * ts), q(size_t(n) * heads * hd);
    for (auto & v : kv) v = __float2bfloat16(nd(gen));
    for (auto & v : q) v = __float2bfloat16(nd(gen));
    __nv_bfloat16 * dkv, * dq, * o1, * o8; void * kc8, * vc8, * kc1, * vc1; float * part; int * dpos;
    const size_t kb = f->key_bytes(ctx, kvh, hd), vb = f->value_bytes(ctx, kvh, hd);
    CK(cudaMalloc(&dkv, kv.size() * 2)); CK(cudaMalloc(&dq, q.size() * 2));
    CK(cudaMalloc(&o1, q.size() * 2)); CK(cudaMalloc(&o8, q.size() * 2));
    CK(cudaMalloc(&kc8, kb)); CK(cudaMalloc(&vc8, vb)); CK(cudaMalloc(&kc1, kb)); CK(cudaMalloc(&vc1, vb));
    CK(cudaMemset(kc8, 0, kb)); CK(cudaMemset(vc8, 0, vb)); CK(cudaMemset(kc1, 0, kb)); CK(cudaMemset(vc1, 0, vb));
    CK(cudaMalloc(&part, size_t(n) * heads * split * (hd + 2) * 4 * 4));
    CK(cudaMalloc(&dpos, 2 * (n + 1) * sizeof(int)));
    std::vector<int> pos; for (int i = 0; i <= n; ++i) { pos.push_back(i); pos.push_back(i + 1); }
    CK(cudaMemcpy(dpos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkv, kv.data(), kv.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    auto step = [&](int first, int count, void * kc, void * vc, __nv_bfloat16 * o) {
        KvStep a{};
        a.k0 = dkv + size_t(first) * ts; a.ts = ts; a.n = count; a.kc = kc; a.vc = vc; a.dpos = dpos + 2 * first;
        a.heads = heads; a.kvheads = kvh; a.head_dim = hd; a.nver = nver; a.split = split; a.context = ctx;
        a.q = dq + size_t(first) * heads * hd; a.o = o + size_t(first) * heads * hd; a.part = part;
        a.scale = 1.f / 16.f; a.stream = nullptr;
        f->write_and_attend(a);
    };
    try {
        step(0, n, kc8, vc8, o8);
        for (int i = 0; i < n; ++i) step(i, 1, kc1, vc1, o1);
    } catch (const std::exception & e) {
        std::printf("FAIL %s heads %d / kv %d n %d: %s\n", name, heads, kvh, n, e.what());
        return 1;
    }
    CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> y8(q.size()), y1(q.size());
    CK(cudaMemcpy(y8.data(), o8, y8.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y1.data(), o1, y1.size() * 2, cudaMemcpyDeviceToHost));
    double md = 0, mx = 0;
    for (size_t i = 0; i < y8.size(); ++i) {
        const double a = __bfloat162float(y8[i]), b = __bfloat162float(y1[i]);
        md = std::fmax(md, std::fabs(a - b)); mx = std::fmax(mx, std::fabs(b));
    }
    for (void * p : {(void *)dkv, (void *)dq, (void *)o1, (void *)o8, kc8, vc8, kc1, vc1, (void *)part, (void *)dpos}) cudaFree(p);
    const bool ok = std::isfinite(md) && mx > 0 && md <= 2e-2 * mx;
    std::printf("%s %s heads %d / kv %d: %d-token step vs %d single steps max diff %.3g (max |y| %.3g)\n", ok ? "PASS" : "FAIL",
                name, heads, kvh, n, n, md, mx);
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    for (const char * name : {"q8k-turbo3v", "t4k-turbo3v", "t4kv"})
        for (auto [h, kv] : {std::pair{24, 4}, std::pair{16, 2}, std::pair{32, 2}})
            for (int n : {8, 20}) fails += run(name, h, kv, n);  // verify width, prefill chunk
    return fails ? 1 : 0;
}
