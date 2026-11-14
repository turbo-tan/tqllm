// fn_gdn_multi (several decode blocks in one launch) must give the same bits as fn_gdn per sequence: outputs and each
// sequence's next state, mixed block sizes (1..8 rows), 1..5 sequences, at the Flash-Next shape.
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    const int kh = 16, vh = 48, dim = 128, channels = (2 * kh + vh) * dim; const float eps = 1e-6f;
    const size_t ns = size_t(vh) * dim * dim; const int R = 40, Q = 5;
    std::vector<float> neg(vh), bias(vh), st(ns * Q), x(size_t(R) * channels), a(size_t(R) * vh), b(a.size());
    for (int i = 0; i < vh; ++i) { neg[i] = -.2f - .017f * i; bias[i] = .1f * std::cos(float(i)); }
    for (size_t i = 0; i < st.size(); ++i) st[i] = .05f * std::sin(float(i) * .021f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.7f * std::sin(float(i) * .0137f) + 0.2f * std::cos(float(i) * .00071f);
    for (size_t i = 0; i < a.size(); ++i) { a[i] = .9f * std::cos(float(i) * 1.3f); b[i] = .8f * std::sin(float(i) * .7f); }
    auto up = [](const std::vector<float> & v) { float * p; cudaMalloc(&p, v.size() * 4); cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice); return p; };
    float * dn = up(neg), * dw = up(bias), * ds = up(st), * dx = up(x), * da = up(a), * db = up(b), * dnext, * dy, * dnext2, * dy2;
    cudaMalloc(&dnext, ns * Q * 4); cudaMalloc(&dy, size_t(R) * vh * dim * 4); cudaMalloc(&dnext2, ns * Q * 4); cudaMalloc(&dy2, size_t(R) * vh * dim * 4);
    const std::vector<std::vector<int>> cases = {{1, 1}, {4, 4, 4, 4}, {3, 1, 4}, {8, 2}, {1, 4, 2, 3, 4}, {5, 8, 6, 7}};
    bool ok = true;
    for (const auto & ns_ : cases) {
        tqllm_cuda::GdnSeqs seqs; int off = 0;
        for (size_t q = 0; q < ns_.size(); ++q) {
            const int n = ns_[q];
            tqllm_cuda::fn_gdn(dx + size_t(off) * channels, da + size_t(off) * vh, db + size_t(off) * vh, dn, dw, ds + q * ns, dnext + q * ns,
                               dy + size_t(off) * vh * dim, kh, vh, dim, n, eps, nullptr);
            seqs.off[q] = off; seqs.n[q] = n; seqs.state[q] = ds + q * ns; seqs.next[q] = dnext2 + q * ns; ++seqs.count; off += n;
        }
        tqllm_cuda::fn_gdn_multi(dx, da, db, dn, dw, seqs, dy2, kh, vh, dim, eps, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> y1(size_t(off) * vh * dim), y2(y1.size()), s1(ns * ns_.size()), s2(s1.size());
        cudaMemcpy(y1.data(), dy, y1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(y2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(s1.data(), dnext, s1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(s2.data(), dnext2, s2.size() * 4, cudaMemcpyDeviceToHost);
        if (std::memcmp(y1.data(), y2.data(), y1.size() * 4) || std::memcmp(s1.data(), s2.data(), s1.size() * 4)) {
            std::printf("FAIL: %zu sequences, %d rows: one launch differs from per-sequence fn_gdn\n", ns_.size(), off); ok = false;
        }
    }
    if (cudaGetLastError() != cudaSuccess) { std::puts("FAIL: CUDA error"); ok = false; }
    std::puts(ok ? "gdn multi: PASS (one launch for several sequences bit-identical to per-sequence fn_gdn)" : "gdn multi: FAIL");
    return ok ? 0 : 1;
}
