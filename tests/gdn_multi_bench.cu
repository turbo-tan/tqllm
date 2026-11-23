// Bench only: fn_gdn_multi at batched decode sizes (Flash-Next shape, S sequences x n rows), state rotating over R sets in
// DRAM: us per call, GB/s of state traffic (read + write; GB10 read+write ceiling ~215), FNV hash of outputs and next states
// (equal hashes across builds and variants = identical bits). usage: gdn-multi-bench [sequences=8] [vpw=0: by device]
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main(int argc, char ** argv) {
    const int S = argc > 1 ? std::atoi(argv[1]) : 8, vpw = argc > 2 ? std::atoi(argv[2]) : 0;
    const int kh = 16, vh = 48, dim = 128, channels = (2 * kh + vh) * dim, nmax = 4; const float eps = 1e-6f;
    const size_t ns = size_t(vh) * dim * dim;
    constexpr int R = 6;  // 6 x S states of 3.1 MB each: DRAM, as decode's 36 layers x sequences
    std::vector<float> neg(vh), bias(vh), st(ns);
    for (int i = 0; i < vh; ++i) { neg[i] = -.2f - .017f * i; bias[i] = .1f * std::cos(float(i)); }
    for (size_t i = 0; i < ns; ++i) st[i] = .05f * std::sin(float(i) * .021f);
    float * dn, * dw, * ds, * dnext, * dx, * da, * db, * dy;
    cudaMalloc(&dn, vh * 4); cudaMalloc(&dw, vh * 4); cudaMalloc(&ds, ns * 4 * R * S); cudaMalloc(&dnext, ns * 4 * R * S);
    cudaMemcpy(dn, neg.data(), vh * 4, cudaMemcpyHostToDevice); cudaMemcpy(dw, bias.data(), vh * 4, cudaMemcpyHostToDevice);
    for (int r = 0; r < R * S; ++r) cudaMemcpy(ds + r * ns, st.data(), ns * 4, cudaMemcpyHostToDevice);
    const int rmax = S * nmax;
    cudaMalloc(&dx, size_t(rmax) * channels * 4); cudaMalloc(&da, rmax * vh * 4); cudaMalloc(&db, rmax * vh * 4); cudaMalloc(&dy, size_t(rmax) * vh * dim * 4);
    std::vector<float> x(size_t(rmax) * channels), a(rmax * vh), b(rmax * vh);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.7f * std::sin(float(i) * .0137f);
    for (size_t i = 0; i < a.size(); ++i) { a[i] = .9f * std::cos(float(i) * 1.3f); b[i] = .8f * std::sin(float(i) * .7f); }
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice);
    if (vpw) tqllm_cuda::fn_gdn_set_vpw(vpw);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int n : {1, 2, 3, 4}) {
        auto seqs = [&](int i) {
            tqllm_cuda::GdnSeqs q; q.count = S;
            for (int j = 0; j < S; ++j) { q.off[j] = j * n; q.n[j] = n; q.state[j] = ds + size_t((i % R) * S + j) * ns; q.next[j] = dnext + size_t((i % R) * S + j) * ns; }
            return q;
        };
        for (int i = 0; i < 12; ++i) tqllm_cuda::fn_gdn_multi(dx, da, db, dn, dw, seqs(i), dy, kh, vh, dim, eps, nullptr);
        const int iters = 120;
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) tqllm_cuda::fn_gdn_multi(dx, da, db, dn, dw, seqs(i), dy, kh, vh, dim, eps, nullptr);
        cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        tqllm_cuda::fn_gdn_multi(dx, da, db, dn, dw, seqs(0), dy, kh, vh, dim, eps, nullptr);
        std::vector<float> y(size_t(S) * n * vh * dim), nx(ns * S);
        cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(nx.data(), dnext, nx.size() * 4, cudaMemcpyDeviceToHost);
        uint64_t h = 1469598103934665603ull;
        for (float v : y) { uint32_t u; std::memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
        for (float v : nx) { uint32_t u; std::memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
        const double us = 1000.0 * ms / iters;
        std::printf("S %d n %d vpw %d: %7.1f us  %6.0f GB/s state  hash %016llx\n", S, n, vpw, us, 2.0 * ns * 4 * S / (us * 1e3), (unsigned long long) h);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
