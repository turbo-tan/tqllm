// Bench only: the per-token GDN recurrence at decode sizes (Flash-Next shape, 1-8 tokens): us per call, GB/s of state
// traffic (read + write), and an FNV hash of the outputs and next state (equal hashes across builds = identical bits).
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
int main() {
    const int kh = 16, vh = 48, dim = 128, channels = (2 * kh + vh) * dim; const float eps = 1e-6f;
    const size_t ns = size_t(vh) * dim * dim;
    std::vector<float> neg(vh), bias(vh), st(ns);
    for (int i = 0; i < vh; ++i) { neg[i] = -.2f - .017f * i; bias[i] = .1f * std::cos(float(i)); }
    for (size_t i = 0; i < ns; ++i) st[i] = .05f * std::sin(float(i) * .021f);
    constexpr int R = 16;  // rotate over 16 states (100 MB): DRAM, as decode's 36 layers x sequences
    float * dn, * dw, * ds, * dnext, * dx, * da, * db, * dy;
    cudaMalloc(&dn, vh * 4); cudaMalloc(&dw, vh * 4); cudaMalloc(&ds, ns * 4 * R); cudaMalloc(&dnext, ns * 4 * R);
    cudaMemcpy(dn, neg.data(), vh * 4, cudaMemcpyHostToDevice); cudaMemcpy(dw, bias.data(), vh * 4, cudaMemcpyHostToDevice);
    for (int r = 0; r < R; ++r) cudaMemcpy(ds + r * ns, st.data(), ns * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dx, size_t(8) * channels * 4); cudaMalloc(&da, 8 * vh * 4); cudaMalloc(&db, 8 * vh * 4); cudaMalloc(&dy, size_t(8) * vh * dim * 4);
    std::vector<float> x(size_t(8) * channels), a(8 * vh), b(8 * vh);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.7f * std::sin(float(i) * .0137f);
    for (size_t i = 0; i < a.size(); ++i) { a[i] = .9f * std::cos(float(i) * 1.3f); b[i] = .8f * std::sin(float(i) * .7f); }
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int n : {1, 4, 8}) {
        for (int i = 0; i < 20; ++i) tqllm_cuda::fn_gdn(dx, da, db, dn, dw, ds + (i % R) * ns, dnext + (i % R) * ns, dy, kh, vh, dim, n, eps, nullptr);
        cudaEventRecord(e0);
        const int iters = 320;
        for (int i = 0; i < iters; ++i) tqllm_cuda::fn_gdn(dx, da, db, dn, dw, ds + (i % R) * ns, dnext + (i % R) * ns, dy, kh, vh, dim, n, eps, nullptr);
        cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        std::vector<float> y(size_t(n) * vh * dim), nx(ns);
        cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(nx.data(), dnext, ns * 4, cudaMemcpyDeviceToHost);
        uint64_t h = 1469598103934665603ull;
        for (float v : y) { uint32_t u; std::memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
        for (float v : nx) { uint32_t u; std::memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
        const double us = 1000.0 * ms / iters;
        std::printf("n %d: %7.1f us  %6.0f GB/s state  hash %016llx\n", n, us, 2.0 * ns * 4 / (us * 1e3), (unsigned long long) h);
    }
}
