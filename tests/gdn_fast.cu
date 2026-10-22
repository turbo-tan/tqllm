// fn_gdn at the Flash-Next decode shape: writes out + next state to argv[1] and prints the time per call.
// Run once without and once with TQFN_GDN_FAST=1, then compare the two files byte for byte.
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test-gdn-fast out.bin\n"); return 2; }
    const int kh = 16, vh = 48, dim = 128, channels = (2 * kh + vh) * dim; const float eps = 1e-6f;
    std::FILE * f = std::fopen(argv[1], "wb");
    for (int n = 1; n <= 4; ++n) {
        std::vector<float> x(size_t(n) * channels), a(n * vh), b(n * vh), neg(vh), bias(vh), st(size_t(vh) * dim * dim);
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.7f * std::sin(float(i) * .0137f + n);
        for (size_t i = 0; i < a.size(); ++i) { a[i] = .9f * std::cos(float(i) * 1.3f); b[i] = .8f * std::sin(float(i) * .7f); }
        for (int i = 0; i < vh; ++i) { neg[i] = -.2f - .017f * i; bias[i] = .1f * std::cos(float(i)); }
        for (size_t i = 0; i < st.size(); ++i) st[i] = .05f * std::sin(float(i) * .021f);
        float *dx, *da, *db, *dn, *dw, *ds, *dnext, *dy;
        cudaMalloc(&dx, x.size() * 4); cudaMalloc(&da, a.size() * 4); cudaMalloc(&db, b.size() * 4); cudaMalloc(&dn, vh * 4);
        cudaMalloc(&dw, vh * 4); cudaMalloc(&ds, st.size() * 4); cudaMalloc(&dnext, st.size() * 4); cudaMalloc(&dy, size_t(n) * vh * dim * 4);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dn, neg.data(), vh * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, bias.data(), vh * 4, cudaMemcpyHostToDevice); cudaMemcpy(ds, st.data(), st.size() * 4, cudaMemcpyHostToDevice);
        tqllm_cuda::fn_gdn(dx, da, db, dn, dw, ds, dnext, dy, kh, vh, dim, n, eps, nullptr); cudaDeviceSynchronize();
        std::vector<float> y(size_t(n) * vh * dim), nx(st.size());
        cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(nx.data(), dnext, nx.size() * 4, cudaMemcpyDeviceToHost);
        std::fwrite(y.data(), 4, y.size(), f); std::fwrite(nx.data(), 4, nx.size(), f);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int w = 0; w < 50; ++w) tqllm_cuda::fn_gdn(dx, da, db, dn, dw, ds, dnext, dy, kh, vh, dim, n, eps, nullptr);
        cudaEventRecord(e0); const int reps = 2000;
        for (int r = 0; r < reps; ++r) tqllm_cuda::fn_gdn(dx, da, db, dn, dw, ds, dnext, dy, kh, vh, dim, n, eps, nullptr);
        cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        std::printf("GDN n=%d %.2f us per call\n", n, 1000.f * ms / reps);
        cudaFree(dx); cudaFree(da); cudaFree(db); cudaFree(dn); cudaFree(dw); cudaFree(ds); cudaFree(dnext); cudaFree(dy);
    }
    std::fclose(f); return 0;
}
