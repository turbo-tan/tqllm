// Chunked gated-delta prefill vs the per-token kernel at the Flash-Next shape: max relative error of the outputs and the
// final state, and the time per call of both. Exit 1 when the error exceeds 1e-3.
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
static float rel(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num = std::max(num, double(std::fabs(a[i] - b[i]))); den = std::max(den, double(std::fabs(a[i]))); }
    return float(num / std::max(den, 1e-12));
}
int main() {
    const int kh = 16, vh = 48, dim = 128, channels = (2 * kh + vh) * dim; const float eps = 1e-6f;
    bool ok = true;
    for (int n : {64, 200, 1024}) {
        std::vector<float> x(size_t(n) * channels), a(size_t(n) * vh), b(size_t(n) * vh), neg(vh), bias(vh), st(size_t(vh) * dim * dim);
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.7f * std::sin(float(i) * .0137f + n) + 0.2f * std::cos(float(i) * .00071f);
        for (size_t i = 0; i < a.size(); ++i) { a[i] = .9f * std::cos(float(i) * 1.3f); b[i] = .8f * std::sin(float(i) * .7f); }
        for (int i = 0; i < vh; ++i) { neg[i] = -.2f - .017f * i; bias[i] = .1f * std::cos(float(i)); }
        for (size_t i = 0; i < st.size(); ++i) st[i] = .05f * std::sin(float(i) * .021f);
        float *dx, *da, *db, *dn, *dw, *ds, *dnext, *dy;
        // a device another process filled (tune_oom before it ran serially) fails here with a message, not inside fn_gdn
        const bool got = cudaMalloc(&dx, x.size() * 4) == cudaSuccess && cudaMalloc(&da, a.size() * 4) == cudaSuccess &&
            cudaMalloc(&db, b.size() * 4) == cudaSuccess && cudaMalloc(&dn, vh * 4) == cudaSuccess && cudaMalloc(&dw, vh * 4) == cudaSuccess &&
            cudaMalloc(&ds, st.size() * 4) == cudaSuccess && cudaMalloc(&dnext, st.size() * 4) == cudaSuccess &&
            cudaMalloc(&dy, size_t(n) * vh * dim * 4) == cudaSuccess;
        if (!got) { std::printf("GDN_CHUNK FAIL: device allocation failed (another process holds the memory?)\n"); return 1; }
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dn, neg.data(), vh * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, bias.data(), vh * 4, cudaMemcpyHostToDevice); cudaMemcpy(ds, st.data(), st.size() * 4, cudaMemcpyHostToDevice);
        std::vector<float> y[2], nx[2]; float us[2];
        for (int mode = 0; mode < 2; ++mode) {
            tqllm_cuda::fn_gdn_set_chunk_min(mode ? 64 : 0);
            tqllm_cuda::fn_gdn(dx, da, db, dn, dw, ds, dnext, dy, kh, vh, dim, n, eps, nullptr);
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { std::printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
            y[mode].resize(size_t(n) * vh * dim); nx[mode].resize(st.size());
            cudaMemcpy(y[mode].data(), dy, y[mode].size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(nx[mode].data(), dnext, nx[mode].size() * 4, cudaMemcpyDeviceToHost);
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            const int reps = 20; cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) tqllm_cuda::fn_gdn(dx, da, db, dn, dw, ds, dnext, dy, kh, vh, dim, n, eps, nullptr);
            cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0; cudaEventElapsedTime(&ms, e0, e1); us[mode] = 1000.f * ms / reps;
        }
        const float ey = rel(y[0], y[1]), es = rel(nx[0], nx[1]);
        ok = ok && ey < 1e-3f && es < 1e-3f;
        std::printf("GDN n=%d per-token %.1f us, chunked %.1f us (%.1fx); max rel err out %.2e, state %.2e\n", n, us[0], us[1], us[0] / us[1], ey, es);
        cudaFree(dx); cudaFree(da); cudaFree(db); cudaFree(dn); cudaFree(dw); cudaFree(ds); cudaFree(dnext); cudaFree(dy);
    }
    std::printf(ok ? "GDN_CHUNK PASS\n" : "GDN_CHUNK FAIL\n");
    return ok ? 0 : 1;
}
