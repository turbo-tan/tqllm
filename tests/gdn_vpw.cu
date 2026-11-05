// The per-token GDN kernel must give the same bits for 2, 4 and 8 value columns per warp (the device picks one):
// outputs and next state at the Flash-Next shape, 1..8 tokens.
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    const int kh = 16, vh = 48, dim = 128, channels = (2 * kh + vh) * dim; const float eps = 1e-6f;
    const size_t ns = size_t(vh) * dim * dim;
    std::vector<float> neg(vh), bias(vh), st(ns), x(size_t(8) * channels), a(8 * vh), b(8 * vh);
    for (int i = 0; i < vh; ++i) { neg[i] = -.2f - .017f * i; bias[i] = .1f * std::cos(float(i)); }
    for (size_t i = 0; i < ns; ++i) st[i] = .05f * std::sin(float(i) * .021f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.7f * std::sin(float(i) * .0137f) + 0.2f * std::cos(float(i) * .00071f);
    for (size_t i = 0; i < a.size(); ++i) { a[i] = .9f * std::cos(float(i) * 1.3f); b[i] = .8f * std::sin(float(i) * .7f); }
    auto up = [](const std::vector<float> & v) { float * p; cudaMalloc(&p, v.size() * 4); cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice); return p; };
    float * dn = up(neg), * dw = up(bias), * ds = up(st), * dx = up(x), * da = up(a), * db = up(b), * dnext, * dy;
    cudaMalloc(&dnext, ns * 4); cudaMalloc(&dy, size_t(8) * vh * dim * 4);
    bool ok = true;
    for (int n = 1; n <= 8; ++n) {
        std::vector<float> ref_y, ref_s;
        for (int v : {8, 4, 2}) {
            tqllm_cuda::fn_gdn_set_vpw(v);
            tqllm_cuda::fn_gdn(dx, da, db, dn, dw, ds, dnext, dy, kh, vh, dim, n, eps, nullptr);
            cudaDeviceSynchronize();
            std::vector<float> y(size_t(n) * vh * dim), s2(ns);
            cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(s2.data(), dnext, ns * 4, cudaMemcpyDeviceToHost);
            if (v == 8) { ref_y = y; ref_s = s2; continue; }
            if (std::memcmp(y.data(), ref_y.data(), y.size() * 4) || std::memcmp(s2.data(), ref_s.data(), ns * 4)) {
                std::printf("FAIL: %d tokens, %d columns per warp differ from 8\n", n, v); ok = false;
            }
        }
    }
    tqllm_cuda::fn_gdn_set_vpw(0);
    if (cudaGetLastError() != cudaSuccess) { std::puts("FAIL: CUDA error"); ok = false; }
    std::puts(ok ? "gdn vpw: PASS (2 / 4 / 8 value columns per warp bit-identical)" : "gdn vpw: FAIL");
    return ok ? 0 : 1;
}
