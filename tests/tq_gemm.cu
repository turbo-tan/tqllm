// tq4_gemm_h / tq4b3_gemm_h (prefill GEMM without an FP16 weight copy) against a host reference built from the
// weights tq4_dequant_h / tq4b3_dequant_h write (the values the old dequant + cuBLAS path multiplied).
#include "quant_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)
static int fails = 0;
static void run(int bits, int rows, int k, int n, std::mt19937 & rng, float xmax = 2.f) {
    const size_t wbytes = bits == 4 ? size_t(rows) * k / 2 : size_t(rows) * (k / 32) * 3 * 4;
    std::vector<uint8_t> w(wbytes);
    for (auto & b : w) b = uint8_t(rng());
    std::vector<__half> sc(size_t(rows) * (k / 128)), x(size_t(n) * k);
    std::uniform_real_distribution<float> us(0.002f, 0.02f), ux(-xmax, xmax);
    for (auto & s : sc) s = __float2half(us(rng));
    for (auto & v : x) v = __float2half(ux(rng));
    uint8_t * dw; __half *dsc, *dx, *dwh; __nv_bfloat16 * dy;
    CK(cudaMalloc(&dw, wbytes)); CK(cudaMalloc(&dsc, sc.size() * 2)); CK(cudaMalloc(&dx, x.size() * 2));
    CK(cudaMalloc(&dwh, size_t(rows) * k * 2)); CK(cudaMalloc(&dy, size_t(n) * rows * 2));
    CK(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsc, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    if (bits == 4) { tqllm_cuda::tq4_dequant_h(dw, dsc, dwh, rows, k, nullptr); tqllm_cuda::tq4_gemm_h(dw, dsc, dx, dy, rows, k, n, rows, nullptr); }
    else { tqllm_cuda::tq4b3_dequant_h(dw, dsc, dwh, rows, k, nullptr); tqllm_cuda::tq4b3_gemm_h(dw, dsc, dx, dy, rows, k, n, rows, nullptr); }
    CK(cudaDeviceSynchronize());
    std::vector<__half> wh(size_t(rows) * k); std::vector<__nv_bfloat16> y(size_t(n) * rows);
    CK(cudaMemcpy(wh.data(), dwh, wh.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost));
    double maxref = 0, maxerr = 0;
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < rows; ++r) {
            double ref = 0;
            for (int i = 0; i < k; ++i) ref += double(__half2float(wh[size_t(r) * k + i])) * __half2float(x[size_t(c) * k + i]);
            maxref = std::max(maxref, std::fabs(ref));
            maxerr = std::max(maxerr, std::fabs(ref - double(__bfloat162float(y[size_t(c) * rows + r]))));
        }
    const double rel = maxerr / std::max(maxref, 1e-9);
    const bool ok = rel < 1e-2;
    std::printf("TQ%d rows %4d k %4d n %3d |x|<=%5.0f: max rel err %.2e %s\n", bits, rows, k, n, xmax, rel, ok ? "ok" : "FAIL");
    fails += !ok;
    cudaFree(dw); cudaFree(dsc); cudaFree(dx); cudaFree(dwh); cudaFree(dy);
}
int main() {
    std::mt19937 rng(20261011);
    for (int bits : {4, 3})
        for (int n : {17, 64, 100, 129}) run(bits, 200, 384, n, rng);  // rows not a multiple of 128, k = 3 quant blocks
    run(4, 512, 1024, 333, rng);
    run(3, 512, 1024, 333, rng);
    // large activations (outlier channels): a 128-weight partial sum exceeds the FP16 range (65504); the GB10 KLD gate
    // showed p99.9 KLD ~7 on 479-token prefills when the per-block accumulation was FP16
    run(4, 256, 512, 96, rng, 300.f);
    run(3, 256, 512, 96, rng, 300.f);
    std::printf("tq_gemm: %s\n", fails ? "FAIL" : "PASS");
    return fails ? EXIT_FAILURE : EXIT_SUCCESS;
}
