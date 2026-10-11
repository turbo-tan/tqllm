// INT8 prefill GEMM (Ampere) vs an FP32 reference (cuBLAS SGEMM on the same FP16 inputs), with outlier weight rows and outlier
// tokens (per-row / per-token scales must keep the other rows accurate), a zero row, and an output column offset.
// Needs an sm_8x GPU without FP8 (Ampere); skipped elsewhere.
#include "int8_gemm.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char * what, double v) {
    std::printf("%-58s %.5f %s\n", what, v, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}
}  // namespace

int main() {
    if (!tqllm_cuda::Int8Gemm::supported()) { std::puts("int8_gemm: skipped (no int8 tensor cores)"); return 0; }
    const int N = 1024, K = 5120, M = 200, LDO = 3 * N, OFF = N;  // output written at column OFF of a wider buffer
    std::mt19937 rng(7);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<__half> hw(size_t(N) * K), hx(size_t(M) * K);
    for (int n = 0; n < N; ++n) {
        const float s = n == 5 ? 0.f : n % 97 == 3 ? 2.f : 0.02f;  // row 5 zero, a few outlier rows 100x larger
        for (int k = 0; k < K; ++k) hw[size_t(n) * K + k] = __float2half(s * g(rng));
    }
    for (int m = 0; m < M; ++m) {
        const float s = m % 50 == 7 ? 30.f : 1.f;  // outlier tokens (massive activations)
        for (int k = 0; k < K; ++k) hx[size_t(m) * K + k] = __float2half(s * g(rng) * (k % 512 == 1 ? 3.5f : 1.f)  /* channel outliers after the engine's 32-group rotation (x20 / sqrt(32)) */);
    }
    __half *dw, *dx, *dy;
    float *fw, *fx, *fy;
    cudaMalloc(&dw, hw.size() * 2); cudaMalloc(&dx, hx.size() * 2); cudaMalloc(&dy, size_t(M) * LDO * 2);
    cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(dy, 0, size_t(M) * LDO * 2);
    std::vector<float> w32(hw.size()), x32(hx.size());
    for (size_t i = 0; i < hw.size(); ++i) w32[i] = __half2float(hw[i]);
    for (size_t i = 0; i < hx.size(); ++i) x32[i] = __half2float(hx[i]);
    cudaMalloc(&fw, w32.size() * 4); cudaMalloc(&fx, x32.size() * 4); cudaMalloc(&fy, size_t(M) * N * 4);
    cudaMemcpy(fw, w32.data(), w32.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(fx, x32.data(), x32.size() * 4, cudaMemcpyHostToDevice);
    cublasHandle_t b;
    cublasCreate(&b);
    const float one = 1.f, zero = 0.f;
    cublasSetMathMode(b, CUBLAS_PEDANTIC_MATH);
    cublasSgemm(b, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, fw, K, fx, K, &zero, fy, N);

    tqllm_cuda::Int8Gemm g8;
    g8.init(size_t(N) * K, size_t(M) * K, N, M);
    int32_t * acc; float * sw_all; __nv_bfloat16 * yb;
    cudaMalloc(&acc, size_t(M) * LDO * 4); cudaMalloc(&sw_all, size_t(LDO) * 4); cudaMalloc(&yb, size_t(M) * LDO * 2);
    cudaMemset(acc, 0, size_t(M) * LDO * 4); cudaMemset(sw_all, 0, size_t(LDO) * 4);
    g8.quant_x(dx, K, M, K, nullptr);
    g8.gemm(dw, N, K, M, acc + OFF, LDO, sw_all + OFF, nullptr);
    tqllm_cuda::int8_i32_to_h_scaled(acc, dy, M, LDO, sw_all, g8.sx(), nullptr);
    tqllm_cuda::int8_i32_to_bf16_scaled(acc, yb, M, LDO, sw_all, g8.sx(), nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::puts("FAIL: CUDA error"); return 1; }

    std::vector<float> ref(size_t(M) * N);
    std::vector<__half> got(size_t(M) * LDO);
    std::vector<__nv_bfloat16> gotb(size_t(M) * LDO);
    cudaMemcpy(ref.data(), fy, ref.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(), dy, got.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(gotb.data(), yb, gotb.size() * 2, cudaMemcpyDeviceToHost);
    float ref_max = 0.f;
    for (float v : ref) ref_max = std::fmax(ref_max, std::fabs(v));
    check(ref_max < 60000.f, "reference fits FP16", ref_max);
    double e[3] = {}, r[3] = {}, eb = 0, rb = 0;
    bool untouched = true, zero_row = true, finite = true;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < LDO; ++n) {
            const float v = __half2float(got[size_t(m) * LDO + n]);
            if (n < OFF || n >= OFF + N) { untouched &= v == 0.f; continue; }
            const int c = n - OFF;
            finite &= std::isfinite(v);
            if (c == 5) { zero_row &= v == 0.f; continue; }
            const double t = ref[size_t(m) * N + c], d = double(v) - t, db = double(__bfloat162float(gotb[size_t(m) * LDO + n])) - t;
            const int grp = c % 97 == 3 ? 1 : m % 50 == 7 ? 2 : 0;
            e[grp] += d * d; r[grp] += t * t; eb += db * db; rb += t * t;
        }
    check(finite, "all outputs finite", 0);
    check(untouched, "columns outside [OFF, OFF + N) zero (sw 0 there)", 0);
    check(zero_row, "zero weight row gives zeros", 0);
    const double rel0 = std::sqrt(e[0] / r[0]), rel1 = std::sqrt(e[1] / r[1]), rel2 = std::sqrt(e[2] / r[2]);
    check(rel0 < 0.05, "relative error, normal rows x normal tokens (fp16 out)", rel0);
    check(rel1 < 0.05, "relative error, outlier weight rows", rel1);
    check(rel2 < 0.05, "relative error, outlier tokens", rel2);
    check(std::sqrt(eb / rb) < 0.05, "relative error, bf16 output", std::sqrt(eb / rb));
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::puts("int8_gemm: ok");
    return 0;
}
