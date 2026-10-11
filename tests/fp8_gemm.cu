// FP8 prefill GEMM vs an FP32 reference (cuBLAS SGEMM on the same FP16 inputs), with outlier weight rows and outlier
// tokens (per-row / per-token scales must keep the other rows accurate), a zero row, and an output column offset.
// Needs FP8 tensor cores (sm_89+); skipped elsewhere.
#include "fp8_gemm.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
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
    if (!tqllm_cuda::Fp8Gemm::supported()) { std::puts("fp8_gemm: skipped (no FP8 tensor cores)"); return 0; }
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
        for (int k = 0; k < K; ++k) hx[size_t(m) * K + k] = __float2half(s * g(rng) * (k % 512 == 1 ? 20.f : 1.f));
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

    tqllm_cuda::Fp8Gemm f8;
    f8.init(size_t(N) * K, size_t(M) * K, N, M);
    f8.run(dw, N, K, dx, K, M, dy + OFF, LDO, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::puts("FAIL: CUDA error"); return 1; }

    std::vector<float> ref(size_t(M) * N);
    std::vector<__half> got(size_t(M) * LDO);
    cudaMemcpy(ref.data(), fy, ref.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(), dy, got.size() * 2, cudaMemcpyDeviceToHost);
    float ref_max = 0.f;
    for (float v : ref) ref_max = std::fmax(ref_max, std::fabs(v));
    check(ref_max < 60000.f, "reference fits FP16 (as the FP16 prefill path requires)", ref_max);
    // relative error per output row-group: normal rows x normal tokens, outlier rows, outlier tokens
    double e[3] = {}, r[3] = {};
    bool untouched = true, zero_row = true, finite = true;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < LDO; ++n) {
            const float v = __half2float(got[size_t(m) * LDO + n]);
            if (n < OFF || n >= OFF + N) { untouched &= v == 0.f; continue; }
            const int c = n - OFF;
            finite &= std::isfinite(v);
            if (c == 5) { zero_row &= v == 0.f; continue; }
            const double d = double(v) - ref[size_t(m) * N + c], t = ref[size_t(m) * N + c];
            const int grp = c % 97 == 3 ? 1 : m % 50 == 7 ? 2 : 0;
            e[grp] += d * d; r[grp] += t * t;
        }
    check(finite, "all outputs finite", 0);
    check(untouched, "columns outside [OFF, OFF + N) untouched", 0);
    check(zero_row, "zero weight row gives zeros", 0);
    // E4M3 has 3 mantissa bits: per-product relative error ~2^-4, the dot product averages it down to a few percent
    const double rel0 = std::sqrt(e[0] / r[0]), rel1 = std::sqrt(e[1] / r[1]), rel2 = std::sqrt(e[2] / r[2]);
    check(rel0 < 0.05, "relative error, normal rows x normal tokens", rel0);
    check(rel1 < 0.05, "relative error, outlier weight rows", rel1);
    check(rel2 < 0.05, "relative error, outlier tokens", rel2);
    // split form, as the engine uses it: X quantised once, resident weights (second call without the FP16 copy),
    // unscaled GEMM, scales applied by the fused BF16 conversion -> must match run() exactly
    {
        tqllm_cuda::Fp8Gemm g;
        g.init(size_t(N) * K, size_t(M) * K, N, M, size_t(N) * K + N * 4);
        float * sw_all;
        __half * yc;
        __nv_bfloat16 * yb;
        cudaMalloc(&sw_all, size_t(N) * 4); cudaMalloc(&yc, size_t(M) * N * 2); cudaMalloc(&yb, size_t(M) * N * 2);
        g.quant_x(dx, K, M, K, nullptr);
        g.gemm(dw, dw, N, K, M, yc, N, sw_all, nullptr);           // first call: quantise and keep
        const bool kept = g.cached(dw);
        g.gemm(dw, nullptr, N, K, M, yc, N, sw_all, nullptr);      // second: resident, no FP16 weights
        tqllm_cuda::fp8_h_to_bf16_scaled(yc, yb, M, N, sw_all, g.sx(), tqllm_cuda::Fp8Gemm::kMul, nullptr);
        cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> hb(size_t(M) * N);
        cudaMemcpy(hb.data(), yb, hb.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                const double a = __bfloat162float(hb[size_t(m) * N + n]), b = __half2float(got[size_t(m) * LDO + OFF + n]);
                maxd = std::max(maxd, std::fabs(a - b) / (std::fabs(b) + 1e-3));
            }
        check(kept, "weights kept resident after the first call", 0);
        check(maxd < 0.01, "split + resident + fused BF16 conversion vs run() (max rel diff, BF16 rounding)", maxd);
        // verify-sized batches (--draft-max > 7 on GB10: 9-16 tree columns reach the prefill GEMM): every token count
        // must run and match run() on the same tokens
        for (int t : {9, 11, 13, 15, 16, 17}) {
            bool threw = false;
            try {
                g.quant_x(dx, K, t, K, nullptr);
                g.gemm(dw, nullptr, N, K, t, yc, N, sw_all, nullptr);
                tqllm_cuda::fp8_h_to_bf16_scaled(yc, yb, t, N, sw_all, g.sx(), tqllm_cuda::Fp8Gemm::kMul, nullptr);
                cudaDeviceSynchronize();
            } catch (const std::exception & e) {
                threw = true;
                std::printf("  %d tokens: %s\n", t, e.what());
            }
            double md = 1e30;
            if (!threw) {
                cudaMemcpy(hb.data(), yb, size_t(t) * N * 2, cudaMemcpyDeviceToHost);
                md = 0;
                for (int m = 0; m < t; ++m)
                    for (int n = 0; n < N; ++n) {
                        const double a = __bfloat162float(hb[size_t(m) * N + n]), b = __half2float(got[size_t(m) * LDO + OFF + n]);
                        md = std::max(md, std::fabs(a - b) / (std::fabs(b) + 1e-3));
                    }
            }
            char what[96];
            std::snprintf(what, sizeof what, "%d-token batch runs and matches run() (max rel diff)", t);
            check(!threw && md < 0.01, what, md);
        }
    }
    // the 27B shape that failed on GB10 with --draft-max 10 (2026-10-09: "matmul rows 16384 k 5120 tok 16 ... status 14"),
    // engine sizing (max tokens of a prefill chunk), resident weights, a verify-sized batch after a prefill-sized one
    {
        const int R = 16384, KK = 5120, T = 16, MAXT = 4096;
        std::vector<__half> w2(size_t(R) * KK), x2(size_t(MAXT) * KK);
        for (auto & v : w2) v = __float2half(0.02f * g(rng));
        for (auto & v : x2) v = __float2half(g(rng));
        __half *dw2, *dx2, *y2;
        float * sw2;
        cudaMalloc(&dw2, w2.size() * 2); cudaMalloc(&dx2, x2.size() * 2); cudaMalloc(&y2, size_t(MAXT) * R * 2); cudaMalloc(&sw2, size_t(R) * 4);
        cudaMemcpy(dw2, w2.data(), w2.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dx2, x2.data(), x2.size() * 2, cudaMemcpyHostToDevice);
        tqllm_cuda::Fp8Gemm g2;
        g2.init(size_t(R) * KK, size_t(MAXT) * KK, R, MAXT, size_t(R) * KK + R * 4);
        for (int t : {MAXT, T, 9, 12}) {
            std::string err;
            try {
                g2.quant_x(dx2, KK, t, KK, nullptr);
                g2.gemm(dw2, dw2, R, KK, t, y2, R, sw2, nullptr);
                cudaDeviceSynchronize();
            } catch (const std::exception & e) {
                err = e.what();
            }
            char what[128];
            std::snprintf(what, sizeof what, "27B shape %d x %d, %d tokens runs%s%s", R, KK, t, err.empty() ? "" : ": ", err.c_str());
            check(err.empty(), what, 0);
        }
        // verify steps are captured into CUDA graphs: a token count first seen during capture must still run (the
        // per-shape algorithm timing cannot synchronise inside a capture), and the replayed graph must match an
        // uncaptured call on the same inputs
        {
            const int t = 10;
            cudaStream_t cs;
            cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking);
            std::string err;
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            try {
                cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
                g2.quant_x(dx2, KK, t, KK, cs);
                g2.gemm(dw2, nullptr, R, KK, t, y2, R, sw2, cs);
            } catch (const std::exception & e) {
                err = e.what();
            }
            const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
            if (err.empty() && ce != cudaSuccess) err = std::string("capture: ") + cudaGetErrorString(ce);
            if (err.empty() && cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) err = "graph instantiate failed";
            std::vector<__half> a(size_t(t) * R), b(size_t(t) * R);
            if (err.empty()) {
                cudaGraphLaunch(exec, cs);
                cudaStreamSynchronize(cs);
                cudaMemcpy(a.data(), y2, a.size() * 2, cudaMemcpyDeviceToHost);
                g2.quant_x(dx2, KK, t, KK, cs);
                g2.gemm(dw2, nullptr, R, KK, t, y2, R, sw2, cs);
                cudaStreamSynchronize(cs);
                cudaMemcpy(b.data(), y2, b.size() * 2, cudaMemcpyDeviceToHost);
            }
            cudaGetLastError();
            const bool same = err.empty() && std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
            char what[160];
            std::snprintf(what, sizeof what, "new shape (%d tokens) inside graph capture runs, replay == direct%s%s", t,
                          err.empty() ? "" : ": ", err.c_str());
            check(same, what, 0);
            if (exec) cudaGraphExecDestroy(exec);
            if (graph) cudaGraphDestroy(graph);
            cudaStreamDestroy(cs);
        }
        cudaFree(dw2); cudaFree(dx2); cudaFree(y2); cudaFree(sw2);
    }
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::puts("fp8_gemm: ok");
    return 0;
}
