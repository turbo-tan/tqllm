// More than 8 tokens (several sequences' verify rows): fn_gemv_bf16_tokens / _tokens_ld run all token tiles in one launch.
// Each (row, token) must equal the old one-launch-per-8-tokens result bit for bit, and the one launch must be faster
// (the weights are read once from DRAM, not once per 8 tokens). Shapes: the hyper-connection mixers (320 x 10240, 10240 x
// 320) and the router (512 x 2560, written strided).
#include <algorithm>
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static int failures = 0;
static void report(bool ok, const char * what) { std::printf("%-86s %s\n", what, ok ? "ok" : "FAIL"); failures += !ok; }

int main() {
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
    struct Shape { int rows, cols; bool ld; const char * name; };
    const Shape shapes[] = {{320, 10240, false, "320 x 10240 (block per row)"}, {10240, 320, false, "10240 x 320 (short rows)"},
                            {512, 2560, true, "512 x 2560 router, strided"}};
    cudaEvent_t a, b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
    for (const auto & sh : shapes) {
        const int rows = sh.rows, cols = sh.cols, maxt = 40; const int64_t ld = sh.ld ? rows + 1 : rows;
        std::vector<__nv_bfloat16> w(size_t(rows) * cols); std::vector<float> x(size_t(cols) * maxt);
        for (auto & v : w) v = __float2bfloat16_rn(0.05f * nd(rng));
        for (auto & v : x) v = nd(rng);
        __nv_bfloat16 * dw; float * dx, * y1, * y2;
        check(cudaMalloc(&dw, w.size() * 2)); check(cudaMalloc(&dx, x.size() * 4));
        check(cudaMalloc(&y1, size_t(ld) * maxt * 4)); check(cudaMalloc(&y2, size_t(ld) * maxt * 4));
        check(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice)); check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        auto call = [&](const float * xx, float * yy, int t) {
            if (sh.ld) tqllm_cuda::fn_gemv_bf16_tokens_ld(dw, xx, yy, rows, cols, t, ld, nullptr);
            else tqllm_cuda::fn_gemv_bf16_tokens(dw, xx, yy, rows, cols, t, nullptr);
        };
        auto per8 = [&](int t) { for (int t0 = 0; t0 < t; t0 += 8) call(dx + size_t(t0) * cols, y2 + size_t(t0) * ld, std::min(8, t - t0)); };
        for (int t : {9, 12, 17, 20, 24, 32, 38}) {
            check(cudaMemset(y1, 0, size_t(ld) * maxt * 4)); check(cudaMemset(y2, 0, size_t(ld) * maxt * 4));
            call(dx, y1, t); per8(t);
            std::vector<float> r1(size_t(ld) * t), r2(r1.size());
            check(cudaMemcpy(r1.data(), y1, r1.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(r2.data(), y2, r2.size() * 4, cudaMemcpyDeviceToHost));
            bool same = true;
            for (int tt = 0; tt < t; ++tt) same &= std::memcmp(&r1[size_t(tt) * ld], &r2[size_t(tt) * ld], size_t(rows) * 4) == 0;
            char msg[160]; std::snprintf(msg, sizeof msg, "%s, %d tokens: one launch == one launch per 8 tokens", sh.name, t);
            report(same, msg);
        }
        for (int t : {16, 24, 38}) {
            // fastest of 7 interleaved trials per arm: ctest -j runs other GPU tests beside this one, and a single timed
            // window could land on their load (a lone failure under ctest -j 2, 11-12; 5/5 alone)
            float ms1 = 1e30f, ms2 = 1e30f; constexpr int reps = 50;
            for (int k = 0; k < 5; ++k) { call(dx, y1, t); per8(t); }
            for (int trial = 0; trial < 7; ++trial) {
                float m = 0;
                check(cudaEventRecord(a)); for (int k = 0; k < reps; ++k) call(dx, y1, t); check(cudaEventRecord(b)); check(cudaEventSynchronize(b)); check(cudaEventElapsedTime(&m, a, b)); ms1 = std::min(ms1, m);
                check(cudaEventRecord(a)); for (int k = 0; k < reps; ++k) per8(t); check(cudaEventRecord(b)); check(cudaEventSynchronize(b)); check(cudaEventElapsedTime(&m, a, b)); ms2 = std::min(ms2, m);
            }
            std::printf("  %s, %d tokens: one launch %.1f us, per 8 tokens %.1f us\n", sh.name, t, 1e3f * ms1 / reps, 1e3f * ms2 / reps);
            if (t == 24) { char msg[160]; std::snprintf(msg, sizeof msg, "%s, 24 tokens: one launch faster", sh.name); report(ms1 < ms2, msg); }
        }
        cudaFree(dw); cudaFree(dx); cudaFree(y1); cudaFree(y2);
    }
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
