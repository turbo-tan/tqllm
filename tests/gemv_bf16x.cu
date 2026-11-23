// fn_gemv_bf16x_tokens (block-per-row BF16 GEMV on the input's BF16 image, half the rows per block) must give
// fn_gemv_bf16_tokens' bits (fp32 input, each product on its BF16 rounding) at the decode / verify widths, and decline the
// shapes it does not cover.
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
int main() {
    struct Shape { int rows, cols; const char * what; };
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    for (const Shape sh : {Shape{320, 10240, "hc down"}, Shape{4, 10240, "hc inject"}, Shape{48, 2560, "ssm alpha/beta"}, Shape{128, 2560, "indexer k"},
                           Shape{512, 2560, "indexer q"}, Shape{10240, 320, "hc up (not covered)"}}) {
        std::vector<__nv_bfloat16> w(size_t(sh.rows) * sh.cols); for (auto & v : w) v = __float2bfloat16_rn(0.05f * nd(rng));
        std::vector<float> x(size_t(8) * sh.cols); for (auto & v : x) v = nd(rng);
        __nv_bfloat16 * dw, * dxb; float * dx, * y1, * y2;
        cudaMalloc(&dw, w.size() * 2); cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dxb, x.size() * 2);
        cudaMalloc(&y1, size_t(8) * sh.rows * 4); cudaMalloc(&y2, size_t(8) * sh.rows * 4);
        cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        tqllm_cuda::fn_round_bf16(dx, dxb, int64_t(x.size()), nullptr);
        const bool covered = sh.rows <= 16 || (sh.rows <= 1024 && sh.cols >= 1024);
        std::printf("%-20s %5d x %-5d:", sh.what, sh.rows, sh.cols);
        for (int t = 1; t <= 8; ++t) {
            cudaMemset(y1, 0, size_t(8) * sh.rows * 4); cudaMemset(y2, 0xff, size_t(8) * sh.rows * 4);
            tqllm_cuda::fn_gemv_bf16_tokens(dw, dx, y1, sh.rows, sh.cols, t, nullptr);
            const bool ran = tqllm_cuda::fn_gemv_bf16x_tokens(dw, dxb, y2, sh.rows, sh.cols, t, nullptr);
            cudaDeviceSynchronize();
            bool ok = ran == covered;
            if (ran) {
                std::vector<float> a(size_t(t) * sh.rows), b(a.size());
                cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost);
                ok = ok && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
            }
            std::printf(" T%d:%s", t, ok ? (ran ? "same" : "declined") : "FAIL");
            fails += !ok;
        }
        std::printf("\n");
        cudaFree(dw); cudaFree(dx); cudaFree(dxb); cudaFree(y1); cudaFree(y2);
    }
    std::printf("gemv_bf16x: %s\n", fails || cudaGetLastError() != cudaSuccess ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
