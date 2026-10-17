// Prefill router GEMV (fn_gemv_bf16_chunk) vs the per-token path it replaces (fn_gemv_bf16_tokens): bitwise equal on the
// Flash-Next router shapes ([512][2560] experts, [1][2560] shared-expert gate) for every chunk length, written with a
// row stride (the router's [n][experts + 1] logits).
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static bool run(int rows, int cols, int tokens, bool bx) {
    std::mt19937 rng(rows * 131 + cols + tokens); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> w(size_t(rows) * cols); std::vector<float> x(size_t(cols) * tokens);
    for (auto & v : w) v = __float2bfloat16_rn(0.05f * nd(rng));
    for (auto & v : x) v = nd(rng);
    const int64_t ldy = rows + 3;
    __nv_bfloat16 * dw; float * dx, * dref, * dy;
    check(cudaMalloc(&dw, w.size() * 2)); check(cudaMalloc(&dx, x.size() * 4));
    check(cudaMalloc(&dref, size_t(rows) * tokens * 4)); check(cudaMalloc(&dy, size_t(ldy) * tokens * 4));
    check(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice)); check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemset(dy, 0xff, size_t(ldy) * tokens * 4));
    tqllm_cuda::fn_gemv_bf16_tokens(dw, dx, dref, rows, cols, tokens, nullptr);
    tqllm_cuda::fn_gemv_bf16_chunk(dw, dx, dy, rows, cols, tokens, ldy, nullptr);
    if (bx) { __nv_bfloat16 * dxb; check(cudaMalloc(&dxb, x.size() * 2)); check(cudaMemset(dy, 0xff, size_t(ldy) * tokens * 4));
              tqllm_cuda::fn_round_bf16(dx, dxb, int64_t(x.size()), nullptr);
              tqllm_cuda::fn_gemv_bf16_chunk_bx(dw, dxb, dy, rows, cols, tokens, ldy, nullptr); check(cudaDeviceSynchronize()); cudaFree(dxb); }
    check(cudaDeviceSynchronize());
    std::vector<float> ref(size_t(rows) * tokens), y(size_t(ldy) * tokens);
    check(cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    int diff = 0, pad = 0;
    for (int t = 0; t < tokens; ++t) {
        if (std::memcmp(&ref[size_t(t) * rows], &y[size_t(t) * ldy], size_t(rows) * 4)) ++diff;
        for (int64_t r = rows; r < ldy; ++r) { uint32_t b; std::memcpy(&b, &y[size_t(t) * ldy + r], 4); if (b != 0xffffffffu) ++pad; }
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dref); cudaFree(dy);
    std::printf("rows %4d cols %5d tokens %4d%s: %d token rows differ, %d stride cells written\n", rows, cols, tokens, bx ? " (bf16 x)" : "", diff, pad);
    return !diff && !pad;
}
int main() {
    bool ok = true;
    for (const int rows : {512, 1, 16, 9})
        for (const int tokens : {1, 3, 4, 5, 8, 63, 64, 65, 1023, 1024})
            for (const bool bx : {false, true}) ok = run(rows, 2560, tokens, bx) && ok;
    std::puts(ok ? "gemv_chunk PASS" : "gemv_chunk FAIL");
    return ok ? 0 : 1;
}
