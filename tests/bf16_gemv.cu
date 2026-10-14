// BF16 GEMV (Flash-Next resident projections) vs a double oracle on the model's shapes; run with and without
// The few-row and short-row layouts must stay within FP32 rounding of the oracle (same bf16-rounded activations).
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static double run(int rows, int cols, int tokens) {
    std::mt19937 rng(rows * 7 + cols + tokens); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> w(size_t(rows) * cols); std::vector<float> x(size_t(cols) * tokens);
    for (auto & v : w) v = __float2bfloat16_rn(0.05f * nd(rng));
    for (auto & v : x) v = nd(rng);
    __nv_bfloat16 * dw; float * dx, * dy;
    check(cudaMalloc(&dw, w.size() * 2)); check(cudaMalloc(&dx, x.size() * 4)); check(cudaMalloc(&dy, size_t(rows) * tokens * 4));
    check(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice)); check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    tqllm_cuda::fn_gemv_bf16_tokens(dw, dx, dy, rows, cols, tokens, nullptr);
    std::vector<float> y(size_t(rows) * tokens); check(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int t = 0; t < tokens; ++t) for (int r = 0; r < rows; ++r) {
        double e = 0, mag = 0;
        for (int c = 0; c < cols; ++c) {
            const double p = double(__bfloat162float(w[size_t(r) * cols + c])) * double(__bfloat162float(__float2bfloat16_rn(x[size_t(t) * cols + c])));
            e += p; mag += std::fabs(p);
        }
        worst = std::max(worst, std::fabs(y[size_t(t) * rows + r] - e) / std::max(mag, 1e-30));  // error relative to sum |terms|
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    return worst;
}
int main() {
    const int shapes[][2] = {{1, 2560}, {4, 10240}, {16, 10240}, {320, 10240}, {512, 2560}, {10240, 320}, {2560, 640}};
    const char * mode = "block-per-row"; double all = 0;
    for (auto & s : shapes) for (int t : {1, 3}) {
        const double e = run(s[0], s[1], t); all = std::max(all, e);
        std::printf("%s rows %5d cols %5d tokens %d: max |y - oracle| / sum|terms| = %.2e\n", mode, s[0], s[1], t, e);
    }
    std::printf("%s worst %.2e %s\n", mode, all, all < 1e-5 ? "PASS" : "FAIL");
    return all < 1e-5 ? 0 : 1;
}
