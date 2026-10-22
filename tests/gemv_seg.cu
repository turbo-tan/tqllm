// fn_gemv_bf16_segments must equal fn_gemv_bf16_tokens bit for bit on the router shapes (512 x 2560 and 1 x 2560).
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
int main() {
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f); int bad = 0, cases = 0;
    const int cols = 2560, E = 512;
    for (int n = 1; n <= 4; ++n) for (int trial = 0; trial < 5; ++trial) {
        std::vector<__nv_bfloat16> w[4]; const int rows[4] = {E, 1, E, 1};
        __nv_bfloat16 * dw[4]; float * dx; float * ref[4]; float * seg[2];
        std::vector<float> x(size_t(n) * cols); for (auto & v : x) v = nd(rng);
        cudaMalloc(&dx, x.size() * 4); cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        for (int i = 0; i < 4; ++i) {
            w[i].resize(size_t(rows[i]) * cols); for (auto & v : w[i]) v = __float2bfloat16(nd(rng) * 0.05f);
            cudaMalloc(&dw[i], w[i].size() * 2); cudaMemcpy(dw[i], w[i].data(), w[i].size() * 2, cudaMemcpyHostToDevice);
            cudaMalloc(&ref[i], size_t(n) * rows[i] * 4);
            tqllm_cuda::fn_gemv_bf16_tokens(dw[i], dx, ref[i], rows[i], cols, n, nullptr);
        }
        for (int j = 0; j < 2; ++j) cudaMalloc(&seg[j], size_t(n) * (E + 1) * 4);
        tqllm_cuda::GemvSegs g; int r = 0;
        for (int i = 0; i < 4; ++i) { g.w[i] = dw[i]; g.y[i] = seg[i / 2] + (i % 2 ? E : 0); g.ld[i] = E + 1; g.start[i] = r; r += rows[i]; g.start[i + 1] = r; }
        g.n = 4; tqllm_cuda::fn_gemv_bf16_segments(g, dx, cols, n, nullptr);
        for (int i = 0; i < 4; ++i) {
            std::vector<float> a(size_t(n) * rows[i]), b(size_t(n) * (E + 1));
            cudaMemcpy(a.data(), ref[i], a.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), seg[i / 2], b.size() * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < n; ++t) for (int k = 0; k < rows[i]; ++k)
                if (std::memcmp(&a[size_t(t) * rows[i] + k], &b[size_t(t) * (E + 1) + (i % 2 ? E : 0) + k], 4)) { ++bad; t = n; break; }
            cudaFree(dw[i]); cudaFree(ref[i]);
        }
        cudaFree(dx); cudaFree(seg[0]); cudaFree(seg[1]); ++cases;
    }
    if (cudaGetLastError() != cudaSuccess) { std::printf("gemv_seg: CUDA error\n"); return 1; }
    std::printf("gemv_seg: %d cases, %d mismatching blocks\n%s\n", cases, bad, bad ? "gemv seg FAIL" : "gemv seg PASS"); return bad ? 1 : 0;
}
