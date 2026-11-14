// embed_multi_q8_0 for more tokens than a grid's y dimension holds (65,535): every token's row must be gathered
// (a contextual embedding document can exceed it). Synthetic Q8_0 table, checked against a host dequantization.
#include "quant_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const int rows = 7, k = 256, n = 70000;  // n > 65,535
    const int blocks = k / 32;
    std::vector<uint8_t> w(size_t(rows) * blocks * 34);
    std::vector<float> ref(size_t(rows) * k);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < blocks; ++b) {
            uint8_t * blk = w.data() + (size_t(r) * blocks + b) * 34;
            const float d = 0.01f * float(1 + r + b);
            const __half hd = __float2half(d);
            std::memcpy(blk, &hd, 2);
            for (int j = 0; j < 32; ++j) {
                const int8_t q = int8_t((r * 37 + b * 11 + j * 5) % 255 - 127);
                blk[2 + j] = uint8_t(q);
                ref[size_t(r) * k + b * 32 + j] = __half2float(hd) * q;
            }
        }
    std::vector<uint32_t> tok(n);
    for (int i = 0; i < n; ++i) tok[i] = uint32_t((i * 3 + i / 1000) % rows);
    uint8_t * dw = nullptr; uint32_t * dt = nullptr; __nv_bfloat16 * dy = nullptr;
    cudaMalloc(&dw, w.size()); cudaMalloc(&dt, size_t(n) * 4); cudaMalloc(&dy, size_t(n) * k * 2);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dt, tok.data(), size_t(n) * 4, cudaMemcpyHostToDevice);
    cudaMemset(dy, 0xff, size_t(n) * k * 2);  // NaN everywhere: a row the kernel skips cannot pass
    tqllm_cuda::embed_multi_q8_0(dw, dt, n, dy, k, 0);
    const cudaError_t e = cudaDeviceSynchronize();
    int failures = 0;
    if (e != cudaSuccess) { std::printf("FAIL: launch for %d tokens: %s\n", n, cudaGetErrorString(e)); ++failures; }
    std::vector<__nv_bfloat16> y(size_t(n) * k);
    cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost);
    size_t bad = 0, first = ~size_t(0);
    for (int i = 0; i < n && !failures; ++i)
        for (int j = 0; j < k; ++j) {
            const float got = __bfloat162float(y[size_t(i) * k + j]), want = ref[size_t(tok[i]) * k + j];
            if (!(std::fabs(got - want) <= std::fabs(want) / 128 + 1e-6f)) { if (!bad++) first = size_t(i); }
        }
    if (bad) { std::printf("FAIL: %zu values differ, first at token %zu\n", bad, first); ++failures; }
    cudaFree(dw); cudaFree(dt); cudaFree(dy);
    std::puts(failures ? "q8 gather long: FAILED" : "q8 gather long: OK");
    return failures ? 1 : 0;
}
