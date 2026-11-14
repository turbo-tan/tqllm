// u4h-family matvec with 9..64 columns (several sequences' verify rows) must give each column the bits of a one-column
// launch: short matrices split their column tiles over CTAs, longer ones keep every tile in one CTA (16 / 24 / 32-column
// kernels; 33..64 columns as two 32-column blocks per row block). Formats 13 (IQ4_NL-like) and 15 (8-bit) at the Flash-Next dense shapes.
#include "formats/weight/quant_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
int main() {
    std::mt19937 rng(7); std::uniform_int_distribution<int> byte(0, 255); std::normal_distribution<float> nd(0.f, 1.f);
    struct Shape { int rows, k; };
    const Shape shapes[] = {{320, 10240}, {640, 2560}, {512, 2560}, {2560, 640}, {10240, 2560}, {1792, 1024}};
    const int widths[] = {9, 12, 16, 20, 24, 32, 33, 40, 48, 56, 64};
    bool ok = true;
    for (int fmt : {13, 15}) for (const auto & sh : shapes) {
        const int rows = sh.rows, k = sh.k, maxc = 64;
        std::vector<uint8_t> w(size_t(rows) * k); for (auto & b : w) b = uint8_t(byte(rng));
        std::vector<__half> aux(size_t(rows) * k / 4); for (auto & a : aux) a = __float2half(0.002f + 0.01f * std::abs(nd(rng)));
        std::vector<__half> x(size_t(maxc) * k); for (auto & v : x) v = __float2half(nd(rng));
        uint8_t * dw; __half * da, * dx; float * y1, * y2;
        cudaMalloc(&dw, w.size()); cudaMalloc(&da, aux.size() * 2); cudaMalloc(&dx, x.size() * 2);
        cudaMalloc(&y1, size_t(maxc) * rows * 4); cudaMalloc(&y2, size_t(maxc) * rows * 4);
        cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice); cudaMemcpy(da, aux.data(), aux.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        for (int c = 0; c < maxc; ++c) tqllm_cuda::u4h_family_matvec_f32(fmt, dw, da, dx + size_t(c) * k, y1 + size_t(c) * rows, rows, k, 1, nullptr);
        for (int n : widths) {
            cudaMemset(y2, 0xff, size_t(maxc) * rows * 4);
            tqllm_cuda::u4h_family_matvec_f32(fmt, dw, da, dx, y2, rows, k, n, nullptr);
            cudaDeviceSynchronize();
            std::vector<float> a(size_t(n) * rows), b(a.size());
            cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost);
            if (std::memcmp(a.data(), b.data(), a.size() * 4)) { std::printf("FAIL: fmt %d, %d x %d, %d columns differ from one-column launches\n", fmt, rows, k, n); ok = false; }
        }
        cudaFree(dw); cudaFree(da); cudaFree(dx); cudaFree(y1); cudaFree(y2);
    }
    if (cudaGetLastError() != cudaSuccess) { std::puts("FAIL: CUDA error"); ok = false; }
    std::puts(ok ? "u4h columns: PASS (9..64 columns bit-identical to one-column launches, split and unsplit grids)" : "u4h columns: FAIL");
    return ok ? 0 : 1;
}
