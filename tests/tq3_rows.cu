// TQ3_4S row slices: the capped prefill GEMM dequantises row blocks (p + r0 * row_bytes); each block must equal the
// same rows of a whole-matrix dequant. Without a row stride the 16 GB fit could not shrink the FP16 weight buffer.
#include "../src/formats/weight_format.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    const auto * f = tqllm::weight_format(46);
    if (!f) { std::puts("FAIL: TQ3_4S not registered"); return 1; }
    const int rows = 768, cols = 512;
    const size_t rb = f->row_bytes(cols);
    if (rb != size_t(cols) / 32 * 16) { std::printf("FAIL: TQ3_4S row_bytes %zu, expected %d\n", rb, cols / 32 * 16); return 1; }
    std::vector<char> raw(size_t(rows) * cols / 32 * 16);
    std::mt19937 rng(5);
    for (auto & b : raw) b = char(rng());
    for (size_t i = 0; i < raw.size(); i += 16) raw[i] = char(1 + rng() % 200);  // nonzero block scales
    std::vector<void *> owned;
    auto up = [&](const void * h, size_t n) -> const void * {
        void * d = nullptr; cudaMalloc(&d, n); cudaMemcpy(d, h, n, cudaMemcpyHostToDevice); owned.push_back(d); return d;
    };
    const tqllm::Mat m = f->load(raw, rows, cols, up, nullptr);
    __half *full, *part;
    cudaMalloc(&full, size_t(rows) * cols * 2);
    cudaMalloc(&part, size_t(rows) * cols * 2);
    m.fmt->dequant_h(m, full, nullptr);
    const int blk = 256;
    for (int r0 = 0; r0 < rows; r0 += blk) {
        tqllm::Mat v = m;
        v.p = static_cast<const uint8_t *>(m.p) + size_t(r0) * m.fmt->row_bytes(cols);
        v.rows = std::min(blk, rows - r0);
        m.fmt->dequant_h(v, part + size_t(r0) * cols, nullptr);
    }
    cudaDeviceSynchronize();
    std::vector<__half> a(size_t(rows) * cols), b(a.size());
    cudaMemcpy(a.data(), full, a.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), part, b.size() * 2, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) { std::puts("FAIL: CUDA error"); return 1; }
    if (std::memcmp(a.data(), b.data(), a.size() * 2)) { std::puts("FAIL: row-block dequant differs from whole"); return 1; }
    for (void * d : owned) cudaFree(d);
    std::puts("tq3_rows: ok (row blocks == whole dequant, bit-exact)");
    return 0;
}
