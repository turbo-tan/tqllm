// tile_expert_rows_device (resident GB10 banks tiled in place) must write the same bytes as the host tile_expert_rows the
// packed-experts path uses, expert by expert, for every tileable format's block size.
#include "quant_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <array>
#include <cstring>
#include <random>
#include <vector>
int main() {
    int bad = 0;
    std::mt19937 g(5);
    for (int fmt : {4, 5, 10, 14, 16, 0, 1}) {
        const int bpb = tqllm_cuda::tiled_block_bytes(fmt);
        if (!bpb) continue;
        for (auto [rows, k, count] : {std::array<int, 3>{640, 2560, 3}, std::array<int, 3>{2560, 640, 2}, std::array<int, 3>{32, 128, 5}}) {
            const size_t stride = size_t(rows) * (k / 32) * bpb;
            std::vector<char> h(stride * count), want(h.size());
            for (auto & c : h) c = char(g() & 255);
            for (int e = 0; e < count; ++e) tqllm_cuda::tile_expert_rows(h.data() + e * stride, want.data() + e * stride, rows, k, bpb);
            char * d = nullptr; cudaMalloc(&d, h.size()); cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice);
            tqllm_cuda::tile_expert_rows_device(d, count, rows, k, bpb, nullptr);
            std::vector<char> got(h.size()); cudaMemcpy(got.data(), d, got.size(), cudaMemcpyDeviceToHost); cudaFree(d);
            if (std::memcmp(got.data(), want.data(), got.size())) { std::printf("FAIL: fmt %d (bpb %d) %d x %d x %d\n", fmt, bpb, count, rows, k); ++bad; }
        }
    }
    if (cudaGetLastError() != cudaSuccess) { std::puts("FAIL: CUDA error"); ++bad; }
    std::puts(bad ? "tile device: FAIL" : "tile device: PASS (in-place device tiling = host tile_expert_rows)");
    return bad ? 1 : 0;
}
