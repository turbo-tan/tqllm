// The paired gate / up dedup launch switches to its prefetching variant from a per-format group count that start-up
// calibration measures (calibrate_dedup_pf). Both variants must give the same bits at any count, so the switch point never
// changes results; the calibration returns a measured count or INT32_MAX (never faster).
#include "weight_format.hpp"
#include "quant_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
int main() {
    const int slots = 128, rows = 640, cols = 2560, ntok = 8, DS = tqllm_cuda::kDedupStride, C = 2;
    std::mt19937 g(9);
    int bad = 0;
    struct Case { unsigned ty; int bb; const char * what; };
    for (const Case cs : {Case{21, 110, "IQ3_S"}, Case{18, 98, "IQ3_XXS"}, Case{22, 82, "IQ2_S"}}) {
        std::vector<char> raw(size_t(rows) * slots * cols / 256 * cs.bb);
        for (auto & c : raw) c = char(g() & 255);
        const __half d = __float2half(0.001f);
        for (size_t b = 0; b < raw.size() / cs.bb; ++b) std::memcpy(raw.data() + b * cs.bb, &d, 2);
        const tqllm::Upload up = [](const void * h, size_t n) { void * p = nullptr; cudaMalloc(&p, n); cudaMemcpy(p, h, n, cudaMemcpyHostToDevice); return static_cast<const void *>(p); };
        const auto m = tqllm::weight_format(cs.ty)->load(raw, rows * slots, cols, up, nullptr);
        const int fmt = m.fmt->u4h_id();
        std::vector<__half> x(size_t(ntok) * cols);
        for (auto & v : x) v = __float2half(0.01f * float(int(g() % 200) - 100));
        __half * dx; cudaMalloc(&dx, x.size() * 2); cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        for (int G : {24, 96, 200}) {
            std::vector<int> map(size_t(G) * DS, -1);
            for (int q = 0; q < G; ++q) {
                int * e = map.data() + size_t(q) * DS; e[0] = (q * 37) % slots; e[1] = C;
                for (int c = 0; c < C; ++c) { e[2 + c] = (q + c) % ntok; e[2 + tqllm_cuda::kDedupCols + c] = q * C + c; }
            }
            int * dm; cudaMalloc(&dm, map.size() * 4); cudaMemcpy(dm, map.data(), map.size() * 4, cudaMemcpyHostToDevice);
            const size_t ny = size_t(G) * C * rows;
            __nv_bfloat16 * a1, * b1, * a2, * b2;
            cudaMalloc(&a1, ny * 2); cudaMalloc(&b1, ny * 2); cudaMalloc(&a2, ny * 2); cudaMalloc(&b2, ny * 2);
            tqllm_cuda::set_dedup_pf_min(fmt, 1);  // always prefetch
            m.fmt->expert_matvec_dedup2(m, m, dx, a1, b1, rows, dm, G, nullptr);
            tqllm_cuda::set_dedup_pf_min(fmt, INT32_MAX);  // never
            m.fmt->expert_matvec_dedup2(m, m, dx, a2, b2, rows, dm, G, nullptr);
            cudaDeviceSynchronize();
            std::vector<char> h1(ny * 2), h2(ny * 2), h3(ny * 2), h4(ny * 2);
            cudaMemcpy(h1.data(), a1, ny * 2, cudaMemcpyDeviceToHost); cudaMemcpy(h2.data(), a2, ny * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(h3.data(), b1, ny * 2, cudaMemcpyDeviceToHost); cudaMemcpy(h4.data(), b2, ny * 2, cudaMemcpyDeviceToHost);
            if (h1 != h2 || h3 != h4) { std::printf("FAIL: %s, %d groups: prefetching variant differs\n", cs.what, G); ++bad; }
            cudaFree(dm); cudaFree(a1); cudaFree(b1); cudaFree(a2); cudaFree(b2);
        }
        const int t = tqllm_cuda::calibrate_dedup_pf(fmt, static_cast<const uint8_t *>(m.p), m.aux, static_cast<const uint8_t *>(m.p), m.aux,
                                                     rows, cols, slots, nullptr);
        if (t != INT32_MAX && (t < 16 || t > slots)) { std::printf("FAIL: %s calibration returned %d\n", cs.what, t); ++bad; }
        std::printf("dedup pf %s: calibrated threshold %s\n", cs.what, t == INT32_MAX ? "never" : std::to_string(t).c_str());
        cudaFree(dx);
    }
    if (cudaGetLastError() != cudaSuccess) { std::puts("FAIL: CUDA error"); ++bad; }
    std::puts(bad ? "dedup pf switch: FAIL" : "dedup pf switch: PASS (prefetch on / off bit-identical; calibration in range)");
    return bad ? 1 : 0;
}
