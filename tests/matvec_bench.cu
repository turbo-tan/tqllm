// Bench only (no tests): n-column matvec of TQ3_4S (GGUF 46), TQ4 (300), TQ3 (301), IQ4_XS (23), IQ3_S (21), IQ3_XXS (18) and
// Q4_K (12) at the two FFN shapes, for timing and ncu.
// usage: matvec-bench [iters=50] [ncols=8] [tq4_variant=0] [rowscale=1] [type=0 (all)]
#include "formats/weight_format.hpp"
#include "quant_kernels.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
// Read ceiling: every thread streams 16-byte loads over the buffer (grid-stride, 4 loads in flight), sums into one word.
__global__ void read_k(const uint4 * __restrict__ p, size_t n, unsigned * out) {
    unsigned acc = 0;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        const uint4 a = __ldg(p + i), b = __ldg(p + i + stride), c = __ldg(p + i + 2 * stride), d = __ldg(p + i + 3 * stride);
        acc ^= a.x ^ b.y ^ c.z ^ d.w;
    }
    for (; i < n; i += stride) acc ^= __ldg(p + i).x;
    if (acc == 0x12345678u) *out = acc;
}
int main(int argc, char ** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 50, nc = argc > 2 ? std::atoi(argv[2]) : 8, var = argc > 3 ? std::atoi(argv[3]) : 0;
    const int rs = argc > 4 ? std::atoi(argv[4]) : 1;
    const unsigned only = argc > 5 ? unsigned(std::atoi(argv[5])) : 0;  // 0: all types, else just this GGUF type  // row scale: bigger matrices separate fixed per-launch cost from per-byte cost
    std::mt19937 g(7);
    for (auto [rows0, cols] : {std::pair{17408, 5120}, std::pair{5120, 17408}}) {
        const int rows = rows0 * rs;
        for (auto [ty, bb] : {std::pair{46u, 128}, std::pair{300u, 132}, std::pair{301u, 100}, std::pair{23u, 136}, std::pair{21u, 110},
                              std::pair{18u, 98}, std::pair{12u, 144}}) {
            if (only && ty != only) continue;
            std::vector<char> raw(size_t(rows) * cols / 256 * bb);
            for (auto & c : raw) c = char(g() & 255);
            for (size_t b = 0; b < raw.size() / bb; ++b) {  // sane scales: TQ4 fp16 scale at offset 0 of each 128-group pair; TQ3_4S u8 scales anywhere
                const __half d = __float2half(0.001f);
                if (ty == 300) std::memcpy(raw.data() + b * bb, &d, 2), std::memcpy(raw.data() + b * bb + 66, &d, 2);
                if (ty == 301) std::memcpy(raw.data() + b * bb, &d, 2), std::memcpy(raw.data() + b * bb + 50, &d, 2);
                if (ty == 23 || ty == 21 || ty == 18 || ty == 12) std::memcpy(raw.data() + b * bb, &d, 2);  // ggml super-block d at offset 0
            }
            const tqllm::Upload up = [](const void * h, size_t n) { void * p = nullptr; cudaMalloc(&p, n); cudaMemcpy(p, h, n, cudaMemcpyHostToDevice); return static_cast<const void *>(p); };
            const auto m = tqllm::weight_format(ty)->load(raw, rows, cols, up, nullptr);
            std::vector<__half> x(size_t(nc) * cols, __float2half(0.01f));
            __half * dx = nullptr; cudaMalloc(&dx, x.size() * 2); cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
            __nv_bfloat16 * dy = nullptr; cudaMalloc(&dy, size_t(nc) * rows * 2);
            auto call = [&] {
                if (ty == 300 && (var == 10 || var == 11)) tqllm_cuda::tq4_matvec_i8_bf16(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), dx, dy, rows, cols, nc, nullptr, var == 11 ? 1 : 2);
                else if (ty == 301 && var) tqllm_cuda::tq4b3_matvec_variant(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), dx, dy, rows, cols, nc, var, nullptr);
                else if (ty == 300 && var) tqllm_cuda::tq4_matvec_variant(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), dx, dy, rows, cols, nc, var, nullptr);
                else m.fmt->matvec(m, dx, dy, nc, nullptr);
            };
            for (int i = 0; i < 10; ++i) call();
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i) call();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            const double gb = double(raw.size()) * iters / (ms / 1000) / 1e9;
            std::printf("bench %-6s %5d x %5d nc=%d v=%d: %7.1f us  %6.0f GB/s\n", m.fmt->name(), rows, cols, nc, ty == 300 ? var : 0, ms * 1000.f / iters, gb);
            if (ty == 300 && var == 10) {  // numeric difference (MT 2; MT 1 shares the arithmetic) vs the FP16 path (random weights, constant activations here)
                std::vector<__half> xr(size_t(nc) * cols);
                std::mt19937 g2(11); std::normal_distribution<float> nd(0.f, 1.f);
                for (auto & v : xr) v = __float2half(nd(g2));
                cudaMemcpy(dx, xr.data(), xr.size() * 2, cudaMemcpyHostToDevice);
                __nv_bfloat16 * dy0 = nullptr; cudaMalloc(&dy0, size_t(nc) * rows * 2);
                m.fmt->matvec(m, dx, dy0, nc, nullptr);
                tqllm_cuda::tq4_matvec_i8_bf16(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), dx, dy, rows, cols, nc, nullptr);
                std::vector<__nv_bfloat16> a(size_t(nc) * rows), bv(a.size());
                cudaMemcpy(a.data(), dy0, a.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(bv.data(), dy, bv.size() * 2, cudaMemcpyDeviceToHost);
                double se = 0, sr = 0, mx = 0;
                for (size_t i = 0; i < a.size(); ++i) { const double r = __bfloat162float(a[i]), q = __bfloat162float(bv[i]); se += (r - q) * (r - q); sr += r * r; mx = std::max(mx, std::fabs(r - q)); }
                std::printf("   i8 vs fp16 path: rel rms %.3e  max abs %.3e\n", std::sqrt(se / sr), mx);
                cudaFree(dy0);
            }
            if (ty == 46) {  // read ceiling on the same bytes
                unsigned * dout = nullptr; cudaMalloc(&dout, 4);
                const size_t n16 = raw.size() / 16;
                for (int blocks : {272, 544, 1088, 2176}) {
                    for (int i = 0; i < 10; ++i) read_k<<<blocks, 256>>>(static_cast<const uint4 *>(m.p), n16, dout);
                    cudaEventRecord(e0);
                    for (int i = 0; i < iters; ++i) read_k<<<blocks, 256>>>(static_cast<const uint4 *>(m.p), n16, dout);
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float rms = 0; cudaEventElapsedTime(&rms, e0, e1);
                    std::printf("ceiling read-only %5d x %5d blocks=%d: %7.1f us  %6.0f GB/s\n", rows, cols, blocks, rms * 1000.f / iters, double(raw.size()) * iters / (rms / 1000) / 1e9);
                }
                cudaFree(dout);
            }
            cudaFree(dx); cudaFree(dy);
        }
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
