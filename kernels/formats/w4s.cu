// W4S weight format kernels (see src/formats/q4.cpp).
#include "common.cuh"
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"
#include "quant_kernels.cuh"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace tqllm_cuda {
using namespace tqllm_dev;

// W4S: symmetric 4-bit (u4h code layout) + one FP16 scale per 128 weights, w = (q - 8) s (4.125 bits/weight, the
// W4A16 group-128 format). Same CTA / quad / k-permutation scheme as u4h_matvec_k; a warp covers MT 16-row tiles so
// every activation fragment it loads feeds MT MMAs.
template <int NCOL, int MT, class OUT = __nv_bfloat16>
__global__ void __launch_bounds__(256) w4s_matvec_k(const uint8_t * w4, const __half * sc, const __half * xh, OUT * y,
                                                    int rows, int k) {
    __shared__ float red[8][32][4 * MT];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT;
    const uint32_t NEG1024 = 0xE400E400u;
    const int nb = k / 32, ng = k / 128;
    float c[MT][4] = {};
    for (int s0 = warp * 4; s0 < nb; s0 += 32) {
        const int s = s0 + t;
        const bool ok = s < nb;
        uint4 x[4] = {};
        if (ok && g < NCOL) {
            const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g) * k + int64_t(s) * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) x[i] = __ldg(&xp[i]);
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int ra = min(row0 + m * 16 + g, rows - 1), rb = min(row0 + m * 16 + g + 8, rows - 1);
            uint4 wa = make_uint4(0, 0, 0, 0), wb = make_uint4(0, 0, 0, 0);
            uint32_t sa = 0, ma = 0, sb = 0, mb = 0;
            if (ok) {
                wa = __ldg(reinterpret_cast<const uint4 *>(w4 + int64_t(ra) * (k / 2) + int64_t(s) * 16));
                wb = __ldg(reinterpret_cast<const uint4 *>(w4 + int64_t(rb) * (k / 2) + int64_t(s) * 16));
                const __half2 sa2 = __half2half2(sc[int64_t(ra) * ng + s / 4]), sb2 = __half2half2(sc[int64_t(rb) * ng + s / 4]);
                const __half2 ma2 = __hmul2(sa2, __float2half2_rn(-8.f)), mb2 = __hmul2(sb2, __float2half2_rn(-8.f));
                sa = *reinterpret_cast<const uint32_t *>(&sa2);
                ma = *reinterpret_cast<const uint32_t *>(&ma2);
                sb = *reinterpret_cast<const uint32_t *>(&sb2);
                mb = *reinterpret_cast<const uint32_t *>(&mb2);
            }
            auto deq = [&](uint32_t w, uint32_t sel, uint32_t su, uint32_t mu) {
                return hfma2u(hadd2u(__byte_perm(w, 0x64646464u, sel), NEG1024), su, mu);
            };
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t a0 = h ? wa.z : wa.x, a1 = h ? wa.w : wa.y, b0 = h ? wb.z : wb.x, b1 = h ? wb.w : wb.y;
                const uint32_t wa4[4] = {a0 & 0x0F0F0F0Fu, (a0 >> 4) & 0x0F0F0F0Fu, a1 & 0x0F0F0F0Fu, (a1 >> 4) & 0x0F0F0F0Fu};
                const uint32_t wb4[4] = {b0 & 0x0F0F0F0Fu, (b0 >> 4) & 0x0F0F0F0Fu, b1 & 0x0F0F0F0Fu, (b1 >> 4) & 0x0F0F0F0Fu};
                const uint4 xa = x[2 * h], xb = x[2 * h + 1];
                const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    mma_f16(c[m], deq(wa4[u], 0x5140, sa, ma), deq(wb4[u], 0x5140, sb, mb), deq(wa4[u], 0x5342, sa, ma),
                            deq(wb4[u], 0x5342, sb, mb), xw[2 * u], xw[2 * u + 1]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][lane][4 * m + i] = c[m][i];
    __syncthreads();
    if (warp < MT) {  // warp m reduces tile m
        const int m = warp;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * m + i];
        const int n0 = 2 * t, n1 = 2 * t + 1, r = row0 + m * 16 + g;
        auto put = [&](int64_t i, float v2) {
            if constexpr (std::is_same_v<OUT, float>) y[i] = v2;
            else y[i] = __float2bfloat16_rn(v2);
        };
        if (r < rows) {
            if (n0 < NCOL) put(int64_t(n0) * rows + r, v[0]);
            if (n1 < NCOL) put(int64_t(n1) * rows + r, v[1]);
        }
        if (r + 8 < rows) {
            if (n0 < NCOL) put(int64_t(n0) * rows + r + 8, v[2]);
            if (n1 < NCOL) put(int64_t(n1) * rows + r + 8, v[3]);
        }
    }
}

__global__ void w4s_dequant_h_k(const uint8_t * w4, const __half * sc, __half * out, int64_t ngroups) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // 16-weight group
    if (g >= ngroups) return;
    const uint2 raw = reinterpret_cast<const uint2 *>(w4)[g];
    const uint8_t * b = reinterpret_cast<const uint8_t *>(&raw);
    const __half s = sc[g / 8], m = __hmul(s, __float2half(-8.f));
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        v[e] = __hfma(__int2half_rn((b[(u / 2) * 4 + j] >> ((u % 2) * 4)) & 0xF), s, m);
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

template <class OUT>
static void w4s_launch(const uint8_t * w4, const __half * sc, const __half * xh, OUT * y, int rows, int k, int ncols,
                       cudaStream_t s) {
    if (k % 128) throw std::invalid_argument("w4s_matvec: k must be a multiple of 128");
    constexpr int MT = 2;
    const int grid = (rows + 16 * MT - 1) / (16 * MT);
    switch (ncols) {
        case 1: w4s_matvec_k<1, MT, OUT><<<grid, 256, 0, s>>>(w4, sc, xh, y, rows, k); break;
        case 2: w4s_matvec_k<2, MT, OUT><<<grid, 256, 0, s>>>(w4, sc, xh, y, rows, k); break;
        case 3: w4s_matvec_k<3, MT, OUT><<<grid, 256, 0, s>>>(w4, sc, xh, y, rows, k); break;
        case 4: w4s_matvec_k<4, MT, OUT><<<grid, 256, 0, s>>>(w4, sc, xh, y, rows, k); break;
        case 5: w4s_matvec_k<5, MT, OUT><<<grid, 256, 0, s>>>(w4, sc, xh, y, rows, k); break;
        case 6: w4s_matvec_k<6, MT, OUT><<<grid, 256, 0, s>>>(w4, sc, xh, y, rows, k); break;
        case 7: w4s_matvec_k<7, MT, OUT><<<grid, 256, 0, s>>>(w4, sc, xh, y, rows, k); break;
        case 8: w4s_matvec_k<8, MT, OUT><<<grid, 256, 0, s>>>(w4, sc, xh, y, rows, k); break;
        default: throw std::invalid_argument("w4s_matvec: 1..8 columns");
    }
}

void w4s_matvec_bf16(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s) {
    w4s_launch(w4, sc, xh, y, rows, k, ncols, s);
}

void w4s_matvec(const uint8_t * w4, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    w4s_launch(w4, sc, xh, y, rows, k, ncols, s);
}

void w4s_dequant_h(const uint8_t * w4, const __half * sc, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    w4s_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w4, sc, out, ng);
}

}  // namespace tqllm_cuda
