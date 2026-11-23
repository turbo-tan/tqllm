// FP8T matrices: E4M3 weights with one F32 scale per row (modelopt FP8 checkpoints carry one per tensor; a row-concatenated
// q | k | v or qkv | z keeps each part's own), w = e4m3 * scale. Device layout: rows of k E4M3 bytes; aux = F32 [rows].
//
// matvec: tensor cores (mma m16n8k16, FP16 in, FP32 accumulate). E4M3 -> FP16 is exact and FP16 x FP16 products are exact
// in FP32, so the only roundings are the FP32 sums and the final scale. A CTA owns 16 rows; its KS warps split K in 64-wide
// chunks (warp w: chunks w, w + KS, ...), and the partial sums meet in shared memory in warp order. One 8-column tile or
// two: a column's bits depend only on its own inputs and the matrix (KS follows the row count), never on how many columns
// the launch holds, so a batched verify equals one column per launch.
// Inside a 64-chunk, lane (g, t) holds row g and g + 8, bytes [16 t, 16 t + 16); mma step s takes bytes 4 s .. 4 s + 3 of them
// as its k slots (2 t, 2 t + 1, 2 t + 8, 2 t + 9), and the activations of column g are read with the same map.
#include "quant_kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <stdexcept>

namespace {
__device__ __forceinline__ uint32_t cvt2(uint32_t two) {  // two E4M3 bytes (low 16 bits) -> FP16x2, low byte -> low half
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(two & 0xFFFFu), __NV_E4M3);
    return uint32_t(h.x) | (uint32_t(h.y) << 16);
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int NT, int KS, bool F32>
__global__ void __launch_bounds__(32 * KS) fp8t_matvec_k(const uint8_t * __restrict__ w, const float * __restrict__ scale,
                                                          const __half * __restrict__ x, void * y, int rows, int k, int n, int ldy) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int r0 = blockIdx.x * 16;
    const int ra = r0 + g, rb = r0 + g + 8;
    const uint8_t * wa = w + size_t(min(ra, rows - 1)) * k + 16 * t;
    const uint8_t * wb = w + size_t(min(rb, rows - 1)) * k + 16 * t;
    const bool oka = ra < rows, okb = rb < rows;
    const __half * xc[NT];
    bool okx[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int col = 8 * j + g;
        okx[j] = col < n;
        xc[j] = x + size_t(okx[j] ? col : 0) * k + 16 * t;
    }
    float acc[NT][4] = {};
    const int nch = k / 64;
#pragma unroll 2
    for (int ch = warp; ch < nch; ch += KS) {
        const int off = ch * 64;
        uint4 A = oka ? __ldg(reinterpret_cast<const uint4 *>(wa + off)) : make_uint4(0, 0, 0, 0);
        uint4 B = okb ? __ldg(reinterpret_cast<const uint4 *>(wb + off)) : make_uint4(0, 0, 0, 0);
        uint4 X[NT][2];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            X[j][0] = okx[j] ? __ldg(reinterpret_cast<const uint4 *>(xc[j] + off)) : make_uint4(0, 0, 0, 0);
            X[j][1] = okx[j] ? __ldg(reinterpret_cast<const uint4 *>(xc[j] + off + 8)) : make_uint4(0, 0, 0, 0);
        }
        const uint32_t aw[4] = {A.x, A.y, A.z, A.w}, bw[4] = {B.x, B.y, B.z, B.w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const uint32_t a0 = cvt2(aw[s]), a2 = cvt2(aw[s] >> 16), a1 = cvt2(bw[s]), a3 = cvt2(bw[s] >> 16);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint4 & xv = X[j][s / 2];
                const uint32_t b0 = (s & 1) ? xv.z : xv.x, b1 = (s & 1) ? xv.w : xv.y;
                mma16816(acc[j], a0, a1, a2, a3, b0, b1);
            }
        }
    }
    // partial sums of the KS warps, added in warp order
    __shared__ float part[KS][NT][16][8];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        part[warp][j][g][2 * t] = acc[j][0];
        part[warp][j][g][2 * t + 1] = acc[j][1];
        part[warp][j][g + 8][2 * t] = acc[j][2];
        part[warp][j][g + 8][2 * t + 1] = acc[j][3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NT * 128; i += 32 * KS) {
        const int j = i / 128, r = (i / 8) % 16, c = i % 8, col = 8 * j + c, row = r0 + r;
        if (row >= rows || col >= n) continue;
        float v = part[0][j][r][c];
#pragma unroll
        for (int q = 1; q < KS; ++q) v += part[q][j][r][c];
        v *= __ldg(scale + row);
        if constexpr (F32) static_cast<float *>(y)[size_t(col) * ldy + row] = v;
        else static_cast<__nv_bfloat16 *>(y)[size_t(col) * ldy + row] = __float2bfloat16(v);
    }
}

__global__ void fp8t_dequant_h_k(const uint8_t * __restrict__ w, const float * __restrict__ scale, __half * out, size_t n, int k) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const __half_raw h = __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t(w[i]), __NV_E4M3);
        out[i] = __float2half(__half2float(__half(h)) * scale[i / size_t(k)]);
    }
}

template <int NT, bool F32>
void launch_nt(const uint8_t * w, const float * sc, const __half * x, void * y, int rows, int k, int n, cudaStream_t s, int ldy) {
    const dim3 grid((rows + 15) / 16);
    // fewer CTAs than ~4 per SM: split K wider so the matrix still keeps the memory system busy (fixed per matrix shape)
    if (rows <= 2048) fp8t_matvec_k<NT, 8, F32><<<grid, 256, 0, s>>>(w, sc, x, y, rows, k, n, ldy);
    else fp8t_matvec_k<NT, 4, F32><<<grid, 128, 0, s>>>(w, sc, x, y, rows, k, n, ldy);
}
}  // namespace

namespace tqllm_cuda {
int fp8t_split_k(int rows) { return rows <= 2048 ? 8 : 4; }
void fp8t_matvec(const void * w, const float * scale, const __half * x, void * y, bool f32_out, int rows, int k, int n,
                 cudaStream_t s, int ldy) {
    if (k % 64) throw std::invalid_argument("fp8t_matvec: columns must be a multiple of 64");
    if (n < 1 || n > 16) throw std::invalid_argument("fp8t_matvec: 1..16 columns");
    if (!ldy) ldy = rows;
    const auto * q = static_cast<const uint8_t *>(w);
    if (n <= 8) f32_out ? launch_nt<1, true>(q, scale, x, y, rows, k, n, s, ldy) : launch_nt<1, false>(q, scale, x, y, rows, k, n, s, ldy);
    else f32_out ? launch_nt<2, true>(q, scale, x, y, rows, k, n, s, ldy) : launch_nt<2, false>(q, scale, x, y, rows, k, n, s, ldy);
}
void fp8t_dequant_h(const void * w, const float * scale, __half * out, int rows, int k, cudaStream_t s) {
    const size_t n = size_t(rows) * k;
    fp8t_dequant_h_k<<<int(std::min<size_t>((n + 255) / 256, 4096)), 256, 0, s>>>(static_cast<const uint8_t *>(w), scale, out, n, k);
}
}  // namespace tqllm_cuda
