#include <cstdint>
#include "gsq_q2.cuh"
#include "common.cuh"
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace tqllm_cuda {
namespace {
// GSQ Q2_0: F16 scale, 16 bytes of adjacent 2-bit codes per 64 weights; value = scale * (code - 1).
__device__ float value(const unsigned char * data, int64_t element) {
    const auto * block = data + (element / 64) * 18;
    const int j = int(element % 64);
    const float scale = __half2float(*reinterpret_cast<const __half *>(block));
    return scale * (int((block[2 + j / 4] >> (2 * (j % 4))) & 3u) - 1);
}
// Warp per (row, assignment): each 64-weight block (18 bytes) is read by the whole warp, 2 weights per lane.
template<class T> __global__ void dot_warp(const unsigned char * weights, const __half * x, T * y,
                                          int rows, int cols, int experts, const int * ids, int topk, bool per_expert) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31, assignment = blockIdx.y;
    if (row >= rows) return;
    const int expert = ids ? ids[assignment] : 0;
    const int input = per_expert ? assignment : assignment / topk;
    float sum = 0.f;
    if (expert >= 0 && expert < experts) {
        const unsigned char * r = weights + (int64_t(expert) * rows + row) * (cols / 64) * 18;
        const __half2 * xv = reinterpret_cast<const __half2 *>(x + int64_t(input) * cols);
        for (int b = 0; b < cols / 64; ++b) {
            const unsigned char * block = r + b * 18;
            const float scale = __half2float(__ushort_as_half(uint16_t(block[0] | block[1] << 8)));
            const unsigned q = block[2 + lane / 2] >> (4 * (lane & 1));
            const float2 xx = __half22float2(xv[b * 32 + lane]);
            sum += scale * (int(q & 3u) - 1) * xx.x;
            sum += scale * (int((q >> 2) & 3u) - 1) * xx.y;
        }
    }
    for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
    if (!lane) {
        if (expert < 0 || expert >= experts) sum = nanf("");
        if constexpr (std::is_same_v<T, float>) y[int64_t(assignment) * rows + row] = sum;
        else y[int64_t(assignment) * rows + row] = __float2bfloat16_rn(sum);
    }
}
// Wide rows (cols >= 3072, MiniMax): warp per (row, assignment), 4 blocks per step: 8 lanes per block, each lane 2 code
// bytes (8 weights) and one 16-byte load of the 8 inputs. 4x the bytes per load of dot_warp, whose byte-wide reads ran a
// 3072 x 6144 expert at ~85 GB/s on a GB10. Its own summation order: not bit-identical to dot_warp.
template<class T> __global__ void dot_wide(const unsigned char * weights, const __half * x, T * y,
                                          int rows, int cols, int experts, const int * ids, int topk, bool per_expert) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31, assignment = blockIdx.y;
    if (row >= rows) return;
    const int expert = ids ? ids[assignment] : 0;
    const int input = per_expert ? assignment : assignment / topk;
    float sum = 0.f;
    if (expert >= 0 && expert < experts) {
        const int nb = cols / 64, sub = lane & 7, which = lane >> 3;
        const unsigned char * r = weights + (int64_t(expert) * rows + row) * nb * 18;
        const __half * xr = x + int64_t(input) * cols;
        for (int b = which; b < nb; b += 4) {
            const unsigned char * block = r + b * 18;
            const float scale = __half2float(*reinterpret_cast<const __half *>(block));
            const unsigned q = *reinterpret_cast<const uint16_t *>(block + 2 + sub * 2);
            const uint4 xv = *reinterpret_cast<const uint4 *>(xr + b * 64 + sub * 8);
            const __half2 * xh = reinterpret_cast<const __half2 *>(&xv);
            float part = 0.f;
            for (int i = 0; i < 4; ++i) {
                const float2 xx = __half22float2(xh[i]);
                part += (int((q >> (4 * i)) & 3u) - 1) * xx.x;
                part += (int((q >> (4 * i + 2)) & 3u) - 1) * xx.y;
            }
            sum += scale * part;
        }
    }
    for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
    if (!lane) {
        if (expert < 0 || expert >= experts) sum = nanf("");
        if constexpr (std::is_same_v<T, float>) y[int64_t(assignment) * rows + row] = sum;
        else y[int64_t(assignment) * rows + row] = __float2bfloat16_rn(sum);
    }
}
// Grouped experts (moe_group lists): warp per (row, expert, up to C of its tokens); the row's blocks are read once for
// all C tokens. Per token the products and their order are those of dot_warp (bit-identical).
// Grouped experts (moe_group lists): block = 64 rows (8 warps x 8 rows) of one expert; the expert's tokens C at a time
// are staged in shared memory once per block (not once per row), each warp then streams its rows' 2-bit blocks.
// Per (row, token) the products and their order are those of dot_warp.
template<int C> __global__ void __launch_bounds__(256) dot_grouped(const unsigned char * weights, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                                            const int * counts, const int * groups, int capacity, int topk, bool per_expert) {
    const int expert = blockIdx.y, count = counts[expert];
    if (count == 0) return;
    extern __shared__ __half2 xs[];  // [C][cols / 2]
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, half_cols = cols / 2;
    const int row0 = blockIdx.x * 64 + warp * 8;
    for (int c0 = 0; c0 < count; c0 += C) {
        const int nc = min(C, count - c0);
        const int * g = groups + int64_t(expert) * capacity + c0;
        __syncthreads();
        for (int i = threadIdx.x; i < nc * half_cols; i += blockDim.x) {
            const int c = i / half_cols, j = i % half_cols, a = g[c];
            xs[i] = reinterpret_cast<const __half2 *>(x + int64_t(per_expert ? a : a / topk) * cols)[j];
        }
        __syncthreads();
        for (int rr = 0; rr < 8; ++rr) {
            const int row = row0 + rr;
            if (row >= rows) break;
            const unsigned char * r = weights + (int64_t(expert) * rows + row) * (cols / 64) * 18;
            float sum[C];
#pragma unroll
            for (int c = 0; c < C; ++c) sum[c] = 0.f;
            for (int b = 0; b < cols / 64; ++b) {
                const unsigned char * block = r + b * 18;
                const float scale = __half2float(__ushort_as_half(uint16_t(block[0] | block[1] << 8)));
                const unsigned q = block[2 + lane / 2] >> (4 * (lane & 1));
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    if (c >= nc) break;
                    const float2 xx = __half22float2(xs[c * half_cols + b * 32 + lane]);
                    sum[c] += scale * (int(q & 3u) - 1) * xx.x;
                    sum[c] += scale * (int((q >> 2) & 3u) - 1) * xx.y;
                }
            }
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= nc) break;
                float v = sum[c];
                for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
                if (!lane) y[int64_t(g[c]) * rows + row] = __float2bfloat16_rn(v);
            }
        }
    }
}
// Grouped experts on tensor cores: CTA = 64 rows x 64 of one expert's tokens, warp w takes row tile w & 3 and k half
// w >> 2 (u4h_gemm_k's layout). Lane t of a warp owns one 32-weight half of a Q2_0 block per 256-k step; scale * (code - 1)
// is exact in FP16, so only the FP32 summation order differs from dot_grouped.
template<int NT>
__global__ void __launch_bounds__(256) q2_gemm_grouped(const unsigned char * weights, const __half * x, __nv_bfloat16 * y, int rows, int k,
                                                       const int * counts, const int * groups, int capacity, int topk, bool per_expert) {
    using namespace tqllm_dev;
    const int expert = blockIdx.z, ncols = counts[expert];
    if (int(blockIdx.y) * (NT*8) >= ncols) return;
    groups += int64_t(expert) * capacity;
    constexpr int XS = 33;
    __shared__ uint4 xt[NT*8 * XS];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int rt = warp & 3, ks = warp >> 2;
    const int row0 = blockIdx.x * 64 + rt * 16, col0 = blockIdx.y * (NT*8), nb = k / 32, row_bytes = k / 64 * 18;
    const unsigned char * wa = weights + (int64_t(expert) * rows + min(row0 + g, rows - 1)) * row_bytes;
    const unsigned char * wb = weights + (int64_t(expert) * rows + min(row0 + g + 8, rows - 1)) * row_bytes;
    float c[NT][4] = {};
    for (int base = 0; base < nb; base += 8) {
        __syncthreads();
        for (int i = threadIdx.x; i < NT*8 * 32; i += 256) {
            const int col = i / 32, q = i % 32, blk = base + q / 4;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (col0 + col < ncols && blk < nb) {
                const int a = groups[col0 + col];
                v = __ldg(reinterpret_cast<const uint4 *>(x + int64_t(per_expert ? a : a / topk) * k + int64_t(base) * 32) + q);
            }
            xt[col * XS + q] = v;
        }
        __syncthreads();
        const int s = base + ks * 4 + t;
        uint32_t af[8][4];
        if (s < nb) {
            const unsigned char * ba = wa + (s >> 1) * 18, * bb = wb + (s >> 1) * 18;
            const __half sa = __ushort_as_half(uint16_t(ba[0] | ba[1] << 8)), sb = __ushort_as_half(uint16_t(bb[0] | bb[1] << 8));
            const int off = 2 + (s & 1) * 8;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const unsigned qa = ba[off + u], qb = bb[off + u];
                const auto v = [](__half sc, unsigned q, int j) { return __hmul(sc, __int2half_rn(int((q >> (2 * j)) & 3u) - 1)); };
                const __half2 a01 = __halves2half2(v(sa, qa, 0), v(sa, qa, 1)), a23 = __halves2half2(v(sa, qa, 2), v(sa, qa, 3));
                const __half2 b01 = __halves2half2(v(sb, qb, 0), v(sb, qb, 1)), b23 = __halves2half2(v(sb, qb, 2), v(sb, qb, 3));
                af[u][0] = *reinterpret_cast<const uint32_t *>(&a01); af[u][1] = *reinterpret_cast<const uint32_t *>(&b01);
                af[u][2] = *reinterpret_cast<const uint32_t *>(&a23); af[u][3] = *reinterpret_cast<const uint32_t *>(&b23);
            }
        } else {
#pragma unroll
            for (int u = 0; u < 8; ++u) af[u][0] = af[u][1] = af[u][2] = af[u][3] = 0u;
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const uint4 * xp = xt + (nt * 8 + g) * XS + (ks * 4 + t) * 4;
            uint4 x4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) x4[i] = xp[i];
            const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
#pragma unroll
            for (int u = 0; u < 8; ++u) mma_f16(c[nt], af[u][0], af[u][1], af[u][2], af[u][3], xw[2 * u], xw[2 * u + 1]);
        }
    }
    __syncthreads();
    float (*red)[32][NT * 4] = reinterpret_cast<float (*)[32][NT * 4]>(xt);
    if (ks == 1)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[rt][lane][nt * 4 + i] = c[nt][i];
    __syncthreads();
    if (ks == 0) {
        const int r = row0 + g;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int n0 = col0 + nt * 8 + 2 * t, n1 = n0 + 1;
            const float v[4] = {c[nt][0] + red[rt][lane][nt * 4 + 0], c[nt][1] + red[rt][lane][nt * 4 + 1],
                                c[nt][2] + red[rt][lane][nt * 4 + 2], c[nt][3] + red[rt][lane][nt * 4 + 3]};
            const auto put = [&](int col, int row, float value) { if (col < ncols && row < rows) y[int64_t(groups[col]) * rows + row] = __float2bfloat16_rn(value); };
            put(n0, r, v[0]); put(n1, r, v[1]); put(n0, r + 8, v[2]); put(n1, r + 8, v[3]);
        }
    }
}
__global__ void decode(const unsigned char * weights, __half * out, int64_t count) {
    const int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < count) out[j] = __float2half_rn(value(weights, j));
}
void shape(const void * w, const void * x, const void * y, int rows, int cols, int n, int topk = 1) {
    if (!w || !x || !y || rows <= 0 || cols <= 0 || cols % 64 || n <= 0 || topk <= 0 || int64_t(n) * topk > 65535)
        throw std::invalid_argument("GSQ Q2_0: invalid pointer, shape or grid");
}
}
void gsq_q2_matvec(const void * w, const __half * x, __nv_bfloat16 * y, int r, int k, int n, cudaStream_t s) {
    shape(w, x, y, r, k, n);
    dot_warp<<<dim3(unsigned((r + 7) / 8), unsigned(n)), 256, 0, s>>>(static_cast<const unsigned char *>(w), x, y, r, k, 1, nullptr, 1, false);
}
void gsq_q2_matvec_f32(const void * w, const __half * x, float * y, int r, int k, int n, cudaStream_t s) {
    shape(w, x, y, r, k, n);
    dot_warp<<<dim3(unsigned((r + 7) / 8), unsigned(n)), 256, 0, s>>>(static_cast<const unsigned char *>(w), x, y, r, k, 1, nullptr, 1, false);
}
void gsq_q2_expert_matvec(const void * w, const __half * x, __nv_bfloat16 * y, int r, int k,
                          int experts, const int * ids, int n, int topk, bool per_expert, cudaStream_t s) {
    shape(w, x, y, r, k, n, topk);
    if (!ids || experts <= 0 || int64_t(experts) * r > INT_MAX)
        throw std::invalid_argument("GSQ Q2_0: invalid expert dimensions");
    if (k >= 3072)
        dot_wide<<<dim3(unsigned((r + 7) / 8), unsigned(n * topk)), 256, 0, s>>>(static_cast<const unsigned char *>(w), x, y, r, k, experts, ids, topk, per_expert);
    else
        dot_warp<<<dim3(unsigned((r + 7) / 8), unsigned(n * topk)), 256, 0, s>>>(static_cast<const unsigned char *>(w), x, y, r, k, experts, ids, topk, per_expert);
}
void gsq_q2_expert_grouped(const void * w, const __half * x, __nv_bfloat16 * y, int r, int k, const int * counts, const int * groups,
                           int capacity, int experts, int topk, bool per_expert, cudaStream_t s) {
    shape(w, x, y, r, k, 1, 1);
    if (!counts || !groups || experts <= 0 || experts > 65535 || capacity <= 0 || int64_t(experts) * r > INT_MAX)
        throw std::invalid_argument("GSQ Q2_0: invalid grouped expert dimensions");
    // TQFN_Q2_DOT_GROUPED=1: the scalar kernel (A/B); default: tensor cores when k is a multiple of 64
    static const bool scalar = std::getenv("TQFN_Q2_DOT_GROUPED") && std::atoi(std::getenv("TQFN_Q2_DOT_GROUPED")) != 0;
    if (!scalar && capacity <= 65535 * 64) {
        if(capacity<=8)q2_gemm_grouped<1><<<dim3(unsigned((r+63)/64),1,unsigned(experts)),256,0,s>>>(
            static_cast<const unsigned char *>(w),x,y,r,k,counts,groups,capacity,topk,per_expert);
        else q2_gemm_grouped<8><<<dim3(unsigned((r + 63) / 64), unsigned((capacity + 63) / 64), unsigned(experts)), 256, 0, s>>>(
            static_cast<const unsigned char *>(w), x, y, r, k, counts, groups, capacity, topk, per_expert);
        return;
    }
    constexpr int C = 16;
    const size_t smem = size_t(C) * k * sizeof(__half);
    static int attr_for = 0;
    if (smem > 48 * 1024 && attr_for < int(smem)) { cudaFuncSetAttribute(dot_grouped<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)); attr_for = int(smem); }
    dot_grouped<C><<<dim3(unsigned((r + 63) / 64), unsigned(experts)), 256, smem, s>>>(
        static_cast<const unsigned char *>(w), x, y, r, k, counts, groups, capacity, topk, per_expert);
}
void gsq_q2_dequant(const void * w, __half * out, int r, int k, cudaStream_t s) {
    shape(w, out, out, r, k, 1);
    const int64_t count = int64_t(r) * k;
    if ((count + 255) / 256 > INT_MAX) throw std::invalid_argument("GSQ Q2_0: dequant grid overflow");
    decode<<<unsigned((count + 255) / 256), 256, 0, s>>>(static_cast<const unsigned char *>(w), out, count);
}
}
