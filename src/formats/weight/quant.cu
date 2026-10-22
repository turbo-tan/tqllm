#include <cstdint>
#include <cstring>
#include "debug.hpp"
#include <cstdio>
#include <utility>
#include <map>
#include <cstdlib>
#include "quant_kernels.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <climits>
#include <math_constants.h>
#include <stdexcept>
#include <type_traits>

namespace {

using namespace tqllm_cuda;

// 8-level codebook for N(0,1) and the RHT sign pattern (llama.cpp-tq3 TQ3_0 definitions).
__constant__ float TQ3_SIGNS[32] = {
    +1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1,
    -1, -1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1,
};
__device__ __constant__ float TQ3_CODEBOOK[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f,
                                                 0.230106f,  0.725222f,  1.277503f,  1.988943f};

// E3M5: 2^((b >> 5) - 9) * (1 + (b & 31) / 32), 0 for b == 0. Built directly as FP32 bits.
__device__ __forceinline__ float e3m5(uint32_t b) {
    return b ? __uint_as_float((((b >> 5) + 118u) << 23) | ((b & 31u) << 18)) : 0.f;
}

__device__ __forceinline__ float wht32(float v, int lane) {
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, s);
        v = (lane & s) ? o - v : v + o;
    }
    return v * 0.17677669529663687f;  // 1/sqrt(32)
}

// One warp per 32-element block: lane = element.
__global__ void tq3_rotate_k(const __nv_bfloat16 * x, float * xr, int total_blocks) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= total_blocks) return;
    const int64_t i = int64_t(warp) * 32 + lane;
    xr[i] = wht32(__bfloat162float(x[i]) * TQ3_SIGNS[lane], lane);
}

__global__ void tq3_rotate_f32_k(const float * x, float * xr, int total_blocks) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= total_blocks) return;
    const int64_t i = int64_t(warp) * 32 + lane;
    xr[i] = wht32(x[i] * TQ3_SIGNS[lane], lane);
}

template<bool Repacked>
__global__ void tq3_matvec_f32_raw_k(const uint4 * weights, const float * xr, float * y, int rows, int k) {
    const int row = blockIdx.x, token = blockIdx.y, blocks = k / 32;
    const float * x = xr + int64_t(token) * k;
    float sum = 0.f;
    for (int b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint4 q = weights[int64_t(row) * blocks + b];
        const uint32_t groups[4] = {q.y, (q.y >> 24) | (q.z << 8), (q.z >> 16) | (q.w << 16), q.w >> 8};
        for (int g = 0; g < 4; ++g) {
            float part = 0.f;
            for (int j = 0; j < 8; ++j) {
                uint32_t index;
                if constexpr (Repacked) {
                    const uint32_t words[3] = {q.y, q.z, q.w};
                    index = g < 3 ? (words[g] >> (4 * j)) & 7u :
                        ((q.y >> (4 * j + 3)) & 1u) | (((q.z >> (4 * j + 3)) & 1u) << 1) |
                        (((q.w >> (4 * j + 3)) & 1u) << 2);
                } else index = (groups[g] >> (3 * j)) & 7u;
                part += TQ3_CODEBOOK[index] * x[b * 32 + g * 8 + j];
            }
            sum += e3m5((q.x >> (8 * g)) & 255) * part;
        }
    }
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
    __shared__ float partial[8];
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0.f;
        for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (threadIdx.x == 0) y[int64_t(token) * rows + row] = sum;
    }
}

__global__ void tq3_row_k(const uint4 * w, int row, __nv_bfloat16 * y, int nblk, const uint32_t * dtok = nullptr) {
    const int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (b >= nblk) return;
    if (dtok) {
        row = int(dtok[blockIdx.y]);
        y += int64_t(blockIdx.y) * nblk * 32;
    }
    const uint4 blk = w[int64_t(row) * nblk + b];
    const int g = lane / 8, j = lane % 8;
    const uint32_t scales = blk.x;
    uint32_t bits;
    if (g == 0) bits = blk.y;
    else if (g == 1) bits = (blk.y >> 24) | (blk.z << 8);
    else if (g == 2) bits = (blk.z >> 16) | (blk.w << 16);
    else bits = blk.w >> 8;
    const float v = TQ3_CODEBOOK[(bits >> (3 * j)) & 7] * e3m5((scales >> (8 * g)) & 255);
    y[int64_t(b) * 32 + lane] = __float2bfloat16_rn(wht32(v, lane) * TQ3_SIGNS[lane]);
}

// Block: 8 warps x RPW rows. K is processed in tiles of 32 blocks (1024 weights); the N activation columns of a
// tile sit in shared memory transposed ([n][element-in-block][block]) so lane l (= block l of the tile) reads
// conflict-free. Each lane decodes its weight block once and applies it to all N columns.
template <int N, int RPW>
__global__ void __launch_bounds__(256) tq3_matvec_k(const uint4 * w, const float * xr, __nv_bfloat16 * y, int rows,
                                                    int k) {
    __shared__ float xs[N][32][33];
    __shared__ float cb[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    const int nblk = k / 32, ntile = (nblk + 31) / 32;
    if (threadIdx.x < 8) cb[threadIdx.x] = TQ3_CODEBOOK[threadIdx.x];

    float acc[RPW][N];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int n = 0; n < N; ++n) acc[r][n] = 0.f;

    for (int t = 0; t < ntile; ++t) {
        const int b0 = t * 32, nb = min(32, nblk - b0);
        __syncthreads();
        for (int i = threadIdx.x; i < N * nb * 32; i += blockDim.x) {
            const int n = i / (nb * 32), rem = i % (nb * 32), bl = rem / 32, e = rem % 32;
            xs[n][e][bl] = xr[int64_t(n) * k + int64_t(b0 + bl) * 32 + e];
        }
        __syncthreads();
        if (lane >= nb) continue;
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = row0 + r;
            if (row >= rows) break;
            const uint4 blk = w[int64_t(row) * nblk + b0 + lane];
            const uint32_t g0 = blk.y, g1 = (blk.y >> 24) | (blk.z << 8), g2 = (blk.z >> 16) | (blk.w << 16),
                           g3 = blk.w >> 8;
            const uint32_t gb[4] = {g0, g1, g2, g3};
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const float d = e3m5((blk.x >> (8 * g)) & 255);
                float part[N];
#pragma unroll
                for (int n = 0; n < N; ++n) part[n] = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float c = cb[(gb[g] >> (3 * j)) & 7];
#pragma unroll
                    for (int n = 0; n < N; ++n) part[n] += c * xs[n][g * 8 + j][lane];
                }
#pragma unroll
                for (int n = 0; n < N; ++n) acc[r][n] += d * part[n];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int n = 0; n < N; ++n) {
            float v = acc[r][n];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && row < rows) y[int64_t(n) * rows + row] = __float2bfloat16_rn(v);
        }
    }
}


// Single-column fast path. Lane l of a warp walks blocks l, l+32, ... of its RPW rows. For each block it loads the
// 32 rotated activations into registers once (8 x float4, coalesced) and reuses them for all RPW rows, whose weight
// blocks are fetched up front to keep RPW 16-byte loads in flight per lane. No shared-memory tile, no block sync.
template <int RPW>
__global__ void __launch_bounds__(256) tq3_matvec1_k(const uint4 * w, const float * xr, __nv_bfloat16 * y, int rows,
                                                     int k) {
    __shared__ float cb[8];
    if (threadIdx.x < 8) cb[threadIdx.x] = TQ3_CODEBOOK[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    if (row0 >= rows) return;
    const int nblk = k / 32;
    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

    for (int b = lane; b < nblk; b += 32) {
        uint4 blk[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) blk[r] = row0 + r < rows ? __ldg(&w[int64_t(row0 + r) * nblk + b]) : make_uint4(0, 0, 0, 0);
        float xv[32];
        const float4 * xp = reinterpret_cast<const float4 *>(xr + int64_t(b) * 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 t = __ldg(&xp[i]);
            xv[4 * i] = t.x;
            xv[4 * i + 1] = t.y;
            xv[4 * i + 2] = t.z;
            xv[4 * i + 3] = t.w;
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const uint4 q = blk[r];
            const uint32_t gb[4] = {q.y, (q.y >> 24) | (q.z << 8), (q.z >> 16) | (q.w << 16), q.w >> 8};
            float s = 0.f;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) part += cb[(gb[g] >> (3 * j)) & 7] * xv[g * 8 + j];
                s += e3m5((q.x >> (8 * g)) & 255) * part;
            }
            acc[r] += s;
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && row0 + r < rows) y[row0 + r] = __float2bfloat16_rn(v);
    }
}


// ---- int8 fast path ----
// Repacked block (16 bytes): x = 4 E3M5 scales (unchanged); y,z,w = W0,W1,W2. Elements 0..23: low 3 bits of
// nibble (e % 8) of W[e / 8], i.e. already a valid PRMT byte selector after masking with 0x7777. Elements 24..31:
// bit t of index (24 + m) is bit 3 of nibble m of W[t]. Codebook lookup = PRMT on an int8 codebook, dot = DP4A.
constexpr float TQ3_CB_Q = 63.5f;  // int8 codebook = round(c * 63.5), |c| <= 1.997

__global__ void tq3_repack_k(uint4 * w, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint4 q = w[i];
    const uint32_t gb[4] = {q.y, (q.y >> 24) | (q.z << 8), (q.z >> 16) | (q.w << 16), q.w >> 8};
    uint32_t W[3] = {0, 0, 0};
    for (int e = 0; e < 24; ++e) W[e / 8] |= ((gb[e / 8] >> (3 * (e % 8))) & 7u) << (4 * (e % 8));
    for (int m = 0; m < 8; ++m) {
        const uint32_t idx = (gb[3] >> (3 * m)) & 7u;
        for (int t = 0; t < 3; ++t) W[t] |= ((idx >> t) & 1u) << (4 * m + 3);
    }
    w[i] = make_uint4(q.x, W[0], W[1], W[2]);
}

// One warp per 32-element block: rotate (sign flip + WHT), then max-abs int8 quantisation.
__global__ void tq3_rotate_i8_k(const __nv_bfloat16 * x, int8_t * xq, float * xs, int nblk) {
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (b >= nblk) return;
    const int64_t i = int64_t(b) * 32 + lane;
    const float v = wht32(__bfloat162float(x[i]) * TQ3_SIGNS[lane], lane);
    float m = fabsf(v);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float scale = m / 127.f;
    xq[i] = int8_t(m > 0.f ? __float2int_rn(v / scale) : 0);
    if (lane == 0) xs[b] = scale;
}

__device__ __forceinline__ float e3m5_byte(uint32_t b) {
    return b ? __uint_as_float((b << 18) + 0x3B000000u) : 0.f;
}

template <int RPW>
__global__ void __launch_bounds__(256) tq3_matvec_i8_k(const uint4 * w, const int4 * xq, const float * xs,
                                                       __nv_bfloat16 * y, int rows, int k, uint32_t cblo,
                                                       uint32_t cbhi) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    if (row0 >= rows) return;
    const int nblk = k / 32;
    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;
    for (int b = lane; b < nblk; b += 32) {
        uint4 blk[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r)
            blk[r] = row0 + r < rows ? __ldg(&w[int64_t(row0 + r) * nblk + b]) : make_uint4(0, 0, 0, 0);
        const int4 xa = __ldg(&xq[2 * b]), xb = __ldg(&xq[2 * b + 1]);
        const float sx = __ldg(&xs[b]);
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const uint4 q = blk[r];
            const uint32_t W0 = q.y, W1 = q.z, W2 = q.w;
            int s0 = __dp4a(int(__byte_perm(cblo, cbhi, W0 & 0x7777)), xa.x, 0);
            s0 = __dp4a(int(__byte_perm(cblo, cbhi, (W0 >> 16) & 0x7777)), xa.y, s0);
            int s1 = __dp4a(int(__byte_perm(cblo, cbhi, W1 & 0x7777)), xa.z, 0);
            s1 = __dp4a(int(__byte_perm(cblo, cbhi, (W1 >> 16) & 0x7777)), xa.w, s1);
            int s2 = __dp4a(int(__byte_perm(cblo, cbhi, W2 & 0x7777)), xb.x, 0);
            s2 = __dp4a(int(__byte_perm(cblo, cbhi, (W2 >> 16) & 0x7777)), xb.y, s2);
            const uint32_t sa = ((W0 >> 3) & 0x1111) | ((W1 >> 2) & 0x2222) | ((W2 >> 1) & 0x4444);
            const uint32_t sb = ((W0 >> 19) & 0x1111) | ((W1 >> 18) & 0x2222) | ((W2 >> 17) & 0x4444);
            int s3 = __dp4a(int(__byte_perm(cblo, cbhi, sa)), xb.z, 0);
            s3 = __dp4a(int(__byte_perm(cblo, cbhi, sb)), xb.w, s3);
            acc[r] += sx * (e3m5_byte(q.x & 255) * float(s0) + e3m5_byte((q.x >> 8) & 255) * float(s1) +
                            e3m5_byte((q.x >> 16) & 255) * float(s2) + e3m5_byte(q.x >> 24) * float(s3));
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && row0 + r < rows) y[row0 + r] = __float2bfloat16_rn(v * (1.f / TQ3_CB_Q));
    }
}

struct Tq3Codebook8 {
    uint32_t lo, hi;
};

Tq3Codebook8 tq3_codebook_i8() {
    static const float c[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f,
                               0.230106f,  0.725222f,  1.277503f,  1.988943f};
    Tq3Codebook8 cb{0, 0};
    for (int i = 0; i < 4; ++i) {
        cb.lo |= uint32_t(uint8_t(int8_t(lrintf(c[i] * TQ3_CB_Q)))) << (8 * i);
        cb.hi |= uint32_t(uint8_t(int8_t(lrintf(c[i + 4] * TQ3_CB_Q)))) << (8 * i);
    }
    return cb;
}


// ---- fused residual / norm / rotate / int8 ----
__device__ float block_sum256(float v) {
    __shared__ float sh[8];
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += sh[i];
    return t;
}

__global__ void add_sumsq_k(__nv_bfloat16 * x, const __nv_bfloat16 * y, float * partial, int k) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    float v = 0.f;
    if (i < k) {
        const __nv_bfloat16 r = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(y[i]));
        x[i] = r;
        v = __bfloat162float(r);
    }
    const float t = block_sum256(v * v);
    if (threadIdx.x == 0) partial[blockIdx.x] = t;
}

__global__ void sumsq_k(const __nv_bfloat16 * x, float * partial, int k) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    const float v = i < k ? __bfloat162float(x[i]) : 0.f;
    const float t = block_sum256(v * v);
    if (threadIdx.x == 0) partial[blockIdx.x] = t;
}

// Block of 256 threads = 8 warps = 8 TQ3 blocks of 32 elements.
__global__ void norm_rotate_i8_k(const __nv_bfloat16 * x, const float * w, const float * partial, int nparts, float eps,
                                 int8_t * xq, float * xs, int k) {
    __shared__ float inv_s;
    if (threadIdx.x < 32) {
        float t = 0.f;
        for (int i = threadIdx.x; i < nparts; i += 32) t += partial[i];
        for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) inv_s = rsqrtf(t / k + eps);
    }
    __syncthreads();
    const int i = blockIdx.x * 256 + threadIdx.x, lane = threadIdx.x % 32, b = i / 32;
    if (i >= k) return;
    const float n = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x[i]) * inv_s * w[i]));
    const float v = wht32(n * TQ3_SIGNS[lane], lane);
    float m = fabsf(v);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float scale = m / 127.f;
    xq[i] = int8_t(m > 0.f ? __float2int_rn(v / scale) : 0);
    if (lane == 0) xs[b] = scale;
}

// ---- two-stage argmax ----
__device__ __forceinline__ void amax_merge(float & v, uint32_t & ix, float ov, uint32_t oi) {
    if (ov > v || (ov == v && oi < ix)) {
        v = ov;
        ix = oi;
    }
}

__global__ void argmax1_k(const float * x, int n, float * pv, uint32_t * pi) {
    float v = -INFINITY;
    uint32_t ix = 0xffffffffu;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) amax_merge(v, ix, x[i], i);
    for (int o = 16; o; o >>= 1) amax_merge(v, ix, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, ix, o));
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    if (threadIdx.x % 32 == 0) {
        sv[threadIdx.x / 32] = v;
        si[threadIdx.x / 32] = ix;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; ++w) amax_merge(v, ix, sv[w], si[w]);
        pv[blockIdx.x] = v;
        pi[blockIdx.x] = ix;
    }
}

__global__ void argmax2_k(const float * pv, const uint32_t * pi, int np, uint32_t * out) {
    float v = -INFINITY;
    uint32_t ix = 0xffffffffu;
    for (int i = threadIdx.x; i < np; i += 32) amax_merge(v, ix, pv[i], pi[i]);
    for (int o = 16; o; o >>= 1) amax_merge(v, ix, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, ix, o));
    if (threadIdx.x == 0) *out = ix;
}


// ---- multi-column verify path ----
__global__ void tq3_rotate_h_k(const __nv_bfloat16 * x, __half * xh, int total_blocks) {
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (b >= total_blocks) return;
    const int64_t i = int64_t(b) * 32 + lane;
    xh[i] = __float2half_rn(wht32(__bfloat162float(x[i]) * TQ3_SIGNS[lane], lane));
}

// grid (k / 256, ncols); partial [ncols][k / 256]
__global__ void norm_rotate_h_k(const __nv_bfloat16 * x, const float * w, const float * partial, int nparts, float eps,
                                __half * xh, int k) {
    __shared__ float inv_s;
    const int n = blockIdx.y;
    if (threadIdx.x < 32) {
        float t = 0.f;
        for (int i = threadIdx.x; i < nparts; i += 32) t += partial[n * nparts + i];
        for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) inv_s = rsqrtf(t / k + eps);
    }
    __syncthreads();
    const int i = blockIdx.x * 256 + threadIdx.x, lane = threadIdx.x % 32;
    if (i >= k) return;
    const int64_t o = int64_t(n) * k + i;
    const float v = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x[o]) * inv_s * w[i]));
    xh[o] = __float2half_rn(wht32(v * TQ3_SIGNS[lane], lane));
}

__device__ __forceinline__ void mma_f16(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16h(uint32_t * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {  // FP16 accumulate: twice the GA10x tensor rate
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ uint32_t hadd2u(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("add.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// FP16 tensor-core matvec over the repacked TQ3_4S layout. CTA = 8 warps on one 16-row tile, split-K across warps.
// k is permuted identically in A and B so lane t of a k16 half owns the 4 consecutive weights 16h + 4t.. (one scale
// group); weights decode in registers: int8 codebook (+128) -> PRMT into fp16 (1024 + u) -> HFMA2 with (s, -1152 s).
// FP16 tensor-core matvec over the repacked TQ3_4S layout. CTA = 8 warps on one 16-row tile, split-K across warps.
// The 4 lanes of a quad take 4 consecutive blocks of the same rows (64 contiguous bytes per row, no redundant
// loads), and each lane decodes its two blocks (rows g, g+8) exactly once. k is permuted identically in A and B:
// MMA u (0..7) takes, from lane t, elements 4u..4u+3 of lane t's block, so the activation fragment of lane t is
// simply the 32 halves of that block. Decode: int8 codebook (+128) -> PRMT into fp16 (1024 + u) -> HFMA2 with
// (s, -1152 s), exact for E3M5 scales.
template <int NCOL, bool EXPERT = false>
__global__ void __launch_bounds__(256) tq3_matvec_h_k(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows,
                                                      int k, uint32_t cblo_u, uint32_t cbhi_u,
                                                      const int * ids = nullptr, int topk = 1, bool per_expert = false) {
    if constexpr (EXPERT) {
        const int assignment = blockIdx.y;
        w += int64_t(ids[assignment]) * rows * (k / 32);
        xh += int64_t(per_expert ? assignment : assignment / topk) * k;
        y += int64_t(assignment) * rows;
    }
    constexpr int NACC = NCOL > 8 ? 8 : 4;
    __shared__ float red[8][32][NACC];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16, nblk = k / 32;
    const uint32_t NEG1152 = 0xE480E480u;
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    float c[NACC] = {};
    for (int b0 = warp * 4; b0 < nblk; b0 += 32) {
        const int b = b0 + t;
        const bool ok = b < nblk;
        const uint4 qa = ok ? __ldg(&w[int64_t(ra) * nblk + b]) : make_uint4(0, 0, 0, 0);
        const uint4 qb = ok ? __ldg(&w[int64_t(rb) * nblk + b]) : make_uint4(0, 0, 0, 0);
        uint4 x4[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        uint4 x42[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if (g < NCOL && ok) {
            const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g) * k + int64_t(b) * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) x4[i] = __ldg(&xp[i]);
        }
        if constexpr (NCOL > 8) {
            if (g < NCOL - 8 && ok) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g + 8) * k + int64_t(b) * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) x42[i] = __ldg(&xp[i]);
            }
        }
        const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
        const uint32_t * xw2 = reinterpret_cast<const uint32_t *>(x42);
        uint32_t sa[4], oa[4], sb[4], ob[4];
#pragma unroll
        for (int gi = 0; gi < 4; ++gi) {
            const uint32_t ba = (qa.x >> (8 * gi)) & 255u, bb = (qb.x >> (8 * gi)) & 255u;
            const uint32_t ha = ba ? (ba << 5) + 0x1800u : 0u, hb = bb ? (bb << 5) + 0x1800u : 0u;
            sa[gi] = ha | (ha << 16);
            sb[gi] = hb | (hb << 16);
            oa[gi] = hmul2u(sa[gi], NEG1152);
            ob[gi] = hmul2u(sb[gi], NEG1152);
        }
        const uint32_t sla = ((qa.y >> 3) & 0x1111u) | ((qa.z >> 2) & 0x2222u) | ((qa.w >> 1) & 0x4444u);
        const uint32_t slb = ((qb.y >> 3) & 0x1111u) | ((qb.z >> 2) & 0x2222u) | ((qb.w >> 1) & 0x4444u);
        const uint32_t sla2 = ((qa.y >> 19) & 0x1111u) | ((qa.z >> 18) & 0x2222u) | ((qa.w >> 17) & 0x4444u);
        const uint32_t slb2 = ((qb.y >> 19) & 0x1111u) | ((qb.z >> 18) & 0x2222u) | ((qb.w >> 17) & 0x4444u);
        const uint32_t sela[8] = {qa.y, qa.y >> 16, qa.z, qa.z >> 16, qa.w, qa.w >> 16, sla, sla2};
        const uint32_t selb[8] = {qb.y, qb.y >> 16, qb.z, qb.z >> 16, qb.w, qb.w >> 16, slb, slb2};
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
            const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
            const int gi = u / 2;
            mma_f16(c, hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]),
                    hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]),
                    hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]),
                    hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]), xw[2 * u], xw[2 * u + 1]);
            if constexpr (NCOL > 8)
                mma_f16(c + 4, hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]),
                        hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]),
                        hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]),
                        hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]), xw2[2 * u], xw2[2 * u + 1]);
        }
    }
#pragma unroll
    for (int i = 0; i < NACC; ++i) red[warp][lane][i] = c[i];
    __syncthreads();
    if (warp == 0) {
        float v[NACC] = {};
        for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
            for (int i = 0; i < NACC; ++i) v[i] += red[w2][lane][i];
        const int n0 = 2 * t, n1 = 2 * t + 1, r = row0 + g;
        const float inv = 1.f / TQ3_CB_Q;
        if (r < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0] * inv);
            if (n1 < NCOL) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1] * inv);
        }
        if (r + 8 < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2] * inv);
            if (n1 < NCOL) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3] * inv);
        }
        if constexpr (NCOL > 8) {
            if (r < rows) {
                if (n0 + 8 < NCOL) y[int64_t(n0 + 8) * rows + r] = __float2bfloat16_rn(v[4] * inv);
                if (n1 + 8 < NCOL) y[int64_t(n1 + 8) * rows + r] = __float2bfloat16_rn(v[5] * inv);
            }
            if (r + 8 < rows) {
                if (n0 + 8 < NCOL) y[int64_t(n0 + 8) * rows + r + 8] = __float2bfloat16_rn(v[6] * inv);
                if (n1 + 8 < NCOL) y[int64_t(n1 + 8) * rows + r + 8] = __float2bfloat16_rn(v[7] * inv);
            }
        }
    }
}

// Wide variant for batched verify (13..32 columns): NT tiles of 8 columns, MT row tiles of 16 per CTA so each
// activation load serves MT row tiles (at 32 columns the 16-row CTA re-read the activations from L2 ~9x over the
// weight bytes). Per row and column the k split, MMA order and 8-warp reduction are those of tq3_matvec_h_k, so the
// results are bit-identical to narrow calls.
template <int NT, int MT>
__global__ void __launch_bounds__(256) tq3_matvec_hw_k(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows,
                                                       int k, int nc, uint32_t cblo_u, uint32_t cbhi_u) {
    __shared__ float red[8][32][4 * NT];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nblk = k / 32;
    const uint32_t NEG1152 = 0xE480E480u;
    float c[MT][NT][4] = {};
    for (int b0 = warp * 4; b0 < nblk; b0 += 32) {
        const int b = b0 + t;
        const bool ok = b < nblk;
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
            uint4 x4[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
            if (8 * tt + g < nc && ok) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(8 * tt + g) * k + int64_t(b) * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) x4[i] = __ldg(&xp[i]);
            }
            const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
#pragma unroll
            for (int m = 0; m < MT; ++m) {  // one activation load serves MT row tiles (weights re-read from L1)
                const int ra = min(row0 + 16 * m + g, rows - 1), rb = min(row0 + 16 * m + g + 8, rows - 1);
                const uint4 qa = ok ? __ldg(&w[int64_t(ra) * nblk + b]) : make_uint4(0, 0, 0, 0);
                const uint4 qb = ok ? __ldg(&w[int64_t(rb) * nblk + b]) : make_uint4(0, 0, 0, 0);
                uint32_t sa[4], oa[4], sb[4], ob[4];
#pragma unroll
                for (int gi = 0; gi < 4; ++gi) {
                    const uint32_t ba = (qa.x >> (8 * gi)) & 255u, bb = (qb.x >> (8 * gi)) & 255u;
                    const uint32_t ha = ba ? (ba << 5) + 0x1800u : 0u, hb = bb ? (bb << 5) + 0x1800u : 0u;
                    sa[gi] = ha | (ha << 16);
                    sb[gi] = hb | (hb << 16);
                    oa[gi] = hmul2u(sa[gi], NEG1152);
                    ob[gi] = hmul2u(sb[gi], NEG1152);
                }
                const uint32_t sla = ((qa.y >> 3) & 0x1111u) | ((qa.z >> 2) & 0x2222u) | ((qa.w >> 1) & 0x4444u);
                const uint32_t slb = ((qb.y >> 3) & 0x1111u) | ((qb.z >> 2) & 0x2222u) | ((qb.w >> 1) & 0x4444u);
                const uint32_t sla2 = ((qa.y >> 19) & 0x1111u) | ((qa.z >> 18) & 0x2222u) | ((qa.w >> 17) & 0x4444u);
                const uint32_t slb2 = ((qb.y >> 19) & 0x1111u) | ((qb.z >> 18) & 0x2222u) | ((qb.w >> 17) & 0x4444u);
                const uint32_t sela[8] = {qa.y, qa.y >> 16, qa.z, qa.z >> 16, qa.w, qa.w >> 16, sla, sla2};
                const uint32_t selb[8] = {qb.y, qb.y >> 16, qb.z, qb.z >> 16, qb.w, qb.w >> 16, slb, slb2};
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
                    const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
                    const int gi = u / 2;
                    mma_f16(c[m][tt], hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]),
                            hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]),
                            hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]),
                            hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]), xw[2 * u], xw[2 * u + 1]);
                }
            }
        }
    }
    const float inv = 1.f / TQ3_CB_Q;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m) __syncthreads();
#pragma unroll
        for (int tt = 0; tt < NT; ++tt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[warp][lane][4 * tt + i] = c[m][tt][i];
        __syncthreads();
        if (warp < NT) {
            const int tt = warp;
            float v[4] = {};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * tt + i];
            const int n0 = 8 * tt + 2 * t, n1 = n0 + 1, r = row0 + 16 * m + g;
            if (r < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1] * inv);
            }
            if (r + 8 < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3] * inv);
            }
        }
    }
}

// <= 8 columns, MT row tiles per CTA, next iteration's weights and activations loaded while the current one computes
// (two loads in flight per warp). Per-column arithmetic identical to tq3_matvec_h_k.
__device__ __forceinline__ void tq3_tile_mma(float * c, const uint4 & qa, const uint4 & qb, const uint32_t * xw,
                                             uint32_t cblo_u, uint32_t cbhi_u) {
    const uint32_t NEG1152 = 0xE480E480u;
    uint32_t sa[4], oa[4], sb[4], ob[4];
#pragma unroll
    for (int gi = 0; gi < 4; ++gi) {
        const uint32_t ba = (qa.x >> (8 * gi)) & 255u, bb = (qb.x >> (8 * gi)) & 255u;
        const uint32_t ha = ba ? (ba << 5) + 0x1800u : 0u, hb = bb ? (bb << 5) + 0x1800u : 0u;
        sa[gi] = ha | (ha << 16);
        sb[gi] = hb | (hb << 16);
        oa[gi] = hmul2u(sa[gi], NEG1152);
        ob[gi] = hmul2u(sb[gi], NEG1152);
    }
    const uint32_t sla = ((qa.y >> 3) & 0x1111u) | ((qa.z >> 2) & 0x2222u) | ((qa.w >> 1) & 0x4444u);
    const uint32_t slb = ((qb.y >> 3) & 0x1111u) | ((qb.z >> 2) & 0x2222u) | ((qb.w >> 1) & 0x4444u);
    const uint32_t sla2 = ((qa.y >> 19) & 0x1111u) | ((qa.z >> 18) & 0x2222u) | ((qa.w >> 17) & 0x4444u);
    const uint32_t slb2 = ((qb.y >> 19) & 0x1111u) | ((qb.z >> 18) & 0x2222u) | ((qb.w >> 17) & 0x4444u);
    const uint32_t sela[8] = {qa.y, qa.y >> 16, qa.z, qa.z >> 16, qa.w, qa.w >> 16, sla, sla2};
    const uint32_t selb[8] = {qb.y, qb.y >> 16, qb.z, qb.z >> 16, qb.w, qb.w >> 16, slb, slb2};
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
        const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
        const int gi = u / 2;
        mma_f16(c, hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]),
                hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]),
                hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]),
                hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]), xw[2 * u], xw[2 * u + 1]);
    }
}

// > 8 columns: NT tiles of 8 columns, MT row tiles per CTA, next iteration's loads in flight while the current one
// computes (as tq3_matvec_p_k), and each weight block decoded once for all NT tiles (tq3_matvec_hw_k decodes per tile).
// Per accumulator the k split, MMA order and 8-warp reduction are tq3_matvec_h_k's: bit-identical per column.
template <int NT, int MT>
__global__ void __launch_bounds__(256) tq3_matvec_pw_k(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows,
                                                       int k, int nc, uint32_t cblo_u, uint32_t cbhi_u) {
    __shared__ float red[8][32][4 * NT];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nblk = k / 32;
    const uint32_t NEG1152 = 0xE480E480u;
    int ra[MT], rb[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        ra[m] = min(row0 + 16 * m + g, rows - 1);
        rb[m] = min(row0 + 16 * m + g + 8, rows - 1);
    }
    float c[MT][NT][4] = {};
    uint4 qa[MT], qb[MT], x4[NT][4];
    auto load = [&](int b0, uint4 (&A)[MT], uint4 (&B)[MT], uint4 (&X)[NT][4]) {
        const int b = b0 + t;
        const bool ok = b < nblk;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            A[m] = ok ? __ldg(&w[int64_t(ra[m]) * nblk + b]) : make_uint4(0, 0, 0, 0);
            B[m] = ok ? __ldg(&w[int64_t(rb[m]) * nblk + b]) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
            const bool on = 8 * tt + g < nc && ok;
            const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(8 * tt + g) * k + int64_t(b) * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) X[tt][i] = on ? __ldg(&xp[i]) : make_uint4(0, 0, 0, 0);
        }
    };
    int b0 = warp * 4;
    if (b0 < nblk) load(b0, qa, qb, x4);
    for (; b0 < nblk; b0 += 32) {
        uint4 na[MT], nb[MT], nx[NT][4];
        const bool more = b0 + 32 < nblk;
        if (more) load(b0 + 32, na, nb, nx);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            uint32_t sa[4], oa[4], sb[4], ob[4];
#pragma unroll
            for (int gi = 0; gi < 4; ++gi) {
                const uint32_t ba = (qa[m].x >> (8 * gi)) & 255u, bb = (qb[m].x >> (8 * gi)) & 255u;
                const uint32_t ha = ba ? (ba << 5) + 0x1800u : 0u, hb = bb ? (bb << 5) + 0x1800u : 0u;
                sa[gi] = ha | (ha << 16);
                sb[gi] = hb | (hb << 16);
                oa[gi] = hmul2u(sa[gi], NEG1152);
                ob[gi] = hmul2u(sb[gi], NEG1152);
            }
            const uint4 A = qa[m], B = qb[m];
            const uint32_t sla = ((A.y >> 3) & 0x1111u) | ((A.z >> 2) & 0x2222u) | ((A.w >> 1) & 0x4444u);
            const uint32_t slb = ((B.y >> 3) & 0x1111u) | ((B.z >> 2) & 0x2222u) | ((B.w >> 1) & 0x4444u);
            const uint32_t sla2 = ((A.y >> 19) & 0x1111u) | ((A.z >> 18) & 0x2222u) | ((A.w >> 17) & 0x4444u);
            const uint32_t slb2 = ((B.y >> 19) & 0x1111u) | ((B.z >> 18) & 0x2222u) | ((B.w >> 17) & 0x4444u);
            const uint32_t sela[8] = {A.y, A.y >> 16, A.z, A.z >> 16, A.w, A.w >> 16, sla, sla2};
            const uint32_t selb[8] = {B.y, B.y >> 16, B.z, B.z >> 16, B.w, B.w >> 16, slb, slb2};
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
                const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
                const int gi = u / 2;
                const uint32_t a0 = hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]);
                const uint32_t a1 = hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]);
                const uint32_t a2 = hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]);
                const uint32_t a3 = hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]);
#pragma unroll
                for (int tt = 0; tt < NT; ++tt) {
                    const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4[tt]);
                    mma_f16(c[m][tt], a0, a1, a2, a3, xw[2 * u], xw[2 * u + 1]);
                }
            }
        }
        if (more) {
#pragma unroll
            for (int m = 0; m < MT; ++m) { qa[m] = na[m]; qb[m] = nb[m]; }
#pragma unroll
            for (int tt = 0; tt < NT; ++tt)
#pragma unroll
                for (int i = 0; i < 4; ++i) x4[tt][i] = nx[tt][i];
        }
    }
    const float inv = 1.f / TQ3_CB_Q;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m) __syncthreads();
#pragma unroll
        for (int tt = 0; tt < NT; ++tt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[warp][lane][4 * tt + i] = c[m][tt][i];
        __syncthreads();
        if (warp < NT) {
            const int tt = warp;
            float v[4] = {};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * tt + i];
            const int n0 = 8 * tt + 2 * t, n1 = n0 + 1, r = row0 + 16 * m + g;
            if (r < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1] * inv);
            }
            if (r + 8 < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3] * inv);
            }
        }
    }
}

template <int MT>
__global__ void __launch_bounds__(256) tq3_matvec_p_k(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows,
                                                      int k, int nc, uint32_t cblo_u, uint32_t cbhi_u) {
    __shared__ float red[8][32][4];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nblk = k / 32;
    int ra[MT], rb[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        ra[m] = min(row0 + 16 * m + g, rows - 1);
        rb[m] = min(row0 + 16 * m + g + 8, rows - 1);
    }
    float c[MT][4] = {};
    uint4 qa[MT], qb[MT], x4[4];
    auto load = [&](int b0, uint4 (&A)[MT], uint4 (&B)[MT], uint4 (&X)[4]) {
        const int b = b0 + t;
        const bool ok = b < nblk;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            A[m] = ok ? __ldg(&w[int64_t(ra[m]) * nblk + b]) : make_uint4(0, 0, 0, 0);
            B[m] = ok ? __ldg(&w[int64_t(rb[m]) * nblk + b]) : make_uint4(0, 0, 0, 0);
        }
        const bool on = g < nc && ok;
        const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g) * k + int64_t(b) * 32);
#pragma unroll
        for (int i = 0; i < 4; ++i) X[i] = on ? __ldg(&xp[i]) : make_uint4(0, 0, 0, 0);
    };
    int b0 = warp * 4;
    if (b0 < nblk) load(b0, qa, qb, x4);
    for (; b0 < nblk; b0 += 32) {
        uint4 na[MT], nb[MT], nx[4];
        const bool more = b0 + 32 < nblk;
        if (more) load(b0 + 32, na, nb, nx);
        const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
#pragma unroll
        for (int m = 0; m < MT; ++m) tq3_tile_mma(c[m], qa[m], qb[m], xw, cblo_u, cbhi_u);
        if (more) {
#pragma unroll
            for (int m = 0; m < MT; ++m) { qa[m] = na[m]; qb[m] = nb[m]; }
#pragma unroll
            for (int i = 0; i < 4; ++i) x4[i] = nx[i];
        }
    }
    const float inv = 1.f / TQ3_CB_Q;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m) __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][lane][i] = c[m][i];
        __syncthreads();
        if (warp == 0) {
            float v[4] = {};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][i];
            const int n0 = 2 * t, n1 = n0 + 1, r = row0 + 16 * m + g;
            if (r < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1] * inv);
            }
            if (r + 8 < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3] * inv);
            }
        }
    }
}

struct Tq3CodebookU {
    uint32_t lo, hi;
};

Tq3CodebookU tq3_codebook_u8() {  // int8 codebook + 128, as unsigned bytes
    static const float c[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f,
                               0.230106f,  0.725222f,  1.277503f,  1.988943f};
    Tq3CodebookU cb{0, 0};
    for (int i = 0; i < 4; ++i) {
        cb.lo |= uint32_t(uint8_t(lrintf(c[i] * TQ3_CB_Q) + 128)) << (8 * i);
        cb.hi |= uint32_t(uint8_t(lrintf(c[i + 4] * TQ3_CB_Q) + 128)) << (8 * i);
    }
    return cb;
}

__global__ void argmax_cols1_k(const float * x, int len, float * pv, uint32_t * pi, int nb) {
    const int n = blockIdx.y;
    const float * xn = x + int64_t(n) * len;
    float v = -INFINITY;
    uint32_t ix = 0xffffffffu;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) amax_merge(v, ix, xn[i], i);
    for (int o = 16; o; o >>= 1) amax_merge(v, ix, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, ix, o));
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    if (threadIdx.x % 32 == 0) {
        sv[threadIdx.x / 32] = v;
        si[threadIdx.x / 32] = ix;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; ++w) amax_merge(v, ix, sv[w], si[w]);
        pv[n * nb + blockIdx.x] = v;
        pi[n * nb + blockIdx.x] = ix;
    }
}

__global__ void argmax_cols2_k(const float * pv, const uint32_t * pi, int nb, uint32_t * out) {
    const int n = blockIdx.x;
    float v = -INFINITY;
    uint32_t ix = 0xffffffffu;
    for (int i = threadIdx.x; i < nb; i += 32) amax_merge(v, ix, pv[n * nb + i], pi[n * nb + i]);
    for (int o = 16; o; o >>= 1) amax_merge(v, ix, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, ix, o));
    if (threadIdx.x == 0) out[n] = ix;
}

// Q6_K: one warp per row; lane l handles, in each 128-half of a 256 block, elements l, l+32, l+64, l+96.
template <int N>
__global__ void __launch_bounds__(256) q6k_matvec_k(const uint8_t * w, const float * x, float * y, int rows, int k) {
    const int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const int nblk = k / Q6K_BLOCK;
    const uint8_t * wr = w + int64_t(row) * nblk * Q6K_BLOCK_BYTES;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; ++n) acc[n] = 0.f;
    const int is = lane / 16;
    for (int b = 0; b < nblk; ++b) {
        const uint8_t * blk = wr + b * Q6K_BLOCK_BYTES;
        const uint8_t *ql = blk, *qh = blk + 128;
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
        const float d = __half2float(*reinterpret_cast<const __half *>(blk + 208));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int l0 = ql[h * 64 + lane], l1 = ql[h * 64 + lane + 32], hb = qh[h * 32 + lane];
            const float q1 = float(((l0 & 0xF) | ((hb & 3) << 4)) - 32) * d * sc[h * 8 + is];
            const float q2 = float(((l1 & 0xF) | (((hb >> 2) & 3) << 4)) - 32) * d * sc[h * 8 + is + 2];
            const float q3 = float(((l0 >> 4) | (((hb >> 4) & 3) << 4)) - 32) * d * sc[h * 8 + is + 4];
            const float q4 = float(((l1 >> 4) | (((hb >> 6) & 3) << 4)) - 32) * d * sc[h * 8 + is + 6];
            const int64_t base = int64_t(b) * Q6K_BLOCK + h * 128 + lane;
#pragma unroll
            for (int n = 0; n < N; ++n) {
                const float * xn = x + int64_t(n) * k + base;
                acc[n] += q1 * xn[0] + q2 * xn[32] + q3 * xn[64] + q4 * xn[96];
            }
        }
    }
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float v = acc[n];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) y[int64_t(n) * rows + row] = v;
    }
}


// Q6_K with word loads: 8 lanes per 128-element half block, lane t owns l = 4t..4t+3 of that half (16 weights:
// l, l+32, l+64, l+96 for each of its 4 l values). A warp covers 2 blocks per iteration.
template <int N>
__global__ void __launch_bounds__(256) q6k_matvec_w_k(const uint8_t * w, const float * x, float * y, int rows, int k) {
    const int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const int nblk = k / Q6K_BLOCK;
    const uint8_t * wr = w + int64_t(row) * nblk * Q6K_BLOCK_BYTES;
    const int half = lane / 8 % 2, sub = lane / 16, t = lane % 8, l = 4 * t, is = l / 16;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; ++n) acc[n] = 0.f;
    for (int b = sub; b < nblk; b += 2) {
        const uint8_t * blk = wr + b * Q6K_BLOCK_BYTES;
        const uint32_t ql0 = *reinterpret_cast<const uint16_t *>(blk + half * 64 + l) |
                             (uint32_t(*reinterpret_cast<const uint16_t *>(blk + half * 64 + l + 2)) << 16);
        const uint32_t ql1 = *reinterpret_cast<const uint16_t *>(blk + half * 64 + l + 32) |
                             (uint32_t(*reinterpret_cast<const uint16_t *>(blk + half * 64 + l + 34)) << 16);
        const uint32_t qh = *reinterpret_cast<const uint16_t *>(blk + 128 + half * 32 + l) |
                            (uint32_t(*reinterpret_cast<const uint16_t *>(blk + 128 + half * 32 + l + 2)) << 16);
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192) + half * 8 + is;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk + 208));
        const float d1 = d * sc[0], d2 = d * sc[2], d3 = d * sc[4], d4 = d * sc[6];
        float w1[4], w2[4], w3[4], w4[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int a = (ql0 >> (8 * e)) & 255, c = (ql1 >> (8 * e)) & 255, h = (qh >> (8 * e)) & 255;
            w1[e] = d1 * float(((a & 0xF) | ((h & 3) << 4)) - 32);
            w2[e] = d2 * float(((c & 0xF) | (((h >> 2) & 3) << 4)) - 32);
            w3[e] = d3 * float(((a >> 4) | (((h >> 4) & 3) << 4)) - 32);
            w4[e] = d4 * float(((c >> 4) | (((h >> 6) & 3) << 4)) - 32);
        }
        const int64_t base = int64_t(b) * Q6K_BLOCK + half * 128 + l;
#pragma unroll
        for (int n = 0; n < N; ++n) {
            const float * xn = x + int64_t(n) * k + base;
            const float4 x1 = *reinterpret_cast<const float4 *>(xn), x2 = *reinterpret_cast<const float4 *>(xn + 32);
            const float4 x3 = *reinterpret_cast<const float4 *>(xn + 64), x4 = *reinterpret_cast<const float4 *>(xn + 96);
            acc[n] += w1[0] * x1.x + w1[1] * x1.y + w1[2] * x1.z + w1[3] * x1.w + w2[0] * x2.x + w2[1] * x2.y +
                      w2[2] * x2.z + w2[3] * x2.w + w3[0] * x3.x + w3[1] * x3.y + w3[2] * x3.z + w3[3] * x3.w +
                      w4[0] * x4.x + w4[1] * x4.y + w4[2] * x4.z + w4[3] * x4.w;
        }
    }
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float v = acc[n];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) y[int64_t(n) * rows + row] = v;
    }
}


// One thread per element of one Q6_K row (ggml dequantize_row_q6_K layout).
__global__ void q6k_row_k(const uint8_t * w, int row, __nv_bfloat16 * y, int k, const uint32_t * dtok = nullptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    if (dtok) {  // multi-token: blockIdx.y selects the token, id read on the device
        row = int(dtok[blockIdx.y]);
        y += int64_t(blockIdx.y) * k;
    }
    const int nblk = k / Q6K_BLOCK, b = i / Q6K_BLOCK, e = i % Q6K_BLOCK;
    const uint8_t * blk = w + (int64_t(row) * nblk + b) * Q6K_BLOCK_BYTES;
    const int h = e / 128, q = (e % 128) / 32, l = e % 32, is = l / 16;
    const uint8_t * ql = blk + h * 64;
    const uint8_t hb = blk[128 + h * 32 + l];
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192) + h * 8;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk + 208));
    int v;
    if (q == 0) v = (ql[l] & 0xF) | ((hb & 3) << 4);
    else if (q == 1) v = (ql[l + 32] & 0xF) | (((hb >> 2) & 3) << 4);
    else if (q == 2) v = (ql[l] >> 4) | (((hb >> 4) & 3) << 4);
    else v = (ql[l + 32] >> 4) | (((hb >> 6) & 3) << 4);
    y[i] = __float2bfloat16_rn(d * sc[is + 2 * q] * float(v - 32));
}


// ---- Q6_K -> int8 + FP16 scales (load time) and the FP16 tensor-core LM head ----
__global__ void q6k_to_i8_k(const uint8_t * w, int8_t * w8, __half * sc, int rows, int k) {
    const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // element index
    if (e >= int64_t(rows) * k) return;
    const int row = int(e / k), i = int(e % k), b = i / Q6K_BLOCK, el = i % Q6K_BLOCK;
    const uint8_t * blk = w + (int64_t(row) * (k / Q6K_BLOCK) + b) * Q6K_BLOCK_BYTES;
    const int h = el / 128, q = (el % 128) / 32, l = el % 32, is = l / 16;
    const uint8_t * ql = blk + h * 64;
    const uint8_t hb = blk[128 + h * 32 + l];
    int v;
    if (q == 0) v = (ql[l] & 0xF) | ((hb & 3) << 4);
    else if (q == 1) v = (ql[l + 32] & 0xF) | (((hb >> 2) & 3) << 4);
    else if (q == 2) v = (ql[l] >> 4) | (((hb >> 4) & 3) << 4);
    else v = (ql[l + 32] >> 4) | (((hb >> 6) & 3) << 4);
    w8[e] = int8_t(v - 32);
    if (el % 16 == 0) {  // one scale per 16 consecutive weights (Q6_K sub-block)
        const int8_t * scl = reinterpret_cast<const int8_t *>(blk + 192) + h * 8;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk + 208));
        sc[e / 16] = __float2half_rn(d * float(scl[is + 2 * q]));
    }
}

// CTA 8 warps on a 16-row tile, split-K over warps. The 4 lanes of a quad take 4 consecutive 16-weight scale groups
// of the same rows (64 contiguous bytes per row per quad, no redundant loads); k is permuted identically in A and B
// so MMA u (0..3) takes elements 4u..4u+3 of each lane's group. Per element: (int8 ^ 0x80) -> fp16 (1024 + u),
// + (-1152) = q exactly, * s (the FP16 d * sc of that group).
template <int NCOL, class OUT = float>
__global__ void __launch_bounds__(256) i8h_matvec_k(const int8_t * w8, const __half * sc, const __half * xh, OUT * y,
                                                    int rows, int k, int ldy = 0) {
    const int64_t ld = ldy ? ldy : rows;  // output column stride
    constexpr int NACC = NCOL > 8 ? 8 : 4;
    __shared__ float red[8][32][NACC];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16, nk = k / 16;
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    const uint32_t NEG1152 = 0xE480E480u;
    float c[NACC] = {};
    for (int s0 = warp * 4; s0 < nk; s0 += 32) {
        const int s = s0 + t;
        const bool ok = s < nk;
        uint4 wa = make_uint4(0, 0, 0, 0), wb = make_uint4(0, 0, 0, 0);
        uint32_t sau = 0, sbu = 0;
        uint4 x0 = make_uint4(0, 0, 0, 0), x1 = make_uint4(0, 0, 0, 0);
        uint4 x02 = make_uint4(0, 0, 0, 0), x12 = make_uint4(0, 0, 0, 0);
        if (ok) {
            wa = __ldg(reinterpret_cast<const uint4 *>(w8 + int64_t(ra) * k + int64_t(s) * 16));
            wb = __ldg(reinterpret_cast<const uint4 *>(w8 + int64_t(rb) * k + int64_t(s) * 16));
            const __half sa = sc[int64_t(ra) * nk + s], sb = sc[int64_t(rb) * nk + s];
            const __half2 sa2 = __halves2half2(sa, sa), sb2 = __halves2half2(sb, sb);
            sau = *reinterpret_cast<const uint32_t *>(&sa2);
            sbu = *reinterpret_cast<const uint32_t *>(&sb2);
            if (g < NCOL) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g) * k + int64_t(s) * 16);
                x0 = __ldg(&xp[0]);
                x1 = __ldg(&xp[1]);
            }
            if constexpr (NCOL > 8) {
                if (g < NCOL - 8) {
                    const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g + 8) * k + int64_t(s) * 16);
                    x02 = __ldg(&xp[0]);
                    x12 = __ldg(&xp[1]);
                }
            }
        }
        const uint32_t wa4[4] = {wa.x ^ 0x80808080u, wa.y ^ 0x80808080u, wa.z ^ 0x80808080u, wa.w ^ 0x80808080u};
        const uint32_t wb4[4] = {wb.x ^ 0x80808080u, wb.y ^ 0x80808080u, wb.z ^ 0x80808080u, wb.w ^ 0x80808080u};
        const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        const uint32_t xw2[8] = {x02.x, x02.y, x02.z, x02.w, x12.x, x12.y, x12.z, x12.w};
        auto deq = [&](uint32_t w, uint32_t sel, uint32_t su) { return hmul2u(hadd2u(__byte_perm(w, 0x64646464u, sel), NEG1152), su); };
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            mma_f16(c, deq(wa4[u], 0x5140, sau), deq(wb4[u], 0x5140, sbu), deq(wa4[u], 0x5342, sau),
                    deq(wb4[u], 0x5342, sbu), xw[2 * u], xw[2 * u + 1]);
            if constexpr (NCOL > 8)
                mma_f16(c + 4, deq(wa4[u], 0x5140, sau), deq(wb4[u], 0x5140, sbu), deq(wa4[u], 0x5342, sau),
                        deq(wb4[u], 0x5342, sbu), xw2[2 * u], xw2[2 * u + 1]);
        }
    }
#pragma unroll
    for (int i = 0; i < NACC; ++i) red[warp][lane][i] = c[i];
    __syncthreads();
    if (warp == 0) {
        float v[NACC] = {};
        for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
            for (int i = 0; i < NACC; ++i) v[i] += red[w2][lane][i];
        const int n0 = 2 * t, n1 = 2 * t + 1, r = row0 + g;
        auto put = [&](int64_t i, float v2) {
            if constexpr (std::is_same_v<OUT, float>) y[i] = v2;
            else y[i] = __float2bfloat16_rn(v2);
        };
        if (r < rows) {
            if (n0 < NCOL) put(int64_t(n0) * ld + r, v[0]);
            if (n1 < NCOL) put(int64_t(n1) * ld + r, v[1]);
        }
        if (r + 8 < rows) {
            if (n0 < NCOL) put(int64_t(n0) * ld + r + 8, v[2]);
            if (n1 < NCOL) put(int64_t(n1) * ld + r + 8, v[3]);
        }
        if constexpr (NCOL > 8) {
            if (r < rows) {
                if (n0 + 8 < NCOL) put(int64_t(n0 + 8) * ld + r, v[4]);
                if (n1 + 8 < NCOL) put(int64_t(n1 + 8) * ld + r, v[5]);
            }
            if (r + 8 < rows) {
                if (n0 + 8 < NCOL) put(int64_t(n0 + 8) * ld + r + 8, v[6]);
                if (n1 + 8 < NCOL) put(int64_t(n1 + 8) * ld + r + 8, v[7]);
            }
        }
    }
}

// u4h: 4-bit codes + FP16 (scale, min) per 32 weights, w = q * s + m (a Q4_K sub-block carried over exactly up to the
// FP16 rounding of s and m). Same CTA / quad / k-permutation scheme as i8h_matvec_k: a lane owns a 16-weight group
// stored as 8 bytes (a lane loads a whole 32-weight block); element e = 4u + j (u = 0..3) of a group is byte j,
// nibble u % 2 of word u / 2. Decode: nibble -> fp16
// (1024 + q), - 1024 = q exactly, then one HFMA2 with (s, m).
// k4 (native Q4_K): the same codes, but per row one 16-byte Q4_K superblock header (FP16 d, dmin, 12 bytes of 6-bit
// scales / mins) per 256 weights instead of an FP16 (scale, min) pair per 32; (d * sc, -dmin * m) is rebuilt per block.
__device__ __forceinline__ __half2 k4_sm(const uint8_t * hdr, int j) {
    const uint4 h = __ldg(reinterpret_cast<const uint4 *>(hdr));
    const uint8_t * b = reinterpret_cast<const uint8_t *>(&h);
    const float d = __half2float(*reinterpret_cast<const __half *>(b)), dm = __half2float(*reinterpret_cast<const __half *>(b + 2));
    const uint8_t * q = b + 4;
    int sc, m;
    if (j < 4) sc = q[j] & 63, m = q[j + 4] & 63;
    else sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    return __floats2half2_rn(d * float(sc), -dm * float(m));
}

// k3 (native Q3_K): per 32 weights three words: L0, L1 (2-bit planes of the two 16-groups, element 4u + j at bits
// 8j + 2u) and H (high bits: group h, element 4u + j at bit 8j + u + 4h); codes 0..7 = Q3_K value + 4. Per row and
// 256 weights a 20-byte record: 16 int8 scales (one per 16 weights) and the FP16 d. w = d * sc * code - 4 d * sc.

// iq3 (native IQ3_S): per 32 weights 12 bytes (8 grid-index low bytes, 4 sign bytes); per row and 256 weights a 16-byte
// record (FP16 d, qh[8] = index bit 8 per 32-block, scales[4] = 4-bit per 32-block). Grid word w of a 32-block is
// elements 4w..4w+3 = one u4h code word; codes 16 +- magnitude, then the u4h HFMA with (db, -16 db), db = d (1 + 2 sc).
__device__ const uint32_t iq3s_grid_d[512] = {
#include "iq3s_grid.inc"
};
__device__ __forceinline__ uint32_t iq3_code(uint32_t g, uint32_t sbits) {  // 4 magnitudes, 4 sign bits -> 16 +- m
    const uint32_t neg = ((sbits * 0x00204081u) & 0x01010101u) * 0xFFu;
    return (__vadd4(0x10101010u, g) & ~neg) | (__vsub4(0x10101010u, g) & neg);
}
__device__ __forceinline__ __half2 iq3_sm(const uint8_t * rec, int ib) {
    const float d = __half2float(__ldg(reinterpret_cast<const __half *>(rec)));
    const float db = d * float(1 + 2 * ((__ldg(rec + 10 + ib / 2) >> 4 * (ib % 2)) & 0xf));
    return __floats2half2_rn(db, -16.f * db);
}

// iq3x (native IQ3_XXS): per 32 weights 12 bytes (8 grid-index bytes into a 256-entry grid, the scale / sign word);
// per row and 256 weights a 4-byte record (FP16 d, pad). Signs: 7-bit index, 8th sign = parity. Codes 64 +- m.
__device__ const uint32_t iq3xxs_grid_d[256] = {
#include "iq3xxs_grid.inc"
};
__device__ __forceinline__ uint32_t iq3x_code(uint32_t g, uint32_t sbits) {  // 4 magnitudes, 4 sign bits -> 64 +- m
    const uint32_t neg = ((sbits * 0x00204081u) & 0x01010101u) * 0xFFu;
    return (__vadd4(0x40404040u, g) & ~neg) | (__vsub4(0x40404040u, g) & neg);
}
__device__ __forceinline__ uint32_t iq3x_signs(uint32_t a, int w) {  // sign bits of grid word w (0..7) of a 32-block
    const uint32_t i7 = (a >> (7 * (w / 2))) & 127, sg = i7 | ((__popc(i7) & 1) << 7);
    return (sg >> (4 * (w % 2))) & 0xF;
}
__device__ __forceinline__ __half2 iq3x_sm(const uint8_t * rec, uint32_t a) {
    const float db = __half2float(__ldg(reinterpret_cast<const __half *>(rec))) * (0.5f + float(a >> 28)) * 0.5f;
    return __floats2half2_rn(db, -64.f * db);
}

// UD low-bit types on the u4h kernel (FMT 8 IQ2_XXS, 9 IQ2_XS, 10 IQ2_S, 11 IQ1_S, 12 Q2_K): per 32 weights 1-2 code
// words as stored in the GGUF block, per row and 256 weights a record of the block's scalars. Grids: llama.cpp (MIT),
// used only to read these files (see NOTICE).
__device__ const uint64_t iq2xxs_grid_d[256] = {
#include "iq2xxs_grid.inc"
};
__device__ const uint64_t iq2xs_grid_d[512] = {
#include "iq2xs_grid.inc"
};
__device__ const uint64_t iq2s_grid_d[1024] = {
#include "iq2s_grid.inc"
};
__device__ const uint64_t iq1s_grid_d[2048] = {
#include "iq1s_grid.inc"
};
__host__ __device__ constexpr int ud_bpb(int f) { return f == 11 ? 4 : 8; }    // code bytes per 32 weights
__host__ __device__ constexpr int ud_rb(int f) { return f == 8 ? 4 : f == 9 ? 12 : 20; }  // record bytes per 256
__host__ __device__ constexpr int ud_grid_words(int f) {
    return f == 4 ? 512 : f == 5 ? 256 : f == 8 ? 512 : f == 9 ? 1024 : f == 10 ? 2048 : f == 11 ? 4096 : 1;
}
template <int FMT>
__device__ __forceinline__ const uint32_t * ud_grid_src() {
    if constexpr (FMT == 8) return reinterpret_cast<const uint32_t *>(iq2xxs_grid_d);
    else if constexpr (FMT == 9) return reinterpret_cast<const uint32_t *>(iq2xs_grid_d);
    else if constexpr (FMT == 10) return reinterpret_cast<const uint32_t *>(iq2s_grid_d);
    else return reinterpret_cast<const uint32_t *>(iq1s_grid_d);
}
// codes (one byte per weight, value = code * s + m) of 16-group h of 32-block s; c[u] holds weights 4u..4u+3 of the group
template <int FMT>
__device__ __forceinline__ void ud_group(uint32_t w0, uint32_t w1, const uint8_t * rec, int s, int h, const uint32_t * grid,
                                         uint32_t c[4], __half2 & sm) {
    if constexpr (FMT == 12) {  // Q2_K: 2-bit codes of group h at bits 2 e of word h; record scales[16], d, dmin
        const uint8_t sc = __ldg(rec + (s % 8) * 2 + h);
        const float d = __half2float(__ldg(reinterpret_cast<const __half *>(rec + 16)));
        const float dm = __half2float(__ldg(reinterpret_cast<const __half *>(rec + 18)));
        const uint32_t v = h ? w1 : w0;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint32_t x = (v >> (8 * u)) & 0xFF;
            c[u] = (x & 3u) | ((x & 0xCu) << 6) | ((x & 0x30u) << 12) | ((x & 0xC0u) << 18);
        }
        sm = __floats2half2_rn(d * float(sc & 0xF), -dm * float(sc >> 4));
        return;
    }
    const float d = __half2float(__ldg(reinterpret_cast<const __half *>(rec)));
    if constexpr (FMT == 11) {  // IQ1_S: 11-bit index (qs byte, 3 qh bits), grid -1 / 0 / 1; code 8 g + 16 +- 1
        const uint32_t qh = __ldg(reinterpret_cast<const uint16_t *>(rec + 2) + s % 8);
        const float dl = d * float(2 * ((qh >> 12) & 7) + 1);
        const uint32_t add = (qh & 0x8000u) ? 0x07070707u : 0x09090909u;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int w = 4 * h + u, l = w / 2;
            const uint32_t idx = ((w0 >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8);
            c[u] = (__vadd4(grid[2 * idx + (w & 1)], 0x01010101u) << 3) + add;
        }
        sm = __floats2half2_rn(0.125f * dl, -2.f * dl);
        return;
    }
    float db;
    if constexpr (FMT == 8) {  // IQ2_XXS: 4 grid bytes (w0); w1 = 4 x 7-bit sign index + 4-bit scale
        db = d * (0.5f + float(w1 >> 28)) * 0.25f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int w = 4 * h + u;
            c[u] = iq3x_code(grid[2 * ((w0 >> (8 * (w / 2))) & 0xFF) + (w & 1)], iq3x_signs(w1, w));
        }
    } else if constexpr (FMT == 9) {  // IQ2_XS: 4 x uint16 (9-bit index, 7-bit sign index); record d, pad, scales[8]
        const uint8_t sc = __ldg(rec + 4 + s % 8);
        db = d * (0.5f + float(h ? sc >> 4 : sc & 0xF)) * 0.25f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int w = 4 * h + u, l = w / 2;
            const uint32_t q = ((l < 2 ? w0 : w1) >> (16 * (l & 1))) & 0xFFFF, i7 = q >> 9;
            const uint32_t sg = i7 | ((__popc(i7) & 1) << 7);
            c[u] = iq3x_code(grid[2 * (q & 511) + (w & 1)], (sg >> (4 * (w & 1))) & 0xF);
        }
    } else {  // IQ2_S: 4 grid low bytes (w0), 4 sign bytes (w1); record d, qh[8], scales[8]
        const uint32_t qh = __ldg(rec + 2 + s % 8);
        const uint8_t sc = __ldg(rec + 10 + s % 8);
        db = d * (0.5f + float(h ? sc >> 4 : sc & 0xF)) * 0.25f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int w = 4 * h + u, l = w / 2;
            const uint32_t idx = ((w0 >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
            c[u] = iq3x_code(grid[2 * idx + (w & 1)], ((w1 >> (8 * l)) >> (4 * (w & 1))) & 0xF);
        }
    }
    sm = __floats2half2_rn(db, -64.f * db);
}

// IQ4_NL codebook (llama.cpp kvalues_iq4nl, MIT; see NOTICE) as level + 128 bytes: 4 nibbles -> 4 code bytes
__device__ __forceinline__ uint32_t iq4nl_codes(uint32_t c) {
    constexpr uint32_t L0 = 0x3F2D1801u, L1 = 0x766A5D4Fu, L2 = 0xA6998D81u, L3 = 0xF1D9C5B5u;
    const uint32_t sel = c & 0x7777u;
    return __byte_perm(__byte_perm(L0, L1, sel), __byte_perm(L2, L3, sel), 0x3210u | ((c >> 1) & 0x4444u));
}
// One lane's 32-weight block s of rows ra and rb in the u4h-family layout: load() reads its codes and (scale, min) pairs,
// codes(h) returns the code bytes of 16-group h (value = code * scale + min). Shared by u4h_matvec_k and u4h_gemm_k.
// TQFN_TILED expert banks: the code bytes of 16-row x 4-block tiles are stored lane-contiguous (row-in-tile major, then
// block-in-quad), so the 32 lanes of a warp load one contiguous run instead of 8 strided row pieces. Same bytes, same math.
__constant__ int c_tiled_experts = 0;
__host__ __device__ constexpr int tiled_bpb(int f) { return f == 4 || f == 5 ? 12 : f == 13 || f == 14 ? 16 : f >= 8 && f <= 12 ? ud_bpb(f) : 0; }
__device__ __forceinline__ int64_t u4h_block_off(int r, int s, int nb, int bpb, bool tiled) {
    if (!tiled) return (int64_t(r) * nb + s) * bpb;
    return (((int64_t(r >> 4) * (nb >> 2) + (s >> 2)) * 16 + (r & 15)) * 4 + (s & 3)) * bpb;
}
template <int FMT, bool GROUPED = false>
struct U4hTile {
    uint4 wa = make_uint4(0, 0, 0, 0), wb = make_uint4(0, 0, 0, 0);
    uint32_t ha = 0, hb = 0, ha2 = 0, hb2 = 0;  // k5: the block's 32 high code bits; k6: 2 high bits per code (2 words)
    const uint8_t * ua = nullptr, * ub = nullptr;  // UD types: the 32-block's record
    uint32_t sa[2] = {}, ma[2] = {}, sb[2] = {}, mb[2] = {};
    uint4 wa2 = make_uint4(0, 0, 0, 0), wb2 = make_uint4(0, 0, 0, 0);  // i8h: the block's second 16 codes
    static __device__ __forceinline__ void pk(__half2 p, uint32_t & sv, uint32_t & mv) {
        const __half2 s2 = __low2half2(p), m2 = __high2half2(p);
        sv = *reinterpret_cast<const uint32_t *>(&s2);
        mv = *reinterpret_cast<const uint32_t *>(&m2);
    }
    __device__ __forceinline__ void load(const uint8_t * w4, const __half2 * sm, int k, int ra, int rb, int s, int nb, bool ok) {
        if (ok) {
            const bool tl = GROUPED && tiled_bpb(FMT) && c_tiled_experts;
            if constexpr (FMT == 13 || FMT == 14) {  // IQ4N / IQ4L: 16 B natural nibbles per 32 weights
                wa = __ldg(reinterpret_cast<const uint4 *>(w4 + u4h_block_off(ra, s, nb, 16, tl)));
                wb = __ldg(reinterpret_cast<const uint4 *>(w4 + u4h_block_off(rb, s, nb, 16, tl)));
                float fa, fb;
                if constexpr (FMT == 13) {  // 8-byte record per 256: d, scales_h, scales_l[4]; scale d * (ls - 32)
                    const auto * kh = reinterpret_cast<const uint8_t *>(sm);
                    auto sc = [&](int r) {
                        const uint2 rr = __ldg(reinterpret_cast<const uint2 *>(kh + (int64_t(r) * (k / 256) + s / 8) * 8));
                        const int sub = s % 8;
                        const int ls = int((rr.y >> (4 * sub)) & 0xF) | int((((rr.x >> 16) >> (2 * sub)) & 3) << 4);
                        return __half2float(__ushort_as_half(uint16_t(rr.x & 0xFFFFu))) * float(ls - 32);
                    };
                    fa = sc(ra), fb = sc(rb);
                } else {  // FP16 scale per 32
                    const auto * sh = reinterpret_cast<const __half *>(sm);
                    fa = __half2float(__ldg(sh + int64_t(ra) * nb + s)), fb = __half2float(__ldg(sh + int64_t(rb) * nb + s));
                }
                pk(__floats2half2_rn(fa, -128.f * fa), sa[0], ma[0]);
                pk(__floats2half2_rn(fb, -128.f * fb), sb[0], mb[0]);
                sa[1] = sa[0], ma[1] = ma[0], sb[1] = sb[0], mb[1] = mb[0];
            } else if constexpr (FMT == 15) {  // I8H: 32 int8 per 32 weights, FP16 scale per 16
                const auto * pa = reinterpret_cast<const uint4 *>(w4 + int64_t(ra) * k + int64_t(s) * 32);
                const auto * pb = reinterpret_cast<const uint4 *>(w4 + int64_t(rb) * k + int64_t(s) * 32);
                wa = __ldg(pa), wa2 = __ldg(pa + 1), wb = __ldg(pb), wb2 = __ldg(pb + 1);
                const auto * sh = reinterpret_cast<const __half *>(sm);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float fa = __half2float(__ldg(sh + int64_t(ra) * (k / 16) + 2 * s + h));
                    const float fb = __half2float(__ldg(sh + int64_t(rb) * (k / 16) + 2 * s + h));
                    pk(__floats2half2_rn(fa, -128.f * fa), sa[h], ma[h]);
                    pk(__floats2half2_rn(fb, -128.f * fb), sb[h], mb[h]);
                }
            } else if constexpr (FMT == 5) {
                const auto * wpa = reinterpret_cast<const uint32_t *>(w4 + u4h_block_off(ra, s, nb, 12, tl));
                const auto * wpb = reinterpret_cast<const uint32_t *>(w4 + u4h_block_off(rb, s, nb, 12, tl));
                const auto * kh = reinterpret_cast<const uint8_t *>(sm);
                wa = make_uint4(__ldg(wpa), __ldg(wpa + 1), __ldg(wpa + 2), 0);
                wb = make_uint4(__ldg(wpb), __ldg(wpb + 1), __ldg(wpb + 2), 0);
                pk(iq3x_sm(kh + (int64_t(ra) * (k / 256) + s / 8) * 4, wa.z), sa[0], ma[0]);
                pk(iq3x_sm(kh + (int64_t(rb) * (k / 256) + s / 8) * 4, wb.z), sb[0], mb[0]);
                sa[1] = sa[0], ma[1] = ma[0], sb[1] = sb[0], mb[1] = mb[0];
            } else if constexpr (FMT == 4) {
                const auto * wpa = reinterpret_cast<const uint32_t *>(w4 + u4h_block_off(ra, s, nb, 12, tl));
                const auto * wpb = reinterpret_cast<const uint32_t *>(w4 + u4h_block_off(rb, s, nb, 12, tl));
                const auto * kh = reinterpret_cast<const uint8_t *>(sm);
                const uint8_t * rca = kh + (int64_t(ra) * (k / 256) + s / 8) * 16, * rcb = kh + (int64_t(rb) * (k / 256) + s / 8) * 16;
                wa = make_uint4(__ldg(wpa), __ldg(wpa + 1), __ldg(wpa + 2), __ldg(rca + 2 + s % 8));
                wb = make_uint4(__ldg(wpb), __ldg(wpb + 1), __ldg(wpb + 2), __ldg(rcb + 2 + s % 8));
                pk(iq3_sm(rca, s % 8), sa[0], ma[0]);
                pk(iq3_sm(rcb, s % 8), sb[0], mb[0]);
                sa[1] = sa[0], ma[1] = ma[0], sb[1] = sb[0], mb[1] = mb[0];
            } else if constexpr (FMT == 2) {
                const auto * wpa = reinterpret_cast<const uint32_t *>(w4 + (int64_t(ra) * (k / 32) + s) * 12);
                const auto * wpb = reinterpret_cast<const uint32_t *>(w4 + (int64_t(rb) * (k / 32) + s) * 12);
                wa = make_uint4(__ldg(wpa), __ldg(wpa + 1), __ldg(wpa + 2), 0);
                wb = make_uint4(__ldg(wpb), __ldg(wpb + 1), __ldg(wpb + 2), 0);
                const auto * kh = reinterpret_cast<const uint8_t *>(sm);
                const uint8_t * rca = kh + (int64_t(ra) * (k / 256) + s / 8) * 20, * rcb = kh + (int64_t(rb) * (k / 256) + s / 8) * 20;
                const float da = __half2float(__ldg(reinterpret_cast<const __half *>(rca + 16)));
                const float db = __half2float(__ldg(reinterpret_cast<const __half *>(rcb + 16)));
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float fa = da * float(int8_t(__ldg(rca + (s % 8) * 2 + h)));
                    const float fb = db * float(int8_t(__ldg(rcb + (s % 8) * 2 + h)));
                    pk(__floats2half2_rn(fa, -4.f * fa), sa[h], ma[h]);
                    pk(__floats2half2_rn(fb, -4.f * fb), sb[h], mb[h]);
                }
            } else if constexpr (FMT >= 8) {
                const auto * wpa = reinterpret_cast<const uint32_t *>(w4 + u4h_block_off(ra, s, nb, ud_bpb(FMT), tl));
                const auto * wpb = reinterpret_cast<const uint32_t *>(w4 + u4h_block_off(rb, s, nb, ud_bpb(FMT), tl));
                wa.x = __ldg(wpa), wb.x = __ldg(wpb);
                if constexpr (ud_bpb(FMT) == 8) wa.y = __ldg(wpa + 1), wb.y = __ldg(wpb + 1);
                const auto * kh = reinterpret_cast<const uint8_t *>(sm);
                ua = kh + (int64_t(ra) * (k / 256) + s / 8) * ud_rb(FMT);
                ub = kh + (int64_t(rb) * (k / 256) + s / 8) * ud_rb(FMT);
            } else if constexpr (FMT == 7) {
                const auto * wpa = reinterpret_cast<const uint32_t *>(w4 + (int64_t(ra) * (k / 32) + s) * 24);
                const auto * wpb = reinterpret_cast<const uint32_t *>(w4 + (int64_t(rb) * (k / 32) + s) * 24);
                wa = make_uint4(__ldg(wpa), __ldg(wpa + 1), __ldg(wpa + 2), __ldg(wpa + 3));
                wb = make_uint4(__ldg(wpb), __ldg(wpb + 1), __ldg(wpb + 2), __ldg(wpb + 3));
                ha = __ldg(wpa + 4), ha2 = __ldg(wpa + 5), hb = __ldg(wpb + 4), hb2 = __ldg(wpb + 5);
                const auto * kh = reinterpret_cast<const uint8_t *>(sm);
                const uint8_t * rca = kh + (int64_t(ra) * (k / 256) + s / 8) * 20, * rcb = kh + (int64_t(rb) * (k / 256) + s / 8) * 20;
                const float da = __half2float(__ldg(reinterpret_cast<const __half *>(rca + 16)));
                const float db = __half2float(__ldg(reinterpret_cast<const __half *>(rcb + 16)));
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float fa = da * float(int8_t(__ldg(rca + (s % 8) * 2 + h)));
                    const float fb = db * float(int8_t(__ldg(rcb + (s % 8) * 2 + h)));
                    pk(__floats2half2_rn(fa, -32.f * fa), sa[h], ma[h]);
                    pk(__floats2half2_rn(fb, -32.f * fb), sb[h], mb[h]);
                }
            } else if constexpr (FMT == 6) {
                const auto * wpa = reinterpret_cast<const uint32_t *>(w4 + (int64_t(ra) * (k / 32) + s) * 20);
                const auto * wpb = reinterpret_cast<const uint32_t *>(w4 + (int64_t(rb) * (k / 32) + s) * 20);
                wa = make_uint4(__ldg(wpa), __ldg(wpa + 1), __ldg(wpa + 2), __ldg(wpa + 3));
                wb = make_uint4(__ldg(wpb), __ldg(wpb + 1), __ldg(wpb + 2), __ldg(wpb + 3));
                ha = __ldg(wpa + 4), hb = __ldg(wpb + 4);
                const auto * kh = reinterpret_cast<const uint8_t *>(sm);
                pk(k4_sm(kh + (int64_t(ra) * (k / 256) + s / 8) * 16, s % 8), sa[0], ma[0]);
                pk(k4_sm(kh + (int64_t(rb) * (k / 256) + s / 8) * 16, s % 8), sb[0], mb[0]);
                sa[1] = sa[0], ma[1] = ma[0], sb[1] = sb[0], mb[1] = mb[0];
            } else {
                wa = __ldg(reinterpret_cast<const uint4 *>(w4 + int64_t(ra) * (k / 2) + int64_t(s) * 16));
                wb = __ldg(reinterpret_cast<const uint4 *>(w4 + int64_t(rb) * (k / 2) + int64_t(s) * 16));
                __half2 pa, pb;
                if constexpr (FMT == 1) {
                    const auto * kh = reinterpret_cast<const uint8_t *>(sm);
                    pa = k4_sm(kh + (int64_t(ra) * (k / 256) + s / 8) * 16, s % 8);
                    pb = k4_sm(kh + (int64_t(rb) * (k / 256) + s / 8) * 16, s % 8);
                } else {
                    pa = sm[int64_t(ra) * nb + s], pb = sm[int64_t(rb) * nb + s];
                }
                pk(pa, sa[0], ma[0]);
                pk(pb, sb[0], mb[0]);
                sa[1] = sa[0], ma[1] = ma[0], sb[1] = sb[0], mb[1] = mb[0];
            }
        }
    }
    // One code word (k 4u..4u+3 of 16-group h) of rows a and b: the int8 kernel's lane t needs only u = t, so the grid
    // formats do one lookup instead of four. Formats without a specialisation take the full decode.
    __device__ __forceinline__ void codes1(int h, int u, int s, bool ok, const uint32_t * grid, uint32_t & ca, uint32_t & cb) {
        if constexpr (FMT == 13 || FMT == 14) {
            const uint32_t a2 = h ? (u < 2 ? wa.z : wa.w) : (u < 2 ? wa.x : wa.y), b2 = h ? (u < 2 ? wb.z : wb.w) : (u < 2 ? wb.x : wb.y);
            ca = iq4nl_codes((a2 >> (16 * (u & 1))) & 0xFFFFu); cb = iq4nl_codes((b2 >> (16 * (u & 1))) & 0xFFFFu);
        } else if constexpr (FMT == 5) {
            const uint32_t qa = h ? wa.y : wa.x, qb = h ? wb.y : wb.x; const int w = 4 * h + u;
            ca = iq3x_code(grid[(qa >> (8 * u)) & 0xFF], iq3x_signs(wa.z, w)); cb = iq3x_code(grid[(qb >> (8 * u)) & 0xFF], iq3x_signs(wb.z, w));
        } else {
            uint32_t wa4[4], wb4[4]; codes(h, s, ok, grid, wa4, wb4); ca = wa4[u]; cb = wb4[u];
        }
    }
    __device__ __forceinline__ void codes(int h, int s, bool ok, const uint32_t * grid, uint32_t wa4[4], uint32_t wb4[4]) {
        if constexpr (FMT == 13 || FMT == 14) {  // weights 16h + 4u .. + 3: nibbles of 16-bit half (u & 1) of word 2h + u / 2
            const uint32_t a2[2] = {h ? wa.z : wa.x, h ? wa.w : wa.y}, b2[2] = {h ? wb.z : wb.x, h ? wb.w : wb.y};
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                wa4[u] = iq4nl_codes((a2[u / 2] >> (16 * (u & 1))) & 0xFFFFu);
                wb4[u] = iq4nl_codes((b2[u / 2] >> (16 * (u & 1))) & 0xFFFFu);
            }
            return;
        } else if constexpr (FMT == 15) {  // int8 -> code + 128
            const uint4 qa = h ? wa2 : wa, qb = h ? wb2 : wb;
            wa4[0] = qa.x ^ 0x80808080u, wa4[1] = qa.y ^ 0x80808080u, wa4[2] = qa.z ^ 0x80808080u, wa4[3] = qa.w ^ 0x80808080u;
            wb4[0] = qb.x ^ 0x80808080u, wb4[1] = qb.y ^ 0x80808080u, wb4[2] = qb.z ^ 0x80808080u, wb4[3] = qb.w ^ 0x80808080u;
            return;
        }
        if constexpr (FMT >= 8) {
            __half2 pa = __floats2half2_rn(0.f, 0.f), pb = pa;
            if (ok) {
                ud_group<FMT>(wa.x, wa.y, ua, s, h, grid, wa4, pa);
                ud_group<FMT>(wb.x, wb.y, ub, s, h, grid, wb4, pb);
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) wa4[u] = wb4[u] = 0;
            }
            pk(pa, sa[h], ma[h]);
            pk(pb, sb[h], mb[h]);
        } else if constexpr (FMT == 5) {
            const uint32_t qa = h ? wa.y : wa.x, qb = h ? wb.y : wb.x;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int w = 4 * h + u;
                wa4[u] = iq3x_code(grid[(qa >> (8 * u)) & 0xFF], iq3x_signs(wa.z, w));
                wb4[u] = iq3x_code(grid[(qb >> (8 * u)) & 0xFF], iq3x_signs(wb.z, w));
            }
        } else if constexpr (FMT == 4) {  // words 4h + u: index = qs byte | qh bit << 8; signs nibble of byte (4h + u) / 2
            const uint32_t qa = h ? wa.y : wa.x, qb = h ? wb.y : wb.x;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int w = 4 * h + u;
                const uint32_t ia = ((qa >> (8 * u)) & 0xFF) | (((wa.w >> w) & 1) << 8);
                const uint32_t ib2 = ((qb >> (8 * u)) & 0xFF) | (((wb.w >> w) & 1) << 8);
                wa4[u] = iq3_code(grid[ia], (wa.z >> (4 * w)) & 0xF);
                wb4[u] = iq3_code(grid[ib2], (wb.z >> (4 * w)) & 0xF);
            }
        } else if constexpr (FMT == 2) {
            const uint32_t la = h ? wa.y : wa.x, lb = h ? wb.y : wb.x;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                wa4[u] = ((la >> (2 * u)) & 0x03030303u) | (((wa.z >> (u + 4 * h)) & 0x01010101u) << 2);
                wb4[u] = ((lb >> (2 * u)) & 0x03030303u) | (((wb.z >> (u + 4 * h)) & 0x01010101u) << 2);
            }
        } else {
            const uint32_t a0 = h ? wa.z : wa.x, a1 = h ? wa.w : wa.y, b0 = h ? wb.z : wb.x, b1 = h ? wb.w : wb.y;
            wa4[0] = a0 & 0x0F0F0F0Fu, wa4[1] = (a0 >> 4) & 0x0F0F0F0Fu, wa4[2] = a1 & 0x0F0F0F0Fu, wa4[3] = (a1 >> 4) & 0x0F0F0F0Fu;
            wb4[0] = b0 & 0x0F0F0F0Fu, wb4[1] = (b0 >> 4) & 0x0F0F0F0Fu, wb4[2] = b1 & 0x0F0F0F0Fu, wb4[3] = (b1 >> 4) & 0x0F0F0F0Fu;
            if constexpr (FMT == 7) {  // high 2 bits of code 4u + j (group h): bits 2 (4u + j) of word h
                auto spread2 = [](uint32_t v) {  // 4 x 2 bits -> bits 4-5 of 4 bytes
                    return ((v & 3u) | ((v & 0xCu) << 6) | ((v & 0x30u) << 12) | ((v & 0xC0u) << 18)) << 4;
                };
                const uint32_t xa = h ? ha2 : ha, xb = h ? hb2 : hb;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    wa4[u] |= spread2((xa >> (8 * u)) & 0xFF);
                    wb4[u] |= spread2((xb >> (8 * u)) & 0xFF);
                }
            }
            if constexpr (FMT == 6) {  // high bit of code 4u + j (group h) is bit 16 h + 4 u + j
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    wa4[u] |= ((((ha >> (16 * h + 4 * u)) & 0xF) * 0x00204081u) & 0x01010101u) << 4;
                    wb4[u] |= ((((hb >> (16 * h + 4 * u)) & 0xF) * 0x00204081u) & 0x01010101u) << 4;
                }
            }
        }
    }
};

#ifndef U4H_DENSE_MINB
#define U4H_DENSE_MINB 1
#endif
template <int NCOL, class OUT = __nv_bfloat16, int MT = 1, int FMT = 0, int PF = 0, bool GROUPED = false>
__global__ void __launch_bounds__(256, GROUPED ? 4 : U4H_DENSE_MINB) u4h_matvec_k(const uint8_t * w4, const __half2 * sm, const __half * xh,
                                                    OUT * y, int rows, int k, int nc, const uint32_t * rid = nullptr,
                                                    int ldy = 0, const int * expert_ids = nullptr, int topk = 1, bool per_expert_input = false,
                                                    const uint8_t * w4b = nullptr, const __half2 * smb = nullptr, OUT * yb = nullptr) {
    if (blockIdx.z) { w4 = w4b; sm = smb; y = yb; }  // second matrix of a paired launch (gate + up)
    int source_row = 0;
    // Dedup (GROUPED, NCOL > 1): expert_ids holds kDedupStride ints per unique expert: slot, columns, input rows[4],
    // output assignments[4]; the expert's weights are read once for all assignments that use it.
    constexpr bool DEDUP = GROUPED && NCOL > 1;
    const int * dmap = nullptr;
    if constexpr (DEDUP) {
        dmap = expert_ids + int64_t(blockIdx.y) * tqllm_cuda::kDedupStride;
        if (dmap[0] < 0) return;
        source_row = dmap[0] * rows;
        nc = dmap[1];
    } else if constexpr (GROUPED) {
        const int group = blockIdx.y;
        if (expert_ids[group] < 0) return;  // assignment skipped by this launch (whole block, before any barrier)
        source_row = expert_ids[group] * rows;
        xh += int64_t(per_expert_input ? group : group / topk) * k;
        y += int64_t(group) * rows;
    }
    const int64_t ld = ldy ? ldy : rows;  // output column stride
    // rid: output row r reads source row rid[r] (a row subset of a larger matrix, results as for a gathered copy)
    // NT tiles of 8 columns share each decoded weight fragment; MT row tiles of 16 per CTA share each activation load
    // (per row the k split, MMA order and 8-warp reduction do not depend on MT: bit-identical results)
    constexpr int NT = (NCOL + 7) / 8;
    __shared__ float red[8][32][4 * NT];
    __shared__ uint32_t grid[ud_grid_words(FMT)];
    if constexpr (FMT == 4) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = iq3s_grid_d[i];
        __syncthreads();
    }
    if constexpr (FMT == 5) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) grid[i] = iq3xxs_grid_d[i];
        __syncthreads();
    }
    if constexpr (FMT >= 8 && FMT <= 11) {
        const uint32_t * src = ud_grid_src<FMT>();
        for (int i = threadIdx.x; i < ud_grid_words(FMT); i += blockDim.x) grid[i] = src[i];
        __syncthreads();
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT;
    const uint32_t NEG1024 = 0xE400E400u;
    const int nb = k / 32;  // a lane owns one 32-weight block (16 B codes + one (s, m) pair) per iteration
    float c[MT][NT][4] = {};
    auto loadx = [&](int s, bool ok, uint4 (&x)[NT][4]) {
#pragma unroll
        for (int tt = 0; tt < NT; ++tt)
#pragma unroll
            for (int i = 0; i < 4; ++i) x[tt][i] = make_uint4(0, 0, 0, 0);
        if (ok) {
#pragma unroll
            for (int tt = 0; tt < NT; ++tt)
                if (8 * tt + g < nc) {
                    const int64_t col = DEDUP ? dmap[2 + 8 * tt + g] : 8 * tt + g;
                    const uint4 * xp = reinterpret_cast<const uint4 *>(xh + col * k + int64_t(s) * 32);
#pragma unroll
                    for (int i = 0; i < 4; ++i) x[tt][i] = __ldg(&xp[i]);
                }
        }
    };
    // PF (MT 1 only): the next block's codes and activations are loaded before this block computes (same math order)
    uint4 xn[PF ? NT : 1][4];
    U4hTile<FMT, GROUPED> qn;
    int ra0 = min(row0 + g, rows - 1), rb0 = min(row0 + g + 8, rows - 1);
    if (rid) ra0 = int(rid[ra0]), rb0 = int(rid[rb0]);
    ra0 += source_row;
    rb0 += source_row;
    if constexpr (PF) {
        const int s = warp * 4 + t;
        loadx(s, s < nb, xn);
        qn.load(w4, sm, k, ra0, rb0, s, nb, s < nb);
    }
    for (int s0 = warp * 4; s0 < nb; s0 += 32) {
        const int s = s0 + t;
        const bool ok = s < nb;
        uint4 x[NT][4];
        U4hTile<FMT, GROUPED> qpf;
        if constexpr (PF) {
#pragma unroll
            for (int tt = 0; tt < NT; ++tt)
#pragma unroll
                for (int i = 0; i < 4; ++i) x[tt][i] = xn[tt][i];
            qpf = qn;
            const int sn = s + 32;
            loadx(sn, sn < nb, xn);
            qn = U4hTile<FMT, GROUPED>();
            qn.load(w4, sm, k, ra0, rb0, sn, nb, sn < nb);
        } else {
            loadx(s, ok, x);
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            int ra = min(row0 + 16 * m + g, rows - 1), rb = min(row0 + 16 * m + g + 8, rows - 1);
            if (rid) ra = int(rid[ra]), rb = int(rid[rb]);
            ra += source_row;
            rb += source_row;
            U4hTile<FMT, GROUPED> q;
            if constexpr (PF) q = qpf;
            else q.load(w4, sm, k, ra, rb, s, nb, ok);
            auto deq = [&](uint32_t w, uint32_t sel, uint32_t su, uint32_t mu) {
                return hfma2u(hadd2u(__byte_perm(w, 0x64646464u, sel), NEG1024), su, mu);
            };
#pragma unroll
            for (int h = 0; h < 2; ++h) {  // the block's two 16-weight groups
                uint32_t wa4[4], wb4[4];
                q.codes(h, s, ok, grid, wa4, wb4);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const uint32_t d0 = deq(wa4[u], 0x5140, q.sa[h], q.ma[h]), d1 = deq(wb4[u], 0x5140, q.sb[h], q.mb[h]),
                                   d2 = deq(wa4[u], 0x5342, q.sa[h], q.ma[h]), d3 = deq(wb4[u], 0x5342, q.sb[h], q.mb[h]);
#pragma unroll
                    for (int tt = 0; tt < NT; ++tt) {
                        const uint4 xa = x[tt][2 * h], xb = x[tt][2 * h + 1];
                        const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                        mma_f16(c[m][tt], d0, d1, d2, d3, xw[2 * u], xw[2 * u + 1]);
                    }
                }
            }
        }
    }
    auto put = [&](int64_t i, float v2) {
        if constexpr (std::is_same_v<OUT, float>) y[i] = v2;
        else y[i] = __float2bfloat16_rn(v2);
    };
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m) __syncthreads();
#pragma unroll
        for (int tt = 0; tt < NT; ++tt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[warp][lane][4 * tt + i] = c[m][tt][i];
        __syncthreads();
        if (warp < NT) {
            const int tt = warp;
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * tt + i];
            const int n0 = 8 * tt + 2 * t, n1 = n0 + 1, r = row0 + 16 * m + g;
            const int64_t o0 = DEDUP ? (n0 < nc ? dmap[6 + n0] : 0) : n0, o1 = DEDUP ? (n1 < nc ? dmap[6 + n1] : 0) : n1;
            if (r < rows) {
                if (n0 < nc) put(o0 * ld + r, v[0]);
                if (n1 < nc) put(o1 * ld + r, v[1]);
            }
            if (r + 8 < rows) {
                if (n0 < nc) put(o0 * ld + r + 8, v[2]);
                if (n1 < nc) put(o1 * ld + r + 8, v[3]);
            }
        }
    }
}

// Prefill GEMM on the u4h-family formats without a dequantised copy: 64 rows x 64 columns per CTA, warp w takes row tile
// w & 3 and k half w >> 2. Each lane decodes its 32-weight block once (U4hTile, the matvec decode) and reuses the
// fragments across 8 n-tiles, so the weights are read once per 64 columns. FP32 accumulate.
template <int FMT, bool GROUPED = false, typename OUT = __nv_bfloat16, bool GATEUP = false>
__global__ void __launch_bounds__(256) u4h_gemm_k(const uint8_t * w4, const __half2 * sm, const __half * xh,
                                                  OUT * y, int rows, int k, int ncols, int ldy, const int * counts = nullptr, const int * groups = nullptr,
                                                  int capacity = 0, int topk = 1, bool per_expert_input = false,
                                                  const uint8_t * w4b = nullptr, const __half2 * smb = nullptr, __half * yh = nullptr) {
    // GATEUP (unused by the engines since the MMQ prefill path; library only): w4/sm = gate, w4b/smb = up (same format and shape); the epilogue writes silu(g) * u as
    // half to yh (the down projection's input), so the gate and up outputs and the separate swiglu pass never exist.
    int source_row = 0;
    if constexpr (GROUPED) {
        const int expert = blockIdx.z;
        ncols = counts[expert];
        if (int(blockIdx.y) * 64 >= ncols) return;
        groups += int64_t(expert) * capacity;
        source_row = expert * rows;
    }
    constexpr int NT = 8, XS = 33;  // x tile: 64 columns x 256 k (32 uint4), row stride padded to 33 uint4
    constexpr bool GRID_SMEM = ud_grid_words(FMT) > 1 && ud_grid_words(FMT) <= 2048;  // IQ1_S's 16 KB grid stays in L1
    __shared__ uint4 xt[64 * XS];
    __shared__ uint32_t gsm[GRID_SMEM ? ud_grid_words(FMT) : 1];
    const uint32_t * grid = gsm;
    if constexpr (FMT == 4) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) gsm[i] = iq3s_grid_d[i];
    } else if constexpr (FMT == 5) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) gsm[i] = iq3xxs_grid_d[i];
    } else if constexpr (FMT >= 8 && FMT <= 11) {
        if constexpr (GRID_SMEM) {
            const uint32_t * src = ud_grid_src<FMT>();
            for (int i = threadIdx.x; i < ud_grid_words(FMT); i += blockDim.x) gsm[i] = src[i];
        } else {
            grid = ud_grid_src<FMT>();
        }
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int rt = warp & 3, ks = warp >> 2;
    const int row0 = blockIdx.x * 64 + rt * 16, col0 = blockIdx.y * 64, nb = k / 32;
    const int ra = source_row + min(row0 + g, rows - 1), rb = source_row + min(row0 + g + 8, rows - 1);
    const uint32_t NEG1024 = 0xE400E400u;
    auto deq = [&](uint32_t w, uint32_t sel, uint32_t su, uint32_t mu) {
        return hfma2u(hadd2u(__byte_perm(w, 0x64646464u, sel), NEG1024), su, mu);
    };
    float c[NT][4] = {};
    [[maybe_unused]] float c2[GATEUP ? NT : 1][4] = {};
    for (int base = 0; base < nb; base += 8) {  // 8 blocks (256 k) per step: warps ks = 0 / 1 take blocks base + 4 ks + t
        __syncthreads();  // previous step's tile reads are done
        for (int i = threadIdx.x; i < 64 * 32; i += 256) {
            const int col = i / 32, q = i % 32, blk = base + q / 4;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (col0 + col < ncols && blk < nb) {
                int input_col = col0 + col;
                if constexpr (GROUPED) {
                    input_col = groups[input_col];
                    if (!per_expert_input) input_col /= topk;
                }
                v = __ldg(reinterpret_cast<const uint4 *>(xh + int64_t(input_col) * k + int64_t(base) * 32) + q);
            }
            xt[col * XS + q] = v;
        }
        __syncthreads();
        const int s = base + ks * 4 + t;
        const bool ok = s < nb;
        U4hTile<FMT, GROUPED> qd;
        qd.load(w4, sm, k, ra, rb, s, nb, ok);
        uint32_t af[8][4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint32_t wa4[4], wb4[4];
            qd.codes(h, s, ok, grid, wa4, wb4);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                af[4 * h + u][0] = deq(wa4[u], 0x5140, qd.sa[h], qd.ma[h]);
                af[4 * h + u][1] = deq(wb4[u], 0x5140, qd.sb[h], qd.mb[h]);
                af[4 * h + u][2] = deq(wa4[u], 0x5342, qd.sa[h], qd.ma[h]);
                af[4 * h + u][3] = deq(wb4[u], 0x5342, qd.sb[h], qd.mb[h]);
            }
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
        if constexpr (GATEUP) {  // the up tile of the same block, same x fragments
            U4hTile<FMT, GROUPED> qu;
            qu.load(w4b, smb, k, ra, rb, s, nb, ok);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t wa4[4], wb4[4];
                qu.codes(h, s, ok, grid, wa4, wb4);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    af[4 * h + u][0] = deq(wa4[u], 0x5140, qu.sa[h], qu.ma[h]);
                    af[4 * h + u][1] = deq(wb4[u], 0x5140, qu.sb[h], qu.mb[h]);
                    af[4 * h + u][2] = deq(wa4[u], 0x5342, qu.sa[h], qu.ma[h]);
                    af[4 * h + u][3] = deq(wb4[u], 0x5342, qu.sb[h], qu.mb[h]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint4 * xp = xt + (nt * 8 + g) * XS + (ks * 4 + t) * 4;
                uint4 x4[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) x4[i] = xp[i];
                const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
#pragma unroll
                for (int u = 0; u < 8; ++u) mma_f16(c2[nt], af[u][0], af[u][1], af[u][2], af[u][3], xw[2 * u], xw[2 * u + 1]);
            }
        }
    }
    __syncthreads();
    float (*red)[32][NT * 4] = reinterpret_cast<float (*)[32][NT * 4]>(xt);  // 16 KB of the 33.8 KB tile
    if (ks == 1)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[rt][lane][nt * 4 + i] = c[nt][i];
    __syncthreads();
    if (ks == 0)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) c[nt][i] += red[rt][lane][nt * 4 + i];
    if constexpr (GATEUP) {  // second round for the up accumulators
        __syncthreads();
        if (ks == 1)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) red[rt][lane][nt * 4 + i] = c2[nt][i];
        __syncthreads();
        if (ks == 0)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) c2[nt][i] += red[rt][lane][nt * 4 + i];
    }
    if (ks == 0) {
        const int r = row0 + g;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int n0 = col0 + nt * 8 + 2 * t, n1 = n0 + 1;
            const auto put = [&](int col, int row, float value, [[maybe_unused]] float up) {
                if (col < ncols) {
                    if constexpr (GROUPED) col = groups[col];
                    if constexpr (GATEUP) {
                        // the same arithmetic as fn_swiglu_bf16_h_groups_k on bf16-rounded gate / up values
                        const float a = __bfloat162float(__float2bfloat16_rn(value)), b = __bfloat162float(__float2bfloat16_rn(up));
                        yh[int64_t(col) * ldy + row] = __float2half_rn(a / (1.f + expf(-a)) * b);
                    } else if constexpr (std::is_same_v<OUT, float>) y[int64_t(col) * ldy + row] = value;
                    else y[int64_t(col) * ldy + row] = __float2bfloat16_rn(value);
                }
            };
            if (r < rows) {
                put(n0, r, c[nt][0], GATEUP ? c2[nt][0] : 0.f);
                put(n1, r, c[nt][1], GATEUP ? c2[nt][1] : 0.f);
            }
            if (r + 8 < rows) {
                put(n0, r + 8, c[nt][2], GATEUP ? c2[nt][2] : 0.f);
                put(n1, r + 8, c[nt][3], GATEUP ? c2[nt][3] : 0.f);
            }
        }
    }
}
template <int FMT, typename OUT = __nv_bfloat16>
static void u4h_gemm_launch(const uint8_t * w4, const void * sm, const __half * xh, OUT * y, int rows, int k,
                            int ncols, cudaStream_t s, int ldy) {
    if (k % 64) throw std::invalid_argument("u4h_gemm: k must be a multiple of 64");
    const dim3 grid((rows + 63) / 64, (ncols + 63) / 64);
    u4h_gemm_k<FMT, false, OUT><<<grid, 256, 0, s>>>(w4, reinterpret_cast<const __half2 *>(sm), xh, y, rows, k, ncols, ldy ? ldy : rows);
}

// ---- INT8-activation prefill GEMM (TQFN_PREFILL_INT8) ----
// Activations quantised per 32-value block (Strata's scheme): x = xs[row][blk] * q (int8), plus the sum of q per 16-group
// (int16) for the weight "min" term. Weights stay in their bank format: value = scale_h * code + min_h per 16-group h
// (U4hTile), with code - 128 as the int8 operand, so dot(value, x) = xs * (scale_h * D + (128 scale_h + min_h) * Q) with
// D = dot(code - 128, q) and Q = sum(q): both exact int32 (|D| <= 127 * 128 * 16), rescaled into fp32 per 16-group, so no
// int32 sum ever crosses a scale boundary. The int32 accumulator starts at the float bits of 12582912 so D's bits are the
// exact float of 12582912 + D (|D| < 2^22). Zero blocks: codes 0, scale 0, output 0.
template <bool TOKEN>
__global__ void quant_act_rows_k(const __half * x, const int * rows_of, const int * counts, int cap, int k,
                                 int8_t * q, float * s, int16_t * qsum) {
    int row;  // rows_of null: row blockIdx.x; else grouped: row rows_of[slot * cap + i] for i < counts[slot]
    if (!rows_of) row = int(blockIdx.x);
    else { const int slot = int(blockIdx.y), i = int(blockIdx.x); if (i >= counts[slot]) return; row = rows_of[slot * cap + i]; }
    const __half * xr = x + int64_t(row) * k; const int nb = k / 32, lane = threadIdx.x % 32;
    __shared__ float tok_m;
    if constexpr (TOKEN) {  // one scale per token: absmax over the row
        float m = 0.f;
        for (int i = threadIdx.x; i < k; i += blockDim.x) m = fmaxf(m, fabsf(__half2float(xr[i])));
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
        __shared__ float red[32];
        if (lane == 0) red[threadIdx.x / 32] = m;
        __syncthreads();
        if (threadIdx.x == 0) { float mm = 0.f; for (unsigned i = 0; i < blockDim.x / 32; ++i) mm = fmaxf(mm, red[i]); tok_m = mm; }
        __syncthreads();
    }
    for (int blk = threadIdx.x / 32; blk < nb; blk += blockDim.x / 32) {
        const float v = __half2float(xr[blk * 32 + lane]);
        float m = fabsf(v);
        if constexpr (TOKEN) m = tok_m;
        else for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
        const float inv = m > 0.f ? 127.f / m : 0.f;
        const int qi = __float2int_rn(v * inv);
        q[int64_t(row) * k + blk * 32 + lane] = int8_t(qi);
        int sum = qi;
        for (int o = 8; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);  // within each 16-lane half
        if (lane == 0) { s[int64_t(row) * nb + blk] = m / 127.f; qsum[(int64_t(row) * nb + blk) * 2] = int16_t(sum); }
        if (lane == 16) qsum[(int64_t(row) * nb + blk) * 2 + 1] = int16_t(sum);
    }
}
__device__ __forceinline__ void mma_s8_k16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    constexpr int M = 0x4B400000;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(b0), "r"(M), "r"(M), "r"(M), "r"(M));
}
__device__ __forceinline__ float h2lo(uint32_t v) { return __half2float(__ushort_as_half(uint16_t(v & 0xFFFFu))); }
// 64 rows x 64 columns per CTA, 8 warps: warp w takes row tile w & 3 and k half w >> 2 (blocks 4 (w >> 2) .. + 3 of each
// 256-k step). Unlike u4h_gemm_k, the 4 lanes of a row group work on the same block (one scale per MMA): lane t holds
// k 4t..4t+3 of each 16-group (wa4[t] of U4hTile::codes). Formats with one scale per 32 use one k32 MMA per block; formats
// with a scale per 16 use two k16 MMAs. Where min = -OFF * scale exactly (power-of-two OFF, exact in half) the int8
// operand is code - OFF and no min term is needed; the other formats use code - 128 and add (128 scale + min) * sum(q).
__device__ __forceinline__ void mma_s8_k32(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    constexpr int M = 0x4B400000;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(M), "r"(M), "r"(M), "r"(M));
}
template <int FMT> __host__ __device__ constexpr int i8_offset() {  // min == -offset * scale: 0 = generic (min term)
    // IQ4_NL / IQ4_XS: code - 128 is exact (verified against the fp16 kernel and an fp64 oracle). Every other format takes the
    // generic path: operand code - 128 plus (128 scale + min) * sum(q) from the kernel's own (scale, min), exact by construction
    // whatever the format's code offset is (the per-format offset table was wrong for IQ2_S and unverified for IQ3_S).
    return (FMT == 13 || FMT == 14) ? 128 : 0;
}
template <int FMT> __host__ __device__ constexpr bool i8_one_scale() { return FMT == 1 || FMT == 5 || FMT == 6 || FMT == 13 || FMT == 14; }
// TOKEN: one activation scale per token (xs[row][0]), applied once at the store; the per-block epilogue is then weight
// scale only (8 scalar instructions per k32 MMA instead of 14). Per-32 scales (TOKEN = false) isolate outliers per block.
template <int FMT, bool GROUPED, bool TOKEN = false, typename OUT = __nv_bfloat16>
__global__ void __launch_bounds__(256) u4h_i8_gemm_k(const uint8_t * w4, const __half2 * sm, const int8_t * xq, const float * xs,
                                                     const int16_t * xsum, OUT * y, int rows, int k, int ncols, int ldy,
                                                     const int * counts, const int * groups, int capacity, int topk, bool per_expert_input) {
    int source_row = 0;
    if constexpr (GROUPED) {
        const int expert = blockIdx.z;
        ncols = counts[expert];
        if (int(blockIdx.y) * 64 >= ncols) return;
        groups += int64_t(expert) * capacity;
        source_row = expert * rows;
    }
    constexpr int KS = 256, XS = KS + 16, NT = 8, NB = KS / 32;
    constexpr int OFF = i8_offset<FMT>();
    constexpr bool ONE = i8_one_scale<FMT>(), MINTERM = OFF == 0;
    constexpr uint32_t SUB = MINTERM ? 0x80808080u : uint32_t(OFF) * 0x01010101u;
    constexpr bool GRID_SMEM = ud_grid_words(FMT) > 1 && ud_grid_words(FMT) <= 2048;
    __shared__ __align__(16) int8_t xt[64 * XS];
    __shared__ float st[64][NB];
    __shared__ int16_t sq[64][NB * 2];
    __shared__ uint32_t gsm[GRID_SMEM ? ud_grid_words(FMT) : 1];
    __shared__ int cols_in[64];
    const uint32_t * grid = gsm;
    if constexpr (FMT == 4) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) gsm[i] = iq3s_grid_d[i];
    } else if constexpr (FMT == 5) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) gsm[i] = iq3xxs_grid_d[i];
    } else if constexpr (FMT >= 8 && FMT <= 11) {
        if constexpr (GRID_SMEM) {
            const uint32_t * src = ud_grid_src<FMT>();
            for (int i = threadIdx.x; i < ud_grid_words(FMT); i += blockDim.x) gsm[i] = src[i];
        } else {
            grid = ud_grid_src<FMT>();
        }
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int rt = warp & 3, ks = warp >> 2;
    const int row0 = blockIdx.x * 64 + rt * 16, col0 = blockIdx.y * 64, nb = k / 32;
    const int ra = source_row + min(row0 + g, rows - 1), rb = source_row + min(row0 + g + 8, rows - 1);
    if (threadIdx.x < 64) {
        int col = col0 + int(threadIdx.x), in = -1;
        if (col < ncols) { in = col; if constexpr (GROUPED) { in = groups[col]; if (!per_expert_input) in /= topk; } }
        cols_in[threadIdx.x] = in;
    }
    auto sub = [](uint32_t code) { return MINTERM || OFF == 128 ? code ^ 0x80808080u : __vsub4(code, SUB); };
    float acc[NT][4] = {};
    for (int base = 0; base < k; base += KS) {
        __syncthreads();
        for (int i = threadIdx.x; i < 64 * (KS / 16); i += 256) {
            const int col = i / (KS / 16), qd16 = i % (KS / 16), in = cols_in[col];
            uint4 v = make_uint4(0, 0, 0, 0);
            if (in >= 0 && base + qd16 * 16 < k) v = __ldg(reinterpret_cast<const uint4 *>(xq + int64_t(in) * k + base) + qd16);
            *reinterpret_cast<uint4 *>(xt + col * XS + qd16 * 16) = v;
        }
        for (int i = threadIdx.x; i < 64 * NB; i += 256) {
            const int col = i / NB, b = i % NB, in = cols_in[col];
            const bool ok = in >= 0 && base + b * 32 < k;
            st[col][b] = ok ? xs[int64_t(in) * nb + base / 32 + b] : 0.f;
            if constexpr (MINTERM) {
                sq[col][2 * b] = ok ? xsum[(int64_t(in) * nb + base / 32 + b) * 2] : int16_t(0);
                sq[col][2 * b + 1] = ok ? xsum[(int64_t(in) * nb + base / 32 + b) * 2 + 1] : int16_t(0);
            }
        }
        __syncthreads();
#pragma unroll
        for (int bb = 0; bb < 4; ++bb) {
            const int b = ks * 4 + bb, s = base / 32 + b;
            if (s >= nb) break;
            U4hTile<FMT, GROUPED> qd;
            qd.load(w4, sm, k, ra, rb, s, nb, true);
            uint32_t wa0[4], wb0[4], wa1[4], wb1[4];  // only element t is used: the lane's own 4 codes per 16-group
            qd.codes1(0, t, s, true, grid, wa0[t], wb0[t]);
            qd.codes1(1, t, s, true, grid, wa1[t], wb1[t]);
            if constexpr (ONE) {  // one k32 MMA per block
                const uint32_t a0 = sub(wa0[t]), a1 = sub(wb0[t]), a2 = sub(wa1[t]), a3 = sub(wb1[t]);
                const float sa = h2lo(qd.sa[0]), sb = h2lo(qd.sb[0]);
                [[maybe_unused]] float ca = 0.f, cb = 0.f;
                if constexpr (MINTERM) { ca = fmaf(128.f, sa, h2lo(qd.ma[0])); cb = fmaf(128.f, sb, h2lo(qd.mb[0])); }
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int8_t * xp = xt + (nt * 8 + g) * XS + b * 32 + 4 * t;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t *>(xp), b1 = *reinterpret_cast<const uint32_t *>(xp + 16);
                    int d[4];
                    mma_s8_k32(d, a0, a1, a2, a3, b0, b1);
                    const int c0 = nt * 8 + 2 * t, c1 = c0 + 1;
                    if constexpr (TOKEN) {
                        acc[nt][0] = fmaf(sa, __int_as_float(d[0]) - 12582912.f, acc[nt][0]); acc[nt][1] = fmaf(sa, __int_as_float(d[1]) - 12582912.f, acc[nt][1]);
                        acc[nt][2] = fmaf(sb, __int_as_float(d[2]) - 12582912.f, acc[nt][2]); acc[nt][3] = fmaf(sb, __int_as_float(d[3]) - 12582912.f, acc[nt][3]);
                        if constexpr (MINTERM) {
                            const float q0 = float(sq[c0][2 * b]) + float(sq[c0][2 * b + 1]), q1 = float(sq[c1][2 * b]) + float(sq[c1][2 * b + 1]);
                            acc[nt][0] = fmaf(ca, q0, acc[nt][0]); acc[nt][1] = fmaf(ca, q1, acc[nt][1]); acc[nt][2] = fmaf(cb, q0, acc[nt][2]); acc[nt][3] = fmaf(cb, q1, acc[nt][3]);
                        }
                        continue;
                    }
                    const float x0 = st[c0][b], x1 = st[c1][b];
                    float d0 = sa * (__int_as_float(d[0]) - 12582912.f), d1 = sa * (__int_as_float(d[1]) - 12582912.f);
                    float d2 = sb * (__int_as_float(d[2]) - 12582912.f), d3 = sb * (__int_as_float(d[3]) - 12582912.f);
                    if constexpr (MINTERM) {
                        const float q0 = float(sq[c0][2 * b]) + float(sq[c0][2 * b + 1]), q1 = float(sq[c1][2 * b]) + float(sq[c1][2 * b + 1]);
                        d0 = fmaf(ca, q0, d0); d1 = fmaf(ca, q1, d1); d2 = fmaf(cb, q0, d2); d3 = fmaf(cb, q1, d3);
                    }
                    acc[nt][0] = fmaf(x0, d0, acc[nt][0]); acc[nt][1] = fmaf(x1, d1, acc[nt][1]);
                    acc[nt][2] = fmaf(x0, d2, acc[nt][2]); acc[nt][3] = fmaf(x1, d3, acc[nt][3]);
                }
            } else {  // a scale per 16: two k16 MMAs per block
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const uint32_t a0 = sub(hh ? wa1[t] : wa0[t]), a1 = sub(hh ? wb1[t] : wb0[t]);
                    const float sa = h2lo(qd.sa[hh]), sb = h2lo(qd.sb[hh]);
                    [[maybe_unused]] float ca = 0.f, cb = 0.f;
                    if constexpr (MINTERM) { ca = fmaf(128.f, sa, h2lo(qd.ma[hh])); cb = fmaf(128.f, sb, h2lo(qd.mb[hh])); }
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        const uint32_t b0 = *reinterpret_cast<const uint32_t *>(xt + (nt * 8 + g) * XS + b * 32 + hh * 16 + 4 * t);
                        int d[4];
                        mma_s8_k16(d, a0, a1, b0);
                        const int c0 = nt * 8 + 2 * t, c1 = c0 + 1;
                        if constexpr (TOKEN) {
                            acc[nt][0] = fmaf(sa, __int_as_float(d[0]) - 12582912.f, acc[nt][0]); acc[nt][1] = fmaf(sa, __int_as_float(d[1]) - 12582912.f, acc[nt][1]);
                            acc[nt][2] = fmaf(sb, __int_as_float(d[2]) - 12582912.f, acc[nt][2]); acc[nt][3] = fmaf(sb, __int_as_float(d[3]) - 12582912.f, acc[nt][3]);
                            if constexpr (MINTERM) {
                                const float q0 = float(sq[c0][2 * b + hh]), q1 = float(sq[c1][2 * b + hh]);
                                acc[nt][0] = fmaf(ca, q0, acc[nt][0]); acc[nt][1] = fmaf(ca, q1, acc[nt][1]); acc[nt][2] = fmaf(cb, q0, acc[nt][2]); acc[nt][3] = fmaf(cb, q1, acc[nt][3]);
                            }
                            continue;
                        }
                        const float x0 = st[c0][b], x1 = st[c1][b];
                        float d0 = sa * (__int_as_float(d[0]) - 12582912.f), d1 = sa * (__int_as_float(d[1]) - 12582912.f);
                        float d2 = sb * (__int_as_float(d[2]) - 12582912.f), d3 = sb * (__int_as_float(d[3]) - 12582912.f);
                        if constexpr (MINTERM) {
                            const float q0 = float(sq[c0][2 * b + hh]), q1 = float(sq[c1][2 * b + hh]);
                            d0 = fmaf(ca, q0, d0); d1 = fmaf(ca, q1, d1); d2 = fmaf(cb, q0, d2); d3 = fmaf(cb, q1, d3);
                        }
                        acc[nt][0] = fmaf(x0, d0, acc[nt][0]); acc[nt][1] = fmaf(x1, d1, acc[nt][1]);
                        acc[nt][2] = fmaf(x0, d2, acc[nt][2]); acc[nt][3] = fmaf(x1, d3, acc[nt][3]);
                    }
                }
            }
        }
    }
    __syncthreads();
    float (*red)[32][NT * 4] = reinterpret_cast<float (*)[32][NT * 4]>(xt);  // 16 KB of the 17 KB tile
    if (ks == 1)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[rt][lane][nt * 4 + i] = acc[nt][i];
    __syncthreads();
    if (ks == 0) {
        const int r = row0 + g;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int n0 = col0 + nt * 8 + 2 * t, n1 = n0 + 1;
            const float v0 = acc[nt][0] + red[rt][lane][nt * 4 + 0], v1 = acc[nt][1] + red[rt][lane][nt * 4 + 1];
            const float v2 = acc[nt][2] + red[rt][lane][nt * 4 + 2], v3 = acc[nt][3] + red[rt][lane][nt * 4 + 3];
            const auto put = [&](int col, int row, float value) {
                if (col < ncols) {
                    if constexpr (TOKEN) { const int in = cols_in[col - col0]; value *= in >= 0 ? xs[int64_t(in) * nb] : 0.f; }
                    if constexpr (GROUPED) col = groups[col];
                    if constexpr (std::is_same_v<OUT, float>) y[int64_t(col) * ldy + row] = value;
                    else y[int64_t(col) * ldy + row] = __float2bfloat16_rn(value);
                }
            };
            if (r < rows) { put(n0, r, v0); put(n1, r, v1); }
            if (r + 8 < rows) { put(n0, r + 8, v2); put(n1, r + 8, v3); }
        }
    }
}

// ---- INT8 expert GEMM v3 (TQFN_PREFILL_INT8=3 per-32 activation scales, 4 per-token) ----
// Structure after Strata's fused IQ kernel (moe_fused_iq.cu, credited; own implementation): the weights of a 128-row x 64-k
// stage are decoded ONCE per CTA into shared memory as int8 (one thread per (row, 32-block)), with their per-16 scale and
// min term, double-buffered; the activations of the 128-token tile arrive by cp.async one stage ahead. 8 warps = 4 row
// groups (32 rows = 2 m16) x 2 token groups (64 tokens = 8 n8): 16 tensor-core ops per warp per 32 k. Per-32-scale formats
// use one m16n8k32 per block, per-16 formats two m16n8k16. Same exact int -> float trick as u4h_i8_gemm_k.
__device__ __forceinline__ void cp_async16(void * smem, const void * gmem, bool pred) {
    const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
template <int FMT, bool TOKEN>
__global__ void __launch_bounds__(256) u4h_i8v3_gemm_k(const uint8_t * w4, const __half2 * sm, const int8_t * xq, const float * xs,
                                                       const int16_t * xsum, __nv_bfloat16 * y, int rows, int k,
                                                       const int * counts, const int * groups, int capacity, int topk, bool per_expert_input) {
    // tokens per CTA: 128 (two warp columns of 64); the generic formats carry min-term tables, so 64 keeps static smem < 48 KB
    constexpr int BM = 128, BN = i8_offset<FMT>() == 0 ? 64 : 128, KS = 64, NBS = KS / 32, WN = BN / 2, NI = WN / 8;
    constexpr int WLD = KS + 16, XLD = KS + 16;                 // padded row strides (bytes)
    constexpr int OFF = i8_offset<FMT>();
    constexpr bool ONE = i8_one_scale<FMT>(), MINTERM = OFF == 0;
    constexpr bool GRID_SMEM = ud_grid_words(FMT) > 1 && ud_grid_words(FMT) <= 1024;  // IQ2_S's 8 KB grid stays in L1 here
    const int expert = blockIdx.z, ncols = counts[expert];
    if (int(blockIdx.y) * BN >= ncols) return;
    groups += int64_t(expert) * capacity;
    const int source_row = expert * rows, row_tile = blockIdx.x * BM, col0 = blockIdx.y * BN, nb = k / 32;
    __shared__ __align__(16) int8_t wsm[2][BM * WLD];
    __shared__ float wsc[2][BM][NBS * 2];                       // weight scale per 16-group
    __shared__ float wmc[2][MINTERM ? BM : 1][NBS * 2];         // (128 scale + min) per 16-group (generic formats)
    __shared__ __align__(16) int8_t xsmem[2][BN * XLD];
    __shared__ float xsc[2][BN][NBS];                           // activation scale per 32 (TOKEN: unused)
    __shared__ float xqs[2][MINTERM ? BN : 1][NBS * 2];         // sum of codes per 16 (generic formats)
    __shared__ int cols_in[BN];
    __shared__ uint32_t gsm[GRID_SMEM ? ud_grid_words(FMT) : 1];
    const uint32_t * grid = gsm;
    if constexpr (FMT == 4) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) gsm[i] = iq3s_grid_d[i];
    } else if constexpr (FMT == 5) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) gsm[i] = iq3xxs_grid_d[i];
    } else if constexpr (FMT >= 8 && FMT <= 11) {
        if constexpr (GRID_SMEM) {
            const uint32_t * src = ud_grid_src<FMT>();
            for (int i = threadIdx.x; i < ud_grid_words(FMT); i += blockDim.x) gsm[i] = src[i];
        } else {
            grid = ud_grid_src<FMT>();
        }
    }
    for (int i = threadIdx.x; i < BN; i += blockDim.x) {
        const int col = col0 + i; int in = -1;
        if (col < ncols) { in = groups[col]; if (!per_expert_input) in /= topk; }
        cols_in[i] = in;
    }
    __syncthreads();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int wr = warp & 3, wc = warp >> 2;                    // row group (32 rows), token group (64 tokens)
    // stage loader: weights decoded by thread (row = tid >> 1, block = tid & 1); activations by cp.async
    auto load_stage = [&](int stage, int buf) {
        const int kb = stage * KS;
        {
            const int r = tid >> 1, j = tid & 1, s = kb / 32 + j, row = row_tile + r;
            const bool ok = row < rows && s < nb;
            const int ra = source_row + min(row, rows - 1);
            U4hTile<FMT, true> qd;
            qd.load(w4, sm, k, ra, ra, ok ? s : 0, nb, ok);
            uint32_t w0[4], w1[4], d0[4], d1[4];
            qd.codes(0, ok ? s : 0, ok, grid, w0, d0);
            qd.codes(1, ok ? s : 0, ok, grid, w1, d1);
            uint32_t * dst = reinterpret_cast<uint32_t *>(&wsm[buf][r * WLD + j * 32]);
#pragma unroll
            for (int u = 0; u < 4; ++u) { dst[u] = ok ? (w0[u] ^ 0x80808080u) : 0u; dst[4 + u] = ok ? (w1[u] ^ 0x80808080u) : 0u; }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float sc = ok ? h2lo(qd.sa[h]) : 0.f;
                wsc[buf][r][2 * j + h] = sc;
                if constexpr (MINTERM) wmc[buf][r][2 * j + h] = ok ? fmaf(128.f, sc, h2lo(qd.ma[h])) : 0.f;
            }
        }
        for (int i = tid; i < BN * (KS / 16); i += 256) {        // 128 tokens x 4 x 16 bytes
            const int c = i / (KS / 16), q = i % (KS / 16), in = cols_in[c];
            const bool ok = in >= 0 && kb + q * 16 < k;
            cp_async16(&xsmem[buf][c * XLD + q * 16], ok ? static_cast<const void *>(xq + int64_t(in) * k + kb + q * 16) : static_cast<const void *>(xq), ok);
        }
        for (int i = tid; i < BN * NBS; i += 256) {
            const int c = i / NBS, j = i % NBS, in = cols_in[c], s = kb / 32 + j;
            const bool ok = in >= 0 && s < nb;
            if constexpr (!TOKEN) xsc[buf][c][j] = ok ? xs[int64_t(in) * nb + s] : 0.f;
            if constexpr (MINTERM) {
                xqs[buf][c][2 * j] = ok ? float(xsum[(int64_t(in) * nb + s) * 2]) : 0.f;
                xqs[buf][c][2 * j + 1] = ok ? float(xsum[(int64_t(in) * nb + s) * 2 + 1]) : 0.f;
            }
        }
        cp_async_commit();
    };
    float acc[2][NI][4] = {};
    const int nstages = (k + KS - 1) / KS;
    load_stage(0, 0);
    for (int st = 0; st < nstages; ++st) {
        const int buf = st & 1;
        cp_async_wait0();
        __syncthreads();
        if (st + 1 < nstages) load_stage(st + 1, buf ^ 1);       // overlaps with the MMAs below
#pragma unroll
        for (int j = 0; j < NBS; ++j) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r0 = wr * 32 + mi * 16 + g, r1 = r0 + 8;
                const int8_t * wa = &wsm[buf][r0 * WLD + j * 32], * wb = &wsm[buf][r1 * WLD + j * 32];
                const uint32_t a0 = *reinterpret_cast<const uint32_t *>(wa + 4 * t), a1 = *reinterpret_cast<const uint32_t *>(wb + 4 * t);
                const uint32_t a2 = *reinterpret_cast<const uint32_t *>(wa + 16 + 4 * t), a3 = *reinterpret_cast<const uint32_t *>(wb + 16 + 4 * t);
                const float s0a = wsc[buf][r0][2 * j], s1a = wsc[buf][r0][2 * j + 1], s0b = wsc[buf][r1][2 * j], s1b = wsc[buf][r1][2 * j + 1];
                [[maybe_unused]] float m0a = 0.f, m1a = 0.f, m0b = 0.f, m1b = 0.f;
                if constexpr (MINTERM) { m0a = wmc[buf][r0][2 * j]; m1a = wmc[buf][r0][2 * j + 1]; m0b = wmc[buf][r1][2 * j]; m1b = wmc[buf][r1][2 * j + 1]; }
#pragma unroll
                for (int ni = 0; ni < NI; ++ni) {
                    const int c0 = wc * WN + ni * 8 + g;                     // token row of the B fragment
                    const int8_t * xp = &xsmem[buf][c0 * XLD + j * 32];
                    const uint32_t b0 = *reinterpret_cast<const uint32_t *>(xp + 4 * t), b1 = *reinterpret_cast<const uint32_t *>(xp + 16 + 4 * t);
                    const int oc0 = wc * WN + ni * 8 + 2 * t, oc1 = oc0 + 1;  // output columns of this lane
                    float x0 = 1.f, x1 = 1.f;
                    if constexpr (!TOKEN) { x0 = xsc[buf][oc0][j]; x1 = xsc[buf][oc1][j]; }
                    float v[4];
                    if constexpr (ONE && !MINTERM) {
                        int d[4]; mma_s8_k32(d, a0, a1, a2, a3, b0, b1);
                        v[0] = s0a * (__int_as_float(d[0]) - 12582912.f); v[1] = s0a * (__int_as_float(d[1]) - 12582912.f);
                        v[2] = s0b * (__int_as_float(d[2]) - 12582912.f); v[3] = s0b * (__int_as_float(d[3]) - 12582912.f);
                    } else {
                        int d[4], e[4]; mma_s8_k16(d, a0, a1, b0); mma_s8_k16(e, a2, a3, b1);
                        v[0] = fmaf(s0a, __int_as_float(d[0]) - 12582912.f, s1a * (__int_as_float(e[0]) - 12582912.f));
                        v[1] = fmaf(s0a, __int_as_float(d[1]) - 12582912.f, s1a * (__int_as_float(e[1]) - 12582912.f));
                        v[2] = fmaf(s0b, __int_as_float(d[2]) - 12582912.f, s1b * (__int_as_float(e[2]) - 12582912.f));
                        v[3] = fmaf(s0b, __int_as_float(d[3]) - 12582912.f, s1b * (__int_as_float(e[3]) - 12582912.f));
                        if constexpr (MINTERM) {
                            const float q00 = xqs[buf][oc0][2 * j], q01 = xqs[buf][oc0][2 * j + 1], q10 = xqs[buf][oc1][2 * j], q11 = xqs[buf][oc1][2 * j + 1];
                            v[0] = fmaf(m0a, q00, fmaf(m1a, q01, v[0])); v[1] = fmaf(m0a, q10, fmaf(m1a, q11, v[1]));
                            v[2] = fmaf(m0b, q00, fmaf(m1b, q01, v[2])); v[3] = fmaf(m0b, q10, fmaf(m1b, q11, v[3]));
                        }
                    }
                    acc[mi][ni][0] = fmaf(x0, v[0], acc[mi][ni][0]); acc[mi][ni][1] = fmaf(x1, v[1], acc[mi][ni][1]);
                    acc[mi][ni][2] = fmaf(x0, v[2], acc[mi][ni][2]); acc[mi][ni][3] = fmaf(x1, v[3], acc[mi][ni][3]);
                }
            }
        }
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int r = row_tile + wr * 32 + mi * 16 + g + (q >= 2 ? 8 : 0);
                const int cl = wc * WN + ni * 8 + 2 * t + (q & 1), col = col0 + cl;
                if (r >= rows || col >= ncols) continue;
                float value = acc[mi][ni][q];
                if constexpr (TOKEN) { const int in = cols_in[cl]; value *= in >= 0 ? xs[int64_t(in) * nb] : 0.f; }
                y[int64_t(groups[col]) * rows + r] = __float2bfloat16_rn(value);
            }
}

// i8h -> u4h (draft-only copies): per 32 weights, asymmetric min / max 4-bit, codes in the u4h nibble order.
__global__ void u4h_from_i8h_k(const int8_t * w8, const __half * sc, uint8_t * w4, __half2 * sm, int64_t nblk) {
    const int64_t b = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    float v[32], lo = 1e30f, hi = -1e30f;
    for (int i = 0; i < 32; ++i) {
        v[i] = float(w8[b * 32 + i]) * __half2float(sc[b * 2 + i / 16]);
        lo = fminf(lo, v[i]);
        hi = fmaxf(hi, v[i]);
    }
    const __half hs = __float2half_rn((hi - lo) / 15.f), hm = __float2half_rn(lo);
    const float s = __half2float(hs), m = __half2float(hm);
    uint8_t out[16] = {};
    for (int e = 0; e < 32; ++e) {
        const int q = s > 0.f ? max(0, min(15, __float2int_rn((v[e] - m) / s))) : 0;
        const int grp = e / 16, ei = e % 16, u = ei / 4, j = ei % 4;
        out[grp * 8 + (u / 2) * 4 + j] |= uint8_t(q << ((u % 2) * 4));
    }
    for (int i = 0; i < 16; ++i) w4[b * 16 + i] = out[i];
    sm[b] = __halves2half2(hs, hm);
}

// Prefill GEMM over the repacked TQ3_4S layout: y[n][r] for up to 64 columns per grid.y slice. CTA = 64 rows (warp w:
// row tile w & 3, k half w >> 2). Weight decode is exactly tq3_matvec_h_k's (same numerics per weight); each lane
// decodes its block once and reuses the A fragments across 8 n-tiles, so the weight stream is read once per 64 columns.
template <bool ACC16, bool EXPERT = false>
__global__ void __launch_bounds__(256) tq3_gemm_h_k(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows,
                                                    int k, int ncols, uint32_t cblo_u, uint32_t cbhi_u,
                                                    const int * counts = nullptr, const int * groups = nullptr,
                                                    int capacity = 0, int topk = 1, bool per_expert = false) {
    if constexpr (EXPERT) {
        ncols = counts[blockIdx.z];
        if (int(blockIdx.y) * 64 >= ncols) return;
        w += int64_t(blockIdx.z) * rows * (k / 32);
        groups += int64_t(blockIdx.z) * capacity;
    }
    constexpr int NT = 8;  // n-tiles of 8 columns
    __shared__ float red[4][32][NT * 4];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int rt = warp & 3, ks = warp >> 2;
    const int row0 = blockIdx.x * 64 + rt * 16, col0 = blockIdx.y * 64, nblk = k / 32;
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    const uint32_t NEG1152 = 0xE480E480u;
    float c[NT][4] = {};
    for (int b0 = ks * 4; b0 < nblk; b0 += 8) {
        const int b = b0 + t;
        const bool ok = b < nblk;
        const uint4 qa = ok ? __ldg(&w[int64_t(ra) * nblk + b]) : make_uint4(0, 0, 0, 0);
        const uint4 qb = ok ? __ldg(&w[int64_t(rb) * nblk + b]) : make_uint4(0, 0, 0, 0);
        uint32_t sa[4], oa[4], sb[4], ob[4];
#pragma unroll
        for (int gi = 0; gi < 4; ++gi) {
            const uint32_t ba = (qa.x >> (8 * gi)) & 255u, bb = (qb.x >> (8 * gi)) & 255u;
            const uint32_t ha = ba ? (ba << 5) + 0x1800u : 0u, hb = bb ? (bb << 5) + 0x1800u : 0u;
            sa[gi] = ha | (ha << 16);
            sb[gi] = hb | (hb << 16);
            oa[gi] = hmul2u(sa[gi], NEG1152);
            ob[gi] = hmul2u(sb[gi], NEG1152);
        }
        const uint32_t sla = ((qa.y >> 3) & 0x1111u) | ((qa.z >> 2) & 0x2222u) | ((qa.w >> 1) & 0x4444u);
        const uint32_t slb = ((qb.y >> 3) & 0x1111u) | ((qb.z >> 2) & 0x2222u) | ((qb.w >> 1) & 0x4444u);
        const uint32_t sla2 = ((qa.y >> 19) & 0x1111u) | ((qa.z >> 18) & 0x2222u) | ((qa.w >> 17) & 0x4444u);
        const uint32_t slb2 = ((qb.y >> 19) & 0x1111u) | ((qb.z >> 18) & 0x2222u) | ((qb.w >> 17) & 0x4444u);
        const uint32_t sela[8] = {qa.y, qa.y >> 16, qa.z, qa.z >> 16, qa.w, qa.w >> 16, sla, sla2};
        const uint32_t selb[8] = {qb.y, qb.y >> 16, qb.z, qb.z >> 16, qb.w, qb.w >> 16, slb, slb2};
        uint32_t af[8][4];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
            const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
            const int gi = u / 2;
            af[u][0] = hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]);
            af[u][1] = hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]);
            af[u][2] = hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]);
            af[u][3] = hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]);
        }
        uint32_t c16[NT][2] = {};  // ACC16: this iteration's 128-k segment accumulated in FP16 (2x tensor rate)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int col = col0 + nt * 8 + g;
            uint4 x4[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
            if (ok && col < ncols) {
                int input_col = col;
                if constexpr (EXPERT) input_col = per_expert ? groups[col] : groups[col] / topk;
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(input_col) * k + int64_t(b) * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) x4[i] = __ldg(&xp[i]);
            }
            const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                if constexpr (ACC16) mma_f16h(c16[nt], af[u][0], af[u][1], af[u][2], af[u][3], xw[2 * u], xw[2 * u + 1]);
                else mma_f16(c[nt], af[u][0], af[u][1], af[u][2], af[u][3], xw[2 * u], xw[2 * u + 1]);
            }
        }
        if constexpr (ACC16)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&c16[nt][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&c16[nt][1]));
                c[nt][0] += lo.x;
                c[nt][1] += lo.y;
                c[nt][2] += hi.x;
                c[nt][3] += hi.y;
            }
    }
    if (ks == 1)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[rt][lane][nt * 4 + i] = c[nt][i];
    __syncthreads();
    if (ks == 0) {
        const float inv = 1.f / TQ3_CB_Q;
        const int r = row0 + g;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int n0 = col0 + nt * 8 + 2 * t, n1 = n0 + 1;
            int out0 = n0, out1 = n1;
            if constexpr (EXPERT) {
                if (n0 < ncols) out0 = groups[n0];
                if (n1 < ncols) out1 = groups[n1];
            }
            const float v0 = (c[nt][0] + red[rt][lane][nt * 4 + 0]) * inv, v1 = (c[nt][1] + red[rt][lane][nt * 4 + 1]) * inv;
            const float v2 = (c[nt][2] + red[rt][lane][nt * 4 + 2]) * inv, v3 = (c[nt][3] + red[rt][lane][nt * 4 + 3]) * inv;
            if (r < rows) {
                if (n0 < ncols) y[int64_t(out0) * rows + r] = __float2bfloat16_rn(v0);
                if (n1 < ncols) y[int64_t(out1) * rows + r] = __float2bfloat16_rn(v1);
            }
            if (r + 8 < rows) {
                if (n0 < ncols) y[int64_t(out0) * rows + r + 8] = __float2bfloat16_rn(v2);
                if (n1 < ncols) y[int64_t(out1) * rows + r + 8] = __float2bfloat16_rn(v3);
            }
        }
    }
}

// TQ3_4S (repacked) -> FP16 [rows][k], true values in the rotated basis (the tq3_matvec_h_k decode / TQ3_CB_Q):
// feeds cuBLAS for large prefill chunks. One thread per 32-weight block, natural k order.
__global__ void tq3_dequant_h_k(const uint4 * w, __half * out, int64_t nblocks, uint32_t cblo_u, uint32_t cbhi_u) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint4 q = w[i];
    const uint32_t NEG1152 = 0xE480E480u;
    const __half2 inv = __float2half2_rn(1.f / TQ3_CB_Q);
    const uint32_t invu = *reinterpret_cast<const uint32_t *>(&inv);
    uint32_t sg[4], og[4];
#pragma unroll
    for (int gi = 0; gi < 4; ++gi) {
        const uint32_t b = (q.x >> (8 * gi)) & 255u, hh = b ? (b << 5) + 0x1800u : 0u;
        sg[gi] = hh | (hh << 16);
        og[gi] = hmul2u(sg[gi], NEG1152);
    }
    const uint32_t sl = ((q.y >> 3) & 0x1111u) | ((q.z >> 2) & 0x2222u) | ((q.w >> 1) & 0x4444u);
    const uint32_t sl2 = ((q.y >> 19) & 0x1111u) | ((q.z >> 18) & 0x2222u) | ((q.w >> 17) & 0x4444u);
    const uint32_t sel[8] = {q.y, q.y >> 16, q.z, q.z >> 16, q.w, q.w >> 16, sl, sl2};
    uint32_t o[16];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sel[u] & 0x7777u);
        o[2 * u]     = hmul2u(hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sg[u / 2], og[u / 2]), invu);
        o[2 * u + 1] = hmul2u(hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sg[u / 2], og[u / 2]), invu);
    }
    uint4 * dst = reinterpret_cast<uint4 *>(out + i * 32);
#pragma unroll
    for (int j = 0; j < 4; ++j) dst[j] = make_uint4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]);
}

__global__ void f32_to_bf16_k(const float * x, __nv_bfloat16 * y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}

__global__ void i8h_quantize_k(const float * x, int8_t * w8, __half * sc, int64_t ngroups) {
    const int64_t gi = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gi >= ngroups) return;
    const float * xg = x + gi * 16;
    float m = 0.f;
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(xg[i]));
    const __half hs = __float2half_rn(m / 127.f);
    const float s = __half2float(hs);
    for (int i = 0; i < 16; ++i) w8[gi * 16 + i] = int8_t(s > 0.f ? max(-127, min(127, __float2int_rn(xg[i] / s))) : 0);
    sc[gi] = hs;
}

// stage 1: grid (chunks, rows), one top-16 per chunk; stage 2: top-16 over chunks*16 candidates per row
__global__ void topk16_chunk_k(const float * x, int len, int chunk, float * cv, uint32_t * ci) {
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    __shared__ float lv_s;
    __shared__ uint32_t li_s;
    const int row = blockIdx.y, c = blockIdx.x;
    const int beg = c * chunk, end = min(len, beg + chunk);
    const float * xr = x + int64_t(row) * len;
    if (threadIdx.x == 0) {
        lv_s = CUDART_INF_F;
        li_s = 0xffffffffu;
    }
    __syncthreads();
    for (int r = 0; r < 16; ++r) {
        float bv = -CUDART_INF_F;
        uint32_t bi = 0xffffffffu;
        const float lv = lv_s;
        const uint32_t li = li_s;
        for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
            const float v = xr[i];
            const bool below = v < lv || (v == lv && uint32_t(i) > li);
            if (below && (v > bv || (v == bv && uint32_t(i) < bi))) {
                bv = v;
                bi = i;
            }
        }
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (threadIdx.x % 32 == 0) {
            sv[threadIdx.x / 32] = bv;
            si[threadIdx.x / 32] = bi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < 8; ++w)
                if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) {
                    bv = sv[w];
                    bi = si[w];
                }
            const int64_t o = (int64_t(row) * gridDim.x + c) * 16 + r;
            cv[o] = bv;
            ci[o] = bi;
            lv_s = bv;
            li_s = bi;
        }
        __syncthreads();
    }
}

__global__ void topk16_merge_k(const float * cv, const uint32_t * ci, int ncand, float * vals, uint32_t * idx) {
    // one warp per row; ncand <= 1024: repeated argmax over (value desc, index asc) strictly after the last pick
    const int row = blockIdx.x, lane = threadIdx.x;
    const float * v = cv + int64_t(row) * ncand;
    const uint32_t * ix = ci + int64_t(row) * ncand;
    float lv = CUDART_INF_F;
    uint32_t li = 0xffffffffu;
    for (int r = 0; r < 16; ++r) {
        float bv = -CUDART_INF_F;
        uint32_t bi = 0xffffffffu;
        for (int i = lane; i < ncand; i += 32) {
            const float a = v[i];
            const uint32_t id = ix[i];
            const bool below = a < lv || (a == lv && id > li);
            if (below && (a > bv || (a == bv && id < bi))) {
                bv = a;
                bi = id;
            }
        }
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            vals[row * 16 + r] = bv;
            idx[row * 16 + r] = bi;
        }
        lv = bv;
        li = bi;
    }
}

__global__ void bf16_to_h_k(const __nv_bfloat16 * x, __half * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(__bfloat162float(x[i]));
}

__global__ void bf16_to_f32_k(const __nv_bfloat16 * x, float * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]);
}

template <int N>
void tq3_launch(const void * w, const float * xr, __nv_bfloat16 * y, int rows, int k, cudaStream_t s) {
    constexpr int RPW = 2;
    const int grid = (rows + 8 * RPW - 1) / (8 * RPW);
    tq3_matvec_k<N, RPW><<<grid, 256, 0, s>>>(static_cast<const uint4 *>(w), xr, y, rows, k);
}

template <int N>
void q6k_launch(const void * w, const float * x, float * y, int rows, int k, cudaStream_t s) {
    q6k_matvec_w_k<N><<<(rows + 7) / 8, 256, 0, s>>>(static_cast<const uint8_t *>(w), x, y, rows, k);
}

} // namespace

namespace tqllm_cuda {
void k_expert_matvec(int fmt, const uint8_t * w, const void * aux, const __half * x,
                     __nv_bfloat16 * y, int rows, int cols, const int * ids, int n, int topk,
                     bool per_expert_input, cudaStream_t stream) {
    if (!w || !aux || !x || !y || !ids || rows <= 0 || cols <= 0 ||
        cols % (fmt == 14 ? 128 : fmt == 15 ? 64 : 256) || n <= 0 || topk <= 0 || int64_t(n) * topk > 65535)
        throw std::invalid_argument("k_expert_matvec: invalid shape");
    const dim3 grid(unsigned((rows + 15) / 16), unsigned(n * topk));
    const auto * scales = static_cast<const __half2 *>(aux);
    // Whole u4h family, not just K4/K5/K6: GROUPED is an indexing change in the same
    // template the dense kernels use (NCOL 1, MT 1, PF 0), so grouped decode stays
    // bit-identical to a row-sliced dense matvec (tests/moe_device.cu).
    switch (fmt) {
    case 0: u4h_matvec_k<1, __nv_bfloat16, 1, 0, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 1: u4h_matvec_k<1, __nv_bfloat16, 1, 1, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 2: u4h_matvec_k<1, __nv_bfloat16, 1, 2, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 4: u4h_matvec_k<1, __nv_bfloat16, 1, 4, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 5: u4h_matvec_k<1, __nv_bfloat16, 1, 5, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 6: u4h_matvec_k<1, __nv_bfloat16, 1, 6, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 7: u4h_matvec_k<1, __nv_bfloat16, 1, 7, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 8: u4h_matvec_k<1, __nv_bfloat16, 1, 8, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 9: u4h_matvec_k<1, __nv_bfloat16, 1, 9, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 10: u4h_matvec_k<1, __nv_bfloat16, 1, 10, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 11: u4h_matvec_k<1, __nv_bfloat16, 1, 11, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 12: u4h_matvec_k<1, __nv_bfloat16, 1, 12, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 13: u4h_matvec_k<1, __nv_bfloat16, 1, 13, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 14: u4h_matvec_k<1, __nv_bfloat16, 1, 14, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 15: u4h_matvec_k<1, __nv_bfloat16, 1, 15, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    default: throw std::invalid_argument("k_expert_matvec: unsupported format");
    }
}


void k_expert_matvec_dedup(int fmt, const uint8_t * w, const void * aux, const __half * x, __nv_bfloat16 * y,
                           int rows, int cols, const int * dmap, int groups, cudaStream_t stream) {
    if (!w || !aux || !x || !y || !dmap || rows <= 0 || cols <= 0 || cols % (fmt == 14 ? 128 : fmt == 15 ? 64 : 256) || groups <= 0 || groups > 65535)
        throw std::invalid_argument("k_expert_matvec_dedup: invalid shape");
    // One 16-row tile per block: the expert's weights are streamed once, the activations are shared.
    const dim3 grid(unsigned((rows + 15) / 16), unsigned(groups));
    const auto * sc = static_cast<const __half2 *>(aux);
#define TQ_DEDUP(F) u4h_matvec_k<tqllm_cuda::kDedupCols, __nv_bfloat16, 1, F, 0, true><<<grid, 256, 0, stream>>>(w, sc, x, y, rows, cols, 0, nullptr, 0, dmap)
    switch (fmt) {
        case 1: TQ_DEDUP(1); break;
        case 4: TQ_DEDUP(4); break;
        case 5: TQ_DEDUP(5); break;
        case 6: TQ_DEDUP(6); break;
        case 7: TQ_DEDUP(7); break;
        case 10: TQ_DEDUP(10); break;
        case 13: TQ_DEDUP(13); break;
        case 14: TQ_DEDUP(14); break;
        case 15: TQ_DEDUP(15); break;
        default: throw std::invalid_argument("k_expert_matvec_dedup: unsupported format");
    }
#undef TQ_DEDUP
}

void set_tiled_experts(bool on) { const int v = on ? 1 : 0; cudaMemcpyToSymbol(c_tiled_experts, &v, sizeof v); }
int tiled_block_bytes(int fmt) { return tiled_bpb(fmt); }
void tile_expert_rows(const char * src, char * dst, int rows, int k, int bpb) {
    const int nb = k / 32;
    for (int r = 0; r < rows; ++r)
        for (int s = 0; s < nb; ++s) {
            const int64_t to = (((int64_t(r >> 4) * (nb >> 2) + (s >> 2)) * 16 + (r & 15)) * 4 + (s & 3)) * bpb;
            std::memcpy(dst + to, src + (int64_t(r) * nb + s) * bpb, size_t(bpb));
        }
}
// gate and up in one launch (grid z = 2): same blocks, same per-row math as two k_expert_matvec_dedup calls
void k_expert_matvec2(int fmt, const uint8_t * wa, const void * auxa, const uint8_t * wb, const void * auxb, const __half * x,
                      __nv_bfloat16 * ya, __nv_bfloat16 * yb, int rows, int cols, const int * ids, int n, int topk, cudaStream_t stream) {
    if (!wa || !auxa || !wb || !auxb || !x || !ya || !yb || !ids || rows <= 0 || cols <= 0 ||
        cols % (fmt == 14 ? 128 : fmt == 15 ? 64 : 256) || n <= 0 || topk <= 0 || int64_t(n) * topk > 65535)
        throw std::invalid_argument("k_expert_matvec2: invalid shape");
    const dim3 grid(unsigned((rows + 15) / 16), unsigned(n * topk), 2u);
    const auto * sa = static_cast<const __half2 *>(auxa); const auto * sb = static_cast<const __half2 *>(auxb);
#define TQ_PAIR(F) u4h_matvec_k<1, __nv_bfloat16, 1, F, 0, true><<<grid, 256, 0, stream>>>(wa, sa, x, ya, rows, cols, 1, nullptr, 0, ids, topk, false, wb, sb, yb)
    switch (fmt) {
        case 0: TQ_PAIR(0); break;
        case 1: TQ_PAIR(1); break;
        case 2: TQ_PAIR(2); break;
        case 4: TQ_PAIR(4); break;
        case 5: TQ_PAIR(5); break;
        case 6: TQ_PAIR(6); break;
        case 7: TQ_PAIR(7); break;
        case 8: TQ_PAIR(8); break;
        case 9: TQ_PAIR(9); break;
        case 10: TQ_PAIR(10); break;
        case 11: TQ_PAIR(11); break;
        case 12: TQ_PAIR(12); break;
        case 13: TQ_PAIR(13); break;
        case 14: TQ_PAIR(14); break;
        case 15: TQ_PAIR(15); break;
        default: throw std::invalid_argument("k_expert_matvec2: unsupported format");
    }
#undef TQ_PAIR
}

void k_expert_matvec_dedup2(int fmt, const uint8_t * wa, const void * auxa, const uint8_t * wb, const void * auxb, const __half * x,
                            __nv_bfloat16 * ya, __nv_bfloat16 * yb, int rows, int cols, const int * dmap, int groups, cudaStream_t stream) {
    if (!wa || !auxa || !wb || !auxb || !x || !ya || !yb || !dmap || rows <= 0 || cols <= 0 || cols % (fmt == 14 ? 128 : fmt == 15 ? 64 : 256) || groups <= 0 || groups > 65535)
        throw std::invalid_argument("k_expert_matvec_dedup2: invalid shape");
    const dim3 grid(unsigned((rows + 15) / 16), unsigned(groups), 2u);
    const auto * sa = static_cast<const __half2 *>(auxa); const auto * sb = static_cast<const __half2 *>(auxb);
#define TQ_DEDUP2(F) u4h_matvec_k<tqllm_cuda::kDedupCols, __nv_bfloat16, 1, F, 0, true><<<grid, 256, 0, stream>>>(wa, sa, x, ya, rows, cols, 0, nullptr, 0, dmap, 1, false, wb, sb, yb)
    switch (fmt) {
        case 1: TQ_DEDUP2(1); break;
        case 4: TQ_DEDUP2(4); break;
        case 5: TQ_DEDUP2(5); break;
        case 6: TQ_DEDUP2(6); break;
        case 7: TQ_DEDUP2(7); break;
        case 10: TQ_DEDUP2(10); break;
        case 13: TQ_DEDUP2(13); break;
        case 14: TQ_DEDUP2(14); break;
        case 15: TQ_DEDUP2(15); break;
        default: throw std::invalid_argument("k_expert_matvec_dedup2: unsupported format");
    }
#undef TQ_DEDUP2
}

void k_expert_gemm(int fmt, const uint8_t * w, const void * aux, const __half * x,
                   __nv_bfloat16 * y, int rows, int cols, const int * counts, const int * groups,
                   int capacity, int experts, int topk, bool per_expert_input, cudaStream_t stream) {
    if (!w || !aux || !x || !y || !counts || !groups || rows <= 0 || cols <= 0 ||
        cols % (fmt == 14 ? 128 : 256) || capacity <= 0 || capacity > 65535 * 64 ||
        experts <= 0 || experts > 65535 || topk <= 0)
        throw std::invalid_argument("k_expert_gemm: invalid shape");
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((capacity + 63) / 64), unsigned(experts));
    const auto * scales = static_cast<const __half2 *>(aux);
    // Same family coverage as k_expert_matvec: u4h_gemm_k<FMT, GROUPED> is generic in FMT
    // (its GROUPED branch only reads counts/groups and offsets source_row), so prefill
    // stays bit-identical to a row-sliced dense gemm().
    switch (fmt) {
    case 0: u4h_gemm_k<0, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 1: u4h_gemm_k<1, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 2: u4h_gemm_k<2, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 4: u4h_gemm_k<4, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 5: u4h_gemm_k<5, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 6: u4h_gemm_k<6, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 7: u4h_gemm_k<7, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 8: u4h_gemm_k<8, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 9: u4h_gemm_k<9, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 10: u4h_gemm_k<10, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 11: u4h_gemm_k<11, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 12: u4h_gemm_k<12, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 13: u4h_gemm_k<13, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 14: u4h_gemm_k<14, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 15: u4h_gemm_k<15, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    default: throw std::invalid_argument("k_expert_gemm: unsupported format");
    }
}

// fmt: 0 u4h, 1 k4, 2 k3, 4 iq3, 5 iq3x, 6 k5, 7 k6, 8-12 UD types (u4h_matvec_k's FMT); gemm only: 13 iq4n, 14 iq4l, 15 i8h
void k_expert_gemm_i8(int fmt, const uint8_t * w, const void * aux, const int8_t * xq, const float * xs, const int16_t * xsum,
                      __nv_bfloat16 * y, int rows, int cols, const int * counts, const int * groups,
                      int capacity, int experts, int topk, bool per_expert_input, cudaStream_t stream, int mode) {
    if (!w || !aux || !xq || !xs || !xsum || !y || !counts || !groups || rows <= 0 || cols <= 0 ||
        cols % (fmt == 14 ? 128 : 256) || capacity <= 0 || capacity > 65535 * 64 ||
        experts <= 0 || experts > 65535 || topk <= 0)
        throw std::invalid_argument("k_expert_gemm_i8: invalid shape");
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((capacity + 63) / 64), unsigned(experts));
    const auto * scales = static_cast<const __half2 *>(aux);
#define TQ_I8G(F) do { if (mode >= 3) { const int bn3 = i8_offset<F>() == 0 ? 64 : 128; const dim3 g3(unsigned((rows + 127) / 128), unsigned((capacity + bn3 - 1) / bn3), unsigned(experts)); \
                         if (mode == 4) u4h_i8v3_gemm_k<F, true><<<g3, 256, 0, stream>>>(w, scales, xq, xs, xsum, y, rows, cols, counts, groups, capacity, topk, per_expert_input); \
                         else u4h_i8v3_gemm_k<F, false><<<g3, 256, 0, stream>>>(w, scales, xq, xs, xsum, y, rows, cols, counts, groups, capacity, topk, per_expert_input); } \
                       else if (mode == 2) u4h_i8_gemm_k<F, true, true><<<grid, 256, 0, stream>>>(w, scales, xq, xs, xsum, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); \
                       else u4h_i8_gemm_k<F, true, false><<<grid, 256, 0, stream>>>(w, scales, xq, xs, xsum, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); } while (0)
    switch (fmt) {
        case 1: TQ_I8G(1); break;
        case 4: TQ_I8G(4); break;
        case 5: TQ_I8G(5); break;
        case 6: TQ_I8G(6); break;
        case 7: TQ_I8G(7); break;
        case 10: TQ_I8G(10); break;
        case 13: TQ_I8G(13); break;
        case 14: TQ_I8G(14); break;
        default: throw std::invalid_argument("k_expert_gemm_i8: unsupported format");
    }
#undef TQ_I8G
}
void k_quant_act_rows(const __half * x, const int * rows_of, const int * counts, int cap, int slots, int nrows, int k,
                      int8_t * q, float * s, int16_t * qsum, cudaStream_t stream, int mode) {
    if (k % 32 || !x || !q || !s || !qsum) throw std::invalid_argument("k_quant_act_rows: invalid shape");
    if (!rows_of) {
        if (nrows <= 0) return;
        if (mode == 2 || mode == 4) quant_act_rows_k<true><<<unsigned(nrows), 256, 0, stream>>>(x, nullptr, nullptr, 0, k, q, s, qsum);
        else quant_act_rows_k<false><<<unsigned(nrows), 256, 0, stream>>>(x, nullptr, nullptr, 0, k, q, s, qsum);
    } else if (cap > 0 && slots > 0) {
        if (mode == 2 || mode == 4) quant_act_rows_k<true><<<dim3(unsigned(cap), unsigned(slots)), 256, 0, stream>>>(x, rows_of, counts, cap, k, q, s, qsum);
        else quant_act_rows_k<false><<<dim3(unsigned(cap), unsigned(slots)), 256, 0, stream>>>(x, rows_of, counts, cap, k, q, s, qsum);
    }
}

// dense (non-grouped) int8-activation GEMM, f32 output: the layer-major dense projections (TQFN_DENSE_INT8)
bool u4h_family_gemm_i8_f32(int fmt, const uint8_t * w, const void * aux, const int8_t * xq, const float * xs, const int16_t * xsum,
                            float * y, int rows, int k, int ncols, cudaStream_t s, int ldy, int mode) {
    if (k % 64 || rows <= 0 || ncols <= 0) return false;
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((ncols + 63) / 64));
    const auto * scales = static_cast<const __half2 *>(aux);
#define TQ_I8D(F) do { if (mode == 2) u4h_i8_gemm_k<F, false, true, float><<<grid, 256, 0, s>>>(w, scales, xq, xs, xsum, y, rows, k, ncols, ldy ? ldy : rows, nullptr, nullptr, 0, 1, false); \
                       else u4h_i8_gemm_k<F, false, false, float><<<grid, 256, 0, s>>>(w, scales, xq, xs, xsum, y, rows, k, ncols, ldy ? ldy : rows, nullptr, nullptr, 0, 1, false); } while (0)
    switch (fmt) {
        case 1: TQ_I8D(1); break;
        case 4: TQ_I8D(4); break;
        case 5: TQ_I8D(5); break;
        case 6: TQ_I8D(6); break;
        case 7: TQ_I8D(7); break;
        case 10: TQ_I8D(10); break;
        case 13: TQ_I8D(13); break;
        case 14: TQ_I8D(14); break;
        default: return false;
    }
#undef TQ_I8D
    return true;
}

// gate + up in one grouped launch with the swiglu epilogue (library only): yh [assignment][rows] half
void k_expert_gemm_gateup(int fmt, const uint8_t * wg, const void * auxg, const uint8_t * wu, const void * auxu, const __half * x,
                          __half * yh, int rows, int cols, const int * counts, const int * groups,
                          int capacity, int experts, int topk, cudaStream_t stream) {
    if (!wg || !auxg || !wu || !auxu || !x || !yh || !counts || !groups || rows <= 0 || cols <= 0 ||
        cols % (fmt == 14 ? 128 : 256) || capacity <= 0 || capacity > 65535 * 64 || experts <= 0 || experts > 65535 || topk <= 0)
        throw std::invalid_argument("k_expert_gemm_gateup: invalid shape");
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((capacity + 63) / 64), unsigned(experts));
    const auto * sg = static_cast<const __half2 *>(auxg); const auto * su = static_cast<const __half2 *>(auxu);
#define TQ_GU(F) u4h_gemm_k<F, true, __nv_bfloat16, true><<<grid, 256, 0, stream>>>(wg, sg, x, nullptr, rows, cols, capacity, rows, counts, groups, capacity, topk, false, wu, su, yh)
    switch (fmt) {
        case 1: TQ_GU(1); break;
        case 4: TQ_GU(4); break;
        case 5: TQ_GU(5); break;
        case 6: TQ_GU(6); break;
        case 7: TQ_GU(7); break;
        case 10: TQ_GU(10); break;
        case 13: TQ_GU(13); break;
        case 14: TQ_GU(14); break;
        default: throw std::invalid_argument("k_expert_gemm_gateup: unsupported format");
    }
#undef TQ_GU
}

void u4h_family_gemm(int fmt, const uint8_t * w, const void * aux, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    switch (fmt) {
        case 0: u4h_gemm_launch<0>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 1: u4h_gemm_launch<1>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 2: u4h_gemm_launch<2>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 4: u4h_gemm_launch<4>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 5: u4h_gemm_launch<5>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 6: u4h_gemm_launch<6>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 7: u4h_gemm_launch<7>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 8: u4h_gemm_launch<8>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 9: u4h_gemm_launch<9>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 10: u4h_gemm_launch<10>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 11: u4h_gemm_launch<11>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 12: u4h_gemm_launch<12>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 13: u4h_gemm_launch<13>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 14: u4h_gemm_launch<14>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 15: u4h_gemm_launch<15>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        default: throw std::invalid_argument("u4h_family_gemm: unknown format");
    }
}
// Same, F32 output (prefill projections that feed F32 activations); false: format without a GEMM kernel.
bool u4h_family_gemm_f32(int fmt, const uint8_t * w, const void * aux, const __half * xh, float * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    switch (fmt) {
        case 0: u4h_gemm_launch<0, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 1: u4h_gemm_launch<1, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 2: u4h_gemm_launch<2, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 4: u4h_gemm_launch<4, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 5: u4h_gemm_launch<5, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 6: u4h_gemm_launch<6, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 7: u4h_gemm_launch<7, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 8: u4h_gemm_launch<8, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 9: u4h_gemm_launch<9, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 10: u4h_gemm_launch<10, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 11: u4h_gemm_launch<11, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 12: u4h_gemm_launch<12, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 13: u4h_gemm_launch<13, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 14: u4h_gemm_launch<14, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 15: u4h_gemm_launch<15, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        default: return false;
    }
    return true;
}


void tq3_repack(void * w, int64_t nblocks, cudaStream_t s) {
    tq3_repack_k<<<int((nblocks + 255) / 256), 256, 0, s>>>(static_cast<uint4 *>(w), nblocks);
}

void tq3_rotate_i8(const __nv_bfloat16 * x, int8_t * xq, float * xs, int k, cudaStream_t s) {
    const int nblk = k / 32;
    tq3_rotate_i8_k<<<(nblk * 32 + 255) / 256, 256, 0, s>>>(x, xq, xs, nblk);
}

void tq3_matvec_i8(const void * w, const int8_t * xq, const float * xs, __nv_bfloat16 * y, int rows, int k,
                   cudaStream_t s) {
    static const Tq3Codebook8 cb = tq3_codebook_i8();
    constexpr int RPW = 2;
    tq3_matvec_i8_k<RPW><<<(rows + 8 * RPW - 1) / (8 * RPW), 256, 0, s>>>(
        static_cast<const uint4 *>(w), reinterpret_cast<const int4 *>(xq), xs, y, rows, k, cb.lo, cb.hi);
}

void tq3_rotate(const __nv_bfloat16 * x, float * xr, int k, int ncols, cudaStream_t s) {
    const int blocks = k / 32 * ncols;
    tq3_rotate_k<<<(blocks * 32 + 255) / 256, 256, 0, s>>>(x, xr, blocks);
}

void tq3_rotate_f32(const float * x, float * xr, int k, int ncols, cudaStream_t s) {
    if (!x || !xr || k <= 0 || k % 32 || ncols <= 0 || int64_t(k) * ncols > INT_MAX)
        throw std::invalid_argument("tq3_rotate_f32: invalid pointer or shape");
    const int blocks = k / 32 * ncols;
    tq3_rotate_f32_k<<<(int64_t(blocks) * 32 + 255) / 256, 256, 0, s>>>(x, xr, blocks);
}

void tq3_matvec_f32_raw(const void * w, const float * xr, float * y, int rows, int k, int ncols, cudaStream_t s) {
    if (!w || !xr || !y || rows <= 0 || k <= 0 || k % 32 || ncols <= 0 || ncols > 65535 ||
        int64_t(k) * ncols > INT_MAX || int64_t(rows) * ncols > INT_MAX)
        throw std::invalid_argument("tq3_matvec_f32_raw: invalid pointer or shape");
    tq3_matvec_f32_raw_k<false><<<dim3(unsigned(rows), unsigned(ncols)), 256, 0, s>>>(
        static_cast<const uint4 *>(w), xr, y, rows, k);
}

void tq3_matvec_f32_repacked(const void * w, const float * xr, float * y, int rows, int k, int ncols, cudaStream_t s) {
    if (!w || !xr || !y || rows <= 0 || k <= 0 || k % 32 || ncols <= 0 || ncols > 65535 ||
        int64_t(k) * ncols > INT_MAX || int64_t(rows) * ncols > INT_MAX)
        throw std::invalid_argument("tq3_matvec_f32_repacked: invalid pointer or shape");
    tq3_matvec_f32_raw_k<true><<<dim3(unsigned(rows), unsigned(ncols)), 256, 0, s>>>(
        static_cast<const uint4 *>(w), xr, y, rows, k);
}

void tq3_matvec(const void * w, const float * xr, __nv_bfloat16 * y, int rows, int k, int ncols, cudaStream_t s) {
    if (k % 32) throw std::invalid_argument("tq3_matvec: k must be a multiple of 32");
    if (ncols == 1) {
        constexpr int RPW = 4;
        tq3_matvec1_k<RPW><<<(rows + 8 * RPW - 1) / (8 * RPW), 256, 0, s>>>(static_cast<const uint4 *>(w), xr, y, rows, k);
        return;
    }
    switch (ncols) {
        case 2: return tq3_launch<2>(w, xr, y, rows, k, s);
        case 3: return tq3_launch<3>(w, xr, y, rows, k, s);
        case 4: return tq3_launch<4>(w, xr, y, rows, k, s);
        case 5: return tq3_launch<5>(w, xr, y, rows, k, s);
        case 6: return tq3_launch<6>(w, xr, y, rows, k, s);
        case 7: return tq3_launch<7>(w, xr, y, rows, k, s);
        case 8: return tq3_launch<8>(w, xr, y, rows, k, s);
        default: throw std::invalid_argument("tq3_matvec: 1..8 columns");
    }
}

void tq3_row(const void * w, int row, __nv_bfloat16 * y, int k, cudaStream_t s) {
    const int nblk = k / 32;
    tq3_row_k<<<(nblk + 7) / 8, 256, 0, s>>>(static_cast<const uint4 *>(w), row, y, nblk);
}

void q6k_matvec(const void * w, const float * x, float * y, int rows, int k, int ncols, cudaStream_t s) {
    if (k % Q6K_BLOCK) throw std::invalid_argument("q6k_matvec: k must be a multiple of 256");
    switch (ncols) {
        case 1: return q6k_launch<1>(w, x, y, rows, k, s);
        case 2: return q6k_launch<2>(w, x, y, rows, k, s);
        case 3: return q6k_launch<3>(w, x, y, rows, k, s);
        case 4: return q6k_launch<4>(w, x, y, rows, k, s);
        case 5: return q6k_launch<5>(w, x, y, rows, k, s);
        case 6: return q6k_launch<6>(w, x, y, rows, k, s);
        case 7: return q6k_launch<7>(w, x, y, rows, k, s);
        case 8: return q6k_launch<8>(w, x, y, rows, k, s);
        case 9: return q6k_launch<9>(w, x, y, rows, k, s);
        case 10: return q6k_launch<10>(w, x, y, rows, k, s);
        case 11: return q6k_launch<11>(w, x, y, rows, k, s);
        case 12: return q6k_launch<12>(w, x, y, rows, k, s);
        case 13: return q6k_launch<13>(w, x, y, rows, k, s);
        case 14: return q6k_launch<14>(w, x, y, rows, k, s);
        case 15: return q6k_launch<15>(w, x, y, rows, k, s);
        case 16: return q6k_launch<16>(w, x, y, rows, k, s);
        default: throw std::invalid_argument("q6k_matvec: 1..16 columns");
    }
}

void add_sumsq(__nv_bfloat16 * x, const __nv_bfloat16 * y, float * partial, int k, cudaStream_t s) {
    add_sumsq_k<<<(k + 255) / 256, 256, 0, s>>>(x, y, partial, k);
}

void sumsq(const __nv_bfloat16 * x, float * partial, int k, cudaStream_t s) {
    sumsq_k<<<(k + 255) / 256, 256, 0, s>>>(x, partial, k);
}

void norm_rotate_i8(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, int8_t * xq,
                    float * xs, int k, cudaStream_t s) {
    norm_rotate_i8_k<<<(k + 255) / 256, 256, 0, s>>>(x, w, partial, (k + 255) / 256, eps, xq, xs, k);
}

void argmax_big(const float * x, int n, uint32_t * out, void * scratch, cudaStream_t s) {
    constexpr int NB = 512;
    auto * pv = static_cast<float *>(scratch);
    auto * pi = reinterpret_cast<uint32_t *>(pv + 1024);
    argmax1_k<<<NB, 256, 0, s>>>(x, n, pv, pi);
    argmax2_k<<<1, 32, 0, s>>>(pv, pi, NB, out);
}

void q6k_row(const void * w, int row, __nv_bfloat16 * y, int k, cudaStream_t s) {
    q6k_row_k<<<(k + 255) / 256, 256, 0, s>>>(static_cast<const uint8_t *>(w), row, y, k);
}

// One thread per element of one IQ3_S row (raw GGUF layout: 110 bytes per 256 weights).
// Block: d(FP16,0:2) | qs[64](2:66) | qh[8](66:74) | signs[32](74:106) | scales[4](106:110).
// Decode mirrors the proven iq3_dequant_h_k path: value = db * (iq3_code(grid, signs) - 16),
// db = d * (1 + 2 * scale_nibble).  Dual-mode: dtok==nullptr → single row; else multi-token.
__global__ void iq3s_row_k(const uint8_t * w, int row, __nv_bfloat16 * y, int k, const uint32_t * dtok = nullptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    if (dtok) {  // multi-token: blockIdx.y selects the token, id read on the device
        row = int(dtok[blockIdx.y]);
        y += int64_t(blockIdx.y) * k;
    }
    const int nblk = k / 256;
    const int b = i / 256, e = i % 256;
    const int ib = e / 32, wi = (e % 32) / 4, j = e % 4;
    const uint8_t * blk = w + (int64_t(row) * nblk + b) * 110;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));  // offset 0: 2-byte aligned (110 % 2 == 0)
    const uint32_t qs_byte = blk[2 + ib * 8 + wi];
    const uint32_t qh_bit  = (blk[66 + ib] >> wi) & 1u;
    // signs word assembled from bytes: raw GGUF blocks are 110-byte-strided, so blk+74+4*ib is
    // NOT 4-byte aligned — a uint32_t load here would fault. Byte loads are always safe.
    const uint32_t sg = uint32_t(blk[74 + ib * 4]) | (uint32_t(blk[75 + ib * 4]) << 8) |
                        (uint32_t(blk[76 + ib * 4]) << 16) | (uint32_t(blk[77 + ib * 4]) << 24);
    const uint32_t sc_nib  = (blk[106 + ib / 2] >> (4 * (ib % 2))) & 0xFu;
    const float db = d * float(1 + 2 * sc_nib);
    const uint32_t c = iq3_code(iq3s_grid_d[qs_byte | (qh_bit << 8)], (sg >> (4 * wi)) & 0xFu);
    const int code = int((c >> (8 * j)) & 0xFFu) - 16;
    y[i] = __float2bfloat16_rn(db * float(code));
}

void iq3s_row(const void * w, int row, __nv_bfloat16 * y, int k, cudaStream_t s) {
    iq3s_row_k<<<(k + 255) / 256, 256, 0, s>>>(static_cast<const uint8_t *>(w), row, y, k);
}

void tq3_rotate_h(const __nv_bfloat16 * x, __half * xh, int k, int ncols, cudaStream_t s) {
    const int blocks = k / 32 * ncols;
    tq3_rotate_h_k<<<(blocks * 32 + 255) / 256, 256, 0, s>>>(x, xh, blocks);
}

void norm_rotate_h(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, __half * xh, int k,
                   int ncols, cudaStream_t s) {
    norm_rotate_h_k<<<dim3((k + 255) / 256, ncols), 256, 0, s>>>(x, w, partial, (k + 255) / 256, eps, xh, k);
}

// ---- W4A16 models: u4h weights take un-rotated FP16 activations ----
__global__ void norm_h_k(const __nv_bfloat16 * x, const float * w, const float * partial, int nparts, float eps,
                         __half * xh, int k) {
    __shared__ float inv_s;
    const int n = blockIdx.y;
    if (threadIdx.x < 32) {
        float t = 0.f;
        for (int i = threadIdx.x; i < nparts; i += 32) t += partial[n * nparts + i];
        for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) inv_s = rsqrtf(t / k + eps);
    }
    __syncthreads();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= k) return;
    const int64_t o = int64_t(n) * k + i;
    xh[o] = __float2half_rn(__bfloat162float(__float2bfloat16_rn(__bfloat162float(x[o]) * inv_s * w[i])));
}

// u4h group of 16 (8 bytes: element 4u + j = byte (u / 2) * 4 + j, nibble u % 2), (scale, min) per 32 -> FP16
__global__ void u4h_dequant_h_k(const uint8_t * w4, const __half2 * sm, __half * out, int64_t ngroups) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const uint2 raw = reinterpret_cast<const uint2 *>(w4)[g];
    const uint8_t * b = reinterpret_cast<const uint8_t *>(&raw);
    const __half2 s = sm[g / 2];
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        const int q = (b[(u / 2) * 4 + j] >> ((u % 2) * 4)) & 0xF;
        v[e] = __hfma(__int2half_rn(q), __low2half(s), __high2half(s));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

// i8h: int8 + FP16 scale per 16 -> FP16 (prefill dequant). One thread per 16-weight group.
__global__ void i8h_dequant_h_k(const int8_t * w8, const __half * sc, __half * out, int64_t ngroups) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const int4 raw = reinterpret_cast<const int4 *>(w8)[g];
    const int8_t * q = reinterpret_cast<const int8_t *>(&raw);
    const __half s = sc[g];
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) v[e] = __hmul(__int2half_rn(q[e]), s);
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

__global__ void iq3x_dequant_h_k(const uint8_t * w3, const uint8_t * rec, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const int64_t blk = g / 2, row = blk / (k / 32);
    const int sblk = int(blk % (k / 32)), h = int(g % 2);
    const auto * wp = reinterpret_cast<const uint32_t *>(w3 + blk * 12);
    const uint32_t q = wp[h], a = wp[2];
    const __half2 s = iq3x_sm(rec + (row * (k / 256) + sblk / 8) * 4, a);
    __align__(16) __half v[16];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const uint32_t c = iq3x_code(iq3xxs_grid_d[(q >> (8 * u)) & 0xFF], iq3x_signs(a, 4 * h + u));
#pragma unroll
        for (int j = 0; j < 4; ++j) v[4 * u + j] = __hfma(__int2half_rn(int((c >> (8 * j)) & 0xFF)), __low2half(s), __high2half(s));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

__global__ void iq3_dequant_h_k(const uint8_t * w3, const uint8_t * rec, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // one 16-weight group
    if (g >= ngroups) return;
    const int64_t blk = g / 2, row = blk / (k / 32);
    const int sblk = int(blk % (k / 32)), h = int(g % 2);
    const auto * wp = reinterpret_cast<const uint32_t *>(w3 + blk * 12);
    const uint8_t * rc = rec + (row * (k / 256) + sblk / 8) * 16;
    const __half2 s = iq3_sm(rc, sblk % 8);
    const uint32_t q = wp[h], sg = wp[2], qh = rc[2 + sblk % 8];
    __align__(16) __half v[16];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int w = 4 * h + u;
        const uint32_t c = iq3_code(iq3s_grid_d[((q >> (8 * u)) & 0xFF) | (((qh >> w) & 1) << 8)], (sg >> (4 * w)) & 0xF);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[4 * u + j] = __hfma(__int2half_rn(int((c >> (8 * j)) & 0xFF)), __low2half(s), __high2half(s));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

__global__ void k3_dequant_h_k(const uint8_t * w3, const uint8_t * kr, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // one 16-weight group
    if (g >= ngroups) return;
    const int64_t blk = g / 2, row = blk / (k / 32);
    const int sblk = int(blk % (k / 32)), h = int(g % 2);
    const auto * wp = reinterpret_cast<const uint32_t *>(w3 + blk * 12);
    const uint32_t L = wp[h], H = wp[2];
    const uint8_t * rc = kr + (row * (k / 256) + sblk / 8) * 20;
    const float sc = __half2float(*reinterpret_cast<const __half *>(rc + 16)) * float(int8_t(rc[(sblk % 8) * 2 + h]));
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        const int q = int(((L >> (8 * j + 2 * u)) & 3) | (((H >> (8 * j + u + 4 * h)) & 1) << 2));
        v[e] = __float2half_rn(sc * float(q - 4));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

__global__ void k4_dequant_h_k(const uint8_t * w4, const uint8_t * kh, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const uint2 raw = reinterpret_cast<const uint2 *>(w4)[g];
    const uint8_t * b = reinterpret_cast<const uint8_t *>(&raw);
    const int64_t blk = g / 2, row = blk / (k / 32);
    const int sblk = int(blk % (k / 32));
    const __half2 s = k4_sm(kh + (row * (k / 256) + sblk / 8) * 16, sblk % 8);
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        const int q = (b[(u / 2) * 4 + j] >> ((u % 2) * 4)) & 0xF;
        v[e] = __hfma(__int2half_rn(q), __low2half(s), __high2half(s));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

__global__ void embed_multi_q8_0_k(const uint8_t * w, const uint32_t * dtok, __nv_bfloat16 * y, int k) {
    const int n = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const uint8_t * blk = w + (int64_t(dtok[n]) * (k / 32) + i / 32) * 34;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));
    y[int64_t(n) * k + i] = __float2bfloat16_rn(d * float(reinterpret_cast<const int8_t *>(blk + 2)[i % 32]));
}

// Q4_K rows (144-byte blocks: d, dmin, scales[12], qs[128]) as in src/formats/weight/dequant.cpp q4k: element i of a block is the low
// (i % 64 < 32) or high nibble of qs[32 * (i / 64) + i % 32], sub-block j = i / 32 with its 6-bit scale and min
__global__ void embed_multi_q4k_k(const uint8_t * w, const uint32_t * dtok, __nv_bfloat16 * y, int k) {
    const int n = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const uint8_t * blk = w + (int64_t(dtok[n]) * (k / 256) + i / 256) * 144;
    const int e = i % 256, j = e / 32, l = e % 32;
    const uint8_t * sc = blk + 4;
    int s, m;
    if (j < 4) {
        s = sc[j] & 63;
        m = sc[j + 4] & 63;
    } else {
        s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
    const uint8_t q = blk[16 + 32 * (e / 64) + l];
    const float d = __half2float(reinterpret_cast<const __half *>(blk)[0]);
    const float dmin = __half2float(reinterpret_cast<const __half *>(blk)[1]);
    y[int64_t(n) * k + i] = __float2bfloat16_rn(d * float(s) * float((e % 64) < 32 ? (q & 0xF) : (q >> 4)) - dmin * float(m));
}

// Q3_K rows (110-byte blocks, layout as src/formats/weight/dequant.cpp q3k): 6-bit scale s of 16 = low nibble of byte s % 4 (+4 for
// s / 4 odd; high nibble for s / 4 >= 2) | 2 bits of byte 8 + s % 4 at 2 * (s / 4), minus 32
__global__ void embed_multi_q3k_k(const uint8_t * w, const uint32_t * dtok, __nv_bfloat16 * y, int k) {
    const int n = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const uint8_t * blk = w + (int64_t(dtok[n]) * (k / 256) + i / 256) * 110;
    const int e = i % 256, h = e / 128, j = (e % 128) / 32, l = e % 32, si = 8 * h + 2 * j + (l >= 16);
    const int g = si / 4, b = si % 4;
    const uint8_t * raw = blk + 96;
    const int lo = (g % 2 ? raw[4 + b] : raw[b]) >> (g >= 2 ? 4 : 0) & 0xF;
    const int sc = (lo | (((raw[8 + b] >> (2 * g)) & 3) << 4)) - 32;
    const int q = ((blk[32 + 32 * h + l] >> (2 * j)) & 3) - ((blk[l] >> (4 * h + j)) & 1 ? 0 : 4);
    const float d = __half2float(*reinterpret_cast<const __half *>(blk + 108));
    y[int64_t(n) * k + i] = __float2bfloat16_rn(d * float(sc) * float(q));
}
void embed_multi_q3k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    embed_multi_q3k_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), dtok, y, k);
}

void embed_multi_q4k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    embed_multi_q4k_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), dtok, y, k);
}

// Q5_K rows (176-byte blocks: d, dmin, scales[12], qh[32], qs[128]) as in q5k_to_i8_k / src/formats/weight/dequant.cpp q5k:
// 5-bit value = low/high nibble of qs[32 * (j / 2) + l] plus the qh[l] bit of sub-block j; affine with per-
// sub-block scale (s) and min (m), value = d * s * c - dmin * m
__global__ void embed_multi_q5k_k(const uint8_t * w, const uint32_t * dtok, __nv_bfloat16 * y, int k) {
    const int n = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const uint8_t * blk = w + (int64_t(dtok[n]) * (k / 256) + i / 256) * 176;
    const int e = i % 256, j = e / 32, l = e % 32;
    const uint8_t * sc = blk + 4;
    int s, m;
    if (j < 4) {
        s = sc[j] & 63;
        m = sc[j + 4] & 63;
    } else {
        s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
    const int q = blk[48 + 32 * (j / 2) + l];
    const int c = (j & 1 ? q >> 4 : q & 0xF) + ((blk[16 + l] >> j) & 1) * 16;
    const float d = __half2float(reinterpret_cast<const __half *>(blk)[0]);
    const float dmin = __half2float(reinterpret_cast<const __half *>(blk)[1]);
    y[int64_t(n) * k + i] = __float2bfloat16_rn(d * float(s) * float(c) - dmin * float(m));
}
void embed_multi_q5k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    embed_multi_q5k_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), dtok, y, k);
}

void norm_h(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, __half * xh, int k, int ncols,
            cudaStream_t s) {
    norm_h_k<<<dim3((k + 255) / 256, ncols), 256, 0, s>>>(x, w, partial, (k + 255) / 256, eps, xh, k);
}

void i8h_dequant_h(const int8_t * w8, const __half * sc, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    i8h_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w8, sc, out, ng);
}

void iq3x_dequant_h(const uint8_t * w3, const uint8_t * rec, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    iq3x_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w3, rec, out, ng, k);
}

void iq3_dequant_h(const uint8_t * w3, const uint8_t * rec, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    iq3_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w3, rec, out, ng, k);
}

void k3_dequant_h(const uint8_t * w3, const uint8_t * kr, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    k3_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w3, kr, out, ng, k);
}

// UD types: one thread per 16-weight group, the matvec kernel's decoder with the grid read from global memory
template <int FMT>
__global__ void udq_dequant_h_k(const uint8_t * w, const uint8_t * kr, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const int64_t blk = g / 2, row = blk / (k / 32);
    const int h = int(g % 2), sblk = int(blk % (k / 32));
    const auto * wp = reinterpret_cast<const uint32_t *>(w + blk * ud_bpb(FMT));
    const uint32_t w0 = wp[0], w1 = ud_bpb(FMT) == 8 ? wp[1] : 0u;
    const uint8_t * rec = kr + (row * (k / 256) + sblk / 8) * ud_rb(FMT);
    const uint32_t * grid = nullptr;
    if constexpr (FMT <= 11) grid = ud_grid_src<FMT>();
    uint32_t c[4];
    __half2 sm;
    ud_group<FMT>(w0, w1, rec, sblk, h, grid, c, sm);
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) v[e] = __hfma(__int2half_rn(int((c[e / 4] >> (8 * (e % 4))) & 0xFF)), __low2half(sm), __high2half(sm));
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}
void udq_dequant_h(int fmt, const uint8_t * w, const uint8_t * rec, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    const unsigned nbk = unsigned((ng + 255) / 256);
    switch (fmt) {
        case 8: udq_dequant_h_k<8><<<nbk, 256, 0, s>>>(w, rec, out, ng, k); break;
        case 9: udq_dequant_h_k<9><<<nbk, 256, 0, s>>>(w, rec, out, ng, k); break;
        case 10: udq_dequant_h_k<10><<<nbk, 256, 0, s>>>(w, rec, out, ng, k); break;
        case 11: udq_dequant_h_k<11><<<nbk, 256, 0, s>>>(w, rec, out, ng, k); break;
        case 12: udq_dequant_h_k<12><<<nbk, 256, 0, s>>>(w, rec, out, ng, k); break;
        default: throw std::invalid_argument("udq_dequant_h: unknown format");
    }
}
// k6: per 32 weights 16 nibble bytes (k4 order) + 2 words of 2-bit highs; 20-byte record (16 int8 scales, d) per 256
__global__ void k6_dequant_h_k(const uint8_t * w6, const uint8_t * kr, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const int64_t blk = g / 2;
    const int h = int(g % 2);
    const uint8_t * b = w6 + blk * 24 + h * 8;
    const uint32_t hi = *reinterpret_cast<const uint32_t *>(w6 + blk * 24 + 16 + 4 * h);
    const int64_t row = blk / (k / 32);
    const int sblk = int(blk % (k / 32));
    const uint8_t * rc = kr + (row * (k / 256) + sblk / 8) * 20;
    const float f = __half2float(*reinterpret_cast<const __half *>(rc + 16)) * float(int8_t(rc[(sblk % 8) * 2 + h]));
    const __half sc = __float2half_rn(f), mn = __float2half_rn(-32.f * f);
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        const int q = ((b[(u / 2) * 4 + j] >> ((u % 2) * 4)) & 0xF) | int(((hi >> (2 * e)) & 3) << 4);
        v[e] = __hfma(__int2half_rn(q), sc, mn);
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}
void k6_dequant_h(const uint8_t * w6, const uint8_t * kr, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    k6_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w6, kr, out, ng, k);
}
// k5: per 32 weights 16 nibble bytes (k4 order) + the 32 high bits; one thread per 16-weight group
__global__ void k5_dequant_h_k(const uint8_t * w5, const uint8_t * kh, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const int64_t blk = g / 2;
    const int h = int(g % 2);
    const uint8_t * b = w5 + blk * 20 + h * 8;
    const uint32_t hi = *reinterpret_cast<const uint32_t *>(w5 + blk * 20 + 16) >> (16 * h);
    const int64_t row = blk / (k / 32);
    const int sblk = int(blk % (k / 32));
    const __half2 s = k4_sm(kh + (row * (k / 256) + sblk / 8) * 16, sblk % 8);
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        const int q = ((b[(u / 2) * 4 + j] >> ((u % 2) * 4)) & 0xF) | int(((hi >> e) & 1) << 4);
        v[e] = __hfma(__int2half_rn(q), __low2half(s), __high2half(s));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}
void k5_dequant_h(const uint8_t * w5, const uint8_t * kh, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    k5_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w5, kh, out, ng, k);
}

void k4_dequant_h(const uint8_t * w4, const uint8_t * kh, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    k4_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w4, kh, out, ng, k);
}

void u4h_dequant_h(const uint8_t * w4, const __half2 * sm, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    u4h_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w4, sm, out, ng);
}

void embed_multi_q8_0(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    embed_multi_q8_0_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), dtok, y, k);
}

void tq3_matvec_h_variant(const void * w, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols, int variant,
                          cudaStream_t s) {
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const auto * wp = static_cast<const uint4 *>(w);
    if (ncols > 8) throw std::invalid_argument("tq3_matvec_h_variant: 1..8 columns");
    switch (variant) {
        case 1: tq3_matvec_p_k<1><<<(rows + 15) / 16, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi); break;
        case 2: tq3_matvec_p_k<2><<<(rows + 31) / 32, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi); break;
        case 3: tq3_matvec_hw_k<1, 2><<<(rows + 31) / 32, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi); break;
        case 4: tq3_matvec_hw_k<1, 4><<<(rows + 63) / 64, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi); break;
        default:
            if (ncols != 8) throw std::invalid_argument("tq3_matvec_h_variant: default variant at 8 columns");
            tq3_matvec_h_k<8><<<(rows + 15) / 16, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi);
    }
}

// Per-shape kernel variant for <= 8 columns, picked by tq3_tune at load (all variants are bit-identical per column):
// Tuning scratch: a full device must fail closed here, not time launches that never ran (or ran on null buffers).
static void tune_scratch(__half ** x, size_t xbytes, __nv_bfloat16 ** y, size_t ybytes, const char * who) {
    if (cudaMalloc(x, xbytes) != cudaSuccess || cudaMalloc(y, ybytes) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(*x);
        *x = nullptr;
        throw std::runtime_error(std::string(who) + ": tuning scratch allocation failed");
    }
}
static void tune_done(__half * x, __nv_bfloat16 * y, const char * who) {
    const cudaError_t e = cudaGetLastError();
    cudaFree(x);
    cudaFree(y);
    if (e != cudaSuccess) throw std::runtime_error(std::string(who) + ": " + cudaGetErrorString(e));
}

// 0 narrow, 1 / 2 prefetching with 1 / 2 row tiles, 3 / 4 row-tiled x2 / x4.
static std::map<std::pair<int, int>, int> & tq3_tuned() {
    static std::map<std::pair<int, int>, int> m;
    return m;
}

// Tuning launches the matvec on the weights it is given: only for device (or managed) memory. A host-side repack (repack_expert's
// Upload keeps the bytes on the host) must not run GPU kernels on host pointers: without HMM that is an illegal access at load.
static bool tune_pointer_ok(const void * p) {
    cudaPointerAttributes a{};
    if (!p || cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return false; }
    return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
}
void tq3_tune(const void * w, int rows, int k, cudaStream_t s) {
    if (tq3_tuned().count({rows, k}) || !tune_pointer_ok(w)) return;
    __half * x = nullptr;
    __nv_bfloat16 * y = nullptr;
    tune_scratch(&x, size_t(8) * k * 2, &y, size_t(8) * rows * 2, "tq3_tune");
    cudaMemsetAsync(x, 0, size_t(8) * k * 2, s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int best = 0;
    float best_ms = 1e30f;
    for (int v : {0, 1, 2, 3, 4}) {
        for (int i = 0; i < 10; ++i) tq3_matvec_h_variant(w, x, y, rows, k, 8, v, s);  // warm-up (clocks, caches)
        cudaEventRecord(e0, s);
        for (int i = 0; i < 30; ++i) tq3_matvec_h_variant(w, x, y, rows, k, 8, v, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms * 0.98f) best_ms = ms, best = v;  // keep the lower variant unless clearly faster
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    tune_done(x, y, "tq3_tune");
    tq3_tuned()[{rows, k}] = best;
    if (tqllm::debug_on("tune")) std::fprintf(stderr, "tq3_tune %d x %d -> variant %d (%.1f us)\n", rows, k, best, best_ms * 1000.f / 30.f);
}

void tq3_matvec_h(const void * w, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols, cudaStream_t s) {
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const auto * wp = static_cast<const uint4 *>(w);
    const int grid = (rows + 15) / 16;
    if (ncols <= 8) {  // tuned shape
        const auto it = tq3_tuned().find({rows, k});
        if (it != tq3_tuned().end() && (it->second != 0 || ncols == 8)) {
            tq3_matvec_h_variant(w, xh, y, rows, k, ncols, it->second, s);
            return;
        }
    }
    // long-K matrices (FFN down) take the prefetching kernel with 2 row tiles per CTA
    // (ncu, base clocks, 27B verify: 5120 x 17408 at 8 columns 72.4 -> 62.3 us)
    if (k >= 12288 && ncols <= 8) {
        tq3_matvec_p_k<2><<<(rows + 31) / 32, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi);
        return;
    }
    switch (ncols) {
        case 1: tq3_matvec_h_k<1><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 2: tq3_matvec_h_k<2><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 3: tq3_matvec_h_k<3><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 4: tq3_matvec_h_k<4><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 5: tq3_matvec_h_k<5><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 6: tq3_matvec_h_k<6><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 7: tq3_matvec_h_k<7><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 8: tq3_matvec_h_k<8><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 9: tq3_matvec_h_k<9><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 10: tq3_matvec_h_k<10><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 11: tq3_matvec_h_k<11><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 12: tq3_matvec_h_k<12><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        default:
            if (ncols > 12 && ncols <= 32) {
                auto go = [&](auto nt_c) {  // 2 row tiles per CTA
                    constexpr int NT = decltype(nt_c)::value;
                    tq3_matvec_pw_k<NT, 2><<<(rows + 31) / 32, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi);
                };
                if (ncols <= 16) go(std::integral_constant<int, 2>{});
                else if (ncols <= 24) go(std::integral_constant<int, 3>{});
                else go(std::integral_constant<int, 4>{});
            }
            else throw std::invalid_argument("tq3_matvec_h: 1..32 columns");
    }
}

void argmax_cols(const float * x, int len, int ncols, uint32_t * out, void * scratch, cudaStream_t s) {
    constexpr int NB = 128;  // scratch: ncols * NB (value, index) pairs
    auto * pv = static_cast<float *>(scratch);
    auto * pi = reinterpret_cast<uint32_t *>(pv + ncols * NB);
    argmax_cols1_k<<<dim3(NB, ncols), 256, 0, s>>>(x, len, pv, pi, NB);
    argmax_cols2_k<<<ncols, 32, 0, s>>>(pv, pi, NB, out);
}

void embed_multi_tq3(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    const int nblk = k / 32;
    tq3_row_k<<<dim3((nblk + 7) / 8, n), 256, 0, s>>>(static_cast<const uint4 *>(w), 0, y, nblk, dtok);
}

void embed_multi_q6k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    q6k_row_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), 0, y, k, dtok);
}

void embed_multi_iq3s(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    iq3s_row_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), 0, y, k, dtok);
}

// Q5_K -> int8 + FP16 scale per 16 (symmetric, max-abs / 127): the affine Q5_K values are not exactly representable,
// int8 adds about 1/8 of a Q5_K step of error. One thread per 16-group.
__global__ void q5k_to_i8_k(const uint8_t * w, int8_t * w8, __half * sc, int64_t ngroups) {
    const int64_t gi = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gi >= ngroups) return;
    const uint8_t * blk = w + (gi / 16) * 176;
    const int e0 = int(gi % 16) * 16, j = e0 / 32;
    const uint8_t *scr = blk + 4, *qh = blk + 16, *qs = blk + 48;
    int s, m;
    if (j < 4) {
        s = scr[j] & 63;
        m = scr[j + 4] & 63;
    } else {
        s = (scr[j + 4] & 0xF) | ((scr[j - 4] >> 6) << 4);
        m = (scr[j + 4] >> 4) | ((scr[j] >> 6) << 4);
    }
    const float d = __half2float(reinterpret_cast<const __half *>(blk)[0]) * float(s);
    const float mn = __half2float(reinterpret_cast<const __half *>(blk)[1]) * float(m);
    float v[16], amax = 0.f;
    for (int t = 0; t < 16; ++t) {
        const int l = (e0 + t) % 32;
        const uint8_t q = qs[32 * (j / 2) + l];
        const int c = (j % 2 ? q >> 4 : q & 0xF) + ((qh[l] >> j) & 1) * 16;
        v[t] = d * float(c) - mn;
        amax = fmaxf(amax, fabsf(v[t]));
    }
    const __half sh = __float2half_rn(amax / 127.f);
    const float st = __half2float(sh), inv = st > 0.f ? 1.f / st : 0.f;
    for (int t = 0; t < 16; ++t) w8[gi * 16 + t] = int8_t(max(-127, min(127, __float2int_rn(v[t] * inv))));
    sc[gi] = sh;
}
void q5k_to_i8(const void * q5k, int8_t * w8, __half * sc, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    q5k_to_i8_k<<<int((ng + 255) / 256), 256, 0, s>>>(static_cast<const uint8_t *>(q5k), w8, sc, ng);
}
void q6k_to_i8(const void * q6k, int8_t * w8, __half * sc, int rows, int k, cudaStream_t s) {
    const int64_t n = int64_t(rows) * k;
    q6k_to_i8_k<<<int((n + 255) / 256), 256, 0, s>>>(static_cast<const uint8_t *>(q6k), w8, sc, rows, k);
}

// Exact int8 re-score of 16 candidate rows per column (candidates from the 4-bit head), argmax (ties -> lower id).
// Weight rounding as in i8h_matvec_k (q * s in FP16), FP32 accumulation. grid = ncols, 16 warps (one per candidate).
__global__ void __launch_bounds__(512) i8h_rescore_argmax_k(const int8_t * w8, const __half * sc, const __half * xh,
                                                            const uint32_t * cand, int k, uint32_t * out, float * scores) {
    __shared__ float sv[16];
    const int c = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t row = cand[c * 16 + warp];
    const int8_t * wr = w8 + int64_t(row) * k;
    const __half * sr = sc + int64_t(row) * (k / 16);
    const __half * xr = xh + int64_t(c) * k;
    float acc = 0.f;
    for (int i = lane * 8; i < k; i += 256) {
        const uint2 wq = __ldg(reinterpret_cast<const uint2 *>(wr + i));
        const uint4 xq = __ldg(reinterpret_cast<const uint4 *>(xr + i));
        const __half s = sr[i / 16];
        const int8_t * wb = reinterpret_cast<const int8_t *>(&wq);
        const __half * xv = reinterpret_cast<const __half *>(&xq);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc += __half2float(__hmul(__int2half_rn(wb[e]), s)) * __half2float(xv[e]);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        sv[warp] = acc;
        if (scores) scores[c * 16 + warp] = acc;  // exact logits of the 16 candidates (sampling)
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bv = sv[0];
        uint32_t bi = cand[c * 16];
        for (int j = 1; j < 16; ++j) {
            const uint32_t id = cand[c * 16 + j];
            if (sv[j] > bv || (sv[j] == bv && id < bi)) bv = sv[j], bi = id;
        }
        out[c] = bi;
    }
}

void i8h_rescore_argmax(const int8_t * w8, const __half * sc, const __half * xh, const uint32_t * cand, int k,
                        int ncols, uint32_t * out, cudaStream_t s, float * scores) {
    if (k % 256) throw std::invalid_argument("i8h_rescore_argmax: k must be a multiple of 256");
    i8h_rescore_argmax_k<<<ncols, 512, 0, s>>>(w8, sc, xh, cand, k, out, scores);
}

void i8h_matvec(const int8_t * w8, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    const int grid = (rows + 15) / 16;
    switch (ncols) {
        case 1: i8h_matvec_k<1><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 2: i8h_matvec_k<2><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 3: i8h_matvec_k<3><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 4: i8h_matvec_k<4><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 5: i8h_matvec_k<5><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 6: i8h_matvec_k<6><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 7: i8h_matvec_k<7><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 8: i8h_matvec_k<8><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 9: i8h_matvec_k<9><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 10: i8h_matvec_k<10><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 11: i8h_matvec_k<11><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 12: i8h_matvec_k<12><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 13: i8h_matvec_k<13><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 14: i8h_matvec_k<14><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 15: i8h_matvec_k<15><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 16: i8h_matvec_k<16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        default: throw std::invalid_argument("i8h_matvec: 1..16 columns");
    }
}

void i8h_quantize(const float * x, int8_t * w8, __half * sc, int64_t n, cudaStream_t s) {
    const int64_t ng = n / 16;
    i8h_quantize_k<<<int((ng + 255) / 256), 256, 0, s>>>(x, w8, sc, ng);
}

static __global__ void bf16_matvec_cols_k(const __nv_bfloat16 * w, const __nv_bfloat16 * x, __nv_bfloat16 * y, int rows,
                                          int k, int ldy) {
    const int r = blockIdx.x * 8 + threadIdx.x / 32, n = blockIdx.y, lane = threadIdx.x % 32;
    if (r >= rows) return;
    const uint4 * wr = reinterpret_cast<const uint4 *>(w + int64_t(r) * k);
    const uint4 * xr = reinterpret_cast<const uint4 *>(x + int64_t(n) * k);
    float acc = 0.f;
    for (int i = lane; i < k / 8; i += 32) {
        const uint4 a = __ldg(&wr[i]), b = __ldg(&xr[i]);
        const auto * ah = reinterpret_cast<const __nv_bfloat162 *>(&a);
        const auto * bh = reinterpret_cast<const __nv_bfloat162 *>(&b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 fa = __bfloat1622float2(ah[j]), fb = __bfloat1622float2(bh[j]);
            acc = fmaf(fa.x, fb.x, acc);
            acc = fmaf(fa.y, fb.y, acc);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (lane == 0) y[int64_t(n) * ldy + r] = __float2bfloat16_rn(acc);
}

void bf16_matvec_cols(const __nv_bfloat16 * w, const __nv_bfloat16 * x, __nv_bfloat16 * y, int rows, int k, int ncols,
                      cudaStream_t s, int ldy) {
    if (k % 256) throw std::invalid_argument("bf16_matvec_cols: k must be a multiple of 256");
    bf16_matvec_cols_k<<<dim3(unsigned((rows + 7) / 8), unsigned(ncols)), 256, 0, s>>>(w, x, y, rows, k, ldy ? ldy : rows);
}

void i8h_matvec_bf16(const int8_t * w8, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    const int grid = (rows + 15) / 16;
    switch (ncols) {
        case 1: i8h_matvec_k<1, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 2: i8h_matvec_k<2, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 3: i8h_matvec_k<3, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 4: i8h_matvec_k<4, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 5: i8h_matvec_k<5, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 6: i8h_matvec_k<6, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 7: i8h_matvec_k<7, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 8: i8h_matvec_k<8, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 9: i8h_matvec_k<9, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 10: i8h_matvec_k<10, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 11: i8h_matvec_k<11, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 12: i8h_matvec_k<12, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 13: i8h_matvec_k<13, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 14: i8h_matvec_k<14, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 15: i8h_matvec_k<15, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 16: i8h_matvec_k<16, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        default: throw std::invalid_argument("i8h_matvec_bf16: 1..16 columns");
    }
}

// Per-shape row tiling for <= 8 columns (u4h_tune; bit-identical for every MT), one map per format.
template <int FMT>
static std::map<std::pair<int, int>, int> & tuned() {
    static std::map<std::pair<int, int>, int> m;
    return m;
}

template <class OUT, int FMT = 0>
static void u4h_launch(const uint8_t * w4, const __half2 * sm, const __half * xh, OUT * y, int rows, int k, int ncols,
                       cudaStream_t s, int mt = -1, const uint32_t * rid = nullptr, int ldy = 0) {
    if (k % 64) throw std::invalid_argument("u4h_matvec: k must be a multiple of 64");
    if (mt < 0 && ncols <= 8) {
        auto & tm = tuned<FMT>();
        const auto it = tm.find({rows, k});
        mt = it == tm.end() ? 1 : it->second;
    }
    if (ncols <= 8 && mt == 5) {  // MT 1 with prefetch
        u4h_matvec_k<8, OUT, 1, FMT, 1><<<(rows + 15) / 16, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
        return;
    }
    if (ncols <= 8 && mt > 1) {
        if (mt >= 4) u4h_matvec_k<8, OUT, 4, FMT><<<(rows + 63) / 64, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
        else u4h_matvec_k<8, OUT, 2, FMT><<<(rows + 31) / 32, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
        return;
    }
    const int grid = (rows + 15) / 16;
    switch (ncols) {
        case 1: u4h_matvec_k<1, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 1, rid, ldy); break;
        case 2: u4h_matvec_k<2, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 2, rid, ldy); break;
        case 3: u4h_matvec_k<3, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 3, rid, ldy); break;
        case 4: u4h_matvec_k<4, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 4, rid, ldy); break;
        case 5: u4h_matvec_k<5, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 5, rid, ldy); break;
        case 6: u4h_matvec_k<6, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 6, rid, ldy); break;
        case 7: u4h_matvec_k<7, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 7, rid, ldy); break;
        case 8: u4h_matvec_k<8, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 8, rid, ldy); break;
        default:
            if (ncols > 8 && ncols <= 16) {
                // 9..16 columns: shared activation loads pay off (3090: head 248320 x 5120 MT 1 / 2 / 4 = 1478 / 1156 /
                // 982 us, 5120 x 17408 116 / 81 / 93 us)
                if (mt < 0) mt = rows >= 65536 ? 4 : 2;
                if (mt >= 4) u4h_matvec_k<16, OUT, 4, FMT><<<(rows + 63) / 64, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
                else if (mt == 2) u4h_matvec_k<16, OUT, 2, FMT><<<(rows + 31) / 32, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
                else u4h_matvec_k<16, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
            }
            else if (ncols > 16 && ncols <= 24) u4h_matvec_k<24, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
            else if (ncols > 24 && ncols <= 32) u4h_matvec_k<32, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
            else throw std::invalid_argument("u4h_matvec: 1..32 columns");
    }
}

void iq3x_matvec_bf16(const uint8_t * w3, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                      int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 5>(w3, reinterpret_cast<const __half2 *>(rec), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void iq3x_matvec(const uint8_t * w3, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols,
                 cudaStream_t s) {
    u4h_launch<float, 5>(w3, reinterpret_cast<const __half2 *>(rec), xh, y, rows, k, ncols, s);
}

void iq3_matvec_bf16(const uint8_t * w3, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 4>(w3, reinterpret_cast<const __half2 *>(rec), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void iq3_matvec(const uint8_t * w3, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    u4h_launch<float, 4>(w3, reinterpret_cast<const __half2 *>(rec), xh, y, rows, k, ncols, s);
}


void k3_matvec_bf16(const uint8_t * w3, const uint8_t * kr, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 2>(w3, reinterpret_cast<const __half2 *>(kr), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void k3_matvec(const uint8_t * w3, const uint8_t * kr, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t s) {
    u4h_launch<float, 2>(w3, reinterpret_cast<const __half2 *>(kr), xh, y, rows, k, ncols, s);
}

void k4_matvec_bf16(const uint8_t * w4, const uint8_t * kh, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 1>(w4, reinterpret_cast<const __half2 *>(kh), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void k4_matvec(const uint8_t * w4, const uint8_t * kh, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t s) {
    u4h_launch<float, 1>(w4, reinterpret_cast<const __half2 *>(kh), xh, y, rows, k, ncols, s);
}

template <int FMT>
static void ud_dispatch_bf16(const uint8_t * w, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                             int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, FMT>(w, reinterpret_cast<const __half2 *>(rec), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void udq_matvec_bf16(int fmt, const uint8_t * w, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    switch (fmt) {
        case 8: ud_dispatch_bf16<8>(w, rec, xh, y, rows, k, ncols, s, ldy); break;
        case 9: ud_dispatch_bf16<9>(w, rec, xh, y, rows, k, ncols, s, ldy); break;
        case 10: ud_dispatch_bf16<10>(w, rec, xh, y, rows, k, ncols, s, ldy); break;
        case 11: ud_dispatch_bf16<11>(w, rec, xh, y, rows, k, ncols, s, ldy); break;
        case 12: ud_dispatch_bf16<12>(w, rec, xh, y, rows, k, ncols, s, ldy); break;
        default: throw std::invalid_argument("udq_matvec: unknown format");
    }
}
void udq_matvec(int fmt, const uint8_t * w, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    const auto * r = reinterpret_cast<const __half2 *>(rec);
    switch (fmt) {
        case 8: u4h_launch<float, 8>(w, r, xh, y, rows, k, ncols, s); break;
        case 9: u4h_launch<float, 9>(w, r, xh, y, rows, k, ncols, s); break;
        case 10: u4h_launch<float, 10>(w, r, xh, y, rows, k, ncols, s); break;
        case 11: u4h_launch<float, 11>(w, r, xh, y, rows, k, ncols, s); break;
        case 12: u4h_launch<float, 12>(w, r, xh, y, rows, k, ncols, s); break;
        default: throw std::invalid_argument("udq_matvec: unknown format");
    }
}
void k6_matvec_bf16(const uint8_t * w6, const uint8_t * kr, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 7>(w6, reinterpret_cast<const __half2 *>(kr), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void k6_matvec(const uint8_t * w6, const uint8_t * kr, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t s) {
    u4h_launch<float, 7>(w6, reinterpret_cast<const __half2 *>(kr), xh, y, rows, k, ncols, s);
}
void k5_matvec_bf16(const uint8_t * w5, const uint8_t * kh, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 6>(w5, reinterpret_cast<const __half2 *>(kh), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void k5_matvec(const uint8_t * w5, const uint8_t * kh, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t s) {
    u4h_launch<float, 6>(w5, reinterpret_cast<const __half2 *>(kh), xh, y, rows, k, ncols, s);
}

void u4h_matvec_bf16(const uint8_t * w4, const __half2 * sm, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    u4h_launch(w4, sm, xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}

void u4h_matvec(const uint8_t * w4, const __half2 * sm, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    u4h_launch(w4, sm, xh, y, rows, k, ncols, s);
}
void u4h_matvec_ids(const uint8_t * w4, const __half2 * sm, const uint32_t * rid, const __half * xh, float * y, int rows,
                    int k, int ncols, cudaStream_t s) {
    u4h_launch(w4, sm, xh, y, rows, k, ncols, s, -1, rid);
}

void u4h_matvec_mt(const uint8_t * w4, const __half2 * sm, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                   int ncols, int mt, cudaStream_t s) {
    u4h_launch(w4, sm, xh, y, rows, k, ncols, s, mt);
}

template <int FMT>
static void tune_impl(const uint8_t * w4, const __half2 * sm, int rows, int k, cudaStream_t s) {
    if (tuned<FMT>().count({rows, k}) || !tune_pointer_ok(w4)) return;
    __half * x = nullptr;
    __nv_bfloat16 * y = nullptr;
    tune_scratch(&x, size_t(8) * k * 2, &y, size_t(8) * rows * 2, "u4h_tune");
    cudaMemsetAsync(x, 0, size_t(8) * k * 2, s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int best = 1;
    float best_ms = 1e30f;
    for (int mt : {1, 2, 4, 5}) {  // 5: MT 1 with prefetch
        for (int i = 0; i < 10; ++i) u4h_launch<__nv_bfloat16, FMT>(w4, sm, x, y, rows, k, 8, s, mt);
        cudaEventRecord(e0, s);
        for (int i = 0; i < 30; ++i) u4h_launch<__nv_bfloat16, FMT>(w4, sm, x, y, rows, k, 8, s, mt);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms * 0.98f) best_ms = ms, best = mt;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    tune_done(x, y, "u4h_tune");
    tuned<FMT>()[{rows, k}] = best;
    if (tqllm::debug_on("tune")) std::fprintf(stderr, "u4h_tune fmt %d %d x %d -> MT %d (%.1f us)\n", FMT, rows, k, best, best_ms * 1000.f / 30.f);
}

void u4h_tune(const uint8_t * w4, const __half2 * sm, int rows, int k, cudaStream_t s) { tune_impl<0>(w4, sm, rows, k, s); }
void iq3x_tune(const uint8_t * w3, const uint8_t * rec, int rows, int k, cudaStream_t s) {
    tune_impl<5>(w3, reinterpret_cast<const __half2 *>(rec), rows, k, s);
}
void iq3_tune(const uint8_t * w3, const uint8_t * rec, int rows, int k, cudaStream_t s) {
    tune_impl<4>(w3, reinterpret_cast<const __half2 *>(rec), rows, k, s);
}
void k4_tune(const uint8_t * w4, const uint8_t * kh, int rows, int k, cudaStream_t s) {
    tune_impl<1>(w4, reinterpret_cast<const __half2 *>(kh), rows, k, s);
}
// matvec through the u4h kernel for any u4h-family fmt (incl. 13 iq4n, 14 iq4l, 15 i8h); mt as u4h_launch (-1: tuned)
void u4h_family_matvec_f32(int fmt, const uint8_t * w, const void * aux, const __half * xh, float * y, int rows, int k,
                           int ncols, cudaStream_t s) {
    const auto * a = reinterpret_cast<const __half2 *>(aux);
    switch (fmt) {
        case 13: u4h_launch<float, 13>(w, a, xh, y, rows, k, ncols, s); break;
        case 14: u4h_launch<float, 14>(w, a, xh, y, rows, k, ncols, s); break;
        case 15: u4h_launch<float, 15>(w, a, xh, y, rows, k, ncols, s); break;
        default: throw std::invalid_argument("u4h_family_matvec_f32: fmt not wired");
    }
}
void u4h_family_tune(int fmt, const uint8_t * w, const void * aux, int rows, int k, cudaStream_t s) {
    const auto * a = reinterpret_cast<const __half2 *>(aux);
    switch (fmt) {
        case 13: tune_impl<13>(w, a, rows, k, s); break;
        case 14: tune_impl<14>(w, a, rows, k, s); break;
        case 15: tune_impl<15>(w, a, rows, k, s); break;
        default: throw std::invalid_argument("u4h_family_tune: fmt not wired");
    }
}
void u4h_family_matvec_bf16(int fmt, const uint8_t * w, const void * aux, const __half * xh, __nv_bfloat16 * y, int rows,
                            int k, int ncols, cudaStream_t s, int ldy, int mt) {
    const auto * a = reinterpret_cast<const __half2 *>(aux);
    switch (fmt) {
        case 13: u4h_launch<__nv_bfloat16, 13>(w, a, xh, y, rows, k, ncols, s, mt, nullptr, ldy); break;
        case 14: u4h_launch<__nv_bfloat16, 14>(w, a, xh, y, rows, k, ncols, s, mt, nullptr, ldy); break;
        case 15: u4h_launch<__nv_bfloat16, 15>(w, a, xh, y, rows, k, ncols, s, mt, nullptr, ldy); break;
        case 6: u4h_launch<__nv_bfloat16, 6>(w, a, xh, y, rows, k, ncols, s, mt, nullptr, ldy); break;
        default: throw std::invalid_argument("u4h_family_matvec: fmt not wired");
    }
}

void udq_tune(int fmt, const uint8_t * w, const uint8_t * rec, int rows, int k, cudaStream_t s) {
    const auto * r = reinterpret_cast<const __half2 *>(rec);
    switch (fmt) {
        case 8: tune_impl<8>(w, r, rows, k, s); break;
        case 9: tune_impl<9>(w, r, rows, k, s); break;
        case 10: tune_impl<10>(w, r, rows, k, s); break;
        case 11: tune_impl<11>(w, r, rows, k, s); break;
        case 12: tune_impl<12>(w, r, rows, k, s); break;
        default: throw std::invalid_argument("udq_tune: unknown format");
    }
}
void k6_tune(const uint8_t * w6, const uint8_t * kr, int rows, int k, cudaStream_t s) {
    tune_impl<7>(w6, reinterpret_cast<const __half2 *>(kr), rows, k, s);
}
void k5_tune(const uint8_t * w5, const uint8_t * kh, int rows, int k, cudaStream_t s) {
    tune_impl<6>(w5, reinterpret_cast<const __half2 *>(kh), rows, k, s);
}
void k3_tune(const uint8_t * w3, const uint8_t * kr, int rows, int k, cudaStream_t s) {
    tune_impl<2>(w3, reinterpret_cast<const __half2 *>(kr), rows, k, s);
}

void u4h_from_i8h(const int8_t * w8, const __half * sc, uint8_t * w4, __half2 * sm, int64_t n, cudaStream_t s) {
    const int64_t nblk = n / 32;
    u4h_from_i8h_k<<<unsigned((nblk + 255) / 256), 256, 0, s>>>(w8, sc, w4, sm, nblk);
}

void tq3_gemm_h(const void * w, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols, cudaStream_t s) {
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const dim3 grid((rows + 63) / 64, (ncols + 63) / 64);
    tq3_gemm_h_k<true><<<grid, 256, 0, s>>>(static_cast<const uint4 *>(w), xh, y, rows, k, ncols, cb.lo, cb.hi);
}

void tq3_expert_matvec(const void * w, const __half * x, __nv_bfloat16 * y, int rows, int k,
                       const int * ids, int n, int topk, bool per_expert, cudaStream_t s) {
    if (!w || !x || !y || !ids || rows <= 0 || k <= 0 || k % 32 || n <= 0 || topk <= 0 ||
        int64_t(n) * topk > 65535)
        throw std::invalid_argument("tq3_expert_matvec: invalid shape or pointer");
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const dim3 grid((rows + 15) / 16, n * topk);
    tq3_matvec_h_k<1, true><<<grid, 256, 0, s>>>(static_cast<const uint4 *>(w), x, y, rows, k,
                                               cb.lo, cb.hi, ids, topk, per_expert);
}

void tq3_expert_gemm(const void * w, const __half * x, __nv_bfloat16 * y, int rows, int k,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert, cudaStream_t s) {
    if (!w || !x || !y || !counts || !groups || rows <= 0 || k <= 0 || k % 32 || capacity <= 0 ||
        experts <= 0 || experts > 65535 || topk <= 0 ||
        (int64_t(capacity) + 63) / 64 > 65535)
        throw std::invalid_argument("tq3_expert_gemm: invalid shape or pointer");
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const dim3 grid((rows + 63) / 64, (capacity + 63) / 64, experts);
    tq3_gemm_h_k<true, true><<<grid, 256, 0, s>>>(static_cast<const uint4 *>(w), x, y, rows, k, capacity,
                                               cb.lo, cb.hi, counts, groups, capacity, topk, per_expert);
}

void tq3_dequant_h(const void * w, __half * out, int rows, int k, cudaStream_t s) {
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const int64_t nb = int64_t(rows) * (k / 32);
    tq3_dequant_h_k<<<unsigned((nb + 255) / 256), 256, 0, s>>>(static_cast<const uint4 *>(w), out, nb, cb.lo, cb.hi);
}

__global__ void h_to_bf16_k(const __half * x, __nv_bfloat16 * y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(__half2float(x[i]));
}

void h_to_bf16(const __half * x, __nv_bfloat16 * y, int64_t n, cudaStream_t s) {
    h_to_bf16_k<<<unsigned((n + 255) / 256), 256, 0, s>>>(x, y, n);
}

void f32_to_bf16(const float * x, __nv_bfloat16 * y, int64_t n, cudaStream_t s) {
    f32_to_bf16_k<<<unsigned((n + 255) / 256), 256, 0, s>>>(x, y, n);
}

void topk16_big(const float * x, int len, int rows, float * vals, uint32_t * idx, void * scratch, cudaStream_t s) {
    constexpr int NCH = 64;
    const int chunk = (len + NCH - 1) / NCH;
    auto * cv = static_cast<float *>(scratch);
    auto * ci = reinterpret_cast<uint32_t *>(cv + size_t(rows) * NCH * 16);
    topk16_chunk_k<<<dim3(NCH, rows), 256, 0, s>>>(x, len, chunk, cv, ci);
    topk16_merge_k<<<rows, 32, 0, s>>>(cv, ci, NCH * 16, vals, idx);
}

void bf16_to_h(const __nv_bfloat16 * x, __half * y, int n, cudaStream_t s) {
    bf16_to_h_k<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
}

void bf16_to_f32(const __nv_bfloat16 * x, float * y, int n, cudaStream_t s) {
    bf16_to_f32_k<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
}

} // namespace tqllm_cuda
