// FNT weight format kernels (host side src/formats/weight/fnt.cpp; format: docs/reference/formats.md, "FNT").
//
// FNT-k (k = KB = 4..7 stream bits per weight pair, k / 2 bits per weight): a tail-biting bitshift-trellis code
// (QTIP-style, V = 2 weights per step, 16-bit window, the table-free code c2 of TQT with a 16-bit window) in the
// 128-block Hadamard basis of TQ4 / TQT. Per 128 weights of a row: KB * 8 stream bytes (KB * 64 bits, little-endian bit
// order; bit index mod KB * 64: the stream wraps inside the block). Weight pair p (weights 2p, 2p + 1) reads the 16-bit
// window w = bits [p * KB, p * KB + 16) and decodes it as
//   y = (w * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFF;  z = y * 0x2C1B3C6D + 0x9E3779B9;  z2 = z * 0x297A2D39 + 0x7F4A7C15;
//   h(v) = (v & 0x8FFF8FFF) ^ 0x3B603B60 (two FP16);  (weight 2p, weight 2p + 1) = FP16x2 h(z) + h(z2) (round to nearest)
// times the FP16 scale of the weight's 640-group (5 blocks). The encoder is a tail-biting Viterbi search (BlockLDLQ).
//
// Device layout (fnt.cpp load): streams [rows][cols / 128][KB * 8 bytes] and scales [rows][cols / 640] FP16.
// The matrix stays in the plain basis for the engine (basis() == Plain): every kernel rotates its input itself, per
// 128-block x' = H D x / sqrt(128) (D = TQ4's signs), in shared memory, before the dot products.
//
// Exactness: all decode-type kernels (expert matvec, dedup, grouped) use one per-lane order: lane l takes pairs 2l, 2l + 1
// of each block (weights 4l .. 4l + 3), part += w * x (FMA, weight order) over the 5 blocks of a group, sum = s * part +
// sum per group, then the warp's xor butterfly 16 .. 1: the same numbers in every kernel (tests/fnt_format.cu).
#include "fnt.cuh"
#include "quant_kernels.cuh"
#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace tqllm_cuda {
using namespace tqllm_dev;

namespace fntk {
constexpr int kGroup = 640;
// TQ4 / TQT rotation signs (tqv12.py sign_vec: bit i % 32 of S1[i / 32] set -> -1)
__constant__ uint32_t kS1[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};  // = tqv12.py S1, tq4k::S1

__device__ __forceinline__ uint32_t code16(uint32_t w) {
    const uint32_t y = (w * 0x9E3779B1u + 0x7F4A7C15u) & 0xFFFFu;
    const uint32_t z = y * 0x2C1B3C6Du + 0x9E3779B9u, z2 = z * 0x297A2D39u + 0x7F4A7C15u;
    return hadd2u((z & 0x8FFF8FFFu) ^ 0x3B603B60u, (z2 & 0x8FFF8FFFu) ^ 0x3B603B60u);
}
// lane l's four unscaled weights (pairs 2l, 2l + 1) of a block whose stream is the 2 * KB words blk
template <int KB> __device__ __forceinline__ void lane_w(const uint32_t * blk, int l, float (&w)[4]) {
    constexpr int NW = 2 * KB;
    const int o = 2 * l * KB, i = o >> 5, sh = o & 31;
    const uint32_t a = blk[i], b = blk[i + 1 == NW ? 0 : i + 1];
    const uint32_t v = __funnelshift_r(a, b, sh);
    const uint32_t c0 = code16(v & 0xFFFFu), c1 = code16((v >> KB) & 0xFFFFu);
    const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&c0));
    const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&c1));
    w[0] = f0.x; w[1] = f0.y; w[2] = f1.x; w[3] = f1.y;
}
__device__ __forceinline__ float sign_of(int i) { return (kS1[i >> 5] >> (i & 31)) & 1u ? -1.f : 1.f; }
// warp-wide forward rotation of one 128-block held as v[4] per lane (element 4l + j): x' = H D x / sqrt(128)
__device__ __forceinline__ void rot128(float (&v)[4], int lane) {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= sign_of(4 * lane + j);
    float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];  // h = 1
    v[0] = a0 + a2; v[2] = a0 - a2; v[1] = a1 + a3; v[3] = a1 - a3;               // h = 2
#pragma unroll
    for (int m = 0; m < 5; ++m) {  // h = 4 << m: lane bit m
        const bool hi = (lane >> m) & 1;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float p = __shfl_xor_sync(0xffffffffu, v[j], 1 << m);
            v[j] = hi ? p - v[j] : v[j] + p;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= 0.08838834764831845f;  // 1 / sqrt(128) in FP32
}
// the inverse (D H y / sqrt(128)): the same butterflies, then the signs
__device__ __forceinline__ void unrot128(float (&v)[4], int lane) {
    float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2; v[2] = a0 - a2; v[1] = a1 + a3; v[3] = a1 - a3;
#pragma unroll
    for (int m = 0; m < 5; ++m) {
        const bool hi = (lane >> m) & 1;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float p = __shfl_xor_sync(0xffffffffu, v[j], 1 << m);
            v[j] = hi ? p - v[j] : v[j] + p;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * 0.08838834764831845f * sign_of(4 * lane + j);
}
// stage nc input rows (FP16, cols each) rotated into shared xs [nc][cols] (FP16); all warps of the block
__device__ __forceinline__ void stage_rot(__half * xs, const __half * x, const int * rows_of, int nc, int cols, int fixed_row) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, nwarps = blockDim.x / 32, nb = cols / 128;
    for (int t = warp; t < nc * nb; t += nwarps) {
        const int c = t / nb, b = t % nb;
        const int64_t r = rows_of ? rows_of[c] : fixed_row;
        const __half2 * src = reinterpret_cast<const __half2 *>(x + r * cols + b * 128 + 4 * lane);
        const float2 p0 = __half22float2(src[0]), p1 = __half22float2(src[1]);
        float v[4] = {p0.x, p0.y, p1.x, p1.y};
        rot128(v, lane);
        __half2 * dst = reinterpret_cast<__half2 *>(xs + int64_t(c) * cols + b * 128 + 4 * lane);
        dst[0] = __floats2half2_rn(v[0], v[1]);
        dst[1] = __floats2half2_rn(v[2], v[3]);
    }
}
// A warp's rows staged through shared memory: the next row's stream is loaded (8-byte coalesced loads into registers; a
// row is a multiple of 8 bytes, not always of 16) while the current one is decoded from shared memory. Rows of up to
// 1280 bytes (cols 2560 at KB 7 = 1120 B).
constexpr int kRowVec = 5, kRowWords = kRowVec * 32 * 2;
struct RowStage {
    uint2 r[kRowVec];
    __device__ __forceinline__ void load(const uint32_t * row, int nvec, int lane) {
        const uint2 * p = reinterpret_cast<const uint2 *>(row);
#pragma unroll
        for (int i = 0; i < kRowVec; ++i) { const int j = lane + 32 * i; r[i] = j < nvec ? __ldg(p + j) : make_uint2(0, 0); }
    }
    __device__ __forceinline__ void store(uint32_t * buf, int nvec, int lane) const {
#pragma unroll
        for (int i = 0; i < kRowVec; ++i) { const int j = lane + 32 * i; if (j < nvec) reinterpret_cast<uint2 *>(buf)[j] = r[i]; }
    }
};
template <class T> __device__ __forceinline__ T out_cast(float v) {
    if constexpr (std::is_same_v<T, float>) return v;
    else return __float2bfloat16_rn(v);
}
// one row's dot products with NC staged columns (nc <= NC present): the per-lane order of the header
template <int KB, int NC>
__device__ __forceinline__ void row_dot(const uint32_t * rw, const __half * __restrict__ sc, const __half * xs,
                                        int cols, int nc, int lane, float (&sum)[NC]) {
    constexpr int NW = 2 * KB;
    const int nb = cols / 128;
#pragma unroll
    for (int c = 0; c < NC; ++c) sum[c] = 0.f;
    for (int g = 0; g < nb / 5; ++g) {
        float part[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) part[c] = 0.f;
        for (int bb = 0; bb < 5; ++bb) {
            const int b = g * 5 + bb;
            float w[4];
            lane_w<KB>(rw + b * NW, lane, w);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                if (c >= nc) break;
                const __half2 * xp = reinterpret_cast<const __half2 *>(xs + c * cols + b * 128 + 4 * lane);
                const float2 x0 = __half22float2(xp[0]), x1 = __half22float2(xp[1]);
                part[c] = __fmaf_rn(w[0], x0.x, part[c]);
                part[c] = __fmaf_rn(w[1], x0.y, part[c]);
                part[c] = __fmaf_rn(w[2], x1.x, part[c]);
                part[c] = __fmaf_rn(w[3], x1.y, part[c]);
            }
        }
        const float s = __half2float(sc[g]);
#pragma unroll
        for (int c = 0; c < NC; ++c) sum[c] = __fmaf_rn(s, part[c], sum[c]);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        for (int d = 16; d; d >>= 1) sum[c] += __shfl_xor_sync(0xffffffffu, sum[c], d);
}

// expert matvec: block = 64 rows (8 warps x 8) of one assignment; ids null -> expert 0 (plain matvec, column a)
template <int KB, class T>
__global__ void __launch_bounds__(256) k_matvec(const uint32_t * __restrict__ w, const __half * __restrict__ sc,
                                                const __half * __restrict__ x, T * __restrict__ y, int rows, int cols,
                                                int experts, const int * __restrict__ ids, int topk, bool per_expert) {
    extern __shared__ __align__(16) unsigned char smem[];
    uint32_t * wbuf = reinterpret_cast<uint32_t *>(smem) + (threadIdx.x / 32) * kRowWords;
    __half * xs = reinterpret_cast<__half *>(smem + 8 * kRowWords * 4);
    const int a = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int expert = ids ? ids[a] : 0;
    const int input = per_expert ? a : a / topk;
    const bool valid = expert >= 0 && expert < experts;
    const int64_t rw_words = int64_t(cols / 128) * 2 * KB, ng = cols / kGroup;
    const int nvec = int(rw_words / 2), row0 = blockIdx.x * 64 + warp * 8, nr = valid ? min(8, rows - row0) : 0;
    RowStage st;
    if (nr > 0) st.load(w + (int64_t(expert) * rows + row0) * rw_words, nvec, lane);
    stage_rot(xs, x, nullptr, 1, cols, input);
    __syncthreads();
    for (int rr = 0; rr < 8; ++rr) {
        const int row = row0 + rr;
        if (row >= rows) break;
        float sum[1];
        if (valid) {
            __syncwarp();
            st.store(wbuf, nvec, lane);
            __syncwarp();
            if (rr + 1 < nr) st.load(w + (int64_t(expert) * rows + row + 1) * rw_words, nvec, lane);
            row_dot<KB, 1>(wbuf, sc + (int64_t(expert) * rows + row) * ng, xs, cols, 1, lane, sum);
        } else sum[0] = nanf("");
        if (!lane) y[int64_t(a) * rows + row] = out_cast<T>(sum[0]);
    }
}
// dedup: per group (quant_kernels.cuh kDedupStride layout) the expert's rows are decoded once for its columns
constexpr int kDedupPass = 4;  // columns staged per pass (a group of up to kDedupCols columns takes 1 or 2 passes)
template <int KB>
__global__ void __launch_bounds__(256) k_dedup(const uint32_t * __restrict__ w, const __half * __restrict__ sc,
                                               const __half * __restrict__ x, __nv_bfloat16 * __restrict__ y, int rows,
                                               int cols, const int * __restrict__ dmap, int groups) {
    constexpr int DC = kDedupCols, NP = kDedupPass;
    extern __shared__ __align__(16) unsigned char smem[];
    uint32_t * wbuf = reinterpret_cast<uint32_t *>(smem) + (threadIdx.x / 32) * kRowWords;
    __half * xs = reinterpret_cast<__half *>(smem + 8 * kRowWords * 4);
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int64_t rw_words = int64_t(cols / 128) * 2 * KB, ng = cols / kGroup;
    const int nvec = int(rw_words / 2), row0 = blockIdx.x * 64 + warp * 8, nr = min(8, rows - row0);
    bool first = true;
    for (int gi = int(blockIdx.y); gi < groups; gi += int(gridDim.y)) {
        const int * e = dmap + int64_t(gi) * kDedupStride;
        if (e[0] < 0) return;
        const uint32_t * we = w + int64_t(e[0]) * rows * rw_words;
        for (int c0 = 0; c0 < e[1]; c0 += NP) {
            const int nc = min(NP, e[1] - c0);
            RowStage st;
            if (nr > 0) st.load(we + row0 * rw_words, nvec, lane);
            if (!first) __syncthreads();
            first = false;
            stage_rot(xs, x, e + 2 + c0, nc, cols, 0);
            __syncthreads();
            for (int rr = 0; rr < nr; ++rr) {
                const int row = row0 + rr;
                __syncwarp();
                st.store(wbuf, nvec, lane);
                __syncwarp();
                if (rr + 1 < nr) st.load(we + (row + 1) * rw_words, nvec, lane);
                float sum[NP];
                row_dot<KB, NP>(wbuf, sc + (int64_t(e[0]) * rows + row) * ng, xs, cols, nc, lane, sum);
#pragma unroll
                for (int c = 0; c < NP; ++c) {
                    if (c >= nc) break;
                    if (!lane) y[int64_t(e[2 + DC + c0 + c]) * rows + row] = __float2bfloat16_rn(sum[c]);
                }
            }
        }
    }
}
// grouped (prefill): block = 64 rows of one expert, its tokens C at a time (moe_group lists)
template <int KB, int C>
__global__ void __launch_bounds__(256) k_grouped(const uint32_t * __restrict__ w, const __half * __restrict__ sc,
                                                 const __half * __restrict__ x, __nv_bfloat16 * __restrict__ y, int rows,
                                                 int cols, const int * __restrict__ counts, const int * __restrict__ groups,
                                                 int capacity, int topk, bool per_expert) {
    extern __shared__ __align__(16) unsigned char smem[];
    uint32_t * wbuf = reinterpret_cast<uint32_t *>(smem) + (threadIdx.x / 32) * kRowWords;
    __half * xs = reinterpret_cast<__half *>(smem + 8 * kRowWords * 4);
    __shared__ int in_rows[C];
    const int expert = blockIdx.y, count = counts[expert];
    if (count == 0) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int64_t rw_words = int64_t(cols / 128) * 2 * KB, ng = cols / kGroup;
    const int nvec = int(rw_words / 2), row0 = blockIdx.x * 64 + warp * 8, nr = min(8, rows - row0);
    const uint32_t * we = w + int64_t(expert) * rows * rw_words;
    const int * g = groups + int64_t(expert) * capacity;
    for (int c0 = 0; c0 < count; c0 += C) {
        const int nc = min(C, count - c0);
        RowStage st;
        if (nr > 0) st.load(we + row0 * rw_words, nvec, lane);
        __syncthreads();
        if (threadIdx.x < nc) { const int a = g[c0 + threadIdx.x]; in_rows[threadIdx.x] = per_expert ? a : a / topk; }
        __syncthreads();
        stage_rot(xs, x, in_rows, nc, cols, 0);
        __syncthreads();
        for (int rr = 0; rr < nr; ++rr) {
            const int row = row0 + rr;
            __syncwarp();
            st.store(wbuf, nvec, lane);
            __syncwarp();
            if (rr + 1 < nr) st.load(we + (row + 1) * rw_words, nvec, lane);
            float sum[C];
            row_dot<KB, C>(wbuf, sc + (int64_t(expert) * rows + row) * ng, xs, cols, nc, lane, sum);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= nc) break;
                if (!lane) y[int64_t(g[c0 + c]) * rows + row] = __float2bfloat16_rn(sum[c]);
            }
        }
    }
}
// FP16 weights in the plain basis: warp per (row, block): decode, scale (FP32 product), unrotate, round
template <int KB>
__global__ void k_dequant(const uint32_t * __restrict__ w, const __half * __restrict__ sc, __half * __restrict__ out,
                          int rows, int cols) {
    const int lane = threadIdx.x & 31;
    const int64_t t = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32, nb = cols / 128;
    if (t >= int64_t(rows) * nb) return;
    const int64_t r = t / nb;
    const int b = int(t % nb);
    float v[4];
    lane_w<KB>(w + t * 2 * KB, lane, v);
    const float s = __half2float(sc[r * (cols / kGroup) + b / 5]);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s;
    unrot128(v, lane);
    __half2 * dst = reinterpret_cast<__half2 *>(out + r * cols + b * 128 + 4 * lane);
    dst[0] = __floats2half2_rn(v[0], v[1]);
    dst[1] = __floats2half2_rn(v[2], v[3]);
}
void check(int kb, const void * w, const void * sc, const void * x, const void * y, int rows, int cols) {
    if (kb < 4 || kb > 7 || !w || !sc || !x || !y || rows <= 0 || cols <= 0 || cols % kGroup)
        throw std::invalid_argument("FNT: invalid pointer, bit rate or shape");
}
}  // namespace fntk

#define FNT_KB(kb, F) switch (kb) { case 4: F(4); break; case 5: F(5); break; case 6: F(6); break; default: F(7); break; }

template <class T>
static void matvec_any(int kb, const void * w, const __half * sc, const __half * x, T * y, int rows, int cols, int experts,
                       const int * ids, int n, int topk, bool per_expert, cudaStream_t s) {
    fntk::check(kb, w, sc, x, y, rows, cols);
    if (cols > 2560) throw std::invalid_argument("FNT: row too long (2560 max)");
    if (n <= 0 || topk <= 0 || int64_t(n) * topk > 65535 || experts <= 0 || int64_t(experts) * rows > INT_MAX)
        throw std::invalid_argument("FNT: invalid expert grid");
    const dim3 grid(unsigned((rows + 63) / 64), unsigned(n * topk));
    const size_t smem = 8 * fntk::kRowWords * 4 + size_t(cols) * sizeof(__half);
#define L(K) fntk::k_matvec<K, T><<<grid, 256, smem, s>>>(static_cast<const uint32_t *>(w), sc, x, y, rows, cols, experts, ids, topk, per_expert)
    FNT_KB(kb, L)
#undef L
}
void fnt_expert_matvec(int kb, const void * w, const __half * sc, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                       int experts, const int * ids, int n, int topk, bool per_expert, cudaStream_t s) {
    matvec_any(kb, w, sc, x, y, rows, cols, experts, ids, n, topk, per_expert, s);
}
void fnt_matvec_f32(int kb, const void * w, const __half * sc, const __half * x, float * y, int rows, int cols, int n, cudaStream_t s) {
    matvec_any(kb, w, sc, x, y, rows, cols, 1, nullptr, n, 1, true, s);
}
void fnt_expert_dedup(int kb, const void * w, const __half * sc, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                      const int * dmap, int groups, cudaStream_t s) {
    fntk::check(kb, w, sc, x, y, rows, cols);
    if (!dmap || groups <= 0 || groups > 65535) throw std::invalid_argument("FNT: invalid dedup map");
    const size_t smem = 8 * fntk::kRowWords * 4 + size_t(fntk::kDedupPass) * cols * sizeof(__half);
    if (smem > 48 * 1024 || cols > 2560) throw std::invalid_argument("FNT: dedup row too long");
    const dim3 grid(unsigned((rows + 63) / 64), unsigned(std::min(groups, kDedupGridGroups)));
#define L(K) fntk::k_dedup<K><<<grid, 256, smem, s>>>(static_cast<const uint32_t *>(w), sc, x, y, rows, cols, dmap, groups)
    FNT_KB(kb, L)
#undef L
}
void fnt_expert_grouped(int kb, const void * w, const __half * sc, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                        const int * counts, const int * groups, int capacity, int experts, int topk, bool per_expert, cudaStream_t s) {
    fntk::check(kb, w, sc, x, y, rows, cols);
    if (!counts || !groups || experts <= 0 || experts > 65535 || capacity <= 0 || int64_t(experts) * rows > INT_MAX)
        throw std::invalid_argument("FNT: invalid grouped expert dimensions");
    constexpr int C = 8;
    const size_t smem = 8 * fntk::kRowWords * 4 + size_t(C) * cols * sizeof(__half);
    if (smem > 96 * 1024 || cols > 2560) throw std::invalid_argument("FNT: grouped row too long");
    static bool attr[4] = {};
    if (smem > 48 * 1024 && !attr[kb - 4]) {
#define A(K) cudaFuncSetAttribute(fntk::k_grouped<K, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)
        FNT_KB(kb, A)
#undef A
        attr[kb - 4] = true;
    }
    const dim3 grid(unsigned((rows + 63) / 64), unsigned(experts));
#define L(K) fntk::k_grouped<K, C><<<grid, 256, smem, s>>>(static_cast<const uint32_t *>(w), sc, x, y, rows, cols, counts, groups, capacity, topk, per_expert)
    FNT_KB(kb, L)
#undef L
}
void fnt_dequant_h(int kb, const void * w, const __half * sc, __half * out, int rows, int cols, cudaStream_t s) {
    fntk::check(kb, w, sc, out, out, rows, cols);
    const int64_t warps = int64_t(rows) * (cols / 128);
    if ((warps + 7) / 8 > INT_MAX) throw std::invalid_argument("FNT: dequant grid overflow");
#define L(K) fntk::k_dequant<K><<<unsigned((warps + 7) / 8), 256, 0, s>>>(static_cast<const uint32_t *>(w), sc, out, rows, cols)
    FNT_KB(kb, L)
#undef L
}
}  // namespace tqllm_cuda
