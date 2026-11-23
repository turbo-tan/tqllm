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
template <class T> __device__ __forceinline__ T out_cast(float v) {
    if constexpr (std::is_same_v<T, float>) return v;
    else return __float2bfloat16_rn(v);
}
// ---- tensor-core kernel (all expert paths) ----
// CTA = 8 warps = RT 16-row tiles x KG k-parts (KG = 640-groups per part: cols 2560 -> KG 4, RT 2; cols 640 -> KG 1,
// RT 8). Lane (g, t) of a warp decodes rows g and g + 8 of its tile, segment t (pairs 16t .. 16t + 15, weights
// 32t .. 32t + 31) of each 128-block; MMA u takes pairs 2u, 2u + 1 (TQT's k order, the inputs loaded to match).
// Per row and column: the 5 blocks of a group in one FP32 MMA accumulator, c = s * tmp + c per group, the k-parts summed
// in part order through shared memory: the same numbers for every kernel, column set and grid (tests/fnt_format.cu).
constexpr int kCols = 4, kXPad = 8;  // columns per pass (half an MMA n tile: 3 CTAs per SM); FP16 row padding of the staged inputs
__device__ __forceinline__ int kparts(int cols) { return cols >= 4 * kGroup ? 4 : cols >= 2 * kGroup ? 2 : 1; }
// 5 stream words of row-block blk starting at segment t's first bit (pre-shifted when it is mid-word)
template <int KB> __device__ __forceinline__ void seg_words(const uint32_t * __restrict__ blk, int t, uint32_t (&E)[5]) {
    constexpr int NW = 2 * KB;
    const int bw = (t * KB) >> 1;
    uint32_t r[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) { int j = bw + i; j -= j >= NW ? NW : 0; r[i] = __ldg(blk + j); }
    if ((t * KB) & 1) {
#pragma unroll
        for (int i = 0; i < 5; ++i) E[i] = __funnelshift_r(r[i], r[i + 1], 16);
    } else {
#pragma unroll
        for (int i = 0; i < 5; ++i) E[i] = r[i];
    }
}
template <int KB> __device__ __forceinline__ uint32_t seg_pair(const uint32_t (&E)[5], int j) {
    const int o = j * KB;  // compile-time with j unrolled
    return code16(__funnelshift_r(E[o >> 5], E[(o >> 5) + 1], o & 31));  // y uses the window's low 16 bits only
}
// One pass: rows [row0, row0 + RT * 16) of expert e against nc <= 8 staged columns xs [8][cols + kXPad]; returns the
// lane's 4 outputs (rows g, g + 8 x columns 2t, 2t + 1) on the k-part-0 warps (valid = true)
template <int KB>
__device__ __forceinline__ void mma_pass(const uint32_t * __restrict__ w, const __half * __restrict__ sc, const __half * xs,
                                         int64_t erow0, int rows_left, int cols, float * red, float (&out)[4], bool & valid) {
    constexpr int NW = 2 * KB;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int KG = kparts(cols), kp = warp % KG, rt = warp / KG;
    const int ng = cols / kGroup, gpp = ng / KG;  // 640-groups per k-part
    const int nb = cols / 128, xstride = cols + kXPad;
    const int ra = min(rt * 16 + g, rows_left - 1), rb = min(rt * 16 + g + 8, rows_left - 1);
    const uint32_t * wa = w + (erow0 + ra) * int64_t(nb) * NW;
    const uint32_t * wb = w + (erow0 + rb) * int64_t(nb) * NW;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int gi = kp * gpp; gi < (kp + 1) * gpp; ++gi) {
        float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int bb = 0; bb < 5; ++bb) {
            const int b = gi * 5 + bb;
            uint32_t Ea[5], Eb[5];
            seg_words<KB>(wa + b * NW, t, Ea);
            seg_words<KB>(wb + b * NW, t, Eb);
            const uint4 * xp = reinterpret_cast<const uint4 *>(xs + min(g, kCols - 1) * xstride + b * 128 + t * 32);
            uint4 x4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) x4[i] = g < kCols ? xp[i] : make_uint4(0, 0, 0, 0);
            const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
#pragma unroll
            for (int u = 0; u < 8; ++u)
                mma_f16(tmp, seg_pair<KB>(Ea, 2 * u), seg_pair<KB>(Eb, 2 * u), seg_pair<KB>(Ea, 2 * u + 1),
                        seg_pair<KB>(Eb, 2 * u + 1), xw[2 * u], xw[2 * u + 1]);
        }
        const float sa = __half2float(sc[(erow0 + ra) * ng + gi]), sb = __half2float(sc[(erow0 + rb) * ng + gi]);
        c[0] = __fadd_rn(c[0], __fmul_rn(sa, tmp[0])); c[1] = __fadd_rn(c[1], __fmul_rn(sa, tmp[1]));  // product, then sum:
        c[2] = __fadd_rn(c[2], __fmul_rn(sb, tmp[2])); c[3] = __fadd_rn(c[3], __fmul_rn(sb, tmp[3]));  // k_gemm's order
    }
    if (KG > 1) {
        __syncthreads();  // red is free (the previous pass read it)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[(warp * 32 + lane) * 4 + i] = c[i];
        __syncthreads();
        if (kp == 0)
            for (int q = 1; q < KG; ++q)
#pragma unroll
                for (int i = 0; i < 4; ++i) c[i] += red[((warp + q) * 32 + lane) * 4 + i];
    }
    valid = kp == 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = c[i];
}
// shared layout: xs [kCols][cols + kXPad] FP16, then red [8 warps][32][4] F32
__host__ __device__ constexpr size_t mma_smem(int cols) { return size_t(kCols) * (cols + kXPad) * 2 + 8 * 32 * 4 * 4; }
// stage nc columns (input rows in_rows[c]) rotated; columns nc..7 zero
__device__ __forceinline__ void stage_cols(__half * xs, const __half * x, const int * in_rows, int nc, int cols) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, nb = cols / 128, xstride = cols + kXPad;
    for (int q = warp; q < kCols * nb; q += 8) {
        const int c = q / nb, b = q % nb;
        __half2 * dst = reinterpret_cast<__half2 *>(xs + c * xstride + b * 128 + 4 * lane);
        if (c >= nc) { dst[0] = dst[1] = __floats2half2_rn(0.f, 0.f); continue; }
        const __half2 * src = reinterpret_cast<const __half2 *>(x + int64_t(in_rows[c]) * cols + b * 128 + 4 * lane);
        const float2 p0 = __half22float2(src[0]), p1 = __half22float2(src[1]);
        float v[4] = {p0.x, p0.y, p1.x, p1.y};
        rot128(v, lane);
        dst[0] = __floats2half2_rn(v[0], v[1]);
        dst[1] = __floats2half2_rn(v[2], v[3]);
    }
}
// work item -> (expert, up to 8 (input row, output row) pairs); MODE 0: assignment a = blockIdx.y (ids, topk);
// 1: dedup map entry; 2: moe_group lists (the expert's tokens 8 at a time)
template <int KB, int MODE, class T>
__global__ void __launch_bounds__(256, KB % 2 ? 2 : 3) k_mma(const uint32_t * __restrict__ w, const __half * __restrict__ sc,
                                                const __half * __restrict__ x, T * __restrict__ y, int rows, int cols,
                                                int experts, const int * __restrict__ ids, int topk, bool per_expert,
                                                const int * __restrict__ dmap, int ngroups, const int * __restrict__ counts,
                                                const int * __restrict__ groups, int capacity) {
    extern __shared__ __align__(16) unsigned char smem[];
    __half * xs = reinterpret_cast<__half *>(smem);
    float * red = reinterpret_cast<float *>(smem + size_t(kCols) * (cols + kXPad) * 2);
    __shared__ int in_rows[kCols], out_rows[kCols];
    const int lane = threadIdx.x & 31, g = lane / 4, t = lane % 4;
    const int KG = kparts(cols), RT = 8 / KG, rt = (threadIdx.x / 32) / KG;
    const int row0 = blockIdx.x * RT * 16;
    if (row0 >= rows) return;
    const int rows_left = rows - row0;
    for (int it = 0; ; ++it) {
        const int item = int(blockIdx.y) + it * int(gridDim.y);
        int expert = 0, total = 0;
        if (MODE == 0) { if (it > 0) break; expert = ids ? ids[blockIdx.y] : 0; total = 1; }
        else if (MODE == 1) { if (item >= ngroups) break; const int * e = dmap + int64_t(item) * kDedupStride; expert = e[0]; total = e[0] < 0 ? 0 : e[1]; if (e[0] < 0) break; }
        else { if (item >= experts) break; expert = item; total = counts[item]; }
        const bool bad = MODE == 0 && (expert < 0 || expert >= experts);
        for (int c0 = 0; c0 < total; c0 += kCols) {
            const int nc = min(kCols, total - c0);
            __syncthreads();  // the previous pass finished reading xs / in_rows
            if (threadIdx.x < nc) {
                int in = 0, o = 0;
                if (MODE == 0) { o = blockIdx.y; in = per_expert ? o : o / topk; }
                else if (MODE == 1) { const int * e = dmap + int64_t(item) * kDedupStride; in = e[2 + c0 + threadIdx.x]; o = e[2 + kDedupCols + c0 + threadIdx.x]; }
                else { o = groups[int64_t(item) * capacity + c0 + threadIdx.x]; in = per_expert ? o : o / topk; }
                in_rows[threadIdx.x] = in; out_rows[threadIdx.x] = o;
            }
            __syncthreads();
            stage_cols(xs, x, in_rows, nc, cols);
            __syncthreads();
            float v[4]; bool valid;
            mma_pass<KB>(w, sc, xs, int64_t(bad ? 0 : expert) * rows + row0, rows_left, cols, red, v, valid);
            if (valid) {
                const int r = rt * 16 + g;
                const auto put = [&](int col, int rr, float val) {
                    if (col < nc && rr < rows_left) y[int64_t(out_rows[col]) * rows + row0 + rr] = out_cast<T>(bad ? nanf("") : val);
                };
                put(2 * t, r, v[0]); put(2 * t + 1, r, v[1]); put(2 * t, r + 8, v[2]); put(2 * t + 1, r + 8, v[3]);
            }
        }
    }
}
// Prefill grouped GEMM (moe_group lists): CTA = 8 warps x one 16-row tile each (128 rows), the expert's tokens 32 at a
// time (4 MMA n tiles), the 640-groups in order with the inputs staged per group. Per row and token the numbers of
// k_mma: the same block MMAs per group, product s * tmp per group, the groups summed in order (cols 640 and 2560).
constexpr int kGemmTok = 32;
__host__ __device__ constexpr size_t gemm_smem() { return size_t(kGemmTok) * (kGroup + kXPad) * 2; }
template <int KB>
__global__ void __launch_bounds__(256, 2) k_gemm(const uint32_t * __restrict__ w, const __half * __restrict__ sc,
                                                 const __half * __restrict__ x, __nv_bfloat16 * __restrict__ y, int rows,
                                                 int cols, const int * __restrict__ counts, const int * __restrict__ groups,
                                                 int capacity, int topk, bool per_expert) {
    constexpr int NW = 2 * KB, NT = kGemmTok / 8, XS = kGroup + kXPad;
    extern __shared__ __align__(16) unsigned char smem[];
    __half * xs = reinterpret_cast<__half *>(smem);
    __shared__ int in_rows[kGemmTok], out_rows[kGemmTok];
    const int expert = blockIdx.z, count = counts[expert];
    const int c0 = blockIdx.y * kGemmTok;
    if (c0 >= count) return;
    const int nc = min(kGemmTok, count - c0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 128 + warp * 16, ng = cols / kGroup, nb = cols / 128;
    if (threadIdx.x < kGemmTok) {
        const int a = threadIdx.x < nc ? groups[int64_t(expert) * capacity + c0 + threadIdx.x] : 0;
        out_rows[threadIdx.x] = a; in_rows[threadIdx.x] = per_expert ? a : a / topk;
    }
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    const uint32_t * wa = w + (int64_t(expert) * rows + ra) * int64_t(nb) * NW;
    const uint32_t * wb = w + (int64_t(expert) * rows + rb) * int64_t(nb) * NW;
    float c[NT][4] = {};
    for (int gi = 0; gi < ng; ++gi) {
        __syncthreads();  // in_rows written / the previous group's inputs read
        for (int q = warp; q < kGemmTok * 5; q += 8) {  // stage the group's 5 blocks of every token, rotated
            const int tok = q / 5, b = q % 5;
            __half2 * dst = reinterpret_cast<__half2 *>(xs + tok * XS + b * 128 + 4 * lane);
            if (tok >= nc) { dst[0] = dst[1] = __floats2half2_rn(0.f, 0.f); continue; }
            const __half2 * src = reinterpret_cast<const __half2 *>(x + int64_t(in_rows[tok]) * cols + gi * kGroup + b * 128 + 4 * lane);
            const float2 p0 = __half22float2(src[0]), p1 = __half22float2(src[1]);
            float v[4] = {p0.x, p0.y, p1.x, p1.y};
            rot128(v, lane);
            dst[0] = __floats2half2_rn(v[0], v[1]);
            dst[1] = __floats2half2_rn(v[2], v[3]);
        }
        __syncthreads();
        float tmp[NT][4] = {};
#pragma unroll
        for (int bb = 0; bb < 5; ++bb) {
            const int b = gi * 5 + bb;
            uint32_t Ea[5], Eb[5];
            seg_words<KB>(wa + b * NW, t, Ea);
            seg_words<KB>(wb + b * NW, t, Eb);
            uint32_t A[8][4];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                A[u][0] = seg_pair<KB>(Ea, 2 * u); A[u][1] = seg_pair<KB>(Eb, 2 * u);
                A[u][2] = seg_pair<KB>(Ea, 2 * u + 1); A[u][3] = seg_pair<KB>(Eb, 2 * u + 1);
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xs + (nt * 8 + g) * XS + bb * 128 + t * 32);
                uint4 x4[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) x4[i] = xp[i];
                const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
#pragma unroll
                for (int u = 0; u < 8; ++u) mma_f16(tmp[nt], A[u][0], A[u][1], A[u][2], A[u][3], xw[2 * u], xw[2 * u + 1]);
            }
        }
        const float sa = __half2float(sc[(int64_t(expert) * rows + ra) * ng + gi]), sb = __half2float(sc[(int64_t(expert) * rows + rb) * ng + gi]);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            c[nt][0] = __fadd_rn(c[nt][0], __fmul_rn(sa, tmp[nt][0])); c[nt][1] = __fadd_rn(c[nt][1], __fmul_rn(sa, tmp[nt][1]));
            c[nt][2] = __fadd_rn(c[nt][2], __fmul_rn(sb, tmp[nt][2])); c[nt][3] = __fadd_rn(c[nt][3], __fmul_rn(sb, tmp[nt][3]));
        }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int col = nt * 8 + 2 * t;
        const auto put = [&](int cc, int r, float v) { if (cc < nc && r < rows) y[int64_t(out_rows[cc]) * rows + r] = __float2bfloat16_rn(v); };
        put(col, row0 + g, c[nt][0]); put(col + 1, row0 + g, c[nt][1]); put(col, row0 + g + 8, c[nt][2]); put(col + 1, row0 + g + 8, c[nt][3]);
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

template <int MODE, class T>
static void launch_mma(int kb, const void * w, const __half * sc, const __half * x, T * y, int rows, int cols, int experts,
                       const int * ids, int topk, bool per_expert, const int * dmap, int ngroups, const int * counts,
                       const int * groups, int capacity, unsigned gy, cudaStream_t s) {
    const int KG = cols >= 2560 ? 4 : cols >= 1280 ? 2 : 1, RT = 8 / KG;
    const dim3 grid(unsigned((rows + RT * 16 - 1) / (RT * 16)), gy);
    const size_t smem = fntk::mma_smem(cols);
    static bool attr[4] = {};
#define L(K) do { \
        auto kern = fntk::k_mma<K, MODE, T>; \
        if (smem > 48 * 1024 && !attr[K - 4]) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)); attr[K - 4] = true; } \
        kern<<<grid, 256, smem, s>>>(static_cast<const uint32_t *>(w), sc, x, y, rows, cols, experts, ids, topk, per_expert, dmap, ngroups, counts, groups, capacity); \
    } while (0)
    FNT_KB(kb, L)
#undef L
}
static void check_cols(int cols) {
    if (cols != 640 && cols != 1280 && cols != 2560) throw std::invalid_argument("FNT: kernels take rows of 640, 1280 or 2560");
}
void fnt_expert_matvec(int kb, const void * w, const __half * sc, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                       int experts, const int * ids, int n, int topk, bool per_expert, cudaStream_t s) {
    fntk::check(kb, w, sc, x, y, rows, cols); check_cols(cols);
    if (n <= 0 || topk <= 0 || int64_t(n) * topk > 65535 || experts <= 0 || int64_t(experts) * rows > INT_MAX)
        throw std::invalid_argument("FNT: invalid expert grid");
    launch_mma<0>(kb, w, sc, x, y, rows, cols, experts, ids, topk, per_expert, nullptr, 0, nullptr, nullptr, 0, unsigned(n * topk), s);
}
void fnt_matvec_f32(int kb, const void * w, const __half * sc, const __half * x, float * y, int rows, int cols, int n, cudaStream_t s) {
    fntk::check(kb, w, sc, x, y, rows, cols); check_cols(cols);
    if (n <= 0 || n > 65535) throw std::invalid_argument("FNT: invalid column count");
    launch_mma<0>(kb, w, sc, x, y, rows, cols, 1, nullptr, 1, true, nullptr, 0, nullptr, nullptr, 0, unsigned(n), s);
}
void fnt_expert_dedup(int kb, const void * w, const __half * sc, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                      const int * dmap, int groups, cudaStream_t s) {
    fntk::check(kb, w, sc, x, y, rows, cols); check_cols(cols);
    if (!dmap || groups <= 0 || groups > 65535) throw std::invalid_argument("FNT: invalid dedup map");
    launch_mma<1>(kb, w, sc, x, y, rows, cols, 0, nullptr, 1, true, dmap, groups, nullptr, nullptr, 0,
                  unsigned(std::min(groups, kDedupGridGroups)), s);
}
void fnt_expert_grouped(int kb, const void * w, const __half * sc, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                        const int * counts, const int * groups, int capacity, int experts, int topk, bool per_expert, cudaStream_t s) {
    fntk::check(kb, w, sc, x, y, rows, cols); check_cols(cols);
    if (!counts || !groups || experts <= 0 || experts > 65535 || capacity <= 0 || int64_t(experts) * rows > INT_MAX)
        throw std::invalid_argument("FNT: invalid grouped expert dimensions");
    const unsigned ty = unsigned((capacity + fntk::kGemmTok - 1) / fntk::kGemmTok);
    if (ty > 65535) throw std::invalid_argument("FNT: grouped capacity too large");
    const dim3 grid(unsigned((rows + 127) / 128), ty, unsigned(experts));
#define L(K) fntk::k_gemm<K><<<grid, 256, fntk::gemm_smem(), s>>>(static_cast<const uint32_t *>(w), sc, x, y, rows, cols, counts, groups, capacity, topk, per_expert)
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
