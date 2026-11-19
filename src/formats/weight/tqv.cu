// TQV-12 weight format kernels (host side src/formats/weight/tqv.cpp; format: docs/reference/formats.md, "TQV-12").
//
// Per 128 weights of a row, in TQ4's rotated basis: one FP16 scale + 4 units of 16 bytes; unit j = weights 32j..32j+31 as
// 4 little-endian uint32 x, y, z, w: x/y/z a 96-bit stream of 8 12-bit codebook indices (group u = weights 4u..4u+3 at
// bits 12u..12u+11), w the 32 sign bits permuted so one shift gives an FP16x2 sign mask: weight 4u at bit 15-u, 4u+1 at
// 31-u, 4u+2 at 7-u, 4u+3 at 23-u. Weight = scale * table[idx][e] * (sign ? -1 : 1), table = the official int8 codebook
// (tqv12_codebook.inc, tag tqv12-v1; values 0..127, decoded exactly to FP16).
//
// Device layout (tqv.cpp load): codes [rows][k / 32] uint4 (k / 2 bytes per row, as TQ4) and scales [rows][k / 128]
// FP16, so a row range is a matrix of its own (row_bytes / aux_row_bytes).
//
// Matvec: a 16-row split-K MMA kernel, int8 table staged in shared memory once
// per CTA: CTA = 8 warps on 16 x MT rows, the 4 lanes of a quad take the 4 segments of one 128-weight scale group, MMA u
// takes group u of each lane's segment for rows g and g + 8; the per-128 scale multiplies the FP32 MMA partials once per
// step. Per row and column the k split, MMA order and 8-warp reduction do not depend on MT, PF, the grid or the column
// count: the same numbers in every configuration (tests/tqv12_format.cu).
#include "common.cuh"
#include "quant_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace tqllm_cuda {
using namespace tqllm_dev;

namespace tqvk {
__device__ const uint32_t TABLE[4096] = {
#include "tqv12_codebook.inc"
};
constexpr int kTableBytes = 4096 * 4;

// group u's 12-bit index from the 3 index words of a segment (u compile-time after unrolling)
__device__ __forceinline__ uint32_t idx12(uint32_t w0, uint32_t w1, uint32_t w2, int u) {
    const int p = 12 * u, wi = p / 32, sh = p % 32;
    const uint32_t lo = wi == 0 ? w0 : wi == 1 ? w1 : w2, hi = wi == 0 ? w1 : w2;
    if (sh + 12 <= 32) return (lo >> sh) & 0xFFFu;
    return __funnelshift_r(lo, hi, sh) & 0xFFFu;
}
// int8x4 entry -> two FP16x2 (weights 4u, 4u+1 / 4u+2, 4u+3) with the group's signs: 1024 + q by PRMT, minus 1024 exact
__device__ __forceinline__ void decode(uint32_t mg, uint32_t sgn, int u, uint32_t & h01, uint32_t & h23) {
    h01 = hadd2u(__byte_perm(mg, 0x64646464u, 0x5140), 0xE400E400u) ^ ((sgn << u) & 0x80008000u);
    h23 = hadd2u(__byte_perm(mg, 0x64646464u, 0x5342), 0xE400E400u) ^ ((sgn << (u + 8)) & 0x80008000u);
}
// sign of weight j (0..31) of a segment in the permuted sign word
__device__ __forceinline__ int sign_bit(int j) {
    const int u = j / 4, e = j % 4;
    return e == 0 ? 15 - u : e == 1 ? 31 - u : e == 2 ? 7 - u : 23 - u;
}
// the 32 FP16 weights of one segment exactly as tqv12_dequant_h writes them
__device__ __forceinline__ void segment_h(uint4 v, __half s, __half (&o)[32]) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const uint32_t mg = __ldg(&TABLE[idx12(v.x, v.y, v.z, u)]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int q = int((mg >> (8 * e)) & 0xFFu), j = 4 * u + e;
            o[j] = __hmul(__int2half_rn((v.w >> sign_bit(j)) & 1u ? -q : q), s);
        }
    }
}
template <class OUT> __device__ __forceinline__ OUT out_cast(float v) {
    if constexpr (std::is_same_v<OUT, float>) return v;
    else return __float2bfloat16_rn(v);
}
}  // namespace tqvk

// y[n][r] (row stride ldy) = sum_k W[r][k] xh[n][k]; NCOL 8 / 16 column capacity, nc <= NCOL present.
// Grid: any number of CTAs; each loops over the 16 x MT row tiles blockIdx.x, + gridDim.x, ... (persistent when the grid
// is smaller than the tile count: the 16 KB table is then staged once per CTA, not once per tile).
template <int NCOL, int MT, int PF, class OUT>
__global__ void __launch_bounds__(256, 2) tqv12_matvec_k(const uint4 * __restrict__ w, const __half * __restrict__ sc,
                                                         const __half * __restrict__ xh, OUT * __restrict__ y, int rows,
                                                         int k, int nc, int ldy) {
    constexpr int NACC = NCOL > 8 ? 8 : 4;
    __shared__ uint32_t tb[4096];
    __shared__ float red[8][32][MT * NACC];
    // the table goes to shared memory after the first tile's first weight loads are issued (its L2 latency overlaps them)
    uint4 tv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) tv[i] = __ldg(reinterpret_cast<const uint4 *>(tqvk::TABLE) + threadIdx.x + 256 * i);
    bool staged = false;

    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int nb = k / 32, ng = k / 128;
    const int ntiles = (rows + 16 * MT - 1) / (16 * MT);
    struct Q { uint4 q[MT][2]; float s[MT][2]; };
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int row0 = tile * 16 * MT;
        int rr[MT][2];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) rr[m][h] = min(row0 + 16 * m + g + 8 * h, rows - 1);
        float c[MT][NACC] = {};
        auto loadq = [&](int b, Q & q) {
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int64_t r = rr[m][h];
                    q.q[m][h] = __ldg(w + r * nb + b);
                    q.s[m][h] = __half2float(sc[r * ng + b / 4]);
                }
        };
        auto loadx = [&](int b, uint4 * x4, uint4 * x42) {
#pragma unroll
            for (int i = 0; i < 4; ++i) x4[i] = x42[i] = make_uint4(0, 0, 0, 0);
            if (g < nc) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g) * k + int64_t(b) * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) x4[i] = __ldg(&xp[i]);
            }
            if constexpr (NCOL > 8) {
                if (g + 8 < nc) {
                    const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g + 8) * k + int64_t(b) * 32);
#pragma unroll
                    for (int i = 0; i < 4; ++i) x42[i] = __ldg(&xp[i]);
                }
            }
        };
        auto step = [&](const Q & cur, const uint4 * x4, const uint4 * x42) {
            const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
            const uint32_t * xw2 = reinterpret_cast<const uint32_t *>(x42);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                float tmp[NACC] = {};
                const uint4 qa = cur.q[m][0], qb = cur.q[m][1];
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    uint32_t a01, a23, b01, b23;
                    tqvk::decode(tb[tqvk::idx12(qa.x, qa.y, qa.z, u)], qa.w, u, a01, a23);
                    tqvk::decode(tb[tqvk::idx12(qb.x, qb.y, qb.z, u)], qb.w, u, b01, b23);
                    mma_f16(tmp, a01, b01, a23, b23, xw[2 * u], xw[2 * u + 1]);
                    if constexpr (NCOL > 8) mma_f16(tmp + 4, a01, b01, a23, b23, xw2[2 * u], xw2[2 * u + 1]);
                }
#pragma unroll
                for (int hh = 0; hh < NACC / 4; ++hh) {
                    c[m][4 * hh + 0] = fmaf(cur.s[m][0], tmp[4 * hh + 0], c[m][4 * hh + 0]);
                    c[m][4 * hh + 1] = fmaf(cur.s[m][0], tmp[4 * hh + 1], c[m][4 * hh + 1]);
                    c[m][4 * hh + 2] = fmaf(cur.s[m][1], tmp[4 * hh + 2], c[m][4 * hh + 2]);
                    c[m][4 * hh + 3] = fmaf(cur.s[m][1], tmp[4 * hh + 3], c[m][4 * hh + 3]);
                }
            }
        };
        Q nxt;
        if (warp * 4 < nb) loadq(warp * 4 + t, nxt);
        if (!staged) {  // block-uniform: first tile of this CTA
#pragma unroll
            for (int i = 0; i < 4; ++i) reinterpret_cast<uint4 *>(tb)[threadIdx.x + 256 * i] = tv[i];
            __syncthreads();
            staged = true;
        }
        for (int b0 = warp * 4; b0 < nb; b0 += 32) {
            const int b = b0 + t;
            const Q cur = nxt;
            if constexpr (PF == 1) {  // next step's weights in flight
                if (b0 + 32 < nb) loadq(b + 32, nxt);
            }
            uint4 x4[4], x42[4];
            loadx(b, x4, x42);
            step(cur, x4, x42);
            if constexpr (PF == 0) {
                if (b0 + 32 < nb) loadq(b + 32, nxt);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int i = 0; i < NACC; ++i) red[warp][lane][m * NACC + i] = c[m][i];
        __syncthreads();
        if (warp < MT) {
            const int m = warp;
            float v[NACC] = {};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < NACC; ++i) v[i] += red[w2][lane][m * NACC + i];
            auto put = [&](int n, int r, float v2) {
                if (n < nc && r < rows) y[int64_t(n) * ldy + r] = tqvk::out_cast<OUT>(v2);
            };
            const int n0 = 2 * t, r = row0 + 16 * m + g;
#pragma unroll
            for (int h = 0; h < NACC / 4; ++h) {
                put(n0 + 8 * h, r, v[4 * h]), put(n0 + 1 + 8 * h, r, v[4 * h + 1]);
                put(n0 + 8 * h, r + 8, v[4 * h + 2]), put(n0 + 1 + 8 * h, r + 8, v[4 * h + 3]);
            }
        }
        __syncthreads();  // red is rewritten by the next tile
    }
}

namespace {
// Kernel choice per shape (3090 measurements in tqv12_pick's comments; same numbers in every choice).
struct TqvPick { int mt, pf, persist; };
TqvPick tqv12_pick(int rows, int k, int nc, int variant) {
    if (variant > 0) {  // tests / benches: variant - 1 = mt2 * 4 + pf * 2 + persist
        const int v = variant - 1;
        return {(v >> 2) & 1 ? 2 : 1, (v >> 1) & 1, v & 1};
    }
    // 3090 (test-tqv12-format --bench, second run after the early table staging; us at 1 / 8 / 16 columns):
    //   few rows: <= 4 columns 1 tile + prefetch (5120 x 17408 59.1 vs 64.1 for 2 tiles; 5120 x 6144 24.5 vs 24.9);
    //   5-8: long k 1 tile + prefetch, one CTA per tile (65.6 vs 74.0), short k 2 tiles + prefetch (26.3 vs 26.8);
    //   9-16: long k 2 tiles + prefetch (105.0 vs 106.8), short k 1 tile + prefetch (37.7 vs 39.4)
    //   many rows (34816 / 16384 / 14336 x 5120): <= 4 columns 1 tile (112.8 / 58.7 / 49.2); 5-8: 2 tiles (123.6 vs
    //   126.0, 55.8 vs 56.3); 9-16: 1 tile + prefetch (197.8 / 98.5 / 85.6 vs 197.5 / 99.2 / 86.6 without)
    if (rows <= 8192) {
        if (nc <= 4) return {1, 1, 1};
        if (nc <= 8) return k > 8192 ? TqvPick{1, 1, 0} : TqvPick{2, 1, 1};
        return k > 8192 ? TqvPick{2, 1, 1} : TqvPick{1, 1, 1};
    }
    if (nc <= 4) return {1, 0, 1};
    if (nc <= 8) return {2, 0, 1};
    return {1, 1, 1};
}

template <int NCOL, int MT, int PF, class OUT>
void tqv12_go(const uint4 * w, const __half * sc, const __half * xh, OUT * y, int rows, int k, int nc, int ldy, bool persist,
              cudaStream_t s) {
    auto kern = tqv12_matvec_k<NCOL, MT, PF, OUT>;
    static const int cap = [&] {
        int per_sm = 0, nsm = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, 256, 0);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        return std::max(1, per_sm * nsm);
    }();
    const int ntiles = (rows + 16 * MT - 1) / (16 * MT);
    const int grid = persist ? std::min(ntiles, cap) : ntiles;
    kern<<<grid, 256, 0, s>>>(w, sc, xh, y, rows, k, nc, ldy);
}

template <class OUT>
void tqv12_launch(const uint8_t * w, const __half * sc, const __half * xh, OUT * y, int rows, int k, int nc, int ldy, int variant,
                  cudaStream_t s) {
    if (k % 128) throw std::invalid_argument("tqv12_matvec: k must be a multiple of 128");
    if (nc < 1 || nc > 16) throw std::invalid_argument("tqv12_matvec: 1..16 columns");
    if (ldy == 0) ldy = rows;
    const TqvPick p = tqv12_pick(rows, k, nc, variant);
    const auto * w4 = reinterpret_cast<const uint4 *>(w);
    auto by = [&](auto ncol_c) {
        constexpr int NC = decltype(ncol_c)::value;
        if (p.mt == 2) {
            if (p.pf) tqv12_go<NC, 2, 1>(w4, sc, xh, y, rows, k, nc, ldy, p.persist, s);
            else tqv12_go<NC, 2, 0>(w4, sc, xh, y, rows, k, nc, ldy, p.persist, s);
        } else {
            if (p.pf) tqv12_go<NC, 1, 1>(w4, sc, xh, y, rows, k, nc, ldy, p.persist, s);
            else tqv12_go<NC, 1, 0>(w4, sc, xh, y, rows, k, nc, ldy, p.persist, s);
        }
    };
    if (nc <= 8) by(std::integral_constant<int, 8>{});
    else by(std::integral_constant<int, 16>{});
}

// FP16 [rows][k]: one thread per 32-weight segment, four 16-byte stores
__global__ void tqv12_dequant_h_k(const uint4 * w, const __half * sc, __half * out, int64_t nseg) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nseg) return;
    __align__(16) __half v[32];
    tqvk::segment_h(__ldg(w + i), sc[i / 4], v);
#pragma unroll
    for (int j = 0; j < 4; ++j) reinterpret_cast<uint4 *>(out)[i * 4 + j] = reinterpret_cast<const uint4 *>(v)[j];
}

// int8 [rows][k] + row scales: one block per row, the row scale amax/127 of the FP16 row tqv12_dequant_h writes (1 for a
// zero row), then the values int8_quant_rows makes of it (src/core/int8_gemm.cu): the dequant_h + int8_quant_rows
// operand without the FP16 round trip. cm (optional): FP16 values times cm[column] first.
template <bool CM>
__global__ void __launch_bounds__(128) tqv12_dequant_i8_k(const uint4 * w, const __half * sc, int k, int8_t * out, float * scale,
                                                          const float * cm) {
    const int64_t r = blockIdx.x, s0 = r * (k / 32);
    const int nseg = k / 32;
    auto seg = [&](int j, float (&f)[32]) {
        __half v[32];
        tqvk::segment_h(__ldg(w + s0 + j), sc[(s0 + j) / 4], v);
#pragma unroll
        for (int e = 0; e < 32; ++e) f[e] = __half2float(v[e]);
        if constexpr (CM) {
#pragma unroll
            for (int e = 0; e < 32; ++e) f[e] *= __ldg(cm + 32 * j + e);
        }
    };
    float m = 0.f;
    for (int j = threadIdx.x; j < nseg; j += 128) {
        float f[32];
        seg(j, f);
#pragma unroll
        for (int e = 0; e < 32; ++e) m = fmaxf(m, fabsf(f[e]));
    }
    __shared__ float red[4];
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float scl = m > 0.f ? m / 127.f : 1.f, inv = 1.f / scl;
    if (threadIdx.x == 0) scale[r] = scl;
    for (int j = threadIdx.x; j < nseg; j += 128) {
        float f[32];
        seg(j, f);
        __align__(16) int8_t q[32];
#pragma unroll
        for (int e = 0; e < 32; ++e) q[e] = int8_t(__float2int_rn(fminf(fmaxf(f[e] * inv, -127.f), 127.f)));
        reinterpret_cast<uint4 *>(out)[(s0 + j) * 2] = reinterpret_cast<const uint4 *>(q)[0];
        reinterpret_cast<uint4 *>(out)[(s0 + j) * 2 + 1] = reinterpret_cast<const uint4 *>(q)[1];
    }
}

// ---- Prefill GEMM without an FP16 copy of the matrix: tq_gemm_h_k of tq4.cu (CTA 128 rows x 64 columns, one 128-weight
// block per step, FP16 MMA accumulation inside the block, FP32 across blocks) with the TQV segment decode in its store
// step (the values tqv12_dequant_h writes).
namespace tqvg {
constexpr int BM = 128, BN = 64, BK = 128, LDS = BK + 8;
constexpr size_t kSmem = size_t(BM + BN) * LDS * sizeof(__half);
__device__ __forceinline__ void mma16816(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
}  // namespace tqvg

__global__ void __launch_bounds__(256) tqv12_gemm_h_k(const uint4 * w, const __half * sc, const __half * xh, __nv_bfloat16 * y,
                                                      int rows, int k, int n, int ldy) {
    using namespace tqvg;
    extern __shared__ __align__(16) __half smem[];
    __half * Ws = smem;               // [BM][LDS]
    __half * Xs = smem + BM * LDS;    // [BN][LDS]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, nb128 = k / BK, nb = k / 32;
    const int wm = (warp & 3) * 32, wn = (warp >> 2) * 32;
    uint4 wraw[2], xraw[4];
    __half wsc[2];
    auto load = [&](int kb) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int u = tid + 256 * j, r = u >> 2, seg = u & 3, row = m0 + r;
            wsc[j] = row < rows ? sc[size_t(row) * nb128 + kb] : __float2half(0.f);
            wraw[j] = row < rows ? __ldg(w + size_t(row) * nb + size_t(kb) * 4 + seg) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = tid + 256 * j, c = u >> 4, part = u & 15, col = n0 + c;
            xraw[j] = col < n ? __ldg(reinterpret_cast<const uint4 *>(xh + size_t(col) * k + size_t(kb) * BK) + part)
                              : make_uint4(0, 0, 0, 0);
        }
    };
    auto store = [&] {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int u = tid + 256 * j, r = u >> 2, seg = u & 3;
            __align__(16) __half v[32];
            tqvk::segment_h(wraw[j], wsc[j], v);
            uint4 * dst = reinterpret_cast<uint4 *>(Ws + r * LDS + seg * 32);
#pragma unroll
            for (int q = 0; q < 4; ++q) dst[q] = reinterpret_cast<const uint4 *>(v)[q];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = tid + 256 * j, c = u >> 4, part = u & 15;
            *reinterpret_cast<uint4 *>(Xs + c * LDS + part * 8) = xraw[j];
        }
    };
    float acc[2][4][4] = {};
    load(0);
    for (int kb = 0; kb < nb128; ++kb) {
        __syncthreads();
        store();
        __syncthreads();
        if (kb + 1 < nb128) load(kb + 1);
        uint32_t hacc[2][4][2] = {};
#pragma unroll
        for (int k16 = 0; k16 < BK; k16 += 16) {
            uint32_t a[2][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const __half * base = Ws + (wm + mi * 16 + g) * LDS + k16 + 2 * t;
                a[mi][0] = *reinterpret_cast<const uint32_t *>(base);
                a[mi][1] = *reinterpret_cast<const uint32_t *>(base + 8 * LDS);
                a[mi][2] = *reinterpret_cast<const uint32_t *>(base + 8);
                a[mi][3] = *reinterpret_cast<const uint32_t *>(base + 8 * LDS + 8);
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const __half * base = Xs + (wn + ni * 8 + g) * LDS + k16 + 2 * t;
                b[ni][0] = *reinterpret_cast<const uint32_t *>(base);
                b[ni][1] = *reinterpret_cast<const uint32_t *>(base + 8);
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma16816(hacc[mi][ni], a[mi], b[ni]);
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&hacc[mi][ni][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&hacc[mi][ni][1]));
                acc[mi][ni][0] += lo.x; acc[mi][ni][1] += lo.y; acc[mi][ni][2] += hi.x; acc[mi][ni][3] += hi.y;
            }
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int r0 = m0 + wm + mi * 16 + g, c0 = n0 + wn + ni * 8 + 2 * t;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = r0 + 8 * h;
                if (r >= rows) continue;
                if (c0 < n) y[size_t(c0) * ldy + r] = __float2bfloat16_rn(acc[mi][ni][2 * h]);
                if (c0 + 1 < n) y[size_t(c0 + 1) * ldy + r] = __float2bfloat16_rn(acc[mi][ni][2 * h + 1]);
            }
        }
}
}  // namespace

void tqv12_matvec_bf16(const uint8_t * w, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols,
                       int ldy, cudaStream_t s) {
    tqv12_launch(w, sc, xh, y, rows, k, ncols, ldy, 0, s);
}
void tqv12_matvec(const uint8_t * w, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols, cudaStream_t s) {
    tqv12_launch(w, sc, xh, y, rows, k, ncols, rows, 0, s);
}
void tqv12_matvec_variant(const uint8_t * w, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols, int variant,
                          cudaStream_t s) {
    tqv12_launch(w, sc, xh, y, rows, k, ncols, rows, variant, s);
}
void tqv12_dequant_h(const uint8_t * w, const __half * sc, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t n = int64_t(rows) * (k / 32);
    if (n > 0) tqv12_dequant_h_k<<<unsigned((n + 255) / 256), 256, 0, s>>>(reinterpret_cast<const uint4 *>(w), sc, out, n);
}
void tqv12_gemm_h(const uint8_t * w, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int n, int ldy,
                  cudaStream_t s) {
    if (k % 128 || n < 1 || rows < 1) throw std::invalid_argument("tqv12_gemm_h: k % 128 == 0, n >= 1");
    static const bool attr = [] {
        cudaFuncSetAttribute(tqv12_gemm_h_k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(tqvg::kSmem));
        return true;
    }();
    (void)attr;
    const dim3 grid(unsigned((rows + tqvg::BM - 1) / tqvg::BM), unsigned((n + tqvg::BN - 1) / tqvg::BN));
    tqv12_gemm_h_k<<<grid, 256, tqvg::kSmem, s>>>(reinterpret_cast<const uint4 *>(w), sc, xh, y, rows, k, n, ldy);
}
void tqv12_dequant_i8(const uint8_t * w, const __half * sc, int8_t * out, float * scale, int rows, int k, cudaStream_t s,
                      const float * cm) {
    if (rows < 1) return;
    const auto * w4 = reinterpret_cast<const uint4 *>(w);
    if (cm) tqv12_dequant_i8_k<true><<<rows, 128, 0, s>>>(w4, sc, k, out, scale, cm);
    else tqv12_dequant_i8_k<false><<<rows, 128, 0, s>>>(w4, sc, k, out, scale, nullptr);
}

}  // namespace tqllm_cuda
