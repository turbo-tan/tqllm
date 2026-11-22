// TQT weight format kernels (host side src/formats/weight/tqt.cpp; format: docs/reference/formats.md, "TQT").
//
// TQT = a tail-biting bitshift-trellis code (QTIP-style, V = 2 weights per 8-bit step, 18-bit window) in TQ4's rotated
// basis. Per 128 weights of a row: one FP16 scale + 64 stream bytes B_0..B_63. Weight pair p (weights 2p, 2p+1) reads
// the 18-bit window w = (B_p | B_{p+1} << 8 | B_{p+2} << 16) & 0x3FFFF (byte indices mod 64: the stream wraps inside the
// block) and decodes it with the table-free code "c2" (tag tqt-c2-v1):
//   x = (w * 0x9E3779B1 + 0x7F4A7C15) << 14;  y = rotl(x, 16);  z = y * 0x2C1B3C6D + 0x9E3779B9;
//   z2 = z * 0x297A2D39 + 0x7F4A7C15;  h(v) = (v & 0x8FFF8FFF) ^ 0x3B603B60 (two FP16);
//   (weight 2p, weight 2p+1) = FP16x2 h(z) + h(z2) (round to nearest), times the scale.
// The low 16 bits are weight 2p. The encoder (tools/convert/qwen35/tqt.py) is a Viterbi search over the 1024-state trellis.
//
// Device layout (tqt.cpp load): codes [rows][k / 32] uint4 (16 stream bytes = 16 pairs per 32-weight segment, k / 2 bytes
// per row, as TQ4) and scales [rows][k / 128] FP16, so a row range is a matrix of its own (row_bytes / aux_row_bytes).
// A segment's last two pairs read the first two bytes of the next segment of the same block (the first after the last).
//
// Matvec: the TQV-12 kernel skeleton (tqv.cu): CTA = 8 warps on 16 x MT rows, the 4 lanes of a quad take the 4 segments
// of one 128-weight block (the next segment's first word by an L1-hit load), MMA u takes pairs 2u, 2u+1 of each lane's
// segment for rows g and g + 8; the per-128 scale multiplies the FP32 MMA partials once per step. Per row and column the
// k split, MMA order and 8-warp reduction do not depend on MT, PF, the grid or the column count: the same numbers in
// every configuration (tests/tqt_format.cu). No table: the decode is 7 integer ops and one FP16x2 add per pair.
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

namespace tqtk {
// pair p (0..15) of a segment: E = {x, y, z, w, first word of the next segment}
__device__ __forceinline__ uint32_t pair(const uint32_t (&E)[5], int p) {
    const int i = p / 4, o = p % 4;
    const uint32_t v = o == 0 ? E[i] : __funnelshift_r(E[i], E[i + 1], 8 * o);  // window in bits 0-17 (bits 18+ shifted out)
    const uint32_t x = v * 0xDE6C4000u + 0x9F054000u;                           // (w * HA + HB) << 14
    const uint32_t y = __funnelshift_l(x, x, 16);
    const uint32_t z = y * 0x2C1B3C6Du + 0x9E3779B9u, z2 = z * 0x297A2D39u + 0x7F4A7C15u;
    return hadd2u((z & 0x8FFF8FFFu) ^ 0x3B603B60u, (z2 & 0x8FFF8FFFu) ^ 0x3B603B60u);
}
// the 32 FP16 weights of one segment exactly as tqt_dequant_h writes them
__device__ __forceinline__ void segment_h(uint4 v, uint32_t nx, __half s, __half (&o)[32]) {
    const uint32_t E[5] = {v.x, v.y, v.z, v.w, nx};
    const __half2 s2 = __half2half2(s);
#pragma unroll
    for (int p = 0; p < 16; ++p) {
        const uint32_t h = hmul2u(pair(E, p), *reinterpret_cast<const uint32_t *>(&s2));
        o[2 * p] = __ushort_as_half((unsigned short)(h & 0xFFFFu));
        o[2 * p + 1] = __ushort_as_half((unsigned short)(h >> 16));
    }
}
// first word of the next segment of segment i's block (i = global segment index, 4 segments per block)
__device__ __forceinline__ uint32_t next_x(const uint4 * w, int64_t i) {
    return __ldg(reinterpret_cast<const uint32_t *>(w + ((i & ~int64_t(3)) | ((i + 1) & 3))));
}
template <class OUT> __device__ __forceinline__ OUT out_cast(float v) {
    if constexpr (std::is_same_v<OUT, float>) return v;
    else return __float2bfloat16_rn(v);
}
}  // namespace tqtk

// y[n][r] (row stride ldy) = sum_k W[r][k] xh[n][k]; NCOL 8 / 16 column capacity, nc <= NCOL present.
// Grid: any number of CTAs; each loops over the 16 x MT row tiles blockIdx.x, + gridDim.x, ... (persistent when the grid
// is smaller than the tile count). L1: the same code as a second instance that prefers L1 over shared memory (prefer_l1,
// 5-8 columns); the carveout is per function, so 1-column decode keeps the driver default on the plain instance.
template <int NCOL, int MT, int PF, class OUT, bool L1 = false>
__global__ void __launch_bounds__(256, 2) tqt_matvec_k(const uint4 * __restrict__ w, const __half * __restrict__ sc,
                                                         const __half * __restrict__ xh, OUT * __restrict__ y, int rows,
                                                         int k, int nc, int ldy) {
    constexpr int NACC = NCOL > 8 ? 8 : 4;
    __shared__ float red[8][32][MT * NACC];

    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int nb = k / 32, ng = k / 128;
    const int ntiles = (rows + 16 * MT - 1) / (16 * MT);
    // nx: the first word of the next segment of the block (another lane's load: an L1 hit). A quad shuffle instead
    // costs 5-10 % here: in this library's relocatable-device-code build the shuffle's divergent-warp fallback is a call
    // and the kernel needs 24 more registers.
    struct Q { uint4 q[MT][2]; float s[MT][2]; uint32_t nx[MT][2]; };
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
                    q.nx[m][h] = __ldg(reinterpret_cast<const uint32_t *>(w + r * nb + ((b & ~3) | ((b + 1) & 3))));
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
                const uint32_t Ea[5] = {qa.x, qa.y, qa.z, qa.w, cur.nx[m][0]};
                const uint32_t Eb[5] = {qb.x, qb.y, qb.z, qb.w, cur.nx[m][1]};
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const uint32_t a01 = tqtk::pair(Ea, 2 * u), a23 = tqtk::pair(Ea, 2 * u + 1);
                    const uint32_t b01 = tqtk::pair(Eb, 2 * u), b23 = tqtk::pair(Eb, 2 * u + 1);
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
                if (n < nc && r < rows) y[int64_t(n) * ldy + r] = tqtk::out_cast<OUT>(v2);
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
// Kernel choice per shape (3090 measurements in tqt_pick's comments; same numbers in every choice).
struct TqtPick { int mt, pf, persist; };
TqtPick tqt_pick(int rows, int k, int nc, int variant) {
    if (variant > 0) {  // tests / benches: variant - 1 = mt2 * 4 + pf * 2 + persist
        const int v = variant - 1;
        return {(v >> 2) & 1 ? 2 : 1, (v >> 1) & 1, v & 1};
    }
    // 3090 (test-tqt-format --bench, us; TQ4 in brackets): 1-8 columns one 16-row tile per CTA, no prefetch, one CTA
    // per tile: 34816 x 5120 113.4 / 133.4 at 1 / 8 columns (109.3 / 130.0), 5120 x 17408 62.0 / 72.0 (60.1 / 63.7),
    // 16384 x 5120 59.0 / 65.4 (56.6 / 59.9), 5120 x 6144 24.2 / 26.9 (24.1 / 26.5); persistent / prefetch variants
    // are 0-12 % slower. 9-16 columns: two tiles per CTA (180.5 / 92.2 / 92.4 / 36.0 at 16 columns vs 205.7 / 96.2 /
    // 103.9 / 39.1 for TQ4; one tile 205.8 / 113.6 / 100.9 / 41.9).
    (void)rows, (void)k;
    return nc <= 8 ? TqtPick{1, 0, 0} : TqtPick{2, 0, 0};
}

template <int NCOL, int MT, int PF, class OUT>
void tqt_go(const uint4 * w, const __half * sc, const __half * xh, OUT * y, int rows, int k, int nc, int ldy, bool persist,
              cudaStream_t s) {
    auto kern = tqt_matvec_k<NCOL, MT, PF, OUT>;
    // 5-8 columns: the activations every 16-row block re-reads (80-280 KB) stay in L1 (16 columns: no gain measured)
    const bool l1 = NCOL <= 8 && nc > 4;
    if (l1) prefer_l1(reinterpret_cast<const void *>(tqt_matvec_k<NCOL, MT, PF, OUT, true>), 256);
    static const int cap = [&] {
        int per_sm = 0, nsm = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, 256, 0);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        return std::max(1, per_sm * nsm);
    }();
    const int ntiles = (rows + 16 * MT - 1) / (16 * MT);
    const int grid = persist ? std::min(ntiles, cap) : ntiles;
    if (l1) tqt_matvec_k<NCOL, MT, PF, OUT, true><<<grid, 256, 0, s>>>(w, sc, xh, y, rows, k, nc, ldy);
    else kern<<<grid, 256, 0, s>>>(w, sc, xh, y, rows, k, nc, ldy);
}

template <class OUT>
void tqt_launch(const uint8_t * w, const __half * sc, const __half * xh, OUT * y, int rows, int k, int nc, int ldy, int variant,
                  cudaStream_t s) {
    if (k % 128) throw std::invalid_argument("tqt_matvec: k must be a multiple of 128");
    if (nc < 1 || nc > 16) throw std::invalid_argument("tqt_matvec: 1..16 columns");
    if (ldy == 0) ldy = rows;
    const TqtPick p = tqt_pick(rows, k, nc, variant);
    const auto * w4 = reinterpret_cast<const uint4 *>(w);
    auto by = [&](auto ncol_c) {
        constexpr int NC = decltype(ncol_c)::value;
        if (p.mt == 2) {
            if (p.pf) tqt_go<NC, 2, 1>(w4, sc, xh, y, rows, k, nc, ldy, p.persist, s);
            else tqt_go<NC, 2, 0>(w4, sc, xh, y, rows, k, nc, ldy, p.persist, s);
        } else {
            if (p.pf) tqt_go<NC, 1, 1>(w4, sc, xh, y, rows, k, nc, ldy, p.persist, s);
            else tqt_go<NC, 1, 0>(w4, sc, xh, y, rows, k, nc, ldy, p.persist, s);
        }
    };
    if (nc <= 8) by(std::integral_constant<int, 8>{});
    else by(std::integral_constant<int, 16>{});
}

// FP16 [rows][k]: one thread per 32-weight segment, four 16-byte stores
__global__ void tqt_dequant_h_k(const uint4 * w, const __half * sc, __half * out, int64_t nseg) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nseg) return;
    __align__(16) __half v[32];
    tqtk::segment_h(__ldg(w + i), tqtk::next_x(w, i), sc[i / 4], v);
#pragma unroll
    for (int j = 0; j < 4; ++j) reinterpret_cast<uint4 *>(out)[i * 4 + j] = reinterpret_cast<const uint4 *>(v)[j];
}

// int8 [rows][k] + row scales: one block per row, the row scale amax/127 of the FP16 row tqt_dequant_h writes (1 for a
// zero row), then the values int8_quant_rows makes of it (src/core/int8_gemm.cu): the dequant_h + int8_quant_rows
// operand without the FP16 round trip. cm (optional): FP16 values times cm[column] first.
template <bool CM>
__global__ void __launch_bounds__(128) tqt_dequant_i8_k(const uint4 * w, const __half * sc, int k, int8_t * out, float * scale,
                                                          const float * cm) {
    const int64_t r = blockIdx.x, s0 = r * (k / 32);
    const int nseg = k / 32;
    auto seg = [&](int j, float (&f)[32]) {
        __half v[32];
        tqtk::segment_h(__ldg(w + s0 + j), tqtk::next_x(w, s0 + j), sc[(s0 + j) / 4], v);
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
// block per step, FP16 MMA accumulation inside the block, FP32 across blocks) with the TQT segment decode in its store
// step (the values tqt_dequant_h writes).
namespace tqtg {
constexpr int BM = 128, BN = 64, BK = 128, LDS = BK + 8;
constexpr size_t kSmem = size_t(BM + BN) * LDS * sizeof(__half);
__device__ __forceinline__ void mma16816(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
}  // namespace tqtg

__global__ void __launch_bounds__(256) tqt_gemm_h_k(const uint4 * w, const __half * sc, const __half * xh, __nv_bfloat16 * y,
                                                      int rows, int k, int n, int ldy) {
    using namespace tqtg;
    extern __shared__ __align__(16) __half smem[];
    __half * Ws = smem;               // [BM][LDS]
    __half * Xs = smem + BM * LDS;    // [BN][LDS]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, nb128 = k / BK, nb = k / 32;
    const int wm = (warp & 3) * 32, wn = (warp >> 2) * 32;
    uint4 wraw[2], xraw[4];
    uint32_t wnx[2];
    __half wsc[2];
    auto load = [&](int kb) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int u = tid + 256 * j, r = u >> 2, seg = u & 3, row = m0 + r;
            wsc[j] = row < rows ? sc[size_t(row) * nb128 + kb] : __float2half(0.f);
            wraw[j] = row < rows ? __ldg(w + size_t(row) * nb + size_t(kb) * 4 + seg) : make_uint4(0, 0, 0, 0);
            wnx[j] = row < rows ? tqtk::next_x(w, int64_t(row) * nb + int64_t(kb) * 4 + seg) : 0u;
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
            tqtk::segment_h(wraw[j], wnx[j], wsc[j], v);
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

void tqt_matvec_bf16(const uint8_t * w, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols,
                       int ldy, cudaStream_t s) {
    tqt_launch(w, sc, xh, y, rows, k, ncols, ldy, 0, s);
}
void tqt_matvec(const uint8_t * w, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols, cudaStream_t s) {
    tqt_launch(w, sc, xh, y, rows, k, ncols, rows, 0, s);
}
void tqt_matvec_variant(const uint8_t * w, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols, int variant,
                          cudaStream_t s) {
    tqt_launch(w, sc, xh, y, rows, k, ncols, rows, variant, s);
}
void tqt_dequant_h(const uint8_t * w, const __half * sc, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t n = int64_t(rows) * (k / 32);
    if (n > 0) tqt_dequant_h_k<<<unsigned((n + 255) / 256), 256, 0, s>>>(reinterpret_cast<const uint4 *>(w), sc, out, n);
}
void tqt_gemm_h(const uint8_t * w, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int n, int ldy,
                  cudaStream_t s) {
    if (k % 128 || n < 1 || rows < 1) throw std::invalid_argument("tqt_gemm_h: k % 128 == 0, n >= 1");
    static const bool attr = [] {
        cudaFuncSetAttribute(tqt_gemm_h_k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(tqtg::kSmem));
        return true;
    }();
    (void)attr;
    const dim3 grid(unsigned((rows + tqtg::BM - 1) / tqtg::BM), unsigned((n + tqtg::BN - 1) / tqtg::BN));
    tqt_gemm_h_k<<<grid, 256, tqtg::kSmem, s>>>(reinterpret_cast<const uint4 *>(w), sc, xh, y, rows, k, n, ldy);
}
void tqt_dequant_i8(const uint8_t * w, const __half * sc, int8_t * out, float * scale, int rows, int k, cudaStream_t s,
                      const float * cm) {
    if (rows < 1) return;
    const auto * w4 = reinterpret_cast<const uint4 *>(w);
    if (cm) tqt_dequant_i8_k<true><<<rows, 128, 0, s>>>(w4, sc, k, out, scale, cm);
    else tqt_dequant_i8_k<false><<<rows, 128, 0, s>>>(w4, sc, k, out, scale, nullptr);
}

}  // namespace tqllm_cuda
