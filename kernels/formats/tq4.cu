// TQ4 weight format kernels (see src/formats/tq4.cpp).
#include "../../src/debug.hpp"
#include "common.cuh"
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"
#include "quant_kernels.cuh"
#include "matvec_rules.hpp"

#include <cstdint>
#include <utility>
#include <cstdio>
#include <map>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace tqllm_cuda {
using namespace tqllm_dev;

// ---- TQ4: rotated 4-bit (per 128: Turbo4 s1 signs, Hadamard / sqrt(128)) on the 16-level code
// (negative << 3) | rank, one FP16 scale per 128; w_rot = L[code] * s (L = int8 Turbo4 levels). Codes in natural nibble
// order (low nibble = even column), so a 16-bit field holds 4 consecutive weights = one PRMT decode.
namespace tq4k {
__device__ __constant__ uint32_t S1[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};
__device__ __forceinline__ uint32_t expand(uint32_t c) {  // 4 nibbles -> 4 int8 levels +-{6,18,31,44,58,75,96,127}
    const uint32_t pm = __byte_perm(0x2C1F1206u, 0x7F604B3Au, c & 0x7777u);
    uint32_t sg = (c >> 3) & 0x1111u;
    sg = (sg & 0x0011u) | ((sg & 0x1100u) << 8);
    sg = (sg & 0x00010001u) | ((sg & 0x00100010u) << 4);
    return __vadd4(pm ^ (sg * 0xFFu), sg);
}
// TQ3 (3-bit, same rotation): codes (negative << 2) | rank on the 8-level code, int8 levels +-{14,45,79,127}.
// Nibble-sliced planes per 32 weights of a row, three uint32: A0 = rank of code j in bits 4j..4j+1 and of code j + 8 in
// bits 4j+2..4j+3 (j = 0..7), A1 the same for codes 16..31, C = sign of code j at bit 4j, j + 8 at 4j+1, j + 16 at 4j+2,
// j + 24 at 4j+3. nibbles3() rebuilds TQ4-layout nibbles (rank in bits 0-1, sign in bit 3) with masks and shifts only,
// so the TQ4 MMA path is reused unchanged.
__device__ __forceinline__ uint4 nibbles3(uint32_t a0, uint32_t a1, uint32_t c) {
    constexpr uint32_t R = 0x33333333u, S = 0x88888888u;
    return make_uint4((a0 & R) | ((c << 3) & S), ((a0 >> 2) & R) | ((c << 2) & S), (a1 & R) | ((c << 1) & S),
                      ((a1 >> 2) & R) | (c & S));
}
// IQ4_NL codebook (llama.cpp kvalues_iq4nl, MIT - reused only to read IQ4_XS files bit-exactly; see NOTICE) as
// level + 128 bytes: 4 nibbles -> 4 offset bytes in two PRMT lookups and one PRMT select on each nibble's bit 3.
constexpr uint32_t IQ4_L0 = 0x3F2D1801u, IQ4_L1 = 0x766A5D4Fu, IQ4_L2 = 0xA6998D81u, IQ4_L3 = 0xF1D9C5B5u;
__device__ __forceinline__ uint32_t iq4_levels(uint32_t c) {
    const uint32_t s = c & 0x7777u;
    const uint32_t lo = __byte_perm(IQ4_L0, IQ4_L1, s), hi = __byte_perm(IQ4_L2, IQ4_L3, s);
    return __byte_perm(lo, hi, 0x3210u | ((c >> 1) & 0x4444u));
}
// IQ4_XS per-32 scale d * (ls - 32) from its 8-byte record per 256 weights (d, scales_h, scales_l[4]), read as one uint2.
__device__ __forceinline__ __half iq4_scale_r(uint2 r, int sub) {
    const __half d = __ushort_as_half(uint16_t(r.x & 0xFFFFu));
    const uint32_t sh = r.x >> 16;
    const int ls = int((r.y >> (4 * sub)) & 0xF) | int(((sh >> (2 * sub)) & 3) << 4);  // scales_l nibble sub = bits 4 sub
    return __float2half(__half2float(d) * float(ls - 32));
}
__device__ __forceinline__ __half iq4_scale(const uint8_t * rec, int sub) {
    return iq4_scale_r(__ldg(reinterpret_cast<const uint2 *>(rec)), sub);
}

// IQ3_S grid (llama.cpp iq3s_grid, MIT - reused only to read IQ3_S files bit-exactly; see NOTICE): 512 groups of 4
// magnitudes 1..15. The split-K kernel stages it in shared memory per CTA.
__device__ const uint32_t IQ3S_GRID_G[512] = {
#include "../iq3s_grid.inc"
};
// IQ3_S group of 4: grid magnitudes with 4 sign bits applied, as level + 128 bytes
__device__ __forceinline__ uint32_t iq3_levels(uint32_t mag, uint32_t sg4) {
    const uint32_t m = ((sg4 * 0x00204081u) & 0x01010101u) * 0xFFu;  // sign bit i -> byte i = 0xFF
    return __vsub4(mag ^ m, m) ^ 0x80808080u;
}
// IQ3 record per 256 weights, bytes: d (0-1), qh[8] (2-9, one byte per 32-weight block), scales[4] (10-13), pad
__device__ __forceinline__ __half iq3_scale_r(uint4 r, int sub) {  // d * (1 + 2 * nibble sub)
    const uint64_t hi = (uint64_t(r.w) << 32) | r.z;                   // bytes 8..15
    const int nib = int((hi >> (16 + 8 * (sub / 2) + 4 * (sub & 1))) & 0xFu);
    return __float2half(__half2float(__ushort_as_half(uint16_t(r.x & 0xFFFFu))) * float(1 + 2 * nib));
}
__device__ __forceinline__ uint32_t iq3_qh_r(uint4 r, int sub) {
    const uint64_t lo = (uint64_t(r.y) << 32) | r.x;                   // bytes 0..7
    return sub < 6 ? uint32_t(lo >> (8 * (2 + sub))) & 0xFFu : (r.z >> (8 * (sub - 6))) & 0xFFu;
}

template <int BITS> __device__ __forceinline__ uint32_t expand_b(uint32_t c) {  // expand() with the BITS level table
    if constexpr (BITS == 4) return expand(c);
    const uint32_t pm = __byte_perm(0x7F4F2D0Eu, 0u, c & 0x7777u);
    uint32_t sg = (c >> 3) & 0x1111u;
    sg = (sg & 0x0011u) | ((sg & 0x1100u) << 8);
    sg = (sg & 0x00010001u) | ((sg & 0x00100010u) << 4);
    return __vadd4(pm ^ (sg * 0xFFu), sg);
}
// expand(c) ^ 0x80808080 (levels + 128) in 6 ops: magnitude from a +128 and a -128 byte table, blended by the nibble
// signs replicated per byte (raw PRMT sign mode; __byte_perm masks the selector to 3 bits). Exhaustively identical,
// but slower in the matvec on the 3090 (24.2 -> 24.7 ms per round: 3 PRMT vs 1), so unused; kept for Blackwell trials.
__device__ __forceinline__ uint32_t expand_u(uint32_t c) {
    const uint32_t sel = c & 0x7777u;
    const uint32_t pos = __byte_perm(0xAC9F9286u, 0xFFE0CBBAu, sel), neg = __byte_perm(0x54616E7Au, 0x01203546u, sel);
    uint32_t m;
    asm("prmt.b32 %0, %1, %2, 0xD9C8;" : "=r"(m) : "r"(c << 4), "r"(c));
    return (pos & ~m) | (neg & m);
}
}

// NCOL: column capacity (NT = NCOL / 8 rounded up tiles of 8 columns share every dequantised weight fragment);
// nc <= NCOL: columns actually present.
// TQ4_MINB (compile time): blocks per SM the register budget must allow for the <= 8-column kernels; 3 (79 regs,
// 50% occupancy) was slower in-model on the 3090 (23.8 -> 24.35 ms per round) than 1 (92 regs, 33%)
#ifndef TQ4_MINB
#define TQ4_MINB 1
#endif
// TQ3_MINB: blocks per SM for the TQ3 (block-per-lane) kernels; bounds the register count so latency is hidden
#ifndef TQ3_MINB
#define TQ3_MINB 2
#endif
// PF: 0 = loads then compute per k block; 1 = next block's weights AND activations in flight (doubles registers);
// 2 = next block's weights in flight, activations loaded just in time (they are L2 resident and shared by all rows);
// 3 = two k blocks per iteration with all their weight loads issued first (four 16-byte loads in flight per lane, no
// buffer carried across iterations). PF >= 2 run under a 2 (MT 2) / 3 (MT 1) blocks-per-SM register bound.
template <int NCOL, int MT, class OUT = __nv_bfloat16, int PF = 0, int BITS = 4>
__global__ void __launch_bounds__(256, BITS == 3 ? TQ3_MINB : PF >= 2 ? (MT == 1 ? 3 : 2) : (NCOL <= 8 && MT <= 2 && PF != 1) ? TQ4_MINB : 1) tq4_matvec_k(const uint8_t * w4, const __half * sc, const __half * xh, OUT * y,
                                                    int rows, int k, int nc, int ldy, int nw) {
    // PF: the next k block's weights, scales and activations are loaded while the current one computes. Per row and
    // column the k split, MMA order and 8-warp reduction do not depend on MT or PF: bit-identical results.
    constexpr int NT = (NCOL + 7) / 8;
    __shared__ float red[8][32][4 * MT * NT];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT;
    const int nb = k / 32, ng = k / 128;
    float c[MT][NT][4] = {};
    struct Ld {
        uint4 wa[MT], wb[MT], x[NT][4];
        uint32_t sa[MT], sb[MT];
    };
    auto load_x = [&](int s0, Ld & L) {
        const int s = s0 + t;
        const bool ok = s < nb;
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
            const bool on = ok && 8 * tt + g < nc;
            const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(8 * tt + g) * k + int64_t(s) * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) L.x[tt][i] = on ? __ldg(&xp[i]) : make_uint4(0, 0, 0, 0);
        }
    };
    auto load_w = [&](int s0, Ld & L) {
        const int s = s0 + t;
        const bool ok = s < nb;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int ra = min(row0 + m * 16 + g, rows - 1), rb = min(row0 + m * 16 + g + 8, rows - 1);
            L.wa[m] = L.wb[m] = make_uint4(0, 0, 0, 0);
            L.sa[m] = L.sb[m] = 0;
            if (ok) {
                if constexpr (BITS == 3) {
                    // device layout: per row, planes [A0 x nb][A1 x nb][C x nb] (coalesced across the quad's segments)
                    const uint32_t * pa = reinterpret_cast<const uint32_t *>(w4) + int64_t(ra) * nb * 3 + s;
                    const uint32_t * pb = reinterpret_cast<const uint32_t *>(w4) + int64_t(rb) * nb * 3 + s;
                    L.wa[m] = tq4k::nibbles3(__ldg(pa), __ldg(pa + nb), __ldg(pa + 2 * nb));
                    L.wb[m] = tq4k::nibbles3(__ldg(pb), __ldg(pb + nb), __ldg(pb + 2 * nb));
                } else {
                    L.wa[m] = __ldg(reinterpret_cast<const uint4 *>(w4 + int64_t(ra) * (k / 2) + int64_t(s) * 16));
                    L.wb[m] = __ldg(reinterpret_cast<const uint4 *>(w4 + int64_t(rb) * (k / 2) + int64_t(s) * 16));
                }
                const __half2 sa2 = __half2half2(sc[int64_t(ra) * ng + s / 4]), sb2 = __half2half2(sc[int64_t(rb) * ng + s / 4]);
                L.sa[m] = *reinterpret_cast<const uint32_t *>(&sa2);
                L.sb[m] = *reinterpret_cast<const uint32_t *>(&sb2);
            }
        }
    };
    auto load = [&](int s0, Ld & L) {
        load_x(s0, L);
        load_w(s0, L);
    };
    // (1152 + L) - 1152 = L exactly in FP16, then one rounding by the scale
    auto deq = [&](uint32_t lv, uint32_t sel, uint32_t su) {
        return hmul2u(hadd2u(__byte_perm(lv, 0x64646464u, sel), 0xE480E480u), su);
    };
    auto compute = [&](const Ld & L) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const uint4 wa = L.wa[m], wb = L.wb[m];
            const uint32_t sa = L.sa[m], sb = L.sb[m];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t a0 = h ? wa.z : wa.x, a1 = h ? wa.w : wa.y, b0 = h ? wb.z : wb.x, b1 = h ? wb.w : wb.y;
                const uint32_t fa[4] = {a0 & 0xFFFFu, a0 >> 16, a1 & 0xFFFFu, a1 >> 16};
                const uint32_t fb[4] = {b0 & 0xFFFFu, b0 >> 16, b1 & 0xFFFFu, b1 >> 16};
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const uint32_t la = tq4k::expand_b<BITS>(fa[u]) ^ 0x80808080u, lb = tq4k::expand_b<BITS>(fb[u]) ^ 0x80808080u;
                    const uint32_t d0 = deq(la, 0x5140, sa), d1 = deq(lb, 0x5140, sb), d2 = deq(la, 0x5342, sa),
                                   d3 = deq(lb, 0x5342, sb);
#pragma unroll
                    for (int tt = 0; tt < NT; ++tt) {
                        const uint4 xa = L.x[tt][2 * h], xb = L.x[tt][2 * h + 1];
                        const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                        mma_f16(c[m][tt], d0, d1, d2, d3, xw[2 * u], xw[2 * u + 1]);
                    }
                }
            }
        }
    };
    if constexpr (BITS == 3) {
        // One 128-weight block per lane and step: three aligned 16-byte loads per row (the A0 / A1 / C words of the
        // block's four segments, row plane-major layout), then its four segments through compute(). nw warps split
        // the k blocks (chosen per shape by the launcher so the last step is not mostly idle).
        const int nb4 = k / 128;
        for (int S0 = warp * 4; S0 < nb4; S0 += 4 * nw) {
            const int S = S0 + t;
            const bool ok = S < nb4;
            uint4 pa[MT][3], pb[MT][3];
            uint32_t sa[MT], sb[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const int ra = min(row0 + m * 16 + g, rows - 1), rb = min(row0 + m * 16 + g + 8, rows - 1);
                const uint4 * qa = reinterpret_cast<const uint4 *>(w4) + int64_t(ra) * nb / 4 * 3 + S;
                const uint4 * qb = reinterpret_cast<const uint4 *>(w4) + int64_t(rb) * nb / 4 * 3 + S;
#pragma unroll
                for (int pl = 0; pl < 3; ++pl) {
                    pa[m][pl] = ok ? __ldg(qa + pl * (nb / 4)) : make_uint4(0, 0, 0, 0);
                    pb[m][pl] = ok ? __ldg(qb + pl * (nb / 4)) : make_uint4(0, 0, 0, 0);
                }
                const __half2 sa2 = __half2half2(ok ? sc[int64_t(ra) * ng + S] : __float2half(0.f));
                const __half2 sb2 = __half2half2(ok ? sc[int64_t(rb) * ng + S] : __float2half(0.f));
                sa[m] = *reinterpret_cast<const uint32_t *>(&sa2);
                sb[m] = *reinterpret_cast<const uint32_t *>(&sb2);
            }
            auto word = [](const uint4 & v, int j) { return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w; };
#pragma unroll 1
            for (int j = 0; j < 4; ++j) {  // not unrolled: one segment's activations live at a time (register budget)
                Ld cur;
                const int sg = 4 * S + j;
#pragma unroll
                for (int tt = 0; tt < NT; ++tt) {
                    const bool on = ok && 8 * tt + g < nc;
                    const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(8 * tt + g) * k + int64_t(sg) * 32);
#pragma unroll
                    for (int i = 0; i < 4; ++i) cur.x[tt][i] = on ? __ldg(&xp[i]) : make_uint4(0, 0, 0, 0);
                }
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    cur.wa[m] = tq4k::nibbles3(word(pa[m][0], j), word(pa[m][1], j), word(pa[m][2], j));
                    cur.wb[m] = tq4k::nibbles3(word(pb[m][0], j), word(pb[m][1], j), word(pb[m][2], j));
                    cur.sa[m] = sa[m], cur.sb[m] = sb[m];
                }
                compute(cur);
            }
        }
    } else if constexpr (PF == 2) {
        Ld cur, nxt;
        int s0 = warp * 4;
        if (s0 < nb) load_w(s0, cur);
        for (; s0 < nb; s0 += 32) {
            const bool more = s0 + 32 < nb;
            if (more) load_w(s0 + 32, nxt);
            load_x(s0, cur);
            compute(cur);
            if (more) {
#pragma unroll
                for (int m = 0; m < MT; ++m) cur.wa[m] = nxt.wa[m], cur.wb[m] = nxt.wb[m], cur.sa[m] = nxt.sa[m], cur.sb[m] = nxt.sb[m];
            }
        }
    } else if constexpr (PF == 3) {
        for (int s0 = warp * 4; s0 < nb; s0 += 64) {
            Ld c0, c1;
            load_w(s0, c0);
            const bool two = s0 + 32 < nb;
            if (two) load_w(s0 + 32, c1);
            load_x(s0, c0);
            compute(c0);
            if (two) {
                load_x(s0 + 32, c1);
                compute(c1);
            }
        }
    } else if constexpr (PF == 1) {
        Ld cur, nxt;
        int s0 = warp * 4;
        if (s0 < nb) load(s0, cur);
        for (; s0 < nb; s0 += 32) {
            const bool more = s0 + 32 < nb;
            if (more) load(s0 + 32, nxt);
            compute(cur);
            if (more) cur = nxt;
        }
    } else {
        for (int s0 = warp * 4; s0 < nb; s0 += 32) {
            Ld cur;
            load(s0, cur);
            compute(cur);
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int tt = 0; tt < NT; ++tt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[warp][lane][4 * (m * NT + tt) + i] = c[m][tt][i];
    __syncthreads();
    if (warp < MT) {
        const int m = warp;
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        for (int w2 = 0; w2 < nw; ++w2)
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * (m * NT + tt) + i];
        const int n0 = 8 * tt + 2 * t, n1 = n0 + 1, r = row0 + m * 16 + g;
        auto put = [&](int64_t i, float v2) {
            if constexpr (std::is_same_v<OUT, float>) y[i] = v2;
            else y[i] = __float2bfloat16_rn(v2);
        };
        if (r < rows) {
            if (n0 < nc) put(int64_t(n0) * ldy + r, v[0]);
            if (n1 < nc) put(int64_t(n1) * ldy + r, v[1]);
        }
        if (r + 8 < rows) {
            if (n0 < nc) put(int64_t(n0) * ldy + r + 8, v[2]);
            if (n1 < nc) put(int64_t(n1) * ldy + r + 8, v[3]);
        }
        }
    }
}

// TQ3 (301) matvec in the shape of tq3_matvec_h_k (TQ3_4S): CTA = 8 warps on one 16-row tile, split-K across warps,
// the 4 lanes of a quad take 4 consecutive 32-weight blocks of the same rows (one 128-weight scale group) and each lane
// decodes its two blocks (rows g, g + 8) once. Codes go straight to (level + 128) bytes with one PRMT over an 8-entry
// table indexed by (negative << 2) | rank (nibbles3cb), then the exact FP16 dequant of the generic kernel:
// ((1152 + L) - 1152) * s, so every weight value is the same as tq4_matvec_k<.., 3>'s. k is permuted identically in A
// and B: MMA u takes elements 4u..4u+3 of lane t's block. NCOL <= 8: one accumulator tile; 9..16: two.
namespace tq4k {
__device__ __forceinline__ uint4 nibbles3cb(uint32_t a0, uint32_t a1, uint32_t c) {  // rank bits 0-1, sign bit 2
    constexpr uint32_t R = 0x33333333u, S = 0x44444444u;
    return make_uint4((a0 & R) | ((c << 2) & S), ((a0 >> 2) & R) | ((c << 1) & S), (a1 & R) | (c & S),
                      ((a1 >> 2) & R) | ((c >> 1) & S));
}
constexpr uint32_t TQ3_CB_LO = 0xFFCFAD8Eu;  // levels + 128 for ranks 0..3: 14, 45, 79, 127
constexpr uint32_t TQ3_CB_HI = 0x01315372u;  // negative: -14, -45, -79, -127
}  // namespace tq4k

// PF: the next step's codes and scales are loaded before this step computes (long k: one wave of CTAs, so every warp
// walks many dependent steps; the prefetch keeps two steps of weight loads in flight). MT: 16-row tiles per CTA; each
// activation fragment (L2) serves MT tiles (the 3-bit weights are fewer bytes than the activations they meet). Per row
// and column the k split, MMA order and 8-warp reduction do not depend on MT or PF: same results.
template <int NCOL, class OUT, int PF = 0, int MT = 1, bool PFX = false, int BITS = 3>
__global__ void __launch_bounds__(256) tq3b_matvec_h_k(const uint32_t * w3, const __half * sc,
                                                                      const __half * xh, OUT * y, int rows, int k, int nc,
                                                                      int ldy) {
    constexpr int NACC = NCOL > 8 ? 8 : 4;
    __shared__ float red[8][32][MT * NACC];
    __shared__ uint32_t g3[BITS == 21 ? 512 : 1];  // IQ3: the 512-group magnitude grid
    if constexpr (BITS == 21) {
        for (int i = threadIdx.x; i < 512; i += 256) g3[i] = tq4k::IQ3S_GRID_G[i];
        __syncthreads();
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nb = k / 32, ng = k / 128;
    int rr[MT][2];
    const uint32_t * pw[MT][2];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            rr[m][h] = min(row0 + 16 * m + g + 8 * h, rows - 1);
            pw[m][h] = w3 + int64_t(rr[m][h]) * nb * (BITS == 3 || BITS == 21 ? 3 : 4);
        }
    float c[MT][NACC] = {};
    // words per 32-weight block: TQ3 planes A0 / A1 / C, TQ4 / IQ4N one uint4, IQ3 8 grid-index bytes + 4 sign bytes
    constexpr int NWD = BITS == 3 || BITS == 21 ? 3 : 4;
    struct Q { uint32_t q[MT][2][NWD]; __half s[MT][2]; uint32_t qh[MT][2]; };
    auto loadq = [&](int b, Q & q) {
        const bool ok = b < nb;
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                if constexpr (BITS == 3) {
#pragma unroll
                    for (int pl = 0; pl < 3; ++pl) q.q[m][h][pl] = ok ? __ldg(pw[m][h] + pl * nb + b) : 0u;
                } else if constexpr (BITS == 21) {
#pragma unroll
                    for (int i = 0; i < 3; ++i) q.q[m][h][i] = ok ? __ldg(pw[m][h] + 3 * b + i) : 0u;
                } else {
                    const uint4 v = ok ? __ldg(reinterpret_cast<const uint4 *>(pw[m][h]) + b) : make_uint4(0, 0, 0, 0);
                    q.q[m][h][0] = v.x, q.q[m][h][1] = v.y, q.q[m][h][2] = v.z, q.q[m][h][3] = v.w;
                }
                if constexpr (BITS == 21) {  // IQ3: 16-byte record per 256 weights
                    const uint4 r = ok ? __ldg(reinterpret_cast<const uint4 *>(reinterpret_cast<const uint8_t *>(sc) +
                                                                                (int64_t(rr[m][h]) * (nb / 8) + b / 8) * 16))
                                       : make_uint4(0, 0, 0, 0);
                    q.s[m][h] = tq4k::iq3_scale_r(r, b % 8);
                    q.qh[m][h] = tq4k::iq3_qh_r(r, b % 8);
                } else if constexpr (BITS == 14) {  // IQ4N: 8-byte record per 256 weights
                    const uint8_t * rec = reinterpret_cast<const uint8_t *>(sc) + (int64_t(rr[m][h]) * (nb / 8) + b / 8) * 8;
                    q.s[m][h] = ok ? tq4k::iq4_scale(rec, b % 8) : __float2half(0.f);
                } else if constexpr (BITS == 15) {  // IQ4L: FP16 scale per 32 weights
                    q.s[m][h] = ok ? sc[int64_t(rr[m][h]) * nb + b] : __float2half(0.f);
                } else {
                    q.s[m][h] = ok ? sc[int64_t(rr[m][h]) * ng + b / 4] : __float2half(0.f);
                }
            }
    };
    auto loadx = [&](int b, uint4 * x4, uint4 * x42) {
        const bool ok = b < nb;
#pragma unroll
        for (int i = 0; i < 4; ++i) x4[i] = x42[i] = make_uint4(0, 0, 0, 0);
        if (g < nc && ok) {
            const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g) * k + int64_t(b) * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) x4[i] = __ldg(&xp[i]);
        }
        if constexpr (NCOL > 8) {
            if (g + 8 < nc && ok) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g + 8) * k + int64_t(b) * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) x42[i] = __ldg(&xp[i]);
            }
        }
    };
    auto step = [&](const Q & cur, const uint4 * x4, const uint4 * x42) {
        const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
        const uint32_t * xw2 = reinterpret_cast<const uint32_t *>(x42);
        auto deq = [](uint32_t lv, uint32_t sel, uint32_t su) {
            return hmul2u(hadd2u(__byte_perm(lv, 0x64646464u, sel), 0xE480E480u), su);
        };
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            uint4 na = make_uint4(0, 0, 0, 0), nbk = make_uint4(0, 0, 0, 0);
            if constexpr (BITS == 3) {
                na = tq4k::nibbles3cb(cur.q[m][0][0], cur.q[m][0][1], cur.q[m][0][2]);
                nbk = tq4k::nibbles3cb(cur.q[m][1][0], cur.q[m][1][1], cur.q[m][1][2]);
            } else if constexpr (BITS != 21) {
                na = make_uint4(cur.q[m][0][0], cur.q[m][0][1], cur.q[m][0][2], cur.q[m][0][3]);
                nbk = make_uint4(cur.q[m][1][0], cur.q[m][1][1], cur.q[m][1][2], cur.q[m][1][3]);
            }
            const __half2 sa2 = __half2half2(cur.s[m][0]), sb2 = __half2half2(cur.s[m][1]);
            const uint32_t sa = *reinterpret_cast<const uint32_t *>(&sa2), sb = *reinterpret_cast<const uint32_t *>(&sb2);
            const uint32_t sela[4] = {na.x, na.y, na.z, na.w}, selb[4] = {nbk.x, nbk.y, nbk.z, nbk.w};
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const uint32_t fa = (u & 1) ? sela[u / 2] >> 16 : sela[u / 2], fb = (u & 1) ? selb[u / 2] >> 16 : selb[u / 2];
                uint32_t ua, ub;
                if constexpr (BITS == 21) {  // group u: 9-bit grid index (byte u of words 0-1, bit u of qh), 4 sign bits
                    const auto lvl = [&](int h) {
                        const uint32_t lo = ((u < 4 ? cur.q[m][h][0] : cur.q[m][h][1]) >> (8 * (u & 3))) & 0xFFu;
                        const uint32_t idx = lo | (((cur.qh[m][h] >> u) & 1u) << 8);
                        return tq4k::iq3_levels(g3[idx], (cur.q[m][h][2] >> (4 * u)) & 0xFu);
                    };
                    ua = lvl(0);
                    ub = lvl(1);
                } else if constexpr (BITS == 3) {
                    ua = __byte_perm(tq4k::TQ3_CB_LO, tq4k::TQ3_CB_HI, fa & 0x7777u);
                    ub = __byte_perm(tq4k::TQ3_CB_LO, tq4k::TQ3_CB_HI, fb & 0x7777u);
                } else if constexpr (BITS == 14 || BITS == 15) {
                    ua = tq4k::iq4_levels(fa);
                    ub = tq4k::iq4_levels(fb);
                } else {
                    ua = tq4k::expand(fa & 0xFFFFu) ^ 0x80808080u;
                    ub = tq4k::expand(fb & 0xFFFFu) ^ 0x80808080u;
                }
                const uint32_t d0 = deq(ua, 0x5140, sa), d1 = deq(ub, 0x5140, sb), d2 = deq(ua, 0x5342, sa),
                               d3 = deq(ub, 0x5342, sb);
                mma_f16(c[m], d0, d1, d2, d3, xw[2 * u], xw[2 * u + 1]);
                if constexpr (NCOL > 8) mma_f16(c[m] + 4, d0, d1, d2, d3, xw2[2 * u], xw2[2 * u + 1]);
            }
        }
        };
    if constexpr (PF == 2) {
        for (int b0 = warp * 4; b0 < nb; b0 += 64) {  // two steps: every load of both issued before either computes
            Q q0, q1;
            uint4 xa[4], xa2[4], xb[4], xb2[4];
            loadq(b0 + t, q0);
            loadq(b0 + 32 + t, q1);
            loadx(b0 + t, xa, xa2);
            loadx(b0 + 32 + t, xb, xb2);
            step(q0, xa, xa2);
            if (b0 + 32 < nb) step(q1, xb, xb2);
        }
    } else {
        Q nxt;
        uint4 nx4[4], nx42[4];
        if constexpr (PF == 1) loadq(warp * 4 + t, nxt);
        if constexpr (PFX) loadx(warp * 4 + t, nx4, nx42);
        for (int b0 = warp * 4; b0 < nb; b0 += 32) {
            const int b = b0 + t;
            Q cur;
            if constexpr (PF == 1) {
                cur = nxt;
                if (b0 + 32 < nb) loadq(b + 32, nxt);
            } else {
                loadq(b, cur);
            }
            uint4 x4[4], x42[4];
            if constexpr (PFX) {
#pragma unroll
                for (int i = 0; i < 4; ++i) x4[i] = nx4[i], x42[i] = nx42[i];
                if (b0 + 32 < nb) loadx(b + 32, nx4, nx42);
            } else {
                loadx(b, x4, x42);
            }
            step(cur, x4, x42);
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
            if (n >= nc || r >= rows) return;
            if constexpr (std::is_same_v<OUT, float>) y[int64_t(n) * ldy + r] = v2;
            else y[int64_t(n) * ldy + r] = __float2bfloat16_rn(v2);
        };
        const int n0 = 2 * t, r = row0 + 16 * m + g;
#pragma unroll
        for (int h = 0; h < NACC / 4; ++h) {
            put(n0 + 8 * h, r, v[4 * h]), put(n0 + 1 + 8 * h, r, v[4 * h + 1]);
            put(n0 + 8 * h, r + 8, v[4 * h + 2]), put(n0 + 1 + 8 * h, r + 8, v[4 * h + 3]);
        }
    }
}

// ---- int8 tensor-core path (prototype): activations quantised per 128-group, expand() levels straight into mma.s8,
// one FP32 FMA per 128-group with (row scale x column scale). Sum order and the int8 activation step differ from the FP16
// path (not byte-identical: KLD-gated).
__global__ void tq4_xquant_k(const __half * xh, int8_t * xq, float * sx, int k, int ncols) {  // one warp per (col, group)
    const int col = blockIdx.x / (k / 128), grp = blockIdx.x % (k / 128), lane = threadIdx.x;
    const __half * src = xh + int64_t(col) * k + grp * 128 + lane * 4;
    const uint2 raw = *reinterpret_cast<const uint2 *>(src);
    const __half2 h0 = *reinterpret_cast<const __half2 *>(&raw.x), h1 = *reinterpret_cast<const __half2 *>(&raw.y);
    const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
    const float v[4] = {f0.x, f0.y, f1.x, f1.y};
    float mx = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float sq = mx / 127.f;
    uint32_t pk = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) pk |= (uint32_t(sq > 0.f ? __float2int_rn(v[j] / sq) : 0) & 0xFFu) << (8 * j);
    *reinterpret_cast<uint32_t *>(xq + int64_t(col) * k + grp * 128 + lane * 4) = pk;
    if (lane == 0) sx[col * (k / 128) + grp] = sq;
    if (ncols == 0) sx[0] = 0.f;  // keeps the parameter used
}

template <int MT, class OUT>
__global__ void __launch_bounds__(256, MT == 1 ? 4 : 2) tq4_matvec_i8_k(const uint8_t * w4, const __half * sc, const int8_t * xq, const float * sx,
                                                          OUT * y, int rows, int k, int nc, int ldy) {
    __shared__ float red[8][32][4 * MT];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nb = k / 32, ng = k / 128;
    float c[MT][4] = {};
    int ra[MT], rb[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) ra[m] = min(row0 + m * 16 + g, rows - 1), rb[m] = min(row0 + m * 16 + g + 8, rows - 1);
    const bool colg = g < nc, col0 = 2 * t < nc, col1 = 2 * t + 1 < nc;
    for (int s0 = warp * 4; s0 < nb; s0 += 32) {
        const int s = s0 + t, grp = s0 / 4;
        const bool ok = s < nb;
        uint4 xa = make_uint4(0, 0, 0, 0), xb = make_uint4(0, 0, 0, 0);
        if (ok && colg) {
            const uint4 * xp = reinterpret_cast<const uint4 *>(xq + int64_t(g) * k + int64_t(s) * 32);
            xa = __ldg(xp), xb = __ldg(xp + 1);
        }
        const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        const float sx0 = ok && col0 ? __ldg(sx + (2 * t) * ng + grp) : 0.f, sx1 = ok && col1 ? __ldg(sx + (2 * t + 1) * ng + grp) : 0.f;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            uint4 wa = make_uint4(0, 0, 0, 0), wb = make_uint4(0, 0, 0, 0);
            float swa = 0.f, swb = 0.f;
            if (ok) {
                wa = __ldg(reinterpret_cast<const uint4 *>(w4 + int64_t(ra[m]) * (k / 2) + int64_t(s) * 16));
                wb = __ldg(reinterpret_cast<const uint4 *>(w4 + int64_t(rb[m]) * (k / 2) + int64_t(s) * 16));
                swa = __half2float(sc[int64_t(ra[m]) * ng + grp]);
                swb = __half2float(sc[int64_t(rb[m]) * ng + grp]);
            }
            const uint32_t fw[4] = {wa.x, wa.y, wa.z, wa.w}, gw[4] = {wb.x, wb.y, wb.z, wb.w};
            int acc[4] = {0, 0, 0, 0};
#pragma unroll
            for (int u = 0; u < 4; ++u) {  // weights 8u..8u+7 of the lane's block: fields 2u (slots 4t..) and 2u+1 (slots 16+4t..)
                const uint32_t a0 = tq4k::expand(fw[u] & 0xFFFFu), a2 = tq4k::expand(fw[u] >> 16);
                const uint32_t a1 = tq4k::expand(gw[u] & 0xFFFFu), a3 = tq4k::expand(gw[u] >> 16);
                mma_s8(acc, a0, a1, a2, a3, xw[2 * u], xw[2 * u + 1]);
            }
            c[m][0] += float(acc[0]) * swa * sx0;
            c[m][1] += float(acc[1]) * swa * sx1;
            c[m][2] += float(acc[2]) * swb * sx0;
            c[m][3] += float(acc[3]) * swb * sx1;
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][lane][4 * m + i] = c[m][i];
    __syncthreads();
    if (warp < MT) {
        const int m = warp;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * m + i];
        const int n0 = 2 * t, n1 = n0 + 1, r = row0 + m * 16 + g;
        auto put = [&](int64_t i, float v2) {
            if constexpr (std::is_same_v<OUT, float>) y[i] = v2;
            else y[i] = __float2bfloat16_rn(v2);
        };
        if (r < rows) {
            if (n0 < nc) put(int64_t(n0) * ldy + r, v[0]);
            if (n1 < nc) put(int64_t(n1) * ldy + r, v[1]);
        }
        if (r + 8 < rows) {
            if (n0 < nc) put(int64_t(n0) * ldy + r + 8, v[2]);
            if (n1 < nc) put(int64_t(n1) * ldy + r + 8, v[3]);
        }
    }
}

void tq4_matvec_i8_bf16(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols,
                        cudaStream_t s, int mt) {
    if (k % 128 || ncols < 1 || ncols > 8) throw std::invalid_argument("tq4_matvec_i8: k % 128 == 0, 1..8 columns");
    static int8_t * xq = nullptr;
    static float * sx = nullptr;
    static size_t cap = 0;
    const size_t need = size_t(8) * k;
    if (need > cap) {  // prototype scratch (grown, never freed)
        if (xq) cudaFree(xq), cudaFree(sx);
        cudaMalloc(&xq, need), cudaMalloc(&sx, need / 128 * 4);
        cap = need;
    }
    tq4_xquant_k<<<ncols * (k / 128), 32, 0, s>>>(xh, xq, sx, k, ncols);
    if (mt == 1) tq4_matvec_i8_k<1, __nv_bfloat16><<<(rows + 15) / 16, 256, 0, s>>>(w4, sc, xq, sx, y, rows, k, ncols, rows);
    else tq4_matvec_i8_k<2, __nv_bfloat16><<<(rows + 31) / 32, 256, 0, s>>>(w4, sc, xq, sx, y, rows, k, ncols, rows);
}

__global__ void tq4_dequant_h_k(const uint8_t * w4, const __half * sc, __half * out, int64_t nfields) {
    const int64_t f = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // 4 weights
    if (f >= nfields) return;
    const uint32_t c = reinterpret_cast<const uint16_t *>(w4)[f];
    const uint32_t lv = tq4k::expand(c);
    const float s = __half2float(sc[f / 32]);
    __align__(8) __half v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __float2half_rn(float(int8_t(lv >> (8 * j))) * s);
    reinterpret_cast<uint2 *>(out)[f] = *reinterpret_cast<const uint2 *>(v);
}

__global__ void tq4b3_dequant_h_k(const uint8_t * w3, const __half * sc, __half * out, int64_t nfields, int k) {
    const int64_t f = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // 4 weights; 8 fields per 32-weight segment
    if (f >= nfields) return;
    const int nb = k / 32;
    const int64_t row = f / (k / 4), seg = (f % (k / 4)) / 8;
    const uint32_t * b = reinterpret_cast<const uint32_t *>(w3) + row * nb * 3 + seg;
    const uint32_t p[3] = {b[0], b[nb], b[2 * nb]};
    const int q = int(f % 8);  // codes 4q .. 4q + 3: nibble field q % 2 of TQ4-layout word q / 2
    const uint4 n = tq4k::nibbles3(p[0], p[1], p[2]);
    const uint32_t word = q / 2 == 0 ? n.x : q / 2 == 1 ? n.y : q / 2 == 2 ? n.z : n.w;
    const uint32_t c = (word >> (16 * (q % 2))) & 0xFFFFu;
    const uint32_t lv = tq4k::expand_b<3>(c);
    const float s = __half2float(sc[f / 32]);
    __align__(8) __half v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __float2half_rn(float(int8_t(lv >> (8 * j))) * s);
    reinterpret_cast<uint2 *>(out)[f] = *reinterpret_cast<const uint2 *>(v);
}

// xh[n] = R x[n] per 128 (x BF16), optionally x = bf16(x * rsqrt(sum(partial) / k + eps) * w) first. Warp per 128.
template <bool NORM>
__global__ void tq4_rotate_h_k(const __nv_bfloat16 * x, const float * w, const float * partial, int nparts, float eps,
                               __half * xh, int k) {
    const int col = blockIdx.y, lane = threadIdx.x % 32, grp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (grp * 128 >= k) return;
    float inv_s = 1.f;
    if constexpr (NORM) {
        float tt = 0.f;
        for (int i = lane; i < nparts; i += 32) tt += partial[col * nparts + i];
        for (int o = 16; o; o >>= 1) tt += __shfl_xor_sync(0xffffffffu, tt, o);
        inv_s = rsqrtf(tt / k + eps);
    }
    const int64_t base = int64_t(col) * k + grp * 128 + lane * 4;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float xv = __bfloat162float(x[base + j]);
        if constexpr (NORM) xv = __bfloat162float(__float2bfloat16_rn(xv * inv_s * w[grp * 128 + lane * 4 + j]));
        v[j] = ((tq4k::S1[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) ? -xv : xv;
    }
    float a = v[0], b = v[1];
    v[0] = a + b, v[1] = a - b;
    a = v[2], b = v[3];
    v[2] = a + b, v[3] = a - b;
    a = v[0], b = v[2];
    v[0] = a + b, v[2] = a - b;
    a = v[1], b = v[3];
    v[1] = a + b, v[3] = a - b;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float other = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? other - v[j] : v[j] + other;
        }
#pragma unroll
    for (int j = 0; j < 4; ++j) xh[base + j] = __float2half_rn(v[j] * 0.08838834764831845f);
}

// Element op fused with the TQ4 128-point input rotation (same transform, signs and scaling as tq4_rotate_h_k):
// OP 1 swiglu (gu [n][2F] -> act), OP 2 attention sigmoid gate (o *= bf16(sigmoid(gate)), o written back),
// OP 3 swiglu on an FP16 gu (the prefill GEMM output, rounded to BF16 first: same values as OP 1 after h_to_bf16).
template <int OP>
__global__ void tq4_rot_op_k(const __nv_bfloat16 * a0, __nv_bfloat16 * o, int ts, int dim, __half * xh, int k) {
    const int col = blockIdx.y, lane = threadIdx.x % 32, grp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (grp * 128 >= k) return;
    const int64_t base = int64_t(col) * k + grp * 128 + lane * 4;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = grp * 128 + lane * 4 + j;
        float xv;
        if constexpr (OP == 1 || OP == 3) {
            float a, u;
            if constexpr (OP == 3) {
                const __half * gh = reinterpret_cast<const __half *>(a0);
                a = __bfloat162float(__float2bfloat16_rn(__half2float(gh[int64_t(col) * 2 * k + e])));
                u = __bfloat162float(__float2bfloat16_rn(__half2float(gh[int64_t(col) * 2 * k + k + e])));
            } else {
                a = __bfloat162float(a0[int64_t(col) * 2 * k + e]), u = __bfloat162float(a0[int64_t(col) * 2 * k + k + e]);
            }
            xv = __bfloat162float(__float2bfloat16_rn(a / (1.f + expf(-a)) * u));
        } else {
            const int h = e / dim, d = e % dim;
            const float g = __bfloat162float(a0[int64_t(col) * ts + int64_t(h) * 2 * dim + dim + d]);
            const float sg = __bfloat162float(__float2bfloat16_rn(1.f / (1.f + expf(-g))));
            const __nv_bfloat16 r = __float2bfloat16_rn(__bfloat162float(o[base + j]) * sg);
            o[base + j] = r;
            xv = __bfloat162float(r);
        }
        v[j] = ((tq4k::S1[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) ? -xv : xv;
    }
    float a = v[0], b = v[1];
    v[0] = a + b, v[1] = a - b;
    a = v[2], b = v[3];
    v[2] = a + b, v[3] = a - b;
    a = v[0], b = v[2];
    v[0] = a + b, v[2] = a - b;
    a = v[1], b = v[3];
    v[1] = a + b, v[3] = a - b;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float other = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? other - v[j] : v[j] + other;
        }
#pragma unroll
    for (int j = 0; j < 4; ++j) xh[base + j] = __float2half_rn(v[j] * 0.08838834764831845f);
}

// Variants (tests / benches; the engine runs 0): 0 MT 2 (default), 1 MT 2 + prefetch, 2 MT 1, 3 MT 1 + prefetch, 4 MT 4,
// 5 MT 4 + prefetch, 6 MT 2 + weights-only prefetch, 7 MT 1 + weights-only prefetch, 8 MT 2 + two blocks per iteration
// (loads hoisted), 9 MT 1 + two blocks per iteration. All bit-identical per column (test tq4_matvec_variants in tests/quant.cu). MT 4 only up to 8 columns (the
// reduction buffer); wider calls fall back to MT 2 with the same prefetch setting.
template <class OUT, int BITS = 4>
static void tq4_launch_v(const uint8_t * w4, const __half * sc, const __half * xh, OUT * y, int rows, int k, int ncols,
                         int v, cudaStream_t s, int ldy = 0) {
    if (ldy == 0) ldy = rows;
    if (k % 128) throw std::invalid_argument("tq4_matvec: k must be a multiple of 128");
    if (ncols < 1 || ncols > 32) throw std::invalid_argument("tq4_matvec: 1..32 columns");
    int mt = v >= 6 ? (v % 2 == 0 ? 2 : 1) : v / 2 == 1 ? 1 : v / 2 == 2 ? 4 : 2;
    const int pf = BITS != 4 ? 0 : v >= 8 ? 3 : v >= 6 ? 2 : v % 2;  // TQ3 has its own block-per-lane load pipeline
    int nw = 8;  // TQ4: 8 warps split the 32-weight segments
    if constexpr (BITS == 3) {  // warps (4..8) that leave the fewest idle lanes on the last 128-weight block step
        const int nb4 = k / 128;
        double best = -1;
        for (int w = 8; w >= 4; --w) {
            const int lanes = 4 * w, steps = (nb4 + lanes - 1) / lanes;
            const double u = double(nb4) / (double(lanes) * steps);
            if (u > best + 1e-9) best = u, nw = w;
        }
    }
    // the 16-row split-K kernel (tq3b_matvec_h_k) for <= 16 columns; prefetch mode 0 / 1 / 2 (see the kernel);
    // variants 20 / 21 / 22 force it with that mode (tests / benches)
    auto launch_new = [&](int pfm) {
        const uint32_t * w = reinterpret_cast<const uint32_t *>(w4);
        const int grid = (rows + 15) / 16;
        auto go3 = [&](auto ncol_c, auto pf_c) {
            constexpr int NC = decltype(ncol_c)::value, P = decltype(pf_c)::value;
            tq3b_matvec_h_k<NC, OUT, P, 1, false, BITS><<<grid, 256, 0, s>>>(w, sc, xh, y, rows, k, ncols, ldy);
        };
        auto by_pf = [&](auto ncol_c) {
            if (pfm == 2) go3(ncol_c, std::integral_constant<int, 2>{});
            else if (pfm == 1) go3(ncol_c, std::integral_constant<int, 1>{});
            else go3(ncol_c, std::integral_constant<int, 0>{});
        };
        if (ncols <= 8) by_pf(std::integral_constant<int, 8>{});
        else by_pf(std::integral_constant<int, 16>{});
    };
    if (v >= 20 && v <= 22 && ncols <= 16) {
        launch_new(v - 20);
        return;
    }
    if (v == 0) {  // default: the rule table (kernels/formats/matvec_rules.hpp)
        const auto pick = mvr::choose(BITS, rows, k, ncols);
        if (pick.kern == mvr::Kern::Split16) {
            launch_new(pick.pf);
            return;
        }
    }
    if constexpr (BITS == 3) {
        if (v == 1) v = 0;  // variant 1: the previous kernel's default configuration
        mt = v >= 6 ? (v % 2 == 0 ? 2 : 1) : v / 2 == 1 ? 1 : v / 2 == 2 ? 4 : 2;
    }
    if (mt == 4 && ncols > 8) mt = 2;
    auto go = [&](auto ncol_c, auto mt_c, auto pf_c) {
        constexpr int NCOL = decltype(ncol_c)::value, MT = decltype(mt_c)::value;
        constexpr int PF = decltype(pf_c)::value;
        tq4_matvec_k<NCOL, MT, OUT, PF, BITS><<<(rows + 16 * MT - 1) / (16 * MT), 32 * nw, 0, s>>>(w4, sc, xh, y, rows, k, ncols,
                                                                                                ldy, nw);
    };
    auto by_mt_pf = [&](auto ncol_c) {
        constexpr int NCOL = decltype(ncol_c)::value;
        auto by_pf = [&](auto mt_c) {
            if (pf == 3) go(ncol_c, mt_c, std::integral_constant<int, 3>{});
            else if (pf == 2) go(ncol_c, mt_c, std::integral_constant<int, 2>{});
            else if (pf == 1) go(ncol_c, mt_c, std::integral_constant<int, 1>{});
            else go(ncol_c, mt_c, std::integral_constant<int, 0>{});
        };
        if (mt == 1) {
            by_pf(std::integral_constant<int, 1>{});
            return;
        }
        if constexpr (NCOL <= 8) {
            if (mt == 4) {
                by_pf(std::integral_constant<int, 4>{});
                return;
            }
        }
        by_pf(std::integral_constant<int, 2>{});
    };
    switch (ncols) {
        case 1: by_mt_pf(std::integral_constant<int, 1>{}); break;
        case 2: by_mt_pf(std::integral_constant<int, 2>{}); break;
        case 3: by_mt_pf(std::integral_constant<int, 3>{}); break;
        case 4: by_mt_pf(std::integral_constant<int, 4>{}); break;
        case 5: by_mt_pf(std::integral_constant<int, 5>{}); break;
        case 6: by_mt_pf(std::integral_constant<int, 6>{}); break;
        case 7: by_mt_pf(std::integral_constant<int, 7>{}); break;
        case 8: by_mt_pf(std::integral_constant<int, 8>{}); break;
        default:
            if (ncols <= 16) by_mt_pf(std::integral_constant<int, 16>{});
            else if (ncols <= 24) by_mt_pf(std::integral_constant<int, 24>{});
            else by_mt_pf(std::integral_constant<int, 32>{});
    }
}

template <class OUT>
static void tq4_launch(const uint8_t * w4, const __half * sc, const __half * xh, OUT * y, int rows, int k, int ncols,
                       cudaStream_t s, int ldy = 0) {
    tq4_launch_v(w4, sc, xh, y, rows, k, ncols, 0, s, ldy);
}

void tq4_matvec_bf16_ld(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                        int ncols, int ldy, cudaStream_t s) {
    tq4_launch(w4, sc, xh, y, rows, k, ncols, s, ldy);
}

void tq4_matvec_variant(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                        int ncols, int v, cudaStream_t s) {
    tq4_launch_v(w4, sc, xh, y, rows, k, ncols, v, s);
}

// IQ4N (engine format 1008): IQ4_XS repacked to natural nibble order (byte j = weights 2j, 2j + 1) + its 8-byte record
// per 256 weights, on the 16-row split-K kernel; <= 16 columns. pf: 0 / 1 / 2 prefetch mode (same results).
template <class OUT>
static void iq4n_launch(const uint8_t * w, const uint8_t * rec, const __half * xh, OUT * y, int rows, int k, int ncols,
                        int pf, cudaStream_t s, int ldy) {
    if (ncols < 1 || ncols > 16) throw std::invalid_argument("iq4n_matvec: 1..16 columns");
    if (k % 256) throw std::invalid_argument("iq4n_matvec: k must be a multiple of 256");
    if (ldy <= 0) ldy = rows;
    const auto * w32 = reinterpret_cast<const uint32_t *>(w);
    const auto * sc = reinterpret_cast<const __half *>(rec);
    const int grid = (rows + 15) / 16;
    auto go = [&](auto ncol_c, auto pf_c) {
        constexpr int NC = decltype(ncol_c)::value, P = decltype(pf_c)::value;
        tq3b_matvec_h_k<NC, OUT, P, 1, false, 14><<<grid, 256, 0, s>>>(w32, sc, xh, y, rows, k, ncols, ldy);
    };
    auto by_pf = [&](auto ncol_c) {
        if (pf == 2) go(ncol_c, std::integral_constant<int, 2>{});
        else if (pf == 1) go(ncol_c, std::integral_constant<int, 1>{});
        else go(ncol_c, std::integral_constant<int, 0>{});
    };
    if (ncols <= 8) by_pf(std::integral_constant<int, 8>{});
    else by_pf(std::integral_constant<int, 16>{});
}
void iq4n_matvec_bf16(const uint8_t * w, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                      int ncols, int pf, cudaStream_t s, int ldy) {
    iq4n_launch(w, rec, xh, y, rows, k, ncols, pf, s, ldy);
}
void iq4n_matvec(const uint8_t * w, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols, int pf,
                 cudaStream_t s) {
    iq4n_launch(w, rec, xh, y, rows, k, ncols, pf, s, rows);
}
// IQ4L (engine format 1009): IQ4_NL in natural nibble order (16 B per 32 weights) + one FP16 scale per 32 weights, on the
// 16-row split-K kernel; <= 16 columns. pf: 0 / 1 / 2 prefetch mode (same results).
template <class OUT>
static void iq4l_launch(const uint8_t * w, const __half * sc, const __half * xh, OUT * y, int rows, int k, int ncols, int pf,
                        cudaStream_t s, int ldy) {
    if (ncols < 1 || ncols > 16) throw std::invalid_argument("iq4l_matvec: 1..16 columns");
    if (k % 128) throw std::invalid_argument("iq4l_matvec: k must be a multiple of 128");
    if (ldy <= 0) ldy = rows;
    const auto * w32 = reinterpret_cast<const uint32_t *>(w);
    const int grid = (rows + 15) / 16;
    auto go = [&](auto ncol_c, auto pf_c) {
        constexpr int NC = decltype(ncol_c)::value, P = decltype(pf_c)::value;
        tq3b_matvec_h_k<NC, OUT, P, 1, false, 15><<<grid, 256, 0, s>>>(w32, sc, xh, y, rows, k, ncols, ldy);
    };
    auto by_pf = [&](auto ncol_c) {
        if (pf == 2) go(ncol_c, std::integral_constant<int, 2>{});
        else if (pf == 1) go(ncol_c, std::integral_constant<int, 1>{});
        else go(ncol_c, std::integral_constant<int, 0>{});
    };
    if (ncols <= 8) by_pf(std::integral_constant<int, 8>{});
    else by_pf(std::integral_constant<int, 16>{});
}
void iq4l_matvec_bf16(const uint8_t * w, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols,
                      int pf, cudaStream_t s, int ldy) {
    iq4l_launch(w, sc, xh, y, rows, k, ncols, pf, s, ldy);
}
void iq4l_matvec(const uint8_t * w, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols, int pf,
                 cudaStream_t s) {
    iq4l_launch(w, sc, xh, y, rows, k, ncols, pf, s, rows);
}
// IQ3 (engine format 1006, IQ3_S: 12 B per 32 weights + 16 B record per 256) on the 16-row split-K kernel, <= 16 columns
template <class OUT>
static void iq3s_launch(const uint8_t * w, const uint8_t * rec, const __half * xh, OUT * y, int rows, int k, int ncols,
                        int pf, cudaStream_t s, int ldy) {
    if (ncols < 1 || ncols > 16) throw std::invalid_argument("iq3_matvec_split: 1..16 columns");
    if (k % 256) throw std::invalid_argument("iq3_matvec_split: k must be a multiple of 256");
    if (ldy <= 0) ldy = rows;
    const auto * w32 = reinterpret_cast<const uint32_t *>(w);
    const auto * sc = reinterpret_cast<const __half *>(rec);
    const int grid = (rows + 15) / 16;
    auto go = [&](auto ncol_c, auto pf_c) {
        constexpr int NC = decltype(ncol_c)::value, P = decltype(pf_c)::value;
        tq3b_matvec_h_k<NC, OUT, P, 1, false, 21><<<grid, 256, 0, s>>>(w32, sc, xh, y, rows, k, ncols, ldy);
    };
    auto by_pf = [&](auto ncol_c) {
        if (pf == 2) go(ncol_c, std::integral_constant<int, 2>{});
        else if (pf == 1) go(ncol_c, std::integral_constant<int, 1>{});
        else go(ncol_c, std::integral_constant<int, 0>{});
    };
    if (ncols <= 8) by_pf(std::integral_constant<int, 8>{});
    else by_pf(std::integral_constant<int, 16>{});
}
void iq3_matvec_split_bf16(const uint8_t * w, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                           int ncols, int pf, cudaStream_t s, int ldy) {
    iq3s_launch(w, rec, xh, y, rows, k, ncols, pf, s, ldy);
}
void iq3_matvec_split(const uint8_t * w, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols,
                      int pf, cudaStream_t s) {
    iq3s_launch(w, rec, xh, y, rows, k, ncols, pf, s, rows);
}

// dequantise IQ4N to FP16 [rows][k] (prefill GEMM operand); one thread per 16-weight group, two 16-byte stores
__global__ void iq4n_dequant_h_k(const uint8_t * w, const uint8_t * rec, __half * out, int64_t ngroups, int k) {
    const int64_t gi = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gi >= ngroups) return;
    const int64_t blk = gi / 2;
    const int h = int(gi % 2), nb = k / 32;
    const int64_t row = blk / nb;
    const int b = int(blk % nb);
    const __half sc = tq4k::iq4_scale(rec + (row * (nb / 8) + b / 8) * 8, b % 8);
    const uint2 q = *reinterpret_cast<const uint2 *>(w + blk * 16 + h * 8);
    const uint32_t qw[2] = {q.x, q.y};
    __align__(16) __half v[16];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const uint32_t lv = tq4k::iq4_levels((u & 1) ? qw[u / 2] >> 16 : qw[u / 2]);
#pragma unroll
        for (int e = 0; e < 4; ++e) v[4 * u + e] = __hmul(__int2half_rn(int((lv >> (8 * e)) & 0xFF) - 128), sc);
    }
    reinterpret_cast<uint4 *>(out)[gi * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[gi * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}
void iq4n_dequant_h(const uint8_t * w, const uint8_t * rec, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t n = int64_t(rows) * (k / 16);
    iq4n_dequant_h_k<<<unsigned((n + 255) / 256), 256, 0, s>>>(w, rec, out, n, k);
}

// dequantise IQ4L to FP16 [rows][k]; one thread per 16-weight group, two 16-byte stores
__global__ void iq4l_dequant_h_k(const uint8_t * w, const __half * sc, __half * out, int64_t ngroups) {
    const int64_t gi = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gi >= ngroups) return;
    const int64_t blk = gi / 2;
    const int h = int(gi % 2);
    const __half d = sc[blk];
    const uint2 q = *reinterpret_cast<const uint2 *>(w + blk * 16 + h * 8);
    const uint32_t qw[2] = {q.x, q.y};
    __align__(16) __half v[16];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const uint32_t lv = tq4k::iq4_levels((u & 1) ? qw[u / 2] >> 16 : qw[u / 2]);
#pragma unroll
        for (int e = 0; e < 4; ++e) v[4 * u + e] = __hmul(__int2half_rn(int((lv >> (8 * e)) & 0xFF) - 128), d);
    }
    reinterpret_cast<uint4 *>(out)[gi * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[gi * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}
void iq4l_dequant_h(const uint8_t * w, const __half * sc, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t n = int64_t(rows) * (k / 16);
    iq4l_dequant_h_k<<<unsigned((n + 255) / 256), 256, 0, s>>>(w, sc, out, n);
}

void tq4_matvec_bf16(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s) {
    tq4_launch(w4, sc, xh, y, rows, k, ncols, s);
}
void tq4_matvec(const uint8_t * w4, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    tq4_launch(w4, sc, xh, y, rows, k, ncols, s);
}
void tq4b3_matvec_bf16_ld(const uint8_t * w3, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                        int ncols, int ldy, cudaStream_t s) {
    tq4_launch_v<__nv_bfloat16, 3>(w3, sc, xh, y, rows, k, ncols, 0, s, ldy);
}
void tq4b3_matvec_variant(const uint8_t * w3, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                          int ncols, int v, cudaStream_t s) {
    tq4_launch_v<__nv_bfloat16, 3>(w3, sc, xh, y, rows, k, ncols, v, s);
}
void tq4b3_matvec(const uint8_t * w3, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    tq4_launch_v<float, 3>(w3, sc, xh, y, rows, k, ncols, 0, s);
}
void tq4b3_dequant_h(const uint8_t * w3, const __half * sc, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t nf = int64_t(rows) * k / 4;
    tq4b3_dequant_h_k<<<unsigned((nf + 255) / 256), 256, 0, s>>>(w3, sc, out, nf, k);
}
void tq4_dequant_h(const uint8_t * w4, const __half * sc, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t nf = int64_t(rows) * k / 4;
    tq4_dequant_h_k<<<unsigned((nf + 255) / 256), 256, 0, s>>>(w4, sc, out, nf);
}
void tq4_rotate_h(const __nv_bfloat16 * x, __half * xh, int k, int ncols, cudaStream_t s) {
    tq4_rotate_h_k<false><<<dim3((k / 128 + 7) / 8, ncols), 256, 0, s>>>(x, nullptr, nullptr, 0, 0.f, xh, k);
}
void tq4_swiglu_rot_h(const __nv_bfloat16 * gu, int n, int F, __half * xh, cudaStream_t s) {
    if (F % 128) throw std::invalid_argument("tq4_swiglu_rot_h: F % 128");
    tq4_rot_op_k<1><<<dim3((F / 128 + 7) / 8, n), 256, 0, s>>>(gu, nullptr, 0, 0, xh, F);
}
void tq4_swiglu_rot_from_h(const __half * gu, int n, int F, __half * xh, cudaStream_t s) {
    if (F % 128) throw std::invalid_argument("tq4_swiglu_rot_from_h: F % 128");
    tq4_rot_op_k<3><<<dim3((F / 128 + 7) / 8, n), 256, 0, s>>>(reinterpret_cast<const __nv_bfloat16 *>(gu), nullptr, 0, 0,
                                                                  xh, F);
}
void tq4_sigmoid_gate_rot_h(__nv_bfloat16 * o, const __nv_bfloat16 * qg0, int ts, int n, int heads, int dim, __half * xh,
                            cudaStream_t s) {
    if ((heads * dim) % 128) throw std::invalid_argument("tq4_sigmoid_gate_rot_h: heads * dim % 128");
    tq4_rot_op_k<2><<<dim3((heads * dim / 128 + 7) / 8, n), 256, 0, s>>>(qg0, o, ts, dim, xh, heads * dim);
}
void tq4_norm_rotate_h(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, __half * xh, int k,
                       int ncols, cudaStream_t s) {
    tq4_rotate_h_k<true><<<dim3((k / 128 + 7) / 8, ncols), 256, 0, s>>>(x, w, partial, (k + 255) / 256, eps, xh, k);
}


// ---- Prefill GEMM on TQ4 / TQ3 weights without an FP16 copy of the matrix ----
// y[c][r] (row stride ldy) = sum_k W[r][k] xh[c][k] for c < n. A CTA owns 128 rows x 64 columns and walks K one quant
// block (128 weights) at a time: the raw codes are decoded with their scale straight into shared memory (W = level * s
// rounded to FP16, the values tq4_dequant_h writes), the activations are staged next to them, and 8 warps run
// mma.m16n8k16 with FP16 accumulation inside the block and FP32 accumulation across blocks. The next block's codes and
// activations are loaded into registers while the current one computes. Replaces dequant + cuBLAS, which writes and
// reads back an FP16 copy of every weight per prefill chunk.
namespace tqg {
constexpr int BM = 128, BN = 64, BK = 128, LDS = BK + 8;  // LDS: padded row (halves) -> conflict-free fragment loads
constexpr size_t kSmem = size_t(BM + BN) * LDS * sizeof(__half);
__device__ __forceinline__ void mma16816(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// 8 int8 levels (two expand() words) times the block scale -> 8 halves
__device__ __forceinline__ uint4 levels_to_h(uint32_t lv0, uint32_t lv1, float s) {
    __align__(16) __half v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = __float2half_rn(float(int8_t(lv0 >> (8 * j))) * s);
        v[4 + j] = __float2half_rn(float(int8_t(lv1 >> (8 * j))) * s);
    }
    return *reinterpret_cast<const uint4 *>(v);
}
}  // namespace tqg

// BITS 4: w = codes [rows][k / 2] (nibble j of a 16-bit field = weight j). BITS 3: w = planes [rows][A0 | A1 | C][k / 32].
template <int BITS>
__global__ void __launch_bounds__(256) tq_gemm_h_k(const uint8_t * w, const __half * sc, const __half * xh, __nv_bfloat16 * y,
                                                   int rows, int k, int n, int ldy) {
    using namespace tqg;
    extern __shared__ __align__(16) __half smem[];
    __half * Ws = smem;               // [BM][LDS]
    __half * Xs = smem + BM * LDS;    // [BN][LDS]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, nb128 = k / BK;
    const int wm = (warp & 3) * 32, wn = (warp >> 2) * 32;
    // per-thread load slots: W - two (row, 32-weight segment) pairs; X - four 8-half pieces
    uint4 wraw[2]; uint32_t w3raw[2][3]; uint4 xraw[4]; float wsc[2];
    auto load = [&](int kb) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int u = tid + 256 * j, r = u >> 2, seg = u & 3, row = m0 + r;
            wsc[j] = row < rows ? __half2float(sc[size_t(row) * nb128 + kb]) : 0.f;
            if constexpr (BITS == 4) {
                wraw[j] = row < rows ? __ldg(reinterpret_cast<const uint4 *>(w + size_t(row) * (k / 2) + size_t(kb) * 64) + seg)
                                     : make_uint4(0, 0, 0, 0);
            } else {
                const int nb = k / 32;
                const uint32_t * p = reinterpret_cast<const uint32_t *>(w) + size_t(row) * nb * 3 + kb * 4 + seg;
#pragma unroll
                for (int pl = 0; pl < 3; ++pl) w3raw[j][pl] = row < rows ? __ldg(p + pl * nb) : 0u;
            }
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
            uint4 words;  // 32 weights as 4 words of 8 nibbles (TQ4 layout)
            if constexpr (BITS == 4) words = wraw[j];
            else words = tq4k::nibbles3(w3raw[j][0], w3raw[j][1], w3raw[j][2]);
            const uint32_t wd[4] = {words.x, words.y, words.z, words.w};
            uint4 * dst = reinterpret_cast<uint4 *>(Ws + r * LDS + seg * 32);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t lo = BITS == 4 ? tq4k::expand(wd[q] & 0xFFFFu) : tq4k::expand_b<3>(wd[q] & 0xFFFFu);
                const uint32_t hi = BITS == 4 ? tq4k::expand(wd[q] >> 16) : tq4k::expand_b<3>(wd[q] >> 16);
                dst[q] = levels_to_h(lo, hi, wsc[j]);
            }
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
        if (kb + 1 < nb128) load(kb + 1);  // overlaps the MMAs below
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
                for (int ni = 0; ni < 4; ++ni) tqg::mma16816(hacc[mi][ni], a[mi], b[ni]);
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
    // C fragment: (row g, cols 2t, 2t + 1) and (row g + 8, same cols)
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

template <int BITS>
static void tq_gemm_h(const uint8_t * w, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int n,
                      int ldy, cudaStream_t s) {
    if (k % 128 || n < 1 || rows < 1) throw std::invalid_argument("tq_gemm_h: k % 128 == 0, n >= 1");
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(tq_gemm_h_k<BITS>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(tqg::kSmem));
        attr = true;
    }
    const dim3 grid(unsigned((rows + tqg::BM - 1) / tqg::BM), unsigned((n + tqg::BN - 1) / tqg::BN));
    tq_gemm_h_k<BITS><<<grid, 256, tqg::kSmem, s>>>(w, sc, xh, y, rows, k, n, ldy);
}
void tq4_gemm_h(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int n, int ldy,
                cudaStream_t s) { tq_gemm_h<4>(w4, sc, xh, y, rows, k, n, ldy, s); }
void tq4b3_gemm_h(const uint8_t * w3, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int n, int ldy,
                  cudaStream_t s) { tq_gemm_h<3>(w3, sc, xh, y, rows, k, n, ldy, s); }
}  // namespace tqllm_cuda
