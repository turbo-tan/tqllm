// u4h-family matvec / GEMM / int8-activation GEMM kernels and the IQ grids. Included by quant.cu inside its anonymous namespace (kernels only; no includes here).
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
    // grids up to 8 KB are staged in shared memory once per CTA; IQ1_S's 16 KB grid is read through L1 instead (staging
    // it cost each 16-row CTA as many bytes as its weights: MiniMax IQ1_S experts ran at ~150 GB/s on a GB10). Same values.
    constexpr bool GRID_SMEM = ud_grid_words(FMT) <= 2048;
    __shared__ uint32_t grid_sm[GRID_SMEM ? ud_grid_words(FMT) : 1];
    const uint32_t * grid = grid_sm;
    if constexpr (FMT == 4) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) grid_sm[i] = iq3s_grid_d[i];
        __syncthreads();
    }
    if constexpr (FMT == 5) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) grid_sm[i] = iq3xxs_grid_d[i];
        __syncthreads();
    }
    if constexpr (FMT >= 8 && FMT <= 11) {
        const uint32_t * src = ud_grid_src<FMT>();
        if constexpr (GRID_SMEM) {
            for (int i = threadIdx.x; i < ud_grid_words(FMT); i += blockDim.x) grid_sm[i] = src[i];
            __syncthreads();
        } else {
            grid = src;
        }
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
