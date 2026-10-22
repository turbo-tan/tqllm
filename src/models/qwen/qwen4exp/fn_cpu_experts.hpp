#pragma once
// Flash-Next CPU expert-miss pool (x86 AVX2): missed experts computed on the CPU from the pinned packed copy.
#include "fn_checkpoint.hpp"
#if defined(__x86_64__)
inline const int8_t kCpuIq4nl[16] = {
#include "kvalues_iq4nl.inc"
};
__attribute__((target("avx2,fma"))) static float cpu_dot_f32(const float * a, const float * b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(); int k = 0;
    for (; k + 16 <= n; k += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    __m128 q = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    q = _mm_add_ps(q, _mm_movehl_ps(q, q)); q = _mm_add_ss(q, _mm_shuffle_ps(q, q, 1));
    float r = _mm_cvtss_f32(q); for (; k < n; ++k) r += a[k] * b[k]; return r;
}
// multi-column IQ4_NL: decode each block once for up to 4 activation vectors
__attribute__((target("avx2,fma"))) static void cpu_dot_iq4nl_q8_mc(const uint8_t * w, const int8_t * const * xqc, const float * const * xdc, int nc, int nblocks, float * out) {
    const __m256i table = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kCpuIq4nl)));
    const __m256i low = _mm256_set1_epi8(0x0f), ones = _mm256_set1_epi16(1);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int b = 0; b < nblocks; ++b, w += 18) {
        uint16_t dh; std::memcpy(&dh, w, 2); __half_raw hr; hr.x = dh; const float d = __half2float(__half(hr));
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w + 2));
        const __m256i wv = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), low));
        const __m256i wa = _mm256_sign_epi8(wv, wv);
        for (int c = 0; c < nc; ++c) {
            const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xqc[c] + b * 32));
            const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(wa, _mm256_sign_epi8(xv, wv)), ones);
            acc[c] = _mm256_fmadd_ps(_mm256_set1_ps(d * xdc[c][b]), _mm256_cvtepi32_ps(p32), acc[c]);
        }
    }
    for (int c = 0; c < nc; ++c) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc[c]), _mm256_extractf128_ps(acc[c], 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1)); out[c] = _mm_cvtss_f32(s);
    }
}
// x -> fp16-rounded fp32 and int8 per 32-block (vectorised; n % 32 == 0)
__attribute__((target("avx2,fma,f16c"))) static void cpu_prepare_x(const float * x, float * xr, int8_t * q, float * d, int n) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int b = 0; b < n / 32; ++b) {
        __m256 v[4], m = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) {
            v[i] = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_loadu_ps(x + b * 32 + 8 * i), _MM_FROUND_TO_NEAREST_INT));
            _mm256_storeu_ps(xr + b * 32 + 8 * i, v[i]); m = _mm256_max_ps(m, _mm256_and_ps(v[i], absmask));
        }
        __m128 l = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        l = _mm_max_ps(l, _mm_movehl_ps(l, l)); l = _mm_max_ss(l, _mm_shuffle_ps(l, l, 1));
        const float mx = _mm_cvtss_f32(l), sc = mx / 127.f; d[b] = sc;
        const __m256 inv = _mm256_set1_ps(sc > 0.f ? 1.f / sc : 0.f);
        __m256i i32[4]; for (int i = 0; i < 4; ++i) i32[i] = _mm256_cvtps_epi32(_mm256_mul_ps(v[i], inv));
        const __m256i p16a = _mm256_packs_epi32(i32[0], i32[1]), p16b = _mm256_packs_epi32(i32[2], i32[3]);
        const __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p16a, p16b), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(q + b * 32), p8);
    }
}
// x as int8 per 32-block with one scale (CPU numerics)
inline void cpu_quant_q8(const float * x, int n, int8_t * q, float * d) {
    for (int b = 0; b < n / 32; ++b) {
        float m = 0.f; for (int j = 0; j < 32; ++j) m = std::max(m, std::fabs(x[b * 32 + j]));
        const float s = m / 127.f, inv = s > 0.f ? 1.f / s : 0.f; d[b] = s;
        for (int j = 0; j < 32; ++j) q[b * 32 + j] = int8_t(std::lrintf(x[b * 32 + j] * inv));
    }
}
// one IQ4_NL row (18-byte blocks: fp16 d, 16 bytes of nibbles; low nibbles are elements 0-15) against a q8 vector
__attribute__((target("avx2,fma"))) static float cpu_dot_iq4nl_q8(const uint8_t * w, const int8_t * xq, const float * xd, int nblocks) {
    const __m256i table = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kCpuIq4nl)));
    const __m256i low = _mm256_set1_epi8(0x0f), ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nblocks; ++b, w += 18) {
        uint16_t dh; std::memcpy(&dh, w, 2);
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w + 2));
        const __m256i nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), low);
        const __m256i wv = _mm256_shuffle_epi8(table, nib);
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq + b * 32));
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
        const __m256i p32 = _mm256_madd_epi16(p16, ones);
        __half_raw hr; hr.x = dh;
        acc = _mm256_fmadd_ps(_mm256_set1_ps(__half2float(__half(hr)) * xd[b]), _mm256_cvtepi32_ps(p32), acc);
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
inline const uint32_t kCpuIq3xxs[256] = {
#include "iq3xxs_grid.inc"
};
inline const uint32_t kCpuIq3s[512] = {
#include "iq3s_grid.inc"
};
inline const uint64_t kCpuIq2s[1024] = {
#include "iq2s_grid.inc"
};
// 32 sign bits -> 32 bytes of 0xff (negative) / 0x00
__attribute__((target("avx2"))) static inline __m256i cpu_sign_mask(uint32_t bits) {
    const __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32(int(bits)),
        _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0x0000000000000000LL));
    const __m256i bit = _mm256_set1_epi64x(int64_t(0x8040201008040201ULL));
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
}
// unsigned magnitudes (32 bytes) with sign mask against 32 int8 activations -> 8 int32 partial sums (4 bytes each)
__attribute__((target("avx2"))) static inline __m256i cpu_mag_dot(__m256i mag, __m256i neg, const int8_t * xq) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq));
    const __m256i xs = _mm256_sub_epi8(_mm256_xor_si256(x, neg), neg);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(mag, xs), _mm256_set1_epi16(1));
}
__attribute__((target("avx2,fma"))) static inline float cpu_hsum(__m256 acc) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1)); return _mm_cvtss_f32(s);
}
inline float cpu_h2f(const uint8_t * p) { uint16_t v; std::memcpy(&v, p, 2); __half_raw r; r.x = v; return __half2float(__half(r)); }
// IQ3_XXS row (98-byte blocks of 256) against q8 x
__attribute__((target("avx2,fma"))) static void cpu_dot_iq3xxs_q8(const uint8_t * w, const int8_t * const * xqc, const float * const * xdc, int nc, int nblocks, float * out) {
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int b = 0, sb = 0; b < nblocks; ++b, w += 98) {
        const float d = cpu_h2f(w); const uint8_t * qs = w + 2;
        for (int ib = 0; ib < 8; ++ib, ++sb) {
            uint32_t a; std::memcpy(&a, w + 66 + 4 * ib, 4);
            uint32_t sbits = 0;
            for (int l = 0; l < 4; ++l) { const uint32_t i7 = (a >> 7 * l) & 127; sbits |= (i7 | ((__builtin_popcount(i7) & 1) << 7)) << (8 * l); }
            const uint8_t * q = qs + ib * 8;
            const __m256i mag = _mm256_set_epi32(int(kCpuIq3xxs[q[7]]), int(kCpuIq3xxs[q[6]]), int(kCpuIq3xxs[q[5]]), int(kCpuIq3xxs[q[4]]),
                                                 int(kCpuIq3xxs[q[3]]), int(kCpuIq3xxs[q[2]]), int(kCpuIq3xxs[q[1]]), int(kCpuIq3xxs[q[0]]));
            const float db = d * (0.5f + float(a >> 28)) * 0.5f; const __m256i neg = cpu_sign_mask(sbits);
            for (int c = 0; c < nc; ++c)
                acc[c] = _mm256_fmadd_ps(_mm256_set1_ps(db * xdc[c][sb]), _mm256_cvtepi32_ps(cpu_mag_dot(mag, neg, xqc[c] + sb * 32)), acc[c]);
        }
    }
    for (int c = 0; c < nc; ++c) out[c] = cpu_hsum(acc[c]);
}
// IQ3_S row (110-byte blocks)
__attribute__((target("avx2,fma"))) static void cpu_dot_iq3s_q8(const uint8_t * w, const int8_t * const * xqc, const float * const * xdc, int nc, int nblocks, float * out) {
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int b = 0, sb = 0; b < nblocks; ++b, w += 110) {
        const float d = cpu_h2f(w); const uint8_t * qs = w + 2, * qh = w + 66, * signs = w + 74, * sc = w + 106;
        for (int ib = 0; ib < 8; ++ib, ++sb) {
            const uint8_t * q = qs + ib * 8; const int hb = qh[ib];
            int g[8]; for (int k = 0; k < 8; ++k) g[k] = int(kCpuIq3s[q[k] | (((hb >> k) & 1) << 8)]);
            const __m256i mag = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(g));
            uint32_t sbits; std::memcpy(&sbits, signs + ib * 4, 4);
            const float db = d * float(1 + 2 * ((sc[ib / 2] >> 4 * (ib % 2)) & 0xf)); const __m256i neg = cpu_sign_mask(sbits);
            for (int c = 0; c < nc; ++c)
                acc[c] = _mm256_fmadd_ps(_mm256_set1_ps(db * xdc[c][sb]), _mm256_cvtepi32_ps(cpu_mag_dot(mag, neg, xqc[c] + sb * 32)), acc[c]);
        }
    }
    for (int c = 0; c < nc; ++c) out[c] = cpu_hsum(acc[c]);
}
// IQ2_S row (82-byte blocks); two scales per 32 (one per 16 weights)
__attribute__((target("avx2,fma"))) static void cpu_dot_iq2s_q8(const uint8_t * w, const int8_t * const * xqc, const float * const * xdc, int nc, int nblocks, float * out) {
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int b = 0, sb = 0; b < nblocks; ++b, w += 82) {
        const float d = cpu_h2f(w); const uint8_t * qs = w + 2, * signs = w + 34, * qh = w + 66, * sc = w + 74;
        for (int ib = 0; ib < 8; ++ib, ++sb) {
            int64_t g[4]; for (int l = 0; l < 4; ++l) g[l] = int64_t(kCpuIq2s[qs[ib * 4 + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)]);
            const __m256i mag = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(g));
            uint32_t sbits; std::memcpy(&sbits, signs + ib * 4, 4);
            const float d0 = d * (0.5f + float(sc[ib] & 0xf)) * 0.25f, d1 = d * (0.5f + float(sc[ib] >> 4)) * 0.25f;
            const __m256 dd = _mm256_setr_ps(d0, d0, d0, d0, d1, d1, d1, d1); const __m256i neg = cpu_sign_mask(sbits);
            for (int c = 0; c < nc; ++c)
                acc[c] = _mm256_fmadd_ps(_mm256_mul_ps(dd, _mm256_set1_ps(xdc[c][sb])), _mm256_cvtepi32_ps(cpu_mag_dot(mag, neg, xqc[c] + sb * 32)), acc[c]);
        }
    }
    for (int c = 0; c < nc; ++c) out[c] = cpu_hsum(acc[c]);
}
// GSQ Q2_0 row (18-byte blocks of 64: fp16 d, 16 bytes of 2-bit codes, value (q - 1) * d; byte k holds elements 4k..4k+3)
// against q8 x permuted per 32 to [s * 8 + k] = x[4k + s], with xs = per-32 sum of the int8 x
__attribute__((target("avx2,fma"))) static float cpu_dot_q2_0_q8p(const uint8_t * w, const int8_t * xq, const float * xd, const int * xs, int nblocks) {
    __m256 acc = _mm256_setzero_ps(); const uint64_t m = 0x0303030303030303ULL; float tail = 0.f;
    for (int b = 0; b < nblocks; ++b, w += 18) {
        const float d = cpu_h2f(w);
        for (int h = 0; h < 2; ++h, xq += 32, ++xd, ++xs) {
            uint64_t v; std::memcpy(&v, w + 2 + 8 * h, 8);
            const __m256i q = _mm256_set_epi64x(int64_t((v >> 6) & m), int64_t((v >> 4) & m), int64_t((v >> 2) & m), int64_t(v & m));
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq));
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), _mm256_set1_epi16(1));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * xd[0]), _mm256_cvtepi32_ps(p), acc);
            tail -= d * xd[0] * float(xs[0]);
        }
    }
    return cpu_hsum(acc) + tail;
}
inline void cpu_permute_q2(const int8_t * xq, int n, int8_t * out, int * sums) {
    for (int b = 0; b < n / 32; ++b) {
        int sum = 0;
        for (int k = 0; k < 8; ++k) for (int s2 = 0; s2 < 4; ++s2) { const int8_t v = xq[b * 32 + 4 * k + s2]; out[b * 32 + s2 * 8 + k] = v; sum += v; }
        sums[b] = sum;
    }
}
// The CPU pool reads missed experts from the pinned packed copy (device layout, optionally tiled) instead of the GGUF
// mapping, so no file pages are touched: each row is rebuilt into its GGUF block bytes for the int8 kernels.
struct CpuPackedView {
    const char * map = nullptr; std::vector<uint64_t> base, block; std::vector<std::array<uint64_t, 6>> sizes;
    std::vector<std::array<int, 3>> fmt; bool tiled = false; int layers = 0;
};
inline CpuPackedView * g_cpu_packed = nullptr;
inline int64_t cpu_block_off(int r, int s, int nb, int bpb, bool tiled) {
    if (!tiled) return (int64_t(r) * nb + s) * bpb;
    return (((int64_t(r >> 4) * (nb >> 2) + (s >> 2)) * 16 + (r & 15)) * 4 + (s & 3)) * bpb;
}
// GGUF bytes of row r (cols weights) of one device-layout matrix; returns nullptr if the format is not handled
inline const char * cpu_gguf_row(int fmt, const char * dev, const char * aux, int r, int cols, bool tiled, char * out) {
    const int nb = cols / 32, kb = cols / 256;
    auto blk32 = [&](int s, int bpb) { return reinterpret_cast<const uint8_t *>(dev + cpu_block_off(r, s, nb, bpb, tiled)); };
    if (fmt == 5) {  // IQ3_XXS 98 B: d, qs[64], signs/scales[32]
        for (int b = 0; b < kb; ++b) {
            uint8_t * o = reinterpret_cast<uint8_t *>(out) + b * 98; const char * rec = aux + (int64_t(r) * kb + b) * 4;
            std::memcpy(o, rec, 2);
            for (int ib = 0; ib < 8; ++ib) { const uint8_t * d = blk32(b * 8 + ib, 12); std::memcpy(o + 2 + ib * 8, d, 8); std::memcpy(o + 66 + ib * 4, d + 8, 4); }
        }
        return out;
    }
    if (fmt == 4) {  // IQ3_S 110 B: d, qs[64], qh[8], signs[32], scales[4]
        for (int b = 0; b < kb; ++b) {
            uint8_t * o = reinterpret_cast<uint8_t *>(out) + b * 110; const char * rec = aux + (int64_t(r) * kb + b) * 16;
            std::memcpy(o, rec, 2); std::memcpy(o + 66, rec + 2, 8); std::memcpy(o + 106, rec + 10, 4);
            for (int ib = 0; ib < 8; ++ib) { const uint8_t * d = blk32(b * 8 + ib, 12); std::memcpy(o + 2 + ib * 8, d, 8); std::memcpy(o + 74 + ib * 4, d + 8, 4); }
        }
        return out;
    }
    if (fmt == 10) {  // IQ2_S 82 B: d, qs[32], signs[32], qh[8], scales[8]
        for (int b = 0; b < kb; ++b) {
            uint8_t * o = reinterpret_cast<uint8_t *>(out) + b * 82; const char * rec = aux + (int64_t(r) * kb + b) * 20;
            std::memcpy(o, rec, 2); std::memcpy(o + 66, rec + 2, 16);
            for (int ib = 0; ib < 8; ++ib) { const uint8_t * d = blk32(b * 8 + ib, 8); std::memcpy(o + 2 + ib * 4, d, 4); std::memcpy(o + 34 + ib * 4, d + 4, 4); }
        }
        return out;
    }
    if (fmt == 14) {  // IQ4_NL 18 B per 32: d, qs[16] (qs[i] = e[i] | e[i + 16] << 4); device: natural nibble pairs + fp16 d
        const auto * dh = reinterpret_cast<const uint16_t *>(aux) + int64_t(r) * nb;
        for (int s = 0; s < nb; ++s) {
            uint8_t * o = reinterpret_cast<uint8_t *>(out) + s * 18; const uint8_t * d = blk32(s, 16);
            std::memcpy(o, dh + s, 2);
            auto e = [&](int i) { return (d[i / 2] >> (4 * (i & 1))) & 0xF; };
            for (int i = 0; i < 16; ++i) o[2 + i] = uint8_t(e(i) | (e(i + 16) << 4));
        }
        return out;
    }
    if (fmt == 13) {  // IQ4_XS 136 B per 256: d, scales_h, scales_l[4], qs[128] (per 32 as IQ4_NL); device: natural pairs + record
        for (int b = 0; b < kb; ++b) {
            uint8_t * o = reinterpret_cast<uint8_t *>(out) + b * 136;
            std::memcpy(o, aux + (int64_t(r) * kb + b) * 8, 8);
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t * d = blk32(b * 8 + ib, 16);
                auto e = [&](int i) { return (d[i / 2] >> (4 * (i & 1))) & 0xF; };
                for (int i = 0; i < 16; ++i) o[8 + ib * 16 + i] = uint8_t(e(i) | (e(i + 16) << 4));
            }
        }
        return out;
    }
    return nullptr;
}
// TQFN_CPU_MISS=T: experts that are not resident in the GPU cache are computed by T pinned CPU threads while the GPU
// runs the hits (full routing, CPU numerics: not bit-identical to the GPU path). Rows come from the GGUF bytes.
class CpuMissPool {
public:
    struct Job { const char * w[3]; uint32_t type[3]; size_t row_bytes[3]; int cols_t[8]; int ncols = 0; int slot = 0;
                 const char * dev[3] = {nullptr, nullptr, nullptr}, * aux[3] = {nullptr, nullptr, nullptr}; int fmt[3] = {-1, -1, -1}; bool tiled = false; };
    CpuMissPool(int threads, int hidden, int ff) : hidden_(hidden), ff_(ff) {
        for (int i = 0; i < threads; ++i) workers_.emplace_back([this, i] { run(i); });
        for (int i = 0; i < threads; ++i) {
            cpu_set_t set; CPU_ZERO(&set); CPU_SET(1 + i, &set);
            pthread_setaffinity_np(workers_[i].native_handle(), sizeof(set), &set);
        }
        cpu_set_t host; CPU_ZERO(&host); CPU_SET(0, &host); pthread_setaffinity_np(pthread_self(), sizeof(host), &host);
    }
    ~CpuMissPool() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; gen_.fetch_add(1); } cv_.notify_all(); for (auto & w : workers_) w.join();
        if (calls_) std::printf("CPU_MISS pool %.3f ms per layer call busy, host blocked %.3f ms per call (%llu calls), %llu rows read from the GGUF mapping\n",
                                busy_ms_ / calls_, blocked_ms_ / calls_, (unsigned long long) calls_, (unsigned long long) gguf_rows_.load());
        if (calls_) std::printf("CPU_MISS per call: %.2f experts, %.2f columns; gate/up phase %.3f ms wall (%.3f ms CPU summed over workers), down phase %.3f ms wall (%.3f ms CPU)\n",
                                double(jobs_sum_) / calls_, double(cols_sum_) / calls_, gu_wall_ms_ / calls_, cpu_ns_gu_.load() / 1e6 / calls_,
                                dn_wall_ms_ / calls_, cpu_ns_dn_.load() / 1e6 / calls_);
        for (auto & [t, e] : check_gu_) std::printf("CPU_CHECK gate/up type %u int8 vs fp32: max abs diff %.3g, max |ref| %.3g\n", t, e.first, e.second);
        if (std::getenv("TQFN_CPU_CHECK")) std::printf("CPU_CHECK packed-row rebuild mismatches vs GGUF: %llu\n", (unsigned long long) row_mismatch_);
        if (check_mag_ > 0) std::printf("CPU_CHECK down int8 vs fp32: max abs diff %.3g, max |ref| %.3g\n", check_err_, check_mag_);
    }
    // x: n tokens x hidden (fp32, rounded to fp16 like the GPU input); out: one bf16 row of `hidden` per job column.
    void start(const float * x, int n, std::vector<Job> jobs, __nv_bfloat16 * out) {
        x_.assign(size_t(n) * hidden_, 0.f);
        xq_.resize(x_.size()); xd_.resize(x_.size() / 32);
        cpu_prepare_x(x, x_.data(), xq_.data(), xd_.data(), int(x_.size()));
        jobs_ = std::move(jobs); out_ = out; int cols = 0;
        for (auto & j : jobs_) { j.slot = cols; cols += j.ncols; }
        jobs_sum_ += jobs_.size(); cols_sum_ += uint64_t(cols);
        act_.assign(size_t(cols) * ff_, 0.f);
        // about two items per worker in each phase, whatever the number of experts (load balance)
        const int k = std::max<int>(1, int(jobs_.size())), tw = 2 * int(workers_.size());
        chunks_[0] = std::clamp((tw + k - 1) / k, 4, 64); chunks_[1] = std::clamp((tw + k - 1) / k, 4, 128);
        phase_items_[0] = int(jobs_.size()) * chunks_[0]; phase_items_[1] = int(jobs_.size()) * chunks_[1];
        next_.store(0); done_.store(0); phase_.store(0); pending_ = !jobs_.empty();
        if (pending_) { t_start_ = std::chrono::steady_clock::now(); { std::lock_guard<std::mutex> l(m_); gen_.fetch_add(1, std::memory_order_release); } if (sleepers_.load()) cv_.notify_all(); }
    }
    void wait() {
        if (!pending_) return;
        const auto t0 = std::chrono::steady_clock::now();
        while (done_.load(std::memory_order_acquire) < phase_items_[0] + phase_items_[1]) _mm_pause();
        const auto t1 = std::chrono::steady_clock::now();
        busy_ms_ += std::chrono::duration<double, std::milli>(t_last_.load() > t_start_ ? t_last_.load() - t_start_ : t1 - t_start_).count();
        blocked_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count(); ++calls_;
        gu_wall_ms_ += std::chrono::duration<double, std::milli>(t_gu_.load() - t_start_).count();
        dn_wall_ms_ += std::chrono::duration<double, std::milli>(t_last_.load() - t_gu_.load()).count();
        pending_ = false;
    }
private:
    int chunks_[2] = {20, 32};
    // the GGUF bytes of row r of matrix m: rebuilt from the pinned packed copy when available, else the GGUF mapping
    const char * row_src(const Job & j, int m, int r, int cols) {
        const char * gguf = j.w[m] + size_t(r) * j.row_bytes[m];
        if (!j.dev[m]) {
            if (gguf_rows_.fetch_add(1, std::memory_order_relaxed) == 0 && std::getenv("TQLLM_DEBUG"))
                std::printf("CPU_MISS first GGUF-mapping row: matrix %d has no packed copy (gguf type %u)\n", m, j.type[m]);
            return gguf;
        }
        if (j.type[m] == GGML_GSQ_Q2_0) return j.dev[m] + size_t(r) * j.row_bytes[m];  // device layout == GGUF rows
        thread_local std::vector<char> rb[3]; rb[m].resize(j.row_bytes[m]);
        const char * q = cpu_gguf_row(j.fmt[m], j.dev[m], j.aux[m], r, cols, j.tiled, rb[m].data());
        if (!q) {
            if (gguf_rows_.fetch_add(1, std::memory_order_relaxed) == 0 && std::getenv("TQLLM_DEBUG"))
                std::printf("CPU_MISS first GGUF-mapping row: matrix %d fmt %d gguf type %u\n", m, j.fmt[m], j.type[m]);
            return gguf;
        }
        static const bool check = std::getenv("TQFN_CPU_CHECK") != nullptr;
        if (check && std::memcmp(q, gguf, j.row_bytes[m]) != 0) { std::lock_guard<std::mutex> l(m_); ++row_mismatch_; }
        return q;
    }
    void work(int item) {
        const bool down = item >= phase_items_[0];
        const int it = down ? item - phase_items_[0] : item, chunks = down ? chunks_[1] : chunks_[0];
        const Job & j = jobs_[it / chunks]; const int part = it % chunks;
        thread_local std::vector<float> row; row.resize(down ? ff_ : hidden_);
        if (!down) {
            const int r0 = part * ff_ / chunks, r1 = (part + 1) * ff_ / chunks;
            thread_local std::vector<float> g; g.assign(size_t(j.ncols) * (r1 - r0), 0.f);
            using DotQ8 = void (*)(const uint8_t *, const int8_t * const *, const float * const *, int, int, float *);
            const int8_t * xqc[8]; const float * xdc[8];
            for (int c = 0; c < j.ncols; ++c) { xqc[c] = xq_.data() + size_t(j.cols_t[c]) * hidden_; xdc[c] = xd_.data() + size_t(j.cols_t[c]) * hidden_ / 32; }
            for (int m = 0; m < 2; ++m) {
                const DotQ8 fast = hidden_ % 256 ? nullptr : j.type[m] == 18 ? cpu_dot_iq3xxs_q8 : j.type[m] == 21 ? cpu_dot_iq3s_q8 : j.type[m] == 22 ? cpu_dot_iq2s_q8 : nullptr;
                for (int r = r0; r < r1; ++r) {
                    const char * wr = row_src(j, m, r, hidden_);
                    float fo[8];
                    if (fast) for (int c0 = 0; c0 < j.ncols; c0 += 4) fast(reinterpret_cast<const uint8_t *>(wr), xqc + c0, xdc + c0, std::min(4, j.ncols - c0), hidden_ / 256, fo + c0);
                    else dequantize_host(j.type[m], wr, row.data(), hidden_);
                    for (int c = 0; c < j.ncols; ++c) {
                        const int t = j.cols_t[c];
                        const float acc = fast ? fo[c] : cpu_dot_f32(row.data(), x_.data() + size_t(t) * hidden_, hidden_);
                        static const bool check = std::getenv("TQFN_CPU_CHECK") != nullptr;
                        if (check && fast) {
                            std::vector<float> ref(hidden_); dequantize_host(j.type[m], wr, ref.data(), hidden_);
                            const float want = cpu_dot_f32(ref.data(), x_.data() + size_t(t) * hidden_, hidden_);
                            std::lock_guard<std::mutex> l(m_); auto & e = check_gu_[j.type[m]];
                            e.first = std::max(e.first, double(std::fabs(acc - want))); e.second = std::max(e.second, double(std::fabs(want)));
                        }
                        const float v = __bfloat162float(__float2bfloat16_rn(acc));
                        float & g_or_out = g[size_t(c) * (r1 - r0) + (r - r0)];
                        if (m == 0) g_or_out = v;
                        else act_[size_t(j.slot + c) * ff_ + r] = __half2float(__float2half_rn(g_or_out / (1.f + std::exp(-g_or_out)) * v));
                    }
                }
            }
        } else {
            const int r0 = part * hidden_ / chunks, r1 = (part + 1) * hidden_ / chunks;
            const bool nl = j.type[2] == 20 && ff_ % 32 == 0, q2 = j.type[2] == GGML_GSQ_Q2_0 && ff_ % 64 == 0;
            thread_local std::vector<int8_t> aq, ap; thread_local std::vector<float> ad; thread_local std::vector<int> as;
            if (nl || q2) {
                aq.resize(size_t(j.ncols) * ff_); ad.resize(size_t(j.ncols) * ff_ / 32);
                for (int c = 0; c < j.ncols; ++c) cpu_quant_q8(act_.data() + size_t(j.slot + c) * ff_, ff_, aq.data() + size_t(c) * ff_, ad.data() + size_t(c) * ff_ / 32);
            }
            if (q2) {
                ap.resize(aq.size()); as.resize(ad.size());
                for (int c = 0; c < j.ncols; ++c) cpu_permute_q2(aq.data() + size_t(c) * ff_, ff_, ap.data() + size_t(c) * ff_, as.data() + size_t(c) * ff_ / 32);
            }
            for (int r = r0; r < r1; ++r) {
                const char * wr = row_src(j, 2, r, ff_);
                if (!nl && !q2) dequantize_host(j.type[2], wr, row.data(), ff_);
                float nlo[8];
                if (nl) {
                    const int8_t * xqc[8]; const float * xdc[8];
                    for (int c = 0; c < j.ncols; ++c) { xqc[c] = aq.data() + size_t(c) * ff_; xdc[c] = ad.data() + size_t(c) * ff_ / 32; }
                    for (int c0 = 0; c0 < j.ncols; c0 += 4) cpu_dot_iq4nl_q8_mc(reinterpret_cast<const uint8_t *>(wr), xqc + c0, xdc + c0, std::min(4, j.ncols - c0), ff_ / 32, nlo + c0);
                }
                for (int c = 0; c < j.ncols; ++c) {
                    const auto * wu = reinterpret_cast<const uint8_t *>(wr);
                    const float acc = nl ? nlo[c]
                                    : q2 ? cpu_dot_q2_0_q8p(wu, ap.data() + size_t(c) * ff_, ad.data() + size_t(c) * ff_ / 32, as.data() + size_t(c) * ff_ / 32, ff_ / 64)
                                         : cpu_dot_f32(row.data(), act_.data() + size_t(j.slot + c) * ff_, ff_);
                    out_[size_t(j.slot + c) * hidden_ + r] = __float2bfloat16_rn(acc);
                    static const bool check = std::getenv("TQFN_CPU_CHECK") != nullptr;
                    if (check && (nl || q2)) {  // int8 kernel vs dequantize_host + fp32 dot
                        std::vector<float> ref(ff_); dequantize_host(j.type[2], wr, ref.data(), ff_);
                        const float want = cpu_dot_f32(ref.data(), act_.data() + size_t(j.slot + c) * ff_, ff_);
                        std::lock_guard<std::mutex> l(m_); check_err_ = std::max(check_err_, double(std::fabs(acc - want)));
                        check_mag_ = std::max(check_mag_, double(std::fabs(want)));
                    }
                }
            }
        }
    }
    void run(int) {
        uint64_t seen = 0;
        for (;;) {
            uint64_t g;
            const auto idle0 = std::chrono::steady_clock::now();
            for (int spin = 0; (g = gen_.load(std::memory_order_acquire)) == seen; ++spin) {
                _mm_pause();
                // decode calls arrive every ~0.5 ms; after 2 ms idle (prefill, between prompts) sleep instead of spinning
                static const int spin_ms = std::getenv("TQFN_CPU_SPIN_MS") ? std::atoi(std::getenv("TQFN_CPU_SPIN_MS")) : 2;
                if ((spin & 1023) == 1023 && std::chrono::steady_clock::now() - idle0 > std::chrono::milliseconds(spin_ms)) {
                    std::unique_lock<std::mutex> l(m_); ++sleepers_;
                    cv_.wait(l, [&] { return gen_.load() != seen; }); --sleepers_;
                }
            }
            seen = g; if (stop_) return;
            for (;;) {
                const int ph = phase_.load(std::memory_order_acquire);
                const int total = ph == 0 ? phase_items_[0] : phase_items_[0] + phase_items_[1];
                const int item = next_.fetch_add(1);
                if (item >= phase_items_[0] + phase_items_[1]) break;
                if (item >= phase_items_[0])  // down needs every gate/up item finished
                    while (done_.load(std::memory_order_acquire) < phase_items_[0]) _mm_pause();
                (void) total;
                const auto w0 = std::chrono::steady_clock::now();
                work(item);
                const auto w1 = std::chrono::steady_clock::now();
                (item >= phase_items_[0] ? cpu_ns_dn_ : cpu_ns_gu_).fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count()));
                const int d = done_.fetch_add(1, std::memory_order_acq_rel) + 1;
                if (d == phase_items_[0]) t_gu_.store(w1);
                if (d == phase_items_[0] + phase_items_[1]) t_last_.store(std::chrono::steady_clock::now());
            }
        }
    }
    int hidden_, ff_; std::vector<std::thread> workers_; std::atomic<uint64_t> gen_{0}; std::atomic<int> next_{0}, done_{0}, phase_{0};
    std::atomic<bool> stop_{false}; bool pending_ = false; int phase_items_[2] = {0, 0};
    std::mutex m_; std::condition_variable cv_; std::atomic<int> sleepers_{0};
    std::chrono::steady_clock::time_point t_start_; std::atomic<std::chrono::steady_clock::time_point> t_last_{}, t_gu_{};
    std::atomic<uint64_t> cpu_ns_gu_{0}, cpu_ns_dn_{0}, gguf_rows_{0}; double gu_wall_ms_ = 0, dn_wall_ms_ = 0; uint64_t jobs_sum_ = 0, cols_sum_ = 0;
    double busy_ms_ = 0, blocked_ms_ = 0, check_err_ = 0, check_mag_ = 0; uint64_t calls_ = 0, row_mismatch_ = 0;
    std::vector<float> x_, act_, xd_; std::vector<int8_t> xq_; std::map<uint32_t, std::pair<double, double>> check_gu_; std::vector<Job> jobs_; __nv_bfloat16 * out_ = nullptr;
};
#else  // not x86: no AVX2 CPU expert kernels; the CPU miss pool is unavailable (TQFN_CPU_MISS refuses to start)
struct CpuPackedView { const char * map = nullptr; std::vector<uint64_t> base, block; std::vector<std::array<uint64_t, 6>> sizes;
                       std::vector<std::array<int, 3>> fmt; bool tiled = false; int layers = 0; };
inline CpuPackedView * g_cpu_packed = nullptr;
class CpuMissPool {
public:
    struct Job { const char * w[3]; uint32_t type[3]; size_t row_bytes[3]; int cols_t[8]; int ncols = 0; int slot = 0;
                 const char * dev[3] = {nullptr, nullptr, nullptr}, * aux[3] = {nullptr, nullptr, nullptr}; int fmt[3] = {-1, -1, -1}; bool tiled = false; };
    CpuMissPool(int, int, int) { throw std::runtime_error("TQFN_CPU_MISS needs an x86 CPU with AVX2"); }
    void start(const float *, int, std::vector<Job>, __nv_bfloat16 *) {}
    void wait() {}
};
#endif
