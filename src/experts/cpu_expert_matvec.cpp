// CPU expert matvec -- IQ4_NL (engine FMT 14, GGUF type 20).
//
// ARCHITECTURES: x86-64 = AVX2+FMA (original). aarch64 = NEON; the int8 path uses the
// dotprod extension via a function-level target attribute (ARMv8.2-A+; GB10/Grace is
// ARMv9 where dotprod is mandatory, so every aarch64 target we ship has it).
//
// Reference semantics copied from src/formats/weight/dequant.cpp:328-337 (iq4nl) and :380-383 (dispatch):
//   18 bytes per 32 weights: uint16 fp16 scale d, then 16 code bytes.
//   byte j low nibble  -> weight j      (j = 0..15)
//   byte j high nibble -> weight j + 16 (j = 0..15)
//   weight value = d * float(kvalues_iq4nl[nibble])
// I link the project's real half_to_float so the scale interpretation cannot drift.
#include "cpu_expert_matvec.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#else
#error "cpu_expert_matvec.cpp: no SIMD path for this architecture (x86-64 or aarch64 only)"
#endif
#include <cstring>
#include <cmath>
#include <vector>   // fp64 diagnostic path scratch

namespace tqllm {
float half_to_float(uint16_t h);   // from src/formats/weight/dequant.cpp (host, no CUDA)
}

namespace tqllm { namespace cpu {
namespace {

// kvalues_iq4nl, copied from src/formats/weight/kvalues_iq4nl.inc (ggml, MIT, (c) ggml authors).
// Non-linear 16-entry codebook -- this is why IQ4_NL is table-lookup-friendly: a 16-entry
// table maps directly onto vpshufb's / TBL's 16-byte lane table.
alignas(16) const int8_t KV16[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

inline float block_scale_f32(const uint8_t *p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return half_to_float(h);
}

#if defined(__x86_64__)

inline __m128i kv_table() { return _mm_load_si128(reinterpret_cast<const __m128i *>(KV16)); }

// Nibble bytes -> signed codebook bytes via vpshufb (indices 0..15, bit7 clear).
inline __m128i nib_to_kv(__m128i tbl, __m128i nib) { return _mm_shuffle_epi8(tbl, nib); }

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x1));
    return _mm_cvtss_f32(lo);
}

// Split 16 code bytes into low nibbles (weights 0..15) and high nibbles (weights 16..31).
// _mm_srli_epi16 then mask: shifting the whole 128-bit reg right by 4 bits per 16-bit word
// is safe because we mask to 0x0F afterwards, discarding any cross-byte bleed.
inline void split_nibbles(__m128i code, __m128i &lo, __m128i &hi) {
    const __m128i m = _mm_set1_epi8(0x0F);
    lo = _mm_and_si128(code, m);
    hi = _mm_and_si128(_mm_srli_epi16(code, 4), m);
}

#else  // __aarch64__

inline int8x16_t kv_table() { return vld1q_s8(KV16); }

// Nibble bytes -> signed codebook bytes via TBL (indices 0..15, always in range).
inline int8x16_t nib_to_kv(int8x16_t tbl, uint8x16_t nib) { return vqtbl1q_s8(tbl, nib); }

inline float hsum128(float32x4_t v) { return vaddvq_f32(v); }

// Split 16 code bytes into low nibbles (weights 0..15) and high nibbles (weights 16..31).
// Logical shift right on unsigned bytes: no cross-byte bleed possible, and the high
// nibble of an unsigned byte lands in 0..15 without needing a mask.
inline void split_nibbles(uint8x16_t code, uint8x16_t &lo, uint8x16_t &hi) {
    lo = vandq_u8(code, vdupq_n_u8(0x0F));
    hi = vshrq_n_u8(code, 4);
}

// Widen int8x16 -> four float32x4 (exact: |kv| <= 127, int->float conversion lossless),
// each pre-multiplied by the block scale d -- one fp32 rounding per weight, matching
// the reference `d * float(kvalues_iq4nl[nib])` exactly.
inline void kv_to_f32x4(int8x16_t kv, float d, float32x4_t out[4]) {
    const int16x8_t w16 = vmovl_s8(vget_low_s8(kv));
    const int16x8_t g16 = vmovl_s8(vget_high_s8(kv));
    out[0] = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))),  d);
    out[1] = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))), d);
    out[2] = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(g16))),  d);
    out[3] = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(g16))), d);
}

#endif  // arch

}  // namespace

#if defined(__x86_64__)

// ---------------------------------------------------------------------------
// EXACT fp32 PATH -- this is the one that must meet <=1e-5 vs dequantize_host.
//
// Per 32-weight block: vpshufb the nibbles to signed int8 codebook values, widen to
// fp32 (4 x 8 lanes), multiply by the block scale d EXACTLY as the reference does
// (y[j] = d * kv[j], rounded), then FMA against x. Reusing the W row across all ntok
// tokens is what keeps this memory-bound rather than read-bound: W is read once, not
// ntok times.
// ---------------------------------------------------------------------------
void iq4nl_expert_matvec_fp32(const uint8_t *W, int rows, int cols, int row_bytes,
                              const float *x, int ntok, float *y) {
    if (ntok < 1 || ntok > 4 || cols % 32 != 0 || !W || !x || !y) return;
    const int nb = cols / 32;
    const __m128i tbl = kv_table();

    for (int r = 0; r < rows; ++r) {
        const uint8_t *wr = W + (int64_t)r * row_bytes;
        __m256 acc[4];
        for (int t = 0; t < ntok; ++t) acc[t] = _mm256_setzero_ps();

        for (int b = 0; b < nb; ++b) {
            const uint8_t *p = wr + (int64_t)b * 18;
            const float d = block_scale_f32(p);
            const __m256 dv = _mm256_set1_ps(d);

            __m128i lo, hi;
            split_nibbles(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 2)), lo, hi);
            const __m128i slo = nib_to_kv(tbl, lo);   // signed codebook, weights 0..15
            const __m128i shi = nib_to_kv(tbl, hi);   // signed codebook, weights 16..31

            // Widen int8 -> int32 -> fp32. _mm256_cvtepi8_epi32 consumes the LOW 8 bytes.
            const __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(slo)), dv);
            const __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(slo, 8))), dv);
            const __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(shi)), dv);
            const __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(shi, 8))), dv);

            const int xb = b * 32;
            for (int t = 0; t < ntok; ++t) {
                const float *xp = x + (int64_t)t * cols + xb;
                acc[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp + 0),  acc[t]);
                acc[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 8),  acc[t]);
                acc[t] = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), acc[t]);
                acc[t] = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), acc[t]);
            }
        }
        for (int t = 0; t < ntok; ++t) y[(int64_t)t * rows + r] = hsum256(acc[t]);
    }
}

// ---------------------------------------------------------------------------
// fp64-ACCUMULATE PATH -- same dequant as fp32, double accumulation. Diagnostic
// split: if this meets 1e-5 and fp32 does not, my dequant is exact and the fp32
// residual is inherent summation error over `cols` terms.
// ---------------------------------------------------------------------------
void iq4nl_expert_matvec_fp64(const uint8_t *W, int rows, int cols, int row_bytes,
                              const float *x, int ntok, float *y) {
    if (ntok < 1 || ntok > 4 || cols % 32 != 0 || !W || !x || !y) return;
    const int nb = cols / 32;
    const __m128i tbl = kv_table();
    // Scratch: fully dequantised W row in fp32 (exact dequant values), then dot in fp64.
    std::vector<float> row(cols);

    for (int r = 0; r < rows; ++r) {
        const uint8_t *wr = W + (int64_t)r * row_bytes;
        for (int b = 0; b < nb; ++b) {
            const uint8_t *p = wr + (int64_t)b * 18;
            const float d = block_scale_f32(p);
            __m128i lo, hi;
            split_nibbles(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 2)), lo, hi);
            const __m128i slo = nib_to_kv(tbl, lo);
            const __m128i shi = nib_to_kv(tbl, hi);
            const __m256 dv = _mm256_set1_ps(d);
            _mm256_storeu_ps(&row[b*32 + 0],  _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(slo)), dv));
            _mm256_storeu_ps(&row[b*32 + 8],  _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(slo, 8))), dv));
            _mm256_storeu_ps(&row[b*32 + 16], _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(shi)), dv));
            _mm256_storeu_ps(&row[b*32 + 24], _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(shi, 8))), dv));
        }
        for (int t = 0; t < ntok; ++t) {
            const float *xp = x + (int64_t)t * cols;
            double acc = 0.0;
            for (int j = 0; j < cols; ++j) acc += (double)row[j] * (double)xp[j];
            y[(int64_t)t * rows + r] = (float)acc;
        }
    }
}

// ---------------------------------------------------------------------------
// FAST int8 PATH -- Claude's "int8/int16 maddubs, the way the GPU dp4a paths do".
// LOSSY: x is requantised to int8 per 32-block. Reported error is measured, and this
// path is deliberately NOT the one held to the <=1e-5 gate.
//
// Uses cvtepi8_epi16 + _mm256_madd_epi16 rather than _mm256_maddubs_epi16:
//   * maddubs treats operand A as UNSIGNED, but our codebook values are signed
//     (-127..113), which would need a +128 bias plus a per-block sum(kv) correction.
//   * maddubs also SATURATES its pairwise int16 sum.
//   * madd_epi16 is signed x signed, no bias, and cannot saturate here
//     (|kv|<=127, |xq|<=127 -> each pair sum <= 32258 < 32767).
// ---------------------------------------------------------------------------
void iq4nl_expert_matvec_i16(const uint8_t *W, int rows, int cols, int row_bytes,
                             const float *x, int ntok, float *y) {
    if (ntok < 1 || ntok > 4 || cols % 32 != 0 || !W || !x || !y) return;
    const int nb = cols / 32;
    const __m128i tbl = kv_table();
    const __m256i lim_hi = _mm256_set1_epi32(127);
    const __m256i lim_lo = _mm256_set1_epi32(-127);

    // PERFORMANCE FIX: x does not depend on the row, so quantising it inside the row loop
    // redid identical work `rows` times (640x for gate/up). Measured cost: at ntok=4 the
    // old layout was SLOWER than the exact fp32 path (1.73 ms vs 0.38 ms), i.e. the "fast"
    // path lost its entire purpose. Quantise once into scratch, reuse for every row.
    // Scratch is tiny: ntok*nb*2 vectors of 16 int16 = 4*80*64B = 20 KB at cols=2560.
    std::vector<__m256i> xq_lo((size_t)ntok * nb), xq_hi((size_t)ntok * nb);
    std::vector<float> rs((size_t)ntok * nb);

    for (int t = 0; t < ntok; ++t) {
        for (int b = 0; b < nb; ++b) {
            const float *xp = x + (int64_t)t * cols + b * 32;
            __m256 v0 = _mm256_loadu_ps(xp + 0),  v1 = _mm256_loadu_ps(xp + 8);
            __m256 v2 = _mm256_loadu_ps(xp + 16), v3 = _mm256_loadu_ps(xp + 24);
            const __m256 av = _mm256_set1_ps(-0.0f);   // abs mask: clear the sign bit
            __m256 m0 = _mm256_max_ps(_mm256_andnot_ps(av, v0), _mm256_andnot_ps(av, v1));
            __m256 m1 = _mm256_max_ps(_mm256_andnot_ps(av, v2), _mm256_andnot_ps(av, v3));
            // Horizontal MAX (not sum) over the per-lane maxima -> block max|x|.
            __m256 mm = _mm256_max_ps(m0, m1);
            __m128 l = _mm_max_ps(_mm256_castps256_ps128(mm), _mm256_extractf128_ps(mm, 1));
            l = _mm_max_ps(l, _mm_movehl_ps(l, l));
            l = _mm_max_ss(l, _mm_shuffle_ps(l, l, 0x1));
            const float maxabs = _mm_cvtss_f32(l);
            const float inv = (maxabs > 0.0f) ? (127.0f / maxabs) : 0.0f;
            const size_t idx = (size_t)t * nb + b;
            rs[idx] = (inv > 0.0f) ? (1.0f / inv) : 0.0f;
            const __m256 sv = _mm256_set1_ps(inv);

            __m256i q0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v0, sv)), lim_lo), lim_hi);
            __m256i q1 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v1, sv)), lim_lo), lim_hi);
            __m256i q2 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v2, sv)), lim_lo), lim_hi);
            __m256i q3 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v3, sv)), lim_lo), lim_hi);
            // packs_epi32 works within 128-bit lanes -> permute to restore order.
            xq_lo[idx] = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0xD8);  // weights 0..15
            xq_hi[idx] = _mm256_permute4x64_epi64(_mm256_packs_epi32(q2, q3), 0xD8);  // weights 16..31
        }
    }

    for (int r = 0; r < rows; ++r) {
        const uint8_t *wr = W + (int64_t)r * row_bytes;
        __m256 accf[4];
        for (int t = 0; t < ntok; ++t) accf[t] = _mm256_setzero_ps();

        for (int b = 0; b < nb; ++b) {
            const uint8_t *p = wr + (int64_t)b * 18;
            // BUG FIX: the weight block scale d was missing from this path entirely.
            // The true weight is d*kv, so  sum(d*kv*x) = d * rs * sum(kv*xq).
            // Without d the result was ~1e5 off -- a correctness defect, not quantisation loss.
            const float d = block_scale_f32(p);

            // --- codebook values as int16, matching the xq lane order ---
            __m128i lo, hi;
            split_nibbles(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 2)), lo, hi);
            const __m256i kv_lo = _mm256_cvtepi8_epi16(nib_to_kv(tbl, lo));
            const __m256i kv_hi = _mm256_cvtepi8_epi16(nib_to_kv(tbl, hi));

            for (int t = 0; t < ntok; ++t) {
                const size_t idx = (size_t)t * nb + b;
                __m256i m0 = _mm256_madd_epi16(kv_lo, xq_lo[idx]);   // 8 x int32 pair sums
                __m256i m1 = _mm256_madd_epi16(kv_hi, xq_hi[idx]);
                __m256i mi = _mm256_add_epi32(m0, m1);
                // d * rs[t] * sum(kv*xq): fold the weight scale and the x scale together.
                accf[t] = _mm256_add_ps(accf[t],
                    _mm256_mul_ps(_mm256_cvtepi32_ps(mi), _mm256_set1_ps(d * rs[idx])));
            }
        }
        for (int t = 0; t < ntok; ++t) y[(int64_t)t * rows + r] = hsum256(accf[t]);
    }
}

#else  // __aarch64__ -- NEON port, same three kernels, same rounding contract.

// ---------------------------------------------------------------------------
// EXACT fp32 PATH (NEON) -- <=1e-5 vs dequantize_host, same as the AVX2 contract.
//
// Per 32-weight block: TBL the nibbles to signed int8 codebook values (vqtbl1q is the
// exact analogue of vpshufb for a 16-byte table), widen int8 -> int16 -> int32 -> fp32
// (every widening exact: |kv| <= 127), multiply by d with ONE fp32 rounding per weight
// (matches the reference `d * float(kv[j])` bit-for-bit), then FMA against x. W is
// decoded once per block and reused across all ntok tokens, same as the x86 path.
// ---------------------------------------------------------------------------
void iq4nl_expert_matvec_fp32(const uint8_t *W, int rows, int cols, int row_bytes,
                              const float *x, int ntok, float *y) {
    if (ntok < 1 || ntok > 4 || cols % 32 != 0 || !W || !x || !y) return;
    const int nb = cols / 32;
    const int8x16_t tbl = kv_table();

    for (int r = 0; r < rows; ++r) {
        const uint8_t *wr = W + (int64_t)r * row_bytes;
        float32x4_t acc[4];
        for (int t = 0; t < ntok; ++t) acc[t] = vdupq_n_f32(0.0f);

        for (int b = 0; b < nb; ++b) {
            const uint8_t *p = wr + (int64_t)b * 18;
            const float d = block_scale_f32(p);

            uint8x16_t lo, hi;
            split_nibbles(vld1q_u8(p + 2), lo, hi);
            float32x4_t wl[4], wh[4];
            kv_to_f32x4(nib_to_kv(tbl, lo), d, wl);   // weights 0..15
            kv_to_f32x4(nib_to_kv(tbl, hi), d, wh);   // weights 16..31

            const int xb = b * 32;
            for (int t = 0; t < ntok; ++t) {
                const float *xp = x + (int64_t)t * cols + xb;
                acc[t] = vfmaq_f32(acc[t], wl[0], vld1q_f32(xp + 0));
                acc[t] = vfmaq_f32(acc[t], wl[1], vld1q_f32(xp + 4));
                acc[t] = vfmaq_f32(acc[t], wl[2], vld1q_f32(xp + 8));
                acc[t] = vfmaq_f32(acc[t], wl[3], vld1q_f32(xp + 12));
                acc[t] = vfmaq_f32(acc[t], wh[0], vld1q_f32(xp + 16));
                acc[t] = vfmaq_f32(acc[t], wh[1], vld1q_f32(xp + 20));
                acc[t] = vfmaq_f32(acc[t], wh[2], vld1q_f32(xp + 24));
                acc[t] = vfmaq_f32(acc[t], wh[3], vld1q_f32(xp + 28));
            }
        }
        for (int t = 0; t < ntok; ++t) y[(int64_t)t * rows + r] = hsum128(acc[t]);
    }
}

// ---------------------------------------------------------------------------
// fp64-ACCUMULATE PATH (NEON) -- identical dequant to the fp32 path (the same
// kv_to_f32x4 helper), double accumulation. GATE 1 in the test requires this to be
// BIT-IDENTICAL (max_abs == 0.0) to dequantize_host(type 20): exact int->float
// widening + one rounded multiply per weight + scalar fp64 dot guarantees that.
// ---------------------------------------------------------------------------
void iq4nl_expert_matvec_fp64(const uint8_t *W, int rows, int cols, int row_bytes,
                              const float *x, int ntok, float *y) {
    if (ntok < 1 || ntok > 4 || cols % 32 != 0 || !W || !x || !y) return;
    const int nb = cols / 32;
    const int8x16_t tbl = kv_table();
    std::vector<float> row(cols);

    for (int r = 0; r < rows; ++r) {
        const uint8_t *wr = W + (int64_t)r * row_bytes;
        for (int b = 0; b < nb; ++b) {
            const uint8_t *p = wr + (int64_t)b * 18;
            const float d = block_scale_f32(p);
            uint8x16_t lo, hi;
            split_nibbles(vld1q_u8(p + 2), lo, hi);
            float32x4_t wl[4], wh[4];
            kv_to_f32x4(nib_to_kv(tbl, lo), d, wl);
            kv_to_f32x4(nib_to_kv(tbl, hi), d, wh);
            vst1q_f32(&row[b*32 + 0],  wl[0]);
            vst1q_f32(&row[b*32 + 4],  wl[1]);
            vst1q_f32(&row[b*32 + 8],  wl[2]);
            vst1q_f32(&row[b*32 + 12], wl[3]);
            vst1q_f32(&row[b*32 + 16], wh[0]);
            vst1q_f32(&row[b*32 + 20], wh[1]);
            vst1q_f32(&row[b*32 + 24], wh[2]);
            vst1q_f32(&row[b*32 + 28], wh[3]);
        }
        for (int t = 0; t < ntok; ++t) {
            const float *xp = x + (int64_t)t * cols;
            double acc = 0.0;
            for (int j = 0; j < cols; ++j) acc += (double)row[j] * (double)xp[j];
            y[(int64_t)t * rows + r] = (float)acc;
        }
    }
}

// ---------------------------------------------------------------------------
// FAST int8 PATH (NEON + dotprod). Same contract as the x86 version: LOSSY, x is
// requantised to int8 per 32-block, gated at rms_rel <= 1e-2, never inside the 1e-5.
//
// Where x86 needs cvtepi8_epi16 + madd_epi16 (two instructions per 16 products, and
// the maddubs saturation/bias trap documented in the x86 section), ARM's SDOT does
// signed int8 x int8 -> int32 lane accumulation in ONE instruction with no bias and
// no saturation: per lane |sum of 4 products| <= 4*127*127 = 64516 << INT32_MAX.
// Integer accumulation is exact, so the only rounding vs the x86 path is the float
// conversion order -- orders of magnitude inside the 1e-2 budget.
//
// target("+dotprod"): dotprod is optional in ARMv8.0/8.1 but mandatory from ARMv8.2
// and in all of ARMv9 (GB10/Grace included), so every aarch64 target we ship has it.
// The attribute scopes the ISA extension to this function alone; the rest of the file
// stays baseline armv8-a.
// ---------------------------------------------------------------------------
__attribute__((target("+dotprod")))
void iq4nl_expert_matvec_i16(const uint8_t *W, int rows, int cols, int row_bytes,
                             const float *x, int ntok, float *y) {
    if (ntok < 1 || ntok > 4 || cols % 32 != 0 || !W || !x || !y) return;
    const int nb = cols / 32;
    const int8x16_t tbl = kv_table();

    // PERFORMANCE FIX (same as x86): quantise x ONCE into scratch, reuse for every row.
    // Scratch: ntok*nb*2 x int8x16 = 4*80*32B = 10 KB at cols=2560.
    std::vector<int8x16_t> xq_lo((size_t)ntok * nb), xq_hi((size_t)ntok * nb);
    std::vector<float> rs((size_t)ntok * nb);

    const int32x4_t lim_hi = vdupq_n_s32(127);
    const int32x4_t lim_lo = vdupq_n_s32(-127);

    // 4 x float32x4 -> clamp(round-to-nearest-even(v*inv)) -> int8x8, two of which
    // combine into one int8x16. vcvtnq_s32_f32 is round-to-nearest-even, matching
    // _mm256_cvtps_epi32's default MXCSR mode exactly.
    auto quant16 = [&](const float32x4_t v[4], float inv) -> int8x16_t {
        int32x4_t q0 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_n_f32(v[0], inv)), lim_lo), lim_hi);
        int32x4_t q1 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_n_f32(v[1], inv)), lim_lo), lim_hi);
        int32x4_t q2 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_n_f32(v[2], inv)), lim_lo), lim_hi);
        int32x4_t q3 = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_n_f32(v[3], inv)), lim_lo), lim_hi);
        const int16x8_t h01 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
        const int16x8_t h23 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
        return vcombine_s8(vmovn_s16(h01), vmovn_s16(h23));   // narrowing is exact: |q| <= 127
    };

    for (int t = 0; t < ntok; ++t) {
        for (int b = 0; b < nb; ++b) {
            const float *xp = x + (int64_t)t * cols + b * 32;
            float32x4_t v[8];
            for (int i = 0; i < 8; ++i) v[i] = vld1q_f32(xp + 4 * i);
            float32x4_t a[8];
            for (int i = 0; i < 8; ++i) a[i] = vabsq_f32(v[i]);
            // block max|x|: tree-max the eight lanes, then horizontal max.
            float32x4_t mm = vmaxq_f32(vmaxq_f32(vmaxq_f32(a[0], a[1]), vmaxq_f32(a[2], a[3])),
                                       vmaxq_f32(vmaxq_f32(a[4], a[5]), vmaxq_f32(a[6], a[7])));
            const float maxabs = vmaxvq_f32(mm);
            const float inv = (maxabs > 0.0f) ? (127.0f / maxabs) : 0.0f;
            const size_t idx = (size_t)t * nb + b;
            rs[idx] = (inv > 0.0f) ? (1.0f / inv) : 0.0f;
            xq_lo[idx] = quant16(v,     inv);   // weights 0..15
            xq_hi[idx] = quant16(v + 4, inv);   // weights 16..31
        }
    }

    for (int r = 0; r < rows; ++r) {
        const uint8_t *wr = W + (int64_t)r * row_bytes;
        float32x4_t accf[4];
        for (int t = 0; t < ntok; ++t) accf[t] = vdupq_n_f32(0.0f);

        for (int b = 0; b < nb; ++b) {
            const uint8_t *p = wr + (int64_t)b * 18;
            // Weight block scale d -- the bug the x86 path also had to fix: the true
            // weight is d*kv, so sum(d*kv*x) = d * rs * sum(kv*xq).
            const float d = block_scale_f32(p);

            uint8x16_t lo, hi;
            split_nibbles(vld1q_u8(p + 2), lo, hi);
            const int8x16_t kv_lo = nib_to_kv(tbl, lo);   // signed codebook, weights 0..15
            const int8x16_t kv_hi = nib_to_kv(tbl, hi);   // signed codebook, weights 16..31

            for (int t = 0; t < ntok; ++t) {
                const size_t idx = (size_t)t * nb + b;
                // SDOT: 16 signed int8 products per instruction, exact int32 lanes.
                int32x4_t mi = vdotq_s32(vdupq_n_s32(0), kv_lo, xq_lo[idx]);
                mi = vdotq_s32(mi, kv_hi, xq_hi[idx]);
                // d * rs[t] * sum(kv*xq): fold the weight scale and the x scale together.
                accf[t] = vfmaq_f32(accf[t], vcvtq_f32_s32(mi), vdupq_n_f32(d * rs[idx]));
            }
        }
        for (int t = 0; t < ntok; ++t) y[(int64_t)t * rows + r] = hsum128(accf[t]);
    }
}

#endif  // arch

}}  // namespace tqllm::cpu
