// AVX2 CPU expert matvec -- IQ4_NL (engine FMT 14, GGUF type 20).
//
// Reference semantics copied from src/dequant.cpp:328-337 (iq4nl) and :380-383 (dispatch):
//   18 bytes per 32 weights: uint16 fp16 scale d, then 16 code bytes.
//   byte j low nibble  -> weight j      (j = 0..15)
//   byte j high nibble -> weight j + 16 (j = 0..15)
//   weight value = d * float(kvalues_iq4nl[nibble])
// I link the project's real half_to_float so the scale interpretation cannot drift.
#include "cpu_expert_matvec.hpp"

#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <vector>   // fp64 diagnostic path scratch

namespace tqllm {
float half_to_float(uint16_t h);   // from src/dequant.cpp (host, no CUDA)
}

namespace tqllm { namespace cpu {
namespace {

// kvalues_iq4nl, copied from kernels/kvalues_iq4nl.inc (ggml, MIT, (c) ggml authors).
// Non-linear 16-entry codebook -- this is why IQ4_NL is vpshufb-friendly: a 16-entry
// table maps directly onto _mm_shuffle_epi8's 16-byte lane table.
alignas(16) const int8_t KV16[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

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

inline float block_scale_f32(const uint8_t *p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return half_to_float(h);
}

}  // namespace

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

}}  // namespace tqllm::cpu
