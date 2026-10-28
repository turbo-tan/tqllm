#pragma once
// Integer dot products of packed GGUF rows against an int8-quantized activation (Arm dotprod).
// The activation is quantized once per projection input: 256-element blocks, one FP32 scale each,
// plus 16-element sums so K-quant minimums and the Q6_K offset cost one multiply per block.
#include "gguf.hpp"
#include "host_iq_dot.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tqllm {
struct HostQ8 {
    int n = 0;
    std::vector<int8_t> q;
    std::vector<float> d;        // one per 256 elements
    std::vector<int16_t> sums;   // one per 16 elements
};

inline bool host_q8_supported(uint32_t type, int cols) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    return cols % 256 == 0 && (type == GGML_Q3_K || type == GGML_Q4_K || type == GGML_Q5_K || type == GGML_Q6_K ||
                               type == GGML_Q8_0 || host_iq_type(type));
#else
    (void) type; (void) cols;
    return false;
#endif
}

inline void host_quantize_q8(const float * x, int n, HostQ8 & out) {
    if (n % 256) throw std::invalid_argument("int8 activation length must be a multiple of 256");
    out.n = n; out.q.resize(n); out.d.resize(n / 256); out.sums.resize(n / 16);
    for (int b = 0; b < n / 256; ++b) {
        const float * xb = x + 256 * b;
        float amax = 0.f;
#if defined(__aarch64__)
        float32x4_t m = vdupq_n_f32(0.f);
        for (int j = 0; j < 256; j += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(xb + j)));
        amax = vmaxvq_f32(m);
#else
        for (int j = 0; j < 256; ++j) amax = std::max(amax, std::abs(xb[j]));
#endif
        const float d = amax / 127.f, id = amax > 0.f ? 127.f / amax : 0.f;
        out.d[b] = d;
        int8_t * qb = out.q.data() + 256 * b;
        for (int j = 0; j < 256; ++j) qb[j] = int8_t(std::lrintf(xb[j] * id));
        for (int g = 0; g < 16; ++g) {
            int s = 0;
            for (int j = 0; j < 16; ++j) s += qb[16 * g + j];
            out.sums[16 * b + g] = int16_t(s);
        }
    }
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
inline float host_fp16(const unsigned char * p) { __fp16 h; std::memcpy(&h, p, 2); return float(h); }

// Q4_K / Q5_K 12-byte scale field -> eight 6-bit scales and eight 6-bit minimums.
inline void host_k4_scales(const unsigned char * p, uint8_t scales[8], uint8_t mins[8]) {
    uint32_t w[3]; std::memcpy(w, p, 12);
    const uint32_t s[2] = {w[0] & 0x3f3f3f3fu, (w[2] & 0x0f0f0f0fu) | (((w[0] >> 6) & 0x03030303u) << 4)};
    const uint32_t m[2] = {w[1] & 0x3f3f3f3fu, ((w[2] >> 4) & 0x0f0f0f0fu) | (((w[1] >> 6) & 0x03030303u) << 4)};
    std::memcpy(scales, s, 8); std::memcpy(mins, m, 8);
}

// Sum each of four int32x4 dot accumulators into one lane: lane i = horizontal sum of a[i].
inline int32x4_t host_hsum4(int32x4_t a0, int32x4_t a1, int32x4_t a2, int32x4_t a3) {
    return vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
}

// One packed row against N int8 activations: each block is decoded once and dotted N times, so a batch reads
// the weights once. Integer block totals are exact; the float accumulation (one fused multiply-add per block
// with the scale product rounded once) is the same as each SMMLA lane in host_q8_mmla.hpp, so single tokens,
// sdot batches and SMMLA batches all give bit-identical results.
template <int N>
inline void host_q8_dot_n(uint32_t type, const unsigned char * row, const HostQ8 * const * xs, float * out) {
    const int blocks = xs[0]->n / 256;
    const int32x4_t zero = vdupq_n_s32(0);
    float32x4_t acc[N];
    float sum[N];
    for (int i = 0; i < N; ++i) { acc[i] = vdupq_n_f32(0.f); sum[i] = 0.f; }
    if (host_iq_type(type)) {
        host_iq_dot_n<N>(type, row, xs, out);
        return;
    } else if (type == GGML_Q3_K) {
        // 110 bytes: hmask[32], qs[64], scales[12] (16 x 6-bit - 32), d. Code = 2 low bits - (high bit ? 0 : 4).
        const uint8x16_t three = vdupq_n_u8(3), four = vdupq_n_u8(4);
        for (int b = 0; b < blocks; ++b, row += 110) {
            __fp16 h; std::memcpy(&h, row + 108, 2);
            const float dw = float(h);
            // Scales: low nibbles from bytes 0..7, high 2 bits from bytes 8..11.
            uint32_t a[3]; std::memcpy(a, row + 96, 12);
            const uint32_t sw[4] = {(a[0] & 0x0f0f0f0fu) | ((a[2] & 0x03030303u) << 4),
                                    (a[1] & 0x0f0f0f0fu) | (((a[2] >> 2) & 0x03030303u) << 4),
                                    ((a[0] >> 4) & 0x0f0f0f0fu) | (((a[2] >> 4) & 0x03030303u) << 4),
                                    ((a[1] >> 4) & 0x0f0f0f0fu) | (((a[2] >> 6) & 0x03030303u) << 4)};
            int8_t sc[16];
            vst1q_s8(sc, vsubq_s8(vreinterpretq_s8_u32(vld1q_u32(sw)), vdupq_n_s8(32)));
            const uint8x16_t hm[2] = {vld1q_u8(row), vld1q_u8(row + 16)};
            int32x4_t isum[N];
            for (int i = 0; i < N; ++i) isum[i] = zero;
#pragma clang loop unroll(full)
            for (int n = 0; n < 2; ++n) {
                const uint8x16_t q[2] = {vld1q_u8(row + 32 + 32 * n), vld1q_u8(row + 48 + 32 * n)};
#pragma clang loop unroll(full)
                for (int j = 0; j < 4; ++j) {
                    const uint8x16_t bit = vdupq_n_u8(uint8_t(1u << (4 * n + j)));
                    int8x16_t c[2];
#pragma clang loop unroll(full)
                    for (int w = 0; w < 2; ++w) {
                        const uint8x16_t low = vandq_u8(j ? vshlq_u8(q[w], vdupq_n_s8(int8_t(-2 * j))) : q[w], three);
                        c[w] = vreinterpretq_s8_u8(vsubq_u8(low, vbicq_u8(four, vtstq_u8(hm[w], bit))));
                    }
                    const int k = 8 * n + 2 * j;
                    for (int i = 0; i < N; ++i) {
                        const int8_t * y = xs[i]->q.data() + 256 * b + 128 * n + 32 * j;
                        isum[i] = vmlaq_n_s32(isum[i], vdotq_s32(zero, c[0], vld1q_s8(y)), sc[k]);
                        isum[i] = vmlaq_n_s32(isum[i], vdotq_s32(zero, c[1], vld1q_s8(y + 16)), sc[k + 1]);
                    }
                }
            }
            for (int i = 0; i < N; ++i) sum[i] = std::fma(float(vaddvq_s32(isum[i])), dw * xs[i]->d[b], sum[i]);
        }
    } else if (type == GGML_Q4_K || type == GGML_Q5_K) {
        const bool five = type == GGML_Q5_K;
        const uint8x16_t low = vdupq_n_u8(15), bit = vdupq_n_u8(16);
        for (int b = 0; b < blocks; ++b, row += five ? 176 : 144) {
            const float dw = host_fp16(row), mw = host_fp16(row + 2);
            uint8_t scales[8], mins[8];
            host_k4_scales(row + 4, scales, mins);
            const uint16x8_t sc = vmovl_u8(vld1_u8(scales));
            const int32x4_t sc_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc)));
            const int32x4_t sc_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc)));
            const int16x8_t mn = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mins)));
            const unsigned char * qs = row + (five ? 48 : 16);
            uint8x16_t h0 = vdupq_n_u8(0), h1 = h0;
            if (five) { h0 = vld1q_u8(row + 16); h1 = vld1q_u8(row + 32); }
            int32x4_t isum[N];
            for (int pair = 0; pair < 2; ++pair) {
                uint8x16_t c[8];  // groups 4 * pair .. 4 * pair + 3, two 16-byte halves each
                for (int jj = 0; jj < 2; ++jj, qs += 32) {
                    const uint8x16_t q0 = vld1q_u8(qs), q1 = vld1q_u8(qs + 16);
                    uint8x16_t a0 = vandq_u8(q0, low), a1 = vandq_u8(q1, low);
                    uint8x16_t b0 = vshrq_n_u8(q0, 4), b1 = vshrq_n_u8(q1, 4);
                    if (five) {
                        a0 = vorrq_u8(a0, vandq_u8(vshlq_n_u8(h0, 4), bit));
                        a1 = vorrq_u8(a1, vandq_u8(vshlq_n_u8(h1, 4), bit));
                        b0 = vorrq_u8(b0, vandq_u8(vshlq_n_u8(h0, 3), bit));
                        b1 = vorrq_u8(b1, vandq_u8(vshlq_n_u8(h1, 3), bit));
                        h0 = vshrq_n_u8(h0, 2); h1 = vshrq_n_u8(h1, 2);
                    }
                    c[4 * jj] = a0; c[4 * jj + 1] = a1; c[4 * jj + 2] = b0; c[4 * jj + 3] = b1;
                }
                for (int i = 0; i < N; ++i) {
                    const int8_t * y = xs[i]->q.data() + 256 * b + 128 * pair;
                    int32x4_t g[4];
                    for (int k = 0; k < 4; ++k)
                        g[k] = vdotq_s32(vdotq_s32(zero, vreinterpretq_s8_u8(c[2 * k]), vld1q_s8(y + 32 * k)),
                                         vreinterpretq_s8_u8(c[2 * k + 1]), vld1q_s8(y + 32 * k + 16));
                    const int32x4_t h = host_hsum4(g[0], g[1], g[2], g[3]);
                    isum[i] = pair ? vmlaq_s32(isum[i], h, sc_hi) : vmulq_s32(h, sc_lo);
                }
            }
            for (int i = 0; i < N; ++i) {
                const int16_t * bsums = xs[i]->sums.data() + 16 * b;
                const int16x8_t group_sums = vpaddq_s16(vld1q_s16(bsums), vld1q_s16(bsums + 8));
                int32x4_t msum = vmull_s16(vget_low_s16(group_sums), vget_low_s16(mn));
                msum = vmlal_s16(msum, vget_high_s16(group_sums), vget_high_s16(mn));
                sum[i] = std::fma(float(vaddvq_s32(isum[i])), dw * xs[i]->d[b], sum[i]);
                sum[i] = std::fma(-float(vaddvq_s32(msum)), mw * xs[i]->d[b], sum[i]);
            }
        }
    } else if (type == GGML_Q6_K) {
        const uint8x16_t low = vdupq_n_u8(15), high = vdupq_n_u8(48);
        for (int b = 0; b < blocks; ++b, row += 210) {
            const float dw = host_fp16(row + 208);
            const int8x16_t scv = vld1q_s8(reinterpret_cast<const int8_t *>(row + 192));
            const int16x8_t s_lo = vmovl_s8(vget_low_s8(scv)), s_hi = vmovl_s8(vget_high_s8(scv));
            const int32x4_t sq[4] = {vmovl_s16(vget_low_s16(s_lo)), vmovl_s16(vget_high_s16(s_lo)),
                                     vmovl_s16(vget_low_s16(s_hi)), vmovl_s16(vget_high_s16(s_hi))};
            int32x4_t isum[N];
            for (int part = 0; part < 2; ++part) {
                const unsigned char * ql = row + 64 * part, * qh = row + 128 + 32 * part;
                const uint8x16_t l[4] = {vld1q_u8(ql), vld1q_u8(ql + 16), vld1q_u8(ql + 32), vld1q_u8(ql + 48)};
                const uint8x16_t hb[2] = {vld1q_u8(qh), vld1q_u8(qh + 16)};
                // c[2 * group + half]: group k of 32 takes qh bits 2k..2k+1, moved to code bits 4..5.
                uint8x16_t c[8];
                for (int half = 0; half < 2; ++half) {
                    c[half] = vorrq_u8(vandq_u8(l[half], low), vandq_u8(vshlq_n_u8(hb[half], 4), high));
                    c[2 + half] = vorrq_u8(vandq_u8(l[2 + half], low), vandq_u8(vshlq_n_u8(hb[half], 2), high));
                    c[4 + half] = vorrq_u8(vshrq_n_u8(l[half], 4), vandq_u8(hb[half], high));
                    c[6 + half] = vorrq_u8(vshrq_n_u8(l[2 + half], 4), vandq_u8(vshrq_n_u8(hb[half], 2), high));
                }
                for (int i = 0; i < N; ++i) {
                    const int8_t * y = xs[i]->q.data() + 256 * b + 128 * part;
                    int32x4_t g[8];
                    for (int k = 0; k < 8; ++k) g[k] = vdotq_s32(zero, vreinterpretq_s8_u8(c[k]), vld1q_s8(y + 16 * k));
                    const int32x4_t lo = vmulq_s32(host_hsum4(g[0], g[1], g[2], g[3]), sq[2 * part]);
                    const int32x4_t sum = vmlaq_s32(lo, host_hsum4(g[4], g[5], g[6], g[7]), sq[2 * part + 1]);
                    isum[i] = part ? vaddq_s32(isum[i], sum) : sum;
                }
            }
            for (int i = 0; i < N; ++i) {
                // Codes are stored +32; remove that offset with the 16-element activation sums.
                const int16_t * bsums = xs[i]->sums.data() + 16 * b;
                int32x4_t offset = vmull_s16(vget_low_s16(s_lo), vld1_s16(bsums));
                offset = vmlal_s16(offset, vget_high_s16(s_lo), vld1_s16(bsums + 4));
                offset = vmlal_s16(offset, vget_low_s16(s_hi), vld1_s16(bsums + 8));
                offset = vmlal_s16(offset, vget_high_s16(s_hi), vld1_s16(bsums + 12));
                sum[i] = std::fma(float(vaddvq_s32(vsubq_s32(isum[i], vshlq_n_s32(offset, 5)))), dw * xs[i]->d[b], sum[i]);
            }
        }
    } else if (type == GGML_Q8_0) {
        for (int b = 0; b < blocks; ++b) {
            float32x4_t part[N];
            for (int i = 0; i < N; ++i) part[i] = vdupq_n_f32(0.f);
            for (int k = 0; k < 8; ++k, row += 34) {
                const int8x16_t w0 = vld1q_s8(reinterpret_cast<const int8_t *>(row + 2));
                const int8x16_t w1 = vld1q_s8(reinterpret_cast<const int8_t *>(row + 18));
                const float dw = host_fp16(row);
                for (int i = 0; i < N; ++i) {
                    const int8_t * y = xs[i]->q.data() + 256 * b + 32 * k;
                    const int32x4_t p = vdotq_s32(vdotq_s32(zero, w0, vld1q_s8(y)), w1, vld1q_s8(y + 16));
                    part[i] = vfmaq_n_f32(part[i], vcvtq_f32_s32(p), dw);
                }
            }
            for (int i = 0; i < N; ++i) acc[i] = vfmaq_n_f32(acc[i], part[i], xs[i]->d[b]);
        }
        for (int i = 0; i < N; ++i) sum[i] = vaddvq_f32(acc[i]);
    } else throw std::invalid_argument("unsupported int8 host dot format");
    for (int i = 0; i < N; ++i) out[i] = sum[i];
}

inline float host_q8_dot(uint32_t type, const unsigned char * row, const HostQ8 & x) {
    const HostQ8 * xs[1] = {&x};
    float out;
    host_q8_dot_n<1>(type, row, xs, &out);
    return out;
}

// Rows against any number of activations, four at a time; out[i] receives the dot with xs[i].
inline void host_q8_dot_batch(uint32_t type, const unsigned char * row, const HostQ8 * const * xs, int count, float * out) {
    for (; count >= 4; count -= 4, xs += 4, out += 4) host_q8_dot_n<4>(type, row, xs, out);
    if (count == 3) host_q8_dot_n<3>(type, row, xs, out);
    else if (count == 2) host_q8_dot_n<2>(type, row, xs, out);
    else if (count == 1) host_q8_dot_n<1>(type, row, xs, out);
}
#endif
}
