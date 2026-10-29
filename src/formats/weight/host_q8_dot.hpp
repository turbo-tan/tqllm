#pragma once
// Integer dot products of packed GGUF rows against an int8-quantized activation (Arm dotprod).
// The activation is quantized once per projection input: 256-element blocks, one FP32 scale each,
// plus 16-element sums so K-quant minimums and the Q6_K offset cost one multiply per block.
#include "gguf.hpp"
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
    return cols % 256 == 0 && (type == GGML_Q4_K || type == GGML_Q5_K || type == GGML_Q6_K || type == GGML_Q8_0);
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

inline float host_q8_dot(uint32_t type, const unsigned char * row, const HostQ8 & x) {
    const int blocks = x.n / 256;
    const int8_t * y = x.q.data();
    const int16_t * bsums = x.sums.data();
    const int32x4_t zero = vdupq_n_s32(0);
    float sum = 0.f;
    if (type == GGML_Q4_K || type == GGML_Q5_K) {
        const bool five = type == GGML_Q5_K;
        const uint8x16_t low = vdupq_n_u8(15), one = vdupq_n_u8(1);
        for (int b = 0; b < blocks; ++b, row += five ? 176 : 144, y += 256, bsums += 16) {
            const float d = host_fp16(row) * x.d[b], dmin = host_fp16(row + 2) * x.d[b];
            uint8_t scales[8], mins[8];
            host_k4_scales(row + 4, scales, mins);
            const int16x8_t group_sums = vpaddq_s16(vld1q_s16(bsums), vld1q_s16(bsums + 8));
            const int16x8_t mn = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mins)));
            int32x4_t msum = vmull_s16(vget_low_s16(group_sums), vget_low_s16(mn));
            msum = vmlal_s16(msum, vget_high_s16(group_sums), vget_high_s16(mn));
            const unsigned char * qs = row + (five ? 48 : 16);
            uint8x16_t h0 = vdupq_n_u8(0), h1 = h0;
            if (five) { h0 = vld1q_u8(row + 16); h1 = vld1q_u8(row + 32); }
            int32x4_t isum = zero;
            for (int j = 0; j < 4; ++j, qs += 32) {
                const uint8x16_t q0 = vld1q_u8(qs), q1 = vld1q_u8(qs + 16);
                uint8x16_t a0 = vandq_u8(q0, low), a1 = vandq_u8(q1, low);
                uint8x16_t b0 = vshrq_n_u8(q0, 4), b1 = vshrq_n_u8(q1, 4);
                if (five) {
                    a0 = vorrq_u8(a0, vshlq_n_u8(vandq_u8(h0, one), 4));
                    a1 = vorrq_u8(a1, vshlq_n_u8(vandq_u8(h1, one), 4));
                    b0 = vorrq_u8(b0, vshlq_n_u8(vandq_u8(vshrq_n_u8(h0, 1), one), 4));
                    b1 = vorrq_u8(b1, vshlq_n_u8(vandq_u8(vshrq_n_u8(h1, 1), one), 4));
                    h0 = vshrq_n_u8(h0, 2); h1 = vshrq_n_u8(h1, 2);
                }
                const int8_t * yj = y + 64 * j;
                int32x4_t p = vdotq_s32(zero, vreinterpretq_s8_u8(a0), vld1q_s8(yj));
                p = vdotq_s32(p, vreinterpretq_s8_u8(a1), vld1q_s8(yj + 16));
                int32x4_t r = vdotq_s32(zero, vreinterpretq_s8_u8(b0), vld1q_s8(yj + 32));
                r = vdotq_s32(r, vreinterpretq_s8_u8(b1), vld1q_s8(yj + 48));
                isum = vmlaq_n_s32(isum, p, scales[2 * j]);
                isum = vmlaq_n_s32(isum, r, scales[2 * j + 1]);
            }
            sum += d * float(vaddvq_s32(isum)) - dmin * float(vaddvq_s32(msum));
        }
    } else if (type == GGML_Q6_K) {
        const uint8x16_t low = vdupq_n_u8(15), three = vdupq_n_u8(3);
        for (int b = 0; b < blocks; ++b, row += 210, y += 256, bsums += 16) {
            const float d = host_fp16(row + 208) * x.d[b];
            const int8_t * sc = reinterpret_cast<const int8_t *>(row + 192);
            // Codes are stored +32; remove that offset with the 16-element activation sums.
            const int8x16_t scv = vld1q_s8(sc);
            int32x4_t offset = vmull_s16(vget_low_s16(vmovl_s8(vget_low_s8(scv))), vld1_s16(bsums));
            offset = vmlal_s16(offset, vget_high_s16(vmovl_s8(vget_low_s8(scv))), vld1_s16(bsums + 4));
            offset = vmlal_s16(offset, vget_low_s16(vmovl_s8(vget_high_s8(scv))), vld1_s16(bsums + 8));
            offset = vmlal_s16(offset, vget_high_s16(vmovl_s8(vget_high_s8(scv))), vld1_s16(bsums + 12));
            int32x4_t isum = zero;
            for (int part = 0; part < 2; ++part) {
                const unsigned char * ql = row + 64 * part, * qh = row + 128 + 32 * part;
                const uint8x16_t l[4] = {vld1q_u8(ql), vld1q_u8(ql + 16), vld1q_u8(ql + 32), vld1q_u8(ql + 48)};
                uint8x16_t h[2] = {vld1q_u8(qh), vld1q_u8(qh + 16)};
                const int8_t * yp = y + 128 * part;
                const int8_t * sp = sc + 8 * part;
                for (int group = 0; group < 4; ++group) {
                    for (int half = 0; half < 2; ++half) {
                        const uint8x16_t lo = group >= 2 ? vshrq_n_u8(l[(group % 2) * 2 + half], 4)
                                                         : vandq_u8(l[(group % 2) * 2 + half], low);
                        const uint8x16_t code = vorrq_u8(lo, vshlq_n_u8(vandq_u8(h[half], three), 4));
                        h[half] = vshrq_n_u8(h[half], 2);
                        const int32x4_t p = vdotq_s32(zero, vreinterpretq_s8_u8(code),
                                                      vld1q_s8(yp + 32 * group + 16 * half));
                        isum = vmlaq_n_s32(isum, p, sp[2 * group + half]);
                    }
                }
            }
            sum += d * float(vaddvq_s32(isum) - 32 * vaddvq_s32(offset));
        }
    } else if (type == GGML_Q8_0) {
        for (int b = 0; b < blocks; ++b, y += 256) {
            float32x4_t acc = vdupq_n_f32(0.f);
            for (int k = 0; k < 8; ++k, row += 34) {
                int32x4_t p = vdotq_s32(zero, vld1q_s8(reinterpret_cast<const int8_t *>(row + 2)), vld1q_s8(y + 32 * k));
                p = vdotq_s32(p, vld1q_s8(reinterpret_cast<const int8_t *>(row + 18)), vld1q_s8(y + 32 * k + 16));
                acc = vfmaq_n_f32(acc, vcvtq_f32_s32(p), host_fp16(row));
            }
            sum += vaddvq_f32(acc) * x.d[b];
        }
    } else throw std::invalid_argument("unsupported int8 host dot format");
    return sum;
}
#endif
}
