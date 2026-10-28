#pragma once
// Arm i8mm (SMMLA) projections: two packed K-quant rows against two int8 activations per instruction. Each
// block is decoded once per row pair and reused for every activation pair, so a speculative verify batch or a
// prefill chunk costs little more than one token. Single tokens use the sdot kernel, whose arithmetic matches
// these lanes exactly, so a token's result never depends on what it was batched with.
#include "host_q8_dot.hpp"
#include <cstring>
#include <vector>

namespace tqllm {
// Two activations interleaved for SMMLA: for every 8 elements, 8 int8 values of the first then of the second.
struct HostQ8Pair {
    int n = 0;
    std::vector<int8_t> q;
    std::vector<float> d;        // per 256 elements: [first, second, first, second]
    std::vector<int32_t> sums;   // per 32 elements: [first, second, first, second]
};

inline bool host_mmla_supported(uint32_t type, int cols) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
    return cols % 256 == 0 && (type == GGML_Q4_K || type == GGML_Q5_K || type == GGML_Q6_K);
#else
    (void) type; (void) cols;
    return false;
#endif
}

// second may be null: its lanes are zero and ignored by the caller.
inline void host_pair_q8(const HostQ8 & first, const HostQ8 * second, HostQ8Pair & out) {
    const int n = first.n;
    out.n = n; out.q.resize(size_t(2) * n); out.d.resize(size_t(n) / 256 * 4); out.sums.resize(size_t(n) / 32 * 4);
    for (int c = 0; c < n / 8; ++c) {
        std::memcpy(out.q.data() + 16 * c, first.q.data() + 8 * c, 8);
        if (second) std::memcpy(out.q.data() + 16 * c + 8, second->q.data() + 8 * c, 8);
        else std::memset(out.q.data() + 16 * c + 8, 0, 8);
    }
    for (int b = 0; b < n / 256; ++b) {
        const float a = first.d[b], s = second ? second->d[b] : 0.f;
        float * d = out.d.data() + 4 * b;
        d[0] = a; d[1] = s; d[2] = a; d[3] = s;
    }
    for (int g = 0; g < n / 32; ++g) {
        const int a = first.sums[2 * g] + first.sums[2 * g + 1];
        const int s = second ? second->sums[2 * g] + second->sums[2 * g + 1] : 0;
        int32_t * sum = out.sums.data() + 4 * g;
        sum[0] = a; sum[1] = s; sum[2] = a; sum[3] = s;
    }
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
inline int32x4_t host_pair_lanes(int a, int b) {
    const int32_t v[4] = {a, a, b, b};
    return vld1q_s32(v);
}
inline float32x4_t host_pair_lanes(float a, float b) {
    const float v[4] = {a, a, b, b};
    return vld1q_f32(v);
}
// 32 codes of two rows (lo = elements 0..15, hi = 16..31) against 32 interleaved activation elements.
inline int32x4_t host_mmla32(int8x16_t lo0, int8x16_t hi0, int8x16_t lo1, int8x16_t hi1, const int8_t * y) {
    const int64x2_t l0 = vreinterpretq_s64_s8(lo0), l1 = vreinterpretq_s64_s8(lo1);
    const int64x2_t h0 = vreinterpretq_s64_s8(hi0), h1 = vreinterpretq_s64_s8(hi1);
    int32x4_t g = vmmlaq_s32(vdupq_n_s32(0), vreinterpretq_s8_s64(vzip1q_s64(l0, l1)), vld1q_s8(y));
    g = vmmlaq_s32(g, vreinterpretq_s8_s64(vzip2q_s64(l0, l1)), vld1q_s8(y + 16));
    g = vmmlaq_s32(g, vreinterpretq_s8_s64(vzip1q_s64(h0, h1)), vld1q_s8(y + 32));
    return vmmlaq_s32(g, vreinterpretq_s8_s64(vzip2q_s64(h0, h1)), vld1q_s8(y + 48));
}

// Rows r0 and r1 against P activation pairs. out[i] lanes: [r0 x first, r0 x second, r1 x first, r1 x second].
template <int P>
inline void host_mmla_rows(uint32_t type, const unsigned char * r0, const unsigned char * r1,
                           const HostQ8Pair * const * ps, float32x4_t * out) {
    const int blocks = ps[0]->n / 256;
    for (int i = 0; i < P; ++i) out[i] = vdupq_n_f32(0.f);
    if (type == GGML_Q4_K || type == GGML_Q5_K) {
        const bool five = type == GGML_Q5_K;
        const int bytes = five ? 176 : 144;
        const uint8x16_t low = vdupq_n_u8(15), bit = vdupq_n_u8(16);
        for (int b = 0; b < blocks; ++b, r0 += bytes, r1 += bytes) {
            const float32x4_t dr = host_pair_lanes(host_fp16(r0), host_fp16(r1));
            const float32x4_t mr = host_pair_lanes(host_fp16(r0 + 2), host_fp16(r1 + 2));
            uint8_t s0[8], m0[8], s1[8], m1[8];
            host_k4_scales(r0 + 4, s0, m0); host_k4_scales(r1 + 4, s1, m1);
            const unsigned char * q0 = r0 + (five ? 48 : 16), * q1 = r1 + (five ? 48 : 16);
            uint8x16_t h[4] = {};
            if (five) { h[0] = vld1q_u8(r0 + 16); h[1] = vld1q_u8(r0 + 32); h[2] = vld1q_u8(r1 + 16); h[3] = vld1q_u8(r1 + 32); }
            int32x4_t isum[P], msum[P];
            for (int i = 0; i < P; ++i) { isum[i] = vdupq_n_s32(0); msum[i] = vdupq_n_s32(0); }
            for (int j = 0; j < 4; ++j, q0 += 32, q1 += 32) {
                // c[row][group]: group 2j (low nibbles) and 2j+1 (high nibbles), 16 + 16 elements each.
                uint8x16_t c[2][4];
                const unsigned char * qs[2] = {q0, q1};
                for (int r = 0; r < 2; ++r) {
                    const uint8x16_t a = vld1q_u8(qs[r]), e = vld1q_u8(qs[r] + 16);
                    c[r][0] = vandq_u8(a, low); c[r][1] = vandq_u8(e, low);
                    c[r][2] = vshrq_n_u8(a, 4); c[r][3] = vshrq_n_u8(e, 4);
                    if (five) {
                        uint8x16_t & ha = h[2 * r], & he = h[2 * r + 1];
                        c[r][0] = vorrq_u8(c[r][0], vandq_u8(vshlq_n_u8(ha, 4), bit));
                        c[r][1] = vorrq_u8(c[r][1], vandq_u8(vshlq_n_u8(he, 4), bit));
                        c[r][2] = vorrq_u8(c[r][2], vandq_u8(vshlq_n_u8(ha, 3), bit));
                        c[r][3] = vorrq_u8(c[r][3], vandq_u8(vshlq_n_u8(he, 3), bit));
                        ha = vshrq_n_u8(ha, 2); he = vshrq_n_u8(he, 2);
                    }
                }
                for (int half = 0; half < 2; ++half) {
                    const int group = 2 * j + half;
                    const int32x4_t sv = host_pair_lanes(int(s0[group]), int(s1[group]));
                    const int32x4_t mv = host_pair_lanes(int(m0[group]), int(m1[group]));
                    const int8x16_t lo0 = vreinterpretq_s8_u8(c[0][2 * half]), hi0 = vreinterpretq_s8_u8(c[0][2 * half + 1]);
                    const int8x16_t lo1 = vreinterpretq_s8_u8(c[1][2 * half]), hi1 = vreinterpretq_s8_u8(c[1][2 * half + 1]);
                    for (int i = 0; i < P; ++i) {
                        const int8_t * y = ps[i]->q.data() + 2 * (256 * b + 32 * group);
                        isum[i] = vmlaq_s32(isum[i], host_mmla32(lo0, hi0, lo1, hi1, y), sv);
                        msum[i] = vmlaq_s32(msum[i], mv, vld1q_s32(ps[i]->sums.data() + 4 * (8 * b + group)));
                    }
                }
            }
            for (int i = 0; i < P; ++i) {
                const float32x4_t dx = vld1q_f32(ps[i]->d.data() + 4 * b);
                out[i] = vfmaq_f32(out[i], vcvtq_f32_s32(isum[i]), vmulq_f32(dr, dx));
                out[i] = vfmsq_f32(out[i], vcvtq_f32_s32(msum[i]), vmulq_f32(mr, dx));
            }
        }
    } else if (type == GGML_Q6_K) {
        const uint8x16_t low = vdupq_n_u8(15), high = vdupq_n_u8(48);
        const int8x16_t offset = vdupq_n_s8(32);
        for (int b = 0; b < blocks; ++b, r0 += 210, r1 += 210) {
            const float32x4_t dr = host_pair_lanes(host_fp16(r0 + 208), host_fp16(r1 + 208));
            const int8_t * sc0 = reinterpret_cast<const int8_t *>(r0 + 192), * sc1 = reinterpret_cast<const int8_t *>(r1 + 192);
            int32x4_t isum[P];
            for (int i = 0; i < P; ++i) isum[i] = vdupq_n_s32(0);
            for (int part = 0; part < 2; ++part) {
                // c[row][k]: 16 signed codes of sixteen-element group 8 * part + k.
                int8x16_t c[2][8];
                const unsigned char * rows[2] = {r0, r1};
                for (int r = 0; r < 2; ++r) {
                    const unsigned char * ql = rows[r] + 64 * part, * qh = rows[r] + 128 + 32 * part;
                    const uint8x16_t l[4] = {vld1q_u8(ql), vld1q_u8(ql + 16), vld1q_u8(ql + 32), vld1q_u8(ql + 48)};
                    const uint8x16_t hb[2] = {vld1q_u8(qh), vld1q_u8(qh + 16)};
                    for (int half = 0; half < 2; ++half) {
                        const uint8x16_t u[4] = {
                            vorrq_u8(vandq_u8(l[half], low), vandq_u8(vshlq_n_u8(hb[half], 4), high)),
                            vorrq_u8(vandq_u8(l[2 + half], low), vandq_u8(vshlq_n_u8(hb[half], 2), high)),
                            vorrq_u8(vshrq_n_u8(l[half], 4), vandq_u8(hb[half], high)),
                            vorrq_u8(vshrq_n_u8(l[2 + half], 4), vandq_u8(vshrq_n_u8(hb[half], 2), high))};
                        for (int group = 0; group < 4; ++group)
                            c[r][2 * group + half] = vsubq_s8(vreinterpretq_s8_u8(u[group]), offset);
                    }
                }
                for (int k = 0; k < 8; ++k) {
                    const int s = 8 * part + k;
                    const int32x4_t sv = host_pair_lanes(int(sc0[s]), int(sc1[s]));
                    const int64x2_t a0 = vreinterpretq_s64_s8(c[0][k]), a1 = vreinterpretq_s64_s8(c[1][k]);
                    const int8x16_t lo = vreinterpretq_s8_s64(vzip1q_s64(a0, a1)), hi = vreinterpretq_s8_s64(vzip2q_s64(a0, a1));
                    for (int i = 0; i < P; ++i) {
                        const int8_t * y = ps[i]->q.data() + 2 * (256 * b + 16 * s);
                        const int32x4_t g = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), lo, vld1q_s8(y)), hi, vld1q_s8(y + 16));
                        isum[i] = vmlaq_s32(isum[i], g, sv);
                    }
                }
            }
            for (int i = 0; i < P; ++i)
                out[i] = vfmaq_f32(out[i], vcvtq_f32_s32(isum[i]), vmulq_f32(dr, vld1q_f32(ps[i]->d.data() + 4 * b)));
        }
    }
}

// Rows r0, r1 against any number of pairs, four pairs at a time.
inline void host_mmla_batch(uint32_t type, const unsigned char * r0, const unsigned char * r1,
                            const HostQ8Pair * const * ps, int count, float32x4_t * out) {
    for (; count >= 4; count -= 4, ps += 4, out += 4) host_mmla_rows<4>(type, r0, r1, ps, out);
    if (count == 3) host_mmla_rows<3>(type, r0, r1, ps, out);
    else if (count == 2) host_mmla_rows<2>(type, r0, r1, ps, out);
    else if (count == 1) host_mmla_rows<1>(type, r0, r1, ps, out);
}
#endif
}
