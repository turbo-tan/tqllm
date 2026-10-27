#pragma once
#include "dequant.hpp"
#include "gguf.hpp"
#include <cstring>
#include <stdexcept>
#if defined(__aarch64__)
#include <arm_neon.h>

namespace tqllm {
inline float32x4_t host_codes16(float32x4_t acc, uint8x16_t code, const float * x,
                              float scale, float minimum, bool signed_codes = false) {
    const int16x8_t a = signed_codes ? vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(code)))
                                   : vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(code)));
    const int16x8_t b = signed_codes ? vmovl_s8(vget_high_s8(vreinterpretq_s8_u8(code)))
                                   : vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(code)));
    const int32x4_t groups[4] = {vmovl_s16(vget_low_s16(a)), vmovl_s16(vget_high_s16(a)),
                                vmovl_s16(vget_low_s16(b)), vmovl_s16(vget_high_s16(b))};
    for (int j = 0; j < 4; ++j) {
        const auto weight = vsubq_f32(vmulq_n_f32(vcvtq_f32_s32(groups[j]), scale), vdupq_n_f32(minimum));
        acc = vfmaq_f32(acc, weight, vld1q_f32(x + 4 * j));
    }
    return acc;
}

// Keep the four-lane dot order of host_dot while decoding packed blocks in registers.
inline float host_packed_dot(uint32_t type, const unsigned char * row, const float * x, int cols) {
    float32x4_t acc = vdupq_n_f32(0.f);
    if (type == GGML_Q4_K || type == GGML_Q5_K) {
        const int bytes = type == GGML_Q4_K ? 144 : 176;
        for (int k = 0; k < cols; k += 256, row += bytes) {
            uint16_t hd, hm;
            std::memcpy(&hd, row, 2); std::memcpy(&hm, row + 2, 2);
            const float d = half_to_float(hd), mn = half_to_float(hm);
            const auto * qs = row + (type == GGML_Q4_K ? 16 : 48);
            for (int group = 0; group < 8; ++group) {
                uint8_t scale, minimum;
                scale_min_k4(group, row + 4, scale, minimum);
                for (int half = 0; half < 2; ++half) {
                    auto code = vld1q_u8(qs + 32 * (group / 2) + 16 * half);
                    code = group % 2 ? vshrq_n_u8(code, 4) : vandq_u8(code, vdupq_n_u8(15));
                    if (type == GGML_Q5_K) {
                        auto hi = vshlq_u8(vld1q_u8(row + 16 + 16 * half), vdupq_n_s8(int8_t(-group)));
                        code = vorrq_u8(code, vshlq_n_u8(vandq_u8(hi, vdupq_n_u8(1)), 4));
                    }
                    acc = host_codes16(acc, code, x + k + 32 * group + 16 * half, d * scale, mn * minimum);
                }
            }
        }
    } else if (type == GGML_Q6_K) {
        for (int k = 0; k < cols; k += 256, row += 210) {
            uint16_t hd; std::memcpy(&hd, row + 208, 2);
            const float d = half_to_float(hd);
            const auto * scales = reinterpret_cast<const int8_t *>(row + 192);
            for (int part = 0; part < 2; ++part) {
                for (int group = 0; group < 4; ++group) {
                    for (int half = 0; half < 2; ++half) {
                        auto lo = vld1q_u8(row + part * 64 + (group % 2) * 32 + half * 16);
                        lo = group >= 2 ? vshrq_n_u8(lo, 4) : vandq_u8(lo, vdupq_n_u8(15));
                        auto hi = vshlq_u8(vld1q_u8(row + 128 + part * 32 + half * 16),
                                          vdupq_n_s8(int8_t(-2 * group)));
                        auto code = vorrq_u8(lo, vshlq_n_u8(vandq_u8(hi, vdupq_n_u8(3)), 4));
                        code = vsubq_u8(code, vdupq_n_u8(32));
                        acc = host_codes16(acc, code, x + k + part * 128 + group * 32 + half * 16,
                                           d * scales[part * 8 + group * 2 + half], 0.f, true);
                    }
                }
            }
        }
    } else if (type == GGML_Q8_0) {
        for (int k = 0; k < cols; k += 32, row += 34) {
            uint16_t hd; std::memcpy(&hd, row, 2);
            const float d = half_to_float(hd);
            acc = host_codes16(acc, vld1q_u8(row + 2), x + k, d, 0.f, true);
            acc = host_codes16(acc, vld1q_u8(row + 18), x + k + 16, d, 0.f, true);
        }
    } else throw std::invalid_argument("unsupported packed host dot format");
    return vaddvq_f32(acc);
}
}
#endif
