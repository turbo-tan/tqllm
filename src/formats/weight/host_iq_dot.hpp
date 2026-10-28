#pragma once
// Integer dot products for the grid / lookup-table GGUF formats (IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS) against int8
// activations (Arm dotprod). Each 256-weight block is unpacked once into signed int8 values and one integer
// scale per 16 weights, then dotted with every activation of the batch. The float step is one fused
// multiply-add per block, so any batch size gives bit-identical results.
#include "gguf.hpp"
#include <cstdint>
#include <cmath>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tqllm::iq {
inline constexpr uint32_t grid3xxs[256] = {
#include "iq3xxs_grid.inc"
};
inline constexpr uint32_t grid3s[512] = {
#include "iq3s_grid.inc"
};
inline constexpr uint64_t grid2s[1024] = {
#include "iq2s_grid.inc"
};
inline constexpr uint8_t signs7[128] = {
#include "ksigns_iq2xs.inc"
};
inline constexpr int8_t kvalues4[16] = {
#include "kvalues_iq4nl.inc"
};
}

namespace tqllm {
inline bool host_iq_type(uint32_t type) {
    return type == GGML_IQ3_XXS || type == GGML_IQ3_S || type == GGML_IQ2_S || type == GGML_IQ4_XS;
}

#if defined(__aarch64__)
inline int host_iq_block_bytes(uint32_t type) {
    return type == GGML_IQ3_XXS ? 98 : type == GGML_IQ3_S ? 110 : type == GGML_IQ2_S ? 82 : 136;
}

// 0xFF in byte j when bit j of the sign byte is set; the IQ3_XXS table takes a 7-bit index plus even parity.
struct HostSignMasks {
    uint64_t byte[256] = {}, parity7[128] = {};
    constexpr HostSignMasks() {
        for (int s = 0; s < 256; ++s)
            for (int j = 0; j < 8; ++j) if (s >> j & 1) byte[s] |= uint64_t(0xff) << (8 * j);
        for (int i = 0; i < 128; ++i) {
            int bits = 0;
            for (int j = 0; j < 7; ++j) bits += i >> j & 1;
            parity7[i] = byte[i | (bits & 1) << 7];
        }
    }
};
inline constexpr HostSignMasks host_sign_masks{};

// Magnitudes (unsigned bytes) negated where mask bytes are 0xFF: (m ^ mask) - mask.
inline int8x16_t host_apply_signs(uint8x16_t m, uint64_t lo, uint64_t hi) {
    const uint8x16_t mask = vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)));
    return vreinterpretq_s8_u8(vsubq_u8(veorq_u8(m, mask), mask));
}
inline uint8x16_t host_grid4(const uint32_t * grid, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    const uint32_t v[4] = {grid[a], grid[b], grid[c], grid[d]};
    return vreinterpretq_u8_u32(vld1q_u32(v));
}

// Rows of grid / table formats against N int8 activations (X has q, d; 256-element blocks). Per 32 weights the
// signed values are built in registers and dotted with every activation; one fused multiply-add per block.
template <int N, class X>
inline void host_iq_dot_n(uint32_t type, const unsigned char * row, const X * const * xs, float * out) {
    const int blocks = xs[0]->n / 256, bytes = type == GGML_IQ3_XXS ? 98 : type == GGML_IQ3_S ? 110 : type == GGML_IQ2_S ? 82 : 136;
    const int32x4_t zero = vdupq_n_s32(0);
    const auto & masks = host_sign_masks;
    const int8x16_t table = vld1q_s8(iq::kvalues4);
    float sum[N];
    for (int i = 0; i < N; ++i) sum[i] = 0.f;
    for (int b = 0; b < blocks; ++b, row += bytes) {
        __fp16 h; std::memcpy(&h, row, 2);
        const float d = float(h);
        float factor = d;
        int32x4_t isum[N];
        for (int i = 0; i < N; ++i) isum[i] = zero;
        for (int ib = 0; ib < 8; ++ib) {
            int8x16_t v0, v1;
            int s0, s1;
            if (type == GGML_IQ3_XXS) {
                const uint8_t * qs = row + 2 + 8 * ib;
                uint32_t a; std::memcpy(&a, row + 66 + 4 * ib, 4);
                s0 = s1 = int(2 * (a >> 28) + 1);  // d * (0.5 + s) / 2 = d / 4 * (2s + 1)
                factor = d * 0.25f;
                v0 = host_apply_signs(host_grid4(iq::grid3xxs, qs[0], qs[1], qs[2], qs[3]),
                                      masks.parity7[a & 127], masks.parity7[a >> 7 & 127]);
                v1 = host_apply_signs(host_grid4(iq::grid3xxs, qs[4], qs[5], qs[6], qs[7]),
                                      masks.parity7[a >> 14 & 127], masks.parity7[a >> 21 & 127]);
            } else if (type == GGML_IQ3_S) {
                const uint8_t * qs = row + 2 + 8 * ib, * signs = row + 74 + 4 * ib, * sc = row + 106;
                const uint32_t qh = uint32_t(row[66 + ib]) << 8;
                s0 = s1 = 1 + 2 * ((sc[ib / 2] >> 4 * (ib % 2)) & 0xf);
                v0 = host_apply_signs(host_grid4(iq::grid3s, qs[0] | (qh & 256), qs[1] | (qh >> 1 & 256),
                                                 qs[2] | (qh >> 2 & 256), qs[3] | (qh >> 3 & 256)),
                                      masks.byte[signs[0]], masks.byte[signs[1]]);
                v1 = host_apply_signs(host_grid4(iq::grid3s, qs[4] | (qh >> 4 & 256), qs[5] | (qh >> 5 & 256),
                                                 qs[6] | (qh >> 6 & 256), qs[7] | (qh >> 7 & 256)),
                                      masks.byte[signs[2]], masks.byte[signs[3]]);
            } else if (type == GGML_IQ2_S) {
                const uint8_t * qs = row + 2 + 4 * ib, * signs = row + 34 + 4 * ib, sc = row[74 + ib];
                const uint32_t qh = row[66 + ib];
                s0 = 2 * (sc & 0xf) + 1; s1 = 2 * (sc >> 4) + 1;  // d * (0.5 + s) / 4 = d / 8 * (2s + 1)
                factor = d * 0.125f;
                const uint64_t g[4] = {iq::grid2s[qs[0] | (qh << 8 & 0x300)], iq::grid2s[qs[1] | (qh << 6 & 0x300)],
                                       iq::grid2s[qs[2] | (qh << 4 & 0x300)], iq::grid2s[qs[3] | (qh << 2 & 0x300)]};
                v0 = host_apply_signs(vreinterpretq_u8_u64(vld1q_u64(g)), masks.byte[signs[0]], masks.byte[signs[1]]);
                v1 = host_apply_signs(vreinterpretq_u8_u64(vld1q_u64(g + 2)), masks.byte[signs[2]], masks.byte[signs[3]]);
            } else {  // IQ4_XS: d, scales_h, scales_l[4], qs[128]
                uint16_t sh; std::memcpy(&sh, row + 2, 2);
                s0 = s1 = (((row[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4)) - 32;
                const uint8x16_t q = vld1q_u8(row + 8 + 16 * ib);
                v0 = vqtbl1q_s8(table, vandq_u8(q, vdupq_n_u8(15)));
                v1 = vqtbl1q_s8(table, vshrq_n_u8(q, 4));
            }
            for (int i = 0; i < N; ++i) {
                const int8_t * y = xs[i]->q.data() + 256 * b + 32 * ib;
                isum[i] = vmlaq_n_s32(isum[i], vdotq_s32(zero, v0, vld1q_s8(y)), s0);
                isum[i] = vmlaq_n_s32(isum[i], vdotq_s32(zero, v1, vld1q_s8(y + 16)), s1);
            }
        }
        for (int i = 0; i < N; ++i) sum[i] = std::fma(float(vaddvq_s32(isum[i])), factor * xs[i]->d[b], sum[i]);
    }
    for (int i = 0; i < N; ++i) out[i] = sum[i];
}
#endif
}
