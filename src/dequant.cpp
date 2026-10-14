#include "dequant.hpp"
#include "gguf.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace tqllm {
float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {  // subnormal
            int e = -1;
            uint32_t m = mant;
            do {
                ++e;
                m <<= 1;
            } while (!(m & 0x400));
            bits = sign | uint32_t(127 - 15 - e) << 23 | (m & 0x3FF) << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | mant << 13;
    } else {
        bits = sign | (exp + 127 - 15) << 23 | mant << 13;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void q3k_scales(const uint8_t * p, int8_t * sc) {  // 12 bytes -> 16 signed scales (6-bit - 32), as ggml
    uint32_t a[4];
    std::memcpy(a, p, 12);
    const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu, t = a[2];
    a[2] = ((a[0] >> 4) & km2) | (((t >> 4) & km1) << 4);
    a[3] = ((a[1] >> 4) & km2) | (((t >> 6) & km1) << 4);
    a[0] = (a[0] & km2) | (((t >> 0) & km1) << 4);
    a[1] = (a[1] & km2) | (((t >> 2) & km1) << 4);
    std::memcpy(sc, a, 16);
    for (int i = 0; i < 16; ++i) sc[i] = int8_t(sc[i] - 32);
}

namespace {

void q4k(const uint8_t * b, float * y) {  // 144 bytes: d, dmin, scales[12], qs[128] -> 256 values
    uint16_t hd, hm;
    std::memcpy(&hd, b, 2);
    std::memcpy(&hm, b + 2, 2);
    const float d = half_to_float(hd), mn = half_to_float(hm);
    const uint8_t * sc = b + 4;
    const uint8_t * q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is, sc, s, m);
        const float d1 = d * s, m1 = mn * m;
        scale_min_k4(is + 1, sc, s, m);
        const float d2 = d * s, m2 = mn * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

// Q5_K 176 bytes: d, dmin, scales[12], qh[32], qs[128]; as Q4_K plus a high bit qh[l] >> sub-block
void q5k(const uint8_t * b, float * y) {
    uint16_t hd, hm;
    std::memcpy(&hd, b, 2);
    std::memcpy(&hm, b + 2, 2);
    const float d = half_to_float(hd), mn = half_to_float(hm);
    const uint8_t *sc = b + 4, *qh = b + 16, *q = b + 48;
    for (int j = 0; j < 8; ++j) {
        uint8_t s, m;
        scale_min_k4(j, sc, s, m);
        for (int l = 0; l < 32; ++l) {
            const int v = (j % 2 ? q[32 * (j / 2) + l] >> 4 : q[32 * (j / 2) + l] & 0xF) + ((qh[l] >> j) & 1) * 16;
            *y++ = d * s * v - mn * m;
        }
    }
}

// Q3_K 110 bytes: hmask[32], qs[64], scales[12] (16 x 6-bit), d. Element e: n = e / 128, j = (e % 128) / 32,
// w = e % 32; code = ((qs[32n + w] >> 2j) & 3) - (hmask[w] bit 4n + j ? 0 : 4); scale index 8n + 2j + (w >= 16).
void q3k(const uint8_t * b, float * y) {
    int8_t sc[16];
    q3k_scales(b + 96, sc);
    uint16_t hd;
    std::memcpy(&hd, b + 108, 2);
    const float d = half_to_float(hd);
    for (int e = 0; e < 256; ++e) {
        const int n = e / 128, j = (e % 128) / 32, w = e % 32;
        const int q = ((b[32 + 32 * n + w] >> (2 * j)) & 3) - ((b[w] >> (4 * n + j)) & 1 ? 0 : 4);
        y[e] = d * float(sc[8 * n + 2 * j + (w >= 16)]) * float(q);
    }
}

// IQ3_S 110 bytes: d, qs[64], qh[8], signs[32], scales[4] (ggml dequantize_row_iq3_s).
const uint32_t iq3s_grid[512] = {
#include "../kernels/iq3s_grid.inc"
};
void iq3s(const uint8_t * b, float * y) {
    uint16_t hd;
    std::memcpy(&hd, b, 2);
    const float d = half_to_float(hd);
    const uint8_t * qs = b + 2, * qh = b + 66, * signs = b + 74, * sc = b + 106;
    for (int ib = 0; ib < 8; ++ib) {
        const float db = d * float(1 + 2 * ((sc[ib / 2] >> 4 * (ib % 2)) & 0xf));
        for (int w = 0; w < 8; ++w, y += 4) {
            const uint32_t g = iq3s_grid[qs[ib * 8 + w] | (((qh[ib] >> w) & 1) << 8)];
            const uint8_t sg = signs[ib * 4 + w / 2] >> (4 * (w % 2));
            for (int j = 0; j < 4; ++j) y[j] = db * float((g >> (8 * j)) & 0xff) * ((sg >> j) & 1 ? -1.f : 1.f);
        }
    }
}

// IQ3_XXS 98 bytes: d, qs[64] grid indices, then per 32 weights a uint32: 4 x 7-bit sign index (8th sign = parity),
// 4-bit scale on top (ggml dequantize_row_iq3_xxs).
const uint32_t iq3xxs_grid[256] = {
#include "../kernels/iq3xxs_grid.inc"
};
void iq3xxs(const uint8_t * b, float * y) {
    uint16_t hd;
    std::memcpy(&hd, b, 2);
    const float d = half_to_float(hd);
    const uint8_t * qs = b + 2;
    for (int ib = 0; ib < 8; ++ib) {
        uint32_t a;
        std::memcpy(&a, b + 66 + 4 * ib, 4);
        const float db = d * (0.5f + float(a >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint32_t i7 = (a >> 7 * l) & 127, sg = i7 | ((__builtin_popcount(i7) & 1) << 7);
            for (int h = 0; h < 2; ++h, y += 4) {
                const uint32_t g = iq3xxs_grid[qs[ib * 8 + 2 * l + h]];
                for (int j = 0; j < 4; ++j) y[j] = db * float((g >> (8 * j)) & 0xff) * ((sg >> (4 * h + j)) & 1 ? -1.f : 1.f);
            }
        }
    }
}

// IQ4_XS 136 bytes: d, scales_h, scales_l[4], qs[128] (ggml dequantize_row_iq4_xs).
void iq4xs(const uint8_t * b, float * y) {
    static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    uint16_t hd, sh;
    std::memcpy(&hd, b, 2);
    std::memcpy(&sh, b + 2, 2);
    const float d = half_to_float(hd);
    const uint8_t * sl = b + 4, * qs = b + 8;
    for (int ib = 0; ib < 8; ++ib, y += 32, qs += 16) {
        const int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
        const float dl = d * float(ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j] = dl * kv[qs[j] & 0xf];
            y[j + 16] = dl * kv[qs[j] >> 4];
        }
    }
}

void q6k(const uint8_t * b, float * y) {  // 210 bytes: ql[128], qh[64], scales[16], d
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t * sc = reinterpret_cast<const int8_t *>(b + 192);
    uint16_t hd;
    std::memcpy(&hd, b + 208, 2);
    const float d = half_to_float(hd);
    for (int n = 0; n < 2; ++n, y += 128, ql += 64, qh += 32, sc += 8)
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            y[l] = d * sc[is] * float(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            y[l + 32] = d * sc[is + 2] * float(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            y[l + 64] = d * sc[is + 4] * float(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            y[l + 96] = d * sc[is + 6] * float(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
}


// ---- 2-bit IQ family, Q2_K, IQ1_M, IQ4_NL: ports of ggml's dequantize_row_* (llama.cpp, MIT licence); tables in kernels/*.inc ----
const uint64_t iq2xxs_grid[256] = {
#include "../kernels/iq2xxs_grid.inc"
};
const uint64_t iq2xs_grid[512] = {
#include "../kernels/iq2xs_grid.inc"
};
const uint64_t iq2s_grid[1024] = {
#include "../kernels/iq2s_grid.inc"
};
const uint64_t iq1s_grid[2048] = {
#include "../kernels/iq1s_grid.inc"
};
const uint8_t ksigns_iq2xs[128] = {
#include "../kernels/ksigns_iq2xs.inc"
};
const int8_t kvalues_iq4nl[16] = {
#include "../kernels/kvalues_iq4nl.inc"
};
const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

// IQ2_XXS 66 bytes: d, 32 x uint16 (per 32 weights: 4 grid-index bytes, then 4 x 7-bit sign index + 4-bit scale in a uint32).
void iq2xxs(const uint8_t * b, float * y) {
    uint16_t hd;
    std::memcpy(&hd, b, 2);
    const float d = half_to_float(hd);
    for (int ib = 0; ib < 8; ++ib) {
        uint32_t a[2];
        std::memcpy(a, b + 2 + 8 * ib, 8);
        const uint8_t * a8 = reinterpret_cast<const uint8_t *>(a);
        const float db = d * (0.5f + float(a[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint64_t g = iq2xxs_grid[a8[l]];
            const uint8_t sg = ksigns_iq2xs[(a[1] >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j) y[j] = db * float((g >> (8 * j)) & 0xff) * ((sg & kmask_iq2xs[j]) ? -1.f : 1.f);
            y += 8;
        }
    }
}

// IQ2_XS 74 bytes: d, 32 x uint16 (9-bit grid index, 7-bit sign index), 8 scale bytes (two 4-bit scales per 32 weights).
void iq2xs(const uint8_t * b, float * y) {
    uint16_t hd;
    std::memcpy(&hd, b, 2);
    const float d = half_to_float(hd);
    const uint8_t * sc = b + 66;
    for (int ib = 0; ib < 8; ++ib) {
        const float db[2] = {d * (0.5f + float(sc[ib] & 0xf)) * 0.25f, d * (0.5f + float(sc[ib] >> 4)) * 0.25f};
        for (int l = 0; l < 4; ++l) {
            uint16_t q;
            std::memcpy(&q, b + 2 + 2 * (4 * ib + l), 2);
            const uint64_t g = iq2xs_grid[q & 511];
            const uint8_t sg = ksigns_iq2xs[q >> 9];
            for (int j = 0; j < 8; ++j) y[j] = db[l / 2] * float((g >> (8 * j)) & 0xff) * ((sg & kmask_iq2xs[j]) ? -1.f : 1.f);
            y += 8;
        }
    }
}

// IQ2_S 82 bytes: d, qs[64] (32 grid-index low bytes, then 32 sign bytes), qh[8], scales[8].
void iq2s(const uint8_t * b, float * y) {
    uint16_t hd;
    std::memcpy(&hd, b, 2);
    const float d = half_to_float(hd);
    const uint8_t * qs = b + 2, * qh = b + 66, * sc = b + 74;
    const uint8_t * signs = qs + 32;
    for (int ib = 0; ib < 8; ++ib) {
        const float db[2] = {d * (0.5f + float(sc[ib] & 0xf)) * 0.25f, d * (0.5f + float(sc[ib] >> 4)) * 0.25f};
        for (int l = 0; l < 4; ++l) {
            const uint64_t g = iq2s_grid[qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300)];
            for (int j = 0; j < 8; ++j) y[j] = db[l / 2] * float((g >> (8 * j)) & 0xff) * ((signs[l] & kmask_iq2xs[j]) ? -1.f : 1.f);
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
}

// Q2_K 84 bytes: scales[16] (4-bit scale, 4-bit min), qs[64], d, dmin.
void q2k(const uint8_t * b, float * y) {
    uint16_t hd, hm;
    std::memcpy(&hd, b + 80, 2);
    std::memcpy(&hm, b + 82, 2);
    const float d = half_to_float(hd), mn = half_to_float(hm);
    const uint8_t * q = b + 16;
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                const uint8_t sc = b[is++];
                const float dl = d * float(sc & 0xF), ml = mn * float(sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * float((q[l + 16 * half] >> shift) & 3) - ml;
            }
            shift += 2;
        }
        q += 32;
    }
}

// IQ1_S 50 bytes: d, qs[32], qh[8] x uint16 (3 index bits per group of 8, 3-bit scale, delta sign).
void iq1s(const uint8_t * b, float * y) {
    const float d = half_to_float(uint16_t(b[0] | (b[1] << 8)));
    const uint8_t * qs = b + 2;
    for (int ib = 0; ib < 8; ++ib) {
        const uint16_t qh = uint16_t(b[34 + 2 * ib] | (b[35 + 2 * ib] << 8));
        const float dl = d * float(2 * ((qh >> 12) & 7) + 1), delta = (qh & 0x8000) ? -0.125f : 0.125f;
        for (int l = 0; l < 4; ++l) {
            const uint64_t g = iq1s_grid[qs[l] | (((qh >> (3 * l)) & 7) << 8)];
            for (int j = 0; j < 8; ++j) y[j] = dl * (float(int8_t((g >> (8 * j)) & 0xff)) + delta);
            y += 8;
        }
        qs += 4;
    }
}
// IQ1_M 56 bytes: qs[32], qh[16], scales[8] (FP16 d scattered over the four top nibbles + 3-bit sub-block scales).
void iq1m(const uint8_t * b, float * y) {
    uint16_t sc[4];
    std::memcpy(sc, b + 48, 8);
    const uint16_t u = uint16_t((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const float d = half_to_float(u);
    const uint8_t * qs = b, * qh = b + 32;
    for (int ib = 0; ib < 8; ++ib) {
        const float dl1 = d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        const float dl2 = d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
        const uint16_t idx[4] = {uint16_t(qs[0] | ((qh[0] << 8) & 0x700)), uint16_t(qs[1] | ((qh[0] << 4) & 0x700)),
                                 uint16_t(qs[2] | ((qh[1] << 8) & 0x700)), uint16_t(qs[3] | ((qh[1] << 4) & 0x700))};
        const float delta[4] = {(qh[0] & 0x08) ? -0.125f : 0.125f, (qh[0] & 0x80) ? -0.125f : 0.125f,
                                (qh[1] & 0x08) ? -0.125f : 0.125f, (qh[1] & 0x80) ? -0.125f : 0.125f};
        for (int l = 0; l < 4; ++l) {
            const float dl = l < 2 ? dl1 : dl2;
            const uint64_t g = iq1s_grid[idx[l]];
            for (int j = 0; j < 8; ++j) y[j] = dl * (float(int8_t((g >> (8 * j)) & 0xff)) + delta[l]);
            y += 8;
        }
        qs += 4;
        qh += 2;
    }
}

// IQ4_NL 18 bytes per 32 weights: d, 16 bytes of two 4-bit indices into a non-linear table.
void iq4nl(const uint8_t * b, float * y) {
    uint16_t hd;
    std::memcpy(&hd, b, 2);
    const float d = half_to_float(hd);
    for (int j = 0; j < 16; ++j) {
        y[j] = d * float(kvalues_iq4nl[b[2 + j] & 0xf]);
        y[j + 16] = d * float(kvalues_iq4nl[b[2 + j] >> 4]);
    }
}

} // namespace

void dequantize_host(uint32_t type, const void * src, float * dst, int64_t n) {
    const auto * s = static_cast<const uint8_t *>(src);
    switch (type) {
        case GGML_GSQ_Q2_0:
            if (n < 0 || n % 64) break;
            for (int64_t b = 0; b < n / 64; ++b) {
                uint16_t scale;
                std::memcpy(&scale, s + b * 18, 2);
                const float d = half_to_float(scale);
                for (int j = 0; j < 64; ++j)
                    dst[b * 64 + j] = (int((s[b * 18 + 2 + j / 4] >> (2 * (j % 4))) & 3) - 1) * d;
            }
            return;
        case 0: std::memcpy(dst, src, size_t(n) * 4); return;
        case 1:
            for (int64_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, s + 2 * i, 2);
                dst[i] = half_to_float(h);
            }
            return;
        case 30:
            for (int64_t i = 0; i < n; ++i) {
                uint32_t bits = uint32_t(s[2 * i] | (s[2 * i + 1] << 8)) << 16;
                std::memcpy(&dst[i], &bits, 4);
            }
            return;
        case 10:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) q2k(s + b * 84, dst + b * 256);
            return;
        case 16:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) iq2xxs(s + b * 66, dst + b * 256);
            return;
        case 17:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) iq2xs(s + b * 74, dst + b * 256);
            return;
        case 20:
            if (n % 32) break;
            for (int64_t b = 0; b < n / 32; ++b) iq4nl(s + b * 18, dst + b * 32);
            return;
        case 8:  // Q8_0: d, 32 int8
            if (n % 32) break;
            for (int64_t b = 0; b < n / 32; ++b) {
                const uint8_t * q = s + b * 34;
                const float d = half_to_float(uint16_t(q[0] | (q[1] << 8)));
                for (int j = 0; j < 32; ++j) dst[b * 32 + j] = d * float(int8_t(q[2 + j]));
            }
            return;
        case 19:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) iq1s(s + b * 50, dst + b * 256);
            return;
        case 22:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) iq2s(s + b * 82, dst + b * 256);
            return;
        case 29:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) iq1m(s + b * 56, dst + b * 256);
            return;
        case 18:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) iq3xxs(s + b * 98, dst + b * 256);
            return;
        case 21:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) iq3s(s + b * 110, dst + b * 256);
            return;
        case 23:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) iq4xs(s + b * 136, dst + b * 256);
            return;
        case 12:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) q4k(s + b * 144, dst + b * 256);
            return;
        case 11:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) q3k(s + b * 110, dst + b * 256);
            return;
        case 13:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) q5k(s + b * 176, dst + b * 256);
            return;
        case 14:
            if (n % 256) break;
            for (int64_t b = 0; b < n / 256; ++b) q6k(s + b * 210, dst + b * 256);
            return;
    }
    throw std::runtime_error("dequantize_host: unsupported type " + std::to_string(type));
}

} // namespace tqllm
