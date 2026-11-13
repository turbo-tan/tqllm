// CPU K-quant expert matvec (see cpu_kquant.hpp). Block layouts follow src/formats/weight/dequant.cpp (q2k / q3k / q4k).
#include "cpu_kquant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__x86_64__)  // GCC / Clang: target attributes and __builtin_cpu_supports (MSVC takes the scalar path, as cpu_expert_matvec.cpp)
#include <immintrin.h>
#define TQ_KQ_X86 1
#endif

namespace tqllm { namespace cpu {
namespace {
constexpr uint32_t kQ2K = 10, kQ3K = 11, kQ4K = 12;

float h2f(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s;
        else {  // subnormal
            int ee = -1;
            uint32_t mm = m;
            do { ++ee; mm <<= 1; } while (!(mm & 0x400));
            f = s | (uint32_t(127 - 15 - ee) << 23) | ((mm & 0x3FF) << 13);
        }
    } else if (e == 31) f = s | 0x7F800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}
uint16_t rd16(const uint8_t * p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

void scale_min_k4(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
void q3k_scales(const uint8_t * p, int8_t * sc) {
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

// one 256-value block of each format against x block b (q: 256 int8, s16: 16 sums), scalar
float q2k_block(const uint8_t * b, const int8_t * q, const int16_t * s16) {
    const uint8_t * qs = b + 16;
    int32_t isum = 0, msum = 0;
    for (int n = 0; n < 2; ++n)
        for (int j = 0; j < 4; ++j)
            for (int half = 0; half < 2; ++half) {
                const int g = n * 8 + j * 2 + half;
                int32_t dot = 0;
                for (int l = 0; l < 16; ++l) dot += ((qs[n * 32 + half * 16 + l] >> (2 * j)) & 3) * q[n * 128 + j * 32 + half * 16 + l];
                isum += (b[g] & 0xF) * dot;
                msum += (b[g] >> 4) * s16[g];
            }
    return h2f(rd16(b + 80)) * float(isum) - h2f(rd16(b + 82)) * float(msum);
}
float q3k_block(const uint8_t * b, const int8_t * q, const int16_t * s16) {
    int8_t sc[16];
    q3k_scales(b + 96, sc);
    int32_t isum = 0;
    for (int n = 0; n < 2; ++n)
        for (int j = 0; j < 4; ++j)
            for (int half = 0; half < 2; ++half) {
                const int g = 8 * n + 2 * j + half;
                int32_t dot = 0;
                for (int l = 0; l < 16; ++l) {
                    const int w = half * 16 + l;
                    const int u = ((b[32 + 32 * n + w] >> (2 * j)) & 3) | (((b[w] >> (4 * n + j)) & 1) << 2);  // code + 4
                    dot += u * q[n * 128 + j * 32 + w];
                }
                isum += sc[g] * (dot - 4 * s16[g]);
            }
    return h2f(rd16(b + 108)) * float(isum);
}
float q4k_block(const uint8_t * b, const int8_t * q, const int16_t * s16) {
    const uint8_t * sc = b + 4, * qs = b + 16;
    int32_t isum = 0, msum = 0;
    for (int jj = 0; jj < 4; ++jj) {
        int s1, m1, s2, m2;
        scale_min_k4(2 * jj, sc, s1, m1);
        scale_min_k4(2 * jj + 1, sc, s2, m2);
        int32_t d1 = 0, d2 = 0;
        for (int l = 0; l < 32; ++l) {
            d1 += (qs[32 * jj + l] & 0xF) * q[64 * jj + l];
            d2 += (qs[32 * jj + l] >> 4) * q[64 * jj + 32 + l];
        }
        isum += s1 * d1 + s2 * d2;
        msum += m1 * (s16[4 * jj] + s16[4 * jj + 1]) + m2 * (s16[4 * jj + 2] + s16[4 * jj + 3]);
    }
    return h2f(rd16(b)) * float(isum) - h2f(rd16(b + 2)) * float(msum);
}

#ifdef TQ_KQ_X86
__attribute__((target("avx2,fma,f16c"))) inline int32_t hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
// int16 lanes 0-7 = a, 8-15 = b (the two 16-value halves of a 32-byte vector)
__attribute__((target("avx2,fma,f16c"))) inline __m256i halves(int a, int b) {
    return _mm256_set_m128i(_mm_set1_epi16(short(b)), _mm_set1_epi16(short(a)));
}
__attribute__((target("avx2,fma,f16c"))) inline float f16(const uint8_t * p) { return _cvtsh_ss(rd16(p)); }
__attribute__((target("avx2,fma,f16c"))) inline float q2k_block_avx2(const uint8_t * b, const int8_t * q, const int16_t * s16) {
    const __m256i m3 = _mm256_set1_epi8(3);
    __m256i acc = _mm256_setzero_si256();
    for (int n = 0; n < 2; ++n) {
        const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + 16 + 32 * n));
        for (int j = 0; j < 4; ++j) {
            const __m256i codes = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m3);
            const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + n * 128 + j * 32));
            const int g = n * 8 + j * 2;
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(codes, xq), halves(b[g] & 0xF, b[g + 1] & 0xF)));
        }
    }
    // mins: 16 x (4-bit min) * sum16, as one madd
    const __m128i sc8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b));
    const __m256i mins = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sc8, 4), _mm_set1_epi8(0xF)));
    const int32_t msum = hsum_epi32(_mm256_madd_epi16(mins, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s16))));
    return f16(b + 80) * float(hsum_epi32(acc)) - f16(b + 82) * float(msum);
}
__attribute__((target("avx2,fma,f16c"))) inline float q3k_block_avx2(const uint8_t * b, const int8_t * q, const int16_t * s16) {
    int8_t sc[16];
    q3k_scales(b + 96, sc);
    const __m256i m3 = _mm256_set1_epi8(3), m1 = _mm256_set1_epi8(1);
    const __m256i hm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b));
    __m256i acc = _mm256_setzero_si256();
    for (int n = 0; n < 2; ++n) {
        const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + 32 + 32 * n));
        for (int j = 0; j < 4; ++j) {
            const __m256i lo = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m3);
            const __m256i hb = _mm256_and_si256(_mm256_srli_epi16(hm, 4 * n + j), m1);
            const __m256i u = _mm256_or_si256(lo, _mm256_slli_epi16(hb, 2));  // code + 4, 0..7
            const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + n * 128 + j * 32));
            const int g = 8 * n + 2 * j;
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, xq), halves(sc[g], sc[g + 1])));
        }
    }
    const __m256i scv = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(sc)));
    const int32_t corr = hsum_epi32(_mm256_madd_epi16(scv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s16))));
    return f16(b + 108) * float(hsum_epi32(acc) - 4 * corr);
}
__attribute__((target("avx2,fma,f16c"))) inline float q4k_block_avx2(const uint8_t * b, const int8_t * q, const int16_t * s16) {
    const uint8_t * sc = b + 4;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256i acc = _mm256_setzero_si256();
    int32_t msum = 0;
    for (int jj = 0; jj < 4; ++jj) {
        int s1, mn1, s2, mn2;
        scale_min_k4(2 * jj, sc, s1, mn1);
        scale_min_k4(2 * jj + 1, sc, s2, mn2);
        const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + 16 + 32 * jj));
        const __m256i lo = _mm256_and_si256(qv, m4), hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), m4);
        const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + 64 * jj));
        const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + 64 * jj + 32));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), _mm256_set1_epi16(short(s1))));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), _mm256_set1_epi16(short(s2))));
        msum += mn1 * (s16[4 * jj] + s16[4 * jj + 1]) + mn2 * (s16[4 * jj + 2] + s16[4 * jj + 3]);
    }
    return f16(b) * float(hsum_epi32(acc)) - f16(b + 2) * float(msum);
}
// rows [r0, r1) with the block inlined (no call per block)
#define TQ_KQ_ROWS(NAME, BLOCK, BB)                                                                                     \
    __attribute__((target("avx2,fma,f16c"))) void NAME(const uint8_t * W, int cols, const Xq & x, int r0, int r1, float * y) { \
        const int nb = cols / 256;                                                                                      \
        for (int r = r0; r < r1; ++r) {                                                                                 \
            const uint8_t * row = W + int64_t(r) * nb * BB;                                                             \
            float s = 0.f;                                                                                              \
            for (int k = 0; k < nb; ++k) s += x.d[size_t(k)] * BLOCK(row + int64_t(k) * BB, x.q.data() + 256 * k, x.sum16.data() + 16 * k); \
            y[r - r0] = s;                                                                                              \
        }                                                                                                               \
    }
TQ_KQ_ROWS(q2k_rows_avx2, q2k_block_avx2, 84)
TQ_KQ_ROWS(q3k_rows_avx2, q3k_block_avx2, 110)
TQ_KQ_ROWS(q4k_rows_avx2, q4k_block_avx2, 144)
bool have_avx2() {
    static const bool v = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return v;
}
#endif

using BlockFn = float (*)(const uint8_t *, const int8_t *, const int16_t *);
int block_bytes(uint32_t t) { return t == kQ2K ? 84 : t == kQ3K ? 110 : t == kQ4K ? 144 : 0; }
void matvec(BlockFn f, int bb, const uint8_t * W, int cols, const Xq & x, int r0, int r1, float * y) {
    const int nb = cols / 256;
    for (int r = r0; r < r1; ++r) {
        const uint8_t * row = W + int64_t(r) * nb * bb;
        float s = 0.f;
        for (int k = 0; k < nb; ++k) s += x.d[size_t(k)] * f(row + int64_t(k) * bb, x.q.data() + 256 * k, x.sum16.data() + 16 * k);
        y[r - r0] = s;
    }
}
}  // namespace

void quantize_x(const float * x, int cols, Xq & out) {
    const int nb = cols / 256;
    out.cols = cols;
    out.q.resize(size_t(cols));
    out.d.resize(size_t(nb));
    out.sum16.resize(size_t(nb) * 16);
    for (int k = 0; k < nb; ++k) {
        const float * xb = x + 256 * k;
        float amax = 0.f;
        for (int i = 0; i < 256; ++i) amax = std::max(amax, std::fabs(xb[i]));
        const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
        out.d[size_t(k)] = d;
        for (int g = 0; g < 16; ++g) {
            int s = 0;
            for (int i = 0; i < 16; ++i) {
                const int v = int(std::lrintf(xb[16 * g + i] * id));
                out.q[size_t(256 * k + 16 * g + i)] = int8_t(std::clamp(v, -127, 127));
                s += out.q[size_t(256 * k + 16 * g + i)];
            }
            out.sum16[size_t(16 * k + g)] = int16_t(s);
        }
    }
}

bool kq_supported(uint32_t type) { return block_bytes(type) != 0; }
int64_t kq_row_bytes(uint32_t type, int cols) { return int64_t(cols / 256) * block_bytes(type); }

void kq_matvec_scalar(uint32_t type, const uint8_t * W, int cols, const Xq & x, int r0, int r1, float * y) {
    const BlockFn f = type == kQ2K ? q2k_block : type == kQ3K ? q3k_block : q4k_block;
    matvec(f, block_bytes(type), W, cols, x, r0, r1, y);
}

void kq_matvec(uint32_t type, const uint8_t * W, int cols, const Xq & x, int r0, int r1, float * y) {
#ifdef TQ_KQ_X86
    if (have_avx2()) {
        (type == kQ2K ? q2k_rows_avx2 : type == kQ3K ? q3k_rows_avx2 : q4k_rows_avx2)(W, cols, x, r0, r1, y);
        return;
    }
#endif
    kq_matvec_scalar(type, W, cols, x, r0, r1, y);
}

}}  // namespace tqllm::cpu
