#pragma once
// Trellis-coded KV quantisation (TCQ), host reference of the CUDA lane's format (src/formats/kv/kv_tq3.cu cache_write_tcq,
// kv_compact.cu tcq_value): per KV-head row of 256 values, two 128-value groups. A group is normalised to unit norm,
// rotated (signs S1, 128-point Walsh-Hadamard / sqrt(128), signs S2) and coded by Viterbi over the right-shift trellis
// ns = (prev >> KB) | (out << 6) with a free initial state: KB = 3 (512 states) for K, KB = 2 (256 states) for V. The group
// stream holds the 6 initial-state bits, then 128 KB-bit symbols; the last word's high half is the group's FP16 norm
// (original norm / reconstructed norm x alpha). Value t of a group = codebook[state after symbol t] x norm, in the rotated
// basis. K: 13 words per group (26 per row, 104 B); V: 9 (18 per row, 72 B) = 176 B per row against 1024 B in BF16.
// The Metal kernels follow this reference; tests compare against it.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace tqllm::mlx_backend::tcq {
#include "../../formats/kv/tcq2_tables.inc"
#include "../../formats/kv/tcq3_tables.inc"
inline constexpr float kCodebookV[256] = TQ_TCQ2_CODEBOOK_V;
inline constexpr float kCodebookK[512] = TQ_TCQ3_CODEBOOK_K;
inline constexpr float kSigns1[128] = TQ_TCQ_SIGNS1;
inline constexpr float kSigns2[128] = TQ_TCQ_SIGNS2;
inline constexpr float kAlphaK = 1.0f, kAlphaV = 1.04f, kInvSqrt128 = 0.08838834764831845f;
inline constexpr int kGroup = 128, kDim = 256;
constexpr int words_per_group(int kb) {return (6 + 128 * kb + 31) / 32;}  // K 13, V 9

inline uint16_t to_half(float f) {  // round to nearest even, finite range only (norms)
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u; int e = int((x >> 23) & 0xFF) - 127 + 15; uint32_t m = x & 0x7FFFFFu;
    if (e <= 0) {if (e < -10) return uint16_t(sign); m |= 0x800000u; const int sh = 14 - e; uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r; return uint16_t(sign | r);}
    if (e >= 31) return uint16_t(sign | 0x7C00u);
    uint32_t r = (uint32_t(e) << 10) | (m >> 13); const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return uint16_t(sign | r);
}
inline float from_half(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF; uint32_t x;
    if (e == 0) {if (m == 0) x = sign; else {int ee = -1; uint32_t mm = m; do {++ee; mm <<= 1;} while (!(mm & 0x400)); x = sign | uint32_t(127 - 15 - ee) << 23 | (mm & 0x3FF) << 13;}}
    else if (e == 31) x = sign | 0x7F800000u | (m << 13);
    else x = sign | (e - 15 + 127) << 23 | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}
// in place: 128-point unnormalised Walsh-Hadamard (the same butterfly order as the CUDA writer)
inline void wht128(float* x) {
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; ++i) if (!(i & h)) {const float a = x[i], b = x[i + h]; x[i] = a + b; x[i + h] = a - b;}
}
// one 128-value group -> words_per_group(kb) words
inline void encode_group(const float* in, int kb, uint32_t* out) {
    const int states = 64 << kb, np = 1 << kb, wpg = words_per_group(kb);
    const float* cb = kb == 2 ? kCodebookV : kCodebookK;
    float x[128]; double ss = 0;
    for (int i = 0; i < 128; ++i) {x[i] = in[i]; ss += double(x[i]) * x[i];}
    const float gnorm = float(std::sqrt(ss)), inv = gnorm > 1e-10f ? 1.0f / gnorm : 0.0f;
    for (int i = 0; i < 128; ++i) x[i] = x[i] * inv * kSigns1[i];
    wht128(x);
    for (int i = 0; i < 128; ++i) x[i] *= kInvSqrt128 * kSigns2[i];
    std::vector<float> cost(states, 0.0f), next(states);
    std::vector<uint8_t> bt(128 * 64);
    for (int t = 0; t < 128; ++t) {
        for (int s = 0; s < states; ++s) {
            const int base = (s & 0x3F) << kb; float best = cost[base]; int bp = 0;
            for (int p = 1; p < np; ++p) if (cost[base | p] < best) {best = cost[base | p]; bp = p;}
            if (s < 64) bt[t * 64 + s] = uint8_t(bp);
            const float d = x[t] - cb[s]; next[s] = best + d * d;
        }
        cost.swap(next);
    }
    int state = 0; for (int s = 1; s < states; ++s) if (cost[s] < cost[state]) state = s;
    uint8_t sym[128];
    for (int t = 127; t >= 0; --t) {sym[t] = uint8_t(state >> 6); state = ((state & 0x3F) << kb) | bt[t * 64 + (state & 0x3F)];}
    const int init = state;
    double rs = 0; int cs = init;
    for (int t = 0; t < 128; ++t) {cs = (cs >> kb) | (int(sym[t]) << 6); rs += double(cb[cs]) * cb[cs];}
    const float rn = float(std::sqrt(rs));
    const float norm = (rn > 1e-10f ? gnorm / rn : gnorm) * (kb == 2 ? kAlphaV : kAlphaK);
    const int ib = (init >> kb) & 0x3F, nbits = 6 + 128 * kb;
    for (int w = 0; w < wpg; ++w) {
        uint32_t word = 0;
        for (int bit = 0; bit < 32; ++bit) {
            const int pos = w * 32 + bit; int v = 0;
            if (pos < 6) v = (ib >> pos) & 1;
            else if (pos < nbits) v = (sym[(pos - 6) / kb] >> ((pos - 6) % kb)) & 1;
            word |= uint32_t(v) << bit;
        }
        if (w == wpg - 1) word |= uint32_t(to_half(norm)) << 16;
        out[w] = word;
    }
}
// value t of a group, rotated basis (the attention reads this; K: queries rotated the same way, V: output rotated back)
inline float group_value(const uint32_t* g, int kb, int t) {
    const int words = words_per_group(kb), lbits = 6 + kb, bit = kb * t, w0 = bit / 32;
    const uint64_t win = uint64_t(g[w0]) | (w0 + 1 < words ? uint64_t(g[w0 + 1]) << 32 : 0);
    const uint32_t st = uint32_t(win >> (bit % 32)) & ((1u << lbits) - 1);
    return (kb == 2 ? kCodebookV : kCodebookK)[st] * from_half(uint16_t(g[words - 1] >> 16));
}
// rotated basis -> original (R^T = S1 x WHT / sqrt(128) x S2 per group)
inline void unrotate_group(float* x) {
    for (int i = 0; i < 128; ++i) x[i] *= kSigns2[i];
    wht128(x);
    for (int i = 0; i < 128; ++i) x[i] *= kInvSqrt128 * kSigns1[i];
}
inline void rotate_group(float* x) {  // original -> rotated (no normalisation): queries for K, so <Rq, Rk> = <q, k>
    for (int i = 0; i < 128; ++i) x[i] *= kSigns1[i];
    wht128(x);
    for (int i = 0; i < 128; ++i) x[i] *= kInvSqrt128 * kSigns2[i];
}
// a 256-value row: K (kb 3, 26 words) or V (kb 2, 18 words)
inline void encode_row(const float* row, int kb, uint32_t* out) {
    encode_group(row, kb, out); encode_group(row + 128, kb, out + words_per_group(kb));
}
inline void decode_row(const uint32_t* rec, int kb, float* out, bool original_basis) {
    for (int g = 0; g < 2; ++g) {
        for (int t = 0; t < 128; ++t) out[g * 128 + t] = group_value(rec + g * words_per_group(kb), kb, t);
        if (original_basis) unrotate_group(out + g * 128);
    }
}
}  // namespace tqllm::mlx_backend::tcq
