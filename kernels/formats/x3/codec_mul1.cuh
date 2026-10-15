#pragma once
// codec_mul1.cuh: the ONE place that knows how mul1 trellis words become weights.
//
// A PAW X3 tile is 16x16 weights packed as `8*KH` uint16 (KH = 2*K, so K 1/1.5/2/2.5/3/3.5/4 ->
// KH 2..8). A lane decodes 8 consecutive weights into 4 half2; 32 lanes cover one tile. Weights
// come from the exllamav3 mul1 ("3-inst") scalar codebook: idx -> fma(uint16(idx*0x83DCD12D),
// 1/147.7, -10.39). The window extraction is ported verbatim (source line comments kept) from
// llama-paw-k25 ggml/src/ggml-cuda/paw-x3.cu.
//
// The gemv/reconstruct skeletons are templated on a Codec like this one and never name mul1 or K,
// so a new codec is a new codec_<name>.cuh plus one row in the host codec table.
#include "ptx.cuh"
#include <cuda_fp16.h>

namespace tqllm {
namespace x3 {

// Decode two mul1 codebook entries from precomputed products x0 = idx0 * 0x83DCD12D,
// x1 = idx1 * 0x83DCD12D (paw-x3.cu decode_mul1_product_2).
__device__ __forceinline__ half2 mul1_decode_2(uint32_t x0, uint32_t x1) {
    const uint32_t acc = 0x6400u;
    uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));   //  0.00677 = 1/147.7
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));  // -10.39
    return __hfma2(__halves2half2(__ushort_as_half((uint16_t) sum0), __ushort_as_half((uint16_t) sum1)),
                   k_inv_h2, k_bias_h2);
}

__host__ __device__ constexpr bool mul1_frac(int KH) { return (KH & 1) != 0; }          // 1.5, 2.5, 3.5
__host__ __device__ constexpr int  mul1_ka(int KH)   { return (KH - 1) / 2; }            // 1, 2, 3
__host__ __device__ constexpr int  mul1_bits(int KH) { return (KH & 1) ? 5 * KH : KH / 2; }  // 1..4 / 15,25,35

// Four consecutive windows whose last one ends at ring bit e (exclusive); they end at
// e - P - KA - 1, e - P, e - KA - 1 and e. Ported from paw-x3.cu ext4_frac.
template <int KA>
__device__ __forceinline__ void ext4_frac(const uint32_t* ptr, int e, uint32_t& x0, uint32_t& x1, uint32_t& x2, uint32_t& x3) {
    constexpr int P = 2 * KA + 1;
    constexpr int NW = 8 * KA + 4;
    const int i2 = (e - 1) >> 5;
    const int i1 = i2 == 0 ? NW - 1 : i2 - 1;
    const uint64_t v = ((uint64_t) ptr[i1] << 32) | (uint64_t) ptr[i2];
    const int s = ((i2 + 1) << 5) - e;
    x3 = (uint32_t) (v >> s) & 0xffff;
    x2 = (uint32_t) (v >> (s + KA + 1)) & 0xffff;
    x1 = (uint32_t) (v >> (s + P)) & 0xffff;
    x0 = (uint32_t) (v >> (s + P + KA + 1)) & 0xffff;
}

// Windows t0 .. t0 + 7 (t0 = 8 * lane) are four pairs; they end at 4 * P * lane + 2 * P and + 4 * P.
template <int KA>
__device__ __forceinline__ void ext8w_frac(const uint32_t* ptr, int t0, uint32_t (&w)[8]) {
    constexpr int P = 2 * KA + 1;
    const int e = (t0 >> 3) * (4 * P);
    ext4_frac<KA>(ptr, e + 2 * P, w[7], w[6], w[5], w[4]);
    ext4_frac<KA>(ptr, e + 4 * P, w[3], w[2], w[1], w[0]);
}

template <int bits>
__device__ __forceinline__ int wrap_idx(int i) {
    constexpr int words = bits * 256 / 32;
    return i >= words ? i - words : i;
}

// Extract the 8 window values (t_offset .. t_offset + 7) as uint16 each. One branch per K, ported
// from paw-x3.cu dq8 / dq8_aligned_{1,2,4}bits / dq8_frac.
template <int KH>
__device__ __forceinline__ void mul1_extract8(const uint32_t* ptr, int t_offset, uint32_t (&w)[8]) {
    constexpr int bits = mul1_bits(KH);
    if constexpr (mul1_frac(KH)) {
        ext8w_frac<mul1_ka(KH)>(ptr, t_offset, w);
    } else if constexpr (bits == 1) {
        const uint32_t i1 = uint32_t(t_offset) >> 5;
        const uint32_t i0 = (i1 + 7) & 7;
        const uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t_offset) & 24));
        w[7] = b & 0xffff;
        BFE16_IMM(w[6], b, 1); BFE16_IMM(w[5], b, 2); BFE16_IMM(w[4], b, 3);
        BFE16_IMM(w[3], b, 4); BFE16_IMM(w[2], b, 5); BFE16_IMM(w[1], b, 6); BFE16_IMM(w[0], b, 7);
    } else if constexpr (bits == 2) {
        const uint32_t i1 = uint32_t(t_offset) >> 4;
        const uint32_t i0 = (i1 + 15) & 15;
        const uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t_offset) & 8) << 1);
        w[7] = b & 0xffff;
        BFE16_IMM(w[6], b, 2); BFE16_IMM(w[5], b, 4); BFE16_IMM(w[4], b, 6);
        BFE16_IMM(w[3], b, 8); BFE16_IMM(w[2], b, 10); BFE16_IMM(w[1], b, 12); BFE16_IMM(w[0], b, 14);
    } else if constexpr (bits == 4) {
        const uint32_t i1 = uint32_t(t_offset) >> 3;
        const uint32_t i0 = (i1 + 31) & 31;
        const uint32_t a = ptr[i0];
        const uint32_t b = ptr[i1];
        uint32_t s;
        FSHF_IMM(s, b, a, 20);
        w[7] = b & 0xffff;
        BFE16_IMM(w[6], b, 4); BFE16_IMM(w[5], b, 8); BFE16_IMM(w[4], b, 12); BFE16_IMM(w[3], b, 16);
        w[2] = s & 0xffff;
        BFE16_IMM(w[1], s, 4); BFE16_IMM(w[0], s, 8);
    } else {  // bits == 3 (generic, align 4)
        const int b1 = (t_offset + 257) * bits;
        const int b0 = b1 - 16;
        const int b2 = b1 + bits * 7;
        const uint32_t i0 = uint32_t(b0 / 32) % uint32_t(bits * 256 / 32);
        const uint32_t i2 = uint32_t((b2 - 1) / 32) % uint32_t(bits * 256 / 32);
        const int s2 = (int(i2) + 1) * 32 - b2;
        const uint32_t a = ptr[i0];
        const uint32_t b = ptr[i2];
        w[7] = fshift(b, a, s2);
        w[6] = w[7] >> bits;
        w[5] = w[6] >> bits;
        w[4] = w[5] >> bits;
        w[3] = fshift(b, a, s2 + bits * 4);
        w[2] = w[3] >> bits;
        w[1] = w[2] >> bits;
        w[0] = w[1] >> bits;
        w[0] &= 0xffff; w[1] &= 0xffff; w[2] &= 0xffff; w[3] &= 0xffff;
        w[4] &= 0xffff; w[5] &= 0xffff; w[6] &= 0xffff; w[7] &= 0xffff;
    }
}

// The mul1 codec. K = KH/2 (KH in {2,3,4,5,6,7,8}); `words` and `lanes` are what the skeletons need.
template <int KH>
struct Mul1Codec {
    static_assert(KH >= 2 && KH <= 8, "mul1 K must be in {1,1.5,2,2.5,3,3.5,4}");
    static constexpr int words = 8 * KH;   // uint16 per 16x16 tile
    static constexpr int lanes = 32;       // 8-weight groups per tile

    __device__ static void decode(const uint16_t* tile, int lane, half2 (&w)[4]) {
        uint32_t x[8];
        mul1_extract8<KH>(reinterpret_cast<const uint32_t*>(tile), lane * 8, x);
        w[0] = mul1_decode_2(x[0] * 0x83DCD12Du, x[1] * 0x83DCD12Du);
        w[1] = mul1_decode_2(x[2] * 0x83DCD12Du, x[3] * 0x83DCD12Du);
        w[2] = mul1_decode_2(x[4] * 0x83DCD12Du, x[5] * 0x83DCD12Du);
        w[3] = mul1_decode_2(x[6] * 0x83DCD12Du, x[7] * 0x83DCD12Du);
    }
};

}  // namespace x3
}  // namespace tqllm
