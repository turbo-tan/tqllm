#pragma once
// TQT decode helpers shared by the TQT kernels (tqt.cu) and the Inkling expert kernels. Format: docs/reference/formats.md
// "TQT", code c2 (tqt-c2-v1); segment = 32 weights = 16 stream bytes, a block of 128 = 4 segments + one FP16 scale.
#include "common.cuh"

#include <cstdint>
#include <type_traits>

namespace tqllm_cuda {
namespace tqtk {
// the two FP16 values (unscaled) of an 18-bit window in bits 0-17 of v (higher bits are shifted out)
__device__ __forceinline__ uint32_t pair_win(uint32_t v) {
    const uint32_t x = v * 0xDE6C4000u + 0x9F054000u;  // (w * HA + HB) << 14
    const uint32_t y = __funnelshift_l(x, x, 16);
    const uint32_t z = y * 0x2C1B3C6Du + 0x9E3779B9u, z2 = z * 0x297A2D39u + 0x7F4A7C15u;
    return tqllm_dev::hadd2u((z & 0x8FFF8FFFu) ^ 0x3B603B60u, (z2 & 0x8FFF8FFFu) ^ 0x3B603B60u);
}
// pair p (0..15) of a segment: E = {x, y, z, w, first word of the next segment}
__device__ __forceinline__ uint32_t pair(const uint32_t (&E)[5], int p) {
    const int i = p / 4, o = p % 4;
    return pair_win(o == 0 ? E[i] : __funnelshift_r(E[i], E[i + 1], 8 * o));
}
// the 32 FP16 weights of one segment exactly as tqt_dequant_h writes them
__device__ __forceinline__ void segment_h(uint4 v, uint32_t nx, __half s, __half (&o)[32]) {
    const uint32_t E[5] = {v.x, v.y, v.z, v.w, nx};
    const __half2 s2 = __half2half2(s);
#pragma unroll
    for (int p = 0; p < 16; ++p) {
        const uint32_t h = tqllm_dev::hmul2u(pair(E, p), *reinterpret_cast<const uint32_t *>(&s2));
        o[2 * p] = __ushort_as_half((unsigned short)(h & 0xFFFFu));
        o[2 * p + 1] = __ushort_as_half((unsigned short)(h >> 16));
    }
}
// first word of the next segment of segment i's block (i = global segment index, 4 segments per block)
__device__ __forceinline__ uint32_t next_x(const uint4 * w, int64_t i) {
    return __ldg(reinterpret_cast<const uint32_t *>(w + ((i & ~int64_t(3)) | ((i + 1) & 3))));
}
template <class OUT> __device__ __forceinline__ OUT out_cast(float v) {
    if constexpr (std::is_same_v<OUT, float>) return v;
    else return __float2bfloat16_rn(v);
}
}  // namespace tqtk
}  // namespace tqllm_cuda
