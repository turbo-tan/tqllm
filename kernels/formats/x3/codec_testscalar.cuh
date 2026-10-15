#pragma once
// codec_testscalar.cuh: a test-only codec, present so the codec seam is provable. It is a plain
// 8-bit scalar codebook (value = (byte - 128) / 128), 256 bytes per 16x16 tile = 128 uint16. It
// implements exactly the Mul1Codec interface (both the smem decode and the register path), so
// gemv_sq / gemv_x3v / reconstruct instantiate for it unchanged and the same G2 matvec-vs-dequant
// tests run on it for free. A future real codec is the same shape: one new codec_<name>.cuh plus
// one host table row.
#include "ptx.cuh"
#include <cuda_fp16.h>

namespace tqllm {
namespace x3 {

struct TestScalarCodec {
    static constexpr int words = 128;  // 256 weights * 1 byte = 256 B = 128 uint16
    static constexpr int lanes = 32;   // 8-weight groups per tile

    __device__ static void decode(const uint16_t* tile, int lane, half2 (&w)[4]) {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(tile) + lane * 8;
        for (int i = 0; i < 4; ++i) {
            const float v0 = (float(int(b[2 * i + 0]) - 128)) * (1.0f / 128.0f);
            const float v1 = (float(int(b[2 * i + 1]) - 128)) * (1.0f / 128.0f);
            w[i] = __floats2half2_rn(v0, v1);
        }
    }

    // ---- register path (gemv_x3v): two 32-word loads per tile ----
    static constexpr int  bits     = 8;
    static constexpr bool frac     = false;
    static constexpr int  ka       = 0;
    static constexpr int  u32words = words / 2;  // 64
    static constexpr int  ld_stride = 32;
    static constexpr int  ld_lanes  = 32;
    static constexpr bool supports_gemv_x3v = true;
    static __host__ __device__ constexpr int loads(int WNT) { return WNT * u32words / ld_stride; }

    // The int8 dp4a "sq" path relies on the mul1 linear codebook, so this codec does not provide it.
    static constexpr bool supports_sq = false;
    static __host__ __device__ constexpr int row_u32(int size_n) { return size_n * u32words / 16; }
    __device__ static void sq_windows(const uint32_t*, int, uint32_t (&)[8]) {}

    __device__ static void regs_decode(const uint32_t* bw, int t, int lane, FragB& f0, FragB& f1) {
        const uint32_t full = 0xffffffffu;
        // Load l holds tile word (l*32 + lane). A lane needs words 2*lane and 2*lane+1; the shfl
        // operand must be uniform across lanes, so fetch both loads and select.
        const int ia = lane * 2, ib = lane * 2 + 1;
        // Call both shuffles unconditionally (a divergent __shfl_sync is undefined) then select.
        const uint32_t a0 = __shfl_sync(full, bw[2 * t + 0], ia % 32);
        const uint32_t a1 = __shfl_sync(full, bw[2 * t + 1], ia % 32);
        const uint32_t b0 = __shfl_sync(full, bw[2 * t + 0], ib % 32);
        const uint32_t b1w = __shfl_sync(full, bw[2 * t + 1], ib % 32);
        const uint32_t wa = (ia < 32) ? a0 : a1;
        const uint32_t wb = (ib < 32) ? b0 : b1w;
        const uint8_t b[8] = {
            uint8_t(wa), uint8_t(wa >> 8), uint8_t(wa >> 16), uint8_t(wa >> 24),
            uint8_t(wb), uint8_t(wb >> 8), uint8_t(wb >> 16), uint8_t(wb >> 24)};
        half2 r[4];
        for (int i = 0; i < 4; ++i)
            r[i] = __floats2half2_rn((float(int(b[2 * i + 0]) - 128)) * (1.0f / 128.0f),
                                     (float(int(b[2 * i + 1]) - 128)) * (1.0f / 128.0f));
        f0[0] = r[0]; f0[1] = r[1]; f1[0] = r[2]; f1[1] = r[3];
    }
};

}  // namespace x3
}  // namespace tqllm
