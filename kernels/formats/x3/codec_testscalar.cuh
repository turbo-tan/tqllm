#pragma once
// codec_testscalar.cuh: a test-only codec, present so the codec seam is provable. It is a plain
// 8-bit scalar codebook (value = (byte - 128) / 128), 256 bytes per 16x16 tile = 128 uint16. It
// implements exactly the Mul1Codec interface, so gemv_sq / gemv_x3v / reconstruct instantiate for
// it unchanged and the same G2 matvec-vs-dequant tests run on it for free. A future real codec is
// the same shape: one new codec_<name>.cuh plus one host table row.
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
};

}  // namespace x3
}  // namespace tqllm
