// NVFP4 (modelopt) weight kernels, the Qwen3.8-Flash-Next expert format: codes [rows][k / 2] (E2M1 pairs, low nibble =
// even column), FP8 E4M3 scale per 16 [rows][k / 16], FP32 global scale; w = e2m1 * fp8(scale) * global.
// First version: correctness reference for the Flash-Next engine (one warp per row, FP32 accumulate), not tuned.
#include "common.cuh"
#include "quant_kernels.cuh"

#include <cuda_fp8.h>

#include <cstdint>

namespace tqllm_cuda {
namespace {
__device__ __constant__ float E2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                          -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// grid (rows / 8), 256 threads: warp = row, lanes stride over the 16-weight groups; up to 8 columns
template <int NCOL>
__global__ void __launch_bounds__(256) nvfp4_matvec_k(const uint8_t * w, const uint8_t * sc, float gscale,
                                                      const __nv_bfloat16 * x, float * y, int rows, int k) {
    const int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const int ng = k / 16;
    const uint8_t * wr = w + int64_t(row) * (k / 2);
    const uint8_t * sr = sc + int64_t(row) * ng;
    float acc[NCOL] = {};
    for (int g = lane; g < ng; g += 32) {
        __nv_fp8_e4m3 s8;
        s8.__x = sr[g];
        const float s = float(s8);
        const uint64_t codes = *reinterpret_cast<const uint64_t *>(wr + g * 8);  // 16 nibbles
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            const float wv = E2M1[(codes >> (4 * e)) & 15u] * s;
#pragma unroll
            for (int c = 0; c < NCOL; ++c) acc[c] += wv * __bfloat162float(x[int64_t(c) * k + g * 16 + e]);
        }
    }
#pragma unroll
    for (int c = 0; c < NCOL; ++c) {
        float v = acc[c];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) y[int64_t(c) * rows + row] = v * gscale;
    }
}
}  // namespace

void nvfp4_matvec(const uint8_t * w, const uint8_t * sc, float gscale, const __nv_bfloat16 * x, float * y, int rows,
                  int k, int ncols, cudaStream_t s) {
    const dim3 grid((rows + 7) / 8);
    switch (ncols) {
        case 1: nvfp4_matvec_k<1><<<grid, 256, 0, s>>>(w, sc, gscale, x, y, rows, k); break;
        case 2: nvfp4_matvec_k<2><<<grid, 256, 0, s>>>(w, sc, gscale, x, y, rows, k); break;
        case 4: nvfp4_matvec_k<4><<<grid, 256, 0, s>>>(w, sc, gscale, x, y, rows, k); break;
        case 8: nvfp4_matvec_k<8><<<grid, 256, 0, s>>>(w, sc, gscale, x, y, rows, k); break;
        default:
            for (int c0 = 0; c0 < ncols; ++c0)
                nvfp4_matvec_k<1><<<grid, 256, 0, s>>>(w, sc, gscale, x + int64_t(c0) * k, y + int64_t(c0) * rows, rows, k);
    }
}
}  // namespace tqllm_cuda
