// MXFP8 matrices (OCP microscaling FP8: E4M3 values, one E8M0 scale per 32 along a row), as the original MiniMax /
// DSpark checkpoints store their projections. Row-major rows of k E4M3 bytes followed by that row's k / 32 scale bytes,
// so stacked matrices and row ranges are plain byte ranges. One warp per row, FP32 accumulation, up to 8 columns.
#include "quant_kernels.cuh"

#include <cuda_fp8.h>

#include <algorithm>
#include <stdexcept>

namespace {
__device__ __forceinline__ float e4m3(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t(b), __NV_E4M3)));
}
__device__ __forceinline__ float e8m0(uint8_t s) { return s ? __int_as_float(int(s) << 23) : 5.877472e-39f; }  // 2^(s-127)

template <int N, bool F32>
__global__ void mxfp8_matvec_k(const uint8_t * __restrict__ w, const __half * __restrict__ x, void * y, int rows, int k, int ldy) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const uint8_t * wr = w + size_t(row) * (k + k / 32);
    const uint8_t * sr = wr + k;
    float acc[N] = {};
    for (int i = lane * 16; i < k; i += 512) {  // 16 weights per lane: half of one 32-block
        const uint4 wv = __ldg(reinterpret_cast<const uint4 *>(wr + i));
        const uint8_t * we = reinterpret_cast<const uint8_t *>(&wv);
        const float s = e8m0(__ldg(sr + i / 32));
        float wf[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) wf[j] = e4m3(we[j]) * s;
#pragma unroll
        for (int c = 0; c < N; ++c) {
            const uint4 x0 = __ldg(reinterpret_cast<const uint4 *>(x + size_t(c) * k + i));
            const uint4 x1 = __ldg(reinterpret_cast<const uint4 *>(x + size_t(c) * k + i + 8));
            const __half * a = reinterpret_cast<const __half *>(&x0);
            const __half * b = reinterpret_cast<const __half *>(&x1);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[c] += wf[j] * __half2float(a[j]) + wf[j + 8] * __half2float(b[j]);
        }
    }
#pragma unroll
    for (int c = 0; c < N; ++c) {
        float v = acc[c];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) {
            if constexpr (F32) static_cast<float *>(y)[size_t(c) * ldy + row] = v;
            else static_cast<__nv_bfloat16 *>(y)[size_t(c) * ldy + row] = __float2bfloat16(v);
        }
    }
}

__global__ void mxfp8_dequant_h_k(const uint8_t * __restrict__ w, __half * out, size_t n, int k) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const size_t r = i / size_t(k), c = i % size_t(k);
        const uint8_t * wr = w + r * size_t(k + k / 32);
        out[i] = __float2half(e4m3(wr[c]) * e8m0(wr[k + c / 32]));
    }
}

template <bool F32>
void launch(const uint8_t * w, const __half * x, void * y, int rows, int k, int n, cudaStream_t s, int ldy) {
    if (!ldy) ldy = rows;
    const dim3 grid((rows + 3) / 4), block(128);
    switch (n) {
        case 1: mxfp8_matvec_k<1, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 2: mxfp8_matvec_k<2, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 3: mxfp8_matvec_k<3, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 4: mxfp8_matvec_k<4, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 5: mxfp8_matvec_k<5, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 6: mxfp8_matvec_k<6, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 7: mxfp8_matvec_k<7, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 8: mxfp8_matvec_k<8, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        default: throw std::invalid_argument("mxfp8_matvec: 1..8 columns");
    }
}
}  // namespace

namespace tqllm_cuda {
void mxfp8_matvec(const void * w, const __half * x, void * y, bool f32_out, int rows, int k, int n, cudaStream_t s, int ldy) {
    if (k % 512) throw std::invalid_argument("mxfp8_matvec: columns must be a multiple of 512");
    const auto * q = static_cast<const uint8_t *>(w);
    f32_out ? launch<true>(q, x, y, rows, k, n, s, ldy) : launch<false>(q, x, y, rows, k, n, s, ldy);
}
void mxfp8_dequant_h(const void * w, __half * out, int rows, int k, cudaStream_t s) {
    const size_t n = size_t(rows) * k;
    const auto * q = static_cast<const uint8_t *>(w);
    mxfp8_dequant_h_k<<<int(std::min<size_t>((n + 255) / 256, 4096)), 256, 0, s>>>(q, out, n, k);
}
}  // namespace tqllm_cuda
