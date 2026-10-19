// Dense BF16 / F16 matrices: one warp per row, FP32 accumulation, up to 8 input columns per launch.
#include "quant_kernels.cuh"

#include <algorithm>
#include <stdexcept>

namespace {
template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T, int N, bool F32>
__global__ void fp_matvec_k(const T * __restrict__ w, const __half * __restrict__ x, void * y, int rows, int k, int ldy) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const T * wr = w + size_t(row) * k;
    float acc[N] = {};
    if (k % 8 == 0) {
        for (int i = lane * 8; i < k; i += 256) {
            const uint4 wv = __ldg(reinterpret_cast<const uint4 *>(wr + i));
            const T * we = reinterpret_cast<const T *>(&wv);
#pragma unroll
            for (int c = 0; c < N; ++c) {
                const uint4 xv = __ldg(reinterpret_cast<const uint4 *>(x + size_t(c) * k + i));
                const __half * xe = reinterpret_cast<const __half *>(&xv);
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[c] += to_f(we[j]) * __half2float(xe[j]);
            }
        }
    } else {
        for (int i = lane; i < k; i += 32) {
            const float wv = to_f(wr[i]);
#pragma unroll
            for (int c = 0; c < N; ++c) acc[c] += wv * __half2float(x[size_t(c) * k + i]);
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

template <typename T>
__global__ void fp_dequant_h_k(const T * __restrict__ w, __half * out, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        out[i] = __float2half(to_f(w[i]));
}

template <typename T, bool F32>
void launch(const T * w, const __half * x, void * y, int rows, int k, int n, cudaStream_t s, int ldy) {
    if (!ldy) ldy = rows;
    const dim3 grid((rows + 3) / 4), block(128);
    switch (n) {
        case 1: fp_matvec_k<T, 1, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 2: fp_matvec_k<T, 2, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 3: fp_matvec_k<T, 3, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 4: fp_matvec_k<T, 4, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 5: fp_matvec_k<T, 5, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 6: fp_matvec_k<T, 6, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 7: fp_matvec_k<T, 7, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 8: fp_matvec_k<T, 8, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        default: throw std::invalid_argument("fp_matvec: 1..8 columns");
    }
}
}  // namespace

namespace tqllm_cuda {
void fp_matvec(bool bf16, const void * w, const __half * x, void * y, bool f32_out, int rows, int k, int n, cudaStream_t s, int ldy) {
    if (bf16) f32_out ? launch<__nv_bfloat16, true>(static_cast<const __nv_bfloat16 *>(w), x, y, rows, k, n, s, ldy)
                      : launch<__nv_bfloat16, false>(static_cast<const __nv_bfloat16 *>(w), x, y, rows, k, n, s, ldy);
    else f32_out ? launch<__half, true>(static_cast<const __half *>(w), x, y, rows, k, n, s, ldy)
                 : launch<__half, false>(static_cast<const __half *>(w), x, y, rows, k, n, s, ldy);
}
void fp_dequant_h(bool bf16, const void * w, __half * out, size_t n, cudaStream_t s) {
    const int blocks = int(std::min<size_t>((n + 255) / 256, 4096));
    if (bf16) fp_dequant_h_k<<<blocks, 256, 0, s>>>(static_cast<const __nv_bfloat16 *>(w), out, n);
    else fp_dequant_h_k<<<blocks, 256, 0, s>>>(static_cast<const __half *>(w), out, n);
}
}  // namespace tqllm_cuda
