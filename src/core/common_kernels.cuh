#pragma once
// FP32 kernels shared by model engines (moved from the GLM-5.3 engine, 11-02, now used by GLM-5.3 and MiniMax-M3): row
// matvec (plain and split-k), RMSNorm, column gather, type conversions, add, argmax; and the block / warp reductions
// their kernels and the engines' own kernels build on.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace tqllm_cuda::common {
#if defined(__CUDACC__) || defined(__HIP__)  // device helpers for the shared kernels and the engines' own kernels
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// block-wide sum, every thread gets the result; blockDim.x multiple of 32, <= 1024
__device__ __forceinline__ float block_sum(float v, float * red) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : 0.f;
    t = warp_sum(t);
    return t;
}
__device__ __forceinline__ float block_max(float v, float * red) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : -FLT_MAX;
    for (int o = 16; o; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    return t;
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
#endif

// y[r] = W[r] . x for an FP32 row-major [rows][cols] matrix
// y[t][r] = W[r] . x[t] for t < n
void mv_f32(const float * w, const float * x, float * y, int rows, int cols, int n, cudaStream_t s);
// same result for matrices with few rows: each row split over `splits` blocks; partials [n][rows][splits] summed in
// order (deterministic, no atomics). partial must hold n * rows * splits floats.
void mv_f32_splitk(const float * w, const float * x, float * y, float * partial, int rows, int cols, int n, int splits,
                       cudaStream_t s);
// out[t][j] = in[t * stride + off + j], j < width, t < n
void gather_cols(const float * in, int stride, int off, int width, int n, float * out, cudaStream_t s);
void bf16_to_f32(const __nv_bfloat16 * x, float * y, int n, cudaStream_t s);
void f32_to_half(const float * x, __half * y, int n, cudaStream_t s);
// y = x / sqrt(mean(x^2) + eps) * w (w may be null); writes FP32 and / or FP16 copies (either may be null)
void rmsnorm(const float * x, const float * w, float * y, __half * yh, int n, float eps, cudaStream_t s, int rows = 1);
// rows of length n: each normalised on its own (KDA output heads)
void rmsnorm_rows(const float * x, const float * w, float * y, int rows, int n, float eps, cudaStream_t s);
// a += b
void add(float * a, const float * b, int n, cudaStream_t s);
// index of the largest value (first on ties)
void argmax(const float * x, int n, int * out, cudaStream_t s);
// per row of x [rows][n]
void argmax_rows(const float * x, int n, int rows, int * out, cudaStream_t s);
}  // namespace tqllm_cuda::common
