// Runtime LoRA kernels (lora.cuh): partial A x per column slice (one block per slice x CB columns x RB ranks), then
// y += scale * B t with t summed over the slices in a fixed order. Decode widths (n <= 16) spread A x over every slice
// of the input instead of one block per column, so the adapter costs a few microseconds per matrix.
#include "lora.cuh"

namespace tqllm_cuda {

namespace {
constexpr int kCB = 4, kRB = 8, kThreads = kLoraSlice / 2;

// part [slices][n][rank]
__global__ void lora_down_k(const float * __restrict__ a, int cols, int rank, const __half * __restrict__ x, int n,
                            const float * __restrict__ scale, float * __restrict__ part) {
    if (*scale == 0.f) return;
    const int sl = blockIdx.x, j0 = blockIdx.y * kCB, k0 = blockIdx.z * kRB;
    const int nj = min(kCB, n - j0), nk = min(kRB, rank - k0);
    const int c = sl * kLoraSlice + 2 * threadIdx.x;
    float acc[kCB][kRB];
#pragma unroll
    for (int jj = 0; jj < kCB; ++jj)
#pragma unroll
        for (int kk = 0; kk < kRB; ++kk) acc[jj][kk] = 0.f;
    if (c < cols) {
        float2 xv[kCB];
#pragma unroll
        for (int jj = 0; jj < kCB; ++jj)
            xv[jj] = jj < nj ? __half22float2(*reinterpret_cast<const __half2 *>(x + size_t(j0 + jj) * cols + c))
                             : make_float2(0.f, 0.f);
#pragma unroll
        for (int kk = 0; kk < kRB; ++kk) {
            if (kk >= nk) break;
            const float2 av = *reinterpret_cast<const float2 *>(a + size_t(k0 + kk) * cols + c);
#pragma unroll
            for (int jj = 0; jj < kCB; ++jj) acc[jj][kk] = fmaf(av.x, xv[jj].x, av.y * xv[jj].y + acc[jj][kk]);
        }
    }
    __shared__ float red[kCB * kRB][kThreads / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int jj = 0; jj < kCB; ++jj)
#pragma unroll
        for (int kk = 0; kk < kRB; ++kk) {
            float v = acc[jj][kk];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[jj * kRB + kk][warp] = v;
        }
    __syncthreads();
    if (threadIdx.x < kCB * kRB) {
        const int jj = threadIdx.x / kRB, kk = threadIdx.x % kRB;
        if (jj < nj && kk < nk) {
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < kThreads / 32; ++w) v += red[threadIdx.x][w];
            part[(size_t(sl) * n + j0 + jj) * rank + k0 + kk] = v;
        }
    }
}

__global__ void lora_up_k(const float * __restrict__ b, int rows, int rank, const float * __restrict__ part, int slices,
                          int n, __nv_bfloat16 * __restrict__ y, int ldy, const float * __restrict__ scale) {
    const float s = *scale;
    if (s == 0.f) return;
    __shared__ float tj[kLoraMaxRank];
    const int j = blockIdx.y;
    if (threadIdx.x < rank) {
        float v = 0.f;
        for (int sl = 0; sl < slices; ++sl) v += part[(size_t(sl) * n + j) * rank + threadIdx.x];
        tj[threadIdx.x] = v;
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const float * bi = b + size_t(i) * rank;
    float acc = 0.f;
    for (int k = 0; k < rank; ++k) acc = fmaf(bi[k], tj[k], acc);
    __nv_bfloat16 & o = y[size_t(j) * ldy + i];
    o = __float2bfloat16(__bfloat162float(o) + s * acc);
}
__global__ void lora_scale_k(const float * __restrict__ t, size_t m, const float * __restrict__ scale, __nv_bfloat16 * __restrict__ tb) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < m) tb[i] = __float2bfloat16(t[i] * *scale);
}
}  // namespace

void lora_apply_gemm(cublasHandle_t h, const __half * a16, const __nv_bfloat16 * b16, int rows, int cols, int rank, const __half * x,
                     __nv_bfloat16 * y, int n, int ldy, const float * scale, float * t, __nv_bfloat16 * tb, cudaStream_t s) {
    if (n <= 0 || rank <= 0) return;
    cublasSetStream(h, s);
    const float one = 1.f, zero = 0.f;
    // column-major: t (rank x n) = a16^T (rank x cols) . x (cols x n)
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rank, n, cols, &one, a16, CUDA_R_16F, cols, x, CUDA_R_16F, cols, &zero, t, CUDA_R_32F, rank,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    const size_t m = size_t(n) * rank;
    lora_scale_k<<<unsigned((m + 255) / 256), 256, 0, s>>>(t, m, scale, tb);
    // y (rows x n, ld ldy) += b16^T (rows x rank) . tb (rank x n)
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, rank, &one, b16, CUDA_R_16BF, rank, tb, CUDA_R_16BF, rank, &one, y, CUDA_R_16BF, ldy,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void lora_apply(const float * a, const float * b, int rows, int cols, int rank, const __half * x, __nv_bfloat16 * y, int n,
                int ldy, const float * scale, float * t, cudaStream_t s) {
    if (n <= 0 || rank <= 0) return;
    const int slices = (cols + kLoraSlice - 1) / kLoraSlice;
    lora_down_k<<<dim3(slices, (n + kCB - 1) / kCB, (rank + kRB - 1) / kRB), kThreads, 0, s>>>(a, cols, rank, x, n, scale, t);
    lora_up_k<<<dim3((rows + 255) / 256, n), 256, 0, s>>>(b, rows, rank, t, slices, n, y, ldy, scale);
}

}  // namespace tqllm_cuda
