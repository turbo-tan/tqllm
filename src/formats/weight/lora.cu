// Runtime LoRA kernels (lora.cuh): t = A x (one block per CB columns x RB ranks), then y += scale * B t.
#include "lora.cuh"

namespace tqllm_cuda {

namespace {
constexpr int kCB = 4, kRB = 8, kThreads = 256;

__global__ void lora_down_k(const float * __restrict__ a, int cols, int rank, const __half * __restrict__ x, int n,
                            const float * __restrict__ scale, float * __restrict__ t) {
    if (*scale == 0.f) return;
    const int j0 = blockIdx.x * kCB, k0 = blockIdx.y * kRB;
    const int nj = min(kCB, n - j0), nk = min(kRB, rank - k0);
    float acc[kCB][kRB];
#pragma unroll
    for (int jj = 0; jj < kCB; ++jj)
#pragma unroll
        for (int kk = 0; kk < kRB; ++kk) acc[jj][kk] = 0.f;
    for (int c = 2 * threadIdx.x; c < cols; c += 2 * kThreads) {
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
            for (int jj = 0; jj < kCB; ++jj) acc[jj][kk] = fmaf(av.x, xv[jj].x, fmaf(av.y, xv[jj].y, acc[jj][kk]));
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
            t[size_t(j0 + jj) * rank + k0 + kk] = v;
        }
    }
}

__global__ void lora_up_k(const float * __restrict__ b, int rows, int rank, const float * __restrict__ t,
                          __nv_bfloat16 * __restrict__ y, int ldy, const float * __restrict__ scale) {
    const float s = *scale;
    if (s == 0.f) return;
    __shared__ float tj[kLoraMaxRank];
    const int j = blockIdx.y;
    if (threadIdx.x < rank) tj[threadIdx.x] = t[size_t(j) * rank + threadIdx.x];
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const float * bi = b + size_t(i) * rank;
    float acc = 0.f;
    for (int k = 0; k < rank; ++k) acc = fmaf(bi[k], tj[k], acc);
    __nv_bfloat16 & o = y[size_t(j) * ldy + i];
    o = __float2bfloat16(__bfloat162float(o) + s * acc);
}
}  // namespace

void lora_apply(const float * a, const float * b, int rows, int cols, int rank, const __half * x, __nv_bfloat16 * y, int n,
                int ldy, const float * scale, float * t, cudaStream_t s) {
    if (n <= 0 || rank <= 0) return;
    lora_down_k<<<dim3((n + kCB - 1) / kCB, (rank + kRB - 1) / kRB), kThreads, 0, s>>>(a, cols, rank, x, n, scale, t);
    lora_up_k<<<dim3((rows + kThreads - 1) / kThreads, n), kThreads, 0, s>>>(b, rows, rank, t, y, ldy, scale);
}

}  // namespace tqllm_cuda
