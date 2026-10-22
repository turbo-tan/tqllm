#include "vision_kernels.cuh"

#include <cmath>

namespace tqllm_cuda {
namespace {

__device__ __forceinline__ float block_sum(float v, float * sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? sh[threadIdx.x] : 0.f;
    if (w == 0)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}

__device__ __forceinline__ float block_max(float v, float * sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? sh[threadIdx.x] : -INFINITY;
    if (w == 0)
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}

__global__ void layer_norm_k(const float * x, const float * w, const float * b, float * y, int d, float eps) {
    __shared__ float sh[32];
    const float * xr = x + int64_t(blockIdx.x) * d;
    float * yr = y + int64_t(blockIdx.x) * d;
    float s = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s, sh) / d;
    float v = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) v += (xr[i] - mean) * (xr[i] - mean);
    const float inv = rsqrtf(block_sum(v, sh) / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

__global__ void bias_k(float * y, const float * b, int64_t n, int d, bool gelu) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = y[i] + b[i % d];
    if (gelu) v = 0.5f * v * (1.f + tanhf(0.7978845608028654f * v * (1.f + 0.044715f * v * v)));
    y[i] = v;
}

__global__ void add_k(float * y, const float * x, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void to_h_k(const float * x, __half * y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}

// one block per row, one thread per (head, pair) of q and of k
__global__ void rope_k(float * qkv, const int * pos, int heads, int hd, int ld, float base) {
    const int r = blockIdx.x, half = hd / 2, quarter = hd / 4;
    float * row = qkv + int64_t(r) * ld;
    for (int t = threadIdx.x; t < 2 * heads * half; t += blockDim.x) {
        const int m = t / (heads * half), h = (t / half) % heads, j = t % half;  // m: 0 = q, 1 = k
        const float p = float(pos[2 * r + (j < quarter ? 0 : 1)]);
        const int e = j < quarter ? j : j - quarter;
        const float theta = p * powf(base, -2.f * float(e) / float(half));
        float s, c;
        sincosf(theta, &s, &c);
        float * v = row + m * heads * hd + h * hd;
        const float x0 = v[j], x1 = v[j + half];
        v[j] = x0 * c - x1 * s;
        v[j + half] = x0 * s + x1 * c;
    }
}

__global__ void softmax_k(float * s, int cols) {
    __shared__ float sh[32];
    float * r = s + int64_t(blockIdx.x) * cols;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, r[i]);
    m = block_max(m, sh);
    float z = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float e = __expf(r[i] - m);
        r[i] = e;
        z += e;
    }
    const float inv = 1.f / block_sum(z, sh);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) r[i] *= inv;
}

__global__ void scatter_k(__nv_bfloat16 * x, const int * tab, const __nv_bfloat16 * img, int h) {
    const int src = tab[4 * blockIdx.x];
    if (src < 0) return;
    for (int i = threadIdx.x; i < h; i += blockDim.x) x[int64_t(blockIdx.x) * h + i] = img[int64_t(src) * h + i];
}

unsigned blocks(int64_t n) { return unsigned((n + 255) / 256); }

}  // namespace

void vis_layer_norm(const float * x, const float * w, const float * b, float * y, int n, int d, float eps, cudaStream_t s) {
    layer_norm_k<<<n, 256, 0, s>>>(x, w, b, y, d, eps);
}
void vis_bias(float * y, const float * b, int n, int d, bool gelu, cudaStream_t s) {
    bias_k<<<blocks(int64_t(n) * d), 256, 0, s>>>(y, b, int64_t(n) * d, d, gelu);
}
void vis_add(float * y, const float * x, int64_t n, cudaStream_t s) { add_k<<<blocks(n), 256, 0, s>>>(y, x, n); }
void vis_f32_to_h(const float * x, __half * y, int64_t n, cudaStream_t s) { to_h_k<<<blocks(n), 256, 0, s>>>(x, y, n); }
void vis_rope(float * qkv, const int * pos, int n, int heads, int hd, int ld, float base, cudaStream_t s) {
    rope_k<<<n, 256, 0, s>>>(qkv, pos, heads, hd, ld, base);
}
void mm_scatter(__nv_bfloat16 * x, const int * tab, int n, const __nv_bfloat16 * img, int h, cudaStream_t s) {
    scatter_k<<<n, 256, 0, s>>>(x, tab, img, h);
}
void vis_softmax_rows(float * s, int rows, int cols, cudaStream_t st) { softmax_k<<<rows, 256, 0, st>>>(s, cols); }

}  // namespace tqllm_cuda
