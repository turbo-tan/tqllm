// Shared FP32 kernels (common_kernels.cuh): the GLM-5.3 engine's originals, unchanged.
#include "common_kernels.cuh"
#include <cfloat>
#include <stdexcept>
#include <string>

namespace tqllm_cuda::common {
namespace {
void check_launch(const char * what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + cudaGetErrorString(e));
}
int blocks(int n, int t) { return (n + t - 1) / t; }

// one warp per row
__global__ void mv_f32_k(const float * w, const float * x, float * y, int rows, int cols) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31, t = blockIdx.y;
    if (r >= rows) return;
    const float * row = w + size_t(r) * cols, * xt = x + size_t(t) * cols;
    float acc = 0.f;
    for (int c = lane; c < cols; c += 32) acc += row[c] * xt[c];
    acc = warp_sum(acc);
    if (lane == 0) y[size_t(t) * rows + r] = acc;
}

// grid (rows, splits, n): block sums its column slice of one row
__global__ void mv_splitk_k(const float * w, const float * x, float * partial, int rows, int cols, int splits) {
    __shared__ float red[32];
    const int r = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int per = (cols + splits - 1) / splits, c0 = sp * per, c1 = min(cols, c0 + per);
    const float * row = w + size_t(r) * cols, * xt = x + size_t(t) * cols;
    float acc = 0.f;
    for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x) acc += row[c] * xt[c];
    acc = block_sum(acc, red);
    if (threadIdx.x == 0) partial[(size_t(t) * rows + r) * splits + sp] = acc;
}

__global__ void splitk_sum_k(const float * partial, float * y, int count, int splits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = 0.f;
    for (int sp = 0; sp < splits; ++sp) v += partial[size_t(i) * splits + sp];
    y[i] = v;
}

__global__ void gather_cols_k(const float * in, int stride, int off, int width, int count, float * out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = in[size_t(i / width) * stride + off + i % width];
}

__global__ void bf16_to_f32_k(const __nv_bfloat16 * x, float * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]);
}

__global__ void f32_to_half_k(const float * x, __half * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

// one block per row
__global__ void rmsnorm_k(const float * x, const float * w, float * y, __half * yh, int n, float eps) {
    __shared__ float red[32];
    const float * xr = x + size_t(blockIdx.x) * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    const float s = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] * s * (w ? w[i] : 1.f);
        if (y) y[size_t(blockIdx.x) * n + i] = v;
        if (yh) yh[size_t(blockIdx.x) * n + i] = __float2half(v);
    }
}

// short rows (per-head q / k norms): one warp per row, 8 rows per block
__global__ void rmsnorm_warp_k(const float * x, const float * w, float * y, __half * yh, int n, float eps, int rows) {
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    const float * xr = x + size_t(r) * n;
    float ss = 0.f;
    for (int i = lane; i < n; i += 32) ss += xr[i] * xr[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float s = rsqrtf(ss / n + eps);
    for (int i = lane; i < n; i += 32) {
        const float v = xr[i] * s * (w ? w[i] : 1.f);
        if (y) y[size_t(r) * n + i] = v;
        if (yh) yh[size_t(r) * n + i] = __float2half(v);
    }
}

__global__ void add_k(float * a, const float * b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void argmax_k(const float * x, int n, int * out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    x += size_t(blockIdx.x) * n; out += blockIdx.x;
    float v = -FLT_MAX;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > v) { v = x[i]; idx = i; }
    bv[threadIdx.x] = v; bi[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) {
        if (threadIdx.x < s) {
            const float o = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) { bv[threadIdx.x] = o; bi[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = bi[0];
}
}  // namespace

void mv_f32(const float * w, const float * x, float * y, int rows, int cols, int n, cudaStream_t s) {
    mv_f32_k<<<dim3(blocks(rows, 8), n), 256, 0, s>>>(w, x, y, rows, cols);
    check_launch("common kernel: ");
}
void mv_f32_splitk(const float * w, const float * x, float * y, float * partial, int rows, int cols, int n, int splits,
                       cudaStream_t s) {
    mv_splitk_k<<<dim3(rows, splits, n), 256, 0, s>>>(w, x, partial, rows, cols, splits);
    check_launch("common kernel: ");
    splitk_sum_k<<<blocks(rows * n, 256), 256, 0, s>>>(partial, y, rows * n, splits);
    check_launch("common kernel: ");
}
void gather_cols(const float * in, int stride, int off, int width, int n, float * out, cudaStream_t s) {
    gather_cols_k<<<blocks(width * n, 256), 256, 0, s>>>(in, stride, off, width, width * n, out);
    check_launch("common kernel: ");
}
void bf16_to_f32(const __nv_bfloat16 * x, float * y, int n, cudaStream_t s) {
    bf16_to_f32_k<<<blocks(n, 256), 256, 0, s>>>(x, y, n);
    check_launch("common kernel: ");
}
void f32_to_half(const float * x, __half * y, int n, cudaStream_t s) {
    f32_to_half_k<<<blocks(n, 256), 256, 0, s>>>(x, y, n);
    check_launch("common kernel: ");
}
void rmsnorm(const float * x, const float * w, float * y, __half * yh, int n, float eps, cudaStream_t s, int rows) {
    if (n <= 256) rmsnorm_warp_k<<<(rows + 7) / 8, 256, 0, s>>>(x, w, y, yh, n, eps, rows);
    else rmsnorm_k<<<rows, 1024, 0, s>>>(x, w, y, yh, n, eps);
    check_launch("common kernel: ");
}
void rmsnorm_rows(const float * x, const float * w, float * y, int rows, int n, float eps, cudaStream_t s) {
    rmsnorm_k<<<rows, 128, 0, s>>>(x, w, y, nullptr, n, eps);
    check_launch("common kernel: ");
}
void add(float * a, const float * b, int n, cudaStream_t s) {
    add_k<<<blocks(n, 256), 256, 0, s>>>(a, b, n);
    check_launch("common kernel: ");
}
void argmax(const float * x, int n, int * out, cudaStream_t s) {
    argmax_k<<<1, 1024, 0, s>>>(x, n, out);
    check_launch("common kernel: ");
}
void argmax_rows(const float * x, int n, int rows, int * out, cudaStream_t s) {
    argmax_k<<<rows, 1024, 0, s>>>(x, n, out);
    check_launch("common kernel: ");
}
}  // namespace tqllm_cuda::common

namespace {
__global__ void debug_device_fault_k(int * p) { p[threadIdx.x] = 1; }
}
namespace tqllm_cuda {
void debug_device_fault(cudaStream_t stream) { debug_device_fault_k<<<1, 32, 0, stream>>>(reinterpret_cast<int *>(uintptr_t(0x10))); }
}  // namespace tqllm_cuda
