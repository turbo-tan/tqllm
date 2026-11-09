#include "dspine_kernels.cuh"
#include <cfloat>
#include <cmath>

namespace tqllm_cuda {
namespace {
__device__ float sum256(float v) {
    __shared__ float sums[8];
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8 ? sums[threadIdx.x] : 0.f;
    if (threadIdx.x < 32)
        for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    if (threadIdx.x == 0) sums[0] = v;
    __syncthreads();
    return sums[0];
}
__global__ void read_norm(const __nv_bfloat16* x, __nv_bfloat16* out, float* rms, int H, float eps) {
    const int row = blockIdx.x;
    float sum = 0;
    for (int h = threadIdx.x; h < H; h += 256) { const float v = float(x[size_t(row) * H + h]); sum += v * v; }
    const float r = sqrtf(fmaxf(sum256(sum) / H, FLT_MIN));
    if (threadIdx.x == 0) rms[row] = r;
    for (int h = threadIdx.x; h < H; h += 256) out[size_t(row) * H + h] = __float2bfloat16_rn(float(x[size_t(row) * H + h]) / (r + eps));
}
__global__ void normalize_codes(__nv_bfloat16* codes, const __nv_bfloat16* book, const uint32_t* tokens,
                                int C, int B, float eps) {
    const int row = blockIdx.x;
    float sum = 0;
    for (int k = threadIdx.x; k < C; k += 256) { const float v = float(codes[size_t(row) * C + k]); sum += v * v; }
    const float scale = sqrtf(float(C)) / (sqrtf(sum256(sum)) + eps);
    for (int k = threadIdx.x; k < C; k += 256)
        codes[size_t(row) * C + k] = __float2bfloat16_rn(row % B ? float(codes[size_t(row) * C + k]) * scale
            : sqrtf(float(C)) * float(book[size_t(tokens[row - row % B]) * C + k]));
}
__global__ void message(const __nv_bfloat16* packed, const float* bias, __nv_bfloat16* out, int R, int B, int N) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(N) * R) return;
    const int row = int(i / R), r = int(i % R);
    float v = 0;
    if (row % B) {
        const float q = float(packed[size_t(row) * 3 * R + r]);
        const float s = float(packed[size_t(row - 1) * 3 * R + R + r]);
        v = float(packed[size_t(row - 1) * 3 * R + 2 * R + r]) / (1.f + expf(-(q + s + bias[r])));
    }
    out[i] = __float2bfloat16_rn(v);
}
__global__ void add(__nv_bfloat16* x, const __nv_bfloat16* delta, const float* rms, int H, int B, int N) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(N) * H || (i / H) % B == 0) return;
    x[i] = __float2bfloat16_rn(float(x[i]) + rms[i / H] * float(delta[i]));
}
__global__ void predecessor_codes(const uint32_t* candidates, const uint32_t* tokens,
                                  const __nv_bfloat16* book, __nv_bfloat16* out, int C, int B, int N) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(N) * 16 * C) return;
    const int row = int(i / (16 * C)), candidate = int((i / C) % 16), k = int(i % C);
    const uint32_t tok = row % B <= 1 ? tokens[row - row % B] : candidates[size_t(row - 1) * 16 + candidate];
    out[i] = __float2bfloat16_rn(sqrtf(float(C)) * float(book[size_t(tok) * C + k]));
}
__global__ void refine_message(const __nv_bfloat16* own, const __nv_bfloat16* pred,
                               const float* bias, __nv_bfloat16* out, int R, int N) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= size_t(N) * 16 * R) return;
    const size_t row = i / R, r = i % R;
    const float z = float(own[(row / 16) * 3 * R + r]) + float(pred[row * 2 * R + r]) + bias[r];
    out[i] = __float2bfloat16_rn(float(pred[row * 2 * R + R + r]) / (1.f + expf(-z)));
}
__global__ void refine_norm(const __nv_bfloat16* raw, const __nv_bfloat16* delta, const float* rms,
                            const float* norm, __nv_bfloat16* out, int H, float eps) {
    const size_t row = blockIdx.x, source = row / 16;
    float sum = 0;
    for (int h = threadIdx.x; h < H; h += 256) {
        const float v = float(__float2bfloat16_rn(float(raw[source * H + h]) + rms[source] * float(delta[row * H + h])));
        sum += v * v;
    }
    const float inv = rsqrtf(sum256(sum) / H + eps);
    for (int h = threadIdx.x; h < H; h += 256) {
        const float v = float(__float2bfloat16_rn(float(raw[source * H + h]) + rms[source] * float(delta[row * H + h])));
        out[row * H + h] = __float2bfloat16_rn(float(__float2bfloat16_rn(v * inv)) * norm[h]);
    }
}
__global__ void score(const int8_t* head, const __half* scales, const __nv_bfloat16* x,
                      const uint32_t* candidates, float* out, int H) {
    const size_t row = blockIdx.x, pred = blockIdx.y, candidate = blockIdx.z;
    const size_t tok = candidates[row * 16 + candidate];
    float sum = 0;
    for (int h = threadIdx.x; h < H; h += 256)
        sum += float(head[tok * H + h]) * __half2float(scales[tok * (H / 16) + h / 16]) * float(x[(row * 16 + pred) * H + h]);
    sum = sum256(sum);
    if (threadIdx.x == 0) out[row * 256 + pred * 16 + candidate] = sum;
}
}
void dspine_read_norm(const __nv_bfloat16* x, __nv_bfloat16* y, float* rms, int H, int N, float e, cudaStream_t s) {
    read_norm<<<N, 256, 0, s>>>(x, y, rms, H, e);
}
void dspine_codes(__nv_bfloat16* x, const __nv_bfloat16* b, const uint32_t* t, int C, int B, int N, float e, cudaStream_t s) {
    normalize_codes<<<N, 256, 0, s>>>(x, b, t, C, B, e);
}
void dspine_message(const __nv_bfloat16* x, const float* b, __nv_bfloat16* y, int R, int B, int N, cudaStream_t s) {
    message<<<(size_t(N) * R + 255) / 256, 256, 0, s>>>(x, b, y, R, B, N);
}
void dspine_add(__nv_bfloat16* x, const __nv_bfloat16* d, const float* r, int H, int B, int N, cudaStream_t s) {
    add<<<(size_t(N) * H + 255) / 256, 256, 0, s>>>(x, d, r, H, B, N);
}
void dspine_predecessor_codes(const uint32_t* c, const uint32_t* t, const __nv_bfloat16* b, __nv_bfloat16* y,
                              int C, int B, int N, cudaStream_t s) {
    predecessor_codes<<<(size_t(N) * 16 * C + 255) / 256, 256, 0, s>>>(c, t, b, y, C, B, N);
}
void dspine_refine_message(const __nv_bfloat16* q, const __nv_bfloat16* p, const float* b, __nv_bfloat16* y,
                           int R, int N, cudaStream_t s) {
    refine_message<<<(size_t(N) * 16 * R + 255) / 256, 256, 0, s>>>(q, p, b, y, R, N);
}
void dspine_refine_norm(const __nv_bfloat16* x, const __nv_bfloat16* d, const float* r, const float* w,
                        __nv_bfloat16* y, int H, int N, float e, cudaStream_t s) {
    refine_norm<<<N * 16, 256, 0, s>>>(x, d, r, w, y, H, e);
}
void dspine_score(const int8_t* w, const __half* s, const __nv_bfloat16* x, const uint32_t* c,
                   float* y, int H, int N, cudaStream_t stream) {
    score<<<dim3(N, 16, 16), 256, 0, stream>>>(w, s, x, c, y, H);
}
}
