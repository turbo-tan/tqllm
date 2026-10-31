// Example plugins for ABI 2 (include/tqllm/plugin.h), one shared library per variant (CMake builds each with a define):
//   EX_NEW     a new format P8 (type 2001): GGUF-style blocks of 32 values, FP16 scale + 32 int8
//   EX_PATCH   an exact patch of the I8H dequant (same arithmetic as the built-in kernel)
//   EX_BROKEN  P8 whose CPU reference is wrong on one value of every block: the contract must refuse it
//   EX_INEXACT an I8H dequant patch that claims exact but scales by 1.001: the exactness check must refuse it
#include "tqllm/plugin.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>

namespace {
constexpr int kP8Values = 32, kP8Bytes = 34;
#if defined(EX_NEW) || defined(EX_BROKEN)

__global__ void p8_dequant_k(const uint8_t * blocks, __half * out, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t * b = blocks + (i / kP8Values) * kP8Bytes;
    __half d; memcpy(&d, b, 2);
    out[i] = __float2half(__half2float(d) * float(int8_t(b[2 + i % kP8Values])));
}
__global__ void p8_matvec_k(const uint8_t * blocks, int rows, int cols, const __half * x, __nv_bfloat16 * y) {
    const int r = blockIdx.x, j = blockIdx.y;
    float acc = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const int64_t i = int64_t(r) * cols + c;
        const uint8_t * b = blocks + (i / kP8Values) * kP8Bytes;
        __half d; memcpy(&d, b, 2);
        acc += __half2float(d) * float(int8_t(b[2 + i % kP8Values])) * __half2float(x[int64_t(j) * cols + c]);
    }
    __shared__ float part[128];
    part[threadIdx.x] = acc; __syncthreads();
    for (int s = 64; s > 0; s >>= 1) { if (int(threadIdx.x) < s) part[threadIdx.x] += part[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) y[int64_t(j) * rows + r] = __float2bfloat16(part[0]);
}
#else
__global__ void i8h_dequant_k(const int8_t * w, const __half * sc, __half * out, int64_t n, float bias) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bias == 1.f ? __hmul(__int2half_rn(w[i]), sc[i / 16]) : __float2half(float(w[i]) * __half2float(sc[i / 16]) * bias);
}

#endif
#if defined(EX_NEW) || defined(EX_BROKEN)
int32_t p8_cpu(const void * blocks, float * out, uint64_t n) {
    const uint8_t * b = static_cast<const uint8_t *>(blocks);
    for (uint64_t i = 0; i < n; ++i) {
        const uint8_t * blk = b + (i / kP8Values) * kP8Bytes;
        __half d; std::memcpy(&d, blk, 2);
        out[i] = __half2float(d) * float(int8_t(blk[2 + i % kP8Values]));
#ifdef EX_BROKEN
        if (i % kP8Values == 5) out[i] += 1.f + std::abs(out[i]);
#endif
    }
    return 0;
}
int32_t p8_gpu_dequant(const void * blocks, const void *, int32_t rows, int32_t cols, void * out, void * stream) {
    const int64_t n = int64_t(rows) * cols;
    p8_dequant_k<<<unsigned((n + 255) / 256), 256, 0, cudaStream_t(stream)>>>(static_cast<const uint8_t *>(blocks), static_cast<__half *>(out), n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
int32_t p8_gpu_matvec(const void * blocks, const void *, int32_t rows, int32_t cols, const void * x, void * y, int32_t ncols, void * stream) {
    p8_matvec_k<<<dim3(unsigned(rows), unsigned(ncols)), 128, 0, cudaStream_t(stream)>>>(static_cast<const uint8_t *>(blocks), rows, cols,
                                                                                       static_cast<const __half *>(x), static_cast<__nv_bfloat16 *>(y));
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
#else
int32_t i8h_gpu_dequant(const void * blocks, const void * aux, int32_t rows, int32_t cols, void * out, void * stream) {
    const int64_t n = int64_t(rows) * cols;
#ifdef EX_INEXACT
    const float bias = 1.001f;
#else
    const float bias = 1.f;
#endif
    i8h_dequant_k<<<unsigned((n + 255) / 256), 256, 0, cudaStream_t(stream)>>>(static_cast<const int8_t *>(blocks), static_cast<const __half *>(aux),
                                                                              static_cast<__half *>(out), n, bias);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
#endif
}  // namespace

extern "C" int32_t tq_plugin_init_v2(const tq_host_v2 * host, tq_error * error) {
    if (!host || host->abi_version != TQ_ABI_VERSION_2) return 1;
#if defined(EX_NEW) || defined(EX_BROKEN)
    static const tq_format_v2 f{sizeof(tq_format_v2), TQ_ABI_VERSION_2, "P8", 2001, kP8Values, kP8Bytes, 0, 0, 0,
                                p8_cpu, p8_gpu_dequant, p8_gpu_matvec};
#else
    static const tq_format_v2 f{sizeof(tq_format_v2), TQ_ABI_VERSION_2, "I8H dequant patch", TQ_PATCH_I8H, 0, 0, 0, 0, 1,
                                nullptr, i8h_gpu_dequant, nullptr};
#endif
    return host->register_format(host->context, &f, error);
}
