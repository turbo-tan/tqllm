#pragma once
// Prefill GEMM input rows: n tokens run as n rounded up to 8 (cuBLAS picks its 16816 kernels), capped at the buffer.
// The padding rows [n, padded) are read by every GEMM path (FP8 / int8 activation quantisation, FP16 cuBLAS), so they are
// zeroed: otherwise they hold whatever the buffer held before (compute-sanitizer initcheck), and on GB10 the 27B's first
// token with a drafter loaded changed between server starts.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace tqllm {
inline int gemm_pad_rows(int n, int cap) { return std::min(cap, (n + 7) / 8 * 8); }
inline cudaError_t zero_gemm_padding(__half * x, int n, int padded, int cols, cudaStream_t s) {
    if (padded <= n) return cudaSuccess;
    return cudaMemsetAsync(x + size_t(n) * cols, 0, size_t(padded - n) * cols * sizeof(__half), s);
}
}  // namespace tqllm
