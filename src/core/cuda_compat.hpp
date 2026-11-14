#pragma once
// CUDA runtime calls whose signature changed between toolkits: the version branches live here, not at the call sites.
// Built and tested with CUDA 13.0 (Linux, Docker); the Windows CI builds with 12.8.
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace tqllm_cuda {

// n copies dst[i] <- src[i] (bytes[i] each) on `stream`, sources read in stream order. CUDA 12.8+: one batched call
// (13.0: 8 arguments; 12.8 / 12.9: 9, with non-const sources and a failure index); older: one cudaMemcpyAsync per copy.
inline cudaError_t memcpy_batch_async(void ** dst, const void * const * src, const size_t * bytes, size_t n,
                                      cudaStream_t stream, bool prefer_overlap = false) {
    if (n == 0) return cudaSuccess;
#if CUDART_VERSION >= 12080
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    if (prefer_overlap) attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t first = 0;
#if CUDART_VERSION >= 13000
    return cudaMemcpyBatchAsync(dst, src, bytes, n, &attr, &first, 1, stream);
#else
    std::vector<void *> src_mut(n);
    for (size_t i = 0; i < n; ++i) src_mut[i] = const_cast<void *>(src[i]);
    size_t fail_idx = 0;
    return cudaMemcpyBatchAsync(dst, src_mut.data(), const_cast<size_t *>(bytes), n, &attr, &first, 1, &fail_idx, stream);
#endif
#else
    (void) prefer_overlap;
    for (size_t i = 0; i < n; ++i)
        if (const cudaError_t e = cudaMemcpyAsync(dst[i], src[i], bytes[i], cudaMemcpyDefault, stream); e != cudaSuccess) return e;
    return cudaSuccess;
#endif
}

// The capture status of `stream` and, while capturing, the nodes the next captured work will depend on.
// CUDA 13.0: 7 arguments (with edge data); 12.x: 6.
inline cudaError_t stream_capture_deps(cudaStream_t stream, cudaStreamCaptureStatus * status, const cudaGraphNode_t ** deps,
                                       size_t * ndeps) {
#if CUDART_VERSION >= 13000
    return cudaStreamGetCaptureInfo(stream, status, nullptr, nullptr, deps, nullptr, ndeps);
#else
    return cudaStreamGetCaptureInfo(stream, status, nullptr, nullptr, deps, ndeps);
#endif
}

}  // namespace tqllm_cuda
