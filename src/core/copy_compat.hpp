#pragma once
// Host -> device copy of many small regions in one call, whatever the runtime offers: cudaMemcpyBatchAsync (CUDA 12.8+,
// two signatures) with stream-ordered, overlap-preferring attributes; one cudaMemcpyAsync per region elsewhere (older
// CUDA, AMD). Callers never branch on the runtime version or the backend (AGENTS.md rule 13).
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "backend_caps.hpp"

namespace tqllm_cuda {
inline cudaError_t copy_batch_h2d(std::vector<void *> & dst, const std::vector<const void *> & src, std::vector<size_t> & bytes,
                                  cudaStream_t stream) {
    if (dst.empty()) return cudaSuccess;
#if CUDART_VERSION >= 12080
    if constexpr (tqllm::caps::kBatchCopyAttributes) {
        cudaMemcpyAttributes attr = {};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
        size_t first = 0;
#if CUDART_VERSION >= 13000
        return cudaMemcpyBatchAsync(dst.data(), src.data(), bytes.data(), dst.size(), &attr, &first, 1, stream);
#else
        // 12.8 / 12.9: 9 arguments, non-const void ** src and an extra size_t * failIdx before the stream
        std::vector<void *> src_mut(src.size());
        for (size_t i = 0; i < src.size(); ++i) src_mut[i] = const_cast<void *>(src[i]);
        size_t fail_idx = 0;
        return cudaMemcpyBatchAsync(dst.data(), src_mut.data(), bytes.data(), dst.size(), &attr, &first, 1, &fail_idx, stream);
#endif
    }
#endif
    for (size_t i = 0; i < dst.size(); ++i)
        if (const cudaError_t e = cudaMemcpyAsync(dst[i], src[i], bytes[i], cudaMemcpyHostToDevice, stream); e != cudaSuccess) return e;
    return cudaSuccess;
}
}  // namespace tqllm_cuda
