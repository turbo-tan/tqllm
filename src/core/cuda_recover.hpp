#pragma once
// Start-of-request recovery for a long-running server. A failed CUDA API call (an out-of-memory cudaMalloc, say) leaves a
// non-sticky error pending; code that checks with cudaPeekAtLastError (QSA, CUB) then reports it on every later request,
// so one failed request broke the server for good. A sticky error (a kernel fault) cannot be cleared: it is returned,
// and the context is lost.
#include <cuda_runtime.h>

namespace tqllm {
inline cudaError_t clear_stale_cuda_error() {
    if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return e;
    (void) cudaGetLastError();
    return cudaSuccess;
}
}  // namespace tqllm
namespace tqllm_cuda {
// TQLLM_DEBUG=cuda_fault:N (tqserve): a kernel writes through a bad address on the N-th request, a sticky error (the
// CUDA context is lost), as a kernel fault after an out-of-memory launch did on a 3090 (11-14, Xid 31)
void debug_device_fault(cudaStream_t stream = nullptr);
}  // namespace tqllm_cuda
