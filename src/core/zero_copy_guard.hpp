#pragma once
// Kernels must not dereference host memory on PCIe GPUs (G2, owner 11-12: zero-copy reads beside the copy engine's DMA
// fault those cards). Launchers whose pointers must be device memory reject a host pointer there; GPUs that read host
// memory coherently (GB10, C2C) pass. Bulk data goes by the copy engine. Deliberate exceptions (not guarded): small
// scattered gathers whose row ids exist only on the device, inside captured graphs: the token embeddings
// (Qwen35Engine::embed_multi, a few rows per verify / draft pass) and the int8 LM head's re-score rows
// (i8h_rescore_argmax, 16 candidate rows per token), when the fit places those tables in host memory.
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tqllm_cuda {
inline bool host_memory_coherent() {
    static const bool v = [] { int d = 0, on = 0; cudaGetDevice(&d);
        return cudaDeviceGetAttribute(&on, cudaDevAttrPageableMemoryAccessUsesHostPageTables, d) == cudaSuccess && on; }();
    return v;
}
inline bool is_device_memory(const void * p) {
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return false; }
    return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
}
inline void require_device_memory(const void * p, const char * what) {
    if (!p || host_memory_coherent() || is_device_memory(p)) return;
    throw std::invalid_argument(std::string(what) + ": a kernel would access host memory (zero-copy) on a PCIe GPU");
}
}  // namespace tqllm_cuda
