#pragma once
// Counted device allocations: the memory fit reads dev_bytes() where the driver cannot report free memory (WSL / WDDM).
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace tqllm_cuda {
inline std::atomic<size_t> & dev_bytes_counter() { static std::atomic<size_t> n{0}; return n; }
inline std::unordered_map<void *, size_t> & dev_sizes() { static std::unordered_map<void *, size_t> m; return m; }
inline std::mutex & dev_mutex() { static std::mutex m; return m; }
inline cudaError_t dev_malloc_v(void ** p, size_t n) {
    const cudaError_t e = cudaMalloc(p, n);
    if (e == cudaSuccess && n) {
        std::lock_guard<std::mutex> lock(dev_mutex());
        dev_sizes()[*p] = n;
        dev_bytes_counter() += n;
    }
    return e;
}
template <class T> cudaError_t dev_malloc(T ** p, size_t n) { return dev_malloc_v(reinterpret_cast<void **>(p), n); }
inline cudaError_t dev_free(void * p) {
    if (p) {
        std::lock_guard<std::mutex> lock(dev_mutex());
        const auto it = dev_sizes().find(p);
        if (it != dev_sizes().end()) dev_bytes_counter() -= it->second, dev_sizes().erase(it);
    }
    return cudaFree(p);
}
inline size_t dev_bytes() { return dev_bytes_counter().load(); }
}  // namespace tqllm_cuda
