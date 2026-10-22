#pragma once
#include "device_free.hpp"

#include <cuda_runtime.h>

#include <fstream>
#include <iterator>

namespace tqllm {
// cudaMemGetInfo corrected for unified memory (device_free.hpp); 0 on error
inline size_t device_free_bytes(size_t * total_out = nullptr) {
    size_t fr = 0, tot = 0;
    if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) { cudaGetLastError(); fr = tot = 0; }
    int dev = 0, integrated = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev);
    std::string text;
    if (integrated) { std::ifstream mi("/proc/meminfo"); text.assign(std::istreambuf_iterator<char>(mi), std::istreambuf_iterator<char>()); }
    if (total_out) *total_out = tot;
    return usable_free_bytes(fr, tot, integrated != 0, text);
}
}  // namespace tqllm
