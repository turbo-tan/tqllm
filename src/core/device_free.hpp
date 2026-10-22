#pragma once
// Free device memory a plan can count on. Unified-memory GPUs (GB10) report the page cache as used in cudaMemGetInfo
// (121 GiB total, 2.9 GiB "free" with 97 GiB available), so their free memory comes from /proc/meminfo MemAvailable.
#include <algorithm>
#include <cstddef>
#include <string>

namespace tqllm {
// MemAvailable from /proc/meminfo text, in bytes (0 if absent).
inline size_t meminfo_available(const std::string & text) {
    const auto at = text.find("MemAvailable:");
    if (at == std::string::npos) return 0;
    return size_t(std::stoull(text.substr(at + 13))) * 1024;
}
inline size_t usable_free_bytes(size_t cuda_free, size_t total, bool integrated, const std::string & meminfo) {
    return integrated ? std::min(total, std::max(cuda_free, meminfo_available(meminfo))) : cuda_free;
}
}  // namespace tqllm
