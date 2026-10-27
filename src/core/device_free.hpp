#pragma once
// Free device memory a plan can count on. Unified-memory GPUs (GB10) report the page cache as used in cudaMemGetInfo
// (121 GiB total, 2.9 GiB "free" with 97 GiB available), so their free memory comes from /proc/meminfo MemAvailable.
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

namespace tqllm {
// MemAvailable from /proc/meminfo text, in bytes (0 if absent).
inline size_t meminfo_available(const std::string & text) {
    const auto at = text.find("MemAvailable:");
    if (at == std::string::npos) return 0;
    return size_t(std::stoull(text.substr(at + 13))) * 1024;
}
// Host RAM a plan can count on: physical memory, capped by the container's cgroup limit (v2 memory.max, v1
// memory.limit_in_bytes; "max" or an absurd v1 value = no limit). A RunPod pod reported 944 GB physical with a 117 GB
// limit, so a pinned expert pack sized from the physical figure pushed the cgroup into reclaim.
inline size_t host_ram_limit(size_t physical, const std::string & cgroup_limit) {
    if (cgroup_limit.empty() || cgroup_limit.compare(0, 3, "max") == 0) return physical;
    size_t limit = 0;
    try { limit = size_t(std::stoull(cgroup_limit)); } catch (...) { return physical; }
    return limit > 0 && limit < physical ? limit : physical;
}
// this process's host RAM limit: physical memory capped by its cgroup (v2, then v1)
inline size_t host_ram_bytes() {
    const size_t physical = size_t(sysconf(_SC_PHYS_PAGES)) * size_t(sysconf(_SC_PAGESIZE));
    for (const char * f : {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"}) {
        std::ifstream in(f); if (!in) continue;
        return host_ram_limit(physical, std::string(std::istreambuf_iterator<char>(in), {}));
    }
    return physical;
}
inline size_t usable_free_bytes(size_t cuda_free, size_t total, bool integrated, const std::string & meminfo) {
    return integrated ? std::min(total, std::max(cuda_free, meminfo_available(meminfo))) : cuda_free;
}
}  // namespace tqllm
