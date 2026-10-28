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
// cgroup v2 path of this process from /proc/self/cgroup text ("0::/user.slice/x.scope" -> "/user.slice/x.scope")
inline std::string cgroup_v2_path(const std::string & proc_self_cgroup) {
    const auto at = proc_self_cgroup.find("0::");
    if (at == std::string::npos) return "";
    const auto end = proc_self_cgroup.find('\n', at);
    return proc_self_cgroup.substr(at + 3, end == std::string::npos ? std::string::npos : end - at - 3);
}
// this process's host RAM limit: physical memory capped by the tightest memory.max on its cgroup path (v2; a container
// with its own cgroup namespace sees "/"), else the v1 limit
inline size_t host_ram_bytes() {
    const auto read = [](const std::string & f) { std::ifstream in(f); return in ? std::string(std::istreambuf_iterator<char>(in), {}) : std::string(); };
    size_t ram = size_t(sysconf(_SC_PHYS_PAGES)) * size_t(sysconf(_SC_PAGESIZE));
    std::string path = cgroup_v2_path(read("/proc/self/cgroup"));
    for (;;) {
        ram = host_ram_limit(ram, read("/sys/fs/cgroup" + (path == "/" ? std::string() : path) + "/memory.max"));
        if (path.empty() || path == "/") break;
        const auto slash = path.rfind('/'); path = slash == 0 ? "/" : path.substr(0, slash);
    }
    return host_ram_limit(ram, read("/sys/fs/cgroup/memory/memory.limit_in_bytes"));
}
inline size_t usable_free_bytes(size_t cuda_free, size_t total, bool integrated, const std::string & meminfo) {
    return integrated ? std::min(total, std::max(cuda_free, meminfo_available(meminfo))) : cuda_free;
}
}  // namespace tqllm
