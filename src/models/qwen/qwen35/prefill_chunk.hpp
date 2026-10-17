#pragma once
// Prefill chunk length (tokens per GEMM pass). The context rule keeps 24 GB cards within memory at long contexts;
// devices with >= 16 GB left after the device-resident KV take 4096 at any context, >= 32 GB (GB10, 128 GB unified) 8192.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace tqllm {
// MemAvailable from /proc/meminfo text, in bytes (0 if absent). Unified-memory GPUs (GB10) report the page cache as
// used in cudaMemGetInfo (121 GiB total, 2.9 GiB "free" with 97 GiB available), so their free memory comes from here.
inline size_t meminfo_available(const std::string & text) {
    const auto at = text.find("MemAvailable:");
    if (at == std::string::npos) return 0;
    return size_t(std::stoull(text.substr(at + 13))) * 1024;
}
inline int prefill_chunk(int64_t context, size_t total_bytes, size_t free_bytes, size_t kv_device_bytes) {
    int n = context <= 131072 ? 2048 : context <= 196608 ? 1024 : 512;  // 256K + 2048 would pass 24 GB
    if (context <= 65536 && total_bytes >= (size_t(20) << 30)) n = 4096;  // 27B on the 3090: +4-5%, peak +1 GB at 32K
    // GB10 TQ_XS prefill, 4096 -> 8192: 8K 1170 -> 1208, 32K 1068 -> 1136, 128K 818 -> 863, 262K 629 -> 659 tok/s
    const size_t room = free_bytes > kv_device_bytes ? free_bytes - kv_device_bytes : 0;
    if (room >= (size_t(16) << 30)) n = std::max(n, 4096);
    if (room >= (size_t(32) << 30)) n = std::max(n, 8192);
    return n;
}
}  // namespace tqllm
