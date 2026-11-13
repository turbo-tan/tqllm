#pragma once
// Prefill chunk length (tokens per GEMM pass). The context rule keeps 24 GB cards within memory at long contexts;
// devices with >= 16 GB left after the device-resident KV take 4096 at any context, >= 32 GB (GB10, 128 GB unified) 8192.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "device_free.hpp"

namespace tqllm {
inline int prefill_chunk(int64_t context, size_t total_bytes, size_t free_bytes, size_t kv_device_bytes) {
    int n = context <= 131072 ? 2048 : context <= 196608 ? 1024 : 512;  // 256K + 2048 would pass 24 GB
    if (context <= 65536 && total_bytes >= (size_t(20) << 30)) n = 4096;  // 27B on the 3090: +4-5%, peak +1 GB at 32K
    // GB10 TQ_XS prefill, 4096 -> 8192: 8K 1170 -> 1208, 32K 1068 -> 1136, 128K 818 -> 863, 262K 629 -> 659 tok/s
    const size_t room = free_bytes > kv_device_bytes ? free_bytes - kv_device_bytes : 0;
    if (room >= (size_t(16) << 30)) n = std::max(n, 4096);
    if (room >= (size_t(32) << 30)) n = std::max(n, 8192);
    return n;
}

// Where a prompt keeps its state for the next turn (0: nowhere): on the chunk grid a cold prefill uses (multiples of
// `chunk` from 0), at least `tail` tokens before the end (the next turn's template re-renders the end), after `from` (the
// position this prefill resumed at). A later turn that resumes there prefills exactly the chunks a cold prefill would,
// so its reply equals the reply with the cache off (the kernels pick their path by chunk width).
// Chunks restart at every message start (starts: sorted positions of the chat template's message-start token, > 0):
// a prefill's chunks are a property of its prompt, the same for a cold prefill and for one resumed at a message start.
// The end of the chunk that begins at t.
inline uint32_t chunk_end(const std::vector<uint32_t> & starts, uint32_t t, uint32_t prompt, uint32_t chunk) {
    const auto next = std::upper_bound(starts.begin(), starts.end(), t);
    const uint32_t seg = next == starts.begin() ? 0 : *(next - 1);
    uint32_t end = seg + ((t - seg) / chunk + 1) * chunk;
    if (next != starts.end()) end = std::min(end, *next);
    return std::min(end, prompt);
}
// Where a chat prompt keeps its state for the next turn: the start of its last message (the generation prompt), which
// the next turn's prompt shares and where every prefill of it starts a chunk; 0 when that is not past `from`. Prompts
// without message starts keep the grid rule (kept_state_at).
inline uint32_t kept_state_msg(const std::vector<uint32_t> & starts, uint32_t prompt, uint32_t from) {
    const auto it = std::lower_bound(starts.begin(), starts.end(), prompt);
    if (it == starts.begin()) return 0;
    const uint32_t last = *(it - 1);
    return last > from ? last : 0;
}
inline uint32_t kept_state_at(uint32_t prompt, uint32_t from, uint32_t chunk, uint32_t tail) {
    if (chunk == 0 || prompt < tail) return 0;
    const uint32_t grid = (prompt - tail) / chunk * chunk;
    return grid > from ? grid : 0;
}
}  // namespace tqllm
