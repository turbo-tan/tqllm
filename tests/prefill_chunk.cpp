// Prefill chunk rule: GB10 (128 GB unified) at long context must not fall back to the 24 GB-card 512-token chunks;
// 24 GB and 16 GB cards keep their sizes.
#include "prefill_chunk.hpp"

#include <cstdio>
#include <string>

namespace {
int failures = 0;
void expect(int got, int want, const char * what) {
    if (got != want) { std::fprintf(stderr, "FAIL %s: %d, want %d\n", what, got, want); ++failures; }
}
constexpr size_t GiB = size_t(1) << 30;
// Qwen3.8-27B, 16 attention layers, tq3 KV: 2 x 4 heads x 66 bytes per position and layer
constexpr size_t kv_tq3_per_pos = 16 * 2 * 4 * 66;
}

int main() {
    using tqllm::prefill_chunk;
    // GB10, TQ_XS loaded (~12 GB of 128), 540K context, all KV on the device: old rule 512
    expect(prefill_chunk(540672, 128 * GiB, 110 * GiB, 540672 * kv_tq3_per_pos), 8192, "GB10 540K device KV");
    expect(prefill_chunk(270336, 128 * GiB, 110 * GiB, 270336 * kv_tq3_per_pos), 8192, "GB10 270K");
    expect(prefill_chunk(8192, 128 * GiB, 110 * GiB, 8192 * kv_tq3_per_pos), 8192, "GB10 8K");
    expect(prefill_chunk(270336, 48 * GiB, 20 * GiB, 270336 * kv_tq3_per_pos), 4096, "48 GB card, 20 GB free");
    // RTX 3090 (24 GB, ~11 GB free after the model): unchanged
    expect(prefill_chunk(32768, 24 * GiB, 11 * GiB, 32768 * kv_tq3_per_pos), 4096, "3090 32K");
    expect(prefill_chunk(131072, 24 * GiB, 11 * GiB, 131072 * kv_tq3_per_pos), 2048, "3090 128K");
    expect(prefill_chunk(262144, 24 * GiB, 11 * GiB, 262144 * kv_tq3_per_pos), 512, "3090 256K");
    // RTX 5060 Ti (16 GB, ~3.5 GB free): unchanged
    expect(prefill_chunk(204800, 16 * GiB, size_t(3.5 * GiB), 204800 * kv_tq3_per_pos), 512, "5060 Ti 200K");
    expect(prefill_chunk(32768, 16 * GiB, size_t(3.5 * GiB), 32768 * kv_tq3_per_pos), 2048, "5060 Ti 32K");
    // KV larger than free memory: no underflow
    expect(prefill_chunk(262144, 24 * GiB, 2 * GiB, 20 * GiB), 512, "KV beyond free");
    // GB10 unified memory: cudaMemGetInfo says 2.9 GiB free (page cache counted as used), /proc/meminfo has the truth
    const std::string gb10 = "MemTotal:       127535116 kB\nMemFree:         3120496 kB\nMemAvailable:   101820244 kB\n";
    const size_t avail = tqllm::meminfo_available(gb10);
    if (avail != size_t(101820244) * 1024) { std::fprintf(stderr, "FAIL meminfo_available: %zu\n", avail); ++failures; }
    expect(prefill_chunk(270336, 121 * GiB, size_t(2.9 * GiB), 270336 * kv_tq3_per_pos), 512, "GB10 cudaMemGetInfo free only");
    expect(prefill_chunk(270336, 121 * GiB, avail, 270336 * kv_tq3_per_pos), 8192, "GB10 MemAvailable");
    if (tqllm::meminfo_available("MemTotal: 1 kB\n") != 0) { std::fprintf(stderr, "FAIL meminfo without MemAvailable\n"); ++failures; }
    if (failures) return 1;
    std::puts("prefill_chunk: ok");
    return 0;
}
