#pragma once
#include <cstdint>

namespace tqllm {
inline uint64_t batch_graph_seed(bool sampled, unsigned stash) {
    // Mix the mode before layout fields; otherwise sampled bit 0 aliases the state index.
    return (1469598103934665603ull ^ uint64_t(sampled) ^ (uint64_t(stash) << 40)) * 1099511628211ull;
}
}  // namespace tqllm
