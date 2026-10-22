#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {
// A placement/precision policy, never a router mask. Original expert ids are preserved.
inline int validate_hot_map(const std::vector<int32_t> & map, int layers, int experts) {
    if (layers <= 0 || experts <= 1 || map.size() != size_t(layers) * experts)
        throw std::invalid_argument("hot map: invalid shape");
    int common = -1;
    for (int l = 0; l < layers; ++l) {
        std::vector<bool> seen(size_t(experts), false); int count = 0;
        for (int e = 0; e < experts; ++e) {
            const int v = map[size_t(l) * experts + e];
            if (v == -1) continue;
            if (v < 0 || v >= experts || seen[size_t(v)]) throw std::invalid_argument("hot map: invalid or duplicate index");
            seen[size_t(v)] = true; ++count;
        }
        if (count == 0 || count == experts) throw std::invalid_argument("hot map: requires both hot and cold experts");
        for (int i = 0; i < count; ++i) if (!seen[size_t(i)]) throw std::invalid_argument("hot map: indices must be contiguous");
        if (common >= 0 && count != common) throw std::invalid_argument("hot map: unequal layer sizes");
        common = count;
    }
    return common;
}
// Precision membership and cache residency are independent; zero preserves the original capacity.
inline int hot_cache_slots(int hot_count, int requested) {
    if (hot_count < 1 || requested < 0 || requested > hot_count)
        throw std::invalid_argument("hot_slots must be zero or1..hot-map count");
    return requested ? requested : hot_count;
}
inline bool expert_is_hot(const std::vector<int32_t> & map, int layer, int experts, int expert) {
    if (experts <= 0 || layer < 0 || expert < 0 || expert >= experts || size_t(layer) >= map.size() / size_t(experts))
        throw std::out_of_range("hot map: expert out of range: layer=" + std::to_string(layer) + " expert=" + std::to_string(expert) + " experts=" + std::to_string(experts) + " map_size=" + std::to_string(map.size()));
    return map[size_t(layer) * experts + expert] >= 0;
}
} // namespace tqllm
