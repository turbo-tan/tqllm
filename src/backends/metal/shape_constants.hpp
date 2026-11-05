#pragma once
// Arrays that depend only on a few shape ints (index ramps, causal masks): made once and shared, so the layers of a
// forward read one array instead of each building its own (TQLLM_DEBUG=count_nodes counts the launches), and a later
// forward of the same shape reads it already evaluated. The most recent kKeep keys are kept.
#include <mlx/mlx.h>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace tqllm::mlx_backend {
inline mlx::core::array shape_constant(const char* what, const std::vector<int>& key, const std::function<mlx::core::array()>& make) {
    constexpr size_t kKeep = 32;
    struct Entry {std::string what; std::vector<int> key; mlx::core::array value;};
    static thread_local auto* cache = new std::deque<Entry>();  // never destroyed: an array must not outlive MLX's allocator
    for (const auto& e : *cache) if (e.key == key && e.what == what) return e.value;
    cache->push_front({what, key, make()});
    if (cache->size() > kKeep) cache->pop_back();
    return cache->front().value;
}
}  // namespace tqllm::mlx_backend
