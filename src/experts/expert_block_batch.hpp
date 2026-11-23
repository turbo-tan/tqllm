#pragma once
#include "cpu_pool.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

namespace tqllm {
// Assemble independent expert blocks in caller order; a failed batch must not be consumed.
template <class Load>
void expert_block_batch(const int * ids, size_t n, char * dst, size_t block, Load load) {
    if (!n) return;
    if (n > size_t(std::numeric_limits<int>::max())) throw std::runtime_error("too many expert blocks");
    const int threads = int(std::min<size_t>(n, std::max(2u, std::min(16u, std::thread::hardware_concurrency() / 2))));
    CpuPool pool(threads);
    pool.run(int(n), [&](int i) {
        const auto parts = load(ids[i]);
        char * d = dst + size_t(i) * block;
        size_t at = 0;
        for (const auto & p : parts) {
            if (p.size() > block - at) throw std::runtime_error("on-demand expert block size mismatch");
            std::memcpy(d + at, p.data(), p.size()); at += p.size();
        }
        if (at != block) throw std::runtime_error("on-demand expert block size mismatch");
    });
}
}  // namespace tqllm
