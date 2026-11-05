#pragma once
// Independent per-block work (GGUF loaders repacking 256-weight blocks) over all CPU cores: contiguous ranges, one thread
// each, every index exactly once. Single-threaded these loops took ~40 s of a 60 s 27B Q4_K_M load on a 5950X (11-02).
#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace tqllm {
template <class F> void parallel_blocks(int64_t n, F && f, int64_t min_per_thread = 4096) {
    const int64_t hw = std::max<int64_t>(1, std::thread::hardware_concurrency());
    const int64_t nt = std::max<int64_t>(1, std::min(hw, n / std::max<int64_t>(1, min_per_thread)));
    if (nt <= 1) {
        for (int64_t i = 0; i < n; ++i) f(i);
        return;
    }
    std::vector<std::thread> th;
    th.reserve(size_t(nt));
    for (int64_t t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            const int64_t a = n * t / nt, b = n * (t + 1) / nt;
            for (int64_t i = a; i < b; ++i) f(i);
        });
    for (auto & x : th) x.join();
}
}  // namespace tqllm
