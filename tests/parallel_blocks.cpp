#include "parallel_blocks.hpp"
#include <atomic>
#include <cstdio>
#include <vector>

int main() {
    int bad = 0;
    for (const int64_t n : {0L, 1L, 7L, 4095L, 4096L, 100000L, 1000003L})
        for (const int64_t grain : {1L, 2L, 4096L}) {
            std::vector<std::atomic<int>> hit(static_cast<size_t>(n));
            for (auto & h : hit) h = 0;
            tqllm::parallel_blocks(n, [&](int64_t i) { ++hit[size_t(i)]; }, grain);
            for (int64_t i = 0; i < n; ++i)
                if (hit[size_t(i)] != 1) { std::fprintf(stderr, "FAIL n %ld grain %ld: index %ld hit %d times\n", long(n), long(grain), long(i), int(hit[size_t(i)])); ++bad; break; }
        }
    if (bad) return 1;
    std::printf("PASS parallel_blocks\n");
    return 0;
}
