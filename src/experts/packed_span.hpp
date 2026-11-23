#pragma once
// No CUDA: tests and CPU-only builds include this without packed_experts.hpp (which needs cuda_bf16.h).
#include <cstdint>
#include <utility>
#include <vector>

namespace tqllm {

// The byte span [begin, end) of layers [lo, hi) in a pack laid out by `base` / `block` (layers in order, `experts` each).
inline std::pair<uint64_t, uint64_t> packed_span(const std::vector<uint64_t> & base, const std::vector<uint64_t> & block,
                                                 int experts, int lo, int hi) {
    if (lo >= hi || base.empty()) return {0, 0};
    const int last = hi - 1;
    return {base[size_t(lo)], base[size_t(last)] + block[size_t(last)] * uint64_t(experts)};
}

}  // namespace tqllm
