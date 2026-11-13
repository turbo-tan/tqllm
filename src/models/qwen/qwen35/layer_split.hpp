#pragma once
// Layer split across GPUs (one rank per GPU, NCCL between ranks; docs plan multi-gpu-20261109): rank r owns the
// contiguous layers [lo, hi). Pure arithmetic, no CUDA: tests/layer_split.cpp.
#include <cstdint>
#include <stdexcept>
#include <string>

namespace tqllm {
struct LayerRange {
    uint32_t lo = 0, hi = 0;
    bool owns(uint32_t l) const { return l >= lo && l < hi; }
    bool first() const { return lo == 0; }
};
// world ranks over `layers`; `split_at` (two ranks) places rank 1's first layer, -1 = half (rounded down for rank 0)
inline LayerRange layer_range(uint32_t layers, int rank, int world, int split_at) {
    if (world < 1 || rank < 0 || rank >= world || layers < uint32_t(world)) throw std::invalid_argument("layer split: bad rank / world");
    if (world == 1) return {0, layers};
    if (world == 2 && split_at >= 0) {
        if (split_at < 1 || uint32_t(split_at) >= layers) throw std::invalid_argument("--split-at must be within 1 .. layers - 1");
        return rank == 0 ? LayerRange{0, uint32_t(split_at)} : LayerRange{uint32_t(split_at), layers};
    }
    const uint32_t lo = uint32_t(uint64_t(layers) * rank / world), hi = uint32_t(uint64_t(layers) * (rank + 1) / world);
    return {lo, hi};
}
}  // namespace tqllm
