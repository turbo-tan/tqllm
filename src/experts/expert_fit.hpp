#pragma once
// Per-layer expert slots that fit a byte budget. Layers differ in expert size (Flash-Next IQ3_S: 1.46-2.54 MiB), so
// the fit is in bytes, not slot counts: every layer's count is scaled by one factor (the routing-trace shape is kept)
// and the largest factor whose banks fit is taken.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tqllm {
inline size_t bank_bytes(const std::vector<int> & caps, const std::vector<size_t> & block) {
    size_t total = 0;
    for (size_t l = 0; l < caps.size() && l < block.size(); ++l) total += size_t(caps[l]) * block[l];
    return total;
}
// caps unchanged when they fit; otherwise each layer gets max(min_cap, floor(cap * f)) for the largest f that fits.
// When even min_cap everywhere does not fit, that is returned and the caller must refuse the configuration.
inline std::vector<int> fit_layer_slots(const std::vector<int> & caps, const std::vector<size_t> & block, size_t bytes,
                                        int min_cap) {
    if (bank_bytes(caps, block) <= bytes) return caps;
    auto scaled = [&](double f) {
        std::vector<int> out(caps.size());
        for (size_t l = 0; l < caps.size(); ++l) out[l] = std::min(caps[l], std::max(min_cap, int(std::floor(caps[l] * f))));
        return out;
    };
    double lo = 0.0, hi = 1.0;
    for (int i = 0; i < 40; ++i) {
        const double mid = 0.5 * (lo + hi);
        (bank_bytes(scaled(mid), block) <= bytes ? lo : hi) = mid;
    }
    return scaled(lo);
}
}  // namespace tqllm
