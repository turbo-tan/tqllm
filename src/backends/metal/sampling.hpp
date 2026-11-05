#pragma once
// Host-side pieces of speculative sampling with a point-mass draft (no MLX dependency, unit-tested on any host).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace tqllm::mlx_backend {
// (logit / temperature, id) candidates of one row (the row's top k) -> the filtered distribution as (probability, id),
// most likely first: softmax over the k candidates, then the smallest prefix reaching top_p, renormalized.
inline std::vector<std::pair<double, uint32_t>> filtered_distribution(const float* scaled, const uint32_t* ids, int k, double top_p) {
    std::vector<std::pair<double, uint32_t>> d;
    d.reserve(size_t(k));
    for (int t = 0; t < k; ++t) d.push_back({double(scaled[t]), ids[t]});
    std::sort(d.begin(), d.end(), [](const auto& x, const auto& y) {return x.first > y.first;});
    const double peak = d[0].first;  // read once: the loop below overwrites d[0]
    double z = 0;
    for (auto& e : d) {e.first = std::exp(e.first - peak); z += e.first;}
    double mass = 0;
    size_t keep = d.size();
    for (size_t t = 0; t < d.size(); ++t) {mass += d[t].first / z; if (mass >= top_p) {keep = t + 1; break;}}
    d.resize(keep);
    z = 0;
    for (auto& e : d) z += e.first;
    for (auto& e : d) e.first /= z;
    return d;
}
// one draw from d, excluding `skip` (renormalized); UINT32_MAX excludes nothing
inline uint32_t draw_excluding(const std::vector<std::pair<double, uint32_t>>& d, uint32_t skip, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    double z = 0;
    for (const auto& e : d) if (e.second != skip) z += e.first;
    if (z <= 0) return d[0].second;
    const double u = unit(rng) * z;
    double c = 0;
    uint32_t last = d[0].second;
    for (const auto& e : d) {
        if (e.second == skip) continue;
        last = e.second;
        c += e.first;
        if (u < c) return e.second;
    }
    return last;
}
// The point-mass draft x at a row with target distribution d: returns (accepted, token). Accept with probability p(x);
// on rejection the token comes from p without x. The emitted token is distributed exactly as p.
inline std::pair<bool, uint32_t> speculative_pick(const std::vector<std::pair<double, uint32_t>>& d, uint32_t x, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    double px = 0;
    for (const auto& e : d) if (e.second == x) px = e.first;
    if (unit(rng) < px) return {true, x};
    return {false, draw_excluding(d, x, rng)};
}
}  // namespace tqllm::mlx_backend
