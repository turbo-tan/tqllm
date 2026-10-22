// row_topk returns row-contiguous ids / values: row r's k entries at r * k are that row's true top k.
#include "sampling_topk.hpp"
#include <mlx/transforms.h>
#include <algorithm>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

namespace mx = mlx::core;
int main() {
    const int rows = 5, vocab = 3000, k = 20;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    std::vector<float> v(size_t(rows) * vocab);
    for (auto& x : v) x = nd(rng);
    auto logits = mx::array(v.data(), {rows, vocab}, mx::float32);
    auto [idx, val] = tqllm::mlx_backend::row_topk(logits, k);
    mx::eval(idx, val);
    const auto* pi = idx.data<uint32_t>();
    const auto* pv = val.data<float>();
    bool ok = true;
    for (int r = 0; r < rows; ++r) {
        std::vector<int> order(vocab);
        for (int t = 0; t < vocab; ++t) order[t] = t;
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) {return v[size_t(r) * vocab + a] > v[size_t(r) * vocab + b];});
        std::set<uint32_t> want(order.begin(), order.begin() + k), got(pi + r * k, pi + (r + 1) * k);
        bool vals = true;
        for (int t = 0; t < k; ++t) vals &= pv[r * k + t] == v[size_t(r) * vocab + pi[r * k + t]];
        const bool pass = want == got && vals;
        ok &= pass;
        std::printf("TOPK row %d %s\n", r, pass ? "PASS" : "FAIL");
    }
    std::printf(ok ? "TOPK_PASS\n" : "TOPK_FAIL\n");
    return ok ? 0 : 1;
}
