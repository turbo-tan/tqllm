// decode_attention_exact (attention_exact.hpp), no model, random BF16 cache:
// (1) a 16-row chain verify equals 16 one-row (AR) calls bit for bit;
// (2) every row of a random 16-row token tree equals the one-row call at its AR position over a cache whose block holds
//     the row's path in depth order (what AR would have written), bit for bit;
// at prefixes inside the first tile, across tile and chunk boundaries, past 4K keys, across the chunk-size step at 32K
// (32846: row 0 on 1024-key chunks, deeper rows on 2048) and on 4096-key chunks (70000).
#include <mlx/mlx.h>

#include <cstdio>
#include <random>
#include <vector>

#include "attention_exact.hpp"
#include "tree_device.hpp"

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

namespace {
int failures = 0;
mx::array row(const mx::array& out, int t) {   // [1, heads, n, dim] -> [heads, dim]
    return mx::reshape(mx::slice(out, {0, 0, t, 0}, {1, out.shape(1), t + 1, out.shape(3)}), {out.shape(1), out.shape(3)});
}
}  // namespace

int main() {
    if (!matrix_unit_available()) {std::printf("ATTENTION_EXACT_SKIP (needs the matrix unit)\n"); return 0;}
    const int heads = 24, kvh = 4, dim = 256, cap = 70144;
    const float scale = 0.0625f;
    mx::random::seed(3);
    auto k = mx::astype(mx::random::normal({1, kvh, cap, dim}), mx::bfloat16);
    auto v = mx::astype(mx::random::normal({1, kvh, cap, dim}), mx::bfloat16);
    auto q = mx::astype(mx::astype(mx::random::normal({1, heads, 16, dim}), mx::bfloat16), mx::float32);
    mx::eval(k, v, q);
    std::mt19937 rng(11);
    for (int prefix : {5, 40, 63, 64, 200, 1000, 1017, 1030, 2047, 4500, 32846, 70000}) {
        // (1) chain
        auto chain = decode_attention_exact(q, k, v, scale, prefix + 16);
        int same = 0;
        for (int t = 0; t < 16; ++t) {
            auto one = decode_attention_exact(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}), k, v, scale, prefix + t + 1);
            same += mx::array_equal(row(chain, t), row(one, 0)).item<bool>();
        }
        failures += same != 16;
        std::printf("%s ATTENTION_EXACT prefix %d: chain rows == one-row calls %d / 16\n", same == 16 ? "PASS" : "FAIL", prefix, same);
        // (2) tree
        std::vector<int32_t> parents(16, -1);
        for (int t = 1; t < 16; ++t) parents[size_t(t)] = int32_t(rng() % uint32_t(t));
        const auto dp = mx::array(parents.data(), {16}, mx::int32);
        auto [visible, depth] = tree_ancestors(dp);
        auto tree = decode_attention_exact(q, k, v, scale, prefix + 16, &visible);
        same = 0;
        for (int t = 0; t < 16; ++t) {
            std::vector<int32_t> path;
            for (int a = t; a >= 0; a = parents[size_t(a)]) path.insert(path.begin(), a);
            // the cache AR would hold: the path's keys at prefix, prefix + 1, ...
            std::vector<int32_t> order(static_cast<size_t>(cap));
            for (int i = 0; i < cap; ++i) order[size_t(i)] = i;
            for (size_t d = 0; d < path.size(); ++d) order[size_t(prefix) + d] = prefix + path[d];
            const auto idx = mx::array(order.data(), {cap}, mx::int32);
            auto ka = mx::take(k, idx, 2), va = mx::take(v, idx, 2);
            const int x = prefix + int(path.size()) - 1;
            auto one = decode_attention_exact(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}), ka, va, scale, x + 1);
            same += mx::array_equal(row(tree, t), row(one, 0)).item<bool>();
        }
        failures += same != 16;
        std::printf("%s ATTENTION_EXACT prefix %d: tree rows == AR calls on their paths %d / 16\n", same == 16 ? "PASS" : "FAIL", prefix, same);
    }
    std::printf(failures ? "ATTENTION_EXACT_FAIL\n" : "ATTENTION_EXACT_PASS\n");
    return failures ? 1 : 0;
}
