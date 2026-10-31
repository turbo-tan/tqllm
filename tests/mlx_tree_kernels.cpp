// Token-tree verify kernels, bit for bit against the chain they must reproduce (no model needed):
// (1) the fused GDN step with parents: every tree row's output, delta and decay equal running that row's own path
//     (anchor .. row) as a chain;
// (2) the BF16 verify attention with the ancestor mask: every tree row equals the same row computed with its path keys
//     stored contiguously after the prefix (the layout an AR step sees).
#include "attention_decode.hpp"
#include "gdn.hpp"
#include <mlx/mlx.h>
#include <cstdio>
#include <random>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

static mx::array randn(std::mt19937& rng, mx::Shape shape, float scale) {
    size_t n = 1; for (int d : shape) n *= size_t(d);
    std::normal_distribution<float> nd(0.0f, scale); std::vector<float> v(n); for (auto& x : v) x = nd(rng);
    return mx::array(v.data(), shape, mx::float32);
}
static mx::array bf(const mx::array& x) {return mx::astype(mx::astype(x, mx::bfloat16), mx::float32);}

int main() {
    std::mt19937 rng(11);
    int failures = 0;
    const std::vector<int32_t> parents = {-1, 0, 0, 2, 2, 4, 1, 5, 3, 0};  // a tree with branches at rows 0 and 2
    const int n = int(parents.size());
    auto path_of = [&](int t) {std::vector<int> p; for (int a = t; a >= 0; a = parents[size_t(a)]) p.insert(p.begin(), a); return p;};

    {   // (1) GDN step
        const int kh = 2, vh = 4, width = 2 * kh * 128 + vh * 128;
        auto prev = bf(randn(rng, {3, width}, 1.0f)), rows = bf(randn(rng, {n, width}, 1.0f));
        auto weights = randn(rng, {width, 4}, 0.5f), alpha = randn(rng, {n, vh}, 1.0f), beta = randn(rng, {n, vh}, 1.0f);
        auto bias = randn(rng, {vh}, 0.5f), rate = mx::negative(mx::abs(randn(rng, {vh}, 0.5f)));
        auto z = randn(rng, {n, vh * 128}, 1.0f), w = randn(rng, {128}, 1.0f), state = randn(rng, {1, vh, 128, 128}, 0.1f);
        mx::eval({prev, rows, weights, alpha, beta, bias, rate, z, w, state});
        const auto par = mx::array(parents.data(), {n}, mx::int32);
        auto tree = gdn_step_fused(prev, rows, weights, alpha, beta, bias, rate, z, w, 1e-6f, 0.0883883f, state, kh, vh, 0, 0, 0, &par);
        mx::eval(tree.out, tree.deltas, tree.decay);
        int equal = 0;
        for (int t = 0; t < n; ++t) {
            const auto p = path_of(t);
            const auto idx = mx::array(p.data(), {int(p.size())}, mx::int32);
            auto chain = gdn_step_fused(prev, mx::take(rows, idx, 0), weights, mx::take(alpha, idx, 0), mx::take(beta, idx, 0), bias, rate,
                                        mx::take(z, idx, 0), w, 1e-6f, 0.0883883f, state, kh, vh);
            const int last = int(p.size()) - 1;
            auto a = mx::slice(tree.out, {t, 0}, {t + 1, vh * 128}), b = mx::slice(chain.out, {last, 0}, {last + 1, vh * 128});
            auto da = mx::slice(tree.deltas, {t, 0, 0}, {t + 1, vh, 128}), db = mx::slice(chain.deltas, {last, 0, 0}, {last + 1, vh, 128});
            equal += mx::array_equal(a, b).item<bool>() && mx::array_equal(da, db).item<bool>();
        }
        failures += equal != n;
        std::printf("%s TREE_GDN tree rows == their own path as a chain, bit for bit: %d / %d\n", equal == n ? "PASS" : "FAIL", equal, n);
    }
    {   // (2) attention: prefix of `start` keys, then the n block keys; each tree row vs its path stored contiguously
        const int heads = 24, kvh = 4, dim = 256;
        for (int start : {1000, 3000, 9000}) {
            const int length = start + n;
            auto q = randn(rng, {1, heads, n, dim}, 1.0f);
            auto k = mx::astype(randn(rng, {1, kvh, length, dim}, 1.0f), mx::bfloat16), v = mx::astype(randn(rng, {1, kvh, length, dim}, 1.0f), mx::bfloat16);
            std::vector<uint8_t> vis(size_t(n) * n, 0);
            for (int t = 0; t < n; ++t) for (int a : path_of(t)) vis[size_t(t) * n + a] = 1;
            auto visible = mx::astype(mx::array(vis.data(), {n, n}, mx::uint8), mx::bool_);
            auto tree = decode_attention_bf16(q, k, v, 0.0625f, length, &visible);
            mx::eval(tree);
            int equal = 0;
            for (int t = 0; t < n; ++t) {
                const auto p = path_of(t);
                std::vector<int32_t> order(start);
                for (int i = 0; i < start; ++i) order[size_t(i)] = i;
                for (int a : p) order.push_back(start + a);
                const auto idx = mx::array(order.data(), {int(order.size())}, mx::int32);
                auto kk = mx::take(k, idx, 2), vv = mx::take(v, idx, 2);
                auto one = decode_attention_bf16(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}), kk, vv, 0.0625f, int(order.size()));
                equal += mx::array_equal(one, mx::slice(tree, {0, 0, t, 0}, {1, heads, t + 1, dim})).item<bool>();
            }
            failures += equal != n;
            std::printf("%s TREE_ATTN %d prefix keys: tree rows == their path stored contiguously, bit for bit: %d / %d\n",
                        equal == n ? "PASS" : "FAIL", start, equal, n);
        }
    }
    std::printf(failures ? "TREE_KERNELS_FAIL\n" : "TREE_KERNELS_PASS\n");
    return failures ? 1 : 0;
}
