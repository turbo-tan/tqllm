// Token-tree verify kernels, bit for bit against the chain they must reproduce (no model needed):
// (1) the fused GDN step with parents: every tree row's output, delta and decay equal running that row's own path
//     (anchor .. row) as a chain;
// (2) the BF16 verify attention with the ancestor mask: every tree row equals the same row computed with its path keys
//     stored contiguously after the prefix (the layout an AR step sees).
// Both for a 10-row tree and for 32 rows (the 17-32 row verify: the GDN step's device-scratch path, attention as two
// 16-row calls through causal_attention_rows): a 32-row tree and a 32-row chain.
#include "attention_rows.hpp"
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

// Verify one tree through the shared mask cache and count the rows that equal their own path stored contiguously.
static int attention_rows_matching_paths(std::mt19937& rng, const std::vector<int32_t>& parents, int start) {
    const int n = int(parents.size()), heads = 24, kvh = 4, dim = 256;
    const int length = start + n;
    auto q = randn(rng, {1, heads, n, dim}, 1.0f);
    auto k = mx::astype(randn(rng, {1, kvh, length, dim}, 1.0f), mx::bfloat16), v = mx::astype(randn(rng, {1, kvh, length, dim}, 1.0f), mx::bfloat16);
    auto path_of = [&](int t) {std::vector<int> p; for (int a = t; a >= 0; a = parents[size_t(a)]) p.insert(p.begin(), a); return p;};
    std::vector<uint8_t> vis(size_t(n) * n, 0);
    for (int t = 0; t < n; ++t) for (int a : path_of(t)) vis[size_t(t) * n + a] = 1;
    auto visible = mx::astype(mx::array(vis.data(), {n, n}, mx::uint8), mx::bool_);
    auto tree = causal_attention_rows(q, k, v, 0.0625f, length, &visible, &parents, true);
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
    return equal;
}

int main() {
    std::mt19937 rng(11);
    int failures = 0;
    std::vector<int32_t> tree32{-1};  // 32 rows in depth-first order: branches at several depths, a 13-deep spine
    for (int r = 1; r < 32; ++r) tree32.push_back(r < 14 ? r - 1 : r % 5 == 0 ? 0 : r % 3 == 0 ? r - 9 : r - 1);
    std::vector<int32_t> chain32(32);
    for (int r = 0; r < 32; ++r) chain32[size_t(r)] = r - 1;
    for (const auto& parents : {std::vector<int32_t>{-1, 0, 0, 2, 2, 4, 1, 5, 3, 0}, tree32, chain32}) {
    const int n = int(parents.size());
    std::printf("-- %d rows%s\n", n, n == 32 && parents == chain32 ? " (chain)" : "");
    auto path_of = [&](int t) {std::vector<int> p; for (int a = t; a >= 0; a = parents[size_t(a)]) p.insert(p.begin(), a); return p;};

    {   // (1) GDN step
        const int kh = 2, vh = 4, width = 2 * kh * 128 + vh * 128;
        auto prev = bf(randn(rng, {3, width}, 1.0f)), rows = bf(randn(rng, {n, width}, 1.0f));
        // alpha / beta projected inside the kernel from the input rows (the served path: x and the ab weights)
        const int hidden = 512;
        auto weights = randn(rng, {width, 4}, 0.5f), input = randn(rng, {n, hidden}, 1.0f), ab = randn(rng, {2 * vh, hidden}, 0.05f);
        auto bias = randn(rng, {vh}, 0.5f), rate = mx::negative(mx::abs(randn(rng, {vh}, 0.5f)));
        auto z = randn(rng, {n, vh * 128}, 1.0f), w = randn(rng, {128}, 1.0f), state = randn(rng, {1, vh, 128, 128}, 0.1f);
        mx::eval({prev, rows, weights, input, ab, bias, rate, z, w, state});
        // small evaluated inputs reach a custom kernel as `constant` buffers; the model passes fresh device results, so
        // every call here does too (one multiply each, not evaluated): an input evaluated by an earlier call would be
        // constant again
        auto fresh = [](const mx::array& x) {return mx::multiply(x, mx::array(1.0f, x.dtype()));};
        const auto par = mx::array(parents.data(), {n}, mx::int32);
        auto tree = gdn_step_fused(fresh(prev), fresh(rows), fresh(weights), fresh(input), fresh(ab), fresh(bias), fresh(rate), fresh(z),
                                   fresh(w), 1e-6f, 0.0883883f, fresh(state), kh, vh, 0, 0, 0, &par);
        mx::eval(tree.out, tree.deltas, tree.decay);
        int equal = 0;
        for (int t = 0; t < n; ++t) {
            const auto p = path_of(t);
            const auto idx = mx::array(p.data(), {int(p.size())}, mx::int32);
            auto chain = gdn_step_fused(fresh(prev), fresh(mx::take(rows, idx, 0)), fresh(weights), fresh(mx::take(input, idx, 0)),
                                        fresh(ab), fresh(bias), fresh(rate), fresh(mx::take(z, idx, 0)), fresh(w), 1e-6f,
                                        0.0883883f, fresh(state), kh, vh);
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
            auto tree = causal_attention_rows(q, k, v, 0.0625f, length, &visible, &parents, true);
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
    }
    {   // (3) the mask cache must never serve a stale mask: two different trees of the same size at the same lengths,
        // verified back to back, then 16 and 32 rows alternating
        const std::vector<int32_t> a16{-1, 0, 0, 1, 3, 1, 5, 2, 7, 0, 9, 4, 11, 6, 13, 8};
        const std::vector<int32_t> b16{-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14};
        for (const auto* t : {&a16, &b16, &a16, &tree32, &chain32, &tree32, &a16}) {
            const int n = int(t->size()), equal = attention_rows_matching_paths(rng, *t, 3000);
            failures += equal != n;
            std::printf("%s MASK_CACHE %d rows after another tree at the same lengths: tree rows == their path, bit for bit: %d / %d\n",
                        equal == n ? "PASS" : "FAIL", n, equal, n);
        }
    }
    std::printf(failures ? "TREE_KERNELS_FAIL\n" : "TREE_KERNELS_PASS\n");
    return failures ? 1 : 0;
}
