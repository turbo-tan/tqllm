// QSA rows of decode_attention_exact (attention_qsa.hpp), no model, random BF16 cache, block keys and indexer queries:
// (0) the indexer projection + norm of 16 rows == each row alone, bit for bit;
// (1) a 16-row chain verify equals 16 one-row (AR) calls bit for bit;
// (2) every row of a random 16-row tree equals the one-row call at its AR position over a cache holding its path, bit
//     for bit;
// (3) the selection: 512 ascending blocks below T / 4, and no unselected block scores above a selected one (scores
//     recomputed here, so near-ties within 1e-4 are allowed);
// (4) a sparse row equals softmax attention over its selected keys and its loop keys (reference in FP32, 2e-3);
// at prefixes where all rows are sparse and where the kQsaFrom switch falls between row 0 and the deeper rows;
// (5) over the int8 cache (decode_attention_qsa_int8): chain rows == one-row calls bit for bit, and a row within 2e-2
//     of the BF16 cache's row (same selection, quantized keys / values).
#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "attention_exact.hpp"
#include "tree_device.hpp"

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

namespace {
int failures = 0;
mx::array row(const mx::array& out, int t) {  // [1, heads, n, dim] -> [heads, dim]
    return mx::reshape(mx::slice(out, {0, 0, t, 0}, {1, out.shape(1), t + 1, out.shape(3)}), {out.shape(1), out.shape(3)});
}
int tof(int x) {return x - 15 > 0 ? (x - 15) / 64 * 64 : 0;}
}  // namespace

int main() {
    if (!segmented_attention_available(256)) {std::printf("ATTENTION_QSA_SKIP (needs the matrix unit)\n"); return 0;}
    const int heads = 24, kvh = 4, dim = 256, cap = kQsaFrom + 8192;
    const float scale = 0.0625f;
    mx::random::seed(5);
    auto k = mx::astype(mx::random::normal({1, kvh, cap, dim}), mx::bfloat16);
    auto v = mx::astype(mx::random::normal({1, kvh, cap, dim}), mx::bfloat16);
    auto q = mx::astype(mx::astype(mx::random::normal({1, heads, 16, dim}), mx::bfloat16), mx::float32);
    auto blocks = mx::random::normal({cap / kQsaBlock, kQsaDim});
    auto iq = mx::random::normal({16, kQsaHeads, kQsaDim});
    mx::eval(k, v, q, blocks, iq);
    std::mt19937 rng(17);
    // (0) the indexer projection + norm: 16 rows at once == each row alone, bit for bit; near an FP32 matmul
    {
        auto x = mx::astype(mx::astype(mx::random::normal({16, 5120}), mx::bfloat16), mx::float32);
        auto w = mx::astype(mx::multiply(mx::random::normal({kQsaDim, 5120}), mx::array(0.02f)), mx::float16);
        auto ones = mx::ones({kQsaDim});
        const auto none = mx::zeros({1}, mx::int32);
        auto all = qsa_norm_rope(qsa_project(x, w, 0, kQsaDim), ones, none, 1e7f, 1e-6f, false, false);
        int same = 0;
        for (int t = 0; t < 16; ++t) {
            auto one = qsa_norm_rope(qsa_project(mx::slice(x, {t, 0}, {t + 1, 5120}), w, 0, kQsaDim), ones, none, 1e7f, 1e-6f, false, false);
            same += mx::array_equal(mx::slice(all, {t, 0}, {t + 1, kQsaDim}), one).item<bool>();
        }
        auto ref = mx::matmul(x, mx::transpose(mx::astype(w, mx::float32)));
        ref = mx::multiply(ref, mx::rsqrt(mx::add(mx::mean(mx::square(ref), -1, true), mx::array(1e-6f))));
        const float err = mx::max(mx::abs(mx::subtract(all, ref))).item<float>();
        const bool ok = same == 16 && err < 1e-3f;
        failures += !ok;
        std::printf("%s ATTENTION_QSA projection: 16 rows == one row at a time %d / 16, vs FP32 matmul + norm %.2e\n",
                    ok ? "PASS" : "FAIL", same, err);
    }
    for (int prefix : {kQsaFrom + 14, kQsaFrom + 3616}) {  // the switch between row 0 and the deeper rows; all rows sparse
        // (1) chain
        const QsaRows all{iq, blocks};
        auto chain = decode_attention_exact(q, k, v, scale, prefix + 16, nullptr, &all);
        int same = 0;
        for (int t = 0; t < 16; ++t) {
            const QsaRows one{mx::slice(iq, {t, 0, 0}, {t + 1, kQsaHeads, kQsaDim}), blocks};
            auto ar = decode_attention_exact(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}), k, v, scale, prefix + t + 1, nullptr, &one);
            same += mx::array_equal(row(chain, t), row(ar, 0)).item<bool>();
        }
        failures += same != 16;
        std::printf("%s ATTENTION_QSA prefix %d: chain rows == one-row calls %d / 16\n", same == 16 ? "PASS" : "FAIL", prefix, same);
        // (2) tree
        std::vector<int32_t> parents(16, -1);
        for (int t = 1; t < 16; ++t) parents[size_t(t)] = int32_t(rng() % uint32_t(t));
        auto [visible, depth] = tree_ancestors(mx::array(parents.data(), {16}, mx::int32));
        auto tree = decode_attention_exact(q, k, v, scale, prefix + 16, &visible, &all);
        same = 0;
        for (int t = 0; t < 16; ++t) {
            std::vector<int32_t> path;
            for (int a = t; a >= 0; a = parents[size_t(a)]) path.insert(path.begin(), a);
            std::vector<int32_t> order(static_cast<size_t>(cap));
            for (int i = 0; i < cap; ++i) order[size_t(i)] = i;
            for (size_t d = 0; d < path.size(); ++d) order[size_t(prefix) + d] = prefix + path[d];
            const auto idx = mx::array(order.data(), {cap}, mx::int32);
            const QsaRows one{mx::slice(iq, {t, 0, 0}, {t + 1, kQsaHeads, kQsaDim}), blocks};
            const int x = prefix + int(path.size()) - 1;
            auto ar = decode_attention_exact(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}), mx::take(k, idx, 2),
                                             mx::take(v, idx, 2), scale, x + 1, nullptr, &one);
            same += mx::array_equal(row(tree, t), row(ar, 0)).item<bool>();
        }
        failures += same != 16;
        std::printf("%s ATTENTION_QSA prefix %d: tree rows == AR calls on their paths %d / 16\n", same == 16 ? "PASS" : "FAIL", prefix, same);
    }
    // (3) / (4) one sparse row, as an AR step at x
    const int x = kQsaFrom + 3616, T = tof(x), nb = T / kQsaBlock;
    const auto plan = exact_rows_plan(x, 1, nullptr);
    const auto q1 = mx::slice(iq, {0, 0, 0}, {1, kQsaHeads, kQsaDim});
    auto sel = qsa_select(mx::concatenate({q1, mx::zeros({15, kQsaHeads, kQsaDim})}, 0), blocks, plan, 1, nb, 2 + kExactTail, kQsaFrom);
    auto sel0 = mx::slice(sel, {0, 0}, {1, kQsaPick});
    // scores as the indexer defines them: sum over heads of relu(q . k), / sqrt(128)
    auto bk = mx::slice(blocks, {0, 0}, {nb, kQsaDim});
    auto scores = mx::multiply(mx::sum(mx::maximum(mx::matmul(mx::reshape(q1, {kQsaHeads, kQsaDim}), mx::transpose(bk)), mx::array(0.0f)), 0),
                               mx::array(0.08838834764831845f));
    mx::eval(sel0, scores);
    std::vector<int32_t> s(kQsaPick);
    std::copy_n(sel0.data<int32_t>(), kQsaPick, s.begin());
    bool ordered = s[0] >= 0;
    for (int i = 1; i < kQsaPick; ++i) ordered = ordered && s[size_t(i)] > s[size_t(i - 1)];
    ordered = ordered && s.back() < nb;
    std::vector<char> picked(size_t(nb), 0);
    float low = 1e30f, high = -1e30f;
    const float* sc = scores.data<float>();
    for (int b : s) if (b >= 0 && b < nb) {picked[size_t(b)] = 1; low = std::min(low, sc[b]);}
    for (int b = 0; b < nb; ++b) if (!picked[size_t(b)]) high = std::max(high, sc[b]);
    const bool top = high <= low + 1e-4f * std::max(1.0f, std::fabs(low));
    failures += !(ordered && top);
    std::printf("%s ATTENTION_QSA select: %d ascending blocks below %d %s, lowest picked %.5f vs highest left %.5f\n",
                ordered && top ? "PASS" : "FAIL", kQsaPick, nb, ordered ? "yes" : "no", low, high);
    // reference attention over the selected keys and the loop keys T .. x
    std::vector<int32_t> keys;
    for (int b : s) for (int j = 0; j < kQsaBlock; ++j) keys.push_back(b * kQsaBlock + j);
    for (int j = T; j <= x; ++j) keys.push_back(j);
    const auto kidx = mx::array(keys.data(), {int(keys.size())}, mx::int32);
    auto kk = mx::astype(mx::take(k, kidx, 2), mx::float32), vv = mx::astype(mx::take(v, kidx, 2), mx::float32);
    auto qq = mx::reshape(mx::astype(mx::astype(mx::slice(q, {0, 0, 0, 0}, {1, heads, 1, dim}), mx::bfloat16), mx::float32), {kvh, 6, dim});
    auto att = mx::softmax(mx::multiply(mx::matmul(qq, mx::transpose(mx::reshape(kk, {kvh, int(keys.size()), dim}), {0, 2, 1})), mx::array(scale)), -1);
    auto ref = mx::reshape(mx::matmul(att, mx::reshape(vv, {kvh, int(keys.size()), dim})), {heads, dim});
    const QsaRows one{q1, blocks};
    auto got = row(decode_attention_exact(mx::slice(q, {0, 0, 0, 0}, {1, heads, 1, dim}), k, v, scale, x + 1, nullptr, &one), 0);
    const float err = mx::max(mx::abs(mx::subtract(got, ref))).item<float>(), mag = mx::max(mx::abs(ref)).item<float>();
    const bool close = err <= 2e-3f * std::max(1.0f, mag);
    failures += !close;
    std::printf("%s ATTENTION_QSA row == softmax over its selected + loop keys: max abs diff %.2e (max %.3f)\n",
                close ? "PASS" : "FAIL", err, mag);
    // (5) int8 cache, chain rows (the int8 KV path verifies chains)
    {
        const auto kv8 = quantize_kv_int8(k, v);
        mx::eval(kv8.k8t, kv8.v8, kv8.kscale, kv8.vscale);
        const int prefix = kQsaFrom + 3616;
        const QsaRows all{iq, blocks};
        auto chain = decode_attention_qsa_int8(q, kv8.k8t, kv8.v8, kv8.kscale, kv8.vscale, scale, prefix + 16, all);
        int same = 0;
        for (int t = 0; t < 16; ++t) {
            const QsaRows one{mx::slice(iq, {t, 0, 0}, {t + 1, kQsaHeads, kQsaDim}), blocks};
            auto ar = decode_attention_qsa_int8(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}), kv8.k8t, kv8.v8, kv8.kscale,
                                                kv8.vscale, scale, prefix + t + 1, one);
            same += mx::array_equal(row(chain, t), row(ar, 0)).item<bool>();
        }
        failures += same != 16;
        std::printf("%s ATTENTION_QSA int8 prefix %d: chain rows == one-row calls %d / 16\n", same == 16 ? "PASS" : "FAIL", prefix, same);
        auto bf = decode_attention_exact(q, k, v, scale, prefix + 16, nullptr, &all);
        const float e8 = mx::max(mx::abs(mx::subtract(chain, bf))).item<float>(), m8 = mx::max(mx::abs(bf)).item<float>();
        const bool near = e8 <= 2e-2f * std::max(1.0f, m8);
        failures += !near;
        std::printf("%s ATTENTION_QSA int8 rows vs BF16 cache rows: max abs diff %.2e (max %.3f)\n", near ? "PASS" : "FAIL", e8, m8);
    }
    std::printf(failures ? "ATTENTION_QSA_FAIL\n" : "ATTENTION_QSA_PASS\n");
    return failures ? 1 : 0;
}
