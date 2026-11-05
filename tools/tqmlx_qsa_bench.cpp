// QSA timing without a model: one full-attention layer's decode / verify work for a 16-row step over random BF16 K / V,
// at each context: dense (decode_attention_exact), the indexer alone (projection of 16 rows, key / query norm + RoPE,
// block keys for 64 new keys, block scores + top-512 select), and QSA attention (indexer + sparse rows). ms per call,
// evaluated, the mean of `reps` after 3 warm-ups.
// usage: tqmlx-qsa-bench [reps] [contexts...]
#include "attention_exact.hpp"
#include <mlx/mlx.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

static double time_ms(int reps, const std::function<mx::array()>& run) {
    double total = 0;
    for (int r = 0; r < reps + 3; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        mx::eval(run());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (r >= 3) total += ms;
    }
    return total / reps;
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    std::vector<int> contexts;
    for (int i = 2; i < argc; ++i) contexts.push_back(std::atoi(argv[i]));
    if (contexts.empty()) contexts = {32768, 65536, 131072};
    const int heads = 24, kvh = 4, dim = 256, hidden = 5120, n = 16;
    const float scale = 0.0625f, theta = 1e7f, eps = 1e-6f;
    mx::random::seed(7);
    const auto w = mx::astype(mx::multiply(mx::random::normal({kQsaHeads * kQsaDim + kQsaDim, hidden}), mx::array(0.02f)), mx::float16);
    const auto ones = mx::ones({kQsaDim});
    const auto x = mx::astype(mx::astype(mx::random::normal({n, hidden}), mx::bfloat16), mx::float32);
    const auto q = mx::astype(mx::astype(mx::random::normal({1, heads, n, dim}), mx::bfloat16), mx::float32);
    const auto none = mx::zeros({1}, mx::int32);
    mx::eval(w, x, q);
    for (const int ctx : contexts) {
        const int cap = (ctx + 64 + 1023) / 1024 * 1024, prefix = ctx;
        auto k = mx::astype(mx::random::normal({1, kvh, cap, dim}), mx::bfloat16), v = mx::astype(mx::random::normal({1, kvh, cap, dim}), mx::bfloat16);
        auto tokens = mx::astype(mx::random::normal({cap, kQsaDim}), mx::float16);
        auto blocks = mx::random::normal({cap / kQsaBlock, kQsaDim});
        mx::eval(k, v, tokens, blocks);
        const auto at = mx::add(mx::arange(n, mx::int32), mx::array(prefix, mx::int32));
        const int bound = (prefix + n - 1 - 15) / 64 * 64;
        const auto plan = exact_rows_plan(prefix, n, nullptr);
        mx::eval(plan);
        auto indexer = [&] {  // what qsa_rows does on a sparse step, past the cache appends
            auto proj = qsa_project(x, w, 0, kQsaHeads * kQsaDim + kQsaDim);
            auto keys = mx::astype(qsa_norm_rope(mx::slice(proj, {0, 0, kQsaHeads * kQsaDim}, {kQsaSplit, n, kQsaHeads * kQsaDim + kQsaDim}),
                                                 ones, none, theta, eps, false, false), mx::float16);
            auto fresh = qsa_norm_rope(mx::slice(tokens, {prefix - 64, 0}, {prefix, kQsaDim}), ones,
                                       mx::multiply(mx::arange(prefix / 4 - 16, prefix / 4, mx::int32), mx::array(4, mx::int32)), theta, eps, true, true);
            auto qi = qsa_norm_rope(mx::reshape(mx::slice(proj, {0, 0, 0}, {kQsaSplit, n, kQsaHeads * kQsaDim}), {kQsaSplit, n * kQsaHeads, kQsaDim}),
                                    ones, at, theta, eps, true, false);
            return std::vector<mx::array>{mx::reshape(qi, {n, kQsaHeads, kQsaDim}), keys, fresh};
        };
        const double dense = time_ms(reps, [&] {return decode_attention_exact(q, k, v, scale, prefix + n);});
        const double proj = time_ms(reps, [&] {return qsa_project(x, w, 0, kQsaHeads * kQsaDim + kQsaDim);});
        const double index = time_ms(reps, [&] {auto r = indexer(); return mx::concatenate({mx::reshape(r[0], {-1}), mx::reshape(mx::astype(r[1], mx::float32), {-1}), mx::reshape(r[2], {-1})});});
        const auto qi = indexer()[0];
        mx::eval(qi);
        const double select = time_ms(reps, [&] {return qsa_select(qi, blocks, plan, n, bound / kQsaBlock, 2 + kExactTail, kQsaFrom);});
        const QsaRows rows{qi, blocks, 4096};
        const double sparse = time_ms(reps, [&] {return decode_attention_exact(q, k, v, scale, prefix + n, nullptr, &rows);});
        const double full = time_ms(reps, [&] {
            auto r = indexer();
            const QsaRows live{r[0], blocks, 4096};
            return decode_attention_exact(q, k, v, scale, prefix + n, nullptr, &live);
        });
        std::printf("QSA_BENCH ctx %d: dense %.3f ms | QSA %.3f ms (indexer %.3f: projection %.3f, select %.3f; sparse attention %.3f) per layer"
                    " -> x16 layers dense %.1f / QSA %.1f ms per step\n",
                    ctx, dense, full, index, proj, select, sparse, dense * 16, full * 16);
    }
    return 0;
}
