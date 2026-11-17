// Issue #55 (M1 Max: "Thread group size (1024) is greater than the maximum allowed threads per threadgroup (384)"):
// (0) MLX turns a too-large threadgroup away with an exception that threadgroup_runs() recognizes, and the next kernel
//     on the stream still gives the right values (the load probes rely on both);
// (1) the fused GDN step at 512 / 256 / 128 / 32 threads per threadgroup equals 1024 bit for bit (every output), for
//     1 / 16 / 17 / 32 rows, gates given or projected in the kernel, chain and token tree (its state output is not
//     written, so not compared); threads = 0 (the fitted size) equals 1024 too;
// (2) the QSA block selection at 256 / 32 threads equals 1024 (ties included);
// (3) the rms pair kernels run at 1024 here (false only under TQLLM_DEBUG=no_tg1024, which caps the fitted sizes).
#include <mlx/mlx.h>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "attention_exact.hpp"
#include "gdn.hpp"
#include "prefill_fused.hpp"
#include "threadgroup_fit.hpp"

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    failures += !ok;
}
mx::array randn(std::mt19937& rng, mx::Shape shape, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    size_t n = 1;
    for (auto s : shape) n *= size_t(s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return mx::array(v.data(), shape, mx::float32);
}
// a fresh device result per call (evaluated small inputs would reach the kernel as constant buffers)
mx::array fresh(const mx::array& x) {return mx::multiply(x, mx::array(1.0f, x.dtype()));}
bool same(const mx::array& a, const mx::array& b) {return a.shape() == b.shape() && mx::array_equal(a, b).item<bool>();}
}  // namespace

int main() {
    std::printf("threadgroup cap %d%s\n", threadgroup_cap(), threadgroup_cap() < 1024 ? " (TQLLM_DEBUG=no_tg1024)" : "");
    {   // (0)
        static auto twice = mx::fast::metal_kernel("tqllm_test_twice", {"x"}, {"y"}, R"METAL(
            const uint i = thread_position_in_grid.x;
            if (i < uint(N)) y[i] = 2.0f * x[i];
)METAL");
        const auto x = mx::arange(64, mx::float32);
        auto run = [&](int t) {
            auto y = twice({fresh(x)}, {{64}}, {mx::float32}, {t, 1, 1}, {t, 1, 1}, {{"N", 64}}, std::nullopt, false,
                           mx::Device(mx::Device::gpu))[0];
            mx::eval(y);
            return y;
        };
        std::string why;
        const bool big = threadgroup_runs([&](int t) {run(t);}, 2048, &why);
        check(!big && !why.empty(), "a 2048-thread threadgroup is turned away (" + why + ")");
        check(same(run(64), mx::multiply(x, mx::array(2.0f))), "the stream gives right values after the refused dispatch");
    }
    std::mt19937 rng(55);
    {   // (1)
        // vh = 8: an input below 8 elements reaches a custom kernel as a `constant` buffer (MLX), which the ABF branch cannot read
        const int kh = 2, vh = 8, width = 2 * kh * 128 + vh * 128, hidden = 256;
        auto prev = randn(rng, {3, width}, 1.0f), weights = randn(rng, {width, 4}, 0.5f), ab = randn(rng, {2 * vh, hidden}, 0.05f);
        auto bias = randn(rng, {vh}, 0.5f), rate = mx::negative(mx::abs(randn(rng, {vh}, 0.5f))), w = randn(rng, {128}, 1.0f);
        auto state = randn(rng, {1, vh, 128, 128}, 0.1f);
        for (int m : {1, 16, 17, 32}) {
            auto rows = randn(rng, {m, width}, 1.0f), z = randn(rng, {m, vh * 128}, 1.0f), input = randn(rng, {m, hidden}, 1.0f);
            auto alpha = randn(rng, {m, vh}, 1.0f), beta = randn(rng, {m, vh}, 1.0f);
            std::vector<int32_t> parents(static_cast<size_t>(m));
            for (int t = 0; t < m; ++t) parents[size_t(t)] = t == 0 ? -1 : (t % 3 == 0 ? 0 : t - 1);  // branches off row 0
            const auto par = mx::array(parents.data(), {m}, mx::int32);
            mx::eval({prev, weights, ab, bias, rate, w, state, rows, z, input, alpha, beta});
            for (int abf : {0, 1})
                for (int tree : {0, 1}) {
                    auto step = [&](int threads) {
                        auto r = gdn_step_fused(fresh(prev), fresh(rows), fresh(weights), fresh(abf ? input : alpha), fresh(abf ? ab : beta),
                                                fresh(bias), fresh(rate), fresh(z), fresh(w), 1e-6f, 0.0883883f, fresh(state), kh, vh, 0, 0, 0,
                                                tree ? &par : nullptr, threads);
                        std::vector<mx::array> all = {r.out, r.state, r.deltas, r.keys, r.decay, r.history};
                        mx::eval(all);
                        return all;
                    };
                    const auto ref = step(1024);
                    for (int threads : {512, 256, 128, 32, 0}) {
                        const auto got = step(threads);
                        bool eq = true;
                        for (size_t i = 0; i < ref.size(); ++i)
                            if (!(tree && i == 1)) eq = eq && same(got[i], ref[i]);  // a branched tree leaves the state unwritten
                        check(eq, "GDN step m=" + std::to_string(m) + " abf=" + std::to_string(abf) + " tree=" + std::to_string(tree) +
                                      " threads=" + std::to_string(threads) + " == 1024, every output bit for bit");
                    }
                }
        }
    }
    {   // (2)
        const int prefix = kQsaFrom + 3616, n = 16, deepest = prefix + n - 1 - 15, bound = deepest / kExactTile * kExactTile;
        const int nbmax = bound / kQsaBlock, pw = 2 + kExactTail;
        // block keys from 997 distinct rows, so many blocks score the same (ties are taken lowest block first)
        auto keys = randn(rng, {997, kQsaDim}, 1.0f);
        auto blocks = mx::take(keys, mx::remainder(mx::arange(nbmax + 64, mx::int32), mx::array(997, mx::int32)), 0);
        auto q = randn(rng, {16, kQsaHeads, kQsaDim}, 1.0f);
        const auto plan = exact_rows_plan(prefix, n, nullptr);
        mx::eval({blocks, q, plan});
        const auto ref = qsa_select(q, blocks, plan, n, nbmax, pw, kQsaFrom, 1024);
        for (int threads : {256, 32, 0})
            check(same(qsa_select(q, blocks, plan, n, nbmax, pw, kQsaFrom, threads), ref),
                  "QSA selection threads=" + std::to_string(threads) + " == 1024");
    }
    {   // (3)
        const bool want = threadgroup_cap() >= 1024;
        check(rms_norm_bf16_pair_runs(5120, false, false) == want && rms_norm_bf16_pair_runs(5120, true, true) == want,
              std::string("rms pair kernels ") + (want ? "run at 1024" : "fall back under the cap"));
    }
    std::printf("THREADGROUP_FIT %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
