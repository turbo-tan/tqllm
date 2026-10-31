#include <mlx/mlx.h>
#include "gdn.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
mx::array rnd(std::mt19937& g, mx::Shape s, float scale) {
    size_t n = 1;
    for (int d : s) n *= size_t(d);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(g);
    return mx::array(v.data(), s, mx::float32);
}
mx::array bf(const mx::array& x) {return mx::astype(mx::astype(x, mx::bfloat16), mx::float32);}
// largest difference in bf16 ulps of the reference (values are bf16-rounded or derived from bf16 values)
double ulps(const mx::array& a, const mx::array& b) {
    auto x = mx::contiguous(a), y = mx::contiguous(b);
    mx::eval(x, y);
    double worst = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double r = y.data<float>()[i], d = std::fabs(double(x.data<float>()[i]) - r);
        worst = std::max(worst, d / (std::fabs(r) * 0.0078125 + 1e-6));
    }
    return worst;
}
}  // namespace

int main() {
    std::mt19937 g(5);
    bool ok = true;
    const int kh = 16, vh = 48, width = 2 * kh * 128 + vh * 128;
    for (int m : {1, 8}) {
        auto joined = rnd(g, {m + 3, width}, 1.0f), weights = rnd(g, {width, 4}, 0.5f);
        auto ref = gdn_convolve_native(joined, weights);
        auto q = mx::reshape(mx::slice(ref, {0, 0}, {m, kh * 128}), {1, m, kh, 128});
        auto k = mx::reshape(mx::slice(ref, {0, kh * 128}, {m, 2 * kh * 128}), {1, m, kh, 128});
        auto v = mx::slice(ref, {0, 2 * kh * 128}, {m, width});
        q = gdn_normalize(q, 1.0f / std::sqrt(128.0f));
        k = gdn_normalize(k, 1.0f);
        auto want = mx::concatenate({mx::reshape(q, {m, kh * 128}), mx::reshape(k, {m, kh * 128}), v}, 1);
        auto got = gdn_pre_fused(joined, weights, kh, 1.0f / std::sqrt(128.0f));
        const double e1 = ulps(got, want);
        auto y = rnd(g, {1, m, vh, 128}, 1.0f), z = rnd(g, {1, m, vh, 128}, 1.0f), w = rnd(g, {128}, 0.3f);
        auto want2 = bf(mx::multiply(mx::fast::rms_norm(bf(y), w, 1e-6f), mx::multiply(z, mx::sigmoid(z))));
        auto got2 = gdn_post_fused(y, z, w, 1e-6f);
        const double e2 = ulps(got2, want2);
        // rows are independent: the m=8 call's row 0 equals an m=1 call on that row
        std::printf("GDN_FUSE m=%d pre_max_bf16_ulps=%.3f post_max_bf16_ulps=%.3f\n", m, e1, e2);
        ok = ok && e1 <= 1.01 && e2 <= 1.01;
    }
    // fused 1-8 token step vs the separate ops, and the exactness invariant spec decode relies on: the state replayed
    // from an m=8 tape after a partial accept of `count` tokens equals `count` single-token steps bit for bit
    auto same = [](const mx::array& a, const mx::array& b) {
        auto x = mx::contiguous(a), y = mx::contiguous(b);
        mx::eval(x, y);
        if (x.size() != y.size()) return false;
        for (size_t i = 0; i < x.size(); ++i) if (std::bit_cast<uint32_t>(x.data<float>()[i]) != std::bit_cast<uint32_t>(y.data<float>()[i])) return false;
        return true;
    };
    auto relerr = [](const mx::array& a, const mx::array& b) {
        auto d = mx::sqrt(mx::sum(mx::square(mx::subtract(a, b)))), n = mx::sqrt(mx::sum(mx::square(b)));
        mx::eval(d, n); return double(d.item<float>() / std::max(n.item<float>(), 1e-20f));
    };
    for (int vh2 : {48, 32}) {
        const int w2 = 2 * kh * 128 + vh2 * 128;
        const float eps = 1e-6f, qscale = 1.0f / std::sqrt(128.0f);
        auto weights = rnd(g, {w2, 4}, 0.5f), bias = rnd(g, {vh2}, 1.0f);
        auto rate = mx::negative(mx::abs(rnd(g, {vh2}, 1.0f))), nw = rnd(g, {128}, 0.3f);
        auto s0 = rnd(g, {1, vh2, 128, 128}, 0.1f);
        for (int m : {8, 16}) {
        auto joined = rnd(g, {m + 3, w2}, 1.0f), alpha = rnd(g, {m, vh2}, 1.0f), beta = rnd(g, {m, vh2}, 1.0f);
        auto z = rnd(g, {m, vh2 * 128}, 1.0f);
        auto step = gdn_step_fused(joined, weights, alpha, beta, bias, rate, z, nw, eps, qscale, s0, kh, vh2);
        // reference from the separate launches
        auto conv = gdn_convolve_native(joined, weights);
        auto q = gdn_normalize(mx::reshape(mx::slice(conv, {0, 0}, {m, kh * 128}), {1, m, kh, 128}), qscale);
        auto k = gdn_normalize(mx::reshape(mx::slice(conv, {0, kh * 128}, {m, 2 * kh * 128}), {1, m, kh, 128}), 1.0f);
        auto v = mx::reshape(mx::slice(conv, {0, 2 * kh * 128}, {m, w2}), {1, m, vh2, 128});
        auto [dec, bet] = gdn_gates_native(alpha, beta, bias, rate);
        auto ref = gdn_scan(q, k, v, mx::reshape(dec, {1, m, vh2}), mx::reshape(bet, {1, m, vh2}), s0, true, true);
        auto want = mx::reshape(bf(mx::multiply(mx::fast::rms_norm(bf(ref.output), nw, eps), mx::multiply(mx::reshape(z, {1, m, vh2, 128}), mx::sigmoid(mx::reshape(z, {1, m, vh2, 128}))))), {m, vh2 * 128});
        const double eo = ulps(mx::astype(step.out, mx::float32), want), es = relerr(step.state, ref.state), ed = relerr(step.deltas, *ref.deltas);
        const double ek = relerr(step.keys, k), ea = relerr(step.decay, mx::reshape(dec, {1, m, vh2}));
        const bool hist = same(step.history, mx::slice(joined, {m, 0}, {m + 3, w2}));
        // replay vs repeated m=1 steps, and m=1 outputs vs the m=8 rows
        bool replay_ok = true, rows_ok = true;
        std::optional<mx::array> seq = s0;
        for (int count = 1; count < m; ++count) {
            const int t = count - 1;
            auto one = gdn_step_fused(mx::slice(joined, {t, 0}, {t + 4, w2}), weights, mx::slice(alpha, {t, 0}, {t + 1, vh2}),
                mx::slice(beta, {t, 0}, {t + 1, vh2}), bias, rate, mx::slice(z, {t, 0}, {t + 1, vh2 * 128}), nw, eps, qscale, seq, kh, vh2);
            seq = one.state;
            rows_ok = rows_ok && same(mx::astype(one.out, mx::float32), mx::astype(mx::slice(step.out, {t, 0}, {t + 1, vh2 * 128}), mx::float32));
            auto replayed = gdn_replay_prefix(step.keys, step.decay, step.deltas, s0, count);
            replay_ok = replay_ok && same(replayed, *seq);
        }
        std::printf("GDN_STEP vh=%d m=%d out_max_bf16_ulps=%.3f state_rel=%.2e deltas_rel=%.2e keys_rel=%.2e decay_rel=%.2e history_exact=%d m1_rows_exact=%d replay_exact=%d\n",
            vh2, m, eo, es, ed, ek, ea, int(hist), int(rows_ok), int(replay_ok));
        ok = ok && eo <= 1.01 && es < 1e-5 && ed < 1e-5 && ek < 1e-6 && ea < 1e-6 && hist && rows_ok && replay_ok;
        }
    }
    // alpha / beta projected inside the step kernel ([m, hidden] input + [2 vh, hidden] weights) vs the projected gates:
    // same up to the BF16 rounding of a differently ordered dot (decay / beta within ~1 bf16 ulp), and the invariants
    // hold on the in-kernel path itself: m=1 rows bitwise equal to the m=16 rows, replay from the tape exact
    for (int m : {8, 16}) {
        const int vh2 = 32, hid = 5120, w2 = 2 * kh * 128 + vh2 * 128;
        const float eps = 1e-6f, qscale = 1.0f / std::sqrt(128.0f);
        auto weights = rnd(g, {w2, 4}, 0.5f), bias = rnd(g, {vh2}, 1.0f), rate = mx::negative(mx::abs(rnd(g, {vh2}, 1.0f))), nw = rnd(g, {128}, 0.3f);
        auto s0 = rnd(g, {1, vh2, 128, 128}, 0.1f), joined = rnd(g, {m + 3, w2}, 1.0f), z = rnd(g, {m, vh2 * 128}, 1.0f);
        auto x = rnd(g, {m, hid}, 1.0f), ab = rnd(g, {2 * vh2, hid}, 0.02f);
        auto prev = mx::slice(joined, {0, 0}, {3, w2}), rows = mx::slice(joined, {3, 0}, {m + 3, w2});
        auto alpha = bf(mx::matmul(x, mx::transpose(mx::slice(ab, {0, 0}, {vh2, hid})))), beta = bf(mx::matmul(x, mx::transpose(mx::slice(ab, {vh2, 0}, {2 * vh2, hid}))));
        auto ref = gdn_step_fused(prev, rows, weights, alpha, beta, bias, rate, z, nw, eps, qscale, s0, kh, vh2);
        auto got = gdn_step_fused(prev, rows, weights, x, ab, bias, rate, z, nw, eps, qscale, s0, kh, vh2);
        const double ea = relerr(got.decay, ref.decay), eo = relerr(mx::astype(got.out, mx::float32), mx::astype(ref.out, mx::float32)), es = relerr(got.state, ref.state);
        bool rows_ok = true, replay_ok = true;
        std::optional<mx::array> seq = s0;
        for (int count = 1; count < m; ++count) {
            const int t = count - 1;
            auto one = gdn_step_fused(mx::slice(joined, {t, 0}, {t + 3, w2}), mx::slice(joined, {t + 3, 0}, {t + 4, w2}), weights,
                mx::slice(x, {t, 0}, {t + 1, hid}), ab, bias, rate, mx::slice(z, {t, 0}, {t + 1, vh2 * 128}), nw, eps, qscale, seq, kh, vh2);
            seq = one.state;
            rows_ok = rows_ok && same(mx::astype(one.out, mx::float32), mx::astype(mx::slice(got.out, {t, 0}, {t + 1, vh2 * 128}), mx::float32)) && same(one.decay, mx::slice(got.decay, {0, t, 0}, {1, t + 1, vh2}));
            replay_ok = replay_ok && same(gdn_replay_prefix(got.keys, got.decay, got.deltas, s0, count), *seq);
        }
        std::printf("GDN_STEP_AB m=%d decay_rel=%.2e out_rel=%.2e state_rel=%.2e m1_rows_exact=%d replay_exact=%d\n", m, ea, eo, es, int(rows_ok), int(replay_ok));
        ok = ok && ea < 2e-2 && eo < 2e-2 && es < 2e-2 && rows_ok && replay_ok;
    }
    // prefill pack: q / k / v and the gated norm read from a strided FP16 product [m, qkv | z | extra] bit for bit as the
    // fused kernels on concatenate(history, round(product)) and round(z slice)
    {
        const int m = 70, zw = vh * 128, ld = width + zw + 32;
        auto product = mx::astype(rnd(g, {m, ld}, 1.0f), mx::float16);
        auto history = bf(rnd(g, {3, width}, 1.0f)), weights = rnd(g, {width, 4}, 0.5f), w = rnd(g, {128}, 0.3f);
        auto round = [](const mx::array& p) {return bf(mx::astype(p, mx::float32));};
        auto joined = mx::concatenate({history, round(mx::slice(product, {0, 0}, {m, width}))}, 0);
        auto ref = gdn_pre_fused(joined, weights, kh, 1.0f / std::sqrt(128.0f));
        auto [q, k, v] = gdn_pre_pack(history, product, width, weights, kh, 1.0f / std::sqrt(128.0f));
        const bool pre = same(mx::concatenate({q, k, v}, 1), ref);
        auto y = rnd(g, {1, m, vh, 128}, 1.0f);
        auto z = mx::reshape(round(mx::slice(product, {0, width}, {m, width + zw})), {1, m, vh, 128});
        const bool post = same(gdn_post_pack(y, product, width, w, 1e-6f), gdn_post_fused(y, z, w, 1e-6f));
        std::printf("GDN_PACK m=%d pre_exact=%d post_exact=%d\n", m, int(pre), int(post));
        ok = ok && pre && post;
    }
    std::puts(ok ? "MLX_GDN_FUSE_PASS" : "MLX_GDN_FUSE_FAIL");
    return ok ? 0 : 1;
}
