#include <mlx/mlx.h>
#include "gdn.hpp"
#include <cmath>
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
    // chunked prefill GDN vs the sequential scan (same inputs, random state), relative error of output and state
    // strong decay: per-token decays down to exp(-20) and exact zeros (model heads with large dt), which overflow
    // an unmasked exp of the cumulative log-decay difference; correlated: near-identical keys (repeated tokens), decay 0.99,
    // beta 0.9, where a power-series inverse of (I + L) cancels to garbage
    struct Case {int m; float strength; bool correlated;};
    for (auto [m, strength, correlated] : std::vector<Case>{{128, 0.05f, false}, {200, 0.05f, false}, {2048, 0.05f, false},
                                                           {200, 5.0f, false}, {2048, 5.0f, false}, {200, 0.0f, true}, {2048, 0.0f, true}}) {
        const int kh2 = 16, vh2 = 48;
        auto nrm = [](const mx::array& x) {return mx::multiply(x, mx::rsqrt(mx::add(mx::sum(mx::square(x), -1, true), mx::array(1e-6f))));};
        auto q = mx::multiply(nrm(rnd(g, {1, m, kh2, 128}, 1.0f)), mx::array(0.0883883f)), k = nrm(rnd(g, {1, m, kh2, 128}, 1.0f));
        if (correlated) k = nrm(mx::add(mx::broadcast_to(rnd(g, {1, 1, kh2, 128}, 1.0f), {1, m, kh2, 128}), rnd(g, {1, m, kh2, 128}, 0.05f)));
        auto v = rnd(g, {1, m, vh2, 128}, 1.0f);
        auto decay = mx::exp(mx::multiply(mx::abs(rnd(g, {1, m, vh2}, 1.0f)), mx::array(-strength)));
        if (strength > 1.0f) decay = mx::multiply(decay, mx::astype(mx::greater(mx::abs(rnd(g, {1, m, vh2}, 1.0f)), mx::array(0.05f)), mx::float32));
        auto beta = mx::sigmoid(rnd(g, {1, m, vh2}, 1.0f));
        if (correlated) {decay = mx::full({1, m, vh2}, 0.99f); beta = mx::full({1, m, vh2}, 0.9f);}
        auto s0 = rnd(g, {1, vh2, 128, 128}, 0.1f);
        auto ref = gdn_scan(q, k, v, decay, beta, s0, false, false);
        auto [o, st] = gdn_chunked(q, k, v, decay, beta, s0, 64);
        auto rel = [](const mx::array& a, const mx::array& b) {
            auto d = mx::sqrt(mx::sum(mx::square(mx::subtract(a, b)))), n = mx::sqrt(mx::sum(mx::square(b)));
            mx::eval(d, n); return double(d.item<float>() / n.item<float>());
        };
        const double eo = rel(o, ref.output), es = rel(st, ref.state);
        std::printf("GDN_CHUNKED m=%d decay_strength=%.2f correlated=%d out_rel_err=%.2e state_rel_err=%.2e\n", m, strength, int(correlated), eo, es);
        ok = ok && eo < 1e-4 && es < 1e-4;  // NaN fails both comparisons
    }
    std::puts(ok ? "MLX_GDN_FUSE_PASS" : "MLX_GDN_FUSE_FAIL");
    return ok ? 0 : 1;
}
