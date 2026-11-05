#include <mlx/mlx.h>
#include "prefill_fused.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
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
bool same(const mx::array& a, const mx::array& b) {
    return a.shape() == b.shape() && a.dtype() == b.dtype() && mx::array_equal(a, b).item<bool>();
}
double median_ms(std::vector<double>& t) {
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
}  // namespace

// the fused prefill kernels give the bits of the MLX op chains they replace (the engine also checks this on device
// at its first prefill chunk and falls back when it does not hold)
int main() {
    std::mt19937 g(11);
    bool ok = true;
    for (int rows : {1, 8, 16, 70, 2048}) {
        auto x = mx::astype(mx::astype(rnd(g, {rows, 5120}, 3.0f), mx::bfloat16), mx::float32), w = rnd(g, {5120}, 0.5f);
        auto got = rms_norm_bf16_pair(x, w, 1e-6f), want = rms_norm_bf16_pair_reference(x, w, 1e-6f);
        const bool n32 = same(got.first, want.first), n16 = same(got.second, want.second);
        auto gu = mx::astype(rnd(g, {rows, 2 * 17408}, 2.0f), mx::float16);
        const bool sw = same(swiglu_pack16(gu), swiglu_pack16_reference(gu));
        std::printf("PREFILL_FUSED rows=%d rms_norm fp32_exact=%d fp16_exact=%d swiglu_exact=%d\n", rows, int(n32), int(n16), int(sw));
        {   // residual add + norm in one launch (decode / verify rows): bits of add_bf + the reference pair
            auto r = rnd(g, {rows, 5120}, 3.0f);
            auto got_r = rms_norm_bf16_pair_residual(x, r, w, 1e-6f), want_r = rms_norm_bf16_pair_residual_reference(x, r, w, 1e-6f);
            const bool s_ok = same(std::get<0>(got_r), std::get<0>(want_r)), n_ok = same(std::get<1>(got_r), std::get<1>(want_r)),
                       h_ok = same(std::get<2>(got_r), std::get<2>(want_r));
            std::printf("DECODE_FUSED rows=%d add_norm sum_exact=%d fp32_exact=%d fp16_exact=%d\n", rows, int(s_ok), int(n_ok), int(h_ok));
            ok = ok && s_ok && n_ok && h_ok;
        }
        ok = ok && n32 && n16 && sw;
    }

    // attention q/k RMS norm + RoPE: Qwen3.8-27B shape (heads 24, kvheads 4, hd 256, partial rotary rot 64 and full 256)
    const int heads = 24, kvheads = 4, hd = 256;
    const float theta = 1e7f, eps = 1e-6f;
    for (int m : {64, 2048}) {
        for (int rot : {64, 256}) {
            for (int offset : {0, 4096}) {
                auto qp = mx::astype(rnd(g, {m, heads * 2 * hd}, 2.0f), mx::float16);
                auto kp = mx::astype(rnd(g, {m, kvheads * hd}, 2.0f), mx::float16);
                auto vp = mx::astype(rnd(g, {m, kvheads * hd}, 2.0f), mx::float16);
                auto qw = rnd(g, {hd}, 0.5f), kw = rnd(g, {hd}, 0.5f);
                auto f = attn_qk_norm_rope(qp, kp, vp, qw, kw, heads, kvheads, hd, rot, theta, eps, offset);
                auto r = attn_qk_norm_rope_reference(qp, kp, vp, qw, kw, heads, kvheads, hd, rot, theta, eps, offset);
                const bool eq = same(f.q, r.q), ek = same(f.k, r.k), ev = same(f.v, r.v), eg = same(f.gate, r.gate);
                std::printf("PREFILL_FUSED attn_qk m=%d rot=%d offset=%d q=%d k=%d v=%d gate=%d\n",
                            m, rot, offset, int(eq), int(ek), int(ev), int(eg));
                ok = ok && eq && ek && ev && eg;
            }
        }
    }

    // timing at m=2048, rot=64, offset=0 (median of 20 after a warmup)
    {
        const int m = 2048, rot = 64, offset = 0;
        auto qp = mx::astype(rnd(g, {m, heads * 2 * hd}, 2.0f), mx::float16);
        auto kp = mx::astype(rnd(g, {m, kvheads * hd}, 2.0f), mx::float16);
        auto vp = mx::astype(rnd(g, {m, kvheads * hd}, 2.0f), mx::float16);
        auto qw = rnd(g, {hd}, 0.5f), kw = rnd(g, {hd}, 0.5f);
        auto time_it = [&](bool fused) {
            auto run = [&] {
                auto r = fused ? attn_qk_norm_rope(qp, kp, vp, qw, kw, heads, kvheads, hd, rot, theta, eps, offset)
                               : attn_qk_norm_rope_reference(qp, kp, vp, qw, kw, heads, kvheads, hd, rot, theta, eps, offset);
                mx::eval({r.q, r.k, r.v, r.gate});
            };
            run();  // warmup / build
            std::vector<double> t;
            for (int i = 0; i < 20; i++) {
                auto t0 = std::chrono::steady_clock::now();
                run();
                t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            return median_ms(t);
        };
        const double tf = time_it(true), tr = time_it(false);
        std::printf("PREFILL_FUSED attn_qk timing m=%d fused=%.3f ms reference=%.3f ms\n", m, tf, tr);
    }

    std::puts(ok ? "MLX_PREFILL_FUSED_PASS" : "MLX_PREFILL_FUSED_FAIL");
    return ok ? 0 : 1;
}
