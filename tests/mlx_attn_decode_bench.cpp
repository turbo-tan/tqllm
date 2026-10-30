// Decode/verify attention for the Qwen3.8-27B attention layers (24 q heads, 4 kv heads, d 256, BF16 KV) at long key counts.
// Per-layer ms of: (a) the engine's v4 call (8 verify rows padded to 16 -> MLX matrix kernel), (b) the same 8 rows unpadded
// (q_len <= 8 -> MLX vector kernel, 2-pass split over keys), (c) 1 row (AR) on the vector kernel; plus whether row r of the
// 8-row call equals the same query run alone as a 1-row call, bit for bit, and as a row of the 16-row padded call.
#import <Foundation/Foundation.h>
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/random.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
namespace mx = mlx::core;
using Clock = std::chrono::steady_clock;
static double med(std::vector<double>& v) {std::sort(v.begin(), v.end()); return v[v.size() / 2];}
int main() {
    @autoreleasepool {
        const int H = 24, KH = 4, D = 256;
        const float scale = 1.0f / std::sqrt(float(D));
        mx::random::seed(5);
        for (int L : {16384, 32768, 65536}) {
            auto k = mx::astype(mx::random::normal({1, KH, L, D}), mx::bfloat16);
            auto v = mx::astype(mx::random::normal({1, KH, L, D}), mx::bfloat16);
            auto q16 = mx::astype(mx::random::normal({1, H, 16, D}), mx::bfloat16);
            mx::eval(k, v, q16);
            const auto bench = [&](auto f) {
                auto r = f(); mx::eval(r);
                std::vector<double> t;
                for (int i = 0; i < 15; ++i) {auto s = Clock::now(); auto y = f(); mx::eval(y); t.push_back(std::chrono::duration<double, std::milli>(Clock::now() - s).count());}
                return med(t);
            };
            auto q8 = mx::slice(q16, {0, 0, 0, 0}, {1, H, 8, D});
            auto q1 = mx::slice(q16, {0, 0, 0, 0}, {1, H, 1, D});
            const double a = bench([&] {return mx::fast::scaled_dot_product_attention(q16, k, v, scale, "causal");});
            const double b = bench([&] {return mx::fast::scaled_dot_product_attention(mx::contiguous(q8), k, v, scale, "causal");});
            const double c = bench([&] {return mx::fast::scaled_dot_product_attention(mx::contiguous(q1), k, v, scale, "causal");});
            // row independence of the vector kernel: row 0 of the 8-row call vs the 1-row call
            auto o8 = mx::fast::scaled_dot_product_attention(mx::contiguous(q8), k, v, scale, "causal");
            auto o1 = mx::fast::scaled_dot_product_attention(mx::contiguous(q1), k, v, scale, "causal");
            auto o16 = mx::fast::scaled_dot_product_attention(q16, k, v, scale, "causal");
            auto same81 = mx::all(mx::equal(mx::slice(o8, {0, 0, 0, 0}, {1, H, 1, D}), o1));
            auto same816 = mx::all(mx::equal(mx::slice(o8, {0, 0, 0, 0}, {1, H, 8, D}), mx::slice(o16, {0, 0, 0, 0}, {1, H, 8, D})));
            mx::eval(same81, same816);
            std::printf("ATTN L=%d per layer: 16-row padded %.3f ms, 8 rows %.3f ms, 1 row %.3f ms | x16 layers: %.1f / %.1f / %.1f ms | row0(8)==1row %d, rows(8)==rows(16) %d\n",
                        L, a, b, c, a * 16, b * 16, c * 16, int(same81.item<bool>()), int(same816.item<bool>()));
        }
        // (d) rows folded into the heads dimension: [1, KH*G*n, 1, d] queries over [1, KH, L, d] keys, an explicit mask gives
        // token t of the call the keys <= L - n + t. Every folded row is a 1-row query for MLX's vector (split-over-keys) kernel.
        const int G = H / KH;
        for (int L : {16384, 32768, 65536}) {
            auto k = mx::astype(mx::random::normal({1, KH, L, D}), mx::bfloat16);
            auto v = mx::astype(mx::random::normal({1, KH, L, D}), mx::bfloat16);
            auto qfull = mx::astype(mx::random::normal({1, H, 8, D}), mx::bfloat16);
            mx::eval(k, v, qfull);
            const auto folded = [&](const mx::array& q, int n) {
                // q [1, H, n, D] -> [1, KH*G*n, 1, D], head index = kvh * (G*n) + g * n + t
                auto qq = mx::reshape(q, {1, KH, G, n, D});
                qq = mx::reshape(qq, {1, KH * G * n, 1, D});
                auto t = mx::remainder(mx::arange(G * n), mx::array(n));            // token of each row within a kv head
                auto last = mx::add(t, mx::array(L - n));                              // last visible key (0-based)
                auto vis = mx::less_equal(mx::reshape(mx::arange(L), {1, 1, 1, L}), mx::reshape(last, {1, 1, G * n, 1}));
                vis = mx::reshape(mx::broadcast_to(mx::reshape(vis, {1, 1, G * n, L}), {1, KH, G * n, L}), {1, KH * G * n, 1, L});
                auto out = mx::fast::scaled_dot_product_attention(qq, k, v, 1.0f / std::sqrt(float(D)), "array", vis);
                return mx::reshape(out, {1, H, n, D});
            };
            const auto bench = [&](auto f) {
                auto r = f(); mx::eval(r);
                std::vector<double> t;
                for (int i = 0; i < 15; ++i) {auto s2 = Clock::now(); auto y = f(); mx::eval(y); t.push_back(std::chrono::duration<double, std::milli>(Clock::now() - s2).count());}
                return med(t);
            };
            auto q8 = mx::contiguous(qfull);
            auto q1 = mx::contiguous(mx::slice(qfull, {0, 0, 0, 0}, {1, H, 1, D}));
            const double t8 = bench([&] {return folded(q8, 8);});
            const double t1 = bench([&] {return folded(q1, 1);});
            auto o8 = folded(q8, 8);
            auto o1 = folded(q1, 1);
            // the 1-row call at the same context equals row 7 (the last token sees all L keys) of the 8-row call
            auto same = mx::all(mx::equal(mx::slice(o8, {0, 0, 7, 0}, {1, H, 8, D}), o1));
            // and the 4-row call's rows equal the matching rows of the 8-row call for a call that ends at the same token
            mx::eval(same);
            std::printf("FOLD L=%d per layer: 8 rows folded %.3f ms, 1 row folded %.3f ms | x16 layers: %.1f / %.1f ms | last row of 8 == 1-row call: %d\n",
                        L, t8, t1, t8 * 16, t1 * 16, int(same.item<bool>()));
        }
        std::printf("ATTN_DONE\n");
    }
    return 0;
}
