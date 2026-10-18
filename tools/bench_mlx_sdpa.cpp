// Prefill attention cost on MLX's fused SDPA: Qwen3.8-27B full-attention shape (24 query heads, 4 KV heads, head dim 256)
// vs the same FLOPs at head dim 128 (48 / 8 heads), one 2048-row chunk against L keys with the causal (bottom-right)
// mask, BF16. Prints TFLOPS counting causal work only.
#include <mlx/mlx.h>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <functional>
#include <tuple>
#include <vector>

namespace mx = mlx::core;
static double ms(const std::function<void()>& f) {
    f(); double b = 1e30;
    for (int i = 0; i < 5; ++i) {auto t = std::chrono::steady_clock::now(); f(); b = std::min(b, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());}
    return b;
}
int main() {
    const int m = 2048;
    for (int L : {2048, 4096, 8192, 16384})
        for (auto [h, kvh, d] : {std::tuple{24, 4, 256}, std::tuple{48, 8, 128}})
            for (auto dt : {mx::bfloat16, mx::float16}) {
                auto q = mx::astype(mx::random::normal({1, h, m, d}), dt), k = mx::astype(mx::random::normal({1, kvh, L, d}), dt);
                auto v = mx::astype(mx::random::normal({1, kvh, L, d}), dt);
                mx::eval(q, k, v);
                const float scale = 1.0f / std::sqrt(float(d));
                const double t = ms([&] {std::vector<mx::array> o; for (int i = 0; i < 4; ++i) o.push_back(mx::fast::scaled_dot_product_attention(q, k, v, scale, "causal")); mx::eval(o);}) / 4;
                // causal work: query row i (absolute L - m + i) sees L - m + i + 1 keys
                const double pairs = double(m) * (L - m) + double(m) * (m + 1) / 2;
                const double flop = 4.0 * pairs * h * d;
                std::printf("SDPA L=%5d heads %d/%d dim %d %s: %.2f ms  %.1f TFLOPS\n", L, h, kvh, d, dt == mx::bfloat16 ? "bf16" : "fp16", t, flop / t * 1e-9);
            }
    std::puts("SDPA_BENCH_DONE");
}
