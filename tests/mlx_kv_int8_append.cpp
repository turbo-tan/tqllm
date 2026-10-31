// int8 KV append (attention_decode.hpp quantize_rows_int8): one kernel gives the same codes and FP16 scales as the
// MLX-op quantizer (quantize_kv_int8's math) bit for bit, K transposed, for decode (1 / 8 / 16 rows) and prefill (300)
// row counts; timing of both per call at 16 rows.
#include "attention_decode.hpp"
#include <mlx/mlx.h>
#include <chrono>
#include <cstdio>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
int main() {
    int failures = 0;
    const int kvh = 4;
    auto ref = [&](const mx::array& x) {
        const int m = x.shape(2);
        auto xf = mx::reshape(mx::astype(x, mx::float32), {kvh, m, 256});
        auto amax = mx::max(mx::abs(xf), -1, true);
        auto s16 = mx::astype(mx::maximum(mx::divide(amax, mx::array(127.0f)), mx::array(1e-8f)), mx::float16);
        auto codes = mx::astype(mx::clip(mx::round(mx::divide(xf, mx::astype(s16, mx::float32))), mx::array(-127.0f), mx::array(127.0f)), mx::int8);
        return std::pair{codes, mx::reshape(s16, {kvh, m})};
    };
    for (int m : {1, 8, 16, 300}) {
        auto noise = mx::random::normal({1, kvh, m, 256}, mx::float32, mx::random::key(m));
        auto spikes = mx::multiply(mx::astype(mx::greater(mx::random::uniform({1, kvh, m, 256}, mx::float32, mx::random::key(m + 7)), mx::array(0.97f)), mx::float32), mx::array(9.0f));
        auto k = mx::astype(mx::add(noise, spikes), mx::bfloat16);
        auto v = mx::astype(mx::multiply(mx::random::normal({1, kvh, m, 256}, mx::float32, mx::random::key(m + 3)), mx::array(0.3f)), mx::bfloat16);
        auto r = quantize_rows_int8(k, v);
        auto [kc, ks] = ref(k); auto [vc, vs] = ref(v);
        auto kt = mx::transpose(kc, {0, 2, 1});
        const bool ok = mx::array_equal(r.kt, kt).item<bool>() && mx::array_equal(r.v, vc).item<bool>() &&
                        mx::array_equal(r.kscale, ks).item<bool>() && mx::array_equal(r.vscale, vs).item<bool>();
        std::printf("%s int8 rows m %d: codes + scales bit for bit vs MLX ops\n", ok ? "PASS" : "FAIL", m);
        failures += !ok;
    }
    auto k = mx::astype(mx::random::normal({1, kvh, 16, 256}), mx::bfloat16), v = mx::astype(mx::random::normal({1, kvh, 16, 256}), mx::bfloat16);
    mx::eval(k, v);
    auto time = [&](auto f) {
        f(); double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 16; ++i) f();
            best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 16);
        }
        return best;
    };
    const double tk = time([&] {auto r = quantize_rows_int8(k, v); mx::eval(r.kt, r.v, r.kscale, r.vscale);});
    const double to = time([&] {auto [a, b] = ref(k); auto [c, d] = ref(v); mx::eval(a, b, c, d);});
    std::printf("BENCH int8 rows m 16 (us per call, eval each): kernel %.0f, MLX ops %.0f\n", tk, to);
    std::printf(failures ? "KV_INT8_APPEND_FAIL\n" : "KV_INT8_APPEND_PASS\n");
    return failures ? 1 : 0;
}
