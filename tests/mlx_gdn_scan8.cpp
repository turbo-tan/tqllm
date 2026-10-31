// Prefill GDN scan with 8 lanes per state row (gdn_scan8) against the 32-lane gdn_scan on Qwen3.8-27B shapes (16 key
// heads, 48 value heads, 128 x 128 state): output and final state agree to FP32 summation-order level (pass: rel < 1e-5),
// and the time for 48 layer calls at T 2048 (one prefill chunk).
#include "gdn.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
constexpr int KH = 16, VH = 48, D = 128;
mx::array random_array(std::mt19937& rng, mx::Shape shape, float lo, float hi) {
    size_t n = 1;
    for (int d : shape) n *= size_t(d);
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = u(rng);
    return mx::array(v.data(), shape, mx::float32);
}
mx::array unit_rows(const mx::array& x) {return mx::multiply(x, mx::rsqrt(mx::sum(mx::square(x), -1, true)));}
double rel(const mx::array& a, const mx::array& b) {
    auto d = mx::max(mx::abs(mx::subtract(a, b))), n = mx::max(mx::abs(b));
    mx::eval(d, n); return double(d.item<float>() / std::max(n.item<float>(), 1e-30f));
}
double ms(const std::function<void()>& f) {
    f(); double b = 1e30;
    for (int i = 0; i < 5; ++i) {auto t = std::chrono::steady_clock::now(); f(); b = std::min(b, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());}
    return b;
}
}  // namespace

int main() {
    std::mt19937 rng(9);
    bool ok = true;
    for (int t : {64, 300, 2048}) {
        auto q = unit_rows(random_array(rng, {1, t, KH, D}, -1, 1)), k = unit_rows(random_array(rng, {1, t, KH, D}, -1, 1));
        auto v = random_array(rng, {1, t, VH, D}, -1, 1), decay = random_array(rng, {1, t, VH}, 0.85f, 1.0f);
        auto beta = random_array(rng, {1, t, VH}, 0.0f, 1.0f), state = random_array(rng, {1, VH, D, D}, -0.1f, 0.1f);
        auto ref = gdn_scan(q, k, v, decay, beta, state, true), got = gdn_scan8(q, k, v, decay, beta, state);
        const double ro = rel(got.output, ref.output), rs = rel(got.state, ref.state);
        bool staged_exact = true;
        for (int rows : {16, 32}) {
            auto st = gdn_scan8_staged(q, k, v, decay, beta, state, rows);
            staged_exact = staged_exact && mx::array_equal(st.output, got.output).item<bool>() && mx::array_equal(st.state, got.state).item<bool>();
        }
        std::printf("GDN_SCAN8_STAGED T=%d equal_to_scan8=%d\n", t, int(staged_exact));
        const bool pass = ro < 1e-5 && rs < 1e-5 && staged_exact;
        std::printf("GDN_SCAN8 T=%d out rel %.2e state rel %.2e %s\n", t, ro, rs, pass ? "PASS" : "FAIL");
        ok = ok && pass;
        if (t == 2048) {
            std::vector<mx::array> layers;
            const double a = ms([&] {std::vector<mx::array> o; for (int l = 0; l < 48; ++l) {auto r = gdn_scan(q, k, v, decay, beta, state); o.push_back(r.output); o.push_back(r.state);} mx::eval(o);});
            const double b = ms([&] {std::vector<mx::array> o; for (int l = 0; l < 48; ++l) {auto r = gdn_scan8(q, k, v, decay, beta, state); o.push_back(r.output); o.push_back(r.state);} mx::eval(o);});
            std::printf("GDN_SCAN8_SPEED T=2048 48 layers: 32-lane %.1f ms, 8-lane %.1f ms -> %.2fx\n", a, b, a / b);
            for (int rows : {16, 32}) {
                const double c = ms([&] {std::vector<mx::array> o; for (int l = 0; l < 48; ++l) {auto r = gdn_scan8_staged(q, k, v, decay, beta, state, rows); o.push_back(r.output); o.push_back(r.state);} mx::eval(o);});
                std::printf("GDN_SCAN8_SPEED T=2048 48 layers: staged %d rows %.1f ms vs 8-lane %.1f ms -> %.2fx\n", rows, c, b, b / c);
            }
        }
    }
    std::puts(ok ? "GDN_SCAN8_PASS" : "GDN_SCAN8_FAIL");
    return ok ? 0 : 1;
}
