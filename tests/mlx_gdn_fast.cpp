// MLX's built-in gated delta recurrence (mx::fast::gated_delta_update, MLX >= 0.32.3) against the engine's gdn_scan on
// Qwen3.8-27B shapes (16 key heads, 48 value heads, 128 x 128): agreement, bitwise behaviour across call splits, and time
// per layer call (48 calls per eval, like the 48 GDN layers). Our value head h reads key head h % KH; MLX reads h / (VH / KH),
// so inputs are permuted into MLX order and outputs back.
#include "gdn.hpp"
#include <mlx/fast.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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
mx::array unit_rows(const mx::array& x) { return mx::multiply(x, mx::rsqrt(mx::sum(mx::square(x), -1, true))); }
struct Inputs { mx::array q, k, v, decay, beta, state; };
Inputs make(std::mt19937& rng, int t) {
    return {unit_rows(random_array(rng, {1, t, KH, D}, -1, 1)), unit_rows(random_array(rng, {1, t, KH, D}, -1, 1)),
            random_array(rng, {1, t, VH, D}, -1, 1), random_array(rng, {1, t, VH}, 0.85f, 1.0f),
            random_array(rng, {1, t, VH}, 0.0f, 1.0f), random_array(rng, {1, VH, D, D}, -0.1f, 0.1f)};
}
// MLX head n reads key n / f; our head h = (n % f) * KH + n / f reads key h % KH = n / f
mx::array to_mlx_order() {
    std::vector<int32_t> idx(VH);
    const int f = VH / KH;
    for (int n = 0; n < VH; ++n) idx[n] = (n % f) * KH + n / f;
    return mx::array(idx.data(), {VH}, mx::int32);
}
mx::array from_mlx_order() {
    std::vector<int32_t> inv(VH);
    const int f = VH / KH;
    for (int n = 0; n < VH; ++n) inv[(n % f) * KH + n / f] = n;
    return mx::array(inv.data(), {VH}, mx::int32);
}
std::pair<mx::array, mx::array> mlx_scan(const Inputs& in) {
    static const auto to = to_mlx_order(), from = from_mlx_order();
    auto r = mx::fast::gated_delta_update(in.q, in.k, mx::take(in.v, to, 2), mx::take(in.decay, to, 2),
                                          mx::take(in.beta, to, 2), mx::take(in.state, to, 1));
    return {mx::take(r[0], from, 2), mx::take(r[1], from, 1)};
}
Inputs slice_t(const Inputs& in, int a, int b, const mx::array& state) {
    auto s4 = [&](const mx::array& x) { auto e = x.shape(); e[1] = b; mx::Shape s(x.ndim(), 0); s[1] = a; return mx::slice(x, s, e); };
    return {s4(in.q), s4(in.k), s4(in.v), s4(in.decay), s4(in.beta), state};
}
double rel(const mx::array& a, const mx::array& b) {
    auto d = mx::max(mx::abs(mx::subtract(a, b))), p = mx::max(mx::abs(b));
    mx::eval(d, p);
    return double(d.item<float>()) / double(p.item<float>());
}
bool same_bits(mx::array a, mx::array b) {
    a = mx::contiguous(a); b = mx::contiguous(b); mx::eval(a, b);
    return a.size() == b.size() && std::memcmp(a.data<float>(), b.data<float>(), a.nbytes()) == 0;
}
double time_us(const std::function<std::vector<mx::array>()>& round, int reps) {
    mx::eval(round());
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) mx::eval(round());
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / reps;
}
}  // namespace

int main() {
    std::mt19937 rng(11);
    bool ok = true;
    for (int t : {1, 8, 15, 16, 64, 2048}) {
        auto in = make(rng, t);
        auto ours = gdn_scan(in.q, in.k, in.v, in.decay, in.beta, in.state);
        auto [y, s] = mlx_scan(in);
        const double ry = rel(y, ours.output), rs = rel(s, ours.state);
        const bool pass = ry < (t >= 16 ? 2e-3 : 2e-5) && rs < (t >= 16 ? 2e-3 : 2e-5);
        ok &= pass;
        std::printf("GDN_FAST agree T=%d out rel %.2e state rel %.2e %s\n", t, ry, rs, pass ? "PASS" : "FAIL");
    }
    {   // sequential path (T < 16): one 15-token call vs 8 + 7 and vs 15 one-token calls, bit for bit
        auto in = make(rng, 15);
        auto [y, s] = mlx_scan(in);
        auto [y8, s8] = mlx_scan(slice_t(in, 0, 8, in.state));
        auto [y7, s7] = mlx_scan(slice_t(in, 8, 15, s8));
        bool split = same_bits(mx::concatenate({y8, y7}, 1), y) && same_bits(s7, s);
        auto st = in.state; std::vector<mx::array> ys;
        for (int i = 0; i < 15; ++i) { auto [yi, si] = mlx_scan(slice_t(in, i, i + 1, st)); ys.push_back(yi); st = si; }
        bool chain = same_bits(mx::concatenate(ys, 1), y) && same_bits(st, s);
        std::printf("GDN_FAST sequential invariance: 15 vs 8+7 %d, 15 vs 15x1 %d\n", int(split), int(chain));
    }
    for (int t : {1, 8, 15, 16, 2048}) {
        std::vector<Inputs> layers;
        for (int l = 0; l < 48; ++l) layers.push_back(make(rng, t));
        for (auto& in : layers) mx::eval(in.q, in.k, in.v, in.decay, in.beta, in.state);
        const int reps = t >= 2048 ? 3 : 20;
        double a = time_us([&] { std::vector<mx::array> o; for (auto& in : layers) { auto r = gdn_scan(in.q, in.k, in.v, in.decay, in.beta, in.state); o.push_back(r.output); o.push_back(r.state); } return o; }, reps);
        double b = time_us([&] { std::vector<mx::array> o; for (auto& in : layers) { auto r = mlx_scan(in); o.push_back(r.first); o.push_back(r.second); } return o; }, reps);
        double c = time_us([&] { std::vector<mx::array> o; for (auto& in : layers) { auto r = mx::fast::gated_delta_update(in.q, in.k, in.v, in.decay, in.beta, in.state); o.push_back(r[0]); o.push_back(r[1]); } return o; }, reps);
        std::printf("GDN_FAST_BENCH T=%d 48 layers: ours %.0f us, mlx %.0f us (with head permutes), mlx bare %.0f us\n", t, a, b, c);
    }
    std::printf(ok ? "GDN_FAST_PASS\n" : "GDN_FAST_FAIL\n");
    return ok ? 0 : 1;
}
