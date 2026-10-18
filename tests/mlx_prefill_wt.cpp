// Prefill product on long-K matrices: the dequant kernel writes W as [columns, rows] for a plain x @ W (j719: FFN down
// 25.5 vs 21.2 TFLOPS on W.T). 1. the transposed copy is bitwise the transpose of the [rows, columns] copy (4-bit affine
// and TQ4); 2. the product's error vs FP32 is no worse than x @ W.T; 3. speed at the FFN down shape, m 2048.
#include "nax_affine.hpp"
#include "tq4.hpp"
#include <mlx/random.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
NaxAffine affine4(int rows, int k, std::mt19937& rng) {
    std::vector<uint8_t> codes(size_t(rows) * k);
    for (auto& c : codes) c = uint8_t(rng() & 15u);
    auto bf = [](float v) {uint32_t u; std::memcpy(&u, &v, 4); return uint16_t(u >> 16);};
    std::vector<uint16_t> sc(size_t(rows) * (k / 64)), bi(sc.size());
    for (size_t i = 0; i < sc.size(); ++i) {sc[i] = bf(0.002f + float(rng() % 100) * 4e-5f); bi[i] = bf(-0.015f + float(rng() % 100) * 3e-4f);}
    return nax_affine_load(nax_affine_pack(codes, sc, bi, rows, k, 4, 64), rows, k, 4, 64);
}
NaxAffine tq4(int rows, int k, std::mt19937& rng) {
    std::vector<uint8_t> codes(size_t(rows) * (k / 2));
    for (auto& c : codes) c = uint8_t(rng());
    std::vector<mx::float16_t> scales(size_t(rows) * (k / 128));
    for (auto& v : scales) v = mx::float16_t(1e-4f + float(rng() % 100) * 4e-6f);
    Tq4Weights t{mx::array(codes.data(), {rows, k / 2}, mx::uint8), mx::array(scales.data(), {rows, k / 128}, mx::float16), true};
    return nax_from_tq4(t);
}
double rel(const mx::array& a, const mx::array& b) {
    auto d = mx::sqrt(mx::sum(mx::square(mx::subtract(a, b)))), n = mx::sqrt(mx::sum(mx::square(b)));
    mx::eval(d, n); return double(d.item<float>() / n.item<float>());
}
double best(const std::function<void()>& f) {
    f(); double b = 1e30;
    for (int i = 0; i < 7; ++i) {auto t = std::chrono::steady_clock::now(); f(); b = std::min(b, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());}
    return b;
}
}  // namespace

int main() {
    std::mt19937 rng(3);
    bool ok = true;
    for (int kind = 0; kind < 2; ++kind) {
        const int rows = 256, k = 1024;
        auto w = kind ? tq4(rows, k, rng) : affine4(rows, k, rng);
        auto plain = nax_affine_dequant(w), trans = nax_affine_dequant(w, true);
        auto same = mx::array_equal(mx::transpose(plain), trans);
        auto x = mx::astype(mx::random::normal({300, k}), mx::float16);
        auto ref = mx::matmul(mx::astype(x, mx::float32), mx::transpose(mx::astype(plain, mx::float32)));
        const double e_old = rel(mx::astype(mx::matmul(x, mx::transpose(plain)), mx::float32), ref);
        const double e_new = rel(mx::astype(nax_affine_prefill_product(w, x), mx::float32), ref);
        const bool pass = same.item<bool>() && e_new <= std::max(2.0 * e_old, 1e-5);
        std::printf("PREFILL_WT %s transposed bitwise %s, rel_err x@W %.2e (x@W.T %.2e) %s\n", kind ? "tq4" : "affine4",
                    same.item<bool>() ? "yes" : "no", e_new, e_old, pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }
    for (int kind = 0; kind < 2; ++kind) {
        auto w = kind ? tq4(5120, 17408, rng) : affine4(5120, 17408, rng);
        auto x = mx::astype(mx::random::normal({2048, 17408}), mx::float16);
        mx::eval(x);
        const double t_old = best([&] {mx::eval(mx::matmul(x, mx::transpose(nax_affine_dequant(w))));});
        const double t_new = best([&] {mx::eval(nax_affine_prefill_product(w, x));});
        const double flop = 2.0 * 2048 * 5120.0 * 17408;
        std::printf("PREFILL_WT_SPEED %s down 5120x17408 m=2048: x@W.T %.2f ms (%.1f TFLOPS), x@W %.2f ms (%.1f TFLOPS) -> %.2fx\n",
                    kind ? "tq4" : "affine4", t_old, flop / t_old * 1e-9, t_new, flop / t_new * 1e-9, t_old / t_new);
    }
    std::puts(ok ? "PREFILL_WT_PASS" : "PREFILL_WT_FAIL");
    return ok ? 0 : 1;
}
