// Prefill through mx::quantized_matmul (nax_affine_to_mlx + nax_affine_product_fp16):
// 1. the repacked words hold every code at MLX's position (row r, column k -> word k / 8, nibble k % 8) and the FP16
//    scales / biases equal the stored BF16 values, for group 64 and 128
// 2. the product's error vs an FP32 reference is no worse than the dequant + matmul path's
// 3. speed at the prefill shapes (m 2048): repack, quantized_matmul and the dequant + matmul path
#include "nax_affine.hpp"
#include <mlx/random.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <random>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
uint16_t bf(float v) {uint32_t u; std::memcpy(&u, &v, 4); return uint16_t(u >> 16);}
float from_bf(uint16_t b) {uint32_t u = uint32_t(b) << 16; float v; std::memcpy(&v, &u, 4); return v;}
struct Case {std::vector<uint8_t> codes; std::vector<uint16_t> sc, bi; std::optional<NaxAffine> nax;
    const NaxAffine& w() const {return *nax;}};
Case affine4(int rows, int k, int group, std::mt19937& rng) {
    Case c;
    c.codes.resize(size_t(rows) * k);
    for (auto& v : c.codes) v = uint8_t(rng() & 15u);
    c.sc.resize(size_t(rows) * (k / group)); c.bi.resize(c.sc.size());
    for (size_t i = 0; i < c.sc.size(); ++i) {c.sc[i] = bf(0.002f + float(rng() % 100) * 4e-5f); c.bi[i] = bf(-0.015f + float(rng() % 100) * 3e-4f);}
    c.nax = nax_affine_load(nax_affine_pack(c.codes, c.sc, c.bi, rows, k, 4, group), rows, k, 4, group);
    return c;
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
    std::mt19937 rng(11);
    bool ok = true;
    for (int group : {64, 128}) {
        const int rows = 256, k = 512;
        auto c = affine4(rows, k, group, rng);
        auto q = nax_affine_to_mlx(c.w());
        mx::eval(q.words, q.scales, q.biases);
        const uint32_t* words = q.words.data<uint32_t>();
        const mx::float16_t* s = q.scales.data<mx::float16_t>();
        const mx::float16_t* b = q.biases.data<mx::float16_t>();
        size_t bad_codes = 0, bad_scales = 0;
        for (int r = 0; r < rows; ++r)
            for (int x = 0; x < k; ++x)
                bad_codes += ((words[size_t(r) * (k / 8) + x / 8] >> (4 * (x % 8))) & 15u) != c.codes[size_t(r) * k + x];
        for (size_t i = 0; i < c.sc.size(); ++i)
            bad_scales += float(s[i]) != from_bf(c.sc[i]) || float(b[i]) != from_bf(c.bi[i]);
        std::printf("NAX_QMM layout group %d: wrong codes %zu / %d, wrong scales or biases %zu / %zu\n", group, bad_codes, rows * k, bad_scales, c.sc.size());
        ok = ok && bad_codes == 0 && bad_scales == 0;
        for (int m : {100, 300}) {
            auto x = mx::astype(mx::random::normal({m, k}), mx::float16);
            auto ref = mx::matmul(mx::astype(x, mx::float32), mx::transpose(mx::astype(nax_affine_dequant(c.w()), mx::float32)));
            auto old = mx::matmul(x, mx::transpose(nax_affine_dequant(c.w())));
            auto got = mx::quantized_matmul(x, q.words, q.scales, q.biases, true, group, 4);
            const double e_old = rel(mx::astype(old, mx::float32), ref), e_new = rel(mx::astype(got, mx::float32), ref);
            std::printf("NAX_QMM group %d m=%d rel_err quantized_matmul %.2e (dequant + matmul %.2e)\n", group, m, e_new, e_old);
            ok = ok && e_new <= std::max(2.0 * e_old, 1e-5);
        }
    }
    for (auto [rows, k] : std::vector<std::pair<int, int>>{{17408, 5120}, {5120, 17408}, {10240, 5120}}) {
        auto c = affine4(rows, k, 64, rng);
        auto x = mx::astype(mx::random::normal({2048, k}), mx::float16);
        mx::eval(x);
        const double t_old = best([&] {mx::eval(mx::matmul(x, mx::transpose(nax_affine_dequant(c.w()))));});
        const double t_pack = best([&] {auto q = nax_affine_to_mlx(c.w()); mx::eval(q.words, q.scales, q.biases);});
        auto q = nax_affine_to_mlx(c.w());
        mx::eval(q.words, q.scales, q.biases);
        const double t_qmm = best([&] {mx::eval(mx::quantized_matmul(x, q.words, q.scales, q.biases, true, 64, 4));});
        const double t_new = best([&] {mx::eval(nax_affine_product_fp16(c.w(), x));});
        const double flop = 2.0 * 2048 * double(rows) * k;
        std::printf("NAX_QMM_SPEED rows=%d k=%d m=2048: dequant+matmul %.2f ms (%.1f TFLOPS), repack %.2f ms + qmm %.2f ms, "
                    "product %.2f ms (%.1f TFLOPS) -> %.2fx\n", rows, k, t_old, flop / t_old * 1e-9, t_pack, t_qmm, t_new,
                    flop / t_new * 1e-9, t_old / t_new);
    }
    std::puts(ok ? "NAX_QMM_PASS" : "NAX_QMM_FAIL");
    return ok ? 0 : 1;
}
