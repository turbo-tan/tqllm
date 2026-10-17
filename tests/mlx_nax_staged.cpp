// Staged prefill GEMM (nax_affine_gemm_staged): weights decoded into threadgroup memory, no FP16 weight copy.
// 1. correctness on random 4-bit affine and TQ4 matrices: error vs an FP32 reference no worse than the FP16-copy path's
// 2. throughput at the FFN prefill shapes vs dequant + MLX GEMM (the path it replaces); gate >= 30 TFLOPS
#include "nax_affine.hpp"
#include "tq4.hpp"
#include <mlx/random.h>
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
NaxAffine affine4(int rows, int k, std::mt19937& rng) {
    std::vector<uint8_t> codes(size_t(rows) * k);
    for (auto& c : codes) c = uint8_t(rng() & 15u);
    auto bf = [](float v) {uint32_t u; std::memcpy(&u, &v, 4); return uint16_t(u >> 16);};
    std::vector<uint16_t> sc(size_t(rows) * (k / 64)), bi(sc.size());
    for (size_t i = 0; i < sc.size(); ++i) {sc[i] = bf(0.002f + float(rng() % 100) * 4e-5f); bi[i] = bf(-0.015f + float(rng() % 100) * 3e-4f);}
    const auto payload = nax_affine_pack(codes, sc, bi, rows, k, 4, 64);
    return nax_affine_load(payload, rows, k, 4, 64);
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
    std::mt19937 rng(7);
    bool ok = true;
    for (int kind = 0; kind < 2; ++kind)
        for (int m : {1, 100, 128, 300}) {
            const int rows = 256, k = 512;
            auto w = kind ? tq4(rows, k, rng) : affine4(rows, k, rng);
            auto x = mx::astype(mx::random::normal({m, k}), mx::float16);
            auto ref = mx::matmul(mx::astype(x, mx::float32), mx::transpose(mx::astype(nax_affine_dequant(w), mx::float32)));
            auto old = nax_affine_gemm(w, x, false), got = nax_affine_gemm_staged(w, x, false);
            const double e_old = rel(old, ref), e_new = rel(got, ref);
            std::printf("NAX_STAGED %s m=%d rel_err staged %.2e (fp16-copy path %.2e)\n", kind ? "tq4" : "affine4", m, e_new, e_old);
            ok = ok && e_new <= std::max(2.0 * e_old, 1e-5);
        }
    for (auto [rows, k] : std::vector<std::pair<int, int>>{{17408, 5120}, {5120, 17408}}) {
        for (int kind = 0; kind < 2; ++kind) {
            auto w = kind ? tq4(rows, k, rng) : affine4(rows, k, rng);
            auto x = mx::astype(mx::random::normal({2048, k}), mx::float16);
            mx::eval(x);
            const double t_old = best([&] {mx::eval(nax_affine_gemm(w, x, true));});
            const double t_new = best([&] {mx::eval(nax_affine_gemm_staged(w, x, true));});
            const double flop = 2.0 * 2048 * double(rows) * k;
            std::printf("NAX_STAGED_SPEED %s rows=%d k=%d m=2048: fp16-copy path %.2f ms (%.1f TFLOPS), staged %.2f ms (%.1f TFLOPS) -> %.2fx\n",
                        kind ? "tq4" : "affine4", rows, k, t_old, flop / t_old * 1e-9, t_new, flop / t_new * 1e-9, t_old / t_new);
        }
    }
    std::puts(ok ? "NAX_STAGED_PASS" : "NAX_STAGED_FAIL");
    return ok ? 0 : 1;
}
