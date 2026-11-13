// TQ4 with integer levels as 8-bit affine weights on MLX's quantized_matmul (affine8_from_tq4): the path for GPUs without
// the M5 matrix unit. The repack holds the same values, so its product must match the TQ4 kernel's.
// `test-mlx-tq4-affine8 bench`: per-matrix time of the TQ4 kernel, the 8-bit repack and a 4-bit affine matrix of the same
// shape (the 27B's FFN shapes, 1 / 16 / 64 rows).
#include "weights.hpp"
#include "prefill.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
Tq4Weights random_tq4(int rows, int k, std::mt19937& rng) {
    std::vector<uint8_t> codes(size_t(rows) * k / 2);
    for (auto& c : codes) c = uint8_t(rng());
    std::uniform_real_distribution<float> sd(0.0002f, 0.002f);
    std::vector<mx::float16_t> s(size_t(rows) * k / 128);
    for (auto& v : s) v = mx::float16_t(sd(rng));
    Tq4Weights w{mx::array(codes.data(), {rows, k / 2}, mx::uint8), mx::array(s.data(), {rows, k / 128}, mx::float16), true};
    mx::eval(w.codes, w.scales);
    return w;
}
mx::array random_rows(int m, int k, std::mt19937& rng) {
    std::uniform_real_distribution<float> xd(-2.0f, 2.0f);
    std::vector<float> x(size_t(m) * k);
    for (auto& v : x) v = float(mx::float16_t(xd(rng)));
    return mx::array(x.data(), {m, k}, mx::float32);
}
std::vector<float> host(const mx::array& y) {
    auto c = mx::contiguous(mx::astype(y, mx::float32)); mx::eval(c);
    return std::vector<float>(c.data<float>(), c.data<float>() + c.size());
}
bool same_product(int rows, int k, int m, std::mt19937& rng) {
    const auto t = random_tq4(rows, k, rng);
    const auto a = affine8_from_tq4(t);
    const auto X = random_rows(m, k, rng);
    // the stored values on the host: scale * level, against the repack's scale * code + bias
    static const float level[16] = {6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};
    mx::eval(a.blocks, a.scales, a.biases);
    const auto* codes = t.codes.data<uint8_t>(); const auto* words = a.blocks.data<uint32_t>();
    const auto* scale = a.scales.data<mx::float16_t>(); const auto* bias = a.biases.data<mx::float16_t>();
    for (int r = 0; r < rows; ++r) for (int c = 0; c < k; ++c) {
        const uint8_t byte = codes[size_t(r) * (k / 2) + c / 2];
        const double want = double(float(scale[size_t(r) * (k / 128) + c / 128])) * level[(c % 2 ? byte >> 4 : byte) & 15];
        const uint32_t code = (words[size_t(r) * (k / 4) + c / 4] >> (8 * (c % 4))) & 255u;
        const double got = double(float(scale[size_t(r) * (k / 128) + c / 128])) * code + double(float(bias[size_t(r) * (k / 128) + c / 128]));
        if (want != got) {std::printf("FAIL value at row %d column %d: %g vs %g\n", r, c, got, want); return false;}
    }
    const auto ref = host(linear(Matrix{t}, X)), y = host(linear(Matrix{a}, X));
    double worst = 0, peak = 0;
    for (size_t i = 0; i < y.size(); ++i) {peak = std::max(peak, double(std::abs(ref[i]))); worst = std::max(worst, double(std::abs(ref[i] - y[i])));}
    // the prefill form on GPUs without the matrix unit: one FP16 staging and a GEMM
    const auto g = host(mx::matmul(mx::astype(tq4_rotate(X), mx::float16), mx::transpose(stage_half(t))));
    double worst_g = 0;
    for (size_t i = 0; i < g.size(); ++i) worst_g = std::max(worst_g, double(std::abs(ref[i] - g[i])));
    std::printf("tq4 as affine8 rows=%d k=%d m=%d values equal, product rel_err_vs_tq4_kernel=%.3e; staged FP16 GEMM rel_err=%.3e\n", rows, k, m,
                worst / peak, worst_g / peak);
    return worst / peak < 2e-3 && worst_g / peak < 2e-3;
}
double time_us(const Matrix& w, const mx::array& X) {
    for (int i = 0; i < 3; ++i) mx::eval(linear(w, X));
    const auto t0 = std::chrono::steady_clock::now();
    const int reps = 20;
    for (int i = 0; i < reps; ++i) mx::eval(linear(w, X));
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
}
void bench(std::mt19937& rng) {
    for (auto [rows, k] : std::vector<std::pair<int, int>>{{17408, 5120}, {5120, 17408}}) {
        const auto t = random_tq4(rows, k, rng);
        auto a8 = affine8_from_tq4(t); mx::eval(a8.blocks, a8.scales, a8.biases);
        std::vector<uint32_t> q(size_t(rows) * k / 8);
        for (auto& v : q) v = uint32_t(rng());
        std::vector<mx::float16_t> s(size_t(rows) * k / 64, mx::float16_t(0.01f));
        AffineWeights a4{mx::array(q.data(), {rows, k / 8}, mx::uint32), mx::array(s.data(), {rows, k / 64}, mx::float16),
                         mx::array(s.data(), {rows, k / 64}, mx::float16), 64, 4};
        mx::eval(a4.blocks, a4.scales, a4.biases);
        for (int m : {1, 16, 64}) {
            const auto X = random_rows(m, k, rng); mx::eval(X);
            std::printf("BENCH %dx%d m %d us: tq4 kernel %.0f, affine8 %.0f, affine4 %.0f\n", rows, k, m,
                        time_us(Matrix{t}, X), time_us(Matrix{a8}, X), time_us(Matrix{a4}, X));
            // the 4-bit-size alternatives: wider column batches of the TQ4 kernel, and one BF16 staging + GEMM per call
            const auto R = tq4_rotate(X); mx::eval(R);
            auto timed = [&](auto&& f) {
                for (int i = 0; i < 3; ++i) mx::eval(f());
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 20; ++i) mx::eval(f());
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20;
            };
            std::printf("BENCH %dx%d m %d us (rotated input): tq4 cols 4 %.0f, 8 %.0f, 16 %.0f; cols 16 simds 1 %.0f; stage + gemm16 %.0f\n", rows, k, m,
                        timed([&] {return tq4_matmul(t, R, 1, 4, 2);}), timed([&] {return tq4_matmul(t, R, 1, 8, 2);}),
                        timed([&] {return tq4_matmul(t, R, 1, 16, 2);}), timed([&] {return tq4_matmul(t, R, 1, 16, 1);}),
                        timed([&] {return mx::matmul(mx::astype(R, mx::float16), mx::transpose(stage_half(t)));}));
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(4);
    if (argc > 1 && std::string(argv[1]) == "bench") {bench(rng); return 0;}
    bool ok = true;
    ok &= same_product(256, 512, 3, rng);
    ok &= same_product(128, 1024, 16, rng);
    ok &= same_product(64, 256, 40, rng);
    std::printf(ok ? "tq4 affine8: PASS\n" : "tq4 affine8: FAIL\n");
    return ok ? 0 : 1;
}
