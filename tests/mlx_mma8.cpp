#include "mma8.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
constexpr int L4[8] = {6, 18, 31, 44, 58, 75, 96, 127};
float half_round(float v) {
    mx::float16_t h(v);
    return float(h);
}
bool run(int rows, int k, int m, std::mt19937& rng) {
    std::vector<uint8_t> codes(size_t(rows) * (k / 2));
    for (auto& c : codes) c = uint8_t(rng());
    std::vector<mx::float16_t> scales(size_t(rows) * (k / 128));
    std::uniform_real_distribution<float> sd(0.01f, 0.05f), xd(-1.0f, 1.0f);
    for (auto& s : scales) s = mx::float16_t(sd(rng));
    std::vector<float> x(size_t(m) * k);
    for (auto& v : x) v = xd(rng);
    Tq4Weights w{mx::array(codes.data(), {rows, k / 2}, mx::uint8), mx::array(scales.data(), {rows, k / 128}, mx::float16), true};
    auto packed = mma8_pack(w);
    auto y = mma8_matmul(packed, mx::array(x.data(), {m, k}, mx::float32));
    auto ref = tq4_matmul(w, mx::array(x.data(), {m, k}, mx::float32));
    y = mx::contiguous(y);
    mx::eval(y, ref);
    if (y.shape() != mx::Shape{m, rows}) throw std::runtime_error("unexpected output shape");
    const float* got = y.data<float>();
    const float* old = ref.data<float>();
    double worst = 0, worst_old = 0, peak = 0;
    for (int r = 0; r < m; ++r)
        for (int n = 0; n < rows; ++n) {
            double acc = 0;
            for (int b = 0; b < k / 128; ++b) {
                double block = 0;
                for (int c = b * 128; c < (b + 1) * 128; ++c) {
                    uint8_t byte = codes[size_t(n) * (k / 2) + c / 2];
                    uint32_t code = (c % 2 ? byte >> 4 : byte) & 15u;
                    double level = (code & 8) ? -L4[code & 7] : L4[code & 7];
                    block += level * half_round(x[size_t(r) * k + c]);
                }
                acc += float(scales[size_t(n) * (k / 128) + b]) * block;
            }
            peak = std::max(peak, std::abs(acc));
            worst = std::max(worst, std::abs(acc - got[size_t(r) * rows + n]));
            worst_old = std::max(worst_old, std::abs(double(old[size_t(r) * rows + n]) - got[size_t(r) * rows + n]));
        }
    std::printf("MMA8 rows=%d k=%d m=%d max_abs_err_vs_cpu=%.3e rel=%.3e vs_old_tq4=%.3e\n", rows, k, m, worst, worst / peak, worst_old / peak);
    return worst / peak < 5e-5 && worst_old / peak < 5e-3;
}
bool run_tq3(int rows, int k, int m, std::mt19937& rng) {
    constexpr int L3[4] = {14, 45, 79, 127};
    const int blocks = k / 32;
    std::vector<uint32_t> planes(size_t(rows) * 3 * blocks);
    for (auto& v : planes) v = uint32_t(rng());
    std::vector<mx::float16_t> scales(size_t(rows) * (k / 128));
    std::uniform_real_distribution<float> sd(0.01f, 0.05f), xd(-1.0f, 1.0f);
    for (auto& v : scales) v = mx::float16_t(sd(rng));
    std::vector<float> x(size_t(m) * k);
    for (auto& v : x) v = xd(rng);
    Tq3Weights w{mx::array(planes.data(), {rows, 3, blocks}, mx::uint32), mx::array(scales.data(), {rows, k / 128}, mx::float16), false};
    auto packed = mma8_pack(w);
    auto input = mx::array(x.data(), {m, k}, mx::float32);
    auto y = mx::contiguous(mma8_matmul(packed, input));
    auto yb = mx::contiguous(mma8_matmul(packed, mma8_input(input), m, true));
    mx::eval(y, yb);
    const float* got = y.data<float>();
    const float* got_bf = yb.data<float>();
    double worst = 0, peak = 0, worst_bf = 0;
    for (int r = 0; r < m; ++r)
        for (int n = 0; n < rows; ++n) {
            double acc = 0;
            for (int b = 0; b < k / 128; ++b) {
                double block = 0;
                for (int c = b * 128; c < (b + 1) * 128; ++c) {
                    const int g = c / 32, j = c % 32;
                    const uint32_t low = planes[(size_t(n) * 3 + 0) * blocks + g], high = planes[(size_t(n) * 3 + 1) * blocks + g],
                                   sign = planes[(size_t(n) * 3 + 2) * blocks + g];
                    const uint32_t ranks = j < 16 ? low : high;
                    const uint32_t rank = (ranks >> (4 * (j % 8) + 2 * ((j % 16) / 8))) & 3u;
                    const uint32_t neg = (sign >> (4 * (j % 8) + j / 8)) & 1u;
                    block += double(neg ? -L3[rank] : L3[rank]) * half_round(x[size_t(r) * k + c]);
                }
                acc += float(scales[size_t(n) * (k / 128) + b]) * block;
            }
            peak = std::max(peak, std::abs(acc));
            worst = std::max(worst, std::abs(acc - got[size_t(r) * rows + n]));
            const uint32_t bits = std::bit_cast<uint32_t>(got[size_t(r) * rows + n]);
            const float expect_bf = std::bit_cast<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
            worst_bf = std::max(worst_bf, std::abs(double(expect_bf) - got_bf[size_t(r) * rows + n]));
        }
    std::printf("MMA8_TQ3 rows=%d k=%d m=%d rel=%.3e bf16_epilogue_diff=%.3e\n", rows, k, m, worst / peak, worst_bf);
    return worst / peak < 5e-5 && worst_bf == 0;
}
}
int main() {
    try {
        std::mt19937 rng(12345);
        bool ok = true;
        for (int m = 1; m <= 8; ++m) ok &= run(200, 512, m, rng);
        for (int m : {1, 3, 8}) ok &= run_tq3(200, 512, m, rng);
        ok &= run_tq3(1024, 5120, 8, rng);
        ok &= run(1024, 5120, 8, rng);
        ok &= run(1024, 5120, 1, rng);
        std::puts(ok ? "MMA8_PASS" : "MMA8_FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("MMA8_ERROR %s\n", e.what());
        return 2;
    }
}
