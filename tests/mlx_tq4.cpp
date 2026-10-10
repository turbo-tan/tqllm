#include "tq4.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <bit>

namespace mx = mlx::core;
using tqllm::mlx_backend::tq4_matmul;

int main() {
    constexpr int rows = 17, k = 256;
    const int levels[] = {6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};
    std::vector<uint8_t> codes(rows * k / 2);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t(i * 37 + 11);
    std::vector<float> scales(rows * k / 128, 0.125f);
    auto c = mx::array(codes.data(), {rows, k / 2}, mx::uint8);
    auto s = mx::astype(mx::array(scales.data(), {rows, k / 128}), mx::float16);
    {
        std::vector<uint8_t> blocks(rows * (k / 128) * 66);
        for (int b = 0; b < rows * k / 128; ++b) {
            blocks[b * 66] = 0;
            blocks[b * 66 + 1] = 0x30; // FP16 0.125
            std::copy_n(codes.data() + b * 64, 64, blocks.data() + b * 66 + 2);
        }
        auto loaded = tqllm::mlx_backend::load_tq4_blocks(blocks, rows, k);
        auto same_codes = mx::all(mx::equal(loaded.codes, c));
        auto same_scales = mx::all(mx::equal(loaded.scales, s));
        if (!same_codes.item<bool>() || !same_scales.item<bool>()) return 5;
        blocks.pop_back();
        bool rejected = false;
        try { tqllm::mlx_backend::load_tq4_blocks(blocks, rows, k); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) return 6;
    }
    int cases = 0;
    {
        std::vector<float> input(2 * k);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i) * 0.07f);
        auto rotated = tqllm::mlx_backend::tq4_rotate(mx::array(input.data(), {2, k}));
        mx::eval(rotated);
        constexpr uint32_t signs[] = {0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9};
        for (int b = 0; b < 4; ++b)
            for (unsigned i = 0; i < 128; ++i) {
                double expected = 0;
                for (unsigned j = 0; j < 128; ++j) {
                    const int sign = 1 - 2 * ((signs[j / 32] >> (j % 32)) & 1);
                    const int h = 1 - 2 * (std::popcount(i & j) & 1);
                    expected += sign * h * input[b * 128 + j];
                }
                if (std::abs(rotated.data<float>()[b * 128 + i] - expected / std::sqrt(128.0)) > 1e-4) {
                    std::cerr << "TQ4 signed rotation failed\n";
                    return 4;
                }
            }
    }
    for (int m : {1, 8, 64}) {
        std::vector<float> x(m * k), expected(m * rows);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.13f);
        for (int a = 0; a < m; ++a)
            for (int r = 0; r < rows; ++r) {
                double sum = 0;
                for (int j = 0; j < k; ++j) {
                    int code = (codes[r * k / 2 + j / 2] >> ((j & 1) * 4)) & 15;
                    sum += double(levels[code]) * (2.7326 / 127.0) * 0.125 * x[a * k + j];
                }
                expected[a * rows + r] = float(sum);
            }
        auto input = mx::array(x.data(), {m, k});
        for (int rps : {1, 2, 4, 8})
            for (int cols : {1, 4, 16})
            for (int simds : {1, 2, 4, 8}) {
                auto out = tq4_matmul(c, s, input, rps, cols, simds);
                mx::eval(out);
                float max_error = 0;
                for (size_t i = 0; i < expected.size(); ++i)
                    max_error = std::max(max_error, std::abs(out.data<float>()[i] - expected[i]));
                if (max_error > 1e-4f) {
                    std::cerr << "TQ4 arithmetic failed: m=" << m << " rps=" << rps << " cols=" << cols << " error=" << max_error << '\n';
                    return 1;
                }
                ++cases;
            }
    }
    auto input = mx::zeros({1, k});
    for (int invalid : {0, 3, 16}) {
        bool rejected = false;
        try { tq4_matmul(c, s, input, invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) return 2;
    }
    bool rejected = false;
    try { tq4_matmul(c, s, mx::zeros({1, k / 2})); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 3;
    std::cout << "MLX_TQ4_PASS " << cases << " arithmetic cases; invalid tiles and shapes rejected\n";
}
