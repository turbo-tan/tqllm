#include "tq4.hpp"
#include "tq3.hpp"
#include "q6.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <bit>

namespace mx = mlx::core;
using tqllm::mlx_backend::tq4_matmul;

static void test_q6() {
    constexpr int rows = 17, k = 512;
    std::vector<uint8_t> raw(rows*(k/256)*210);
    std::vector<float> weights(rows*k);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < k/256; ++b) {
            auto* p = raw.data()+(r*(k/256)+b)*210;
            p[208] = 0; p[209] = 0x18; // FP16 1/512
            for (int g = 0; g < 16; ++g) p[192+g] = uint8_t(int8_t((g&1)?-g-1:g+1));
            for (int h = 0; h < 2; ++h)
                for (int c = 0; c < 4; ++c)
                    for (int l = 0; l < 32; ++l) {
                        const int j = h*128+c*32+l;
                        const int q = (r*13+b*7+j*3)%64;
                        p[h*64+(c%2)*32+l] |= uint8_t((q&15) << (c/2*4));
                        p[128+h*32+l] |= uint8_t((q>>4) << (2*c));
                        const int scale = int8_t(p[192+h*8+c*2+l/16]);
                        weights[r*k+b*256+j] = (q-32)*float(scale)/512.0f;
                    }
        }
    auto packed = tqllm::mlx_backend::load_q6_blocks(raw, rows, k);
    for (int m : {1, 8, 64}) {
        std::vector<float> input(m*k);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i)*0.1f);
        auto out = tqllm::mlx_backend::q6_matmul(packed, mx::array(input.data(), {m,k}));
        mx::eval(out);
        for (int a = 0; a < m; ++a)
            for (int r = 0; r < rows; ++r) {
                double expected = 0;
                for (int j = 0; j < k; ++j) expected += double(weights[r*k+j])*input[a*k+j];
                if (std::abs(out.data<float>()[a*rows+r]-expected) > 3e-4)
                    throw std::runtime_error("Q6_K code, signed scale or arithmetic failed");
            }
    }
    raw.pop_back();
    bool rejected = false;
    try { tqllm::mlx_backend::load_q6_blocks(raw, rows, k); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Q6_K truncated payload accepted");
}

static void test_tq3() {
    constexpr int rows = 17, k = 256;
    constexpr int levels[] = {14, 45, 79, 127, -14, -45, -79, -127};
    std::vector<uint8_t> codes(rows * k), blocks(rows * k / 128 * 50);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < k; ++j) codes[r*k+j] = uint8_t((r*17+j*3+(j/32)*5)&7);
    for (int g = 0; g < rows*k/128; ++g) {
        blocks[g*50] = 0;
        blocks[g*50+1] = (g&1) ? 0x30 : 0x2c; // 0.125 or 0.0625
        for (int b = 0; b < 4; ++b) {
            uint32_t planes[3] = {};
            for (int j = 0; j < 32; ++j) {
                const int c = codes[g*128+b*32+j];
                planes[j/16] |= uint32_t(c&3) << ((j%8)*4+((j%16)/8)*2);
                planes[2] |= uint32_t(c>>2) << ((j%8)*4+j/8);
            }
            for (int p = 0; p < 3; ++p)
                for (int byte = 0; byte < 4; ++byte)
                    blocks[g*50+2+(b*3+p)*4+byte] = uint8_t(planes[p] >> (byte*8));
        }
    }
    const auto weights = tqllm::mlx_backend::load_tq3_blocks(blocks, rows, k);
    for (int m : {1, 8, 64}) {
        std::vector<float> x(m*k);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i)*0.09f);
        auto out = tqllm::mlx_backend::tq3_matmul(weights, mx::array(x.data(), {m,k}));
        mx::eval(out);
        for (int a = 0; a < m; ++a)
            for (int r = 0; r < rows; ++r) {
                double expected = 0;
                for (int j = 0; j < k; ++j) {
                    const float s = ((r*k+j)/128)&1 ? 0.125f : 0.0625f;
                    expected += levels[codes[r*k+j]] * double(s) * x[a*k+j];
                }
                if (std::abs(out.data<float>()[a*rows+r]-expected) > 3e-4)
                    throw std::runtime_error("TQ3 plane decoder or scale arithmetic failed");
            }
    }
    blocks.pop_back();
    bool rejected = false;
    try { tqllm::mlx_backend::load_tq3_blocks(blocks, rows, k); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("TQ3 truncated payload accepted");
}

int main() {
    test_q6();
    test_tq3();
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
        std::vector<float> input(k);
        for (int j = 0; j < k; ++j) input[j] = std::sin(float(j) * 0.11f);
        auto physical = tq4_matmul(loaded, mx::array(input.data(), {1, k}));
        mx::eval(physical);
        for (int r = 0; r < rows; ++r) {
            double expected = 0;
            for (int j = 0; j < k; ++j) {
                int code = (codes[r * k / 2 + j / 2] >> ((j & 1) * 4)) & 15;
                expected += double(levels[code]) * 0.125 * input[j];
            }
            if (std::abs(physical.data<float>()[r] - expected) > 1e-4) {
                std::cerr << "GGUF integer-codebook scale convention failed\n";
                return 7;
            }
        }
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
