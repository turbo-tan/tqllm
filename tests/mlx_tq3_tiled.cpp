#include "tq3_tiled.hpp"
#include "weights.hpp"
#include "../src/dequant.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm;
using namespace tqllm::mlx_backend;
static uint32_t word(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
static std::vector<uint8_t> fixture(int rows, int k, int bits = 3) {
    const int payload = bits == 3 ? 50 : 66;
    std::vector<uint8_t> raw(size_t(rows) * (k / 128) * payload);
    for (int r = 0; r < rows; ++r) for (int g = 0; g < k / 128; ++g) {
        auto* p = raw.data() + (size_t(r) * (k / 128) + g) * payload;
        uint16_t scale = uint16_t(0x1c00 + (r * 17 + g * 31) % 1024);
        if (r % 3 == 1) scale |= 0x8000;
        p[0] = uint8_t(scale); p[1] = uint8_t(scale >> 8);
        for (int b = 0; b < 4; ++b) {
            uint32_t planes[4] = {};
            for (int j = 0; j < 32; ++j) {
                uint32_t code = uint32_t((r * 3 + g * 5 + b * 7 + j) % (1 << bits));
                if (bits == 3) {
                    planes[j / 16] |= (code & 3u) << (4 * (j % 8) + 2 * ((j % 16) / 8));
                    planes[2] |= (code >> 2) << (4 * (j % 8) + j / 8);
                } else planes[j / 8] |= code << (4 * (j % 8));
            }
            for (int a = 0; a < bits; ++a) for (int byte = 0; byte < 4; ++byte)
                p[2 + b * (bits * 4) + a * 4 + byte] = uint8_t(planes[a] >> (8 * byte));
        }
    }
    return raw;
}
static void check_layout(const Tq3TiledWeights& tiled, const std::vector<uint8_t>& raw) {
    mx::eval(tiled.planes, tiled.scales);
    const auto* codes = tiled.planes.data<uint32_t>();
    auto bits = mx::view(tiled.scales, mx::uint16); mx::eval(bits);
    const auto* scales = bits.data<uint16_t>();
    const int blocks = tiled.columns / 32;
    for (int r = 0; r < ((tiled.rows + 31) / 32) * 32; ++r) {
        for (int b = 0; b < blocks; ++b) for (int a = 0; a < tiled.bits; ++a) {
            uint32_t expected = 0;
            if (r < tiled.rows) {
                const auto* p = raw.data() + (size_t(r) * (blocks / 4) + b / 4) * (tiled.bits == 3 ? 50 : 66);
                expected = word(p + 2 + (b % 4) * (tiled.bits * 4) + a * 4);
            }
            const size_t physical = ((size_t(r / 32) * blocks + b) * tiled.bits + a) * 32 + r % 32;
            if (codes[physical] != expected) throw std::runtime_error("tiled code payload differs from raw GGUF");
        }
        for (int g = 0; g < blocks / 4; ++g) {
            uint16_t expected = 0;
            if (r < tiled.rows) {
                const auto* p = raw.data() + (size_t(r) * (blocks / 4) + g) * (tiled.bits == 3 ? 50 : 66);
                expected = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
            }
            if (scales[(size_t(r / 32) * (blocks / 4) + g) * 32 + r % 32] != expected)
                throw std::runtime_error("tiled scale payload differs from raw GGUF");
        }
    }
}
static std::vector<float> inputs(int m, int k) {
    std::vector<float> values(size_t(m) * k);
    for (size_t i = 0; i < values.size(); ++i) values[i] = .1f * std::sin(float(i) * .037f);
    return values;
}
static double rel_l2(const mx::array& a, const mx::array& b) {
    mx::eval(a, b);
    double error = 0, norm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a.data<float>()[i]) - b.data<float>()[i];
        error += d * d; norm += double(b.data<float>()[i]) * b.data<float>()[i];
    }
    return std::sqrt(error / std::max(norm, 1e-30));
}
static void check_projection(const std::vector<uint8_t>& raw, int rows, int k, bool original_gate,
    int bits = 3, bool integer_levels = true, bool compensated = false, bool expanded_screen = false, bool cooperative_screen = false, bool register_screen = false, bool integer_screen = false, bool packed_cache = false, bool compensated_input = false, bool group_scaled = false) {
    auto prepare = [&] {
        if (bits == 3) return tq3_tile(load_tq3_blocks(raw, rows, k));
        auto w = load_tq4_blocks(raw, rows, k); w.integer_levels = integer_levels;
        return tq4_tile(w);
    };
    auto tiled = prepare();
    check_layout(tiled, raw);
    auto values = inputs(8, k);
    if (compensated) {
        for (size_t i = 0; i < values.size(); ++i) {
            if (i % 19 == 0) values[i] *= 100.0f;
            if (i % 23 == 0) values[i] *= 1e-5f;
            const float hi = float(_Float16(values[i]));
            const float lo = float(_Float16((values[i] - hi) * 2048.0f));
            const double reconstructed = double(hi) + double(lo) / 2048.0;
            if (std::abs(reconstructed - values[i]) > 2e-7 * std::abs(values[i]) + 2e-11)
                throw std::runtime_error("compensated input reconstruction failed");
        }
    }
    auto x = mx::array(values.data(), {8, k});
    constexpr int levels[] = {14, 45, 79, 127, -14, -45, -79, -127};
    constexpr int levels4[] = {6,18,31,44,58,75,96,127,-6,-18,-31,-44,-58,-75,-96,-127};
    constexpr float procedural[] = {0.129099213f,0.387297638f,0.667012598f,0.946727559f,1.247959055f,1.613740157f,2.065587402f,2.732600000f,-0.129099213f,-0.387297638f,-0.667012598f,-0.946727559f,-1.247959055f,-1.613740157f,-2.065587402f,-2.732600000f};
    std::vector<float> original(size_t(8) * rows), rounded(original.size()), input_rounded(original.size());
    for (int m = 0; m < 8; ++m) for (int r = 0; r < rows; ++r) {
        double sum = 0, half_sum = 0, input_half_sum = 0;
        for (int j = 0; j < k; ++j) {
            const auto* p = raw.data() + (size_t(r) * (k / 128) + j / 128) * (bits == 3 ? 50 : 66);
            const float scale = half_to_float(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
            float weight;
            if (bits == 3) {
                const auto* b = p + 2 + ((j % 128) / 32) * 12;
                const int t = j % 32;
                const uint32_t ranks = word(b + (t / 16) * 4), signs = word(b + 8);
                const int code = int((ranks >> (4 * (t % 8) + 2 * ((t % 16) / 8))) & 3u) +
                    4 * int((signs >> (4 * (t % 8) + t / 8)) & 1u);
                weight = scale * levels[code];
            } else {
                const int t = j % 128, code = (p[2 + t / 2] >> (4 * (t % 2))) & 15;
                weight = scale * (integer_levels ? float(levels4[code]) : procedural[code]);
            }
            const float w = weight, a = values[size_t(m) * k + j];
            sum += double(w) * a;
            half_sum += double(float(_Float16(w))) * float(_Float16(a));
            input_half_sum += double(w) * float(_Float16(a));
        }
        original[size_t(m) * rows + r] = float(sum);
        rounded[size_t(m) * rows + r] = float(half_sum);
        input_rounded[size_t(m) * rows + r] = float(input_half_sum);
    }
    auto cpu = mx::array(rounded.data(), {8, rows}), full = mx::array(original.data(), {8, rows});
    if (group_scaled) {
        auto expected = mx::array(input_rounded.data(), {8, rows});
        for (int bk : {32,128}) for (int split : {1,2,4}) {
            if (k % (128*split)) continue;
            auto reference = tq_compensated_matmul(tiled,x,split,bk,false);
            const double error = rel_l2(reference,expected);
            if (!std::isfinite(error) || error>2e-5)
                throw std::runtime_error("group-scaled projection differs from CPU exact weights/half input");
            for (int width : {1,3,5,8}) {
                auto got=tq_compensated_matmul(tiled,mx::array(values.data(),{width,k}),split,bk,false);
                mx::eval(got,reference);
                if (std::memcmp(got.data<float>(),reference.data<float>(),size_t(width)*rows*sizeof(float)))
                    throw std::runtime_error("group-scaled projection changes with column count");
            }
            std::cout<<"GROUP_SCALED_CPU_PASS bits="<<bits<<" rows="<<rows<<" k="<<k
                     <<" bk="<<bk<<" splits="<<split<<" contract_rel_l2="<<error
                     <<" original_rel_l2="<<rel_l2(reference,full)<<std::endl;
        }
        return;
    }
    if (cooperative_screen || register_screen) {
        for (int split : {1, 2, 4}) {
            if (k % (128 * split)) continue;
            auto reference = register_screen ? tq_register_matmul(tiled, x, split, integer_screen, packed_cache, compensated_input) :
                tq_tiled_matmul(tiled, x, split, true, 32, true, std::nullopt, 4);
            const double error = rel_l2(reference, full);
            if (!std::isfinite(error) || error > 2e-5)
                throw std::runtime_error("cooperative projection failed original CPU oracle");
            for (int width : {1, 3, 5, 8}) for (bool prefetch : {false, true}) {
                auto got = register_screen ? tq_register_matmul(tiled, mx::array(values.data(), {width, k}), split, integer_screen, packed_cache, compensated_input) :
                    tq_tiled_matmul(tiled, mx::array(values.data(), {width, k}), split, prefetch, 32, true, std::nullopt, 4);
                mx::eval(got);
                if (std::memcmp(got.data<float>(), reference.data<float>(), size_t(width) * rows * sizeof(float)))
                    throw std::runtime_error("cooperative projection changed with width or prefetch");
            }
            std::cout << "COOPERATIVE_ORACLE_PASS bits=" << bits << " rows=" << rows << " k=" << k
                      << " group_scaled=" << group_scaled << " compensated_input=" << compensated_input << " packed_cache=" << packed_cache << " integer_operand=" << integer_screen << " register_operand=" << register_screen << " splits=" << split << " original_cpu_rel_l2=" << error << std::endl;
        }
        return;
    }
    if (expanded_screen) {
        auto expanded = tq_expand_tiled_codes(tiled); mx::eval(expanded);
        const auto* words = expanded.data<uint32_t>();
        for (int r = 0; r < ((rows + 31) / 32) * 32; ++r) for (int j = 0; j < k; ++j) {
            int expected = 0;
            if (r < rows) {
                const auto* p = raw.data() + (size_t(r) * (k / 128) + j / 128) * (bits == 3 ? 50 : 66);
                if (bits == 3) {
                    const int t = j % 32;
                    const auto* b = p + 2 + ((j % 128) / 32) * 12;
                    const uint32_t ranks = word(b + (t / 16) * 4), sign = word(b + 8);
                    const int code = int((ranks >> (4 * (t % 8) + 2 * ((t % 16) / 8))) & 3u) +
                        4 * int((sign >> (4 * (t % 8) + t / 8)) & 1u);
                    expected = levels[code];
                } else {
                    const int t = j % 128, code = (p[2 + t / 2] >> (4 * (t % 2))) & 15;
                    expected = levels4[code];
                }
            }
            const size_t at = (size_t(r / 32) * (k / 4) + j / 4) * 32 + r % 32;
            const uint32_t byte = (words[at] >> (8 * (j % 4))) & 255u;
            const int actual = byte < 128 ? int(byte) : int(byte) - 256;
            if (actual != expected) throw std::runtime_error("expanded code differs from raw GGUF level");
        }
        for (int split : {1, 2, 4}) {
            if (k % (128 * split)) continue;
            auto packed = tq_tiled_matmul(tiled, x, split, true, 32, true);
            auto candidate = tq_tiled_matmul(tiled, x, split, true, 32, true, expanded);
            const double error = rel_l2(candidate, full);
            if (!std::isfinite(error) || error > 2e-5 || rel_l2(candidate, packed) != 0.0)
                throw std::runtime_error("expanded projection failed CPU or packed arithmetic oracle");
            for (int width : {1, 3, 5, 8}) for (bool prefetch : {false, true}) {
                auto got = tq_tiled_matmul(tiled, mx::array(values.data(), {width, k}), split, prefetch, 32, true, expanded);
                mx::eval(got);
                if (std::memcmp(got.data<float>(), packed.data<float>(), size_t(width) * rows * sizeof(float)))
                    throw std::runtime_error("expanded projection changed with width or prefetch");
            }
            std::cout << "EXPANDED_ORACLE_PASS bits=" << bits << " rows=" << rows << " k=" << k
                      << " splits=" << split << " original_cpu_rel_l2=" << error << std::endl;
        }
        return;
    }
    if (compensated) {
        for (int split : {1, 2, 4}) for (int tile_k : {32, 128}) {
            if (k % (128 * split)) continue;
            auto reference = tq_compensated_matmul(tiled, x, split, tile_k);
            const double error = rel_l2(reference, full);
            if (!std::isfinite(error) || error > 2e-5)
                throw std::runtime_error("compensated projection failed original GGUF CPU oracle");
            for (int width : {1, 3, 5, 8}) {
                auto got = tq_compensated_matmul(tiled, mx::array(values.data(), {width, k}), split, tile_k);
                mx::eval(got);
                if (std::memcmp(got.data<float>(), reference.data<float>(), size_t(width) * rows * sizeof(float)))
                    throw std::runtime_error("compensated projection changed with width");
            }
            std::cout << "COMPENSATED_ORACLE_PASS bits=" << bits << " rows=" << rows << " k=" << k
                      << " splits=" << split << " tile_k=" << tile_k << " original_cpu_rel_l2=" << error << std::endl;
        }
        return;
    }
    for (int split : {1, 2, 4}) for (int tile_k : {32, 64, 128}) for (bool fp32 : {false, true}) {
        if (fp32 && tile_k == 128) continue;
        if (k % (128 * split)) continue;
        auto reference = tq_tiled_matmul(tiled, x, split, true, tile_k, fp32);
        const double numerical = rel_l2(reference, fp32 ? full : cpu), precision = rel_l2(reference, full);
        std::cout << "TILED_ORACLE_CHECK bits=" << bits << " integer_levels=" << integer_levels << " rows=" << rows << " k=" << k << " split=" << split << " tile_k=" << tile_k
                  << " fp32_operands=" << fp32 << " cpu_rel_l2=" << numerical << " original_rel_l2=" << precision << std::endl;
        if (!std::isfinite(numerical) || numerical > 5e-5 || !std::isfinite(precision) ||
            (original_gate && precision > 1e-3))
            throw std::runtime_error("tiled projection failed independent FP16 or original-weight CPU oracle");
        for (int width : {1, 3, 5, 8}) {
            auto slice = mx::array(values.data(), {width, k});
            for (bool prefetch : {false, true}) {
                auto got = tq_tiled_matmul(tiled, slice, split, prefetch, tile_k, fp32); mx::eval(got);
                if (std::memcmp(got.data<float>(), reference.data<float>(), size_t(width) * rows * sizeof(float)))
                    throw std::runtime_error("tiled projection changed with width or prefetch");
            }
        }
        std::cout << "TILED_ORACLE_PASS bits=" << bits << " integer_levels=" << integer_levels << " rows=" << rows << " k=" << k << " split=" << split << " tile_k=" << tile_k
                  << " fp32_operands=" << fp32 << " cpu_rel_l2=" << numerical << " original_rel_l2=" << precision << std::endl;
    }
}
template<class F> static double measure(F operation) {
    mx::eval(operation());
    std::vector<double> times;
    for (int i = 0; i < 7; ++i) {
        auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < 10; ++repeat) mx::eval(operation());
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 10);
    }
    std::sort(times.begin(), times.end()); return times[3];
}
static void bench(const GgufFile& file, const std::string& name, bool one_point = false) {
    const auto& t = file.tensors.at(name);
    const auto data = file.read(t);
    const std::vector<uint8_t> raw(data.begin(), data.end());
    std::optional<Tq3Weights> packed3;
    std::optional<Tq4Weights> packed4;
    if (t.type == GGML_TQ3) packed3 = load_tq3_blocks(raw, int(t.ne[1]), int(t.ne[0]));
    else packed4 = load_tq4_blocks(raw, int(t.ne[1]), int(t.ne[0]));
    auto tiled = packed3 ? tq3_tile(*packed3) : tq4_tile(*packed4); mx::eval(tiled.planes, tiled.scales);
    check_layout(tiled, {raw.begin(), raw.end()});
    for (int width : {1, 3, 5, 8}) {
        auto values = inputs(width, tiled.columns);
        auto x = mx::array(values.data(), {width, tiled.columns}); mx::eval(x);
        auto native = [&] { return packed3 ? tq3_matmul(*packed3, x) : tq4_matmul(*packed4, x); };
        auto expected = native(); mx::eval(expected);
        const double baseline = measure(native);
        for (int tile_k : {32, 64, 128}) for (int split : {1, 2, 4}) for (bool prefetch : {false, true}) for (bool fp32 : {false, true}) {
            if (fp32 && tile_k == 128) continue;
            if (one_point && (tile_k == 128 || split != 4 || !prefetch || !fp32)) continue;
            auto result = tq_tiled_matmul(tiled, x, split, prefetch, tile_k, fp32);
            double error = rel_l2(result, expected);
            if (!std::isfinite(error) || error > (fp32 ? 2e-5 : 1e-3)) throw std::runtime_error("full-shape projection precision gate failed");
            const double time = measure([&] { return tq_tiled_matmul(tiled, x, split, prefetch, tile_k, fp32); });
            std::cout << "TILED_BENCH tensor=" << name << " bits=" << tiled.bits << " rows=" << tiled.rows << " k=" << tiled.columns
                      << " width=" << width << " split=" << split << " prefetch=" << prefetch << " tile_k=" << tile_k
                      << " fp32_operands=" << fp32 << " baseline_ms=" << baseline << " tiled_ms=" << time
                      << " original_rel_l2=" << error << std::endl;
        }
    }
}
static void bench_compensated(const GgufFile& file, const std::string& name, bool expanded_screen = false, bool cooperative_screen = false, bool register_screen = false, bool integer_screen = false, bool packed_cache = false, bool compensated_input = false, bool group_scaled = false) {
    const auto& tensor = file.tensors.at(name);
    auto data = file.read(tensor);
    std::vector<uint8_t> raw(data.begin(), data.end());
    auto weights = tensor.type == GGML_TQ3 ?
        tq3_tile(load_tq3_blocks(raw, int(tensor.ne[1]), int(tensor.ne[0]))) :
        tq4_tile(load_tq4_blocks(raw, int(tensor.ne[1]), int(tensor.ne[0])));
    mx::eval(weights.planes, weights.scales);
    std::optional<mx::array> expanded;
    if (expanded_screen) { expanded = tq_expand_tiled_codes(weights); mx::eval(*expanded); }
    for (int width : {1, 8}) for (int tile_k : {32, 128}) {
        if ((expanded_screen || cooperative_screen || register_screen) && tile_k != 32) continue;
        auto values = inputs(width, weights.columns);
        auto x = mx::array(values.data(), {width, weights.columns}); mx::eval(x);
        auto baseline = [&] { return tq_tiled_matmul(weights, x, 4, true, 32, true); };
        auto candidate = [&] { return register_screen ? tq_register_matmul(weights, x, 4, integer_screen, packed_cache, compensated_input) : cooperative_screen ?
            tq_tiled_matmul(weights, x, 4, true, 32, true, std::nullopt, 4) : expanded_screen ?
            tq_tiled_matmul(weights, x, 4, true, 32, true, expanded) :
            tq_compensated_matmul(weights, x, 4, tile_k, !group_scaled); };
        const double error = rel_l2(candidate(), baseline());
        if (!std::isfinite(error) || error > (group_scaled ? 1e-3 : 2e-5))
            throw std::runtime_error("integer-stage full shape precision failed");
        for (int arm = 0; arm < 4; ++arm) {
            const bool comp = arm == 1 || arm == 2;
            const double elapsed = comp ? measure(candidate) : measure(baseline);
            std::cout << "COMPENSATED_BENCH tensor=" << name << " rows=" << weights.rows
                      << " k=" << weights.columns << " width=" << width << " arm=" << arm
                      << " group_scaled=" << group_scaled << " compensated_input=" << compensated_input << " packed_cache=" << packed_cache << " integer_operand=" << integer_screen << " register_operand=" << register_screen << " cooperative=" << cooperative_screen << " expanded=" << expanded_screen << " expanded_bytes=" << (expanded ? expanded->nbytes() : 0)
                      << " tile_k=" << tile_k << " compensated=" << comp << " elapsed_ms=" << elapsed
                      << " fp32_baseline_rel_l2=" << error << std::endl;
        }
    }
}
int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 2 && argc != 3) throw std::invalid_argument("usage: test-mlx-tq3-tiled [model.gguf [--cpu-only|--one-point|--mixer-cpu|--compensated|--expanded|--cooperative|--register|--register-int8|--packed-register-int8|--compensated-register-int8]]");
        if (argc == 3 && std::string(argv[2]) != "--cpu-only" && std::string(argv[2]) != "--one-point" &&
            std::string(argv[2]) != "--mixer-cpu" && std::string(argv[2]) != "--compensated" && std::string(argv[2]) != "--expanded" && std::string(argv[2]) != "--cooperative" && std::string(argv[2]) != "--register" && std::string(argv[2]) != "--register-int8" && std::string(argv[2]) != "--packed-register-int8" && std::string(argv[2]) != "--compensated-register-int8" && std::string(argv[2]) != "--group-scaled")
            throw std::invalid_argument("unknown test option");
        const bool group_scaled = argc == 3 && std::string(argv[2]) == "--group-scaled";
        const bool compensated = argc == 3 && std::string(argv[2]) == "--compensated";
        const bool expanded_screen = argc == 3 && std::string(argv[2]) == "--expanded";
        const bool cooperative_screen = argc == 3 && std::string(argv[2]) == "--cooperative";
        const bool compensated_input = argc == 3 && std::string(argv[2]) == "--compensated-register-int8";
        const bool packed_cache = compensated_input || (argc == 3 && std::string(argv[2]) == "--packed-register-int8");
        const bool integer_screen = packed_cache || (argc == 3 && std::string(argv[2]) == "--register-int8");
        const bool register_screen = integer_screen || (argc == 3 && std::string(argv[2]) == "--register");
        if (compensated || expanded_screen || cooperative_screen || register_screen || group_scaled) {
            for (int bits : {3, 4}) for (int rows : {1, 17, 33, 64}) for (int k : {128, 512}) {
                auto raw = fixture(rows, k, bits);
                const int stride = bits == 3 ? 50 : 66;
                for (int r = 0; r < rows; ++r) for (int group = 0; group < k / 128; ++group) {
                    const uint16_t edge[] = {0x0000,0x0001,0x8001,0x0400,0x8400};
                    if ((r + group) % 7 < 5) {
                        uint16_t scale = edge[(r + group) % 7];
                        size_t at = (size_t(r) * (k / 128) + group) * stride;
                        raw[at] = uint8_t(scale); raw[at + 1] = uint8_t(scale >> 8);
                    }
                }
                check_projection(raw, rows, k, false, bits, true, compensated, expanded_screen, cooperative_screen, register_screen, integer_screen, packed_cache, compensated_input, group_scaled);
            }
            auto tiled = tq3_tile(load_tq3_blocks(fixture(1, 512), 1, 512));
            for (int bad = 0; bad < 5; ++bad) {
                bool rejected = false;
                try {
                    auto w = tiled;
                    auto x = mx::zeros({1, 512}, mx::float32);
                    int split = 4;
                    if (bad == 0) w.integer_levels = false;
                    if (bad == 1) x = mx::zeros({9, 512}, mx::float32);
                    if (bad == 2) x = mx::zeros({1, 512}, mx::float16);
                    if (bad == 3) split = 3;
                    if (bad == 4) w.scales = mx::zeros({1, 4, 31}, mx::float16);
                    if (register_screen) tq_register_matmul(w, x, split, integer_screen, packed_cache, compensated_input);
                    else tq_compensated_matmul(w, x, split, 32, !group_scaled);
                } catch (const std::invalid_argument&) { rejected = true; }
                if (!rejected) throw std::runtime_error("invalid compensated layout accepted");
            }
            if (integer_screen) {
                auto procedural = load_tq4_blocks(fixture(1,512,4),1,512);
                procedural.integer_levels = false;
                bool rejected = false;
                try {tq_register_matmul(tq4_tile(procedural), mx::zeros({1,512},mx::float32),4,true);}
                catch (const std::invalid_argument&) {rejected = true;}
                if (!rejected) throw std::runtime_error("noninteger codebook accepted by INT8 operand");
            }
            std::cout << "COMPENSATED_SYNTHETIC_PASS" << std::endl;
        } else {
        // Cancellation fixtures validate the stated FP16 contract, while reporting original FP32 drift.
        for (int rows : {1, 17, 33, 64}) for (int k : {128, 512})
            check_projection(fixture(rows, k), rows, k, false);
        for (int rows : {1, 17, 33, 64}) for (int k : {128, 512})
            for (bool integer : {false, true}) check_projection(fixture(rows, k, 4), rows, k, false, 4, integer);
        auto small = load_tq3_blocks(fixture(1, 128), 1, 128);
        for (int bad = 0; bad < 3; ++bad) {
            bool caught = false;
            try {
                auto invalid = small;
                if (bad == 0) invalid.interleaved = true;
                if (bad == 1) invalid.scales = mx::zeros({2, 1}, mx::float16);
                if (bad == 2) invalid.planes = mx::zeros({1, 3, 5}, mx::uint32);
                tq3_tile(invalid);
            } catch (const std::invalid_argument&) { caught = true; }
            if (!caught) throw std::runtime_error("invalid planar layout accepted");
        }
        bool caught = false;
        try { tq3_tiled_matmul(tq3_tile(small), mx::zeros({9, 128}, mx::float32)); }
        catch (const std::invalid_argument&) { caught = true; }
        if (!caught) throw std::runtime_error("unsupported verifier width accepted");
        std::cout << "TILED_LAYOUT_AND_SYNTHETIC_PASS" << std::endl;
        }
        if (argc >= 2) {
            auto file = gguf_open(argv[1]);
            std::vector<std::string> names;
            const bool mixer=argc==3 && std::string(argv[2])=="--mixer-cpu";
            for (const auto& [name, t] : file.tensors)
                if ((t.type == GGML_TQ3 || t.type == GGML_TQ4) && t.ne.size() == 2 &&
                    (mixer?(name.ends_with(".attn_qkv.weight") || name.ends_with(".attn_gate.weight") ||
                            name.ends_with(".ssm_out.weight")):
                           (name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_down.weight"))) &&
                    t.ne[0] * t.ne[1] > 1000000) {
                    bool seen = false;
                    for (const auto& n : names)
                        if (file.tensors.at(n).ne[0] == t.ne[0] && file.tensors.at(n).type == t.type) seen = true;
                    if (!seen) names.push_back(name);
                }
            if (names.size() < 2) throw std::runtime_error("missing real projection shapes");
            for (const auto& name : names) {
                auto t = file.tensors.at(name); t.bytes = t.bytes / t.ne[1] * 33; t.ne[1] = 33;
                auto raw = file.read(t); check_projection({raw.begin(), raw.end()}, 33, int(t.ne[0]), true,
                    t.type == GGML_TQ3 ? 3 : 4, true, compensated, expanded_screen, cooperative_screen, register_screen, integer_screen, packed_cache, compensated_input, group_scaled);
            }
            std::cout << "TILED_REAL_CPU_GATE_PASS" << std::endl;
            if (compensated || expanded_screen || cooperative_screen || register_screen || group_scaled) {
                for (const auto& name : names) bench_compensated(file, name, expanded_screen, cooperative_screen, register_screen, integer_screen, packed_cache, compensated_input, group_scaled);
                std::cout << (group_scaled ? "GROUP_SCALED_SCREEN_COMPLETE" : register_screen ? "REGISTER_SCREEN_COMPLETE" : cooperative_screen ? "COOPERATIVE_SCREEN_COMPLETE" : expanded_screen ? "EXPANDED_SCREEN_COMPLETE" : "COMPENSATED_SCREEN_COMPLETE") << std::endl;
            }
            if (argc == 2 || (argc == 3 && std::string(argv[2]) == "--one-point"))
                for (const auto& name : names) bench(file, name, argc == 3);
        }
        std::cout << "TILED_SCREEN_COMPLETE" << std::endl;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
