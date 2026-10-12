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
    int bits = 3, bool integer_levels = true) {
    auto prepare = [&] {
        if (bits == 3) return tq3_tile(load_tq3_blocks(raw, rows, k));
        auto w = load_tq4_blocks(raw, rows, k); w.integer_levels = integer_levels;
        return tq4_tile(w);
    };
    auto tiled = prepare();
    check_layout(tiled, raw);
    const auto values = inputs(8, k);
    auto x = mx::array(values.data(), {8, k});
    constexpr int levels[] = {14, 45, 79, 127, -14, -45, -79, -127};
    constexpr int levels4[] = {6,18,31,44,58,75,96,127,-6,-18,-31,-44,-58,-75,-96,-127};
    constexpr float procedural[] = {0.129099213f,0.387297638f,0.667012598f,0.946727559f,1.247959055f,1.613740157f,2.065587402f,2.732600000f,-0.129099213f,-0.387297638f,-0.667012598f,-0.946727559f,-1.247959055f,-1.613740157f,-2.065587402f,-2.732600000f};
    std::vector<float> original(size_t(8) * rows), rounded(original.size());
    for (int m = 0; m < 8; ++m) for (int r = 0; r < rows; ++r) {
        double sum = 0, half_sum = 0;
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
        }
        original[size_t(m) * rows + r] = float(sum);
        rounded[size_t(m) * rows + r] = float(half_sum);
    }
    auto cpu = mx::array(rounded.data(), {8, rows}), full = mx::array(original.data(), {8, rows});
    for (int split : {1, 2, 4}) for (int tile_k : {32, 128}) for (bool fp32 : {false, true}) {
        if (fp32 && tile_k != 32) continue;
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
static void bench(const GgufFile& file, const std::string& name) {
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
        for (int tile_k : {32, 128}) for (int split : {1, 2, 4}) for (bool prefetch : {false, true}) for (bool fp32 : {false, true}) {
            if (fp32 && tile_k != 32) continue;
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
int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 2 && argc != 3) throw std::invalid_argument("usage: test-mlx-tq3-tiled [model.gguf [--cpu-only]]");
        if (argc == 3 && std::string(argv[2]) != "--cpu-only") throw std::invalid_argument("unknown test option");
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
        if (argc >= 2) {
            auto file = gguf_open(argv[1]);
            std::vector<std::string> names;
            for (const auto& [name, t] : file.tensors)
                if ((t.type == GGML_TQ3 || t.type == GGML_TQ4) && t.ne.size() == 2 &&
                    (name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_down.weight")) &&
                    t.ne[0] * t.ne[1] > 1000000) {
                    bool seen = false;
                    for (const auto& n : names)
                        if (file.tensors.at(n).ne[0] == t.ne[0] && file.tensors.at(n).type == t.type) seen = true;
                    if (!seen) names.push_back(name);
                }
            if (names.size() < 2) throw std::runtime_error("missing real FFN shapes");
            for (const auto& name : names) {
                auto t = file.tensors.at(name); t.bytes = t.bytes / t.ne[1] * 33; t.ne[1] = 33;
                auto raw = file.read(t); check_projection({raw.begin(), raw.end()}, 33, int(t.ne[0]), true,
                    t.type == GGML_TQ3 ? 3 : 4);
            }
            std::cout << "TILED_REAL_CPU_GATE_PASS" << std::endl;
            if (argc == 2) for (const auto& name : names) bench(file, name);
        }
        std::cout << "TILED_SCREEN_COMPLETE" << std::endl;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
