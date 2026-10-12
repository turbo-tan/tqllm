#include "weights.hpp"
#include "tq3_tiled.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

template<class F> double median_ms(F operation) {
    mx::eval(operation());
    std::vector<double> samples;
    for (int i = 0; i < 7; ++i) {
        auto start = std::chrono::steady_clock::now();
        mx::eval(operation());
        samples.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

float relative_l2(const mx::array& value, const mx::array& reference) {
    auto delta = mx::subtract(mx::astype(value, mx::float32), reference);
    return mx::sqrt(mx::divide(mx::sum(mx::square(delta)),
        mx::maximum(mx::sum(mx::square(reference)), mx::array(1e-20f)))).item<float>();
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: bench-mlx-repack model.gguf tensor");
        auto file = tqllm::gguf_open(argv[1]);
        auto original = load_matrix(file, argv[2]);
        auto exact = [&] {
            if (auto w = std::get_if<Tq3Weights>(&original)) return tq3_affine8(*w);
            if (auto w = std::get_if<Tq4Weights>(&original)) return tq4_affine8(*w);
            throw std::invalid_argument("repack screen requires TQ3 or TQ4");
        }();
        auto tiled = [&] {
            if (auto w = std::get_if<Tq3Weights>(&original)) return tq3_tile(*w);
            return tq4_tile(std::get<Tq4Weights>(original));
        }();
        // Reconstruct in the original rotated basis, before any low-precision rounding.
        auto dense = mx::dequantize(exact.codes, exact.scales, exact.biases, 128, 8, "affine");
        mx::eval(dense);
        const int rows = dense.shape(0), k = dense.shape(1);
        for (auto dtype : {mx::float16, mx::bfloat16}) {
            auto staged = mx::astype(dense, dtype);
            mx::eval(staged);
            for (int bits : {4, 6, 8}) {
                const int group = 64;
                auto packed = mx::quantize(staged, group, bits, "affine");
                mx::eval(packed);
                auto rebuilt = mx::astype(mx::dequantize(packed[0], packed[1], packed[2],
                    group, bits, "affine"), mx::float32);
                const float weight_error = relative_l2(rebuilt, dense);
                if (!std::isfinite(weight_error)) throw std::runtime_error("nonfinite repacked weights");
                size_t bytes = 0;
                for (const auto& a : packed) bytes += a.nbytes();
                for (int m : {1, 8, 32}) {
                    std::vector<float> input(size_t(m) * k);
                    for (size_t i = 0; i < input.size(); ++i)
                        input[i] = .1f * std::sin(float(i) * .037f) + .03f * std::cos(float(i) * .117f);
                    auto x = mx::array(input.data(), {m, k});
                    auto low = mx::astype(x, dtype);
                    mx::eval(x, low);
                    auto control = [&] { return tq_tiled_matmul(tiled, x, 4, true, 32, true); };
                    auto candidate = [&] { return mx::quantized_matmul(low, packed[0], packed[1],
                        packed[2], true, group, bits, "affine"); };
                    auto reference = control();
                    auto truth = mx::matmul(x, mx::transpose(dense));
                    const float control_error = relative_l2(reference, truth);
                    if (!std::isfinite(control_error) || control_error > 2e-5f)
                        throw std::runtime_error("packed control differs from original dense projection");
                    const float output_error = relative_l2(candidate(), reference);
                    if (!std::isfinite(output_error)) throw std::runtime_error("nonfinite repack projection");
                    for (int arm : {0, 1, 1, 0}) {
                        const double elapsed = arm ? median_ms(candidate) : median_ms(control);
                        std::cout << "REPACK_SCREEN tensor=" << argv[2] << " rows=" << rows
                            << " k=" << k << " width=" << m
                            << " dtype=" << (dtype == mx::float16 ? "fp16" : "bf16")
                            << " bits=" << bits << " group=" << group << " arm=" << arm
                            << " elapsed_ms=" << elapsed << " bytes=" << bytes
                            << " weight_rel_l2=" << weight_error << " output_rel_l2=" << output_error
                            << " control_rel_l2=" << control_error << std::endl;
                    }
                }
            }
        }
        std::cout << "REPACK_SCREEN_COMPLETE" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
