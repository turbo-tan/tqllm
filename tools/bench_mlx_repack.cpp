#include "weights.hpp"
#include "tq3_tiled.hpp"
#include "prefill.hpp"
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
        const bool dense_screen = argc == 4 && std::string(argv[3]) == "--dense";
        if (argc != 3 && !dense_screen)
            throw std::invalid_argument("usage: bench-mlx-repack model.gguf tensor [--dense]");
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
        uint64_t ffn_elements = 0, ffn_original_bytes = 0;
        for (const auto& [name, tensor] : file.tensors) {
            if (name.find(".ffn_") == std::string::npos || !name.ends_with(".weight") || tensor.ne.size() != 2)
                continue;
            ffn_elements += tensor.ne[0] * tensor.ne[1];
            // Only the formats supported by this screen contribute to the replacement estimate.
            if (tensor.type == tqllm::GGML_TQ3) ffn_original_bytes += tensor.ne[0] * tensor.ne[1] / 128 * 50;
            else if (tensor.type == tqllm::GGML_TQ4) ffn_original_bytes += tensor.ne[0] * tensor.ne[1] / 128 * 66;
        }
        std::cout << "REPACK_MEMORY ffn_elements=" << ffn_elements
                  << " dense_fp16_ffn_bytes=" << ffn_elements * 2
                  << " replaced_tq_ffn_bytes=" << ffn_original_bytes << std::endl;
        for (auto dtype : {mx::float16, mx::bfloat16}) {
            auto staged = mx::astype(dense, dtype);
            mx::eval(staged);
            for (int bits : (dense_screen ? std::vector<int>{16} : std::vector<int>{4, 6, 8})) {
                const int group = bits == 16 ? 0 : 64;
                auto packed = bits == 16 ? std::vector<mx::array>{staged}
                    : mx::quantize(staged, group, bits, "affine");
                mx::eval(packed);
                auto rebuilt = mx::astype(bits == 16 ? staged
                    : mx::dequantize(packed[0], packed[1], packed[2], group, bits, "affine"), mx::float32);
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
                    auto control = [&] { return m <= 8 ? tq_tiled_matmul(tiled, x, 4, true, 32, true)
                        : mx::astype(mx::matmul(mx::astype(x, mx::bfloat16),
                            mx::transpose(stage_bf16(original))), mx::float32); };
                    auto reference = control();
                    auto truth = mx::matmul(x, mx::transpose(dense));
                    const float control_error = relative_l2(reference, truth);
                    if (!std::isfinite(control_error) || control_error > (m <= 8 ? 2e-5f : .02f))
                        throw std::runtime_error("packed control differs from original dense projection");
                    for (int pad : (m == 8 ? std::vector<int>{0, 32} : std::vector<int>{0})) {
                    auto candidate = [&] {
                        auto input = pad ? mx::concatenate({low, mx::zeros({pad-m, k}, dtype)}, 0) : low;
                        auto out = bits == 16 ? mx::matmul(input, mx::transpose(staged))
                            : mx::quantized_matmul(input, packed[0], packed[1], packed[2],
                                true, group, bits, "affine");
                        return pad ? mx::slice(out, {0,0}, {m,rows}) : out;
                    };
                    const float output_error = relative_l2(candidate(), truth);
                    if (!std::isfinite(output_error)) throw std::runtime_error("nonfinite repack projection");
                    for (int arm : {0, 1, 1, 0}) {
                        const double elapsed = arm ? median_ms(candidate) : median_ms(control);
                        std::cout << "REPACK_SCREEN tensor=" << argv[2] << " rows=" << rows
                            << " k=" << k << " width=" << m
                            << " dtype=" << (dtype == mx::float16 ? "fp16" : "bf16")
                            << " bits=" << bits << " group=" << group << " arm=" << arm
                            << " padded_rows=" << pad << " control=" << (m <= 8 ? "packed_fp32" : "staged_bf16")
                            << " elapsed_ms=" << elapsed << " bytes=" << bytes
                            << " weight_rel_l2=" << weight_error << " output_rel_l2=" << output_error
                            << " control_rel_l2=" << control_error << std::endl;
                    }
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
