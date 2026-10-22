#include "prefill.hpp"
#include "verify.hpp"
#include "tensorops.hpp"
#include "int8_tiles.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <optional>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
template<class F> double measure(F operation) {
    mx::eval(operation());
    std::vector<double> times;
    for (int i = 0; i < 5; ++i) {
        auto start = std::chrono::steady_clock::now();
        mx::eval(operation());
        times.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[2];
}
int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4) throw std::invalid_argument("usage: bench-mlx-linear model.gguf tensor");
        auto file = tqllm::gguf_open(argv[1]);
        const auto& tensor = file.tensors.at(argv[2]);
        if (tensor.ne.size() != 2) throw std::invalid_argument("matrix required");
        const int k = tensor.ne[0], rows = tensor.ne[1];
        auto weights = load_matrix(file, argv[2]);
        std::optional<Tq3Affine8> affine;
        std::optional<Q6AffineSplit> q6_split;
        if (std::getenv("TQLLM_MLX_Q6_AFFINE") && std::holds_alternative<Q6Weights>(weights))
            q6_split = q6_affine_split(std::get<Q6Weights>(weights));
        std::optional<Tq3Expanded> expanded;
        if(auto w=std::get_if<Tq3Weights>(&weights)) {expanded=tq3_expand(*w);affine=tq3_affine8(*w);}
        if (std::getenv("TQLLM_MLX_AFFINE_TQ4"))
            if (auto w = std::get_if<Tq4Weights>(&weights)) affine = tq4_affine8(*w);
        auto dense = stage_bf16(weights);
        mx::eval(dense);
        const double unpack = measure([&] { return stage_bf16(weights); });
        for (int m : (argc==4 ? std::vector<int>{1,3,5,8,16} : std::vector<int>{1,4,8,16,64,512})) {
            std::vector<float> values(size_t(m) * k);
            for (size_t i = 0; i < values.size(); ++i) values[i] = .1f * std::sin(float(i) * .037f);
            mx::array input(values.data(), {m, k}, mx::float32);
            auto rotated = tensor.type == tqllm::GGML_TQ3 || tensor.type == tqllm::GGML_TQ4
                ? tq4_rotate(input) : input;
            mx::eval(rotated);
            auto half_input = mx::astype(rotated, mx::bfloat16);
            mx::eval(half_input);
            const double rotation = measure([&] { return tq4_rotate(input); });
            const double compressed = measure([&] { return linear(weights, input); });
            const double verify_ms = measure([&] { return verify_tiled(weights, input); });
            double q6_affine_ms = -1, q6_affine_error = -1;
            if (q6_split) {
                auto expected = linear(weights, input), got = q6_affine_matmul(*q6_split, input);
                q6_affine_error = mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(got, expected))),
                    mx::maximum(mx::sum(mx::square(expected)), mx::array(1e-20f)))).item<float>();
                if (!std::isfinite(q6_affine_error) || q6_affine_error > 2e-5)
                    throw std::runtime_error("Q6 affine projection drift exceeds FP32 gate");
                q6_affine_ms = measure([&] { return q6_affine_matmul(*q6_split, input); });
            }
            double int8_ms = -1, int8_error = -1;
            if (std::getenv("TQLLM_MLX_INT8_TILE") && std::holds_alternative<Tq3Weights>(weights)) {
                const auto& w = std::get<Tq3Weights>(weights);
                auto expected = linear(weights, input), got = verify_tq3_int8(w, input);
                int8_error = mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(got, expected))),
                    mx::maximum(mx::sum(mx::square(expected)), mx::array(1e-20f)))).item<float>();
                if (!std::isfinite(int8_error) || int8_error > 2e-5)
                    throw std::runtime_error("residual INT8 projection drift exceeds FP32 gate");
                int8_ms = measure([&] { return verify_tq3_int8(w, input); });
            }
            const double affine_ms = affine ? measure([&] { return tq3_affine8_matmul(*affine,tq4_rotate(input)); }) : 0;
            double affine_error=-1;
            if(affine && (std::getenv("TQLLM_MLX_AFFINE_TQ4") ||
               (std::getenv("MLX_ENABLE_TF32") && std::string(std::getenv("MLX_ENABLE_TF32"))=="1"))) {
                auto expected=linear(weights,input);
                auto got=tq3_affine8_matmul(*affine,tq4_rotate(input));
                affine_error=mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(got,expected))),
                    mx::maximum(mx::sum(mx::square(expected)),mx::array(1e-20f)))).item<float>();
                const bool rounded = std::getenv("TQLLM_MLX_AFFINE_HALF") ||
                    (std::getenv("MLX_ENABLE_TF32") && std::string(std::getenv("MLX_ENABLE_TF32"))=="1");
                if(!std::isfinite(affine_error) || affine_error>(rounded?1e-3:2e-5))
                    throw std::runtime_error("affine projection drift exceeds precision gate");
            }
            const double decoded_tensor_ms = expanded && !std::getenv("TQLLM_MLX_TENSOR_MEMORY") ? measure([&] { return verify_expanded_tensorops(*expanded,input); }) : 0;
            const double decoded_ms = expanded ? measure([&] { return tq3_expanded_matmul(*expanded,tq4_rotate(input)); }) : 0;
            const double tensorops = measure([&] { return verify_tensorops(weights, input, true); });
            double tensor_error = 0;
            if (std::getenv("TQLLM_MLX_TENSOR_MEMORY")) {
                auto expected = linear(weights, input);
                auto got = verify_tensorops(weights, input, true);
                auto delta = mx::subtract(got, expected);
                auto error = mx::sqrt(mx::divide(mx::sum(mx::square(delta)),
                    mx::maximum(mx::sum(mx::square(expected)), mx::array(1e-20f))));
                tensor_error = error.item<float>();
                if (!std::isfinite(tensor_error) || tensor_error > 1e-3)
                    throw std::runtime_error("packed tensor tile differs from compressed projection");
            }
            const double staged = measure([&] { return prefill_linear(weights, input); });
            const double gemm = measure([&] { return mx::matmul(half_input, mx::transpose(dense)); });
            std::cout << "LINEAR_BENCH tensor=" << argv[2] << " type=" << tensor.type
                      << " rows=" << rows << " k=" << k << " m=" << m
                      << " rotate_ms=" << rotation << " unpack_ms=" << unpack
                      << " compressed_ms=" << compressed << " staged_ms=" << staged
                      << " q6_affine_ms=" << q6_affine_ms << " q6_affine_rel_l2=" << q6_affine_error
                      << " verify_tiled_ms=" << verify_ms
                      << " int8_tile_ms=" << int8_ms << " int8_tile_rel_l2=" << int8_error
                      << " affine8_ms=" << affine_ms << " affine8_rel_l2=" << affine_error << " decoded_tensor_ms=" << decoded_tensor_ms << " decoded_ms=" << decoded_ms << " tensorops_ms=" << tensorops << " tensorops_rel_l2=" << tensor_error << " resident_gemm_ms=" << gemm << std::endl;
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
