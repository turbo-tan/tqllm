#include "prefill.hpp"
#include "verify.hpp"
#include "tensorops.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
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
        std::optional<Tq3Expanded> expanded;
        if(auto w=std::get_if<Tq3Weights>(&weights)) {expanded=tq3_expand(*w);affine=tq3_affine8(*w);}
        auto dense = stage_bf16(weights);
        mx::eval(dense);
        const double unpack = measure([&] { return stage_bf16(weights); });
        for (int m : (argc==4 ? std::vector<int>{1,8,16} : std::vector<int>{1,4,8,16,64,512})) {
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
            const double affine_ms = affine ? measure([&] { return tq3_affine8_matmul(*affine,tq4_rotate(input)); }) : 0;
            const double decoded_tensor_ms = expanded ? measure([&] { return verify_expanded_tensorops(*expanded,input); }) : 0;
            const double decoded_ms = expanded ? measure([&] { return tq3_expanded_matmul(*expanded,tq4_rotate(input)); }) : 0;
            const double tensorops = measure([&] { return verify_tensorops(weights, input, true); });
            const double staged = measure([&] { return prefill_linear(weights, input); });
            const double gemm = measure([&] { return mx::matmul(half_input, mx::transpose(dense)); });
            std::cout << "LINEAR_BENCH tensor=" << argv[2] << " type=" << tensor.type
                      << " rows=" << rows << " k=" << k << " m=" << m
                      << " rotate_ms=" << rotation << " unpack_ms=" << unpack
                      << " compressed_ms=" << compressed << " staged_ms=" << staged
                      << " affine8_ms=" << affine_ms << " decoded_tensor_ms=" << decoded_tensor_ms << " decoded_ms=" << decoded_ms << " tensorops_ms=" << tensorops << " resident_gemm_ms=" << gemm << std::endl;
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
