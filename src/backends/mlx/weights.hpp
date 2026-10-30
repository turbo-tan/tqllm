#pragma once
#include "tq3.hpp"
#include "tq4.hpp"
#include "q6.hpp"
#include "q8.hpp"
#include "kquant.hpp"
#include "q4_embedding.hpp"
#include "gguf.hpp"
#include <variant>

namespace tqllm::mlx_backend {
using Matrix = std::variant<mlx::core::array, Tq3Weights, Tq4Weights, Q6Weights, Q8Weights, KQuantWeights>;
// GGUF dimensions are reversed into MLX row-major dimensions.
mlx::core::array load_plain_tensor(const GgufTensor& tensor, std::span<const uint8_t> payload);
Matrix load_matrix(const GgufTensor& tensor, std::span<const uint8_t> payload);
Matrix load_matrix(const GgufFile& file, const std::string& name);
mlx::core::array linear(const Matrix& weights, const mlx::core::array& input);
// In a TQ4 model the dense matrices (TQ_L: BF16 ssm_alpha / ssm_beta) are stored in the TQ4 rotated basis, like the TQ4
// matrices they share an input with: tq4_basis rotates the input of a dense matrix (j945 / C15as); other matrices unchanged.
mlx::core::array basis_input(const Matrix& weights, const mlx::core::array& input, bool tq4_basis);
mlx::core::array linear(const Matrix& weights, const mlx::core::array& input, bool tq4_basis);
}
