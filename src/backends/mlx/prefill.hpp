#pragma once
#include "weights.hpp"
namespace tqllm::mlx_backend {
// Explicit BF16 compute staging for a single projection; storage is unchanged.
mlx::core::array stage_bf16(const Matrix& weights);
mlx::core::array prefill_linear(const Matrix& weights,const mlx::core::array& input);
}
