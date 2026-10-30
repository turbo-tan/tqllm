#pragma once
#include "weights.hpp"
namespace tqllm::mlx_backend {
// Explicit BF16 compute staging for a single projection; storage is unchanged.
mlx::core::array stage_bf16(const Matrix& weights);
}
