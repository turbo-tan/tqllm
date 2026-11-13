#pragma once
#include "weights.hpp"
namespace tqllm::mlx_backend {
// Explicit BF16 compute staging for a single projection; storage is unchanged.
mlx::core::array stage_bf16(const Matrix& weights);
// TQ4 as an FP16 matrix (scale * level rounded once to FP16): a prefill chunk's GEMM operand where the matrix has no
// matrix-unit copy (M1-M4)
mlx::core::array stage_half(const Tq4Weights& weights);
}
