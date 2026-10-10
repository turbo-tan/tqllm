#pragma once
#include <mlx/mlx.h>

namespace tqllm::mlx_backend {
mlx::core::array tq4_rotate(const mlx::core::array& input);
// Packed codebook weights in the rotated basis; input is already rotated.
mlx::core::array tq4_matmul(const mlx::core::array& codes,
                          const mlx::core::array& scales,
                          const mlx::core::array& rotated_input,
                          int rows_per_simd = 1, int columns_per_tile = 4);
}
