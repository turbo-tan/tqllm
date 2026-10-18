#pragma once
#include <mlx/mlx.h>

namespace tqllm::mlx_backend {
struct Affine8Weights {
    mlx::core::array codes, scales, biases, half_scales, half_biases;
};
mlx::core::array affine8_matmul(const Affine8Weights& weights,
                              const mlx::core::array& rotated_input, bool half = false);
}
