#pragma once
#include <mlx/mlx.h>
#include <span>
#include <cstdint>

namespace tqllm::mlx_backend {
struct Q6Weights {
    mlx::core::array blocks;
};
Q6Weights load_q6_blocks(std::span<const uint8_t> blocks, int rows, int columns);
mlx::core::array q6_matmul(const Q6Weights& weights, const mlx::core::array& input);
}
