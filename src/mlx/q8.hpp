#pragma once
#include <mlx/mlx.h>
#include <span>
#include <cstdint>
namespace tqllm::mlx_backend {
struct Q8Weights { mlx::core::array blocks; };
Q8Weights load_q8_blocks(std::span<const uint8_t> blocks, int rows, int columns);
mlx::core::array q8_matmul(const Q8Weights& weights, const mlx::core::array& input);
}
