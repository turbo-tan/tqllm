#pragma once
#include <mlx/mlx.h>
#include <span>
#include <cstdint>
namespace tqllm::mlx_backend {
struct Q4Embedding { mlx::core::array blocks; };
Q4Embedding load_q4_embedding(std::span<const uint8_t> blocks, int rows, int columns);
mlx::core::array q4_embedding(const Q4Embedding& weights, std::span<const uint32_t> ids);
}
