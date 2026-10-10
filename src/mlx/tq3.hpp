#pragma once
#include <mlx/mlx.h>
#include <span>
#include <cstdint>

namespace tqllm::mlx_backend {
struct Tq3Weights {
    mlx::core::array planes; // [rows, 3, columns/32], nibble-sliced rank and sign planes
    mlx::core::array scales;
};
// GGUF301: FP16 integer-level scale + four 12-byte plane records per128 weights.
Tq3Weights load_tq3_blocks(std::span<const uint8_t> blocks, int rows, int columns);
mlx::core::array tq3_matmul(const Tq3Weights& weights, const mlx::core::array& rotated_input);
}
