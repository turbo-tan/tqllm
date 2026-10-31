#pragma once
#include <mlx/mlx.h>
#include <span>
#include <cstdint>

namespace tqllm::mlx_backend {
struct Tq3Weights {
    mlx::core::array planes; // Planar [rows,3,K/32] or interleaved [rows,K/32,4] uint32 records.
    mlx::core::array scales;
    bool interleaved=false;
};
// GGUF301: FP16 integer-level scale + four 12-byte plane records per128 weights.
Tq3Weights load_tq3_blocks(std::span<const uint8_t> blocks, int rows, int columns);
mlx::core::array tq3_matmul(const Tq3Weights& weights, const mlx::core::array& rotated_input);
}
