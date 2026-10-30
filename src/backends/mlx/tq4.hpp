#pragma once
#include <mlx/mlx.h>
#include <span>
#include <cstdint>

namespace tqllm::mlx_backend {
struct Tq4Weights {
    mlx::core::array codes;
    mlx::core::array scales;
    bool integer_levels = true;
};
// GGUF type 300: one FP16 scale followed by 64 packed bytes per block.
Tq4Weights load_tq4_blocks(std::span<const uint8_t> blocks, int rows, int columns);
mlx::core::array tq4_rotate(const mlx::core::array& input);
// tq4_rotate fused into one kernel, FP16 result (input of the matrix-unit TQ4 path).
mlx::core::array tq4_rotate_half(const mlx::core::array& input);
// Packed codebook weights in the rotated basis; input is already rotated.
mlx::core::array tq4_matmul(const mlx::core::array& codes,
                          const mlx::core::array& scales,
                          const mlx::core::array& rotated_input,
                          int rows_per_simd = 1, int columns_per_tile = 4,
                          int simd_groups_per_threadgroup = 2, bool integer_levels = false);
mlx::core::array tq4_matmul(const Tq4Weights& weights,
                          const mlx::core::array& rotated_input,
                          int rows_per_simd = 1, int columns_per_tile = 4,
                          int simd_groups_per_threadgroup = 2);
}
