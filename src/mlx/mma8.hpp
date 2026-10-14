#pragma once
#include "tq3.hpp"
#include "tq4.hpp"

namespace tqllm::mlx_backend {
// TQ3/TQ4 codes regrouped as nibbles so each lane of an 8x8 simdgroup MMA fragment reads its own codes.
struct Mma8Weights {
    mlx::core::array words;  // [rows_padded/16, K/128, 2, 32 lanes, 4] uint32
    mlx::core::array scales; // [rows_padded/16, K/128, 2, 8] float16
    mlx::core::array lut;    // 256 uint32: FP16 level pair for each code byte
    int rows;
    int rows_padded;
    int columns;
};
Mma8Weights mma8_pack(const Tq4Weights& weights);
Mma8Weights mma8_pack(const Tq3Weights& weights);
void mma8_clear_pack_cache();
// Rotated FP32 activations (1..8 rows) as zero-padded [8, K] FP16, shared by every projection of one input.
mlx::core::array mma8_input(const mlx::core::array& rotated_input);
// Rows 0..m-1 of the product; round_bf16 applies the engine's BF16 rounding in the kernel epilogue.
mlx::core::array mma8_matmul(const Mma8Weights& weights, const mlx::core::array& input8, int m, bool round_bf16);
mlx::core::array mma8_matmul(const Mma8Weights& weights, const mlx::core::array& rotated_input);
}
