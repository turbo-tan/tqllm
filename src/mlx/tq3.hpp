#pragma once
#include <mlx/mlx.h>
#include <span>
#include <cstdint>
#include "affine8.hpp"

namespace tqllm::mlx_backend {
struct Tq3Weights {
    mlx::core::array planes; // Planar [rows,3,K/32] or interleaved [rows,K/32,4] uint32 records.
    mlx::core::array scales;
    bool interleaved=false;
};
// GGUF301: FP16 integer-level scale + four 12-byte plane records per128 weights.
Tq3Weights load_tq3_blocks(std::span<const uint8_t> blocks, int rows, int columns);
struct Tq3Expanded {
    mlx::core::array values;
    mlx::core::array scales;
};
using Tq3Affine8 = Affine8Weights;
Tq3Affine8 tq3_affine8(const Tq3Weights& weights);
Tq3Affine8 tq3_affine8_join(const Tq3Affine8& gate,const Tq3Affine8& up);
mlx::core::array tq3_affine8_matmul(const Tq3Affine8& weights,const mlx::core::array& rotated_input);
Tq3Expanded tq3_expand(const Tq3Weights& weights);
mlx::core::array tq3_expanded_matmul(const Tq3Expanded& weights,const mlx::core::array& rotated_input);
Tq3Weights tq3_interleave(const Tq3Weights& weights);
mlx::core::array tq3_matmul(const Tq3Weights& weights, const mlx::core::array& rotated_input);
}
