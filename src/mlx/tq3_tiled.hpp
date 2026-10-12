#pragma once
#include "tq3.hpp"

namespace tqllm::mlx_backend {
struct Tq3TiledWeights {
    mlx::core::array planes; // [row_tiles, K/32, 3, 32]
    mlx::core::array scales; // [row_tiles, K/128, 32]
    int rows;
    int columns;
};
Tq3TiledWeights tq3_tile(const Tq3Weights& weights);
// Input is already rotated. FP16 operands and FP32 accumulation are explicit.
mlx::core::array tq3_tiled_matmul(const Tq3TiledWeights& weights,
    const mlx::core::array& input, int splits = 1, bool prefetch = true);
}
