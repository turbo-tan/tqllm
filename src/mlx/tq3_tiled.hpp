#pragma once
#include "tq3.hpp"
#include "tq4.hpp"
#include <optional>

namespace tqllm::mlx_backend {
struct TqTiledWeights {
    mlx::core::array planes; // [row_tiles, K/32, plane_count, 32]
    mlx::core::array scales; // [row_tiles, K/128, 32]
    int rows;
    int columns;
    int bits = 3;
    bool integer_levels = true;
};
using Tq3TiledWeights = TqTiledWeights;
Tq3TiledWeights tq3_tile(const Tq3Weights& weights);
TqTiledWeights tq4_tile(const Tq4Weights& weights);
// Input is already rotated. Operand precision is explicit; accumulation is FP32.
mlx::core::array tq_tiled_matmul(const TqTiledWeights& weights,
    const mlx::core::array& input, int splits = 1, bool prefetch = true, int tile_k = 32,
    bool full_precision = false,
    const std::optional<mlx::core::array>& expanded = std::nullopt, int simdgroups = 1, bool transpose_stage = false, int padded_rows = 8);
mlx::core::array tq_register_matmul(const TqTiledWeights& weights,
    const mlx::core::array& input, int splits = 4, bool integer_operand = false, bool packed_cache = false, bool compensated_input = false);
// Lossless int8 code levels, packed four per word in output-coalesced order.
mlx::core::array tq_expand_tiled_codes(const TqTiledWeights& weights);
// Integer codes stay exact in FP16; high/residual input rows share one tile.
mlx::core::array tq_compensated_matmul(const TqTiledWeights& weights,
    const mlx::core::array& input, int splits = 4, int tile_k = 32, bool residual = true);
inline mlx::core::array tq3_tiled_matmul(const Tq3TiledWeights& weights,
    const mlx::core::array& input, int splits = 1, bool prefetch = true, int tile_k = 32,
    bool full_precision = false) {
    return tq_tiled_matmul(weights, input, splits, prefetch, tile_k, full_precision);
}
}
