#pragma once
#include <mlx/mlx.h>
#include <span>
#include <cstdint>
#include <vector>

namespace tqllm::mlx_backend {
struct Q6Weights {
    mlx::core::array blocks;
};
Q6Weights load_q6_blocks(std::span<const uint8_t> blocks, int rows, int columns);
mlx::core::array q6_matmul(const Q6Weights& weights, const mlx::core::array& input);
mlx::core::array q6_rows_matmul(const Q6Weights& weights, const mlx::core::array& input,
                                const mlx::core::array& ids);
mlx::core::array sparse_logits(const mlx::core::array& ids, const mlx::core::array& scores, int vocab);
mlx::core::array shard_topk_ids(const mlx::core::array& logits, int candidates);
struct Q6AffineSplit {
    std::vector<mlx::core::array> codes, scales, biases;
    int bits = 8;
};
// Exact weight expansion into two group-32 matrices, each with 16 zero columns.
Q6AffineSplit q6_affine_split(const Q6Weights& weights, int bits = 8);
mlx::core::array q6_affine_matmul(const Q6AffineSplit& weights, const mlx::core::array& input);
}
