#pragma once
#include <mlx/mlx.h>
#include <cstdint>
#include <span>

namespace tqllm::mlx_backend {
// GGUF K-quant and IQ blocks kept as stored (Q2_K, Q3_K, Q4_K, Q5_K, IQ3_S, IQ4_XS, IQ4_NL), decoded on the GPU with
// the element values of dequantize_host (formats/weight/dequant.cpp), bit for bit
struct KQuantWeights {
    mlx::core::array blocks;  // uint8 [rows, columns / block elements, block bytes]
    uint32_t type = 0;
};
bool kquant_supported(uint32_t type);
KQuantWeights load_kquant_blocks(std::span<const uint8_t> blocks, uint32_t type, int rows, int columns);
// FP32 [rows, columns]
mlx::core::array kquant_dequant(const KQuantWeights& weights);
// FP32 input [m, columns] -> [m, rows]; up to 16 rows decode the blocks in the product, more rows multiply the
// dequantized FP32 matrix
mlx::core::array kquant_matmul(const KQuantWeights& weights, const mlx::core::array& input);
}
