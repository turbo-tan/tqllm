#pragma once
#include <mlx/mlx.h>
#include <cstdint>
#include <span>
#include <vector>

namespace tqllm::mlx_backend {
// Affine weights (scale * code + bias, 4 or 6 bits, BF16 scale and bias decoded in FP32) in the lane order of the M5 matrix unit's 16x32x16 MPP matmul2d:
// each lane reads the codes of its own right-operand elements, so any number of input rows decodes into registers.
struct NaxAffine {
    mlx::core::array words;   // [rows/32, columns/16, 32 lanes, 2] uint32, low nibbles
    mlx::core::array high;    // [rows/32, columns/16, 32 lanes] uint32, high bit pairs (6-bit only)
    mlx::core::array scales;  // [rows/32, columns/group, 32] BF16 bits (uint16)
    mlx::core::array biases;  // [rows/32, columns/group, 32] BF16 bits (uint16)
    int rows, columns, bits, group;
};
uint64_t nax_affine_bytes(uint64_t rows, uint64_t columns, int bits, int group);
NaxAffine nax_affine_load(std::span<const uint8_t> payload, int rows, int columns, int bits, int group);
// codes [rows, columns] in 0..2^bits-1, scales/biases [rows, columns/group]: the stored payload.
std::vector<uint8_t> nax_affine_pack(std::span<const uint8_t> codes, std::span<const uint16_t> scales_bf16,
                                     std::span<const uint16_t> biases_bf16, int rows, int columns, int bits, int group);
// [m, columns] input (any float dtype, m >= 1) times the stored matrix, FP32 [m, rows]. Every output row depends
// only on its own input row, so one row and a verify batch give bitwise equal results.
mlx::core::array nax_affine_matmul(const NaxAffine& weights, const mlx::core::array& input);
int nax_affine_splits(const NaxAffine& weights);

// 8-bit affine, group 64, row-major codes: a token-embedding table read by gather.
struct Affine8Embedding {
    mlx::core::array codes;   // [rows, columns] uint8
    mlx::core::array scales;  // [rows, columns/64] bfloat16
    mlx::core::array biases;  // [rows, columns/64] bfloat16
    int rows, columns;
};
uint64_t affine8_embedding_bytes(uint64_t rows, uint64_t columns);
Affine8Embedding affine8_embedding_load(std::span<const uint8_t> payload, int rows, int columns);
mlx::core::array affine8_embedding(const Affine8Embedding& table, std::span<const uint32_t> tokens);
}
