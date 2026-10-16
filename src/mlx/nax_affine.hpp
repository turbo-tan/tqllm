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
    bool tq4 = false;  // TQLLM TQ4 codebook: value = FP16 scale * signed level (scales hold FP16 bits, no biases)
    bool q6 = false;   // Q6_K head: 6-bit codes, group 16, FP32 scales (two uint16 words each), value = scale * (code - 32)
};
uint64_t nax_affine_bytes(uint64_t rows, uint64_t columns, int bits, int group);
NaxAffine nax_affine_load(std::span<const uint8_t> payload, int rows, int columns, int bits, int group);
// codes [rows, columns] in 0..2^bits-1, scales/biases [rows, columns/group]: the stored payload.
std::vector<uint8_t> nax_affine_pack(std::span<const uint8_t> codes, std::span<const uint16_t> scales_bf16,
                                     std::span<const uint16_t> biases_bf16, int rows, int columns, int bits, int group);
// [m, columns] input (any float dtype, rounded to FP16; m >= 1) times the stored matrix, FP32 [m, rows], optionally
// rounded to BF16 values in the kernel epilogue. Every output row depends only on its own input
// row, so one row and a verify batch give bitwise equal results.
mlx::core::array nax_affine_matmul(const NaxAffine& weights, const mlx::core::array& input, bool round_bf16 = false);
int nax_affine_splits(const NaxAffine& weights);
// The stored matrix as FP16 [rows, columns], the values the matmul kernel multiplies.
mlx::core::array nax_affine_dequant(const NaxAffine& weights);
// Prefill: dequantize once, then one FP16 GEMM (FP32 accumulate, FP16 output) for many input rows. Not row-invariant
// with nax_affine_matmul (accumulation order), so decode and verify keep the row kernel.
mlx::core::array nax_affine_gemm(const NaxAffine& weights, const mlx::core::array& input, bool round_bf16 = false);
// Matrices with the same columns, bits and group stacked along the output rows (one launch for projections that
// share an input).
// TQ4 (GGUF type 300, integer levels, rotated basis) repacked into the matrix-unit lane order; the input must be
// tq4_rotate()d like for every TQ4 kernel.
struct Tq4Weights;
NaxAffine nax_from_tq4(const Tq4Weights& weights);
// Q6_K (GGUF head) repacked for the matrix unit: same codes, FP32 d * sub-block scale per 16 columns.
struct Q6Weights;
NaxAffine nax_from_q6(const Q6Weights& weights);
NaxAffine nax_affine_concat(const std::vector<const NaxAffine*>& parts);
// Rows `keep` (in that order, a multiple of 128) of a stored payload, repacked as their own matrix.
NaxAffine nax_affine_select_rows(std::span<const uint8_t> payload, int rows, int columns, int bits, int group,
                                 std::span<const uint32_t> keep);

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
