#pragma once
#include <mlx/mlx.h>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
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
    int split = 0;     // split-K override (0: by shape); a pack of stacked matrices keeps its members' largest split
};
uint64_t nax_affine_bytes(uint64_t rows, uint64_t columns, int bits, int group);
// MLX affine 4-bit quantize output (codes [rows, columns / 8] uint32, BF16 scales / biases [rows, columns / group])
// repacked on the GPU into the NAX lane order: the same words as nax_affine_pack of the unpacked codes
NaxAffine nax_affine_from_quantized(const mlx::core::array& codes, const mlx::core::array& scales,
                                    const mlx::core::array& biases, int group);
NaxAffine nax_affine_load(std::span<const uint8_t> payload, int rows, int columns, int bits, int group);
// codes [rows, columns] in 0..2^bits-1, scales/biases [rows, columns/group]: the stored payload.
std::vector<uint8_t> nax_affine_pack(std::span<const uint8_t> codes, std::span<const uint16_t> scales_bf16,
                                     std::span<const uint16_t> biases_bf16, int rows, int columns, int bits, int group);
// [m, columns] input (any float dtype, rounded to FP16; m >= 1) times the stored matrix, FP32 [m, rows], optionally
// rounded to BF16 values in the kernel epilogue. Every output row depends only on its own input
// row, so one row and a verify batch give bitwise equal results.
mlx::core::array nax_affine_matmul(const NaxAffine& weights, const mlx::core::array& input, bool round_bf16 = false);
int nax_affine_splits(const NaxAffine& weights);
// Verify / decode FFN: h = bf16(bf16(silu(bf16(x Wg^T))) x bf16(x Wu^T)), FP32 [m, rows] for m <= 16 rows, gate and up
// in one launch (4-bit affine or TQ4, matching shapes). Row-invariant like nax_affine_matmul.
mlx::core::array nax_affine_swiglu(const NaxAffine& gate, const NaxAffine& up, const mlx::core::array& input);
// The stored matrix as FP16 [rows, columns], the values the matmul kernel multiplies.
mlx::core::array nax_affine_dequant(const NaxAffine& weights);
// Prefill: dequantize once, then one FP16 GEMM (FP32 accumulate, FP16 output) for many input rows. Not row-invariant
// with nax_affine_matmul (accumulation order), so decode and verify keep the row kernel.
mlx::core::array nax_affine_gemm(const NaxAffine& weights, const mlx::core::array& input, bool round_bf16 = false);
// A 4-bit affine matrix in MLX's quantized layout: codes [rows, columns / 8] uint32 (column c at bits 4 (c % 8)), FP16
// scales / biases [rows, columns / group].
struct MlxQuantized {mlx::core::array codes, scales, biases; int group;};
// Repack of the stored codes (no requantization); BF16 scales and biases rounded to FP16 (exact in FP16's normal range).
MlxQuantized nax_affine_to_mlx(const NaxAffine& weights);
// Prefill: FP16 [m, columns] input times the repacked matrix on MLX's quantized_matmul, FP16 [m, rows]. It reads 4-bit
// weights instead of an FP16 copy (fewer DRAM bytes). Bench / test only: as the prefill path it lost on tqserve (j791 doc8k 387 vs 414 tok/s).
mlx::core::array nax_affine_qmm(const MlxQuantized& weights, const mlx::core::array& input);
// Prefill GEMM on the stored codes, no FP16 weight copy: FP16 [m, columns] input (TQ4: rotated) times the matrix, FP16
// [m, rows]. Epilogue ep: 0 plain, 1 silu(aux) * product (aux = the gate output, SwiGLU fused into the up projection),
// 2 aux + product (residual); aux FP16 [m, rows]. Tile: SGS simdgroups of RS rows over TN output columns, 64 inputs per
// staged step; tb 1 stages the weights K-contiguous per column (transposed right operand), pf 1 prefetches the next
// step's codes during the matmul. Not row-invariant with nax_affine_matmul. 4-bit affine and TQ4, group a multiple of 64.
mlx::core::array nax_affine_prefill(const NaxAffine& weights, const mlx::core::array& input, int ep = 0,
                                    const std::optional<mlx::core::array>& aux = std::nullopt, int rs = 32, int tn = 64, int sgs = 4,
                                    int tb = 1, int pf = 1, int rf = 1);
// the same GEMM on weights nax_affine_dequant already returned (one dequant shared by several inputs)
mlx::core::array nax_affine_gemm_dequantized(const mlx::core::array& weights, const mlx::core::array& input, bool round_bf16 = false);
// TQ4 (GGUF type 300, integer levels, rotated basis) repacked into the matrix-unit lane order; the input must be
// tq4_rotate()d like for every TQ4 kernel.
struct Tq4Weights;
NaxAffine nax_from_tq4(const Tq4Weights& weights);
// Q6_K (GGUF head) repacked for the matrix unit: same codes, FP32 d * sub-block scale per 16 columns.
struct Q6Weights;
NaxAffine nax_from_q6(const Q6Weights& weights);
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
mlx::core::array affine8_embedding(const Affine8Embedding& table, const mlx::core::array& ids);
}
