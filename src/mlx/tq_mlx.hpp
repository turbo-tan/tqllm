#pragma once
#include "mma8.hpp"
#include "tq3_tiled.hpp"
#include "../gguf.hpp"
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace tqllm::mlx_backend {
// tq_mlx: a GGUF whose TQ3/TQ4 matrices are stored in the layout the Metal kernels read, so loading
// is a copy with no repack. Tiled (FFN and the big mixers) and MMA8 (the rest) are the two layouts.
using TqMlxMatrix = std::variant<TqTiledWeights, Mma8Weights>;
inline constexpr const char* kTqMlxVersionKey = "tq_mlx.version";
bool is_tq_mlx_type(uint32_t type);
bool is_tq_mlx_file(const GgufFile& file);
TqMlxMatrix tq_mlx_load(const GgufTensor& tensor, std::span<const uint8_t> payload);
// BF16 [rows, K] weights (scale * integer level), the operand for GEMM prefill.
mlx::core::array tq_mlx_dense(const TqMlxMatrix& weights);
// Rotated FP32 input (any row count) times the stored matrix, FP32 result.
mlx::core::array tq_mlx_prefill(const TqMlxMatrix& weights, const mlx::core::array& rotated_input);
// Payload of one TQ3/TQ4 tensor in its tq_mlx layout; `type` receives the new GGUF type.
std::vector<uint8_t> tq_mlx_pack_tensor(const std::string& name, const GgufTensor& tensor,
                                        std::span<const uint8_t> payload, uint32_t& type);
struct TqMlxStats {
    int tiled = 0, mma8 = 0, copied = 0;
    uint64_t bytes_in = 0, bytes_out = 0;
};
TqMlxStats tq_mlx_convert(const std::string& source, const std::string& destination);
}
