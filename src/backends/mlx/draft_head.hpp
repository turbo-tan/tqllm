#pragma once
#include "weights.hpp"
#include <optional>
namespace tqllm::mlx_backend {
// Proposal-only affine head. Target verification always retains the original head.
struct DraftHead4 {
    mlx::core::array codes,scales,biases;
    // Fragment-order repack for the 8-row simdgroup-matrix kernel (TQLLM_MLX_HEAD_FRAG).
    std::optional<mlx::core::array> frag_codes,frag_scales,frag_biases;
};
DraftHead4 draft_head4(const Matrix& original);
mlx::core::array draft_head4_matmul(const DraftHead4& head,const mlx::core::array& input);
}
