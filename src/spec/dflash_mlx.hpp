#pragma once
// DFlash2 drafter kernels on MLX shared by the Metal engines (Qwen: src/backends/metal/dflash.cpp; Inkling-2:
// src/models/inkling2/loop/inkling2_mlx_draft.cpp): the grouped two-tap dynamic convolution, the top-16 candidates and
// the selector's greedy walk.
#include <mlx/mlx.h>
#include <cstdint>
#include <optional>
#include <utility>

namespace tqllm::mlx_backend {
// side 0: input taps of the block, side 1: output taps; with a residual, returns bf(residual + conv)
mlx::core::array dflash_conv(const mlx::core::array& input,const mlx::core::array& dynamic,
                            const mlx::core::array& base,int group,int side,
                            const std::optional<mlx::core::array>& residual=std::nullopt);
mlx::core::array dflash_select(const mlx::core::array& mapped,const mlx::core::array& unary,
    const mlx::core::array& gate,const mlx::core::array& successors,
    const mlx::core::array& predecessors,uint32_t anchor,int count);
// the same selection from an anchor id already on the GPU (uint32 scalar)
mlx::core::array dflash_select(const mlx::core::array& mapped,const mlx::core::array& unary,
    const mlx::core::array& gate,const mlx::core::array& successors,
    const mlx::core::array& predecessors,const mlx::core::array& anchor,int count);
// the 16 largest logits per row, largest first (ties: lower column first): {column ids (uint32), values (float32)} [rows, 16]
std::pair<mlx::core::array,mlx::core::array> dflash_top16(const mlx::core::array& logits);
}  // namespace tqllm::mlx_backend
