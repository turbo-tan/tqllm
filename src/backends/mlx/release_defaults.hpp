#pragma once
#include <cstdlib>

namespace tqllm::mlx_backend {
// The settings the shipped configuration needs before the first MLX call, applied by every entry point (tqserve's Model
// and each Qwen35): exact FP32 matmuls (no TF32), 16-row drafts and the gated speed defaults (decode C15f / j892).
// Explicit environment settings still win (=0 turns one off).
inline void apply_release_defaults() {
    setenv("MLX_ENABLE_TF32", "0", 0);
    setenv("TQLLM_MLX_DFLASH_BLOCK", "16", 0);
    setenv("TQLLM_MLX_GDN_STEP", "1", 0);
    setenv("TQLLM_MLX_DRAFT_AHEAD", "1", 0);
    setenv("TQLLM_MLX_DRAFT_HEAD4", "2", 0);
}
}  // namespace tqllm::mlx_backend
