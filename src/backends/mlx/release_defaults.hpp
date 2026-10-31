#pragma once
#include <cstdlib>

namespace tqllm::mlx_backend {
// The MLX setting the shipped configuration needs before the first MLX call, applied by every entry point (tqserve's
// Model and each Qwen35): exact FP32 matmuls (no TF32); an explicit MLX_ENABLE_TF32 still wins. The engine's own served
// choices (16-row drafts, fused GDN step, draft-ahead, 4-bit proposal head; j892) are code defaults, not variables.
inline void apply_release_defaults() {
    setenv("MLX_ENABLE_TF32", "0", 0);
}
}  // namespace tqllm::mlx_backend
