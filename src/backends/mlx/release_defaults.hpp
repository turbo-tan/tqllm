#pragma once
#include <cstdlib>
#include <initializer_list>

namespace tqllm::mlx_backend {
// The shipped configuration, the measured exact set (j690-j694, release candidate j707): scan GDN with the compact prefix
// tape, batch-invariant decode-rows attention, serial dense rows, device-side draft selection, compiled elementwise
// epilogues, 16-row drafts. Every entry point applies it before backend code reads a flag (several cache their flag on
// first use); explicit environment settings still win.
inline void apply_release_defaults() {
    setenv("MLX_ENABLE_TF32", "0", 0);
    for (const char* name : {"TQLLM_MLX_GDN_SCAN", "TQLLM_MLX_SERIAL_VERIFY_ATTENTION", "TQLLM_MLX_SERIAL_VERIFY_DENSE",
                             "TQLLM_MLX_COMPACT_GDN_TAPE", "TQLLM_MLX_DEVICE_SELECTOR", "TQLLM_MLX_BATCH_DECODE_DENSE",
                             "TQLLM_MLX_ATTN_DECODE_ROWS", "TQLLM_MLX_DRAFT_HEAD_UNPAD", "TQLLM_MLX_FUSE_EW"})
        setenv(name, "1", 0);
    setenv("TQLLM_MLX_DFLASH_BLOCK", "16", 0);
}
}  // namespace tqllm::mlx_backend
