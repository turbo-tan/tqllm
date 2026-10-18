// The shipped MLX configuration comes from one place (apply_release_defaults), applied by tqserve's Model and by every
// Qwen35 engine: the 10-18 bench tools ran without TQLLM_MLX_FUSE_EW / DFLASH_BLOCK because only tqserve set them.
// Explicit environment settings still win.
#include "release_defaults.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    const char* on[] = {"TQLLM_MLX_GDN_SCAN", "TQLLM_MLX_SERIAL_VERIFY_ATTENTION", "TQLLM_MLX_SERIAL_VERIFY_DENSE",
                        "TQLLM_MLX_COMPACT_GDN_TAPE", "TQLLM_MLX_DEVICE_SELECTOR", "TQLLM_MLX_BATCH_DECODE_DENSE",
                        "TQLLM_MLX_ATTN_DECODE_ROWS", "TQLLM_MLX_DRAFT_HEAD_UNPAD", "TQLLM_MLX_FUSE_EW"};
    for (const char* n : on) unsetenv(n);
    unsetenv("MLX_ENABLE_TF32");
    setenv("TQLLM_MLX_DFLASH_BLOCK", "8", 1);
    tqllm::mlx_backend::apply_release_defaults();
    bool ok = true;
    for (const char* n : on) {const char* v = std::getenv(n); ok &= v && std::strcmp(v, "1") == 0;}
    ok &= std::getenv("MLX_ENABLE_TF32") && std::strcmp(std::getenv("MLX_ENABLE_TF32"), "0") == 0;
    ok &= std::strcmp(std::getenv("TQLLM_MLX_DFLASH_BLOCK"), "8") == 0;
    std::puts(ok ? "MLX_RELEASE_DEFAULTS_PASS" : "MLX_RELEASE_DEFAULTS_FAIL");
    return ok ? 0 : 1;
}
