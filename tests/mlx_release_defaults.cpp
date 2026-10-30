// The settings the shipped MLX configuration needs come from one place (apply_release_defaults), applied by tqserve's
// Model and by every Qwen35 engine; explicit environment settings still win.
#include "release_defaults.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    unsetenv("MLX_ENABLE_TF32");
    setenv("TQLLM_MLX_DFLASH_BLOCK", "8", 1);
    tqllm::mlx_backend::apply_release_defaults();
    bool ok = std::getenv("MLX_ENABLE_TF32") && std::strcmp(std::getenv("MLX_ENABLE_TF32"), "0") == 0;
    ok &= std::strcmp(std::getenv("TQLLM_MLX_DFLASH_BLOCK"), "8") == 0;
    unsetenv("TQLLM_MLX_DFLASH_BLOCK");
    tqllm::mlx_backend::apply_release_defaults();
    ok &= std::strcmp(std::getenv("TQLLM_MLX_DFLASH_BLOCK"), "16") == 0;
    std::puts(ok ? "MLX_RELEASE_DEFAULTS_PASS" : "MLX_RELEASE_DEFAULTS_FAIL");
    return ok ? 0 : 1;
}
