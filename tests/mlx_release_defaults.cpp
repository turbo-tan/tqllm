// The MLX setting the shipped configuration needs (no TF32) comes from one place (apply_release_defaults), applied by
// tqserve's Model and by every Qwen35 engine; an explicit MLX_ENABLE_TF32 still wins.
#include "release_defaults.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    unsetenv("MLX_ENABLE_TF32");
    tqllm::mlx_backend::apply_release_defaults();
    bool ok = std::getenv("MLX_ENABLE_TF32") && std::strcmp(std::getenv("MLX_ENABLE_TF32"), "0") == 0;
    setenv("MLX_ENABLE_TF32", "1", 1);
    tqllm::mlx_backend::apply_release_defaults();
    ok &= std::strcmp(std::getenv("MLX_ENABLE_TF32"), "1") == 0;
    std::puts(ok ? "MLX_RELEASE_DEFAULTS_PASS" : "MLX_RELEASE_DEFAULTS_FAIL");
    return ok ? 0 : 1;
}
