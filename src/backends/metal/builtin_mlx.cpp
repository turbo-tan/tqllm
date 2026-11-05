// The engines a Metal (MLX) build serves, in match order (first match wins).
#include "../../engine/registry.hpp"
#include "engine_mlx.hpp"

namespace tqllm {
void register_builtin_engines() {
    register_engine({"qwen35-mlx", [](const std::filesystem::path & p) { return p.extension() == ".gguf"; }, &make_mlx_qwen35_engine});
}
}  // namespace tqllm
