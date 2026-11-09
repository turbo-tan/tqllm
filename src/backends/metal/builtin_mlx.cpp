// The engines a Metal (MLX) build serves, in match order (first match wins).
#include "../../engine/registry.hpp"
#include "engine_mlx.hpp"
#include "gguf.hpp"

namespace tqllm {
void register_builtin_engines() {
    register_engine({"qwen3-mlx", [](const std::filesystem::path& p) {
                         return p.extension() == ".gguf" && gguf_architecture(p.string()) == "qwen3";
                     }, &make_mlx_qwen35_engine, false});
    register_engine({"qwen35-mlx", [](const std::filesystem::path & p) { return p.extension() == ".gguf"; }, &make_mlx_qwen35_engine});
}
}  // namespace tqllm
