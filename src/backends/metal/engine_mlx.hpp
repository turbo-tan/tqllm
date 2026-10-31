#pragma once
// The Metal engine (engine_mlx.cpp): Qwen3.8 27B GGUF + the DFlash2 drafter on MLX, speculative decoding only.
#include "../../engine/engine.hpp"
#include <memory>
#include <string>

namespace tqllm {
std::unique_ptr<Engine> make_mlx_qwen35_engine(const std::string& weights, const std::string& tokenizer, uint32_t context, int slots,
                                               const EngineOptions& options);
}  // namespace tqllm
