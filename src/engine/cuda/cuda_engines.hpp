#pragma once
// CUDA builds: the loaders src/engine/builtin_cuda.cpp registers.
#include "../engine.hpp"
#include <memory>
#include <string>

namespace tqllm {
// qwen35 GGUF with the memory fit: leaves -fitt free, applies the fit steps and reloads; the engine's note says what it chose
std::unique_ptr<Engine> load_fitted(const std::string & path, const std::string & tokenizer, uint32_t context, int slots,
                                    EngineOptions options);
// converted Qwen2 (DIR/weights.tqbin)
std::unique_ptr<Engine> make_qwen2_engine(const std::string & weights, const std::string & tokenizer, uint32_t context);
}  // namespace tqllm
