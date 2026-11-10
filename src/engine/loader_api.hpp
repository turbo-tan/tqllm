#pragma once
// The engine assembly both GPU backends compile (AGENTS.md: a backend folder holds the registration list and
// the platform's own kernels, not a copy of the loader). Implementation: src/engine/load_fitted.cpp and
// src/models/qwen/qwen2/qwen2_engine.cpp.
#include "engine.hpp"

#include <memory>
#include <string>

namespace tqllm {
// qwen35 GGUF with the memory fit: leaves -fitt free, applies the fit steps and reloads; the engine's note says what it chose
std::unique_ptr<Engine> load_fitted(const std::string & path, const std::string & tokenizer, uint32_t context, int slots,
                                    EngineOptions options);
// converted Qwen2 (DIR/weights.tqbin)
std::unique_ptr<Engine> make_qwen2_engine(const std::string & weights, const std::string & tokenizer, uint32_t context);
}  // namespace tqllm