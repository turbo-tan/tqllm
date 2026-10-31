#pragma once
// Engine registry: the engines a build serves, matched in registration order (first match wins). Each backend's build
// lists its engines in register_builtin_engines() (src/engine/builtin_cuda.cpp, src/backends/mlx/builtin_mlx.cpp): an
// explicit list, not static initializers, which a static library would drop with an unreferenced object file.
#include "engine.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace tqllm {
struct EngineEntry {
    std::string name;
    bool (*matches)(const std::filesystem::path & model) = nullptr;
    std::unique_ptr<Engine> (*make)(const std::string & model, const std::string & tokenizer, uint32_t context, int slots,
                                    const EngineOptions & options) = nullptr;
    bool exact_path = true;  // false: no exact path yet (with --exact on it warns and runs its own path)
};
void register_engine(EngineEntry entry);           // an entry with the same name replaces the earlier one
const std::vector<EngineEntry> & engines();        // registers the built-in engines on first use
const EngineEntry * find_engine(const std::filesystem::path & model);  // nullptr: no engine in this build serves it
void register_builtin_engines();                   // one definition per backend build
}  // namespace tqllm
