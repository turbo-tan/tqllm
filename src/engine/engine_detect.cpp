// Engine detection from GGUF metadata, host-only and shared by every backend (AGENTS.md rule 15): a backend that does not
// build an engine still recognises its files and refuses them by name, instead of letting them fall through to another loader.
#include "engine.hpp"
#include "gguf.hpp"

namespace tqllm {
bool is_glm5next_gguf(const std::string & path) {
    const std::string a = gguf_architecture(path);
    return a == "glm5-next" || a == "glm5next";
}
bool is_minimax_gguf(const std::string & path) { return gguf_architecture(path) == "minimax-m3"; }
}  // namespace tqllm
