// Builds without the MoE engines (TQLLM_MOE_ENGINES=OFF, the native Windows default): the engine list still recognises
// those models by GGUF architecture and says why they cannot load, instead of trying another engine.
#include "engine.hpp"
#include "gguf.hpp"

#include <stdexcept>

namespace tqllm {
namespace {
[[noreturn]] void unavailable(const char * model) {
    throw std::runtime_error(std::string(model) + " needs the MoE engines, which this build leaves out (TQLLM_MOE_ENGINES=OFF; "
                             "native Windows builds do so for now): run it under Linux or WSL 2");
}
}  // namespace
bool is_glm5next_gguf(const std::string & path) {
    const std::string a = gguf_architecture(path);
    return a == "glm5-next" || a == "glm5next";
}
bool is_minimax_gguf(const std::string & path) { return gguf_architecture(path) == "minimax-m3"; }
std::unique_ptr<Engine> make_glm5next_engine(const std::string &, uint32_t) { unavailable("GLM-5.3-Flash"); }
std::unique_ptr<Engine> make_minimax_engine(const std::string &, uint32_t, const std::string &, const std::string &, const TpConfig &,
                                            const std::string &, const std::string &) {
    unavailable("MiniMax-M3.1-Flash");
}
}  // namespace tqllm
