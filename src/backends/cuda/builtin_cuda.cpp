// The engines a CUDA build serves, in match order (first match wins). Flash-Next (.tqfn) registers itself from
// src/models/qwen/qwen4exp/tqserve_flashnext.cpp, which only tqserve links.
#include "../../engine/lockstep.hpp"
#include "../../engine/registry.hpp"
#include "../../core/tp_comm.hpp"
#include "cuda_engines.hpp"
#include "hf_folder.hpp"
#include "nlohmann/json.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tqllm {
namespace {
// a GGUF file or a Hugging Face checkpoint folder (src/core/hf) is dispatched by its architecture
bool gguf_like(const std::filesystem::path & p) { return p.extension() == ".gguf" || hf_folder(p.string()); }
std::filesystem::path model_dir(const std::filesystem::path & p) { return std::filesystem::is_directory(p) ? p : p.parent_path(); }
std::string model_type(const std::filesystem::path & dir) {
    std::ifstream in(dir / "config.json");
    if (!in) throw std::runtime_error("cannot open model config: " + (dir / "config.json").string());
    std::stringstream ss;
    ss << in.rdbuf();
    const Json root = parse_json(ss.str());
    const Json * t = root.get("model_type");
    return t ? t->str : std::string();
}
bool is_qwen35_checkpoint(const std::filesystem::path & p) {
    try { return model_type(model_dir(p)) == "qwen3_5"; } catch (const std::exception &) { return false; }
}
struct TpChannel final : ByteChannel {
    explicit TpChannel(std::shared_ptr<TpComm> t) : tp(std::move(t)) {}
    void send(const void * p, size_t n) override { tp->ctl_send(p, n); }
    void recv(void * p, size_t n) override { tp->ctl_recv(p, n); }
    std::shared_ptr<TpComm> tp;
};
// --split layer: one fixed load per rank (the memory fit's retries would desynchronise the ranks)
std::unique_ptr<Engine> split_engine(const std::string & m, const std::string & t, uint32_t c, int slots, const EngineOptions & o) {
    if (o.tp_world != 2) throw std::invalid_argument("--split layer: two GPUs (--tp-rank 0 / 1) so far");
    if (o.tp_rank == 1 && o.tp_peer.empty()) throw std::invalid_argument("--split layer: rank 1 needs --tp-peer (rank 0's address)");
    auto comm = std::make_shared<TpComm>(o.tp_rank, o.tp_world, o.tp_peer.empty() ? "127.0.0.1" : o.tp_peer, o.tp_port);
    auto e = make_qwen35_split_engine(m, t, c, slots, o, comm);
    return std::make_unique<LockstepEngine>(std::move(e), std::make_shared<TpChannel>(comm), o.tp_rank);
}
void one_slot(int slots, const char * what) {
    if (slots != 1) throw std::invalid_argument(std::string(what) + " runs one slot (--np 1)");
}
}  // namespace

void register_builtin_engines() {
    register_engine({"minimax", [](const std::filesystem::path & p) { return gguf_like(p) && is_minimax_gguf(p.string()); },
                     [](const std::string & m, const std::string & t, uint32_t c, int slots, const EngineOptions & o) {
                         one_slot(slots, "MiniMax-M3.1-Flash");
                         TpConfig tp;
                         tp.rank = o.tp_rank; tp.world = o.tp_world; tp.peer = o.tp_peer; tp.port = o.tp_port;
                         if (tp.world > 1 && tp.rank != 0) throw std::invalid_argument("tqserve is rank 0; rank 1 runs glm-gpu --tp-rank 1");
                         return make_minimax_engine(m, c, o.draft_model, t, tp, o.hot_map, o.draft_vocab);  // -md: the DSpark draft GGUF
                     },
                     false});
    register_engine({"glm5next", [](const std::filesystem::path & p) { return gguf_like(p) && is_glm5next_gguf(p.string()); },
                     [](const std::string & m, const std::string &, uint32_t c, int slots, const EngineOptions &) {
                         one_slot(slots, "GLM-5.3-Flash");
                         return make_glm5next_engine(m, c);
                     },
                     false});
    register_engine({"qwen3-gguf", [](const std::filesystem::path& p) {
                         return p.extension() == ".gguf" && gguf_architecture(p.string()) == "qwen3";
                     }, [](const std::string& m, const std::string& t, uint32_t c, int slots, const EngineOptions& o) {
                         return make_qwen35_engine(m, t, c, slots, o);
                     }, false});  // exactness gate pending for the 128-wide Qwen3 target
    // llama.cpp-tq3 GGUF, or a Qwen3.8-27B folder hf_arch.cpp maps (exl3 / BF16 as published), before the old checkpoint path
    register_engine({"qwen35-gguf", [](const std::filesystem::path & p) { return p.extension() == ".gguf" || hf_folder_arch(p.string()) == "qwen35"; },
                     [](const std::string & m, const std::string & t, uint32_t c, int slots, const EngineOptions & o) {
                         if (o.split_layers) return split_engine(m, t, c, slots, o);
                         return load_fitted(m, t, c, slots, o);
                     }});
    register_engine({"qwen35-checkpoint", is_qwen35_checkpoint,
                     [](const std::string & m, const std::string & t, uint32_t c, int slots, const EngineOptions & o) {
                         if (slots != 1) throw std::invalid_argument("multiple slots need a qwen35 GGUF model");
                         return make_qwen35_engine(model_dir(m).string(), t, c, 1, o);
                     }});
    register_engine({"qwen2", [](const std::filesystem::path & p) { return p.extension() != ".tqfn" && p.extension() != ".gguf"; },
                     [](const std::string & m, const std::string & t, uint32_t c, int slots, const EngineOptions &) {
                         if (slots != 1) throw std::invalid_argument("multiple slots need a qwen35 GGUF model");
                         return make_qwen2_engine(m, t, c);  // DIR/weights.tqbin
                     }});
}
}  // namespace tqllm
