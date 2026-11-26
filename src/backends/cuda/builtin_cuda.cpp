// The engines a CUDA build serves, in match order (first match wins). Flash-Next (.tqfn) registers itself from
// src/models/qwen/qwen4exp/tqserve_flashnext.cpp, which only tqserve links.
#include "split_fit.hpp"
#include "fit_context.hpp"
#include "../../engine/lockstep.hpp"
#include "../../engine/registry.hpp"
#include "../../core/tp_comm.hpp"
#include "../../core/split_bounds.hpp"
#include "cuda_engines.hpp"
#include "hf_folder.hpp"
#include "nlohmann/json.hpp"

#include <cuda_runtime.h>
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
// --split layer: each rank loads its layers; the ranks first agree on the layer ranges (by each card's free memory, or
// --split-at). With the memory fit (-fit, default on) all ranks load at the same context, gather {loaded, free} over the
// control sockets and step together (src/engine/split_fit.hpp): keep, load at another context, or stop with what each
// rank needs. -fit off: one load at the asked context, as before.
std::unique_ptr<Engine> split_engine(const std::string & m, const std::string & t, uint32_t c, int slots, const EngineOptions & o_in) {
    if (o_in.tp_world < 2) throw std::invalid_argument("--split layer: two or more GPUs (--tp-rank R, --tp-world N)");
    if (o_in.tp_rank > 0 && o_in.tp_peer.empty())
        throw std::invalid_argument("--split layer: rank " + std::to_string(o_in.tp_rank) + " needs --tp-peer (rank 0's address)");
    auto comm = std::make_shared<TpComm>(o_in.tp_rank, o_in.tp_world, o_in.tp_peer.empty() ? "127.0.0.1" : o_in.tp_peer, o_in.tp_port);
    EngineOptions o = o_in;
    {
        std::vector<uint64_t> cost; uint64_t fixed = 0;
        qwen35_split_costs(m, c, slots, o, cost, fixed);
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) fr = 0;
        o.split_at = range_bounds(agree_split(*comm, cost, fixed, fr, o.split_at));
    }
    auto wrap = [&](std::unique_ptr<Engine> e) {
        return std::make_unique<LockstepEngine>(std::move(e), std::make_shared<TpChannel>(comm), o.tp_rank);
    };
    if (!o.fit) return wrap(make_qwen35_split_engine(m, t, c, slots, o, comm));
    const long reserve_mb = fit_slot_reserve_mb(o.slot_reserve_mb, !o.draft_model.empty() || o.native_mtp);
    const uint64_t margin = fit_margin(true, o.fit_target_mb, false) + (slots > 1 ? uint64_t(slots) * (uint64_t(reserve_mb) << 20) : 0);
    SplitFit fit(c, o.fit_min_ctx, margin);
    struct Wire { uint32_t ctx; uint32_t loaded; uint64_t free_b, total_b; };
    for (uint32_t ctx = c;;) {
        auto e = fit_try_load([&] { return make_qwen35_split_engine(m, t, ctx, slots, o, comm); }, [] { cudaGetLastError(); });
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) fr = tot = 0;
        int dev = 0, integrated = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev);
        if (integrated) {  // GB10: the page cache the OS reclaims on demand is free for the device too (as load_fitted)
            size_t avail = 0, mfree = 0;
            std::ifstream mi("/proc/meminfo");
            for (std::string k; mi >> k;) {
                size_t v = 0; std::string unit; mi >> v >> unit;
                if (k == "MemAvailable:") avail = v << 10; else if (k == "MemFree:") mfree = v << 10;
            }
            fr = unified_free(fr, tot, avail, mfree);
        }
        const Wire mine{ctx, e ? 1u : 0u, e ? uint64_t(fr) : 0, uint64_t(tot)};
        std::vector<Wire> all(static_cast<size_t>(o.tp_world));
        comm->ctl_all_gather(&mine, all.data(), sizeof mine);
        std::vector<SplitLoad> ranks;
        for (size_t r = 0; r < all.size(); ++r) {
            if (all[r].ctx != ctx)
                throw std::runtime_error("--split layer: the ranks lost step in the memory fit (context " + std::to_string(ctx) + " here, " +
                                         std::to_string(all[r].ctx) + " on rank " + std::to_string(r) + ")");
            ranks.push_back({all[r].loaded != 0, all[r].free_b, all[r].total_b});
        }
        const SplitFitStep step = fit.next(ctx, ranks);
        if (step.kind != SplitFitStep::keep || ctx != c)
            for (size_t r = 0; r < ranks.size(); ++r) {
                char line[160];
                if (!ranks[r].loaded) std::snprintf(line, sizeof line, "out of memory while loading");
                else std::snprintf(line, sizeof line, "%.2f GB free (%s %.2f GB margin)", double(ranks[r].free_b) / 1e9,
                                   ranks[r].free_b < margin ? "<" : ">=", double(margin) / 1e9);
                std::fprintf(stderr, "tqllm: fit: --split layer rank %zu at context %u: %s\n", r, ctx, line);
            }
        if (step.kind == SplitFitStep::keep) {
            if (ctx != c) {
                e->note = "to fit " + std::to_string(slots) + " slot(s) x " + std::to_string(c) + " on all " + std::to_string(o.tp_world) +
                          " GPUs (--split layer): context " + std::to_string(ctx);
                std::fprintf(stderr, "tqserve: fit: %s (choose yourself with a smaller --context or --split-at)\n", e->note.c_str());
            }
            return wrap(std::move(e));
        }
        e.reset();
        cudaGetLastError();
        if (step.kind == SplitFitStep::fail) throw std::runtime_error(fit.failure(ctx));
        std::fprintf(stderr, "tqllm: fit: --split layer: every rank loads again at context %u\n", step.ctx);
        ctx = step.ctx;
    }
}
void one_slot(int slots, const char * what) {
    if (slots != 1) throw std::invalid_argument(std::string(what) + " runs one slot (--np 1)");
}
}  // namespace

void register_builtin_engines() {
    register_engine({"inkling2", [](const std::filesystem::path & p) { return hf_folder_arch(p.string()) == "inkling2-loop"; },
                     [](const std::string & m, const std::string &, uint32_t c, int slots, const EngineOptions & o) {
                         one_slot(slots, "Inkling-2-Mini");
                         return make_inkling2_engine(m, c, o.native_mtp, o.exact, o.draft_model, o.expert_kernels);  // --mtp: the DSpark head in mtp.safetensors; -md: a DFlash2 folder
                     },
                     true});  // exact: spec == plain, verify blocks == single steps
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
