// Flash-Next behind the tqllm Engine interface (tqserve), registered as the .tqfn model type.
#include "registry.hpp"
#include "fn_engine.hpp"
#include "core/tp_comm.hpp"

namespace {
using tqllm::flashnext::FlashNextEngine;
struct TpChannel final : tqllm::ByteChannel {  // the lockstep calls go over the link's host control socket
    explicit TpChannel(std::shared_ptr<tqllm::TpComm> t) : tp(std::move(t)) {}
    void send(const void * p, size_t n) override { tp->ctl_send(p, n); }
    void recv(void * p, size_t n) override { tp->ctl_recv(p, n); }
    std::shared_ptr<tqllm::TpComm> tp;
};
std::unique_ptr<tqllm::Engine> make_flashnext(const std::string & manifest, const std::string &, uint32_t context, int slots,
                                              const tqllm::EngineOptions & o) {
    FnConfig cfg; cfg.exact = o.exact; cfg.prompt_cache = o.prompt_cache; cfg.prompt_cache_set = o.prompt_cache_set; cfg.prompt_park = o.prompt_park;
    if (o.prefill_chunk > 0) { cfg.tune.chunk = o.prefill_chunk; cfg.tune.chunk_set = true; }
    if (o.prefill_sub > 0) cfg.tune.sub = o.prefill_sub;
    if (o.prefill_stage > 0) { cfg.tune.stage = o.prefill_stage; cfg.tune.stage_set = true; }
    if (o.prefill_reserve_mb >= 0) cfg.tune.reserve_mb = o.prefill_reserve_mb;
    if (o.prefill_arena_mb >= 0) cfg.tune.arena_mb = o.prefill_arena_mb;
    if (o.dense_cache_mb >= 0) { cfg.tune.dense_cache_mb = o.dense_cache_mb; cfg.tune.dense_cache_set = true; }
    if (o.prefill_borrow == "all") cfg.tune.borrow = tqllm::BorrowMode::All;
    else if (!o.prefill_borrow.empty() && o.prefill_borrow != "auto") { cfg.tune.borrow = tqllm::BorrowMode::Count; cfg.tune.borrow_count = std::atoi(o.prefill_borrow.c_str()); }
    if (o.split_experts) {
        if (o.tp_world != 2) throw std::invalid_argument("--split experts: two ranks (--tp-rank 0 / 1)");
        if (o.tp_rank == 1 && o.tp_peer.empty()) throw std::invalid_argument("--split experts: rank 1 needs --tp-peer (rank 0's address)");
        cfg.ep = std::make_shared<tqllm::TpComm>(o.tp_rank, 2, o.tp_peer.empty() ? "127.0.0.1" : o.tp_peer, o.tp_port);
    }
    if (o.split_layers) {  // --split layer: one tqserve per GPU, rank 0 serves, the others follow (Model wraps the engine)
        if (o.split_experts) throw std::invalid_argument("--split layer and --split experts cannot be combined");
        if (o.tp_world < 2) throw std::invalid_argument("--split layer: two or more GPUs (--tp-rank R, --tp-world N)");
        if (o.tp_rank > 0 && o.tp_peer.empty())
            throw std::invalid_argument("--split layer: rank " + std::to_string(o.tp_rank) + " needs --tp-peer (rank 0's address)");
        cfg.split = std::make_shared<tqllm::TpComm>(o.tp_rank, o.tp_world, o.tp_peer.empty() ? "127.0.0.1" : o.tp_peer, o.tp_port);
        cfg.split_at = o.split_at;
        auto comm = cfg.split;
        auto e = std::make_unique<FlashNextEngine>(manifest, context, slots, cfg);
        e->lockstep_channel = std::make_shared<TpChannel>(comm); e->lockstep_rank = o.tp_rank;  // Model wraps it
        return e;
    }
    return std::make_unique<FlashNextEngine>(manifest, context, slots, cfg);
}
// a model path ending in .tqfn (a manifest of shard / expert / MTP files) selects Flash-Next; this object file is linked
// into tqserve directly (not through the static library), so its registration runs
const bool registered_flashnext = (tqllm::register_engine({"flashnext", [](const std::filesystem::path & p) { return p.extension() == ".tqfn"; },
                                                           &make_flashnext}), true);
}  // namespace
