#pragma once
// Flash-Next behind the tqllm Engine interface (tqserve). The model is a manifest file NAME.tqfn of key=value lines:
//   gguf=<shard> (one line per shard, in order), mtp=<MTP gguf>, experts=<packed experts>, draft_vocab=<file>,
//   optional overrides: slots=N|auto (default auto: sized from the device), budget_gb (fixed slots only), drafts=3,
//   chunk (--prefill-chunk wins), gsq_rco=1, layer_slots=<per-layer routing plan, scaled to the slots>
// Pilot only: hot_map=<int32 map>, cold_slots=16, cold_host_mb=3072, cold_readers=0; no packed bank or pruning.
// One slot, greedy MTP speculative decoding (exact: tokens identical to plain greedy). A fresh DecodeModel per request
// resets the recurrent, PLE, attention and MTP state; the expert caches persist across requests.
#include "fn_model.hpp"
#include "engine.hpp"
#include "cold_expert_store.hpp"
#include "cold_expert_memo.hpp"
#include "expert_read_pool.hpp"
#include "expert_fetch_queue.hpp"

namespace tqllm::flashnext {
using ColdFetchQueue=ExpertFetchQueue<std::array<ExpertPayload,3>>;
struct FnServe {
    std::vector<std::string> ggufs; std::string mtp, experts, draft_vocab, hot_map_path, cold_store_path;
    std::string layer_slots;  // manifest layer_slots=: per-layer expert slots when no experts file carries a .slots plan
    std::unique_ptr<ColdExpertStore> cold_store;
    bool cold_fast = false, cold_reclaim_hot = false, cold_mmap = false, cold_streaming = false;
    int cold_overlap = 0;
    int cold_readers = 0,cold_prefetch_k=0;
    std::unique_ptr<ColdFetchQueue> cold_fetch_queue;
    std::unique_ptr<ExpertReadPool> cold_read_pool;
    std::vector<int32_t> hot_map; int cold_slots = 16, hot_slots = 0;
    std::unique_ptr<MixedExpertCache> cold_cache;
    ColdExpertMemo cold_host{size_t(3) << 30};
    int slots = 176, drafts = 3, chunk = 8; double budget_gb = 18; bool gsq_rco = true, slots_auto = false;
    std::vector<std::string> arg_store; std::vector<char *> argv;
    std::unique_ptr<Fixtures> fixtures; std::unique_ptr<Checkpoint> cp;
    std::unique_ptr<MixedExpertCache> cache, mtp_cache; PackedExperts packed; char * mtp_packed = nullptr; size_t mtp_block = 0;
    ScratchArena arena; size_t arena_prefill = 0, arena_decode = 0; int kv_capacity_full = 0;
    size_t state_full = 0; int state_tokens = 0;  // one sequence's full-context state (context_keep) and its token count
    size_t plan_budget = 0;  // free + bank bytes after the warm-up (pool trimmed, 256 MB steps): a prompt length always plans the same chunk / block
    float * res_host = nullptr; size_t res_host_cap = 0;  // pinned prompt residuals of a multi-chunk prefill (one prefill at a time)
};
class FlashNextEngine final : public tqllm::Engine {
    FnServe f_; uint32_t context_;
    // one sequence per slot (KV, recurrent and MTP state, sampling); weights, expert caches and scratch are shared
    struct SlotState {
        std::unique_ptr<DecodeModel> model_; std::unique_ptr<Device<float>> prompt_res_;
        std::unique_ptr<tqllm::BankLedger> ledger_;  // this request's borrowed expert banks
        tqllm::ExpertCacheStats stats_at_decode_{}, mtp_stats_at_decode_{};  // TQLLM_DEBUG=mem: cache counters when the decode of this request began
        ExpertCacheStats cold_at_decode_{};
        ColdExpertStore::Stats cold_store_at_decode_{};
        ColdFetchQueue::Stats cold_fetch_at_decode_{};
        size_t plan_catchup_bytes_ = 0;             // the plan's catch-up peak: what an overlapped refill must leave free
        std::vector<int> out_; mutable std::vector<uint32_t> out_view_; int generate_ = 0, d1_ = 0; bool done_ = true; uint64_t steps_ = 0, toks_ = 0; uint64_t prefill_us_ = 0, decode_us_ = 0;
        std::chrono::steady_clock::time_point t_begin_, t_decode_;
        float temp_ = 0.f, top_p_ = 1.f; int top_k_ = 0; std::mt19937_64 rng_{1};
    };
    std::vector<SlotState> slots_; int cur_ = 0;
    SlotState & S() { return slots_[size_t(cur_)]; }
    uint32_t slot_context() const { return context_; }  // Model passes the per-slot context (tqserve: --context / --np)
    // the shared caches copy misses on the active slot's stream
    void bind() { if (S().model_ && S().model_->miss_stream) { f_.cache->set_miss_stream(S().model_->miss_stream); f_.mtp_cache->set_miss_stream(S().model_->miss_stream); } }
    static bool is_eos(int t) { return t == 248046 || t == 248044; }
    // T > 0: sample each verify position from the target's own distribution; a draft is accepted only if it equals that
    // sample, so the output follows the target exactly (greedy drafts just lower the acceptance)
    std::vector<int> pick(const float * rows, int n);
public:
    FlashNextEngine(const std::string & manifest, uint32_t context, int nslots);
    // TQLLM_DEBUG=mem: where the device memory is (MiB); "other" = KV, QSA keys, recurrent state, CUDA context
    void ledger(const char * at) const;
    // borrowed banks back, filled with the prompt's most-routed experts (Strata does the same)
    // after release_prefill (buffers freed, pool trimmed): the ledger rebuilds the borrowed banks; a bank that no longer
    // fits refills on its first use in decode
    void refill_borrowed();
    void arena_resize(size_t bytes);
    void spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int) override;
    bool spec_advance() override;
    std::vector<uint32_t> spec_finish(uint64_t * stats) override {
        bind();
        while (spec_advance()) {}
        // steps, drafted, accepted drafts, decode us, prefill us (tqserve timings)
        if (stats) { stats[0] = S().steps_; stats[1] = S().steps_ * uint64_t(f_.drafts); stats[2] = S().toks_ - S().steps_; stats[3] = S().decode_us_; stats[4] = S().prefill_us_; }
        std::vector<uint32_t> r(S().out_.begin(), S().out_.end());
        if (tqllm::debug_on("mem")) {  // the decode phase's expert cache traffic (the warm-prompt decode question of 10-17)
            const auto s = f_.cache->stats(), & b = S().stats_at_decode_, m = f_.mtp_cache->stats(), & mb = S().mtp_stats_at_decode_;
            std::fprintf(stderr, "LEDGER decode of %zu tokens: expert hits %llu misses %llu (%.1f %% hit) transfer %.2f GB, banks %.0f MB | MTP hits %llu misses %llu transfer %.2f GB, bank %.0f MB\n", r.size(),
                         (unsigned long long) (s.hits - b.hits), (unsigned long long) (s.misses - b.misses),
                         100.0 * double(s.hits - b.hits) / std::max<double>(1, double(s.hits - b.hits + s.misses - b.misses)),
                         double(s.transfer_bytes - b.transfer_bytes) / 1e9, s.device_bytes / 1048576.0,
                         (unsigned long long) (m.hits - mb.hits), (unsigned long long) (m.misses - mb.misses), double(m.transfer_bytes - mb.transfer_bytes) / 1e9, m.device_bytes / 1048576.0);
            std::fprintf(stderr, "LEDGER decode rounds %llu | since start: prefetched %llu used %llu, pf pending %llu running %llu waited-unread %llu skipped %llu\n",
                         (unsigned long long) S().steps_, (unsigned long long) f_.cache->prefetched(), (unsigned long long) f_.cache->prefetch_used(),
                         (unsigned long long) s.pf_pending, (unsigned long long) s.pf_running, (unsigned long long) s.pf_waited_unread, (unsigned long long) s.pf_skipped);
        }
        S().model_.reset(); S().prompt_res_.reset();
        return r;
    }
    // out_ holds int ids: an append-only uint32 mirror per slot (only new ids converted; rebuilt after a reset)
    const std::vector<uint32_t> & spec_tokens(int s) const override {
        const auto & o = slots_[size_t(s)].out_; auto & v = slots_[size_t(s)].out_view_;
        if (v.size() > o.size() || (!v.empty() && v.back() != uint32_t(o[v.size() - 1]))) v.clear();
        for (size_t i = v.size(); i < o.size(); ++i) v.push_back(uint32_t(o[i]));
        return v;
    }
    int slots() const override { return int(slots_.size()); }
    void select_slot(int s) override { require(s >= 0 && s < int(slots_.size()), "Flash-Next slot out of range"); cur_ = s; }
    // one step for every listed slot: each drafts with its own MTP state, one verify pass covers all of them
    // (TQFN_SLOT_BATCH=0, a single active slot, more than 64 rows or more experts than cache slots: one slot at a time)
    void spec_step(const int * ids, int n, bool * done) override;
    bool incremental() const override { return true; }
    bool has_drafter() const override { return true; }
    void set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence = 0.f) override {
        (void) presence;
        S().temp_ = temperature; S().top_p_ = top_p > 0.f && top_p <= 1.f ? top_p : 1.f; S().top_k_ = top_k; S().rng_.seed(seed ? seed : 0x9e3779b97f4a7c15ull);
    }
    uint32_t step(uint32_t, uint32_t, bool, float *) override { throw std::runtime_error("Flash-Next serves through spec_begin / spec_advance"); }
    void verify(const uint32_t *, int, uint32_t, uint32_t *) override { throw std::runtime_error("Flash-Next: verify is internal"); }
    void commit(int) override { throw std::runtime_error("Flash-Next: commit is internal"); }
    const float * logits() const override { return nullptr; }
    void synchronize() override { ck(cudaDeviceSynchronize()); }
    uint32_t layers() const override { return uint32_t(f_.cp->config.layers); }
    uint32_t hidden() const override { return uint32_t(f_.cp->config.hidden); }
    uint32_t vocab() const override { return uint32_t(f_.cp->head_resident().m.rows); }
    uint32_t eos() const override { return 248046; }
    uint32_t context() const override { return context_; }
};
}  // namespace tqllm::flashnext
