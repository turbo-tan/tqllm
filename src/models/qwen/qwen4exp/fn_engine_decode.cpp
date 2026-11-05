// Flash-Next decode: speculative MTP steps per slot and token selection.
#include "fn_engine.hpp"

namespace tqllm::flashnext {

std::vector<int> FlashNextEngine::pick(const float * rows, int n) {
    auto & checkpoint = *f_.cp;
    if (S().temp_ <= 0.f) return checkpoint.head_argmax_rows(rows, n);
    const auto lg = checkpoint.head_logits_device(rows, n); const size_t V = lg.size() / size_t(n);
    std::vector<int> r(n); std::vector<std::pair<float, int>> cand;
    for (int i = 0; i < n; ++i) {
        const float * l = lg.data() + size_t(i) * V; float mx = l[0]; for (size_t v = 1; v < V; ++v) mx = std::max(mx, l[v]);
        cand.clear();
        for (size_t v = 0; v < V; ++v) { const float p = std::exp((l[v] - mx) / S().temp_); if (p > 1e-9f) cand.push_back({p, int(v)}); }
        std::sort(cand.begin(), cand.end(), [](auto a, auto b) { return a.first > b.first; });
        if (S().top_k_ > 0 && int(cand.size()) > S().top_k_) cand.resize(size_t(S().top_k_));
        double total = 0; for (auto & c : cand) total += c.first;
        if (S().top_p_ < 1.f) { double acc = 0; size_t keep = 0; for (; keep < cand.size(); ++keep) { acc += cand[keep].first / total; if (acc >= S().top_p_) { ++keep; break; } }
                            cand.resize(std::max<size_t>(1, std::min(keep, cand.size()))); total = 0; for (auto & c : cand) total += c.first; }
        double u = std::uniform_real_distribution<double>(0, total)(S().rng_), acc = 0; r[i] = cand.back().second;
        for (auto & c : cand) { acc += c.first; if (acc >= u) { r[i] = c.second; break; } }
    }
    return r;
}

// TQLLM_DEBUG=steptrace: one line per decode step: slots in the step, drafts, decode layers on the waves path, fewest bank
// slots, then per slot its output length after the step and the tokens it accepted (an np N run against np 1 finds the
// first step whose tokens differ and what that step did differently)
void FlashNextEngine::trace_step(int slots, int drafts, const std::vector<std::pair<size_t, std::vector<int>>> & per_slot, const std::vector<int> & slot_ids) {
    static const bool on = tqllm::debug_on("steptrace");
    if (on) {
        static uint64_t n = 0;
        std::string line = "STEPTRACE " + std::to_string(++n) + " slots " + std::to_string(slots) + " drafts " + std::to_string(drafts) +
                           " waves " + std::to_string(f_.cp->rt.trace_waves) + " minslots " + std::to_string(f_.cp->rt.trace_min_slots);
        for (size_t i = 0; i < per_slot.size(); ++i) { const auto & [len, toks] = per_slot[i]; line += " |s" + std::to_string(i < slot_ids.size() ? slot_ids[i] : -1) + ":" + std::to_string(len); for (int t : toks) line += " " + std::to_string(t); }
        std::fprintf(stderr, "%s\n", line.c_str());
    }
    f_.cp->rt.trace_waves = 0; f_.cp->rt.trace_min_slots = 1 << 30;
}
// a finished request's prompt + output becomes copy history for the next ones (the last 4, at most 32K tokens each)
void FlashNextEngine::remember_copy(SlotState & sl) {
    auto & c = sl.ctx_; c.insert(c.end(), sl.out_.begin() + long(std::min(c.size() - sl.plen_, sl.out_.size())), sl.out_.end());
    if (c.size() > (size_t(32) << 10)) c.erase(c.begin(), c.end() - long(size_t(32) << 10));
    copy_history_.push_back(std::move(c)); c.clear();
    if (copy_history_.size() > 4) copy_history_.erase(copy_history_.begin());
}
bool FlashNextEngine::spec_advance() {
    if (S().done_) return false;
    if (S().steps_ == 8 && tqllm::debug_on("mem")) ledger("decode step 8");
    bind();
    refill_borrowed();  // no-op unless an earlier refill ran out of memory
    auto & checkpoint = *f_.cp; const int drafts = f_.drafts;
    const int y = S().out_.back();
    // copy drafts: a continuation of an earlier occurrence of the last 3 tokens (this request or the last few) that starts
    // with the MTP's own first draft replaces the MTP chain (exact either way: the verify keeps what the model emits)
    if (S().ctx_.size() < S().plen_ + S().out_.size()) S().ctx_.insert(S().ctx_.end(), S().out_.begin() + long(S().ctx_.size() - S().plen_), S().out_.end());
    std::vector<const std::vector<int> *> hist;
    for (auto it = copy_history_.rbegin(); it != copy_history_.rend(); ++it) hist.push_back(&*it);
    std::vector<int> dr = drafts > 0 ? tqllm::flashnext::copy_draft(S().ctx_, hist, 3, drafts) : std::vector<int>{};
    if (!dr.empty() && dr[0] == S().d1_) ++S().copied_;
    else dr = S().model_->mtp_drafts(S().d1_, drafts);
    std::vector<int> block{y}; block.insert(block.end(), dr.begin(), dr.end());
    auto rows = S().model_->step(block);
    const auto v = pick(rows->p, int(block.size()));
    int m = 1; while (m <= int(dr.size()) && v[m - 1] == dr[m - 1]) ++m;
    for (int i = 0; i < m && int(S().out_.size()) < S().generate_; ++i) { S().out_.push_back(v[i]); if (is_eos(v[i])) { m = i + 1; break; } }
    S().model_->accept(m); ++S().steps_; S().toks_ += m; S().offered_ += dr.size(); ++S().taken_[size_t(std::min(m, 16))];
    trace_step(1, int(dr.size()), {{S().out_.size(), std::vector<int>(v.begin(), v.begin() + m)}}, {-1});
    // after the request's first step (decode's lazy buffers, e.g. the host-K/V gather, exist now): the banks grow into what
    // is free beyond 1 GB; the start-up keep is sized for a prefill, which borrows the banks back anyway (10-28 128K: 3.6 GB
    // idle in decode at 133 slots / layer vs Strata's ~172). Growing before the first step left that gather no room (OOM)
    // only after a prompt that borrowed the banks: grown after a short request, the next long prefill had no room (OOM)
    if (S().steps_ == 1 && S().refilled_ && slots_.size() == 1 && f_.slots_auto && !f_.cp->run.small_card && kv_host_enabled()) {
        { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
        if (const int grown = f_.cache->grow_into(size_t(1024) << 20, f_.cp->config.experts))
            std::fprintf(stderr, "tqllm: expert banks grown for decode: %d layers, up to %d slots\n", grown, f_.cache->slots());
    }
    S().d1_ = S().model_->mtp_batch(std::vector<int>(v.begin(), v.begin() + m), S().model_->res_keep->p);
    S().done_ = int(S().out_.size()) >= S().generate_ || is_eos(S().out_.back());
    if (S().done_ && f_.cp->rt.profile_decode_req) { ck(cudaDeviceSynchronize()); ck(cudaProfilerStop()); f_.cp->rt.profile_decode_req = false; }
    if (S().done_) remember_copy(S());
    if (S().done_) { S().model_->set_spec(false); ck(cudaDeviceSynchronize());
                 S().decode_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - S().t_decode_).count()); }
    if (S().done_ && f_.cold_cache && tqllm::debug_on("mem")) {
        const auto c = f_.cold_cache->stats(); const auto b = S().cold_at_decode_;
        const auto d = f_.cold_store ? f_.cold_store->stats() : ColdExpertStore::Stats{};
        const auto a = S().cold_store_at_decode_;
        std::fprintf(stderr, "COLD_DECODE tokens=%zu steps=%llu hits=%llu misses=%llu h2d_bytes=%llu loads=%llu read_bytes=%llu read_ms=%.3f checksum_ms=%.3f mapped_fault_verify_ms=%.3f\n",
            S().out_.size()-1, (unsigned long long)S().steps_, (unsigned long long)(c.hits-b.hits),
            (unsigned long long)(c.misses-b.misses), (unsigned long long)(c.transfer_bytes-b.transfer_bytes),
            (unsigned long long)(d.loads-a.loads), (unsigned long long)(d.bytes-a.bytes),
            (d.read_ns-a.read_ns)/1e6, (d.checksum_ns-a.checksum_ns)/1e6, (d.mapped_ns-a.mapped_ns)/1e6);
    }
    if (S().done_ && f_.cold_fetch_queue && tqllm::debug_on("mem")) {
        const auto q = f_.cold_fetch_queue->stats(), b = S().cold_fetch_at_decode_;
        std::fprintf(stderr, "COLD_QUEUE prefetched=%llu used=%llu promoted=%llu deduplicated=%llu expired=%llu rejected=%llu demand_reads=%llu completed=%llu\n",
            (unsigned long long)(q.prefetched-b.prefetched), (unsigned long long)(q.used-b.used),
            (unsigned long long)(q.promoted-b.promoted), (unsigned long long)(q.deduplicated-b.deduplicated),
            (unsigned long long)(q.expired-b.expired), (unsigned long long)(q.rejected-b.rejected),
            (unsigned long long)(q.demand_reads-b.demand_reads), (unsigned long long)(q.completed-b.completed));
    }
    return !S().done_;
}

void FlashNextEngine::spec_step(const int * ids, int n, bool * done) {
    const int topk = f_.cp->config.topk;
    std::vector<int> act;
    for (int i = 0; i < n; ++i) { done[i] = slots_[size_t(ids[i])].done_; if (!done[i]) act.push_back(i); }
    // fewer drafts per slot when more slots run, so their rows still share one verify pass (each draft row is exact: the
    // text does not depend on the draft count)
    // the fewest slots of any layer (per-layer slot tiers): a step whose routed experts exceed one layer's bank takes the
    // waves path there, which sums the down projection in another order than np 1 (F39; np 8, 11-03: 86 slots in the
    // smallest layer vs 122 on average, 1-2 of 8 answers differed at a near tie)
    int fewest = f_.cp->config.experts;
    for (int l = 0; l < f_.cp->config.layers; ++l) if (f_.cache->has_layout(l)) fewest = std::min(fewest, f_.cache->slots(l));
    const int drafts = batch_drafts(int(act.size()), f_.drafts, topk, fewest, f_.cp->config.experts);
    if (act.size() < 2 || drafts < 0) {
        serial_steps_ += act.size() > 1;
        for (int i : act) { select_slot(ids[i]); done[i] = !spec_advance(); }
        return;
    }
    ++batched_steps_;
    // every slot drafts and catches up in shared MTP passes
    std::vector<DecodeModel *> ms; std::vector<std::vector<int>> blocks, drs; std::vector<int> d1s;
    for (int i : act) { select_slot(ids[i]); refill_borrowed(); ms.push_back(S().model_.get()); d1s.push_back(S().d1_); }
    select_slot(ids[act[0]]); bind();
    drs = S().model_->mtp_drafts_multi(ms, d1s, drafts);
    std::vector<const std::vector<int> *> hist;  // copy drafts per slot, as in spec_advance (at most the slot's draft count)
    for (auto it = copy_history_.rbegin(); it != copy_history_.rend(); ++it) hist.push_back(&*it);
    for (size_t a = 0; a < act.size(); ++a) {
        auto & sl = slots_[size_t(ids[act[a]])];
        if (sl.ctx_.size() < sl.plen_ + sl.out_.size()) sl.ctx_.insert(sl.ctx_.end(), sl.out_.begin() + long(sl.ctx_.size() - sl.plen_), sl.out_.end());
        auto c = drafts > 0 ? tqllm::flashnext::copy_draft(sl.ctx_, hist, 3, drafts) : std::vector<int>{};
        if (!c.empty() && c[0] == drs[a][0]) { drs[a] = std::move(c); ++sl.copied_; }
    }
    for (size_t a = 0; a < act.size(); ++a) {
        std::vector<int> block{slots_[size_t(ids[act[a]])].out_.back()}; block.insert(block.end(), drs[a].begin(), drs[a].end());
        blocks.push_back(block);
    }
    select_slot(ids[act[0]]); bind();
    f_.cp->rt.block_seqs = int(act.size());
    const auto rows_d = S().model_->step_multi(ms, blocks);
    f_.cp->rt.block_seqs = 1;
    std::vector<std::vector<int>> vs(act.size()); std::vector<int> ms_acc(act.size());
    // greedy slots: every slot's verify rows through the head in one launch and one readback (a readback per slot
    // drained the stream each time)
    bool greedy = true; size_t total = 0;
    for (size_t a = 0; a < act.size(); ++a) { greedy = greedy && slots_[size_t(ids[act[a]])].temp_ <= 0.f; total += blocks[a].size(); }
    const std::vector<int> all = greedy ? f_.cp->head_argmax_rows(rows_d->p, int(total)) : std::vector<int>{};
    size_t off = 0;
    for (size_t a = 0; a < act.size(); ++a) {  // pick and roll back each slot before any MTP catch-up reuses the arena
        select_slot(ids[act[a]]); auto & sl = S();
        const auto v = greedy ? std::vector<int>(all.begin() + long(off), all.begin() + long(off + blocks[a].size()))
                              : pick(rows_d->p + off * size_t(f_.cp->config.hidden), int(blocks[a].size()));
        off += blocks[a].size();
        int m = 1; while (m <= int(drs[a].size()) && v[m - 1] == drs[a][m - 1]) ++m;
        for (int i = 0; i < m && int(sl.out_.size()) < sl.generate_; ++i) { sl.out_.push_back(v[i]); if (is_eos(v[i])) { m = i + 1; break; } }
        sl.model_->accept(m); ++sl.steps_; sl.toks_ += m; sl.offered_ += drs[a].size(); ++sl.taken_[size_t(std::min(m, 16))]; vs[a] = v; ms_acc[a] = m;
    }
    { std::vector<std::pair<size_t, std::vector<int>>> tr;
      for (size_t a = 0; a < act.size(); ++a) tr.push_back({slots_[size_t(ids[act[a]])].out_.size(), std::vector<int>(vs[a].begin(), vs[a].begin() + ms_acc[a])});
      std::vector<int> sid; for (int i : act) sid.push_back(ids[i]);
      trace_step(int(act.size()), drafts, tr, sid); }
    std::vector<int> d1n(act.size());
    {
        std::vector<std::vector<int>> t(act.size()); std::vector<const float *> hp(act.size());
        for (size_t a = 0; a < act.size(); ++a) { t[a].assign(vs[a].begin(), vs[a].begin() + ms_acc[a]); hp[a] = ms[a]->res_keep->p; }
        select_slot(ids[act[0]]); bind();
        d1n = S().model_->mtp_batch_multi(ms, t, hp);
    }
    for (size_t a = 0; a < act.size(); ++a) {
        select_slot(ids[act[a]]); bind(); auto & sl = S();
        sl.d1_ = d1n[a];
        sl.done_ = int(sl.out_.size()) >= sl.generate_ || is_eos(sl.out_.back());
        if (sl.done_ && f_.cp->rt.profile_decode_req) { ck(cudaDeviceSynchronize()); ck(cudaProfilerStop()); f_.cp->rt.profile_decode_req = false; }
        if (sl.done_) remember_copy(sl);
        if (sl.done_) { sl.model_->set_spec(false); ck(cudaDeviceSynchronize());
                        sl.decode_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sl.t_decode_).count()); }
        done[act[a]] = sl.done_;
    }
}

}  // namespace tqllm::flashnext
