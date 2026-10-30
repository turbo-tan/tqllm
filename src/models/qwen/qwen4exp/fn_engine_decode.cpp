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

bool FlashNextEngine::spec_advance() {
    if (S().done_) return false;
    if (S().steps_ == 8 && tqllm::debug_on("mem")) ledger("decode step 8");
    bind();
    refill_borrowed();  // no-op unless an earlier refill ran out of memory
    auto & checkpoint = *f_.cp; const int drafts = f_.drafts;
    const int y = S().out_.back();
    const std::vector<int> dr = S().model_->mtp_drafts(S().d1_, drafts);
    std::vector<int> block{y}; block.insert(block.end(), dr.begin(), dr.end());
    auto rows = S().model_->step(block);
    const auto v = pick(rows->p, int(block.size()));
    int m = 1; while (m <= int(dr.size()) && v[m - 1] == dr[m - 1]) ++m;
    for (int i = 0; i < m && int(S().out_.size()) < S().generate_; ++i) { S().out_.push_back(v[i]); if (is_eos(v[i])) { m = i + 1; break; } }
    S().model_->accept(m); ++S().steps_; S().toks_ += m;
    S().d1_ = S().model_->mtp_batch(std::vector<int>(v.begin(), v.begin() + m), S().model_->res_keep->p);
    S().done_ = int(S().out_.size()) >= S().generate_ || is_eos(S().out_.back());
    if (S().done_ && g_profile_decode_req) { ck(cudaDeviceSynchronize()); ck(cudaProfilerStop()); g_profile_decode_req = false; }
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
    if (S().done_ && f_.hot_victim_mb && tqllm::debug_on("mem")) {
        const auto v=f_.cache->host_victim_stats(), b=S().host_victim_at_decode_;
        std::fprintf(stderr,"HOT_VICTIM bytes=%zu hits=%llu source_loads=%llu spills=%llu d2h_bytes=%llu\n",v.bytes,
            (unsigned long long)(v.hits-b.hits),(unsigned long long)(v.source_loads-b.source_loads),
            (unsigned long long)(v.spills-b.spills),(unsigned long long)(v.d2h_bytes-b.d2h_bytes));
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
    const int drafts = f_.drafts, topk = f_.cp->config.topk;
    std::vector<int> act;
    for (int i = 0; i < n; ++i) { done[i] = slots_[size_t(ids[i])].done_; if (!done[i]) act.push_back(i); }
    const int rows = int(act.size()) * (drafts + 1);
    if (act.size() < 2 || rows > 64 || (f_.cache->slots() < f_.cp->config.experts && rows * topk > f_.cache->slots())) {
        for (int i : act) { select_slot(ids[i]); done[i] = !spec_advance(); }
        return;
    }
    // every slot drafts and catches up in shared MTP passes
    std::vector<DecodeModel *> ms; std::vector<std::vector<int>> blocks, drs; std::vector<int> d1s;
    for (int i : act) { select_slot(ids[i]); refill_borrowed(); ms.push_back(S().model_.get()); d1s.push_back(S().d1_); }
    select_slot(ids[act[0]]); bind();
    drs = S().model_->mtp_drafts_multi(ms, d1s, drafts);
    for (size_t a = 0; a < act.size(); ++a) {
        std::vector<int> block{slots_[size_t(ids[act[a]])].out_.back()}; block.insert(block.end(), drs[a].begin(), drs[a].end());
        blocks.push_back(block);
    }
    select_slot(ids[act[0]]); bind();
    const auto rows_d = S().model_->step_multi(ms, blocks);
    std::vector<std::vector<int>> vs(act.size()); std::vector<int> ms_acc(act.size());
    size_t off = 0;
    for (size_t a = 0; a < act.size(); ++a) {  // pick and roll back each slot before any MTP catch-up reuses the arena
        select_slot(ids[act[a]]); auto & sl = S();
        const auto v = pick(rows_d->p + off * size_t(f_.cp->config.hidden), int(blocks[a].size())); off += blocks[a].size();
        int m = 1; while (m <= drafts && v[m - 1] == drs[a][m - 1]) ++m;
        for (int i = 0; i < m && int(sl.out_.size()) < sl.generate_; ++i) { sl.out_.push_back(v[i]); if (is_eos(v[i])) { m = i + 1; break; } }
        sl.model_->accept(m); ++sl.steps_; sl.toks_ += m; vs[a] = v; ms_acc[a] = m;
    }
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
        if (sl.done_) { sl.model_->set_spec(false); ck(cudaDeviceSynchronize());
                        sl.decode_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sl.t_decode_).count()); }
        done[act[a]] = sl.done_;
    }
}

}  // namespace tqllm::flashnext
