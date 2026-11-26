// Flash-Next request prefill: memory plan, expert bank borrow and refill, chunked prompt, MTP catch-up.
#include "debug.hpp"
#include "fn_engine.hpp"
#include <cmath>
#include <map>
#include "decode_rules.hpp"
#include "prefill_observer_cuda.hpp"

namespace tqllm::flashnext {

void FlashNextEngine::ledger(const char * at) const {
    size_t fr = 0, tot = 0; ck(cudaMemGetInfo(&fr, &tot));
    cudaMemPool_t pool; uint64_t res = 0, used = 0; ck(cudaDeviceGetDefaultMemPool(&pool, 0));
    ck(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &res)); ck(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
    const double M = 1048576.0, trunk = f_.cache->stats().device_bytes / M, mtp = f_.mtp_cache->stats().device_bytes / M,
                 stage = 2.0 * DecodeModel::stage().half_bytes / M, arena = f_.arena.size / M, gdn = tqllm_cuda::fn_gdn_workspace_bytes() / M,
                 inuse = (tot - fr) / M, other = inuse - res / M - trunk - mtp - stage - arena - gdn;
    std::fprintf(stderr, "LEDGER %s: in use %.0f = pool %.0f (used %.0f) + trunk banks %.0f + MTP banks %.0f + staging %.0f + arena %.0f + "
                 "GDN ws %.0f + other %.0f | free %.0f\n", at, inuse, res / M, used / M, trunk, mtp, stage, arena, gdn, other, fr / M);
}

void FlashNextEngine::refill_borrowed() {
    auto & led = S().ledger_; if (!led) return;
    using pclk = std::chrono::steady_clock; const auto r0 = pclk::now();
    const size_t all = led->borrowed_total();
    if (led->state() == tqllm::BankLedger::State::Borrowed) led->buffers_released();
    // a lent layer whose bank was built during the prompt (its decode-path tail: the prompt cache's snapshot kTail before
    // the end) got it from the little memory free mid-prefill; the refill would only load experts into that small bank
    // (RTX 3090, 32K with --prompt-cache on: 11.6 GB of banks and 3.6 GB left free, 83 % hit vs 91 % without). It goes, so
    // the refill rebuilds it at its share of what is free now.
    for (int l : refill_rebuilds(led->borrowed_layers(), [this](int l) { return f_.cache->bank_ready(l); })) f_.cache->release_bank(l);
    // small cards: stop while the warm-up's keep (512 MB) is still free; a refill to the last megabyte left the first
    // decode step's bank upload 88 MiB (12 GB, 16K prompt in 2 x 10240): the rest come back as ordinary misses. With slot
    // tiers each refilled bank is sized to what is free (MixedExpertCache choose_cap keeps 256 MB).
    { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
    const int done = led->refill(f_.cp->run.small_card ? size_t(512) << 20 : 0); cudaGetLastError();
    ck(cudaDeviceSynchronize());
    if (all) std::fprintf(stderr, "tqllm: refilled %d of %zu borrowed expert banks in %.0f ms\n", done, all, std::chrono::duration<double, std::milli>(pclk::now() - r0).count());
    if (all) S().refilled_ = true;
    if (S().model_) S().model_->pf_refill.clear();
    led.reset();
}

void FlashNextEngine::arena_resize(size_t bytes) {
    if (f_.arena.size == bytes || !f_.arena.base) return;
    if (bytes < f_.arena.size && keep_prefill_buffers(f_.cp->run.unified_memory, tqllm::device_free_bytes())) return;  // GB10: the prefill size stays
    ck(cudaDeviceSynchronize()); ck(cudaFree(f_.arena.base)); f_.arena.base = nullptr;
    ck(cudaMalloc(&f_.arena.base, bytes)); f_.arena.size = bytes; f_.arena.used = 0;
}

void FlashNextEngine::take_snapshot(SlotState & s) {
    auto & m = *s.model_; const size_t L = m.rec.size();
    if (s.stash_) {  // prefill_ahead: the restored snapshot stays as alt_ (fresh buffers below)
        s.alt_state_.swap(s.snap_state_); s.alt_conv_.swap(s.snap_conv_); s.alt_ple_.swap(s.snap_ple_);
        s.snap_state_.clear(); s.snap_conv_.clear(); s.snap_ple_.clear(); s.stash_ = false;
    }
    s.snap_state_.resize(L); s.snap_conv_.resize(L); s.snap_ple_.resize(L);
    auto copy = [](std::unique_ptr<Device<float>> & dst, const std::unique_ptr<Device<float>> & src) {
        if (!src) { dst.reset(); return; }
        if (!dst || dst->count != src->count) { PersistentAlloc keep; dst.reset(); dst = std::make_unique<Device<float>>(src->count); }
        ck(cudaMemcpyAsync(dst->p, src->p, src->count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
    };
    for (size_t l = 0; l < L; ++l) { copy(s.snap_state_[l], m.rec[l].state); copy(s.snap_conv_[l], m.rec[l].history); copy(s.snap_ple_[l], m.ple_past[l]); }
}

bool FlashNextEngine::grow_kept(DecodeModel & m, size_t rows) {
    const int cap = int((rows + 1023) / 1024 * 1024), old = m.kv_capacity;
    if (cap <= old) return true;
    auto & cp = *f_.cp; const auto & c = cp.config;
    using clk = std::chrono::steady_clock; const auto t0 = clk::now();
    const size_t row = size_t(cp.attention_kvheads) * cp.attention_dim * sizeof(__nv_bfloat16);
    bool vmm = false; for (const auto & a : m.att) vmm = vmm || (a.k && a.k->vmm);
    // room: the larger state, plus one layer's new buffers while its old ones still live (a plain K/V buffer is copied);
    // what is short comes from the banks, as for a new sequence beside running ones (never below the verify floor)
    {
        auto one = f_.mem_shape; one.attention_layers = 1;
        const size_t grow = tqllm::flashnext::sequence_kv_bytes(f_.mem_shape, cap) - tqllm::flashnext::sequence_kv_bytes(f_.mem_shape, old);
        const size_t layer = vmm ? size_t(cap) / 4 * size_t(one.index_dim) * 4 * 2 : tqllm::flashnext::sequence_kv_bytes(one, cap);
        const size_t need = grow + layer + (size_t(256) << 20);
        { cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
        const size_t fr = tqllm::device_free_bytes();
        if (fr < need) {
            const int floor = std::max(int(slots_.size()) * c.topk, tqllm::flashnext::verify_slot_min(f_.drafts, c.topk));
            const int n = f_.cache->shrink_to_free(need - fr, floor);
            std::fprintf(stderr, "tqllm: prompt cache: the kept K/V needs %zu MB more: %d expert banks gave slots back\n", (need - fr) >> 20, n);
        }
    }
    try {
        ck(cudaDeviceSynchronize());  // the QSA host mirrors' queued copies have landed
        auto grow = [&](AttentionState & a) {
            if (a.k) for (auto * b : {a.k.get(), a.v.get()}) if (!b->grow(size_t(cap) * row, size_t(a.len) * row)) return false;
            if (a.qsa) {  // the indexer keys in a larger indexer: the same cells, blocks and ring tail (import_cells)
                PersistentAlloc keep;
                auto q = std::make_unique<tqllm_cuda::QsaIndexer>(cap, 4, 2048, 4, 128, std::min(cp.rotary, 128), cp.theta, float(cp.epsilon));
                q->keys().import_from(a.qsa->keys(), a.qsa->size(), nullptr);
                ck(cudaStreamSynchronize(nullptr)); a.qsa = std::move(q);
            }
            a.cap = cap;
            return true;
        };
        for (auto & a : m.att) if (!grow(a)) return false;
        if (!grow(m.mtp_att)) return false;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "tqllm: prompt cache: growing the kept K/V failed (%s)\n", e.what());
        cudaDeviceSynchronize(); cudaGetLastError();
        return false;
    }
    m.kv_capacity = cap; m.history.reserve(size_t(cap));
    std::fprintf(stderr, "tqllm: prompt cache: kept K/V grown from %d to %d rows in %.0f ms\n", old, cap,
                 std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    return true;
}

void FlashNextEngine::restore_snapshot(SlotState & s, int p) {
    auto & m = *s.model_;
    m.set_spec(false);
    auto back = [](std::unique_ptr<Device<float>> & dst, const std::unique_ptr<Device<float>> & src) {
        require(bool(dst) == bool(src) && (!src || dst->count == src->count), "prompt cache: snapshot does not match the sequence state");
        if (src) ck(cudaMemcpyAsync(dst->p, src->p, src->count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
    };
    for (size_t l = 0; l < m.rec.size(); ++l) { back(m.rec[l].state, s.snap_state_[l]); back(m.rec[l].history, s.snap_conv_[l]); back(m.ple_past[l], s.snap_ple_[l]); }
    for (auto & a : m.att) if (a.k) a.len = p;  // QSA keys past len are cut at the next append
    if (m.mtp_att.k) m.mtp_att.len = p;          // MTP pairs (token i + 1, residual i) for i < p
    m.history.rollback(size_t(p)); m.position = p;
}

void FlashNextEngine::pc_verify(const std::vector<int> & ids, int p) {
    auto & checkpoint = *f_.cp; ck(cudaDeviceSynchronize());
    auto ref = std::make_unique<DecodeModel>(checkpoint, *f_.fixtures, *f_.cache, size_t(S().model_->kv_capacity));  // one slot: the kept sequence's size, not the slot context
    ref->mtp_experts = f_.mtp_cache.get();
    for (int b = 0; b < p; b += 64) { if (g_arena) g_arena->used = 0; (void) ref->step(std::vector<int>(ids.begin() + b, ids.begin() + std::min(p, b + 64))); }
    ck(cudaDeviceSynchronize());
    auto & m = *S().model_;
    auto host = [](const float * d, size_t n) { std::vector<float> h(n); ck(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost)); return h; };
    auto host16 = [](const __nv_bfloat16 * d, size_t n) { std::vector<__nv_bfloat16> b(n); ck(cudaMemcpy(b.data(), d, n * 2, cudaMemcpyDeviceToHost));
                                                          std::vector<float> h(n); for (size_t i = 0; i < n; ++i) h[i] = __bfloat162float(b[i]); return h; };
    auto rel = [](const std::vector<float> & a, const std::vector<float> & b) { double d = 0, r = 0;
        for (size_t i = 0; i < a.size(); ++i) { d += double(a[i] - b[i]) * (a[i] - b[i]); r += double(b[i]) * b[i]; } return std::sqrt(d / std::max(r, 1e-30)); };
    double worst = 0; std::string where = "-";
    auto note = [&](const std::string & what, double e) { if (e > 2e-2) std::fprintf(stderr, "PCV %s rel %.3g\n", what.c_str(), e); if (e > worst) { worst = e; where = what; } };
    const size_t row = size_t(checkpoint.attention_kvheads) * checkpoint.attention_dim;
    for (size_t l = 0; l < m.rec.size(); ++l) {
        const std::string L = "layer " + std::to_string(l);
        auto & a = m.rec[l]; auto & b = ref->rec[l];
        if (a.state && b.state) note(L + " gdn state", rel(host(a.state->p, a.state->count), host(b.state->p, b.state->count)));
        if (a.history && b.history) note(L + " gdn conv", rel(host(a.history->p, a.history->count), host(b.history->p, b.history->count)));
        if (bool(a.state) != bool(b.state)) note(L + " gdn state presence", 1);
        if (m.ple_past[l] && ref->ple_past[l]) note(L + " ple past", rel(host(m.ple_past[l]->p, m.ple_past[l]->count), host(ref->ple_past[l]->p, ref->ple_past[l]->count)));
        if (l < m.att.size() && m.att[l].k && ref->att[l].k) {
            note(L + " k", rel(host16(m.att[l].k->p, size_t(p) * row), host16(ref->att[l].k->p, size_t(p) * row)));
            note(L + " v", rel(host16(m.att[l].v->p, size_t(p) * row), host16(ref->att[l].v->p, size_t(p) * row)));
            if (m.att[l].len != ref->att[l].len) note(L + " len " + std::to_string(m.att[l].len) + " vs " + std::to_string(ref->att[l].len), 1);
            if (m.att[l].qsa && ref->att[l].qsa && m.att[l].qsa->size() < ref->att[l].qsa->size()) note(L + " qsa size", 1);
        }
    }
    if (m.position != ref->position || m.history.size() != ref->history.size())
        note("position " + std::to_string(m.position) + " / " + std::to_string(ref->position) + ", history " + std::to_string(m.history.size()) + " / " + std::to_string(ref->history.size()), 1);
    std::fprintf(stderr, "PCV restored state at %d vs a fresh prefill: worst rel %.3g (%s)\n", p, worst, where.c_str());
    ref.reset(); bind();
}

void FlashNextEngine::abort_prefill() {
    const auto lk = ep_lock();
    try {
        (void) cudaDeviceSynchronize(); (void) cudaGetLastError();
        if (auto & m = S().model_) { m->release_prefill(); m->pf_borrowed.clear(); m->prompt_emb.clear(); m->keep_embeddings = false; }
        arena_resize(f_.arena_decode);
        refill_borrowed();
    } catch (const std::exception & e) { std::fprintf(stderr, "tqllm: cleanup after a failed prefill: %s\n", e.what()); }
    f_.cp->rt.trim_below = SIZE_MAX;
}

void FlashNextEngine::spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int) {
    if (auto * e = ep_helper()) e->request_begin();  // before the lock: rank 0's forwards already sent finish first
    const auto lk = ep_lock(); S().serving_ = true;
    if (sharing()) park_swap(prompt);
    static std::map<int, std::vector<int>> bank_keep;  // lent trunk banks: their resident experts before the prompt
    auto & checkpoint = *f_.cp; const auto & c = checkpoint.config;
    require(!prompt.empty() && prompt.size() + count + 16 <= slot_context(), "prompt plus max_tokens exceeds the slot context");
    // a chat prompt (turn chunks) resumes only at a kept message start, never from the last reply's own state (live_)
    const bool chat = turns() && turn_prompt(prompt, f_.cp->run.im_start);
    int P0 = !caching() ? 0 : chat ? (S().model_ ? reusable_prefix(S().seq_, S().snap_, prompt) : 0) : reuse(S(), prompt);  // prompt positions [0, P0) come from this slot's last request
    // the state before an idle prefill (prefill_ahead): the client sent the answer back otherwise than expected
    if (chat && S().model_ && !P0 && (P0 = reusable_prefix(S().alt_seq_, S().alt_snap_, prompt))) S().swap_alt();
    if (!S().stash_) S().drop_alt();
    // a kept sequence has the capacity of the request that built it (cached_sequence_rows / multi_sequence_rows): a request
    // it does not hold grows it (kept_sequence_rows); only when the memory cannot hold that is the prompt read afresh
    if (P0 && !kept_sequence_fits(prompt.size(), count, size_t(S().model_->kv_capacity))) {
        const size_t rows = kept_sequence_rows(prompt.size(), count, size_t(S().model_->kv_capacity), size_t(slot_context()));
        int ok = rows && grow_kept(*S().model_, rows) ? 1 : 0;
        if (f_.cp->run.split) DecodeModel::split_min(*f_.cp->run.split, &ok);  // --split layer: every rank keeps the prefix, or none
        if (!ok) { std::fprintf(stderr, "tqllm: prompt cache: the kept K/V cannot grow to this request: the prompt is read afresh\n"); P0 = 0; }
    }
    const size_t seq_rows = slots_.size() > 1 ? tqllm::flashnext::multi_sequence_rows(caching(), prompt.size(), count, size_t(slot_context()))
                          : P0 ? size_t(S().model_->kv_capacity)
                          : caching() ? cached_sequence_rows(prompt.size(), count, size_t(slot_context())) : prompt.size() + count + 16;
    const bool live = !chat && P0 && P0 == S().live_;  // the last reply's own state: no restore
    S().snap_ = S().live_ = 0;  // set again once this prefill has completed
    if (!P0) { S().model_.reset(); if (caching()) S().conv_ = ++conv_clock_; }  // a new conversation in this slot
    S().prompt_res_.reset();
    // banks a long prompt left below their slot target grow back once its state is gone, unless this request's own state
    // needs the room (regrowing then would throw the banks' experts away and rebuild them at the same smaller size)
    // with TQFN_KV_HOST the K/V older than the kept window lives in host memory after the prompt, so only the window's
    // state keeps device room (a long prefill borrows banks for the rest and gives them back)
    static const bool kv_host = kv_host_enabled();
    static const size_t kv_keep = std::getenv("TQFN_KV_HOST_KEEP") ? size_t(std::atoi(std::getenv("TQFN_KV_HOST_KEEP"))) : 32768;
    const size_t state_rows = kv_host ? std::min(seq_rows, 2 * kv_keep) : seq_rows;
    if (slots_.size() == 1 && !P0) {
        // what the last request's state left goes first, so regrow reads the device's real free memory: the pool's cached
        // blocks, and the host K/V gathers (device row caches a long prompt's kv_to_host made, ~0.9 GB after 128K; nothing
        // reads host K/V now, and kv_to_host makes them again before a refill)
        ck(cudaDeviceSynchronize());
        for (auto & g : host_gathers()) g.reset();
        if (f_.tier && f_.tier->chunks()) {  // a long prompt's host buffers take the elastic chunks' RAM; decode gives it back
            const uint64_t prompt_host = tqllm::flashnext::prompt_host_bytes(uint64_t(prompt.size() + count), f_.tier_rows_per_token,
                                                                            f_.tier_kv_per_token, kv_host ? kv_keep : 0);
            // the prefill stream ring grows on first use: what it still lacks is kept too
            const uint64_t ring_left = 2 * (f_.tier_ring_half - std::min<uint64_t>(f_.tier_ring_half, DecodeModel::stage().ring_bytes));
            f_.fit_tier(tqllm::flashnext::host_tier_keep(prompt.size() > size_t(16384), prompt_host) + ring_left);
        }
        { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
        const auto g0 = std::chrono::steady_clock::now();
        const int grown = f_.cache->regrow(regrow_keep_bytes(size_t(768) << 20, f_.state_full, f_.state_tokens, state_rows));
        if (grown && tqllm::debug_on("mem"))
            std::fprintf(stderr, "tqllm: %d expert banks grew back to their target in %.0f ms\n", grown,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count());
    }
    if (slots_.size() > 1) {
        // several sequences: the request's K/V (prompt + max_tokens) and what the running requests will still grow into
        // must fit beside the reserve; if not, the largest banks give slots back now, between steps (never below np x
        // topk, the batching floor), instead of a running request failing for memory later (3090 np 12 x 8K, 11-05)
        // a new sequence allocates seq_rows (with the prompt cache more than prompt + max_tokens; 3090 np 4 11-18: counting
        // only prompt + max_tokens admitted requests whose K/V then failed to map, HTTP 500)
        const size_t own = S().model_ ? std::min(prompt.size() + count + 16, size_t(slot_context())) : seq_rows;
        size_t grow = 0;
        for (const auto & sl : slots_) {
            if (&sl == &S() || sl.done_ || !sl.model_) continue;
            const size_t total = sl.plen_ + size_t(sl.generate_) + 16, at = size_t(std::max(0, sl.model_->position));
            if (total > at) grow += tqllm::flashnext::sequence_kv_bytes(f_.mem_shape, int(total)) - tqllm::flashnext::sequence_kv_bytes(f_.mem_shape, int(at));
        }
        // a slot that starts a new sequence allocates its PLE histories / GDN state on the first forward (11-09)
        const size_t fixed = S().model_ ? 0 : f_.seq_fixed;
        { cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
        const size_t fr = tqllm::device_free_bytes();
        if (const size_t give = tqllm::flashnext::request_shrink_bytes(f_.mem_shape, own, fr > grow ? fr - grow : 0, size_t(256) << 20, fixed)) {
            const int n = f_.cache->shrink_to_free(give, int(slots_.size()) * c.topk);
            std::fprintf(stderr, "tqllm: a request needs %zu MB more for its K/V: %d expert banks gave slots back\n", give >> 20, n);
        }
        // still short after the banks gave what they could (small cards: 12 / 16 GB, np 4, 11-08: the step then failed for
        // memory and every running request answered HTTP 500): refuse it now, before any of its state exists, so the server
        // queues it until a running request finishes ("out of memory" -> tqserve's admission requeue, admission.hpp)
        bool others = false;
        for (const auto & sl : slots_) others = others || (&sl != &S() && !sl.done_ && sl.model_);
        const size_t fr2 = tqllm::device_free_bytes();
        if (others && tqllm::flashnext::request_shrink_bytes(f_.mem_shape, own, fr2 > grow ? fr2 - grow : 0, size_t(256) << 20, fixed))
            throw std::runtime_error("out of memory: this request's K/V does not fit beside the running requests; it waits for one to finish");
    }
    // only once admitted: a refused request must not reset what the running ones share (their K/V row caches, the trim mark)
    f_.cp->rt.trim_below = SIZE_MAX; ++f_.cp->rt.kv_epoch;
    if (slots_.size() == 1 && f_.kv_capacity_full) {  // one sequence: its attention state is sized by the request, not the slot context
        const int need = int(std::min<size_t>(size_t(f_.kv_capacity_full), (seq_rows + 1023) / 1024 * 1024));
        if (need != f_.cp->rt.kv_capacity) { if (f_.cp->rt.qsa_batch) { ck(cudaDeviceSynchronize()); f_.cp->rt.qsa_batch.reset(); } f_.cp->rt.kv_capacity = need; }
    }
    if (g_arena) g_arena->used = 0;
    // the pool keeps what the last request freed (release threshold max); hand it back so plain cudaMalloc users (expert
    // banks, staging, attention workspaces) can have it
    // a failed earlier request may have left a non-sticky CUDA error pending (QSA / CUB would report it on every later
    // request); a sticky one means the context is lost
    if (const cudaError_t e = tqllm::clear_stale_cuda_error(); e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA context lost by an earlier request (") + cudaGetErrorString(e) + "); restart tqserve");
    { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
    // GB10: prefill buffers an earlier request kept go back once the margin is gone (decode and the slots' states grew after
    // that request's check; a .44-shaped np 4 x 131K brain sat at 16.3 GB free with them kept)
    if (f_.cp->run.unified_memory && !keep_prefill_buffers(true, tqllm::device_free_bytes())) {
        auto & st = DecodeModel::stage();
        if (f_.cp->rt.dhc.base || st.half[0] || st.half[1] || f_.arena.size > f_.arena_decode) {
            ck(cudaDeviceSynchronize()); f_.cp->rt.dhc.release();
            for (auto & hp : st.half) if (hp) { ck(cudaFree(hp)); hp = nullptr; } st.half_bytes = 0; st.K = 0;
            f_.cp->rt.qsa_batch.reset(); tqllm_cuda::fn_gdn_release_workspaces();
            arena_resize(f_.arena_decode);
            cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0));
        }
    }
    if (tqllm::debug_on("mem")) { std::fprintf(stderr, "LEDGER request of %zu prompt tokens\n", prompt.size()); ledger("request start"); }
    if (live) S().model_->set_spec(false);
    else if (P0) restore_snapshot(S(), P0);  // snap_ is already cleared
    else S().model_ = std::make_unique<DecodeModel>(checkpoint, *f_.fixtures, *f_.cache, seq_rows);
    S().model_->mtp_experts = f_.mtp_cache.get();
    bind();
    const std::vector<int> ids(prompt.begin(), prompt.end());
    if (P0) std::fprintf(stderr, "tqllm: prompt cache: %d of %zu prompt tokens reused\n", P0, prompt.size());
    if (P0 && tqllm::debug_on("pc_verify")) pc_verify(ids, P0);
    // TQLLM_DEBUG=profile_req:N: cudaProfilerStart at the N-th request's prefill, stop after it (nsys --capture-range=cudaProfilerApi)
    static const int prof_req = tqllm::debug_cstr("profile_req") ? std::atoi(tqllm::debug_cstr("profile_req")) : 0; static int req_no = 0;
    // TQLLM_DEBUG=profile_decode:1: the capture covers that request's decode instead of its prefill
    static const bool prof_decode = tqllm::debug_cstr("profile_decode") && std::atoi(tqllm::debug_cstr("profile_decode")) != 0;
    const bool profiling = prof_req > 0 && ++req_no == prof_req && !prof_decode;
    f_.cp->rt.profile_decode_req = prof_req > 0 && req_no == prof_req && prof_decode;
    if (profiling) ck(cudaProfilerStart());
    S().t_begin_ = std::chrono::steady_clock::now(); S().prefill_us_ = S().decode_us_ = 0;
    static const bool req_log = tqllm::debug_cstr("profile_req") != nullptr;  // TQLLM_DEBUG=profile_req: host timeline of the request
    const auto stamp = [&](const char * what) { if (req_log) std::fprintf(stderr, "tqllm: req +%.3f s %s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - S().t_begin_).count(), what); };
    S().ctx_ = ids; S().plen_ = ids.size(); S().copied_ = 0;
    S().out_.clear(); S().out_view_.clear(); S().generate_ = int(count); S().steps_ = S().toks_ = S().offered_ = 0; S().taken_ = {}; S().refilled_ = false; S().pick_ = {};
    // Trunk prefill first, then the MTP catch-up, so their working sets do not add up: a prompt longer than one chunk
    // keeps its residuals in pinned host memory, drops the MTP bank before the trunk and the layer-major buffers
    // after it. The catch-up runs the same batches in the same order as when it followed each chunk.
    const int P = int(ids.size()), T = P - P0; const int W = S().model_->width;
    // turn chunks (chat prompts with the prompt cache, decode_rules.hpp): chunks restart at every message start from
    // kTurnMin, a chunk of kTurnShort tokens or more runs layer-major and a shorter one as kTurnStep-row decode steps, every
    // piece writes its MTP keys / values itself, and the state is kept at the last message start, so a prompt's reply is
    // the same cold and resumed. A chat prompt with no message start past kTurnMin is read as before and keeps nothing.
    const std::vector<int> starts = chat ? turn_starts(ids, f_.cp->run.im_start) : std::vector<int>{};
    const bool turn = chat && (!starts.empty() || P0 > 0);
    const bool turn_lm = turn && turn_has_long(starts, P0, P);  // a layer-major chunk: the prefill plans (else decode steps only)
    S().chat_ = chat;
    // the snapshot for the next request: the last message start (chat), or a chunk boundary kTail before the prompt's end
    int snap = chat ? (turn ? turn_kept_at(starts, P0) : 0) : caching() ? snapshot_at(P, P0, kTail) : 0;
    int keep_snap = P0;  // no new snapshot: the restored one still holds P0
    const bool snap_at_start = live && !snap;
    if (snap_at_start) take_snapshot(S());  // a short next message: the same prompt again restores here
    int grid = 0;  // turn: the chunk grid inside a message (a constant of the server, set with the plan)
    auto chunk_end = [&](int b, int ch) {
        if (turn) return turn_piece_end(starts, b, P, grid > 0 ? std::min(grid, ch) : ch);
        const int e = std::min(P, b + ch); return snap > b && snap < e ? snap : e; };
    int outer = 0, outer_group = 0;  // leading chunks run layer-outer (step_outer) in groups: experts read from the GGUF on demand
    std::vector<std::pair<int, int>> spans;
    stamp("model ready");
    // the prefill chunk for this request: the engine's chunk, halved while the plan cannot fit even with every bank
    // released (a 128K prompt at 104 slots does not fit 16K chunks on a 24 GB card)
    // Small cards: the chunk starts at the whole prompt (every chunk streams every expert once, and at 12 GB the link is
    // the bottleneck: 5 x 5,120 streams 327 GB in the 15 s a 3090 takes to compute 25K tokens) and the plan below shrinks
    // it until it fits, with the LM heads parked. 24 GB keeps the engine chunk: there the streams already hide under the
    // kernels (nsys 10-19: 25K in 2 x 16K 90.7 % busy, in one chunk 86.3 % and 0.8 s slower). A prompt shorter than the
    // engine chunk gets its own length: the same experts stream either way, but the activation and staging buffers of a
    // 16K chunk cost 6.5 GB there, and giving up banks for them only adds the refill (11-11: 1.2K prompt 2.38 -> 2.16 s).
    constexpr int kMaxChunk = 32768;
    // a DFlash2 drafter captures every chunk row's features (5 x hidden BF16 each): the chunk is capped at its buffer
    const int chunk_cap = f_.dfl_feat ? int(f_.dfl_feat_rows) : kMaxChunk;
    // a turn's short new part (a follow-up message) plans its own rows (turn_min_chunk)
    const int min_chunk = turn ? turn_min_chunk(T) : 2048;
    const int chunk0 = turn && T < 2048 ? min_chunk
                     : std::min(f_.cp->run.tune.chunk_set || !f_.cp->run.small_card ? f_.chunk : tqllm::flashnext::small_card_chunk(T, kMaxChunk), chunk_cap);
    int chunk = chunk0, block = 0; size_t arena_req = f_.arena_prefill; size_t heads_parked = 0;  // the arena a chunk needs: the start-up size, or less once the chunk has halved
    struct UnparkGuard { Checkpoint & cp; size_t & n; ~UnparkGuard() { if (n) { try { cp.unpark_heads(); } catch (...) {} n = 0; } } } unpark_guard{checkpoint, heads_parked};
    std::unique_ptr<Device<float>> last_row;  // heads parked: the last chunk's head row, picked once they are back
    if ((turn ? turn_lm : T >= 256) && chunk >= 256) {
        const auto & c = checkpoint.config;
        tqllm::PrefillShape sh; sh.hidden = c.hidden; sh.hc = c.hc; sh.expert_ff = c.expert_ff; sh.topk = c.topk; sh.experts = c.experts; sh.layers = c.layers;
        sh.wide_bf16 = S().model_->prefill_wide_bf16(std::min(checkpoint.run.tune.sub, chunk));
        // the staging halves are sized for the largest expert block of any layer (prefill_layer): mixed-format files have
        // bigger blocks in some layers (the first layer's block undercounted staging by 170 MB at 245K)
        for (int l = 0; l < c.layers; ++l) if (f_.cache->has_layout(l)) {
            const auto lay = f_.cache->bank_layout(l); size_t blk = 0;
            for (int i = 0; i < 6; ++i) blk += lay.piece[i];
            sh.expert_block_bytes = std::max(sh.expert_block_bytes, std::max(blk, size_t(lay.block)) + 6 * 256 / std::max(1, f_.cp->run.tune.stage));
        }
        f_.cp->rt.prefill_stage = tqllm::flashnext::small_card_stage(f_.cp->run.small_card, f_.cp->run.tune.stage_set,
                                                                  f_.cp->run.tune.stage, T, f_.chunk);
        tqllm::PrefillTuning tu; tu.sub = f_.cp->run.tune.sub; tu.stage = f_.cp->rt.prefill_stage;
        tu.dense_cache_bytes = dense_half_pool_bytes(f_.cp->run.tune); tu.mtp_bank_bytes = f_.mtp_cache->peak_device_bytes();
        tu.workspace_bytes = size_t(1024 + std::max(0L, f_.cp->run.tune.reserve_mb)) << 20;  // --prefill-reserve-mb: headroom for the kernels' own scratch
        // above 128K keep a small margin beyond the plan's own terms (allocator granularity, kernel scratch)
        if (T > 131072) tu.workspace_bytes += size_t(448) << 20;
        int attn = 0; for (int l = 0; l < c.layers; ++l) if (checkpoint.has_tensor("blk." + std::to_string(l) + ".indexer.k_proj.weight")) ++attn;
        const int seq_cap = S().model_ && S().model_->kv_capacity ? S().model_->kv_capacity : f_.cp->rt.kv_capacity;
        int plan_cap = seq_cap, plan_T = T;  // the context and prompt the plan sizes for (turn grid: the slot context, below)
        auto context_term = [&](int block) { return tqllm::prefill_context_bytes(plan_cap, block, attn, checkpoint.attention_kvheads, checkpoint.attention_dim, c.index_dim); };
        // the plan counts the QSA selection scratch in context_bytes: release the old one so the free memory it reads is right
        if (f_.cp->rt.qsa_batch) { ck(cudaDeviceSynchronize()); f_.cp->rt.qsa_batch.reset(); cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
        auto free_now = [] { return tqllm::device_free_bytes(); };
        std::vector<size_t> bank_bytes; size_t banks_total = 0;
        for (int l : f_.cache->banks_lru()) { bank_bytes.push_back(f_.cache->bank_bytes(l)); banks_total += bank_bytes.back(); }
        const int block0 = std::clamp(f_.cp->run.tune.sub, 512, 2048);
        // the plan's room is fixed for the process (plan_room: the free + bank bytes measured at start, pool trimmed,
        // 256 MB steps), so a prompt length always plans the same chunk and block: with the free memory of the moment the
        // same 25.3K prompt took one chunk of 25600 or two of 21504 (+-1 s of TTFT, 11-12), and the tokens follow the
        // chunk boundary. The pool's cached blocks are handed back first so the borrow works on the device's real free memory.
        { cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
        const size_t mtp_bank = f_.mtp_bank_bytes;  // the MTP bank, snapshotted at start (a multi-chunk prompt drops it)
        auto room_for = [&](size_t mtp, size_t heads) {
            return tqllm::plan_room(f_.plan_budget, free_now(), banks_total, f_.cache->device_staging_bytes(), mtp, heads);
        };
        auto tune_for = [&](int chunk, int block) {
            tqllm::PrefillTuning t = tu;
            t.chunk = chunk; t.sub = std::min(f_.cp->run.tune.sub, block); t.context_bytes = context_term(block);
            // the QSA score scratch holds one block of queries: a prompt shorter than the block has only its own (sizing only)
            t.arena_bytes = tqllm::prefill_arena_req(f_.arena_prefill, chunk, std::min(block, plan_T), checkpoint.attention_heads);
            return t;
        };
        const tqllm::PlanRoom base_room = room_for(0, 0);
        if (f_.plan_budget && !base_room.from_budget)  // should not happen: the banks can always give the budget back
            std::fprintf(stderr, "tqllm: prefill plan: %.0f MB free + %.0f MB of banks + %.0f MB staging is below the %.0f MB fixed plan room; this request plans with what is there, its chunk may differ\n",
                         free_now() / 1048576.0, banks_total / 1048576.0, f_.cache->device_staging_bytes() / 1048576.0, f_.plan_budget / 1048576.0);
        tqllm::PrefillChoice choice = tqllm::prefill_choice(
            sh, tune_for, T, base_room.room, mtp_bank, f_.cp->run.small_card ? checkpoint.heads_bytes() : 0, chunk0, block0, min_chunk);
        // the LM heads (needed only for the first token) in pinned host memory, when their room makes the whole prompt one
        // chunk (a >32K prompt): the pick then hangs on a host copy, so the trunk reads it back after its buffers are gone.
        // A one-chunk prompt picks in place, so parking there buys nothing and is not done (11-12: it was parked and
        // unparked again for 1.1 s to plan the same two chunks).
        if (choice.park_heads && T > choice.chunk && (heads_parked = checkpoint.park_heads())) {
            cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0));
        } else if (choice.park_heads) {
            choice = tqllm::prefill_choice(sh, tune_for, T, base_room.room, mtp_bank, 0, chunk0, block0, min_chunk);
        }
        chunk = choice.chunk; block = choice.block;
        if (turn) {
            // the grid and the selection block of every turn prompt are the ones a full-context prompt plans on the fixed
            // plan room: a constant of the server, so a turn's chunks and block do not depend on the prompt's length and
            // a conversation's earlier turns were read as a cold read of its latest prompt reads them
            const size_t fixed_room = f_.plan_budget ? f_.plan_budget : base_room.room;
            const int whole_T = int(slot_context()) - 16;
            const int grid0 = std::min(f_.cp->run.tune.chunk_set || !f_.cp->run.small_card ? f_.chunk : tqllm::flashnext::small_card_chunk(whole_T, kMaxChunk), chunk_cap);
            tqllm::PrefillShape sh_whole = sh; sh_whole.wide_bf16 = S().model_->prefill_wide_bf16(std::min(checkpoint.run.tune.sub, grid0));
            const tqllm::PrefillTuning tu_now = tu;
            tu.stage = tqllm::flashnext::small_card_stage(f_.cp->run.small_card, f_.cp->run.tune.stage_set, f_.cp->run.tune.stage, whole_T, f_.chunk);
            tu.workspace_bytes = size_t(1024 + std::max(0L, f_.cp->run.tune.reserve_mb)) << 20; if (whole_T > 131072) tu.workspace_bytes += size_t(448) << 20;
            if (mtp_bank) tu.mtp_bank_bytes = mtp_bank;  // the start-up snapshot, not the bank's peak of the moment
            plan_cap = int(slot_context()); plan_T = whole_T;
            const auto whole = tqllm::prefill_choice(sh_whole, tune_for, whole_T, fixed_room, mtp_bank, 0, grid0, block0);
            plan_cap = seq_cap; plan_T = T; tu = tu_now;
            grid = whole.chunk; block = whole.block;
            // the plan's chunk below the grid only matters for a message longer than that chunk
            bool cut = false;
            for (int b = P0, e; b < P && !cut; b = e) { e = turn_chunk_end(starts, b, P, grid); cut = e - b > chunk; }
            static int told_grid = 0, told_block = 0;  // printed when it is first set (or would ever change) and when a message is cut
            if (grid != told_grid || block != told_block || cut || tqllm::debug_on("fn_prefill_log"))
                std::fprintf(stderr, "tqllm: turn chunks: grid %d, selection block %d (this prompt: %d of %d tokens, %zu message starts, chunk %d)%s\n", grid, block, T, P, starts.size(), chunk,
                             cut ? ": a message is longer than the plan's chunk, this read can differ from a cold one" : "");
            told_grid = grid; told_block = block;
            chunk = std::min(chunk, grid);
        }
        if (f_.cp->run.split) { int cb[2] = {chunk, block}; DecodeModel::split_agree(*f_.cp->run.split, cb, 2); chunk = cb[0]; block = cb[1]; }
        tu = tune_for(chunk, block); arena_req = tu.arena_bytes;
        tqllm::PrefillPlan plan;
        {   // a multi-chunk prompt clears the MTP bank before its trunk, so its room comes back to the plan
            const size_t mtp_room = T > chunk ? (mtp_bank ? mtp_bank : f_.mtp_cache->stats().device_bytes) : 0;
            const size_t room = room_for(mtp_room, heads_parked).room;
            plan = tqllm::prefill_plan(sh, tu, T, std::max(room, banks_total) - banks_total);
            if (tqllm::debug_on("mem") || tqllm::debug_on("fn_prefill_log"))
                std::fprintf(stderr, "tqllm: prefill plan: %d tokens -> chunk %d, block %d%s: needs %.0f MB (room %.0f MB%s + mtp %.0f, banks %.0f MB, %.0f MB free)\n",
                             T, chunk, block, heads_parked ? ", heads parked" : "", plan.need / 1048576.0, base_room.room / 1048576.0,
                             base_room.from_budget ? "" : " measured", mtp_room / 1048576.0, banks_total / 1048576.0, free_now() / 1048576.0);
        }
        // experts read from the GGUF on demand: the chunks up to the snapshot run layer-outer when the device holds a whole
        // layer's streamed experts and every chunk's rows (the chunk and block above stay as chosen: the text follows them)
        if (f_.cache->host_tier() && !f_.dfl_feat && !f_.cp->run.split && slots_.size() == 1 && T > chunk) {
            const int rows = turn ? chunk : reused_read_rows(live, P0 ? S().snap_rows_ : 0, T, chunk);
            for (int b = P0, e = 0; b < P; b = e) { e = chunk_end(b, rows); spans.push_back({b, e}); }
            if (const int n = tqllm::flashnext::layer_outer_chunks(spans, snap)) {
                // the largest group of chunks whose rows (BF16 or fp32 wide rows and PLE rows) fit beside the plan, with a
                // whole layer staged on the device when that fits (one upload per layer), else the engine's stage groups
                // uploaded again for each chunk from the pinned ring (the drive still read once per layer and group);
                // the groups run one after another
                const size_t mtp_room = mtp_bank ? mtp_bank : f_.mtp_cache->stats().device_bytes;
                const size_t room = room_for(mtp_room, heads_parked).room, free_b = std::max(room, banks_total) - banks_total;
                const size_t per_token = size_t(W) * (sh.wide_bf16 ? 2 : 4) + size_t(c.hidden) * 4;
                tqllm::PrefillPlan p2; int g = 0, k2 = 0;
                for (const int k : {c.experts, tu.stage}) {
                    tqllm::PrefillTuning t2 = tu; t2.stage = k; p2 = tqllm::prefill_plan(sh, t2, T, free_b);
                    for (g = n; g >= 2; --g) if (p2.need + size_t(spans[size_t(g) - 1].second - spans[0].first) * per_token <= room) break;
                    if (g >= 2) { k2 = k; break; }
                }
                if (g >= 2) {
                    outer = n; outer_group = g; plan = p2; f_.cp->rt.prefill_stage = k2;
                    plan.need += size_t(spans[size_t(g) - 1].second - spans[0].first) * per_token;
                    plan.release = plan.need > free_b ? plan.need - free_b : 0;
                    plan.host_copies = plan.host_copies / size_t(std::max(1, plan.chunks)) * size_t((n + g - 1) / g + plan.chunks - n);
                }
                std::fprintf(stderr, "tqllm: prefill: %d chunks layer-outer%s (groups of %d, %d experts staged at once: plan %.0f MB + %.0f MB of rows per chunk, room %.0f MB)\n",
                             outer, outer ? "" : " does not fit, chunk by chunk", g, k2, p2.need / 1048576.0,
                             double(spans[0].second - spans[0].first) * per_token / 1048576.0, room / 1048576.0);
            }
        }
        // small cards release every bank for a multi-chunk prompt: a kept bank grows by its misses during the prompt
        // (slots=auto banks start below their cap) and OOMed at "expert bank upload ... 54 MiB free"
        const bool borrow_all = f_.cp->run.small_card && T > chunk && f_.cp->run.tune.borrow == tqllm::BorrowMode::Auto;
        f_.cp->rt.qsa_block = block; f_.cp->rt.sub_req = block;
        // the first chunk's host prep (embedding + PLE gathers, ~0.4 s at 16K) runs while the banks, KV and arena are set up
        { DecodeModel * m = S().model_.get(); std::vector<int> first(ids.begin() + P0, ids.begin() + chunk_end(P0, chunk));
          m->prep_next_ = std::async(std::launch::async, [m, first] { return m->prepare_host(first); }); }
        if (T > chunk) f_.mtp_cache->clear();  // a multi-chunk prompt needs the room; one chunk keeps the MTP bank
        tqllm::BankLedger::Ops ops;
        ops.lru_layers = [this] { return f_.cache->banks_lru(); };
        ops.bank_bytes = [this](int l) { return f_.cache->bank_bytes(l); };
        // A lent bank comes back with the experts it held before the prompt (the decode-warmed set): the answer routes
        // like the earlier answers, not like the prompt. A bank that held nothing takes the prompt's most-routed experts.
        bank_keep.clear();
        ops.release = [this](int l) {
            auto & keep = bank_keep[l];
            for (int e = 0; e < f_.cache->experts(); ++e) if (f_.cache->resident(l, e)) keep.push_back(e);
            f_.cache->release_bank(l); };
        ops.refill = [this](int l) {
            const auto k = bank_keep.find(l);
            std::vector<int> top;
            if (k != bank_keep.end() && !k->second.empty()) top = k->second;
            else { auto & m = *S().model_; const auto it = m.pf_refill.find(l); if (it == m.pf_refill.end()) return; top = it->second; }
            const int cap = f_.cache->plan_slots(l);
            top.resize(refill_load_count(std::min(top.size(), size_t(cap)), cap, f_.cache->experts())); f_.cache->prepare(l, top); };
        ops.free_bytes = free_now;
        S().ledger_ = std::make_unique<tqllm::BankLedger>(std::move(ops));
        std::vector<int> bankless; for (int l = 0; l < c.layers; ++l) if (!f_.cache->bank_ready(l) && f_.cache->has_layout(l)) bankless.push_back(l);
        const auto ev = tqllm::borrow_banks(*S().ledger_, plan, borrow_all ? tqllm::BorrowMode::All : f_.cp->run.tune.borrow, f_.cp->run.tune.borrow_count, bankless); S().plan_catchup_bytes_ = plan.catchup_peak;
        f_.cp->rt.trim_below = S().ledger_->borrowed_bytes() + f_.mtp_cache->peak_device_bytes() + (size_t(512) << 20);
        S().model_->pf_borrowed.assign(size_t(c.layers), 0);
        for (int l : ev) S().model_->pf_borrowed[size_t(l)] = 1;
        if (!ev.empty() || tqllm::debug_on("mem"))
            std::fprintf(stderr, "tqllm: prefill of %d tokens in %d chunks of %d%s: needs %.0f MB (chunk %.0f, catch-up %.0f), released %zu expert banks (%.0f MB), %.0f MB free, %.1f GB of experts to stream\n",
                         T, plan.chunks, chunk, heads_parked ? " (heads parked)" : "", plan.need / 1048576.0, plan.trunk_peak / 1048576.0, plan.catchup_peak / 1048576.0, ev.size(),
                         S().ledger_->borrowed_bytes() / 1048576.0, free_now() / 1048576.0, plan.host_copies / 1e9);
    }
    const bool host_res = T > chunk;
    using pclk = std::chrono::steady_clock; double trunk_ms = 0, mtp_ms = 0; f_.cp->rt.qsa_ms = 0; const auto p0 = pclk::now();
    auto lap = [&](double & acc, pclk::time_point & t) { if (!kPrefillLog) return; ck(cudaDeviceSynchronize()); const auto u = pclk::now();
                                                         acc += std::chrono::duration<double, std::milli>(u - t).count(); t = u; };
    auto mem_note = [&](const char * at) {
        if (tqllm::debug_on("mem")) ledger(at);
        if (kPrefillLog) { ck(cudaDeviceSynchronize());
            std::fprintf(stderr, "tqllm: prefill +%.2f s %s\n", std::chrono::duration<double>(pclk::now() - p0).count(), at); }
    };
    mem_note("banks borrowed"); stamp("plan + banks borrowed");
    if (const size_t back = (turn ? turn_lm : kv_back_for_prompt(T, chunk)) ? S().model_->kv_to_device() : 0)
        std::fprintf(stderr, "tqllm: attention KV: %.0f MB back on the device for this prompt\n", back / 1048576.0);
    if (host_res) {
        f_.mtp_cache->clear();
        if (tqllm::flashnext::prefill_residuals_on_host(f_.cp->run.head_here(), chunk) && f_.res_host_cap < size_t(P) * W) {  // kv_inline below needs none
            if (f_.res_host) ck(cudaFreeHost(f_.res_host));
            ck(cudaMallocHost(reinterpret_cast<void **>(&f_.res_host), size_t(P) * W * sizeof(float))); f_.res_host_cap = size_t(P) * W;
        }
    }
    mem_note("residual host buffer ready"); stamp("KV + residual host buffer ready");
    if ((turn ? turn_lm : T >= 256) && chunk >= 256) arena_resize(arena_req);  // layer-major prefill; decode shrinks it again
    stamp("KV + arena ready");
    mem_note("prefill arena ready");
    // tokens per MTP catch-up batch (each needs topk unique MTP slots)
    // a cache holding every expert takes any width (the whole chunk: grouped GEMM path); else unique experts <= slots
    static const int cw = f_.mtp_cache->slots() >= checkpoint.config.experts
        ? chunk
        : std::max(1, std::min(16,
                               f_.mtp_cache->slots() / checkpoint.config.topk));
    // the catch-up runs the MTP layer layer-major in 1024-position pieces
    constexpr bool mtp_lm = true;  // MTP catch-up layer-major
    auto catch_up = [&](int b, int e, const float * res) {  // res: residual rows b..e-1 on the device
        if (mtp_lm && chunk >= 256 && e - b >= 256) {  // layer-major MTP straight from the chunk's rows
            const int piece = tqllm::flashnext::mtp_piece_tokens(e - b);
            for (int q = b; q < e; q += piece) {
                const int qe = std::min(e, q + piece); std::vector<int> toks;
                for (int i = q; i < qe; ++i) toks.push_back(i + 1 < P ? ids[i + 1] : S().out_[0]);
                if (g_arena) g_arena->used = 0;
                S().d1_ = S().model_->mtp_prefill(toks, res + size_t(q - b) * W);
            }
            return;
        }
        for (int q = b; q < e; q += cw) {
            std::vector<int> toks;
            for (int i = q; i < std::min(e, q + cw); ++i) toks.push_back(i + 1 < P ? ids[i + 1] : S().out_[0]);
            if (g_arena) g_arena->used = 0;  // the catch-up only reads persistent residual rows
            S().d1_ = S().model_->mtp_batch(toks, res + size_t(q - b) * W);
        }
    };
    S().model_->keep_embeddings = host_res && mtp_lm && chunk >= 256; S().model_->prompt_emb.clear();  // indexed by position
    if (S().model_->keep_embeddings) S().model_->prompt_emb.alloc(size_t(P) * size_t(c.hidden));
    const bool dfl = f_.drafter && f_.dfl_feat && slots_.size() == 1;
    // head_only (--split layer, one sequence): the first token, the MTP catch-up and the drafter's K/V on the last rank only
    auto & run = f_.cp->run; const bool head = run.head_here();
    // a multi-chunk prompt publishes its first token once decode can start (after the last position's MTP layer, the K/V
    // move and the bank refill): published at the pick, the second token followed it 0.25-1.2 s later (3090, 32K / 128K)
    const bool hold_first = host_res;
    auto first_token = [&](const float * row) {
        std::vector<int> t = head ? pick(row, 1) : std::vector<int>(1);
        if (run.head_only) {
            int buf[64] = {t[0]};
            DecodeModel::split_from_last(*run.split, buf, 64);
            t[0] = buf[0];
        }
        S().out_.push_back(t[0]);
        if (first_token_hook_ && !hold_first) first_token_hook_();
    };
    if (dfl) { S().model_->cap_layers = &f_.drafter->target_layers(); S().model_->cap_out = f_.dfl_feat->p; S().model_->cap_stride = f_.drafter->features(); }
    // a multi-chunk prompt (residuals in host memory, MTP bank dropped) writes the MTP layer's keys / values chunk by chunk
    // inside the trunk loop: run after the trunk, the catch-up held the second token 0.3 s behind the first (3090, 32K)
    // turn chunks: every chunk, one-chunk prompts too (the same pieces cold and resumed; the catch-up below runs others)
    const bool kv_inline = (host_res || turn) && head && mtp_lm && chunk >= 256;
    std::unique_ptr<Device<float>> last_res;  // kv_inline: the prompt's last residual row, for its MTP layer after the trunk
    const int step_rows = turn ? chunk : reused_read_rows(live, P0 ? S().snap_rows_ : 0, T, chunk);
    // after a chunk's trunk: the first token, the MTP layer's K/V, the host residual copy, the snapshot
    auto after_chunk = [&](int b, int e, const Device<float> * rows, pclk::time_point tl) {
        if (e == P && (heads_parked || kv_inline)) {  // picked after the trunk buffers go and the heads are back, or after the K/V catch-up below
            PersistentAlloc keep; last_row = std::make_unique<Device<float>>(size_t(c.hidden));
            ck(cudaMemcpyAsync(last_row->p, rows->p + size_t(e - b - 1) * c.hidden, size_t(c.hidden) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        } else if (e == P) first_token(rows->p + size_t(e - b - 1) * checkpoint.config.hidden);
        if (kv_inline) {
            // the MTP layer's keys / values for this chunk's rows, from the trunk's rows still on the device (no host round
            // trip); the next tokens' embeddings from the trunk's prep, the chunk's last row gathers its own (its next token
            // opens the next chunk, whose prep has not run). The prompt's last position (next token = the first output)
            // runs the whole MTP layer after the trunk.
            const auto & pemb = S().model_->prompt_emb; const size_t hd = size_t(c.hidden);
            const int ke = std::min(e, P - 1), split = e < P ? e - 1 : ke;
            const float * res = S().model_->res_keep->p;
            auto toks = [&](int x, int y) { return std::vector<int>(ids.begin() + x + 1, ids.begin() + y + 1); };
            if (g_arena) g_arena->used = 0;
            if (split > b) S().model_->mtp_kv_prefill(toks(b, split), res, pemb.size() == size_t(P) * hd ? pemb.data() + size_t(b + 1) * hd : nullptr);
            if (ke > split) { if (g_arena) g_arena->used = 0; S().model_->mtp_kv_prefill(toks(split, ke), res + size_t(split - b) * W, nullptr); }
            if (e == P) {
                PersistentAlloc keep; last_res = std::make_unique<Device<float>>(size_t(W));
                ck(cudaMemcpyAsync(last_res->p, res + size_t(P - 1 - b) * W, size_t(W) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
                if (!heads_parked) { first_token(last_row->p); last_row.reset(); }
            }
            lap(mtp_ms, tl);
        }
        stamp("first token picked");
        if (host_res && head && !kv_inline) ck(cudaMemcpyAsync(f_.res_host + size_t(b) * W, S().model_->res_keep->p, size_t(e - b) * W * sizeof(float), cudaMemcpyDeviceToHost, nullptr));
        if (host_res) mem_note("chunk done");
        else if (head && !kv_inline) { catch_up(b, e, S().model_->res_keep->p); lap(mtp_ms, tl); }
        if (e == snap) take_snapshot(S());
        tqllm::prefill_completed(nullptr, uint32_t(e), uint32_t(P));
    };
    int b0 = P0;
    for (int g0 = 0; g0 < outer; g0 += outer_group) {
        std::vector<std::vector<int>> toks; const int g1 = std::min(outer, g0 + outer_group);
        for (int i = g0; i < g1; ++i) toks.emplace_back(ids.begin() + spans[size_t(i)].first, ids.begin() + spans[size_t(i)].second);
        auto tl = pclk::now();
        S().model_->step_outer(toks, [&](size_t i, const Device<float> & rows) {
            const auto & sp = spans[size_t(g0) + i]; lap(trunk_ms, tl); after_chunk(sp.first, sp.second, &rows, tl); });
        b0 = spans[size_t(g1) - 1].second;
    }
    bool stepped = false;  // turn: a short chunk ran as decode steps since the last layer-major chunk
    for (int b = b0, e = 0; b < P; b = e) {
        e = chunk_end(b, step_rows);
        // decode steps build the banks of lent layers from the memory the plan had counted for its layer-major chunks
        // (3090, a cold 15K chat read: the next chunk out of memory, 27 MiB free): they go again before the next one, as
        // the refill releases a bank built mid-prompt (refill_rebuilds)
        if (turn && e - b >= kTurnShort && stepped && S().ledger_) {
            for (int l : S().ledger_->borrowed_layers()) if (f_.cache->bank_ready(l)) f_.cache->release_bank(l);
            cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0));
        }
        stepped = turn && e - b < kTurnShort ? true : e - b >= kTurnShort ? false : stepped;
        auto tl = pclk::now();
        const std::vector<int> next(ids.begin() + e, ids.begin() + (e < P ? chunk_end(e, step_rows) : P));
        auto rows = S().model_->step(std::vector<int>(ids.begin() + b, ids.begin() + e), &next);
        if (dfl && head) f_.drafter->inject(f_.dfl_feat->p, e - b, uint32_t(b));  // the chunk's features into the drafter's K/V
        lap(trunk_ms, tl);
        stamp("step returned");
        after_chunk(b, e, rows.get(), tl);
    }
    if (run.split) DecodeModel::split_release_staging();
    if (host_res) {
        mem_note("after trunk prefill");
        // the trunk buffers go first: the MTP bank (650 MB) needs their room (keeping them OOMed the bank upload at 140 slots)
        S().model_->release_prefill(); S().model_->pf_borrowed.clear();
        if (heads_parked) { checkpoint.unpark_heads(); heads_parked = 0; first_token(last_row->p); last_row.reset(); }
        mem_note("prefill buffers released");
        if (tqllm::debug_cstr("profile_req")) std::fprintf(stderr, "tqllm: req +%.3f s buffers released\n", std::chrono::duration<double>(pclk::now() - p0).count());
        auto tl = pclk::now();
        // residual rows staged through a small device buffer, a whole number of catch-up batches, starting at each
        // chunk boundary so the batches are the ones the interleaved order ran
        const int rb = cw * std::max(1, 256 / cw);
        std::unique_ptr<Device<float>> res;
        { PersistentAlloc keep; res = std::make_unique<Device<float>>(size_t(rb) * W); }
        if (!head) {
            S().model_->prompt_emb.clear(); S().model_->keep_embeddings = false;
        } else if (kv_inline) {
            if (g_arena) g_arena->used = 0;
            S().d1_ = S().model_->mtp_batch({S().out_[0]}, last_res->p); last_res.reset();
            S().model_->release_prefill();
            S().model_->prompt_emb.clear(); S().model_->keep_embeddings = false;
        } else if (mtp_lm && chunk >= 256) {
            const int mb = std::min(P, chunk);  // one piece per chunk: the MTP experts are staged once per piece
            std::unique_ptr<Device<float>> rows;
            { PersistentAlloc keep; rows = std::make_unique<Device<float>>(size_t(mb) * W); }
            const auto & pemb = S().model_->prompt_emb; const size_t hd = size_t(c.hidden);
            std::vector<float> pe;
            // keys / values for every position but the last, then the full MTP layer for the last one (mtp_kv_prefill)
            for (int q = P0; q < P; q += mb) {
                const int qe = std::min(P, q + mb);
                ck(cudaMemcpyAsync(rows->p, f_.res_host + size_t(q) * W, size_t(qe - q) * W * sizeof(float), cudaMemcpyHostToDevice, nullptr));
                std::vector<int> toks;
                for (int i = q; i < qe; ++i) toks.push_back(i + 1 < P ? ids[i + 1] : S().out_[0]);
                const float * emb = nullptr;
                if (pemb.size() == size_t(P) * hd) {  // the next tokens' embeddings from the trunk's prep; the last one (the first output) is gathered
                    pe.resize(size_t(qe - q) * hd);
                    const int kept = std::min(qe, P - 1) - q;
                    std::memcpy(pe.data(), pemb.data() + size_t(q + 1) * hd, size_t(kept) * hd * sizeof(float));
                    if (qe == P) { const auto last = checkpoint.file("token_embd.weight").gather_rows("token_embd.weight", {uint64_t(S().out_[0])});
                                   std::memcpy(pe.data() + size_t(kept) * hd, last.data(), hd * sizeof(float)); }
                    emb = pe.data();
                }
                if (g_arena) g_arena->used = 0;
                const int keep = qe == P ? qe - q - 1 : qe - q;  // the last position runs the whole layer below
                if (keep > 0) S().model_->mtp_kv_prefill(std::vector<int>(toks.begin(), toks.begin() + keep), rows->p, emb);
                if (qe == P) { if (g_arena) g_arena->used = 0; S().d1_ = S().model_->mtp_batch({toks.back()}, rows->p + size_t(qe - q - 1) * W); }
            }
            rows.reset(); S().model_->release_prefill();
            S().model_->prompt_emb.clear(); S().model_->keep_embeddings = false;
        } else
        for (int b = P0; b < P; b += chunk)
            for (int q = b, e = std::min(P, b + chunk); q < e; q += rb) {
                const int qe = std::min(e, q + rb);
                ck(cudaMemcpyAsync(res->p, f_.res_host + size_t(q) * W, size_t(qe - q) * W * sizeof(float), cudaMemcpyHostToDevice, nullptr));
                catch_up(q, qe, res->p);
            }
        lap(mtp_ms, tl);
        mem_note("after MTP catch-up");
        if (tqllm::debug_cstr("profile_req")) std::fprintf(stderr, "tqllm: req +%.3f s after MTP catch-up\n", std::chrono::duration<double>(pclk::now() - p0).count());
    }
    if (kPrefillLog) std::fprintf(stderr, "tqllm: prefill %d tokens %.1f s: trunk %.1f s (QSA %.1f s), MTP catch-up %.1f s\n", P,
        std::chrono::duration<double>(pclk::now() - p0).count(), trunk_ms / 1000, f_.cp->rt.qsa_ms / 1000, mtp_ms / 1000);
    if (!host_res && kv_inline) {  // a turn prompt in one chunk: the whole MTP layer for its last position
        if (g_arena) g_arena->used = 0;
        S().d1_ = S().model_->mtp_batch({S().out_[0]}, last_res->p); last_res.reset();
    }
    if (!host_res && (turn ? turn_lm : T >= 256) && chunk >= 256) S().model_->release_prefill();  // single-chunk layer-major prompt
    arena_resize(f_.arena_decode);
    f_.arena.high = 0;  // decode high-water from here (LEDGER decode of: arena high)
    if (P >= 256) {  // long prompt: the old K/V rows to host memory, the refill below takes the freed device memory
        using kclk = std::chrono::steady_clock; const auto k0 = kclk::now();
        if (const size_t freed = S().model_->kv_to_host()) {
            std::fprintf(stderr, "tqllm: attention KV: %.0f MB of old positions moved to host memory in %.0f ms\n", freed / 1048576.0,
                         std::chrono::duration<double, std::milli>(kclk::now() - k0).count());
            // banks an earlier long prompt left below their slot target grow back into the freed memory with the refill
            if (slots_.size() == 1) f_.cache->regrow(size_t(768) << 20);
        }
    }
    stamp("before the bank refill"); if (tqllm::debug_on("mem")) ledger("before the bank refill");
    refill_borrowed(); f_.cp->rt.trim_below = SIZE_MAX; stamp("banks refilled");
    if (tqllm::debug_on("mem")) ledger("banks refilled");
    S().model_->set_spec(true);
    ck(cudaDeviceSynchronize()); S().t_decode_ = std::chrono::steady_clock::now();
    if (hold_first && first_token_hook_ && !S().out_.empty()) first_token_hook_();
    if (tqllm::debug_on("mem")) { S().stats_at_decode_ = f_.cache->stats(); S().mtp_stats_at_decode_ = f_.mtp_cache->stats(); }
    S().prefill_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(S().t_decode_ - S().t_begin_).count());
    if (profiling) ck(cudaProfilerStop());
    if (f_.cold_cache) S().cold_at_decode_ = f_.cold_cache->stats();
    if (f_.cold_store) S().cold_store_at_decode_ = f_.cold_store->stats();
    if (f_.cold_fetch_queue) S().cold_fetch_at_decode_ = f_.cold_fetch_queue->stats();
    if (f_.cp->rt.profile_decode_req) ck(cudaProfilerStart());
    if (g_arena && tqllm::debug_cstr("arena")) std::fprintf(stderr, "tqllm: prefill %d tokens, arena high %.0f MB of %.0f\n", P, g_arena->high / 1048576.0, g_arena->size / 1048576.0);
    S().done_ = int(S().out_.size()) >= S().generate_ || is_eos(S().out_.back());
    if (caching()) { S().seq_ = ids; S().snap_ = snap ? snap : keep_snap; if (snap || snap_at_start) S().snap_rows_ = step_rows; }
    (void) c;
}

bool FlashNextEngine::prefill_ahead(int s, const std::vector<uint32_t> & prefix) {
    if (!turns() || slots_.size() != 1 || s != 0 || f_.cp->run.split || f_.cp->run.experts) return false;
    const auto lk = ep_lock();
    auto & sl = slots_[0];
    if (!sl.model_ || sl.serving_ || !sl.done_ || sl.snap_ <= 0) return false;
    std::vector<uint32_t> p = prefix; p.push_back(f_.cp->run.im_start);  // the next message's start: the state is kept there
    const int P0 = reusable_prefix(sl.seq_, sl.snap_, p);
    if (!P0 || int(p.size()) < P0 + 3 || p.size() + 1 + 16 > slot_context()) return false;
    cur_ = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<int> old_seq = sl.seq_; const int old_snap = sl.snap_;
    auto hook = std::move(first_token_hook_); first_token_hook_ = nullptr;
    sl.stash_ = true;
    try { spec_begin(p, 1, 0); }
    catch (const std::exception & e) {
        std::fprintf(stderr, "tqllm: idle prefill of the returned answer failed: %s\n", e.what());
        abort_prefill();
        sl.stash_ = false; sl.done_ = true; sl.serving_ = false; sl.model_.reset(); sl.snap_ = sl.live_ = 0; sl.drop_alt();
        first_token_hook_ = std::move(hook); ep_note_state(); return false;
    }
    first_token_hook_ = std::move(hook);
    sl.done_ = true; sl.serving_ = false; ep_note_state();
    if (sl.stash_ || sl.alt_state_.empty()) { sl.stash_ = false; sl.drop_alt(); }  // no new snapshot was taken
    else { sl.alt_seq_ = old_seq; sl.alt_snap_ = old_snap; }
    std::fprintf(stderr, "tqllm: prefilled the returned answer while idle: %zu tokens in %.2f s (kept at %d)\n", p.size() - size_t(P0),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), sl.snap_);
    return true;
}

}  // namespace tqllm::flashnext
