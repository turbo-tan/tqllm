// Flash-Next request prefill: memory plan, expert bank borrow and refill, chunked prompt, MTP catch-up.
#include "fn_engine.hpp"
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
    // small cards: stop while the warm-up's keep (512 MB) is still free; a refill to the last megabyte left the first
    // decode step's bank upload 88 MiB (12 GB, 16K prompt in 2 x 10240): the rest come back as ordinary misses. With slot
    // tiers each refilled bank is sized to what is free (MixedExpertCache choose_cap keeps 256 MB).
    { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
    const int done = led->refill(g_small_card ? size_t(512) << 20 : 0); cudaGetLastError();
    ck(cudaDeviceSynchronize());
    if (all) std::fprintf(stderr, "tqllm: refilled %d of %zu borrowed expert banks in %.0f ms\n", done, all, std::chrono::duration<double, std::milli>(pclk::now() - r0).count());
    S().model_->pf_refill.clear(); led.reset();
}

void FlashNextEngine::arena_resize(size_t bytes) {
    if (f_.arena.size == bytes || !f_.arena.base) return;
    ck(cudaDeviceSynchronize()); ck(cudaFree(f_.arena.base)); f_.arena.base = nullptr;
    ck(cudaMalloc(&f_.arena.base, bytes)); f_.arena.size = bytes; f_.arena.used = 0;
}

void FlashNextEngine::spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int) {
    static std::map<int, std::vector<int>> bank_keep;  // lent trunk banks: their resident experts before the prompt
    auto & checkpoint = *f_.cp; const auto & c = checkpoint.config;
    require(!prompt.empty() && prompt.size() + count + 16 <= slot_context(), "prompt plus max_tokens exceeds the slot context");
    S().model_.reset(); S().prompt_res_.reset(); g_trim_below = SIZE_MAX; ++g_kv_epoch;
    // banks a long prompt left below their slot target grow back once its state is gone, unless this request's own state
    // needs the room (regrowing then would throw the banks' experts away and rebuild them at the same smaller size)
    // with TQFN_KV_HOST the K/V older than the kept window lives in host memory after the prompt, so only the window's
    // state keeps device room (a long prefill borrows banks for the rest and gives them back)
    static const bool kv_host = std::getenv("TQFN_KV_HOST") && std::atoi(std::getenv("TQFN_KV_HOST")) != 0;
    static const size_t kv_keep = std::getenv("TQFN_KV_HOST_KEEP") ? size_t(std::atoi(std::getenv("TQFN_KV_HOST_KEEP"))) : 32768;
    const size_t state_rows = kv_host ? std::min(prompt.size() + count + 16, 2 * kv_keep) : prompt.size() + count + 16;
    if (slots_.size() == 1) f_.cache->regrow(regrow_keep_bytes(size_t(768) << 20, f_.state_full, f_.state_tokens, state_rows));
    if (slots_.size() == 1 && f_.kv_capacity_full) {  // one sequence: its attention state is sized by the request, not the slot context
        const int need = int(std::min<size_t>(size_t(f_.kv_capacity_full), (prompt.size() + count + 16 + 1023) / 1024 * 1024));
        if (need != g_kv_capacity) { if (g_qsa_batch) { ck(cudaDeviceSynchronize()); g_qsa_batch.reset(); } g_kv_capacity = need; }
    }
    if (g_arena) g_arena->used = 0;
    // the pool keeps what the last request freed (release threshold max); hand it back so plain cudaMalloc users (expert
    // banks, staging, attention workspaces) can have it
    // a failed earlier request may have left a non-sticky CUDA error pending (QSA / CUB would report it on every later
    // request); a sticky one means the context is lost
    if (const cudaError_t e = tqllm::clear_stale_cuda_error(); e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA context lost by an earlier request (") + cudaGetErrorString(e) + "); restart tqserve");
    { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
    if (tqllm::debug_on("mem")) { std::fprintf(stderr, "LEDGER request of %zu prompt tokens\n", prompt.size()); ledger("request start"); }
    S().model_ = std::make_unique<DecodeModel>(checkpoint, *f_.fixtures, *f_.cache, prompt.size() + count + 16);
    S().model_->mtp_experts = f_.mtp_cache.get();
    bind();
    const std::vector<int> ids(prompt.begin(), prompt.end());
    // TQFN_PROFILE_REQ=N: cudaProfilerStart at the N-th request's prefill, stop after it (nsys --capture-range=cudaProfilerApi)
    static const int prof_req = std::getenv("TQFN_PROFILE_REQ") ? std::atoi(std::getenv("TQFN_PROFILE_REQ")) : 0; static int req_no = 0;
    // TQFN_PROFILE_DECODE=1: the capture covers that request's decode instead of its prefill
    static const bool prof_decode = std::getenv("TQFN_PROFILE_DECODE") && std::atoi(std::getenv("TQFN_PROFILE_DECODE")) != 0;
    const bool profiling = prof_req > 0 && ++req_no == prof_req && !prof_decode;
    g_profile_decode_req = prof_req > 0 && req_no == prof_req && prof_decode;
    if (profiling) ck(cudaProfilerStart());
    S().t_begin_ = std::chrono::steady_clock::now(); S().prefill_us_ = S().decode_us_ = 0;
    static const bool req_log = std::getenv("TQFN_PROFILE_REQ") != nullptr;  // TQFN_PROFILE_REQ: host timeline of the request
    const auto stamp = [&](const char * what) { if (req_log) std::fprintf(stderr, "tqllm: req +%.3f s %s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - S().t_begin_).count(), what); };
    S().out_.clear(); S().out_view_.clear(); S().generate_ = int(count); S().steps_ = S().toks_ = 0;
    // Trunk prefill first, then the MTP catch-up, so their working sets do not add up: a prompt longer than one chunk
    // keeps its residuals in pinned host memory, drops the MTP bank before the trunk and the layer-major buffers
    // after it. The catch-up runs the same batches in the same order as when it followed each chunk.
    const int P = int(ids.size()); const int W = S().model_->width;
    stamp("model ready");
    // the prefill chunk for this request: the engine's chunk, halved while the plan cannot fit even with every bank
    // released (a 128K prompt at 104 slots does not fit 16K chunks on a 24 GB card)
    // Small cards: the chunk starts at the whole prompt (every chunk streams every expert once, and at 12 GB the link is
    // the bottleneck: 5 x 5,120 streams 327 GB in the 15 s a 3090 takes to compute 25K tokens) and the plan below shrinks
    // it until it fits, with the LM heads parked. 24 GB keeps the engine chunk: there the streams already hide under the
    // kernels (nsys 10-19: 25K in 2 x 16K 90.7 % busy, in one chunk 86.3 % and 0.8 s slower).
    constexpr int kMaxChunk = 32768;
    const int chunk0 = g_tune.chunk_set || !g_small_card ? f_.chunk : std::max(f_.chunk, std::min(kMaxChunk, (P + 1023) / 1024 * 1024));
    int chunk = chunk0; size_t arena_req = f_.arena_prefill; size_t heads_parked = 0;  // the arena a chunk needs: the start-up size, or less once the chunk has halved
    struct UnparkGuard { Checkpoint & cp; size_t & n; ~UnparkGuard() { if (n) { try { cp.unpark_heads(); } catch (...) {} n = 0; } } } unpark_guard{checkpoint, heads_parked};
    std::unique_ptr<Device<float>> last_row;  // heads parked: the last chunk's head row, picked once they are back
    if (P >= 256 && chunk >= 256) {
        const auto & c = checkpoint.config;
        tqllm::PrefillShape sh; sh.hidden = c.hidden; sh.hc = c.hc; sh.expert_ff = c.expert_ff; sh.topk = c.topk; sh.experts = c.experts; sh.layers = c.layers;
        // the staging halves are sized for the largest expert block of any layer (prefill_layer): mixed-format files have
        // bigger blocks in some layers (the first layer's block undercounted staging by 170 MB at 245K)
        for (int l = 0; l < c.layers; ++l) if (f_.cache->has_layout(l)) {
            const auto lay = f_.cache->bank_layout(l); size_t blk = 0;
            for (int i = 0; i < 6; ++i) blk += lay.piece[i];
            sh.expert_block_bytes = std::max(sh.expert_block_bytes, std::max(blk, size_t(lay.block)) + 6 * 256 / std::max(1, g_tune.stage));
        }
        tqllm::PrefillTuning tu; tu.sub = g_tune.sub; tu.stage = g_tune.stage;
        tu.dense_cache_bytes = dense_half_pool_bytes(); tu.mtp_bank_bytes = f_.mtp_cache->peak_device_bytes();
        tu.workspace_bytes = size_t(1024 + std::max(0L, g_tune.reserve_mb)) << 20;  // --prefill-reserve-mb: headroom for the kernels' own scratch
        // above 128K keep a small margin beyond the plan's own terms (allocator granularity, kernel scratch)
        if (P > 131072) tu.workspace_bytes += size_t(448) << 20;
        int attn = 0; for (int l = 0; l < c.layers; ++l) if (checkpoint.has_tensor("blk." + std::to_string(l) + ".indexer.k_proj.weight")) ++attn;
        auto context_term = [&](int block) { return tqllm::prefill_context_bytes(g_kv_capacity, block, attn, checkpoint.attention_kvheads, checkpoint.attention_dim, c.index_dim); };
        // the plan counts the QSA selection scratch in context_bytes: release the old one so the free memory it reads is right
        if (g_qsa_batch) { ck(cudaDeviceSynchronize()); g_qsa_batch.reset(); cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
        auto free_now = [] { return tqllm::device_free_bytes(); };
        std::vector<size_t> bank_bytes; size_t banks_total = 0;
        for (int l : f_.cache->banks_lru()) { bank_bytes.push_back(f_.cache->bank_bytes(l)); banks_total += bank_bytes.back(); }
        const int block0 = std::clamp(g_tune.sub, 512, 2048);
        int block = block0;
        tqllm::PrefillPlan plan; bool park_tried = false, borrow_all = false;
        for (;;) {
            tu.chunk = chunk; tu.sub = std::min(g_tune.sub, block); tu.context_bytes = context_term(block);
            const size_t mtp_room = P > chunk ? f_.mtp_cache->stats().device_bytes : 0;  // a multi-chunk prompt clears the MTP bank first
            arena_req = std::min<size_t>(f_.arena_prefill, tqllm::prefill_arena_bytes(chunk));  // the measured rule at every chunk size
            arena_req += size_t(block) * size_t(checkpoint.attention_heads) * 2052 * 8 + (size_t(64) << 20);  // one block's QSA score scratch + headroom
            tu.arena_bytes = arena_req;
            // the plan sees min(measured, budget): the same prompt length plans the same chunk and block (the same text)
            // whatever earlier requests left behind; parked heads and the MTP bank a multi-chunk prompt drops add to the budget
            size_t avail = free_now() + mtp_room + banks_total;
            if (const size_t cap = f_.plan_budget ? f_.plan_budget + mtp_room + heads_parked : 0; cap && avail > cap) avail = cap;
            else if (cap && tqllm::debug_on("mem")) std::fprintf(stderr, "tqllm: prefill plan: %zu MB available, below the %zu MB budget\n", avail >> 20, cap >> 20);
            plan = tqllm::prefill_plan(sh, tu, P, std::max(avail, banks_total) - banks_total);
            // small cards release every bank for a multi-chunk prompt: a kept bank grows by its misses during the prompt
            // (slots=auto banks start below their cap) and OOMed at "expert bank upload ... 54 MiB free"; the fit check
            // below still asks whether the plan fits with every bank gone
            borrow_all = g_small_card && P > chunk && g_tune.borrow == tqllm::BorrowMode::Auto;
            try { (void) tqllm::banks_to_borrow(plan, bank_bytes, g_tune.borrow, g_tune.borrow_count);
                // small cards: a multi-chunk prompt parks the LM heads (needed only for the first token) in pinned host
                // memory and plans again with their room (~635 MB: a bigger chunk streams the experts fewer times)
                if (g_small_card && P > chunk && !park_tried) {
                    park_tried = true; heads_parked = checkpoint.park_heads();
                    if (heads_parked) {  // the freed buffers sit in the async pool until trimmed; the plan reads the device's free memory
                        cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0));
                        chunk = chunk0; block = block0; continue;
                    }
                }
                // one chunk after all: that path picks and catches up with the heads in place; plan again without parking
                if (heads_parked && P <= chunk) { checkpoint.unpark_heads(); heads_parked = 0; chunk = chunk0; block = block0; continue; }
                break; }
            catch (const std::runtime_error & e) {
                const auto & cb = plan.chunk;
                if (tqllm::debug_on("mem"))
                    std::fprintf(stderr, "tqllm: prefill of %d tokens: %s (chunk %d, block %d: activations %zu expert_io %zu staging %zu residual %zu arena %zu dense %zu workspace %zu, context %zu MB)\n",
                                 P, e.what(), chunk, block, cb.activations >> 20, cb.expert_io >> 20, cb.staging >> 20, cb.residual >> 20, cb.arena >> 20, cb.dense_cache >> 20, cb.workspace >> 20, tu.context_bytes >> 20);
                // a smaller selection block first (~0.5 s at 32K); halving the chunk doubles the expert bytes streamed per prompt
                if (block > 512) { block = block > 1024 ? 1024 : 512; continue; }
                if (chunk <= 2048) throw;
                chunk = std::max(2048, chunk * 7 / 8 / 1024 * 1024); block = block0;  // 16K -> 14K -> 12K -> 10.5K ...: every chunk streams all experts, so take the largest that fits
            }
        }
        g_qsa_block = block; g_sub_req = block;
        // the first chunk's host prep (embedding + PLE gathers, ~0.4 s at 16K) runs while the banks, KV and arena are set up
        { DecodeModel * m = S().model_.get(); std::vector<int> first(ids.begin(), ids.begin() + std::min(P, chunk));
          m->prep_next_ = std::async(std::launch::async, [m, first] { return m->prepare_host(first); }); }
        if (P > chunk) f_.mtp_cache->clear();  // a multi-chunk prompt needs the room; one chunk keeps the MTP bank
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
            top.resize(std::min(top.size(), size_t(f_.cache->plan_slots(l)))); f_.cache->prepare(l, top); };
        ops.free_bytes = free_now;
        S().ledger_ = std::make_unique<tqllm::BankLedger>(std::move(ops));
        std::vector<int> bankless; for (int l = 0; l < c.layers; ++l) if (!f_.cache->bank_ready(l) && f_.cache->has_layout(l)) bankless.push_back(l);
        const auto ev = tqllm::borrow_banks(*S().ledger_, plan, borrow_all ? tqllm::BorrowMode::All : g_tune.borrow, g_tune.borrow_count, bankless); S().plan_catchup_bytes_ = plan.catchup_peak;
        g_trim_below = S().ledger_->borrowed_bytes() + f_.mtp_cache->peak_device_bytes() + (size_t(512) << 20);
        S().model_->pf_borrowed.assign(size_t(c.layers), 0);
        for (int l : ev) S().model_->pf_borrowed[size_t(l)] = 1;
        if (!ev.empty() || tqllm::debug_on("mem"))
            std::fprintf(stderr, "tqllm: prefill of %d tokens in %d chunks of %d%s: needs %.0f MB (chunk %.0f, catch-up %.0f), released %zu expert banks (%.0f MB), %.0f MB free, %.1f GB of experts to stream\n",
                         P, plan.chunks, chunk, heads_parked ? " (heads parked)" : "", plan.need / 1048576.0, plan.trunk_peak / 1048576.0, plan.catchup_peak / 1048576.0, ev.size(),
                         S().ledger_->borrowed_bytes() / 1048576.0, free_now() / 1048576.0, plan.host_copies / 1e9);
    }
    const bool host_res = P > chunk;
    using pclk = std::chrono::steady_clock; double trunk_ms = 0, mtp_ms = 0; g_qsa_ms = 0; const auto p0 = pclk::now();
    auto lap = [&](double & acc, pclk::time_point & t) { if (!g_prefill_log) return; ck(cudaDeviceSynchronize()); const auto u = pclk::now();
                                                         acc += std::chrono::duration<double, std::milli>(u - t).count(); t = u; };
    auto mem_note = [&](const char * at) {
        if (tqllm::debug_on("mem")) ledger(at);
        if (g_prefill_log) { ck(cudaDeviceSynchronize());
            std::fprintf(stderr, "tqllm: prefill +%.2f s %s\n", std::chrono::duration<double>(pclk::now() - p0).count(), at); }
    };
    mem_note("banks borrowed"); stamp("plan + banks borrowed");
    if (const size_t back = S().model_->kv_to_device())
        std::fprintf(stderr, "tqllm: attention KV: %.0f MB back on the device for this prompt\n", back / 1048576.0);
    if (host_res) {
        f_.mtp_cache->clear();
        if (f_.res_host_cap < size_t(P) * W) {
            if (f_.res_host) ck(cudaFreeHost(f_.res_host));
            ck(cudaMallocHost(reinterpret_cast<void **>(&f_.res_host), size_t(P) * W * sizeof(float))); f_.res_host_cap = size_t(P) * W;
        }
    }
    mem_note("residual host buffer ready");
    if (P >= 256 && chunk >= 256) arena_resize(arena_req);  // layer-major prefill; decode shrinks it again
    stamp("KV + arena ready");
    mem_note("prefill arena ready");
    // TQFN_MTP_CATCHUP: tokens per MTP catch-up batch (each needs topk unique MTP slots)
    // a cache holding every expert takes any width (the whole chunk: grouped GEMM path); else unique experts <= slots
    static const int cw = f_.mtp_cache->slots() >= checkpoint.config.experts
        ? (std::getenv("TQFN_MTP_CATCHUP") ? std::atoi(std::getenv("TQFN_MTP_CATCHUP")) : chunk)
        : std::max(1, std::min(std::getenv("TQFN_MTP_CATCHUP") ? std::atoi(std::getenv("TQFN_MTP_CATCHUP")) : 16,
                               f_.mtp_cache->slots() / checkpoint.config.topk));
    // TQFN_MTP_LAYER_MAJOR (default on): the catch-up runs the MTP layer layer-major in 1024-position pieces
    static const bool mtp_lm = !(std::getenv("TQFN_MTP_LAYER_MAJOR") && std::atoi(std::getenv("TQFN_MTP_LAYER_MAJOR")) == 0);
    auto catch_up = [&](int b, int e, const float * res) {  // res: residual rows b..e-1 on the device
        if (mtp_lm && chunk >= 256 && e - b >= 256) {  // layer-major MTP straight from the chunk's rows
            for (int q = b; q < e; q += 1024) {
                const int qe = std::min(e, q + 1024); std::vector<int> toks;
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
    S().model_->keep_embeddings = host_res && mtp_lm && chunk >= 256; S().model_->prompt_emb.clear();
    if (S().model_->keep_embeddings) S().model_->prompt_emb.alloc(size_t(P) * size_t(c.hidden));
    for (int b = 0; b < P; b += chunk) {
        const int e = std::min(P, b + chunk);
        auto tl = pclk::now();
        const std::vector<int> next(ids.begin() + e, ids.begin() + std::min(P, e + chunk));
        auto rows = S().model_->step(std::vector<int>(ids.begin() + b, ids.begin() + e), &next);
        lap(trunk_ms, tl);
        stamp("step returned");
        if (e == P && heads_parked) {  // picked after the trunk buffers go and the heads are back
            PersistentAlloc keep; last_row = std::make_unique<Device<float>>(size_t(c.hidden));
            ck(cudaMemcpyAsync(last_row->p, rows->p + size_t(e - b - 1) * c.hidden, size_t(c.hidden) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        } else if (e == P) S().out_.push_back(pick(rows->p + size_t(e - b - 1) * checkpoint.config.hidden, 1)[0]);
        stamp("first token picked");
        if (host_res) ck(cudaMemcpyAsync(f_.res_host + size_t(b) * W, S().model_->res_keep->p, size_t(e - b) * W * sizeof(float), cudaMemcpyDeviceToHost, nullptr));
        if (host_res) mem_note("chunk done");
        else { catch_up(b, e, S().model_->res_keep->p); lap(mtp_ms, tl); }
        tqllm::prefill_completed(nullptr, uint32_t(e), uint32_t(P));
    }
    if (host_res) {
        mem_note("after trunk prefill");
        // the trunk buffers go first: the MTP bank (650 MB) needs their room (keeping them OOMed the bank upload at 140 slots)
        S().model_->release_prefill(); S().model_->pf_borrowed.clear();
        if (heads_parked) { checkpoint.unpark_heads(); heads_parked = 0; S().out_.push_back(pick(last_row->p, 1)[0]); last_row.reset(); }
        mem_note("prefill buffers released");
        if (std::getenv("TQFN_PROFILE_REQ")) std::fprintf(stderr, "tqllm: req +%.3f s buffers released\n", std::chrono::duration<double>(pclk::now() - p0).count());
        auto tl = pclk::now();
        // residual rows staged through a small device buffer, a whole number of catch-up batches, starting at each
        // chunk boundary so the batches are the ones the interleaved order ran
        const int rb = cw * std::max(1, 256 / cw);
        std::unique_ptr<Device<float>> res;
        { PersistentAlloc keep; res = std::make_unique<Device<float>>(size_t(rb) * W); }
        if (mtp_lm && chunk >= 256) {
            const int mb = std::min(P, chunk);  // one piece per chunk: the MTP experts are staged once per piece
            std::unique_ptr<Device<float>> rows;
            { PersistentAlloc keep; rows = std::make_unique<Device<float>>(size_t(mb) * W); }
            const auto & pemb = S().model_->prompt_emb; const size_t hd = size_t(c.hidden);
            std::vector<float> pe;
            // keys / values for every position but the last, then the full MTP layer for the last one (mtp_kv_prefill)
            for (int q = 0; q < P; q += mb) {
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
        for (int b = 0; b < P; b += chunk)
            for (int q = b, e = std::min(P, b + chunk); q < e; q += rb) {
                const int qe = std::min(e, q + rb);
                ck(cudaMemcpyAsync(res->p, f_.res_host + size_t(q) * W, size_t(qe - q) * W * sizeof(float), cudaMemcpyHostToDevice, nullptr));
                catch_up(q, qe, res->p);
            }
        lap(mtp_ms, tl);
        mem_note("after MTP catch-up");
        if (std::getenv("TQFN_PROFILE_REQ")) std::fprintf(stderr, "tqllm: req +%.3f s after MTP catch-up\n", std::chrono::duration<double>(pclk::now() - p0).count());
    }
    if (g_prefill_log) std::fprintf(stderr, "tqllm: prefill %d tokens %.1f s: trunk %.1f s (QSA %.1f s), MTP catch-up %.1f s\n", P,
        std::chrono::duration<double>(pclk::now() - p0).count(), trunk_ms / 1000, g_qsa_ms / 1000, mtp_ms / 1000);
    if (!host_res && P >= 256 && chunk >= 256) S().model_->release_prefill();  // single-chunk layer-major prompt
    arena_resize(f_.arena_decode);
    if (P >= 256) {  // long prompt: the old K/V rows to host memory, the refill below takes the freed device memory
        using kclk = std::chrono::steady_clock; const auto k0 = kclk::now();
        if (const size_t freed = S().model_->kv_to_host()) {
            std::fprintf(stderr, "tqllm: attention KV: %.0f MB of old positions moved to host memory in %.0f ms\n", freed / 1048576.0,
                         std::chrono::duration<double, std::milli>(kclk::now() - k0).count());
            // banks an earlier long prompt left below their slot target grow back into the freed memory with the refill
            if (slots_.size() == 1) f_.cache->regrow(size_t(768) << 20);
        }
    }
    stamp("before the bank refill"); refill_borrowed(); g_trim_below = SIZE_MAX; stamp("banks refilled");
    S().model_->set_spec(true);
    ck(cudaDeviceSynchronize()); S().t_decode_ = std::chrono::steady_clock::now();
    if (tqllm::debug_on("mem")) { S().stats_at_decode_ = f_.cache->stats(); S().mtp_stats_at_decode_ = f_.mtp_cache->stats(); }
    S().prefill_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(S().t_decode_ - S().t_begin_).count());
    if (profiling) ck(cudaProfilerStop());
    if (f_.cold_cache) S().cold_at_decode_ = f_.cold_cache->stats();
    if (f_.cold_store) S().cold_store_at_decode_ = f_.cold_store->stats();
    if (f_.cold_fetch_queue) S().cold_fetch_at_decode_ = f_.cold_fetch_queue->stats();
    if (g_profile_decode_req) ck(cudaProfilerStart());
    if (g_arena && std::getenv("TQFN_ARENA_LOG")) std::fprintf(stderr, "tqllm: prefill %d tokens, arena high %.0f MB of %.0f\n", P, g_arena->high / 1048576.0, g_arena->size / 1048576.0);
    S().done_ = int(S().out_.size()) >= S().generate_ || is_eos(S().out_.back());
    (void) c;
}

}  // namespace tqllm::flashnext
