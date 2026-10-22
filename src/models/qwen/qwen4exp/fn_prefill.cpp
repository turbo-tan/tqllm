// Flash-Next layer-major prefill: dense sub-chunks, then the routed experts once per layer from the expert banks and the staged groups.
#include "fn_model.hpp"
#include "decode_rules.hpp"

std::vector<int> DecodeModel::stage_plan(const MixedExpertCache & cache, int layer) {
    std::vector<int> plan; const bool bank = cache.bank_ready(layer);
    for (int e = 0; e < cache.experts(); ++e) if (!bank || !cache.resident(layer, e)) plan.push_back(e);
    return plan;
}

void DecodeModel::stage_alloc(PrefillStage & st, size_t need, int K) {
    if (st.half_bytes >= need && st.K == K) return;
    ck(cudaDeviceSynchronize());
    for (auto & hp : st.half) if (hp) { ck(cudaFree(hp)); hp = nullptr; }
    // + a tail: MMQ reads the last row up to a multiple of 512 columns (a 640-column IQ4_NL down row ran off the end)
    for (auto & hp : st.half) {
        const size_t bytes = need + (size_t(64) << 10);
        if (cudaMalloc(reinterpret_cast<void **>(&hp), bytes) != cudaSuccess) {
            (void) cudaGetLastError();  // blocks freed to the async pool are not visible to cudaMalloc until trimmed
            cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0));
            ck(cudaMalloc(reinterpret_cast<void **>(&hp), bytes));
        }
    }
    st.half_bytes = need; st.K = K; st.layer = -1;
    if (!st.copy) {
        ck(cudaStreamCreateWithFlags(&st.copy, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) { ck(cudaEventCreateWithFlags(&st.ready[b], cudaEventDisableTiming)); ck(cudaEventCreateWithFlags(&st.done[b], cudaEventDisableTiming)); }
    }
    for (int b = 0; b < 2; ++b) ck(cudaEventRecord(st.done[b], nullptr));
}

void DecodeModel::stage_upload(PrefillStage & st, const MixedExpertCache & cache, int layer, int g) {
    const auto lay = cache.bank_layout(layer); size_t off[6]; stage_offsets(lay, st.K, off);
    const int b = g & 1; ck(cudaStreamWaitEvent(st.copy, st.done[b], 0));
    const size_t g0 = size_t(g) * st.K, g1 = std::min(st.plan.size(), g0 + size_t(st.K));
    for (size_t r0 = g0; r0 < g1;) {
        size_t r1 = r0 + 1; while (r1 < g1 && st.plan[r1] == st.plan[r1 - 1] + 1) ++r1;
        const char * src = lay.src0 + size_t(st.plan[r0]) * lay.block; size_t po = 0;
        for (int i = 0; i < 6; ++i) {
            if (lay.piece[i]) ck(cudaMemcpy2DAsync(st.half[b] + off[i] + (r0 - g0) * lay.piece[i], lay.piece[i], src + po, lay.block,
                                                   lay.piece[i], r1 - r0, cudaMemcpyHostToDevice, st.copy));
            po += lay.piece[i];
        }
        st.copy_bytes += double(r1 - r0) * lay.block; r0 = r1;
    }
    ck(cudaEventRecord(st.ready[b], st.copy));
}

void DecodeModel::stage_begin(PrefillStage & st, const MixedExpertCache & cache, int layer) {
    if (st.layer == layer && st.cache == &cache) return;
    st.cache = &cache; st.layer = layer; st.issued = 0;
    // no prepacked file (a GB10 keeps every expert in its bank): nothing to stage, prefill_layer fills the bank instead
    if (!cache.bank_layout(layer).src0) { st.plan.clear(); return; }
    st.plan = stage_plan(cache, layer);
    const int groups = int((st.plan.size() + size_t(st.K) - 1) / size_t(st.K));
    for (; st.issued < std::min(groups, 2); ++st.issued) stage_upload(st, cache, layer, st.issued);
}

void DecodeModel::release_prefill() {
    dense_half_release();
    ck(cudaDeviceSynchronize());
    // TQFN_PF_KEEP: keep the layer-major buffers and staging halves across requests (no free / re-allocate per prompt)
    static const bool keep = std::getenv("TQFN_PF_KEEP") && std::atoi(std::getenv("TQFN_PF_KEEP")) != 0;
    if (!keep) {
        pf_inject.reset(); pf_weights.reset(); pf_sw.reset(); pf_sd.reset(); pf_xr.reset(); pf_ax.reset();
        pf_routed.reset(); pf_idx.reset(); pf_eg.reset(); pf_eu.reset(); pf_ed.reset();
        pf_xq.reset(); pf_axq.reset(); pf_xs.reset(); pf_axs.reset(); pf_xsum.reset(); pf_axsum.reset();
    }
    auto & st = stage();
    if (!keep) { for (auto & hp : st.half) if (hp) { ck(cudaFree(hp)); hp = nullptr; } st.half_bytes = 0; st.K = 0; }
    st.layer = -1; st.cache = nullptr; st.plan.clear(); st.issued = 0;
    g_qsa_batch.reset(); tqllm_cuda::fn_gdn_release_workspaces(); ++g_prefill_epoch;
    if (res_keep && res_keep->count > size_t(64) * width) { PersistentAlloc keep; res_keep.reset(); res_keep = std::make_unique<Device<float>>(size_t(64) * width); }
    size_t fr = 0, tot = 0; ck(cudaMemGetInfo(&fr, &tot));
    if (fr < g_trim_below) { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
}

void DecodeModel::prefill_layer(int layer, float * wide, int n, const float * ple_rows, MixedExpertCache & cache, AttentionState * att_override) {
    const int sub = std::max(16, g_sub_req > 0 ? std::min(g_tune.sub, g_sub_req) : g_tune.sub), stage_k = std::max(1, g_tune.stage);
    const int K = c.topk, A = n * K, E = c.experts, ff = c.expert_ff;
    auto & st = stage();
    if (!st.copy || st.K != stage_k) {  // staging banks sized for the largest expert block of any layer
        size_t need = 0;
        for (int l = 0; l < c.layers; ++l) {
            if (!cache.bank_ready(l) && !cache.has_layout(l)) continue;
            size_t off[6]; const auto lay = cache.bank_layout(l); stage_offsets(lay, stage_k, off); need = std::max(need, off[5] + align256(size_t(stage_k) * lay.piece[5]));
        }
        if (need) stage_alloc(st, need, stage_k);
    }
    grow(pf_inject, size_t(n) * c.hc); grow(pf_weights, size_t(A)); grow(pf_sw, size_t(n)); grow(pf_xr, size_t(n) * h);
    grow(pf_routed, size_t(A)); grow(pf_eg, size_t(A) * ff); grow(pf_eu, size_t(A) * ff); grow(pf_ax, size_t(A) * ff);
    grow(pf_ed, size_t(A) * h); grow(pf_sd, size_t(n) * h);
    static const int pf_int8 = std::getenv("TQFN_PREFILL_INT8") ? std::atoi(std::getenv("TQFN_PREFILL_INT8")) : 0;  // 1 per-32 scales, 2 per token
    if (pf_int8) { grow(pf_xq, size_t(n) * h); grow(pf_xs, size_t(n) * (h / 32)); grow(pf_xsum, size_t(n) * (h / 32) * 2);
                   grow(pf_axq, size_t(A) * ff); grow(pf_axs, size_t(A) * (ff / 32)); grow(pf_axsum, size_t(A) * (ff / 32) * 2); }
    // a layer's streamed experts are queued before its routing is known (borrowed layers too: no bank, every expert
    // streamed), so their copies overlap the dense part instead of starting after the routing readback
    auto stage_early = [&](int l) {
        return cache.bank_ready(l) || (size_t(l) < pf_borrowed.size() && pf_borrowed[size_t(l)] && cache.has_layout(l));
    };
    if (st.copy && stage_early(layer)) stage_begin(st, cache, layer);  // usually queued by the previous layer already
    if (tqllm::debug_on("mem")) {
        size_t fr = 0, tot = 0; ck(cudaMemGetInfo(&fr, &tot));
        cudaMemPool_t pool; uint64_t res = 0, used = 0; ck(cudaDeviceGetDefaultMemPool(&pool, 0));
        ck(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &res)); ck(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
        std::fprintf(stderr, "MEM layer %d n %d free %.0f MB banks %.0f MB staging %.0f MB pool %.0f/%.0f MB dev %.0f MB\n", layer, n,
                     fr / 1048576.0, cache.stats().device_bytes / 1048576.0, 2.0 * st.half_bytes / 1048576.0, used / 1048576.0,
                     res / 1048576.0, tqllm_cuda::dev_bytes_counter().load() / 1048576.0);
    }
    using lclk = std::chrono::steady_clock; const auto lt0 = lclk::now(); double lt[4] = {};
    auto lmark = [&](int i) { if (!g_prefill_log) return; ck(cudaDeviceSynchronize()); lt[i] = std::chrono::duration<double, std::milli>(lclk::now() - lt0).count(); };
    if (st.ids_cap < size_t(A)) { if (st.ids_host) ck(cudaFreeHost(st.ids_host)); st.ids_cap = std::max({size_t(A), 2 * st.ids_cap, size_t(1) << 20});
                                 ck(cudaMallocHost(reinterpret_cast<void **>(&st.ids_host), st.ids_cap * 4)); }
    const PrefillSink base{wide, pf_inject->p, pf_xr->p, pf_routed->p, pf_weights->p, pf_sw->p, pf_sd->p, 0};
    for (int d = 0; d < n; d += sub) {
        const int nd = std::min(sub, n - d); ArenaMark scratch;
        auto rows = std::make_unique<Device<float>>(Device<float>::View{}, wide + size_t(d) * width, size_t(nd) * width);
        PrefillSink sk = base; sk.off = d;
        layer_forward(layer, rows, nd, ple_rows ? ple_rows + size_t(d) * h : nullptr, att_override, cache, nullptr, &sk);
    }
    lmark(0);
    ck(cudaMemcpyAsync(st.ids_host, pf_routed->p, size_t(A) * 4, cudaMemcpyDeviceToHost, nullptr)); ck(cudaStreamSynchronize(nullptr));
    ck(cudaGetLastError());  // a launch that failed (full device) leaves stale routing behind: stop before it is used
    const int * ids = st.ids_host;
    { int bad = 0, first = -1; for (int a = 0; a < A; ++a) if (ids[a] < 0 || ids[a] >= E) { if (!bad) first = a; ++bad; }
      require(!bad, "layer-major prefill: layer " + std::to_string(layer) + " routed " + std::to_string(bad) + " of " + std::to_string(A) +
                    " assignments to an expert outside 0.." + std::to_string(E - 1) + " (first at " + std::to_string(first) + ": " +
                    std::to_string(bad ? ids[first] : 0) + ")"); }
    std::vector<int> cnt(size_t(E), 0); for (int a = 0; a < A; ++a) ++cnt[size_t(ids[a])];
    const bool borrowed = size_t(layer) < pf_borrowed.size() && pf_borrowed[size_t(layer)] && cache.has_layout(layer);
    if (!cache.bank_ready(layer) && borrowed) {  // every routed expert from staging; refilled right after the prompt
        pf_refill[layer] = tqllm::flashnext::bank_experts(cnt, layer, size_t(cache.slots(layer)), g_cpu_owned);  // most routed first: the refill keeps what its bank holds
        size_t off[6]; const auto lay = cache.bank_layout(layer); stage_offsets(lay, stage_k, off);
        const size_t need = off[5] + align256(size_t(stage_k) * lay.piece[5]);
        if (!st.copy || st.K != stage_k || need > st.half_bytes) stage_alloc(st, std::max(st.half_bytes, need), stage_k);
        if (st.layer != layer || st.cache != &cache) { st.cache = &cache; st.layer = layer; st.plan.clear(); st.issued = 0; }
    } else if (!cache.bank_ready(layer)) {  // first use of this layer: its bank takes the most routed experts (later decode hits)
        std::vector<int> top = tqllm::flashnext::bank_experts(cnt, layer, size_t(cache.plan_slots(layer)), g_cpu_owned); std::sort(top.begin(), top.end());
        cache.prepare(layer, top); ck(cudaStreamSynchronize(nullptr));
        { size_t off[6]; const auto lay = cache.bank_layout(layer); stage_offsets(lay, stage_k, off);
          const size_t need = off[5] + align256(size_t(stage_k) * lay.piece[5]);
          if (!st.copy || st.K != stage_k || need > st.half_bytes) stage_alloc(st, std::max(st.half_bytes, need), stage_k); }
        stage_begin(st, cache, layer);
    }
    const auto lay = cache.bank_layout(layer);
    size_t need = 0; { size_t off[6]; stage_offsets(lay, st.K, off); need = off[5] + align256(size_t(st.K) * lay.piece[5]); }
    require(need <= st.half_bytes, "layer-major prefill: staging bank too small for this layer");
    // the groups already queued stay (copied before the routing was known); the rest of the plan is the routed experts
    // the bank does not hold
    st.plan.resize(std::min(st.plan.size(), size_t(st.issued) * size_t(st.K)));
    std::vector<char> planned(size_t(E), 0); for (int e : st.plan) planned[size_t(e)] = 1;
    std::vector<int> hot;
    // without a prepacked file every routed expert goes through the bank (prepare loads the missing ones)
    for (int e = 0; e < E; ++e) if (cnt[size_t(e)] && !planned[size_t(e)]) (cache.resident(layer, e) || !lay.src0 ? hot : st.plan).push_back(e);
    require(st.plan.empty() || lay.src0, "layer-major prefill: experts to stream but no prepacked expert file (--experts)");
    const int groups = int((st.plan.size() + size_t(st.K) - 1) / size_t(st.K));
    for (; st.issued < std::min(groups, 2); ++st.issued) stage_upload(st, cache, layer, st.issued);
    // per-slot assignment lists of every pass, uploaded once: [counts W][groups W x cap]
    struct Pass { int W, cap; size_t at; std::vector<int> slot_of; };
    std::vector<Pass> passes; size_t total = 0;
    auto plan_pass = [&](const std::vector<int> & experts, const std::vector<int> & slots, int W) {
        Pass ps{W, 1, 0, std::vector<int>(size_t(E), -1)};
        for (size_t i = 0; i < experts.size(); ++i) { ps.slot_of[size_t(experts[i])] = slots[i]; ps.cap = std::max(ps.cap, cnt[size_t(experts[i])]); }
        ps.at = total; total += size_t(W) * (1 + ps.cap); passes.push_back(std::move(ps));
    };
    if (!hot.empty()) { const int cap = cache.plan_slots(layer); const auto slots = cache.prepare(layer, hot); plan_pass(hot, slots, cap); }
    for (int g = 0; g < groups; ++g) {
        const size_t g0 = size_t(g) * st.K, g1 = std::min(st.plan.size(), g0 + size_t(st.K));
        std::vector<int> ex, sl;
        for (size_t i = g0; i < g1; ++i) if (cnt[size_t(st.plan[i])]) { ex.push_back(st.plan[i]); sl.push_back(int(i - g0)); }
        plan_pass(ex, sl, st.K);
    }
    if (st.idx_cap < total) { if (st.idx_host) ck(cudaFreeHost(st.idx_host)); st.idx_cap = std::max({total, 2 * st.idx_cap, size_t(4) << 20});
                              ck(cudaMallocHost(reinterpret_cast<void **>(&st.idx_host), st.idx_cap * 4)); }
    for (const auto & ps : passes) {
        int * cn = st.idx_host + ps.at, * gr = cn + ps.W; std::fill(cn, cn + ps.W, 0);
        for (int a = 0; a < A; ++a) { const int sl = ps.slot_of[size_t(ids[a])]; if (sl >= 0) gr[size_t(sl) * ps.cap + size_t(cn[sl]++)] = a; }
    }
    grow(pf_idx, total);
    ck(cudaMemcpyAsync(pf_idx->p, st.idx_host, total * 4, cudaMemcpyHostToDevice, nullptr));
    lmark(1);
    auto gemm = [&](const Mat & mm, const __half * x, __nv_bfloat16 * y, int rows, const Pass & ps, bool per_expert) {
        require(mm.fmt->expert_gemm(mm, x, y, rows, pf_idx->p + ps.at, pf_idx->p + ps.at + ps.W, ps.cap, ps.W, K, per_expert, nullptr),
                std::string("layer-major prefill: no grouped expert kernel for ") + mm.fmt->name() + " (TQFN_LAYER_MAJOR=0)");
    };
    if (pf_int8) tqllm_cuda::k_quant_act_rows(pf_xr->p, nullptr, nullptr, 0, 0, n, h, pf_xq->p, pf_xs->p, pf_xsum->p, nullptr, pf_int8);
    auto gemm_i8 = [&](const Mat & mm, const int8_t * xq, const float * xs, const int16_t * xsum, __nv_bfloat16 * y, int rows, const Pass & ps, bool per_expert) {
        return pf_int8 && mm.fmt->expert_gemm_i8(mm, xq, xs, xsum, y, rows, pf_idx->p + ps.at, pf_idx->p + ps.at + ps.W, ps.cap, ps.W, K, per_expert, nullptr, pf_int8);
    };
    // llama.cpp's MMQ on our banks (src/experts/mmq_experts.cu) for the formats it covers, the grouped fp16 / int8 kernels for
    // the rest. Gate/up read the layer's tokens quantised once; the down input is quantised per pass into pf_eg (dead
    // after the swiglu: the next pass rewrites its own gate rows first).
    auto mmq_on = [&](const Mat & mm) { return tqllm_cuda::mmq_supports(mm.fmt->u4h_id(), mm.cols); };
    // the q8_1 tiles of the gate/up input and the tile lists live in the prefill arena: the dense part handed it back
    ArenaMark mmq_scratch; std::unique_ptr<Device<uint8_t>> pf_mq; std::unique_ptr<Device<int2>> pf_tiles;
    {
        size_t nt = 1; for (const auto & ps : passes) nt = std::max(nt, size_t(tqllm_cuda::mmq_max_tiles(ps.cap, ps.W)));
        pf_mq = std::make_unique<Device<uint8_t>>(tqllm_cuda::mmq_act_bytes(n, h)); pf_tiles = std::make_unique<Device<int2>>(nt);
        tqllm_cuda::mmq_quantize_act(pf_xr->p, n, h, pf_mq->p, nullptr);
    }
    static int mmq_checks = std::getenv("TQFN_PF_MMQ_CHECK") ? std::atoi(std::getenv("TQFN_PF_MMQ_CHECK")) : 0;
    auto gemm_mmq = [&](const Mat & mm, const void * yq, int ny, int div, __nv_bfloat16 * y, int rows, const Pass & ps,
                        const __nv_bfloat16 * gate = nullptr, __half * act = nullptr, int fmt_override = -1) {
        const int id = fmt_override >= 0 ? fmt_override : mm.fmt->u4h_id();
        if (!tqllm_cuda::mmq_supports(id, mm.cols)) return false;
        const bool ok = tqllm_cuda::mmq_expert_gemm(id, static_cast<const uint8_t *>(mm.p), mm.aux, fmt_override < 0 && g_tile_experts && tqllm_cuda::tiled_block_bytes(id),
                                           rows, mm.cols, pf_idx->p + ps.at, st.idx_host + ps.at, pf_idx->p + ps.at + ps.W, ps.cap, ps.W,
                                           div, yq, ny, y, pf_tiles->p, int(pf_tiles->count), nullptr, gate, act);
        // TQFN_PF_MMQ_CHECK=N: up to N MMQ GEMMs against the fp16 grouped kernel on the same pass, the first of each
        // (format, projection shape) anywhere in the model
        static std::set<std::pair<int, int>> mmq_checked;
        if (ok && mmq_checks > 0 && mmq_checked.insert({id, rows}).second) {
            --mmq_checks;
            const __half * x = div == 1 ? pf_ax->p : pf_xr->p;
            ArenaMark check_scratch;
            Device<__nv_bfloat16> ref(size_t(A) * rows);
            require(mm.fmt->expert_gemm(mm, x, ref.p, rows, pf_idx->p + ps.at, pf_idx->p + ps.at + ps.W, ps.cap, ps.W, K, div == 1, nullptr), "check: fp16 kernel");
            std::vector<__nv_bfloat16> a(size_t(A) * rows), b(a.size());
            ck(cudaMemcpy(a.data(), y, a.size() * 2, cudaMemcpyDeviceToHost)); ck(cudaMemcpy(b.data(), ref.p, b.size() * 2, cudaMemcpyDeviceToHost));
            double num = 0, den = 0; size_t cols = 0;
            const int * cn = st.idx_host + ps.at, * gr = cn + ps.W;
            for (int z = 0; z < ps.W; ++z) for (int j = 0; j < cn[z]; ++j, ++cols) {
                const size_t o = size_t(gr[size_t(z) * ps.cap + j]) * rows;
                for (int r = 0; r < rows; ++r) { const double u = __bfloat162float(a[o + r]), v = __bfloat162float(b[o + r]); num += (u - v) * (u - v); den += v * v; }
            }
            std::fprintf(stderr, "MMQ_CHECK layer %d fmt %d rows %d k %d tiled %d slots %d cols %zu: rel L2 %.4g\n", layer, id, rows, mm.cols,
                         int(g_tile_experts && tqllm_cuda::tiled_block_bytes(id)), ps.W, cols, std::sqrt(num / std::max(den, 1e-30)));
        }
        return ok;
    };
    auto run_pass = [&](const Mat (&m)[3], const Pass & ps) {
        // int8 per projection where the format has the kernel; the fp16 kernel otherwise (same inputs, same outputs)
        const int8_t * xq = pf_int8 ? pf_xq->p : nullptr; const float * xs = pf_int8 ? pf_xs->p : nullptr; const int16_t * xsum = pf_int8 ? pf_xsum->p : nullptr;
        if (!gemm_mmq(m[0], pf_mq->p, n, K, pf_eg->p, ff, ps, nullptr, nullptr, -1) && !gemm_i8(m[0], xq, xs, xsum, pf_eg->p, ff, ps, false))
            gemm(m[0], pf_xr->p, pf_eg->p, ff, ps, false);
        // up on MMQ: the swiglu runs in its epilogue (pf_ax written directly, pf_eu untouched); the other paths keep the pass
        if (!gemm_mmq(m[1], pf_mq->p, n, K, pf_eu->p, ff, ps, pf_eg->p, pf_ax->p, -1)) {
            if (!gemm_i8(m[1], xq, xs, xsum, pf_eu->p, ff, ps, false)) gemm(m[1], pf_xr->p, pf_eu->p, ff, ps, false);
            tqllm_cuda::fn_swiglu_bf16_h_groups(pf_eg->p, pf_eu->p, pf_ax->p, pf_idx->p + ps.at, pf_idx->p + ps.at + ps.W, ps.W, ps.cap, ff, nullptr);
        }
        if (mmq_on(m[2])) {
            static_assert(sizeof(__nv_bfloat16) == 2, "pf_eg holds the down q8_1 tiles");
            require(tqllm_cuda::mmq_act_bytes(A, ff) <= size_t(A) * ff * 2, "layer-major MMQ: down tiles larger than pf_eg");
            tqllm_cuda::mmq_quantize_act_rows(pf_ax->p, A, ff, pf_idx->p + ps.at + ps.W, pf_idx->p + ps.at, ps.cap, ps.W, pf_eg->p, nullptr);
            require(gemm_mmq(m[2], pf_eg->p, A, 1, pf_ed->p, h, ps, nullptr, nullptr, -1), "layer-major MMQ: down GEMM failed");
            return;
        }
        if (pf_int8) tqllm_cuda::k_quant_act_rows(pf_ax->p, pf_idx->p + ps.at + ps.W, pf_idx->p + ps.at, ps.cap, ps.W, 0, ff, pf_axq->p, pf_axs->p, pf_axsum->p, nullptr, pf_int8);
        if (!gemm_i8(m[2], pf_int8 ? pf_axq->p : nullptr, pf_int8 ? pf_axs->p : nullptr, pf_int8 ? pf_axsum->p : nullptr, pf_ed->p, h, ps, true)) gemm(m[2], pf_ax->p, pf_ed->p, h, ps, true);
    };
    size_t pi = 0;
    if (!hot.empty()) { const Mat m[3] = {cache.bank_mat(layer, 0), cache.bank_mat(layer, 1), cache.bank_mat(layer, 2)}; run_pass(m, passes[pi++]); }
    size_t off[6]; stage_offsets(lay, st.K, off);
    const int next = layer + 1 < c.layers ? layer + 1 : -1;
    for (int g = 0; g < groups; ++g) {
        const int b = g & 1; ck(cudaStreamWaitEvent(nullptr, st.ready[b], 0));
        const Pass & ps = passes[pi++];
        if (ps.cap > 0 && std::any_of(st.idx_host + ps.at, st.idx_host + ps.at + ps.W, [](int x) { return x > 0; })) {
            Mat m[3];
            for (int q = 0; q < 3; ++q) {
                m[q] = cache.bank_mat(layer, q); m[q].p = st.half[b] + off[2 * q]; m[q].rows = st.K * (q == 2 ? h : ff);
                if (lay.piece[2 * q + 1]) m[q].aux = st.half[b] + off[2 * q + 1];
            }
            run_pass(m, ps);
        }
        ck(cudaEventRecord(st.done[b], nullptr));
        if (g + 2 < groups) { stage_upload(st, cache, layer, g + 2); st.issued = g + 3; }
    }
    lmark(2);
    // queue the next layer's first groups now: they land while this layer finishes and the next one's dense part runs
    if (next >= 0 && stage_early(next)) stage_begin(st, cache, next); else st.layer = -1;
    // combine and the hyper-connection update per sub-chunk (rows of `wide` replaced)
    for (int d = 0; d < n; d += sub) {
        const int nd = std::min(sub, n - d); ArenaMark scratch;
        Device<float> fo(size_t(nd) * h), upd(size_t(nd) * width);
        tqllm_cuda::fn_moe_combine(pf_ed->p + size_t(d) * K * h, pf_weights->p + size_t(d) * K, pf_sd->p + size_t(d) * h, pf_sw->p + d, fo.p, h, K, nd, nullptr);
        float * w = wide + size_t(d) * width;
        tqllm_cuda::hc_combine(w, fo.p, pf_inject->p + size_t(d) * c.hc, upd.p, h, c.hc, nd, nullptr);
        ck(cudaMemcpyAsync(w, upd.p, size_t(nd) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
    }
    if (g_prefill_log) { lmark(3); if (lt[3] > 0)
        std::fprintf(stderr, "tqllm: prefill layer %d n %d %.0f ms: dense %.0f, routing + plan %.0f, experts %.0f, combine %.0f\n",
                     layer, n, lt[3], lt[0], lt[1] - lt[0], lt[2] - lt[1], lt[3] - lt[2]); }
}
