// Flash-Next layer forward (decode, verify and short prefill chunks) and the router.
#include "fn_model.hpp"
#include "decode_rules.hpp"
#include <map>
#include <tuple>
#include "hot_cold_policy.hpp"

// A step whose captured part fails (an operation not allowed in a capture: a weight uploaded on its first use, a handle
// created) runs again without a graph; nothing in an open capture has executed, so the plain run starts from the same
// state. Its layer key then takes two plain calls again before the next capture.
void DecodeModel::layer_forward(int layer, std::unique_ptr<Device<float>> & residual, int n, const float * ple_rows,
                   AttentionState * attn, MixedExpertCache & cache, const std::vector<Seg> * segs,
                   const PrefillSink * sink) {
    try { layer_forward_step(layer, residual, n, ple_rows, attn, cache, segs, sink, false); }
    catch (const std::exception & e) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) != cudaSuccess || cs == cudaStreamCaptureStatusNone) throw;
        cudaGraph_t cg = nullptr; (void) cudaStreamEndCapture(cudaStreamPerThread, &cg);
        if (cg) (void) cudaGraphDestroy(cg);
        (void) cudaGetLastError();
        static int logged = 0;
        if (logged++ < 4) std::fprintf(stderr, "tqllm: layer %d graph capture failed (%s); the step runs without a graph\n", layer, e.what());
        layer_forward_step(layer, residual, n, ple_rows, attn, cache, segs, sink, true);
    }
}

void DecodeModel::layer_forward_step(int layer, std::unique_ptr<Device<float>> & residual, int n, const float * ple_rows,
                   AttentionState * attn, MixedExpertCache & cache, const std::vector<Seg> * segs,
                   const PrefillSink * sink, bool plain) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    // The part of a decode / verify step before the routing readback (PLE, norms, mixer, attention or GDN core, routers,
    // top-k) is captured into a CUDA graph each step, updated in place and launched. Unified-memory GPUs only: GB10
    // decode +5-6 % (text identical); on PCIe cards the per-step re-capture of ~48 graphs costs more host time than the
    // launch gaps it removes (3090 + Ryzen 5950X neutral, RTX 5090 + EPYC 7543 chat 128 -> 154, 64K 118 -> 134 tok/s with
    // graphs off). One sequence only (the multi-slot path interleaves per-sequence state).
    static const bool graph_on = [] { int dev = 0, on = 0; cudaGetDevice(&dev);
        return cudaDeviceGetAttribute(&on, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) == cudaSuccess && on; }();
    // Every step updates all kernel parameters (weights included), so layers of one kind share an exec, keyed by its
    // topology: the node count changes with the position in attention layers and with the draft in the MTP layer.
    // One exec per layer and variant (~11 x 49) held ~1.3 GB the expert banks had taken (out of memory on a 3090).
    struct GraphKey { const void * model; int kind, n; size_t nodes; bool operator<(const GraphKey & o) const { return std::tie(model, kind, n, nodes) < std::tie(o.model, o.kind, o.n, o.nodes); } };
    static std::map<GraphKey, int> calls;  // kind = layer here: calls 1-2 of each layer run plain
    struct GraphSlot { cudaGraphExec_t exec = nullptr; cudaEvent_t done = nullptr; };
    static std::map<GraphKey, std::vector<GraphSlot>> graphs;
    const GraphKey gkey{this, layer, n, 0};
    bool capturing = false;
    if (graph_on && spec && !segs && !sink && n <= 8 && g_arena && !g_persistent) {
        static int epoch = -1; if (epoch != g_prefill_epoch) { epoch = g_prefill_epoch; calls.clear(); }  // lazy scratch is rebuilt by plain calls
        if (plain) calls[gkey] = 0;
        if (++calls[gkey] > 2) { ck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); capturing = true; }  // calls 1-2 run plain: lazy allocations happen outside capture
    }
    const bool trunk = layer < c.layers, attention = !trunk || c.compress_ratios[layer];
    Device<float> norm(size_t(width) * n), mixed(size_t(h) * n), inject(size_t(c.hc) * n), block(size_t(h) * n), combined(size_t(width) * n);
    if (trunk && ple_rows && std::find(c.ple_layers.begin(), c.ple_layers.end(), layer) != c.ple_layers.end()) {
        Device<float> key(size_t(width) * n), value(size_t(h) * n), kn(key.count), qn(key.count), gates(size_t(c.hc) * n), gated(key.count);
        {
            const Checkpoint::SharedInput shared_in(cp, ple_rows, size_t(n) * h);
            cp.project(prefix + "ple_key.weight", ple_rows, key.p, h, width, n);
            cp.project(prefix + "ple_value.weight", ple_rows, value.p, h, h, n);
        }
        tqllm_cuda::hc_norm(key.p, cp.dense_device(prefix + "ple_norm_key.weight"), kn.p, h, c.hc, n, float(cp.epsilon), nullptr);
        tqllm_cuda::hc_norm(residual->p, cp.dense_device(prefix + "ple_norm_query.weight"), qn.p, h, c.hc, n, float(cp.epsilon), nullptr);
        tqllm_cuda::ple_gate(kn.p, qn.p, gates.p, h, c.hc, n, nullptr);
        tqllm_cuda::ple_broadcast(value.p, gates.p, gated.p, h, c.hc, n, nullptr);
        const int hist = (c.ple_kernel - 1) * c.ple.ngram;
        Device<float> pn(key.count), conv(key.count);
        auto out = std::make_unique<Device<float>>(key.count);
        tqllm_cuda::hc_norm(gated.p, cp.dense_device(prefix + "ple_norm_conv.weight"), pn.p, h, c.hc, n, float(cp.epsilon), nullptr);
        auto ple_state = [&](DecodeModel & m, int off, int nn) {  // the conv history is per sequence
            if (!m.ple_past[layer]) { PersistentAlloc keep; m.ple_past[layer] = std::make_unique<Device<float>>(size_t(hist) * width); ck_zero(*m.ple_past[layer]); }
            Device<float> next(m.ple_past[layer]->count);
            const size_t rows = size_t(nn) * width; const float * pseg = pn.p + size_t(off) * width;
            if (m.spec) {
                PersistentAlloc keep;
                if (!m.ple_old[layer]) m.ple_old[layer] = std::make_unique<Device<float>>(m.ple_past[layer]->count);
                if (!m.ple_saved[layer] || m.ple_saved[layer]->count < rows) m.ple_saved[layer] = std::make_unique<Device<float>>(rows);
                ck_copy_n(*m.ple_old[layer], m.ple_past[layer]->p, m.ple_past[layer]->count);
                ck_copy_n(*m.ple_saved[layer], pseg, rows);
            }
            tqllm_cuda::ple_conv(pseg, m.ple_past[layer]->p, cp.dense_device(prefix + "ple_conv1d.weight"), conv.p + size_t(off) * width, next.p,
                                 width, nn, c.ple_kernel, c.ple.ngram, nullptr);
            ck_copy(*m.ple_past[layer], next);
        };
        if (segs) for (const auto & sg : *segs) ple_state(*sg.m, sg.off, sg.n);
        else ple_state(*this, 0, n);
        tqllm_cuda::ple_apply(residual->p, gated.p, conv.p, out->p, int(out->count), nullptr); residual = std::move(out);
    }
    // the norms write their BF16 image for the mixer's BF16 projections (no f32 -> bf16 passes); prefill sub-chunks of more
    // than 64 tokens with a BF16 mixer keep no fp32 normalized rows at all (row scales only, the mix recomputes them; exact)
    auto normb = std::make_unique<Device<__nv_bfloat16>>(size_t(width) * n);
    const bool hc_rs = tqllm::hc_row_scales_only(n, cp.is_bf16(prefix + "hc_attn_down.weight") && cp.is_bf16(prefix + "hc_ffn_down.weight"),
                                                 cp.is_bf16(prefix + "hc_attn_inject.weight") && cp.is_bf16(prefix + "hc_ffn_inject.weight"));
    std::unique_ptr<Device<float>> rs; if (hc_rs) rs = std::make_unique<Device<float>>(size_t(c.hc) * n);
    const float * attn_gamma = cp.dense_device(prefix + "hc_attn_norm.weight");
    if (hc_rs) tqllm_cuda::hc_norm_rs_bf16(residual->p, attn_gamma, rs->p, normb->p, h, c.hc, n, float(cp.epsilon), nullptr);
    else tqllm_cuda::hc_norm_bf16(residual->p, attn_gamma, norm.p, normb->p, h, c.hc, n, float(cp.epsilon), nullptr);
    mix_device(cp, prefix + "hc_attn_", norm.p, n, mixed.p, inject.p, normb->p,
               hc_rs ? residual->p : nullptr, hc_rs ? rs->p : nullptr, hc_rs ? attn_gamma : nullptr);
    if (segs) {  // row-wise projections once for every sequence of the batch
        // row-wise projections once for all sequences; only the stateful middle (conv + recurrence, KV + attention) per sequence
        const std::string pf = "blk." + std::to_string(layer) + ".";
        if (attention) {
            const int heads = cp.attention_heads, kvh = cp.attention_kvheads, ad = cp.attention_dim;
            Device<float> qgB(size_t(n) * heads * ad * 2), kB(size_t(n) * kvh * ad), vB(kB.count), aoB(size_t(n) * heads * ad);
            {
                const Checkpoint::SharedInput shared_in(cp, mixed.p, size_t(n) * h);
                cp.project(pf + "attn_q.weight", mixed.p, qgB.p, h, heads * ad * 2, n);
                cp.project(pf + "attn_k.weight", mixed.p, kB.p, h, kvh * ad, n);
                cp.project(pf + "attn_v.weight", mixed.p, vB.p, h, kvh * ad, n);
            }
            for (const auto & sg : *segs) {
                const AttnPre pre{qgB.p + size_t(sg.off) * heads * ad * 2, kB.p + size_t(sg.off) * kvh * ad, vB.p + size_t(sg.off) * kvh * ad};
                attention_core(cp, mixed.p + size_t(sg.off) * h, nullptr, sg.n, layer, nullptr, trunk ? &sg.m->att[layer] : &sg.m->mtp_att,
                               &pre, aoB.p + size_t(sg.off) * heads * ad);
            }
            tqllm_cuda::fn_attention_gate(aoB.p, qgB.p, heads, ad, n, nullptr);
            cp.project(pf + "attn_output.weight", aoB.p, block.p, heads * ad, h, n);
        } else {
            const int kh = cp.key_heads, vh = cp.value_heads, sd = cp.state_dim, channels = (2 * kh + vh) * sd;
            Device<float> qkvB(size_t(n) * channels), zB(size_t(n) * vh * sd), aB(size_t(n) * vh), bB(aB.count), roB(zB.count), gB(zB.count);
            {
                const Checkpoint::SharedInput shared_in(cp, mixed.p, size_t(n) * h);
                cp.project(pf + "attn_qkv.weight", mixed.p, qkvB.p, h, channels, n);
                cp.project(pf + "attn_gate.weight", mixed.p, zB.p, h, vh * sd, n);
                cp.project(pf + "ssm_alpha.weight", mixed.p, aB.p, h, vh, n);
                cp.project(pf + "ssm_beta.weight", mixed.p, bB.p, h, vh, n);
            }
            for (const auto & sg : *segs) {
                const GdnPre pre{qkvB.p + size_t(sg.off) * channels, zB.p + size_t(sg.off) * vh * sd, aB.p + size_t(sg.off) * vh, bB.p + size_t(sg.off) * vh};
                recurrent_core(cp, nullptr, nullptr, sg.n, layer, nullptr, &sg.m->rec[layer], &pre, roB.p + size_t(sg.off) * vh * sd);
            }
            tqllm_cuda::fn_gated_rms(roB.p, zB.p, cp.dense_device(pf + "ssm_norm.weight"), gB.p, vh, sd, n, float(cp.epsilon), nullptr);
            cp.project(pf + "ssm_out.weight", gB.p, block.p, vh * sd, h, n);
        }
    } else if (segs) {
        for (const auto & sg : *segs) {
            const size_t o = size_t(sg.off) * h;
            if (attention) attention_core(cp, mixed.p + o, block.p + o, sg.n, layer, nullptr, trunk ? &sg.m->att[layer] : &sg.m->mtp_att);
            else recurrent_core(cp, mixed.p + o, block.p + o, sg.n, layer, nullptr, &sg.m->rec[layer]);
        }
    } else if (attention) attention_core(cp, mixed.p, block.p, n, layer, nullptr, attn ? attn : &att[layer]);
    else recurrent_core(cp, mixed.p, block.p, n, layer, nullptr, &rec[layer]);
    // a prefill sub-chunk writes its combined rows straight into the layer's wide rows (combine is elementwise, so in place
    // over the residual view is safe)
    float * comb = sink ? sink->wide + size_t(sink->off) * width : combined.p;
    if (hc_rs) tqllm_cuda::hc_combine_norm_rs_bf16(residual->p, block.p, inject.p, comb, cp.dense_device(prefix + "hc_ffn_norm.weight"), rs->p, normb->p, h, c.hc, n, float(cp.epsilon), nullptr);
    else {
        tqllm_cuda::hc_combine(residual->p, block.p, inject.p, comb, h, c.hc, n, nullptr);
        tqllm_cuda::hc_norm_bf16(comb, cp.dense_device(prefix + "hc_ffn_norm.weight"), norm.p, normb->p, h, c.hc, n, float(cp.epsilon), nullptr);
    }
    mix_device(cp, prefix + "hc_ffn_", norm.p, n, mixed.p, inject.p, normb->p,
               hc_rs ? comb : nullptr, hc_rs ? rs->p : nullptr, hc_rs ? cp.dense_device(prefix + "hc_ffn_norm.weight") : nullptr);
    Device<float> logits(size_t(n) * (c.experts + 1)), weights(size_t(n) * c.topk), shared_weights(n);
    Device<int> routed(weights.count);
    // TQFN_PREDICT_N=N (default 4): next-layer prediction (prefetch) for verify blocks of up to N tokens; above 4 the routers
    // run unfused (placement only: the output does not change)
    static const int predict_n = std::getenv("TQFN_PREDICT_N") ? std::atoi(std::getenv("TQFN_PREDICT_N")) : 4;
    const bool predict = (!g_hot_map || bool(cp.cold_prefetch)) && prefetch_on && n <= predict_n && trunk && layer + 1 < c.layers;
    const std::string np = "blk." + std::to_string(layer + 1) + ".";
    std::unique_ptr<Device<float>> nl;
    if (predict) nl = std::make_unique<Device<float>>(size_t(n) * (c.experts + 1));
    // Both routers + shared gates in one segmented launch and both top-k in one launch (same per-row and per-token
    // arithmetic: test-gemv-seg, test-topk2; +2.7 % 3090 5-chat at --slots 168)
    auto is_bf16 = [&](const std::string & pre) { return cp.file(pre + "ffn_gate_inp.weight").file().tensors.at(pre + "ffn_gate_inp.weight").type == GGML_BF16; };
    static const bool f32_router = std::getenv("TQFN_F32_ROUTER") != nullptr;
    const bool fused = !f32_router && n <= 8 && c.experts <= 1024 && h >= 1024 && is_bf16(prefix) && (!predict || is_bf16(np));
    if (fused) {
        tqllm_cuda::GemvSegs g; int rows = 0;
        auto add = [&](const std::string & name, int r, float * y) {
            g.w[g.n] = cp.bf16_device(name); g.y[g.n] = y; g.ld[g.n] = c.experts + 1; g.start[g.n] = rows; rows += r; g.start[++g.n] = rows;
        };
        add(prefix + "ffn_gate_inp.weight", c.experts, logits.p); add(prefix + "ffn_gate_inp_shexp.weight", 1, logits.p + c.experts);
        if (predict) { add(np + "ffn_gate_inp.weight", c.experts, nl->p); add(np + "ffn_gate_inp_shexp.weight", 1, nl->p + c.experts); }
        tqllm_cuda::fn_gemv_bf16_segments(g, mixed.p, h, n, nullptr);
    } else {
        router(prefix, mixed.p, logits.p, n);
        if (predict) router(np, mixed.p, nl->p, n);
    }
    if (g_keep_mask && trunk) {  // TQFN_KEEP_MAP: pruned experts never win the top-k
        tqllm_cuda::logits_mask(logits.p, g_keep_mask + size_t(layer) * c.experts, n, c.experts + 1, c.experts, nullptr);
        if (predict) tqllm_cuda::logits_mask(nl->p, g_keep_mask + size_t(layer + 1) * c.experts, n, c.experts + 1, c.experts, nullptr);
    }
    std::unique_ptr<Device<int>> next_ids;
    if (predict) {
        Device<float> nw(size_t(n) * c.topk), nsw(n); next_ids = std::make_unique<Device<int>>(nw.count);
        if (fused) tqllm_cuda::moe_topk2(logits.p, routed.p, weights.p, shared_weights.p, nl->p, next_ids->p, nw.p, nsw.p, c.experts, c.topk, n, 1.f, nullptr);
        else {
            tqllm_cuda::moe_topk(logits.p, routed.p, weights.p, shared_weights.p, c.experts, c.topk, n, 1.f, nullptr);
            tqllm_cuda::moe_topk(nl->p, next_ids->p, nw.p, nsw.p, c.experts, c.topk, n, 1.f, nullptr);
        }
    } else tqllm_cuda::moe_topk(logits.p, routed.p, weights.p, shared_weights.p, c.experts, c.topk, n, 1.f, nullptr);
    if (capturing) {
        cudaGraph_t cg = nullptr;
        const cudaError_t e = cudaStreamEndCapture(cudaStreamPerThread, &cg);
        if (e != cudaSuccess || !cg) throw std::runtime_error(std::string("CUDA graph: capture of layer ") + std::to_string(layer) + " failed: " + cudaGetErrorString(e));
        size_t nodes = 0; ck(cudaGraphGetNodes(cg, nullptr, &nodes));
        const bool ple = trunk && ple_rows && std::find(c.ple_layers.begin(), c.ple_layers.end(), layer) != c.ple_layers.end();
        const int kind = !trunk ? -1 : (attention ? 2 * c.compress_ratios[layer] : 0) + ple;
        // an exec still queued on the GPU (another layer's launch) is not updated: each shape keeps a few, one per launch in flight
        auto & pool = graphs[GraphKey{this, kind, n, nodes}];
        GraphSlot * slot = nullptr;
        for (auto & g : pool) if (cudaEventQuery(g.done) == cudaSuccess) { slot = &g; break; }
        if (!slot && pool.size() < 8) { pool.push_back({}); slot = &pool.back(); ck(cudaEventCreateWithFlags(&slot->done, cudaEventDisableTiming)); }
        if (!slot) { slot = &pool.front(); ck(cudaEventSynchronize(slot->done)); }
        if (slot->exec) {
            cudaGraphExecUpdateResultInfo info;
            if (cudaGraphExecUpdate(slot->exec, cg, &info) != cudaSuccess) { cudaGetLastError(); ck(cudaGraphExecDestroy(slot->exec)); slot->exec = nullptr; }
        }
        if (!slot->exec) ck(cudaGraphInstantiate(&slot->exec, cg, 0));
        ck(cudaGraphDestroy(cg));
        ck(cudaGraphLaunch(slot->exec, cudaStreamPerThread));
        ck(cudaEventRecord(slot->done, cudaStreamPerThread));
    }
    using clk = std::chrono::steady_clock; auto h0 = clk::now();
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    Device<float> sg(size_t(n) * c.shared_ff), su(sg.count), sa(sg.count), sd(size_t(n) * h), fo(size_t(n) * h);
    auto shared_expert = [&](__half * have = nullptr) {
        const Checkpoint::SharedInput shared_in(cp, mixed.p, size_t(n) * h, have);
        cp.project(prefix + "ffn_gate_shexp.weight", mixed.p, sg.p, h, c.shared_ff, n);
        cp.project(prefix + "ffn_up_shexp.weight", mixed.p, su.p, h, c.shared_ff, n);
        tqllm_cuda::fn_swiglu_f32(sg.p, su.p, sa.p, int64_t(sg.count), nullptr);
        cp.project(prefix + "ffn_down_shexp.weight", sa.p, sd.p, c.shared_ff, h, n);
    };
    if (sink) {
        const size_t o = size_t(sink->off);
        if (comb != sink->wide + o * width) ck(cudaMemcpyAsync(sink->wide + o * width, combined.p, size_t(n) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpyAsync(sink->inject + o * c.hc, inject.p, size_t(n) * c.hc * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        tqllm_cuda::vis_f32_to_h(mixed.p, sink->xr + o * h, int64_t(n) * h, nullptr);
        ck(cudaMemcpyAsync(sink->routed + o * c.topk, routed.p, size_t(n) * c.topk * sizeof(int), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpyAsync(sink->weights + o * c.topk, weights.p, size_t(n) * c.topk * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpyAsync(sink->shared_weights + o, shared_weights.p, size_t(n) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        shared_expert();
        ck(cudaMemcpyAsync(sink->sd + o * h, sd.p, size_t(n) * h * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        return;
    }
    // TQFN_PREFILL_WAVES (default on): a prefill chunk that routes to more experts than the cache holds runs its MoE in
    // waves of at most `slots` experts; each expert is loaded once per layer per chunk (not once per small chunk).
    std::vector<int> wave_ids;
    // prefill (spec off) always takes waves, decode verify (spec on) the split path when it fits: the two paths sum the
    // down projection in a different order (1 ulp in a few values), so a slot-dependent choice changed greedy text
    // with the slot count (F2: a 23-token prompt ran waves at 140 slots and split at 150 in layer 1)
    // F39 rule (decode_rules.hpp): a verify block takes the split path whenever its distinct experts fit (TQFN_VERIFY_WAVES=1: old rule)
    static const bool old_rule = std::getenv("TQFN_VERIFY_WAVES") != nullptr;
    if (trunk && (g_hot_map || n * c.topk > cache.slots(layer) || (!spec && n > 1))) {  // incl. the short last chunk of a prompt
        wave_ids = routed.get();
        if(predict && g_hot_map && cp.cold_prefetch)cp.cold_prefetch(layer+1,next_ids->get());
        if (g_hot_map) for (int id : wave_ids) if (id < 0 || id >= c.experts) {
            const auto ls = logits.get(); const auto xs = mixed.get(); size_t bad_l = 0, bad_x = 0;
            for (float v : ls) bad_l += !std::isfinite(v);
            for (float v : xs) bad_x += !std::isfinite(v);
            throw std::runtime_error("hot/cold invalid route: layer=" + std::to_string(layer) + " id=" + std::to_string(id) +
                    " nonfinite_logits=" + std::to_string(bad_l) + " nonfinite_input=" + std::to_string(bad_x));
        }
        // one wave when the unique experts fit: the split path's miss accounting assumes small steps
        if (!g_hot_map && !tqllm::flashnext::layer_takes_waves(wave_ids, n, c.topk, cache.slots(layer), spec, old_rule)) wave_ids.clear();
    }
    if (!wave_ids.empty()) {
        const int A = n * c.topk; constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
        std::vector<int> u(wave_ids); std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end()), u.end());
        Device<__half> xr(mixed.count); tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr);
        Device<__nv_bfloat16> eg(size_t(A) * c.expert_ff), eu(eg.count), ed(size_t(A) * h);
        Device<__half> ax(eg.count);
        bool shared_queued = false;
        // Each assignment is written by exactly one bank; weights and original router ids are untouched.
        for (int bank_index = 0; bank_index < (g_hot_map && trunk ? 2 : 1); ++bank_index) {
            // Keep independent shared-expert work on the GPU while the CPU prepares cold demand reads.
            if (cp.cold_overlap >= 2 && bank_index == 1) { shared_expert(xr.p); shared_queued = true; }
            MixedExpertCache & selected = bank_index ? *g_cold_cache : cache;
            std::vector<int> bank_ids;
            for (int id : u) if (!g_hot_map || !trunk || expert_is_hot(*g_hot_map, layer, c.experts, id) == (bank_index == 0)) bank_ids.push_back(id);
            const int W = selected.plan_slots(layer);
            // many tokens per expert (GB10 chunk 1024: ~20): grouped GEMM, weights read once per 64 tokens instead of
            // once per 4; few (3090 chunk 128: ~3) keep the dedup kernel. TQFN_WAVE_GEMM_AVG: the switch-over average.
            // Decided per layer from slot-independent counts, and only when every slot count runs waves (A > experts):
            // the GEMM rounds differently from the dedup kernel the split path uses, so a slot-dependent switch changed
            // greedy text with the slot count (a 23-token prompt: 230 rows, ~8 per expert).
            static const int gemm_avg = std::getenv("TQFN_WAVE_GEMM_AVG") ? std::atoi(std::getenv("TQFN_WAVE_GEMM_AVG")) : 8;
            const bool wave_gemm = gemm_avg > 0 && A > c.experts && A >= gemm_avg * int(u.size());
            for (size_t w0 = 0; w0 < bank_ids.size(); w0 += size_t(W)) {
                const std::vector<int> wave(bank_ids.begin() + long(w0), bank_ids.begin() + long(std::min(bank_ids.size(), w0 + size_t(W))));
                if (bank_index) selected.set_miss_stream(miss_stream);
                const auto wslots = selected.prepare(layer, wave);
                if (selected.split()) selected.wait_misses();
                std::vector<int> slot_of(size_t(c.experts), -1);
                for (size_t i = 0; i < wave.size(); ++i) slot_of[size_t(wave[i])] = wslots[i];
                std::vector<int> hm(size_t(2) * A * DS); int groups[2] = {0, 0};
                for (int part = 0; part < 2; ++part) {  // 0: gate/up (token input), 1: down (assignment input)
                    int * out = hm.data() + size_t(part) * A * DS; std::vector<int> open_at(size_t(W) + 1, -1); int g = 0;
                    for (int a = 0; a < A; ++a) {
                        const int sl = slot_of[size_t(wave_ids[size_t(a)])];
                        if (sl < 0) continue;
                        int & at = open_at[size_t(sl)];
                        if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = sl; out[at * DS + 1] = 0; }
                        int * e = out + at * DS; const int col = e[1]++;
                        e[2 + col] = part ? a : a / c.topk; e[2 + DC + col] = a;
                    }
                    groups[part] = g;
                }
                Device<int> dmap(hm.size()); dmap.put(hm);
                // formats without a dedup kernel (GSQ Q2_0) take the grouped GEMM path: per-slot lists of this wave
                std::vector<int> cnt(size_t(W), 0); int wcap = 1;
                for (int a = 0; a < A; ++a) { const int sl = slot_of[size_t(wave_ids[size_t(a)])]; if (sl >= 0) wcap = std::max(wcap, ++cnt[size_t(sl)]); }
                std::vector<int> grp(size_t(W) * wcap, 0), fill(size_t(W), 0);
                for (int a = 0; a < A; ++a) { const int sl = slot_of[size_t(wave_ids[size_t(a)])]; if (sl >= 0) grp[size_t(sl) * wcap + size_t(fill[size_t(sl)]++)] = a; }
                Device<int> dcnt(cnt.size()), dgrp(grp.size()); dcnt.put(cnt); dgrp.put(grp);
                auto gemm_fmt = [](const Mat & mm) {
                    const int id = mm.fmt->u4h_id();
                    return (id == 1 || id == 4 || id == 5 || id == 6 || id == 7 || id == 10 || id == 13 || id == 14) && mm.cols % (id == 14 ? 128 : 256) == 0;
                };
                auto proj = [&](const Mat & mm, const __half * x, __nv_bfloat16 * y, int rows, bool per_expert) {
                    if (mm.fmt->has_expert_dedup() && !(wave_gemm && gemm_fmt(mm))) mm.fmt->expert_matvec_dedup(mm, x, y, rows, dmap.p + (per_expert ? size_t(A) * DS : 0), groups[per_expert], nullptr);
                    else require(mm.fmt->expert_gemm(mm, x, y, rows, dcnt.p, dgrp.p, wcap, W, c.topk, per_expert, nullptr),
                                 std::string("prefill waves: no grouped kernel for ") + mm.fmt->name());
                };
                auto m = selected.gate(), u2 = selected.up();
                if (m.fmt == u2.fmt && m.fmt->has_expert_dedup() && !(wave_gemm && gemm_fmt(m))) m.fmt->expert_matvec_dedup2(m, u2, xr.p, eg.p, eu.p, c.expert_ff, dmap.p, groups[0], nullptr);
                else { proj(m, xr.p, eg.p, c.expert_ff, false); proj(u2, xr.p, eu.p, c.expert_ff, false); }
                tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
                proj(selected.down(), ax.p, ed.p, h, true);
                // Same-bank waves still wait before reusing slots. The separate cold bank may prepare
                // its demand reads and DMA while the final hot wave remains in flight.
                if (!cp.cold_overlap || !g_hot_map || !trunk || bank_index != 0 || w0 + size_t(W) < bank_ids.size())
                    ck(cudaStreamSynchronize(nullptr));
        }
        }
        if (!shared_queued) shared_expert();
        tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
        if (g_hot_map && trunk) {
            const auto out = fo.get();
            if (std::any_of(out.begin(), out.end(), [](float v) { return !std::isfinite(v); })) {
                const auto a = ed.get(), b = eg.get(), d = eu.get(); size_t ea = 0, eb = 0, eu_bad = 0;
                for (auto v : a) ea += !std::isfinite(float(v));
                for (auto v : b) eb += !std::isfinite(float(v));
                for (auto v : d) eu_bad += !std::isfinite(float(v));
                throw std::runtime_error("hot/cold nonfinite MoE: layer=" + std::to_string(layer) + " down=" + std::to_string(ea) +
                        " gate=" + std::to_string(eb) + " up=" + std::to_string(eu_bad));
            }
        }
    } else if (cache.split()) {
        // Shared expert and hit experts run while the host prepares the cache and misses land on the miss stream.
        const int A = n * c.topk;
        static PinnedInts ids_host; ids_host.need(size_t(2) * A);
        // The routed ids go to the host on their own stream, so the default stream runs the FP16 input and the shared
        // expert during the copy instead of waiting on a copy-engine -> SM dependency.
        static cudaStream_t ids_stream = [] { cudaStream_t q = nullptr; ck(cudaStreamCreateWithFlags(&q, cudaStreamNonBlocking)); return q; }();
        static cudaEvent_t routed_done = [] { cudaEvent_t e; ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e; }();
        ck(cudaEventRecord(routed_done, nullptr)); ck(cudaStreamWaitEvent(ids_stream, routed_done, 0));
        const cudaStream_t cs = ids_stream;
        ck(cudaMemcpyAsync(ids_host.p, routed.p, size_t(A) * 4, cudaMemcpyDeviceToHost, cs));
        if (predict) ck(cudaMemcpyAsync(ids_host.p + A, next_ids->p, size_t(A) * 4, cudaMemcpyDeviceToHost, cs));
        static cudaEvent_t ids_ready = [] { cudaEvent_t e; ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e; }();
        static const int cpu_miss = std::getenv("TQFN_CPU_MISS") ? std::atoi(std::getenv("TQFN_CPU_MISS"))
                                   : g_small_card ? std::max(1, std::min(15, int(std::thread::hardware_concurrency()) / 2 - 1)) : 0;
        static std::unique_ptr<CpuMissPool> cpu_pool = cpu_miss > 0 ? std::make_unique<CpuMissPool>(cpu_miss, h, c.expert_ff) : nullptr;
        g_cpu_owned = cpu_pool != nullptr;
        static float * x_host = nullptr; static __nv_bfloat16 * cpu_out = nullptr;
        const bool use_cpu = cpu_pool && trunk && n <= tqllm::flashnext::kCpuPoolRows;
        if (use_cpu) {
            if (!x_host) { ck(cudaMallocHost(reinterpret_cast<void **>(&x_host), size_t(tqllm::flashnext::kCpuPoolRows) * h * 4));
                           ck(cudaMallocHost(reinterpret_cast<void **>(&cpu_out), size_t(tqllm::flashnext::kCpuPoolRows) * c.topk * h * 2)); }
            ck(cudaMemcpyAsync(x_host, mixed.p, size_t(n) * h * 4, cudaMemcpyDeviceToHost, cs));
        }
        ck(cudaEventRecord(ids_ready, cs));
        Device<__half> xr(mixed.count);  // one FP16 copy of the FFN input for the shared and routed experts
        tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr); shared_expert(xr.p);
        ck(cudaEventSynchronize(ids_ready)); auto h1 = clk::now();
        std::vector<int> routed_ids(ids_host.p, ids_host.p + A);
        // TQFN_TAIL_SKIP=R (APPROXIMATE, changes the output; idea: Strata v0.1.42 route-tail-skip): an expert that is
        // not resident and that every token of this call ranks R-th or lower is neither copied nor computed. Its
        // assignments get weight 0 (other weights unchanged) and point at the token's resident first choice.
        static const int tail_skip = std::getenv("TQFN_TAIL_SKIP") ? std::atoi(std::getenv("TQFN_TAIL_SKIP")) : 0;
        if (tail_skip > 0 && trunk && n <= 4) {
            std::vector<int> best_rank(c.experts, c.topk);
            for (int a = 0; a < A; ++a) best_rank[routed_ids[a]] = std::min(best_rank[routed_ids[a]], a % c.topk);
            for (int a = 0; a < A; ++a) {
                const int e = routed_ids[a];
                if (best_rank[e] < tail_skip - 1 || cache.resident(layer, e)) continue;
                const int first = routed_ids[(a / c.topk) * c.topk];
                if (best_rank[first] >= tail_skip - 1 && !cache.resident(layer, first)) continue;  // first choice itself skipped: keep
                routed_ids[a] = first;
                ck(cudaMemsetAsync(weights.p + a, 0, sizeof(float), nullptr));
                static uint64_t skipped = 0; ++skipped;
                static struct Report { ~Report() { std::printf("TAIL_SKIP assignments skipped %llu\n", (unsigned long long) skipped); } } report;
            }
        }
        if (static FILE * dump = std::getenv("TQFN_ROUTE_DUMP") ? std::fopen(std::getenv("TQFN_ROUTE_DUMP"), "ab") : nullptr; dump && trunk) {
            const int32_t hdr[2] = {layer, n}; std::fwrite(hdr, 4, 2, dump); std::fwrite(routed_ids.data(), 4, routed_ids.size(), dump);
        }
        // Routed and predicted next-layer ids per trunk layer call, for offline cache simulation.
        if (static FILE * dump = std::getenv("TQFN_PRED_DUMP") ? std::fopen(std::getenv("TQFN_PRED_DUMP"), "ab") : nullptr; dump && trunk) {
            const int32_t hdr[3] = {layer, n, predict ? A : 0}; std::fwrite(hdr, 4, 3, dump);
            std::fwrite(routed_ids.data(), 4, routed_ids.size(), dump);
            if (predict) std::fwrite(ids_host.p + A, 4, size_t(A), dump);
        }
        // TQFN_CPU_MISS: a non-resident expert goes to the CPU pool; its GPU assignment points at a resident expert of the
        // same token (output row overwritten before the combine), so the cache neither copies nor inserts it.
        std::vector<std::pair<int, int>> cpu_rows;  // (assignment, row in cpu_out)
        std::vector<int> cpu_experts;  // TQFN_CPU_REFILL: copied into the cache after this layer, off the critical path
        if (use_cpu) {
            std::vector<CpuMissPool::Job> jobs; std::map<int, int> job_of;
            for (int a = 0; a < A; ++a) {
                const int e = routed_ids[a], t = a / c.topk;
                if (!tqllm::flashnext::runs_on_cpu(layer, e, cache.resident(layer, e), true)) continue;
                int sub = -1;
                for (int q = 0; q < c.topk && sub < 0; ++q) if (cache.resident(layer, routed_ids[t * c.topk + q])) sub = routed_ids[t * c.topk + q];
                // nothing resident for this token: any resident expert of the layer is the placeholder (its output row is
                // overwritten by the CPU result before the combine), so the miss is not copied over PCIe
                if (sub < 0) { static std::vector<int> any_res(c.layers + 1, -1); int & r = any_res[layer];
                               if (r < 0 || !cache.resident(layer, r)) { r = -1; for (int x = 0; x < c.experts && r < 0; ++x) if (cache.resident(layer, x)) r = x; }
                               sub = r; }
                if (sub < 0) sub = e;  // nothing resident in the whole layer: the expert loads as a miss (its CPU row still overwrites it)
                // TQFN_CPU_MISS_MAX=K: at most K distinct experts per layer call on the CPU, the rest are GPU misses
                static const int cpu_max = std::getenv("TQFN_CPU_MISS_MAX") ? std::atoi(std::getenv("TQFN_CPU_MISS_MAX")) : g_small_card ? 5 : 1 << 30;
                if (!job_of.count(e) && int(jobs.size()) >= cpu_max) continue;
                auto [it, fresh] = job_of.try_emplace(e, int(jobs.size()));
                if (fresh) {
                    CpuMissPool::Job j; const auto pl = cp.expert_payload(layer, e);
                    const int rows[3] = {c.expert_ff, c.expert_ff, h};
                    for (int m = 0; m < 3; ++m) { j.w[m] = pl[m].bytes.data; j.type[m] = pl[m].type; j.row_bytes[m] = pl[m].bytes.size / size_t(rows[m]); }
                    static const bool gguf_rows = std::getenv("TQFN_CPU_GGUF_ROWS") != nullptr;  // debug A/B: GGUF rows, no rebuild from the packed copy
                    if (const auto * pv = g_cpu_packed; pv && layer < pv->layers && !gguf_rows) {
                        const char * blk = pv->map + pv->base[layer] + uint64_t(e) * pv->block[layer]; uint64_t off = 0;
                        for (int m = 0; m < 3; ++m) {
                            j.dev[m] = blk + off; j.aux[m] = blk + off + pv->sizes[layer][2 * m]; j.fmt[m] = pv->fmt[layer][m];
                            off += pv->sizes[layer][2 * m] + pv->sizes[layer][2 * m + 1];
                        }
                        j.tiled = pv->tiled;
                    }
                    jobs.push_back(j);
                }
                if (fresh) cpu_experts.push_back(e);
                auto & j = jobs[it->second]; j.cols_t[j.ncols++] = t;
                cpu_rows.push_back({a, it->second * 1000 + j.ncols - 1});
                routed_ids[a] = sub;
            }
            std::vector<int> base(jobs.size(), 0);
            for (size_t q = 1; q < jobs.size(); ++q) base[q] = base[q - 1] + jobs[q - 1].ncols;
            for (auto & [a, r] : cpu_rows) r = base[r / 1000] + r % 1000;
            static uint64_t cpu_assign = 0, all_assign = 0; cpu_assign += cpu_rows.size(); all_assign += A;
            static struct Report { ~Report() { std::printf("CPU_MISS assignments on the CPU %llu of %llu\n", (unsigned long long) cpu_assign, (unsigned long long) all_assign); } } report;
            cpu_pool->start(x_host, n, std::move(jobs), cpu_out);
        }
        const auto slots = cache.prepare(layer, routed_ids); const auto & missed = cache.last_missed();
        require(cache.gate().fmt->has_expert_matvec(), "split expert path needs a grouped expert kernel");
        static std::vector<PinnedInts> slot_host(c.layers + 1); auto & sh = slot_host[layer]; sh.need(size_t(3) * A);
        int hit_n = 0, miss_n = 0;
        for (int a = 0; a < A; ++a) {
            sh.p[a] = slots[a];
            sh.p[A + a] = missed[a] ? -1 : slots[a]; hit_n += !missed[a];
            sh.p[2 * A + a] = missed[a] ? slots[a] : -1; miss_n += missed[a];
        }
        int * slots_p = nullptr, * hit_p = nullptr, * miss_p = nullptr;  // set from the parameter maps below
        auto h2 = clk::now();
        auto issue_prefetch = [&] {
            if (!predict) return;
            // TQFN_PF_K: prefetch only each token's k best predicted experts (ids are in descending score order)
            static const int pf_k = std::getenv("TQFN_PF_K") ? std::atoi(std::getenv("TQFN_PF_K")) : 6;  // 6 of 10: +3.3 % vs all (10-11)
            std::vector<int> want; want.reserve(A);
            constexpr int high = 3;  // ranks 1-3 are admitted first, ranks 4..pf_k only over stale or rarely used victims
            for (int t = 0; t < n; ++t) for (int j = 0; j < std::min(high, c.topk); ++j) if (const int id = ids_host.p[A + t * c.topk + j]; !(cpu_pool && tqllm::flashnext::cpu_owned(layer + 1, id))) want.push_back(id);
            cache.set_prefetch_strict(int(want.size()));
            for (int t = 0; t < n; ++t) for (int j = high; j < std::min(pf_k, c.topk); ++j) if (const int id = ids_host.p[A + t * c.topk + j]; !(cpu_pool && tqllm::flashnext::cpu_owned(layer + 1, id))) want.push_back(id);
            cache.prefetch(layer + 1, want, copy_stream);
        };
        const auto h3 = clk::now();
        g_host_ms[0] += ms(h0, h1); g_host_ms[1] += ms(h1, h2); g_host_ms[2] += ms(h2, h3);
        Device<__nv_bfloat16> eg(size_t(n) * c.topk * c.expert_ff), eu(eg.count), ed(size_t(n) * c.topk * h);
        Device<__half> ax(eg.count);
        // Dedup: one block per unique expert for all verify tokens that route to it (weights read once).
        const bool dedup_gu = n > 1 && cache.gate().fmt->has_expert_dedup() && cache.up().fmt->has_expert_dedup();
        const bool dedup_dn = n > 1 && cache.down().fmt->has_expert_dedup();
        constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
        static std::vector<PinnedInts> dedup_host(c.layers + 1); auto & dh = dedup_host[layer]; dh.need(size_t(5) * A * DS);
        // The hits' SwiGLU and down projection run before the wait for the demand misses (each output row is one
        // assignment's own sum, so splitting the down launch into hit and miss groups is exact; +1.8 % 3090 5-chat)
        const bool split_down = dedup_dn && hit_n && miss_n;
        int groups[5] = {0, 0, 0, 0, 0};  // hit, miss, all (down), hit (down), miss (down)
        if (dedup_gu || dedup_dn) {
            auto build = [&](int which, int * out, bool per_expert) {
                int g = 0;
                std::vector<int> open_at(c.experts + 1, -1);
                for (int a = 0; a < A; ++a) {
                    if ((which == 0 && missed[a]) || (which == 1 && !missed[a])) continue;
                    const int slot = slots[a]; int & at = open_at[slot];
                    if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = slot; out[at * DS + 1] = 0; }
                    int * e = out + at * DS; const int col = e[1]++;
                    e[2 + col] = per_expert ? a : a / c.topk; e[2 + DC + col] = a;
                }
                return g;
            };
            if (dedup_gu) { groups[0] = build(0, dh.p, false); groups[1] = build(1, dh.p + A * DS, false); }
            if (dedup_dn && !split_down) groups[2] = build(2, dh.p + 2 * A * DS, true);
            if (split_down) { groups[3] = build(0, dh.p + 3 * A * DS, true); groups[4] = build(1, dh.p + 4 * A * DS, true); }
        }
        int * dmap[5] = {};
        const int A4 = (A + 3) / 4 * 4;
        // slot and dedup maps go up in one async copy from this layer's own pinned buffer (F25: sending them in a kernel's
        // parameters, param_copy, faulted the 3090's front end every ~15-25 min of long decode); only the used groups of each
        // dedup map are sent. The buffer is rewritten one step later, after this layer's routed-id sync has passed the copy.
        int dpos[5] = {}, used = 3 * A4;
        for (int q = 0; q < 5; ++q) { dpos[q] = used; used += groups[q] * DS; }
        Device<int> maps(size_t(used) + 4);
        {
            static std::vector<PinnedInts> map_host(c.layers + 1); auto & mh = map_host[layer];
            static std::vector<int> blob; blob.assign(size_t(used), -1);
            for (int q = 0; q < 3; ++q) std::memcpy(blob.data() + size_t(q) * A4, sh.p + size_t(q) * A, size_t(A) * 4);
            for (int q = 0; q < 5; ++q) if (groups[q]) std::memcpy(blob.data() + dpos[q], dh.p + size_t(q) * A * DS, size_t(groups[q]) * DS * 4);
            blob.resize((blob.size() + 3) / 4 * 4, -1);
            mh.need(blob.size()); std::memcpy(mh.p, blob.data(), blob.size() * 4);
            ck(cudaMemcpyAsync(maps.p, mh.p, blob.size() * 4, cudaMemcpyHostToDevice, nullptr));
            slots_p = maps.p; hit_p = maps.p + A4; miss_p = maps.p + 2 * A4;
            for (int q = 0; q < 5; ++q) dmap[q] = maps.p + dpos[q];
        }
        auto gate_up = [&](const int * which, int part) {
            auto m = cache.gate(), u = cache.up();
            if (dedup_gu) {
                if (!groups[part]) return;
                if (m.fmt == u.fmt) { m.fmt->expert_matvec_dedup2(m, u, xr.p, eg.p, eu.p, c.expert_ff, dmap[part], groups[part], nullptr); return; }  // gate + up in one launch
                m.fmt->expert_matvec_dedup(m, xr.p, eg.p, c.expert_ff, dmap[part], groups[part], nullptr);
                u.fmt->expert_matvec_dedup(u, xr.p, eu.p, c.expert_ff, dmap[part], groups[part], nullptr);
                return;
            }
            m.fmt->expert_matvec(m, xr.p, eg.p, c.expert_ff, which, n, c.topk, false, nullptr);
            u.fmt->expert_matvec(u, xr.p, eu.p, c.expert_ff, which, n, c.topk, false, nullptr);
        };
        if (hit_n) gate_up(hit_p, 0);
        issue_prefetch();  // after the hit launches: the GPU has work while the host issues copies
        auto m = cache.down();
        if (split_down) {  // miss rows of ax are rewritten below before their down projection reads them
            tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
            m.fmt->expert_matvec_dedup(m, ax.p, ed.p, h, dmap[3], groups[3], nullptr);
        }
        cache.wait_misses();
        if (miss_n) gate_up(miss_p, 1);
        tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
        if (split_down) m.fmt->expert_matvec_dedup(m, ax.p, ed.p, h, dmap[4], groups[4], nullptr);
        else if (dedup_dn) m.fmt->expert_matvec_dedup(m, ax.p, ed.p, h, dmap[2], groups[2], nullptr);
        else m.fmt->expert_matvec(m, ax.p, ed.p, h, slots_p, n, c.topk, true, nullptr);
        if (!cpu_rows.empty()) {
            cpu_pool->wait();
            std::vector<tqllm_cuda::ScatterItem> items;
            for (auto [a, r] : cpu_rows) items.push_back({reinterpret_cast<const char *>(cpu_out + size_t(r) * h), reinterpret_cast<char *>(ed.p + size_t(a) * h), size_t(h) * 2});
            tqllm_cuda::scatter_copy(items.data(), int(items.size()), nullptr, 1);
        }
        tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
        static const bool cpu_refill = std::getenv("TQFN_CPU_REFILL") != nullptr;
        if (cpu_refill && !cpu_experts.empty()) {  // the cache still learns the experts the CPU served (copies after this layer)
            cache.set_prefetch_strict(1 << 30); cache.prefetch(layer, cpu_experts, copy_stream);
        }
    } else {
        const auto routed_ids = routed.get(); auto h1 = clk::now();
        if (static FILE * dump = std::getenv("TQFN_ROUTE_DUMP") ? std::fopen(std::getenv("TQFN_ROUTE_DUMP"), "ab") : nullptr; dump && trunk) {
            const int32_t hdr[2] = {layer, n}; std::fwrite(hdr, 4, 2, dump); std::fwrite(routed_ids.data(), 4, routed_ids.size(), dump);
        }
        const auto slots = cache.prepare(layer, routed_ids); routed.put(slots); auto h2 = clk::now();
        if (predict) cache.prefetch(layer + 1, next_ids->get(), copy_stream);
        const auto h3 = clk::now();
        g_host_ms[0] += ms(h0, h1); g_host_ms[1] += ms(h1, h2); g_host_ms[2] += ms(h2, h3);
        Device<__half> xr(mixed.count); tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr);
        Device<__nv_bfloat16> eg(size_t(n) * c.topk * c.expert_ff), eu(eg.count), ed(size_t(n) * c.topk * h);
        Device<__half> ax(eg.count);
        // Formats without a grouped expert kernel (MTP's Q8_0 -> I8H): one row-sliced matvec per assignment.
        auto sliced = [&](Mat mat, const __half * x, __nv_bfloat16 * y, int rows, int cols, bool per_assignment) {
            const size_t data = size_t(rows) * mat.fmt->row_bytes(cols), aux = size_t(rows) * mat.fmt->aux_row_bytes(cols);
            for (int a = 0; a < n * c.topk; ++a) {
                Mat v = mat; v.rows = rows;
                v.p = static_cast<const char *>(mat.p) + size_t(slots[a]) * data;
                if (mat.aux) v.aux = static_cast<const char *>(mat.aux) + size_t(slots[a]) * aux;
                v.fmt->matvec(v, x + size_t(per_assignment ? a : a / c.topk) * cols, y + size_t(a) * rows, 1, nullptr);
            }
        };
        auto m = cache.gate();
        // TQFN_NS_DEDUP=0: one block per assignment. Default: one block per unique expert (weights read once per step)
        const bool dd = n > 1 && m.fmt->has_expert_dedup() && cache.up().fmt->has_expert_dedup() && cache.down().fmt->has_expert_dedup();
        // TQFN_EXPERT_GEMM (default on, chunks of TQFN_EXPERT_GEMM_MIN+ tokens): grouped GEMM per cache slot, each expert's
        // weights read once per 64 of its tokens (the dedup kernel reads them once per 4). Prefill only: decode and the batched
        // verify (<= 64 rows) keep the exact dedup path.
        static const int egemm_min = std::getenv("TQFN_EXPERT_GEMM_MIN") ? std::atoi(std::getenv("TQFN_EXPERT_GEMM_MIN")) : 65;  // above the 64-row batched verify
        auto gemm_ok = [](const Mat & mm) {
            const int id = mm.fmt->u4h_id();
            if (std::string(mm.fmt->name()) == "GSQ_Q2_0") return mm.cols % 64 == 0;
            return (id == 1 || id == 4 || id == 5 || id == 6 || id == 7 || id == 10 || id == 13 || id == 14) && mm.cols % (id == 14 ? 128 : 256) == 0;
        };
        (void) dd;
        // per projection: 0 grouped GEMM, 1 dedup, 2 one block per assignment, 3 row-sliced matvecs (a format without a
        // grouped kernel no longer pulls the layer's other projections onto the slow path)
        // per-slot token lists built on the host with stride = the largest count (not n): the grouped kernels' grids
        // then cover only real tokens (n = 1024 routes ~20 tokens to an expert, not 1024)
        const bool gav = n >= egemm_min;  // the grouped loader reads TQFN_TILED banks too (tests/expert_gemm_tiled.cu)
        int gcap = 1; std::vector<int> hcnt, hgrp;
        if (gav) {
            hcnt.assign(size_t(cache.slots()), 0);
            for (int a = 0; a < n * c.topk; ++a) gcap = std::max(gcap, ++hcnt[size_t(slots[a])]);
            hgrp.assign(size_t(cache.slots()) * gcap, 0); std::vector<int> fill(size_t(cache.slots()), 0);
            for (int a = 0; a < n * c.topk; ++a) hgrp[size_t(slots[a]) * gcap + size_t(fill[size_t(slots[a])]++)] = a;
        }
        Device<int> gcounts(gav ? hcnt.size() : 1), ggroups(gav ? hgrp.size() : 1);
        if (gav) { gcounts.put(hcnt); ggroups.put(hgrp); }
        const Mat mg = cache.gate(), mu = cache.up(), md = cache.down();
        auto mode = [&](const Mat & mm) {
            if (gav && gemm_ok(mm)) return 0;
            if (n > 1 && mm.fmt->has_expert_dedup()) return 1;
            return mm.fmt->has_expert_matvec() ? 2 : 3;
        };
        const int kg = mode(mg), ku = mode(mu), kd = mode(md);

        constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
        const int A = n * c.topk; std::vector<int> hm; int groups[2] = {0, 0};
        if (kg == 1 || ku == 1 || kd == 1) {
            hm.assign(size_t(2) * A * DS, 0);
            for (int part = 0; part < 2; ++part) {  // 0: gate/up (token input), 1: down (assignment input)
                int * out = hm.data() + size_t(part) * A * DS; std::vector<int> open_at(size_t(cache.slots()) + 1, -1); int g = 0;
                for (int a = 0; a < A; ++a) {
                    int & at = open_at[slots[a]];
                    if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = slots[a]; out[at * DS + 1] = 0; }
                    int * e = out + at * DS; const int col = e[1]++;
                    e[2 + col] = part ? a : a / c.topk; e[2 + DC + col] = a;
                }
                groups[part] = g;
            }
        }
        Device<int> dmap(hm.empty() ? 1 : hm.size()); if (!hm.empty()) dmap.put(hm);
        auto run = [&](const Mat & mm, int k, const __half * x, __nv_bfloat16 * y, int rows, int cols, bool per_assignment) {
            if (k == 0) require(mm.fmt->expert_gemm(mm, x, y, rows, gcounts.p, ggroups.p, gcap, cache.slots(), c.topk, per_assignment, nullptr),
                               "expert_gemm unsupported");
            else if (k == 1) mm.fmt->expert_matvec_dedup(mm, x, y, rows, dmap.p + (per_assignment ? size_t(A) * DS : 0), groups[per_assignment], nullptr);
            else if (k == 2) mm.fmt->expert_matvec(mm, x, y, rows, routed.p, n, c.topk, per_assignment, nullptr);
            else sliced(mm, x, y, rows, cols, per_assignment);
        };
        if (kg == 1 && ku == 1 && mg.fmt == mu.fmt) mg.fmt->expert_matvec_dedup2(mg, mu, xr.p, eg.p, eu.p, c.expert_ff, dmap.p, groups[0], nullptr);
        else { run(mg, kg, xr.p, eg.p, c.expert_ff, h, false); run(mu, ku, xr.p, eu.p, c.expert_ff, h, false); }
        tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
        run(md, kd, ax.p, ed.p, h, c.expert_ff, true);
        shared_expert();
        tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
    }
    auto updated = std::make_unique<Device<float>>(size_t(width) * n);
    tqllm_cuda::hc_combine(combined.p, fo.p, inject.p, updated->p, h, c.hc, n, nullptr); residual = std::move(updated);
}

void DecodeModel::router(const std::string & prefix, const float * x, float * logits, int n) {
    const bool bf16 = cp.file(prefix + "ffn_gate_inp.weight").file().tensors.at(prefix + "ffn_gate_inp.weight").type == GGML_BF16;
    // prefill chunks: the tiled GEMV (exact, the per-token kernel's sums; an fp32 cuBLAS GEMM flipped near-tie top-k)
    if (bf16 && n > 64) {
        Device<__nv_bfloat16> xb(size_t(n) * h);  // x rounded once for both GEMVs (they round it per product anyway)
        tqllm_cuda::fn_round_bf16(x, xb.p, int64_t(n) * h, nullptr);
        tqllm_cuda::fn_gemv_bf16_chunk_bx(cp.bf16_device(prefix + "ffn_gate_inp.weight"), xb.p, logits, c.experts, h, n, c.experts + 1, nullptr);
        tqllm_cuda::fn_gemv_bf16_chunk_bx(cp.bf16_device(prefix + "ffn_gate_inp_shexp.weight"), xb.p, logits + c.experts, 1, h, n, c.experts + 1, nullptr);
    } else if (bf16 && std::getenv("TQFN_F32_ROUTER") == nullptr) {
        // straight into the [n][experts + 1] logits (no staging copies)
        tqllm_cuda::fn_gemv_bf16_tokens_ld(cp.bf16_device(prefix + "ffn_gate_inp.weight"), x, logits, c.experts, h, n, c.experts + 1, nullptr);
        tqllm_cuda::fn_gemv_bf16_tokens_ld(cp.bf16_device(prefix + "ffn_gate_inp_shexp.weight"), x, logits + c.experts, 1, h, n, c.experts + 1, nullptr);
    } else if (n > 64 && std::getenv("TQFN_F32_ROUTER") == nullptr) {
        // f32 router (the MTP layer): one cuBLAS fp32 GEMM for a prefill chunk instead of the per-token kernel (127 ms per 32K
        // tokens in the catch-up); fp32 sums in another order, and the MTP's routing only shapes the drafts, never the output
        const float one = 1.f, zero = 0.f; const int ld = c.experts + 1;
        require(cublasSgemm(cp.blas(), CUBLAS_OP_T, CUBLAS_OP_N, c.experts, n, h, &one, cp.dense_device(prefix + "ffn_gate_inp.weight"), h,
                            x, h, &zero, logits, ld) == CUBLAS_STATUS_SUCCESS, "f32 router GEMM failed: " + prefix);
        require(cublasSgemm(cp.blas(), CUBLAS_OP_T, CUBLAS_OP_N, 1, n, h, &one, cp.dense_device(prefix + "ffn_gate_inp_shexp.weight"), h,
                            x, h, &zero, logits + c.experts, ld) == CUBLAS_STATUS_SUCCESS, "f32 shared-gate GEMM failed: " + prefix);
    } else tqllm_cuda::moe_router_f32(cp.dense_device(prefix + "ffn_gate_inp.weight"), cp.dense_device(prefix + "ffn_gate_inp_shexp.weight"),
                                      x, logits, h, c.experts, n, nullptr);
}
