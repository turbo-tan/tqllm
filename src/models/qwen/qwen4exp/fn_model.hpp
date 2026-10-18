#pragma once
// Flash-Next model: DecodeModel (decode, layer-major prefill, MTP drafting) and the packed expert file.
#include "fn_layers.hpp"
#include "fn_cpu_experts.hpp"
struct PinnedInts { int * p = nullptr; size_t n = 0; void need(size_t k) { if (k > n) { if (p) cudaFreeHost(p); ck(cudaMallocHost(reinterpret_cast<void **>(&p), k * sizeof(int))); n = k; } } };
inline double g_host_ms[3];  // decode host time blocked in: routed ids D2H, cache prepare, prefetch issue
struct DecodeModel {
    const Checkpoint & cp; const Fixtures & fx; const Qwen4ExpConfig & c; int h, width;
    MixedExpertCache & experts; MixedExpertCache * mtp_experts = nullptr;
    std::vector<RecurrentState> rec; std::vector<AttentionState> att; std::vector<std::unique_ptr<Device<float>>> ple_past;
    PleTokenHistory history; int position = 0;
    bool prefetch_on = std::getenv("TQFN_NO_PREFETCH") == nullptr; cudaStream_t copy_stream = nullptr, miss_stream = nullptr;
    // speculative decoding: per-token rollback state, final wide residual rows of the last step, MTP head state
    bool spec = false; int spec_pos0 = 0, spec_n = 0; size_t spec_hist0 = 0;
    std::vector<std::unique_ptr<Device<float>>> ple_saved, ple_old;
    std::unique_ptr<Device<float>> pingpong[2];  // prefill residual between layers (layer scratch is reset)
    std::unique_ptr<Device<float>> res_keep;  // [n][width] of the last step, persistent (grows with the prefill chunk)
    AttentionState mtp_att; std::unique_ptr<Device<float>> mtp_h;  // MTP block KV and its last wide output
    // After a long prompt: the trunk's K/V pages wholly below len - keep move to pinned host memory (same addresses);
    // returns the device bytes freed. TQFN_KV_HOST_KEEP: positions kept on the device (32768); TQFN_KV_HOST=0: off.
    size_t kv_to_host() {
        static const int keep = std::getenv("TQFN_KV_HOST_KEEP") ? std::atoi(std::getenv("TQFN_KV_HOST_KEEP")) : 32768;
        size_t freed = 0;
        for (auto & a : att) {
            if (!a.k || !a.k->vmm || a.len <= 2 * keep) continue;
            const size_t row = a.k->bytes / size_t(g_kv_capacity), per = a.k->page_rows(row), cut = size_t(a.len - keep) / per;
            for (size_t i = 0; i < cut; ++i) for (KvBuf * b : {a.k.get(), a.v.get()})
                if (!b->on_host[i] && b->move(i, true)) freed += b->page;
        }
        return freed;
    }
    // Before a long prefill writes and re-reads its context: every host page back on the device (with its data, for a
    // reused prefix); returns the bytes moved. Pages that do not fit stay on the host (read over PCIe).
    size_t kv_to_device() {
        size_t moved = 0;
        for (auto & a : att) if (a.k && a.k->vmm)
            for (KvBuf * b : {a.k.get(), a.v.get()}) for (size_t i = 0; i < b->on_host.size(); ++i)
                if (b->on_host[i] && b->move(i, false)) moved += b->page;
        return moved;
    }
    DecodeModel(const Checkpoint & k, const Fixtures & f, MixedExpertCache & e, size_t capacity)
        : cp(k), fx(f), c(k.config), h(c.hidden), width(c.hidden * c.hc), experts(e), rec(c.layers), att(c.layers),
          ple_past(c.layers), history(c.ple, capacity), ple_saved(c.layers), ple_old(c.layers) {
        ck(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        ck(cudaStreamCreateWithFlags(&miss_stream, cudaStreamNonBlocking)); experts.set_miss_stream(miss_stream);
        PersistentAlloc keep; res_keep = std::make_unique<Device<float>>(size_t(64) * width); mtp_h = std::make_unique<Device<float>>(width);
    }
    ~DecodeModel() { if (copy_stream) { cudaStreamSynchronize(copy_stream); cudaStreamDestroy(copy_stream); }
                     if (miss_stream) { experts.set_miss_stream(nullptr); if (mtp_experts) mtp_experts->set_miss_stream(nullptr);
                                        cudaStreamSynchronize(miss_stream); cudaStreamDestroy(miss_stream); } }
    void set_spec(bool on) { spec = on; for (auto & r : rec) r.spec = on; }

    // One block (trunk layer 0..L-1, or the MTP block L) on n tokens; residual [n][width] is replaced.
    // a batched step over several sequences: rows [off, off + n) belong to model m (its PLE history, KV, recurrent state)
    struct Seg { DecodeModel * m; int off, n; };
    // Layer-major prefill: the dense part of a layer runs on sub-chunks and stops before the routed experts, leaving its
    // rows at `off` of these whole-chunk buffers (combined residual in place of the input rows, FFN inject, FP16 FFN
    // input, routing, shared expert output); the experts then run once for the whole chunk (prefill_layer)
    struct PrefillSink { float * wide; float * inject; __half * xr; int * routed; float * weights, * shared_weights, * sd; int off; };
    void layer_forward(int layer, std::unique_ptr<Device<float>> & residual, int n, const float * ple_rows,
                       AttentionState * attn, MixedExpertCache & cache, const std::vector<Seg> * segs = nullptr,
                       const PrefillSink * sink = nullptr) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
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
        const bool predict = prefetch_on && n <= 4 && trunk && layer + 1 < c.layers;
        const std::string np = "blk." + std::to_string(layer + 1) + ".";
        std::unique_ptr<Device<float>> nl;
        if (predict) nl = std::make_unique<Device<float>>(size_t(n) * (c.experts + 1));
        // Both routers + shared gates in one segmented launch and both top-k in one launch (same per-row and per-token
        // arithmetic: test-gemv-seg, test-topk2; +2.7 % 3090 5-chat at --slots 168)
        auto is_bf16 = [&](const std::string & pre) { return cp.file(pre + "ffn_gate_inp.weight").file().tensors.at(pre + "ffn_gate_inp.weight").type == GGML_BF16; };
        static const bool f32_router = std::getenv("TQFN_F32_ROUTER") != nullptr;
        const bool fused = !f32_router && n <= 4 && c.experts <= 1024 && h >= 1024 && is_bf16(prefix) && (!predict || is_bf16(np));
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
        std::unique_ptr<Device<int>> next_ids;
        if (predict) {
            Device<float> nw(size_t(n) * c.topk), nsw(n); next_ids = std::make_unique<Device<int>>(nw.count);
            if (fused) tqllm_cuda::moe_topk2(logits.p, routed.p, weights.p, shared_weights.p, nl->p, next_ids->p, nw.p, nsw.p, c.experts, c.topk, n, 1.f, nullptr);
            else {
                tqllm_cuda::moe_topk(logits.p, routed.p, weights.p, shared_weights.p, c.experts, c.topk, n, 1.f, nullptr);
                tqllm_cuda::moe_topk(nl->p, next_ids->p, nw.p, nsw.p, c.experts, c.topk, n, 1.f, nullptr);
            }
        } else tqllm_cuda::moe_topk(logits.p, routed.p, weights.p, shared_weights.p, c.experts, c.topk, n, 1.f, nullptr);
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
        if (trunk && n * c.topk > cache.slots(layer)) {  // incl. the short last chunk of a prompt
            wave_ids = routed.get();  // one wave when the unique experts fit: the split path's miss accounting assumes small steps
        }
        if (!wave_ids.empty()) {
            const int A = n * c.topk; constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
            std::vector<int> u(wave_ids); std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end()), u.end());
            Device<__half> xr(mixed.count); tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr);
            Device<__nv_bfloat16> eg(size_t(A) * c.expert_ff), eu(eg.count), ed(size_t(A) * h);
            Device<__half> ax(eg.count);
            const int W = cache.slots(layer);
            for (size_t w0 = 0; w0 < u.size(); w0 += size_t(W)) {
                const std::vector<int> wave(u.begin() + long(w0), u.begin() + long(std::min(u.size(), w0 + size_t(W))));
                const auto wslots = cache.prepare(layer, wave);
                if (cache.split()) cache.wait_misses();
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
                // many tokens per expert (GB10 chunk 1024: ~20): grouped GEMM, weights read once per 64 tokens instead of
                // once per 4; few (3090 chunk 128: ~3) keep the dedup kernel. TQFN_WAVE_GEMM_AVG: the switch-over average
                static const int gemm_avg = std::getenv("TQFN_WAVE_GEMM_AVG") ? std::atoi(std::getenv("TQFN_WAVE_GEMM_AVG")) : 8;
                int wave_a = 0; for (int x : cnt) wave_a += x;
                const bool wave_gemm = gemm_avg > 0 && wave_a >= gemm_avg * int(wave.size());
                auto gemm_fmt = [](const Mat & mm) {
                    const int id = mm.fmt->u4h_id();
                    return (id == 1 || id == 4 || id == 5 || id == 6 || id == 7 || id == 10 || id == 13 || id == 14) && mm.cols % (id == 14 ? 128 : 256) == 0;
                };
                auto proj = [&](const Mat & mm, const __half * x, __nv_bfloat16 * y, int rows, bool per_expert) {
                    if (mm.fmt->has_expert_dedup() && !(wave_gemm && gemm_fmt(mm))) mm.fmt->expert_matvec_dedup(mm, x, y, rows, dmap.p + (per_expert ? size_t(A) * DS : 0), groups[per_expert], nullptr);
                    else require(mm.fmt->expert_gemm(mm, x, y, rows, dcnt.p, dgrp.p, wcap, W, c.topk, per_expert, nullptr),
                                 std::string("prefill waves: no grouped kernel for ") + mm.fmt->name());
                };
                auto m = cache.gate(), u2 = cache.up();
                if (m.fmt == u2.fmt && m.fmt->has_expert_dedup() && !(wave_gemm && gemm_fmt(m))) m.fmt->expert_matvec_dedup2(m, u2, xr.p, eg.p, eu.p, c.expert_ff, dmap.p, groups[0], nullptr);
                else { proj(m, xr.p, eg.p, c.expert_ff, false); proj(u2, xr.p, eu.p, c.expert_ff, false); }
                tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
                proj(cache.down(), ax.p, ed.p, h, true);
                ck(cudaStreamSynchronize(nullptr));  // the next wave's prepare may evict these slots
            }
            shared_expert();
            tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
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
            static const int cpu_miss = std::getenv("TQFN_CPU_MISS") ? std::atoi(std::getenv("TQFN_CPU_MISS")) : 0;
            static std::unique_ptr<CpuMissPool> cpu_pool = cpu_miss > 0 ? std::make_unique<CpuMissPool>(cpu_miss, h, c.expert_ff) : nullptr;
            static float * x_host = nullptr; static __nv_bfloat16 * cpu_out = nullptr;
            const bool use_cpu = cpu_pool && trunk && n <= 4;
            if (use_cpu) {
                if (!x_host) { ck(cudaMallocHost(reinterpret_cast<void **>(&x_host), size_t(4) * h * 4));
                               ck(cudaMallocHost(reinterpret_cast<void **>(&cpu_out), size_t(4) * c.topk * h * 2)); }
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
                    if (cache.resident(layer, e)) continue;
                    int sub = -1;
                    for (int q = 0; q < c.topk && sub < 0; ++q) if (cache.resident(layer, routed_ids[t * c.topk + q])) sub = routed_ids[t * c.topk + q];
                    if (sub < 0) continue;  // nothing resident for this token: the GPU miss path serves it
                    // TQFN_CPU_MISS_MAX=K: at most K distinct experts per layer call on the CPU, the rest are GPU misses
                    static const int cpu_max = std::getenv("TQFN_CPU_MISS_MAX") ? std::atoi(std::getenv("TQFN_CPU_MISS_MAX")) : 1 << 30;
                    if (!job_of.count(e) && int(jobs.size()) >= cpu_max) continue;
                    auto [it, fresh] = job_of.try_emplace(e, int(jobs.size()));
                    if (fresh) {
                        CpuMissPool::Job j; const auto pl = cp.expert_payload(layer, e);
                        const int rows[3] = {c.expert_ff, c.expert_ff, h};
                        for (int m = 0; m < 3; ++m) { j.w[m] = pl[m].bytes.data; j.type[m] = pl[m].type; j.row_bytes[m] = pl[m].bytes.size / size_t(rows[m]); }
                        if (const auto * pv = g_cpu_packed; pv && layer < pv->layers) {
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
                for (int t = 0; t < n; ++t) for (int j = 0; j < std::min(high, c.topk); ++j) want.push_back(ids_host.p[A + t * c.topk + j]);
                cache.set_prefetch_strict(int(want.size()));
                for (int t = 0; t < n; ++t) for (int j = high; j < std::min(pf_k, c.topk); ++j) want.push_back(ids_host.p[A + t * c.topk + j]);
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
            // slot and dedup maps travel in one kernel's parameters (no host-memory reads, which crawl while demand-miss
            // copies hold the SMs); only the used groups of each dedup map are sent
            int dpos[5] = {}, used = 3 * A4;
            for (int q = 0; q < 5; ++q) { dpos[q] = used; used += groups[q] * DS; }
            Device<int> maps(size_t(used) + 4);
            {
                static std::vector<int> blob; blob.assign(size_t(used), -1);
                for (int q = 0; q < 3; ++q) std::memcpy(blob.data() + size_t(q) * A4, sh.p + size_t(q) * A, size_t(A) * 4);
                for (int q = 0; q < 5; ++q) if (groups[q]) std::memcpy(blob.data() + dpos[q], dh.p + size_t(q) * A * DS, size_t(groups[q]) * DS * 4);
                blob.resize((blob.size() + 3) / 4 * 4, -1);
                tqllm_cuda::param_copy(blob.data(), blob.size() * 4, maps.p, nullptr);
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

    // Layer-major prefill: staging banks for the experts a layer's cache bank does not hold, filled by the copy engine.
    // The next layer's first two groups are queued before its routing is known (its non-resident experts by id), so
    // they land while this layer finishes and the next one runs its attention / recurrence; the rest of its plan is the
    // routed experts the bank does not hold. Shared by every sequence (one copy stream).
    struct PrefillStage {
        int K = 0; size_t half_bytes = 0; char * half[2] = {nullptr, nullptr};
        cudaStream_t copy = nullptr; cudaEvent_t ready[2] = {}, done[2] = {};
        int * idx_host = nullptr; size_t idx_cap = 0;
        int * ids_host = nullptr; size_t ids_cap = 0;
        const MixedExpertCache * cache = nullptr; int layer = -1; std::vector<int> plan; int issued = 0;  // copies queued ahead
        double copy_bytes = 0;
    };
    static PrefillStage & stage() { static PrefillStage st; return st; }
    std::unique_ptr<Device<float>> pf_inject, pf_weights, pf_sw, pf_sd; std::unique_ptr<Device<__half>> pf_xr, pf_ax;
    std::unique_ptr<Device<int>> pf_routed, pf_idx; std::unique_ptr<Device<__nv_bfloat16>> pf_eg, pf_eu, pf_ed;
    // TQFN_PREFILL_INT8: int8 activations for the layer-major expert GEMMs (per-32-block scales, per-16 sums)
    std::unique_ptr<Device<int8_t>> pf_xq, pf_axq; std::unique_ptr<Device<float>> pf_xs, pf_axs; std::unique_ptr<Device<int16_t>> pf_xsum, pf_axsum;
    std::vector<char> pf_borrowed;  // layers whose bank was evicted for this prefill: experts from staging, no bank created
    std::map<int, std::vector<int>> pf_refill;  // borrowed layer -> its most-routed experts, refilled in one go after the prompt
    template<class T> static void grow(std::unique_ptr<Device<T>> & d, size_t count) {
        if (!d || d->count < count) { PersistentAlloc keep; d.reset(); d = std::make_unique<Device<T>>(count); }
    }
    static size_t align256(size_t x) { return (x + 255) / 256 * 256; }
    // offsets of the six expert pieces (gate/up/down data + aux) in a staging bank of K experts of this layer
    static void stage_offsets(const MixedExpertCache::BankLayout & lay, int K, size_t off[6]) {
        size_t o = 0; for (int i = 0; i < 6; ++i) { off[i] = o; o += align256(size_t(K) * lay.piece[i]); }
    }
    // the layer's copy plan (sorted): experts its bank does not hold (all of them before the bank exists)
    static std::vector<int> stage_plan(const MixedExpertCache & cache, int layer) {
        std::vector<int> plan; const bool bank = cache.bank_ready(layer);
        for (int e = 0; e < cache.experts(); ++e) if (!bank || !cache.resident(layer, e)) plan.push_back(e);
        return plan;
    }
    static void stage_alloc(PrefillStage & st, size_t need, int K) {
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
    // copy group g of the plan into staging bank g & 1 once the compute stream is done with it (runs of consecutive ids:
    // one 2D copy per piece)
    static void stage_upload(PrefillStage & st, const MixedExpertCache & cache, int layer, int g) {
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
    // start the copies of `layer` (plan + first two groups) unless they are already queued
    static void stage_begin(PrefillStage & st, const MixedExpertCache & cache, int layer) {
        if (st.layer == layer && st.cache == &cache) return;
        st.cache = &cache; st.layer = layer; st.issued = 0;
        // no prepacked file (a GB10 keeps every expert in its bank): nothing to stage, prefill_layer fills the bank instead
        if (!cache.bank_layout(layer).src0) { st.plan.clear(); return; }
        st.plan = stage_plan(cache, layer);
        const int groups = int((st.plan.size() + size_t(st.K) - 1) / size_t(st.K));
        for (; st.issued < std::min(groups, 2); ++st.issued) stage_upload(st, cache, layer, st.issued);
    }
    // Hands back what only layer-major prefill uses (its buffers, both staging banks, the pool's free blocks): the MTP
    // catch-up and decode that follow then have that memory, e.g. for their expert banks (plain cudaMalloc).
    void release_prefill() {
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
        g_qsa_batch.reset(); tqllm_cuda::fn_gdn_release_workspaces();
        if (res_keep && res_keep->count > size_t(64) * width) { PersistentAlloc keep; res_keep.reset(); res_keep = std::make_unique<Device<float>>(size_t(64) * width); }
        size_t fr = 0, tot = 0; ck(cudaMemGetInfo(&fr, &tot));
        if (fr < g_trim_below) { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
    }
    // One trunk layer for a whole prefill chunk (rows of `wide`, replaced in place): the dense part in sub-chunks of
    // --prefill-sub tokens (KV, recurrent and PLE state carried in order), then the routed experts once for every row:
    // the ones in the layer's cache bank run from it (nothing is evicted), the others from the staging banks.
    void prefill_layer(int layer, float * wide, int n, const float * ple_rows, MixedExpertCache & cache, AttentionState * att_override = nullptr) {
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
            { std::vector<int> top; for (int e = 0; e < E; ++e) if (cnt[size_t(e)]) top.push_back(e);
              std::stable_sort(top.begin(), top.end(), [&](int a, int b) { return cnt[size_t(a)] > cnt[size_t(b)]; });
              top.resize(std::min(top.size(), size_t(cache.slots(layer)))); std::sort(top.begin(), top.end()); pf_refill[layer] = top; }
            size_t off[6]; const auto lay = cache.bank_layout(layer); stage_offsets(lay, stage_k, off);
            const size_t need = off[5] + align256(size_t(stage_k) * lay.piece[5]);
            if (!st.copy || st.K != stage_k || need > st.half_bytes) stage_alloc(st, std::max(st.half_bytes, need), stage_k);
            if (st.layer != layer || st.cache != &cache) { st.cache = &cache; st.layer = layer; st.plan.clear(); st.issued = 0; }
        } else if (!cache.bank_ready(layer)) {  // first use of this layer: its bank takes the most routed experts (later decode hits)
            std::vector<int> top; for (int e = 0; e < E; ++e) if (cnt[size_t(e)]) top.push_back(e);
            std::stable_sort(top.begin(), top.end(), [&](int a, int b) { return cnt[size_t(a)] > cnt[size_t(b)]; });
            top.resize(std::min(top.size(), size_t(cache.slots(layer)))); std::sort(top.begin(), top.end());
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
        if (!hot.empty()) plan_pass(hot, cache.prepare(layer, hot), cache.slots(layer));
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

    // The host side of a chunk: token embeddings and per-layer embeddings gathered from the mapped files (the PLE history is
    // appended here, so chunks must be prepared in prompt order). A multi-chunk prefill prepares the next chunk on a thread
    // while the current one runs on the GPU (step(tokens, &next)); the gathers were 0.25 s of GPU idle per 16K chunk.
    struct HostPrep { std::vector<int> tokens; std::vector<float> embedding, ple; size_t hist0 = 0; };
    HostPrep prepare_host(const std::vector<int> & tokens) {
        HostPrep p; p.tokens = tokens; p.hist0 = history.size();
        std::vector<uint64_t> rows(tokens.begin(), tokens.end());
        p.embedding = cp.file("token_embd.weight").gather_rows("token_embd.weight", rows);
        const auto ids = history.append_batch(tokens);
        std::vector<uint64_t> ple_rows(ids.begin(), ids.end());
        p.ple = cp.file("per_layer_token_embd.weight").gather_rows("per_layer_token_embd.weight", ple_rows);
        return p;
    }
    std::future<HostPrep> prep_next_;
    bool keep_embeddings = false;      // a multi-chunk prefill keeps the prompt's embeddings for the MTP catch-up
    // [prompt tokens][hidden] fp32 while keep_embeddings; left uninitialised so each chunk's rows are first touched by the
    // copy thread, not the launching thread (150 MB of page faults per 16K chunk held the GPU idle ~0.1 s per chunk)
    struct HostRows { std::unique_ptr<float[]> p; size_t n = 0;
                      void alloc(size_t k) { p.reset(new float[k]); n = k; } void clear() { p.reset(); n = 0; }
                      size_t size() const { return n; } const float * data() const { return p.get(); } } prompt_emb;

    // Runs n new tokens through the trunk; returns the head-input rows [n][hidden]; keeps the wide rows in res_keep.
    // next: the tokens of the chunk that follows (its host prep starts now, on a thread).
    std::unique_ptr<Device<float>> step(const std::vector<int> & tokens, const std::vector<int> * next = nullptr) {
        const int n = int(tokens.size());
        if (g_arena) g_arena->used = 0;
        const auto t_step0 = std::chrono::steady_clock::now();
        HostPrep prep;
        if (prep_next_.valid()) { prep = prep_next_.get(); require(prep.tokens == tokens, "prefill host prep was made for other tokens"); }
        else prep = prepare_host(tokens);
        if (next && !next->empty()) prep_next_ = std::async(std::launch::async, [this, nx = *next] { return prepare_host(nx); });
        spec_pos0 = position; spec_n = n; spec_hist0 = prep.hist0;
        Device<float> de(prep.embedding.size()); de.put(prep.embedding);
        std::future<void> emb_copy;  // joined before prep goes out of scope
        if (keep_embeddings) emb_copy = std::async(std::launch::async, [dst = prompt_emb.p.get() + size_t(position) * h, &prep] {
            std::memcpy(dst, prep.embedding.data(), prep.embedding.size() * sizeof(float)); });
        auto residual = std::make_unique<Device<float>>(size_t(width) * n);
        tqllm_cuda::hc_init(de.p, residual->p, h, c.hc, n, nullptr);
        Device<float> dp(prep.ple.size()); dp.put(prep.ple);
        if (g_prefill_log && n >= 256) std::fprintf(stderr, "tqllm: prefill chunk host prep (embedding + PLE) %.0f ms\n",
                                                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_step0).count());
        // prefill: hand back each layer's scratch; the residual moves through two persistent buffers (speculative steps keep
        // their per-layer rollback state in the arena, so they keep one arena span per step)
        // TQFN_LAYER_MAJOR (default on): prefill chunks of TQFN_LAYER_MAJOR_MIN+ tokens run layer by layer over the whole
        // chunk (routed experts copied once per layer, prefill_layer); the residual stays in res_keep, updated in place
        static const bool layer_major = tqfn_default_on("TQFN_LAYER_MAJOR");
        static const int lm_min = std::getenv("TQFN_LAYER_MAJOR_MIN") ? std::atoi(std::getenv("TQFN_LAYER_MAJOR_MIN")) : 256;
        if (layer_major && !spec && n >= lm_min && g_arena) {
            if (res_keep->count < size_t(n) * width) { PersistentAlloc keep; res_keep.reset(); res_keep = std::make_unique<Device<float>>(size_t(n) * width); }
            ck(cudaMemcpyAsync(res_keep->p, residual->p, size_t(n) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
            residual.reset(); const size_t base = g_arena->used;
            for (int layer = 0; layer < c.layers; ++layer) { prefill_layer(layer, res_keep->p, n, dp.p, experts); g_arena->used = base; }
            position += n;
            Device<float> norm(size_t(width) * n);
            tqllm_cuda::hc_norm(res_keep->p, cp.dense_device("output_hc_norm.weight"), norm.p, h, c.hc, n, float(cp.epsilon), nullptr);
            auto out = std::make_unique<Device<float>>(size_t(h) * n);
            mix_device(cp, "output_hc_", norm.p, n, out->p, nullptr);
            return out;
        }
        const bool layer_reset = g_arena && !spec && !std::getenv("TQFN_NO_LAYER_RESET");
        const size_t base = g_arena ? g_arena->used : 0;
        for (int layer = 0; layer < c.layers; ++layer) {
            layer_forward(layer, residual, n, dp.p, nullptr, experts);
            if (!layer_reset) continue;
            auto & pp = pingpong[layer & 1];
            if (!pp || pp->count < size_t(n) * width) { PersistentAlloc keep; pp = std::make_unique<Device<float>>(size_t(n) * width); }
            ck(cudaMemcpyAsync(pp->p, residual->p, size_t(n) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
            residual = std::make_unique<Device<float>>(Device<float>::View{}, pp->p, size_t(n) * width);
            g_arena->used = base;
        }
        position += n;
        if (res_keep->count < size_t(n) * width) { PersistentAlloc keep; res_keep = std::make_unique<Device<float>>(size_t(n) * width); }
        ck(cudaMemcpyAsync(res_keep->p, residual->p, size_t(n) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        Device<float> norm(size_t(width) * n);
        tqllm_cuda::hc_norm(residual->p, cp.dense_device("output_hc_norm.weight"), norm.p, h, c.hc, n, float(cp.epsilon), nullptr);
        auto out = std::make_unique<Device<float>>(size_t(h) * n);
        mix_device(cp, "output_hc_", norm.p, n, out->p, nullptr);
        return out;
    }

    // One verify pass over several sequences (this model first): row-independent work (projections, router, experts)
    // runs once for all rows, PLE history and the attention / recurrent cores run per sequence. Returns the head-input
    // rows of every sequence, concatenated in the order given; each model keeps its wide rows in res_keep.
    std::unique_ptr<Device<float>> step_multi(const std::vector<DecodeModel *> & ms, const std::vector<std::vector<int>> & toks) {
        if (g_arena) g_arena->used = 0;
        std::vector<Seg> segs; std::vector<uint64_t> rows, ple_rows; int n = 0;
        for (size_t i = 0; i < ms.size(); ++i) {
            DecodeModel & m = *ms[i]; const int nn = int(toks[i].size());
            m.spec_pos0 = m.position; m.spec_n = nn; m.spec_hist0 = m.history.size();
            segs.push_back({&m, n, nn}); n += nn;
            rows.insert(rows.end(), toks[i].begin(), toks[i].end());
            const auto ids = m.history.append_batch(toks[i]); ple_rows.insert(ple_rows.end(), ids.begin(), ids.end());
        }
        const auto embedding = cp.file("token_embd.weight").gather_rows("token_embd.weight", rows);
        Device<float> de(embedding.size()); de.put(embedding);
        auto residual = std::make_unique<Device<float>>(size_t(width) * n);
        tqllm_cuda::hc_init(de.p, residual->p, h, c.hc, n, nullptr);
        const auto ple = cp.file("per_layer_token_embd.weight").gather_rows("per_layer_token_embd.weight", ple_rows);
        Device<float> dp(ple.size()); dp.put(ple);
        for (int layer = 0; layer < c.layers; ++layer) layer_forward(layer, residual, n, dp.p, nullptr, experts, &segs);
        for (const auto & sg : segs) {
            DecodeModel & m = *sg.m; m.position += sg.n;
            if (m.res_keep->count < size_t(sg.n) * width) { PersistentAlloc keep; m.res_keep = std::make_unique<Device<float>>(size_t(sg.n) * width); }
            ck(cudaMemcpyAsync(m.res_keep->p, residual->p + size_t(sg.off) * width, size_t(sg.n) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        }
        Device<float> norm(size_t(width) * n);
        tqllm_cuda::hc_norm(residual->p, cp.dense_device("output_hc_norm.weight"), norm.p, h, c.hc, n, float(cp.epsilon), nullptr);
        auto out = std::make_unique<Device<float>>(size_t(h) * n);
        mix_device(cp, "output_hc_", norm.p, n, out->p, nullptr);
        return out;
    }

    // Keep the first m of the last step's n tokens (speculative verify): restore every carried state to token m-1.
    void accept(int m) {
        const int n = spec_n;
        if (m >= n) return;
        require(spec && m >= 1, "accept: speculative state was not recorded");
        for (int layer = 0; layer < c.layers; ++layer) {
            auto & r = rec[layer];
            if (r.state && r.conv_saved) {
                const size_t ch = size_t(r.channels), taps1 = r.history->count / ch;  // (the saved buffers may be sized for a wider step)
                // rerun the same recurrence for m tokens from the pre-block state (bit-identical to a per-token state)
                tqllm_cuda::fn_gdn(r.conv_saved->p, r.a_saved->p, r.b_saved->p, r.neg_a, r.bias, r.state2->p, r.state->p,
                                   r.replay_out->p, r.kh, r.vh, r.dim, m, r.eps, nullptr);
                for (size_t j = 0; j < taps1; ++j) {  // history = last taps-1 rows of (old history ++ block inputs[0..m))
                    const size_t row = size_t(m) + j;
                    const float * src = row < taps1 ? r.old_history->p + row * ch : r.qkv_saved->p + (row - taps1) * ch;
                    ck(cudaMemcpyAsync(r.history->p + j * ch, src, ch * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
                }
            }
            if (ple_past[layer] && ple_saved[layer]) {
                const size_t hist_rows = ple_past[layer]->count / size_t(width);
                for (size_t j = 0; j < hist_rows; ++j) {
                    const size_t row = size_t(m) + j;
                    const float * src = row < hist_rows ? ple_old[layer]->p + row * width : ple_saved[layer]->p + (row - hist_rows) * width;
                    ck(cudaMemcpyAsync(ple_past[layer]->p + j * width, src, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
                }
            }
            if (layer < int(att.size()) && att[layer].k) att[layer].len = spec_pos0 + m;
        }
        history.rollback(spec_hist0 + size_t(m));
        position = spec_pos0 + m;
    }

    // MTP block on n (token, wide residual) pairs: appends n MTP KV entries; returns the argmax after the last pair and
    // keeps its wide output in mtp_h (input of the next chained draft).
    int mtp_batch(const std::vector<int> & tokens, const float * h_rows) {
        require(mtp_experts, "MTP not loaded (--mtp)");
        const int n = int(tokens.size()), L = c.layers;
        const std::string nx = "blk." + std::to_string(L) + ".nextn.";
        std::vector<uint64_t> rows(tokens.begin(), tokens.end());
        const auto embedding = cp.file("token_embd.weight").gather_rows("token_embd.weight", rows);
        Device<float> de(embedding.size()), en(embedding.size()), hn(size_t(n) * width), cat(size_t(n) * c.hc * 2 * h); de.put(embedding);
        tqllm_cuda::hc_norm(de.p, cp.dense_device(nx + "enorm.weight"), en.p, h, 1, n, float(cp.epsilon), nullptr);
        tqllm_cuda::hc_norm(h_rows, cp.dense_device(nx + "hnorm.weight"), hn.p, h, c.hc, n, float(cp.epsilon), nullptr);
        for (int sidx = 0; sidx < c.hc; ++sidx)
            ck(cudaMemcpy2DAsync(cat.p + size_t(sidx) * 2 * h, size_t(c.hc) * 2 * h * sizeof(float), en.p, size_t(h) * sizeof(float),
                                 size_t(h) * sizeof(float), size_t(n), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpy2DAsync(cat.p + h, size_t(2) * h * sizeof(float), hn.p, size_t(h) * sizeof(float), size_t(h) * sizeof(float),
                             size_t(n) * c.hc, cudaMemcpyDeviceToDevice, nullptr));
        auto residual = std::make_unique<Device<float>>(size_t(n) * width);
        cp.project(nx + "eh_proj.weight", cat.p, residual->p, 2 * h, h, n * c.hc);
        layer_forward(L, residual, n, nullptr, &mtp_att, *mtp_experts);
        ck(cudaMemcpyAsync(mtp_h->p, residual->p + size_t(n - 1) * width, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        Device<float> norm{size_t(width)}, out{size_t(h)};
        tqllm_cuda::hc_norm(residual->p + size_t(n - 1) * width, cp.dense_device(nx + "hc_head_norm.weight"), norm.p, h, c.hc, 1, float(cp.epsilon), nullptr);
        mix_device(cp, nx + "hc_head_", norm.p, 1, out.p, nullptr);
        return cp.draft_head_argmax_device(out.p, 1);
    }
    // The prompt's MTP keys and values without the rest of the MTP layer. The catch-up uses only the last position's MTP
    // output (the first draft), and the attention keys / values of a row depend on that row's input alone, so the other
    // rows need no attention, experts or head. Rows go in 2048-row pieces; the caller runs mtp_batch on the last position.
    // Changes drafts at most (the GEMMs may round differently), never the output.
    void mtp_kv_prefill(const std::vector<int> & tokens, const float * h_rows, const float * emb) {
        const int n = int(tokens.size()), L = c.layers;
        const std::string nx = "blk." + std::to_string(L) + ".nextn.", prefix = "blk." + std::to_string(L) + ".";
        const int kvh = cp.attention_kvheads, ad = cp.attention_dim; const size_t row = size_t(kvh) * ad;
        AttentionState & a = mtp_att;
        if (!a.k) { PersistentAlloc keep; a.k = std::make_unique<KvBuf>(size_t(g_kv_capacity) * row * sizeof(__nv_bfloat16)); a.v = std::make_unique<KvBuf>(size_t(g_kv_capacity) * row * sizeof(__nv_bfloat16)); }
        require(a.len + n <= g_kv_capacity, "MTP KV exceeds the attention KV capacity");
        const float * attn_gamma = cp.dense_device(prefix + "hc_attn_norm.weight");
        const bool qsa_keys = tqfn_default_on("TQFN_QSA") && cp.has_tensor(prefix + "indexer.k_proj.weight");
        for (int d = 0; d < n; d += 2048) {
            const int m = std::min(2048, n - d); ArenaMark scratch;
            Device<float> de(size_t(m) * h), en(size_t(m) * h), hn(size_t(m) * width), cat(size_t(m) * c.hc * 2 * h), res(size_t(m) * width);
            if (emb) ck(cudaMemcpy(de.p, emb + size_t(d) * h, size_t(m) * h * sizeof(float), cudaMemcpyHostToDevice));
            else { std::vector<uint64_t> ids(tokens.begin() + d, tokens.begin() + d + m); de.put(cp.file("token_embd.weight").gather_rows("token_embd.weight", ids)); }
            tqllm_cuda::hc_norm(de.p, cp.dense_device(nx + "enorm.weight"), en.p, h, 1, m, float(cp.epsilon), nullptr);
            tqllm_cuda::hc_norm(h_rows + size_t(d) * width, cp.dense_device(nx + "hnorm.weight"), hn.p, h, c.hc, m, float(cp.epsilon), nullptr);
            for (int sidx = 0; sidx < c.hc; ++sidx)
                ck(cudaMemcpy2DAsync(cat.p + size_t(sidx) * 2 * h, size_t(c.hc) * 2 * h * sizeof(float), en.p, size_t(h) * sizeof(float),
                                     size_t(h) * sizeof(float), size_t(m), cudaMemcpyDeviceToDevice, nullptr));
            ck(cudaMemcpy2DAsync(cat.p + h, size_t(2) * h * sizeof(float), hn.p, size_t(h) * sizeof(float), size_t(h) * sizeof(float),
                                 size_t(m) * c.hc, cudaMemcpyDeviceToDevice, nullptr));
            cp.project(nx + "eh_proj.weight", cat.p, res.p, 2 * h, h, m * c.hc);
            // the attention input as layer_forward makes it (hc_attn norm + mix)
            Device<float> norm(size_t(width) * m), mixed(size_t(h) * m), inject(size_t(c.hc) * m);
            Device<__nv_bfloat16> normb(size_t(width) * m);
            const bool hc_rs = tqllm::hc_row_scales_only(m, cp.is_bf16(prefix + "hc_attn_down.weight"), cp.is_bf16(prefix + "hc_attn_inject.weight"));
            std::unique_ptr<Device<float>> rs; if (hc_rs) rs = std::make_unique<Device<float>>(size_t(c.hc) * m);
            if (hc_rs) tqllm_cuda::hc_norm_rs_bf16(res.p, attn_gamma, rs->p, normb.p, h, c.hc, m, float(cp.epsilon), nullptr);
            else tqllm_cuda::hc_norm_bf16(res.p, attn_gamma, norm.p, normb.p, h, c.hc, m, float(cp.epsilon), nullptr);
            mix_device(cp, prefix + "hc_attn_", norm.p, m, mixed.p, inject.p, normb.p, hc_rs ? res.p : nullptr, hc_rs ? rs->p : nullptr, hc_rs ? attn_gamma : nullptr);
            Device<float> k(size_t(m) * row), v(size_t(m) * row), kn(size_t(m) * row);
            {
                const Checkpoint::SharedInput shared_in(cp, mixed.p, size_t(m) * h);
                cp.project(prefix + "attn_k.weight", mixed.p, k.p, h, kvh * ad, m);
                cp.project(prefix + "attn_v.weight", mixed.p, v.p, h, kvh * ad, m);
            }
            tqllm_cuda::fn_norm_rope(k.p, cp.dense_device(prefix + "attn_k_norm.weight"), kn.p, kvh, ad, m, a.len, cp.rotary, cp.theta,
                                    float(cp.epsilon), false, nullptr);
            Device<__nv_bfloat16> kb(k.count), vb(v.count);
            tqllm_cuda::f32_to_bf16(kn.p, kb.p, kn.count, nullptr);
            tqllm_cuda::f32_to_bf16(v.p, vb.p, v.count, nullptr);
            ck(cudaMemcpyAsync(a.k->p + size_t(a.len) * row, kb.p, kb.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, nullptr));
            ck(cudaMemcpyAsync(a.v->p + size_t(a.len) * row, vb.p, vb.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, nullptr));
            if (qsa_keys) {  // the indexer keys as attention_core appends them
                if (!a.qsa) { PersistentAlloc keep; a.qsa = std::make_unique<tqllm_cuda::QsaIndexer>(g_kv_capacity, 4, 2048, 4, 128,
                                  std::min(cp.rotary, 128), cp.theta, float(cp.epsilon)); }
                if (a.qsa->size() > a.len) a.qsa->truncate(a.len);
                Device<float> ik(size_t(m) * 128);
                cp.project(prefix + "indexer.k_proj.weight", mixed.p, ik.p, c.hidden, 128, m);
                a.qsa->append(ik.p, m, cp.dense_device(prefix + "indexer.k_norm.weight"), nullptr);
            }
            a.len += m;
        }
    }
    // mtp_batch for a long run of prompt positions: the MTP layer runs layer-major (routed experts staged from the pinned
    // MTP copy once per call, grouped GEMMs) instead of in batches small enough for the MTP cache. Same positions, same
    // order; the GEMM path may round differently, which can change drafts but never the output.
    // emb: the tokens' embeddings already on the host ([n][hidden] fp32, from the trunk's prep) instead of a gather
    int mtp_prefill(const std::vector<int> & tokens, const float * h_rows, const float * emb = nullptr) {
        require(mtp_experts, "MTP not loaded (--mtp)");
        const int n = int(tokens.size()), L = c.layers;
        const std::string nx = "blk." + std::to_string(L) + ".nextn.";
        std::vector<float> embedding;
        if (!emb) { std::vector<uint64_t> rows(tokens.begin(), tokens.end()); embedding = cp.file("token_embd.weight").gather_rows("token_embd.weight", rows); }
        const size_t emb_count = size_t(n) * size_t(h);
        std::unique_ptr<Device<float>> residual;
        { PersistentAlloc keep; residual = std::make_unique<Device<float>>(size_t(n) * width); }
        {
            ArenaMark scratch;
            Device<float> de(emb_count), en(emb_count), hn(size_t(n) * width), cat(size_t(n) * c.hc * 2 * h);
            if (emb) ck(cudaMemcpy(de.p, emb, emb_count * sizeof(float), cudaMemcpyHostToDevice)); else de.put(embedding);
            tqllm_cuda::hc_norm(de.p, cp.dense_device(nx + "enorm.weight"), en.p, h, 1, n, float(cp.epsilon), nullptr);
            tqllm_cuda::hc_norm(h_rows, cp.dense_device(nx + "hnorm.weight"), hn.p, h, c.hc, n, float(cp.epsilon), nullptr);
            for (int sidx = 0; sidx < c.hc; ++sidx)
                ck(cudaMemcpy2DAsync(cat.p + size_t(sidx) * 2 * h, size_t(c.hc) * 2 * h * sizeof(float), en.p, size_t(h) * sizeof(float),
                                     size_t(h) * sizeof(float), size_t(n), cudaMemcpyDeviceToDevice, nullptr));
            ck(cudaMemcpy2DAsync(cat.p + h, size_t(2) * h * sizeof(float), hn.p, size_t(h) * sizeof(float), size_t(h) * sizeof(float),
                                 size_t(n) * c.hc, cudaMemcpyDeviceToDevice, nullptr));
            cp.project(nx + "eh_proj.weight", cat.p, residual->p, 2 * h, h, n * c.hc);
        }
        prefill_layer(L, residual->p, n, nullptr, *mtp_experts, &mtp_att);
        ck(cudaMemcpyAsync(mtp_h->p, residual->p + size_t(n - 1) * width, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        ArenaMark scratch;
        Device<float> norm{size_t(width)}, out{size_t(h)};
        tqllm_cuda::hc_norm(residual->p + size_t(n - 1) * width, cp.dense_device(nx + "hc_head_norm.weight"), norm.p, h, c.hc, 1, float(cp.epsilon), nullptr);
        mix_device(cp, nx + "hc_head_", norm.p, 1, out.p, nullptr);
        return cp.draft_head_argmax_device(out.p, 1);
    }
    // Chained draft from the MTP's own last output; its KV entry is dropped (the next catch-up recomputes it).
    int mtp_chain(int token) {
        Device<float> hcopy{size_t(width)};
        ck(cudaMemcpyAsync(hcopy.p, mtp_h->p, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        const int d = mtp_batch({token}, hcopy.p);
        mtp_att.len -= 1;
        return d;
    }
    // mtp_batch over several sequences (this model first): one MTP-layer pass, per-sequence MTP KV and mtp_h. Returns each
    // sequence's argmax after its last pair. Sequences are grouped so the routed experts of a group fit the MTP cache.
    std::vector<int> mtp_batch_multi(const std::vector<DecodeModel *> & ms, const std::vector<std::vector<int>> & toks,
                                     const std::vector<const float *> & hs) {
        require(mtp_experts, "MTP not loaded (--mtp)");
        std::vector<int> result(ms.size());
        const int cap = std::max(1, mtp_experts->slots() / c.topk);
        for (size_t g0 = 0; g0 < ms.size();) {
            size_t g1 = g0; int N = 0;
            while (g1 < ms.size() && (g1 == g0 || N + int(toks[g1].size()) <= cap)) N += int(toks[g1++].size());
            const int L = c.layers; const std::string nx = "blk." + std::to_string(L) + ".nextn.";
            std::vector<Seg> segs; std::vector<uint64_t> rows;
            for (size_t i = g0, off = 0; i < g1; ++i) { segs.push_back({ms[i], int(off), int(toks[i].size())}); off += toks[i].size(); rows.insert(rows.end(), toks[i].begin(), toks[i].end()); }
            const auto embedding = cp.file("token_embd.weight").gather_rows("token_embd.weight", rows);
            Device<float> de(embedding.size()), en(embedding.size()), hn(size_t(N) * width), cat(size_t(N) * c.hc * 2 * h); de.put(embedding);
            tqllm_cuda::hc_norm(de.p, cp.dense_device(nx + "enorm.weight"), en.p, h, 1, N, float(cp.epsilon), nullptr);
            for (size_t i = g0; i < g1; ++i) {
                const auto & sg = segs[i - g0];
                tqllm_cuda::hc_norm(hs[i], cp.dense_device(nx + "hnorm.weight"), hn.p + size_t(sg.off) * width, h, c.hc, sg.n, float(cp.epsilon), nullptr);
            }
            for (int sidx = 0; sidx < c.hc; ++sidx)
                ck(cudaMemcpy2DAsync(cat.p + size_t(sidx) * 2 * h, size_t(c.hc) * 2 * h * sizeof(float), en.p, size_t(h) * sizeof(float),
                                     size_t(h) * sizeof(float), size_t(N), cudaMemcpyDeviceToDevice, nullptr));
            ck(cudaMemcpy2DAsync(cat.p + h, size_t(2) * h * sizeof(float), hn.p, size_t(h) * sizeof(float), size_t(h) * sizeof(float),
                                 size_t(N) * c.hc, cudaMemcpyDeviceToDevice, nullptr));
            auto residual = std::make_unique<Device<float>>(size_t(N) * width);
            cp.project(nx + "eh_proj.weight", cat.p, residual->p, 2 * h, h, N * c.hc);
            layer_forward(L, residual, N, nullptr, nullptr, *mtp_experts, &segs);
            for (size_t i = g0; i < g1; ++i) {
                const auto & sg = segs[i - g0]; DecodeModel & m = *sg.m;
                const float * last = residual->p + size_t(sg.off + sg.n - 1) * width;
                ck(cudaMemcpyAsync(m.mtp_h->p, last, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
                Device<float> norm{size_t(width)}, out{size_t(h)};
                tqllm_cuda::hc_norm(last, cp.dense_device(nx + "hc_head_norm.weight"), norm.p, h, c.hc, 1, float(cp.epsilon), nullptr);
                mix_device(cp, nx + "hc_head_", norm.p, 1, out.p, nullptr);
                result[i] = cp.draft_head_argmax_device(out.p, 1);
            }
            g0 = g1;
        }
        return result;
    }
    // Drafts d2..dD from d1 for several sequences at once (same chain as mtp_drafts, one MTP pass per chain step).
    std::vector<std::vector<int>> mtp_drafts_multi(const std::vector<DecodeModel *> & ms, const std::vector<int> & d1, int count) {
        std::vector<std::vector<int>> d(ms.size());
        for (size_t i = 0; i < ms.size(); ++i) d[i] = {d1[i]};
        std::vector<std::unique_ptr<Device<float>>> hc(ms.size());
        for (auto & x : hc) x = std::make_unique<Device<float>>(size_t(width));
        for (int k = 1; k < count; ++k) {
            std::vector<std::vector<int>> t(ms.size()); std::vector<const float *> hp(ms.size());
            for (size_t i = 0; i < ms.size(); ++i) {
                ck(cudaMemcpyAsync(hc[i]->p, ms[i]->mtp_h->p, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
                t[i] = {d[i].back()}; hp[i] = hc[i]->p;
            }
            const auto r = mtp_batch_multi(ms, t, hp);
            for (size_t i = 0; i < ms.size(); ++i) d[i].push_back(r[i]);
        }
        for (auto * m : ms) m->mtp_att.len -= count - 1;
        return d;
    }

    // Drafts d2..dD from d1: each chain step sees the previous drafts' MTP KV; all chain entries are dropped at the end.
    std::vector<int> mtp_drafts(int d1, int count) {
        std::vector<int> d{d1};
        Device<float> hcopy{size_t(width)};
        for (int i = 1; i < count; ++i) {
            ck(cudaMemcpyAsync(hcopy.p, mtp_h->p, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
            d.push_back(mtp_batch({d.back()}, hcopy.p));
        }
        mtp_att.len -= count - 1;
        return d;
    }

    // Router logits [n][experts + 1] (last = shared-expert gate). BF16 router weights: per-token GEMVs on the raw weights.
    void router(const std::string & prefix, const float * x, float * logits, int n) {
        const bool bf16 = cp.file(prefix + "ffn_gate_inp.weight").file().tensors.at(prefix + "ffn_gate_inp.weight").type == GGML_BF16;
        // default: the tiled GEMV (exact, the per-token kernel's sums); TQFN_ROUTER_GEMM=1: fp32 cuBLAS GEMM (faster, but
        // its summation order flips near-tie top-k choices: 64K text changed on 10-17)
        static const bool router_gemv = !(std::getenv("TQFN_ROUTER_GEMM") && std::atoi(std::getenv("TQFN_ROUTER_GEMM")) != 0);
        if (bf16 && router_gemv && n > 64) {
            Device<__nv_bfloat16> xb(size_t(n) * h);  // x rounded once for both GEMVs (they round it per product anyway)
            tqllm_cuda::fn_round_bf16(x, xb.p, int64_t(n) * h, nullptr);
            tqllm_cuda::fn_gemv_bf16_chunk_bx(cp.bf16_device(prefix + "ffn_gate_inp.weight"), xb.p, logits, c.experts, h, n, c.experts + 1, nullptr);
            tqllm_cuda::fn_gemv_bf16_chunk_bx(cp.bf16_device(prefix + "ffn_gate_inp_shexp.weight"), xb.p, logits + c.experts, 1, h, n, c.experts + 1, nullptr);
        } else if (bf16 && n > 64) {
            // prefill chunks: fp32 GEMM on an exact f32 copy of the bf16 weights (4 MB per layer, resident), products exact,
            // sums in cuBLAS order; TF32 off for these calls so the logits stay within f32 rounding of the per-token GEMV
            const float one = 1.f, zero = 0.f; const int ld = c.experts + 1;
            cublasMath_t mode = CUBLAS_DEFAULT_MATH; cublasGetMathMode(cp.blas(), &mode); cublasSetMathMode(cp.blas(), CUBLAS_PEDANTIC_MATH);
            require(cublasSgemm(cp.blas(), CUBLAS_OP_T, CUBLAS_OP_N, c.experts, n, h, &one, cp.dense_device(prefix + "ffn_gate_inp.weight"), h,
                                x, h, &zero, logits, ld) == CUBLAS_STATUS_SUCCESS, "router GEMM failed: " + prefix);
            require(cublasSgemm(cp.blas(), CUBLAS_OP_T, CUBLAS_OP_N, 1, n, h, &one, cp.dense_device(prefix + "ffn_gate_inp_shexp.weight"), h,
                                x, h, &zero, logits + c.experts, ld) == CUBLAS_STATUS_SUCCESS, "shared-gate GEMM failed: " + prefix);
            cublasSetMathMode(cp.blas(), mode);
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
    static void ck_zero(Device<float> & d) { ck(cudaMemset(d.p, 0, d.count * sizeof(float))); }
    static void ck_copy(Device<float> & dst, const Device<float> & src) { ck(cudaMemcpy(dst.p, src.p, src.count * sizeof(float), cudaMemcpyDeviceToDevice)); }
    static void ck_copy_n(Device<float> & dst, const float * src, size_t count) { ck(cudaMemcpyAsync(dst.p, src, count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr)); }
};

// Packed expert file (TQFNEXP1): header, then per layer [6 x uint64 sizes: gate data/aux, up data/aux, down data/aux],
// then every expert of every layer as one contiguous block in the decode cache's row layout (no repack on a miss).
struct PackedExperts {
    int layers = 0, experts = 0; std::vector<std::array<uint64_t, 6>> sizes; std::vector<uint64_t> base, block;
    const char * map = nullptr; size_t bytes = 0; bool tiled = false;
    const char * at(int layer, int id) const { return map + base[layer] + uint64_t(id) * block[layer]; }
};
inline PackedExperts open_packed(const std::string & path) {
    PackedExperts pe; FILE * f = std::fopen(path.c_str(), "rb"); require(f, "cannot open " + path);
    char magic[8]; int32_t dims[2];
    require(std::fread(magic, 1, 8, f) == 8 && (std::memcmp(magic, "TQFNEXP1", 8) == 0 || std::memcmp(magic, "TQFNEXPT", 8) == 0) &&
            std::fread(dims, 4, 2, f) == 2, "not a TQFNEXP1 / TQFNEXPT file");
    pe.tiled = std::memcmp(magic, "TQFNEXPT", 8) == 0;
    pe.layers = dims[0]; pe.experts = dims[1]; pe.sizes.resize(pe.layers);
    for (auto & z : pe.sizes) require(std::fread(z.data(), 8, 6, f) == 6, "packed header truncated");
    uint64_t off = 16 + uint64_t(pe.layers) * 48;
    for (int l = 0; l < pe.layers; ++l) { uint64_t b = 0; for (auto v : pe.sizes[l]) b += v; pe.base.push_back(off); pe.block.push_back(b); off += b * pe.experts; }
    std::fseek(f, 0, SEEK_END); pe.bytes = size_t(std::ftell(f)); std::fclose(f);
    require(pe.bytes == off, "packed file size differs from its header");
    const int fd = ::open(path.c_str(), O_RDONLY); require(fd >= 0, "cannot open " + path);
    void * m = ::mmap(nullptr, pe.bytes, PROT_READ, MAP_SHARED, fd, 0); ::close(fd);
    require(m != MAP_FAILED, "cannot map " + path); pe.map = static_cast<const char *>(m); return pe;
}
// u4h kernel id of each projection's device format (what the grouped kernels see), from one expert's load
inline std::array<int, 3> expert_u4h_ids(const Checkpoint & checkpoint, int layer) {
    const auto & c = checkpoint.config; std::array<int, 3> ids{}; const auto payload = checkpoint.expert_payload(layer, 0);
    for (int p = 0; p < 3; ++p) {
        const int cols = p == 2 ? c.expert_ff : c.hidden, rows = p == 2 ? c.hidden : c.expert_ff;
        std::vector<std::vector<char>> keep_bytes;
        Upload keep = [&](const void * data, size_t bytes) {
            keep_bytes.emplace_back(static_cast<const char *>(data), static_cast<const char *>(data) + bytes); return keep_bytes.back().data(); };
        ids[p] = weight_format(payload[p].type)->load(
            std::vector<char>(payload[p].bytes.data, payload[p].bytes.data + payload[p].bytes.size), rows, cols, keep, nullptr).fmt->u4h_id();
    }
    return ids;
}
// One expert in the device layout (data, aux per projection), as the packed file and the MTP prepack store it.
inline std::array<std::vector<char>, 6> repack_expert(const Checkpoint & checkpoint, int layer, int id) {
    const auto & c = checkpoint.config;
    std::array<std::vector<char>, 6> parts; const auto payload = checkpoint.expert_payload(layer, id);
    for (int p = 0; p < 3; ++p) {
        const int cols = p == 2 ? c.expert_ff : c.hidden, rows = p == 2 ? c.hidden : c.expert_ff;
        std::vector<std::vector<char>> pieces; std::vector<const void *> where; pieces.reserve(4);
        Upload keep = [&](const void * data, size_t bytes) {
            pieces.emplace_back(static_cast<const char *>(data), static_cast<const char *>(data) + bytes);
            where.push_back(pieces.back().data()); return pieces.back().data();
        };
        const auto m = weight_format(payload[p].type)->load(
            std::vector<char>(payload[p].bytes.data, payload[p].bytes.data + payload[p].bytes.size), rows, cols, keep, nullptr);
        for (size_t k = 0; k < pieces.size(); ++k) {
            if (where[k] == m.p) parts[2 * p] = std::move(pieces[k]);
            else if (where[k] == m.aux) parts[2 * p + 1] = std::move(pieces[k]);
        }
        require(!parts[2 * p].empty(), "loader produced no row data");
        // TQFN_TILED: lane-contiguous code tiles for the grouped expert kernels (bytes permuted, values unchanged)
        const int bpb = g_tile_experts ? tqllm_cuda::tiled_block_bytes(m.fmt->u4h_id()) : 0;
        if (bpb && rows % 16 == 0 && (cols / 32) % 4 == 0 && parts[2 * p].size() == size_t(rows) * (cols / 32) * bpb) {
            std::vector<char> t(parts[2 * p].size());
            tqllm_cuda::tile_expert_rows(parts[2 * p].data(), t.data(), rows, cols, bpb);
            parts[2 * p] = std::move(t);
        }
    }
    return parts;
}
