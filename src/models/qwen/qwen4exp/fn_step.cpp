// Flash-Next model step: host prep of a chunk, the per-layer pass over a chunk or a batched step, speculative accept.
#include "fn_model.hpp"

DecodeModel::HostPrep DecodeModel::prepare_host(const std::vector<int> & tokens) {
    HostPrep p; p.tokens = tokens; p.hist0 = history.size();
    std::vector<uint64_t> rows(tokens.begin(), tokens.end());
    p.embedding = cp.file("token_embd.weight").gather_rows("token_embd.weight", rows);
    const auto ids = history.append_batch(tokens);
    std::vector<uint64_t> ple_rows(ids.begin(), ids.end());
    p.ple = cp.file("per_layer_token_embd.weight").gather_rows("per_layer_token_embd.weight", ple_rows);
    return p;
}

std::unique_ptr<Device<float>> DecodeModel::step(const std::vector<int> & tokens, const std::vector<int> * next) {
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
    if (kPrefillLog && n >= 256) std::fprintf(stderr, "tqllm: prefill chunk host prep (embedding + PLE) %.0f ms\n",
                                                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_step0).count());
    // prefill: hand back each layer's scratch; the residual moves through two persistent buffers (speculative steps keep
    // their per-layer rollback state in the arena, so they keep one arena span per step)
    // prefill chunks of lm_min+ tokens run layer by layer over the whole
    // chunk (routed experts copied once per layer, prefill_layer); the residual stays in res_keep, updated in place
    constexpr bool layer_major = true;
    constexpr int lm_min = 256;  // tokens from which a prefill chunk runs layer-major
    if (!cp.rt.hot_map && layer_major && !spec && n >= lm_min && g_arena) {
        if (res_keep->count < size_t(n) * width) { PersistentAlloc keep; res_keep.reset(); res_keep = std::make_unique<Device<float>>(size_t(n) * width); }
        ck(cudaMemcpyAsync(res_keep->p, residual->p, size_t(n) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        residual.reset(); const size_t base = g_arena->used;
        // BF16 wide rows through the layers when every layer takes the FP16 row-scale path (8-bit mixers; numeric), back to fp32
        // for the head, the MTP catch-up and the host copy
        const int sub = std::max(16, cp.rt.sub_req > 0 ? std::min(cp.run.tune.sub, cp.rt.sub_req) : cp.run.tune.sub);
        if (prefill_wide_bf16(std::min(sub, n))) {
            if (!res_keep_b || res_keep_b->count < size_t(n) * width) { PersistentAlloc keep; res_keep_b.reset(); res_keep_b = std::make_unique<Device<__nv_bfloat16>>(size_t(n) * width); }
            tqllm_cuda::wide_pack(res_keep->p, res_keep_b->p, int64_t(n) * width, nullptr);
            for (int layer = 0; layer < c.layers; ++layer) { prefill_layer(layer, nullptr, n, dp.p, experts, nullptr, res_keep_b->p); g_arena->used = base; }
            tqllm_cuda::wide_unpack(res_keep_b->p, res_keep->p, int64_t(n) * width, nullptr);
        } else
        for (int layer = 0; layer < c.layers; ++layer) { prefill_layer(layer, res_keep->p, n, dp.p, experts); g_arena->used = base; }
        position += n;
        Device<float> norm(size_t(width) * n);
        tqllm_cuda::hc_norm(res_keep->p, cp.dense_device("output_hc_norm.weight"), norm.p, h, c.hc, n, float(cp.epsilon), nullptr);
        auto out = std::make_unique<Device<float>>(size_t(h) * n);
        mix_device(cp, "output_hc_", norm.p, n, out->p, nullptr);
        return out;
    }
    const bool layer_reset = g_arena && !spec;
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

std::unique_ptr<Device<float>> DecodeModel::step_multi(const std::vector<DecodeModel *> & ms, const std::vector<std::vector<int>> & toks) {
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

void DecodeModel::accept(int m) {
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

void DecodeModel::accept_multi(const std::vector<std::pair<DecodeModel *, int>> & accepts) {
    std::vector<std::pair<DecodeModel *, int>> back;  // sequences that keep fewer tokens than their block
    for (const auto & [d, m] : accepts) if (m < d->spec_n) { require(d->spec && m >= 1, "accept: speculative state was not recorded"); back.push_back({d, m}); }
    if (back.size() < 2 || back.size() > size_t(tqllm_cuda::GdnSeqs::kMax)) { for (const auto & [d, m] : accepts) d->accept(m); return; }
    const DecodeModel & d0 = *back[0].first;
    const int layers = d0.c.layers;
    int rows = 0; for (const auto & [d, m] : back) rows += m;
    // GDN replay inputs: per layer the first m rows of each sequence's saved conv / a / b, gathered contiguously
    std::vector<size_t> base(size_t(layers), SIZE_MAX); size_t ws = 0, out = 0;
    for (int l = 0; l < layers; ++l) {
        int have = 0; for (const auto & [d, m] : back) have += d->rec[size_t(l)].state && d->rec[size_t(l)].conv_saved;
        if (have && have < int(back.size())) { for (const auto & [d, m] : accepts) d->accept(m); return; }
        if (!have) continue;
        const auto & r = d0.rec[size_t(l)];
        base[size_t(l)] = ws; ws += size_t(rows) * (size_t(r.channels) + 2 * size_t(r.vh)); out = std::max(out, size_t(rows) * r.vh * r.dim);
    }
    Device<float> gathered(std::max<size_t>(ws, 1)), replay(std::max<size_t>(out, 1));
    std::vector<tqllm_cuda::ScatterItem> items;
    auto copy = [&](const float * src, float * dst, size_t count) {
        items.push_back({reinterpret_cast<const char *>(src), reinterpret_cast<char *>(dst), count * sizeof(float)});
    };
    for (int l = 0; l < layers; ++l) {
        int off = 0;
        for (const auto & [d, m] : back) {
            auto & r = d->rec[size_t(l)];
            if (r.state && r.conv_saved) {
                const size_t ch = size_t(r.channels), taps1 = r.history->count / ch;
                if (base[size_t(l)] != SIZE_MAX) {
                    float * cx = gathered.p + base[size_t(l)], * ca = cx + size_t(rows) * ch, * cb = ca + size_t(rows) * r.vh;
                    copy(r.conv_saved->p, cx + size_t(off) * ch, size_t(m) * ch);
                    copy(r.a_saved->p, ca + size_t(off) * r.vh, size_t(m) * r.vh);
                    copy(r.b_saved->p, cb + size_t(off) * r.vh, size_t(m) * r.vh);
                }
                for (size_t j = 0; j < taps1; ++j) {  // history = last taps-1 rows of (old history ++ block inputs[0..m))
                    const size_t row = size_t(m) + j;
                    copy(row < taps1 ? r.old_history->p + row * ch : r.qkv_saved->p + (row - taps1) * ch, r.history->p + j * ch, ch);
                }
            }
            if (d->ple_past[size_t(l)] && d->ple_saved[size_t(l)]) {
                const size_t w = size_t(d->width), hist_rows = d->ple_past[size_t(l)]->count / w;
                for (size_t j = 0; j < hist_rows; ++j) {
                    const size_t row = size_t(m) + j;
                    copy(row < hist_rows ? d->ple_old[size_t(l)]->p + row * w : d->ple_saved[size_t(l)]->p + (row - hist_rows) * w, d->ple_past[size_t(l)]->p + j * w, w);
                }
            }
            off += m;
        }
    }
    tqllm_cuda::scatter_copy(items.data(), int(items.size()), nullptr);
    for (int l = 0; l < layers; ++l) {
        if (base[size_t(l)] != SIZE_MAX) {
            const auto & r = d0.rec[size_t(l)];
            const size_t ch = size_t(r.channels);
            const float * cx = gathered.p + base[size_t(l)], * ca = cx + size_t(rows) * ch, * cb = ca + size_t(rows) * r.vh;
            tqllm_cuda::GdnSeqs seqs; int off = 0;
            for (const auto & [d, m] : back) {
                auto & rq = d->rec[size_t(l)];
                seqs.off[seqs.count] = off; seqs.n[seqs.count] = m; seqs.state[seqs.count] = rq.state2->p; seqs.next[seqs.count] = rq.state->p; ++seqs.count; off += m;
            }
            // rerun the same recurrence for m tokens from the pre-block state (bit-identical to fn_gdn per sequence)
            tqllm_cuda::fn_gdn_multi(cx, ca, cb, r.neg_a, r.bias, seqs, replay.p, r.kh, r.vh, r.dim, r.eps, nullptr);
        }
        for (const auto & [d, m] : back) if (l < int(d->att.size()) && d->att[size_t(l)].k) d->att[size_t(l)].len = d->spec_pos0 + m;
    }
    for (const auto & [d, m] : back) { d->history.rollback(d->spec_hist0 + size_t(m)); d->position = d->spec_pos0 + m; }
}
