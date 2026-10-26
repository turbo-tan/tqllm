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
    if (g_prefill_log && n >= 256) std::fprintf(stderr, "tqllm: prefill chunk host prep (embedding + PLE) %.0f ms\n",
                                                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_step0).count());
    // prefill: hand back each layer's scratch; the residual moves through two persistent buffers (speculative steps keep
    // their per-layer rollback state in the arena, so they keep one arena span per step)
    // TQFN_LAYER_MAJOR (default on): prefill chunks of TQFN_LAYER_MAJOR_MIN+ tokens run layer by layer over the whole
    // chunk (routed experts copied once per layer, prefill_layer); the residual stays in res_keep, updated in place
    static const bool layer_major = tqfn_default_on("TQFN_LAYER_MAJOR");
    static const int lm_min = std::getenv("TQFN_LAYER_MAJOR_MIN") ? std::atoi(std::getenv("TQFN_LAYER_MAJOR_MIN")) : 256;
    if (!g_hot_map && layer_major && !spec && n >= lm_min && g_arena) {
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
