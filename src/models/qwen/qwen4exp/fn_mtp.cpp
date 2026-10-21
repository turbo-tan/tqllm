// Flash-Next MTP drafting: the MTP layer on prompt rows (catch-up) and drafts from its own output.
#include "fn_model.hpp"
#include "decode_rules.hpp"

int DecodeModel::mtp_batch(const std::vector<int> & tokens, const float * h_rows) {
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
    static const bool conf_on = std::getenv("TQFN_DRAFT_TAU") != nullptr;
    return cp.draft_head_argmax_device(out.p, 1, conf_on ? &draft_conf : nullptr);
}

void DecodeModel::mtp_kv_prefill(const std::vector<int> & tokens, const float * h_rows, const float * emb) {
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

int DecodeModel::mtp_prefill(const std::vector<int> & tokens, const float * h_rows, const float * emb) {
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

int DecodeModel::mtp_chain(int token) {
    Device<float> hcopy{size_t(width)};
    ck(cudaMemcpyAsync(hcopy.p, mtp_h->p, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
    const int d = mtp_batch({token}, hcopy.p);
    mtp_att.len -= 1;
    return d;
}

std::vector<int> DecodeModel::mtp_batch_multi(const std::vector<DecodeModel *> & ms, const std::vector<std::vector<int>> & toks,
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

std::vector<std::vector<int>> DecodeModel::mtp_drafts_multi(const std::vector<DecodeModel *> & ms, const std::vector<int> & d1, int count) {
    std::vector<std::vector<int>> d(ms.size());
    if (count <= 0) return d;  // drafts=0: plain decode (the length fix-up below assumes at least one draft)
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

std::vector<int> DecodeModel::mtp_drafts(int d1, int count) {
    if (count <= 0) return {};  // drafts=0: plain decode (the length fix-up below assumes at least one draft)
    std::vector<int> d{d1};
    Device<float> hcopy{size_t(width)};
    // TQFN_DRAFT_TAU=t: stop drafting once the last draft's top probability is below t (at least one draft). The verify
    // output does not depend on the block length (greedy text identical for drafts 2-5), so the text is unchanged.
    static const float tau = std::getenv("TQFN_DRAFT_TAU") ? std::strtof(std::getenv("TQFN_DRAFT_TAU"), nullptr) : 0.f;
    for (int i = 1; i < count && tqllm::flashnext::keep_drafting(tau, draft_conf); ++i) {
        ck(cudaMemcpyAsync(hcopy.p, mtp_h->p, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        d.push_back(mtp_batch({d.back()}, hcopy.p));
    }
    mtp_att.len -= int(d.size()) - 1;
    return d;
}
