#pragma once
// Flash-Next layer cores: GDN recurrence, attention (dense and QSA) and the hyper-connection mixer.
#include "debug.hpp"
#include "fn_state.hpp"
#include "experts/moe_kernels.cuh"
#include <vector>
// conv / defer (several sequences): the conv rows go to conv (the batch buffer, row off), and the recurrence is recorded in
// defer for one tqllm_cuda::fn_gdn_multi launch over the batch instead of running here
// copies (several sequences): the rollback saves (history / qkv / conv / a / b) go into this list for one scatter_copy launch per
// layer after the batch (their sources are final once the conv ran; the saves are read only by a later step's rollback)
struct GdnPre { const float * qkv, * z, * a, * b; float * conv = nullptr; int off = 0; tqllm_cuda::GdnSeqs * defer = nullptr;
                std::vector<tqllm_cuda::ScatterItem> * copies = nullptr; };
inline void recurrent_core(const Checkpoint & checkpoint, const float * input_p, float * result, int n,
                           int layer, const Fixtures * fixtures, RecurrentState * carry, const GdnPre * pre = nullptr, float * raw_out = nullptr,
                           const __half * input_h = nullptr, const __nv_bfloat16 * input_b = nullptr);
inline std::vector<float> recurrent_forward(const Checkpoint & checkpoint, const std::vector<float> & mixed,
                                        int layer, const Fixtures * fixtures = nullptr, RecurrentState * carry = nullptr) {
    const int n = int(mixed.size() / checkpoint.config.hidden);
    Device<float> input(mixed.size()), out(size_t(n) * checkpoint.config.hidden); input.put(mixed);
    recurrent_core(checkpoint, input.p, out.p, n, layer, fixtures, carry); return out.get();
}
inline void recurrent_core(const Checkpoint & checkpoint, const float * input_p, float * result, int n,
                           int layer, const Fixtures * fixtures, RecurrentState * carry, const GdnPre * pre, float * raw_out,
                           const __half * input_h, const __nv_bfloat16 * input_b) {
    const auto & c = checkpoint.config;
    const std::string suffix = "-" + std::to_string(layer), prefix = "blk." + std::to_string(layer) + ".";
    const int kh = checkpoint.key_heads, vh = checkpoint.value_heads, dim = checkpoint.state_dim;
    const int channels = (2 * kh + vh) * dim, taps = checkpoint.conv_kernel;
    require(kh > 0 && vh % kh == 0 && dim == 128 && taps == 4, "unsupported recurrent configuration");
    struct In { const float * p; } input{input_p};
    auto mk = [](const float * p, size_t count) {
        return p ? std::make_unique<Device<float>>(Device<float>::View{}, const_cast<float *>(p), count) : std::make_unique<Device<float>>(count);
    };
    // speculative single-sequence step: the projections and the conv output land in the rollback buffers (no copies)
    const bool direct = carry && carry->spec && !pre;
    if (direct) {
        PersistentAlloc keep;
        auto grow = [](std::unique_ptr<Device<float>> & d, size_t count) { if (!d || d->count < count) d = std::make_unique<Device<float>>(count); };
        grow(carry->qkv_saved, size_t(n) * channels); grow(carry->conv_saved, size_t(n) * channels);
        grow(carry->a_saved, size_t(n) * vh); grow(carry->b_saved, size_t(n) * vh);
    }
    auto qkv_d = mk(pre ? pre->qkv : direct ? carry->qkv_saved->p : nullptr, size_t(n) * channels), z_d = mk(pre ? pre->z : nullptr, size_t(n) * vh * dim);
    auto a_d = mk(pre ? pre->a : direct ? carry->a_saved->p : nullptr, size_t(n) * vh), b_d = mk(pre ? pre->b : direct ? carry->b_saved->p : nullptr, size_t(n) * vh);
    auto & qkv = *qkv_d; auto & z = *z_d; auto & a = *a_d; auto & b = *b_d;
    // plain prefill blocks: the qkv and gate projections write FP16 rows (the conv and the gated RMS read them; numeric)
    const bool half_io = !pre && !direct && !(carry && carry->spec) && !raw_out && n > 64 && !fixtures && !tqllm::debug_cstr("dump_gdn") &&
                         checkpoint.takes_half_input(prefix + "attn_qkv.weight") && checkpoint.takes_half_input(prefix + "attn_gate.weight") &&
                         checkpoint.takes_half_input(prefix + "ssm_out.weight");
    std::unique_ptr<Device<__half>> qkvh, zh;
    if (half_io) { qkvh = std::make_unique<Device<__half>>(size_t(n) * channels); zh = std::make_unique<Device<__half>>(size_t(n) * vh * dim); }
    if (!pre) {
        const Checkpoint::SharedInput shared_in(checkpoint, input.p, size_t(n) * c.hidden, const_cast<__half *>(input_h));
        // prefill blocks: one BF16 image of the input for the BF16 alpha / beta projections (each converted it before)
        // (input_b: the caller's, e.g. the mixer's)
        std::unique_ptr<Device<__nv_bfloat16>> xb; const __nv_bfloat16 * xbp = input_b;
        if (!xbp && n > 64 && checkpoint.is_bf16(prefix + "ssm_alpha.weight") && checkpoint.is_bf16(prefix + "ssm_beta.weight")) {
            xb = std::make_unique<Device<__nv_bfloat16>>(size_t(n) * c.hidden); tqllm_cuda::f32_to_bf16(input.p, xb->p, xb->count, nullptr); xbp = xb->p; }
        const Checkpoint::SharedBf16 shared_b(checkpoint, xbp ? input.p : nullptr, size_t(n) * c.hidden, xbp);
        if (half_io) {
            checkpoint.project(prefix + "attn_qkv.weight", input.p, nullptr, c.hidden, channels, n, qkvh->p);
            checkpoint.project(prefix + "attn_gate.weight", input.p, nullptr, c.hidden, vh * dim, n, zh->p);
        } else {
            checkpoint.project(prefix + "attn_qkv.weight", input.p, qkv.p, c.hidden, channels, n);
            checkpoint.project(prefix + "attn_gate.weight", input.p, z.p, c.hidden, vh * dim, n);
        }
        checkpoint.project(prefix + "ssm_alpha.weight", input.p, a.p, c.hidden, vh, n);
        checkpoint.project(prefix + "ssm_beta.weight", input.p, b.p, c.hidden, vh, n);
    }
    if (fixtures && fixtures->has("linear_attn_qkv_mixed" + suffix)) {
        compare("gdn_qkv_projection" + suffix, qkv.get(), fixtures->values("linear_attn_qkv_mixed" + suffix), .02);
        compare("gdn_z_projection" + suffix, z.get(), fixtures->values("z" + suffix), .02);
        const double dense_tolerance = checkpoint.cpu_q8_projection ? .002 : 2e-5;
        compare("gdn_alpha_projection" + suffix, a.get(), fixtures->values("alpha" + suffix), dense_tolerance);
        compare("gdn_beta_projection" + suffix, b.get(), fixtures->values("beta" + suffix), dense_tolerance);
    }
    const float * conv_weights = checkpoint.dense_device(prefix + "ssm_conv1d.weight");
    Device<float> history(size_t(taps - 1) * channels), next_history(history.count);
    auto convolved_d = mk(direct ? carry->conv_saved->p : pre && pre->conv ? pre->conv : nullptr, qkv.count); auto & convolved = *convolved_d;
    if (carry && !carry->history) {
        PersistentAlloc keep;
        carry->history = std::make_unique<Device<float>>(history.count);
        ck(cudaMemset(carry->history->p, 0, history.count * sizeof(float)));
    }
    if (!carry) ck(cudaMemsetAsync(history.p, 0, history.count * sizeof(float), nullptr));
    if (carry && carry->spec) {
        PersistentAlloc keep;
        if (!carry->old_history) carry->old_history = std::make_unique<Device<float>>(history.count);
        if (!carry->qkv_saved || carry->qkv_saved->count < qkv.count) carry->qkv_saved = std::make_unique<Device<float>>(qkv.count);
        // the conv below writes history2 and swaps, so this step leaves the pre-step history buffer alone: its save may be deferred
        if (pre && pre->copies) pre->copies->push_back({reinterpret_cast<const char *>(carry->history->p), reinterpret_cast<char *>(carry->old_history->p), history.count * sizeof(float)});
        else ck(cudaMemcpyAsync(carry->old_history->p, carry->history->p, history.count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        if (!direct) {
            if (pre && pre->copies) pre->copies->push_back({reinterpret_cast<const char *>(qkv.p), reinterpret_cast<char *>(carry->qkv_saved->p), qkv.count * sizeof(float)});
            else ck(cudaMemcpyAsync(carry->qkv_saved->p, qkv.p, qkv.count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        }
        carry->last_n = n; carry->channels = channels;
    }
    // The conv history and recurrent state go to a second buffer and the pointers swap (no D2D copy).
    if (carry && !carry->history2) { PersistentAlloc keep; carry->history2 = std::make_unique<Device<float>>(carry->history->count); }
    float * hist_out = carry ? carry->history2->p : next_history.p;
    if (half_io) tqllm_cuda::ple_conv(qkvh->p, carry ? carry->history->p : history.p, conv_weights, convolved.p, hist_out, channels, n, taps, 1, nullptr);
    else tqllm_cuda::ple_conv(qkv.p, carry ? carry->history->p : history.p, conv_weights, convolved.p, hist_out, channels, n, taps, 1, nullptr);
    if (carry) std::swap(carry->history, carry->history2);
    if (fixtures && fixtures->has("conv_output_silu" + suffix))
        compare("gdn_convolution" + suffix, convolved.get(), fixtures->values("conv_output_silu" + suffix), .02);
    const size_t state_count = size_t(vh) * dim * dim, temp_count = carry && !fixtures ? 1 : state_count;  // unused with carry + swap
    Device<float> state(temp_count), next_state(temp_count);
    auto out_d = raw_out ? std::make_unique<Device<float>>(Device<float>::View{}, raw_out, size_t(n) * vh * dim) : std::make_unique<Device<float>>(size_t(n) * vh * dim);
    auto & out = *out_d;
    const float * neg_a = checkpoint.dense_device(prefix + "ssm_a"), * bias = checkpoint.dense_device(prefix + "ssm_dt.bias");
    if (carry && !carry->state) {
        PersistentAlloc keep;
        carry->state = std::make_unique<Device<float>>(state_count);
        ck(cudaMemset(carry->state->p, 0, state_count * sizeof(float)));
    }
    if (!carry) ck(cudaMemsetAsync(state.p, 0, state.count * sizeof(float), nullptr));
    if (carry && !carry->state2) { PersistentAlloc keep; carry->state2 = std::make_unique<Device<float>>(state_count); }
    float * state_out = carry ? carry->state2->p : next_state.p;
    if (carry && carry->spec) {
        PersistentAlloc keep;
        auto keep_copy = [pre](std::unique_ptr<Device<float>> & dst, const Device<float> & src) {
            if (!dst || dst->count < src.count) dst = std::make_unique<Device<float>>(src.count);
            if (pre && pre->copies) pre->copies->push_back({reinterpret_cast<const char *>(src.p), reinterpret_cast<char *>(dst->p), src.count * sizeof(float)});
            else ck(cudaMemcpyAsync(dst->p, src.p, src.count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        };
        if (!direct) { keep_copy(carry->conv_saved, convolved); keep_copy(carry->a_saved, a); keep_copy(carry->b_saved, b); }
        if (!carry->replay_out || carry->replay_out->count < out.count) carry->replay_out = std::make_unique<Device<float>>(out.count);
        carry->neg_a = neg_a; carry->bias = bias; carry->kh = kh; carry->vh = vh; carry->dim = dim; carry->eps = float(checkpoint.epsilon);
    }
    const float * state_in = carry ? carry->state->p : state.p;
    if (pre && pre->defer && carry) {
        auto & d = *pre->defer;
        require(d.count < tqllm_cuda::GdnSeqs::kMax, "GDN: too many sequences in one batch");
        d.off[d.count] = pre->off; d.n[d.count] = n; d.state[d.count] = state_in; d.next[d.count] = state_out; ++d.count;
    } else
        tqllm_cuda::fn_gdn(convolved.p, a.p, b.p, neg_a, bias, state_in, state_out, out.p, kh, vh, dim, n, float(checkpoint.epsilon), nullptr);
    if (carry) std::swap(carry->state, carry->state2);
    if (fixtures) compare("gdn_recurrent_output" + suffix, out.get(), fixtures->values("attn_output" + suffix), .02);
    if (fixtures && fixtures->has("new_state" + suffix))
        compare("gdn_recurrent_state" + suffix, next_state.get(), fixtures->values("new_state" + suffix), .02);
    if (raw_out) return;
    struct Out { float * p; } projected{result};
    // prefill blocks: the gated rows go to ssm_out as its FP16 input directly (no fp32 rows, no conversion; same values)
    if (n > 64 && !fixtures && !tqllm::debug_cstr("dump_gdn") && checkpoint.takes_half_input(prefix + "ssm_out.weight")) {
        Device<__half> gh(out.count);
        if (zh) tqllm_cuda::fn_gated_rms_h(out.p, zh->p, checkpoint.dense_device(prefix + "ssm_norm.weight"), gh.p, vh, dim, n, float(checkpoint.epsilon), nullptr);
        else tqllm_cuda::fn_gated_rms_h(out.p, z.p, checkpoint.dense_device(prefix + "ssm_norm.weight"), gh.p, vh, dim, n, float(checkpoint.epsilon), nullptr);
        const float * key = reinterpret_cast<const float *>(gh.p);  // never read: the projection takes the FP16 rows
        const Checkpoint::SharedInput shared(checkpoint, key, out.count, gh.p);
        checkpoint.project(prefix + "ssm_out.weight", key, projected.p, vh * dim, c.hidden, n);
        return;
    }
    Device<float> gated(out.count);
    tqllm_cuda::fn_gated_rms(out.p, z.p, checkpoint.dense_device(prefix + "ssm_norm.weight"), gated.p, vh, dim, n, float(checkpoint.epsilon), nullptr);
    if (fixtures && fixtures->has("final_output" + suffix))
        compare("gdn_sigmoid_norm" + suffix, gated.get(), fixtures->values("final_output" + suffix), .02);
    checkpoint.project(prefix + "ssm_out.weight", gated.p, projected.p, vh * dim, c.hidden, n);
    if (const char * dump = tqllm::debug_cstr("dump_gdn")) {
        std::vector<float> pv(size_t(n) * c.hidden); ck(cudaMemcpy(pv.data(), result, pv.size() * 4, cudaMemcpyDeviceToHost));
        for (const auto & [tag, values] : {std::pair<std::string, std::vector<float>>{"gated", gated.get()}, {"projected", pv}}) {
            std::ofstream file(std::filesystem::path(dump) / ("gdn-" + tag + suffix + ".f32"), std::ios::binary);
            file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
        }
    }
    if (fixtures) compare("gdn_projected_output" + suffix, [&] { std::vector<float> v(size_t(n) * c.hidden); ck(cudaMemcpy(v.data(), result, v.size() * 4, cudaMemcpyDeviceToHost)); return v; }(), fixtures->values("linear_attn_out" + suffix), .02);
}
// Decode reads of host K/V (TQFN_KV_HOST): one gather per layer, each with a 16K-row device cache of host rows (consecutive
// steps select mostly the same positions: at 128K 81.6 % of the host rows a step reads were read recently). Sized for the
// slot context and an 8-token verify block of attention_core's widest selection (top_k 2048 + ratio 4 - 1): no allocation
// inside graph capture. kv_to_host makes them before the bank refill sizes the banks around them (made at the first
// decode step, after the refill took the free memory, a 145K prompt on a 3090 at slots=auto ran out of memory).
inline std::unique_ptr<tqllm_cuda::QsaGather> & host_gather(int layer) {
    static std::vector<std::unique_ptr<tqllm_cuda::QsaGather>> gathers;
    if (size_t(layer) >= gathers.size()) gathers.resize(size_t(layer) + 1);
    return gathers[size_t(layer)];
}
inline void make_host_gather(const Checkpoint & checkpoint, int layer) {
    auto & gather = host_gather(layer); if (gather) return;
    const size_t row = size_t(checkpoint.attention_kvheads) * checkpoint.attention_dim * sizeof(__nv_bfloat16);
    PersistentAlloc keep;
    gather = std::make_unique<tqllm_cuda::QsaGather>(std::max(checkpoint.rt.kv_capacity, checkpoint.rt.kv_capacity_full), row, 16384, 8 * (2048 + 4 - 1));
}
// pre: q (with gate), k, v of this sequence's rows projected for a whole batch; raw_out: write the attention output
// there and stop before the gate and the output projection (both row-wise, done for the batch)
struct AttnPre { const float * qg, * k, * v; };
inline void attention_core(const Checkpoint & checkpoint, const float * input_p, float * result, int n,
                           int layer, const Fixtures * fixtures, AttentionState * carry, const AttnPre * pre = nullptr, float * raw_out = nullptr,
                           const __half * input_h = nullptr, const __nv_bfloat16 * input_b = nullptr);
inline std::vector<float> attention_forward(const Checkpoint & checkpoint, const std::vector<float> & mixed,
                                        int layer, const Fixtures * fixtures = nullptr, AttentionState * carry = nullptr) {
    const int n = int(mixed.size() / checkpoint.config.hidden);
    Device<float> input(mixed.size()), out(size_t(n) * checkpoint.config.hidden); input.put(mixed);
    attention_core(checkpoint, input.p, out.p, n, layer, fixtures, carry); return out.get();
}
inline void attention_core(const Checkpoint & checkpoint, const float * input_p, float * result, int n,
                           int layer, const Fixtures * fixtures, AttentionState * carry, const AttnPre * pre, float * raw_out,
                           const __half * input_h, const __nv_bfloat16 * input_b) {
    const auto & c = checkpoint.config;
    const std::string suffix = "-" + std::to_string(layer), prefix = "blk." + std::to_string(layer) + ".";
    const int heads = checkpoint.attention_heads, kvheads = checkpoint.attention_kvheads, dim = checkpoint.attention_dim;
    require(n > 0 && n <= 2048, "dense fixture exceeds QSA budget");
    struct In { const float * p; } input{input_p};
    auto mk = [](const float * p, size_t count) {
        return p ? std::make_unique<Device<float>>(Device<float>::View{}, const_cast<float *>(p), count) : std::make_unique<Device<float>>(count);
    };
    auto qg_d = mk(pre ? pre->qg : nullptr, size_t(n) * heads * dim * 2), k_d = mk(pre ? pre->k : nullptr, size_t(n) * kvheads * dim);
    auto v_d = mk(pre ? pre->v : nullptr, size_t(n) * kvheads * dim);
    auto & qg = *qg_d; auto & k = *k_d; auto & v = *v_d;
    Device<float> q(size_t(n) * heads * dim), kn(k.count);
    if (!pre) {
        const Checkpoint::SharedInput shared_in(checkpoint, input.p, size_t(n) * c.hidden, const_cast<__half *>(input_h));
        checkpoint.project(prefix + "attn_q.weight", input.p, qg.p, c.hidden, heads * dim * 2, n);
        checkpoint.project(prefix + "attn_k.weight", input.p, k.p, c.hidden, kvheads * dim, n);
        checkpoint.project(prefix + "attn_v.weight", input.p, v.p, c.hidden, kvheads * dim, n);
    }
    if (fixtures && fixtures->has("Qcur_full" + suffix)) {
        compare("attention_query_projection" + suffix, qg.get(), fixtures->values("Qcur_full" + suffix), .02);
        compare("attention_key_projection" + suffix, k.get(), fixtures->values("Kcur" + suffix, 0), .02);
        compare("attention_value_projection" + suffix, v.get(), fixtures->values("Vcur" + suffix, 0), .02);
    }
    const int pos = carry ? carry->len : 0;
    tqllm_cuda::fn_norm_rope(qg.p, checkpoint.dense_device(prefix + "attn_q_norm.weight"), q.p, heads, dim, n, pos, checkpoint.rotary, checkpoint.theta,
                            float(checkpoint.epsilon), true, nullptr);
    tqllm_cuda::fn_norm_rope(k.p, checkpoint.dense_device(prefix + "attn_k_norm.weight"), kn.p, kvheads, dim, n, pos, checkpoint.rotary, checkpoint.theta,
                            float(checkpoint.epsilon), false, nullptr);
    Device<__nv_bfloat16> kb(k.count), vb(v.count);
    tqllm_cuda::f32_to_bf16(kn.p, kb.p, kn.count, nullptr);
    tqllm_cuda::f32_to_bf16(v.p, vb.p, v.count, nullptr);
    auto out_d = raw_out ? std::make_unique<Device<float>>(Device<float>::View{}, raw_out, q.count) : std::make_unique<Device<float>>(q.count);
    auto & out = *out_d; struct Out { float * p; } projected{result};
    if (carry) {
        const size_t row = size_t(kvheads) * dim;
        const int cap = kv_cap(*carry, checkpoint.rt.kv_capacity);
        if (!carry->k) { PersistentAlloc keep; carry->k = std::make_unique<KvBuf>(size_t(cap) * row * sizeof(__nv_bfloat16)); carry->v = std::make_unique<KvBuf>(size_t(cap) * row * sizeof(__nv_bfloat16)); }
        require(pos + n <= cap, "decode context exceeds the attention KV capacity (pos " + std::to_string(pos) + " + " + std::to_string(n) + " > " + std::to_string(cap) + ")");
        ck(cudaMemcpyAsync(carry->k->p + size_t(pos) * row, kb.p, kb.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpyAsync(carry->v->p + size_t(pos) * row, vb.p, vb.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, nullptr));
        // QSA (the model's own long-context attention): every token's indexer key is kept; once the context passes the
        // indexer budget each query attends to its top-k selected positions (dense attention is the same below it)
        constexpr bool qsa_on = true;  // the model's sparse attention (dense is the same below the top-k budget)
        const bool has_indexer = qsa_on && checkpoint.has_tensor(prefix + "indexer.k_proj.weight");
        // prefill blocks: one BF16 image of the input for the BF16 indexer k / q projections
        std::unique_ptr<Device<__nv_bfloat16>> ixb; const __nv_bfloat16 * ixp = has_indexer ? input_b : nullptr;
        if (!ixp && has_indexer && n > 64 && checkpoint.is_bf16(prefix + "indexer.k_proj.weight") && checkpoint.is_bf16(prefix + "indexer.q_proj.weight")) {
            ixb = std::make_unique<Device<__nv_bfloat16>>(size_t(n) * c.hidden); tqllm_cuda::f32_to_bf16(input.p, ixb->p, ixb->count, nullptr); ixp = ixb->p; }
        const Checkpoint::SharedBf16 shared_ib(checkpoint, ixp ? input.p : nullptr, size_t(n) * c.hidden, ixp);
        const int top_k = 2048, ratio = 4, iheads = 4, idim = 128;
        if (has_indexer) {
            ArenaMark scratch;
            if (!carry->qsa) {
                // the indexer uses plain cudaMalloc: hand back what the async pool keeps reserved first when memory is short
                // (a trim per attention layer cost 0.3 s of idle GPU at 128K; the indexer state is tens of MB)
                { size_t fr = 0, tot = 0; ck(cudaMemGetInfo(&fr, &tot));
                  if (fr < (size_t(1) << 30)) { cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); } }
                PersistentAlloc keep; carry->qsa = std::make_unique<tqllm_cuda::QsaIndexer>(cap, ratio, top_k, iheads, idim,
                                   std::min(checkpoint.rotary, idim), checkpoint.theta, float(checkpoint.epsilon)); }
            if (carry->qsa->size() > pos) carry->qsa->truncate(pos);  // speculative rollback
            Device<float> ik(size_t(n) * idim);
            checkpoint.project(prefix + "indexer.k_proj.weight", input.p, ik.p, c.hidden, idim, n);
            carry->qsa->append(ik.p, n, checkpoint.dense_device(prefix + "indexer.k_norm.weight"), nullptr);
        }
        if (has_indexer && pos + n > top_k) {
            ArenaMark scratch;
            struct QsaTimer { double & ms; std::chrono::steady_clock::time_point t0;
                explicit QsaTimer(double & m) : ms(m) { if (kPrefillLog) { cudaDeviceSynchronize(); t0 = std::chrono::steady_clock::now(); } }
                ~QsaTimer() { if (kPrefillLog) { cudaDeviceSynchronize(); ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } } qsa_timer(checkpoint.rt.qsa_ms);
            Device<float> iq(size_t(n) * iheads * idim);
            checkpoint.project(prefix + "indexer.q_proj.weight", input.p, iq.p, c.hidden, iheads * idim, n);
            const float * qnorm = checkpoint.dense_device(prefix + "indexer.q_norm.weight");
            auto & batch = checkpoint.rt.qsa_batch;  // shared by every layer (one stream)
            // the selection runs in blocks of checkpoint.rt.qsa_block queries: its scratch scales with queries x context (4 GB at 245K for a
            // 2048-token sub-chunk), and every query's top-k is independent of the block it is scored in
            const int qb = std::min(n, checkpoint.rt.qsa_block);
            if (!batch || batch->max_queries < qb) {
                batch.reset(); cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0));
                batch = std::make_unique<tqllm_cuda::QsaBatchScratch>(std::max(qb, 16), checkpoint.rt.kv_capacity, ratio, top_k);
            }
            for (int q0 = 0; q0 < n; q0 += qb) {
                const int nb = std::min(qb, n - q0);
                const auto sel = carry->qsa->select_batch(iq.p + size_t(q0) * iheads * idim, qnorm, pos + q0, nb, *batch, nullptr);
                const float * qp = q.p + size_t(q0) * heads * dim; float * op = out.p + size_t(q0) * heads * dim;
                tqllm::KvSelected att{carry->k->p, carry->v->p, qp, op, sel.positions, sel.valid, sel.width, nb, pos + q0 + 1, heads, kvheads, dim};
                ArenaMark block_scratch;
                std::unique_ptr<Device<float>> sc, inv;
                if (nb >= 32) {  // grouped kernel, K/V read once per kv head; few tokens (verify blocks): the per-query path has more blocks
                    sc = std::make_unique<Device<float>>(size_t(nb) * heads * sel.width); inv = std::make_unique<Device<float>>(size_t(nb) * heads);
                    att.scores = sc->p; att.inv = inv->p;
                }
                // part of this layer's K/V in host memory (TQFN_KV_HOST): the selected rows go to one compact device copy
                // first (read once, wide loads), then the attention reads that copy with the same rows in the same order
                if (nb < 32 && carry->k->vmm && carry->k->any_host()) {
                    static std::vector<uint64_t> epochs;
                    if (size_t(layer) >= epochs.size()) epochs.resize(size_t(layer) + 1, 0);
                    const size_t row = size_t(kvheads) * dim * sizeof(__nv_bfloat16);
                    make_host_gather(checkpoint, layer); auto & gather = host_gather(layer);
                    const int written_from = epochs[size_t(layer)] == checkpoint.rt.kv_epoch ? pos : 0;  // a new request: the cache starts empty
                    epochs[size_t(layer)] = checkpoint.rt.kv_epoch;
                    gather->run(sel.positions, sel.valid, nb, sel.width, pos + q0 + 1, carry->k->p, carry->v->p, nullptr,
                                carry->k->host_rows(row), written_from);
                    att.keys = gather->keys(); att.values = gather->values(); att.positions = gather->positions();
                    att.first_cells = nb * sel.width + 1;  // renumbered cells all lie below this; the causal cut is already applied
                }
                kv_fmt().attend_selected(att);
            }
        } else if (n < 32 || !tqllm_cuda::qsa_attention_dense_tc(q.p, carry->k->p, carry->v->p, out.p, n, pos + 1, heads, kvheads, dim, nullptr)) {
            // prefill blocks below the indexer budget take the tensor-core kernel the selected path uses above it; decode and
            // verify blocks keep the split-K kernel
            Device<float> ws(tqllm_cuda::fn_dense_attention_workspace(heads, dim, n, pos));
            tqllm_cuda::fn_dense_attention(q.p, carry->k->p, carry->v->p, out.p, heads, kvheads, dim, n, pos, nullptr, ws.p);
        }
        carry->len += n;
    } else {
        Device<float> ws(tqllm_cuda::fn_dense_attention_workspace(heads, dim, n, 0));
        tqllm_cuda::fn_dense_attention(q.p, kb.p, vb.p, out.p, heads, kvheads, dim, n, 0, nullptr, ws.p);
    }
    if (fixtures && fixtures->has("attn_pregate" + suffix))
        compare("attention_causal_output" + suffix, out.get(), fixtures->values("attn_pregate" + suffix), .02);
    if (raw_out) return;
    if (n > 64 && !fixtures && checkpoint.takes_half_input(prefix + "attn_output.weight")) {  // prefill: the gated rows in FP16 only
        Device<__half> oh(out.count);
        tqllm_cuda::fn_attention_gate_h(out.p, qg.p, oh.p, heads, dim, n, nullptr);
        const float * key = reinterpret_cast<const float *>(oh.p);
        const Checkpoint::SharedInput shared(checkpoint, key, out.count, oh.p);
        checkpoint.project(prefix + "attn_output.weight", key, projected.p, heads * dim, c.hidden, n);
        return;
    }
    tqllm_cuda::fn_attention_gate(out.p, qg.p, heads, dim, n, nullptr);
    if (fixtures && fixtures->has("attn_gated" + suffix))
        compare("attention_gated_output" + suffix, out.get(), fixtures->values("attn_gated" + suffix), .02);
    checkpoint.project(prefix + "attn_output.weight", out.p, projected.p, heads * dim, c.hidden, n);
    if (fixtures) compare("attention_projected_output" + suffix, [&] { std::vector<float> v2(size_t(n) * c.hidden); ck(cudaMemcpy(v2.data(), result, v2.size() * 4, cudaMemcpyDeviceToHost)); return v2; }(), fixtures->values("attn_output" + suffix), .02);
}
// Decode blocks of several sequences, every context within the indexer budget: norm / RoPE, the K/V appends, the indexer
// keys and the attention of all sequences in one launch each instead of one per sequence. Per sequence the same operations
// as attention_core (bit-identical); false (nothing done) when a sequence needs the per-sequence path.
struct AttnSeg { AttentionState * carry; int off, n; };
inline bool attention_multi(const Checkpoint & checkpoint, const float * input_p, const std::vector<AttnSeg> & segs, int n, int layer,
                            const AttnPre & pre, float * raw_out) {
    const auto & c = checkpoint.config;
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const int heads = checkpoint.attention_heads, kvheads = checkpoint.attention_kvheads, dim = checkpoint.attention_dim;
    const int top_k = 2048, ratio = 4, iheads = 4, idim = 128;  // attention_core's QSA shape
    const bool has_indexer = checkpoint.has_tensor(prefix + "indexer.k_proj.weight");
    if (segs.size() < 2 || segs.size() > size_t(tqllm_cuda::AttnSeqs::kMax) || dim % 32 || dim > 256) return false;
    int longest = 0;
    for (const auto & sg : segs) {
        if (!sg.carry || sg.n >= 32 || (has_indexer && sg.carry->len + sg.n > top_k) ||
            sg.carry->len + sg.n > kv_cap(*sg.carry, checkpoint.rt.kv_capacity)) return false;
        longest = std::max(longest, sg.carry->len + sg.n);
    }
    if (!tqllm::flashnext::attention_multi_fits(n, heads, dim, longest)) return false;
    const size_t row = size_t(kvheads) * dim;
    tqllm_cuda::AttnSeqs seqs; seqs.count = int(segs.size());
    for (size_t i = 0; i < segs.size(); ++i) {
        AttentionState & carry = *segs[i].carry;
        if (!carry.k) {
            const int cap = kv_cap(carry, checkpoint.rt.kv_capacity);
            PersistentAlloc keep; carry.k = std::make_unique<KvBuf>(size_t(cap) * row * sizeof(__nv_bfloat16)); carry.v = std::make_unique<KvBuf>(size_t(cap) * row * sizeof(__nv_bfloat16));
        }
        seqs.off[i] = segs[i].off; seqs.n[i] = segs[i].n; seqs.pos[i] = carry.len; seqs.k[i] = carry.k->p; seqs.v[i] = carry.v->p;
    }
    Device<float> q(size_t(n) * heads * dim), kn(size_t(n) * row);
    Device<float2> table(size_t(n) * (checkpoint.rotary / 2));
    tqllm_cuda::fn_rope_table_multi(table.p, seqs, checkpoint.rotary, checkpoint.theta, nullptr);
    tqllm_cuda::fn_norm_rope_table(pre.qg, checkpoint.dense_device(prefix + "attn_q_norm.weight"), q.p, table.p, heads, dim, n, checkpoint.rotary,
                                   float(checkpoint.epsilon), true, nullptr);
    tqllm_cuda::fn_norm_rope_table(pre.k, checkpoint.dense_device(prefix + "attn_k_norm.weight"), kn.p, table.p, kvheads, dim, n, checkpoint.rotary,
                                   float(checkpoint.epsilon), false, nullptr);
    Device<__nv_bfloat16> kb(kn.count), vb(kn.count);
    tqllm_cuda::f32_to_bf16(kn.p, kb.p, kn.count, nullptr);
    tqllm_cuda::f32_to_bf16(pre.v, vb.p, vb.count, nullptr);
    std::vector<tqllm_cuda::ScatterItem> appends;
    for (int i = 0; i < seqs.count; ++i) {
        const size_t bytes = size_t(seqs.n[i]) * row * sizeof(__nv_bfloat16), at = size_t(seqs.pos[i]) * row, from = size_t(seqs.off[i]) * row;
        appends.push_back({reinterpret_cast<const char *>(kb.p + from), reinterpret_cast<char *>(segs[size_t(i)].carry->k->p + at), bytes});
        appends.push_back({reinterpret_cast<const char *>(vb.p + from), reinterpret_cast<char *>(segs[size_t(i)].carry->v->p + at), bytes});
    }
    tqllm_cuda::scatter_copy(appends.data(), int(appends.size()), nullptr);
    if (has_indexer) {
        ArenaMark scratch;
        Device<float> ik(size_t(n) * idim);
        checkpoint.project(prefix + "indexer.k_proj.weight", input_p, ik.p, c.hidden, idim, n);
        const float * knorm = checkpoint.dense_device(prefix + "indexer.k_norm.weight");
        std::vector<tqllm_cuda::QsaIndexer *> ixs; std::vector<int> offs, ns;
        for (size_t i = 0; i < segs.size(); ++i) {
            AttentionState & carry = *segs[i].carry;
            const int pos = carry.len;
            if (!carry.qsa) {
                { size_t fr = 0, tot = 0; ck(cudaMemGetInfo(&fr, &tot));
                  if (fr < (size_t(1) << 30)) { cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); } }
                PersistentAlloc keep; carry.qsa = std::make_unique<tqllm_cuda::QsaIndexer>(kv_cap(carry, checkpoint.rt.kv_capacity), ratio, top_k, iheads, idim,
                                   std::min(checkpoint.rotary, idim), checkpoint.theta, float(checkpoint.epsilon)); }
            if (carry.qsa->size() > pos) carry.qsa->truncate(pos);
            ixs.push_back(carry.qsa.get()); offs.push_back(segs[i].off); ns.push_back(segs[i].n);
        }
        tqllm_cuda::QsaIndexer::append_multi(ixs, ik.p, offs, ns, knorm, nullptr);
    }
    Device<float> ws(tqllm_cuda::fn_dense_attention_multi_workspace(heads, dim, seqs));
    tqllm_cuda::fn_dense_attention_multi(q.p, seqs, raw_out, heads, kvheads, dim, ws.p, nullptr);
    for (const auto & sg : segs) sg.carry->len += sg.n;
    return true;
}
struct MixerResult { std::vector<float> mixed, inject; };
// Device GR Read for decode: mixed [n][h] and inject [n][streams] stay on the device.
// normalized_bf16: the BF16 image of `normalized`, made by hc_norm_bf16; the down / inject projections read it and
// the low-rank activation is written with its image too, so no projection converts its input
// rs (prefill sub-chunks > 64 tokens): `normalized` is only a key for the BF16 image; the mix recomputes the normalized rows from
// (rs_x, rs, rs_gamma), the inputs the norm read
inline void mix_device(const Checkpoint & checkpoint, const std::string & prefix, const float * normalized, int n,
                       float * mixed, float * inject, const __nv_bfloat16 * normalized_bf16 = nullptr,
                       const float * rs_x = nullptr, const float * rs = nullptr, const float * rs_gamma = nullptr) {
    const auto & c = checkpoint.config; const int h = c.hidden, streams = c.hc, width = h * streams;
    Device<float> low(size_t(n) * c.hc_low_rank), logits(size_t(n) * width);
    if (normalized_bf16) {
        Device<__nv_bfloat16> lowb(low.count);
        { const Checkpoint::SharedBf16 img(checkpoint, normalized, size_t(n) * width, normalized_bf16);
          checkpoint.project(prefix + "down.weight", normalized, low.p, width, c.hc_low_rank, n);
          tqllm_cuda::hc_low_silu_bf16(low.p, lowb.p, int(low.count), streams, nullptr);
          if (inject) checkpoint.project(prefix + "inject.weight", normalized, inject, width, streams, n); }
        { const Checkpoint::SharedBf16 img(checkpoint, low.p, low.count, lowb.p);
          checkpoint.project(prefix + "up.weight", low.p, logits.p, c.hc_low_rank, width, n); }
        if (rs) tqllm_cuda::hc_mix_rs(rs_x, rs, rs_gamma, logits.p, mixed, h, streams, n, nullptr);
        else tqllm_cuda::hc_mix(normalized, logits.p, mixed, h, streams, n, nullptr);
        return;
    }
    checkpoint.project(prefix + "down.weight", normalized, low.p, width, c.hc_low_rank, n);
    tqllm_cuda::hc_low_silu(low.p, int(low.count), streams, nullptr);
    checkpoint.project(prefix + "up.weight", low.p, logits.p, c.hc_low_rank, width, n);
    tqllm_cuda::hc_mix(normalized, logits.p, mixed, h, streams, n, nullptr);
    if (inject) checkpoint.project(prefix + "inject.weight", normalized, inject, width, streams, n);
}
// quantised mixer weights on row scales: the down / inject projections read the FP16 image as their converted input (no
// fp32 normalized rows, no conversion of the wide rows); the mix recomputes the normalized values like hc_mix_rs
// img_bf16: the BF16 image for BF16 inject weights; null: the inject weights read img too (quantised, or BF16 through their FP16
// copy: no second image of the wide rows)
inline void mix_device_h(const Checkpoint & checkpoint, const std::string & prefix, __half * img, const __nv_bfloat16 * img_bf16, int n,
                         float * mixed, float * inject, const float * rs_x, const float * rs, const float * rs_gamma, __half * mixed_h = nullptr,
                         __nv_bfloat16 * mixed_b = nullptr, const __nv_bfloat16 * rs_xb = nullptr) {
    const auto & c = checkpoint.config; const int h = c.hidden, streams = c.hc, width = h * streams;
    // rs_xb: the rows in BF16 storage (layer-major prefill) instead of rs_x
    auto mix = [&](const __half * lg) {
        if (rs_xb) tqllm_cuda::hc_mix_rs_h(rs_xb, rs, rs_gamma, lg, mixed, h, streams, n, nullptr, mixed_h, mixed_b);
        else tqllm_cuda::hc_mix_rs_h(rs_x, rs, rs_gamma, lg, mixed, h, streams, n, nullptr, mixed_h, mixed_b);
    };
    Device<__half> logits(size_t(n) * width);  // the gate logits in FP16 (sigmoid input)
    const float * key = reinterpret_cast<const float *>(img);  // never read: project takes the shared FP16 rows
    // prefill blocks: down and inject in one GEMM over the image, then the up projection reads the split's FP16 rows
    if (inject && n > 64) {
        const int lr = c.hc_low_rank, ld = Checkpoint::stacked_ld(lr, streams);
        Device<float> stacked(size_t(n) * ld);
        if (checkpoint.project_stacked_h(prefix + "down.weight", prefix + "inject.weight", img, stacked.p, width, lr, streams, n)) {
            Device<__half> lowh(size_t(n) * lr);
            tqllm_cuda::hc_low_split_h(stacked.p, ld, lr, streams, lowh.p, inject, n, nullptr);
            checkpoint.project_hh(prefix + "up.weight", lowh.p, logits.p, lr, width, n);
            mix(logits.p);
            return;
        }
    }
    Device<float> low(size_t(n) * c.hc_low_rank);
    { const Checkpoint::SharedInput shared(checkpoint, key, size_t(n) * width, img);
      require(shared.set, "HC mixer (fp16 image): projections are not resident GPU projections");
      const Checkpoint::SharedBf16 bimg(checkpoint, img_bf16 ? key : nullptr, size_t(n) * width, img_bf16);
      checkpoint.project(prefix + "down.weight", key, low.p, width, c.hc_low_rank, n);
      if (inject && !img_bf16 && checkpoint.is_bf16(prefix + "inject.weight"))
          checkpoint.project_h_bf16w(prefix + "inject.weight", img, inject, width, streams, n);
      else if (inject) checkpoint.project(prefix + "inject.weight", key, inject, width, streams, n); }
    tqllm_cuda::hc_low_silu(low.p, int(low.count), streams, nullptr);
    checkpoint.project(prefix + "up.weight", low.p, nullptr, c.hc_low_rank, width, n, logits.p);
    mix(logits.p);
}
inline MixerResult replay_mix(const Fixtures & fixtures, const Checkpoint & checkpoint, const std::string & suffix,
                              size_t occurrence, const Device<float> & normalized, bool head = false, bool screen = true) {
    const auto & c = checkpoint.config;
    const int h = c.hidden, streams = c.hc, width = h * streams, n = int(normalized.count / width);
    const std::string prefix = head ? "output_hc_" : "blk." + suffix.substr(1) + (occurrence ? ".hc_ffn_" : ".hc_attn_");
    Device<float> low(size_t(n) * c.hc_low_rank), logits(normalized.count), mixed(size_t(n) * h);
    checkpoint.project(prefix + "down.weight", normalized.p, low.p, width, c.hc_low_rank, n);
    tqllm_cuda::hc_low_silu(low.p, int(low.count), streams, nullptr);
    checkpoint.project(prefix + "up.weight", low.p, logits.p, c.hc_low_rank, width, n);
    auto gates = logits.get();
    for (auto & value : gates) value = 1.f / (1.f + std::exp(-value));
    const std::string tag = head ? "head" : suffix + "/" + std::to_string(occurrence);
    // A 2 percent stage screen covers the reference's Q8 activation rounding, not full-model KLD.
    if (screen) compare("hc_projected_gate" + tag, gates, fixtures.values("hc_gate" + suffix, occurrence), .02);
    tqllm_cuda::hc_mix(normalized.p, logits.p, mixed.p, h, streams, n, nullptr);
    MixerResult result{mixed.get(), {}};
    if (head && fixtures.has("result_norm")) {
        if (screen) compare("hc_head_collapse", result.mixed, fixtures.values("result_norm"), .02);
    } else if (!head) {
        if (screen) compare("hc_projected_mix" + tag, result.mixed, fixtures.values("hc_mixed" + suffix, occurrence), .02);
        Device<float> inject(size_t(n) * streams);
        checkpoint.project(prefix + "inject.weight", normalized.p, inject.p, width, streams, n);
        result.inject = inject.get();
        if (screen) compare("hc_projected_inject" + tag, result.inject, fixtures.values("hc_inject" + suffix, occurrence), .02);
    }
    return result;
}
