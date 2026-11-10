// qwen35 engine: speculative decoding (DFlash, MTP, verify, slots). The class is declared in qwen35_engine.hpp.
#include "qwen35_rules.hpp"
#include "qwen35_engine.hpp"

namespace tqllm {
namespace qwen35_detail {

void Qwen35Engine::verify(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) {
    if (!fmt_target) throw std::runtime_error("verify() needs a quantized GGUF model (a registered weight format)");
    if (n < 1 || n > NVER || pos + n > context_) throw std::runtime_error("verify: bad token count or position");
    for (int i = 0; i < n; ++i) h_tok[i] = tokens[i];
    h_pos[0] = int(pos);
    h_pos[1] = int(pos) + 1;
    verify_async(h_tok, h_pos, n, pos, next, true);
    if (sampling()) {
        ck(cudaMemcpyAsync(h_sid.data(), s_id.p, size_t(n) * 16 * 4, cudaMemcpyDeviceToHost, stream));
        ck(cudaMemcpyAsync(h_slg.data(), s_lg.p, size_t(n) * 16 * 4, cudaMemcpyDeviceToHost, stream));
    }
    sync_ck();
}

void Qwen35Engine::verify_body(int n, int max_seq) {
    const int H = int(cfg.hidden), lkh = int(cfg.lin_kheads), lvh = int(cfg.lin_vheads);
    const int dk = int(cfg.lin_dk), dv = int(cfg.lin_dv);
    const int qkv_dim = 2 * lkh * dk + lvh * dv, zdim = lvh * dv;
    auto *X = v_x.as<bf16>(), *M = v_m.as<bf16>(), *T = v_in.as<bf16>();
    auto *O = v_o.as<bf16>(), *CORE = v_core.as<bf16>();
    auto * XH = v_xh.as<__half>();
    const auto * dtok = v_tokens.as<uint32_t>();
    embed_multi(dtok, n, X);
    if (mm_chunk) tqllm_cuda::mm_scatter(X, mm_tab.as<int>(), n, mm_img.as<bf16>(), H, stream);  // image tokens
    tqllm_cuda::sumsq(X, v_px.as<float>(), n * H, stream);
    st("embed");
    const uint32_t n_attn = uint32_t(key_cache.size());
    if (kv_staged) {  // the first two attention layers' old positions, one per staging buffer, behind earlier work
        ck(cudaEventRecord(kv_fork, stream));
        ck(cudaStreamWaitEvent(kv_side, kv_fork, 0));
        kv_prefetch(0);
        if (n_attn > 1) kv_prefetch(1);
    }
    uint32_t attn_i = 0, lin_i = 0;
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        const auto & L = layer[l];
        capture_feature(l, X, v_feat.as<bf16>(), n);
        nrot_h(X, L.in_norm, v_px.as<float>(), XH, H, n);
        st("in_norm_rot(+feature)");
        if (cfg.full[l]) {
            if (kv_staged) ck(cudaStreamWaitEvent(stream, kv_ready[attn_i & 1], 0));
            cmp_layer = int(attn_i);
            if (segs.empty()) attn_mixer(L, key_cache[attn_i].as<bf16>(), value_cache[attn_i].as<bf16>(), n, max_seq);
            else attn_mixer(L, nullptr, nullptr, n, max_seq, int(attn_i));
            cmp_layer = -1;
            if (kv_staged && attn_i + 2 < n_attn) {  // layer attn_i + 2 shares this staging buffer: refill it now
                ck(cudaEventRecord(kv_done[attn_i & 1], stream));
                ck(cudaStreamWaitEvent(kv_side, kv_done[attn_i & 1], 0));
                kv_prefetch(attn_i + 2);
            }
            ++attn_i;
        } else {
            if (!L.w_mix.p) throw std::runtime_error("verify path needs fused Gated DeltaNet weights");
            const int rows = mix_rows(L);
            // verify blocks project straight into this layer's stash, which commit(k < n) replays (no copy);
            // prefill chunks are always committed whole
            bf16 * TL = n <= NVER || !segs.empty() ? v_stash2[sp][lin_i].as<bf16>() : T;
            mix_mm(L, TL, n);
            st("gdn_mix_matvec(+ba)");
            bf16 * QKVC = v_qkvc.as<bf16>();
            if (segs.empty()) {
                tqllm_cuda::ConvReplay crp;
                tqllm_cuda::GdnReplay grp;
                if (defer && vpar == d_tpar.as<int>()) {  // replay the previous step's path first (deferred commit)
                    const bf16 * RT = v_stash2[sp ^ 1][lin_i].as<bf16>();
                    crp.x = RT, crp.pend = d_pend.as<int>(), crp.smid = conv_slot(lin_i, cur + 1);
                    crp.y = d_rep_qkvc.as<bf16>();
                    grp.qkv = d_rep_qkvc.as<bf16>(), grp.b0 = RT + qkv_dim + zdim, grp.a0 = grp.b0 + lvh;
                    grp.abs_ = rows, grp.pend = d_pend.as<int>(), grp.smid = rec_slot(lin_i, cur + 1);
                }
                tqllm_cuda::conv_multi(TL, rows, n, conv_slot(lin_i, cur), conv_slot(lin_i, cur + 1), L.conv_w, QKVC,
                                       qkv_dim, stream, vpar, crp);
                st("gdn_conv");
                const bf16 * B0 = TL + qkv_dim + zdim;  // row layout [qkv | z | b | a]
                tqllm_cuda::gdn_multi(QKVC, n, B0 + lvh, B0, rows, L.neg_a, L.dt_bias, rec_slot(lin_i, cur),
                                      rec_slot(lin_i, cur + 1), CORE, lkh, lvh, dk, dv, cfg.tiled, stream, vpar, grp);
            } else {  // batched verify: the recurrence runs per segment on that slot's state
                for_segs([&](size_t i) {
                    const auto & g = segs[i];
                    const int c = cur_of(g.slot);
                    auto & cs = conv_of(g.slot);
                    auto & rs = rec_of(g.slot);
                    const size_t b = size_t(lin_i) * NSLOT;
                    const bf16 * Tg = TL + size_t(g.col0) * rows;
                    bf16 * Qg = QKVC + size_t(g.col0) * qkv_dim;
                    const int * gp = g.tree ? d_tpar.as<int>() + g.col0 : tree_mode ? d_chain.as<int>() : nullptr;
                    tqllm_cuda::conv_multi(Tg, rows, g.n, cs[b + c % NSLOT].as<bf16>(), cs[b + (c + 1) % NSLOT].as<bf16>(),
                                           L.conv_w, Qg, qkv_dim, stream, gp);
                    const bf16 * B0 = Tg + qkv_dim + zdim;
                    tqllm_cuda::gdn_multi(Qg, g.n, B0 + lvh, B0, rows, L.neg_a, L.dt_bias, rs[b + c % NSLOT].as<float>(),
                                          rs[b + (c + 1) % NSLOT].as<float>(), CORE + size_t(g.col0) * zdim, lkh, lvh,
                                          dk, dv, cfg.tiled, stream, gp);
                });
            }
            st("gdn_recurrence");
            if (basis_ == Basis::Tq3Rot32) {
                tqllm_cuda::gated_rms_rot_h(CORE, TL + qkv_dim, rows, n, L.gnorm_w, XH, lvh, dv, cfg.eps, stream);
            } else {
                tqllm_cuda::gated_rms_multi(CORE, TL + qkv_dim, rows, n, L.gnorm_w, O, lvh, dv, cfg.eps, stream);
                rot_h(O, XH, zdim, n);
            }
            st("gdn_gated_rms+rot");
            mm_h(L.w_out, M, n);
            st("gdn_out_matvec");
            ++lin_i;
        }
        ffn_block(L, n, X);
    }
    // final hidden states (pre-norm) feed the MTP drafter
    ck(cudaMemcpyAsync(v_hid.p, X, size_t(n) * H * 2, cudaMemcpyDeviceToDevice, stream));
    if (!segs.empty()) {  // batched verify: one head pass over every slot's columns (per-column results unchanged)
        head_argmax(out_norm, n, 0);
    } else if (n > NVER) {  // prefill chunk: only the final chunk's last position needs the head
        if (prefill_final) {
            ck(cudaMemcpyAsync(X, X + size_t(n - 1) * H, size_t(H) * 2, cudaMemcpyDeviceToDevice, stream));
            head_argmax(out_norm, 1);
        }
    } else {
        head_argmax(out_norm, n);
    }
    st("head");
}

void Qwen35Engine::dflash_inject_body(const bf16 * feats, int n) {  // positions from d_pos
    const int H = int(cfg.hidden), kd = dfl.kvheads * dfl.hd, nl = int(dfl.L.size());
    auto *G = d_inj_g.as<bf16>(), *KV = d_inj_kv.as<bf16>();
    inj_mm(dfl.fc, feats, G, n);
    tqllm_cuda::rms_f_multi(G, H, 0, dfl.enc_norm, G, H, 0, H, 1, n, cfg.eps, stream);
    inj_mm(dfl.w_kv_all, G, KV, n);
    const int ts = dfl.w_kv_all.rows;
    for (int l = 0; l < nl; ++l) {
        bf16 * K = KV + size_t(l) * 2 * kd;
        tqllm_cuda::rms_f_multi(K, ts, dfl.hd, dfl.L[l].k_norm, K, ts, dfl.hd, dfl.hd, dfl.kvheads, n, cfg.eps, stream);
        tqllm_cuda::rope_partial_multi(K, ts, n, dfl.kvheads, dfl.hd, dfl.hd, d_pos.as<int>(), dfl.theta, stream);
        tqllm_cuda::cache_write_multi(K, K + kd, ts, n, dfl_kc[l].as<bf16>(), dfl_vc[l].as<bf16>(), d_pos.as<int>(),
                                      dfl.kvheads, dfl.hd, stream, dfl.ring);
    }
}

void Qwen35Engine::dflash_begin(const std::vector<uint32_t> & prompt, uint32_t count, int D) {
    const auto P = uint32_t(prompt.size());
    const int H = int(cfg.hidden), nf = int(dfl.target_ids.size());
    // image requests decode without MTP (exact either way), so no request fails for asking for them. Sampled requests
    // keep MTP: its drafts are deterministic (argmax), so accept() / accept_drafts give exact samples of the target's p.
    const bool use_mtp = opt.native_mtp && mtp.present && mm_pending.empty();
    const bool dfl_on = dfl.present;
    if (use_mtp) {
        if (!mtp.present) throw std::runtime_error("--mtp needs a model with an MTP head");
        if (nslots != 1) throw std::runtime_error("--mtp currently needs --np 1");
        if (D < 1 || D >= NVER) throw std::runtime_error("MTP draft length exceeds verify capacity");
    }
    // no drafter: one verified token per step; a tree may spend up to NVER - 1 nodes (depth still <= block - 1)
    if (D <= 0 && dfl_on) D = dfl.dspark ? 3 : 7;  // GB10 MiniCPM-V-4.7 + DSpark chain: 3 -> 147.9 / 86.6 / 163.6, 7 -> 140.2 / 66.1 / 149.2
    D = dfl_on ? std::min(D, tree_mode && !sampling() ? NVER - 1 : dfl.block - 1) : use_mtp ? D : 0;
    const int reserve = use_mtp ? D + 1 : dfl.block + 1;
    if (P == 0 || count == 0 || uint64_t(P) + count + reserve > context_)
        throw std::runtime_error("request exceeds context capacity");
    if (use_mtp) {
        ck(cudaMemsetAsync(mtp_kc.p, 0, kvf->key_bytes(context_, int(cfg.kvheads), int(cfg.head_dim)), stream));
        ck(cudaMemsetAsync(mtp_vc.p, 0, kvf->value_bytes(context_, int(cfg.kvheads), int(cfg.head_dim)), stream));
    }
    gen.rdelta = 0;
    // image prompt: per-token table (source row, t, h, w); an image takes nx * ny cache cells but advances the RoPE
    // position by max(nx, ny) (Qwen3-VL / llama.cpp mtmd)
    const auto images = std::exchange(mm_pending, {});
    HostBuf staged_mm(images.empty() ? 0 : size_t(P) * 4 * sizeof(int));
    int rdelta = 0;
    if (!images.empty()) {
        auto * tab = static_cast<int *>(staged_mm.p);
        int p = 0;
        size_t k = 0;
        for (uint32_t i = 0; i < P;) {
            if (k < images.size() && i == images[k].at) {
                const auto & im = images[k++];
                const uint32_t nt = uint32_t(im.nx) * uint32_t(im.ny);
                if (i + nt > P) throw std::runtime_error("image placeholders run past the prompt");
                for (uint32_t r = 0; r < nt; ++r) {
                    int * e = tab + 4 * size_t(i + r);
                    e[0] = int(im.row0 + r);
                    e[1] = p;
                    e[2] = p + int(r) / im.nx;
                    e[3] = p + int(r) % im.nx;
                }
                i += nt;
                p += std::max(im.nx, im.ny);
            } else {
                int * e = tab + 4 * size_t(i);
                e[0] = -1, e[1] = e[2] = e[3] = p;
                ++i, ++p;
            }
        }
        if (k != images.size()) throw std::runtime_error("image placeholders out of order or beyond the prompt");
        rdelta = int(P) - p;
    }
    // prompt reuse: positions [0, P0) are already cached; a boundary at snap keeps the state for the next request
    const uint32_t P0 = pc_on && images.empty() ? pc_match(prompt) : 0;
    if (P0 < pc_snap) pc_snap = 0;
    pc_live = false;
    const uint32_t snap = pc_on && P >= P0 + 2 * kTail ? P - kTail : 0;
    if (pc_on) pc_seq.assign(prompt.begin(), prompt.end());
    if (P0) std::fprintf(stderr, "tqllm: prompt cache: %u of %u prompt tokens reused\n", P0, P);
    if (kv_cmp && opt.kv_compact_turn && cmp_pos) cmp_reset();  // --kv-compact-at turn: the new tokens see exact KV
    // Queue prefill chunks on one stream and read back only the final next token.
    uint32_t next = 0;
    uint32_t first_draft = 0;
    {
        // TQLLM_DEBUG=tf_dump:FILE: teacher-forced argmax at every prompt position ("pos argmax next_prompt_token")
        const std::string * tf_dump = tqllm::debug_value("tf_dump");
        const char * tf_path = tf_dump && !tf_dump->empty() ? tf_dump->c_str() : nullptr;
        // prefill chunk: NPF tokens through the GEMM path (the dump needs every argmax)
        const int CH = tf_path ? NVER : std::clamp(NPF, 1, NBUF);
        const size_t chunks = (P - P0 + CH - 1) / CH + 1;  // + 1: the split at snap
        HostBuf staged_tokens(size_t(P) * sizeof(uint32_t));
        HostBuf staged_pos(chunks * 2 * sizeof(int));
        std::memcpy(staged_tokens.p, prompt.data(), size_t(P) * sizeof(uint32_t));
        auto * positions = static_cast<int *>(staged_pos.p);
        HostBuf tf_buf(tf_path ? size_t(P) * 4 : 4);
        uint32_t * tf_host = tf_path ? static_cast<uint32_t *>(tf_buf.p) : nullptr;
        const auto tp0 = std::chrono::steady_clock::now();
        const bool show_progress = tqllm::debug_on("prefill");
        int m_last = 0;
        // drafter K/V ring: positions older than P - ring are overwritten before the first draft, so skip them
        const uint32_t inj0 = dfl.ring > 0 && P > uint32_t(dfl.ring) ? P - uint32_t(dfl.ring) : 0;
        try {
            for (uint32_t t = P0, ci = 0; t < P; t += uint32_t(m_last), ++ci) {
                const int m = int(std::min<uint32_t>(std::min<uint32_t>(uint32_t(CH), P - t), snap > t ? snap - t : P));
                m_last = m;
                positions[2 * ci] = int(t);
                positions[2 * ci + 1] = int(t) + 1;
                const bool final = t + uint32_t(m) == P;
                prefill_final = final;
                mm_chunk = !images.empty();
                if (mm_chunk)
                    ck(cudaMemcpyAsync(mm_tab.p, static_cast<int *>(staged_mm.p) + 4 * size_t(t), size_t(m) * 4 * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
                verify_async(static_cast<const uint32_t *>(staged_tokens.p) + t, positions + 2 * ci,
                             m, t, final ? h_tok : nullptr, false);
                if (tf_host) ck(cudaMemcpyAsync(tf_host + t, v_tok.p, size_t(m) * 4, cudaMemcpyDeviceToHost, stream));
                commit(m);
                if (use_mtp) {
                    if (final) {
                        sync_ck();
                        next = h_tok[0];
                        if (sampling()) {  // sample the first token now: the first MTP draft below must follow it, and
                                           // the draft's head pass may overwrite column 0's top-16 on the device
                            ck(cudaMemcpy(h_sid.data(), s_id.p, 16 * 4, cudaMemcpyDeviceToHost));
                            ck(cudaMemcpy(h_slg.data(), s_lg.p, 16 * 4, cudaMemcpyDeviceToHost));
                            next = draw(dist(0), rng);
                        }
                    }
                    const int pairs = int(std::min<uint32_t>(uint32_t(m), P - t - 1));
                    for (int off = 0; off < pairs; off += NVER) {
                        const int n = std::min(NVER, pairs - off);
                        mtp_run(prompt.data() + t + uint32_t(off) + 1,
                                v_hid.as<bf16>() + size_t(off) * H, n, t + uint32_t(off) + 1, false);
                    }
                    if (final && count > 1 && next != eos_)
                        first_draft = mtp_run(&next, v_hid.as<bf16>() + size_t(m - 1) * H, 1, P);
                }
                for (int off = 0; off < m && dfl_on; off += DFlash::injBatch) {
                    const int count = std::min(DFlash::injBatch, m - off);
                    if (t + uint32_t(off + count) <= inj0) continue;
                    dflash_inject(v_feat.as<bf16>() + size_t(off) * nf * H, count, t + uint32_t(off));
                }
                if (t + uint32_t(m) == snap) pc_copy(true), pc_snap = snap;
                tqllm::prefill_completed(stream,t+uint32_t(m),P,P0);
                if (show_progress && (final || (t + uint32_t(m)) % 8192 < uint32_t(m))) {
                    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
                    const uint32_t done = t + uint32_t(m);  // rate counts this request's new tokens (not reused ones)
                    std::fprintf(stderr, "prefill progress %u/%u tokens (%.0f%%, %.0f tok/s)\n", done, P,
                                 100.0 * done / P, (done - P0) / elapsed);
                }
            }
            sync_ck();
            mm_chunk = false;
        } catch (...) {
            cudaStreamSynchronize(stream);
            mm_chunk = false;
            throw;
        }
        if (tf_host) {
            if (FILE * f = std::fopen(tf_path, "w")) {
                for (uint32_t i = 0; i < P; ++i)
                    std::fprintf(f, "%u %u %d\n", i, tf_host[i], i + 1 < P ? int(prompt[i + 1]) : -1);
                std::fclose(f);
            }
        }
        if (!use_mtp) next = h_tok[0];
        if (sampling() && !use_mtp) {  // the final chunk's head wrote column 0's top-16 (MTP: sampled in the loop)
            ck(cudaMemcpy(h_sid.data(), s_id.p, 16 * 4, cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(h_slg.data(), s_lg.p, 16 * 4, cudaMemcpyDeviceToHost));
            next = draw(dist(0), rng);
        }
        if (kv_cmp && opt.kv_compact_turn) {  // refit the old blocks on this request's queries before decoding
            const auto tc = std::chrono::steady_clock::now();
            int nb = 0;
            for (; cmp_pending(); ++nb) kv_idle_step();
            if (nb) std::fprintf(stderr, "tqllm: TqCache: %d blocks compacted for this turn in %.0f ms\n", nb,
                                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count());
        }
        last_prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
    }
    gen = Gen{};
    gen.out.assign(1, next);
    if (opt.lookup) gen.hist.assign(prompt.begin(), prompt.end());
    gen.P = P;
    gen.count = count;
    gen.D = D;
    gen.native_mtp = use_mtp;
    if (use_mtp && count > 1 && next != eos_) gen.mtp_draft.assign(1, first_draft);
    gen.pos = P;
    gen.rdelta = rdelta;
    gen.pending = next;
    gen.done = false;
    gen.prefill_ms = last_prefill_ms;
    gen.decode_start = std::chrono::steady_clock::now();
    dec_log_tp = gen.decode_start;
}

int Qwen35Engine::accept(const std::vector<uint32_t> & draft, uint32_t * res, int col0) {
    int j = 0;
    if (sampling()) {  // speculative sampling with a deterministic draft: exact samples of the target's p
        std::vector<uint32_t> recent;  // penalty window: the output so far, then each draft token as it is accepted
        if (samp.penalized()) {
            const size_t n = gen.out.size(), r0 = n > size_t(kPresenceLastN) ? n - kPresenceLastN : 0;
            recent.assign(gen.out.begin() + long(r0), gen.out.end());
        }
        j = accept_drafts(draft.data(), int(draft.size()), [&](int i, const std::vector<uint32_t> & r) { return dist(col0 + i, r); },
                          rng, res, recent, samp.penalized());
    } else {
        while (j < int(draft.size()) && draft[j] == res[j]) ++j;
    }
    return j;
}

void Qwen35Engine::dflash_step(const int * slots, int ns, bool * done) {
    flush_pending();  // only the active slot can hold a deferred commit (select_slot flushes before parking)
    struct Plan { std::vector<uint32_t> draft; };
    std::vector<Plan> plan;
    static const bool prof = tqllm::debug_on("steps");
    const auto p0 = std::chrono::steady_clock::now();
    segs.clear();
    batch_topk = false;
    int col = 0;
    // Column budget per device class (qwen35_rules.hpp): past it the verify matmuls cost per column, so with many slots
    // each drafts fewer tokens.
    static const int budget = [this] {
        int dev = 0, major = 0, integrated = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev);
        return tqllm::qwen35::verify_column_budget(major, integrated != 0, NB);
    }();
    int live = 0;
    for (int i = 0; i < ns; ++i) {
        select_slot(slots[i]);
        if (gen.done || gen.out.size() >= gen.count || gen.pending == eos_) gen.done = true;
        live += !gen.done;
    }
    const int per = std::max(2, budget / std::max(1, live));
    std::vector<DraftReq> rq;
    for (int i = 0; i < ns; ++i) {
        select_slot(slots[i]);
        if (!gen.done) rq.push_back({slots[i], gen.pending, gen.pos, std::min(gen.D, per - 1)});
    }
    if (tree_mode && dfl.present && rq.size() >= 2 && tree_step_batch(rq, slots, ns, done, p0))
        return;
    std::vector<std::vector<uint32_t>> drafts;
    // device chain: anchors uploaded first, the drafter's chains land in v_tokens and the verify is queued behind
    // them with no host round trip; the drafts are read back with the verify results
    std::vector<int> ccols;
    bool on_dev = false;
    if (dfl.present && rq.size() >= 2 && dfl.block > 1) {
        int c = 0;
        for (auto & r : rq) {
            r.want = std::min(r.want, dfl.block - 1);
            ccols.push_back(c);
            h_tok[c] = r.anchor;
            c += 1 + r.want;
        }
        if (c > 0) ck(cudaMemcpyAsync(v_tokens.p, h_tok, size_t(c) * 4, cudaMemcpyHostToDevice, stream));
        on_dev = dflash_draft_batch(rq, drafts, nullptr, &ccols);
        if (on_dev)
            for (const auto & r : rq) drafts.emplace_back(size_t(r.want), 0u);  // filled after the verify
    }
    const bool batched = on_dev || (dfl.present && dflash_draft_batch(rq, drafts));
    size_t ri = 0;
    for (int i = 0; i < ns; ++i) {
        select_slot(slots[i]);
        done[i] = gen.done;
        if (gen.done) continue;
        auto draft = batched ? std::move(drafts[ri++])
                     : gen.D > 0 ? dflash_draft(gen.pending, gen.pos, std::min(gen.D, per - 1))
                                 : std::vector<uint32_t>{};
        const int n = 1 + int(draft.size());
        if (gen.pos + uint32_t(n) > context_) throw std::runtime_error("verify: position beyond context");
        if (!on_dev) {
            h_tok[col] = gen.pending;
            std::copy(draft.begin(), draft.end(), h_tok + col + 1);
        }
        h_pos[2 * segs.size()] = int(gen.pos);
        h_pos[2 * segs.size() + 1] = int(gen.pos) + 1;
        segs.push_back({slots[i], col, n, gen.pos});
        plan.push_back({std::move(draft)});
        batch_topk |= sampling();
        col += n;
    }
    if (segs.empty()) return;
    const auto p1 = std::chrono::steady_clock::now();
    if (!on_dev) ck(cudaMemcpyAsync(v_tokens.p, h_tok, size_t(col) * 4, cudaMemcpyHostToDevice, stream));
    ck(cudaMemcpyAsync(v_pos.p, h_pos, 2 * segs.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    upload_rpos(int(segs.size()), [&](int i) { return rdelta_of(segs[size_t(i)].slot); });
    for (const auto & g : segs) kv_track(g.pos, g.n);
    uint64_t key = batch_graph_seed(batch_topk, unsigned(sp));  // stash buffer baked in
    for (const auto & g : segs) key = (key ^ uint64_t((g.slot * 64 + g.n) * 4 + cur_of(g.slot) % NSLOT)) * 1099511628211ull;
    ck(cudaGraphLaunch(cached_graph(kGraphBatch, key, [&] { verify_body(col, int(context_)); }), stream));
    std::vector<uint32_t> res(size_t(col) + 1);
    ck(cudaMemcpyAsync(res.data(), v_tok.p, size_t(col) * 4, cudaMemcpyDeviceToHost, stream));
    if (on_dev) ck(cudaMemcpyAsync(h_tok, v_tokens.p, size_t(col) * 4, cudaMemcpyDeviceToHost, stream));
    if (batch_topk) {
        ck(cudaMemcpyAsync(h_sid.data(), s_id.p, size_t(col) * 16 * 4, cudaMemcpyDeviceToHost, stream));
        ck(cudaMemcpyAsync(h_slg.data(), s_lg.p, size_t(col) * 16 * 4, cudaMemcpyDeviceToHost, stream));
    }
    sync_ck();
    const auto p2 = std::chrono::steady_clock::now();
    const auto batch = segs;
    segs.clear();
    batch_topk = false;
    if (on_dev)
        for (size_t b = 0; b < batch.size(); ++b)
            std::copy(h_tok + batch[b].col0 + 1, h_tok + batch[b].col0 + batch[b].n, plan[b].draft.begin());
    const int nf = int(dfl.target_ids.size()), H = int(cfg.hidden);
    size_t pi = 0;
    for (int i = 0; i < ns; ++i) {
        if (done[i]) continue;
        const auto & g = batch[pi];
        const auto & draft = plan[pi++].draft;
        select_slot(g.slot);
        uint32_t * r = res.data() + g.col0;
        const int j = accept(draft, r, g.col0);
        ++gen.steps;
        gen.drafted += draft.size();
        gen.accepted += uint64_t(j);
        bool hit_eos = false;
        for (int k = 0; k < j && gen.out.size() < gen.count; ++k) {
            gen.out.push_back(draft[size_t(k)]);
            hit_eos |= draft[size_t(k)] == eos_;
        }
        if (gen.out.size() < gen.count) gen.out.push_back(r[j]);
        last_n = g.n;
        stash_col0 = g.col0;
        commit(j + 1);
        stash_col0 = 0;
        if (dfl.present) dflash_inject(v_feat.as<bf16>() + size_t(g.col0) * nf * H, j + 1, gen.pos);
        gen.pending = r[j];
        gen.pos += uint32_t(j + 1);
        if (hit_eos || gen.out.size() >= gen.count || gen.pending == eos_) gen.done = true;
        done[i] = gen.done;
    }
    if (prof) {
        sync_ck();
        const auto p3 = std::chrono::steady_clock::now();
        auto msd = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::fprintf(stderr, "bstep segs %zu cols %d draft %.2f verify %.2f commit %.2f ms\n", batch.size(), col,
                     msd(p0, p1), msd(p1, p2), msd(p2, p3));
    }
}

int Qwen35Engine::lookup(uint32_t * dst, int maxlen) const {
    constexpr int ng = 6;
    const auto & h = gen.hist;
    const auto & o = gen.out;
    const size_t len = h.size() + o.size();
    auto at = [&](size_t i) { return i < h.size() ? h[i] : o[i - h.size()]; };
    if (maxlen <= 0 || len < size_t(ng) + 1) return 0;
    for (size_t end = len - 1; end-- > size_t(ng) - 1;) {  // candidate match ends at `end`, strictly before the suffix
        bool eq = true;
        for (int k = 0; k < ng && eq; ++k) eq = at(end - k) == at(len - 1 - k);
        if (!eq) continue;
        int n = 0;
        for (size_t i = end + 1; i < len && n < maxlen; ++i) dst[n++] = at(i);
        return n;
    }
    return 0;
}

bool Qwen35Engine::dflash_advance() {
    if (gen.done) return false;
    auto & out = gen.out;
    auto & pending = gen.pending;
    auto & pos = gen.pos;
    const uint32_t count = gen.count;
    const int D = gen.D;
    if (out.size() >= count || pending == eos_) {
        gen.done = true;
        return false;
    }
    std::vector<uint32_t> res(NVER), in;
    const bool prof = tqllm::debug_on("profile");
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    static const bool tree_chain = tqllm::debug_on("tree_chain");  // tree kernels, chain drafts (draft-independence check)
    if (tree_mode && !sampling() && D > 1 && !tree_chain) {  // token-tree step (greedy)
        // draft -> tree -> verify queued back to back on the device; one host sync after the verify
        const auto t0 = now();
        uint32_t lk[NVER];
        constexpr int NV = std::min(NVER, tqllm::kGdnReplayCols);
        int nl = opt.lookup ? lookup(lk, tqllm::lookup_limit(opt.lookup, NV)) : 0;
        // lookup-first (--lookup-first k): a context match of >= k tokens takes the whole verify block
        bool lf = false;
        if (opt.lookup_first > 0 && opt.lookup) {
            uint32_t lk2[NVER];
            const int n2 = lookup(lk2, tqllm::lookup_first_limit(NV));
            if (n2 >= opt.lookup_first) std::copy(lk2, lk2 + n2, lk), nl = n2, lf = true;
        }
        const int nt = tqllm::tree_cols(D, nl, lf, NV), n = nt + nl;
        dflash_draft(pending, pos, 0, nullptr, nullptr, true);
        tqllm_cuda::dflash_tree(d_idx.as<uint32_t>(), d_score.as<float>(), d_tok.as<uint32_t>(), dfl.block, nt - 1,
                                v_tokens.as<uint32_t>(), d_tpar.as<int>(), d_tdep.as<int>(), stream);
        if (nl) {  // lookup chain: columns nt .. n - 1, the first under the root
            int lp[NVER], ld[NVER];
            for (int i = 0; i < nl; ++i) lp[i] = i ? nt + i - 1 : 0, ld[i] = i + 1;
            ck(cudaMemcpyAsync(v_tokens.as<uint32_t>() + nt, lk, size_t(nl) * 4, cudaMemcpyHostToDevice, stream));
            ck(cudaMemcpyAsync(d_tpar.as<int>() + nt, lp, size_t(nl) * 4, cudaMemcpyHostToDevice, stream));
            ck(cudaMemcpyAsync(d_tdep.as<int>() + nt, ld, size_t(nl) * 4, cudaMemcpyHostToDevice, stream));
        }
        const auto t1 = now();
        h_pos[0] = int(pos);
        h_pos[1] = int(pos) + 1;
        tree_step = true;
        verify_async(nullptr, h_pos, n, pos, res.data(), true);
        tree_step = false;
        std::vector<int> tpar(static_cast<size_t>(n) + 0);
        in.resize(size_t(n));
        ck(cudaMemcpyAsync(in.data(), v_tokens.p, size_t(n) * 4, cudaMemcpyDeviceToHost, stream));
        ck(cudaMemcpyAsync(tpar.data(), d_tpar.p, size_t(n) * 4, cudaMemcpyDeviceToHost, stream));
        sync_ck();
        const auto t2 = now();
        gen.t_draft += ms(t0, t1);
        gen.t_verify += ms(t1, t2);
        // follow the target's argmax down the tree; siblings may repeat a token (tree vs lookup chain): keep the deepest
        std::vector<int> path{0}, best;
        std::function<void(int)> walk = [&](int node) {
            bool leaf = true;
            for (int c = node + 1; c < n; ++c)
                if (tpar[size_t(c)] == node && in[size_t(c)] == res[size_t(node)]) {
                    leaf = false;
                    path.push_back(c);
                    walk(c);
                    path.pop_back();
                }
            if (leaf && path.size() > best.size()) best = path;
        };
        if (nl) {
            walk(0);
            path = best;
        } else {  // no lookup chain: sibling tokens are distinct, the first match is the only one
            for (int node = 0;;) {
                int next = -1;
                for (int c = node + 1; c < n && next < 0; ++c)
                    if (tpar[size_t(c)] == node && in[size_t(c)] == res[size_t(node)]) next = c;
                if (next < 0) break;
                path.push_back(next);
                node = next;
            }
        }
        const int j = int(path.size()) - 1;
        ++gen.steps;
        gen.drafted += uint64_t(n - 1);
        gen.accepted += uint64_t(j);
        bool hit_eos = false;
        for (int i = 1; i <= j && out.size() < count; ++i) {
            out.push_back(in[size_t(path[size_t(i)])]);
            hit_eos |= in[size_t(path[size_t(i)])] == eos_;
        }
        const uint32_t bonus = res[size_t(path.back())];
        if (out.size() < count) out.push_back(bonus);
        const auto t3 = now();
        bool contiguous = true;
        for (int i = 0; i <= j; ++i) contiguous &= path[size_t(i)] == i;
        if (!contiguous) {  // accepted path -> columns 0 .. j: GDN stashes, drafter features, attention K/V
            std::vector<tqllm_cuda::PathGather> g;
            const int nf = int(dfl.target_ids.size()), H = int(cfg.hidden);
            uint32_t lin_i = 0, attn_i = 0;
            for (uint32_t l = 0; l < cfg.layers; ++l) {
                if (cfg.full[l]) {
                    g.push_back({key_cache[attn_i].p, kv_row_bytes(true), 1});
                    g.push_back({value_cache[attn_i].p, kv_row_bytes(false), 1});
                    ++attn_i;
                } else {
                    g.push_back({v_stash2[sp][lin_i].p, int64_t(mix_rows(layer[l])) * 2, 0});
                    ++lin_i;
                }
            }
            g.push_back({v_feat.p, int64_t(nf) * H * 2, 0});
            ck(cudaMemcpyAsync(d_gdesc.p, g.data(), g.size() * sizeof(g[0]), cudaMemcpyHostToDevice, stream));
            ck(cudaMemcpyAsync(d_tcols.p, path.data(), path.size() * 4, cudaMemcpyHostToDevice, stream));
            tqllm_cuda::gather_path(d_gdesc.as<tqllm_cuda::PathGather>(), int(g.size()), d_tcols.as<int>(), j + 1,
                                    int(pos), stream);
        }
        if (defer) {  // deferred commit: the next verify replays this path in its GDN / conv kernels
            if (pend_k) cur = (cur + 1) % NSLOT;  // this verify wrote the previous path's state into slot cur + 1
            pend_sp = sp;
            sp ^= 1;
            set_pending(j + 1);
            last_n = 0;
            dflash_inject(v_feat.as<bf16>(), j + 1, pos);
        } else if (dfl_fixed_grids()) {
            // commit + inject as one graph: ~110 small launches otherwise leave the GPU waiting on the host
            const int k = j + 1;
            if (k > last_n) throw std::runtime_error("commit: accepted count out of range");
            set_dpos(pos);
            const uint64_t key = uint64_t(k) | (uint64_t(k < last_n) << 8) | (uint64_t(cur) << 16) |
                                 (uint64_t(sp) << 24) | (uint64_t(active) << 32);
            ck(cudaGraphLaunch(cached_graph(kGraphCommit, key, [&] {
                if (k < last_n) commit_body(k);
                dflash_inject_body(v_feat.as<bf16>(), k);
            }), stream));
            cur = (cur + 1) % NSLOT;
            last_n = 0;
        } else {
            commit(j + 1);
            dflash_inject(v_feat.as<bf16>(), j + 1, pos);
        }
        if (prof) {
            sync_ck();
            gen.t_tail += ms(t3, now());
        }
        pending = bonus;
        pos += uint32_t(j + 1);
        if (hit_eos) gen.done = true;
    } else {
        const auto t0 = now();
        const auto draft = dflash_draft(pending, pos, D);
        const auto t1 = now();
        in.assign(1, pending);
        in.insert(in.end(), draft.begin(), draft.end());
        verify(in.data(), int(in.size()), pos, res.data());
        const auto t2 = now();
        gen.t_draft += ms(t0, t1);
        gen.t_verify += ms(t1, t2);
        const int j = accept(draft, res.data(), 0);
        if (tqllm::debug_on("draft") && gen.steps < 6) {  // DEBUG
            std::fprintf(stderr, "step %llu pos %u pending %u\n  draft:", (unsigned long long) gen.steps, pos, pending);
            for (auto d : draft) std::fprintf(stderr, " %u", d);
            std::fprintf(stderr, "\n  truth:");
            for (size_t i = 0; i < draft.size() + 1; ++i) std::fprintf(stderr, " %u", res[i]);
            std::fprintf(stderr, "\n");
        }
        ++gen.steps;
        gen.drafted += draft.size();
        gen.accepted += j;
        bool hit_eos = false;
        for (int i = 0; i < j && out.size() < count; ++i) {
            out.push_back(draft[i]);
            hit_eos |= draft[i] == eos_;
        }
        if (out.size() < count) out.push_back(res[j]);
        const auto t3 = now();
        commit(j + 1);
        dflash_inject(v_feat.as<bf16>(), j + 1, pos);  // real features of the accepted positions
        if (prof) {
            sync_ck();
            gen.t_tail += ms(t3, now());
        }
        pending = res[j];
        pos += uint32_t(j + 1);
        if (hit_eos) gen.done = true;
    }
    if (out.size() >= count || pending == eos_) gen.done = true;
    dec_progress();
    return !gen.done;
}

std::vector<uint32_t> Qwen35Engine::dflash_finish(uint64_t * stats) {
    auto now = [] { return std::chrono::steady_clock::now(); };
    const bool prof = tqllm::debug_on("profile");
    const uint64_t steps = gen.steps;
    const uint32_t P = gen.P, count = gen.count;
    const uint64_t drafted = gen.drafted, accepted = gen.accepted;
    const double t_draft = gen.t_draft, t_verify = gen.t_verify, t_tail = gen.t_tail, prefill_ms = gen.prefill_ms;
    const auto decode_start = gen.decode_start;
    if (pc_on && pc_seq.size() == P && gen.pos >= P && gen.pos - P <= gen.out.size()) {  // fed: prompt + out[0, pos - P)
        pc_seq.insert(pc_seq.end(), gen.out.begin(), gen.out.begin() + (gen.pos - P));
        pc_live = true;
    }
    auto out = std::move(gen.out);
    gen = Gen{};
    if (tqllm::debug_on("steps"))
        std::fprintf(stderr, "finish prompt %u out %zu steps %llu drafted %llu accepted %llu\n", P, out.size(),
                     (unsigned long long) steps, (unsigned long long) drafted, (unsigned long long) accepted);
    if (prof && steps)
        std::fprintf(stderr, "prefill %u tokens %.1f ms (%.0f tok/s) | per step: draft %.2f ms, verify %.2f ms, "
                     "commit+inject %.2f ms (%llu steps)\n", P, last_prefill_ms, 1000.0 * P / last_prefill_ms,
                     t_draft / steps, t_verify / steps, t_tail / steps, (unsigned long long) steps);
    if (stage_prof && stage_acc.count("_steps")) {
        const double ns = double(stage_acc["_steps"].second);
        double tot = 0;
        for (const auto & [k, v] : stage_acc) tot += v.first;
        std::fprintf(stderr, "stage profile (synchronised, %.0f verify blocks, %.2f ms per block):\n", ns, tot / ns / 1000.0);
        for (const auto & [k, v] : stage_acc)
            if (k != "_steps")
                std::fprintf(stderr, "  %-30s %8.1f us/block  %5.1f%%  %6.1f calls/block  %7.2f us/call\n", k.c_str(),
                             v.first / ns, 100.0 * v.first / tot, double(v.second) / ns, v.first / double(v.second));
        stage_acc.clear();
    }
    if (stats) {
        stats[0] = steps;
        stats[1] = drafted;
        stats[2] = accepted;
        stats[3] = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(now() - decode_start).count());
        stats[4] = uint64_t(prefill_ms * 1000.0);
    }
    if (out.size() > count) out.resize(count);
    return out;
}

uint32_t Qwen35Engine::mtp_run(const uint32_t * tokens, const bf16 * hidden, int n, uint32_t pos, bool sample) {
    if (!mtp.present) throw std::runtime_error("model has no MTP head");
    struct RestoreBasis {
        Basis & current;
        Basis saved;
        ~RestoreBasis() { current = saved; }
    } restore{basis_, basis_};
    // The nextn block can retain plain Q6_K weights beside a rotated TQ4 target.
    basis_ = mtp.eh_proj.fmt ? mtp.eh_proj.fmt->basis() : Basis::Plain;
    const int H = int(cfg.hidden);
    auto *X = v_x.as<bf16>(), *R = v_r.as<bf16>(), *CAT = v_cat.as<bf16>(), *E = v_np.as<bf16>();
    auto * XH = v_xh.as<__half>();
    for (int i = 0; i < n; ++i) h_tok[i] = tokens[i];
    h_pos[0] = int(pos);
    h_pos[1] = int(pos) + 1;
    ck(cudaMemcpyAsync(v_tokens.p, h_tok, size_t(n) * 4, cudaMemcpyHostToDevice, stream));
    ck(cudaMemcpyAsync(v_pos.p, h_pos, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
    upload_rpos(1, [&](int) { return gen.rdelta; });
    uint32_t bucket = 256;
    while (bucket < pos + n) bucket *= 2;
    bucket = std::min(bucket, context_);
    vpar = chain2 && n <= NVER ? d_chain.as<int>() : nullptr;  // the MTP layer's attention: two-part chain like the verify
    vdep = vpar ? d_cdep.as<int>() : nullptr;                   // (> 8 queries need depths; tree mode crashed here)
    auto layer_kernels = [&] {
        embed_multi(v_tokens.as<uint32_t>(), n, E);
        tqllm_cuda::rms_f_multi(E, H, 0, mtp.enorm, CAT, 2 * H, 0, H, 1, n, cfg.eps, stream);
        tqllm_cuda::rms_f_multi(hidden ? hidden : mtp_h.as<bf16>(), H, 0, mtp.hnorm, CAT + H, 2 * H, 0, H, 1, n, cfg.eps,
                                stream);
        rot_h(CAT, XH, 2 * H, n);
        mm_h(mtp.eh_proj, X, n);
        tqllm_cuda::sumsq(X, v_px.as<float>(), n * H, stream);
        ck(cudaMemcpyAsync(R, X, size_t(n) * H * 2, cudaMemcpyDeviceToDevice, stream));
        nrot_h(X, mtp.layer.in_norm, v_px.as<float>(), XH, H, n);
        attn_mixer(mtp.layer, mtp_kc.as<bf16>(), mtp_vc.as<bf16>(), n, int(bucket));
        ffn_block(mtp.layer, n);
        ck(cudaMemcpyAsync(mtp_h.p, X + size_t(n - 1) * H, size_t(H) * 2, cudaMemcpyDeviceToDevice, stream));
    };
    auto subset_head_kernels = [&] {  // draft-vocabulary rows of the 4-bit head, then the int8 re-score of its top 16
        auto * NP = v_np.as<bf16>();
        tqllm_cuda::rms_f(X + size_t(n - 1) * H, H, mtp.shared_norm, NP, H, H, 1, cfg.eps, stream);
        tqllm_cuda::bf16_to_h(NP, v_xh.as<__half>(), H, stream);
        dsub_matvec(v_xh.as<__half>(), v_logits.as<float>(), 1);
        tqllm_cuda::topk16_big(v_logits.as<float>(), dsub_rows, 1, h2_vals.as<float>(), h2_idx.as<uint32_t>(), h2_scr.p,
                               stream);
        tqllm_cuda::map_ids(h2_idx.as<uint32_t>(), dsub_ids.as<uint32_t>(), 16, stream);
        tqllm_cuda::i8h_rescore_argmax(w8(), s8(), v_xh.as<__half>(), h2_idx.as<uint32_t>(), H, 1, v_tok.as<uint32_t>(),
                                       stream);
    };
    // one-token draft passes (the chain after the first draft) as one captured graph per KV bucket: the same kernels
    // in the same order, without ~25 launches and their host gaps per draft. TQLLM_DEBUG=mtp_eager runs them eagerly (A/B).
    static const bool draft_graph = !tqllm::debug_on("mtp_eager");
    if (draft_graph && sample && n == 1 && !hidden && mtp_head4_sub && has_head8() && segs.empty()) {
        const uint64_t key = (uint64_t(1) << 41) | uint64_t(bucket) | (uint64_t(active) << 32);
        ck(cudaGraphLaunch(cached_graph(kGraphDraft, key, [&] { layer_kernels(); subset_head_kernels(); }), stream));
        ck(cudaMemcpyAsync(h_tok, v_tok.p, 4, cudaMemcpyDeviceToHost, stream));
        sync_main();
        return h_tok[0];
    }
    layer_kernels();
    if (!sample) {
        sync_main();
        return 0;
    }
    if (mtp_head4_sub) {  // draft vocabulary rows of the 4-bit head (dsub_ids), FP16 tensor-core matvec
        auto * NP = v_np.as<bf16>();
        tqllm_cuda::rms_f(X + size_t(n - 1) * H, H, mtp.shared_norm, NP, H, H, 1, cfg.eps, stream);
        tqllm_cuda::bf16_to_h(NP, v_xh.as<__half>(), H, stream);
        dsub_matvec(v_xh.as<__half>(), v_logits.as<float>(), 1);
        if (has_head8()) {  // as the verify head: 4-bit top-16 of the subset, then the int8 head re-scores those rows
            tqllm_cuda::topk16_big(v_logits.as<float>(), dsub_rows, 1, h2_vals.as<float>(), h2_idx.as<uint32_t>(), h2_scr.p,
                                   stream);
            tqllm_cuda::map_ids(h2_idx.as<uint32_t>(), dsub_ids.as<uint32_t>(), 16, stream);
            tqllm_cuda::i8h_rescore_argmax(w8(), s8(), v_xh.as<__half>(), h2_idx.as<uint32_t>(), H, 1,
                                           v_tok.as<uint32_t>(), stream);
            ck(cudaMemcpyAsync(h_tok, v_tok.p, 4, cudaMemcpyDeviceToHost, stream));
            sync_main();
            return h_tok[0];  // a full-vocabulary id
        }
        tqllm_cuda::argmax_cols(v_logits.as<float>(), dsub_rows, 1, v_tok.as<uint32_t>(), v_amax.p, stream);
        ck(cudaMemcpyAsync(h_tok, v_tok.p, 4, cudaMemcpyDeviceToHost, stream));
        sync_main();
        return mtp.draft_ids[h_tok[0]];
    }
    if (mtp.draft_head.p) {  // argmax over the draft vocabulary only (the verifier still sees the full vocab)
        auto * NP = v_np.as<bf16>();
        tqllm_cuda::rms_f(X + size_t(n - 1) * H, H, mtp.shared_norm, NP, H, H, 1, cfg.eps, stream);
        tqllm_cuda::bf16_to_f32(NP, v_xf.as<float>(), H, stream);
        tqllm_cuda::q6k_matvec(mtp.draft_head.p, v_xf.as<float>(), v_logits.as<float>(), mtp.draft_head.rows, H, 1,
                               stream);
        tqllm_cuda::argmax_cols(v_logits.as<float>(), mtp.draft_head.rows, 1, v_tok.as<uint32_t>(), v_amax.p, stream);
        ck(cudaMemcpyAsync(h_tok, v_tok.p, 4, cudaMemcpyDeviceToHost, stream));
        sync_main();
        return mtp.draft_ids[h_tok[0]];
    }
    // Only the final repaired hidden state proposes the next draft token.
    head_argmax(mtp.shared_norm, 1, n - 1);
    ck(cudaMemcpyAsync(h_tok, v_tok.as<uint32_t>() + (n - 1), 4, cudaMemcpyDeviceToHost, stream));
    sync_main();
    return h_tok[0];
}

void Qwen35Engine::select_slot(int s) {
    if (s < 0 || s >= nslots) throw std::out_of_range("slot out of range");
    if (s == active) return;
    flush_pending();
    swap_seq(parked[size_t(active)]);
    swap_seq(parked[size_t(s)]);
    active = s;
}

bool Qwen35Engine::mtp_advance() {
    if (gen.done || gen.out.size() >= gen.count || gen.pending == eos_) {
        gen.done = true;
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto & draft = gen.mtp_draft;
    int D = gen.D;
    // --lookup L: a match of the last tokens earlier in prompt + output that runs longer than the MTP chain (or >= K
    // with --lookup-first K) is the draft instead; the MTP cache is rebuilt from the verified tokens either way
    if (opt.lookup > 0) {
        uint32_t lk[NVER];
        const int n = lookup(lk, std::min(opt.lookup, NVER - 1));
        if (n >= (opt.lookup_first > 0 ? opt.lookup_first : D + 1)) {
            draft.assign(lk, lk + n);
            D = n;
        }
    }
    while (int(draft.size()) < D)
        draft.push_back(mtp_run(&draft.back(), nullptr, 1, gen.pos + uint32_t(draft.size())));
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<uint32_t> in{gen.pending}, res(size_t(D) + 1);
    in.insert(in.end(), draft.begin(), draft.end());
    verify(in.data(), int(in.size()), gen.pos, res.data());
    const auto t2 = std::chrono::steady_clock::now();
    // greedy: longest prefix matching the target's argmax; sampling: speculative sampling with the deterministic MTP
    // draft (accept draft j with the target's probability of it, else resample without it), res[j] the next token
    const int j = accept(draft, res.data(), 0);
    ++gen.steps;
    gen.drafted += uint64_t(D);
    gen.accepted += uint64_t(j);
    for (int i = 0; i < j && gen.out.size() < gen.count; ++i) {
        gen.out.push_back(draft[size_t(i)]);
        if (draft[size_t(i)] == eos_) {
            gen.done = true;
            break;
        }
    }
    if (!gen.done && gen.out.size() < gen.count) gen.out.push_back(res[size_t(j)]);
    commit(j + 1);
    const uint32_t repair_pos = gen.pos + 1;
    gen.pos += uint32_t(j + 1);
    gen.pending = res[size_t(j)];
    gen.done |= gen.out.size() >= gen.count || gen.pending == eos_;
    if (!gen.done) {
        std::vector<uint32_t> repaired(draft.begin(), draft.begin() + j);
        repaired.push_back(gen.pending);
        // Pair each committed token with the target hidden of its preceding position.
        const uint32_t next = mtp_run(repaired.data(), verify_hidden(), int(repaired.size()), repair_pos);
        draft.assign(1, next);
    } else sync_ck();
    const auto t3 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    gen.t_draft += ms(t0, t1);
    gen.t_verify += ms(t1, t2);
    gen.t_tail += ms(t2, t3);
    dec_progress();
    return !gen.done;
}

bool Qwen35Engine::spec_advance() {
    if (gen.native_mtp) return mtp_advance();  // per request: an MTP request with DFlash2 loaded too
    if (dfl.present) return dflash_advance();
    bool done = false;
    const int s = active;
    dflash_step(&s, 1, &done);
    return !done;
}

int Qwen35Engine::drafter_features(const std::vector<uint32_t> & tokens, std::vector<uint16_t> & feats) {
    if (!dfl.present) return 0;
    const auto P = uint32_t(tokens.size());
    if (P == 0 || P + 1 > context_) throw std::runtime_error("drafter_features: sequence exceeds the context");
    const int H = int(cfg.hidden), nf = int(dfl.target_ids.size()), CH = NPF;
    const size_t chunks = (P + CH - 1) / CH, row = size_t(nf) * H;
    feats.assign(size_t(P) * row, 0);
    HostBuf staged_tokens(size_t(P) * sizeof(uint32_t)), staged_pos(chunks * 2 * sizeof(int)),
        staged_feat(size_t(CH) * row * 2);
    std::memcpy(staged_tokens.p, tokens.data(), size_t(P) * sizeof(uint32_t));
    auto * positions = static_cast<int *>(staged_pos.p);
    for (size_t i = 0; i < chunks; ++i) positions[2 * i] = int(i * CH), positions[2 * i + 1] = int(i * CH) + 1;
    for (uint32_t t = 0; t < P; t += uint32_t(CH)) {
        const int m = int(std::min<uint32_t>(uint32_t(CH), P - t));
        prefill_final = false;  // no LM head: only the captured features are needed
        verify_async(static_cast<const uint32_t *>(staged_tokens.p) + t, positions + 2 * (t / CH), m, t, nullptr, false);
        ck(cudaMemcpyAsync(staged_feat.p, v_feat.p, size_t(m) * row * 2, cudaMemcpyDeviceToHost, stream));
        commit(m);
        sync_ck();
        std::memcpy(feats.data() + size_t(t) * row, staged_feat.p, size_t(m) * row * 2);
    }
    return nf;
}

void Qwen35Engine::commit_body(int k) {  // replay the first k verified tokens into the next state slot
    {
        const int lkh = int(cfg.lin_kheads), lvh = int(cfg.lin_vheads), dk = int(cfg.lin_dk), dv = int(cfg.lin_dv);
        const int qkv_dim = 2 * lkh * dk + lvh * dv, zdim = lvh * dv;
        uint32_t lin_i = 0;
        for (uint32_t l = 0; l < cfg.layers; ++l) {
            if (cfg.full[l]) continue;
            const auto & L = layer[l];
            const int rows = mix_rows(L);
            const bf16 * T = v_stash2[sp][lin_i].as<bf16>() + size_t(stash_col0) * rows;
            const int * chain = tree_mode ? d_chain.as<int>() : nullptr;
            tqllm_cuda::conv_multi(T, rows, k, conv_slot(lin_i, cur), conv_slot(lin_i, cur + 1), L.conv_w,
                                   v_qkvc.as<bf16>(), qkv_dim, stream, chain);
            const bf16 * B0 = T + qkv_dim + zdim;
            tqllm_cuda::gdn_multi(v_qkvc.as<bf16>(), k, B0 + lvh, B0, rows, L.neg_a, L.dt_bias, rec_slot(lin_i, cur),
                                  rec_slot(lin_i, cur + 1), v_core.as<bf16>(), lkh, lvh, dk, dv, cfg.tiled, stream,
                                  chain);
            ++lin_i;
        }
    }
}

}  // namespace qwen35_detail
}  // namespace tqllm
