// qwen35 engine: loading and allocation. The class is declared in qwen35_engine.hpp.
#include "backend_caps.hpp"
#include "qwen35_engine.hpp"
#include <cstdio>
#include <cstring>

namespace tqllm {
namespace qwen35_detail {

void Qwen35Engine::load_safetensors(const std::string & dir) {
    const auto idx = safetensors_index(dir);
    const std::string P = idx.count("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.";
    const int H = int(cfg.hidden), F = int(cfg.ffn), hd = int(cfg.head_dim);
    const int lk = int(cfg.lin_kheads * cfg.lin_dk), lv = int(cfg.lin_vheads * cfg.lin_dv);
    const int vh = int(cfg.lin_vheads), qkvd = 2 * lk + lv;
    embed = st_mat(idx, P + "embed_tokens.weight", int(cfg.vocab), H);
    out_norm = put_f32(st_f32(idx, P + "norm.weight", H, 1.f));
    lm_head = cfg.tied ? embed : st_mat(idx, "lm_head.weight", int(cfg.vocab), H);
    layer.resize(cfg.layers);
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        const auto p = P + "layers." + std::to_string(l) + ".";
        auto & L = layer[l];
        L = {};
        L.in_norm = put_f32(st_f32(idx, p + "input_layernorm.weight", H, 1.f));
        L.post_norm = put_f32(st_f32(idx, p + "post_attention_layernorm.weight", H, 1.f));
        L.w_gate = st_mat(idx, p + "mlp.gate_proj.weight", F, H);
        L.w_up = st_mat(idx, p + "mlp.up_proj.weight", F, H);
        L.w_down = st_mat(idx, p + "mlp.down_proj.weight", H, F);
        if (cfg.full[l]) {
            L.wq = st_mat(idx, p + "self_attn.q_proj.weight", 2 * int(cfg.heads) * hd, H);
            L.wk = st_mat(idx, p + "self_attn.k_proj.weight", int(cfg.kvheads) * hd, H);
            L.wv = st_mat(idx, p + "self_attn.v_proj.weight", int(cfg.kvheads) * hd, H);
            L.wo = st_mat(idx, p + "self_attn.o_proj.weight", H, int(cfg.heads) * hd);
            L.q_norm = put_f32(st_f32(idx, p + "self_attn.q_norm.weight", hd, 1.f));
            L.k_norm = put_f32(st_f32(idx, p + "self_attn.k_norm.weight", hd, 1.f));
        } else {
            L.w_qkv = st_mat(idx, p + "linear_attn.in_proj_qkv.weight", qkvd, H);
            L.w_z = st_mat(idx, p + "linear_attn.in_proj_z.weight", lv, H);
            L.w_a = st_mat(idx, p + "linear_attn.in_proj_a.weight", vh, H);
            L.w_b = st_mat(idx, p + "linear_attn.in_proj_b.weight", vh, H);
            L.w_out = st_mat(idx, p + "linear_attn.out_proj.weight", H, lv);
            L.conv_w = put_f32(st_f32(idx, p + "linear_attn.conv1d.weight", uint64_t(qkvd) * cfg.conv_k));
            auto a = st_f32(idx, p + "linear_attn.A_log", vh);
            for (auto & v : a) v = -std::exp(v);
            L.neg_a = put_f32(a);
            L.dt_bias = put_f32(st_f32(idx, p + "linear_attn.dt_bias", vh));
            L.gnorm_w = put_f32(st_f32(idx, p + "linear_attn.norm.weight", cfg.lin_dv));
        }
    }
}

void Qwen35Engine::load_gguf(const GgufFile & f) {
    const int H = int(cfg.hidden), F = int(cfg.ffn), hd = int(cfg.head_dim);
    const int lk = int(cfg.lin_kheads * cfg.lin_dk), lv = int(cfg.lin_vheads * cfg.lin_dv);
    const int vh = int(cfg.lin_vheads), qkvd = 2 * lk + lv;
    {
        // Token-embedding lookups have kernels for TQ3_4S / Q6_K / Q8_0 / Q4_K / Q3_K / Q5_K / IQ3_S / IQ4_XS / BF16.
        // Any other GGUF type is dequantized on the host to BF16 once at load;
        // the BF16 table works on both the single-token (gather) and multi-token paths and can go to host memory.
        const auto & et = gt(f, "token_embd.weight");
        const auto ty = et.type;
        const bool lookup_ok = ty == GGML_TQ3_4S || ty == GGML_Q6_K || ty == GGML_Q8_0 || ty == GGML_Q4_K ||
                               ty == GGML_Q3_K || ty == GGML_Q5_K || ty == GGML_IQ3_S || ty == GGML_BF16 ||
                               ty == GGML_F32 || ty == GGML_F16;
        if (lookup_ok) embed = g_mat(f, "token_embd.weight", int(cfg.vocab), H, false);
        else if (ty == GGML_IQ4_XS && !cfg.tied) {
            // raw IQ4_XS rows, read by iq4xs_row / embed_multi_iq4xs (no conversion, same size as on disk)
            if (et.ne.size() != 2 || et.ne[0] != uint64_t(H) || et.ne[1] != uint64_t(cfg.vocab) || H % 256)
                throw std::runtime_error("unexpected shape for token_embd.weight");
            const auto raw = f.read(et);
            embed = {put(raw.data(), raw.size()), GGML_IQ4_XS, int(cfg.vocab), H};
        } else {
            if (et.ne.size() != 2 || et.ne[0] != uint64_t(H) || et.ne[1] != uint64_t(cfg.vocab))
                throw std::runtime_error("unexpected shape for token_embd.weight");
            const auto raw = f.read(et);
            const int64_t n = int64_t(cfg.vocab) * H;
            std::vector<float> emb_f32(static_cast<size_t>(n));
            dequantize_host(ty, raw.data(), emb_f32.data(), n);
            std::vector<char> b(size_t(n) * 2);
            for (int64_t i = 0; i < n; ++i) {
                const __nv_bfloat16 v = __float2bfloat16_rn(emb_f32[size_t(i)]);
                std::memcpy(b.data() + 2 * i, &v, 2);
            }
            std::fprintf(stderr, "tqserve: token_embd.weight type %u has no lookup kernel; dequantized to BF16 (%.2f GB)\n",
                         unsigned(ty), double(b.size()) / 1e9);
            embed = {put(b.data(), b.size()), GGML_BF16, int(cfg.vocab), H};
        }
    }
    out_norm = g_f32(f, "output_norm.weight", H);
    const auto ht = gt(f, "output.weight").type;  // Q6_K / Q5_K heads stay raw (read through the int8 copy)
    lm_head = cfg.tied ? embed : g_mat(f, "output.weight", int(cfg.vocab), H, ht != GGML_Q6_K && ht != GGML_Q5_K);
    if (cfg.tied && embed.type == GGML_TQ3_4S) throw std::runtime_error("tied TQ3_4S LM head not supported yet");
    // routed + shared experts of one MoE block (the trunk's layers and the MTP block): router, VRAM / host / split
    // expert matrices, the shared expert fused into the routed launches where its type and shape allow
    auto load_moe = [&](Layer & L, const std::string & p, const std::vector<char> & cold, bool force_host) {
        const int E = int(cfg.expert_count), FE = int(cfg.expert_ff), FS = int(cfg.shared_ff);
        L.router = g_f32(f, p + "ffn_gate_inp.weight", uint64_t(E) * H);
        const int nc = int(std::count(cold.begin(), cold.end(), char(1)));
        const bool host = nc == E || force_host;
        // the shared expert joins the VRAM experts for decode when its matrices have the experts' type and shape
        // (TQLLM_MOE_FUSE_SHARED=0: off); prefill keeps its dense path
        auto same = [&](const char * sh, const char * ex, int rows, int cols) {
            if (!f.tensors.count(p + sh)) return false;
            const auto & a = gt(f, p + sh);
            const auto & b = gt(f, p + ex);
            return a.type == b.type && a.ne.size() == 2 && a.ne[0] == uint64_t(cols) && a.ne[1] == uint64_t(rows) && FS == FE;
        };
        const bool fuse = !host && same("ffn_gate_shexp.weight", "ffn_gate_exps.weight", FE, H) &&
                          same("ffn_up_shexp.weight", "ffn_up_exps.weight", FE, H) &&
                          same("ffn_down_shexp.weight", "ffn_down_exps.weight", H, FE);
        auto x = [&](const char * sh) { return fuse ? p + sh : std::string(); };
        if (nc == 0 || host) {
            L.ex_gate = g_exps(f, p + "ffn_gate_exps.weight", E, FE, H, host, x("ffn_gate_shexp.weight"));
            L.ex_up = g_exps(f, p + "ffn_up_exps.weight", E, FE, H, host, x("ffn_up_shexp.weight"));
            L.ex_down = g_exps(f, p + "ffn_down_exps.weight", E, H, FE, host, x("ffn_down_shexp.weight"));
            if (fuse) L.sh_slot = E;
        } else {
            std::tie(L.ex_gate, L.cx_gate) = g_exps_split(f, p + "ffn_gate_exps.weight", E, FE, H, cold, x("ffn_gate_shexp.weight"));
            std::tie(L.ex_up, L.cx_up) = g_exps_split(f, p + "ffn_up_exps.weight", E, FE, H, cold, x("ffn_up_shexp.weight"));
            std::tie(L.ex_down, L.cx_down) = g_exps_split(f, p + "ffn_down_exps.weight", E, H, FE, cold, x("ffn_down_shexp.weight"));
            std::vector<int> hs(size_t(E), -1), cs(size_t(E), -1);
            for (int e = 0; e < E; ++e) {
                if (cold[size_t(e)]) cs[size_t(e)] = L.n_cold++;
                else hs[size_t(e)] = L.n_hot++;
            }
            if (fuse) L.sh_slot = L.n_hot;  // after the hot routed experts
            L.hot_slot = static_cast<const int *>(put(hs.data(), hs.size() * sizeof(int)));
            L.cold_slot = static_cast<const int *>(put(cs.data(), cs.size() * sizeof(int)));
        }
        L.sh_gu = g_cat(f, {{p + "ffn_gate_shexp.weight", FS}, {p + "ffn_up_shexp.weight", FS}}, H);
        if (!L.sh_gu.p) throw std::runtime_error(p + "ffn_{gate,up}_shexp: types cannot be fused");
        L.sh_down = g_mat(f, p + "ffn_down_shexp.weight", H, FS);
        L.sh_gate = g_f32(f, p + "ffn_gate_inp_shexp.weight", H);
        for (const Mat * m : {&L.ex_gate, &L.ex_up, &L.ex_down, &L.cx_gate, &L.cx_up, &L.cx_down})
            check_grouped_expert_format(*m, p);
    };
    const auto mp = "blk." + std::to_string(cfg.layers) + ".";
    // the MTP head loads only for --mtp (tqllm::drafters: one drafter per run, never beside DFlash2)
    if (f.tensors.count(mp + "nextn.eh_proj.weight") && tqllm::drafters(opt).mtp) {
        mtp.present = true;
        mtp.eh_proj = g_mat(f, mp + "nextn.eh_proj.weight", H, 2 * H);
        mtp.enorm = g_f32(f, mp + "nextn.enorm.weight", H);
        mtp.hnorm = g_f32(f, mp + "nextn.hnorm.weight", H);
        mtp.shared_norm = g_f32(f, mp + "nextn.shared_head_norm.weight", H);
        auto & L = mtp.layer;
        L.in_norm = g_f32(f, mp + "attn_norm.weight", H);
        L.post_norm = g_f32(f, mp + "post_attention_norm.weight", H);
        L.w_mix = g_cat(f, {{mp + "attn_q.weight", 2 * int(cfg.heads) * hd}, {mp + "attn_k.weight", int(cfg.kvheads) * hd},
                            {mp + "attn_v.weight", int(cfg.kvheads) * hd}}, H);
        L.wo = g_mat(f, mp + "attn_output.weight", H, int(cfg.heads) * hd);
        L.q_norm = g_f32(f, mp + "attn_q_norm.weight", hd);
        L.k_norm = g_f32(f, mp + "attn_k_norm.weight", hd);
        if (cfg.moe) load_moe(L, mp, std::vector<char>(size_t(cfg.expert_count), 0), false);  // the MTP block's experts stay in VRAM
        else {
            L.w_gu = g_cat(f, {{mp + "ffn_gate.weight", F}, {mp + "ffn_up.weight", F}}, H);
            L.w_down = g_mat(f, mp + "ffn_down.weight", H, F);
        }
        // Draft vocabulary (llama.cpp map format: header "llama-mtp-vocab-v1 <vocab> <n>", then one id per line).
        if (const char * dv = opt.draft_vocab.c_str(); *dv && opt.draft_model.empty()) {  // DFlash builds its own subset
            std::ifstream in(dv);
            std::string magic;
            uint64_t nv = 0, nd = 0;
            in >> magic >> nv >> nd;
            if (!in || nv != cfg.vocab) throw std::runtime_error(std::string("bad draft vocab file ") + dv);
            const auto & ot = gt(f, "output.weight");
            if (ot.type != GGML_Q6_K) throw std::runtime_error("draft vocab needs a Q6_K LM head");
            const size_t row_bytes = size_t(H) / 256 * 210;
            const auto all = f.read(ot);
            std::vector<char> sub(nd * row_bytes);
            for (uint64_t i = 0; i < nd; ++i) {
                uint32_t id = 0;
                in >> id;
                if (!in || id >= cfg.vocab) throw std::runtime_error("bad id in draft vocab file");
                mtp.draft_ids.push_back(id);
                std::copy(all.begin() + id * row_bytes, all.begin() + (id + 1) * row_bytes, sub.begin() + i * row_bytes);
            }
            mtp.draft_head = {put(sub.data(), sub.size()), GGML_Q6_K, int(nd), H};
        }
    }
    layer.resize(cfg.layers);
    const auto cold_of = cfg.moe ? moe_placement(f) : std::vector<std::vector<char>>();
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        const auto p = "blk." + std::to_string(l) + ".";
        auto & L = layer[l];
        L = {};
        L.in_norm = g_f32(f, p + "attn_norm.weight", H);
        L.post_norm = g_f32(f, p + "post_attention_norm.weight", H);
        if (cfg.moe) {
            load_moe(L, p, cold_of[l], int(l) < opt.n_cpu_moe);
        } else {
            L.w_gu = g_cat(f, {{p + "ffn_gate.weight", F}, {p + "ffn_up.weight", F}}, H);
            if (!L.w_gu.p) {
                L.w_gate = g_mat(f, p + "ffn_gate.weight", F, H);
                L.w_up = g_mat(f, p + "ffn_up.weight", F, H);
            }
            L.w_down = g_mat(f, p + "ffn_down.weight", H, F);
        }
        if (cfg.full[l]) {
            L.w_mix = g_cat(f, {{p + "attn_q.weight", 2 * int(cfg.heads) * hd}, {p + "attn_k.weight", int(cfg.kvheads) * hd},
                                {p + "attn_v.weight", int(cfg.kvheads) * hd}}, H);
        }
        if (cfg.full[l] && !L.w_mix.p) {
            L.wq = g_mat(f, p + "attn_q.weight", 2 * int(cfg.heads) * hd, H);
            L.wk = g_mat(f, p + "attn_k.weight", int(cfg.kvheads) * hd, H);
            L.wv = g_mat(f, p + "attn_v.weight", int(cfg.kvheads) * hd, H);
        }
        if (cfg.full[l]) {
            L.wo = g_mat(f, p + "attn_output.weight", H, int(cfg.heads) * hd);
            L.q_norm = g_f32(f, p + "attn_q_norm.weight", hd);
            L.k_norm = g_f32(f, p + "attn_k_norm.weight", hd);
        } else {
            L.w_mix = g_cat(f, {{p + "attn_qkv.weight", qkvd}, {p + "attn_gate.weight", lv},
                                {p + "ssm_beta.weight", vh}, {p + "ssm_alpha.weight", vh}}, H);
            const auto bt = gt(f, p + "ssm_beta.weight").type;
            if (!L.w_mix.p && (bt == GGML_BF16 || bt == GGML_F32 || bt == GGML_F16) && gt(f, p + "attn_qkv.weight").type != GGML_BF16 &&
                weight_format(gt(f, p + "attn_qkv.weight").type)) {
                L.w_mix = g_cat(f, {{p + "attn_qkv.weight", qkvd}, {p + "attn_gate.weight", lv}}, H);
                L.w_ba = g_cat(f, {{p + "ssm_beta.weight", vh}, {p + "ssm_alpha.weight", vh}}, H);
                if (!L.w_mix.p || L.w_ba.type != GGML_BF16) throw std::runtime_error("W4A16: unexpected Gated DeltaNet types");
            }
            if (!L.w_mix.p) {
                L.w_qkv = g_mat(f, p + "attn_qkv.weight", qkvd, H);
                L.w_z = g_mat(f, p + "attn_gate.weight", lv, H);
                L.w_a = g_mat(f, p + "ssm_alpha.weight", vh, H);
                L.w_b = g_mat(f, p + "ssm_beta.weight", vh, H);
            }
            L.w_out = g_mat(f, p + "ssm_out.weight", H, lv);
            L.conv_w = g_f32(f, p + "ssm_conv1d.weight", uint64_t(qkvd) * cfg.conv_k);
            L.neg_a = g_f32(f, p + "ssm_a", vh);
            L.dt_bias = g_f32(f, p + "ssm_dt.bias", vh);
            L.gnorm_w = g_f32(f, p + "ssm_norm.weight", cfg.lin_dv);
        }
    }
}

void Qwen35Engine::alloc_seq_state() {
    const size_t lk = size_t(cfg.lin_kheads) * cfg.lin_dk, lv = size_t(cfg.lin_vheads) * cfg.lin_dv;
    const KvStorePolicy * kvs = kv_store_policy(opt.kv_store);
    if (!kv_pool) {  // first slot: one pool for every slot
        if (std::string(kvs->name()) != "flat")
            throw std::runtime_error(std::string("--kv-store ") + kvs->name() + ": the attention kernels read flat KV only so far");
        const int attn_layers = int(std::count(cfg.full.begin(), cfg.full.end(), true));
        kv_mem = make_cuda_kv_memory(stream);
        const int64_t host = std::min<int64_t>(std::max(0L, opt.kv_host), context_);
        // staged: the newest positions stay in a device window (>= 2048: a prefill chunk plus a verify block always
        // fit), old ones move to host and are copied back per layer; the single-token step path reads uncompressed
        // KV without the copies, so staging needs a compressed format
        const bool staged = host && opt.kv_stage && nslots == 1 && kvf->compressed();
        KvGeometry g{attn_layers, size_t(kv_row_bytes(true)), size_t(kv_row_bytes(false)), kvs->page(context_),
                     host ? std::max<int64_t>(int64_t(context_) - host, staged ? 2048 : 0) : -1, staged};
        kv_pool = std::make_unique<KvPool>(*kv_mem, g, nslots, 0);
        kv_staged = staged && kv_pool->bytes(KvTier::host) > 0;
        if (kv_staged) {
            d_stage_bytes = Buf(2 * sizeof(int64_t));
            ck(cudaMemsetAsync(d_stage_bytes.p, 0, 2 * sizeof(int64_t), stream));
            ck(cudaStreamCreateWithFlags(&kv_side, cudaStreamNonBlocking));
            ck(cudaEventCreateWithFlags(&kv_fork, cudaEventDisableTiming));
            for (auto * e : {&kv_ready[0], &kv_ready[1], &kv_done[0], &kv_done[1]})
                ck(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
            int sms = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
            kv_blocks = 4 * sms;
        }
        // staged (old KV in host memory) or device-resident (all KV on the GPU: old blocks compacted in place,
        // attention reads stand-ins + the exact positions from cmp_pos)
        const bool cmp_dev = !host && nslots == 1;
        kv_cmp = (kv_staged || cmp_dev) && opt.kv_compact > 0 && kvf->compact_ok();
        if (opt.kv_compact > 0 && !kv_cmp)
            std::fprintf(stderr, "tqllm: --kv-compact needs one slot (all KV on the GPU or the staged host tail) and tq3 or tcq3/tcq2 KV; off\n");
        if (!tqllm::caps::kDenseSolver && kv_cmp) {  // the stand-in solve is cuSOLVER Spotrf/Spotrs; hipsolver is not linked in this PR
            std::fprintf(stderr, "tqllm: --kv-compact (TqCache) is not supported on the AMD backend yet (no cuSOLVER); off\n");
            kv_cmp = false;
        }
        const int64_t cmp_span = kv_staged ? host : std::max<int64_t>(0, int64_t(context_) - kCmpWindow);
        if (kv_cmp) {
            // middle layers [A/4, 5A/8) at half the ratio: attention-output error at 8x on Qwen3.5-9B was +29..49 %
            // there against +3..16 % at either end
            for (int a = 0; a < attn_layers; ++a)
                cmp_r.push_back(4 * a >= attn_layers && 8 * a < 5 * attn_layers ? std::max(2, opt.kv_compact / 2) : opt.kv_compact);
            const int r_min = *std::min_element(cmp_r.begin(), cmp_r.end());
            cmp_max = cmp_span;
            const size_t kvh = cfg.kvheads, hdz = cfg.head_dim;
            size_t sb = 0;
            for (int a = 0; a < attn_layers; ++a) {
                const size_t rows = size_t(cmp_span / cmp_r[size_t(a)] + 1);
                q_ring.emplace_back(size_t(kRing) * cfg.heads * hdz * 2);
                cmp_k.emplace_back(rows * kvh * sizeof(tqllm_cuda::KvStandIn256));
                cmp_v.emplace_back(rows * kvh * sizeof(tqllm_cuda::KvStandIn256));
                cmp_b.emplace_back(rows * kvh * 4);
                sb += rows * kvh * (2 * sizeof(tqllm_cuda::KvStandIn256) + 4);
            }
            d_cmp_i = Buf((1 + size_t(attn_layers)) * sizeof(int));
            d_stage_off = Buf(2 * sizeof(int64_t));
            ck(cudaMemsetAsync(d_cmp_i.p, 0, (1 + size_t(attn_layers)) * sizeof(int), stream));
            ck(cudaMemsetAsync(d_stage_off.p, 0, 2 * sizeof(int64_t), stream));
            for (int a = 0; a < attn_layers; ++a)
                cmp_seg.push_back({d_cmp_i.as<int>(), cmp_k[size_t(a)].as<tqllm_cuda::KvStandIn256>(),
                                   cmp_v[size_t(a)].as<tqllm_cuda::KvStandIn256>(), cmp_b[size_t(a)].as<float>(),
                                   d_cmp_i.as<int>() + 1 + a});
            // row-split partials of the stand-in kernel (verify blocks; prefill chunks use one split)
            cmp_split = Buf(tqllm_cuda::kv_compact_split_floats(NVER, int(cfg.heads)) * sizeof(float));
            for (auto & sg : cmp_seg) sg.split_buf = cmp_split.as<float>(), sg.split_cap = NVER * int(cfg.heads) * 32;
            cmp_scratch = Buf(tqllm_cuda::kv_compact_scratch_bytes(kCmpBlock, kRing * int(cfg.heads / cfg.kvheads),
                                                                   kCmpBlock / r_min, int(kvh)));
            if (cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER init failed");
            std::fprintf(stderr, "tqllm: TqCache compaction %dx (%dx in the middle layers): %.2f GB of stand-ins\n",
                         opt.kv_compact, r_min, sb / 1e9);
        }
        if (host)
            std::fprintf(stderr, "tqllm: KV: %s %lld of %u positions per sequence in host memory (%.2f GB host, %.2f GB device)\n",
                         kv_staged ? "oldest" : "last", (long long)host, context_,
                         double(kv_pool->bytes(KvTier::host)) / 1e9, double(kv_pool->bytes(KvTier::device)) / 1e9);
    }
    kv_seq = kvs->make(*kv_pool, context_);
    if (!kv_seq->reserve(context_)) throw std::runtime_error("out of memory");
    for (uint32_t l = 0, a = 0; l < cfg.layers; ++l) {
        if (cfg.full[l]) {
            key_cache.push_back({kv_seq->base(int(a), true)});
            value_cache.push_back({kv_seq->base(int(a), false)});
            ++a;
        } else {
            for (int sl = 0; sl < NSLOT; ++sl) {
                conv_state.emplace_back(size_t(cfg.conv_k - 1) * (2 * lk + lv) * 2);
                rec_state.emplace_back(size_t(cfg.lin_vheads) * cfg.lin_dk * cfg.lin_dv * 4);
            }
        }
    }
}

void Qwen35Engine::allocate() {
    const size_t H = cfg.hidden, F = cfg.ffn, hd = cfg.head_dim;
    const size_t lk = size_t(cfg.lin_kheads) * cfg.lin_dk, lv = size_t(cfg.lin_vheads) * cfg.lin_dv;
    const size_t widest = std::max({H, F, lv, size_t(cfg.heads) * hd, 2 * H, 8 * H});  // 8H: DFlash fc input
    {
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) fr = tot = 0;
        int dev = 0, integrated = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev);
        if (integrated) {
            std::ifstream mi("/proc/meminfo");
            const std::string text((std::istreambuf_iterator<char>(mi)), std::istreambuf_iterator<char>());
            fr = std::min(tot, std::max(fr, meminfo_available(text)));
        }
        const int attn_layers = int(std::count(cfg.full.begin(), cfg.full.end(), true));
        const int64_t dev_pos = int64_t(context_) - std::min<int64_t>(std::max(0L, opt.kv_host), context_);
        const size_t kv_dev = size_t(attn_layers) * size_t(dev_pos) * size_t(kv_row_bytes(true) + kv_row_bytes(false));
        NPF = NBUF = prefill_chunk(int64_t(context_), tot, fr, kv_dev);  // GB10 prefill 8K: 650 -> 1135 tok/s
        prefill_room = fr > kv_dev ? fr - kv_dev : 0;
    }
    if (opt.ubatch) NPF = NBUF = std::max(NVER, opt.ubatch);
    x = Buf(H * 2);
    norm = Buf(H * 2);
    resid = Buf(H * 2);
    mix = Buf(H * 2);
    gate = Buf(F * 2);
    up = Buf(F * 2);
    act = Buf(F * 2);
    xq = Buf(widest);
    xs = Buf(widest / 32 * 4);
    xf = Buf(H * 4);
    part_x = Buf((H + 255) / 256 * 4);
    part_m = Buf((H + 255) / 256 * 4);
    amax = Buf(2048 * 4);
    logits_buf = Buf(size_t(cfg.vocab) * 4);
    token = Buf(sizeof(uint32_t));
    part = Buf(size_t(tqllm_cuda::attention_partial_floats(int(cfg.heads), int(context_), int(hd))) * 4);
    t_in = Buf(std::max({2 * lk + 2 * lv + 2 * size_t(cfg.lin_vheads), 2 * size_t(cfg.heads) * hd + 2 * size_t(cfg.kvheads) * hd, 2 * F}) * 2);
    t_qg = Buf(2 * cfg.heads * hd * 2);
    t_q = Buf(cfg.heads * hd * 2);
    t_k = Buf(cfg.kvheads * hd * 2);
    t_v = Buf(cfg.kvheads * hd * 2);
    t_o = Buf(cfg.heads * hd * 2);
    t_qkv = Buf((2 * lk + lv) * 2);
    t_qkvc = Buf((2 * lk + lv) * 2);
    t_z = Buf(lv * 2);
    t_a = Buf(cfg.lin_vheads * 2);
    t_b = Buf(cfg.lin_vheads * 2);
    t_core = Buf(lv * 2);
    t_gn = Buf(lv * 2);
    alloc_seq_state();
    const size_t mix_rows = std::max(2 * lk + 2 * lv + 2 * size_t(cfg.lin_vheads),
                                     2 * size_t(cfg.heads) * hd + 2 * size_t(cfg.kvheads) * hd);
    v_x = Buf(NBUF * H * 2);
    v_m = Buf(NBUF * H * 2);
    v_r = Buf(NBUF * H * 2);
    v_np = Buf(NBUF * H * 2);
    v_in = Buf(NBUF * std::max(mix_rows, 2 * F) * 2);
    v_o = Buf(NBUF * std::max(size_t(cfg.heads) * hd, lv) * 2);
    v_core = Buf(NBUF * lv * 2);
    v_act = Buf(NBUF * F * 2);
    v_xh = Buf(NBUF * widest * 2);
    if (cfg.moe) {
        const size_t K = cfg.expert_used, FE = cfg.expert_ff;
        moe_logits = Buf(NBUF * (cfg.expert_count + 1) * 4);
        moe_ids = Buf(NBUF * K * sizeof(int));
        moe_counts = Buf(cfg.expert_count * sizeof(int));
        moe_groups = Buf(NBUF * cfg.expert_count * sizeof(int));
        moe_xids = Buf(NBUF * (K + 1) * sizeof(int));
        if (opt.moe_host_mb > 0) {
            moe_hot_ids = Buf(NBUF * (K + 1) * sizeof(int)); moe_cold_ids = Buf(NBUF * (K + 1) * sizeof(int));
            moe_counts2 = Buf(cfg.expert_count * sizeof(int)); moe_groups2 = Buf(NBUF * cfg.expert_count * sizeof(int));
            alloc_moe_stage();
            alloc_moe_cache();
        }
        if (const char * pf = tqllm::debug_cstr("moe_profile")) {
            moe_prof_path = pf; moe_prof = Buf(size_t(cfg.layers) * cfg.expert_count * 8);
            ck(cudaMemset(moe_prof.p, 0, size_t(cfg.layers) * cfg.expert_count * 8));
        }
        moe_weights = Buf(NBUF * K * 4);
        moe_shared_weights = Buf(NBUF * 4);
        moe_gate = Buf(NBUF * (K + 1) * FE * 2);
        moe_up = Buf(NBUF * (K + 1) * FE * 2);
        moe_act = Buf(NBUF * (K + 1) * FE * 2);
        moe_out = Buf(NBUF * (K + 1) * H * 2);
        moe_shared = Buf(NBUF * H * 2);
    }
    v_xf = Buf(NBUF * H * 4);
    v_px = Buf(NBUF * ((H + 255) / 256) * 4);
    v_pm = Buf(NBUF * ((H + 255) / 256) * 4);
    v_logits = Buf(size_t(std::max(NVER, NVER * nslots)) * cfg.vocab * 4);  // batched verify: head over every slot
    NB = NVER * nslots;  // columns of one batched verify (every slot drafting NVER - 1)
    if (NB > NBUF) throw std::runtime_error("too many slots for the column buffers (TQLLM_PREFILL_BUF)");
    v_tok = Buf(size_t(NB) * 4);
    s_id = Buf(size_t(NB) * 16 * 4);
    s_lg = Buf(size_t(NB) * 16 * 4);
    s_scr = Buf(size_t(NB) * 64 * 16 * 8);
    h_sid.resize(size_t(NB) * 16);
    h_slg.resize(size_t(NB) * 16);
    v_amax = Buf(size_t(NB) * 128 * 8);
    v_mix_rows = mix_rows;
    const bool bf16_ba = std::any_of(layer.begin(), layer.end(), [](const Layer & L) { return L.w_ba.p; });
    const MixStaging stage = mix_staging(basis_, bf16_ba, fmt_target);
    if (stage.quantized) {  // BF16 Gated DeltaNet b/a next to 4-bit matrices
        v_xb = Buf(NBUF * widest * 2);
        v_mixa = Buf(NBUF * (2 * lk + 2 * lv) * 2);
        v_mixb = Buf(NBUF * 2 * size_t(cfg.lin_vheads) * 2);
    }
    if (stage.unrotated) v_xu = Buf(NBUF * H * 2);  // the BF16 b/a rows of a rotated-basis model
    v_q = Buf(NBUF * size_t(cfg.heads) * hd * 2);
    v_qkvc = Buf(NBUF * (2 * lk + lv) * 2);
    part_floats = std::max({!kvf->needs_context_scratch() ? size_t(0) : size_t(NVER) * tqllm_cuda::attention_partial_floats(int(cfg.heads), int(context_), int(hd)),
                            size_t(tqllm_cuda::attention_mq_partial_floats(int(cfg.heads), NVER, kv_cmp ? 257 : 256)),
                            size_t(tqllm_cuda::attention_mq_partial_floats(int(cfg.heads), NBUF, kv_cmp ? 2 : 1))});  // + stand-in slot
    v_part = Buf(part_floats * 4);
    for (uint32_t l = 0; l < cfg.layers; ++l)
        if (!cfg.full[l])
            for (auto & st : v_stash2) st.emplace_back(size_t(NB) * mix_rows * 2);
    v_hid = Buf(NBUF * H * 2);
    v_cat = Buf(NVER * 2 * H * 2);
    mtp_h = Buf(H * 2);
    if (mtp.present) {
        mtp_kc = Buf(kvf->key_bytes(context_, int(cfg.kvheads), int(hd)));
        mtp_vc = Buf(kvf->value_bytes(context_, int(cfg.kvheads), int(hd)));
    }
    v_tokens = Buf(NBUF * 4);
    d_tpar = Buf(size_t(NB) * 4);
    d_tdep = Buf(size_t(NB) * 4);
    d_tcols = Buf(size_t(NB) * 4);
    d_chain = Buf(size_t(NB) * 4);
    d_cdep = Buf(size_t(NB) * 4);  // chain depths 0, 1, ...: two-part attention of chains longer than 8 queries
    d_gdesc = Buf(256 * sizeof(tqllm_cuda::PathGather));
    d_pend = Buf(sizeof(int));
    ck(cudaMemset(d_pend.p, 0, sizeof(int)));
    d_rep_qkvc = Buf(size_t(NVER) * (2 * cfg.lin_kheads * cfg.lin_dk + cfg.lin_vheads * cfg.lin_dv) * 2);
    {
        std::vector<int> ch(static_cast<size_t>(NB) + 0);
        for (int i = 0; i < NB; ++i) ch[size_t(i)] = i - 1;
        ck(cudaMemcpy(d_chain.p, ch.data(), ch.size() * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(d_tpar.p, ch.data(), ch.size() * 4, cudaMemcpyHostToDevice));
        for (int i = 0; i < NB; ++i) ch[size_t(i)] = i;
        ck(cudaMemcpy(d_tdep.p, ch.data(), ch.size() * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(d_cdep.p, ch.data(), ch.size() * 4, cudaMemcpyHostToDevice));
    }
    if (opt.native_mtp && nslots != 1)  // refused at start-up, not per request
        throw std::runtime_error("--mtp drafts for one slot only for now: use --np 1 (or -md for --np > 1)");
    chain2 = kvf->tree_ok() && kv_row_bytes(true) % 16 == 0 && kv_row_bytes(false) % 16 == 0 && !tqllm::debug_on("attn_one_part");
    // tree verify: with a DFlash drafter when the KV format has tree attention (--no-tree: chain)
    if (opt.tree && !opt.draft_model.empty() && kvf->tree_ok()) {
        if (kv_row_bytes(true) % 16 || kv_row_bytes(false) % 16)
            throw std::runtime_error("tree verify: KV rows must be a multiple of 16 bytes");
        tree_mode = true;
        {  // deferred commit: GB10 (memory-bound) 73.6 -> 72.8 ms per round, no change on the 3090
            int dev = 0, major = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            defer = major >= 10;
        }
    }
    v_pos = Buf(2 * size_t(nslots) * sizeof(int));
    v_rpos = Buf(2 * size_t(nslots) * sizeof(int));
    ck(cudaHostAlloc(reinterpret_cast<void **>(&h_tok), NBUF * 4, cudaHostAllocDefault));
    ck(cudaHostAlloc(reinterpret_cast<void **>(&h_pos), 2 * size_t(nslots) * sizeof(int), cudaHostAllocDefault));
    ck(cudaHostAlloc(reinterpret_cast<void **>(&h_rpos), 2 * size_t(nslots) * sizeof(int), cudaHostAllocDefault));
    // any quantized target (TQ3 or a registered format such as TQ4) with a Q6_K head: the drafter and verify heads read head8
    const bool head_host = fmt_target && opt.head_host;
    if (k_head() && (quantized || fmt_target)) {
        // head in host memory: the int8 copy is written straight into mapped host memory, so a load never holds it on
        // the device beside the Q6_K original (1.4 GB more at the load's peak than after it; the memory fit plans the end state)
        if (head_host) head8_host_alloc();
        else {
            head8 = Buf(size_t(lm_head.rows) * H);
            head8s = Buf(size_t(lm_head.rows) * H / 16 * 2);
        }
        if (lm_head.type == GGML_Q5_K)
            tqllm_cuda::q5k_to_i8(lm_head.p, const_cast<int8_t *>(w8()), const_cast<__half *>(s8()), lm_head.rows, int(H), stream);
        else tqllm_cuda::q6k_to_i8(lm_head.p, const_cast<int8_t *>(w8()), const_cast<__half *>(s8()), lm_head.rows, int(H), stream);
        sync_ck();
        // every head read now goes through head8 (draft-vocab subsets are built from head8 rows or read from the
        // file): free the 1 GB Q6_K original unless tied to the embedding
        if (!cfg.tied)
            for (auto & b : tensors)
                if (b.p == lm_head.p) {
                    b = Buf();
                    lm_head.p = nullptr;
                    break;
                }
        // two-stage head (4-bit first pass, exact int8 re-score of the top 16) for every registered-format target:
        // single-token decode then reads 0.64 GB instead of the 1.27 GB int8 head (Qwen3.8), with or without a drafter
        if (fmt_target) build_head4();
    }
    if (fmt_target && !cfg.tied && opt.embed_host) {
        size_t bytes = 0;  // raw GGUF rows: one token's row is read per lookup
        const size_t n = size_t(embed.rows) * embed.cols;
        if (embed.type == GGML_Q4_K) bytes = n / 256 * 144;
        else if (embed.type == GGML_Q6_K) bytes = n / 256 * 210;
        else if (embed.type == GGML_Q8_0) bytes = n / 32 * 34;
        else if (embed.type == GGML_Q3_K) bytes = n / 256 * 110;
        else if (embed.type == GGML_Q5_K) bytes = n / 256 * 176;
        else if (embed.type == GGML_IQ3_S) bytes = n / 256 * 110;
        else if (embed.type == GGML_BF16) bytes = n * 2;
        else if (embed.type == GGML_IQ4_XS) bytes = n / 256 * 136;
        if (bytes) {
            const void * old = embed.p;
            embed.p = const_cast<void *>(to_mapped_host(old, bytes));
            for (auto & b : tensors)
                if (b.p == old) { b = Buf(); break; }
            placed_embed_b = bytes;
            std::fprintf(stderr, "tqserve: token embeddings in host memory (%.2f GB of VRAM freed)\n", double(bytes) / 1e9);
        }
    }
    if (fmt_target && NPF >= NGEMM_BLAS) {
        size_t mx = 0, mr = 0;
        for (const auto & L : layer)
            for (const Mat * m : {&L.w_gate, &L.w_up, &L.w_down, &L.wq, &L.wk, &L.wv, &L.wo, &L.w_qkv, &L.w_z,
                                  &L.w_a, &L.w_b, &L.w_out, &L.w_mix, &L.w_gu, &L.sh_gu, &L.sh_down})
                if (m->p && m->fmt) {
                    mx = std::max(mx, size_t(m->rows) * m->cols);
                    mr = std::max(mr, size_t(m->rows));
                }
        if (mx) {
            // gemm_rows caps the FP16 weight buffer (row blocks of row-sliceable formats; tqserve's fit sets it)
            if (const int gemm_rows_req = opt.gemm_rows) {
                size_t mc = 0;
                for (const auto & L : layer)
                    for (const Mat * m : {&L.w_down, &L.w_gu, &L.w_mix, &L.wo, &L.w_out, &L.sh_gu, &L.sh_down})
                        if (m->p && m->fmt) mc = std::max(mc, size_t(m->cols));
                std::vector<GemmMat> mats;
                for (const auto & L : layer)
                    for (const Mat * m : {&L.w_gate, &L.w_up, &L.w_down, &L.wq, &L.wk, &L.wv, &L.wo, &L.w_qkv, &L.w_z,
                                          &L.w_a, &L.w_b, &L.w_out, &L.w_mix, &L.w_gu})
                        if (m->p && m->fmt)
                            mats.push_back({size_t(m->rows), size_t(m->cols), m->fmt->row_bytes(m->cols) != 0});
                const auto plan = gemm_buffer_plan(gemm_rows_req, mc, mats);
                gemm_w_rows = plan.block_rows;
                gemm_w_elems = plan.elems;
                mx = gemm_w_elems;
                if (!gemm_w_rows)
                    std::fprintf(stderr, "tqllm: prefill weight blocks skipped (formats cannot be split by rows)\n");
            }
            gemm_w = Buf(mx * 2);
            // opt-in (--prefill-fp8 on, sm_89+): faster prefill but KLD ~0.01 vs FP16 prefill, so off by default
            if ((opt.prefill_fp8 == 1 || (opt.prefill_fp8 < 0 && !opt.exact && tqllm_cuda::Fp8Gemm::default_on())) && !gemm_w_rows) {
                if (tqllm_cuda::Fp8Gemm::supported()) {
                    size_t mk = 0;
                    for (const auto & L : layer)
                        for (const Mat * m : {&L.w_gate, &L.w_up, &L.w_down, &L.wq, &L.wk, &L.wv, &L.wo, &L.w_qkv, &L.w_z,
                                              &L.w_a, &L.w_b, &L.w_out, &L.w_mix, &L.w_gu})
                            if (m->p && m->fmt) mk = std::max(mk, size_t(m->cols));
                    // quantised weights stay resident when they fit with 24 GB to spare (GB10): no FP16 dequant and
                    // weight quantisation per chunk (128K prefill: ~10 % of the kernel time)
                    size_t w8 = 0;
                    for (const auto & L : layer)
                        for (const Mat * m : {&L.w_gate, &L.w_up, &L.w_down, &L.wq, &L.wk, &L.wv, &L.wo, &L.w_qkv, &L.w_z,
                                              &L.w_a, &L.w_b, &L.w_out, &L.w_mix, &L.w_gu})
                            if (m->p && m->fmt) w8 += size_t(m->rows) * (m->cols + 4);
                    const size_t resident = prefill_room >= w8 + (size_t(24) << 30) ? w8 : 0;
                    fp8 = std::make_unique<tqllm_cuda::Fp8Gemm>();
                    fp8->init(mx, size_t(NBUF) * mk, int(mr), NBUF, resident);
                    fp8_sw = Buf(size_t(mr) * 4);
                    std::fprintf(stderr, "tqllm: prefill GEMMs in FP8 (E4M3, per-row x per-token scales%s; --prefill-fp8 off for FP16)\n",
                                 resident ? ", weights resident" : "");
                } else if (opt.prefill_fp8 == 1) {
                    std::fprintf(stderr, "tqllm: --prefill-fp8 needs FP8 tensor cores (sm_89+); FP16 prefill\n");
                }
            }
            // int8 path (Ampere): int32 results live in gemm_c ([NBUF][mr] x 4 bytes); the FP16 copy the fused SwiGLU
            // reads goes through gemm_w, which must hold NBUF x mr halves
            if (!fp8 && (opt.prefill_int8 == 1 || (opt.prefill_int8 < 0 && !opt.exact && tqllm_cuda::Int8Gemm::default_on())) && !gemm_w_rows &&
                tqllm_cuda::Int8Gemm::supported() &&
                size_t(NBUF) * mr <= mx) {  // gemm_c below is [NBUF][mr] x 4 bytes: room for int32
                size_t mk = 0;
                for (const auto & L : layer)
                    for (const Mat * m : {&L.w_gate, &L.w_up, &L.w_down, &L.wq, &L.wk, &L.wv, &L.wo, &L.w_qkv, &L.w_z,
                                          &L.w_a, &L.w_b, &L.w_out, &L.w_mix, &L.w_gu})
                        if (m->p && m->fmt) mk = std::max(mk, size_t(m->cols));
                i8 = std::make_unique<tqllm_cuda::Int8Gemm>();
                i8->init(mx, size_t(NBUF) * mk, int(mr), NBUF);
                if (!fp8_sw.p) fp8_sw = Buf(size_t(mr) * 4);
                std::fprintf(stderr, "tqllm: prefill GEMMs in int8 (per-row x per-token scales; --prefill-int8 off for FP16)\n");
            }
            gemm_c_bytes = size_t(NBUF) * mr * 4;
            gemm_c = Buf(gemm_c_bytes);
            blas_ws = Buf(size_t(32) << 20);
            cb(cublasSetWorkspace(blas, blas_ws.p, size_t(32) << 20));
        }
    }
    reset_state();
}

void Qwen35Engine::alloc_moe_cache() {
    if (opt.moe_cache_mb <= 0 || !moe_stage_slots) return;
    moe_cache_of.assign(layer.size(), -1), moe_cold_base.assign(layer.size(), 0);
    int total = 0;
    for (size_t l = 0; l < layer.size(); ++l) {
        const auto & L = layer[l];
        moe_cold_base[l] = total;
        if (L.n_cold <= 0) continue;
        total += L.n_cold;
        MoeCache sig;
        const Mat * m[3] = {&L.cx_gate, &L.cx_up, &L.cx_down};
        for (int j = 0; j < 3; ++j) {
            const long long per = m[j]->rows / L.n_cold;
            sig.bytes[2 * j] = per * (long long)m[j]->fmt->row_bytes(m[j]->cols);
            sig.bytes[2 * j + 1] = m[j]->aux ? per * (long long)m[j]->fmt->aux_row_bytes(m[j]->cols) : 0;
            sig.types[j] = m[j]->type;
        }
        int pool = -1;
        for (size_t i = 0; i < moe_cache.size() && pool < 0; ++i)
            if (std::equal(sig.bytes, sig.bytes + tqllm_cuda::kMoeStageSpans, moe_cache[i].bytes) &&
                std::equal(sig.types, sig.types + 3, moe_cache[i].types)) pool = int(i);
        if (pool < 0) { pool = int(moe_cache.size()); moe_cache.push_back(std::move(sig)); }
        moe_cache[size_t(pool)].weight += L.n_cold;
        moe_cache_of[l] = pool;
    }
    long wsum = 0;
    for (const auto & c : moe_cache) wsum += c.weight;
    const double budget = double(opt.moe_cache_mb) * 1048576.0;
    size_t used = 0;
    for (size_t i = 0; i < moe_cache.size(); ++i) {
        auto & c = moe_cache[i];
        long long slot_b = 0;
        for (long long b : c.bytes) slot_b += b;
        c.slots = int(budget * double(c.weight) / double(wsum) / double(slot_b));
        if (c.slots < 2 * moe_stage_slots) {  // too small to keep a step's experts: those layers use the stage
            for (auto & p : moe_cache_of) if (p == int(i)) p = -1;
            c.slots = 0;
            continue;
        }
        for (int j = 0; j < tqllm_cuda::kMoeStageSpans; ++j)
            if (c.bytes[j]) c.buf[j] = Buf(size_t(c.bytes[j]) * c.slots), used += size_t(c.bytes[j]) * c.slots;
        c.key = Buf(size_t(c.slots) * sizeof(int)); ck(cudaMemset(c.key.p, 0xff, size_t(c.slots) * sizeof(int)));
        c.ref = Buf(size_t(c.slots) * sizeof(int)); ck(cudaMemset(c.ref.p, 0, size_t(c.slots) * sizeof(int)));
        c.hand = Buf(sizeof(int)); ck(cudaMemset(c.hand.p, 0, sizeof(int)));
    }
    moe_cache_table = Buf(size_t(std::max(1, total)) * sizeof(int));
    ck(cudaMemset(moe_cache_table.p, 0xff, size_t(std::max(1, total)) * sizeof(int)));
    moe_cache_slot = Buf(size_t(moe_stage_slots) * sizeof(int));
    moe_cache_fill = Buf(size_t(moe_stage_slots) * sizeof(int));
    int slots = 0;
    for (const auto & c : moe_cache) slots += c.slots;
    std::fprintf(stderr, "tqllm: MoE host-expert cache: %d slots in %zu pool(s), %.0f MiB of VRAM (--moe-cache-mb)\n", slots,
                 moe_cache.size(), used / 1048576.0);
}

void Qwen35Engine::alloc_moe_stage() {
    long long most[tqllm_cuda::kMoeStageSpans] = {};
    for (const auto & L : layer) {
        if (L.n_cold <= 0) continue;
        const Mat * m[3] = {&L.cx_gate, &L.cx_up, &L.cx_down};
        for (int j = 0; j < 3; ++j) {
            const long long per = m[j]->rows / L.n_cold;
            const long long pb = per * (long long)m[j]->fmt->row_bytes(m[j]->cols);
            const long long ab = m[j]->aux ? per * (long long)m[j]->fmt->aux_row_bytes(m[j]->cols) : 0;
            if (pb % 16 || ab % 16) return;
            most[2 * j] = std::max(most[2 * j], pb), most[2 * j + 1] = std::max(most[2 * j + 1], ab);
        }
    }
    if (!most[0]) return;
    moe_stage_slots = 2 * (int(cfg.expert_used) + 1);  // two tokens, routed + fused shared expert
    size_t total = 0;
    for (int i = 0; i < tqllm_cuda::kMoeStageSpans; ++i)
        if (most[i]) { moe_stage_buf[i] = Buf(size_t(most[i]) * moe_stage_slots); total += size_t(most[i]) * moe_stage_slots; }
    moe_stage_ids = Buf(size_t(moe_stage_slots) * sizeof(int));
    std::fprintf(stderr, "tqllm: MoE host experts staged through %.0f MiB of VRAM (%d assignments)\n", total / 1048576.0,
                 moe_stage_slots);
}

void Qwen35Engine::upload_rpos(int nseg, const std::function<int(int)> & delta) {
    for (int i = 0; i < nseg; ++i) {
        h_rpos[2 * i] = h_pos[2 * i] - delta(i);
        h_rpos[2 * i + 1] = h_pos[2 * i + 1] - delta(i);
    }
    ck(cudaMemcpyAsync(v_rpos.p, h_rpos, 2 * size_t(nseg) * sizeof(int), cudaMemcpyHostToDevice, stream));
}

void Qwen35Engine::alloc_dfl_state() {
    const size_t kd = size_t(dfl.kvheads) * dfl.hd, rows = dfl.ring ? size_t(dfl.ring) : context_;
    for (size_t l = 0; l < dfl.L.size(); ++l) {
        dfl_kc.emplace_back(rows * kd * 2);
        dfl_vc.emplace_back(rows * kd * 2);
    }
}

void Qwen35Engine::load_dflash(const std::string & path) {
    const auto f = gguf_open(path);
    const std::string A = "dflash.";
    const int H = int(cfg.hidden);
    if (int(f.num(A + "embedding_length")) != H) throw std::runtime_error("DFlash drafter hidden size mismatch");
    dfl.dspark = f.tensors.count("markov_w1.weight") > 0;
    dfl.ffn = int(f.num(A + "feed_forward_length"));
    dfl.theta = dfl.dspark ? float(f.num(A + "rope.freq_base", cfg.theta)) : cfg.theta;  // DFlash2: the target's, as before
    const int F = dfl.ffn;
    dfl.heads = int(f.num(A + "attention.head_count"));
    dfl.kvheads = int(f.num(A + "attention.head_count_kv"));
    dfl.hd = int(f.num(A + "attention.key_length"));
    dfl.block = int(f.num(A + "block_size"));
    dfl.nb = dfl.block;
    if (dfl.dspark) {
        if (!f.num(A + "sample_from_anchor", 1)) throw std::runtime_error("DSpark drafter: only sample_from_anchor is supported");
        dfl.nb = std::min(dfl.nb, NVER - 1);  // a runtime block may be shorter than the trained one; lattice <= verify width
        dfl.block = dfl.nb + 1;
    }
    dfl.group = dfl.dspark ? 0 : int(f.num(A + "conv_group_size"));
    dfl.window = int(f.num(A + "attention.sliding_window", 0));
    if (opt.draft_window >= 0) {
        std::fprintf(stderr, "tqllm: drafter window %d -> %d (--draft-window)\n", dfl.window, opt.draft_window);
        dfl.window = opt.draft_window;
    }
    if (opt.draft_retrieve > 0 && dfl.window > 0) {
        dfl.retr = std::min(opt.draft_retrieve, 128);
        dfl.recent = std::max(64, dfl.window - dfl.retr * 16);
        std::fprintf(stderr, "tqllm: drafter retrieval window: %d far chunks of 16 + %d recent positions (--draft-retrieve)\n",
                     dfl.retr, dfl.recent);
    }
    dfl.topk = dfl.dspark ? 16 : int(f.num(A + "selector_top_k"));
    dfl.rank = dfl.dspark ? int(gt(f, "markov_w1.weight").ne[0]) : int(f.num(A + "selector_rank"));
    dfl.mask_id = uint32_t(f.num("tokenizer.ggml.mask_token_id"));
    for (double v : f.kv.at(A + "target_layers").arr) dfl.target_ids.push_back(int(v));
    const int nl = int(f.num(A + "block_count"));
    if (dfl.dspark ? dfl.hd != 256 || dfl.rank != 256 || dfl.heads / dfl.kvheads > 8
                   : dfl.hd != 128 || dfl.topk != 16 || int(f.num(A + "conv_kernel_size")) != 2 || F != int(cfg.ffn))
        throw std::runtime_error(dfl.dspark ? "unsupported DSpark drafter shape" : "unsupported DFlash2 drafter shape");
    const int nf = int(dfl.target_ids.size()), qd = dfl.heads * dfl.hd, kd = dfl.kvheads * dfl.hd;
    dfl.fc = to_tq3(f, {"fc.weight"}, nf * H);
    dfl.enc_norm = f32_of(f, "enc.output_norm.weight", H);
    dfl.out_norm = f32_of(f, "output_norm.weight", H);
    if (dfl.dspark) {
        dfl.sel_prev = vocab_table(f, "markov_w1.weight", dfl.rank);
        dfl.sel_next = vocab_table(f, "markov_w2.weight", dfl.rank);
    } else {
        dfl.sel_hidden = to_tq3(f, {"selector_hidden.weight"}, H);
        dfl.sel_prev = bf16_table(f, "selector_predecessor.weight", uint64_t(cfg.vocab) * dfl.rank);
        dfl.sel_next = bf16_table(f, "selector_successor.weight", uint64_t(cfg.vocab) * dfl.rank);
    }
    std::vector<std::string> kv_names;
    for (int l = 0; l < nl; ++l) {
        const auto p = "blk." + std::to_string(l) + ".";
        DFlashLayer L;
        L.attn_norm = f32_of(f, p + "attn_norm.weight", H);
        L.q_norm = f32_of(f, p + "attn_q_norm.weight", dfl.hd);
        L.k_norm = f32_of(f, p + "attn_k_norm.weight", dfl.hd);
        L.ffn_norm = f32_of(f, p + "ffn_norm.weight", H);
        if (!dfl.dspark) {
            L.attn_conv_base = f32_of(f, p + "attn_conv_base", uint64_t(H) * 4);
            L.ffn_conv_base = f32_of(f, p + "ffn_conv_base", uint64_t(H) * 4);
            L.conv_attn = to_tq3(f, {p + "attn_conv_proj.weight"}, H);
            L.conv_ffn = to_tq3(f, {p + "ffn_conv_proj.weight"}, H);
        }
        L.w_qkv = to_tq3(f, {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}, H);
        L.wo = to_tq3(f, {p + "attn_output.weight"}, qd);
        L.w_gu = to_tq3(f, {p + "ffn_gate.weight", p + "ffn_up.weight"}, H);
        L.down = to_tq3(f, {p + "ffn_down.weight"}, F);
        dfl.L.push_back(L);
        kv_names.push_back(p + "attn_k.weight");
        kv_names.push_back(p + "attn_v.weight");
    }
    dfl.w_kv_all = to_tq3(f, kv_names, H);  // [l0 k | l0 v | l1 k | ...], each kd rows
    // sliding-window drafter: K/V ring of the window (+ one verify and one draft block, chunk-rounded)
    dfl.ring = dfl.window > 0 ? (dfl.window + NVER + dfl.block + 64 + 63) / 64 * 64 : 0;
    if (dfl.ring >= int(context_)) dfl.ring = 0;
    if (dfl.retr) {  // retrieval reads back at most the newest kDflRetrMaxChunks chunks: a ring that holds them, and
                     // scratch for [retrieved | recent | block]
        dfl.ring = (tqllm_cuda::kDflRetrMaxChunks * tqllm_cuda::kDflRetrChunk + dfl.recent + NVER + dfl.block + 64 + 63) / 64 * 64;
        if (dfl.ring >= int(context_)) dfl.ring = 0;
        const size_t kdb = size_t(dfl.kvheads) * dfl.hd;
        dfl_rscore = Buf(size_t(dfl.kvheads) * 4096 * 4);
        dfl_rsel = Buf(size_t(dfl.kvheads) * 128 * 4);
        dfl_rk = Buf((size_t(128) * 16 + dfl.recent + dfl.block + 64) * kdb * 2);
        dfl_rv = Buf((size_t(128) * 16 + dfl.recent + dfl.block + 64) * kdb * 2);
        dfl_rpos = Buf(sizeof(int));
    }
    alloc_dfl_state();
    const size_t B = size_t(dfl.block) * size_t(nslots);  // batched drafting: one block per slot
    if (size_t(NVER) < B) v_logits = Buf(B * cfg.vocab * 4);
    v_feat = Buf(size_t(NBUF) * nf * H * 2);
    feat1 = Buf(size_t(nf) * H * 2);
    d_x = Buf(B * H * 2);
    d_r = Buf(B * H * 2);
    d_a = Buf(B * H * 2);
    d_b = Buf(B * H * 2);
    d_dyn = Buf(B * size_t(L_conv_rows()) * 2);
    d_dynf = Buf(B * size_t(L_conv_rows()) * 2);
    d_te = Buf(B * H * 2);
    d_gate = Buf(B * dfl.rank * 2);
    if (dfl.dspark) {  // Markov head: unit gate on every rank; head-dim-256 drafter attention partials
        const std::vector<bf16> ones(B * dfl.rank, __float2bfloat16_rn(1.f));
        ck(cudaMemcpy(d_gate.p, ones.data(), ones.size() * 2, cudaMemcpyHostToDevice));
        dfl_part = Buf(size_t(tqllm_cuda::attention_gen256_partial_floats(dfl.heads, int(context_), dfl.nb)) * 4);
    }
    d_vals = Buf(B * 16 * 4);
    d_idx = Buf(B * 16 * 4);
    d_cand = Buf(B * 16 * 4);
    d_score = Buf(B * 256 * 4);
    d_pos = Buf(2 * size_t(nslots) * sizeof(int));
    d_tok = Buf(B * 4);
    d_kv = Buf(size_t(NVER) * nl * 2 * kd * 2);
    // inject scratch for batches wider than the draft buffers (d_x / d_kv hold a draft block and NVER rows): prefill
    // injects in batches of DFlash::injBatch tokens, so the projections read each weight once per batch
    d_inj_g = Buf(size_t(DFlash::injBatch) * H * 2);
    d_inj_kv = Buf(size_t(DFlash::injBatch) * nl * 2 * kd * 2);
    d_topk = Buf(B * 64 * 16 * 8);
    if (const char * dv = opt.draft_vocab.c_str(); *dv && has_head8()) {
        // draft vocabulary (llama.cpp map format): those head8 rows in 4 bits; ids mapped on the device
        std::ifstream in(dv);
        std::string magic;
        uint64_t nv = 0, nd = 0;
        in >> magic >> nv >> nd;
        if (!in || nv != cfg.vocab || nd == 0) throw std::runtime_error(std::string("bad draft vocab file ") + dv);
        std::vector<uint32_t> ids(nd);
        for (auto & id : ids)
            if (!(in >> id) || id >= cfg.vocab) throw std::runtime_error("bad id in draft vocab file");
        dsub_ids = Buf(nd * 4);
        ck(cudaMemcpyAsync(dsub_ids.p, ids.data(), nd * 4, cudaMemcpyHostToDevice, stream));
        dsub_rows = int(nd);
        // the 4-bit head holds the same rows (both u4h of head8): read them through dsub_ids, no 0.2 GB copy
        if (!head4.p) build_head4();
        if (head4.p) {
            dsub_in_head4 = true;
            sync_ck();
        }
    }
    if (dsub_in_head4) {
    } else if (!opt.draft_vocab.empty() && has_head8()) {
        const uint64_t nd = uint64_t(dsub_rows);
        Buf w8(nd * H), sc8(nd * H / 16 * 2);
        tqllm_cuda::gather_rows(this->w8(), w8.p, dsub_ids.as<uint32_t>(), int(nd), int64_t(H), stream);
        tqllm_cuda::gather_rows(s8(), sc8.p, dsub_ids.as<uint32_t>(), int(nd), int64_t(H) / 16 * 2, stream);
        dsub_w = Buf(nd * H / 2);
        dsub_s = Buf(nd * H / 32 * 4);
        tqllm_cuda::u4h_from_i8h(w8.as<int8_t>(), sc8.as<__half>(), dsub_w.as<uint8_t>(), dsub_s.as<__half2>(),
                                 int64_t(nd) * H, stream);
        tqllm_cuda::u4h_tune(dsub_w.as<uint8_t>(), dsub_s.as<__half2>(), int(nd), H, stream);
        sync_ck();
    } else if (const char * dv2 = opt.draft_vocab.c_str(); *dv2 && lm_head.fmt && lm_head.type == GGML_TQ4) {
        // TQ4 LM head: the draft vocabulary rows gathered into a smaller TQ4 matrix (codes [rows][H / 2], scales [rows][H / 128])
        std::ifstream in(dv2);
        std::string magic;
        uint64_t nv = 0, nd = 0;
        in >> magic >> nv >> nd;
        if (!in || nv != cfg.vocab || nd == 0) throw std::runtime_error(std::string("bad draft vocab file ") + dv2);
        std::vector<uint32_t> ids(nd);
        for (auto & id : ids)
            if (!(in >> id) || id >= cfg.vocab) throw std::runtime_error("bad id in draft vocab file");
        dsub_ids = Buf(nd * 4);
        ck(cudaMemcpyAsync(dsub_ids.p, ids.data(), nd * 4, cudaMemcpyHostToDevice, stream));
        dsub_w = Buf(nd * H / 2);
        dsub_s = Buf(nd * H / 128 * 2);
        tqllm_cuda::gather_rows(lm_head.p, dsub_w.p, dsub_ids.as<uint32_t>(), int(nd), int64_t(H) / 2, stream);
        tqllm_cuda::gather_rows(lm_head.aux, dsub_s.p, dsub_ids.as<uint32_t>(), int(nd), int64_t(H) / 128 * 2, stream);
        sync_ck();
        dsub_mat = lm_head;
        dsub_mat.p = dsub_w.p;
        dsub_mat.aux = dsub_s.p;
        dsub_mat.rows = int(nd);
        dsub_rows = int(nd);
    }
    if (!head4.p && head8.p) build_head4();
    if (dfl.dspark && tree_mode) {  // tree verify is not yet greedy-identical to plain decode on this path (GB10 MoE)
        std::fprintf(stderr, "tqllm: DSpark drafter: chain verify (tree verify off)\n");
        tree_mode = defer = false;
    }
    dfl.present = true;
}

void Qwen35Engine::build_head4() {
    const int64_t n = int64_t(lm_head.rows) * cfg.hidden;
    head4 = Buf(size_t(n / 2));
    head4s = Buf(size_t(n / 32) * 4);
    tqllm_cuda::u4h_from_i8h(w8(), s8(), head4.as<uint8_t>(), head4s.as<__half2>(), n, stream);
    tqllm_cuda::u4h_tune(head4.as<uint8_t>(), head4s.as<__half2>(), lm_head.rows, int(cfg.hidden), stream);
    h2_vals = Buf(size_t(NB) * 16 * 4);
    h2_idx = Buf(size_t(NB) * 16 * 4);
    h2_scr = Buf(size_t(NB) * 64 * 16 * 8);
    sync_ck();
}

}  // namespace qwen35_detail
}  // namespace tqllm
