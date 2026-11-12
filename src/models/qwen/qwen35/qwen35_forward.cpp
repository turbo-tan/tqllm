// qwen35 engine: the forward pass (decode, prefill, layers). The class is declared in qwen35_engine.hpp.
#include "qwen35_engine.hpp"

namespace tqllm {
namespace qwen35_detail {

void Qwen35Engine::release_raw() {
    if (stream) cudaStreamSynchronize(stream);
    glru.clear([](cudaGraphExec_t g) { cudaGraphExecDestroy(g); });
    if (side) cudaStreamDestroy(side);
    if (kv_side) cudaStreamDestroy(kv_side);
    if (solver) cusolverDnDestroy(solver);
    solver = nullptr;
    for (auto e : {kv_fork, kv_ready[0], kv_ready[1], kv_done[0], kv_done[1]})
        if (e) cudaEventDestroy(e);
    kv_side = nullptr, kv_fork = kv_ready[0] = kv_ready[1] = kv_done[0] = kv_done[1] = nullptr;
    if (ev_fork) cudaEventDestroy(ev_fork);
    if (ev_join) cudaEventDestroy(ev_join);
    if (moe_fork) cudaEventDestroy(moe_fork);
    if (moe_join) cudaEventDestroy(moe_join);
    if (moe_side) cudaStreamDestroy(moe_side);
    moe_side = nullptr, moe_fork = moe_join = nullptr;
    for (auto st : seg_side) cudaStreamDestroy(st);
    for (auto ev : seg_join) cudaEventDestroy(ev);
    if (seg_fork) cudaEventDestroy(seg_fork);
    if (h_tok) cudaFreeHost(h_tok);
    if (h8_host) cudaFreeHost(h8_host);
    for (void * h : host_blocks) cudaFreeHost(h);
    if (h_pos) cudaFreeHost(h_pos);
    if (h_rpos) cudaFreeHost(h_rpos);
    if (blas) cublasDestroy(blas);
    if (stream) cudaStreamDestroy(stream);
    host_blocks.clear(), seg_side.clear(), seg_join.clear();
    side = nullptr, ev_fork = ev_join = seg_fork = nullptr, h_tok = nullptr, h8_host = nullptr, h_pos = nullptr;
    h_rpos = nullptr;
    blas = nullptr, stream = nullptr;
}

uint32_t Qwen35Engine::step(uint32_t id, uint32_t pos, bool sample, float * capture) {
    if (id >= cfg.vocab || pos >= context_) throw std::runtime_error("token or context outside configured bounds");
    {  // weights the single-token path cannot read (TQ4, W4A16, ...): one-column verify + commit, the same
       // per-column arithmetic as verify(n) (tqverify then checks verify(n) against this chain)
        const Mat & probe = layer[0].w_gu.p ? layer[0].w_gu : layer[0].w_down;
        if (cfg.moe || (probe.type != GGML_TQ3_4S && probe.type != GGML_BF16)) {
            if (capture) throw std::runtime_error("hidden-state capture needs TQ3_4S / BF16 weights");
            uint32_t next = 0;
            verify(&id, 1, pos, &next);
            commit(1);
            return sample ? next : 0;
        }
    }
    if (kvf->compressed()) throw std::runtime_error("compressed KV formats require the multi-token verify path");
    const int H = int(cfg.hidden), F = int(cfg.ffn), nh = int(cfg.heads), kvh = int(cfg.kvheads);
    const int hd = int(cfg.head_dim), lkh = int(cfg.lin_kheads), lvh = int(cfg.lin_vheads);
    const int dk = int(cfg.lin_dk), dv = int(cfg.lin_dv);
    const int qkv_dim = 2 * lkh * dk + lvh * dv, zdim = lvh * dv;
    auto *xp = x.as<bf16>(), *np = norm.as<bf16>(), *rp = resid.as<bf16>(), *mp = mix.as<bf16>();
    if (pos == 0) reset_state(), pc_drop();  // recurrent layers carry state; a new sequence starts from zero

    if (embed.type == GGML_TQ3_4S) tqllm_cuda::tq3_row(embed.p, int(id), xp, H, stream);
    else if (embed.type == GGML_Q6_K) tqllm_cuda::q6k_row(embed.p, int(id), xp, H, stream);
    else if (embed.type == GGML_IQ3_S) tqllm_cuda::iq3s_row(embed.p, int(id), xp, H, stream);
    else if (embed.type == GGML_IQ4_XS) tqllm_cuda::iq4xs_row(embed.p, int(id), xp, H, stream);
    else tqllm_cuda::gather(static_cast<const bf16 *>(embed.p), xp, int(id), H, stream);
    if (quantized) tqllm_cuda::sumsq(xp, part_x.as<float>(), H, stream);
    uint32_t attn_i = 0, lin_i = 0;
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        const auto & L = layer[l];
        capture_feature(l, xp, feat1.as<bf16>(), 1);
        ck(cudaMemcpyAsync(rp, xp, size_t(H) * 2, cudaMemcpyDeviceToDevice, stream));
        norm_prep(xp, L.in_norm, part_x.as<float>(), np, H);
        if (cfg.full[l]) {
            auto *qg = t_qg.as<bf16>(), *q = t_q.as<bf16>(), *k = t_k.as<bf16>(), *v = t_v.as<bf16>();
            auto *o = t_o.as<bf16>();
            auto *kc = key_cache[attn_i].as<bf16>(), *vc = value_cache[attn_i].as<bf16>();
            if (L.w_mix.p) {
                mm(L.w_mix, np, t_in.as<bf16>());
                qg = t_in.as<bf16>();
                k = qg + (cfg.gated_attention ? 2 : 1) * nh * hd;
                v = k + kvh * hd;
            } else {
                mm(L.wq, np, qg);
                mm(L.wk, np, k);
                mm(L.wv, np, v);
            }
            tqllm_cuda::rms_f(qg, (cfg.gated_attention ? 2 : 1) * hd, L.q_norm, q, hd, hd, nh, cfg.eps, stream);
            tqllm_cuda::rms_f(k, hd, L.k_norm, k, hd, hd, kvh, cfg.eps, stream);
            tqllm_cuda::rope_partial(q, nh, hd, int(cfg.rot_dim), int(pos), cfg.theta, stream);
            tqllm_cuda::rope_partial(k, kvh, hd, int(cfg.rot_dim), int(pos), cfg.theta, stream);
            tqllm_cuda::cache_write(k, v, kc, vc, int(pos), kvh, hd, stream);
            if (!cfg.gated_attention) {
                h_pos[0] = int(pos); h_pos[1] = int(pos) + 1;
                ck(cudaMemcpyAsync(v_pos.p, h_pos, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
                tqllm_cuda::attention_gen128(q, kc, vc, o, part.as<float>(), v_pos.as<int>(), 1, int(context_), nh, kvh,
                                            1.f / std::sqrt(float(hd)), true, 0, stream);
            } else {
            tqllm_cuda::attention_decode(q, kc, vc, o, part.as<float>(), int(pos + 1), nh, kvh, hd,
                                         1.f / std::sqrt(float(hd)), stream);
            }
            if (cfg.gated_attention) tqllm_cuda::sigmoid_gate(o, qg, nh, hd, stream);
            prep(o, nh * hd);
            mm(L.wo, o, mp);
            ++attn_i;
        } else {
            auto *qkv = t_qkv.as<bf16>(), *qkvc = t_qkvc.as<bf16>(), *z = t_z.as<bf16>();
            auto *a = t_a.as<bf16>(), *b = t_b.as<bf16>(), *core = t_core.as<bf16>(), *gn = t_gn.as<bf16>();
            if (L.w_mix.p) {
                mm(L.w_mix, np, t_in.as<bf16>());
                qkv = t_in.as<bf16>();
                z = qkv + qkv_dim;
                b = z + zdim;
                a = b + lvh;
            } else {
                mm(L.w_qkv, np, qkv);
                mm(L.w_z, np, z);
                mm(L.w_b, np, b);
                mm(L.w_a, np, a);
            }
            tqllm_cuda::conv_step(qkv, conv_slot(lin_i, cur), L.conv_w, qkvc, qkv_dim, stream);
            tqllm_cuda::gdn_step(qkvc, a, b, L.neg_a, L.dt_bias, rec_slot(lin_i, cur), core, lkh, lvh, dk, dv,
                                 cfg.tiled, stream);
            tqllm_cuda::gated_rms(core, z, L.gnorm_w, gn, lvh, dv, cfg.eps, stream);
            prep(gn, zdim);
            mm(L.w_out, gn, mp);
            ++lin_i;
        }
        if (quantized) tqllm_cuda::add_sumsq(mp, rp, part_m.as<float>(), H, stream);  // mix = residual + mixer
        else tqllm_cuda::residual(mp, rp, H, stream);

        norm_prep(mp, L.post_norm, part_m.as<float>(), np, H);
        if (cfg.moe) throw std::runtime_error("qwen35moe: expert FFN forward is not implemented yet (C2 step 2)");
        const bf16 *gp = gate.as<bf16>(), *upp = up.as<bf16>();
        if (L.w_gu.p) {
            mm(L.w_gu, np, t_in.as<bf16>());
            gp = t_in.as<bf16>();
            upp = gp + F;
        } else {
            mm(L.w_gate, np, gate.as<bf16>());
            mm(L.w_up, np, up.as<bf16>());
        }
        tqllm_cuda::swiglu(gp, upp, act.as<bf16>(), F, stream);
        prep(act.as<bf16>(), F);
        mm(L.w_down, act.as<bf16>(), xp);
        if (quantized) tqllm_cuda::add_sumsq(xp, mp, part_x.as<float>(), H, stream);
        else tqllm_cuda::residual(xp, mp, H, stream);
        if (capture && l + 1 < cfg.layers) to_host(xp, H, capture + size_t(l) * H);
    }
    if (!sample) {
        sync_ck();
        return 0;
    }

    tqllm_cuda::rms_f(xp, 0, out_norm, np, 0, H, 1, cfg.eps, stream);
    if (capture) to_host(np, H, capture + size_t(cfg.layers - 1) * H);
    if (k_head() && has_head8()) {  // same int8 copy as the verify path (the Q6_K original is freed)
        tqllm_cuda::bf16_to_h(np, v_xh.as<__half>(), H, stream);
        tqllm_cuda::i8h_matvec(w8(), s8(), v_xh.as<__half>(), logits_buf.as<float>(),
                               lm_head.rows, H, 1, stream);
    } else if (lm_head.type == GGML_Q6_K) {
        tqllm_cuda::bf16_to_f32(np, xf.as<float>(), H, stream);
        tqllm_cuda::q6k_matvec(lm_head.p, xf.as<float>(), logits_buf.as<float>(), lm_head.rows, H, 1, stream);
    } else if (lm_head.type == GGML_TQ3_4S) {
        throw std::runtime_error("TQ3_4S LM head not supported yet");
    } else {
        const float alpha = 1.f, beta = 0.f;
        cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, int(cfg.vocab), 1, H, &alpha, lm_head.p, CUDA_R_16BF, H, np,
                        CUDA_R_16BF, H, &beta, logits_buf.as<float>(), CUDA_R_32F, int(cfg.vocab),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    tqllm_cuda::argmax_big(logits_buf.as<float>(), int(cfg.vocab), token.as<uint32_t>(), amax.p, stream);
    uint32_t next = 0;
    ck(cudaMemcpyAsync(&next, token.p, sizeof(next), cudaMemcpyDeviceToHost, stream));
    sync_ck();
    return next;
}

bool Qwen35Engine::cmp_pending() const {
    if (!kv_cmp || ring_used < 64 || cmp_pos + kCmpBlock > cmp_max) return false;
    // staged: blocks already in host memory; device-resident: blocks older than the recent exact window
    const int64_t ev = kv_staged ? std::min(int64_t(kv_pool->staged_bytes(true) / size_t(kv_row_bytes(true))),
                                            int64_t(kv_pool->staged_bytes(false) / size_t(kv_row_bytes(false))))
                                 : int64_t(kv_end) - kCmpWindow;
    return ev >= cmp_pos + kCmpBlock;
}

bool Qwen35Engine::kv_idle_step() {
    if (!cmp_pending()) return false;
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t kr = kv_row_bytes(true), vr = kv_row_bytes(false);
    for (size_t a = 0; a < key_cache.size(); ++a) {
        const int m = kCmpBlock / cmp_r[a];
        const char * sk = kv_staged ? static_cast<const char *>(kv_pool->host_copy(int(a), true)) : static_cast<const char *>(key_cache[a].p);
        const char * sv = kv_staged ? static_cast<const char *>(kv_pool->host_copy(int(a), false)) : static_cast<const char *>(value_cache[a].p);
        const void * hk = sk + cmp_pos * kr, * hv = sv + cmp_pos * vr;
        int kcod = 0, vcod = 0;
        kvf->compact_codecs(kcod, vcod);
        tqllm_cuda::kv_compact_block(hk, hv, tqllm_cuda::KvCodec(kcod), tqllm_cuda::KvCodec(vcod), kCmpBlock, int(cfg.kvheads), int(cfg.heads), q_ring[a].as<bf16>(), ring_used,
                                     1.f / std::sqrt(float(cfg.head_dim)), m, int(cmp_pos / cmp_r[a]), cmp_k[a].as<tqllm_cuda::KvStandIn256>(),
                                     cmp_v[a].as<tqllm_cuda::KvStandIn256>(), cmp_b[a].as<float>(), cmp_scratch.p, blas, solver, stream, 10, 1e-3f, 0, kRing, ring_end);
    }
    cmp_pos += kCmpBlock;
    cmp_upload();
    ck(cudaStreamSynchronize(stream));
    std::fprintf(stderr, "tqllm: TqCache: positions below %lld compacted %dx (%dx in the middle layers), %.0f ms\n",
                 (long long)cmp_pos, cmp_r[0], *std::min_element(cmp_r.begin(), cmp_r.end()),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return cmp_pending();
}

void Qwen35Engine::cmp_upload() {
    std::vector<int> ci{int(cmp_pos)};
    for (int r : cmp_r) ci.push_back(int(cmp_pos / r));
    const int64_t off[2] = {cmp_pos * kv_row_bytes(true), cmp_pos * kv_row_bytes(false)};
    ck(cudaMemcpyAsync(d_cmp_i.p, ci.data(), ci.size() * sizeof(int), cudaMemcpyHostToDevice, stream));  // pageable: staged now
    ck(cudaMemcpyAsync(d_stage_off.p, off, sizeof(off), cudaMemcpyHostToDevice, stream));
}

void Qwen35Engine::pc_copy(bool save) {  // recurrent state of slot cur <-> pc_host
    const size_t cb = conv_bytes(), rb = rec_bytes(), nl = conv_state.size() / NSLOT;
    if (!pc_host) pc_host = std::make_unique<HostBuf>(nl * (cb + rb));
    auto * h = static_cast<char *>(pc_host->p);
    for (uint32_t l = 0; l < nl; ++l, h += cb + rb) {
        ck(cudaMemcpyAsync(save ? h : static_cast<void *>(conv_slot(l, cur)), save ? static_cast<void *>(conv_slot(l, cur)) : h,
                           cb, cudaMemcpyDefault, stream));
        ck(cudaMemcpyAsync(save ? h + cb : static_cast<void *>(rec_slot(l, cur)), save ? static_cast<void *>(rec_slot(l, cur)) : h + cb,
                           rb, cudaMemcpyDefault, stream));
    }
}

uint32_t Qwen35Engine::pc_match(const std::vector<uint32_t> & prompt) {
    const auto P = uint32_t(prompt.size());
    auto shared = [&](uint32_t n) {
        return n > 0 && n < P && n <= pc_seq.size() && std::equal(pc_seq.begin(), pc_seq.begin() + n, prompt.begin());
    };
    if (pc_live && shared(uint32_t(pc_seq.size()))) return uint32_t(pc_seq.size());
    if (!shared(pc_snap)) return 0;
    if (pend_k) set_pending(0);  // the live state's deferred path is discarded with it
    pc_copy(false);
    return pc_snap;
}

void Qwen35Engine::mm(const Mat & m, const bf16 * input, bf16 * output) {
    if (m.type == GGML_TQ3_4S) {
        tqllm_cuda::tq3_matvec_i8(m.p, xq.as<int8_t>(), xs.as<float>(), output, m.rows, m.cols, stream);
        return;
    }
    if (m.type != GGML_BF16) throw std::runtime_error("unsupported matrix type in mm()");
    const float alpha = 1.f, beta = 0.f;
    cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, m.rows, 1, m.cols, &alpha, m.p, CUDA_R_16BF, m.cols, input,
                    CUDA_R_16BF, m.cols, &beta, output, CUDA_R_16BF, m.rows, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

const void * Qwen35Engine::put(const void * host, size_t bytes) {
    Buf b(bytes);
    // cudaMemcpy from pageable memory may return before the DMA lands; copy on our stream and wait, so
    // kernels that post-process the upload (tq3_repack) never see a partial tensor.
    ck(cudaMemcpyAsync(b.p, host, bytes, cudaMemcpyHostToDevice, stream));
    sync_ck();
    const void * p = b.p;
    tensors.push_back(std::move(b));
    return p;
}

const SafeTensor & Qwen35Engine::st(const std::map<std::string, SafeTensor> & idx, const std::string & name, uint64_t elems) {
    const auto it = idx.find(name);
    if (it == idx.end()) throw std::runtime_error("missing tensor: " + name);
    uint64_t have = 1;
    for (auto d : it->second.shape) have *= d;
    if (have != elems) throw std::runtime_error("unexpected shape for " + name);
    return it->second;
}

Mat Qwen35Engine::adopt(Mat m) {
    if (!m.fmt) return m;
    if (m.fmt->basis() == Basis::Tq3Rot32) quantized = true;
    if (fmt_target && basis_ != m.fmt->basis() && m.fmt->basis() != Basis::Plain && basis_ != Basis::Plain)
        throw std::runtime_error(std::string("mixed rotated weight formats in one model: ") + m.fmt->name());
    if (!fmt_target || m.fmt->basis() != Basis::Plain) basis_ = m.fmt->basis();
    fmt_target = true;
    return m;
}

std::vector<char> Qwen35Engine::f32_to_bf16_host(const std::vector<char> & raw) {
    std::vector<char> out(raw.size() / 2);
    for (size_t i = 0; i < raw.size() / 4; ++i) {
        float f;
        std::memcpy(&f, raw.data() + 4 * i, 4);
        const __nv_bfloat16 b = __float2bfloat16_rn(f);
        std::memcpy(out.data() + 2 * i, &b, 2);
    }
    return out;
}

float Qwen35Engine::f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t f32;
    if (exp == 0) {
        if (man == 0) f32 = sign;                       // +-0
        else {                                          // subnormal: normalise man into 1.xxx
            uint32_t m = man, e = 127 - 15 + 1;         // exponent once leading 1 is implicit
            while (!(m & 0x400u)) { m <<= 1; --e; }     // shift until bit 10 set
            f32 = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        f32 = sign | 0x7F800000u | (man << 13);         // inf (man==0) or nan
    } else {
        f32 = sign | ((exp + 112u) << 23) | (man << 13);  // normal: bias -15 -> +127 is +112
    }
    float out;
    std::memcpy(&out, &f32, 4);
    return out;
}

std::vector<char> Qwen35Engine::f16_to_bf16_host(const std::vector<char> & raw) {
    std::vector<char> out(raw.size());  // 2 bytes in -> 2 bytes out
    for (size_t i = 0; i < raw.size() / 2; ++i) {
        uint16_t h;
        std::memcpy(&h, raw.data() + 2 * i, 2);
        const __nv_bfloat16 b = __float2bfloat16_rn(f16_bits_to_f32(h));
        std::memcpy(out.data() + 2 * i, &b, 2);
    }
    return out;
}

Mat Qwen35Engine::g_mat(const GgufFile & f, const std::string & name, int rows, int cols, bool repack) {
    const auto & t = gt(f, name);
    if (t.ne.size() != 2 || t.ne[0] != uint64_t(cols) || t.ne[1] != uint64_t(rows))
        throw std::runtime_error("unexpected shape for " + name);
    const auto host = f.read(t);
    if (t.type == GGML_F32) {
        const auto b = f32_to_bf16_host(host);
        return {put(b.data(), b.size()), GGML_BF16, rows, cols};
    }
    if (t.type == GGML_F16) {
        const auto b = f16_to_bf16_host(host);
        return {put(b.data(), b.size()), GGML_BF16, rows, cols};
    }
    if (repack && t.type != GGML_BF16)  // dense BF16 stays on the engine cuBLAS path
        if (const auto * wf = weight_format(t.type)) return adopt(wf->load(host, rows, cols, uploader(), stream));
    if (t.type != GGML_TQ3_4S && t.type != GGML_Q6_K && t.type != GGML_BF16 &&
        !((t.type == GGML_Q8_0 || t.type == GGML_Q4_K || t.type == GGML_Q3_K || t.type == GGML_Q5_K ||
           t.type == GGML_IQ3_S) && !repack))
        throw std::runtime_error("unsupported matrix type " + std::to_string(t.type) + " for " + name);
    if (t.type == GGML_TQ3_4S) quantized = true;
    return {put(host.data(), host.size()), t.type, rows, cols};
}

std::vector<std::vector<char>> Qwen35Engine::moe_placement(const GgufFile & f) {
    const int L = int(cfg.layers), E = int(cfg.expert_count);
    std::vector<std::vector<char>> cold(size_t(L), std::vector<char>(size_t(E), 0));
    if (opt.moe_host_mb <= 0) return cold;
    const std::string path = opt.moe_profile.empty() ? f.path + ".experts" : opt.moe_profile;
    std::ifstream in(path);
    int pl = 0, pe = 0;
    const bool have = bool(in >> pl >> pe);
    if (have && (pl != L || pe != E)) throw std::runtime_error("routing profile " + path + " is for another model");
    std::vector<std::tuple<unsigned long long, int, int>> order;
    unsigned long long total = 0, placed = 0;
    if (!have)  // no profile: the same share of every layer (rank e of each layer before rank e + 1 of any)
        std::fprintf(stderr, "tqllm: no MoE routing profile (%s): host experts spread evenly over the layers (TQLLM_DEBUG=moe_profile:FILE "
                     "writes one, then --moe-profile FILE)\n", path.c_str());
    for (int l = 0; l < L; ++l) for (int e = 0; e < E; ++e) {
        unsigned long long c = 0;
        if (have && !(in >> c)) throw std::runtime_error("truncated routing profile " + path);
        order.emplace_back(have ? c : (unsigned long long)(E - 1 - e), l, e); total += c;
    }
    std::stable_sort(order.begin(), order.end());
    const double want = double(opt.moe_host_mb) * 1048576.0;
    double bytes = 0; int n = 0;
    for (const auto & [c, l, e] : order) {
        if (bytes >= want) break;
        const auto p = "blk." + std::to_string(l) + ".";
        for (const char * w : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})
            bytes += double(gt(f, p + w).bytes) / E;
        cold[size_t(l)][size_t(e)] = 1; placed += c; ++n;
    }
    std::fprintf(stderr, "tqllm: MoE experts in host memory: %d of %d (%.0f MiB), %.1f %% of the profiled routing (%s)\n", n, L * E,
                 bytes / 1048576.0, total ? 100.0 * double(placed) / double(total) : 0.0, path.c_str());
    return cold;
}

Mat Qwen35Engine::expert_rows(const Mat & m, int e, int rows) {
    Mat v = m;
    v.p = static_cast<const char *>(m.p) + size_t(e) * rows * m.fmt->row_bytes(m.cols);
    if (m.aux) v.aux = static_cast<const char *>(m.aux) + size_t(e) * rows * m.fmt->aux_row_bytes(m.cols);
    v.rows = rows;
    return v;
}

Mat Qwen35Engine::g_cat(const GgufFile & f, const std::vector<std::pair<std::string, int>> & parts, int cols) {
    uint32_t type = 0;
    bool mixed = false;
    for (const auto & [name, r] : parts) {
        const auto & t = gt(f, name);
        if (t.ne.size() != 2 || t.ne[0] != uint64_t(cols) || t.ne[1] != uint64_t(r))
            throw std::runtime_error("unexpected shape for " + name);
        mixed |= type && t.type != type;
        type = t.type;
    }
    // types whose loaders produce Plain-basis formats with strided output
    static const std::set<uint32_t> ok = {GGML_Q8_0, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_IQ3_XXS, GGML_IQ3_S, GGML_IQ4_XS, GGML_IQ4_NL, GGML_Q2_K, GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ1_S};
    const bool combinable =
        std::all_of(parts.begin(), parts.end(), [&](const auto & pr) { return ok.count(gt(f, pr.first).type) > 0; });
    if (mixed && !combinable) return {};
    if (combinable && mixed) {
        std::vector<Mat> ms;
        for (const auto & [name, r] : parts) ms.push_back(g_mat(f, name, r, cols));
        auto c = composite_format(std::move(ms));
        if (!c) throw std::runtime_error("cannot combine the formats of " + parts[0].first + " and its group");
        composites.push_back(std::move(c));
        return composite_mat(*composites.back());
    }
    int rows = 0;
    std::vector<char> host;
    for (const auto & [name, r] : parts) {
        const auto d = f.read(gt(f, name));
        host.insert(host.end(), d.begin(), d.end());
        rows += r;
    }
    if (type == GGML_BF16) return {put(host.data(), host.size()), type, rows, cols};
    if (type == GGML_F32) {
        const auto b = f32_to_bf16_host(host);
        return {put(b.data(), b.size()), GGML_BF16, rows, cols};
    }
    if (type == GGML_F16 && !mixed) {  // e.g. llama-quantize COPY keeps ssm_alpha / ssm_beta in F16
        const auto b = f16_to_bf16_host(host);
        return {put(b.data(), b.size()), GGML_BF16, rows, cols};
    }
    if (const auto * wf = weight_format(type)) return adopt(wf->load(host, rows, cols, uploader(), stream));
    return {};
}

const float * Qwen35Engine::g_f32(const GgufFile & f, const std::string & name, uint64_t n) {
    const auto & t = gt(f, name);
    if (t.type == GGML_BF16 && t.bytes == n * 2) {  // BF16 vectors (e.g. the MTP block's router in Unsloth MTP files): exact widening
        const auto raw = f.read(t);
        std::vector<float> w(n);
        const auto * h = reinterpret_cast<const uint16_t *>(raw.data());
        for (uint64_t i = 0; i < n; ++i) { const uint32_t u = uint32_t(h[i]) << 16; std::memcpy(&w[i], &u, 4); }
        return static_cast<const float *>(put(w.data(), w.size() * 4));
    }
    if (t.type != GGML_F32 || t.bytes != n * 4) throw std::runtime_error("expected F32[" + std::to_string(n) + "] " + name);
    const auto host = f.read(t);
    return static_cast<const float *>(put(host.data(), host.size()));
}

void Qwen35Engine::reset_state() {
    const size_t conv_bytes = this->conv_bytes(), rec_bytes = this->rec_bytes();
    cur = 0;
    if (pend_k) set_pending(0);
    for (uint32_t l = 0; l < conv_state.size() / NSLOT; ++l) {
        ck(cudaMemsetAsync(conv_slot(l, 0), 0, conv_bytes, stream));
        ck(cudaMemsetAsync(rec_slot(l, 0), 0, rec_bytes, stream));
    }
}

void Qwen35Engine::set_rope() {
    RopeScaling rs = opt.rope;
    if (rs.type == "none" && opt.rope.factor == 1.f && cfg.rope_type != "none") rs.type = cfg.rope_type, rs.factor = cfg.rope_factor;
    const long orig = rs.orig_ctx ? rs.orig_ctx : cfg.rope_orig_ctx ? long(cfg.rope_orig_ctx) : long(cfg.ctx_train);
    if (rs.on() && rs.type != "linear" && rs.type != "yarn") throw std::invalid_argument("unknown --rope-scaling " + rs.type);
    if (rs.on() && rs.type == "yarn" && orig <= 0) throw std::invalid_argument("YaRN needs --yarn-orig-ctx (no context_length in the model)");
    const RopeTable t = rope_scaling_table(rs, int(cfg.rot_dim), cfg.theta, orig);
    tqllm_cuda::set_rope_scaling(t.mul.data(), int(t.mul.size()), t.mscale);
    if (rs.on())
        std::fprintf(stderr, "tqllm: RoPE %s x%.2f from %ld positions (cos/sin scale %.4f)\n", rs.type.c_str(), rs.factor, orig, t.mscale);
    else if (cfg.ctx_train && context_ > cfg.ctx_train)
        std::fprintf(stderr, "tqllm: WARNING context %u is past the training context %u without RoPE scaling (--rope-scaling yarn "
                             "--rope-scale %.0f)\n", context_, cfg.ctx_train, std::ceil(double(context_) / cfg.ctx_train));
}

void Qwen35Engine::moe_profile_write() {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    ck(cudaStreamIsCapturing(stream, &cs));
    if (cs != cudaStreamCaptureStatusNone) return;
    const size_t n = size_t(cfg.layers) * cfg.expert_count;
    std::vector<unsigned long long> c(n);
    ck(cudaMemcpyAsync(c.data(), moe_prof.p, n * 8, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream));
    const std::string tmp = moe_prof_path + ".tmp";
    if (FILE * o = std::fopen(tmp.c_str(), "w")) {
        std::fprintf(o, "%u %u\n", unsigned(cfg.layers), unsigned(cfg.expert_count));
        for (size_t i = 0; i < n; ++i) std::fprintf(o, "%llu%c", c[i], (i + 1) % cfg.expert_count ? ' ' : '\n');
        std::fclose(o); std::rename(tmp.c_str(), moe_prof_path.c_str());
    }
}

void Qwen35Engine::swap_seq(SeqState & s) {
    std::swap(kv_seq, s.kv_seq);
    std::swap(key_cache, s.key_cache);
    std::swap(value_cache, s.value_cache);
    std::swap(conv_state, s.conv_state);
    std::swap(rec_state, s.rec_state);
    std::swap(dfl_kc, s.dfl_kc);
    std::swap(dfl_vc, s.dfl_vc);
    std::swap(dfl_graph, s.dfl_graph);
    std::swap(cur, s.cur);
    std::swap(samp, s.samp);
    std::swap(rng, s.rng);
    std::swap(gen, s.gen);
}

void Qwen35Engine::st(const char * name) {
    if (!stage_on) return;
    cudaStreamSynchronize(stream);
    const auto t = std::chrono::steady_clock::now();
    auto & a = stage_acc[name];
    a.first += std::chrono::duration<double, std::micro>(t - stage_t).count();
    ++a.second;
    stage_t = t;
}

void Qwen35Engine::mm_h(const Mat & m, bf16 * out, int n, Basis in) {
    if (m.type == GGML_BF16) {  // small BF16 matrices of W4A16 models (Gated DeltaNet b/a): input is un-rotated
        tqllm_cuda::h_to_bf16(v_xh.as<__half>(), v_xb.as<bf16>(), int64_t(n) * m.cols, stream);
        // fixed-order kernel, not cuBLAS: a column's result must not depend on the verify width
        tqllm_cuda::bf16_matvec_cols(static_cast<const bf16 *>(m.p), v_xb.as<bf16>(), out, m.rows, m.cols, n, stream);
        return;
    }
    if (!m.fmt) throw std::runtime_error("verify path needs a registered weight format");
    if (m.fmt->basis() != in)
        throw std::runtime_error(std::string("matrix format ") + m.fmt->name() + " got its input in the wrong basis");
    auto * xh = v_xh.as<__half>();
    const auto batched = [&] {  // the matvec kernel over column batches (also: batched verify, np = 1 numerics)
        const int mc = m.fmt->max_batch_cols();
        for (int c0 = 0; c0 < n; c0 += mc)
            m.fmt->matvec(m, xh + size_t(c0) * m.cols, out + size_t(c0) * m.rows, std::min(mc, n - c0), stream);
    };
    switch (path_of(m, n)) {
    case PrefillPath::matvec: m.fmt->matvec(m, xh, out, n, stream); return;
    case PrefillPath::batched: batched(); return;
    case PrefillPath::native:
        if (m.fmt->gemm(m, xh, out, n, stream)) return;
        if (!gemm_w.p) { batched(); return; }
        break;
    case PrefillPath::dequant: break;
    }
    gemm_rows(m, out, n, m.rows);
}

void Qwen35Engine::calibrate_prefill() {
    if (!fmt_target || !gemm_w.p || NPF < NGEMM_BLAS) return;
    if (const std::string * v = tqllm::debug_value("prefill_path"); v && *v == "old") return;
    std::map<const WeightFormat *, const Mat *> rep;
    size_t total = 0;
    std::map<const WeightFormat *, size_t> elems;
    for (const auto & L : layer)
        for (const Mat * m : {&L.w_gate, &L.w_up, &L.w_down, &L.wq, &L.wk, &L.wv, &L.wo, &L.w_qkv, &L.w_z,
                              &L.w_a, &L.w_b, &L.w_out, &L.w_mix, &L.w_gu})
            if (m->p && m->fmt && !m->fmt->parts() && m->fmt->max_batch_cols() > 0) {
                const size_t e = size_t(m->rows) * m->cols;
                elems[m->fmt] += e; total += e;
                auto & r = rep[m->fmt];
                if (!r || e > size_t(r->rows) * r->cols) r = m;
            }
    const int n1 = NGEMM_BLAS, n2 = std::min(NBUF, 512);
    if (n2 <= n1) return;
    size_t mr = 0;
    for (const auto & [f, m] : rep) mr = std::max(mr, size_t(m->rows));
    Buf out(size_t(n2) * mr * 2);
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0)); ck(cudaEventCreate(&e1));
    auto time_ms = [&](auto && run) {
        run(); sync_ck();
        ck(cudaEventRecord(e0, stream));
        for (int i = 0; i < 3; ++i) run();
        ck(cudaEventRecord(e1, stream)); sync_ck();
        float ms = 0; ck(cudaEventElapsedTime(&ms, e0, e1));
        return double(ms) / 3;
    };
    for (const auto & [f, m] : rep) {
        auto * y = out.as<bf16>();
        const auto * xh = v_xh.as<__half>();
        const int b = f->max_batch_cols();
        const double scale = double(elems[f]) / (double(m->rows) * m->cols);  // representative -> all matrices of f
        PrefillCosts c{};
        c.pass_cols = b;
        c.pass_ms = scale * time_ms([&] { f->matvec(*m, xh, y, b, stream); });
        if (f->has_gemm()) {
            const double t1 = time_ms([&] { f->gemm(*m, xh, y, n1, stream); }), t2 = time_ms([&] { f->gemm(*m, xh, y, n2, stream); });
            c.native_per_col = scale * std::max(1e-6, (t2 - t1) / (n2 - n1));
            c.native_fix = std::max(0.0, scale * t1 - n1 * c.native_per_col);
        }
        const double d1 = time_ms([&] { gemm_rows(*m, y, n1, m->rows); }), d2 = time_ms([&] { gemm_rows(*m, y, n2, m->rows); });
        c.dq_col = scale * std::max(1e-6, (d2 - d1) / (n2 - n1));
        c.dq_fix = std::max(0.0, scale * d1 - n1 * c.dq_col);
        const auto x = prefill_crossover(c, f->max_matvec_cols(), NPF);
        pf_cross[f] = x;
        std::fprintf(stderr, "tqllm: prefill %s: batched pass %.1f ms / %d cols%s, dequant %.1f + %.3f ms/col -> %s up to %d columns\n",
                     f->name(), c.pass_ms, b, f->has_gemm() ? (", native " + std::to_string(c.native_per_col).substr(0, 5) + " ms/col").c_str() : "",
                     c.dq_fix, c.dq_col, x.native ? "native" : "batched", x.cross);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

PrefillPath Qwen35Engine::path_of(const Mat & m, int n) const {
    const bool verify_width = n <= NVER || !segs.empty() || wide_mm;
    if (const std::string * v = tqllm::debug_value("prefill_path"); v && *v == "old") {  // A/B against the old rule
        if (n <= m.fmt->max_matvec_cols()) return PrefillPath::matvec;
        if (verify_width) return PrefillPath::batched;
        if ((n < NGEMM_BLAS || !gemm_w.p) && m.fmt->has_gemm()) return PrefillPath::native;
        return PrefillPath::dequant;
    }
    return prefill_path(n, prefill_fmt(m), gemm_w.p != nullptr, verify_width);
}

// int8 SmoothQuant scales for this gemm_rows call (one s for all parts of a composite, so the activations are quantised
// once). Per-token int8 activations are where the int8 prefill loses accuracy (the per-row int8 weights do not: the
// FP16 GEMM on int8-rounded weights alone is at the FP16 KLD); moving the outlier channels into the weights recovers it.
const float * Qwen35Engine::smooth_aw(const Mat & m) {
    static const bool off = tqllm::debug_on("i8_nosmooth");
    if (off || !i8 || !m.p || !m.fmt || m.cols % 16) return nullptr;
    std::vector<Mat> parts;
    if (const auto * ps = m.fmt->parts()) parts.assign(ps->begin(), ps->end());
    else parts.push_back(m);
    for (const Mat & p : parts) {
        // rotated formats (TQ3 / TQ4: Walsh-Hadamard input basis) already spread the activation outliers: smoothing buys no
        // accuracy there (int8 drafter features on TQ_XS unchanged) and costs ~2 % of prefill
        if (p.fmt->basis() != Basis::Plain) return nullptr;
        // a format with its own int8 reconstruct but no column-scaled one keeps it, unsmoothed
        if (p.fmt->has_dequant_i8() && !p.fmt->has_dequant_i8_cm()) return nullptr;
    }
    auto it = sm_aw.find(m.p);
    if (it == sm_aw.end()) {  // the weights' column abs-max, once per matrix (at load: smooth_prebuild)
        Buf aw(size_t(m.cols) * 4);
        ck(cudaMemsetAsync(aw.p, 0, size_t(m.cols) * 4, stream));
        for (const Mat & p : parts) {
            p.fmt->dequant_h(p, gemm_w.as<__half>(), stream);
            tqllm_cuda::int8_colmax_h(gemm_w.as<__half>(), p.rows, p.cols, p.cols, aw.as<float>(), stream);
        }
        it = sm_aw.emplace(m.p, std::move(aw)).first;
    }
    return it->second.as<float>();
}

void Qwen35Engine::smooth_prebuild() {
    if (!i8) return;
    for (const auto & L : layer)
        for (const Mat * m : {&L.w_gate, &L.w_up, &L.w_down, &L.wq, &L.wk, &L.wv, &L.wo, &L.w_qkv, &L.w_z,
                              &L.w_a, &L.w_b, &L.w_out, &L.w_mix, &L.w_gu})
            smooth_aw(*m);
    sync_ck();
}

// int8 SmoothQuant scales for this gemm_rows call (one s for all parts of a composite, so the activations are quantised
// once). Per-token int8 activations are where the int8 prefill loses accuracy (the per-row int8 weights do not: the
// FP16 GEMM on int8-rounded weights alone is at the FP16 KLD); moving the outlier channels into the weights recovers it.
bool Qwen35Engine::smooth_setup(const Mat & m, int n) {
    static const float alpha = [] {
        const std::string * v = tqllm::debug_value("i8_alpha");
        return v ? std::stof(*v) : 0.75f;
    }();
    if (!i8 || gemm_pad_rows(n, NBUF) < 1024) return false;
    const float * aw = smooth_aw(m);
    if (!aw) return false;
    ck(cudaMemsetAsync(sm_ax.p, 0, size_t(m.cols) * 4, stream));
    tqllm_cuda::int8_colmax_h(v_xh.as<__half>(), n, m.cols, m.cols, sm_ax.as<float>(), stream);
    tqllm_cuda::int8_smooth_scales(sm_ax.as<float>(), aw, sm_s.as<float>(), m.cols, alpha, stream);
    return true;
}

void Qwen35Engine::gemm_rows(const Mat & m, bf16 * out, int n, int ldo) {
    fp8_x_ready = fp8_used = i8_used = false;
    sm_on = smooth_setup(m, n);
    if (const auto * ps = m.fmt->parts()) {  // composite: each part's GEMM into its own row block
        int r0 = 0;
        for (const auto & p : *ps) {
            gemm_c_rows(p, n, ldo, r0);
            r0 += p.rows;
        }
    } else {
        gemm_c_rows(m, n, ldo, 0);
    }
    if (i8_used) {  // int32 in gemm_c: scale while converting (bf16 out, or FP16 back into gemm_c via gemm_w)
        const auto * acc = gemm_c.as<int32_t>();
        if (out) {
            tqllm_cuda::int8_i32_to_bf16_scaled(acc, out, n, ldo, fp8_sw.as<float>(), i8->sx(), stream);
        } else {
            tqllm_cuda::int8_i32_to_h_scaled(acc, gemm_w.as<__half>(), n, ldo, fp8_sw.as<float>(), i8->sx(), stream);
            ck(cudaMemcpyAsync(gemm_c.p, gemm_w.p, size_t(n) * ldo * 2, cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }
    if (fp8_used) {  // apply the FP8 row x token scales in the pass that reads the output anyway
        if (out) tqllm_cuda::fp8_h_to_bf16_scaled(gemm_c.as<__half>(), out, n, ldo, fp8_sw.as<float>(), fp8->sx(),
                                                  tqllm_cuda::Fp8Gemm::kMul, stream);
        else tqllm_cuda::fp8_scale_out(gemm_c.as<__half>(), ldo, ldo, n, fp8_sw.as<float>(), fp8->sx(),
                                       tqllm_cuda::Fp8Gemm::kMul, stream);
        return;
    }
    if (!out) return;
    tqllm_cuda::h_to_bf16(gemm_c.as<__half>(), out, int64_t(n) * ldo, stream);
}

void Qwen35Engine::gemm_c_rows(const Mat & m, int n, int ldo, int r_off) {
    const size_t rb = m.fmt->row_bytes(m.cols), ab = m.fmt->aux_row_bytes(m.cols);
    const int np = gemm_pad_rows(n, NBUF);
    ck(zero_gemm_padding(v_xh.as<__half>(), n, np, m.cols, stream));
    // int8 only on chunks of >= 1024 tokens: per-chunk weight quantisation does not pay off on the 512-token chunks of
    // 24 GB cards at 256K (3090 192K prompt: int8 641 vs FP16 668 tok/s; 54K: 1760 vs 1445)
    if (i8 && np >= 1024 && m.cols % 4 == 0 && ldo % 4 == 0 && r_off % 4 == 0) {
        // Formats that can quantise their weights to int8 themselves (X3's straight-to-int8 reconstruct with
        // load-time scales) skip the FP16 round trip and the per-chunk weight quantise pass entirely.
        float* sw = fp8_sw.as<float>() + r_off;
        const float * cm = sm_on ? sm_s.as<float>() : nullptr;  // SmoothQuant column scales (smooth_setup)
        const bool preq = cm ? m.fmt->dequant_i8_cm(m, cm, i8->w8(), sw, stream) : m.fmt->dequant_i8(m, i8->w8(), sw, stream);
        if (!preq) m.fmt->dequant_h(m, gemm_w.as<__half>(), stream);
        if (!fp8_x_ready) {
            i8->quant_x(v_xh.as<__half>(), m.cols, np, m.cols, stream, cm);
            fp8_x_ready = true;
        }
        if (preq) i8->gemm_prequant(m.rows, m.cols, np, gemm_c.as<int32_t>() + r_off, ldo, stream);
        else i8->gemm(gemm_w.as<__half>(), m.rows, m.cols, np, gemm_c.as<int32_t>() + r_off, ldo, sw, stream, cm);
        i8_used = true;
        return;
    }
    if (fp8 && m.cols % 16 == 0) {
        const bool res = fp8->cached(m.p);
        if (!res) m.fmt->dequant_h(m, gemm_w.as<__half>(), stream);
        if (!fp8_x_ready) {
            fp8->quant_x(v_xh.as<__half>(), m.cols, np, m.cols, stream);
            fp8_x_ready = true;
        }
        fp8->gemm(m.p, res ? nullptr : gemm_w.as<__half>(), m.rows, m.cols, np, gemm_c.as<__half>() + r_off, ldo,
                  fp8_sw.as<float>() + r_off, stream);
        fp8_used = true;
        return;
    }
    // blocks of a multiple of 256 rows: aligned output offsets (cuBLAS otherwise drops to align1 kernels, 1.5x slower)
    const int rc = gemm_w_rows && rb
                       ? int(std::min<size_t>(m.rows, gemm_w_elems / size_t(m.cols) / 256 * 256)) : m.rows;
    if (rc < m.rows) {  // capped FP16 buffer: dequantise + GEMM in row blocks (results written in place in gemm_c)
        const __half alpha = __float2half(1.f), beta = __float2half(0.f);
        for (int r0 = 0; r0 < m.rows; r0 += rc) {
            Mat v = m;
            v.p = static_cast<const uint8_t *>(m.p) + size_t(r0) * rb;
            v.aux = static_cast<const uint8_t *>(m.aux) + size_t(r0) * ab;
            v.rows = std::min(rc, m.rows - r0);
            m.fmt->dequant_h(v, gemm_w.as<__half>(), stream);
            cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, v.rows, np, m.cols, &alpha, gemm_w.p, CUDA_R_16F, m.cols,
                            v_xh.p, CUDA_R_16F, m.cols, &beta, gemm_c.as<__half>() + r_off + r0, CUDA_R_16F, ldo,
                            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        return;
    }
    if (gemm_w_rows && size_t(m.rows) * m.cols > gemm_w_elems)
        throw std::runtime_error(std::string(m.fmt->name()) + ": capped prefill GEMM buffer needs a row-sliceable format");
    m.fmt->dequant_h(m, gemm_w.as<__half>(), stream);
    const __half alpha = __float2half(1.f), beta = __float2half(0.f);
    cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, m.rows, np, m.cols, &alpha, gemm_w.p, CUDA_R_16F, m.cols,
                    v_xh.p, CUDA_R_16F, m.cols, &beta, gemm_c.as<__half>() + r_off, CUDA_R_16F, ldo,
                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void Qwen35Engine::nrot_h(const bf16 * x, const float * w, const float * partial, __half * xh, int k, int n) {
    switch (basis_) {
        case Basis::Tq4Rot128: tqllm_cuda::tq4_norm_rotate_h(x, w, partial, cfg.eps, xh, k, n, stream); break;
        case Basis::Tq3Rot32: tqllm_cuda::norm_rotate_h(x, w, partial, cfg.eps, xh, k, n, stream); break;
        case Basis::Plain: tqllm_cuda::norm_h(x, w, partial, cfg.eps, xh, k, n, stream); break;
    }
}

void Qwen35Engine::prep_input(Basis b, const bf16 * x, __half * xh, int k, int n) {
    switch (b) {
        case Basis::Tq4Rot128: tqllm_cuda::tq4_rotate_h(x, xh, k, n, stream); break;
        case Basis::Tq3Rot32: tqllm_cuda::tq3_rotate_h(x, xh, k, n, stream); break;
        case Basis::Plain: tqllm_cuda::bf16_to_h(x, xh, int64_t(n) * k, stream); break;
    }
}

void Qwen35Engine::head_mv(const bf16 * in, int n) {
    const int H = int(cfg.hidden), mc = lm_head.fmt->max_batch_cols();
    for (int c0 = 0; c0 < n; c0 += mc) {  // chunks of the format's batch width (per-column results unchanged)
        const int nc = std::min(mc, n - c0);
        prep_input(lm_head.fmt->basis(), in + size_t(c0) * H, v_xh.as<__half>(), H, nc);
        lm_head.fmt->matvec_f32(lm_head, v_xh.as<__half>(), v_logits.as<float>() + size_t(c0) * lm_head.rows, nc,
                                stream);
    }
}

void Qwen35Engine::ba_bf16(const Layer & L, int n, cudaStream_t s) {
    if (!tqllm::bf16_ba_unrotated(basis_) || tqllm::debug_on("rot_ba")) {
        tqllm_cuda::h_to_bf16(v_xh.as<__half>(), v_xb.as<bf16>(), int64_t(n) * L.w_ba.cols, s);
        return;
    }
    tqllm_cuda::norm_h(v_x.as<bf16>(), L.in_norm, v_px.as<float>(), cfg.eps, v_xu.as<__half>(), int(cfg.hidden), n, s);
    tqllm_cuda::h_to_bf16(v_xu.as<__half>(), v_xb.as<bf16>(), int64_t(n) * L.w_ba.cols, s);
}

void Qwen35Engine::mix_mm(const Layer & L, bf16 * T, int n) {
    if (!L.w_ba.p) { mm_h(L.w_mix, T, n); return; }
    const int ra = L.w_mix.rows, rb = L.w_ba.rows, rows = ra + rb;
    // straight into the [qkv | z | b | a] rows when the format can write strided output (no interleaving copies)
    if (L.w_mix.fmt && L.w_ba.type == GGML_BF16 &&
        (n <= L.w_mix.fmt->max_matvec_cols() || ((!segs.empty() || wide_mm) && n <= L.w_mix.fmt->max_batch_cols())) &&
        L.w_mix.fmt->basis() == basis_) {
        // [b | a] (BF16, fixed order) on a side branch while [qkv | z] runs: both only read the prepared input
        if (!side) {
            ck(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
            ck(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
            ck(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming));
        }
        ck(cudaEventRecord(ev_fork, stream));  // before the [qkv | z] matvec: the side branch overlaps it
        ck(cudaStreamWaitEvent(side, ev_fork, 0));
        if (L.w_mix.fmt->matvec_ld(L.w_mix, v_xh.as<__half>(), T, n, rows, stream)) {
            ba_bf16(L, n, side);
            tqllm_cuda::bf16_matvec_cols(static_cast<const bf16 *>(L.w_ba.p), v_xb.as<bf16>(), T + ra, rb, L.w_ba.cols,
                                         n, side, rows);
            ck(cudaEventRecord(ev_join, side));
            ck(cudaStreamWaitEvent(stream, ev_join, 0));
            return;
        }
    }
    // prefill chunk: both GEMMs write straight into the [qkv | z | b | a] rows ([b | a] BF16 on cuBLAS: prefill
    // numerics only, verify keeps the fixed-order kernel)
    if (n >= NGEMM_BLAS && segs.empty() && !wide_mm && gemm_w.p && L.w_mix.fmt &&
        L.w_mix.fmt->basis() == basis_ && L.w_ba.type == GGML_BF16 && n > L.w_mix.fmt->max_matvec_cols() &&
        size_t(n + 8) * rows * 4 <= gemm_c_bytes) {
        gemm_rows(L.w_mix, T, n, rows);
        ba_bf16(L, n, stream);
        const float alpha = 1.f, beta = 0.f;
        cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, rb, n, L.w_ba.cols, &alpha, L.w_ba.p, CUDA_R_16BF, L.w_ba.cols,
                        v_xb.p, CUDA_R_16BF, L.w_ba.cols, &beta, T + ra, CUDA_R_16BF, rows, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return;
    }
    mm_h(L.w_mix, v_mixa.as<bf16>(), n);
    ba_bf16(L, n, stream);
    tqllm_cuda::bf16_matvec_cols(static_cast<const bf16 *>(L.w_ba.p), v_xb.as<bf16>(), v_mixb.as<bf16>(), rb,
                                 L.w_ba.cols, n, stream);
    ck(cudaMemcpy2DAsync(T, size_t(rows) * 2, v_mixa.p, size_t(ra) * 2, size_t(ra) * 2, n, cudaMemcpyDeviceToDevice, stream));
    ck(cudaMemcpy2DAsync(T + ra, size_t(rows) * 2, v_mixb.p, size_t(rb) * 2, size_t(rb) * 2, n, cudaMemcpyDeviceToDevice, stream));
}

void Qwen35Engine::set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence) {
    samp.temperature = temperature;
    samp.presence = presence > 0.f ? presence : 0.f;
    samp.top_p = top_p > 0.f && top_p <= 1.f ? top_p : 1.f;
    samp.top_k = top_k;
    rng.seed(seed);
}

void Qwen35Engine::set_penalties(float repeat, float frequency) {
    samp.repeat = repeat > 0.f ? repeat : 1.f;
    samp.frequency = frequency > 0.f ? frequency : 0.f;
}

void Qwen35Engine::kv_stage_window(uint32_t pos, int n) {
    if (kv_cmp && int64_t(pos) < cmp_pos) cmp_reset();  // a new request rewrites the compacted prefix
    if (!kv_pool->stage_ensure(int64_t(pos), int64_t(pos) + n))
        throw std::runtime_error("KV host tail: the device window is smaller than a write block");
    if (kv_cmp) ring_used = std::min(kRing, ring_used + n), ring_end = int((pos + uint32_t(n)) % kRing);
    const int64_t b[2] = {int64_t(kv_pool->staged_bytes(true)), int64_t(kv_pool->staged_bytes(false))};
    ck(cudaMemcpyAsync(d_stage_bytes.p, b, sizeof(b), cudaMemcpyHostToDevice, stream));
}

void Qwen35Engine::kv_track(uint32_t pos, int n) {
    if (kv_staged) kv_stage_window(pos, n);
    else if (kv_cmp) {
        if (int64_t(pos) < cmp_pos) cmp_reset();  // a new request rewrites the compacted prefix
        ring_used = std::min(kRing, ring_used + n), ring_end = int((pos + uint32_t(n)) % kRing);
    }
    kv_end = pos + uint32_t(n);
}

void Qwen35Engine::kv_prefetch(uint32_t a) {  // attention layer a's old positions host -> its staging buffer (parity a & 1)
    const auto * nb = d_stage_bytes.as<int64_t>();
    const auto * off = kv_cmp ? d_stage_off.as<int64_t>() : nullptr;  // the compacted prefix is not staged
    tqllm_cuda::kv_stage_copy(kv_pool->host_copy(int(a), true), key_cache[a].p, nb, kv_blocks, kv_side, off);
    tqllm_cuda::kv_stage_copy(kv_pool->host_copy(int(a), false), value_cache[a].p, nb + 1, kv_blocks, kv_side, off ? off + 1 : nullptr);
    ck(cudaEventRecord(kv_ready[a & 1], kv_side));
}

void Qwen35Engine::attn_mixer(const Layer & L, bf16 * kc, bf16 * vc, int n, int max_seq, int li) {
    const int nh = int(cfg.heads), hd = int(cfg.head_dim);
    auto *T = v_in.as<bf16>(), *O = v_o.as<bf16>(), *M = v_m.as<bf16>();
    auto * XH = v_xh.as<__half>();
    const bool fuse = cfg.gated_attention && (basis_ == Basis::Tq3Rot32 || basis_ == Basis::Tq4Rot128);  // gate + wo input rotation in one kernel
    if (!L.w_mix.p) throw std::runtime_error("multi-token path needs fused attention weights");
    const int rows = L.w_mix.rows;
    mm_h(L.w_mix, T, n);
    st("attn_qkv_matvec");
    bf16 * Q = v_q.as<bf16>();
    (void) max_seq;  // grid is context-independent: fixed split count, K/V tiles shared by all n queries
    if (li < 0) {
        attn_core(L, T, rows, Q, O, kc, vc, n, v_pos.as<int>(), vpar, vdep, fuse ? XH : nullptr, nullptr,  // v_pos: [0], [1] = + 1
                  v_rpos.as<int>(), mm_chunk ? mm_tab.as<int>() : nullptr);
    } else {
        for_segs([&](size_t i) {
            const auto & g = segs[i];
            attn_core(L, T + size_t(g.col0) * rows, rows, Q + size_t(g.col0) * nh * hd, O + size_t(g.col0) * nh * hd,
                      kc_of(g.slot)[size_t(li)].as<bf16>(), vc_of(g.slot)[size_t(li)].as<bf16>(), g.n,
                      v_pos.as<int>() + 2 * i, g.tree ? d_tpar.as<int>() + g.col0 : chain2 ? d_chain.as<int>() : nullptr,
                      g.tree ? d_tdep.as<int>() + g.col0 : chain2 ? d_cdep.as<int>() : nullptr,
                      fuse ? XH + size_t(g.col0) * nh * hd : nullptr, part_of(i), v_rpos.as<int>() + 2 * i);
        });
    }
    if (!fuse) rot_h(O, XH, nh * hd, n);
    st("attn_core(norm,rope,kv,gate)");
    mm_h(L.wo, M, n);
    st("attn_wo_matvec");
}

void Qwen35Engine::ensure_mq_split() {
    if (mq_split) return;
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    mq_split = std::clamp(kvf->split_factor() * sms / int(cfg.kvheads), 1, 256);
}

void Qwen35Engine::ffn_block(const Layer & L, int n, const bf16 * resid) {
    const int H = int(cfg.hidden), F = int(cfg.ffn);
    auto *X = v_x.as<bf16>(), *M = v_m.as<bf16>(), *R = v_r.as<bf16>(), *T = v_in.as<bf16>();
    auto * ACT = v_act.as<bf16>();
    auto * XH = v_xh.as<__half>();
    tqllm_cuda::add_sumsq(M, resid ? resid : R, v_pm.as<float>(), n * H, stream);
    nrot_h(M, L.post_norm, v_pm.as<float>(), XH, H, n);
    const __half * XIN = XH;  // DIAG: TQLLM_DEBUG=rot_router forces the rotated copy for the router
    if (tqllm::bf16_ba_unrotated(basis_) && v_xu.p && !tqllm::debug_on("rot_router")) {
        tqllm_cuda::norm_h(M, L.post_norm, v_pm.as<float>(), cfg.eps, v_xu.as<__half>(), H, n, stream);
        XIN = v_xu.as<__half>();
    }
    if (cfg.moe) {
        const int E = int(cfg.expert_count), K = int(cfg.expert_used), FE = int(cfg.expert_ff), FS = int(cfg.shared_ff);
        tqllm_cuda::moe_router(L.router, L.sh_gate, XIN, moe_logits.as<float>(), H, E, n, stream);
        tqllm_cuda::moe_topk(moe_logits.as<float>(), moe_ids.as<int>(), moe_weights.as<float>(),
                             moe_shared_weights.as<float>(), E, K, n, 1.f, stream);
        const long li = &L - layer.data();
        if (moe_prof.p && li >= 0 && li < long(cfg.layers)) {
            tqllm_cuda::moe_count(moe_ids.as<int>(), moe_prof.as<unsigned long long>() + size_t(li) * E, n * K, stream);
            // decode replays a captured graph (the count kernel runs, host code does not): write on uncaptured calls
            if (li + 1 == long(cfg.layers)) moe_profile_write();
        }
        const bool split = L.n_cold > 0;
        const bool grouped_gemm = n > NVER && segs.empty() && !wide_mm;
        // decode: the shared expert rides the routed launches as assignment K of every token (appended to the hot
        // experts at load; same rows and kernels as its dense matvec), ids widened to K + 1 per token
        const bool fuse = L.sh_slot >= 0 && !grouped_gemm;
        const int KV = fuse ? K + 1 : K;
        const int * IDS = moe_ids.as<int>();
        if (fuse) {
            tqllm_cuda::moe_extend_ids(moe_ids.as<int>(), split ? L.hot_slot : nullptr, split ? L.cold_slot : nullptr,
                                       split ? moe_hot_ids.as<int>() : moe_xids.as<int>(), split ? moe_cold_ids.as<int>() : nullptr,
                                       K, n, L.sh_slot, stream);
            if (!split) IDS = moe_xids.as<int>();
        } else if (split) {
            tqllm_cuda::moe_split_ids(moe_ids.as<int>(), L.hot_slot, L.cold_slot, moe_hot_ids.as<int>(), moe_cold_ids.as<int>(),
                                      n * K, stream);
        }
        if (grouped_gemm && !split)
            tqllm_cuda::moe_group(moe_ids.as<int>(), moe_counts.as<int>(), moe_groups.as<int>(), E, n, KV, stream);
        if (grouped_gemm && split) {
            tqllm_cuda::moe_group(moe_hot_ids.as<int>(), moe_counts.as<int>(), moe_groups.as<int>(), L.n_hot, n, KV, stream);
            tqllm_cuda::moe_group(moe_cold_ids.as<int>(), moe_counts2.as<int>(), moe_groups2.as<int>(), L.n_cold, n, KV, stream);
        }
        auto one = [&](const Mat & mat, const int * ids, int experts, const Buf & counts, const Buf & groups,
                       const __half * input, bf16 * output, int rows, bool per_expert) {
            if (grouped_gemm && mat.fmt->expert_gemm(mat, input, output, rows, counts.as<int>(),
                                                    groups.as<int>(), n, experts, K, per_expert, stream)) return;
            mat.fmt->expert_matvec(mat, input, output, rows, ids, n, KV, per_expert, stream);
        };
        auto project = [&](const Mat & mat, const Mat & cold, const __half * input, bf16 * output, int rows, bool per_expert) {
            if (!split) { one(mat, IDS, E, moe_counts, moe_groups, input, output, rows, per_expert); return; }
            one(mat, moe_hot_ids.as<int>(), L.n_hot, moe_counts, moe_groups, input, output, rows, per_expert);
            one(cold, moe_cold_ids.as<int>(), L.n_cold, moe_counts2, moe_groups2, input, output, rows, per_expert);
        };
        const auto paired = [](const Mat & g, const Mat & u) { return g.fmt == u.fmt && g.fmt->u4h_id() >= 0; };
        // decode with host experts: the cold set (PCIe reads) runs on a side branch beside the hot set. Both write
        // disjoint assignments (the u4h-family expert kernels leave id -1 untouched), joined before each consumer.
        const bool overlap = split && !grouped_gemm && paired(L.ex_gate, L.ex_up) && paired(L.cx_gate, L.cx_up) &&
                             L.ex_down.fmt->u4h_id() >= 0 && L.cx_down.fmt->u4h_id() >= 0;
        if (overlap) {
            if (!moe_side) {
                ck(cudaStreamCreateWithFlags(&moe_side, cudaStreamNonBlocking));
                ck(cudaEventCreateWithFlags(&moe_fork, cudaEventDisableTiming));
                ck(cudaEventCreateWithFlags(&moe_join, cudaEventDisableTiming));
            }
            auto fork = [&] { ck(cudaEventRecord(moe_fork, stream)); ck(cudaStreamWaitEvent(moe_side, moe_fork, 0)); };
            auto join = [&] { ck(cudaEventRecord(moe_join, moe_side)); ck(cudaStreamWaitEvent(stream, moe_join, 0)); };
            fork();
            Mat sg = L.cx_gate, su = L.cx_up, sd = L.cx_down;
            const int * cids = moe_cold_ids.as<int>();
            const int pool = moe_cache_of.empty() ? -1 : moe_cache_of[size_t(&L - layer.data())];
            if (pool >= 0 && n * KV <= moe_stage_slots) {  // cached: copy only the misses, read from the cache slots
                auto & c = moe_cache[size_t(pool)];
                tqllm_cuda::moe_cache_lookup(moe_cold_ids.as<int>(), n * KV, moe_cold_base[size_t(&L - layer.data())],
                                             moe_cache_table.as<int>(), {c.key.as<int>(), c.ref.as<int>(), c.hand.as<int>(), c.slots},
                                             moe_cache_slot.as<int>(), moe_cache_fill.as<int>(), moe_side);
                tqllm_cuda::MoeStageSpan sp[tqllm_cuda::kMoeStageSpans];
                int ns = 0;
                Mat * sm[3] = {&sg, &su, &sd};
                for (int j = 0; j < 3; ++j) {
                    Mat & m = *sm[j];
                    const long long per = m.rows / L.n_cold;
                    sp[ns++] = {m.p, c.buf[2 * j].p, c.bytes[2 * j]};
                    m.p = c.buf[2 * j].p;
                    if (m.aux) { sp[ns++] = {m.aux, c.buf[2 * j + 1].p, c.bytes[2 * j + 1]}; m.aux = c.buf[2 * j + 1].p; }
                    m.rows = int(per) * c.slots;
                }
                tqllm_cuda::moe_cache_fill(sp, ns, moe_cache_fill.as<int>(), moe_cache_slot.as<int>(), n * KV, moe_side);
                cids = moe_cache_slot.as<int>();
            } else if (moe_stage_slots && n * KV <= moe_stage_slots) {
                tqllm_cuda::MoeStageSpan sp[tqllm_cuda::kMoeStageSpans];
                int ns = 0;
                Mat * sm[3] = {&sg, &su, &sd};
                for (int j = 0; j < 3; ++j) {
                    Mat & m = *sm[j];
                    const long long per = m.rows / L.n_cold;
                    sp[ns++] = {m.p, moe_stage_buf[2 * j].p, per * (long long)m.fmt->row_bytes(m.cols)};
                    m.p = moe_stage_buf[2 * j].p;
                    if (m.aux) {
                        sp[ns++] = {m.aux, moe_stage_buf[2 * j + 1].p, per * (long long)m.fmt->aux_row_bytes(m.cols)};
                        m.aux = moe_stage_buf[2 * j + 1].p;
                    }
                    m.rows = int(per) * moe_stage_slots;
                }
                tqllm_cuda::moe_stage(sp, ns, moe_cold_ids.as<int>(), moe_stage_ids.as<int>(), n * KV, moe_side);
                cids = moe_stage_ids.as<int>();
            }
            sg.fmt->expert_matvec2(sg, su, XH, moe_gate.as<bf16>(), moe_up.as<bf16>(), FE, cids, n, KV, moe_side);
            L.ex_gate.fmt->expert_matvec2(L.ex_gate, L.ex_up, XH, moe_gate.as<bf16>(), moe_up.as<bf16>(), FE,
                                          moe_hot_ids.as<int>(), n, KV, stream);
            join();
            tqllm_cuda::moe_swiglu(moe_gate.as<bf16>(), moe_up.as<bf16>(), moe_act.as<__half>(), n * KV * FE, stream);
            fork();
            sd.fmt->expert_matvec(sd, moe_act.as<__half>(), moe_out.as<bf16>(), H, cids, n, KV, true, moe_side);
            L.ex_down.fmt->expert_matvec(L.ex_down, moe_act.as<__half>(), moe_out.as<bf16>(), H, moe_hot_ids.as<int>(), n, KV,
                                         true, stream);
            join();
        } else if (!grouped_gemm && !split && paired(L.ex_gate, L.ex_up)) {
            L.ex_gate.fmt->expert_matvec2(L.ex_gate, L.ex_up, XH, moe_gate.as<bf16>(), moe_up.as<bf16>(), FE,
                                          IDS, n, KV, stream);
        } else {
            project(L.ex_gate, L.cx_gate, XH, moe_gate.as<bf16>(), FE, false);
            project(L.ex_up, L.cx_up, XH, moe_up.as<bf16>(), FE, false);
        }
        if (!overlap) {
            tqllm_cuda::moe_swiglu(moe_gate.as<bf16>(), moe_up.as<bf16>(), moe_act.as<__half>(), n * KV * FE, stream);
            project(L.ex_down, L.cx_down, moe_act.as<__half>(), moe_out.as<bf16>(), H, true);
        }
        if (fuse) {
            tqllm_cuda::moe_sum_strided(moe_out.as<bf16>(), KV, moe_out.as<bf16>() + size_t(K) * H, KV * H, moe_weights.as<float>(),
                                        moe_shared_weights.as<float>(), X, H, K, n, stream);
        } else {
            mm_h(L.sh_gu, T, n);
            tqllm_cuda::swiglu_multi(T, n, FS, ACT, stream);
            rot_h(ACT, XH, FS, n);
            mm_h(L.sh_down, moe_shared.as<bf16>(), n);
            tqllm_cuda::moe_sum(moe_out.as<bf16>(), moe_shared.as<bf16>(), moe_weights.as<float>(),
                                moe_shared_weights.as<float>(), X, H, K, n, stream);
        }
        tqllm_cuda::add_sumsq(X, M, v_px.as<float>(), n * H, stream);
        return;
    }
    if (!L.w_gu.p) throw std::runtime_error("multi-token path needs fused gate/up weights");
    if (basis_ == Basis::Tq4Rot128 && prefill_gemm(L.w_gu, n) && L.w_gu.fmt->basis() == basis_) {
        gemm_rows(L.w_gu, nullptr, n, L.w_gu.rows);  // FP16 gate / up straight into the fused SwiGLU + rotation
        tqllm_cuda::tq4_swiglu_rot_from_h(gemm_c.as<__half>(), n, F, XH, stream);
        mm_h(L.w_down, X, n);
        tqllm_cuda::add_sumsq(X, M, v_px.as<float>(), n * H, stream);
        return;
    }
    st("ffn_resid+norm_rot");
    mm_h(L.w_gu, T, n);
    st("ffn_gu_matvec");
    if (basis_ == Basis::Tq3Rot32) {
        tqllm_cuda::swiglu_rot_h(T, n, F, XH, stream);
    } else if (basis_ == Basis::Tq4Rot128) {
        tqllm_cuda::tq4_swiglu_rot_h(T, n, F, XH, stream);
    } else {
        tqllm_cuda::swiglu_multi(T, n, F, ACT, stream);
        rot_h(ACT, XH, F, n);
    }
    st("ffn_swiglu_rot");
    mm_h(L.w_down, X, n);
    st("ffn_down_matvec");
    tqllm_cuda::add_sumsq(X, M, v_px.as<float>(), n * H, stream);
    st("ffn_resid_out");
}

void Qwen35Engine::head_argmax(const float * norm_w, int n, int col0) {
    const int H = int(cfg.hidden);
    auto * NP = v_np.as<bf16>();
    uint32_t * tok_out = v_tok.as<uint32_t>() + col0;
    float * lg_out = s_lg.as<float>() + size_t(col0) * 16;
    uint32_t * id_out = s_id.as<uint32_t>() + size_t(col0) * 16;
    tqllm_cuda::rms_f(v_x.as<bf16>() + size_t(col0) * H, H, norm_w, NP, H, H, n, cfg.eps, stream);
    if (lm_head.fmt) {  // LM head of a registered weight format (W4A16, TQ4, ...)
        head_mv(NP, n);
        tqllm_cuda::argmax_cols(v_logits.as<float>(), int(cfg.vocab), n, tok_out, v_amax.p, stream);
        if (want_topk())
            tqllm_cuda::topk16_big(v_logits.as<float>(), lm_head.rows, n, lg_out, id_out, s_scr.p, stream);
        return;
    }
    if (!k_head()) throw std::runtime_error("multi-token path needs a Q6_K / Q5_K LM head");
    const bool two_stage = !full_logits;
    if (two_stage && head4.p && has_head8() && (n <= NVER || !segs.empty())) {  // same head path for decode and verify
        tqllm_cuda::bf16_to_h(NP, v_xh.as<__half>(), n * H, stream);
        for (int c0 = 0; c0 < n; c0 += 32)  // batched verify: every slot's columns in one pass per 32
            tqllm_cuda::u4h_matvec(head4.as<uint8_t>(), head4s.as<__half2>(), v_xh.as<__half>() + size_t(c0) * H,
                                   v_logits.as<float>() + size_t(c0) * lm_head.rows, lm_head.rows, H,
                                   std::min(32, n - c0), stream);
        tqllm_cuda::topk16_big(v_logits.as<float>(), lm_head.rows, n, h2_vals.as<float>(), h2_idx.as<uint32_t>(),
                               h2_scr.p, stream);
        tqllm_cuda::i8h_rescore_argmax(w8(), s8(), v_xh.as<__half>(),
                                       h2_idx.as<uint32_t>(), H, n, tok_out, stream,
                                       want_topk() ? lg_out : nullptr);
        if (want_topk())
            ck(cudaMemcpyAsync(id_out, h2_idx.p, size_t(n) * 16 * 4, cudaMemcpyDeviceToDevice, stream));
        return;
    }
    if (has_head8()) {  // int8 + FP16-scale copy of the Q6_K head on FP16 tensor cores
        tqllm_cuda::bf16_to_h(NP, v_xh.as<__half>(), n * H, stream);
        tqllm_cuda::i8h_matvec(w8(), s8(), v_xh.as<__half>(), v_logits.as<float>(),
                               lm_head.rows, H, n, stream);
    } else {
        tqllm_cuda::bf16_to_f32(NP, v_xf.as<float>(), n * H, stream);
        tqllm_cuda::q6k_matvec(lm_head.p, v_xf.as<float>(), v_logits.as<float>(), lm_head.rows, H, n, stream);
    }
    tqllm_cuda::argmax_cols(v_logits.as<float>(), int(cfg.vocab), n, tok_out, v_amax.p, stream);
    if (want_topk())
        tqllm_cuda::topk16_big(v_logits.as<float>(), lm_head.rows, n, lg_out, id_out, s_scr.p, stream);
}

void Qwen35Engine::embed_multi(const uint32_t * dtok, int n, bf16 * Y) {
    const int H = int(cfg.hidden);
    if (embed.type == GGML_TQ3_4S) tqllm_cuda::embed_multi_tq3(embed.p, dtok, n, Y, H, stream);
    else if (embed.type == GGML_Q6_K) tqllm_cuda::embed_multi_q6k(embed.p, dtok, n, Y, H, stream);
    else if (embed.type == GGML_Q8_0) tqllm_cuda::embed_multi_q8_0(embed.p, dtok, n, Y, H, stream);
    else if (embed.type == GGML_Q4_K) tqllm_cuda::embed_multi_q4k(embed.p, dtok, n, Y, H, stream);
    else if (embed.type == GGML_Q3_K) tqllm_cuda::embed_multi_q3k(embed.p, dtok, n, Y, H, stream);
    else if (embed.type == GGML_IQ3_S) tqllm_cuda::embed_multi_iq3s(embed.p, dtok, n, Y, H, stream);
    else if (embed.type == GGML_IQ4_XS) tqllm_cuda::embed_multi_iq4xs(embed.p, dtok, n, Y, H, stream);
    else if (embed.type == GGML_Q5_K) tqllm_cuda::embed_multi_q5k(embed.p, dtok, n, Y, H, stream);
    else if (embed.type == GGML_BF16) tqllm_cuda::embed_multi_bf16(embed.p, dtok, n, Y, H, stream);
    else throw std::runtime_error("multi-token path needs a TQ3_4S, Q6_K, Q8_0, Q4_K, Q3_K, Q5_K, IQ3_S, IQ4_XS or BF16 embedding table");
}

void Qwen35Engine::capture_feature(uint32_t l, const bf16 * x, bf16 * feats, int n) {
    if (!dfl.present) return;
    const size_t H = cfg.hidden, nf = dfl.target_ids.size();
    for (size_t k = 0; k < nf; ++k)
        if (uint32_t(dfl.target_ids[k]) == l)
            ck(cudaMemcpy2DAsync(feats + k * H, nf * H * 2, x, H * 2, H * 2, size_t(n), cudaMemcpyDeviceToDevice,
                                 stream));
}

Mat Qwen35Engine::q4k_to_u4h(const GgufFile & f, const std::vector<std::string> & names, int cols) {
    std::vector<uint8_t> w4;
    std::vector<__half2> sm;
    int rows = 0;
    for (const auto & name : names) {
        const auto & t = gt(f, name);
        const auto raw = f.read(t);
        const int64_t nb = int64_t(t.ne[0] * t.ne[1]) / 256;
        const size_t o4 = w4.size(), osm = sm.size();
        w4.resize(o4 + nb * 128);
        sm.resize(osm + nb * 8);
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 144;
            uint16_t hd, hm;
            std::memcpy(&hd, blk, 2);
            std::memcpy(&hm, blk + 2, 2);
            const float d = half_to_float(hd), mn = half_to_float(hm);
            uint8_t codes[256];
            for (int c = 0; c < 4; ++c)
                for (int l = 0; l < 32; ++l) {
                    codes[c * 64 + l] = blk[16 + c * 32 + l] & 0xF;
                    codes[c * 64 + 32 + l] = blk[16 + c * 32 + l] >> 4;
                }
            for (int j = 0; j < 8; ++j) {
                uint8_t sc, m;
                scale_min_k4(j, blk + 4, sc, m);
                sm[osm + b * 8 + j] = __floats2half2_rn(d * sc, -mn * m);
            }
            uint8_t * out = w4.data() + o4 + b * 128;
            std::memset(out, 0, 128);
            for (int e = 0; e < 256; ++e) {  // group of 16: element 4u + j -> byte (u / 2) * 4 + j, nibble u % 2
                const int grp = e / 16, ei = e % 16, u = ei / 4, j = ei % 4;
                out[grp * 8 + (u / 2) * 4 + j] |= codes[e] << ((u % 2) * 4);
            }
        }
        rows += int(t.ne[1]);
    }
    Buf bw(w4.size()), bs(sm.size() * 4);
    ck(cudaMemcpyAsync(bw.p, w4.data(), w4.size(), cudaMemcpyHostToDevice, stream));
    ck(cudaMemcpyAsync(bs.p, sm.data(), sm.size() * 4, cudaMemcpyHostToDevice, stream));
    sync_ck();
    Mat m{bw.p, TYPE_U4H, rows, cols, bs.p, weight_format(TYPE_U4H)};
    tqllm_cuda::u4h_tune(bw.as<uint8_t>(), bs.as<__half2>(), rows, cols, stream);
    tensors.push_back(std::move(bw));
    tensors.push_back(std::move(bs));
    return m;
}

Mat Qwen35Engine::to_tq3(const GgufFile & f, const std::vector<std::string> & names, int cols) {
    {  // TQ4 / TQ3 drafter tensors (tools/dflash2_to_tq4.py): row-concatenated blocks through that weight format
        const uint32_t ty = gt(f, names[0]).type;
        bool all_tq4 = ty == GGML_TQ4 || ty == GGML_TQ3;
        for (const auto & name : names) {
            const auto & t = gt(f, name);
            if (all_tq4 && t.type != ty && (t.type == GGML_TQ4 || t.type == GGML_TQ3))
                throw std::runtime_error("DFlash2 drafter: fused tensors must share one type (TQ4 / TQ3): " + name);
            all_tq4 = all_tq4 && t.type == ty && t.ne.size() == 2 && t.ne[0] == uint64_t(cols);
        }
        if (all_tq4) {
            std::vector<char> host;
            int rows = 0;
            for (const auto & name : names) {
                const auto & t = gt(f, name);
                const auto raw = f.read(t);
                host.insert(host.end(), raw.begin(), raw.end());
                rows += int(t.ne[1]);
            }
            return weight_format(ty)->load(host, rows, cols, uploader(), stream);  // not adopt(): drafter only
        }
    }
    bool all_q4k = true;
    for (const auto & name : names) {
        const auto & t = gt(f, name);
        all_q4k = all_q4k && t.type == GGML_Q4_K && t.ne.size() == 2 && t.ne[0] == uint64_t(cols) && cols % 256 == 0;
    }
    if (all_q4k) return q4k_to_u4h(f, names, cols);
    std::vector<float> all;
    int rows = 0;
    for (const auto & name : names) {
        const auto & t = gt(f, name);
        if (t.ne.size() != 2 || t.ne[0] != uint64_t(cols)) throw std::runtime_error("unexpected shape for " + name);
        const auto raw = f.read(t);
        const size_t off = all.size();
        all.resize(off + size_t(t.ne[0]) * t.ne[1]);
        dequantize_host(t.type, raw.data(), all.data() + off, int64_t(t.ne[0] * t.ne[1]));
        rows += int(t.ne[1]);
    }
    // 4-bit (scale, min) per 32 via int8 + FP16 scale per 16 (half the bytes of int8; same or higher acceptance on
    // .77; 3-bit TQ3 cost acceptance)
    Buf tmp(all.size() * 4);
    ck(cudaMemcpyAsync(tmp.p, all.data(), all.size() * 4, cudaMemcpyHostToDevice, stream));
    Buf w8(all.size()), sc(all.size() / 16 * 2);
    tqllm_cuda::i8h_quantize(tmp.as<float>(), w8.as<int8_t>(), sc.as<__half>(), int64_t(all.size()), stream);
    Buf bw(all.size() / 2), bs(all.size() / 32 * 4);
    tqllm_cuda::u4h_from_i8h(w8.as<int8_t>(), sc.as<__half>(), bw.as<uint8_t>(), bs.as<__half2>(), int64_t(all.size()),
                             stream);
    sync_ck();
    Mat m{bw.p, TYPE_U4H, rows, cols, bs.p, weight_format(TYPE_U4H)};
    tqllm_cuda::u4h_tune(bw.as<uint8_t>(), bs.as<__half2>(), rows, cols, stream);
    tensors.push_back(std::move(bw));
    tensors.push_back(std::move(bs));
    return m;
}

const float * Qwen35Engine::f32_of(const GgufFile & f, const std::string & name, uint64_t n) {
    const auto & t = gt(f, name);
    std::vector<float> v(n);
    const auto raw = f.read(t);
    dequantize_host(t.type, raw.data(), v.data(), int64_t(n));
    return put_f32(v);
}

const bf16 * Qwen35Engine::bf16_table(const GgufFile & f, const std::string & name, uint64_t n) {
    const auto & t = gt(f, name);
    std::vector<float> v(n);
    const auto raw = f.read(t);
    dequantize_host(t.type, raw.data(), v.data(), int64_t(n));
    std::vector<bf16> b(n);
    for (uint64_t i = 0; i < n; ++i) b[i] = __float2bfloat16_rn(v[i]);
    return static_cast<const bf16 *>(put(b.data(), b.size() * 2));
}

const bf16 * Qwen35Engine::vocab_table(const GgufFile & f, const std::string & name, int rank) {
    const auto & t = gt(f, name);
    if (int(t.ne[0]) != rank || t.ne[1] > cfg.vocab) throw std::runtime_error(name + ": unexpected shape");
    const uint64_t n = uint64_t(t.ne[0]) * t.ne[1];
    std::vector<float> v(n);
    const auto raw = f.read(t);
    dequantize_host(t.type, raw.data(), v.data(), int64_t(n));
    std::vector<bf16> b(uint64_t(cfg.vocab) * rank, __float2bfloat16_rn(0.f));
    for (uint64_t i = 0; i < n; ++i) b[i] = __float2bfloat16_rn(v[i]);
    return static_cast<const bf16 *>(put(b.data(), b.size() * 2));
}

void Qwen35Engine::dsub_matvec(const __half * x, float * y, int n) {
    if (dsub_in_head4)
        tqllm_cuda::u4h_matvec_ids(head4.as<uint8_t>(), head4s.as<__half2>(), dsub_ids.as<uint32_t>(), x, y, dsub_rows,
                                   int(cfg.hidden), n, stream);
    else
        tqllm_cuda::u4h_matvec(dsub_w.as<uint8_t>(), dsub_s.as<__half2>(), x, y, dsub_rows, int(cfg.hidden), n, stream);
}

const void * Qwen35Engine::to_mapped_host(const void * dev, size_t bytes) {
    void * h = nullptr;
    ck(cudaHostAlloc(&h, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    ck(cudaMemcpy(h, dev, bytes, cudaMemcpyDeviceToHost));
    host_blocks.push_back(h);
    void * d = nullptr;
    ck(cudaHostGetDevicePointer(&d, h, 0));
    return d;
}

const void * Qwen35Engine::host_put(const void * host, size_t bytes) {
    void * h = nullptr;
    ck(cudaHostAlloc(&h, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memcpy(h, host, bytes);
    host_blocks.push_back(h);
    void * d = nullptr;
    ck(cudaHostGetDevicePointer(&d, h, 0));
    return d;
}

void Qwen35Engine::head8_host_alloc() {  // the int8 head copy and its scales in mapped host memory (exact re-score rows)
    const size_t bw = size_t(lm_head.rows) * cfg.hidden, bs = bw / 16 * 2;
    ck(cudaHostAlloc(&h8_host, bw + bs, cudaHostAllocMapped | cudaHostAllocPortable));
    void * d = nullptr;
    ck(cudaHostGetDevicePointer(&d, h8_host, 0));
    h8_dev_w = static_cast<const int8_t *>(d);
    h8_dev_s = reinterpret_cast<const __half *>(static_cast<const char *>(d) + bw);
    placed_head_b = bw + bs;
    std::fprintf(stderr, "tqserve: LM head int8 copy in host memory (%.2f GB of VRAM freed)\n", double(bw + bs) / 1e9);
}

void Qwen35Engine::mmx(const Mat & m, const bf16 * in, bf16 * out, int n) {
    if (m.fmt) {  // registered weight format: input prepared in the format's basis
        prep_input(m.fmt->basis(), in, v_xh.as<__half>(), m.cols, n);
        mm_h(m, out, n, m.fmt->basis());
        return;
    }
    if (m.type == GGML_BF16) {  // [rows][cols] x [n][cols] -> [n][rows]
        const float alpha = 1.f, beta = 0.f;
        cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, m.rows, n, m.cols, &alpha, m.p, CUDA_R_16BF, m.cols, in,
                        CUDA_R_16BF, m.cols, &beta, out, CUDA_R_16BF, m.rows, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return;
    }
    throw std::runtime_error("drafter matrix without a weight format");
}

void Qwen35Engine::inj_mm(const Mat & m, const bf16 * in, bf16 * out, int n) {
    if (n > DFlash::injBatch) throw std::runtime_error("dflash inject: batch wider than the scratch");
    const int mc = m.fmt->max_batch_cols();
    for (int c0 = 0; c0 < n; c0 += mc) {
        const int nc = std::min(mc, n - c0);
        prep_input(m.fmt->basis(), in + size_t(c0) * m.cols, v_xh.as<__half>(), m.cols, nc);
        m.fmt->matvec(m, v_xh.as<__half>(), out + size_t(c0) * m.rows, nc, stream);
    }
}

void Qwen35Engine::dspark_forward(uint32_t pos) {
    const int H = int(cfg.hidden), NB = dfl.nb, qd = dfl.heads * dfl.hd, kd = dfl.kvheads * dfl.hd;
    const int qkv = qd + 2 * kd, F = dfl.ffn;
    if (!ds_t.p) {
        ds_t = Buf(size_t(NB) * std::max(qkv, 2 * F) * 2);
        ds_act = Buf(size_t(NB) * F * 2);
        ds_q = Buf(size_t(NB) * qd * 2);
        ds_o = Buf(size_t(NB) * qd * 2);
    }
    auto *X = d_x.as<bf16>(), *R = d_r.as<bf16>(), *A = d_a.as<bf16>(), *Bb = d_b.as<bf16>();
    auto *T = ds_t.as<bf16>(), *ACT = ds_act.as<bf16>(), *Q = ds_q.as<bf16>(), *O = ds_o.as<bf16>();
    embed_multi(d_tok.as<uint32_t>(), NB, X);
    uint32_t bucket = 256;
    while (bucket < pos + NB) bucket *= 2;
    bucket = std::min(bucket, context_);
    for (size_t l = 0; l < dfl.L.size(); ++l) {
        const auto & L = dfl.L[l];
        ck(cudaMemcpyAsync(R, X, size_t(NB) * H * 2, cudaMemcpyDeviceToDevice, stream));
        tqllm_cuda::rms_f_multi(X, H, 0, L.attn_norm, A, H, 0, H, 1, NB, cfg.eps, stream);
        mmx(L.w_qkv, A, T, NB);
        bf16 * K0 = T + qd;
        tqllm_cuda::rms_f_multi(T, qkv, dfl.hd, L.q_norm, Q, qd, dfl.hd, dfl.hd, dfl.heads, NB, cfg.eps, stream);
        tqllm_cuda::rms_f_multi(K0, qkv, dfl.hd, L.k_norm, K0, qkv, dfl.hd, dfl.hd, dfl.kvheads, NB, cfg.eps, stream);
        tqllm_cuda::rope_partial_multi(Q, qd, NB, dfl.heads, dfl.hd, dfl.hd, d_pos.as<int>(), dfl.theta, stream);
        tqllm_cuda::rope_partial_multi(K0, qkv, NB, dfl.kvheads, dfl.hd, dfl.hd, d_pos.as<int>(), dfl.theta, stream);
        tqllm_cuda::cache_write_multi(K0, K0 + kd, qkv, NB, dfl_kc[l].as<bf16>(), dfl_vc[l].as<bf16>(), d_pos.as<int>(),
                                      dfl.kvheads, dfl.hd, stream, 0);
        tqllm_cuda::attention_gen256(Q, dfl_kc[l].as<bf16>(), dfl_vc[l].as<bf16>(), O, dfl_part.as<float>(),
                                     d_pos.as<int>(), NB, int(bucket), dfl.heads, dfl.kvheads,
                                     1.f / std::sqrt(float(dfl.hd)), false, stream);
        mmx(L.wo, O, Bb, NB);
        tqllm_cuda::residual(Bb, R, NB * H, stream);  // Bb = ffn_inp
        tqllm_cuda::rms_f_multi(Bb, H, 0, L.ffn_norm, A, H, 0, H, 1, NB, cfg.eps, stream);
        mmx(L.w_gu, A, T, NB);
        tqllm_cuda::swiglu_multi(T, NB, F, ACT, stream);
        mmx(L.down, ACT, X, NB);
        tqllm_cuda::residual(X, Bb, NB * H, stream);
    }
    auto * TE = d_te.as<bf16>();
    tqllm_cuda::rms_f_multi(X, H, 0, dfl.out_norm, TE, H, 0, H, 1, NB, cfg.eps, stream);
    const int head_rows = lm_head.rows;
    tqllm_cuda::bf16_to_h(TE, v_xh.as<__half>(), NB * H, stream);
    if (head4.p)
        tqllm_cuda::u4h_matvec(head4.as<uint8_t>(), head4s.as<__half2>(), v_xh.as<__half>(), v_logits.as<float>(),
                               head_rows, H, NB, stream);
    else if (lm_head.fmt)
        head_mv(TE, NB);
    else
        tqllm_cuda::i8h_matvec(w8(), s8(), v_xh.as<__half>(), v_logits.as<float>(), head_rows, H, NB, stream);
    // block row r predicts position pos + 1 + r: lattice row r + 1 (row 0 is the anchor)
    tqllm_cuda::topk16_big(v_logits.as<float>(), head_rows, NB, d_vals.as<float>() + 16, d_idx.as<uint32_t>() + 16,
                           d_topk.p, stream);
    tqllm_cuda::dflash_lattice(d_idx.as<uint32_t>(), d_vals.as<float>(), d_gate.as<bf16>(), dfl.rank, dfl.sel_next,
                               dfl.sel_prev, 0, dfl.block, dfl.rank, d_score.as<float>(), stream, d_tok.as<uint32_t>());
}

void Qwen35Engine::dec_progress() {
    static const bool on = tqllm::debug_on("decode");
    if (!on) return;
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - dec_log_tp).count() < 2.5) return;
    dec_log_tp = now;
    const double total = std::chrono::duration<double>(now - gen.decode_start).count();
    const uint32_t n = uint32_t(gen.out.size());
    std::fprintf(stderr, "decode progress %u tokens (%.0f tok/s), drafts accepted %llu of %llu (%.0f%%)\n", n,
                 total > 0 ? n / total : 0.0, (unsigned long long) gen.accepted, (unsigned long long) gen.drafted,
                 gen.drafted ? 100.0 * gen.accepted / gen.drafted : 0.0);
}

void Qwen35Engine::set_prompt_images(const std::vector<PromptImage> & ims) {
    if (!cfg.msec[0]) throw std::runtime_error("model has no M-RoPE sections (rope.dimension_sections): no image input");
    const int H = int(cfg.hidden);
    size_t rows = 0;
    for (const auto & im : ims) rows += size_t(im.nx) * size_t(im.ny);
    if (rows > mm_img_rows) {
        mm_img = Buf(rows * size_t(H) * 2);
        mm_img_rows = rows;
    }
    if (!mm_tab.p) mm_tab = Buf(size_t(NBUF) * 4 * sizeof(int));
    mm_pending.clear();
    size_t r0 = 0;
    for (const auto & im : ims) {
        const size_t nt = size_t(im.nx) * size_t(im.ny);
        tqllm_cuda::f32_to_bf16(im.emb, mm_img.as<bf16>() + r0 * size_t(H), int64_t(nt) * H, stream);
        mm_pending.push_back({im.at, im.nx, im.ny, r0});
        r0 += nt;
    }
    sync_ck();
}

bool Qwen35Engine::prefill_state(const std::vector<uint32_t> & tokens, uint32_t pos0) {
    const auto P = uint32_t(tokens.size());
    if (P == 0) return true;
    if (pos0 + P + 1 > context_) throw std::runtime_error("prefill_state: sequence exceeds the context");
    pc_drop();
    const int CH = NPF;
    const size_t chunks = (P + CH - 1) / CH;
    HostBuf staged_tokens(size_t(P) * sizeof(uint32_t)), staged_pos(chunks * 2 * sizeof(int));
    std::memcpy(staged_tokens.p, tokens.data(), size_t(P) * sizeof(uint32_t));
    auto * positions = static_cast<int *>(staged_pos.p);
    for (size_t i = 0; i < chunks; ++i) positions[2 * i] = int(pos0 + i * CH), positions[2 * i + 1] = int(pos0 + i * CH) + 1;
    for (uint32_t t = 0; t < P; t += uint32_t(CH)) {
        const int m = int(std::min<uint32_t>(uint32_t(CH), P - t));
        prefill_final = false;
        verify_async(static_cast<const uint32_t *>(staged_tokens.p) + t, positions + 2 * (t / CH), m, pos0 + t, nullptr, false);
        commit(m);
        sync_ck();
    }
    prefill_final = true;
    return true;
}

void Qwen35Engine::flush_pending() {
    if (!pend_k) return;
    const int lkh = int(cfg.lin_kheads), lvh = int(cfg.lin_vheads), dk = int(cfg.lin_dk), dv = int(cfg.lin_dv);
    const int qkv_dim = 2 * lkh * dk + lvh * dv, zdim = lvh * dv;
    uint32_t lin_i = 0;
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        if (cfg.full[l]) continue;
        const auto & L = layer[l];
        const int rows = mix_rows(L);
        const bf16 * T = v_stash2[pend_sp][lin_i].as<bf16>();
        tqllm_cuda::conv_multi(T, rows, pend_k, conv_slot(lin_i, cur), conv_slot(lin_i, cur + 1), L.conv_w,
                               v_qkvc.as<bf16>(), qkv_dim, stream, d_chain.as<int>());
        const bf16 * B0 = T + qkv_dim + zdim;
        tqllm_cuda::gdn_multi(v_qkvc.as<bf16>(), pend_k, B0 + lvh, B0, rows, L.neg_a, L.dt_bias, rec_slot(lin_i, cur),
                              rec_slot(lin_i, cur + 1), v_core.as<bf16>(), lkh, lvh, dk, dv, cfg.tiled, stream,
                              d_chain.as<int>());
        ++lin_i;
    }
    cur = (cur + 1) % NSLOT;
    set_pending(0);
}

}  // namespace qwen35_detail
}  // namespace tqllm
