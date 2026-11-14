#pragma once
// qwen35 engine internals: device buffers, config readers, layer / MTP / DFlash records. Included by qwen35_engine.hpp.
// Expects that header's includes before it.
namespace tqllm {
namespace qwen35_detail {

using bf16 = __nv_bfloat16;

inline void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

inline void cb(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS call failed: " + std::to_string(int(e)));
}

struct Buf {
    void * p = nullptr;
    Buf() = default;
    explicit Buf(size_t bytes) {
        if (bytes) ck(tqllm_cuda::dev_malloc(&p, bytes));
    }
    ~Buf() {
        if (p) tqllm_cuda::dev_free(p);
    }
    Buf(Buf && o) noexcept : p(std::exchange(o.p, nullptr)) {}
    Buf & operator=(Buf && o) noexcept {
        if (this != &o) {
            if (p) tqllm_cuda::dev_free(p);
            p = std::exchange(o.p, nullptr);
        }
        return *this;
    }
    Buf(const Buf &) = delete;
    Buf & operator=(const Buf &) = delete;
    template <class T> T * as() const { return static_cast<T *>(p); }
};

struct KvRef {  // a view of one layer's KV rows in the KV store's pool (src/kv/kv_store.hpp); the store owns the memory
    void * p = nullptr;
    template <class T> T * as() const { return static_cast<T *>(p); }
};

struct HostBuf {
    void * p = nullptr;
    explicit HostBuf(size_t bytes) { if (bytes) ck(cudaHostAlloc(&p, bytes, cudaHostAllocPortable)); }
    ~HostBuf() { if (p) cudaFreeHost(p); }
    HostBuf(const HostBuf &) = delete;
    HostBuf & operator=(const HostBuf &) = delete;
};

struct Config {
    uint32_t vocab, layers, hidden, ffn, heads, kvheads, head_dim, rot_dim;
    uint32_t lin_kheads, lin_vheads, lin_dk, lin_dv, conv_k, full_interval;
    float eps, theta;
    uint32_t ctx_train = 0;  // context_length (training context; RoPE scaling reaches past it)
    std::string rope_type = "none";  // GGUF rope.scaling.*; CLI options override
    float rope_factor = 1.f;
    uint32_t rope_orig_ctx = 0;
    std::string architecture = "qwen35";
    bool gated_attention = true;
    bool tied;
    bool tiled = false;      // GGUF stores Gated DeltaNet value heads in tiled order
    int msec[3] = {0, 0, 0};  // interleaved M-RoPE sections (t, h, w rotary pairs) for image tokens
    std::vector<bool> full;  // per layer: true = full attention, false = Gated DeltaNet
    // Mixture-of-experts (arch `qwen35moe`). Every block is MoE in that file - there is no
    // dense-vs-MoE layer split - so `moe` alone selects the path and `layers` still counts blocks.
    // Defaults keep the dense qwen35 config untouched: both construction sites value-init
    // `Config c{}` and then assign by name, so these fields are 0/false on the dense path.
    bool moe = false;
    uint32_t expert_count = 0;      // qwen35moe.expert_count           (256)
    uint32_t expert_used = 0;       // qwen35moe.expert_used_count      (top-8)
    uint32_t expert_ff = 0;         // qwen35moe.expert_feed_forward_length (512, per expert)
    uint32_t shared_ff = 0;         // qwen35moe.expert_shared_feed_forward_length (512)
    // A dense qwen35 file declares `qwen35.feed_forward_length`; the MoE file does not - it has
    // only the two expert FFN lengths above. `ffn` therefore means "per-expert FFN" when moe is
    // set, and the shared expert uses `shared_ff`.
};

inline uint32_t u32(const Json & o, const char * k) {
    const Json * v = o.get(k);
    if (!v) throw std::runtime_error(std::string("config.json missing ") + k);
    return uint32_t(v->num);
}

inline Config read_config(const std::filesystem::path & dir) {
    std::ifstream in(dir / "config.json");
    std::stringstream ss;
    ss << in.rdbuf();
    const Json root = parse_json(ss.str());
    const Json * tp = root.get("text_config");
    const Json & t = tp ? *tp : root;
    Config c{};
    c.vocab = u32(t, "vocab_size");
    c.layers = u32(t, "num_hidden_layers");
    c.hidden = u32(t, "hidden_size");
    c.ffn = u32(t, "intermediate_size");
    c.heads = u32(t, "num_attention_heads");
    c.kvheads = u32(t, "num_key_value_heads");
    c.head_dim = u32(t, "head_dim");
    c.lin_kheads = u32(t, "linear_num_key_heads");
    c.lin_vheads = u32(t, "linear_num_value_heads");
    c.lin_dk = u32(t, "linear_key_head_dim");
    c.lin_dv = u32(t, "linear_value_head_dim");
    c.conv_k = u32(t, "linear_conv_kernel_dim");
    c.eps = float(t.get("rms_norm_eps")->num);
    const Json * rp = t.get("rope_parameters");
    if (!rp) throw std::runtime_error("config.json missing rope_parameters");
    c.theta = float(rp->get("rope_theta")->num);
    const Json * prf = rp->get("partial_rotary_factor");
    c.rot_dim = uint32_t(c.head_dim * (prf ? prf->num : 1.0));
    const Json * tie = root.get("tie_word_embeddings");
    c.tied = tie && tie->num != 0;
    const Json * lt = t.get("layer_types");
    if (!lt || lt->arr.size() != c.layers) throw std::runtime_error("config.json layer_types missing or wrong length");
    for (const auto & x : lt->arr) c.full.push_back(x.str == "full_attention");
    if (c.conv_k != 4) throw std::runtime_error("Gated DeltaNet conv kernel must be 4");
    if (c.lin_vheads % c.lin_kheads) throw std::runtime_error("linear value heads must be a multiple of key heads");
    return c;
}

// A weight matrix [rows][cols] in one of the supported storage types.
constexpr uint32_t TYPE_I8H = 1000;  // int8 + FP16 scale per 16 (engine-internal)
constexpr uint32_t TYPE_U4H = 1001;  // 4-bit + FP16 (scale, min) per 32, from Q4_K (engine-internal)
constexpr uint32_t TYPE_TQ4 = 1003;  // TQ4 (codes [rows][cols / 2] natural nibbles, FP16 scale per 128), rotated inputs
constexpr uint32_t TYPE_W4S = 1002;  // symmetric 4-bit (u4h codes) + FP16 scale per 128, from W4A16 Q4_0 (engine-internal)


struct Layer {
    const float *in_norm, *post_norm;  // RMSNorm weights, 1 + w baked in
    Mat w_gate, w_up, w_down;
    // full attention
    Mat wq, wk, wv, wo;
    const float *q_norm, *k_norm;
    // Gated DeltaNet
    Mat w_qkv, w_z, w_a, w_b, w_out;
    const float *conv_w, *neg_a, *dt_bias, *gnorm_w;
    // GGUF path: matrices sharing an input are concatenated row-wise at load and run as one matvec.
    // w_mix = [q|k|v] (attention) or [qkv|z|b|a] (Gated DeltaNet); w_gu = [gate|up].
    Mat w_mix, w_gu;
    Mat w_ba;  // W4A16: [b|a] kept BF16 (w_mix = [qkv|z] u4h), interleaved into the mix rows after the matmuls
    // qwen35moe: router [E][H]; routed experts stacked per projection (expert e = rows [e*ff, (e+1)*ff)); shared expert
    Mat ex_gate, ex_up, ex_down, sh_gu, sh_down;
    int sh_slot = -1;  // decode: the shared expert's index in ex_gate / ex_up / ex_down (-1: dense shared path)
    // --moe-host-mb: cx_* hold this layer's least-routed experts in pinned host memory and ex_* the rest (VRAM);
    // hot_slot / cold_slot [E] map an expert to its row block in either set (-1 in the other)
    Mat cx_gate, cx_up, cx_down;
    const int * hot_slot = nullptr;
    const int * cold_slot = nullptr;
    int n_hot = 0, n_cold = 0;
    const float * router = nullptr;
    const float * sh_gate = nullptr;  // ffn_gate_inp_shexp [H]: sigmoid gate of the shared expert output
};

struct MtpHead {
    bool present = false;
    Mat eh_proj;
    Mat draft_head;                  // optional: LM-head rows of a draft vocabulary (e.g. 40,960 frequent tokens)
    std::vector<uint32_t> draft_ids;  // row -> token id
    const float *enorm = nullptr, *hnorm = nullptr, *shared_norm = nullptr;
    Layer layer;
};

struct DFlashLayer {
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *ffn_norm = nullptr;
    const float *attn_conv_base = nullptr, *ffn_conv_base = nullptr;  // [2 sides][2 taps][C]
    Mat dspine_read, dspine_write;
    Mat conv_attn, conv_ffn, w_qkv, wo, w_gu, down;                  // TQ3_4S (re-quantised at load)
};

// DFlash2 block-diffusion drafter (z-lab), weights re-quantised to TQ3_4S at load (drafter precision only
// affects acceptance; the target verifies every token).
struct DFlash {
    bool present = false;
    // DSpark (DFlash + Markov head, anchor-first): nb tokens [anchor, MASK x (nb - 1)] predict positions pos + 1 ..
    // pos + nb, laid out as lattice rows 1 .. nb (block = nb + 1); sel_prev / sel_next = markov_w1 / markov_w2, gate 1
    bool dspark = false, dspine = false;
    spec::DSpineConfig spine;
    Mat spine_packed, spine_pred;
    const bf16 *spine_codebook = nullptr;
    const float *spine_bias = nullptr;
    int nb = 0, ffn = 0;
    float theta = 0.f;
    int heads = 0, kvheads = 0, hd = 0, block = 0, group = 0, window = 0, topk = 0, rank = 0, ring = 0;
    int retr = 0, recent = 0;  // retrieval window: retr chunks of 16 far positions + `recent` latest positions
    uint32_t mask_id = 0;
    std::vector<int> target_ids;  // target layers whose *input* hidden states feed the drafter
    static constexpr int injBatch = 32;  // tokens per inject matvec batch (the u4h NCOL cap); the inject scratch is sized to it
    Mat fc, sel_hidden, w_kv_all;  // w_kv_all: [layer][k | v] rows of every layer, one matvec for injection
    const float *enc_norm = nullptr, *out_norm = nullptr;
    const __nv_bfloat16 *sel_prev = nullptr, *sel_next = nullptr;  // [vocab][rank]
    std::vector<DFlashLayer> L;
};

// Arch strings this engine can parse. `qwen35` is the dense hybrid; `qwen35moe` is the same
// hybrid with a mixture-of-experts FFN in every block. Anything else must be rejected here: a
// silently mis-parsed arch produces a model that loads, runs, and emits fluent wrong text.
inline bool arch_supported(const std::string & a) { return a == "qwen35" || a == "qwen35moe" || a == "qwen3"; }

inline Config config_from_gguf(const GgufFile & f) {
    // The metadata namespace comes from the file, not from a hardcoded prefix. The dense and MoE
    // variants use different ones (`qwen35.` vs `qwen35moe.`), and the MoE file contains no
    // `qwen35.` key at all - reading it with the dense prefix failed as
    // "GGUF missing key qwen35.block_count", which named the wrong key rather than the real cause.
    std::string arch;
    if (f.has("general.architecture")) arch = f.str("general.architecture");
    if (arch.empty() || !arch_supported(arch))
        throw std::runtime_error("unsupported GGUF architecture '" + arch +
                                 "' (this engine parses qwen3, qwen35 and qwen35moe)");
    const std::string A = arch + ".";
    Config c{};
    c.moe = arch == "qwen35moe";
    c.architecture = arch;
    c.gated_attention = arch != "qwen3";
    if (f.has("pplx.embedding_length"))  // pplx-embed-v2-context: bidirectional attention, no LM head
        throw std::runtime_error("a contextual embedding model, not a chat model: tqserve serves it on POST /v1/embeddings");
    const uint32_t blocks = uint32_t(f.num(A + "block_count"));
    const uint32_t nextn = uint32_t(f.num(A + "nextn_predict_layers", 0));
    if (nextn > blocks) throw std::runtime_error("nextn_predict_layers exceeds block_count");
    c.layers = blocks - nextn;
    c.hidden = uint32_t(f.num(A + "embedding_length"));
    c.heads = uint32_t(f.num(A + "attention.head_count"));
    c.kvheads = uint32_t(f.num(A + "attention.head_count_kv"));
    c.head_dim = uint32_t(f.num(A + "attention.key_length", c.hidden / c.heads));
    c.rot_dim = uint32_t(f.num(A + "rope.dimension_count", c.head_dim));
    c.theta = float(f.num(A + "rope.freq_base"));
    c.ctx_train = uint32_t(f.num(A + "context_length", 0));
    if (const auto it = f.kv.find(A + "rope.scaling.type"); it != f.kv.end()) c.rope_type = it->second.str;
    c.rope_factor = float(f.num(A + "rope.scaling.factor", 1.0));
    c.rope_orig_ctx = uint32_t(f.num(A + "rope.scaling.original_context_length", 0));
    if (const auto it = f.kv.find(A + "rope.dimension_sections"); it != f.kv.end() && it->second.arr.size() >= 3)
        for (int i = 0; i < 3; ++i) c.msec[i] = int(it->second.arr[size_t(i)]);
    c.eps = float(f.num(A + "attention.layer_norm_rms_epsilon"));
    if (c.gated_attention) {
    c.lin_kheads = uint32_t(f.num(A + "ssm.group_count"));
    c.lin_vheads = uint32_t(f.num(A + "ssm.time_step_rank"));
    c.lin_dk = uint32_t(f.num(A + "ssm.state_size"));
    c.lin_dv = uint32_t(f.num(A + "ssm.inner_size")) / c.lin_vheads;
    c.conv_k = uint32_t(f.num(A + "ssm.conv_kernel"));
    c.full_interval = uint32_t(f.num(A + "full_attention_interval"));
    } else {
        c.lin_kheads = c.lin_vheads = 1; c.lin_dk = c.lin_dv = 32; c.conv_k = 4;
        c.full_interval = 1;
        if (c.head_dim != 128) throw std::runtime_error("Qwen3 currently requires head_dim 128");
    }

    if (!c.moe) {
        c.ffn = uint32_t(f.num(A + "feed_forward_length"));
    } else {
        // The MoE file declares no `feed_forward_length` - only the per-expert and shared-expert
        // widths. `ffn` carries the per-expert width so the existing FFN plumbing keeps meaning
        // "the intermediate width of the FFN being applied".
        c.expert_count = uint32_t(f.num(A + "expert_count"));
        c.expert_used = uint32_t(f.num(A + "expert_used_count"));
        c.expert_ff = uint32_t(f.num(A + "expert_feed_forward_length"));
        c.shared_ff = uint32_t(f.num(A + "expert_shared_feed_forward_length"));
        c.ffn = std::max(c.expert_ff, c.shared_ff);  // per-token FFN buffers serve the routed and the shared expert
        if (c.expert_count == 0 || c.expert_used == 0 || c.expert_used > c.expert_count)
            throw std::runtime_error("qwen35moe: expert_count/expert_used_count inconsistent (" +
                                     std::to_string(c.expert_count) + " experts, top-" +
                                     std::to_string(c.expert_used) + ")");
        if (c.expert_ff == 0 || c.shared_ff == 0)
            throw std::runtime_error("qwen35moe: zero expert FFN width");

        // Cross-check the metadata against the tensor table. The expert tensors are 3-D with the
        // expert index last (ne[2]); gate/up are [hidden, expert_ff, n_expert] and down is
        // [expert_ff, hidden, n_expert]. If the metadata disagrees with the file's own shapes the
        // router would index experts that do not exist, so refuse rather than continue.
        auto check = [&](const char * name, uint64_t n0, uint64_t n1) {
            const auto it = f.tensors.find(name);
            if (it == f.tensors.end())
                throw std::runtime_error(std::string("qwen35moe: missing expert tensor ") + name);
            const GgufTensor & t = it->second;
            if (t.ne.size() != 3 || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != c.expert_count)
                throw std::runtime_error(std::string("qwen35moe: ") + name + " expected [" +
                                         std::to_string(n0) + "," + std::to_string(n1) + "," +
                                         std::to_string(c.expert_count) + "] but the file declares " +
                                         gguf_shape_string(t));
        };
        check("blk.0.ffn_gate_exps.weight", c.hidden, c.expert_ff);
        check("blk.0.ffn_up_exps.weight", c.hidden, c.expert_ff);
        check("blk.0.ffn_down_exps.weight", c.expert_ff, c.hidden);
        // The shared expert is 2-D and uses the shared width, not the per-expert width.
        for (const char * n : {"blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight"}) {
            const auto it = f.tensors.find(n);
            if (it == f.tensors.end())
                throw std::runtime_error(std::string("qwen35moe: missing shared expert tensor ") + n);
            if (it->second.ne.size() != 2 || it->second.ne[0] != c.hidden ||
                it->second.ne[1] != c.shared_ff)
                throw std::runtime_error(std::string("qwen35moe: ") + n + " expected [" +
                                         std::to_string(c.hidden) + "," + std::to_string(c.shared_ff) +
                                         "] but the file declares " + gguf_shape_string(it->second));
        }
        // Every block must be MoE in this arch; a block without expert tensors would mean the
        // layer count or the arch string is wrong.
        for (uint32_t l = 0; l < c.layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".ffn_gate_exps.weight";
            if (!f.tensors.count(p))
                throw std::runtime_error("qwen35moe: block " + std::to_string(l) +
                                         " has no ffn_gate_exps.weight; expected every block to be MoE");
        }
    }

    const auto & emb = f.tensors.at("token_embd.weight");
    c.vocab = uint32_t(emb.ne[1]);
    c.tied = !f.tensors.count("output.weight");
    c.tiled = true;
    for (uint32_t l = 0; l < c.layers; ++l) c.full.push_back((l + 1) % c.full_interval == 0);
    return c;
}

}  // namespace qwen35_detail
}  // namespace tqllm
