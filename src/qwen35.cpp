// Qwen3.5 dense hybrid decode engine (text path): Gated DeltaNet + gated full attention, BF16 weights.
#include "dev_alloc.cuh"
#include "debug.hpp"
#include "batch_graph_key.hpp"
#include "engine.hpp"
#include <random>
#include "json.hpp"
#include "qwen2_kernels.cuh"
#include "gguf.hpp"
#include "prefill_chunk.hpp"
#include "gemm_buffer.hpp"
#include "gemm_pad.hpp"
#include "prefill_path.hpp"
#include "graph_capture.hpp"
#include "graph_lru.hpp"
#include "tree_columns.hpp"
#include "dev_alloc.cuh"
#include "fp8_gemm.cuh"
#include "int8_gemm.cuh"
#include "qwen35_kernels.cuh"
#include "quant_kernels.cuh"
#include "moe_kernels.cuh"
#include "vision_kernels.cuh"
#include "safetensors.hpp"
#include "dequant.hpp"
#include "dflash_kernels.cuh"
#include "formats/kv_format.hpp"
#include "kv_store.hpp"
#include "tqllm/rope_scaling.hpp"
#include "formats/weight_format.hpp"
#include "tqllm/tokenizer.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef TQLLM_VERIFY_COLUMNS
#define TQLLM_VERIFY_COLUMNS 12
#endif

namespace tqllm {
namespace {

using bf16 = __nv_bfloat16;

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

void cb(cublasStatus_t e) {
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

struct KvRef {  // a view of one layer's KV rows in the KV store's pool (src/kv_store.hpp); the store owns the memory
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

uint32_t u32(const Json & o, const char * k) {
    const Json * v = o.get(k);
    if (!v) throw std::runtime_error(std::string("config.json missing ") + k);
    return uint32_t(v->num);
}

Config read_config(const std::filesystem::path & dir) {
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
    Mat conv_attn, conv_ffn, w_qkv, wo, w_gu, down;                  // TQ3_4S (re-quantised at load)
};

// DFlash2 block-diffusion drafter (z-lab), weights re-quantised to TQ3_4S at load (drafter precision only
// affects acceptance; the target verifies every token).
struct DFlash {
    bool present = false;
    // DSpark (DFlash + Markov head, anchor-first): nb tokens [anchor, MASK x (nb - 1)] predict positions pos + 1 ..
    // pos + nb, laid out as lattice rows 1 .. nb (block = nb + 1); sel_prev / sel_next = markov_w1 / markov_w2, gate 1
    bool dspark = false;
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
bool arch_supported(const std::string & a) { return a == "qwen35" || a == "qwen35moe"; }

Config config_from_gguf(const GgufFile & f) {
    // The metadata namespace comes from the file, not from a hardcoded prefix. The dense and MoE
    // variants use different ones (`qwen35.` vs `qwen35moe.`), and the MoE file contains no
    // `qwen35.` key at all - reading it with the dense prefix failed as
    // "GGUF missing key qwen35.block_count", which named the wrong key rather than the real cause.
    std::string arch;
    if (f.has("general.architecture")) arch = f.str("general.architecture");
    if (arch.empty() || !arch_supported(arch))
        throw std::runtime_error("unsupported GGUF architecture '" + arch +
                                 "' (this engine parses qwen35 and qwen35moe only)");
    const std::string A = arch + ".";
    Config c{};
    c.moe = arch == "qwen35moe";
    const uint32_t blocks = uint32_t(f.num(A + "block_count"));
    const uint32_t nextn = uint32_t(f.num(A + "nextn_predict_layers", 0));
    if (nextn > blocks) throw std::runtime_error("nextn_predict_layers exceeds block_count");
    c.layers = blocks - nextn;
    c.hidden = uint32_t(f.num(A + "embedding_length"));
    c.heads = uint32_t(f.num(A + "attention.head_count"));
    c.kvheads = uint32_t(f.num(A + "attention.head_count_kv"));
    c.head_dim = uint32_t(f.num(A + "attention.key_length"));
    c.rot_dim = uint32_t(f.num(A + "rope.dimension_count"));
    c.theta = float(f.num(A + "rope.freq_base"));
    c.ctx_train = uint32_t(f.num(A + "context_length", 0));
    if (const auto it = f.kv.find(A + "rope.scaling.type"); it != f.kv.end()) c.rope_type = it->second.str;
    c.rope_factor = float(f.num(A + "rope.scaling.factor", 1.0));
    c.rope_orig_ctx = uint32_t(f.num(A + "rope.scaling.original_context_length", 0));
    if (const auto it = f.kv.find(A + "rope.dimension_sections"); it != f.kv.end() && it->second.arr.size() >= 3)
        for (int i = 0; i < 3; ++i) c.msec[i] = int(it->second.arr[size_t(i)]);
    c.eps = float(f.num(A + "attention.layer_norm_rms_epsilon"));
    c.lin_kheads = uint32_t(f.num(A + "ssm.group_count"));
    c.lin_vheads = uint32_t(f.num(A + "ssm.time_step_rank"));
    c.lin_dk = uint32_t(f.num(A + "ssm.state_size"));
    c.lin_dv = uint32_t(f.num(A + "ssm.inner_size")) / c.lin_vheads;
    c.conv_k = uint32_t(f.num(A + "ssm.conv_kernel"));
    c.full_interval = uint32_t(f.num(A + "full_attention_interval"));

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

class Qwen35Engine final : public Engine {
public:
    // `path`: a checkpoint directory (safetensors, BF16) or a .gguf file (TQ3_4S / Q6_K / F32).
    Qwen35Engine(const std::string & path, const std::string & tokpath, uint32_t ctx, int slots, const EngineOptions & o)
        : opt(o), context_(ctx) {
        glru.set_cap(size_t(std::max(1, opt.graph_cache)));
        graph_keep = size_t(std::max(0L, opt.graph_keep_mb)) << 20;
        if (slots < 1 || slots > 64) throw std::invalid_argument("slots must be in 1..64");
        if (!context_) throw std::invalid_argument("context must be positive");
        const bool gguf = path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
        std::unique_ptr<GgufFile> gf;
        if (gguf) {
            gf = std::make_unique<GgufFile>(gguf_open(path));
            cfg = config_from_gguf(*gf);
        } else {
            cfg = read_config(path);
        }
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        set_rope();
        try {
            tokenizer.load(tokpath);
            eos_ = tokenizer.eos_id();
            cb(cublasCreate(&blas));
            cb(cublasSetStream(blas, stream));
            cb(cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH));
            auto memlog = [](const char * stage) {  // TQLLM_DEBUG=memlog: device memory in use after each load stage
                static const bool on = tqllm::debug_on("memlog");
                size_t fr = 0, tot = 0;
                if (on && cudaMemGetInfo(&fr, &tot) == cudaSuccess)
                    std::fprintf(stderr, "tqllm: %-9s %.2f GB in use\n", stage, double(tot - fr) / 1e9);
            };
            memlog("start");
            if (gguf) load_gguf(*gf);
            else load_safetensors(path);
            memlog("weights");
            nslots = slots;
            allocate();
            memlog("buffers");
            if (!opt.draft_model.empty() && gguf) load_dflash(opt.draft_model.c_str());
            memlog("drafter");
            calibrate_prefill();
            pc_on = (opt.prompt_cache || kv_cmp) && slots == 1;
            if (slots > 1 && !fmt_target) throw std::runtime_error("multiple slots need a quantized GGUF model (verify path)");
            parked.resize(size_t(slots));
            for (int s = 1; s < slots; ++s) {  // allocate slot s in the members, then park it; slot 0 stays active
                SeqState tmp;
                swap_seq(tmp);
                alloc_seq_state();
                alloc_dfl_state();
                reset_state();
                swap_seq(tmp);
                parked[size_t(s)] = std::move(tmp);
            }
        } catch (...) {
            release_raw();
            throw;
        }
    }

    ~Qwen35Engine() override {
        parked.clear();  // KV stores return their pages, then the pool unmaps, while the stream still exists
        kv_seq.reset();
        kv_pool.reset();
        release_raw();
    }

    // CUDA objects and pinned host memory not owned by RAII members; also run when the constructor throws (a load that
    // runs out of memory must hand back the mapped host blocks: on WSL they count against the device budget, so
    // tqserve's auto-fit retries would otherwise lose ~2 GB per attempt)
    void release_raw() {
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

    uint32_t step(uint32_t id, uint32_t pos, bool sample, float * capture) override {
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
                    k = qg + 2 * nh * hd;
                    v = k + kvh * hd;
                } else {
                    mm(L.wq, np, qg);
                    mm(L.wk, np, k);
                    mm(L.wv, np, v);
                }
                tqllm_cuda::rms_f(qg, 2 * hd, L.q_norm, q, hd, hd, nh, cfg.eps, stream);
                tqllm_cuda::rms_f(k, hd, L.k_norm, k, hd, hd, kvh, cfg.eps, stream);
                tqllm_cuda::rope_partial(q, nh, hd, int(cfg.rot_dim), int(pos), cfg.theta, stream);
                tqllm_cuda::rope_partial(k, kvh, hd, int(cfg.rot_dim), int(pos), cfg.theta, stream);
                tqllm_cuda::cache_write(k, v, kc, vc, int(pos), kvh, hd, stream);
                tqllm_cuda::attention_decode(q, kc, vc, o, part.as<float>(), int(pos + 1), nh, kvh, hd,
                                             1.f / std::sqrt(float(hd)), stream);
                tqllm_cuda::sigmoid_gate(o, qg, nh, hd, stream);
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

    const float * logits() const override { return logits_buf.as<float>(); }
    void synchronize() override { sync_ck(); }
    uint32_t layers() const override { return cfg.layers; }
    uint32_t hidden() const override { return cfg.hidden; }
    uint32_t vocab() const override { return cfg.vocab; }
    uint32_t eos() const override { return eos_; }
    uint32_t context() const override { return context_; }
    size_t kv_token_bytes() const override { return size_t(nslots) * kv_pool_layers() * size_t(kv_row_bytes(true) + kv_row_bytes(false)); }
    size_t kv_host_bytes() const override { return kv_pool ? kv_pool->bytes(KvTier::host) : 0; }
    size_t kv_device_bytes() const override { return kv_pool ? kv_pool->bytes(KvTier::device) : 0; }
    size_t kv_round_bytes() const override { return kv_mem ? size_t(nslots) * kv_pool_layers() * 2 * kv_mem->granule() : 0; }
    // TqCache compaction: the next 8K block at cmp_pos is fully in host memory and there are enough recent queries
    bool kv_idle_pending() const override { return !opt.kv_compact_turn && cmp_pending(); }
    bool cmp_pending() const {
        if (!kv_cmp || ring_used < 64 || cmp_pos + kCmpBlock > cmp_max) return false;
        // staged: blocks already in host memory; device-resident: blocks older than the recent exact window
        const int64_t ev = kv_staged ? std::min(int64_t(kv_pool->staged_bytes(true) / size_t(kv_row_bytes(true))),
                                                int64_t(kv_pool->staged_bytes(false) / size_t(kv_row_bytes(false))))
                                     : int64_t(kv_end) - kCmpWindow;
        return ev >= cmp_pos + kCmpBlock;
    }
    bool kv_idle_step() override {
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
    void cmp_upload() {
        std::vector<int> ci{int(cmp_pos)};
        for (int r : cmp_r) ci.push_back(int(cmp_pos / r));
        const int64_t off[2] = {cmp_pos * kv_row_bytes(true), cmp_pos * kv_row_bytes(false)};
        ck(cudaMemcpyAsync(d_cmp_i.p, ci.data(), ci.size() * sizeof(int), cudaMemcpyHostToDevice, stream));  // pageable: staged now
        ck(cudaMemcpyAsync(d_stage_off.p, off, sizeof(off), cudaMemcpyHostToDevice, stream));
    }
    void cmp_reset() {
        cmp_pos = 0, ring_used = 0;
        cmp_upload();
    }
    void pc_drop() { pc_live = false, pc_snap = 0; }
    void pc_copy(bool save) {  // recurrent state of slot cur <-> pc_host
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
    // cache positions of prompt that can be kept (the recurrent state is set to match); the rest is prefilled
    uint32_t pc_match(const std::vector<uint32_t> & prompt) {
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
    size_t kv_pool_layers() const { return size_t(std::count(cfg.full.begin(), cfg.full.end(), true)); }

private:
    Config cfg;
    const EngineOptions opt;
    uint32_t context_, eos_ = 0;
    cudaStream_t stream{};
    cudaStream_t side{};                // parallel branch for small independent matvecs (GDN b / a), graph-captured
    cudaEvent_t ev_fork{}, ev_join{};
    cudaStream_t moe_side{};             // --moe-host-mb decode: cold (host) experts overlap the hot ones, graph-captured
    cudaEvent_t moe_fork{}, moe_join{};
    std::vector<cudaStream_t> seg_side;  // batched verify: one branch per extra segment (per-slot GDN / attention)
    std::vector<cudaEvent_t> seg_join;
    cudaEvent_t seg_fork{};
    size_t part_floats = 0;
    size_t gemm_c_bytes = 0;
    int gemm_w_rows = 0;      // > 0: FP16 weight buffer capped (EngineOptions::gemm_rows)
    size_t gemm_w_elems = 0;
    cublasHandle_t blas{};
    Tokenizer tokenizer;
    std::vector<Buf> tensors;
    // Target KV-cache format (src/formats/kv_*.cpp; -ctk / -ctv / --kv)
    const KvFormat * kvf = kv_format(opt.kv_format());
    // quality experiments: quantise->dequantise K / V before the cache write (see kv_fakequant)
    int fq_k = tqllm::debug_int("fakeq_k");
    int fq_v = tqllm::debug_int("fakeq_v");
    int mq_split = 0;  // verify attention splits: one wave of kvheads x splits blocks
    Mat embed, lm_head;
    const float * out_norm = nullptr;
    bool quantized = false;
    bool fmt_target = false;     // target matrices come from a registered weight format (src/formats)
    Basis basis_ = Basis::Plain;  // how verify/prefill matmul inputs are prepared (from the target's weight format)
    std::vector<Layer> layer;
    std::unique_ptr<KvMemory> kv_mem;  // declared before every KvStore so stores return their pages first
    std::unique_ptr<KvPool> kv_pool;
    std::unique_ptr<KvStore> kv_seq;   // the active slot's KV rows; key_cache / value_cache view them per attention layer
    bool kv_staged = false;            // old positions in host memory, staged back per attention layer (KvGeometry::staged)
    Buf d_stage_bytes;                 // [K, V] bytes per layer to stage (device: captured graphs read it at run time)
    cudaStream_t kv_side{};
    cudaEvent_t kv_fork{}, kv_ready[2]{}, kv_done[2]{};
    int kv_blocks = 0;
    // TqCache compaction (--kv-compact R, opt-in): per attention layer a ring of recent queries and the stand-in rows of
    // the compacted prefix [0, cmp_pos); attention reads cache positions from cmp_pos and the stand-ins for the rest
    static constexpr int kRing = 256, kCmpBlock = 8192, kCmpWindow = 16384;  // window: newest positions kept exact (device mode)
    uint32_t kv_end = 0;  // end of the positions written so far (device-resident compaction)
    bool kv_cmp = false;
    int cmp_layer = -1, ring_used = 0, ring_end = 0;  // the ring is written at position % kRing; newest row before ring_end
    std::vector<int> cmp_r;  // per attention layer; the middle ones lose most to compaction and keep twice the rows
    int64_t cmp_pos = 0, cmp_max = 0;  // cmp_max: positions the stand-in buffers can cover
    // Prompt reuse (one slot): the cache holds pc_seq. With pc_live the recurrent state is at pc_seq.size(); pc_host
    // holds the state at pc_snap, a prefill chunk boundary kTail tokens before the last prompt's end, for a prompt that
    // shares only that much (a chat template re-renders the previous answer without its think tags).
    static constexpr uint32_t kTail = 64;
    bool pc_on = false, pc_live = false;
    uint32_t pc_snap = 0;
    std::vector<uint32_t> pc_seq;
    std::unique_ptr<HostBuf> pc_host;
    std::vector<Buf> q_ring, cmp_k, cmp_v, cmp_b;
    std::vector<tqllm_cuda::KvCompactSeg> cmp_seg;
    Buf d_cmp_i, d_stage_off, cmp_scratch, cmp_split;  // d_cmp_i: [start position, stand-in rows]; d_stage_off: [K, V] bytes
    cusolverDnHandle_t solver{};
    std::vector<KvRef> key_cache, value_cache;
    std::vector<Buf> conv_state, rec_state;
    Buf x, norm, resid, mix, gate, up, act, logits_buf, token, part, xq, xs, xf, part_x, part_m, amax;
    Buf t_in, t_qg, t_q, t_k, t_v, t_o, t_qkv, t_qkvc, t_z, t_a, t_b, t_core, t_gn;

    // Quantized matrices consume the activation in the TQ3 rotated basis; prep() computes it once per input
    // vector and every following mm() on that input reuses it.
    void prep(const bf16 * input, int k) {
        if (quantized) tqllm_cuda::tq3_rotate_i8(input, xq.as<int8_t>(), xs.as<float>(), k, stream);
    }

    // Norm feeding a group of matmuls: quantized models go straight to the int8 rotated activation (the BF16
    // normalised vector is never materialised); BF16 models keep the reference RMSNorm.
    void norm_prep(const bf16 * x, const float * w, const float * partial, bf16 * np, int k) {
        if (quantized) tqllm_cuda::norm_rotate_i8(x, w, partial, cfg.eps, xq.as<int8_t>(), xs.as<float>(), k, stream);
        else tqllm_cuda::rms_f(x, 0, w, np, 0, k, 1, cfg.eps, stream);
    }

    void mm(const Mat & m, const bf16 * input, bf16 * output) {
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

    void to_host(const bf16 * src, size_t n, float * dst) {
        std::vector<bf16> tmp(n);
        ck(cudaMemcpyAsync(tmp.data(), src, n * sizeof(bf16), cudaMemcpyDeviceToHost, stream));
        sync_ck();
        for (size_t i = 0; i < n; ++i) dst[i] = __bfloat162float(tmp[i]);
    }

    const void * put(const void * host, size_t bytes) {
        Buf b(bytes);
        // cudaMemcpy from pageable memory may return before the DMA lands; copy on our stream and wait, so
        // kernels that post-process the upload (tq3_repack) never see a partial tensor.
        ck(cudaMemcpyAsync(b.p, host, bytes, cudaMemcpyHostToDevice, stream));
        sync_ck();
        const void * p = b.p;
        tensors.push_back(std::move(b));
        return p;
    }

    const float * put_f32(const std::vector<float> & v) {
        return static_cast<const float *>(put(v.data(), v.size() * 4));
    }

    // ---- safetensors (HF BF16) ----
    const SafeTensor & st(const std::map<std::string, SafeTensor> & idx, const std::string & name, uint64_t elems) {
        const auto it = idx.find(name);
        if (it == idx.end()) throw std::runtime_error("missing tensor: " + name);
        uint64_t have = 1;
        for (auto d : it->second.shape) have *= d;
        if (have != elems) throw std::runtime_error("unexpected shape for " + name);
        return it->second;
    }

    Mat st_mat(const std::map<std::string, SafeTensor> & idx, const std::string & name, int rows, int cols) {
        const auto & t = st(idx, name, uint64_t(rows) * cols);
        if (t.dtype != "BF16") throw std::runtime_error("expected BF16 for " + name);
        const auto host = safetensors_read(t);
        return {put(host.data(), host.size()), GGML_BF16, rows, cols};
    }

    // Any 1-D/small tensor as FP32, optionally + 1 (zero-centred norm weights).
    std::vector<float> st_f32(const std::map<std::string, SafeTensor> & idx, const std::string & name, uint64_t n,
                              float add = 0.f) {
        const auto & t = st(idx, name, n);
        const auto host = safetensors_read(t);
        std::vector<float> out(n);
        for (uint64_t i = 0; i < n; ++i) {
            if (t.dtype == "F32") out[i] = reinterpret_cast<const float *>(host.data())[i];
            else if (t.dtype == "BF16") out[i] = __bfloat162float(reinterpret_cast<const bf16 *>(host.data())[i]);
            else throw std::runtime_error("unexpected dtype for " + name);
            out[i] += add;
        }
        return out;
    }

    void load_safetensors(const std::string & dir) {
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

    // ---- GGUF (llama.cpp-tq3 layout: norms already 1 + w, ssm_a = -exp(A_log), tiled value heads) ----
    const GgufTensor & gt(const GgufFile & f, const std::string & name) {
        const auto it = f.tensors.find(name);
        if (it == f.tensors.end()) throw std::runtime_error("missing GGUF tensor: " + name);
        return it->second;
    }

    // A matrix of a registered weight format: remember the basis its matmul inputs need (one per model).
    Mat adopt(Mat m) {
        if (!m.fmt) return m;
        if (m.fmt->basis() == Basis::Tq3Rot32) quantized = true;
        if (fmt_target && basis_ != m.fmt->basis() && m.fmt->basis() != Basis::Plain && basis_ != Basis::Plain)
            throw std::runtime_error(std::string("mixed rotated weight formats in one model: ") + m.fmt->name());
        if (!fmt_target || m.fmt->basis() != Basis::Plain) basis_ = m.fmt->basis();
        fmt_target = true;
        return m;
    }
    Upload uploader() {
        return [this](const void * h, size_t b) { return put(h, b); };
    }

    // F32 GGUF matrices (llama.cpp keeps small ones such as ssm_alpha / ssm_beta in F32) run as BF16
    static std::vector<char> f32_to_bf16_host(const std::vector<char> & raw) {
        std::vector<char> out(raw.size() / 2);
        for (size_t i = 0; i < raw.size() / 4; ++i) {
            float f;
            std::memcpy(&f, raw.data() + 4 * i, 4);
            const __nv_bfloat16 b = __float2bfloat16_rn(f);
            std::memcpy(out.data() + 2 * i, &b, 2);
        }
        return out;
    }

    // F16 GGUF matrices (llama-quantize COPY mode keeps ssm_alpha / ssm_beta in F16) also run as BF16.
    // F16 -> F32 by pure IEEE-754 bit manipulation (no <cmath> dependency in this TU), then round to BF16.
    static inline float f16_bits_to_f32(uint16_t h) {
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
    static std::vector<char> f16_to_bf16_host(const std::vector<char> & raw) {
        std::vector<char> out(raw.size());  // 2 bytes in -> 2 bytes out
        for (size_t i = 0; i < raw.size() / 2; ++i) {
            uint16_t h;
            std::memcpy(&h, raw.data() + 2 * i, 2);
            const __nv_bfloat16 b = __float2bfloat16_rn(f16_bits_to_f32(h));
            std::memcpy(out.data() + 2 * i, &b, 2);
        }
        return out;
    }

    // Matmul matrices load through their weight format (src/formats); `repack` = false loads raw bytes (embeddings).
    Mat g_mat(const GgufFile & f, const std::string & name, int rows, int cols, bool repack = true) {
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
        if (repack)
            if (const auto * wf = weight_format(t.type)) return adopt(wf->load(host, rows, cols, uploader(), stream));
        if (t.type != GGML_TQ3_4S && t.type != GGML_Q6_K && t.type != GGML_BF16 &&
            !((t.type == GGML_Q8_0 || t.type == GGML_Q4_K || t.type == GGML_Q3_K || t.type == GGML_Q5_K ||
               t.type == GGML_IQ3_S) && !repack))
            throw std::runtime_error("unsupported matrix type " + std::to_string(t.type) + " for " + name);
        if (t.type == GGML_TQ3_4S) quantized = true;
        return {put(host.data(), host.size()), t.type, rows, cols};
    }

    // 3-D MoE expert tensor [cols, rows, E] loaded as ONE stacked matrix of E * rows rows (the expert index is the outermost
    // dimension, so expert e is the contiguous row range [e * rows, (e + 1) * rows)); expert_rows() views one expert.
    // `extra`: a 2-D tensor of the same type and expert shape appended as expert `experts` (the fused shared expert)
    Mat g_exps(const GgufFile & f, const std::string & name, int experts, int rows, int cols,
               bool to_host = false, const std::string & extra = {}) {
        const auto & t = gt(f, name);
        if (t.ne.size() != 3 || t.ne[0] != uint64_t(cols) || t.ne[1] != uint64_t(rows) || t.ne[2] != uint64_t(experts))
            throw std::runtime_error("unexpected expert tensor shape for " + name);
        const auto * wf = weight_format(t.type);
        if (!wf) throw std::runtime_error("no weight format for expert tensor type " + std::to_string(t.type) + " (" + name + ")");
        auto host = f.read(t);
        if (!extra.empty()) {
            const auto x = f.read(gt(f, extra));
            host.insert(host.end(), x.begin(), x.end());
        }
        Mat m = adopt(wf->load(host, (experts + !extra.empty()) * rows, cols, to_host ? host_uploader() : uploader(), stream));
        if (!m.fmt || !m.fmt->has_expert_matvec())
            throw std::runtime_error(name + ": its device format has no grouped expert projection");
        return m;
    }
    // g_exps with the experts flagged in `cold` loaded into pinned host memory (UVA) and the others into VRAM, each set
    // stacked in expert order; returns {hot, cold}
    std::pair<Mat, Mat> g_exps_split(const GgufFile & f, const std::string & name, int experts, int rows, int cols,
                                     const std::vector<char> & cold, const std::string & extra = {}) {
        const auto & t = gt(f, name);
        if (t.ne.size() != 3 || t.ne[0] != uint64_t(cols) || t.ne[1] != uint64_t(rows) || t.ne[2] != uint64_t(experts))
            throw std::runtime_error("unexpected expert tensor shape for " + name);
        const auto * wf = weight_format(t.type);
        if (!wf) throw std::runtime_error("no weight format for expert tensor type " + std::to_string(t.type) + " (" + name + ")");
        const auto host = f.read(t);
        const size_t eb = host.size() / size_t(experts);
        std::vector<char> hb, cb;
        for (int e = 0; e < experts; ++e) {
            auto & dst = cold[size_t(e)] ? cb : hb;
            dst.insert(dst.end(), host.begin() + long(size_t(e) * eb), host.begin() + long(size_t(e + 1) * eb));
        }
        if (!extra.empty()) {
            const auto x = f.read(gt(f, extra));
            hb.insert(hb.end(), x.begin(), x.end());
        }
        const int nc = int(cb.size() / eb), nh = experts - nc + !extra.empty();
        Mat hot = adopt(wf->load(hb, nh * rows, cols, uploader(), stream));
        Mat cm = adopt(wf->load(cb, nc * rows, cols, host_uploader(), stream));
        for (const Mat * m : {&hot, &cm})
            if (!m->fmt || !m->fmt->has_expert_matvec()) throw std::runtime_error(name + ": its device format has no grouped expert projection");
        return {hot, cm};
    }
    // --moe-host-mb: the least-routed experts over all layers (counts from the profile) until about M MiB are placed
    std::vector<std::vector<char>> moe_placement(const GgufFile & f) {
        const int L = int(cfg.layers), E = int(cfg.expert_count);
        std::vector<std::vector<char>> cold(size_t(L), std::vector<char>(size_t(E), 0));
        if (opt.moe_host_mb <= 0) return cold;
        const std::string path = opt.moe_profile.empty() ? f.path + ".experts" : opt.moe_profile;
        std::ifstream in(path);
        int pl = 0, pe = 0;
        if (!(in >> pl >> pe) || pl != L || pe != E)
            throw std::runtime_error("--moe-host-mb needs a routing profile for this model (" + path + "; write one with TQLLM_MOE_PROFILE=FILE)");
        std::vector<std::tuple<unsigned long long, int, int>> order;
        unsigned long long total = 0, placed = 0;
        for (int l = 0; l < L; ++l) for (int e = 0; e < E; ++e) {
            unsigned long long c = 0;
            if (!(in >> c)) throw std::runtime_error("truncated routing profile " + path);
            order.emplace_back(c, l, e); total += c;
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
    static Mat expert_rows(const Mat & m, int e, int rows) {
        Mat v = m;
        v.p = static_cast<const char *>(m.p) + size_t(e) * rows * m.fmt->row_bytes(m.cols);
        if (m.aux) v.aux = static_cast<const char *>(m.aux) + size_t(e) * rows * m.fmt->aux_row_bytes(m.cols);
        v.rows = rows;
        return v;
    }

    // Row-concatenation of same-width matrices into one device matrix: one buffer when the types agree, else a composite
    // of per-part formats ({} when the types cannot be combined: the caller falls back to separate matrices).
    std::vector<std::unique_ptr<WeightFormat>> composites;
    Mat g_cat(const GgufFile & f, const std::vector<std::pair<std::string, int>> & parts, int cols) {
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

    const float * g_f32(const GgufFile & f, const std::string & name, uint64_t n) {
        const auto & t = gt(f, name);
        if (t.type != GGML_F32 || t.bytes != n * 4) throw std::runtime_error("expected F32[" + std::to_string(n) + "] " + name);
        const auto host = f.read(t);
        return static_cast<const float *>(put(host.data(), host.size()));
    }

    void load_gguf(const GgufFile & f) {
        const int H = int(cfg.hidden), F = int(cfg.ffn), hd = int(cfg.head_dim);
        const int lk = int(cfg.lin_kheads * cfg.lin_dk), lv = int(cfg.lin_vheads * cfg.lin_dv);
        const int vh = int(cfg.lin_vheads), qkvd = 2 * lk + lv;
        embed = g_mat(f, "token_embd.weight", int(cfg.vocab), H, false);
        out_norm = g_f32(f, "output_norm.weight", H);
        const auto ht = gt(f, "output.weight").type;  // Q6_K / Q5_K heads stay raw (read through the int8 copy)
        lm_head = cfg.tied ? embed : g_mat(f, "output.weight", int(cfg.vocab), H, ht != GGML_Q6_K && ht != GGML_Q5_K);
        if (cfg.tied && embed.type == GGML_TQ3_4S) throw std::runtime_error("tied TQ3_4S LM head not supported yet");
        const auto mp = "blk." + std::to_string(cfg.layers) + ".";
        // the MTP head drafts without -md, or beside DFlash2 with --mtp (--mtp-from N: prompts of N+ tokens use MTP)
        if (f.tensors.count(mp + "nextn.eh_proj.weight") && (opt.draft_model.empty() || opt.native_mtp) && !cfg.moe) {  // MoE MTP is not wired yet
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
            L.w_gu = g_cat(f, {{mp + "ffn_gate.weight", F}, {mp + "ffn_up.weight", F}}, H);
            L.w_down = g_mat(f, mp + "ffn_down.weight", H, F);
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
                const int E = int(cfg.expert_count), FE = int(cfg.expert_ff), FS = int(cfg.shared_ff);
                L.router = g_f32(f, p + "ffn_gate_inp.weight", uint64_t(E) * H);
                const auto & cold = cold_of[l];
                const int nc = int(std::count(cold.begin(), cold.end(), char(1)));
                const bool host = nc == E || int(l) < opt.n_cpu_moe;
                // the shared expert joins the VRAM experts for decode when its matrices have the experts' type and shape
                // (TQLLM_MOE_FUSE_SHARED=0: off); prefill keeps its dense path
                auto same = [&](const char * sh, const char * ex, int rows, int cols) {
                    if (!f.tensors.count(p + sh)) return false;
                    const auto & a = gt(f, p + sh);
                    const auto & b = gt(f, p + ex);
                    return a.type == b.type && a.ne.size() == 2 && a.ne[0] == uint64_t(cols) && a.ne[1] == uint64_t(rows) && FS == FE;
                };
                const char * fz = std::getenv("TQLLM_MOE_FUSE_SHARED");
                const bool fuse = !host && !(fz && fz[0] == '0') && same("ffn_gate_shexp.weight", "ffn_gate_exps.weight", FE, H) &&
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
                    if (m->p && (!m->fmt->has_expert_matvec() || m->fmt->basis() != Basis::Plain))
                        throw std::runtime_error(p + "unsupported grouped expert format: " + m->fmt->name());
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
                if (!L.w_mix.p && (bt == GGML_BF16 || bt == GGML_F32 || bt == GGML_F16) && weight_format(gt(f, p + "attn_qkv.weight").type)) {
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

    // Gated DeltaNet state is double-buffered per layer: the current state is slot `cur`; verify() writes the state
    // after all n tokens to the other slot and commit(k) either flips `cur` (k == n) or replays k tokens into it.
    static constexpr int NVER = TQLLM_VERIFY_COLUMNS, NSLOT = 2;
    // prefill chunk: the verify graph over NPF prompt tokens, projections as one GEMM (weights read once per chunk).
    // Larger chunks amortise the per-chunk FP16 weight dequant; the size follows the context budget (buffer memory).
    int NPF = 512, NBUF = 512;
    static constexpr int NGEMM_BLAS = 128;  // columns from which projections dequantize to FP16 and run on cuBLAS
    static_assert(NVER >= 8 && NVER <= 16);
    int cur = 0;
    bf16 * conv_slot(uint32_t lin, int slot) { return conv_state[lin * NSLOT + (slot % NSLOT)].as<bf16>(); }
    float * rec_slot(uint32_t lin, int slot) { return rec_state[lin * NSLOT + (slot % NSLOT)].as<float>(); }

    size_t conv_bytes() const { return size_t(cfg.conv_k - 1) * (2 * cfg.lin_kheads * cfg.lin_dk + cfg.lin_vheads * cfg.lin_dv) * 2; }
    size_t rec_bytes() const { return size_t(cfg.lin_vheads) * cfg.lin_dk * cfg.lin_dv * 4; }
    void reset_state() {
        const size_t conv_bytes = this->conv_bytes(), rec_bytes = this->rec_bytes();
        cur = 0;
        if (pend_k) set_pending(0);
        for (uint32_t l = 0; l < conv_state.size() / NSLOT; ++l) {
            ck(cudaMemsetAsync(conv_slot(l, 0), 0, conv_bytes, stream));
            ck(cudaMemsetAsync(rec_slot(l, 0), 0, rec_bytes, stream));
        }
    }

    // RoPE scaling from the CLI, else the GGUF; the tables are global to the qwen35 kernels (one model per process).
    void set_rope() {
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

    // Per-sequence KV and Gated DeltaNet state of the active slot.
    void alloc_seq_state() {
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

    void allocate() {
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
            if (const char * pf = std::getenv("TQLLM_MOE_PROFILE")) {
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
        if (fmt_target && basis_ != Basis::Tq3Rot32) {  // BF16 Gated DeltaNet b/a next to 4-bit matrices
            v_xb = Buf(NBUF * widest * 2);
            v_mixa = Buf(NBUF * (2 * lk + 2 * lv) * 2);
            v_mixb = Buf(NBUF * 2 * size_t(cfg.lin_vheads) * 2);
        }
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
            head8 = Buf(size_t(lm_head.rows) * H);
            head8s = Buf(size_t(lm_head.rows) * H / 16 * 2);
            if (lm_head.type == GGML_Q5_K)
                tqllm_cuda::q5k_to_i8(lm_head.p, head8.as<int8_t>(), head8s.as<__half>(), lm_head.rows, int(H), stream);
            else tqllm_cuda::q6k_to_i8(lm_head.p, head8.as<int8_t>(), head8s.as<__half>(), lm_head.rows, int(H), stream);
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
            if (head_host) head8_to_host();  // exact re-score rows from host memory
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
            if (bytes) {
                const void * old = embed.p;
                embed.p = const_cast<void *>(to_mapped_host(old, bytes));
                for (auto & b : tensors)
                    if (b.p == old) { b = Buf(); break; }
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
                if ((opt.prefill_fp8 == 1 || (opt.prefill_fp8 < 0 && tqllm_cuda::Fp8Gemm::default_on())) && !gemm_w_rows) {
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
                if (!fp8 && (opt.prefill_int8 == 1 || (opt.prefill_int8 < 0 && tqllm_cuda::Int8Gemm::default_on())) && !gemm_w_rows &&
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

    Buf v_x, v_m, v_r, v_np, v_in, v_o, v_core, v_act, v_xh, v_xf, v_px, v_pm, v_logits, v_tok, v_amax, v_q, v_qkvc, v_part;
    Buf moe_counts, moe_groups;
    Buf moe_logits, moe_ids, moe_weights, moe_shared_weights, moe_gate, moe_up, moe_act, moe_out, moe_shared;
    Buf moe_hot_ids, moe_cold_ids, moe_counts2, moe_groups2;  // --moe-host-mb split
    Buf moe_xids;  // decode ids widened by the fused shared expert (no split)
    // --moe-host-mb decode: routed cold experts copied to the device (gate, up, down; weights and scales) before their
    // projections, stage_slots assignments at most; empty when a format's per-expert bytes are not 16-byte multiples
    Buf moe_stage_buf[tqllm_cuda::kMoeStageSpans], moe_stage_ids;
    long long moe_stage_bytes[tqllm_cuda::kMoeStageSpans] = {};
    int moe_stage_slots = 0;
    // --moe-cache-mb: CLOCK cache of host experts in VRAM, one pool per expert layout (slot stride == the kernels' expert
    // stride), keyed by a global host-expert index (layer base + cold slot)
    struct MoeCache {
        long long bytes[tqllm_cuda::kMoeStageSpans] = {};
        uint32_t types[3] = {};
        Buf buf[tqllm_cuda::kMoeStageSpans], key, ref, hand;
        int slots = 0;
        long weight = 0;
    };
    std::vector<MoeCache> moe_cache;
    std::vector<int> moe_cache_of, moe_cold_base;  // per layer: pool (-1 none), first global host-expert index
    Buf moe_cache_table, moe_cache_slot, moe_cache_fill;
    void alloc_moe_cache() {
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
    void alloc_moe_stage() {
        if (std::getenv("TQLLM_MOE_STAGE") && std::string(std::getenv("TQLLM_MOE_STAGE")) == "0") return;
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
    Buf moe_prof; std::string moe_prof_path;  // TQLLM_MOE_PROFILE: routing counts [L][E]
    void moe_profile_write() {
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
    // per Gated DeltaNet layer: [n][mix rows] projected inputs of a verify (replayed at commit), double-buffered so a
    // deferred commit (tree mode) can replay the previous step's path while the next verify writes the other buffer
    std::vector<Buf> v_stash2[2];
    int sp = 0;                          // stash buffer of the next verify
    int pend_k = 0, pend_sp = 0;         // deferred commit: accepted path length and its stash buffer (0: none)
    Buf d_pend, d_rep_qkvc;              // device copy of pend_k; conv outputs of the replayed path
    bool defer = false;                  // tree mode: commit replay folded into the next verify's GDN / conv kernels
    std::vector<Buf> & v_stash_cur() { return v_stash2[sp]; }
    Buf v_tokens, v_pos, v_hid, v_cat, mtp_h, mtp_kc, mtp_vc;
    MtpHead mtp;
    DFlash dfl;
    std::vector<Buf> dfl_kc, dfl_vc;
    Buf dfl_rscore, dfl_rsel, dfl_rk, dfl_rv, dfl_rpos;  // drafter retrieval window (--draft-retrieve)
    int retr_vpos = 0;  // host source of dfl_rpos (stable address for the async copy)
    Buf head8, head8s;  // verify-path LM head: Q6_K re-laid out as int8 + FP16 scales per 16
    // where head8 is read from: the device buffers, or pinned host memory mapped into the device (TQLLM_HEAD_HOST)
    const int8_t * w8() const { return h8_host ? h8_dev_w : head8.as<int8_t>(); }
    const __half * s8() const { return h8_host ? h8_dev_s : head8s.as<__half>(); }
    bool has_head8() const { return h8_host || head8.p; }
    bool k_head() const { return lm_head.type == GGML_Q6_K || lm_head.type == GGML_Q5_K; }
    void * h8_host = nullptr;
    const int8_t * h8_dev_w = nullptr;
    const __half * h8_dev_s = nullptr;
    cudaGraphExec_t dfl_graph = nullptr;  // captured dflash_draft body (K/V ring mode)
    Buf head4, head4s;  // draft-only 4-bit copy of head8 (DFlash2 full-vocabulary draft head)
    Buf h2_vals, h2_idx, h2_scr;  // two-stage verify head (default; TQLLM_HEAD8=1 opts out)
    // Sampling (temperature > 0): the head also returns each column's top-16 ids with exact logits; the host runs
    // speculative sampling (accept draft d with p(d), else resample from p without d; bonus token from p).
    struct Sampling { float temperature = 0.f, top_p = 1.f, presence = 0.f; int top_k = 0; } samp;
    static constexpr int kPresenceLastN = 64;  // llama.cpp's default penalty window (repeat_last_n)
    // DFlash generation in progress (spec_begin / spec_advance / spec_finish).
    struct Gen {
        std::vector<uint32_t> out, hist;  // hist: the prompt (prompt-lookup drafting reads hist + out)
        std::vector<uint32_t> mtp_draft;
        bool native_mtp = false;
        uint32_t P = 0, count = 0, pos = 0, pending = 0;
        int D = 0;
        int rdelta = 0;  // KV position - RoPE position: an image spans nx * ny cells but max(nx, ny) positions
        bool done = true;
        uint64_t steps = 0, drafted = 0, accepted = 0;
        double t_draft = 0, t_verify = 0, t_tail = 0, prefill_ms = 0;
        std::chrono::steady_clock::time_point decode_start;
    } gen;
    // Multi-slot serving: the active slot's sequence state lives in the members above (key_cache, conv_state, ...);
    // the others are parked here and swapped in by select_slot, so single-slot code paths are unchanged.
    struct SeqState {
        std::unique_ptr<KvStore> kv_seq;
        std::vector<KvRef> key_cache, value_cache;
        std::vector<Buf> conv_state, rec_state, dfl_kc, dfl_vc;
        cudaGraphExec_t dfl_graph = nullptr;
        int cur = 0;
        Sampling samp;
        std::mt19937_64 rng{0};
        Gen gen;
    };
    std::vector<SeqState> parked;  // parked[s] holds slot s while it is not active
    int active = 0, nslots = 1, NB = NVER;
    // Batched verify (spec_step): segment i = one slot's columns [col0, col0 + n) at positions pos.. (v_pos[2i]).
    struct Seg { int slot, col0, n; uint32_t pos; bool tree = false; };  // tree: drafted token tree (d_tpar + col0)
    std::vector<Seg> segs;  // empty: single-sequence verify on the active slot
    int stash_col0 = 0;     // commit(k) replays from this column of v_stash
    std::vector<KvRef> & kc_of(int s) { return s == active ? key_cache : parked[size_t(s)].key_cache; }
    std::vector<KvRef> & vc_of(int s) { return s == active ? value_cache : parked[size_t(s)].value_cache; }
    std::vector<Buf> & conv_of(int s) { return s == active ? conv_state : parked[size_t(s)].conv_state; }
    std::vector<Buf> & rec_of(int s) { return s == active ? rec_state : parked[size_t(s)].rec_state; }
    int cur_of(int s) const { return s == active ? cur : parked[size_t(s)].cur; }
    std::vector<Buf> & dkc_of(int s) { return s == active ? dfl_kc : parked[size_t(s)].dfl_kc; }
    std::vector<Buf> & dvc_of(int s) { return s == active ? dfl_vc : parked[size_t(s)].dfl_vc; }
    bool wide_mm = false;                          // mm_h: chunk columns by the format's batch width (batched drafts)
    // Tree verification (TQLLM_TREE=1, a tree_ok KV format, greedy): the verify block is a token tree (column parents d_tpar,
    // depths d_tdep, parent < child). Every verify and replay then runs the tree-capable kernels (chains included), so a
    // node's result is exactly that of its path verified as a chain: outputs never depend on the drafts.
    bool tree_mode = false;
    bool chain2 = false;  // KV format with two-part attention: every verify-sized block uses it (see vpar)
    Buf d_tpar, d_tdep, d_chain, d_cdep, d_tcols, d_gdesc;
    const int * vpar = nullptr, * vdep = nullptr;
    bool tree_step = false;  // the current verify block is a drafted tree (d_tpar / d_tdep)
    void swap_seq(SeqState & s) {
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
    std::mt19937_64 rng{0};
    Buf s_id, s_lg, s_scr;
    std::vector<uint32_t> h_sid;
    std::vector<float> h_slg;
    bool sampling() const { return samp.temperature > 0.f; }
    bool batch_topk = false;  // batched verify with at least one sampling slot: every column returns its top-16
    bool want_topk() const { return batch_topk || sampling(); }
    Buf dsub_ids, dsub_w, dsub_s;  // optional DFlash2 draft-vocabulary head (4-bit) + row -> token id
    int dsub_rows = 0;
    bool dsub_in_head4 = false;  // draft vocabulary read from head4 rows through dsub_ids
    Mat dsub_mat{};  // draft vocabulary head in the LM head's own weight format (TQ4); else dsub_w / dsub_s are u4h
    bool prefill_final = true;
    std::unique_ptr<tqllm_cuda::Fp8Gemm> fp8;  // --prefill-fp8 on (sm_89+): FP8 prefill GEMMs
    std::unique_ptr<tqllm_cuda::Int8Gemm> i8;  // --prefill-int8 on (Ampere): int8 prefill GEMMs into gemm_c as int32
    bool i8_used = false;
    Buf fp8_sw;                     // row scales of the current gemm_rows output, [ldo]
    bool fp8_x_ready = false, fp8_used = false;
    size_t prefill_room = 0;        // free device memory after the device-resident KV (chunk rule)
    Buf gemm_w, gemm_c, blas_ws;  // prefill cuBLAS path: FP16 weight scratch, FP32 output, capture-safe workspace  // prefill chunks (n > NVER): run the LM head only on the final chunk
    Buf v_feat, feat1, d_x, d_r, d_a, d_b, d_dyn, d_dynf, d_te, d_gate, d_vals, d_idx, d_cand, d_score, d_pos, d_tok, d_kv, d_inj_g, d_inj_kv,
        d_topk;
    Buf dfl_part, ds_t, ds_act, ds_q, ds_o;  // DSpark drafter attention partials and block scratch
    uint32_t * h_tok = nullptr;  // pinned staging for the graph inputs
    int * h_pos = nullptr;
    // RoPE positions (v_pos minus the slot's rdelta; same layout), and for prefill chunks with image tokens the per-column
    // table mm_tab [n][4] (source row in mm_img or -1, t, h, w)
    int * h_rpos = nullptr;
    Buf v_rpos, mm_tab, mm_img;
    bool mm_chunk = false;
    size_t mm_img_rows = 0;
    struct PendingImage { uint32_t at; int nx, ny; size_t row0; };
    std::vector<PendingImage> mm_pending;
    // v_pos rows [0, nseg) were just staged in h_pos: stage the RoPE copy (delta per segment) next to it
    void upload_rpos(int nseg, const std::function<int(int)> & delta) {
        for (int i = 0; i < nseg; ++i) {
            h_rpos[2 * i] = h_pos[2 * i] - delta(i);
            h_rpos[2 * i + 1] = h_pos[2 * i + 1] - delta(i);
        }
        ck(cudaMemcpyAsync(v_rpos.p, h_rpos, 2 * size_t(nseg) * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    int rdelta_of(int slot) const { return slot == active ? gen.rdelta : parked[size_t(slot)].gen.rdelta; }
    // captured graphs, one bounded cache (src/graph_lru.hpp): verify per slot / shape, batched verify keyed by the segment
    // layout, batched drafts keyed by the slot list, tree commit + drafter inject keyed by k and the slot state
    enum : int { kGraphVerify, kGraphBatch, kGraphDraft, kGraphCommit };
    tqllm::GraphLru<cudaGraphExec_t> glru;
    size_t graph_keep = 0;  // a miss evicts while less than this is free (0: by count only, e.g. WSL reads ~0 free)
    template <class Body>
    cudaGraphExec_t cached_graph(int family, uint64_t key, Body && body) {
        if (cudaGraphExec_t e = glru.find(family, key)) return e;
        return glru.insert(
            family, key, [&] { return tqllm::instantiate_graph(tqllm::capture_graph(stream, body)); },
            [&] { size_t fr = 0, tot = 0; return graph_keep && cudaMemGetInfo(&fr, &tot) == cudaSuccess && fr < graph_keep; },
            [&] { cudaStreamSynchronize(stream); }, [](cudaGraphExec_t g) { cudaGraphExecDestroy(g); });
    }
    // TQLLM_DEBUG=stageprof: verify blocks run eagerly with a stream sync after every stage; per-stage wall time (kernel +
    // its launch latency in isolation) is printed with the per-step profile. Perturbs the step; for attribution only.
    const bool stage_prof = tqllm::debug_on("stageprof");
    bool stage_on = false;
    std::map<std::string, std::pair<double, uint64_t>> stage_acc;
    std::chrono::steady_clock::time_point stage_t;
    void st(const char * name) {
        if (!stage_on) return;
        cudaStreamSynchronize(stream);
        const auto t = std::chrono::steady_clock::now();
        auto & a = stage_acc[name];
        a.first += std::chrono::duration<double, std::micro>(t - stage_t).count();
        ++a.second;
        stage_t = t;
    }
    size_t v_mix_rows = 0;
    Buf v_xb, v_mixa, v_mixb;  // W4A16: BF16 copy of the activation for BF16 matrices; [qkv|z] and [b|a] partials
    int last_n = 0;
    double last_prefill_ms = 0;
    std::chrono::steady_clock::time_point dec_log_tp;  // last TQLLM_DEBUG=decode progress print

    // v_xh holds the input prepared in `in` (the model's basis unless a caller prepared its own)
    void mm_h(const Mat & m, bf16 * out, int n) { mm_h(m, out, n, basis_); }
    void mm_h(const Mat & m, bf16 * out, int n, Basis in) {
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
    // Prefill dispatch by memory traffic (prefill_path.hpp): short chunks of formats without a native GEMM no longer pay
    // the per-chunk FP16 dequantisation of every weight.
    std::map<const WeightFormat *, PrefillCross> pf_cross;  // measured per format at load (calibrate_prefill)
    PrefillFormat prefill_fmt(const Mat & m) const {
        PrefillFormat f{m.fmt->max_matvec_cols(), m.fmt->max_batch_cols(), m.fmt->has_gemm()};
        if (const auto it = pf_cross.find(m.fmt); it != pf_cross.end()) f.cross = it->second.cross, f.cross_native = it->second.native;
        return f;
    }
    // Time the prefill paths on each format's largest projection and keep the column count up to which the batched
    // matvec (or native kernel) beats dequant + cuBLAS on this GPU. TQLLM_DEBUG=prefill_path:old skips it (old rule).
    void calibrate_prefill() {
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
    PrefillPath path_of(const Mat & m, int n) const {
        if (const std::string * v = tqllm::debug_value("prefill_path"); v && *v == "old") {  // A/B against the old rule
            if (n <= m.fmt->max_matvec_cols()) return PrefillPath::matvec;
            if (!segs.empty() || wide_mm) return PrefillPath::batched;
            if ((n < NGEMM_BLAS || !gemm_w.p) && m.fmt->has_gemm()) return PrefillPath::native;
            return PrefillPath::dequant;
        }
        return prefill_path(n, prefill_fmt(m), gemm_w.p != nullptr, !segs.empty() || wide_mm);
    }

    // Prefill chunk: FP16 weights once per chunk + cuBLAS GEMM; column j of the result goes to out + j * ldo (ldo >= rows;
    // the rows past m.rows receive unspecified values, for the caller to overwrite).
    // FP16 accumulate (2x tensor rate on GA10x; same answers on the 7K/64K probes)
    // m on n columns takes the prefill GEMM (dequantise + cuBLAS) path of mm_h
    bool prefill_gemm(const Mat & m, int n) const {
        return m.fmt && gemm_w.p && path_of(m, n) == PrefillPath::dequant;
    }
    // out == nullptr (FP16 accumulate only): the FP16 result stays in gemm_c, [n][ldo]
    void gemm_rows(const Mat & m, bf16 * out, int n, int ldo) {
        fp8_x_ready = fp8_used = i8_used = false;
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
    // rows of m into rows [r_off, r_off + m.rows) of gemm_c ([n][ldo], FP16)
    void gemm_c_rows(const Mat & m, int n, int ldo, int r_off) {
        const size_t rb = m.fmt->row_bytes(m.cols), ab = m.fmt->aux_row_bytes(m.cols);
        const int np = gemm_pad_rows(n, NBUF);
        ck(zero_gemm_padding(v_xh.as<__half>(), n, np, m.cols, stream));
        // int8 only on chunks of >= 1024 tokens: per-chunk weight quantisation does not pay off on the 512-token chunks of
        // 24 GB cards at 256K (3090 192K prompt: int8 641 vs FP16 668 tok/s; 54K: 1760 vs 1445)
        if (i8 && np >= 1024 && m.cols % 4 == 0 && ldo % 4 == 0 && r_off % 4 == 0) {
            // Formats that can quantise their weights to int8 themselves (X3's straight-to-int8 reconstruct with
            // load-time scales) skip the FP16 round trip and the per-chunk weight quantise pass entirely.
            float* sw = fp8_sw.as<float>() + r_off;
            const bool preq = m.fmt->dequant_i8(m, i8->w8(), sw, stream);
            if (!preq) m.fmt->dequant_h(m, gemm_w.as<__half>(), stream);
            if (!fp8_x_ready) {
                i8->quant_x(v_xh.as<__half>(), m.cols, np, m.cols, stream);
                fp8_x_ready = true;
            }
            if (preq) i8->gemm_prequant(m.rows, m.cols, np, gemm_c.as<int32_t>() + r_off, ldo, stream);
            else i8->gemm(gemm_w.as<__half>(), m.rows, m.cols, np, gemm_c.as<int32_t>() + r_off, ldo, sw, stream);
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

    void rot_h(const bf16 * x, __half * xh, int k, int n) {
        prep_input(basis_, x, xh, k, n);
    }
    void nrot_h(const bf16 * x, const float * w, const float * partial, __half * xh, int k, int n) {
        switch (basis_) {
            case Basis::Tq4Rot128: tqllm_cuda::tq4_norm_rotate_h(x, w, partial, cfg.eps, xh, k, n, stream); break;
            case Basis::Tq3Rot32: tqllm_cuda::norm_rotate_h(x, w, partial, cfg.eps, xh, k, n, stream); break;
            case Basis::Plain: tqllm_cuda::norm_h(x, w, partial, cfg.eps, xh, k, n, stream); break;
        }
    }
    // logits [n][vocab] of a 4-bit (W4A16) LM head
    // BF16 [n][k] -> FP16 matmul input in `b`
    void prep_input(Basis b, const bf16 * x, __half * xh, int k, int n) {
        switch (b) {
            case Basis::Tq4Rot128: tqllm_cuda::tq4_rotate_h(x, xh, k, n, stream); break;
            case Basis::Tq3Rot32: tqllm_cuda::tq3_rotate_h(x, xh, k, n, stream); break;
            case Basis::Plain: tqllm_cuda::bf16_to_h(x, xh, int64_t(n) * k, stream); break;
        }
    }
    // logits [n][vocab] from BF16 hidden states when the LM head is a registered weight format
    void head_mv(const bf16 * in, int n) {
        const int H = int(cfg.hidden), mc = lm_head.fmt->max_batch_cols();
        for (int c0 = 0; c0 < n; c0 += mc) {  // chunks of the format's batch width (per-column results unchanged)
            const int nc = std::min(mc, n - c0);
            prep_input(lm_head.fmt->basis(), in + size_t(c0) * H, v_xh.as<__half>(), H, nc);
            lm_head.fmt->matvec_f32(lm_head, v_xh.as<__half>(), v_logits.as<float>() + size_t(c0) * lm_head.rows, nc,
                                    stream);
        }
    }
    int mix_rows(const Layer & L) const { return L.w_mix.rows + (L.w_ba.p ? L.w_ba.rows : 0); }
    // [qkv|z] (u4h) and [b|a] (BF16) into the [qkv|z|b|a] mix rows of T
    void mix_mm(const Layer & L, bf16 * T, int n) {
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
                tqllm_cuda::h_to_bf16(v_xh.as<__half>(), v_xb.as<bf16>(), int64_t(n) * L.w_ba.cols, side);
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
            tqllm_cuda::h_to_bf16(v_xh.as<__half>(), v_xb.as<bf16>(), int64_t(n) * L.w_ba.cols, stream);
            const float alpha = 1.f, beta = 0.f;
            cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, rb, n, L.w_ba.cols, &alpha, L.w_ba.p, CUDA_R_16BF, L.w_ba.cols,
                            v_xb.p, CUDA_R_16BF, L.w_ba.cols, &beta, T + ra, CUDA_R_16BF, rows, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            return;
        }
        mm_h(L.w_mix, v_mixa.as<bf16>(), n);
        mm_h(L.w_ba, v_mixb.as<bf16>(), n);
        ck(cudaMemcpy2DAsync(T, size_t(rows) * 2, v_mixa.p, size_t(ra) * 2, size_t(ra) * 2, n, cudaMemcpyDeviceToDevice, stream));
        ck(cudaMemcpy2DAsync(T + ra, size_t(rows) * 2, v_mixb.p, size_t(rb) * 2, size_t(rb) * 2, n, cudaMemcpyDeviceToDevice, stream));
    }

public:
    void verify(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) override {
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
    // kernel launches are not checked one by one: a launch that fails (the driver or a library cannot allocate on a
    // nearly full device) would silently skip its kernel, so every stream sync also collects the last launch error
    void sync_ck() {
        ck(cudaStreamSynchronize(stream));
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(e));
    }
    bool full_logits = tqllm::debug_value("full_head") != nullptr;  // TQLLM_DEBUG=full_head: full-vocabulary int8 head, the reference for the shortlist
    void set_full_logits(bool on) override { full_logits = on; }
    const float * verify_logits() const override { return v_logits.as<float>(); }
    void set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence) override {
        samp.temperature = temperature;
        samp.presence = presence > 0.f ? presence : 0.f;
        samp.top_p = top_p > 0.f && top_p <= 1.f ? top_p : 1.f;
        samp.top_k = top_k;
        rng.seed(seed);
    }

    // positions [pos, pos + n) are written next: evict old granules to host if the device window needs room (between
    // launches; remaps keep every address, so captured graphs stay valid) and publish the bytes each layer stages back
    void kv_stage_window(uint32_t pos, int n) {
        if (kv_cmp && int64_t(pos) < cmp_pos) cmp_reset();  // a new request rewrites the compacted prefix
        if (!kv_pool->stage_ensure(int64_t(pos), int64_t(pos) + n))
            throw std::runtime_error("KV host tail: the device window is smaller than a write block");
        if (kv_cmp) ring_used = std::min(kRing, ring_used + n), ring_end = int((pos + uint32_t(n)) % kRing);
        const int64_t b[2] = {int64_t(kv_pool->staged_bytes(true)), int64_t(kv_pool->staged_bytes(false))};
        ck(cudaMemcpyAsync(d_stage_bytes.p, b, sizeof(b), cudaMemcpyHostToDevice, stream));
    }
    // every KV write of n positions at pos: staging window (host tail) or the device-resident TqCache bookkeeping
    void kv_track(uint32_t pos, int n) {
        if (kv_staged) kv_stage_window(pos, n);
        else if (kv_cmp) {
            if (int64_t(pos) < cmp_pos) cmp_reset();  // a new request rewrites the compacted prefix
            ring_used = std::min(kRing, ring_used + n), ring_end = int((pos + uint32_t(n)) % kRing);
        }
        kv_end = pos + uint32_t(n);
    }
    void kv_prefetch(uint32_t a) {  // attention layer a's old positions host -> its staging buffer (parity a & 1)
        const auto * nb = d_stage_bytes.as<int64_t>();
        const auto * off = kv_cmp ? d_stage_off.as<int64_t>() : nullptr;  // the compacted prefix is not staged
        tqllm_cuda::kv_stage_copy(kv_pool->host_copy(int(a), true), key_cache[a].p, nb, kv_blocks, kv_side, off);
        tqllm_cuda::kv_stage_copy(kv_pool->host_copy(int(a), false), value_cache[a].p, nb + 1, kv_blocks, kv_side, off ? off + 1 : nullptr);
        ck(cudaEventRecord(kv_ready[a & 1], kv_side));
    }

    void verify_async(const uint32_t * tokens, const int * pos_host, int n, uint32_t pos,
                      uint32_t * next, bool copy_all) {
        if (pos == 0) reset_state(), pc_drop();
        kv_track(pos, n);
        if (!(defer && tree_step && n <= NVER)) flush_pending();
        // every verify-sized block (decode included) goes through the two-part attention when the KV format has it: a tree
        // with its parents, anything else as a chain. One arithmetic for plain decode, chain / MTP / tree verify, so greedy
        // speculative output equals plain decoding (one-part causal attention groups the sums differently: tests/tree.cu)
        vpar = n <= NVER && chain2 ? (tree_mode && tree_step ? d_tpar.as<int>() : d_chain.as<int>()) : nullptr;
        vdep = !vpar ? nullptr : tree_mode && tree_step ? d_tdep.as<int>() : d_cdep.as<int>();
        uint32_t bucket = 256;
        while (bucket < pos + n) bucket *= 2;
        bucket = std::min(bucket, context_);
        if (tokens) ck(cudaMemcpyAsync(v_tokens.p, tokens, size_t(n) * 4, cudaMemcpyHostToDevice, stream));  // else on the device
        ck(cudaMemcpyAsync(v_pos.p, pos_host, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        if (gen.rdelta == 0) {  // prefill chunks are queued unsynchronised: copy from the caller's staged positions
            ck(cudaMemcpyAsync(v_rpos.p, pos_host, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        } else {  // decode after an image (one verify per host round trip)
            ck(cudaStreamSynchronize(stream));
            h_rpos[0] = pos_host[0] - gen.rdelta;
            h_rpos[1] = pos_host[1] - gen.rdelta;
            ck(cudaMemcpyAsync(v_rpos.p, h_rpos, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        }
        // captured graphs bake in the slot's cache and state pointers, so the slot is part of the key
        const uint64_t key = (uint64_t(n) << 40) | (uint64_t(cur) << 32) | bucket | (uint64_t(n > NVER && prefill_final) << 63) |
                             (uint64_t(sampling()) << 62) | (uint64_t(active) << 54) | (uint64_t(tree_step) << 53) |
                             (uint64_t(sp) << 52);
        if (stage_prof && n <= NVER && !mm_chunk && segs.empty()) {
            ck(cudaStreamSynchronize(stream));
            stage_on = true;
            stage_t = std::chrono::steady_clock::now();
            verify_body(n, int(bucket));
            st("zz_tail");
            stage_on = false;
            ++stage_acc["_steps"].second;
        } else if (n > NVER || mm_chunk) {
            // prefill chunks run eagerly: their sizes and positions vary per prompt, so cached graphs would grow without
            // bound (~12 MB each; a 2-slot server OOMed after ~800 requests), and launch cost is negligible at this size
            verify_body(n, int(bucket));
        } else {
            ck(cudaGraphLaunch(cached_graph(kGraphVerify, key, [&] { verify_body(n, int(bucket)); }), stream));
        }
        if (next) {
            const auto * src = v_tok.as<uint32_t>() + (copy_all || n > NVER ? 0 : n - 1);
            ck(cudaMemcpyAsync(next, src, size_t(copy_all ? n : 1) * 4, cudaMemcpyDeviceToHost, stream));
        }
        last_n = n;
    }

    // ---- shared multi-token layer pieces (verify and MTP) ----
    // Full-attention token mixer on n tokens: XH holds the normalised, rotated input; result (pre-residual) -> M.
    // `li` >= 0 (batched verify): attention layer index; each segment then uses its own slot's cache instead of kc / vc.
    void attn_mixer(const Layer & L, bf16 * kc, bf16 * vc, int n, int max_seq, int li = -1) {
        const int nh = int(cfg.heads), hd = int(cfg.head_dim);
        auto *T = v_in.as<bf16>(), *O = v_o.as<bf16>(), *M = v_m.as<bf16>();
        auto * XH = v_xh.as<__half>();
        const bool fuse = basis_ == Basis::Tq3Rot32 || basis_ == Basis::Tq4Rot128;  // gate + wo input rotation in one kernel
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

    // Batched verify: fn(i) for every segment, segment 0 on the main stream and the others on their own branches
    // (graph-captured fork / join; `stream` points at the branch while fn runs). Segments touch disjoint slot state,
    // columns and partial buffers.
    template <class F> void for_segs(F && fn) {
        if (segs.size() < 2 || !segs_fit_part()) {
            for (size_t i = 0; i < segs.size(); ++i) fn(i);
            return;
        }
        if (!seg_fork) ck(cudaEventCreateWithFlags(&seg_fork, cudaEventDisableTiming));
        while (seg_side.size() + 1 < segs.size()) {
            seg_side.emplace_back();
            seg_join.emplace_back();
            ck(cudaStreamCreateWithFlags(&seg_side.back(), cudaStreamNonBlocking));
            ck(cudaEventCreateWithFlags(&seg_join.back(), cudaEventDisableTiming));
        }
        const cudaStream_t main = stream;
        struct Restore {
            cudaStream_t & s;
            cudaStream_t v;
            ~Restore() { s = v; }
        } restore{stream, main};
        ck(cudaEventRecord(seg_fork, main));
        for (size_t i = 1; i < segs.size(); ++i) {
            ck(cudaStreamWaitEvent(seg_side[i - 1], seg_fork, 0));
            stream = seg_side[i - 1];
            fn(i);
            ck(cudaEventRecord(seg_join[i - 1], stream));
        }
        stream = main;
        fn(0);
        for (size_t i = 1; i < segs.size(); ++i) ck(cudaStreamWaitEvent(main, seg_join[i - 1], 0));
    }

    // Concurrent segments each take an equal share of v_part; it must hold one verify launch (<= 8 queries).
    void ensure_mq_split() {
        if (mq_split) return;
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        mq_split = std::clamp(kvf->split_factor() * sms / int(cfg.kvheads), 1, 256);
    }
    bool segs_fit_part() {
        ensure_mq_split();
        return size_t(tqllm_cuda::attention_mq_partial_floats(int(cfg.heads), 8, mq_split)) <= part_floats / segs.size();
    }
    int64_t kv_row_bytes(bool key) const {  // one position, all KV heads, in the cache's format
        const int kvh = int(cfg.kvheads), hd = int(cfg.head_dim);
        return int64_t(key ? kvf->key_bytes(1, kvh, hd) : kvf->value_bytes(1, kvh, hd));
    }
    float * part_of(size_t i) {
        return segs.size() > 1 && segs_fit_part() ? v_part.as<float>() + i * (part_floats / segs.size()) : nullptr;
    }

    // Per-sequence part of the attention mixer on n tokens at positions dpos[0]..: q/k norms, RoPE, K/V write +
    // attention, output gate. T: [n][rows] fused projections (token i: q|gate, then k, then v); Q / O: [n][nh * hd].
    // rpos: RoPE positions when they differ from the cache positions dpos (after an image); mpos: per-token M-RoPE table
    void attn_core(const Layer & L, bf16 * T, int rows, bf16 * Q, bf16 * O, bf16 * kc, bf16 * vc, int n,
                   const int * dpos, const int * par = nullptr, const int * dep = nullptr, __half * xh = nullptr,
                   float * part = nullptr, const int * rpos = nullptr, const int * mpos = nullptr) {
        const int nh = int(cfg.heads), kvh = int(cfg.kvheads), hd = int(cfg.head_dim);
        const int * rp = rpos ? rpos : dpos;
        const int3 sec = {cfg.msec[0], cfg.msec[1], cfg.msec[2]};
        bf16 * K0 = T + 2 * nh * hd;
        // q / k RMSNorm per head, then RoPE on the first rot_dim dims
        tqllm_cuda::rms_rope_multi(T, rows, 2 * hd, L.q_norm, Q, nh * hd, hd, hd, nh, n, cfg.eps, int(cfg.rot_dim), rp,
                                   cfg.theta, stream, dep, mpos, sec);
        tqllm_cuda::rms_rope_multi(K0, rows, hd, L.k_norm, K0, rows, hd, hd, kvh, n, cfg.eps, int(cfg.rot_dim), rp,
                                   cfg.theta, stream, dep, mpos, sec);
        ensure_mq_split();
        if (fq_k) tqllm_cuda::kv_fakequant(K0, rows, n, kvh, hd, fq_k, stream);
        if (fq_v) tqllm_cuda::kv_fakequant(K0 + kvh * hd, rows, n, kvh, hd, fq_v, stream);
        if (!part) part = v_part.as<float>();
        const bool cmp = kv_cmp && cmp_layer >= 0;
        if (cmp) tqllm_cuda::q_ring_write(Q, n, nh * hd, dpos, kRing, q_ring[size_t(cmp_layer)].as<bf16>(), stream);
        kvf->write_and_attend({K0, rows, n, kc, vc, dpos, nh, kvh, hd, NVER, mq_split, int64_t(context_), Q, O, part,
                               1.f / std::sqrt(float(hd)), stream, par, dep, cmp ? &cmp_seg[size_t(cmp_layer)] : nullptr});
        if (xh && basis_ == Basis::Tq4Rot128) tqllm_cuda::tq4_sigmoid_gate_rot_h(O, T, rows, n, nh, hd, xh, stream);
        else if (xh) tqllm_cuda::sigmoid_gate_rot_h(O, T, rows, n, nh, hd, xh, stream);  // + the wo input rotation
        else tqllm_cuda::sigmoid_gate_multi(O, T, rows, n, nh, hd, stream);
    }

    // M += R (residual), then FFN on M, X = M + FFN(M); refreshes the partial sums for the next norm.
    // resid: the residual stream (default: the copy in v_r); the verify passes X itself, which the mixer leaves intact
    void ffn_block(const Layer & L, int n, const bf16 * resid = nullptr) {
        const int H = int(cfg.hidden), F = int(cfg.ffn);
        auto *X = v_x.as<bf16>(), *M = v_m.as<bf16>(), *R = v_r.as<bf16>(), *T = v_in.as<bf16>();
        auto * ACT = v_act.as<bf16>();
        auto * XH = v_xh.as<__half>();
        tqllm_cuda::add_sumsq(M, resid ? resid : R, v_pm.as<float>(), n * H, stream);
        nrot_h(M, L.post_norm, v_pm.as<float>(), XH, H, n);
        if (cfg.moe) {
            const int E = int(cfg.expert_count), K = int(cfg.expert_used), FE = int(cfg.expert_ff), FS = int(cfg.shared_ff);
            tqllm_cuda::moe_router(L.router, L.sh_gate, XH, moe_logits.as<float>(), H, E, n, stream);
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

    // logits + per-column argmax (-> v_tok) of the n hidden vectors in X, normalised with `norm_w`.
    // Head on columns [col0, col0 + n) of v_x; writes v_tok (and the sampling top-16) at the same columns.
    void head_argmax(const float * norm_w, int n, int col0 = 0) {
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
        if (two_stage && head4.p && has_head8() && (n <= 8 || !segs.empty())) {  // 4-bit head -> top-16 -> exact int8 re-score
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

    // Column i's sampling distribution: top-16 by exact logit, presence penalty against `recent` (the last output
    // tokens before this column), temperature, top-k (<= 16), top-p.
    struct Dist { uint32_t id[16]; double p[16]; int n = 0; };
    Dist dist(int col, const std::vector<uint32_t> & recent = {}) const {
        Dist d;
        int ord[16];
        for (int i = 0; i < 16; ++i) ord[i] = i;
        float lg[16];
        const uint32_t * id = h_sid.data() + size_t(col) * 16;
        std::copy_n(h_slg.data() + size_t(col) * 16, 16, lg);
        if (samp.presence > 0.f) {
            const size_t r0 = recent.size() > size_t(kPresenceLastN) ? recent.size() - kPresenceLastN : 0;
            for (int i = 0; i < 16; ++i)
                if (std::find(recent.begin() + long(r0), recent.end(), id[i]) != recent.end()) lg[i] -= samp.presence;
        }
        std::sort(ord, ord + 16, [&](int a, int b) { return lg[a] > lg[b]; });
        const int k = samp.top_k > 0 ? std::min(samp.top_k, 16) : 16;
        double sum = 0;
        for (int i = 0; i < k; ++i) {
            d.id[i] = id[ord[i]];
            d.p[i] = std::exp(double(lg[ord[i]] - lg[ord[0]]) / samp.temperature);
            sum += d.p[i];
        }
        double cum = 0;
        d.n = k;
        for (int i = 0; i < k; ++i) {
            cum += d.p[i] / sum;
            if (cum >= samp.top_p) { d.n = i + 1; break; }
        }
        double s2 = 0;
        for (int i = 0; i < d.n; ++i) s2 += d.p[i];
        for (int i = 0; i < d.n; ++i) d.p[i] /= s2;
        return d;
    }
    uint32_t draw(const Dist & d, uint32_t exclude = UINT32_MAX) {
        double total = 0;
        for (int i = 0; i < d.n; ++i) total += d.id[i] == exclude ? 0 : d.p[i];
        double u = std::uniform_real_distribution<double>(0, total)(rng);
        for (int i = 0; i < d.n; ++i) {
            if (d.id[i] == exclude) continue;
            if ((u -= d.p[i]) <= 0) return d.id[i];
        }
        for (int i = d.n - 1; i >= 0; --i) if (d.id[i] != exclude) return d.id[i];
        return d.id[0];
    }

    // token embeddings of n device-resident ids -> Y [n][hidden] (BF16)
    void embed_multi(const uint32_t * dtok, int n, bf16 * Y) {
        const int H = int(cfg.hidden);
        if (embed.type == GGML_TQ3_4S) tqllm_cuda::embed_multi_tq3(embed.p, dtok, n, Y, H, stream);
        else if (embed.type == GGML_Q6_K) tqllm_cuda::embed_multi_q6k(embed.p, dtok, n, Y, H, stream);
        else if (embed.type == GGML_Q8_0) tqllm_cuda::embed_multi_q8_0(embed.p, dtok, n, Y, H, stream);
        else if (embed.type == GGML_Q4_K) tqllm_cuda::embed_multi_q4k(embed.p, dtok, n, Y, H, stream);
        else if (embed.type == GGML_Q3_K) tqllm_cuda::embed_multi_q3k(embed.p, dtok, n, Y, H, stream);
        else if (embed.type == GGML_IQ3_S) tqllm_cuda::embed_multi_iq3s(embed.p, dtok, n, Y, H, stream);
        else if (embed.type == GGML_Q5_K) tqllm_cuda::embed_multi_q5k(embed.p, dtok, n, Y, H, stream);
        else throw std::runtime_error("multi-token path needs a TQ3_4S, Q6_K, Q8_0, Q4_K, Q3_K, Q5_K or IQ3_S embedding table");
    }

    void verify_body(int n, int max_seq) {
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


    // Copies the input of target layer l (if it is a DFlash feature layer) into feats [n][n_feat * H].
    void capture_feature(uint32_t l, const bf16 * x, bf16 * feats, int n) {
        if (!dfl.present) return;
        const size_t H = cfg.hidden, nf = dfl.target_ids.size();
        for (size_t k = 0; k < nf; ++k)
            if (uint32_t(dfl.target_ids[k]) == l)
                ck(cudaMemcpy2DAsync(feats + k * H, nf * H * 2, x, H * 2, H * 2, size_t(n), cudaMemcpyDeviceToDevice,
                                     stream));
    }

    // FP32 host tensor(s) -> TQ3_4S on the device (row-concatenated), repacked for the fast kernels.
    // Q4_K -> u4h: every 32-weight sub-block keeps its codes, scale d * sc and min -dmin * m (FP16).
    Mat q4k_to_u4h(const GgufFile & f, const std::vector<std::string> & names, int cols) {
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

    Mat to_tq3(const GgufFile & f, const std::vector<std::string> & names, int cols) {
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

    const float * f32_of(const GgufFile & f, const std::string & name, uint64_t n) {
        const auto & t = gt(f, name);
        std::vector<float> v(n);
        const auto raw = f.read(t);
        dequantize_host(t.type, raw.data(), v.data(), int64_t(n));
        return put_f32(v);
    }

    const bf16 * bf16_table(const GgufFile & f, const std::string & name, uint64_t n) {
        const auto & t = gt(f, name);
        std::vector<float> v(n);
        const auto raw = f.read(t);
        dequantize_host(t.type, raw.data(), v.data(), int64_t(n));
        std::vector<bf16> b(n);
        for (uint64_t i = 0; i < n; ++i) b[i] = __float2bfloat16_rn(v[i]);
        return static_cast<const bf16 *>(put(b.data(), b.size() * 2));
    }

    // [rows][rank] table (GGUF ne = {rank, rows}) as BF16 [cfg.vocab][rank], rows past the file's vocabulary zero
    const bf16 * vocab_table(const GgufFile & f, const std::string & name, int rank) {
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

    // Drafter kernel grids independent of the position (sliding-window ring, no retrieval): drafts and commits can be
    // captured once as CUDA graphs. Retrieval sizes its launches by the position, so it runs eagerly.
    bool dfl_fixed_grids() const { return dfl.ring > 0 && !dfl.retr; }

    // Drafter K/V of the active slot: the sliding-window (or retrieval) ring, else the full context.
    void alloc_dfl_state() {
        const size_t kd = size_t(dfl.kvheads) * dfl.hd, rows = dfl.ring ? size_t(dfl.ring) : context_;
        for (size_t l = 0; l < dfl.L.size(); ++l) {
            dfl_kc.emplace_back(rows * kd * 2);
            dfl_vc.emplace_back(rows * kd * 2);
        }
    }

    void load_dflash(const std::string & path) {
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
    // 4-bit copy of head8: the two-stage verify head's first pass and the DFlash2 full-vocabulary draft head
    // logits of the draft-vocabulary rows (dsub_ids order) for n FP16 inputs
    void dsub_matvec(const __half * x, float * y, int n) {
        if (dsub_in_head4)
            tqllm_cuda::u4h_matvec_ids(head4.as<uint8_t>(), head4s.as<__half2>(), dsub_ids.as<uint32_t>(), x, y, dsub_rows,
                                       int(cfg.hidden), n, stream);
        else
            tqllm_cuda::u4h_matvec(dsub_w.as<uint8_t>(), dsub_s.as<__half2>(), x, y, dsub_rows, int(cfg.hidden), n, stream);
    }
    void build_head4() {
        const int64_t n = int64_t(lm_head.rows) * cfg.hidden;
        head4 = Buf(size_t(n / 2));
        head4s = Buf(size_t(n / 32) * 4);
        tqllm_cuda::u4h_from_i8h(head8.as<int8_t>(), head8s.as<__half>(), head4.as<uint8_t>(), head4s.as<__half2>(), n, stream);
        tqllm_cuda::u4h_tune(head4.as<uint8_t>(), head4s.as<__half2>(), lm_head.rows, int(cfg.hidden), stream);
        h2_vals = Buf(size_t(NB) * 16 * 4);
        h2_idx = Buf(size_t(NB) * 16 * 4);
        h2_scr = Buf(size_t(NB) * 64 * 16 * 8);
        sync_ck();
    }
    // Pinned host memory mapped into the device (low-VRAM cards: embeddings and the int8 head are read sparsely)
    const void * to_mapped_host(const void * dev, size_t bytes) {
        void * h = nullptr;
        ck(cudaHostAlloc(&h, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        ck(cudaMemcpy(h, dev, bytes, cudaMemcpyDeviceToHost));
        host_blocks.push_back(h);
        void * d = nullptr;
        ck(cudaHostGetDevicePointer(&d, h, 0));
        return d;
    }
    // Pinned mapped-host copy of raw bytes, returning the device-accessible (UVA) pointer: the -ncmoe
    // expert loader. The weight-format load() repack kernel and the decode expert kernels read m.p / m.aux
    // through this pointer over PCIe (same mechanism as embed_host / head8_to_host); no kernel changes.
    const void * host_put(const void * host, size_t bytes) {
        void * h = nullptr;
        ck(cudaHostAlloc(&h, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        std::memcpy(h, host, bytes);
        host_blocks.push_back(h);
        void * d = nullptr;
        ck(cudaHostGetDevicePointer(&d, h, 0));
        return d;
    }
    Upload host_uploader() {
        return [this](const void * h, size_t b) { return host_put(h, b); };
    }
    std::vector<void *> host_blocks;

    // Verify rescoring reads 16 candidate rows per column, and the drafter uses head4 / its subset: the full int8 head
    // (1.27 GB for Qwen3.8) can live in pinned host memory the device reads over PCIe (~0.6 MB per 8-column round).
    void head8_to_host() {
        const size_t bw = size_t(lm_head.rows) * cfg.hidden, bs = bw / 16 * 2;
        ck(cudaHostAlloc(&h8_host, bw + bs, cudaHostAllocMapped | cudaHostAllocPortable));
        ck(cudaMemcpy(h8_host, head8.p, bw, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(static_cast<char *>(h8_host) + bw, head8s.p, bs, cudaMemcpyDeviceToHost));
        void * d = nullptr;
        ck(cudaHostGetDevicePointer(&d, h8_host, 0));
        h8_dev_w = static_cast<const int8_t *>(d);
        h8_dev_s = reinterpret_cast<const __half *>(static_cast<const char *>(d) + bw);
        head8 = Buf();
        head8s = Buf();
        std::fprintf(stderr, "tqserve: LM head int8 copy in host memory (%.2f GB of VRAM freed)\n", double(bw + bs) / 1e9);
    }
    int L_conv_rows() const { return dfl.L.empty() ? 0 : dfl.L[0].conv_attn.rows; }

    // generic TQ3 matvec on n rows of a BF16 input (rotates into v_xh first)
    void mmx(const Mat & m, const bf16 * in, bf16 * out, int n) {
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

    // Pageable source on purpose: the copy is staged before cudaMemcpyAsync returns, so back-to-back calls cannot
    // race on a shared pinned buffer (the injection after a verify is not followed by a sync).
    void set_dpos(uint32_t pos) {
        const int v[2] = {int(pos), int(pos) + 1};
        ck(cudaMemcpyAsync(d_pos.p, v, sizeof(v), cudaMemcpyHostToDevice, stream));
    }

    // The inject projections as matvec batches of the format's max_batch_cols(): per-column results are identical to
    // narrower calls (the batched-matvec contract), while each weight is read once per batch of tokens instead of once
    // per 8-token group - the whole cost of the old prefill injection.
    void inj_mm(const Mat & m, const bf16 * in, bf16 * out, int n) {
        if (n > DFlash::injBatch) throw std::runtime_error("dflash inject: batch wider than the scratch");
        const int mc = m.fmt->max_batch_cols();
        for (int c0 = 0; c0 < n; c0 += mc) {
            const int nc = std::min(mc, n - c0);
            prep_input(m.fmt->basis(), in + size_t(c0) * m.cols, v_xh.as<__half>(), m.cols, nc);
            m.fmt->matvec(m, v_xh.as<__half>(), out + size_t(c0) * m.rows, nc, stream);
        }
    }
    // Encodes n rows of target features (device [n][nf*H]) for positions pos.. and writes every drafter layer's K/V.
    void dflash_inject(const bf16 * feats, int n, uint32_t pos) {
        set_dpos(pos);
        dflash_inject_body(feats, n);
    }
    void dflash_inject_body(const bf16 * feats, int n) {  // positions from d_pos
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

    // One DFlash2 block [anchor, MASK x (block-1)] at positions pos..; returns up to block-1 greedy drafts.
    // Drafts for several slots in one pass over the drafter weights (block of B rows per slot); the position-dependent
    // pieces (RoPE, K/V write, attention, the block's causal conv, lattice) run per slot. Needs the K/V ring and a
    // 4-bit draft head; otherwise empty (callers draft per slot).
    struct DraftReq { int slot; uint32_t anchor, pos; int want; };
    // tree_cols: instead of chain drafts on the host, build each slot's token tree on the device (dflash_tree) into
    // v_tokens / d_tpar / d_tdep at column tree_cols[i] with rq[i].want nodes; no host sync
    // chain_cols: the same for the greedy chain (min(want, block - 1) tokens at column chain_cols[i] + 1); no host sync
    bool dflash_draft_batch(const std::vector<DraftReq> & rq, std::vector<std::vector<uint32_t>> & out,
                            const std::vector<int> * tree_cols = nullptr, const std::vector<int> * chain_cols = nullptr) {
        const bool subset = mtp.draft_head.p != nullptr && !dsub_rows;
        if (rq.size() < 2 || !dfl_fixed_grids() || dfl.dspark || subset || !(dsub_rows || head4.p || lm_head.fmt))
            return false;
        const int H = int(cfg.hidden), B = dfl.block, qd = dfl.heads * dfl.hd, kd = dfl.kvheads * dfl.hd;
        const int ns = int(rq.size()), DB = ns * B;
        std::vector<uint32_t> blk(size_t(DB), dfl.mask_id);
        std::vector<int> pos(2 * size_t(ns));
        for (int i = 0; i < ns; ++i) {
            blk[size_t(i) * B] = rq[size_t(i)].anchor;
            pos[2 * i] = int(rq[size_t(i)].pos);
            pos[2 * i + 1] = int(rq[size_t(i)].pos) + 1;
        }
        ck(cudaMemcpyAsync(d_tok.p, blk.data(), blk.size() * 4, cudaMemcpyHostToDevice, stream));  // pageable: staged
        ck(cudaMemcpyAsync(d_pos.p, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        auto body = [&] {
            auto *X = d_x.as<bf16>(), *R = d_r.as<bf16>(), *A = d_a.as<bf16>(), *Bb = d_b.as<bf16>();
            auto *T = v_in.as<bf16>(), *O = v_o.as<bf16>(), *Q = v_q.as<bf16>(), *ACT = v_act.as<bf16>();
            auto *DYN = d_dyn.as<bf16>(), *DYNF = d_dynf.as<bf16>();
            const int * dp = d_pos.as<int>();
            embed_multi(d_tok.as<uint32_t>(), DB, X);
            const int cr = L_conv_rows();
            auto conv = [&](const bf16 * in, const bf16 * dyn, const float * base, int side, bf16 * y) {
                for (int i = 0; i < ns; ++i)
                    tqllm_cuda::dflash_conv(in + size_t(i) * B * H, dyn + size_t(i) * B * cr, cr, base, side,
                                            y + size_t(i) * B * H, B, H, dfl.group, stream);
            };
            for (size_t l = 0; l < dfl.L.size(); ++l) {
                const auto & L = dfl.L[l];
                ck(cudaMemcpyAsync(R, X, size_t(DB) * H * 2, cudaMemcpyDeviceToDevice, stream));
                tqllm_cuda::rms_f_multi(X, H, 0, L.attn_norm, A, H, 0, H, 1, DB, cfg.eps, stream);
                mmx(L.conv_attn, A, DYN, DB);
                conv(A, DYN, L.attn_conv_base, 0, Bb);
                mmx(L.w_qkv, Bb, T, DB);
                const int rows = L.w_qkv.rows;
                bf16 * K0 = T + qd;
                tqllm_cuda::rms_f_multi(T, rows, dfl.hd, L.q_norm, Q, qd, dfl.hd, dfl.hd, dfl.heads, DB, cfg.eps, stream);
                tqllm_cuda::rms_f_multi(K0, rows, dfl.hd, L.k_norm, K0, rows, dfl.hd, dfl.hd, dfl.kvheads, DB, cfg.eps,
                                        stream);
                for (int i = 0; i < ns; ++i) {
                    const int sl = rq[size_t(i)].slot;
                    bf16 *Qi = Q + size_t(i) * B * qd, *Ki = K0 + size_t(i) * B * rows, *Oi = O + size_t(i) * B * qd;
                    tqllm_cuda::rope_partial_multi(Qi, qd, B, dfl.heads, dfl.hd, dfl.hd, dp + 2 * i, dfl.theta, stream);
                    tqllm_cuda::rope_partial_multi(Ki, rows, B, dfl.kvheads, dfl.hd, dfl.hd, dp + 2 * i, dfl.theta, stream);
                    bf16 *kc = dkc_of(sl)[l].as<bf16>(), *vc = dvc_of(sl)[l].as<bf16>();
                    tqllm_cuda::cache_write_multi(Ki, Ki + kd, rows, B, kc, vc, dp + 2 * i, dfl.kvheads, dfl.hd, stream,
                                                  dfl.ring);
                    tqllm_cuda::attention_gen128(Qi, kc, vc, Oi, v_part.as<float>(), dp + 2 * i, B, int(context_),
                                                 dfl.heads, dfl.kvheads, 1.f / std::sqrt(float(dfl.hd)), false,
                                                 dfl.window, stream, dfl.ring);
                }
                mmx(L.wo, O, A, DB);
                conv(A, DYN, L.attn_conv_base, 1, Bb);
                tqllm_cuda::residual(Bb, R, DB * H, stream);
                tqllm_cuda::rms_f_multi(Bb, H, 0, L.ffn_norm, A, H, 0, H, 1, DB, cfg.eps, stream);
                mmx(L.conv_ffn, A, DYNF, DB);
                conv(A, DYNF, L.ffn_conv_base, 0, X);
                mmx(L.w_gu, X, T, DB);
                tqllm_cuda::swiglu_multi(T, DB, int(cfg.ffn), ACT, stream);
                mmx(L.down, ACT, A, DB);
                conv(A, DYNF, L.ffn_conv_base, 1, X);
                tqllm_cuda::residual(X, Bb, DB * H, stream);
            }
            auto * TE = d_te.as<bf16>();
            tqllm_cuda::rms_f_multi(X, H, 0, dfl.out_norm, TE, H, 0, H, 1, DB, cfg.eps, stream);
            const int head_rows = dsub_rows ? dsub_rows : lm_head.rows;
            if ((dsub_rows && !dsub_mat.p) || (!dsub_rows && head4.p)) {
                tqllm_cuda::bf16_to_h(TE, v_xh.as<__half>(), DB * H, stream);
                for (int c0 = 0; c0 < DB; c0 += 32) {  // one pass over the 4-bit head per 32 rows
                    const __half * xc = v_xh.as<__half>() + size_t(c0) * H;
                    float * yc = v_logits.as<float>() + size_t(c0) * head_rows;
                    if (dsub_rows) dsub_matvec(xc, yc, std::min(32, DB - c0));
                    else tqllm_cuda::u4h_matvec(head4.as<uint8_t>(), head4s.as<__half2>(), xc, yc, head_rows, H,
                                                std::min(32, DB - c0), stream);
                }
            } else {  // LM head (or its draft-vocabulary rows) in a registered weight format (TQ4, W4A16, ...)
                const Mat & hm = dsub_mat.p ? dsub_mat : lm_head;
                const int mc = hm.fmt->max_batch_cols();
                for (int c0 = 0; c0 < DB; c0 += mc) {
                    const int nc = std::min(mc, DB - c0);
                    prep_input(hm.fmt->basis(), TE + size_t(c0) * H, v_xh.as<__half>(), H, nc);
                    hm.fmt->matvec_f32(hm, v_xh.as<__half>(), v_logits.as<float>() + size_t(c0) * head_rows, nc, stream);
                }
            }
            tqllm_cuda::topk16_big(v_logits.as<float>(), head_rows, DB, d_vals.as<float>(), d_idx.as<uint32_t>(),
                                   d_topk.p, stream);
            if (dsub_rows) tqllm_cuda::map_ids(d_idx.as<uint32_t>(), dsub_ids.as<uint32_t>(), DB * 16, stream);
            mmx(dfl.sel_hidden, TE, d_gate.as<bf16>(), DB);
            for (int i = 0; i < ns; ++i)
                tqllm_cuda::dflash_lattice(d_idx.as<uint32_t>() + size_t(i) * B * 16, d_vals.as<float>() + size_t(i) * B * 16,
                                           d_gate.as<bf16>() + size_t(i) * B * dfl.rank, dfl.rank, dfl.sel_next,
                                           dfl.sel_prev, 0, B, dfl.rank, d_score.as<float>() + size_t(i) * B * 256,
                                           stream, d_tok.as<uint32_t>() + size_t(i) * B);
        };
        uint64_t key = 1469598103934665603ull;
        for (const auto & r : rq) key = (key ^ uint64_t(r.slot + 1)) * 1099511628211ull;
        ck(cudaGraphLaunch(cached_graph(kGraphDraft, key, [&] {
            wide_mm = true;
            struct Reset { bool & f; ~Reset() { f = false; } } reset{wide_mm};
            body();
        }), stream));
        if (tree_cols) {
            for (int i = 0; i < ns; ++i) {
                const int c0 = (*tree_cols)[size_t(i)];
                tqllm_cuda::dflash_tree(d_idx.as<uint32_t>() + size_t(i) * B * 16, d_score.as<float>() + size_t(i) * B * 256,
                                        d_tok.as<uint32_t>() + size_t(i) * B, B, rq[size_t(i)].want,
                                        v_tokens.as<uint32_t>() + c0, d_tpar.as<int>() + c0, d_tdep.as<int>() + c0, stream);
            }
            return true;
        }
        if (chain_cols) {
            for (int i = 0; i < ns; ++i)
                tqllm_cuda::dflash_chain(d_idx.as<uint32_t>() + size_t(i) * B * 16, d_score.as<float>() + size_t(i) * B * 256,
                                         std::min(rq[size_t(i)].want, B - 1),
                                         v_tokens.as<uint32_t>() + (*chain_cols)[size_t(i)] + 1, stream);
            return true;
        }
        std::vector<uint32_t> idx(size_t(DB) * 16);
        std::vector<float> score(size_t(DB) * 256);
        ck(cudaMemcpyAsync(idx.data(), d_idx.p, idx.size() * 4, cudaMemcpyDeviceToHost, stream));
        ck(cudaMemcpyAsync(score.data(), d_score.p, score.size() * 4, cudaMemcpyDeviceToHost, stream));
        sync_ck();
        out.assign(size_t(ns), {});
        for (int s2 = 0; s2 < ns; ++s2) {
            int pred = 0;
            const size_t o = size_t(s2) * B;
            for (int i = 1; i < B && int(out[size_t(s2)].size()) < rq[size_t(s2)].want; ++i) {
                const float * row = score.data() + ((o + i) * 16 + pred) * 16;
                pred = int(std::max_element(row, row + 16) - row);
                out[size_t(s2)].push_back(idx[(o + i) * 16 + pred]);
            }
        }
        return true;
    }

    // launch_only: queue the draft (lattice left in d_idx / d_score) and return without synchronising
    // DSpark drafter forward of the nb-token block at d_pos (tokens in d_tok) -> the lattice rows 1 .. nb (top-16 ids /
    // logits in d_idx / d_vals, Markov scores in d_score); bidirectional attention over [0, pos + nb).
    void dspark_forward(uint32_t pos) {
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

    std::vector<uint32_t> dflash_draft(uint32_t anchor, uint32_t pos, int want, std::vector<uint32_t> * idx_out = nullptr,
                                       std::vector<float> * score_out = nullptr, bool launch_only = false) {
        const int H = int(cfg.hidden), B = dfl.block, qd = dfl.heads * dfl.hd, kd = dfl.kvheads * dfl.hd;
        set_dpos(pos);
        const int NT = dfl.dspark ? dfl.nb : B;  // drafter tokens in the block
        std::vector<uint32_t> blk(size_t(NT), dfl.mask_id);
        blk[0] = anchor;
        ck(cudaMemcpyAsync(d_tok.p, blk.data(), size_t(NT) * 4, cudaMemcpyHostToDevice, stream));
        // Device-resident body (positions in d_pos, tokens / anchor in d_tok): with the K/V ring the kernel grids do not
        // depend on pos, so the whole draft is captured once as a CUDA graph and replayed.
        const bool subset = mtp.draft_head.p != nullptr && !dsub_rows;
        const bool graph = dfl_fixed_grids() && !subset;
        auto body = [&] {
        if (dfl.dspark) {
            dspark_forward(pos);
            return;
        }
        auto *X = d_x.as<bf16>(), *R = d_r.as<bf16>(), *A = d_a.as<bf16>(), *Bb = d_b.as<bf16>();
        auto *T = v_in.as<bf16>(), *O = v_o.as<bf16>(), *Q = v_q.as<bf16>(), *ACT = v_act.as<bf16>();
        auto *DYN = d_dyn.as<bf16>(), *DYNF = d_dynf.as<bf16>();
        embed_multi(d_tok.as<uint32_t>(), B, X);
        uint32_t bucket = 256;
        while (bucket < pos + B) bucket *= 2;
        bucket = std::min(bucket, context_);
        const int cr = L_conv_rows();
        for (size_t l = 0; l < dfl.L.size(); ++l) {
            const auto & L = dfl.L[l];
            ck(cudaMemcpyAsync(R, X, size_t(B) * H * 2, cudaMemcpyDeviceToDevice, stream));
            tqllm_cuda::rms_f_multi(X, H, 0, L.attn_norm, A, H, 0, H, 1, B, cfg.eps, stream);
            mmx(L.conv_attn, A, DYN, B);
            tqllm_cuda::dflash_conv(A, DYN, cr, L.attn_conv_base, 0, Bb, B, H, dfl.group, stream);
            mmx(L.w_qkv, Bb, T, B);
            const int rows = L.w_qkv.rows;
            bf16 * K0 = T + qd;
            tqllm_cuda::rms_f_multi(T, rows, dfl.hd, L.q_norm, Q, qd, dfl.hd, dfl.hd, dfl.heads, B, cfg.eps, stream);
            tqllm_cuda::rms_f_multi(K0, rows, dfl.hd, L.k_norm, K0, rows, dfl.hd, dfl.hd, dfl.kvheads, B, cfg.eps, stream);
            tqllm_cuda::rope_partial_multi(Q, qd, B, dfl.heads, dfl.hd, dfl.hd, d_pos.as<int>(), dfl.theta, stream);
            tqllm_cuda::rope_partial_multi(K0, rows, B, dfl.kvheads, dfl.hd, dfl.hd, d_pos.as<int>(), dfl.theta, stream);
            tqllm_cuda::cache_write_multi(K0, K0 + kd, rows, B, dfl_kc[l].as<bf16>(), dfl_vc[l].as<bf16>(),
                                          d_pos.as<int>(), dfl.kvheads, dfl.hd, stream, dfl.ring);
            const int rr = dfl.retr && int(pos) > dfl.recent + 16
                               ? tqllm_cuda::dflash_retrieve_build(Q, dfl_kc[l].as<bf16>(), dfl_vc[l].as<bf16>(), kd, dfl.heads,
                                                                   dfl.kvheads, int(pos), B, dfl.recent, dfl.retr, dfl.theta,
                                                                   dfl_rscore.as<float>(), dfl_rsel.as<int>(), dfl_rk.as<bf16>(),
                                                                   dfl_rv.as<bf16>(), stream, dfl.ring)
                               : -1;
            if (rr >= 0) {  // scratch [retrieved | recent | block]: the query block sits at scratch row rr + recent
                if (l == 0) {
                    retr_vpos = rr + dfl.recent;
                    ck(cudaMemcpyAsync(dfl_rpos.p, &retr_vpos, sizeof(int), cudaMemcpyHostToDevice, stream));
                }
                const int rows = rr + dfl.recent + B;
                tqllm_cuda::attention_gen128(Q, dfl_rk.as<bf16>(), dfl_rv.as<bf16>(), O, v_part.as<float>(), dfl_rpos.as<int>(), B,
                                             (rows + 63) / 64 * 64, dfl.heads, dfl.kvheads, 1.f / std::sqrt(float(dfl.hd)),
                                             false, rows, stream, 0);
            } else {
                tqllm_cuda::attention_gen128(Q, dfl_kc[l].as<bf16>(), dfl_vc[l].as<bf16>(), O, v_part.as<float>(),
                                             d_pos.as<int>(), B, int(bucket), dfl.heads, dfl.kvheads,
                                             1.f / std::sqrt(float(dfl.hd)), false, dfl.window, stream, dfl.ring);
            }
            mmx(L.wo, O, A, B);
            tqllm_cuda::dflash_conv(A, DYN, cr, L.attn_conv_base, 1, Bb, B, H, dfl.group, stream);
            tqllm_cuda::residual(Bb, R, B * H, stream);  // Bb = ffn_inp
            tqllm_cuda::rms_f_multi(Bb, H, 0, L.ffn_norm, A, H, 0, H, 1, B, cfg.eps, stream);
            mmx(L.conv_ffn, A, DYNF, B);
            tqllm_cuda::dflash_conv(A, DYNF, cr, L.ffn_conv_base, 0, X, B, H, dfl.group, stream);
            mmx(L.w_gu, X, T, B);
            tqllm_cuda::swiglu_multi(T, B, int(cfg.ffn), ACT, stream);
            mmx(L.down, ACT, A, B);
            tqllm_cuda::dflash_conv(A, DYNF, cr, L.ffn_conv_base, 1, X, B, H, dfl.group, stream);
            tqllm_cuda::residual(X, Bb, B * H, stream);
        }
        auto * TE = d_te.as<bf16>();
        tqllm_cuda::rms_f_multi(X, H, 0, dfl.out_norm, TE, H, 0, H, 1, B, cfg.eps, stream);
        // logits (draft vocabulary head if configured), top-16 per position
        int head_rows;
        if (subset) {  // draft vocabulary subset (Q6_K rows)
            head_rows = mtp.draft_head.rows;
            tqllm_cuda::bf16_to_f32(TE, v_xf.as<float>(), B * H, stream);
            tqllm_cuda::q6k_matvec(mtp.draft_head.p, v_xf.as<float>(), v_logits.as<float>(), head_rows, H, B, stream);
        } else if (dsub_rows && dsub_mat.p) {  // draft vocabulary in the LM head's format (TQ4)
            head_rows = dsub_rows;
            prep_input(dsub_mat.fmt->basis(), TE, v_xh.as<__half>(), H, B);
            dsub_mat.fmt->matvec_f32(dsub_mat, v_xh.as<__half>(), v_logits.as<float>(), B, stream);
        } else if (dsub_rows) {  // DFlash2 draft vocabulary: 4-bit subset head, ids mapped after top-16
            head_rows = dsub_rows;
            tqllm_cuda::bf16_to_h(TE, v_xh.as<__half>(), B * H, stream);
            dsub_matvec(v_xh.as<__half>(), v_logits.as<float>(), B);
        } else {  // full vocabulary on the int8 LM-head copy (tensor cores)
            head_rows = lm_head.rows;
            tqllm_cuda::bf16_to_h(TE, v_xh.as<__half>(), B * H, stream);
            if (head4.p)
                tqllm_cuda::u4h_matvec(head4.as<uint8_t>(), head4s.as<__half2>(), v_xh.as<__half>(), v_logits.as<float>(),
                                       head_rows, H, B, stream);
            else if (lm_head.fmt)
                head_mv(TE, B);
            else
                tqllm_cuda::i8h_matvec(w8(), s8(), v_xh.as<__half>(), v_logits.as<float>(),
                                       head_rows, H, B, stream);
        }
        tqllm_cuda::topk16_big(v_logits.as<float>(), head_rows, B, d_vals.as<float>(), d_idx.as<uint32_t>(), d_topk.p,
                               stream);
        if (dsub_rows) tqllm_cuda::map_ids(d_idx.as<uint32_t>(), dsub_ids.as<uint32_t>(), B * 16, stream);
        mmx(dfl.sel_hidden, TE, d_gate.as<bf16>(), B);
        if (!subset)  // candidates are vocabulary ids already: score the lattice on the device
            tqllm_cuda::dflash_lattice(d_idx.as<uint32_t>(), d_vals.as<float>(), d_gate.as<bf16>(), dfl.rank, dfl.sel_next,
                                       dfl.sel_prev, anchor, B, dfl.rank, d_score.as<float>(), stream,
                                       d_tok.as<uint32_t>());
        };
        if (graph) {
            if (!dfl_graph) {
                cudaGraph_t g = tqllm::capture_graph(stream, [&] {
                    wide_mm = true;  // drafter formats with an 8-column matvec (TQ4) take the block through chunked matvecs
                    struct Reset { bool & f; ~Reset() { f = false; } } reset{wide_mm};
                    body();
                });
                dfl_graph = tqllm::instantiate_graph(g);
            }
            ck(cudaGraphLaunch(dfl_graph, stream));
        } else {
            wide_mm = true;
            body();
            wide_mm = false;
        }
        if (launch_only) {
            if (subset) throw std::runtime_error("device-side drafting needs the device draft vocabulary");
            return {};
        }
        std::vector<uint32_t> idx(size_t(B) * 16);
        ck(cudaMemcpyAsync(idx.data(), d_idx.p, idx.size() * 4, cudaMemcpyDeviceToHost, stream));
        if (subset) {  // draft-vocabulary subset: map rows to ids on the host, then score
            sync_ck();
            for (auto & v : idx) v = mtp.draft_ids[v];
            ck(cudaMemcpyAsync(d_cand.p, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice, stream));
            tqllm_cuda::dflash_lattice(d_cand.as<uint32_t>(), d_vals.as<float>(), d_gate.as<bf16>(), dfl.rank,
                                       dfl.sel_next, dfl.sel_prev, anchor, B, dfl.rank, d_score.as<float>(), stream);
        }
        std::vector<float> score(size_t(B) * 256);
        ck(cudaMemcpyAsync(score.data(), d_score.p, score.size() * 4, cudaMemcpyDeviceToHost, stream));
        sync_ck();
        std::vector<uint32_t> out;
        int pred = 0;
        for (int i = 1; i < B && int(out.size()) < want; ++i) {
            const float * row = score.data() + (size_t(i) * 16 + pred) * 16;
            pred = int(std::max_element(row, row + 16) - row);
            out.push_back(idx[size_t(i) * 16 + pred]);
        }
        if (idx_out) *idx_out = std::move(idx);
        if (score_out) *score_out = std::move(score);
        return out;
    }


    std::vector<uint32_t> generate_dflash(const std::vector<uint32_t> & prompt, uint32_t count, int D, uint64_t * stats) {
        dflash_begin(prompt, count, D);
        while (dflash_advance()) {}
        return dflash_finish(stats);
    }

    // Prefill the active slot and set up its generation; dflash_advance() then runs one draft + verify step.
    void dflash_begin(const std::vector<uint32_t> & prompt, uint32_t count, int D) {
        const auto P = uint32_t(prompt.size());
        const int H = int(cfg.hidden), nf = int(dfl.target_ids.size());
        // --mtp with DFlash2 loaded: MTP for prompts of --mtp-from tokens or more, at --mtp-depth drafts
        // sampled or image requests decode without MTP (exact either way), so no request fails for asking for them
        const bool use_mtp = opt.native_mtp && mtp.present && !sampling() && mm_pending.empty() &&
                             (!dfl.present || (opt.mtp_from > 0 && P >= opt.mtp_from));
        const bool dfl_on = dfl.present && !use_mtp;
        if (use_mtp && dfl.present) D = opt.mtp_depth;
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
            if (sampling()) {  // the final chunk's head wrote column 0's top-16
                ck(cudaMemcpy(h_sid.data(), s_id.p, 16 * 4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(h_slg.data(), s_lg.p, 16 * 4, cudaMemcpyDeviceToHost));
                next = draw(dist(0));
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

    // Accepted draft count j for the active slot; res[j] becomes the token after them. Column col0 + i of the verify
    // holds the target's prediction after draft token i - 1 (greedy: res[i]; sampling: h_sid / h_slg top-16).
    int accept(const std::vector<uint32_t> & draft, uint32_t * res, int col0) {
        int j = 0;
        if (sampling()) {  // speculative sampling with a deterministic draft: exact samples of the target's p
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            std::vector<uint32_t> recent;  // presence window: the output so far, then each draft token as it is accepted
            if (samp.presence > 0.f) {
                const size_t n = gen.out.size(), r0 = n > size_t(kPresenceLastN) ? n - kPresenceLastN : 0;
                recent.assign(gen.out.begin() + long(r0), gen.out.end());
            }
            for (; j < int(draft.size()); ++j) {
                const Dist d = dist(col0 + j, recent);
                double q = 0;
                for (int i = 0; i < d.n; ++i) if (d.id[i] == draft[j]) q = d.p[i];
                if (uni(rng) >= q) {
                    res[j] = draw(d, draft[j]);
                    break;
                }
                if (samp.presence > 0.f) recent.push_back(draft[j]);
            }
            if (j == int(draft.size())) res[j] = draw(dist(col0 + j, recent));
        } else {
            while (j < int(draft.size()) && draft[j] == res[j]) ++j;
        }
        return j;
    }

    // Batched decode step: each slot drafts, one verify runs every slot's columns (weights read once), then each
    // slot accepts, commits and injects its drafter features. done[i]: slot slots[i] finished (call spec_finish).
    // Batched token-tree step (greedy slots): every slot's DFlash2 lattice -> its own tree on the device, one verify
    // over all trees (per-segment parents / depths), then per slot the accepted path is gathered, committed and injected.
    // false: not applicable (sampling slot, batched drafting unavailable), nothing was queued.
    bool tree_step_batch(std::vector<DraftReq> rq, const int * slots, int ns, bool * done,
                         std::chrono::steady_clock::time_point p0) {
        for (const auto & r : rq) {
            select_slot(r.slot);
            if (sampling()) return false;
        }
        std::vector<int> cols(rq.size());
        int col = 0;
        for (size_t i = 0; i < rq.size(); ++i) {
            rq[i].want = std::min(rq[i].want, NVER - 1);
            cols[i] = col;
            col += 1 + rq[i].want;
        }
        std::vector<std::vector<uint32_t>> unused;
        if (!dflash_draft_batch(rq, unused, &cols)) return false;
        const auto p1 = std::chrono::steady_clock::now();
        segs.clear();
        batch_topk = false;
        for (size_t i = 0; i < rq.size(); ++i) {
            h_pos[2 * i] = int(rq[i].pos);
            h_pos[2 * i + 1] = int(rq[i].pos) + 1;
            segs.push_back({rq[i].slot, cols[i], 1 + rq[i].want, rq[i].pos, true});
        }
        ck(cudaMemcpyAsync(v_pos.p, h_pos, 2 * segs.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        upload_rpos(int(segs.size()), [&](int i) { return rdelta_of(rq[size_t(i)].slot); });
        for (const auto & g : segs) kv_track(g.pos, g.n);
        uint64_t key = 0x9E3779B97F4A7C15ull ^ (uint64_t(sp) << 40);
        for (const auto & g : segs) key = (key ^ uint64_t((g.slot * 64 + g.n) * 4 + cur_of(g.slot) % NSLOT)) * 1099511628211ull;
        ck(cudaGraphLaunch(cached_graph(kGraphBatch, key, [&] { verify_body(col, int(context_)); }), stream));
        const size_t ncol = static_cast<size_t>(col);
        std::vector<uint32_t> res(ncol, 0u), in(ncol, 0u);
        std::vector<int> tpar(ncol, 0);
        ck(cudaMemcpyAsync(res.data(), v_tok.p, size_t(col) * 4, cudaMemcpyDeviceToHost, stream));
        ck(cudaMemcpyAsync(in.data(), v_tokens.p, size_t(col) * 4, cudaMemcpyDeviceToHost, stream));
        ck(cudaMemcpyAsync(tpar.data(), d_tpar.p, size_t(col) * 4, cudaMemcpyDeviceToHost, stream));
        sync_ck();
        const auto p2 = std::chrono::steady_clock::now();
        const auto batch = segs;
        segs.clear();
        const int nf = int(dfl.target_ids.size()), H = int(cfg.hidden);
        for (const auto & g : batch) {
            select_slot(g.slot);
            const int c0 = g.col0, n = g.n;
            std::vector<int> path{0};  // follow the target's argmax down this slot's tree (local column indices)
            for (int node = 0;;) {
                int next = -1;
                for (int c = node + 1; c < n; ++c)
                    if (tpar[size_t(c0 + c)] == node && in[size_t(c0 + c)] == res[size_t(c0 + node)]) {
                        next = c;
                        break;
                    }
                if (next < 0) break;
                path.push_back(next);
                node = next;
            }
            const int j = int(path.size()) - 1;
            ++gen.steps;
            gen.drafted += uint64_t(n - 1);
            gen.accepted += uint64_t(j);
            bool hit_eos = false;
            for (int i = 1; i <= j && gen.out.size() < gen.count; ++i) {
                gen.out.push_back(in[size_t(c0 + path[size_t(i)])]);
                hit_eos |= in[size_t(c0 + path[size_t(i)])] == eos_;
            }
            const uint32_t bonus = res[size_t(c0 + path.back())];
            if (gen.out.size() < gen.count) gen.out.push_back(bonus);
            bool contiguous = true;
            for (int i = 0; i <= j; ++i) contiguous &= path[size_t(i)] == i;
            if (!contiguous) {  // accepted path -> columns c0 .. c0 + j of this slot's stash / features, K/V at pos ..
                std::vector<tqllm_cuda::PathGather> gd;
                uint32_t lin_i = 0, attn_i = 0;
                for (uint32_t l = 0; l < cfg.layers; ++l) {
                    if (cfg.full[l]) {
                        gd.push_back({kc_of(g.slot)[attn_i].p, kv_row_bytes(true), 1});
                        gd.push_back({vc_of(g.slot)[attn_i].p, kv_row_bytes(false), 1});
                        ++attn_i;
                    } else {
                        const int64_t rb = int64_t(mix_rows(layer[l])) * 2;
                        gd.push_back({static_cast<uint8_t *>(v_stash2[sp][lin_i].p) + c0 * rb, rb, 0});
                        ++lin_i;
                    }
                }
                const int64_t fb = int64_t(nf) * H * 2;
                gd.push_back({static_cast<uint8_t *>(v_feat.p) + c0 * fb, fb, 0});
                ck(cudaMemcpyAsync(d_gdesc.p, gd.data(), gd.size() * sizeof(gd[0]), cudaMemcpyHostToDevice, stream));
                ck(cudaMemcpyAsync(d_tcols.p, path.data(), path.size() * 4, cudaMemcpyHostToDevice, stream));
                tqllm_cuda::gather_path(d_gdesc.as<tqllm_cuda::PathGather>(), int(gd.size()), d_tcols.as<int>(), j + 1,
                                        int(gen.pos), stream);
            }
            last_n = n;
            stash_col0 = c0;
            commit(j + 1);
            stash_col0 = 0;
            dflash_inject(v_feat.as<bf16>() + size_t(c0) * nf * H, j + 1, gen.pos);
            gen.pending = bonus;
            gen.pos += uint32_t(j + 1);
            if (hit_eos || gen.out.size() >= gen.count || gen.pending == eos_) gen.done = true;
        }
        for (int i = 0; i < ns; ++i) {
            select_slot(slots[i]);
            done[i] = gen.done;
        }
        static const bool prof = tqllm::debug_on("steps");
        if (prof) {
            sync_ck();
            auto msd = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            std::fprintf(stderr, "bstep segs %zu cols %d draft %.2f verify %.2f commit %.2f ms (tree)\n", batch.size(), col,
                         msd(p0, p1), msd(p1, p2), msd(p2, std::chrono::steady_clock::now()));
        }
        return true;
    }

    void dflash_step(const int * slots, int ns, bool * done) {
        flush_pending();  // only the active slot can hold a deferred commit (select_slot flushes before parking)
        struct Plan { std::vector<uint32_t> draft; };
        std::vector<Plan> plan;
        static const bool prof = tqllm::debug_on("steps");
        const auto p0 = std::chrono::steady_clock::now();
        segs.clear();
        batch_topk = false;
        int col = 0;
        // Column budget: past ~8-12 columns the verify matmuls turn tensor-core bound, so with many slots each drafts
        // fewer tokens (the greedy output does not depend on the draft length).
        // Blackwell (GB10 LPDDR5X, 5060 Ti) stays memory-bound up to the full NB columns; GA10x GeForce FP32-accumulate
        // tensor rate caps useful width near 16 (3090, c=4 mixed code / essays: budget 16 -> 283 / 198, 24 -> 281 / 173,
        // 32 -> 249 / 138 tok/s; four identical prime prompts favour 24: 300 / 341 / 314).
        static const int budget = [this] {
            int dev = 0, major = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            return major >= 10 ? NB : 16;
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

    // Prompt-lookup drafting (--lookup L, tree mode): the latest earlier occurrence of the last 6 tokens of prompt +
    // output proposes the tokens that followed it as an extra chain under the tree root (up to L columns, taken from the
    // DFlash tree's budget). Verification is unchanged, so outputs never depend on it.
    int lookup(uint32_t * dst, int maxlen) const {
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

    // TQLLM_DEBUG=decode: a progress line every ~2.5 s while decoding (tokens / wall time, same basis as reqlog).
    void dec_progress() {
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

    // One draft + verify step on the active slot; false once the generation is complete.
    bool dflash_advance() {
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

    std::vector<uint32_t> dflash_finish(uint64_t * stats) {
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

public:
    bool has_mtp() const { return mtp.present; }

    // MTP draft head (Qwen3.5/3.8 nextn block). Runs n (token, previous hidden) pairs at positions pos, pos+1, ...:
    // x = eh_proj([enorm(embed(t)) ; hnorm(h)]), one full-attention decoder layer with its own KV cache, then the
    // shared LM head on shared_head_norm(x). Returns the greedy token after the last pair (the next draft) and keeps
    // the last pre-norm output as the hidden for chaining. `hidden`: device [n][H] (nullptr = chain from last MTP).
    uint32_t mtp_run(const uint32_t * tokens, const bf16 * hidden, int n, uint32_t pos, bool sample = true) {
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
        embed_multi(v_tokens.as<uint32_t>(), n, E);
        tqllm_cuda::rms_f_multi(E, H, 0, mtp.enorm, CAT, 2 * H, 0, H, 1, n, cfg.eps, stream);
        tqllm_cuda::rms_f_multi(hidden ? hidden : mtp_h.as<bf16>(), H, 0, mtp.hnorm, CAT + H, 2 * H, 0, H, 1, n, cfg.eps,
                                stream);
        rot_h(CAT, XH, 2 * H, n);
        mm_h(mtp.eh_proj, X, n);
        tqllm_cuda::sumsq(X, v_px.as<float>(), n * H, stream);
        ck(cudaMemcpyAsync(R, X, size_t(n) * H * 2, cudaMemcpyDeviceToDevice, stream));
        nrot_h(X, mtp.layer.in_norm, v_px.as<float>(), XH, H, n);
        vpar = chain2 && n <= NVER ? d_chain.as<int>() : nullptr;  // the MTP layer's attention: two-part chain like the verify
        vdep = vpar ? d_cdep.as<int>() : nullptr;                   // (> 8 queries need depths; tree mode crashed here)
        attn_mixer(mtp.layer, mtp_kc.as<bf16>(), mtp_vc.as<bf16>(), n, int(bucket));
        ffn_block(mtp.layer, n);
        ck(cudaMemcpyAsync(mtp_h.p, X + size_t(n - 1) * H, size_t(H) * 2, cudaMemcpyDeviceToDevice, stream));
        if (!sample) {
            sync_ck();
            return 0;
        }
        if (mtp.draft_head.p) {  // argmax over the draft vocabulary only (the verifier still sees the full vocab)
            auto * NP = v_np.as<bf16>();
            tqllm_cuda::rms_f(X + size_t(n - 1) * H, H, mtp.shared_norm, NP, H, H, 1, cfg.eps, stream);
            tqllm_cuda::bf16_to_f32(NP, v_xf.as<float>(), H, stream);
            tqllm_cuda::q6k_matvec(mtp.draft_head.p, v_xf.as<float>(), v_logits.as<float>(), mtp.draft_head.rows, H, 1,
                                   stream);
            tqllm_cuda::argmax_cols(v_logits.as<float>(), mtp.draft_head.rows, 1, v_tok.as<uint32_t>(), v_amax.p, stream);
            ck(cudaMemcpyAsync(h_tok, v_tok.p, 4, cudaMemcpyDeviceToHost, stream));
            sync_ck();
            return mtp.draft_ids[h_tok[0]];
        }
        // Only the final repaired hidden state proposes the next draft token.
        head_argmax(mtp.shared_norm, 1, n - 1);
        ck(cudaMemcpyAsync(h_tok, v_tok.as<uint32_t>() + (n - 1), 4, cudaMemcpyDeviceToHost, stream));
        sync_ck();
        return h_tok[0];
    }

    int slots() const override { return nslots; }
    bool incremental() const override { return fmt_target; }
    bool has_drafter() const override { return dfl.present; }
    bool native_mtp() const override { return opt.native_mtp && mtp.present && !dfl.present; }  // MTP only (no -md)
    void select_slot(int s) override {
        if (s < 0 || s >= nslots) throw std::out_of_range("slot out of range");
        if (s == active) return;
        flush_pending();
        swap_seq(parked[size_t(active)]);
        swap_seq(parked[size_t(s)]);
        active = s;
    }
    // Without the DFlash drafter every step verifies one token per slot (verify-path numerics, batched over slots).
    void set_prompt_images(const std::vector<PromptImage> & ims) override {
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
    bool mtp_advance() {
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
        int j = 0;
        while (j < D && draft[size_t(j)] == res[size_t(j)]) ++j;
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

    void spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int D) override {
        if (!fmt_target) throw std::runtime_error("incremental decoding needs a quantized GGUF model");
        dflash_begin(prompt, count, D);
    }
    bool spec_advance() override {
        if (gen.native_mtp) return mtp_advance();  // per request: an MTP request with DFlash2 loaded too
        if (dfl.present) return dflash_advance();
        bool done = false;
        const int s = active;
        dflash_step(&s, 1, &done);
        return !done;
    }
    void spec_step(const int * slots, int n, bool * done) override {
        if (gen.native_mtp) {
            if (n != 1 || slots[0] != active) throw std::runtime_error("MTP step needs the active single slot");
            done[0] = !mtp_advance();
        } else dflash_step(slots, n, done);
    }
    std::vector<uint32_t> spec_finish(uint64_t * stats) override { return dflash_finish(stats); }
    std::vector<uint32_t> spec_tokens(int s) const override { return s == active ? gen.out : parked[size_t(s)].gen.out; }
    int drafter_features(const std::vector<uint32_t> & tokens, std::vector<uint16_t> & feats) override {
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

    bool prefill_state(const std::vector<uint32_t> & tokens, uint32_t pos0) override {
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

    std::vector<uint32_t> generate_spec(const std::vector<uint32_t> & prompt, uint32_t count, int D,
                                        uint64_t * stats) override {
        if (dfl.present) return generate_dflash(prompt, count, D, stats);
        if (opt.native_mtp) {
            spec_begin(prompt, count, D);
            while (spec_advance()) {}
            return spec_finish(stats);
        }
        if (!mtp.present) throw std::runtime_error("model has no MTP head");
        if (D < 1 || D > NVER - 1) throw std::runtime_error("draft length exceeds verify capacity");
        const auto P = uint32_t(prompt.size());
        if (P == 0 || P + count + D + 1 > context_) throw std::runtime_error("request exceeds context capacity");
        const int H = int(cfg.hidden);
        // Prefill: main model token by token; MTP KV at position t is built from (h_{t-1}, x_t).
        std::vector<uint32_t> ptoks;
        Buf hbuf(size_t(NVER) * H * 2);
        uint32_t pstart = 1;
        uint32_t next = 0;
        for (uint32_t t = 0; t < P; ++t) {
            next = step(prompt[t], t, t + 1 == P, nullptr);
            if (t + 1 < P) {  // h_t pairs with x_{t+1}
                ck(cudaMemcpyAsync(hbuf.as<bf16>() + ptoks.size() * H, x.p, size_t(H) * 2, cudaMemcpyDeviceToDevice,
                                   stream));
                ptoks.push_back(prompt[t + 1]);
                if (int(ptoks.size()) == NVER || t + 2 == P) {
                    mtp_run(ptoks.data(), hbuf.as<bf16>(), int(ptoks.size()), pstart);
                    pstart += uint32_t(ptoks.size());
                    ptoks.clear();
                }
            }
        }
        // First drafts: pair (h_{P-1}, next) at position P, then chain.
        std::vector<uint32_t> out{next}, draft;
        const auto decode_start = std::chrono::steady_clock::now();
        ck(cudaMemcpyAsync(hbuf.p, x.p, size_t(H) * 2, cudaMemcpyDeviceToDevice, stream));
        draft.push_back(mtp_run(&next, hbuf.as<bf16>(), 1, P));
        uint64_t steps = 0, drafted = 0, accepted = 0;
        uint32_t pos = P, pending = next;
        std::vector<uint32_t> in, res(NVER);
        while (out.size() < count && pending != eos_) {
            while (int(draft.size()) < D) draft.push_back(mtp_run(&draft.back(), nullptr, 1, pos + uint32_t(draft.size())));
            in.assign(1, pending);
            in.insert(in.end(), draft.begin(), draft.end());
            verify(in.data(), int(in.size()), pos, res.data());
            int j = 0;
            while (j < D && draft[j] == res[j]) ++j;
            ++steps;
            drafted += D;
            accepted += j;
            for (int i = 0; i < j && out.size() < count; ++i) out.push_back(draft[i]);
            if (out.size() < count) out.push_back(res[j]);
            commit(j + 1);
            // MTP: pairs (h_{t-1}, x_t) for t = pos+1 .. pos+j+1 (accepted drafts, then the bonus token)
            std::vector<uint32_t> mt(draft.begin(), draft.begin() + j);
            mt.push_back(res[j]);
            pending = res[j];
            pos += uint32_t(j + 1);
            if (std::find(out.end() - (j + 1 < int(out.size()) ? j + 1 : int(out.size())), out.end(), eos_) != out.end())
                break;
            draft.assign(1, mtp_run(mt.data(), verify_hidden(), int(mt.size()), pos));
        }
        if (stats) {
            stats[0] = steps;
            stats[1] = drafted;
            stats[2] = accepted;
            stats[3] = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - decode_start).count());
            stats[4] = uint64_t(last_prefill_ms * 1000.0);
        }
        if (out.size() > count) out.resize(count);
        return out;
    }

    // Hidden states (pre-norm) of the last verify, [n][H] on the device.
    const bf16 * verify_hidden() const { return v_hid.as<bf16>(); }
    // Pre-norm hidden of the last single step().
    const bf16 * step_hidden() const { return x.as<bf16>(); }

private:
    // verify() stored only the state after all n tokens (slot cur + 1). Accepting fewer replays the recurrent
    // layers over the accepted prefix from slot cur, using the inputs kept in v_stash.
    void set_pending(int k) {
        pend_k = k;
        ck(cudaMemcpyAsync(d_pend.p, &pend_k, sizeof(int), cudaMemcpyHostToDevice, stream));  // pageable: staged now
    }
    // Deferred commit -> real state: replay the pending path (as commit(k < n) does) from slot cur into cur + 1.
    void flush_pending() {
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

    void commit(int k) override {
        if (k < 1 || k > last_n) throw std::runtime_error("commit: accepted count out of range");
        if (k < last_n) commit_body(k);
        cur = (cur + 1) % NSLOT;
        last_n = 0;
    }
    void commit_body(int k) {  // replay the first k verified tokens into the next state slot
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
};

} // namespace

std::unique_ptr<Engine> make_qwen35_engine(const std::string & path, const std::string & tokenizer, uint32_t context,
                                           int slots, const EngineOptions & options) {
    return std::make_unique<Qwen35Engine>(path, tokenizer, context, slots, options);
}

} // namespace tqllm
