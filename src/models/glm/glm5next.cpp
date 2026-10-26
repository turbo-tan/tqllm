// GLM-5.3-Flash (GGUF arch `glm5-next` / `glm5next`) engine: decode one token per step on the GPU.
// The forward pass is the one tools/glm_ref.cpp computes in FP32 on the CPU (and that matches llama.cpp):
//   4 mHC streams; per layer hc_pre -> RMSNorm -> KDA (3 of 4 layers) or nope MLA -> hc_post, then
//   hc_pre -> RMSNorm -> dense SwiGLU (first layers) or sigmoid MoE + shared expert -> hc_post; mean -> head.
// Matrices go through the weight-format registry (Q8_0, Q6_K, Q2_K, IQ2_S, ...); tensors without a format
// (F32 / BF16 / F16) run as FP32. TQLLM_GLM_DUMP=1 prints the per-stage sums glm-ref prints (`dump name-layer sum`).
#include "dequant.hpp"
#include "dev_alloc.cuh"
#include "engine.hpp"
#include "expert_bank_plan.hpp"
#include "glm5next_kernels.cuh"
#include "glm_experts.hpp"
#include "gguf.hpp"
#include "gguf_mapped.hpp"
#include "kv_store.hpp"
#include "moe_kernels.cuh"
#include "quant_kernels.cuh"
#include "packed_experts.hpp"
#include "weight_format.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace tqllm {
namespace {

using bf16 = __nv_bfloat16;

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("glm5next: ") + cudaGetErrorString(e));
}
// config/flags.yaml name, plus the short GLM_* name the bring-up scripts in ref/ use
bool env_on(const char * registered, const char * legacy) { return std::getenv(registered) || std::getenv(legacy); }

struct Buf {
    void * p = nullptr;
    size_t bytes = 0;  // what the allocation holds, so a buffer can grow in place
    Buf() = default;
    explicit Buf(size_t n) : bytes(n) { if (bytes) ck(tqllm_cuda::dev_malloc(&p, bytes)); }
    ~Buf() { if (p) tqllm_cuda::dev_free(p); }
    Buf(Buf && o) noexcept : p(std::exchange(o.p, nullptr)), bytes(std::exchange(o.bytes, 0)) {}
    Buf & operator=(Buf && o) noexcept {
        if (this != &o) { if (p) tqllm_cuda::dev_free(p); p = std::exchange(o.p, nullptr); bytes = std::exchange(o.bytes, 0); }
        return *this;
    }
    Buf(const Buf &) = delete;
    Buf & operator=(const Buf &) = delete;
    template <class T> T * as() const { return static_cast<T *>(p); }
};

// A matrix: a weight-format Mat, or FP32 rows for tensors no format loads.
struct M {
    Mat m;
    const float * f32 = nullptr;
    int rows = 0, cols = 0;
    mutable int f32_out = -1;  // the format has matvec_f32 (-1: not tried yet)
    explicit operator bool() const { return rows != 0; }
};

struct Layer {
    bool recurrent = false;
    const float * attn_norm = nullptr, * ffn_norm = nullptr;
    M hc_fn[2];
    const float * hc_base[2] = {}, * hc_scale[2] = {};
    // KDA
    M wqkv, wo, fgb, f_b, g_b;  // wqkv = q | k | v rows; fgb = f_a | g_a | beta rows
    const float * conv = nullptr, * dt_bias = nullptr, * A = nullptr, * o_norm = nullptr;
    Buf hist, state;
    // speculative rollback (verify + commit): the state / history before the block, and the block's own KDA inputs, so
    // commit(k) reruns the recurrence for the first k tokens and rebuilds the conv history - bit-identical to k single
    // steps. `saved` is the number of rows kept in kqkv / kconv / kg / kbeta (the raw projections rebuild the conv history,
    // the conv output feeds the recurrence); 0 = no block recorded. A verify longer than the chunk size saves its rows
    // chunk by chunk into the same buffers, so one snapshot covers the whole block.
    Buf hist0, state0, kqkv, kconv, kg, kbeta;
    int saved = 0;
    // MLA
    M q_a, q_b, kv_a;
    const float * q_a_norm = nullptr, * kv_a_norm = nullptr;
    const __half * kb = nullptr, * vb = nullptr;
    int lat = -1;  // this layer's row set in the shared latent cache (src/kv), -1 for a KDA layer
    // DSA k-pool indexer (MLA layers): key / gate / query / head-weight projections, the key LayerNorm, the pool position
    // bias; idx_kg [ctx][2D] FP16 key | gate per position, idx_pool [ctx / r][D] FP32 pool keys
    M idx_k, idx_g, idx_q, idx_w;
    const float * idx_ln_w = nullptr, * idx_ln_b = nullptr, * idx_ape = nullptr;
    Buf idx_kg, idx_pool;
    // FFN
    M gate, up, down, router, sh_gate, sh_up, sh_down;
    Mat e_gate, e_up, e_down;
    const float * probs_bias = nullptr;
    float clamp_exp = 0, clamp_sh = 0;
    int paired = -1;  // gate + up experts in one launch works for this format (-1: not tried yet)
};

class Glm5NextEngine final : public Engine {
public:
    Glm5NextEngine(const std::string & path, uint32_t context) : context_(context ? context : 4096) {
        ck(cudaStreamCreate(&stream_));
        meta_ = gguf_open(path);
        configure();
        open_maps();
        load();
        alloc();
        dump_ = env_on("TQLLM_GLM_DUMP", "GLM_DUMP");
        // the bank path decides on the host which experts a layer needs, so it cannot be one captured graph
        use_graph_ = !dump_ && !bank_mode_ && !env_on("TQLLM_GLM_NO_GRAPH", "GLM_NO_GRAPH");
    }
    ~Glm5NextEngine() override {
        // the bank frees its device banks through the engine's stream, so it goes before that stream is destroyed
        report_bank();
        bank_.reset();
        // the latent cache's memory object holds the engine's stream, so the store, pool and memory go first
        lat_store_.reset();
        lat_pool_.reset();
        kvmem_.reset();
        if (cublas_) cublasDestroy(cublas_);
        if (copy_) cudaStreamDestroy(copy_);
        if (packed_ && packed_->map) {
            if (packed_pinned_) cudaHostUnregister(const_cast<char *>(packed_->map));
            ::munmap(const_cast<char *>(packed_->map), packed_->bytes);
        }
        if (graph_) cudaGraphExecDestroy(graph_);
        if (h_pos_) cudaFreeHost(h_pos_);
        if (h_embed_) cudaFreeHost(h_embed_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    uint32_t step(uint32_t token, uint32_t pos, bool sample, float * capture) override {
        if (capture) throw std::runtime_error("glm5next: hidden-state capture is not supported");
        uint32_t next = 0;
        run(&token, 1, pos, sample ? &next : nullptr);
        return next;
    }
    // Prompt / verify: chunks of up to chunk_ tokens; every matrix is read once per chunk and every position gets its
    // argmax. Same arithmetic as one step() per token (the formats' matvec is width-independent). A block of 1 < n <=
    // kMaxSavedBlock also snapshots what commit(k) needs to roll the recurrent state back.
    // the tokens' state (KDA, latent, indexer caches) in prompt chunks; decoding continues at pos0 + tokens.size()
    bool prefill_state(const std::vector<uint32_t> & tokens, uint32_t pos0) override {
        if (tokens.empty()) return true;
        if (pos0 + uint32_t(tokens.size()) + 1 > context_) throw std::runtime_error("glm5next: prefill beyond the context");
        run(tokens.data(), int(tokens.size()), pos0, nullptr);
        return true;
    }
    void verify(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) override {
        begin_block(n, pos);
        run(tokens, n, pos, next);
        in_block_ = false;
        verified_ = n;
    }
    // keep the state after the first k tokens of the last verify: replay the KDA recurrence for those k tokens from the
    // pre-block state and rebuild the conv history from the pre-block history plus the k accepted inputs. The recurrence
    // of token t reads only the state and token t, and the conv output of token t reads only the K-1 inputs before it, so
    // both are the same numbers a single-token step would have produced: commit(k) is bit-identical to k step() calls.
    // The MLA latent rows of the rejected tokens need nothing: attention reads rows [0, position) and the next tokens
    // overwrite them.
    void commit(int k) override {
        if (k == verified_) return;
        if (k < 1 || k > verified_) throw std::runtime_error("glm5next: commit(" + std::to_string(k) + ") outside the verified block of " +
                                                             std::to_string(verified_));
        // check every layer first: a half-rolled-back model is worse than a throw
        for (const auto & Y : L_)
            if (Y.recurrent && Y.saved < k)
                throw std::runtime_error("glm5next: commit(" + std::to_string(k) + ") after a block of " +
                                         std::to_string(verified_) + " tokens that recorded " + std::to_string(Y.saved) +
                                         " KDA rows (a block longer than " + std::to_string(kMaxSavedBlock) + " is not recorded)");
        const int d = heads_ * kda_dim_;
        for (auto & Y : L_) {
            if (!Y.recurrent) continue;
            tqllm_cuda::glm_kda_recur(Y.kconv.as<float>(), Y.kconv.as<float>() + d, Y.kconv.as<float>() + 2 * d,
                                      Y.kg.as<float>(), Y.kbeta.as<float>(), Y.state0.as<float>(), Y.state.as<float>(),
                                      rb_out_.as<float>(), heads_, kda_dim_, stream_, k, 3 * d);
            tqllm_cuda::glm_kda_hist_rollback(Y.hist0.as<float>(), Y.kqkv.as<float>(), Y.hist.as<float>(), d, conv_, k, stream_,
                                              3 * d, d);
            Y.saved = 0;
        }
        verified_ = k;
    }
    const float * logits() const override { return logits_.as<float>() + size_t(last_rows_ - 1) * vocab_; }
    void synchronize() override { ck(cudaStreamSynchronize(stream_)); }
    uint32_t layers() const override { return uint32_t(L_.size()); }
    uint32_t hidden() const override { return uint32_t(H_); }
    uint32_t vocab() const override { return uint32_t(vocab_); }
    uint32_t eos() const override { return eos_; }
    uint32_t context() const override { return context_; }

private:
    GgufFile meta_;
    std::string a_;
    cudaStream_t stream_ = nullptr;
    uint32_t context_ = 0, eos_ = 0;
    int H_ = 0, heads_ = 0, dense_lead_ = 0, experts_ = 0, topk_ = 0, q_lora_ = 0, kv_lora_ = 0, dk_ = 0, dv_ = 0;
    int kda_dim_ = 0, conv_ = 0, iters_ = 0, vocab_ = 0, expert_ff_ = 0, ff_ = 0, sh_ff_ = 0, index_topk_ = 0;
    float rms_eps_ = 0, hc_eps_ = 0, gate_lb_ = 0, w_scale_ = 0;
    // DSA: indexer heads / key size / pool size, LayerNorm eps; dsa_ = the file has indexer weights (past index_topk_
    // positions the MLA layers attend to the indexer's cells, below it to every position: the same set)
    int idx_heads_ = 0, idx_dim_ = 0, kpool_ = 0;
    float ln_eps_ = 1e-5f;
    bool dsa_ = false;
    Buf dsa_k_, dsa_g_, dsa_q_, dsa_w_, dsa_cells_, dsa_count_, dsa_scr_;
    size_t dsa_scr_bytes_ = 0;
    static constexpr int kDsaGroup = 64;  // queries per selection pass (bounds the [queries][pools] score scratch)
    // indexer key | gate rows kept: a call's tokens plus one pool (keys are read only to build their pool)
    int dsa_ring() const { return std::max(chunk_, pchunk_) + 2 * kpool_; }
    bool w_norm_ = false, dump_ = false;
    int verified_ = 0, chunk_ = 8, last_rows_ = 1;
    // prompt chunks: up to pchunk_ tokens per forward. Matrices past their matvec width run as FP16 dequant + cuBLAS
    // GEMM, the routed experts as grouped int8 GEMMs (MMQ), the head on the chunk's last row only; pchunk_ == chunk_
    // when the experts have no grouped GEMM (bank mode, a format without MMQ)
    int pchunk_ = 512;
    Buf wdq_, moe_counts_, moe_groups_, mq_x_, mq_a_, mq_tiles_;
    cublasHandle_t cublas_ = nullptr;
    // the verify block being recorded: rows saved so far across its chunks (a single step records nothing)
    bool in_block_ = false;
    int rec_off_ = 0;
    Buf rb_out_;  // commit()'s replay output scratch: a recorded block can be wider than chunk_
    static constexpr int kMaxSavedBlock = 32;  // a longer verify block (a prompt) is not recorded: see begin_block()
    // decode graph: one captured single-token forward, replayed with the position read on the device
    Buf d_pos_, lat_stage_;
    int * h_pos_ = nullptr;
    float * h_embed_ = nullptr;
    cudaGraphExec_t graph_ = nullptr;
    bool use_graph_ = true, warmed_ = false;
    std::vector<std::unique_ptr<MappedGguf>> maps_;
    std::map<std::string, const MappedGguf *> map_of_;
    std::vector<Buf> owned_;
    std::vector<Layer> L_;
    // the nope-MLA latent cache goes through src/kv (rule 4): one row per position (kv_lora FP16) that is both the key and
    // the value, so the pool geometry has value_row 0. One flat page of the context per sequence today; the tiered /
    // paged policies of the same store are what put the oldest rows in host RAM on the 3090.
    std::unique_ptr<KvMemory> kvmem_;
    std::unique_ptr<KvPool> lat_pool_;
    std::unique_ptr<KvStore> lat_store_;
    M out_;
    const float * out_norm_ = nullptr;
    // activations / scratch
    Buf S_, S2_, x_, xn_, xh_, mix_, post_, comb_, fo_, bfa_, bfb_, scratch_h_, proj_, cv_, g1_, fa_, beta_, o_, og_, g2_,
        qr_, q_, qabs_, kvh_, att_, logits_, d_arg_, d_ids_, d_w_, ye_g_, ye_u_, act_, yd_, sh_, embed_;
    int max_rows_ = 0, fdim_ = 0, gdim_ = 0;
    size_t max_split_ = 0;
    Buf partial_;
    // routed experts: the whole expert stack of every MoE layer on the device (default), or a bank of experts per layer in
    // the shared MixedExpertCache (TQLLM_GLM_EXPERT_SLOTS slots for every layer, or TQLLM_GLM_EXPERT_BUDGET_MB alone: the
    // shared plan of src/experts/expert_bank_plan.hpp gives the hot layers more slots than the cold ones). This is the
    // 3090 plan, where the experts live in host RAM. The bank path reads the routed ids on the host (prepare() decides
    // what to load), so it does not capture a graph.
    int slots_ = 0;
    bool bank_mode_ = false;
    size_t budget_ = 0;         // device bytes the banks plus loader staging may hold
    size_t bank_bytes_ = 0;     // what a full set of banks costs: the default budget
    std::vector<size_t> slot_bytes_;  // per layer: one slot's bytes in the bank row layout (0: no bank there)
    std::vector<double> spread_;      // per layer: experts its routing reached (0: not measured yet)
    std::unique_ptr<MixedExpertCache> bank_;
    std::unique_ptr<PackedExperts> packed_;
    bool packed_pinned_ = false;  // the packed file's mapping is cudaHostRegister-ed, so the cache may DMA / kernel-read it
    cudaStream_t copy_ = nullptr;  // prefetch stream: the next token's experts land while this layer runs its attention
    bool pf_ = false;
    std::vector<std::vector<int>> pf_hist_;  // per MoE layer: the experts its last tokens routed (the prefetch hint)
    static constexpr int kPfHistory = 3;  // tokens of routing to re-prefetch after a bank eviction
    std::vector<std::vector<uint8_t>> seen_;  // per MoE layer: the experts its routing reached this prompt
    Buf d_slots_;
    static constexpr int kSmallRows = 512;
    static int splits(int cols) { return std::max(1, std::min(64, cols / 512)); }

    double num(const std::string & k) const { return meta_.num(a_ + k); }
    std::vector<double> arr(const std::string & k, size_t n) const {
        const auto it = meta_.kv.find(a_ + k);
        if (it == meta_.kv.end() || it->second.arr.size() < n) throw std::runtime_error("glm5next: missing or short " + a_ + k);
        return it->second.arr;
    }

    void configure() {
        const std::string arch = meta_.str("general.architecture");
        if (arch != "glm5-next" && arch != "glm5next") throw std::runtime_error("glm5next: wrong architecture " + arch);
        a_ = arch + ".";
        const int nl = int(num("block_count")) - int(meta_.num(a_ + "nextn_predict_layers", 0));
        H_ = int(num("embedding_length")); heads_ = int(num("attention.head_count"));
        dense_lead_ = int(num("leading_dense_block_count")); experts_ = int(num("expert_count"));
        topk_ = int(num("expert_used_count")); q_lora_ = int(num("attention.q_lora_rank"));
        kv_lora_ = int(num("attention.kv_lora_rank")); dk_ = int(num("attention.key_length_mla"));
        dv_ = int(num("attention.value_length_mla")); kda_dim_ = int(num("kda.head_dim"));
        conv_ = int(num("ssm.conv_kernel")); iters_ = int(num("hyper_connection.sinkhorn_iterations"));
        vocab_ = int(num("vocab_size")); expert_ff_ = int(num("expert_feed_forward_length"));
        ff_ = int(num("feed_forward_length")); index_topk_ = int(num("attention.indexer.top_k"));
        sh_ff_ = int(meta_.num(a_ + "expert_shared_feed_forward_length", expert_ff_));
        rms_eps_ = float(num("attention.layer_norm_rms_epsilon")); hc_eps_ = float(num("hyper_connection.epsilon"));
        ln_eps_ = float(meta_.num(a_ + "attention.layer_norm_epsilon", 1e-5));
        idx_heads_ = int(meta_.num(a_ + "attention.indexer.head_count", 0)); idx_dim_ = int(meta_.num(a_ + "attention.indexer.key_length", 0));
        kpool_ = int(meta_.num(a_ + "attention.indexer.kpool", meta_.num(a_ + "attention.indexer.block_size", 1)));
        gate_lb_ = float(num("kda.gate_lower_bound")); w_scale_ = float(meta_.num(a_ + "expert_weights_scale", 0));
        w_norm_ = meta_.num(a_ + "expert_weights_norm", 0) != 0;
        eos_ = uint32_t(meta_.num("tokenizer.ggml.eos_token_id", 0));
        if (int(num("hyper_connection.count")) != 4) throw std::runtime_error("glm5next: mHC with hc != 4");
        if (meta_.num(a_ + "expert_gating_func", 2) != 2) throw std::runtime_error("glm5next: only sigmoid MoE gating");
        if (meta_.num(a_ + "rope.dimension_count", 0) != 0) throw std::runtime_error("glm5next: only the nope-only MLA");
        if (meta_.tensors.count("blk.0.attn_q.weight") && heads_ * kda_dim_ != int(meta_.tensors.at("blk.0.attn_q.weight").ne[1]))
            throw std::runtime_error("glm5next: KDA heads x head_dim differs from attn_q");
        for (const auto & [n, t] : meta_.tensors) dsa_ = dsa_ || (idx_dim_ > 0 && n.find(".indexer.attn_k.weight") != std::string::npos);
        if (context_ > uint32_t(index_topk_)) {
            if (!dsa_) throw std::runtime_error("glm5next: context " + std::to_string(context_) + " > indexer top_k " +
                                                std::to_string(index_topk_) + " needs the indexer weights (none in this file)");
            if (idx_heads_ != 32 || kpool_ < 1 || kpool_ > 8 || index_topk_ % kpool_ || idx_dim_ % 32)
                throw std::runtime_error("glm5next: unsupported DSA indexer shape");
        }
        const auto kv = arr("attention.head_count_kv", size_t(nl)), ce = arr("swiglu_clamp_exp", size_t(nl));
        const auto cs = meta_.kv.count(a_ + "swiglu_clamp_shexp") ? arr("swiglu_clamp_shexp", size_t(nl)) : ce;
        L_.resize(size_t(nl));
        for (int l = 0; l < nl; ++l) {
            L_[size_t(l)].recurrent = kv[size_t(l)] == 0;
            L_[size_t(l)].clamp_exp = float(ce[size_t(l)]);
            L_[size_t(l)].clamp_sh = float(cs[size_t(l)]);
        }
        // TQLLM_GLM_EXPERT_SLOTS: keep only this many experts of every MoE layer resident (0: the whole stack).
        if (const char * v = std::getenv("TQLLM_GLM_EXPERT_SLOTS")) slots_ = std::atoi(v);
        if (slots_ < 0 || slots_ > experts_) throw std::runtime_error("glm5next: TQLLM_GLM_EXPERT_SLOTS out of range");
        if (slots_ > 0 && slots_ < topk_)
            throw std::runtime_error("glm5next: TQLLM_GLM_EXPERT_SLOTS " + std::to_string(slots_) +
                                     " is below expert_used_count " + std::to_string(topk_) + ": one token's experts must fit");
        // TQLLM_GLM_EXPERT_BUDGET_MB without a slot count: the shared plan turns the device budget into per-layer slots.
        bank_mode_ = slots_ > 0 || std::getenv("TQLLM_GLM_EXPERT_BUDGET_MB") != nullptr;
    }

    void open_maps() {
        for (const auto & [name, t] : meta_.tensors) {
            const std::string f = t.file.empty() ? meta_.path : t.file;
            if (!map_of_.count(f)) { maps_.push_back(std::make_unique<MappedGguf>(f)); map_of_[f] = maps_.back().get(); }
        }
    }
    const GgufTensor & tensor(const std::string & n) const {
        const auto it = meta_.tensors.find(n);
        if (it == meta_.tensors.end()) throw std::runtime_error("glm5next: missing tensor " + n);
        return it->second;
    }
    GgufBytes bytes(const std::string & n) const {
        const auto & t = tensor(n);
        return map_of_.at(t.file.empty() ? meta_.path : t.file)->tensor(n);
    }
    const void * put(const void * host, size_t n) {
        owned_.emplace_back(n);
        ck(cudaMemcpyAsync(owned_.back().p, host, n, cudaMemcpyHostToDevice, stream_));
        ck(cudaStreamSynchronize(stream_));
        return owned_.back().p;
    }
    std::vector<float> host_f32(const std::string & n) const {
        const auto & t = tensor(n);
        uint64_t e = 1;
        for (auto d : t.ne) e *= d;
        std::vector<float> v(static_cast<size_t>(e));
        dequantize_host(t.type, bytes(n).data, v.data(), int64_t(e));
        return v;
    }
    const float * vec(const std::string & n) {
        const auto v = host_f32(n);
        return static_cast<const float *>(put(v.data(), v.size() * 4));
    }
    // several matrices with the same input, stacked by rows into one (same GGUF type and columns): one launch, the input
    // read once; per output row the same arithmetic as the separate matrices
    M mat_rows(const std::vector<std::string> & names) {
        const auto & t0 = tensor(names[0]);
        M r;
        r.cols = int(t0.ne[0]);
        bool same_type = true;
        for (const auto & n : names) {
            const auto & t = tensor(n);
            if (t.ne.size() != 2 || int(t.ne[0]) != r.cols) throw std::runtime_error("glm5next: cannot stack " + n);
            same_type = same_type && t.type == t0.type;
            r.rows += int(t.ne[1]);
        }
        // FP32 rows take any mix of source types; a weight format needs one type
        if (r.rows > kSmallRows && !same_type) throw std::runtime_error("glm5next: stacked matrices of different types: " + names[0]);
        const WeightFormat * wf = r.rows <= kSmallRows ? nullptr : weight_format(t0.type);
        if (wf) {
            std::vector<char> raw;
            for (const auto & n : names) { const auto b = bytes(n); raw.insert(raw.end(), b.data, b.data + b.size); }
            r.m = wf->load(raw, r.rows, r.cols, [this](const void * h, size_t s) { return put(h, s); }, stream_);
            if (!r.m.fmt || r.m.fmt->basis() != Basis::Plain) throw std::runtime_error("glm5next: stacked " + names[0] + " format");
            chunk_ = std::min(chunk_, r.m.fmt->max_matvec_cols());
        } else {
            std::vector<float> v;
            for (const auto & n : names) { const auto x = host_f32(n); v.insert(v.end(), x.begin(), x.end()); }
            r.f32 = static_cast<const float *>(put(v.data(), v.size() * 4));
            if (r.rows <= kSmallRows) max_split_ = std::max(max_split_, size_t(r.rows) * splits(r.cols));
        }
        max_rows_ = std::max(max_rows_, r.rows);
        return r;
    }
    M mat(const std::string & n) {
        const auto & t = tensor(n);
        if (t.ne.size() != 2) throw std::runtime_error("glm5next: " + n + " is not a matrix");
        M r;
        r.cols = int(t.ne[0]); r.rows = int(t.ne[1]);
        // few rows (mHC mixes 24, KDA gates 64 / 128, MLA latent 512): FP32 rows and a split-K kernel; the format kernels
        // give such a matrix only a handful of blocks
        const WeightFormat * wf = r.rows <= kSmallRows ? nullptr : weight_format(t.type);
        if (wf) {
            const auto b = bytes(n);
            const std::vector<char> raw(b.data, b.data + b.size);
            r.m = wf->load(raw, r.rows, r.cols, [this](const void * h, size_t s) { return put(h, s); }, stream_);
            if (!r.m.fmt || r.m.fmt->basis() != Basis::Plain)
                throw std::runtime_error("glm5next: " + n + " loads into a rotated or engine-handled format");
        } else {
            const auto v = host_f32(n);
            r.f32 = static_cast<const float *>(put(v.data(), v.size() * 4));
            if (r.rows <= kSmallRows) max_split_ = std::max(max_split_, size_t(r.rows) * splits(r.cols));
        }
        max_rows_ = std::max(max_rows_, r.rows);
        if (r.m.fmt) chunk_ = std::min(chunk_, r.m.fmt->max_matvec_cols());
        return r;
    }
    Mat experts(const std::string & n) {
        const auto & t = tensor(n);
        if (t.ne.size() != 3 || int(t.ne[2]) != experts_) throw std::runtime_error("glm5next: " + n + " is not an expert stack");
        const WeightFormat * wf = weight_format(t.type);
        if (!wf) throw std::runtime_error("glm5next: no weight format for expert type " + std::to_string(t.type));
        const auto b = bytes(n);
        const std::vector<char> raw(b.data, b.data + b.size);
        Mat m = wf->load(raw, int(t.ne[1] * t.ne[2]), int(t.ne[0]), [this](const void * h, size_t s) { return put(h, s); }, stream_);
        if (!m.fmt || !m.fmt->has_expert_matvec() || m.fmt->basis() != Basis::Plain)
            throw std::runtime_error("glm5next: " + n + " has no grouped-expert kernel");
        chunk_ = std::min(chunk_, m.fmt->max_matvec_cols());
        return m;
    }
    // one expert's raw slices, from the shard that holds them (the bank's source)
    std::array<ExpertPayload, 3> expert_payload(int layer, int expert) const {
        return glm_expert_payload("blk." + std::to_string(layer) + ".",
                                  [this](const std::string & n) -> const MappedGguf & {
                                      const auto & t = tensor(n);
                                      return *map_of_.at(t.file.empty() ? meta_.path : t.file);
                                  },
                                  expert);
    }
    // the shared expert bank of src/experts (one per MoE layer), plus an optional packed expert file (TQLLM_GLM_EXPERTS,
    // written by glm-pack-experts) so a miss is a copy instead of a loader repack. `slots_` fixes the slots of every
    // layer; with only TQLLM_GLM_EXPERT_BUDGET_MB the shared plan spreads the bytes over the layers instead.
    std::vector<int> plan_caps() {
        std::vector<int> caps(L_.size(), 1);  // a layer without an expert bank (dense FFN) never opens one
        if (slots_ > 0) { for (auto & c : caps) c = slots_; return caps; }
        const auto p = expert_layer_slots(budget_, slot_bytes_, spread_, topk_, experts_);
        if (!p.fits)
            throw std::runtime_error("glm5next: TQLLM_GLM_EXPERT_BUDGET_MB " + std::to_string(budget_ >> 20) +
                                     " MB does not hold " + std::to_string(topk_) + " experts of every MoE layer");
        for (size_t l = 0; l < caps.size(); ++l) {
            if (!slot_bytes_[l]) continue;
            if (p.slots[l] <= 0)  // the plan skips a layer whose single expert is bigger than the whole budget
                throw std::runtime_error("glm5next: TQLLM_GLM_EXPERT_BUDGET_MB " + std::to_string(budget_ >> 20) +
                                         " MB is below one expert of layer " + std::to_string(l) + " (" +
                                         std::to_string(slot_bytes_[l] >> 20) + " MB): no bank can hold a token's experts");
            caps[l] = p.slots[l];
        }
        std::string line;
        for (size_t l = 0; l < caps.size(); ++l)
            if (slot_bytes_[l]) line += " " + std::to_string(caps[l]) + "/" + std::to_string(int(spread_[l]));
        std::fprintf(stderr, "glm5next: expert banks from %.2f GB: slots/spread per MoE layer:%s (%.2f GB of banks)\n", budget_ / 1e9,
                     line.c_str(), p.bytes / 1e9);
        return caps;
    }
    void open_bank() {
        const char * budget_env = std::getenv("TQLLM_GLM_EXPERT_BUDGET_MB");
        budget_ = budget_env ? size_t(std::atoll(budget_env)) << 20 : bank_bytes_ + bank_bytes_ / 4;
        const auto caps = plan_caps();
        int cap = 1;
        for (int c : caps) cap = std::max(cap, c);
        bank_ = std::make_unique<MixedExpertCache>(int(L_.size()), experts_, H_, expert_ff_, cap, budget_,
                                                   [this](int l, int e) { return expert_payload(l, e); }, stream_);
        bank_->set_layer_slots(caps);
        const char * path = std::getenv("TQLLM_GLM_EXPERTS");
        if (!path) return;
        packed_ = std::make_unique<PackedExperts>(open_packed(path));
        if (packed_->layers != int(L_.size()) || packed_->experts != experts_)
            throw std::runtime_error("glm5next: the packed expert file is for another model");
        if (packed_->tiled) throw std::runtime_error("glm5next: tiled packed experts (TQFNEXPT) are not supported here");
        // page-lock the mapping: a demand miss is then a DMA (or a kernel read) of the packed block, and the next token's
        // experts can be prefetched on a copy stream while this layer runs its attention. Only while the file is small
        // enough to pin beside the model: a 100 GB packed file on a 128 GB box would take the host down instead.
        const size_t ram = size_t(sysconf(_SC_PHYS_PAGES)) * size_t(sysconf(_SC_PAGESIZE));
        const bool pinnable = packed_->bytes <= ram / 2;
        if (pinnable &&
            cudaHostRegister(const_cast<char *>(packed_->map), packed_->bytes, cudaHostRegisterDefault) == cudaSuccess)
            packed_pinned_ = true;
        else
            cudaGetLastError();
        bank_->set_prepacked([this](int l, int e) -> const char * {
            return packed_->block[size_t(l)] ? packed_->at(l, e) : nullptr;  // a dense layer packs nothing
        }, packed_pinned_);
        const char * pf_env = std::getenv("TQLLM_GLM_EXPERT_PF");
        pf_ = packed_pinned_ && !dump_ && !(pf_env && pf_env[0] == '0');
        if (packed_pinned_) {  // split mode: misses upload on the copy stream, the compute stream waits only in wait_misses()
            ck(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
            bank_->set_miss_stream(copy_);
        }
        std::fprintf(stderr, "glm5next: packed experts %s (%.1f GB, %s)\n", path, packed_->bytes / 1e9,
                     packed_pinned_ ? (pf_ ? "pinned: misses DMA, next token prefetched" : "pinned: misses DMA")
                                    : (pinnable ? "not pinned: a miss copies through staging"
                                                : "more than half the host RAM, not page-locked: a miss copies through staging"));
    }
    // The prompt's routing is known: hand the bytes to the layers whose routing spreads over more experts. Placement
    // only - the tokens do not change - and a no-op when the slot count was fixed by hand.
    void replan_banks() {
        if (!bank_ || slots_ > 0) return;
        for (int l : bank_->banks_lru()) bank_->release_bank(l);
        bank_->set_layer_slots(plan_caps());
    }
    void report_bank() {
        if (!bank_) return;
        const auto s = bank_->stats();
        std::fprintf(stderr,
                     "\nglm5next: expert banks: hits %llu misses %llu evictions %llu, %.2f GB moved, prefetch %llu used %llu, "
                     "%.2f GB on the device (peak %.2f)%s\n",
                     (unsigned long long)s.hits, (unsigned long long)s.misses, (unsigned long long)s.evictions,
                     s.transfer_bytes / 1e9, (unsigned long long)bank_->prefetched(), (unsigned long long)bank_->prefetch_used(),
                     s.device_bytes / 1e9, bank_->peak_device_bytes() / 1e9, pf_ ? ", prefetch on" : "");
    }
    // per-head matrices of the latent MLA ([heads][rows][cols] FP16 on the device)
    const __half * heads_f16(const std::string & n) {
        const auto v = host_f32(n);
        std::vector<__half> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
        return static_cast<const __half *>(put(h.data(), h.size() * sizeof(__half)));
    }

    void load() {
        slot_bytes_.assign(L_.size(), 0);
        spread_.assign(L_.size(), 0.0);
        seen_.assign(L_.size(), std::vector<uint8_t>(size_t(experts_), 0));
        pf_hist_.assign(L_.size(), {});
        for (size_t l = 0; l < L_.size(); ++l) {
            auto & Y = L_[l];
            const std::string p = "blk." + std::to_string(l) + ".";
            Y.attn_norm = vec(p + "attn_norm.weight"); Y.ffn_norm = vec(p + "ffn_norm.weight");
            const char * kinds[2] = {"attn", "ffn"};
            for (int k = 0; k < 2; ++k) {
                const std::string h = p + "hc_" + kinds[k];
                Y.hc_fn[k] = mat(h + "_fn.weight"); Y.hc_base[k] = vec(h + "_base.weight"); Y.hc_scale[k] = vec(h + "_scale.weight");
            }
            if (Y.recurrent) {
                Y.wqkv = mat_rows({p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"});
                Y.fgb = mat_rows({p + "ssm_f_a.weight", p + "ssm_g_a.weight", p + "ssm_beta.weight"});
                Y.f_b = mat(p + "ssm_f_b.weight"); Y.g_b = mat(p + "ssm_g_b.weight");
                fdim_ = int(tensor(p + "ssm_f_a.weight").ne[1]); gdim_ = int(tensor(p + "ssm_g_a.weight").ne[1]);
                std::vector<float> cw;
                for (const char * c : {"ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight"}) {
                    const auto v = host_f32(p + c);
                    if (v.size() != size_t(heads_) * kda_dim_ * conv_) throw std::runtime_error("glm5next: conv shape " + p + c);
                    cw.insert(cw.end(), v.begin(), v.end());
                }
                Y.conv = static_cast<const float *>(put(cw.data(), cw.size() * 4));
                Y.dt_bias = vec(p + "ssm_dt.bias"); Y.A = vec(p + "ssm_a"); Y.o_norm = vec(p + "ssm_norm.weight");
            } else {
                Y.q_a = mat(p + "attn_q_a.weight"); Y.q_a_norm = vec(p + "attn_q_a_norm.weight");
                Y.q_b = mat(p + "attn_q_b.weight"); Y.kv_a = mat(p + "attn_kv_a_mqa.weight");
                Y.kv_a_norm = vec(p + "attn_kv_a_norm.weight");
                Y.kb = heads_f16(p + "attn_k_b.weight"); Y.vb = heads_f16(p + "attn_v_b.weight");
                if (dsa_ && meta_.tensors.count(p + "indexer.attn_k.weight")) {
                    Y.idx_k = mat(p + "indexer.attn_k.weight"); Y.idx_g = mat(p + "indexer_compressor_gate.weight");
                    Y.idx_q = mat(p + "indexer.attn_q_b.weight"); Y.idx_w = mat(p + "indexer.proj.weight");
                    Y.idx_ln_w = vec(p + "indexer.k_norm.weight"); Y.idx_ln_b = vec(p + "indexer.k_norm.bias");
                    Y.idx_ape = vec(p + "indexer_compressor_ape.weight");
                }
            }
            Y.wo = mat(p + "attn_output.weight");
            if (int(l) < dense_lead_) {
                Y.gate = mat(p + "ffn_gate.weight"); Y.up = mat(p + "ffn_up.weight"); Y.down = mat(p + "ffn_down.weight");
            } else {
                Y.router = mat(p + "ffn_gate_inp.weight"); Y.probs_bias = vec(p + "exp_probs_b.bias");
                if (bank_mode_) {  // the experts come from the shared bank: only their loaded layout matters here
                    std::array<ExpertFormat, 3> fmt{};
                    const auto parts = repack_expert(expert_payload(l, 0), H_, expert_ff_, false, &fmt);
                    for (int pr = 0; pr < 3; ++pr) {
                        if (!fmt[pr].bank_row_layout)
                            throw std::runtime_error("glm5next: " + p + kGlmExpertProjection[pr] +
                                                     " has no grouped-expert row layout for a bank");
                        chunk_ = std::min(chunk_, fmt[pr].max_matvec_cols);
                    }
                    for (const auto & part : parts) {
                        slot_bytes_[l] += part.size();
                        bank_bytes_ += size_t(slots_) * part.size();  // the default budget of the fixed-slot path
                    }
                } else {
                    Y.e_gate = experts(p + "ffn_gate_exps.weight"); Y.e_up = experts(p + "ffn_up_exps.weight");
                    Y.e_down = experts(p + "ffn_down_exps.weight");
                }
                Y.sh_gate = mat(p + "ffn_gate_shexp.weight"); Y.sh_up = mat(p + "ffn_up_shexp.weight");
                Y.sh_down = mat(p + "ffn_down_shexp.weight");
            }
            std::fprintf(stderr, "\rglm5next: loaded layer %zu / %zu", l + 1, L_.size());
        }
        if (bank_mode_) open_bank();
        out_norm_ = vec("output_norm.weight");
        out_ = mat("output.weight");
        std::fprintf(stderr, "\n");
    }

    // the routed experts as grouped GEMMs (the whole stack on the device: not the bank's slots)
    bool experts_have_mmq() const {
        if (bank_mode_) return false;
        for (size_t l = size_t(dense_lead_); l < L_.size(); ++l)
            if (!L_[l].e_gate.fmt || !L_[l].e_up.fmt || !L_[l].e_down.fmt) return false;
        return true;
    }
    void alloc() {
        if (!experts_have_mmq()) pchunk_ = chunk_;
        pchunk_ = std::max(pchunk_, chunk_);
        const int d = heads_ * kda_dim_, N = pchunk_;
        auto f = [](size_t n) { return Buf(n * sizeof(float)); };
        if (pchunk_ > chunk_) {  // prompt chunks: one FP16 image of the widest non-head matrix, the expert grouping, MMQ inputs
            size_t widest = 0;
            for (const auto & Y : L_)
                for (const M * w : {&Y.wqkv, &Y.wo, &Y.fgb, &Y.f_b, &Y.g_b, &Y.q_a, &Y.q_b, &Y.kv_a, &Y.idx_q, &Y.gate, &Y.up,
                                    &Y.down, &Y.sh_gate, &Y.sh_up, &Y.sh_down})
                    if (w->rows && !w->f32 && w->m.fmt) widest = std::max(widest, size_t(w->rows) * w->cols);
            wdq_ = Buf(std::max<size_t>(1, widest) * sizeof(__half));
            const int fe = L_[size_t(dense_lead_)].e_gate.rows / experts_;
            moe_counts_ = Buf(size_t(experts_) * sizeof(int)); moe_groups_ = Buf(size_t(experts_) * N * sizeof(int));
            mq_x_ = Buf(tqllm_cuda::mmq_act_bytes(N, H_)); mq_a_ = Buf(tqllm_cuda::mmq_act_bytes(N * topk_, fe));
            mq_tiles_ = Buf(size_t(tqllm_cuda::mmq_max_tiles(N, experts_)) * sizeof(int2));
            tqllm_cuda::mmq_warm();
            if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("glm5next: cuBLAS init failed");
            cublasSetStream(cublas_, stream_);
            std::fprintf(stderr, "glm5next: prompt chunks of %d tokens (GEMM + grouped MMQ experts)\n", pchunk_);
        }
        S_ = f(size_t(N) * 4 * H_); S2_ = f(size_t(N) * 4 * H_); x_ = f(size_t(N) * H_); xn_ = f(size_t(N) * 4 * H_);
        xh_ = Buf(size_t(N) * 4 * H_ * sizeof(__half)); mix_ = f(size_t(N) * 64); post_ = f(size_t(N) * 4);
        comb_ = f(size_t(N) * 16); fo_ = f(size_t(N) * H_);
        const int widest = std::max({max_rows_, topk_ * std::max(expert_ff_, H_), 4 * H_, d, ff_});
        bfa_ = Buf(size_t(N) * widest * sizeof(bf16)); bfb_ = Buf(size_t(N) * widest * sizeof(bf16));
        scratch_h_ = Buf(size_t(N) * std::max({4 * H_, d, ff_, q_lora_, heads_ * dv_, topk_ * expert_ff_}) * sizeof(__half));
        proj_ = f(size_t(N) * 3 * d); cv_ = f(size_t(N) * 3 * d); g1_ = f(size_t(N) * d); fa_ = f(size_t(N) * 1024);
        beta_ = f(size_t(N) * heads_); o_ = f(size_t(N) * std::max({d, ff_, sh_ff_})); og_ = f(size_t(N) * std::max({d, ff_, sh_ff_})); g2_ = f(size_t(N) * d);
        qr_ = f(size_t(N) * q_lora_); q_ = f(size_t(N) * heads_ * dk_); qabs_ = f(size_t(N) * heads_ * kv_lora_);
        kvh_ = f(size_t(N) * heads_ * kv_lora_); att_ = f(size_t(N) * heads_ * dv_);
        logits_ = f(std::max(size_t(chunk_) * vocab_, size_t(N) * experts_)); d_arg_ = Buf(size_t(N) * sizeof(int));
        d_ids_ = Buf(size_t(N) * topk_ * sizeof(int)); d_w_ = f(size_t(N) * topk_);
        d_slots_ = Buf(size_t(N) * topk_ * sizeof(int));  // the bank's slot ids of one chunk's assignments
        ye_g_ = Buf(size_t(N) * topk_ * expert_ff_ * sizeof(bf16)); ye_u_ = Buf(size_t(N) * topk_ * expert_ff_ * sizeof(bf16));
        act_ = Buf(size_t(N) * topk_ * std::max(expert_ff_, std::max(ff_, sh_ff_)) * sizeof(__half));
        yd_ = Buf(size_t(N) * topk_ * H_ * sizeof(bf16)); sh_ = f(size_t(N) * H_);
        partial_ = f(std::max<size_t>(1, size_t(N) * max_split_));
        d_pos_ = Buf(sizeof(int)); lat_stage_ = f(size_t(N) * kv_lora_);
        ck(cudaHostAlloc(&h_pos_, sizeof(int), cudaHostAllocDefault));
        ck(cudaHostAlloc(&h_embed_, size_t(N) * 4 * H_ * sizeof(float), cudaHostAllocDefault));
        int nlat = 0;
        for (auto & Y : L_) if (!Y.recurrent) Y.lat = nlat++;
        kvmem_ = make_cuda_kv_memory(stream_);
        lat_pool_ = std::make_unique<KvPool>(*kvmem_, KvGeometry{nlat, size_t(kv_lora_) * sizeof(__half), 0, int64_t(context_)}, 1, 0);
        lat_store_ = kv_store_policy("flat")->make(*lat_pool_, int64_t(context_));
        if (!lat_store_->reserve(int64_t(context_))) throw std::runtime_error("glm5next: the latent cache does not fit");
        if (dsa_) {
            const int D = idx_dim_, pools = int(context_) / kpool_ + 1;
            for (auto & Y : L_)
                if (!Y.recurrent && Y.idx_k.rows) {
                    Y.idx_kg = Buf(size_t(dsa_ring()) * 2 * D * sizeof(__half));
                    Y.idx_pool = f(size_t(pools) * D);
                }
            dsa_k_ = f(size_t(N) * D); dsa_g_ = f(size_t(N) * D);
            dsa_q_ = f(size_t(N) * idx_heads_ * D); dsa_w_ = f(size_t(N) * idx_heads_);
            const int stride = index_topk_ + kpool_ - 1;
            dsa_cells_ = Buf(size_t(N) * stride * sizeof(int)); dsa_count_ = Buf(size_t(N) * sizeof(int));
            dsa_scr_bytes_ = tqllm_cuda::glm_dsa_select_bytes(kDsaGroup, pools);
            dsa_scr_ = Buf(dsa_scr_bytes_);
        }
        for (auto & Y : L_) {
            if (Y.recurrent) {
                Y.hist = f(size_t(3) * (conv_ - 1) * d);
                Y.state = f(size_t(heads_) * kda_dim_ * kda_dim_);
            }
        }
        reset();
    }
    void reset() {
        const int d = heads_ * kda_dim_;
        for (auto & Y : L_) if (Y.recurrent) {
            Y.saved = 0;
            ck(cudaMemsetAsync(Y.hist.p, 0, size_t(3) * (conv_ - 1) * d * sizeof(float), stream_));
            ck(cudaMemsetAsync(Y.state.p, 0, size_t(heads_) * kda_dim_ * kda_dim_ * sizeof(float), stream_));
        }
        verified_ = 0;
        in_block_ = false;
        rec_off_ = 0;
        // a new prompt: the routing spread is measured again, and the plan follows it
        for (auto & v : seen_) std::fill(v.begin(), v.end(), 0);
        std::fill(spread_.begin(), spread_.end(), 0.0);
    }

    void dump(const char * name, int l, const float * v, size_t n) {
        if (!dump_) return;
        std::vector<float> h(n);
        ck(cudaMemcpyAsync(h.data(), v, n * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        ck(cudaStreamSynchronize(stream_));
        double s = 0;
        for (float x : h) s += x;
        std::printf("dump %s-%d %.6f\n", name, l, s);
    }

    // y[n][rows] = W x[n] (x FP32 [n][cols])
    void mv(const M & w, const float * x, float * y, int n) {
        if (w.f32) {
            if (w.rows <= kSmallRows)
                tqllm_cuda::glm_mv_f32_splitk(w.f32, x, y, partial_.as<float>(), w.rows, w.cols, n, splits(w.cols), stream_);
            else tqllm_cuda::glm_mv_f32(w.f32, x, y, w.rows, w.cols, n, stream_);
            return;
        }
        tqllm_cuda::glm_f32_to_half(x, scratch_h_.as<__half>(), w.cols * n, stream_);
        if (n > w.m.fmt->max_matvec_cols()) { gemm_h(w, scratch_h_.as<__half>(), y, n); return; }
        if (w.f32_out != 0) {  // FP32 outputs where the format has them (BF16 rounds every output to 8 mantissa bits)
            try {
                w.m.fmt->matvec_f32(w.m, scratch_h_.as<__half>(), y, n, stream_);
                w.f32_out = 1;
                return;
            } catch (const std::runtime_error &) {
                if (w.f32_out == 1) throw;
                w.f32_out = 0;
            }
        }
        w.m.fmt->matvec(w.m, scratch_h_.as<__half>(), bfa_.as<bf16>(), n, stream_);
        tqllm_cuda::glm_bf16_to_f32(bfa_.as<bf16>(), y, w.rows * n, stream_);
    }
    // a prompt chunk wider than the matvec: the matrix dequantised to FP16 once, one cuBLAS GEMM (FP32 accumulate) into y
    void gemm_h(const M & w, const __half * xh, float * y, int n) {
        if (!cublas_ || size_t(w.rows) * w.cols * sizeof(__half) > wdq_.bytes)
            throw std::runtime_error("glm5next: no prompt GEMM for a " + std::to_string(w.rows) + " x " + std::to_string(w.cols) + " matrix");
        w.m.fmt->dequant_h(w.m, wdq_.as<__half>(), stream_);
        const float one = 1.f, zero = 0.f;
        if (cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, n, w.cols, &one, wdq_.p, CUDA_R_16F, w.cols, xh, CUDA_R_16F, w.cols,
                         &zero, y, CUDA_R_32F, w.rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("glm5next: cuBLAS FP16 GEMM failed");
    }
    // BF16 output, FP16 input (FFN matrices: always a weight format)
    void mv_bf16(const M & w, const __half * xh, bf16 * y, int n) {
        if (w.f32) throw std::runtime_error("glm5next: FFN matrix without a weight format");
        w.m.fmt->matvec(w.m, xh, y, n, stream_);
    }

    void hc_pre(Layer & Y, int k, int l, const char * name, int n) {
        tqllm_cuda::glm_rmsnorm(S_.as<float>(), nullptr, xn_.as<float>(), nullptr, 4 * H_, rms_eps_, stream_, n);
        mv(Y.hc_fn[k], xn_.as<float>(), mix_.as<float>(), n);
        dump("hc_mixes", l, mix_.as<float>(), size_t(n) * Y.hc_fn[k].rows);
        tqllm_cuda::glm_hc_pre(mix_.as<float>(), Y.hc_scale[k], Y.hc_base[k], S_.as<float>(), x_.as<float>(), post_.as<float>(),
                               comb_.as<float>(), H_, iters_, hc_eps_, stream_, n, Y.hc_fn[k].rows);
        dump(name, l, x_.as<float>(), size_t(n) * H_);
    }
    void hc_post(int l, const char * name, int n) {
        tqllm_cuda::glm_hc_post(fo_.as<float>(), S_.as<float>(), post_.as<float>(), comb_.as<float>(), S2_.as<float>(), H_, stream_, n);
        std::swap(S_, S2_);
        dump(name, l, S_.as<float>(), size_t(n) * 4 * H_);
    }

    // A verify block of n > 1 is what commit(k) may roll back: snapshot the state and the conv history once, then every
    // chunk saves its own KDA inputs below the previous chunk's rows. A single decode step records nothing.
    // A block longer than kMaxSavedBlock is not recorded: the saved rows cost ~7 * heads * kda_dim floats per token per
    // recurrent layer, which a prompt (a verify of thousands of tokens) cannot pay. commit(k) on an unrecorded block
    // throws, so only drafting blocks - which are what rolls back - get the snapshot.
    void begin_block(int n, uint32_t pos) {
        if (pos == 0) reset();  // forward() skips its own reset while a block is being recorded
        in_block_ = n > 1 && n <= kMaxSavedBlock;
        rec_off_ = 0;
        if (!in_block_) return;
        const int d = heads_ * kda_dim_;
        auto keep = [](Buf & b, size_t floats) {
            if (b.bytes < floats * sizeof(float)) b = Buf(floats * sizeof(float));
        };
        keep(rb_out_, size_t(n) * d);  // the replay's scratch output: a block can be wider than the chunk scratch
        for (auto & Y : L_) {
            if (!Y.recurrent) continue;
            keep(Y.hist0, size_t(3) * (conv_ - 1) * d);
            keep(Y.state0, size_t(heads_) * kda_dim_ * kda_dim_);
            keep(Y.kqkv, size_t(n) * 3 * d);
            keep(Y.kconv, size_t(n) * 3 * d);
            keep(Y.kg, size_t(n) * d);
            keep(Y.kbeta, size_t(n) * heads_);
            ck(cudaMemcpyAsync(Y.hist0.p, Y.hist.p, size_t(3) * (conv_ - 1) * d * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            ck(cudaMemcpyAsync(Y.state0.p, Y.state.p, size_t(heads_) * kda_dim_ * kda_dim_ * sizeof(float), cudaMemcpyDeviceToDevice,
                               stream_));
            Y.saved = 0;
        }
    }

    // the block's own inputs of this chunk: commit(k) replays the first k tokens from these, no re-projection
    void save_kda(Layer & Y, int n) {
        const int d = heads_ * kda_dim_;
        const size_t off = size_t(rec_off_);
        ck(cudaMemcpyAsync(Y.kqkv.as<float>() + off * 3 * d, proj_.p, size_t(n) * 3 * d * sizeof(float), cudaMemcpyDeviceToDevice,
                           stream_));
        ck(cudaMemcpyAsync(Y.kconv.as<float>() + off * 3 * d, cv_.p, size_t(n) * 3 * d * sizeof(float), cudaMemcpyDeviceToDevice,
                           stream_));
        ck(cudaMemcpyAsync(Y.kg.as<float>() + off * d, g1_.p, size_t(n) * d * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        ck(cudaMemcpyAsync(Y.kbeta.as<float>() + off * heads_, beta_.p, size_t(n) * heads_ * sizeof(float), cudaMemcpyDeviceToDevice,
                           stream_));
        Y.saved += n;
    }

    void kda(Layer & Y, int l, int n) {
        const int d = heads_ * kda_dim_;
        const float * xn = xn_.as<float>();
        float * pq = proj_.as<float>();  // [n][q | k | v][d]
        mv(Y.wqkv, xn, pq, n);
        const bool rb = in_block_;
        float * cv = cv_.as<float>();  // same layout
        tqllm_cuda::glm_kda_conv(pq, Y.hist.as<float>(), Y.conv, cv, d, conv_, stream_, n, 3 * d, d);
        if (dump_) {
            const char * nm[3] = {"kda_q_conv", "kda_k_conv", "kda_v_conv"};
            for (int i = 0; i < 3; ++i) {
                tqllm_cuda::glm_gather_cols(cv, 3 * d, i * d, d, n, o_.as<float>(), stream_);
                dump(nm[i], l, o_.as<float>(), size_t(n) * d);
            }
        }
        // f_a | g_a | beta in one pass: [n][fdim + gdim + heads]
        const int fg = fdim_ + gdim_ + heads_;
        float * fgb = fa_.as<float>();
        mv(Y.fgb, xn, fgb, n);
        // gather each part into its own [n][rows] block for the next matvec / kernels
        tqllm_cuda::glm_gather_cols(fgb, fg, 0, fdim_, n, g2_.as<float>(), stream_);           // f_a -> g2_ scratch
        mv(Y.f_b, g2_.as<float>(), g1_.as<float>(), n);
        tqllm_cuda::glm_kda_gate(g1_.as<float>(), Y.dt_bias, Y.A, gate_lb_, heads_, kda_dim_, stream_, n);
        dump("kda_g1", l, g1_.as<float>(), size_t(n) * d);
        tqllm_cuda::glm_gather_cols(fgb, fg, fdim_ + gdim_, heads_, n, beta_.as<float>(), stream_);
        tqllm_cuda::glm_sigmoid(beta_.as<float>(), n * heads_, stream_);
        dump("kda_beta", l, beta_.as<float>(), size_t(n) * heads_);
        if (rb) save_kda(Y, n);
        // the state updates in place, exactly as in a single step: the pre-block snapshot is in state0, and commit(k)
        // replays the first k tokens of the block into `state` from it
        tqllm_cuda::glm_kda_recur(cv, cv + d, cv + 2 * d, g1_.as<float>(), beta_.as<float>(), Y.state.as<float>(), nullptr,
                                  o_.as<float>(), heads_, kda_dim_, stream_, n, 3 * d);
        dump("kda_scan_out", l, o_.as<float>(), size_t(n) * d);
        tqllm_cuda::glm_gather_cols(fgb, fg, fdim_, gdim_, n, og_.as<float>(), stream_);     // g_a -> og_ scratch
        mv(Y.g_b, og_.as<float>(), g2_.as<float>(), n);
        dump("kda_g2", l, g2_.as<float>(), size_t(n) * d);
        tqllm_cuda::glm_kda_out(o_.as<float>(), g2_.as<float>(), Y.o_norm, og_.as<float>(), nullptr, heads_, kda_dim_, rms_eps_,
                                stream_, n);
        dump("kda_gated", l, og_.as<float>(), size_t(n) * d);
        mv(Y.wo, og_.as<float>(), fo_.as<float>(), n);
        dump("kda_out", l, fo_.as<float>(), size_t(n) * H_);
    }

    // one row per cached position of this MLA layer, from the shared latent cache
    __half * latent(int l) const { return static_cast<__half *>(lat_store_->base(L_[size_t(l)].lat, true)); }

    void mla(Layer & Y, int l, uint32_t pos, int n) {
        const float * xn = xn_.as<float>();
        mv(Y.q_a, xn, qr_.as<float>(), n);
        tqllm_cuda::glm_rmsnorm(qr_.as<float>(), Y.q_a_norm, qr_.as<float>(), nullptr, q_lora_, rms_eps_, stream_, n);
        dump("q_resid", l, qr_.as<float>(), size_t(n) * q_lora_);
        mv(Y.q_b, qr_.as<float>(), q_.as<float>(), n);
        // new latents go to cache rows *d_pos_ .. + n - 1 (the position lives on the device so a graph can replay this)
        float * lat = lat_stage_.as<float>();
        mv(Y.kv_a, xn, lat, n);
        tqllm_cuda::glm_rmsnorm(lat, Y.kv_a_norm, lat, nullptr, kv_lora_, rms_eps_, stream_, n);
        dump("kv_cmpr", l, lat, size_t(n) * kv_lora_);
        __half * cache = latent(l);
        tqllm_cuda::glm_copy_rows_at(lat, cache, kv_lora_, n, d_pos_.as<int>(), stream_);
        tqllm_cuda::glm_mla_qabs(q_.as<float>(), Y.kb, qabs_.as<float>(), heads_, kv_lora_, dk_, stream_, n);
        const float scale = 1.f / std::sqrt(float(dk_));
        if (Y.idx_k.rows) {  // indexer key | gate rows and the pools these tokens complete (positions on the device: graphs)
            mv(Y.idx_k, xn, dsa_k_.as<float>(), n);
            mv(Y.idx_g, xn, dsa_g_.as<float>(), n);
            tqllm_cuda::glm_dsa_store(dsa_k_.as<float>(), dsa_g_.as<float>(), Y.idx_ln_w, Y.idx_ln_b, Y.idx_kg.as<__half>(), Y.idx_ape,
                                      Y.idx_pool.as<float>(), idx_dim_, kpool_, ln_eps_, n, d_pos_.as<int>(), 0, dsa_ring(), stream_);
        }
        if (Y.idx_k.rows && pos + uint32_t(n) > uint32_t(index_topk_)) {  // past index_topk_ positions: the indexer's cells
            mv(Y.idx_q, qr_.as<float>(), dsa_q_.as<float>(), n);
            mv(Y.idx_w, xn, dsa_w_.as<float>(), n);
            const int D = idx_dim_, stride = index_topk_ + kpool_ - 1;
            for (int t0 = 0; t0 < n; t0 += kDsaGroup) {
                const int nt = std::min(kDsaGroup, n - t0);
                tqllm_cuda::glm_dsa_select(dsa_q_.as<float>() + size_t(t0) * idx_heads_ * D, dsa_w_.as<float>() + size_t(t0) * idx_heads_,
                                           Y.idx_pool.as<float>(), D, kpool_, index_topk_ / kpool_, nt, int(pos) + t0, dsa_scr_.p,
                                           dsa_scr_bytes_, dsa_cells_.as<int>() + size_t(t0) * stride, dsa_count_.as<int>() + t0, stride,
                                           stream_);
            }
            tqllm_cuda::glm_mla_attn_sparse(qabs_.as<float>(), cache, kvh_.as<float>(), heads_, kv_lora_, dsa_cells_.as<int>(),
                                            dsa_count_.as<int>(), stride, scale, n, stream_);
        } else {
            // dense: every position before the token (the decode graph's bound is index_topk_ when the file has an indexer:
            // the graph only runs below it)
            const uint32_t bound = Y.idx_k.rows ? std::min(context_, uint32_t(index_topk_)) : context_;
            tqllm_cuda::glm_mla_attn(qabs_.as<float>(), cache, kvh_.as<float>(), heads_, kv_lora_,
                                     int(n == 1 ? bound : pos + 1), scale, stream_, n, n == 1 ? d_pos_.as<int>() : nullptr);
        }
        (void) pos;
        tqllm_cuda::glm_mla_vb(kvh_.as<float>(), Y.vb, att_.as<float>(), nullptr, heads_, kv_lora_, dv_, stream_, n);
        dump("kqv_out", l, att_.as<float>(), size_t(n) * heads_ * dv_);
        mv(Y.wo, att_.as<float>(), fo_.as<float>(), n);
        dump("attn_out", l, fo_.as<float>(), size_t(n) * H_);
    }

    // SwiGLU FFN; x FP32 [n][cols], result FP32 into `out` [n][dn.rows]
    void ffn(const M & g, const M & u, const M & dn, const float * x, float limit, float * out, int n) {
        float * a = o_.as<float>(), * b = og_.as<float>();  // free here: [n][d] >= [n][ff]
        mv(g, x, a, n);
        mv(u, x, b, n);
        tqllm_cuda::glm_swiglu_f32(a, b, act_.as<__half>(), g.rows * n, limit, stream_);
        if (dn.f32) throw std::runtime_error("glm5next: FFN down matrix without a weight format");
        if (n > dn.m.fmt->max_matvec_cols()) { gemm_h(dn, act_.as<__half>(), out, n); return; }
        if (dn.f32_out != 0) {
            try { dn.m.fmt->matvec_f32(dn.m, act_.as<__half>(), out, n, stream_); dn.f32_out = 1; return; }
            catch (const std::runtime_error &) { if (dn.f32_out == 1) throw; dn.f32_out = 0; }
        }
        dn.m.fmt->matvec(dn.m, act_.as<__half>(), bfa_.as<bf16>(), n, stream_);
        tqllm_cuda::glm_bf16_to_f32(bfa_.as<bf16>(), out, dn.rows * n, stream_);
    }

    // the whole expert stack of the layer on the device (TQLLM_GLM_EXPERT_SLOTS unset)
    void moe_stack(Layer & Y, int n) {
        const float * xn = xn_.as<float>();
        __half * xh = xh_.as<__half>();
        tqllm_cuda::glm_f32_to_half(xn, xh, H_ * n, stream_);
        mv(Y.router, xn, logits_.as<float>(), n);  // scratch: the head overwrites it later
        tqllm_cuda::glm_route(logits_.as<float>(), Y.probs_bias, d_ids_.as<int>(), d_w_.as<float>(), experts_, topk_, w_norm_,
                              w_scale_, stream_, n);
        const int fe = Y.e_gate.rows / experts_;
        if (n > chunk_) { moe_grouped(Y, n, fe); return; }
        bool done = false;
        if (Y.e_gate.fmt == Y.e_up.fmt && Y.paired != 0) {  // one launch for gate and up (same kernel per row)
            try {
                Y.e_gate.fmt->expert_matvec2(Y.e_gate, Y.e_up, xh, ye_g_.as<bf16>(), ye_u_.as<bf16>(), fe, d_ids_.as<int>(), n,
                                             topk_, stream_);
                Y.paired = 1;
                done = true;
            } catch (const std::runtime_error &) {
                if (Y.paired == 1) throw;
                Y.paired = 0;
            }
        }
        if (!done) {
            Y.e_gate.fmt->expert_matvec(Y.e_gate, xh, ye_g_.as<bf16>(), fe, d_ids_.as<int>(), n, topk_, false, stream_);
            Y.e_up.fmt->expert_matvec(Y.e_up, xh, ye_u_.as<bf16>(), fe, d_ids_.as<int>(), n, topk_, false, stream_);
        }
        tqllm_cuda::glm_swiglu(ye_g_.as<bf16>(), ye_u_.as<bf16>(), act_.as<__half>(), n * topk_ * fe, Y.clamp_exp, stream_);
        Y.e_down.fmt->expert_matvec(Y.e_down, act_.as<__half>(), yd_.as<bf16>(), H_, d_ids_.as<int>(), n, topk_, true, stream_);
        ffn(Y.sh_gate, Y.sh_up, Y.sh_down, xn, Y.clamp_sh, sh_.as<float>(), n);
        tqllm_cuda::glm_moe_sum(yd_.as<bf16>(), d_w_.as<float>(), sh_.as<float>(), fo_.as<float>(), topk_, H_, stream_, n);
    }

    // a prompt chunk: tokens grouped by routed expert, each expert's rows read once for all its tokens (int8 tensor-core
    // GEMM on the GGUF blocks, MiniMax's moe_grouped); the per-assignment layout [n * topk][rows] of expert_matvec
    void moe_grouped(Layer & Y, int n, int fe) {
        const int A = n * topk_;
        const int * counts = moe_counts_.as<int>(), * groups = moe_groups_.as<int>();
        tqllm_cuda::moe_group(d_ids_.as<int>(), moe_counts_.as<int>(), moe_groups_.as<int>(), experts_, n, topk_, stream_);
        const int ig = Y.e_gate.fmt->gemm_f32_id(), iu = Y.e_up.fmt->gemm_f32_id(), id = Y.e_down.fmt->gemm_f32_id();
        std::vector<int> ch(static_cast<size_t>(experts_));
        ck(cudaMemcpyAsync(ch.data(), moe_counts_.p, ch.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        ck(cudaStreamSynchronize(stream_));
        const int tcap = tqllm_cuda::mmq_max_tiles(n, experts_);
        auto mmq = [&](const Mat & m, int fid, int rows, const void * yq, int ny, int div, bf16 * y) {
            if (!tqllm_cuda::mmq_expert_gemm(fid, static_cast<const uint8_t *>(m.p), m.aux, false, rows, m.cols, counts, ch.data(), groups,
                                             n, experts_, div, yq, ny, y, mq_tiles_.as<int2>(), tcap, stream_, nullptr, nullptr))
                throw std::runtime_error(std::string("glm5next: MMQ expert GEMM refused ") + m.fmt->name());
        };
        // a projection takes MMQ when its format has it, else the format's own grouped expert GEMM (K4: Q4_K layers)
        auto run_proj = [&](const Mat & m, int fid, int rows, const __half * x, int nx, bool per_slot, bf16 * y, void * yq, bool & quantised) {
            if (tqllm_cuda::mmq_supports(fid, m.cols)) {
                if (!quantised) { tqllm_cuda::mmq_quantize_act(x, nx, m.cols, yq, stream_, fid == 11); quantised = true; }
                mmq(m, fid, rows, yq, nx, per_slot ? 1 : topk_, y);
                return;
            }
            if (!m.fmt->expert_gemm(m, x, y, rows, counts, groups, n, experts_, topk_, per_slot, stream_))
                throw std::runtime_error(std::string("glm5next: no grouped expert GEMM for ") + m.fmt->name());
        };
        bool qx = false, qa = false;
        run_proj(Y.e_gate, ig, fe, xh_.as<__half>(), n, false, ye_g_.as<bf16>(), mq_x_.p, qx);
        if (iu != ig) qx = false;
        run_proj(Y.e_up, iu, fe, xh_.as<__half>(), n, false, ye_u_.as<bf16>(), mq_x_.p, qx);
        tqllm_cuda::glm_swiglu(ye_g_.as<bf16>(), ye_u_.as<bf16>(), act_.as<__half>(), A * fe, Y.clamp_exp, stream_);
        run_proj(Y.e_down, id, H_, act_.as<__half>(), A, true, yd_.as<bf16>(), mq_a_.p, qa);
        ffn(Y.sh_gate, Y.sh_up, Y.sh_down, xn_.as<float>(), Y.clamp_sh, sh_.as<float>(), n);
        tqllm_cuda::glm_moe_sum(yd_.as<bf16>(), d_w_.as<float>(), sh_.as<float>(), fo_.as<float>(), topk_, H_, stream_, n);
    }

    // TQLLM_GLM_EXPERT_SLOTS: the routed ids go to the host, prepare() loads what the layer's bank lacks, and the grouped
    // expert kernels run on the bank's slot ids (the same rows, another index). A chunk that routes past the slots runs in
    // waves, each expert loaded once per wave; the output is the same token either way.
    void moe_bank(Layer & Y, int l, int n) {
        const float * xn = xn_.as<float>();
        __half * xh = xh_.as<__half>();
        tqllm_cuda::glm_f32_to_half(xn, xh, H_ * n, stream_);
        mv(Y.router, xn, logits_.as<float>(), n);  // scratch: the head overwrites it later
        tqllm_cuda::glm_route(logits_.as<float>(), Y.probs_bias, d_ids_.as<int>(), d_w_.as<float>(), experts_, topk_, w_norm_,
                              w_scale_, stream_, n);
        std::vector<int> routed(size_t(n) * topk_);
        ck(cudaMemcpyAsync(routed.data(), d_ids_.p, routed.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        ck(cudaStreamSynchronize(stream_));
        // what this layer's routing reaches is the plan's per-layer spread, and the last token's experts are the hint for
        // what the next token will want. Placement only: the tokens do not change.
        auto & seen = seen_[size_t(l)];
        for (int id : routed) seen[size_t(id)] = 1;
        int reached = 0;
        for (uint8_t s : seen) reached += s ? 1 : 0;
        spread_[size_t(l)] = double(reached);
        // the hint: what this layer's last tokens routed. The cache skips the ones still in the bank, so the copies it
        // takes are the ones it dropped in between - those overlap this layer's attention on the next call.
        auto & hist = pf_hist_[size_t(l)];
        hist.insert(hist.end(), routed.end() - size_t(std::min(n, kPfHistory)) * topk_, routed.end());
        if (hist.size() > size_t(kPfHistory) * size_t(topk_))
            hist.erase(hist.begin(), hist.begin() + long(hist.size() - size_t(kPfHistory) * topk_));
        const int fe = expert_ff_;
        for (const auto & wv : glm_expert_waves(routed, topk_, bank_->slots(l))) {
            const auto slot = bank_->prepare(l, wv.ids);
            bank_->wait_misses();  // split mode: the misses of this wave land on the copy stream
            std::vector<int> where(size_t(experts_), -1), phys;
            for (size_t i = 0; i < wv.ids.size(); ++i) where[size_t(wv.ids[i])] = slot[i];
            phys.reserve(size_t(wv.tokens) * topk_);
            for (int a = wv.first * topk_; a < (wv.first + wv.tokens) * topk_; ++a) phys.push_back(where[size_t(routed[size_t(a)])]);
            ck(cudaMemcpyAsync(d_slots_.p, phys.data(), phys.size() * sizeof(int), cudaMemcpyHostToDevice, stream_));
            const size_t off = size_t(wv.first) * topk_;
            const Mat g = bank_->gate(), u = bank_->up(), dn = bank_->down();
            const __half * xw = xh + size_t(wv.first) * H_;
            bf16 * gw = ye_g_.as<bf16>() + off * fe, * uw = ye_u_.as<bf16>() + off * fe;
            bool done = false;
            if (g.fmt == u.fmt && Y.paired != 0) {
                try {
                    g.fmt->expert_matvec2(g, u, xw, gw, uw, fe, d_slots_.as<int>(), wv.tokens, topk_, stream_);
                    Y.paired = 1;
                    done = true;
                } catch (const std::runtime_error &) {
                    if (Y.paired == 1) throw;
                    Y.paired = 0;
                }
            }
            if (!done) {
                g.fmt->expert_matvec(g, xw, gw, fe, d_slots_.as<int>(), wv.tokens, topk_, false, stream_);
                u.fmt->expert_matvec(u, xw, uw, fe, d_slots_.as<int>(), wv.tokens, topk_, false, stream_);
            }
            tqllm_cuda::glm_swiglu(gw, uw, act_.as<__half>() + off * fe, wv.tokens * topk_ * fe, Y.clamp_exp, stream_);
            dn.fmt->expert_matvec(dn, act_.as<__half>() + off * fe, yd_.as<bf16>() + off * H_, H_, d_slots_.as<int>(),
                                  wv.tokens, topk_, true, stream_);
        }
        ffn(Y.sh_gate, Y.sh_up, Y.sh_down, xn, Y.clamp_sh, sh_.as<float>(), n);
        tqllm_cuda::glm_moe_sum(yd_.as<bf16>(), d_w_.as<float>(), sh_.as<float>(), fo_.as<float>(), topk_, H_, stream_, n);
    }

    // tokens [n] at positions pos .. pos + n - 1; next[i] = argmax after token i (null: no argmax)
    // a prompt (n > chunk_, not inside a verify block) runs in chunks of pchunk_ tokens and returns next[] for its last
    // position only (the others ~0u); a verify block gets every row's argmax
    void run(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) {
        const int step = n > chunk_ && !in_block_ ? pchunk_ : chunk_;
        if (step > chunk_ && next) std::fill(next, next + n, ~0u);
        for (int i = 0; i < n; i += step) {
            const int c = std::min(step, n - i);
            forward(tokens + i, c, pos + uint32_t(i));
            if (!next) continue;
            const int rows = last_rows_;
            tqllm_cuda::glm_argmax_rows(logits_.as<float>(), vocab_, rows, d_arg_.as<int>(), stream_);
            std::vector<int> ids(static_cast<size_t>(rows));
            ck(cudaMemcpyAsync(ids.data(), d_arg_.p, size_t(rows) * sizeof(int), cudaMemcpyDeviceToHost, stream_));
            ck(cudaStreamSynchronize(stream_));
            for (int j = 0; j < rows; ++j) next[i + c - rows + j] = uint32_t(ids[size_t(j)]);
        }
        // a prompt (not a decode step) just ran: its routing says which layers reach further, and the bytes follow it
        if (n > 1) replan_banks();
    }

    void forward(const uint32_t * tokens, int n, uint32_t pos) {
        if (pos + uint32_t(n) > context_) throw std::runtime_error("glm5next: position beyond the context");
        if (pos == 0 && !in_block_) reset();  // begin_block() reset before snapshotting for a block at position 0
        const auto & et = tensor("token_embd.weight");
        uint64_t el = 0, by = 0;
        gguf_block_layout(et.type, el, by);
        for (int t = 0; t < n; ++t) {
            if (int(tokens[t]) >= vocab_) throw std::runtime_error("glm5next: token id beyond the vocabulary");
            float * r = h_embed_ + size_t(t) * 4 * H_;
            dequantize_host(et.type, bytes("token_embd.weight").data + size_t(tokens[t]) * (uint64_t(H_) / el * by), r, H_);
            for (int i = 1; i < 4; ++i) std::copy(r, r + H_, r + size_t(i) * H_);
        }
        ck(cudaStreamSynchronize(stream_));  // h_embed_ / h_pos_ are reused: the previous step's copies are done
        ck(cudaMemcpyAsync(S_.p, h_embed_, size_t(n) * 4 * H_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
        *h_pos_ = int(pos);
        ck(cudaMemcpyAsync(d_pos_.p, h_pos_, sizeof(int), cudaMemcpyHostToDevice, stream_));
        // a block being recorded runs its chunks (including a trailing chunk of one token) through the same code, so the
        // captured single-token graph is not an option inside one
        if (n == 1 && use_graph_ && !in_block_ && (!dsa_ || pos + 1 <= uint32_t(index_topk_))) {
            if (graph_) { ck(cudaGraphLaunch(graph_, stream_)); last_rows_ = 1; return; }
            if (warmed_) {  // capture the body once; probes (FP32 outputs, paired experts) are settled by the warm step
                cudaGraph_t g = nullptr;
                ck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
                body(1, pos);
                ck(cudaStreamEndCapture(stream_, &g));
                ck(cudaGraphInstantiate(&graph_, g, 0));
                cudaGraphDestroy(g);
                ck(cudaGraphLaunch(graph_, stream_));
                last_rows_ = 1;
                return;
            }
            warmed_ = true;
        }
        body(n, pos);
        if (in_block_) rec_off_ += n;
    }

    void body(int n, uint32_t pos) {
        for (size_t li = 0; li < L_.size(); ++li) {
            auto & Y = L_[li];
            const int l = int(li);
            // the experts this layer's last tokens routed go to its bank on the copy stream now, so the ones it dropped
            // in between land while this layer runs its attention (MixedExpertCache orders the copies after its last read)
            if (pf_ && l >= dense_lead_ && !pf_hist_[li].empty()) bank_->prefetch_released(l, pf_hist_[li], copy_);
            hc_pre(Y, 0, l, "hc_attn_pre", n);
            tqllm_cuda::glm_rmsnorm(x_.as<float>(), Y.attn_norm, xn_.as<float>(), nullptr, H_, rms_eps_, stream_, n);
            dump("attn_norm", l, xn_.as<float>(), size_t(n) * H_);
            if (Y.recurrent) kda(Y, l, n); else mla(Y, l, pos, n);
            hc_post(l, "hc_attn_post", n);
            hc_pre(Y, 1, l, "hc_ffn_pre", n);
            tqllm_cuda::glm_rmsnorm(x_.as<float>(), Y.ffn_norm, xn_.as<float>(), nullptr, H_, rms_eps_, stream_, n);
            dump("ffn_norm", l, xn_.as<float>(), size_t(n) * H_);
            if (l < dense_lead_) {
                ffn(Y.gate, Y.up, Y.down, xn_.as<float>(), Y.clamp_sh, fo_.as<float>(), n);
            } else if (bank_) {
                moe_bank(Y, l, n);
            } else {
                moe_stack(Y, n);
            }
            dump("ffn_out", l, fo_.as<float>(), size_t(n) * H_);
            hc_post(l, "l_out", n);
        }
        tqllm_cuda::glm_hc_mean(S_.as<float>(), x_.as<float>(), H_, stream_, n);
        dump("hc_head", -1, x_.as<float>(), size_t(n) * H_);
        tqllm_cuda::glm_rmsnorm(x_.as<float>(), out_norm_, xn_.as<float>(), nullptr, H_, rms_eps_, stream_, n);
        dump("result_norm", -1, xn_.as<float>(), size_t(n) * H_);
        const int rows = n > chunk_ ? 1 : n;  // a prompt chunk needs its last token's logits only
        mv(out_, xn_.as<float>() + size_t(n - rows) * H_, logits_.as<float>(), rows);
        dump("result_output", -1, logits_.as<float>(), size_t(rows) * vocab_);
        last_rows_ = rows;
    }
};

}  // namespace

bool is_glm5next_gguf(const std::string & path) {
    const std::string a = gguf_architecture(path);
    return a == "glm5-next" || a == "glm5next";
}

std::unique_ptr<Engine> make_glm5next_engine(const std::string & path, uint32_t context) {
    return std::make_unique<Glm5NextEngine>(path, context);
}

}  // namespace tqllm
