// MiniMax-M3.1-Flash (GGUF arch `minimax-m3`, written by tools/minimax_convert.cpp) engine on the GPU.
// Per layer: RMSNorm -> GQA attention (per-head RMSNorm on q / k, partial NeoX RoPE) -> residual; RMSNorm -> dense
// swigluoai FFN (first layers) or sigmoid MoE (top-k of sigmoid + bias, weights normalised x scale) + shared expert ->
// residual; final RMSNorm -> head. Norm weights in the file are already 1 + w. The reference is tools/minimax_ref.py.
// The lightning indexer keeps the top 16 blocks of 128 keys (the query's own block always): up to 16 blocks it keeps
// every block, so attention runs dense there; past that, on the selected blocks only. KV and index keys are FP16.
#include "debug.hpp"
#include "dequant.hpp"
#include "dev_alloc.cuh"
#include "engine.hpp"
#include "expert_bank_plan.hpp"
#include "glm_experts.hpp"
#include "packed_experts.hpp"
#include "glm5next_kernels.cuh"
#include "gguf.hpp"
#include "gguf_mapped.hpp"
#include "minimax_kernels.cuh"
#include "minimax_memory.hpp"
#include "minimax_spec.hpp"
#include "moe_kernels.cuh"
#include "tp_comm.hpp"
#include "quant_kernels.cuh"
#include "weight_format.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tqllm {
namespace {

using bf16 = __nv_bfloat16;

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("minimax: ") + cudaGetErrorString(e));
}

struct Buf {
    void * p = nullptr;
    size_t bytes = 0;
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

struct M {
    Mat m;
    const float * f32 = nullptr;
    int rows = 0, cols = 0;
    mutable int f32_out = -1;
};

struct Layer {
    const float * attn_norm = nullptr, * ffn_norm = nullptr, * q_norm = nullptr, * k_norm = nullptr;
    const float * iq_norm = nullptr, * ik_norm = nullptr;
    M wqkv, wo, iq, ik;
    M gate, up, down, router, sh_gate, sh_up, sh_down;
    M gate_up, sh_gate_up;  // gate | up rows stacked: one launch reads the input once (the down projection stays apart)
    Mat e_gate, e_up, e_down;
    const float * probs_bias = nullptr;
    int paired = -1;
    Buf kc, vc;  // [context][kv heads][head dim] FP16
    Buf ikc;     // index keys [context][index dim] FP16 (null: no indexer in this layer)
};

class MiniMaxEngine final : public Engine {
public:
    MiniMaxEngine(const std::string & path, uint32_t context, const std::string & draft = "", const std::string & tokenizer = "",
                  const TpConfig & tp = {}, const std::string & hot_map = "")
        : context_(context ? context : 32768), hot_map_(hot_map) {
        if (tp.world > 1) {  // both boxes reach this before loading: NCCL comes up first, then each loads its share
            tp_ = std::make_unique<TpComm>(tp.rank, tp.world, tp.peer, tp.port);
            R_ = tp.rank; W_ = tp.world;
        }
        ck(cudaStreamCreate(&stream_));
        int dev = 0, integrated = 0;
        ck(cudaGetDevice(&dev));
        ck(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev));
        integrated_ = integrated != 0;
        meta_ = gguf_open(path);
        configure();
        if (integrated_) {  // a GB10 holds the weights in the host's memory: refuse before loading, not halfway (a second
                            // 94 GB load beside a running one swapped the box until it hung, 10-23)
            size_t base = 0;  // weights + drafter + reserve; the KV cache and the prefill activations come on top
            for (const auto & [n, t] : meta_.tensors) if (minimax_resident(n)) base += size_t(t.bytes) / size_t(W_);
            if (!draft.empty()) for (const auto & [n, t] : gguf_open(draft).tensors) base += size_t(t.bytes);
            base += kMiniMaxHostReserve;
            const size_t avail = mem_available();
            const bool drafting = !draft.empty();
            auto kv_of = [&](size_t elem) { return L_.size() * minimax_kv_layer_bytes(context_, hkl_, hd_, idim_, elem); };
            // the largest prefill chunk whose activations fit beside the rest (8192 .. 1024: every chunk reconstructs each
            // routed X3 expert it touches, so a 4x larger chunk cuts that cost 4x); a FP16 KV cache when the context
            // fits in it, else FP8 E4M3 (half the bytes: the longer context is what the box can hold)
            pchunk_ = chunk_;
            if (W_ == 1) {
                for (size_t elem : {size_t(2), size_t(1)}) {
                    for (int p : {8192, 4096, 2048, 1024})
                        if (!avail || base + kv_of(elem) + act_bytes(p, drafting) <= avail) { pchunk_ = p; break; }
                    if (pchunk_ >= 1024) { kv8_ = elem == 1; break; }
                }
                if (pchunk_ < 1024) {  // never a silent 8-token prefill (6x slower): name the context that fits instead
                    const size_t per_token = L_.size() * minimax_kv_layer_bytes(1, hkl_, hd_, idim_, 1);
                    const size_t fixed = base + act_bytes(1024, drafting);
                    const size_t fit = avail > fixed ? (avail - fixed) / per_token / 1024 * 1024 : 0;
                    throw std::runtime_error("minimax: context " + std::to_string(context_) + " does not fit even with an FP8 KV cache (" +
                                             std::to_string((base + kv_of(1) + act_bytes(1024, drafting)) >> 30) + " GiB needed with the " +
                                             std::to_string(kMiniMaxHostReserve >> 30) + " GiB reserve, " + std::to_string(avail >> 30) +
                                             " GiB available); largest context that fits: " + std::to_string(fit));
                }
            }
            const size_t need = base + kv_of(kv8_ ? 1 : 2) + act_bytes(pchunk_, drafting);
            if (avail && need > avail)
                throw std::runtime_error("minimax: needs " + std::to_string(need >> 30) + " GiB (weights, KV, " + std::to_string(kMiniMaxHostReserve >> 30) + " GiB reserve), " +
                                         std::to_string(avail >> 30) + " GiB available: is another model running?");
            if (kv8_) std::fprintf(stderr, "minimax: KV cache FP8 E4M3 (context %u does not fit in FP16 beside the weights)\n", context_);
            std::fprintf(stderr, "minimax: prefill chunk %d tokens\n", pchunk_);
        }
        map_files(meta_);
        load();
        if (!tokenizer.empty() && !meta_.has("tokenizer.ggml.eos_token_id")) {  // the converter writes no tokenizer keys
            const GgufFile tg = gguf_open_shard(tokenizer);
            if (tg.has("tokenizer.ggml.eos_token_id")) eos_ = uint32_t(tg.num("tokenizer.ggml.eos_token_id"));
        }
        if (!draft.empty()) load_draft(draft);
        if (!integrated_ && W_ == 1) pchunk_ = 2048;
        if (bank_ || tp_) pchunk_ = chunk_;  // the bank and two-box paths keep the verify chunk
        pchunk_ = std::max(pchunk_, chunk_);
        alloc();
        std::fprintf(stderr, "minimax: prefill chunk %d tokens (activations %.2f GiB)\n", pchunk_,
                     double(act_bytes(pchunk_, has_draft())) / double(size_t(1) << 30));
    }
    ~MiniMaxEngine() override {
        try { stop_followers(); } catch (...) {}
        write_routes();
        report_bank();
        bank_.reset();
        L_.clear();
        owned_.clear();
        if (h_embed_) cudaFreeHost(h_embed_);
        if (cublas_) cublasDestroy(cublas_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    uint32_t step(uint32_t token, uint32_t pos, bool sample, float * capture) override {
        if (capture) throw std::runtime_error("minimax: hidden-state capture is not supported");
        uint32_t next = 0;
        run(&token, 1, pos, sample ? &next : nullptr);
        return next;
    }
    void verify(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) override { run(tokens, n, pos, next); }
    void commit(int) override {}  // attention only: rejected positions are overwritten by the next block
    const float * logits() const override { return logits_.as<float>() + size_t(last_rows_ - 1) * vocab_; }
    bool has_drafter() const override { return has_draft(); }
    // Sampling (temperature > 0): each head row's top-16 exact logits -> temperature, top-k (<= 16), top-p; with the
    // DSpark draft, speculative sampling of a deterministic draft (accept d with p(d), else resample from p without d,
    // bonus token from p after a full accept), so the tokens are distributed exactly as plain sampling.
    void set_ignore_eos(bool on) override { ignore_eos_ = on; }
    void set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence) override {
        (void) presence;
        samp_.temperature = temperature > 0.f ? temperature : 0.f;
        samp_.top_p = top_p > 0.f && top_p <= 1.f ? top_p : 1.f;
        samp_.top_k = top_k;
        rng_.seed(seed);
    }
    // Greedy decode, chunked prefill, with the DSpark draft when one is loaded: per step the block [last token, d draft
    // tokens] is verified and the agreeing prefix kept, so the tokens are the verify path's greedy decode (d = 0: one
    // verified token per step). spec_begin prefills, spec_advance runs one step, spec_finish returns the tokens.
    bool incremental() const override { return true; }
    void spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int d) override {
        if (prompt.empty()) throw std::runtime_error("minimax: empty prompt");
        if (prompt.size() >= context_) throw std::runtime_error("minimax: prompt fills the context");
        gen_ = Gen{};
        gen_.count = count;
        // d <= 0: the draft length follows the measured acceptance (minimax_draft_len, up to the DSpark block); d > 0: fixed
        gen_.adaptive = has_draft() && d <= 0;
        gen_.d = has_draft() ? std::max(0, std::min({d > 0 ? d : dblock_, dblock_, chunk_ - 1})) : 0;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<uint32_t> nx(prompt.size());
        run(prompt.data(), int(prompt.size()), 0, nx.data());
        gen_.tok = nx.back();
        gen_.pos = uint32_t(prompt.size());
        gen_.t1 = std::chrono::steady_clock::now();
        gen_.prefill_us = uint64_t(std::chrono::duration<double, std::micro>(gen_.t1 - t0).count());
        gen_.done = count == 0;
    }
    bool spec_advance() override {
        if (gen_.done) return false;
        const int dk = std::min<int>(gen_.adaptive ? minimax_draft_len(gen_.acc.rate(), gen_.d) : gen_.d, int(context_ - gen_.pos) - 1);
        std::vector<uint32_t> blk(size_t(dk) + 1), v(size_t(dk) + 1);
        blk[0] = gen_.tok;
        if (dk > 0) { const auto pr = draft(gen_.tok, gen_.pos); std::copy(pr.begin(), pr.begin() + dk, blk.begin() + 1); }
        run(blk.data(), dk + 1, gen_.pos, v.data());
        int k = 0;
        if (sampling()) {  // v holds no greedy ids here: accept / resample against each column's distribution
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            for (; k < dk; ++k) {
                const Dist d = dist(k);
                double q = 0;
                for (int i = 0; i < d.n; ++i) if (d.id[i] == blk[size_t(k) + 1]) q = d.p[i];
                if (uni(rng_) >= q) { v[size_t(k)] = draw(d, blk[size_t(k) + 1]); break; }
            }
            if (k == dk) v[size_t(k)] = draw(dist(k));
        } else {
            while (k < dk && blk[size_t(k) + 1] == v[size_t(k)]) ++k;
        }
        for (int j = 0; j <= k && !gen_.done; ++j) {
            gen_.out.push_back(blk[size_t(j)]);
            if ((blk[size_t(j)] == eos_ && !ignore_eos_) || gen_.out.size() >= gen_.count) gen_.done = true;
        }
        if (tqllm::debug_on("draft")) std::fprintf(stderr, "minimax draft pos %u drafted %d kept %d\n", gen_.pos, dk, k);
        ++gen_.steps; gen_.drafted += uint64_t(dk); gen_.accepted += uint64_t(k);
        gen_.acc.update(dk, k);
        gen_.tok = v[size_t(k)];
        gen_.pos += uint32_t(k + 1);
        if (gen_.pos + 1 >= context_) gen_.done = true;
        return !gen_.done;
    }
    std::vector<uint32_t> spec_finish(uint64_t * stats) override {
        if (stats) {
            stats[0] = gen_.steps; stats[1] = gen_.drafted; stats[2] = gen_.accepted;
            stats[3] = uint64_t(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - gen_.t1).count());
            stats[4] = gen_.prefill_us;
        }
        auto out = std::move(gen_.out);
        gen_ = Gen{};
        return out;
    }
    const std::vector<uint32_t> & spec_tokens(int) const override { return gen_.out; }
    std::vector<uint32_t> generate_spec(const std::vector<uint32_t> & prompt, uint32_t count, int d, uint64_t * stats) override {
        spec_begin(prompt, count, d);
        while (spec_advance()) {}
        return spec_finish(stats);
    }
    void synchronize() override { ck(cudaStreamSynchronize(stream_)); }
    uint32_t layers() const override { return uint32_t(L_.size()); }
    uint32_t hidden() const override { return uint32_t(H_); }
    uint32_t vocab() const override { return uint32_t(vocab_); }
    uint32_t eos() const override { return eos_; }
    uint32_t context() const override { return context_; }

private:
    GgufFile meta_;
    const GgufFile * cur_ = &meta_;  // the file tensor() / bytes() read: the model, or the draft while it loads
    std::string a_;
    cudaStream_t stream_ = nullptr;
    uint32_t context_ = 0, eos_ = 0;
    int H_ = 0, hq_ = 0, hk_ = 0, hd_ = 0, rot_ = 0, experts_ = 0, topk_ = 0, eff_ = 0, ff_ = 0, sh_ff_ = 0, dense_lead_ = 0;
    int vocab_ = 0, chunk_ = 8, pchunk_ = 8, last_rows_ = 1, max_rows_ = 0;  // pchunk_: prompt tokens per forward
    cublasHandle_t cublas_ = nullptr;  // F32 weights x many tokens (router, indexer keys) in a prompt chunk
    int ih_ = 0, idim_ = 0, itopk_ = 0, iblock_ = 0, ilocal_ = 0;
    // tensor parallelism over W_ boxes (rank R_): this rank's query / KV heads and ff widths; partial sums all-reduced
    int R_ = 0, W_ = 1, hql_ = 0, hkl_ = 0, effl_ = 0, ffl_ = 0, shl_ = 0;
    std::unique_ptr<TpComm> tp_;
    Buf hdr_;  // rank 0 -> rank 1: {n, pos, tokens...} of each forward
    void reduce(float * x, size_t n) { if (tp_) tp_->all_reduce_sum(x, n, stream_); }
    float eps_ = 0, theta_ = 0, w_scale_ = 0, alpha_ = 0, limit_ = 0;
    size_t max_split_ = 0;
    std::vector<std::unique_ptr<MappedGguf>> maps_;
    std::map<std::string, const MappedGguf *> map_of_;
    mutable std::map<std::string, std::vector<char>> built_;
    bool kv8_ = false;  // KV cache (and the indexer keys, and the draft's K / V) in FP8 E4M3
    std::vector<Buf> owned_;
    std::vector<Layer> L_;
    M out_;
    const float * out_norm_ = nullptr;
    Buf iqv_, ikv_, iscore_, iids_;
    // DSpark draft (DFlash-style block drafter): the target's states after layers target_layers_ are concatenated per
    // position, fused (fc + hidden_norm) and projected to each draft layer's context K / V once per position; a draft
    // pass runs the block [anchor token, mask x (B - 1)] at positions anchor .. anchor + B - 1 through the dense layers
    // (block queries see context positions < anchor within the sliding window, and the whole block), then the target's
    // head plus the Markov bias W2 W1[previous token], greedily left to right.
    struct DraftLayer {
        const float * attn_norm = nullptr, * ffn_norm = nullptr, * q_norm = nullptr, * k_norm = nullptr;
        M wq, wkv, wo, gate_up, down;
        Buf kc, vc;  // context K / V [context][kv heads][head dim] FP16
    };
    GgufFile dmeta_;
    std::vector<DraftLayer> D_;
    std::vector<int> target_layers_;
    M dfc_, dw2_;
    const float * dhid_norm_ = nullptr, * dout_norm_ = nullptr, * dw1_ = nullptr;
    int dblock_ = 0, dmask_ = 0, drank_ = 0, dwindow_ = 0, dff_ = 0;
    Buf feat_, fused_, dh_, dxn_, dq_, dkv_, dk_, dv_, datt_, dfo_, dlog_, dbias_, dlat_, dtok_;
    bool has_draft() const { return !D_.empty(); }
    struct Gen {
        std::vector<uint32_t> out;
        uint32_t count = 0, tok = 0, pos = 0;
        int d = 0;
        bool done = true, adaptive = false;
        MinimaxAcceptRate acc;
        uint64_t steps = 0, drafted = 0, accepted = 0, prefill_us = 0;
        std::chrono::steady_clock::time_point t1;
    } gen_;
    Buf x_, xn_, xh_, qkv_, q_, k_, v_, att_, fo_, a_f_, b_f_, c_f_, act_, bfa_, scratch_h_, logits_, d_arg_, d_ids_, d_w_, ye_g_,
        ye_u_, yd_, sh_, partial_;
    float * h_embed_ = nullptr;
    // routed experts: the whole stack of every MoE layer on the device when it fits beside the rest, else one bank per
    // layer in the shared MixedExpertCache sized from the free device memory (hot experts stay, the others are read
    // from the GGUF on a miss). The bank path reads the routed ids on the host.
    bool integrated_ = false;
    size_t budget_ = 0;
    std::vector<size_t> slot_bytes_;
    std::vector<double> spread_;
    std::vector<std::vector<uint8_t>> seen_;
    std::unique_ptr<MixedExpertCache> bank_;
    Buf d_slots_, dmap_, moe_counts_, moe_groups_, mq_x_, mq_a_, mq_tiles_, wdq_;  // wdq_: one dense matrix dequantised to FP16; mq_: int8 (MMQ) expert GEMM inputs / tile lists
    static constexpr int kSmallRows = 512;
    static int splits(int cols) { return std::max(1, std::min(64, cols / 512)); }

    double num(const std::string & k) const { return meta_.num(a_ + k); }

    void configure() {
        const std::string arch = meta_.str("general.architecture");
        if (arch != "minimax-m3") throw std::runtime_error("minimax: wrong architecture " + arch);
        a_ = arch + ".";
        H_ = int(num("embedding_length")); hq_ = int(num("attention.head_count")); hk_ = int(num("attention.head_count_kv"));
        hd_ = int(num("attention.key_length")); rot_ = int(num("rope.dimension_count")); theta_ = float(num("rope.freq_base"));
        experts_ = int(num("expert_count")); topk_ = int(num("expert_used_count")); eff_ = int(num("expert_feed_forward_length"));
        ff_ = int(num("feed_forward_length")); sh_ff_ = int(num("expert_shared_feed_forward_length"));
        dense_lead_ = int(num("leading_dense_block_count")); w_scale_ = float(num("expert_weights_scale"));
        alpha_ = float(num("swiglu_alpha")); limit_ = float(num("swiglu_limit"));
        eps_ = float(num("attention.layer_norm_rms_epsilon")); vocab_ = int(num("vocab_size"));
        eos_ = uint32_t(meta_.num("tokenizer.ggml.eos_token_id", 0));
        ih_ = int(num("indexer.head_count")); idim_ = int(num("indexer.key_length")); itopk_ = int(num("indexer.top_k_blocks"));
        iblock_ = int(num("indexer.block_size")); ilocal_ = int(num("indexer.local_blocks"));
        if (context_ > uint32_t(num("context_length"))) throw std::runtime_error("minimax: context beyond the model's");
        L_.resize(size_t(num("block_count")));
        if (hq_ % W_ || hk_ % W_ || eff_ % W_ || ff_ % W_ || sh_ff_ % W_)
            throw std::runtime_error("minimax: heads / ff widths do not split over " + std::to_string(W_) + " ranks");
        hql_ = hq_ / W_; hkl_ = hk_ / W_; effl_ = eff_ / W_; ffl_ = ff_ / W_; shl_ = sh_ff_ / W_;
    }
    const GgufTensor & tensor(const std::string & n) const {
        const auto it = cur_->tensors.find(n);
        if (it == cur_->tensors.end()) throw std::runtime_error("minimax: missing tensor " + n);
        return it->second;
    }
    const MappedGguf & map_for(const GgufTensor & t) const { return *map_of_.at(t.file.empty() ? cur_->path : t.file); }
    // a built tensor (an EXL3 folder's X3 matrices gathered from the shards, its norms 1 + w) lives on the host until
    // its upload; every other tensor is a view of its mapped file
    GgufBytes bytes(const std::string & n) const {
        const auto & t = tensor(n);
        if (!t.built()) return map_for(t).view(t);
        auto it = built_.find(n);
        if (it == built_.end()) it = built_.emplace(n, cur_->read(t)).first;
        return {it->second.data(), it->second.size()};
    }
    void discard(const std::string & n, GgufBytes b) const {
        const auto & t = tensor(n);
        if (t.built()) built_.erase(n);
        else map_for(t).discard_pages(b);
    }
    void map_files(const GgufFile & g) {
        for (const auto & [name, t] : g.tensors) {
            if (t.built()) continue;  // read from its parts' files when built
            const std::string f = t.file.empty() ? g.path : t.file;
            if (map_of_.count(f)) continue;
            const bool st = f.size() > 12 && f.compare(f.size() - 12, 12, ".safetensors") == 0;
            maps_.push_back(st ? MappedGguf::raw(f) : std::make_unique<MappedGguf>(f));
            map_of_[f] = maps_.back().get();
        }
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
        dequantize_host_rows(t.type, bytes(n).data, v.data(), int64_t(e / t.ne[0]), int64_t(t.ne[0]));
        return v;
    }
    const float * vec(const std::string & n) {
        const auto v = host_f32(n);
        return static_cast<const float *>(put(v.data(), v.size() * 4));
    }
    // A tensor, or a row / column range of it (per expert of a stack): tensor parallelism loads each rank's share of the
    // same GGUF. Column ranges fall on the type's block boundaries, so every quantised row slices into whole blocks.
    struct Slice {
        std::string name;
        int row0 = 0, rows = -1, col0 = 0, cols = -1;
    };
    Slice whole(const std::string & n) const { return {n}; }
    Slice rows_of(const std::string & n, int part, int parts) const {
        const int r = int(tensor(n).ne[1]) / parts;
        return {n, part * r, r, 0, -1};
    }
    Slice cols_of(const std::string & n, int part, int parts) const {
        const int c = int(tensor(n).ne[0]) / parts;
        return {n, 0, -1, part * c, c};
    }
    int slice_rows(const Slice & sl) const { return sl.rows < 0 ? int(tensor(sl.name).ne[1]) : sl.rows; }
    int slice_cols(const Slice & sl) const { return sl.cols < 0 ? int(tensor(sl.name).ne[0]) : sl.cols; }
    // the slice's raw bytes, part (expert) after part
    std::vector<char> slice_bytes(const Slice & sl) const {
        const auto & t = tensor(sl.name);
        const auto b = bytes(sl.name);
        const int64_t cols = int64_t(t.ne[0]), rows = int64_t(t.ne[1]), parts = t.ne.size() > 2 ? int64_t(t.ne[2]) : 1;
        const size_t row_bytes = b.size / size_t(rows * parts);
        const int nr = slice_rows(sl), nc = slice_cols(sl);
        // the whole tensor (one box): its bytes as stored, whatever the layout (X3 records are not row-sliceable)
        if (sl.row0 == 0 && sl.col0 == 0 && nr == rows && nc == cols) return std::vector<char>(b.data, b.data + b.size);
        uint64_t el = 0, by = 0;
        if (!gguf_block_layout(t.type, el, by) || !el) throw std::runtime_error("minimax: " + sl.name + " cannot be sliced (tensor parallel)");
        if (sl.col0 % int(el) || nc % int(el)) throw std::runtime_error("minimax: column slice of " + sl.name + " splits a block");
        const size_t c0 = size_t(sl.col0) / el * by, cb = size_t(nc) / el * by;
        if (sl.row0 == 0 && nr == rows && cb == row_bytes) return std::vector<char>(b.data, b.data + b.size);
        std::vector<char> out(size_t(parts) * nr * cb);
        char * o = out.data();
        for (int64_t p = 0; p < parts; ++p)
            for (int r = 0; r < nr; ++r, o += cb)
                std::memcpy(o, b.data + (size_t(p * rows + sl.row0 + r) * row_bytes + c0), cb);
        return out;
    }
    std::vector<float> slice_f32(const Slice & sl) const {
        const auto & t = tensor(sl.name);
        const auto v = host_f32(sl.name);
        const int64_t cols = int64_t(t.ne[0]);
        const int nr = slice_rows(sl), nc = slice_cols(sl);
        std::vector<float> out(size_t(nr) * nc);
        for (int r = 0; r < nr; ++r)
            std::memcpy(out.data() + size_t(r) * nc, v.data() + size_t(sl.row0 + r) * cols + sl.col0, size_t(nc) * 4);
        return out;
    }
    M load_mat(const std::vector<Slice> & parts, bool format_rows = false) {
        const auto & t0 = tensor(parts[0].name);
        M r;
        r.cols = slice_cols(parts[0]);
        for (const auto & sl : parts) {
            const auto & t = tensor(sl.name);
            if (t.ne.size() != 2 || slice_cols(sl) != r.cols || t.type != t0.type) throw std::runtime_error("minimax: cannot stack " + sl.name);
            r.rows += slice_rows(sl);
        }
        const WeightFormat * wf = r.rows <= kSmallRows && !format_rows ? nullptr : weight_format(t0.type);
        if (wf) {
            std::vector<char> raw;
            for (const auto & sl : parts) { const auto b = slice_bytes(sl); raw.insert(raw.end(), b.begin(), b.end()); }
            r.m = wf->load(raw, r.rows, r.cols, [this](const void * h, size_t s) { return put(h, s); }, stream_);
            for (const auto & sl : parts) discard(sl.name, bytes(sl.name));
            if (!r.m.fmt || r.m.fmt->basis() != Basis::Plain) throw std::runtime_error("minimax: " + parts[0].name + " format");
            chunk_ = std::min(chunk_, r.m.fmt->max_matvec_cols());
        } else {
            std::vector<float> v;
            for (const auto & sl : parts) { const auto x = slice_f32(sl); v.insert(v.end(), x.begin(), x.end()); }
            r.f32 = static_cast<const float *>(put(v.data(), v.size() * 4));
            if (r.rows <= kSmallRows) max_split_ = std::max(max_split_, size_t(r.rows) * splits(r.cols));
        }
        max_rows_ = std::max(max_rows_, r.rows);
        return r;
    }
    M load_mat(const std::vector<std::string> & names, bool format_rows = false) {
        std::vector<Slice> parts;
        for (const auto & n : names) parts.push_back(whole(n));
        return load_mat(parts, format_rows);
    }
    M mat(const std::string & n) { return load_mat(std::vector<std::string>{n}); }
    Mat experts(const Slice & sl) {
        const std::string & n = sl.name;
        const auto & t = tensor(n);
        if (t.ne.size() != 3 || int(t.ne[2]) != experts_) throw std::runtime_error("minimax: " + n + " is not an expert stack");
        const WeightFormat * wf = weight_format(t.type);
        if (!wf) throw std::runtime_error("minimax: no weight format for expert type " + std::to_string(t.type));
        const auto b = bytes(n);
        const auto raw = slice_bytes(sl);
        Mat m = wf->load(raw, slice_rows(sl) * experts_, slice_cols(sl), [this](const void * h, size_t s) { return put(h, s); }, stream_);
        // the device copy is what the kernels read; the file's pages would hold the same bytes twice (a GB10 shares one memory)
        discard(n, b);
        if (!m.fmt || !m.fmt->has_expert_matvec() || m.fmt->basis() != Basis::Plain)
            throw std::runtime_error("minimax: " + n + " has no grouped-expert kernel");
        chunk_ = std::min(chunk_, m.fmt->max_matvec_cols());
        return m;
    }

    void load() {
        for (size_t l = 0; l < L_.size(); ++l) {
            auto & Y = L_[l];
            const std::string p = "blk." + std::to_string(l) + ".";
            Y.attn_norm = vec(p + "attn_norm.weight"); Y.ffn_norm = vec(p + "ffn_norm.weight");
            Y.q_norm = vec(p + "attn_q_norm.weight"); Y.k_norm = vec(p + "attn_k_norm.weight");
            // tensor parallel: this rank's query heads and KV heads (rows), and their columns of the output projection
            Y.wqkv = load_mat(std::vector<Slice>{rows_of(p + "attn_q.weight", R_, W_), rows_of(p + "attn_k.weight", R_, W_),
                                                 rows_of(p + "attn_v.weight", R_, W_)});
            Y.wo = load_mat(std::vector<Slice>{cols_of(p + "attn_output.weight", R_, W_)});
            if (meta_.tensors.count(p + "indexer.attn_q.weight")) {
                Y.iq = load_mat({p + "indexer.attn_q.weight"}, true); Y.ik = mat(p + "indexer.attn_k.weight");
                Y.iq_norm = vec(p + "indexer.q_norm.weight"); Y.ik_norm = vec(p + "indexer.k_norm.weight");
            }
            if (int(l) < dense_lead_) {
                Y.gate_up = load_mat(std::vector<Slice>{rows_of(p + "ffn_gate.weight", R_, W_), rows_of(p + "ffn_up.weight", R_, W_)});
                Y.down = load_mat(std::vector<Slice>{cols_of(p + "ffn_down.weight", R_, W_)});
            } else {
                Y.router = mat(p + "ffn_gate_inp.weight"); Y.probs_bias = vec(p + "exp_probs_b.bias");
                Y.sh_gate_up = load_mat(std::vector<Slice>{rows_of(p + "ffn_gate_shexp.weight", R_, W_),
                                                           rows_of(p + "ffn_up_shexp.weight", R_, W_)});
                Y.sh_down = load_mat(std::vector<Slice>{cols_of(p + "ffn_down_shexp.weight", R_, W_)});
            }
            std::fprintf(stderr, "\rminimax: loaded layer %zu / %zu", l + 1, L_.size());
        }
        out_norm_ = vec("output_norm.weight");
        if (R_ == 0) out_ = mat("output.weight");  // the head (and the drafter) live on rank 0
        std::fprintf(stderr, "\n");
        load_experts();
    }
    void load_draft(const std::string & path) {
        dmeta_ = gguf_open(path);
        if (dmeta_.str("general.architecture") != "minimax-m3") throw std::runtime_error("minimax: " + path + " is not a MiniMax draft");
        map_files(dmeta_);
        cur_ = &dmeta_;
        const std::string a = "minimax-m3.";
        dblock_ = int(dmeta_.num(a + "dspark.block_size")); dmask_ = int(dmeta_.num(a + "dspark.mask_token_id"));
        drank_ = int(dmeta_.num(a + "dspark.markov_rank")); dwindow_ = int(dmeta_.num(a + "dspark.sliding_window"));
        dff_ = int(dmeta_.num(a + "dspark.feed_forward_length"));
        const std::string ids = dmeta_.str(a + "dspark.target_layers");
        for (size_t at = 0; at < ids.size();) {
            const size_t c = ids.find(',', at);
            target_layers_.push_back(std::stoi(ids.substr(at, c == std::string::npos ? std::string::npos : c - at)));
            at = c == std::string::npos ? ids.size() : c + 1;
        }
        for (int l : target_layers_) if (l < 0 || l >= int(L_.size())) throw std::runtime_error("minimax: draft target layer beyond the model");
        dfc_ = mat("dspark.fc.weight");
        if (dfc_.cols != int(target_layers_.size()) * H_) throw std::runtime_error("minimax: draft fc width");
        dhid_norm_ = vec("dspark.hidden_norm.weight"); dout_norm_ = vec("dspark.output_norm.weight");
        D_.resize(size_t(dmeta_.num(a + "block_count")));
        for (size_t l = 0; l < D_.size(); ++l) {
            auto & Y = D_[l];
            const std::string p = "dspark.blk." + std::to_string(l) + ".";
            Y.attn_norm = vec(p + "attn_norm.weight"); Y.ffn_norm = vec(p + "ffn_norm.weight");
            Y.q_norm = vec(p + "attn_q_norm.weight"); Y.k_norm = vec(p + "attn_k_norm.weight");
            Y.wq = mat(p + "attn_q.weight"); Y.wkv = load_mat({p + "attn_k.weight", p + "attn_v.weight"});
            Y.wo = mat(p + "attn_output.weight");
            Y.gate_up = load_mat({p + "ffn_gate.weight", p + "ffn_up.weight"}); Y.down = mat(p + "ffn_down.weight");
        }
        dw1_ = vec("dspark.markov_w1.weight");
        dw2_ = mat("dspark.markov_w2.weight");
        cur_ = &meta_;
        std::fprintf(stderr, "minimax: DSpark draft %s: %zu layers, block %d, Markov rank %d, window %d\n", path.c_str(), D_.size(),
                     dblock_, drank_, dwindow_);
    }
    static size_t mem_available() {
        std::FILE * f = std::fopen("/proc/meminfo", "r");
        if (!f) return 0;
        char key[64]; unsigned long long kb = 0; size_t out = 0;
        while (std::fscanf(f, "%63s %llu kB", key, &kb) == 2)
            if (std::string(key) == "MemAvailable:") { out = size_t(kb) << 10; break; }
        std::fclose(f);
        return out;
    }
    std::array<ExpertPayload, 3> expert_payload(int layer, int expert) const {
        return glm_expert_payload("blk." + std::to_string(layer) + ".",
                                  [this](const std::string & n) -> const MappedGguf & {
                                      const auto & t = tensor(n);
                                      return *map_of_.at(t.file.empty() ? meta_.path : t.file);
                                  },
                                  expert);
    }
    // the device decides: every expert stack when it fits beside the activations and the KV cache, else the banks
    void load_experts() {
        size_t stack = 0;
        for (size_t l = size_t(dense_lead_); l < L_.size(); ++l)
            for (const char * pj : kGlmExpertProjection) stack += size_t(tensor("blk." + std::to_string(l) + "." + pj).bytes) / size_t(W_);
        size_t free = 0, total = 0;
        ck(cudaMemGetInfo(&free, &total));
        if (integrated_) free = std::max(free, mem_available());  // a GB10's free memory leaves out the reclaimable page cache
        // a GB10 shares its memory with the host: keep kMiniMaxHostReserve for it; a discrete card keeps 2 GB
        // after the dense weights are in, an FP16 KV cache the early guard allowed may leave too little for the experts:
        // FP8 then, as the early guard does for contexts that do not fit (GB10 32K: 99 GiB of experts, 99 free)
        const size_t fixed = (integrated_ ? kMiniMaxHostReserve : size_t(2) << 30) + (size_t(2) << 30);
        const size_t kv16 = L_.size() * minimax_kv_layer_bytes(context_, hkl_, hd_, idim_, 2), kv8 = L_.size() * minimax_kv_layer_bytes(context_, hkl_, hd_, idim_, 1);
        const size_t elem = minimax_expert_kv_elem(stack, free, fixed, kv16, kv8, kv8_);
        if (elem == 1 && !kv8_) {
            kv8_ = true;
            std::fprintf(stderr, "minimax: KV cache FP8 E4M3 (the experts need the room FP16 would take at context %u)\n", context_);
        }
        const size_t kv = kv8_ ? kv8 : kv16;
        const size_t avail = free > fixed + kv ? free - fixed - kv : 0;
        if (stack <= avail) {
            for (size_t l = size_t(dense_lead_); l < L_.size(); ++l) {
                auto & Y = L_[l];
                const std::string p = "blk." + std::to_string(l) + ".";
                // every expert's ff split between the ranks: whatever the routing, both read half of each routed expert
                Y.e_gate = experts(rows_of(p + "ffn_gate_exps.weight", R_, W_)); Y.e_up = experts(rows_of(p + "ffn_up_exps.weight", R_, W_));
                Y.e_down = experts(cols_of(p + "ffn_down_exps.weight", R_, W_));
            }
            std::fprintf(stderr, "minimax: all experts on the device (%.1f GB)\n", stack / 1e9);
            return;
        }
        if (W_ > 1) throw std::runtime_error("minimax: tensor parallel needs every expert share resident (no banks)");
        if (tensor("blk." + std::to_string(dense_lead_) + ".ffn_gate_exps.weight").built())
            throw std::runtime_error("minimax: experts " + std::to_string(stack >> 30) + " GiB > " + std::to_string(avail >> 30) +
                                     " GiB free: hot / cold banks read GGUF expert rows, a checkpoint folder's experts must fit");
        budget_ = avail;
        slot_bytes_.assign(L_.size(), 0);
        spread_.assign(L_.size(), 0.0);
        seen_.assign(L_.size(), std::vector<uint8_t>(size_t(experts_), 0));
        for (size_t l = size_t(dense_lead_); l < L_.size(); ++l) {
            std::array<ExpertFormat, 3> fmt{};
            const auto parts = repack_expert(expert_payload(int(l), 0), H_, eff_, false, &fmt);
            for (int pr = 0; pr < 3; ++pr) {
                if (!fmt[size_t(pr)].bank_row_layout) throw std::runtime_error("minimax: expert format has no bank row layout");
                chunk_ = std::min(chunk_, fmt[size_t(pr)].max_matvec_cols);
            }
            for (const auto & part : parts) slot_bytes_[l] += part.size();
        }
        const auto caps = plan_caps();
        int cap = 1;
        for (int c : caps) cap = std::max(cap, c);
        bank_ = std::make_unique<MixedExpertCache>(int(L_.size()), experts_, H_, eff_, cap, budget_,
                                                   [this](int l, int e) { return expert_payload(l, e); }, stream_);
        bank_->set_layer_slots(caps);
        std::fprintf(stderr, "minimax: experts %.1f GB > %.1f GB free beside the rest: banks of %.1f GB\n", stack / 1e9,
                     avail / 1e9, budget_ / 1e9);
        preload_hot(caps);
    }
    // Hot / cold: with a routing profile (TQLLM_MINIMAX_ROUTES output: int64 [layers][experts]) each layer's bank starts
    // with its most-routed experts; every other expert stays in the GGUF on disk until a token routes to it.
    std::string hot_map_;
    void preload_hot(const std::vector<int> & caps) {
        if (hot_map_.empty()) return;
        std::FILE * f = std::fopen(hot_map_.c_str(), "rb");
        if (!f) throw std::runtime_error("minimax: cannot read hot map " + hot_map_);
        std::vector<int64_t> c(L_.size() * size_t(experts_));
        const size_t got = std::fread(c.data(), 8, c.size(), f);
        std::fclose(f);
        if (got != c.size()) throw std::runtime_error("minimax: hot map is not [layers][experts] int64");
        const auto t0 = std::chrono::steady_clock::now();
        size_t hot = 0;
        for (size_t l = size_t(dense_lead_); l < L_.size(); ++l) {
            std::vector<int> ids(static_cast<size_t>(experts_));
            for (int e = 0; e < experts_; ++e) ids[size_t(e)] = e;
            std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return c[l * experts_ + size_t(a)] > c[l * experts_ + size_t(b)]; });
            ids.resize(size_t(std::min(caps[l], experts_)));
            std::sort(ids.begin(), ids.end());
            bank_->prepare(int(l), ids);
            bank_->wait_misses();
            hot += ids.size();
        }
        ck(cudaStreamSynchronize(stream_));
        std::fprintf(stderr, "minimax: hot set %zu of %zu experts preloaded in %.0f s (cold experts stay on disk)\n", hot,
                     (L_.size() - size_t(dense_lead_)) * size_t(experts_),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        const auto b = bank_->stats();  // the report counts from here: decode, not the preload
        base_hits_ = b.hits; base_misses_ = b.misses; base_moved_ = b.transfer_bytes; base_read_ = disk_read_bytes();
    }
    uint64_t base_hits_ = 0, base_misses_ = 0, base_read_ = 0;
    double base_moved_ = 0;
    static uint64_t disk_read_bytes() {
        std::FILE * f = std::fopen("/proc/self/io", "r");
        if (!f) return 0;
        char key[64]; unsigned long long v = 0, out = 0;
        while (std::fscanf(f, "%63s %llu", key, &v) == 2) if (std::string(key) == "read_bytes:") out = v;
        std::fclose(f);
        return out;
    }
    std::vector<int> plan_caps() {
        std::vector<int> caps(L_.size(), 1);
        const auto p = expert_layer_slots(budget_, slot_bytes_, spread_, topk_, experts_);
        if (!p.fits) throw std::runtime_error("minimax: the free device memory does not hold one token's experts per layer");
        for (size_t l = 0; l < caps.size(); ++l) if (slot_bytes_[l]) caps[l] = std::max(p.slots[l], topk_);
        return caps;
    }
    void report_bank() {
        if (!bank_) return;
        const auto s = bank_->stats();
        std::fprintf(stderr, "\nminimax: expert banks: hits %llu misses %llu (%.1f %% hit), %.2f GB moved, %.2f GB on the device, "
                     "process disk reads %.2f GB\n",
                     (unsigned long long)(s.hits - base_hits_), (unsigned long long)(s.misses - base_misses_),
                     100.0 * double(s.hits - base_hits_) / double(std::max<uint64_t>(1, s.hits - base_hits_ + s.misses - base_misses_)),
                     (double(s.transfer_bytes) - base_moved_) / 1e9, s.device_bytes / 1e9, (disk_read_bytes() - base_read_) / 1e9);
    }

    // device + pinned bytes of the per-token buffers at P tokens per forward (head-sized buffers stay at the verify chunk)
    size_t act_bytes(int P, bool draft) const {
        const size_t qd = size_t(hq_) * hd_, kd = size_t(hk_) * hd_, wf = size_t(std::max({ff_, sh_ff_, eff_}));
        const size_t nblocks = (size_t(context_) + size_t(std::max(1, iblock_)) - 1) / size_t(std::max(1, iblock_));
        size_t f = 4 * size_t(H_) + qd + 2 * kd + 2 * qd + 2 * kd + 2 * wf + 2 * wf + size_t(topk_) * 2 + size_t(ih_) * idim_ +
                   idim_ + nblocks + itopk_ + size_t(experts_) + size_t(H_);  // FP32 rows (the last: pinned embeddings)
        if (draft) f += 8 * size_t(H_) + H_ + 6 * kd;
        size_t h = H_ + std::max(size_t(topk_) * eff_, wf) + std::max({size_t(H_), qd, wf, size_t(8) * H_}) +
                   std::max({qd + 2 * kd, size_t(H_), 2 * wf, size_t(ih_) * idim_, size_t(topk_) * std::max(eff_, H_)}) +
                   2 * size_t(topk_) * eff_ + size_t(topk_) * H_;  // half / bf16 rows
        return size_t(P) * (f * 4 + h * 2 + size_t(topk_) * 4 + size_t(experts_) * 4 + (size_t(H_) + size_t(topk_) * eff_) * 2 + size_t(experts_)) +
               std::max({qd + 2 * kd, 2 * wf, size_t(H_)}) * std::max({size_t(H_), wf, qd}) * 2;  // wdq_
    }
    void alloc() {
        const int N = chunk_, P = std::max(chunk_, pchunk_), qd = hq_ * hd_, kd = hk_ * hd_;
        auto f = [](size_t n) { return Buf(n * sizeof(float)); };
        x_ = f(size_t(P) * H_); xn_ = f(size_t(P) * H_); xh_ = Buf(size_t(P) * H_ * sizeof(__half));
        qkv_ = f(size_t(P) * (qd + 2 * kd)); q_ = f(size_t(P) * qd); k_ = f(size_t(P) * kd); v_ = f(size_t(P) * kd);
        att_ = f(size_t(P) * qd); fo_ = f(size_t(P) * H_);
        const int widest_ff = std::max({ff_, sh_ff_, eff_});
        a_f_ = f(size_t(P) * 2 * widest_ff); b_f_ = f(size_t(P) * widest_ff); c_f_ = f(size_t(P) * widest_ff);
        act_ = Buf(size_t(P) * std::max(topk_ * eff_, widest_ff) * sizeof(__half));
        const int widest = std::max({max_rows_, topk_ * std::max(eff_, H_), H_, widest_ff});
        // the head's vocab rows run on at most N tokens; every other projection on up to P
        const int widest_p = std::max({qd + 2 * kd, H_, 2 * widest_ff, ih_ * idim_, idim_, topk_ * std::max(eff_, H_), experts_});
        bfa_ = Buf(std::max(size_t(N) * widest, size_t(P) * widest_p) * sizeof(bf16));
        scratch_h_ = Buf(size_t(P) * std::max({H_, qd, widest_ff, dfc_.cols}) * sizeof(__half));
        logits_ = f(std::max(size_t(N) * std::max(vocab_, experts_), size_t(P) * experts_)); d_arg_ = Buf(size_t(N) * sizeof(int));
        d_ids_ = Buf(size_t(P) * topk_ * sizeof(int)); d_w_ = f(size_t(P) * topk_);
        ye_g_ = Buf(size_t(P) * topk_ * eff_ * sizeof(bf16)); ye_u_ = Buf(size_t(P) * topk_ * eff_ * sizeof(bf16));
        yd_ = Buf(size_t(P) * topk_ * H_ * sizeof(bf16)); sh_ = f(size_t(P) * H_);
        d_slots_ = Buf(size_t(N) * topk_ * sizeof(int));
        moe_counts_ = Buf(size_t(std::max(1, experts_)) * sizeof(int)); moe_groups_ = Buf(size_t(std::max(1, experts_)) * P * sizeof(int));
        if (P > N) {
            mq_x_ = Buf(tqllm_cuda::mmq_act_bytes(P, H_)); mq_a_ = Buf(tqllm_cuda::mmq_act_bytes(P * topk_, eff_));
            mq_tiles_ = Buf(size_t(tqllm_cuda::mmq_max_tiles(P, experts_)) * sizeof(int2));
        }
        if (P > N) {
            size_t dq = 0;
            auto upd = [&](const M & w) { if (!w.f32 && w.m.fmt) dq = std::max(dq, size_t(w.rows) * w.cols); };
            for (const auto & Y : L_) for (const M * w : {&Y.wqkv, &Y.wo, &Y.iq, &Y.ik, &Y.gate_up, &Y.down, &Y.sh_gate_up, &Y.sh_down}) upd(*w);
            upd(dfc_);
            wdq_ = Buf(dq * sizeof(__half));
        }
        if (P > N) tqllm_cuda::mmq_warm();  // the int8 expert GEMMs' shared-memory limits
        if (P > N) { if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("minimax: cuBLAS init failed");
                     cublasSetStream(cublas_, stream_); }
        hdr_ = Buf(size_t(2 + N) * sizeof(int32_t));
        partial_ = f(std::max<size_t>(1, size_t(N) * max_split_));
        ck(cudaHostAlloc(&h_embed_, size_t(P) * H_ * sizeof(float), cudaHostAllocDefault));
        const size_t h2 = kv8_ ? 1 : sizeof(__half);  // KV cache element bytes
        for (auto & Y : L_) {
            Y.kc = Buf(size_t(context_) * hkl_ * hd_ * h2); Y.vc = Buf(size_t(context_) * hkl_ * hd_ * h2);
            if (Y.iq.rows) Y.ikc = Buf(size_t(context_) * idim_ * h2);
        }
        const int nblocks = int((context_ + uint32_t(iblock_) - 1) / uint32_t(iblock_));
        iqv_ = f(size_t(P) * ih_ * idim_); ikv_ = f(size_t(P) * idim_); iscore_ = f(size_t(P) * nblocks);
        iids_ = Buf(size_t(P) * itopk_ * sizeof(int));
        if (has_draft()) {
            const int B = dblock_;
            if (B > N) throw std::runtime_error("minimax: draft block wider than the chunk");
            feat_ = f(size_t(P) * target_layers_.size() * H_); fused_ = f(size_t(P) * H_);
            dh_ = f(size_t(B) * H_); dxn_ = f(size_t(B) * H_); dq_ = f(size_t(B) * qd); dkv_ = f(size_t(P) * 2 * kd);
            dk_ = f(size_t(P) * kd); dv_ = f(size_t(P) * kd); datt_ = f(size_t(B) * qd); dfo_ = f(size_t(B) * H_);
            dlog_ = f(size_t(B) * vocab_); dbias_ = f(size_t(vocab_)); dlat_ = f(size_t(drank_)); dtok_ = Buf(size_t(B + 1) * sizeof(int));
            for (auto & Y : D_) { Y.kc = Buf(size_t(context_) * kd * h2); Y.vc = Buf(size_t(context_) * kd * h2); }
        }
    }

    void mv(const M & w, const float * x, float * y, int n) {
        if (n > chunk_) {  // a prompt chunk: one pass over the weights for all n tokens
            if (w.f32) {
                const float one = 1.f, zero = 0.f;  // y[n][rows] = x[n][cols] W^T (column-major: W^T x)
                if (cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, n, w.cols, &one, w.f32, w.cols, x, w.cols, &zero, y,
                                w.rows) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("minimax: cuBLAS SGEMM failed");
                return;
            }
            tqllm_cuda::glm_f32_to_half(x, scratch_h_.as<__half>(), w.cols * n, stream_);
            gemm(w.m, scratch_h_.as<__half>(), y, n);
            return;
        }
        if (w.f32) {
            if (w.rows <= kSmallRows)
                tqllm_cuda::glm_mv_f32_splitk(w.f32, x, y, partial_.as<float>(), w.rows, w.cols, n, splits(w.cols), stream_);
            else tqllm_cuda::glm_mv_f32(w.f32, x, y, w.rows, w.cols, n, stream_);
            return;
        }
        tqllm_cuda::glm_f32_to_half(x, scratch_h_.as<__half>(), w.cols * n, stream_);
        if (w.f32_out != 0) {
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
    // y FP32 [n][rows] = W x (half), every column in the format's GEMM; no per-token fallback (it would read W n times).
    // FP32 output first: a BF16 output rounded every projection, and over 60 layers the residual drifted from the
    // per-token path (rel 0.7 at the last layer on a 12-token prompt).
    void gemm(const Mat & m, const __half * xh, float * y, int n) {
        const bool fits = size_t(m.rows) * m.cols * sizeof(__half) <= wdq_.bytes;
        auto dequant_gemm = [&] {  // dequantise once, FP16 tensor-core GEMM
            m.fmt->dequant_h(m, wdq_.as<__half>(), stream_);
            const float one = 1.f, zero = 0.f;
            if (cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, m.rows, n, m.cols, &one, wdq_.p, CUDA_R_16F, m.cols, xh, CUDA_R_16F, m.cols,
                             &zero, y, CUDA_R_32F, m.rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("minimax: cuBLAS FP16 GEMM failed");
        };
        // past the format's own fused-vs-cuBLAS crossover (GB10 gemm-bench)
        if (n > m.fmt->fused_gemm_cols() && m.fmt->fused_gemm_cols() > 0 && fits) { dequant_gemm(); return; }
        const int fid = m.fmt->gemm_f32_id();
        if (fid >= 0 && m.cols % 64 == 0 &&
            tqllm_cuda::u4h_family_gemm_f32(fid, static_cast<const uint8_t *>(m.p), m.aux, xh, y, m.rows, m.cols, n, stream_, m.rows))
            return;
        if (m.fmt->gemm(m, xh, bfa_.as<bf16>(), n, stream_)) { tqllm_cuda::glm_bf16_to_f32(bfa_.as<bf16>(), y, m.rows * n, stream_); return; }
        // a format without a GEMM kernel (X3 trellis): dequantise + cuBLAS at every width
        if (!fits) throw std::runtime_error(std::string("minimax: no prefill GEMM for ") + m.fmt->name());
        dequant_gemm();
    }
    // down(swigluoai(gate x, up x)) -> out FP32; gu = gate | up rows stacked
    void ffn(const M & gu, const M & dn, const float * x, float * out, int n) {
        const int ff = gu.rows / 2;
        mv(gu, x, a_f_.as<float>(), n);  // [n][gate | up]
        tqllm_cuda::glm_gather_cols(a_f_.as<float>(), 2 * ff, 0, ff, n, b_f_.as<float>(), stream_);
        tqllm_cuda::glm_gather_cols(a_f_.as<float>(), 2 * ff, ff, ff, n, c_f_.as<float>(), stream_);
        tqllm_cuda::mm_swigluoai_f32(b_f_.as<float>(), c_f_.as<float>(), act_.as<__half>(), ff * n, alpha_, limit_, stream_);
        if (n > chunk_) { gemm(dn.m, act_.as<__half>(), out, n); return; }
        if (dn.f32_out != 0) {
            try { dn.m.fmt->matvec_f32(dn.m, act_.as<__half>(), out, n, stream_); dn.f32_out = 1; return; }
            catch (const std::runtime_error &) { if (dn.f32_out == 1) throw; dn.f32_out = 0; }
        }
        dn.m.fmt->matvec(dn.m, act_.as<__half>(), bfa_.as<bf16>(), n, stream_);
        tqllm_cuda::glm_bf16_to_f32(bfa_.as<bf16>(), out, dn.rows * n, stream_);
    }
    void moe(Layer & Y, int layer, int n) {
        const float * xn = xn_.as<float>();
        __half * xh = xh_.as<__half>();
        tqllm_cuda::glm_f32_to_half(xn, xh, H_ * n, stream_);
        mv(Y.router, xn, logits_.as<float>(), n);
        tqllm_cuda::glm_route(logits_.as<float>(), Y.probs_bias, d_ids_.as<int>(), d_w_.as<float>(), experts_, topk_, true,
                              w_scale_, stream_, n);
        count_routes(layer, n);
        if (bank_) { moe_bank(Y, layer, n); return; }
        if (n > chunk_) { moe_grouped(Y, n); return; }
        if (n > 1 && Y.e_gate.fmt->has_expert_dedup() && Y.e_gate.fmt == Y.e_up.fmt && Y.e_down.fmt->has_expert_dedup()) {
            moe_dedup(Y, n);
            return;
        }
        bool done = false;
        if (Y.e_gate.fmt == Y.e_up.fmt && Y.paired != 0) {
            try {
                Y.e_gate.fmt->expert_matvec2(Y.e_gate, Y.e_up, xh, ye_g_.as<bf16>(), ye_u_.as<bf16>(), effl_, d_ids_.as<int>(), n,
                                             topk_, stream_);
                Y.paired = 1;
                done = true;
            } catch (const std::runtime_error &) {
                if (Y.paired == 1) throw;
                Y.paired = 0;
            }
        }
        if (!done) {
            Y.e_gate.fmt->expert_matvec(Y.e_gate, xh, ye_g_.as<bf16>(), effl_, d_ids_.as<int>(), n, topk_, false, stream_);
            Y.e_up.fmt->expert_matvec(Y.e_up, xh, ye_u_.as<bf16>(), effl_, d_ids_.as<int>(), n, topk_, false, stream_);
        }
        tqllm_cuda::mm_swigluoai_bf16(ye_g_.as<bf16>(), ye_u_.as<bf16>(), act_.as<__half>(), n * topk_ * effl_, alpha_, limit_,
                                      stream_);
        Y.e_down.fmt->expert_matvec(Y.e_down, act_.as<__half>(), yd_.as<bf16>(), H_, d_ids_.as<int>(), n, topk_, true, stream_);
        ffn(Y.sh_gate_up, Y.sh_down, xn, sh_.as<float>(), n);
        tqllm_cuda::glm_moe_sum(yd_.as<bf16>(), d_w_.as<float>(), sh_.as<float>(), fo_.as<float>(), topk_, H_, stream_, n);
        reduce(fo_.as<float>(), size_t(n) * H_);  // routed + shared partial sums of this rank's ff share
    }
    // A prompt chunk: tokens grouped by routed expert, each expert's rows read once for all its tokens (grouped GEMM);
    // the same per-assignment layout [n * topk][rows] as expert_matvec. swigluoai (alpha, clamp) runs between, so the
    // fused gate-up epilogue (plain SiLU) does not apply.
    void moe_grouped(Layer & Y, int n) {
        const int A = n * topk_;
        const int * counts = moe_counts_.as<int>(), * groups = moe_groups_.as<int>();
        tqllm_cuda::moe_group(d_ids_.as<int>(), moe_counts_.as<int>(), moe_groups_.as<int>(), experts_, n, topk_, stream_);
        const __half * xh = xh_.as<__half>();
        // int8 tensor-core GEMM (llama.cpp's MMQ on our banks) when all three projections have it: gate / up read the
        // tokens quantised once (group entry = assignment, row = entry / topk), down the assignments' activations
        const int ig = Y.e_gate.fmt->gemm_f32_id(), iu = Y.e_up.fmt->gemm_f32_id(), id = Y.e_down.fmt->gemm_f32_id();
        if (tqllm_cuda::mmq_supports(ig, H_) && tqllm_cuda::mmq_supports(iu, H_) && tqllm_cuda::mmq_supports(id, effl_)) {
            std::vector<int> ch(static_cast<size_t>(experts_));
            ck(cudaMemcpyAsync(ch.data(), moe_counts_.p, ch.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_));
            ck(cudaStreamSynchronize(stream_));
            const int tcap = tqllm_cuda::mmq_max_tiles(n, experts_);
            auto mmq = [&](const Mat & m, int fid, int rows, const void * yq, int ny, int div, bf16 * y) {
                if (!tqllm_cuda::mmq_expert_gemm(fid, static_cast<const uint8_t *>(m.p), m.aux, false, rows, m.cols, counts, ch.data(), groups,
                                                 n, experts_, div, yq, ny, y, mq_tiles_.as<int2>(), tcap, stream_, nullptr, nullptr))
                    throw std::runtime_error(std::string("minimax: MMQ expert GEMM refused ") + m.fmt->name());
            };
            // IQ1_S reads each 32-value block's sum (DS4 layout); gate and up share one format per layer
            if (ig != iu) throw std::runtime_error("minimax: gate and up in different formats");
            tqllm_cuda::mmq_quantize_act(xh, n, H_, mq_x_.p, stream_, ig == 11);
            mmq(Y.e_gate, ig, effl_, mq_x_.p, n, topk_, ye_g_.as<bf16>());
            mmq(Y.e_up, iu, effl_, mq_x_.p, n, topk_, ye_u_.as<bf16>());
            tqllm_cuda::mm_swigluoai_bf16(ye_g_.as<bf16>(), ye_u_.as<bf16>(), act_.as<__half>(), A * effl_, alpha_, limit_, stream_);
            tqllm_cuda::mmq_quantize_act(act_.as<__half>(), A, effl_, mq_a_.p, stream_, id == 11);
            mmq(Y.e_down, id, H_, mq_a_.p, A, 1, yd_.as<bf16>());
            ffn(Y.sh_gate_up, Y.sh_down, xn_.as<float>(), sh_.as<float>(), n);
            tqllm_cuda::glm_moe_sum(yd_.as<bf16>(), d_w_.as<float>(), sh_.as<float>(), fo_.as<float>(), topk_, H_, stream_, n);
            reduce(fo_.as<float>(), size_t(n) * H_);
            return;
        }
        auto need = [](bool ok, const Mat & m) { if (!ok) throw std::runtime_error(std::string("minimax: no grouped expert GEMM for ") + m.fmt->name()); };
        need(Y.e_gate.fmt->expert_gemm(Y.e_gate, xh, ye_g_.as<bf16>(), effl_, counts, groups, n, experts_, topk_, false, stream_), Y.e_gate);
        need(Y.e_up.fmt->expert_gemm(Y.e_up, xh, ye_u_.as<bf16>(), effl_, counts, groups, n, experts_, topk_, false, stream_), Y.e_up);
        tqllm_cuda::mm_swigluoai_bf16(ye_g_.as<bf16>(), ye_u_.as<bf16>(), act_.as<__half>(), A * effl_, alpha_, limit_, stream_);
        need(Y.e_down.fmt->expert_gemm(Y.e_down, act_.as<__half>(), yd_.as<bf16>(), H_, counts, groups, n, experts_, topk_, true, stream_),
             Y.e_down);
        ffn(Y.sh_gate_up, Y.sh_down, xn_.as<float>(), sh_.as<float>(), n);
        tqllm_cuda::glm_moe_sum(yd_.as<bf16>(), d_w_.as<float>(), sh_.as<float>(), fo_.as<float>(), topk_, H_, stream_, n);
        reduce(fo_.as<float>(), size_t(n) * H_);
    }
    // A verify block (n > 1): each distinct routed expert is read once for all its tokens (up to kDedupCols per group)
    // instead of once per assignment; same per-assignment outputs as expert_matvec.
    void moe_dedup(Layer & Y, int n) {
        const int A = n * topk_, DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
        std::vector<int> ids(static_cast<size_t>(A));
        ck(cudaMemcpyAsync(ids.data(), d_ids_.p, ids.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        ck(cudaStreamSynchronize(stream_));
        std::vector<int> map(size_t(2) * A * DS, -1);
        auto build = [&](int * out, bool per_expert) {
            std::vector<int> open(size_t(experts_), -1);
            int g = 0;
            for (int a = 0; a < A; ++a) {
                int & at = open[size_t(ids[size_t(a)])];
                if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = ids[size_t(a)]; out[at * DS + 1] = 0; }
                int * e = out + at * DS;
                const int col = e[1]++;
                e[2 + col] = per_expert ? a : a / topk_;
                e[2 + DC + col] = a;
            }
            return g;
        };
        const int g_in = build(map.data(), false), g_dn = build(map.data() + size_t(A) * DS, true);
        if (dmap_.bytes < map.size() * sizeof(int)) dmap_ = Buf(map.size() * sizeof(int));
        ck(cudaMemcpyAsync(dmap_.p, map.data(), map.size() * sizeof(int), cudaMemcpyHostToDevice, stream_));
        ck(cudaStreamSynchronize(stream_));  // `map` is a host vector: the copy must land before it goes away
        const int * dm = dmap_.as<int>();
        Y.e_gate.fmt->expert_matvec_dedup2(Y.e_gate, Y.e_up, xh_.as<__half>(), ye_g_.as<bf16>(), ye_u_.as<bf16>(), effl_, dm, g_in, stream_);
        tqllm_cuda::mm_swigluoai_bf16(ye_g_.as<bf16>(), ye_u_.as<bf16>(), act_.as<__half>(), A * effl_, alpha_, limit_, stream_);
        Y.e_down.fmt->expert_matvec_dedup(Y.e_down, act_.as<__half>(), yd_.as<bf16>(), H_, dm + size_t(A) * DS, g_dn, stream_);
        ffn(Y.sh_gate_up, Y.sh_down, xn_.as<float>(), sh_.as<float>(), n);
        tqllm_cuda::glm_moe_sum(yd_.as<bf16>(), d_w_.as<float>(), sh_.as<float>(), fo_.as<float>(), topk_, H_, stream_, n);
        reduce(fo_.as<float>(), size_t(n) * H_);  // routed + shared partial sums of this rank's ff share
    }

    // the routed ids go to the host, prepare() loads what the layer's bank lacks, and the grouped expert kernels run on
    // the bank's slot ids (the same rows, another index): the output is the stack path's
    void moe_bank(Layer & Y, int l, int n) {
        const float * xn = xn_.as<float>();
        const __half * xh = xh_.as<__half>();
        std::vector<int> routed(size_t(n) * topk_);
        ck(cudaMemcpyAsync(routed.data(), d_ids_.p, routed.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        ck(cudaStreamSynchronize(stream_));
        auto & seen = seen_[size_t(l)];
        for (int id : routed) seen[size_t(id)] = 1;
        spread_[size_t(l)] = double(std::count(seen.begin(), seen.end(), uint8_t(1)));
        for (const auto & wv : glm_expert_waves(routed, topk_, bank_->slots(l))) {
            const auto slot = bank_->prepare(l, wv.ids);
            bank_->wait_misses();
            std::vector<int> where(size_t(experts_), -1), phys;
            for (size_t i = 0; i < wv.ids.size(); ++i) where[size_t(wv.ids[i])] = slot[i];
            for (int a = wv.first * topk_; a < (wv.first + wv.tokens) * topk_; ++a) phys.push_back(where[size_t(routed[size_t(a)])]);
            ck(cudaMemcpyAsync(d_slots_.p, phys.data(), phys.size() * sizeof(int), cudaMemcpyHostToDevice, stream_));
            const size_t off = size_t(wv.first) * topk_;
            const Mat g = bank_->gate(), u = bank_->up(), dn = bank_->down();
            const __half * xw = xh + size_t(wv.first) * H_;
            bf16 * gw = ye_g_.as<bf16>() + off * eff_, * uw = ye_u_.as<bf16>() + off * eff_;
            g.fmt->expert_matvec(g, xw, gw, eff_, d_slots_.as<int>(), wv.tokens, topk_, false, stream_);
            u.fmt->expert_matvec(u, xw, uw, eff_, d_slots_.as<int>(), wv.tokens, topk_, false, stream_);
            tqllm_cuda::mm_swigluoai_bf16(gw, uw, act_.as<__half>() + off * eff_, wv.tokens * topk_ * eff_, alpha_, limit_, stream_);
            dn.fmt->expert_matvec(dn, act_.as<__half>() + off * eff_, yd_.as<bf16>() + off * H_, H_, d_slots_.as<int>(), wv.tokens,
                                  topk_, true, stream_);
        }
        ffn(Y.sh_gate_up, Y.sh_down, xn, sh_.as<float>(), n);
        tqllm_cuda::glm_moe_sum(yd_.as<bf16>(), d_w_.as<float>(), sh_.as<float>(), fo_.as<float>(), topk_, H_, stream_, n);
        reduce(fo_.as<float>(), size_t(n) * H_);  // routed + shared partial sums of this rank's ff share
    }

    void attention(Layer & Y, uint32_t pos, int n) {
        const int qd = hql_ * hd_, kd = hkl_ * hd_, all = qd + 2 * kd;
        mv(Y.wqkv, xn_.as<float>(), qkv_.as<float>(), n);
        tqllm_cuda::glm_gather_cols(qkv_.as<float>(), all, 0, qd, n, q_.as<float>(), stream_);
        tqllm_cuda::glm_gather_cols(qkv_.as<float>(), all, qd, kd, n, k_.as<float>(), stream_);
        tqllm_cuda::glm_gather_cols(qkv_.as<float>(), all, qd + kd, kd, n, v_.as<float>(), stream_);
        tqllm_cuda::glm_rmsnorm(q_.as<float>(), Y.q_norm, q_.as<float>(), nullptr, hd_, eps_, stream_, n * hql_);
        tqllm_cuda::glm_rmsnorm(k_.as<float>(), Y.k_norm, k_.as<float>(), nullptr, hd_, eps_, stream_, n * hkl_);
        tqllm_cuda::mm_rope(q_.as<float>(), n, hql_, hd_, rot_, int(pos), theta_, stream_);
        tqllm_cuda::mm_rope(k_.as<float>(), n, hkl_, hd_, rot_, int(pos), theta_, stream_);
        tqllm_cuda::mm_put_rows_h(k_.as<float>(), Y.kc.p, kd, n, int(pos), stream_, kv8_);
        tqllm_cuda::mm_put_rows_h(v_.as<float>(), Y.vc.p, kd, n, int(pos), stream_, kv8_);
        const int* ids = nullptr;
        const int nblocks = int((pos + uint32_t(n) + uint32_t(iblock_) - 1) / uint32_t(iblock_));
        if (Y.iq.rows) {
            // index keys for every position (a later, longer context ranks them); index queries only once there are more
            // blocks than the indexer keeps (up to top-k blocks it keeps all, so attention stays dense)
            mv(Y.ik, xn_.as<float>(), ikv_.as<float>(), n);
            tqllm_cuda::glm_rmsnorm(ikv_.as<float>(), Y.ik_norm, ikv_.as<float>(), nullptr, idim_, eps_, stream_, n);
            tqllm_cuda::mm_rope(ikv_.as<float>(), n, 1, idim_, rot_, int(pos), theta_, stream_);
            tqllm_cuda::mm_put_rows_h(ikv_.as<float>(), Y.ikc.p, idim_, n, int(pos), stream_, kv8_);
            if (nblocks > itopk_) {
                mv(Y.iq, xn_.as<float>(), iqv_.as<float>(), n);
                tqllm_cuda::glm_rmsnorm(iqv_.as<float>(), Y.iq_norm, iqv_.as<float>(), nullptr, idim_, eps_, stream_, n * ih_);
                tqllm_cuda::mm_rope(iqv_.as<float>(), n, ih_, idim_, rot_, int(pos), theta_, stream_);
                // prompt chunks on tensor cores (4 index heads); decode / verify keep the FP32 kernel
                (n >= 64 && ih_ == 4 && iblock_ <= 128 && iblock_ % 8 == 0 ? tqllm_cuda::mm_index_scores_mma : tqllm_cuda::mm_index_scores)(
                    iqv_.as<float>(), Y.ikc.p, iscore_.as<float>(), n, ih_, idim_, int(pos), iblock_, nblocks, ilocal_, stream_, kv8_);
                tqllm_cuda::mm_topk_blocks(iscore_.as<float>(), iids_.as<int>(), n, nblocks, itopk_, stream_);
                ids = iids_.as<int>();
            }
        }
        // grouped: the query heads of a KV head share one read of its keys / values (16 per KV head here), decode and
        // verify included; a token's result does not depend on how many tokens the call holds
        // prompt chunks (not decode / verify) on tensor cores when a KV head has exactly 16 query heads
        const bool mma = n >= 64 && hql_ == 16 * hkl_ && iblock_ % 64 == 0 && nblocks <= 2048;
        (mma ? tqllm_cuda::mm_attn_mma : hql_ / hkl_ <= 32 ? tqllm_cuda::mm_attn_hg : tqllm_cuda::mm_attn_h)(
            q_.as<float>(), Y.kc.p, Y.vc.p, att_.as<float>(), n, hql_, hkl_, hd_, int(pos),
            1.f / std::sqrt(float(hd_)), ids, itopk_, iblock_, stream_, kv8_);
        mv(Y.wo, att_.as<float>(), fo_.as<float>(), n);
        reduce(fo_.as<float>(), size_t(n) * H_);  // each rank's heads add their part of the output projection
    }

    void capture(int layer, const float * x, int n) {
        const auto it = std::find(target_layers_.begin(), target_layers_.end(), layer);
        if (it == target_layers_.end()) return;
        const size_t k = size_t(it - target_layers_.begin()), row = target_layers_.size() * H_;
        ck(cudaMemcpy2DAsync(feat_.as<float>() + k * H_, row * sizeof(float), x, size_t(H_) * sizeof(float), size_t(H_) * sizeof(float),
                             size_t(n), cudaMemcpyDeviceToDevice, stream_));
    }
    // positions pos .. pos + n - 1 join every draft layer's context K / V (a rejected row is overwritten later)
    void draft_context(uint32_t pos, int n) {
        const int kd = hk_ * hd_;
        mv(dfc_, feat_.as<float>(), fused_.as<float>(), n);
        tqllm_cuda::glm_rmsnorm(fused_.as<float>(), dhid_norm_, fused_.as<float>(), nullptr, H_, eps_, stream_, n);
        for (auto & Y : D_) {
            mv(Y.wkv, fused_.as<float>(), dkv_.as<float>(), n);
            tqllm_cuda::glm_gather_cols(dkv_.as<float>(), 2 * kd, 0, kd, n, dk_.as<float>(), stream_);
            tqllm_cuda::glm_gather_cols(dkv_.as<float>(), 2 * kd, kd, kd, n, dv_.as<float>(), stream_);
            tqllm_cuda::glm_rmsnorm(dk_.as<float>(), Y.k_norm, dk_.as<float>(), nullptr, hd_, eps_, stream_, n * hk_);
            tqllm_cuda::mm_rope(dk_.as<float>(), n, hk_, hd_, rot_, int(pos), theta_, stream_);
            tqllm_cuda::mm_put_rows_h(dk_.as<float>(), Y.kc.p, kd, n, int(pos), stream_, kv8_);
            tqllm_cuda::mm_put_rows_h(dv_.as<float>(), Y.vc.p, kd, n, int(pos), stream_, kv8_);
        }
    }
    // one draft pass: block [anchor, mask ...] at positions anchor_pos ..; returns the block's greedy tokens
    std::vector<uint32_t> draft(uint32_t anchor, uint32_t anchor_pos) {
        const int B = dblock_, kd = hk_ * hd_, qd = hq_ * hd_;
        const auto & et = tensor("token_embd.weight");
        uint64_t el = 0, by = 0;
        gguf_block_layout(et.type, el, by);
        ck(cudaStreamSynchronize(stream_));
        for (int t = 0; t < B; ++t)
            dequantize_host(et.type, bytes("token_embd.weight").data + size_t(t ? uint32_t(dmask_) : anchor) * (uint64_t(H_) / el * by),
                            h_embed_ + size_t(t) * H_, H_);
        ck(cudaMemcpyAsync(dh_.p, h_embed_, size_t(B) * H_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
        for (auto & Y : D_) {
            tqllm_cuda::glm_rmsnorm(dh_.as<float>(), Y.attn_norm, dxn_.as<float>(), nullptr, H_, eps_, stream_, B);
            mv(Y.wq, dxn_.as<float>(), dq_.as<float>(), B);
            tqllm_cuda::glm_rmsnorm(dq_.as<float>(), Y.q_norm, dq_.as<float>(), nullptr, hd_, eps_, stream_, B * hq_);
            tqllm_cuda::mm_rope(dq_.as<float>(), B, hq_, hd_, rot_, int(anchor_pos), theta_, stream_);
            mv(Y.wkv, dxn_.as<float>(), dkv_.as<float>(), B);
            tqllm_cuda::glm_gather_cols(dkv_.as<float>(), 2 * kd, 0, kd, B, dk_.as<float>(), stream_);
            tqllm_cuda::glm_gather_cols(dkv_.as<float>(), 2 * kd, kd, kd, B, dv_.as<float>(), stream_);
            tqllm_cuda::glm_rmsnorm(dk_.as<float>(), Y.k_norm, dk_.as<float>(), nullptr, hd_, eps_, stream_, B * hk_);
            tqllm_cuda::mm_rope(dk_.as<float>(), B, hk_, hd_, rot_, int(anchor_pos), theta_, stream_);
            tqllm_cuda::mm_attn_draft(dq_.as<float>(), Y.kc.p, Y.vc.p, dk_.as<float>(), dv_.as<float>(),
                                      datt_.as<float>(), B, hq_, hk_, hd_, int(anchor_pos), dwindow_, 1.f / std::sqrt(float(hd_)), stream_, kv8_);
            mv(Y.wo, datt_.as<float>(), dfo_.as<float>(), B);
            tqllm_cuda::glm_add(dh_.as<float>(), dfo_.as<float>(), B * H_, stream_);
            tqllm_cuda::glm_rmsnorm(dh_.as<float>(), Y.ffn_norm, dxn_.as<float>(), nullptr, H_, eps_, stream_, B);
            ffn(Y.gate_up, Y.down, dxn_.as<float>(), dfo_.as<float>(), B);
            tqllm_cuda::glm_add(dh_.as<float>(), dfo_.as<float>(), B * H_, stream_);
        }
        (void)qd;
        tqllm_cuda::glm_rmsnorm(dh_.as<float>(), dout_norm_, dxn_.as<float>(), nullptr, H_, eps_, stream_, B);
        mv(out_, dxn_.as<float>(), dlog_.as<float>(), B);
        const int a = int(anchor);
        ck(cudaMemcpyAsync(dtok_.p, &a, sizeof(int), cudaMemcpyHostToDevice, stream_));
        for (int k = 0; k < B; ++k) {  // Markov: each slot's logits + W2 W1[the token chosen before it]
            tqllm_cuda::mm_gather_row(dw1_, dtok_.as<int>() + k, dlat_.as<float>(), drank_, stream_);
            mv(dw2_, dlat_.as<float>(), dbias_.as<float>(), 1);
            tqllm_cuda::mm_add(dlog_.as<float>() + size_t(k) * vocab_, dbias_.as<float>(), vocab_, stream_);
            tqllm_cuda::glm_argmax(dlog_.as<float>() + size_t(k) * vocab_, vocab_, dtok_.as<int>() + k + 1, stream_);
        }
        std::vector<int> ids(size_t(B + 1));
        ck(cudaMemcpyAsync(ids.data(), dtok_.p, ids.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        ck(cudaStreamSynchronize(stream_));
        return std::vector<uint32_t>(ids.begin() + 1, ids.end());
    }

    // TQLLM_MINIMAX_ROUTES=FILE: per-layer routed-expert counts (int64 [layers][experts]) written at exit, for hot / cold
    // expert maps (the counts also include verify rows that are rejected later)
    std::vector<int64_t> route_counts_;
    void count_routes(int layer, int n) {
        static const char * path = std::getenv("TQLLM_MINIMAX_ROUTES");
        if (!path) return;
        if (route_counts_.empty()) route_counts_.assign(L_.size() * size_t(experts_), 0);
        std::vector<int> ids(static_cast<size_t>(n) * topk_);
        ck(cudaMemcpyAsync(ids.data(), d_ids_.p, ids.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        ck(cudaStreamSynchronize(stream_));
        for (int id : ids) if (id >= 0 && id < experts_) ++route_counts_[size_t(layer) * experts_ + size_t(id)];
    }
    void write_routes() const {
        const char * path = std::getenv("TQLLM_MINIMAX_ROUTES");
        if (!path || route_counts_.empty()) return;
        if (std::FILE * f = std::fopen(path, "wb")) { std::fwrite(route_counts_.data(), 8, route_counts_.size(), f); std::fclose(f); }
    }

    // TQLLM_MINIMAX_DUMP=FILE: the residual stream after every layer, as records {int32 layer, pos, n; float [n][H]}
    // (tools/minimax_ref.py writes the same states as h1..hN)
    void dump(int layer, uint32_t pos, int n) {
        static const char * path = std::getenv("TQLLM_MINIMAX_DUMP");
        if (!path) return;
        static std::FILE * f = std::fopen(path, "wb");
        if (!f) throw std::runtime_error(std::string("minimax: cannot write ") + path);
        std::vector<float> h(size_t(n) * H_);
        ck(cudaMemcpyAsync(h.data(), x_.p, h.size() * 4, cudaMemcpyDeviceToHost, stream_));
        ck(cudaStreamSynchronize(stream_));
        const int32_t head[3] = {layer, int32_t(pos), n};
        std::fwrite(head, 4, 3, f);
        std::fwrite(h.data(), 4, h.size(), f);
        std::fflush(f);
    }

    // rank 0 -> rank 1 before every forward: {n, pos, tokens[chunk]} (n < 0: stop)
    void send_header(int n, int pos, const uint32_t * tokens) {
        std::vector<int32_t> h(size_t(2 + chunk_), 0);
        h[0] = n; h[1] = pos;
        for (int t = 0; t < n; ++t) h[size_t(2 + t)] = int32_t(tokens[t]);
        ck(cudaMemcpyAsync(hdr_.p, h.data(), h.size() * 4, cudaMemcpyHostToDevice, stream_));
        tp_->broadcast(hdr_.p, h.size() * 4, 0, stream_);
        ck(cudaStreamSynchronize(stream_));  // `h` goes out of scope
    }
public:
    // rank > 0: mirror rank 0's forwards until it stops
    void follow() override {
        if (!tp_ || R_ == 0) throw std::runtime_error("minimax: follow() is for tensor-parallel ranks > 0");
        std::vector<int32_t> h(size_t(2 + chunk_));
        std::vector<uint32_t> toks(static_cast<size_t>(chunk_));
        for (;;) {
            tp_->broadcast(hdr_.p, h.size() * 4, 0, stream_);
            ck(cudaMemcpyAsync(h.data(), hdr_.p, h.size() * 4, cudaMemcpyDeviceToHost, stream_));
            ck(cudaStreamSynchronize(stream_));
            if (h[0] < 0) return;
            for (int t = 0; t < h[0]; ++t) toks[size_t(t)] = uint32_t(h[size_t(2 + t)]);
            forward(toks.data(), h[0], uint32_t(h[1]), 0);
        }
    }
    void stop_followers() override {
        if (tp_ && R_ == 0) { const uint32_t none = 0; send_header(-1, 0, &none); }
    }
private:
    // n > chunk_ is a prompt: pchunk_ tokens per forward and the head only on the last token, so only next[n - 1] is
    // produced (the other rows are set to ~0u); a verify block (n <= chunk_) gets every row's argmax.
    void run(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) {
        const bool prompt = n > chunk_;
        const int step = prompt ? pchunk_ : chunk_;
        if (prompt && next) std::fill(next, next + n, ~0u);
        for (int i = 0; i < n; i += step) {
            const int c = std::min(step, n - i), rows = prompt ? (i + c == n ? 1 : 0) : c;
            forward(tokens + i, c, pos + uint32_t(i), rows);
            if (!next || rows == 0) continue;
            if (sampling()) {  // top-16 per row to the host; a prompt's last row is drawn here, a verify block's in spec_advance
                top16(rows);
                if (prompt) next[n - 1] = draw(dist(0));
                continue;
            }
            tqllm_cuda::glm_argmax_rows(logits_.as<float>(), vocab_, rows, d_arg_.as<int>(), stream_);
            std::vector<int> ids(static_cast<size_t>(rows));
            ck(cudaMemcpyAsync(ids.data(), d_arg_.p, size_t(rows) * sizeof(int), cudaMemcpyDeviceToHost, stream_));
            ck(cudaStreamSynchronize(stream_));
            for (int j = 0; j < rows; ++j) next[i + c - rows + j] = uint32_t(ids[size_t(j)]);
        }
    }
    struct Sampling { float temperature = 0.f, top_p = 1.f; int top_k = 0; } samp_;
    bool ignore_eos_ = false;
    std::mt19937_64 rng_{0};
    Buf s_val_, s_idx_, s_scr_;
    std::vector<float> h_slg_;
    std::vector<uint32_t> h_sid_;
    bool sampling() const { return samp_.temperature > 0.f; }
    // the head rows' top-16 exact logits and ids -> h_slg_ / h_sid_ [row][16]
    void top16(int rows) {
        if (!s_scr_.p) {
            const size_t R = size_t(std::max(chunk_, 1));
            s_val_ = Buf(R * 16 * 4); s_idx_ = Buf(R * 16 * 4); s_scr_ = Buf(size_t(8) * 64 * 16 * 8);
            h_slg_.resize(R * 16); h_sid_.resize(R * 16);
        }
        for (int r0 = 0; r0 < rows; r0 += 8) {  // topk16_big takes up to 8 rows
            const int nr = std::min(8, rows - r0);
            tqllm_cuda::topk16_big(logits_.as<float>() + size_t(r0) * vocab_, vocab_, nr, s_val_.as<float>() + size_t(r0) * 16,
                                   s_idx_.as<uint32_t>() + size_t(r0) * 16, s_scr_.p, stream_);
        }
        ck(cudaMemcpyAsync(h_slg_.data(), s_val_.p, size_t(rows) * 16 * 4, cudaMemcpyDeviceToHost, stream_));
        ck(cudaMemcpyAsync(h_sid_.data(), s_idx_.p, size_t(rows) * 16 * 4, cudaMemcpyDeviceToHost, stream_));
        ck(cudaStreamSynchronize(stream_));
    }
    struct Dist { uint32_t id[16]; double p[16]; int n = 0; };
    Dist dist(int row) const {
        Dist d;
        int ord[16];
        for (int i = 0; i < 16; ++i) ord[i] = i;
        const float * lg = h_slg_.data() + size_t(row) * 16;
        const uint32_t * id = h_sid_.data() + size_t(row) * 16;
        std::sort(ord, ord + 16, [&](int a, int b) { return lg[a] > lg[b]; });
        const int k = samp_.top_k > 0 ? std::min(samp_.top_k, 16) : 16;
        double sum = 0;
        for (int i = 0; i < k; ++i) {
            d.id[i] = id[ord[i]];
            d.p[i] = std::exp(double(lg[ord[i]] - lg[ord[0]]) / samp_.temperature);
            sum += d.p[i];
        }
        double cum = 0;
        d.n = k;
        for (int i = 0; i < k; ++i) {
            cum += d.p[i] / sum;
            if (cum >= samp_.top_p) { d.n = i + 1; break; }
        }
        double s2 = 0;
        for (int i = 0; i < d.n; ++i) s2 += d.p[i];
        for (int i = 0; i < d.n; ++i) d.p[i] /= s2;
        return d;
    }
    uint32_t draw(const Dist & d, uint32_t exclude = UINT32_MAX) {
        double total = 0;
        for (int i = 0; i < d.n; ++i) total += d.id[i] == exclude ? 0 : d.p[i];
        double u = std::uniform_real_distribution<double>(0, total)(rng_);
        for (int i = 0; i < d.n; ++i) {
            if (d.id[i] == exclude) continue;
            if ((u -= d.p[i]) <= 0) return d.id[i];
        }
        for (int i = d.n - 1; i >= 0; --i) if (d.id[i] != exclude) return d.id[i];
        return d.id[0];
    }
    // rows: how many of the last tokens get logits (0 inside a prompt; at most chunk_)
    void forward(const uint32_t * tokens, int n, uint32_t pos, int rows) {
        if (rows > chunk_ || rows > n) throw std::runtime_error("minimax: head rows beyond the verify chunk");
        if (pos + uint32_t(n) > context_) throw std::runtime_error("minimax: position beyond the context");
        if (tp_ && R_ == 0) send_header(n, int(pos), tokens);
        const auto & et = tensor("token_embd.weight");
        uint64_t el = 0, by = 0;
        gguf_block_layout(et.type, el, by);
        ck(cudaStreamSynchronize(stream_));
        for (int t = 0; t < n; ++t) {
            if (int(tokens[t]) >= vocab_) throw std::runtime_error("minimax: token id beyond the vocabulary");
            dequantize_host(et.type, bytes("token_embd.weight").data + size_t(tokens[t]) * (uint64_t(H_) / el * by),
                            h_embed_ + size_t(t) * H_, H_);
        }
        ck(cudaMemcpyAsync(x_.p, h_embed_, size_t(n) * H_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
        for (size_t li = 0; li < L_.size(); ++li) {
            auto & Y = L_[li];
            tqllm_cuda::glm_rmsnorm(x_.as<float>(), Y.attn_norm, xn_.as<float>(), nullptr, H_, eps_, stream_, n);
            attention(Y, pos, n);
            tqllm_cuda::glm_add(x_.as<float>(), fo_.as<float>(), n * H_, stream_);
            tqllm_cuda::glm_rmsnorm(x_.as<float>(), Y.ffn_norm, xn_.as<float>(), nullptr, H_, eps_, stream_, n);
            if (int(li) < dense_lead_) { ffn(Y.gate_up, Y.down, xn_.as<float>(), fo_.as<float>(), n); reduce(fo_.as<float>(), size_t(n) * H_); }
            else moe(Y, int(li), n);
            tqllm_cuda::glm_add(x_.as<float>(), fo_.as<float>(), n * H_, stream_);
            dump(int(li), pos, n);
            if (has_draft() && li + 1 < L_.size()) capture(int(li), x_.as<float>(), n);
        }
        tqllm_cuda::glm_rmsnorm(x_.as<float>(), out_norm_, xn_.as<float>(), nullptr, H_, eps_, stream_, n);
        // the last layer's state as HF's hidden_states hands it to the drafter: after the final norm
        if (has_draft()) { capture(int(L_.size()) - 1, xn_.as<float>(), n); draft_context(pos, n); }
        if (R_ == 0 && rows > 0) mv(out_, xn_.as<float>() + size_t(n - rows) * H_, logits_.as<float>(), rows);
        if (rows > 0) last_rows_ = rows;
    }
};

}  // namespace

bool is_minimax_gguf(const std::string & path) { return gguf_architecture(path) == "minimax-m3"; }

std::unique_ptr<Engine> make_minimax_engine(const std::string & path, uint32_t context, const std::string & draft,
                                            const std::string & tokenizer, const TpConfig & tp, const std::string & hot_map) {
    return std::make_unique<MiniMaxEngine>(path, context, draft, tokenizer, tp, hot_map);
}

}  // namespace tqllm
