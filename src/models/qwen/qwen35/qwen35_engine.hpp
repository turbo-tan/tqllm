#pragma once
#include "hf_folder.hpp"
// The qwen35 engine class (declarations; small and template methods inline). Definitions: qwen35.cpp (factory),
// qwen35_load.cpp, qwen35_forward.cpp, qwen35_spec.cpp. Internal to src/models/qwen/qwen35/.
#include "prefill_observer_cuda.hpp"
// Qwen3.5 dense hybrid decode engine (text path): Gated DeltaNet + gated full attention, BF16 weights.
#include "dev_alloc.cuh"
#include "debug.hpp"
#include "batch_graph_key.hpp"
#include "engine.hpp"
#include <random>
#include "nlohmann/json.hpp"
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
#include "qwen35_guard.hpp"
#include "qwen35_kernels.cuh"
#include "quant_kernels.cuh"
#include "moe_kernels.cuh"
#include "vision_kernels.cuh"
#include "safetensors.hpp"
#include "dequant.hpp"
#include "dflash_kernels.cuh"
#include "kv_format.hpp"
#include "kv_store.hpp"
#include "tqllm/rope_scaling.hpp"
#include "weight_format.hpp"
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
void check_grouped_expert_format(const Mat & m, const std::string & prefix);
MixStaging mix_staging(Basis basis, bool bf16_ba, bool quantized_model);
}  // namespace tqllm
#include "qwen35_types.hpp"
#include "token_sampling.hpp"
namespace tqllm {
namespace qwen35_detail {

class Qwen35Engine final : public Engine {
public:
    // `path`: a checkpoint directory (safetensors, BF16), a .gguf file (TQ3_4S / Q6_K / F32), or a mapped folder (exl3).
    Qwen35Engine(const std::string & path, const std::string & tokpath, uint32_t ctx, int slots, const EngineOptions & o)
        : opt(o), context_(ctx) {
        glru.set_cap(size_t(std::max(1, opt.graph_cache)));
        graph_keep = size_t(std::max(0L, opt.graph_keep_mb)) << 20;
        if (slots < 1 || slots > 64) throw std::invalid_argument("slots must be in 1..64");
        if (!context_) throw std::invalid_argument("context must be positive");
        // a .gguf file, or a checkpoint folder hf_arch.cpp maps (exl3 as published): both read through gguf_open
        const bool gguf = (path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0) || hf_folder_arch(path) == "qwen35";
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
            auto memlog = [](const char * stage) {  // TQLLM_DEBUG=memlog: device memory in use after each load stage, seconds since the first load began
                static const bool on = tqllm::debug_on("memlog");
                static const auto t0 = std::chrono::steady_clock::now();
                size_t fr = 0, tot = 0;
                if (on && cudaMemGetInfo(&fr, &tot) == cudaSuccess)
                    std::fprintf(stderr, "tqllm: %-9s %.2f GB in use, %.1f s\n", stage, double(tot - fr) / 1e9,
                                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            };
            memlog("start");
            if (gguf) load_gguf(*gf);
            else load_safetensors(path);
            memlog("weights");
            nslots = slots;
            allocate();
            memlog("buffers");
            if (tqllm::drafters(opt).dflash && gguf) load_dflash(opt.draft_model.c_str());
            memlog("drafter");
            calibrate_prefill();
            memlog("calibrate");
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
    void release_raw();

    uint32_t step(uint32_t id, uint32_t pos, bool sample, float * capture) override;

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
    size_t host_placed_bytes(bool head) const override { return head ? placed_head_b : placed_embed_b; }
    size_t placed_head_b = 0, placed_embed_b = 0;
    size_t kv_round_bytes() const override { return kv_mem ? size_t(nslots) * kv_pool_layers() * 2 * kv_mem->granule() : 0; }
    // TqCache compaction: the next 8K block at cmp_pos is fully in host memory and there are enough recent queries
    bool kv_idle_pending() const override { return !opt.kv_compact_turn && cmp_pending(); }
    bool cmp_pending() const;
    bool kv_idle_step() override;
    void cmp_upload();
    void cmp_reset() {
        cmp_pos = 0, ring_used = 0;
        cmp_upload();
    }
    void pc_drop() { pc_live = false, pc_snap = 0; }
    void pc_copy(bool save);
    // cache positions of prompt that can be kept (the recurrent state is set to match); the rest is prefilled
    uint32_t pc_match(const std::vector<uint32_t> & prompt);
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

    void mm(const Mat & m, const bf16 * input, bf16 * output);

    void to_host(const bf16 * src, size_t n, float * dst) {
        std::vector<bf16> tmp(n);
        ck(cudaMemcpyAsync(tmp.data(), src, n * sizeof(bf16), cudaMemcpyDeviceToHost, stream));
        sync_ck();
        for (size_t i = 0; i < n; ++i) dst[i] = __bfloat162float(tmp[i]);
    }

    const void * put(const void * host, size_t bytes);

    const float * put_f32(const std::vector<float> & v) {
        return static_cast<const float *>(put(v.data(), v.size() * 4));
    }

    // ---- safetensors (HF BF16) ----
    const SafeTensor & st(const std::map<std::string, SafeTensor> & idx, const std::string & name, uint64_t elems);

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

    void load_safetensors(const std::string & dir);

    // ---- GGUF (llama.cpp-tq3 layout: norms already 1 + w, ssm_a = -exp(A_log), tiled value heads) ----
    const GgufTensor & gt(const GgufFile & f, const std::string & name) {
        const auto it = f.tensors.find(name);
        if (it == f.tensors.end()) throw std::runtime_error("missing GGUF tensor: " + name);
        return it->second;
    }

    // A matrix of a registered weight format: remember the basis its matmul inputs need (one per model).
    Mat adopt(Mat m);
    Upload uploader() {
        return [this](const void * h, size_t b) { return put(h, b); };
    }

    // F32 GGUF matrices (llama.cpp keeps small ones such as ssm_alpha / ssm_beta in F32) run as BF16
    static std::vector<char> f32_to_bf16_host(const std::vector<char> & raw);

    // F16 GGUF matrices (llama-quantize COPY mode keeps ssm_alpha / ssm_beta in F16) also run as BF16.
    // F16 -> F32 by pure IEEE-754 bit manipulation (no <cmath> dependency in this TU), then round to BF16.
    static inline float f16_bits_to_f32(uint16_t h);
    static std::vector<char> f16_to_bf16_host(const std::vector<char> & raw);

    // Matmul matrices load through their weight format (src/formats); `repack` = false loads raw bytes (embeddings).
    Mat g_mat(const GgufFile & f, const std::string & name, int rows, int cols, bool repack = true);

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
    std::vector<std::vector<char>> moe_placement(const GgufFile & f);
    static Mat expert_rows(const Mat & m, int e, int rows);

    // Row-concatenation of same-width matrices into one device matrix: one buffer when the types agree, else a composite
    // of per-part formats ({} when the types cannot be combined: the caller falls back to separate matrices).
    std::vector<std::unique_ptr<WeightFormat>> composites;
    Mat g_cat(const GgufFile & f, const std::vector<std::pair<std::string, int>> & parts, int cols);

    const float * g_f32(const GgufFile & f, const std::string & name, uint64_t n);

    void load_gguf(const GgufFile & f);

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
    void reset_state();

    // RoPE scaling from the CLI, else the GGUF; the tables are global to the qwen35 kernels (one model per process).
    void set_rope();

    // Per-sequence KV and Gated DeltaNet state of the active slot.
    void alloc_seq_state();

    void allocate();

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
    void alloc_moe_cache();
    void alloc_moe_stage();
    Buf moe_prof; std::string moe_prof_path;  // TQLLM_DEBUG=moe_profile: routing counts [L][E]
    void moe_profile_write();
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
    // Sampling (temperature > 0): the head also returns each column's top-16 ids with exact logits (token_sampling.hpp)
    SamplingParams samp;
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
        SamplingParams samp;
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
    void swap_seq(SeqState & s);
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
    bool mtp_head4_sub = false;  // --mtp --draft-vocab: the MTP draft head scores dsub_ids rows of head4
    cudaStream_t commit_side = nullptr;  // TQLLM_DEBUG=commit_overlap: commit replay stream
    cudaEvent_t commit_fork = nullptr, commit_done = nullptr;
    bool commit_inflight = false;
    void join_commit() {
        if (commit_inflight) { ck(cudaStreamWaitEvent(stream, commit_done, 0)); commit_inflight = false; }
    }
    std::vector<char> mtp_draft_q6k_host;  // Q6_K draft-vocabulary rows until allocate() decides (head4 or Q6_K)
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
    void upload_rpos(int nseg, const std::function<int(int)> & delta);
    int rdelta_of(int slot) const { return slot == active ? gen.rdelta : parked[size_t(slot)].gen.rdelta; }
    // captured graphs, one bounded cache (src/core/graph_lru.hpp): verify per slot / shape, batched verify keyed by the segment
    // layout, batched drafts keyed by the slot list, tree commit + drafter inject keyed by k and the slot state
    enum : int { kGraphVerify, kGraphBatch, kGraphDraft, kGraphCommit };
    tqllm::GraphLru<cudaGraphExec_t> glru;
    size_t graph_keep = 0;  // a miss evicts while less than this is free (0: by count only, e.g. WSL reads ~0 free)
    template <class Body>
    cudaGraphExec_t cached_graph(int family, uint64_t key, Body && body) {
        if (cudaGraphExec_t e = glru.find(family, key)) return e;
        static const bool dbg_graphs = tqllm::debug_on("graphs");
        if (dbg_graphs) {
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            std::fprintf(stderr, "graph miss family %d key %016llx (cached %zu, evictions %zu, free %zu MB)\n", family,
                         (unsigned long long) key, glru.size(), glru.evictions(), fr >> 20);
        }
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
    void st(const char * name);
    size_t v_mix_rows = 0;
    Buf v_xb, v_mixa, v_mixb;  // W4A16: BF16 copy of the activation for BF16 matrices; [qkv|z] and [b|a] partials
    Buf v_xu;                  // rotated-basis models: the un-rotated FP16 normalised activation the BF16 [b|a] rows read
    int last_n = 0;
    double last_prefill_ms = 0;
    std::chrono::steady_clock::time_point dec_log_tp;  // last TQLLM_DEBUG=decode progress print

    // v_xh holds the input prepared in `in` (the model's basis unless a caller prepared its own)
    void mm_h(const Mat & m, bf16 * out, int n) { mm_h(m, out, n, basis_); }
    void mm_h(const Mat & m, bf16 * out, int n, Basis in);
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
    void calibrate_prefill();
    PrefillPath path_of(const Mat & m, int n) const;

    // Prefill chunk: FP16 weights once per chunk + cuBLAS GEMM; column j of the result goes to out + j * ldo (ldo >= rows;
    // the rows past m.rows receive unspecified values, for the caller to overwrite).
    // FP16 accumulate (2x tensor rate on GA10x; same answers on the 7K/64K probes)
    // m on n columns takes the prefill GEMM (dequantise + cuBLAS) path of mm_h
    bool prefill_gemm(const Mat & m, int n) const {
        return m.fmt && gemm_w.p && path_of(m, n) == PrefillPath::dequant;
    }
    // out == nullptr (FP16 accumulate only): the FP16 result stays in gemm_c, [n][ldo]
    void gemm_rows(const Mat & m, bf16 * out, int n, int ldo);
    // rows of m into rows [r_off, r_off + m.rows) of gemm_c ([n][ldo], FP16)
    void gemm_c_rows(const Mat & m, int n, int ldo, int r_off);

    void rot_h(const bf16 * x, __half * xh, int k, int n) {
        prep_input(basis_, x, xh, k, n);
    }
    void nrot_h(const bf16 * x, const float * w, const float * partial, __half * xh, int k, int n);
    // logits [n][vocab] of a 4-bit (W4A16) LM head
    // BF16 [n][k] -> FP16 matmul input in `b`
    void prep_input(Basis b, const bf16 * x, __half * xh, int k, int n);
    // logits [n][vocab] from BF16 hidden states when the LM head is a registered weight format
    void head_mv(const bf16 * in, int n);
    int mix_rows(const Layer & L) const { return L.w_mix.rows + (L.w_ba.p ? L.w_ba.rows : 0); }
    // v_xb = the current layer's normalised activation in BF16, in the basis the BF16 [b|a] rows are stored in
    // (bf16_ba_unrotated): the rotated copy the quantized [qkv|z] rows read, or a second, un-rotated FP16 copy
    // (norm_h writes exactly the values norm_rotate_h rotates).
    void ba_bf16(const Layer & L, int n, cudaStream_t s);
    // [qkv|z] (u4h) and [b|a] (BF16) into the [qkv|z|b|a] mix rows of T
    void mix_mm(const Layer & L, bf16 * T, int n);

public:
    void verify(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) override;
    // kernel launches are not checked one by one: a launch that fails (the driver or a library cannot allocate on a
    // nearly full device) would silently skip its kernel, so every stream sync also collects the last launch error
    // sync without joining an in-flight commit replay: MTP draft passes read neither the recurrent states nor its scratch
    void sync_main() {
        ck(cudaStreamSynchronize(stream));
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(e));
    }
    void sync_ck() {
        join_commit();
        ck(cudaStreamSynchronize(stream));
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(e));
    }
    bool full_logits = tqllm::debug_value("full_head") != nullptr;  // TQLLM_DEBUG=full_head: full-vocabulary int8 head, the reference for the shortlist
    void set_full_logits(bool on) override { full_logits = on; }
    const float * verify_logits() const override { return v_logits.as<float>(); }
    void set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence) override;
    void set_penalties(float repeat, float frequency) override;

    // positions [pos, pos + n) are written next: evict old granules to host if the device window needs room (between
    // launches; remaps keep every address, so captured graphs stay valid) and publish the bytes each layer stages back
    void kv_stage_window(uint32_t pos, int n);
    // every KV write of n positions at pos: staging window (host tail) or the device-resident TqCache bookkeeping
    void kv_track(uint32_t pos, int n);
    void kv_prefetch(uint32_t a);

    void verify_async(const uint32_t * tokens, const int * pos_host, int n, uint32_t pos,
                      uint32_t * next, bool copy_all) {
        join_commit();  // a commit replay on the side stream writes the state this verify reads
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
    void attn_mixer(const Layer & L, bf16 * kc, bf16 * vc, int n, int max_seq, int li = -1);

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
    void ensure_mq_split();
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
    void ffn_block(const Layer & L, int n, const bf16 * resid = nullptr);

    // logits + per-column argmax (-> v_tok) of the n hidden vectors in X, normalised with `norm_w`.
    // Head on columns [col0, col0 + n) of v_x; writes v_tok (and the sampling top-16) at the same columns.
    void head_argmax(const float * norm_w, int n, int col0 = 0);

    // Column col's sampling distribution (token_sampling.hpp top_dist); recent: the last output tokens before this column
    TopDist dist(int col, const std::vector<uint32_t> & recent = {}) const {
        return top_dist(h_slg.data() + size_t(col) * 16, h_sid.data() + size_t(col) * 16, samp, recent);
    }

    // token embeddings of n device-resident ids -> Y [n][hidden] (BF16)
    void embed_multi(const uint32_t * dtok, int n, bf16 * Y);

    void verify_body(int n, int max_seq);


    // Copies the input of target layer l (if it is a DFlash feature layer) into feats [n][n_feat * H].
    void capture_feature(uint32_t l, const bf16 * x, bf16 * feats, int n);

    // FP32 host tensor(s) -> TQ3_4S on the device (row-concatenated), repacked for the fast kernels.
    // Q4_K -> u4h: every 32-weight sub-block keeps its codes, scale d * sc and min -dmin * m (FP16).
    Mat q4k_to_u4h(const GgufFile & f, const std::vector<std::string> & names, int cols);

    Mat to_tq3(const GgufFile & f, const std::vector<std::string> & names, int cols);

    const float * f32_of(const GgufFile & f, const std::string & name, uint64_t n);

    const bf16 * bf16_table(const GgufFile & f, const std::string & name, uint64_t n);

    // [rows][rank] table (GGUF ne = {rank, rows}) as BF16 [cfg.vocab][rank], rows past the file's vocabulary zero
    const bf16 * vocab_table(const GgufFile & f, const std::string & name, int rank);

    // Drafter kernel grids independent of the position (sliding-window ring, no retrieval): drafts and commits can be
    // captured once as CUDA graphs. Retrieval sizes its launches by the position, so it runs eagerly.
    bool dfl_fixed_grids() const { return dfl.ring > 0 && !dfl.retr; }

    // Drafter K/V of the active slot: the sliding-window (or retrieval) ring, else the full context.
    void alloc_dfl_state();

    void load_dflash(const std::string & path);
    // 4-bit copy of head8: the two-stage verify head's first pass and the DFlash2 full-vocabulary draft head
    // logits of the draft-vocabulary rows (dsub_ids order) for n FP16 inputs
    void dsub_matvec(const __half * x, float * y, int n);
    void build_head4();
    // Pinned host memory mapped into the device (low-VRAM cards: embeddings and the int8 head are read sparsely)
    const void * to_mapped_host(const void * dev, size_t bytes);
    // Pinned mapped-host copy of raw bytes, returning the device-accessible (UVA) pointer: the -ncmoe
    // expert loader. The weight-format load() repack kernel and the decode expert kernels read m.p / m.aux
    // through this pointer over PCIe (same mechanism as embed_host / head8_host_alloc); no kernel changes.
    const void * host_put(const void * host, size_t bytes);
    Upload host_uploader() {
        return [this](const void * h, size_t b) { return host_put(h, b); };
    }
    std::vector<void *> host_blocks;

    // Verify rescoring reads 16 candidate rows per column, and the drafter uses head4 / its subset: the full int8 head
    // (1.27 GB for Qwen3.8) can live in pinned host memory the device reads over PCIe (~0.6 MB per 8-column round).
    void head8_host_alloc();
    int L_conv_rows() const { return dfl.L.empty() ? 0 : dfl.L[0].conv_attn.rows; }

    // generic TQ3 matvec on n rows of a BF16 input (rotates into v_xh first)
    void mmx(const Mat & m, const bf16 * in, bf16 * out, int n);

    // Pageable source on purpose: the copy is staged before cudaMemcpyAsync returns, so back-to-back calls cannot
    // race on a shared pinned buffer (the injection after a verify is not followed by a sync).
    void set_dpos(uint32_t pos) {
        const int v[2] = {int(pos), int(pos) + 1};
        ck(cudaMemcpyAsync(d_pos.p, v, sizeof(v), cudaMemcpyHostToDevice, stream));
    }

    // The inject projections as matvec batches of the format's max_batch_cols(): per-column results are identical to
    // narrower calls (the batched-matvec contract), while each weight is read once per batch of tokens instead of once
    // per 8-token group - the whole cost of the old prefill injection.
    void inj_mm(const Mat & m, const bf16 * in, bf16 * out, int n);
    // Encodes n rows of target features (device [n][nf*H]) for positions pos.. and writes every drafter layer's K/V.
    void dflash_inject(const bf16 * feats, int n, uint32_t pos) {
        set_dpos(pos);
        dflash_inject_body(feats, n);
    }
    void dflash_inject_body(const bf16 * feats, int n);

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
    void dspark_forward(uint32_t pos);

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
    void dflash_begin(const std::vector<uint32_t> & prompt, uint32_t count, int D);

    // Accepted draft count j for the active slot; res[j] becomes the token after them. Column col0 + i of the verify
    // holds the target's prediction after draft token i - 1 (greedy: res[i]; sampling: h_sid / h_slg top-16).
    int accept(const std::vector<uint32_t> & draft, uint32_t * res, int col0);

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

    void dflash_step(const int * slots, int ns, bool * done);

    // Prompt-lookup drafting (--lookup L, tree mode): the latest earlier occurrence of the last 6 tokens of prompt +
    // output proposes the tokens that followed it as an extra chain under the tree root (up to L columns, taken from the
    // DFlash tree's budget). Verification is unchanged, so outputs never depend on it.
    int lookup(uint32_t * dst, int maxlen) const;

    // TQLLM_DEBUG=decode: a progress line every ~2.5 s while decoding (tokens / wall time, same basis as reqlog).
    void dec_progress();

    // One draft + verify step on the active slot; false once the generation is complete.
    bool dflash_advance();

    std::vector<uint32_t> dflash_finish(uint64_t * stats);

public:
    bool has_mtp() const { return mtp.present; }

    // MTP draft head (Qwen3.5/3.8 nextn block). Runs n (token, previous hidden) pairs at positions pos, pos+1, ...:
    // x = eh_proj([enorm(embed(t)) ; hnorm(h)]), one full-attention decoder layer with its own KV cache, then the
    // shared LM head on shared_head_norm(x). Returns the greedy token after the last pair (the next draft) and keeps
    // the last pre-norm output as the hidden for chaining. `hidden`: device [n][H] (nullptr = chain from last MTP).
    uint32_t mtp_run(const uint32_t * tokens, const bf16 * hidden, int n, uint32_t pos, bool sample = true);

    int slots() const override { return nslots; }
    bool incremental() const override { return fmt_target; }
    bool has_drafter() const override { return dfl.present; }
    bool native_mtp() const override { return opt.native_mtp && mtp.present && !dfl.present; }  // MTP only (no -md)
    void select_slot(int s) override;
    // Without the DFlash drafter every step verifies one token per slot (verify-path numerics, batched over slots).
    void set_prompt_images(const std::vector<PromptImage> & ims) override;
    bool mtp_advance();

    void spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int D) override {
        if (!fmt_target) throw std::runtime_error("incremental decoding needs a quantized GGUF model");
        dflash_begin(prompt, count, D);
    }
    bool spec_advance() override;
    void spec_step(const int * slots, int n, bool * done) override {
        if (gen.native_mtp) {
            if (n != 1 || slots[0] != active) throw std::runtime_error("MTP step needs the active single slot");
            done[0] = !mtp_advance();
        } else dflash_step(slots, n, done);
    }
    std::vector<uint32_t> spec_finish(uint64_t * stats) override { return dflash_finish(stats); }
    const std::vector<uint32_t> & spec_tokens(int s) const override { return s == active ? gen.out : parked[size_t(s)].gen.out; }
    int drafter_features(const std::vector<uint32_t> & tokens, std::vector<uint16_t> & feats) override;

    bool prefill_state(const std::vector<uint32_t> & tokens, uint32_t pos0) override;

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
    void flush_pending();

    void commit(int k) override {
        if (k < 1 || k > last_n) throw std::runtime_error("commit: accepted count out of range");
        if (k < last_n) {
            // TQLLM_DEBUG=commit_overlap (opt-in until test-native-mtp covers --np > 1): replay on a side stream so it overlaps the MTP repair pass that follows (that pass
            // reads neither the recurrent states nor the replay's scratch); the next verify waits for it (join_commit)
            static const bool overlap = tqllm::debug_on("commit_overlap");
            if (overlap) {
                if (!commit_side) {
                    ck(cudaStreamCreateWithFlags(&commit_side, cudaStreamNonBlocking));
                    ck(cudaEventCreateWithFlags(&commit_fork, cudaEventDisableTiming));
                    ck(cudaEventCreateWithFlags(&commit_done, cudaEventDisableTiming));
                }
                ck(cudaEventRecord(commit_fork, stream));
                ck(cudaStreamWaitEvent(commit_side, commit_fork, 0));
                const cudaStream_t main_stream = stream;
                stream = commit_side;  // cached_graph captures on `stream`
                const uint64_t key = (uint64_t(1) << 40) | uint64_t(k) | (uint64_t(cur) << 16) | (uint64_t(sp) << 24) |
                                     (uint64_t(active) << 32) | (uint64_t(stash_col0) << 44);
                ck(cudaGraphLaunch(cached_graph(kGraphCommit, key, [&] { commit_body(k); }), commit_side));
                stream = main_stream;
                ck(cudaEventRecord(commit_done, commit_side));
                commit_inflight = true;
            } else commit_body(k);
        }
        cur = (cur + 1) % NSLOT;
        last_n = 0;
    }
    void commit_body(int k);
};
}  // namespace qwen35_detail
}  // namespace tqllm
