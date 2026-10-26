#pragma once
// Flash-Next model: DecodeModel (decode, layer-major prefill, MTP drafting) and the packed expert file.
#include "fn_layers.hpp"
#include "fn_cpu_experts.hpp"
#include <atomic>
#include <thread>
struct PinnedInts { int * p = nullptr; size_t n = 0; void need(size_t k) { if (k > n) { if (p) cudaFreeHost(p); ck(cudaMallocHost(reinterpret_cast<void **>(&p), k * sizeof(int))); n = k; } } };
inline double g_host_ms[3];  // decode host time blocked in: routed ids D2H, cache prepare, prefetch issue
struct DecodeModel {
    const Checkpoint & cp; const Fixtures & fx; const Qwen4ExpConfig & c; int h, width;
    MixedExpertCache & experts; MixedExpertCache * mtp_experts = nullptr;
    std::vector<RecurrentState> rec; std::vector<AttentionState> att; std::vector<std::unique_ptr<Device<float>>> ple_past;
    PleTokenHistory history; int position = 0;
    bool prefetch_on = std::getenv("TQFN_NO_PREFETCH") == nullptr; cudaStream_t copy_stream = nullptr, miss_stream = nullptr;
    // speculative decoding: per-token rollback state, final wide residual rows of the last step, MTP head state
    bool spec = false; int spec_pos0 = 0, spec_n = 0; size_t spec_hist0 = 0;
    std::vector<std::unique_ptr<Device<float>>> ple_saved, ple_old;
    std::unique_ptr<Device<float>> pingpong[2];  // prefill residual between layers (layer scratch is reset)
    std::unique_ptr<Device<float>> res_keep;  // [n][width] of the last step, persistent (grows with the prefill chunk)
    AttentionState mtp_att; std::unique_ptr<Device<float>> mtp_h;  // MTP block KV and its last wide output
    float draft_conf = 1.f;  // top probability of the last MTP draft (TQFN_DRAFT_TAU)
    // After a long prompt: the trunk's K/V pages wholly below len - keep move to pinned host memory (same addresses);
    // returns the device bytes freed. TQFN_KV_HOST_KEEP: positions kept on the device (32768); TQFN_KV_HOST=0: off.
    size_t kv_to_host() {
        static const int keep = std::getenv("TQFN_KV_HOST_KEEP") ? std::atoi(std::getenv("TQFN_KV_HOST_KEEP")) : 32768;
        size_t freed = 0;
        for (auto & a : att) {
            if (!a.k || !a.k->vmm || a.len <= 2 * keep) continue;
            const size_t row = a.k->bytes / size_t(g_kv_capacity), per = a.k->page_rows(row), cut = size_t(a.len - keep) / per;
            for (size_t i = 0; i < cut; ++i) for (KvBuf * b : {a.k.get(), a.v.get()})
                if (!b->on_host[i] && b->move(i, true)) freed += b->page;
        }
        return freed;
    }
    // Before a long prefill writes and re-reads its context: every host page back on the device (with its data, for a
    // reused prefix); returns the bytes moved. Pages that do not fit stay on the host (read over PCIe).
    size_t kv_to_device() {
        size_t moved = 0;
        for (auto & a : att) if (a.k && a.k->vmm)
            for (KvBuf * b : {a.k.get(), a.v.get()}) for (size_t i = 0; i < b->on_host.size(); ++i)
                if (b->on_host[i] && b->move(i, false)) moved += b->page;
        return moved;
    }
    DecodeModel(const Checkpoint & k, const Fixtures & f, MixedExpertCache & e, size_t capacity)
        : cp(k), fx(f), c(k.config), h(c.hidden), width(c.hidden * c.hc), experts(e), rec(c.layers), att(c.layers),
          ple_past(c.layers), history(c.ple, capacity), ple_saved(c.layers), ple_old(c.layers) {
        ck(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        ck(cudaStreamCreateWithFlags(&miss_stream, cudaStreamNonBlocking)); experts.set_miss_stream(miss_stream);
        PersistentAlloc keep; res_keep = std::make_unique<Device<float>>(size_t(64) * width); mtp_h = std::make_unique<Device<float>>(width);
    }
    ~DecodeModel() { if (copy_stream) { cudaStreamSynchronize(copy_stream); cudaStreamDestroy(copy_stream); }
                     if (miss_stream) { experts.set_miss_stream(nullptr); if (mtp_experts) mtp_experts->set_miss_stream(nullptr);
                                        cudaStreamSynchronize(miss_stream); cudaStreamDestroy(miss_stream); } }
    void set_spec(bool on) { spec = on; for (auto & r : rec) r.spec = on; }

    // One block (trunk layer 0..L-1, or the MTP block L) on n tokens; residual [n][width] is replaced.
    // a batched step over several sequences: rows [off, off + n) belong to model m (its PLE history, KV, recurrent state)
    struct Seg { DecodeModel * m; int off, n; };
    // Layer-major prefill: the dense part of a layer runs on sub-chunks and stops before the routed experts, leaving its
    // rows at `off` of these whole-chunk buffers (combined residual in place of the input rows, FFN inject, FP16 FFN
    // input, routing, shared expert output); the experts then run once for the whole chunk (prefill_layer)
    struct PrefillSink { float * wide; float * inject; __half * xr; int * routed; float * weights, * shared_weights, * sd; int off; };
    void layer_forward(int layer, std::unique_ptr<Device<float>> & residual, int n, const float * ple_rows,
                       AttentionState * attn, MixedExpertCache & cache, const std::vector<Seg> * segs = nullptr,
                       const PrefillSink * sink = nullptr);
    void layer_forward_step(int layer, std::unique_ptr<Device<float>> & residual, int n, const float * ple_rows,
                            AttentionState * attn, MixedExpertCache & cache, const std::vector<Seg> * segs,
                            const PrefillSink * sink, bool plain);

    // Layer-major prefill: staging banks for the experts a layer's cache bank does not hold, filled by the copy engine.
    // The next layer's first two groups are queued before its routing is known (its non-resident experts by id), so
    // they land while this layer finishes and the next one runs its attention / recurrence; the rest of its plan is the
    // routed experts the bank does not hold. Shared by every sequence (one copy stream).
    struct PrefillStage {
        int K = 0; size_t half_bytes = 0; char * half[2] = {nullptr, nullptr};
        cudaStream_t copy = nullptr; cudaEvent_t ready[2] = {}, done[2] = {};
        int * idx_host = nullptr; size_t idx_cap = 0;
        int * ids_host = nullptr; size_t ids_cap = 0;
        const MixedExpertCache * cache = nullptr; int layer = -1; std::vector<int> plan; int issued = 0;  // copies queued ahead
        double copy_bytes = 0;
    };
    static PrefillStage & stage() { static PrefillStage st; return st; }
    std::unique_ptr<Device<float>> pf_inject, pf_weights, pf_sw, pf_sd; std::unique_ptr<Device<__half>> pf_xr, pf_ax;
    std::unique_ptr<Device<int>> pf_routed, pf_idx; std::unique_ptr<Device<__nv_bfloat16>> pf_eg, pf_eu, pf_ed;
    // TQFN_PREFILL_INT8: int8 activations for the layer-major expert GEMMs (per-32-block scales, per-16 sums)
    std::unique_ptr<Device<int8_t>> pf_xq, pf_axq; std::unique_ptr<Device<float>> pf_xs, pf_axs; std::unique_ptr<Device<int16_t>> pf_xsum, pf_axsum;
    std::vector<char> pf_borrowed;  // layers whose bank was evicted for this prefill: experts from staging, no bank created
    std::map<int, std::vector<int>> pf_refill;  // borrowed layer -> its most-routed experts, refilled in one go after the prompt
    template<class T> static void grow(std::unique_ptr<Device<T>> & d, size_t count) {
        if (!d || d->count < count) { PersistentAlloc keep; d.reset(); d = std::make_unique<Device<T>>(count); }
    }
    static size_t align256(size_t x) { return (x + 255) / 256 * 256; }
    // offsets of the six expert pieces (gate/up/down data + aux) in a staging bank of K experts of this layer
    static void stage_offsets(const MixedExpertCache::BankLayout & lay, int K, size_t off[6]) {
        size_t o = 0; for (int i = 0; i < 6; ++i) { off[i] = o; o += align256(size_t(K) * lay.piece[i]); }
    }
    // the layer's copy plan (sorted): experts its bank does not hold (all of them before the bank exists)
    static std::vector<int> stage_plan(const MixedExpertCache & cache, int layer);
    static void stage_alloc(PrefillStage & st, size_t need, int K);
    // copy group g of the plan into staging bank g & 1 once the compute stream is done with it (runs of consecutive ids:
    // one 2D copy per piece)
    static void stage_upload(PrefillStage & st, const MixedExpertCache & cache, int layer, int g);
    // start the copies of `layer` (plan + first two groups) unless they are already queued
    static void stage_begin(PrefillStage & st, const MixedExpertCache & cache, int layer);
    // Hands back what only layer-major prefill uses (its buffers, both staging banks, the pool's free blocks): the MTP
    // catch-up and decode that follow then have that memory, e.g. for their expert banks (plain cudaMalloc).
    void release_prefill();
    // One trunk layer for a whole prefill chunk (rows of `wide`, replaced in place): the dense part in sub-chunks of
    // --prefill-sub tokens (KV, recurrent and PLE state carried in order), then the routed experts once for every row:
    // the ones in the layer's cache bank run from it (nothing is evicted), the others from the staging banks.
    void prefill_layer(int layer, float * wide, int n, const float * ple_rows, MixedExpertCache & cache, AttentionState * att_override = nullptr);

    // The host side of a chunk: token embeddings and per-layer embeddings gathered from the mapped files (the PLE history is
    // appended here, so chunks must be prepared in prompt order). A multi-chunk prefill prepares the next chunk on a thread
    // while the current one runs on the GPU (step(tokens, &next)); the gathers were 0.25 s of GPU idle per 16K chunk.
    struct HostPrep { std::vector<int> tokens; std::vector<float> embedding, ple; size_t hist0 = 0; };
    HostPrep prepare_host(const std::vector<int> & tokens);
    std::future<HostPrep> prep_next_;
    bool keep_embeddings = false;      // a multi-chunk prefill keeps the prompt's embeddings for the MTP catch-up
    // [prompt tokens][hidden] fp32 while keep_embeddings; left uninitialised so each chunk's rows are first touched by the
    // copy thread, not the launching thread (150 MB of page faults per 16K chunk held the GPU idle ~0.1 s per chunk)
    struct HostRows { std::unique_ptr<float[]> p; size_t n = 0;
                      void alloc(size_t k) { p.reset(new float[k]); n = k; } void clear() { p.reset(); n = 0; }
                      size_t size() const { return n; } const float * data() const { return p.get(); } } prompt_emb;

    // Runs n new tokens through the trunk; returns the head-input rows [n][hidden]; keeps the wide rows in res_keep.
    // next: the tokens of the chunk that follows (its host prep starts now, on a thread).
    std::unique_ptr<Device<float>> step(const std::vector<int> & tokens, const std::vector<int> * next = nullptr);

    // One verify pass over several sequences (this model first): row-independent work (projections, router, experts)
    // runs once for all rows, PLE history and the attention / recurrent cores run per sequence. Returns the head-input
    // rows of every sequence, concatenated in the order given; each model keeps its wide rows in res_keep.
    std::unique_ptr<Device<float>> step_multi(const std::vector<DecodeModel *> & ms, const std::vector<std::vector<int>> & toks);

    // Keep the first m of the last step's n tokens (speculative verify): restore every carried state to token m-1.
    void accept(int m);

    // MTP block on n (token, wide residual) pairs: appends n MTP KV entries; returns the argmax after the last pair and
    // keeps its wide output in mtp_h (input of the next chained draft).
    int mtp_batch(const std::vector<int> & tokens, const float * h_rows);
    // The prompt's MTP keys and values without the rest of the MTP layer. The catch-up uses only the last position's MTP
    // output (the first draft), and the attention keys / values of a row depend on that row's input alone, so the other
    // rows need no attention, experts or head. Rows go in 2048-row pieces; the caller runs mtp_batch on the last position.
    // Changes drafts at most (the GEMMs may round differently), never the output.
    void mtp_kv_prefill(const std::vector<int> & tokens, const float * h_rows, const float * emb);
    // mtp_batch for a long run of prompt positions: the MTP layer runs layer-major (routed experts staged from the pinned
    // MTP copy once per call, grouped GEMMs) instead of in batches small enough for the MTP cache. Same positions, same
    // order; the GEMM path may round differently, which can change drafts but never the output.
    // emb: the tokens' embeddings already on the host ([n][hidden] fp32, from the trunk's prep) instead of a gather
    int mtp_prefill(const std::vector<int> & tokens, const float * h_rows, const float * emb = nullptr);
    // Chained draft from the MTP's own last output; its KV entry is dropped (the next catch-up recomputes it).
    int mtp_chain(int token);
    // mtp_batch over several sequences (this model first): one MTP-layer pass, per-sequence MTP KV and mtp_h. Returns each
    // sequence's argmax after its last pair. Sequences are grouped so the routed experts of a group fit the MTP cache.
    std::vector<int> mtp_batch_multi(const std::vector<DecodeModel *> & ms, const std::vector<std::vector<int>> & toks,
                                     const std::vector<const float *> & hs);
    // Drafts d2..dD from d1 for several sequences at once (same chain as mtp_drafts, one MTP pass per chain step).
    std::vector<std::vector<int>> mtp_drafts_multi(const std::vector<DecodeModel *> & ms, const std::vector<int> & d1, int count);

    // Drafts d2..dD from d1: each chain step sees the previous drafts' MTP KV; all chain entries are dropped at the end.
    std::vector<int> mtp_drafts(int d1, int count);

    // Router logits [n][experts + 1] (last = shared-expert gate). BF16 router weights: per-token GEMVs on the raw weights.
    void router(const std::string & prefix, const float * x, float * logits, int n);
    // stream-ordered on the thread's default stream (capturable; a device-to-device copy never waited for the host anyway)
    static void ck_zero(Device<float> & d) { ck(cudaMemsetAsync(d.p, 0, d.count * sizeof(float), nullptr)); }
    static void ck_copy(Device<float> & dst, const Device<float> & src) { ck(cudaMemcpyAsync(dst.p, src.p, src.count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr)); }
    static void ck_copy_n(Device<float> & dst, const float * src, size_t count) { ck(cudaMemcpyAsync(dst.p, src, count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr)); }
};

// The packed expert file (TQFNEXP1 / TQFNEXPT), its reader and the per-expert repack are shared components now
// (src/experts/packed_experts.hpp): GLM-5.3-Flash writes and reads the same file. These are the Checkpoint-shaped
// wrappers the Flash-Next engine and tqfn-stage call.
inline std::array<int, 3> expert_u4h_ids(const Checkpoint & checkpoint, int layer) {
    return tqllm::expert_u4h_ids(checkpoint.expert_payload(layer, 0), checkpoint.config.hidden, checkpoint.config.expert_ff);
}
inline std::array<std::vector<char>, 6> repack_expert(const Checkpoint & checkpoint, int layer, int id) {
    return tqllm::repack_expert(checkpoint.expert_payload(layer, id), checkpoint.config.hidden, checkpoint.config.expert_ff,
                                g_tile_experts);
}
// The packed experts built in pinned host memory from the GGUF: the bytes tqfn-pack writes (untiled), without the file
inline PackedExperts build_packed(const Checkpoint & checkpoint, int threads) {
    const auto & c = checkpoint.config; PackedExperts pe; pe.layers = c.layers; pe.experts = c.experts; pe.sizes.resize(size_t(c.layers));
    g_tile_experts = false;  // untiled, as in the file: the loader tiles both the same way
    uint64_t off = 0;
    for (int l = 0; l < c.layers; ++l) {
        const auto parts = repack_expert(checkpoint, l, 0); uint64_t b = 0;
        for (int k = 0; k < 6; ++k) { pe.sizes[size_t(l)][size_t(k)] = parts[size_t(k)].size(); b += parts[size_t(k)].size(); }
        pe.base.push_back(off); pe.block.push_back(b); off += b * uint64_t(c.experts);
    }
    pe.bytes = size_t(off); char * host = nullptr; ck(cudaMallocHost(reinterpret_cast<void **>(&host), pe.bytes));
    std::atomic<int64_t> next{0}; const int64_t total = int64_t(c.layers) * c.experts; std::vector<std::thread> pool;
    for (int t = 0; t < std::max(1, threads); ++t) pool.emplace_back([&] {
        for (int64_t i; (i = next++) < total;) {
            const int l = int(i / c.experts), e = int(i % c.experts);
            const auto parts = repack_expert(checkpoint, l, e); char * d = host + pe.base[size_t(l)] + uint64_t(e) * pe.block[size_t(l)];
            for (int k = 0; k < 6; ++k) {
                require(parts[size_t(k)].size() == pe.sizes[size_t(l)][size_t(k)], "expert layout varies within a layer");
                std::memcpy(d, parts[size_t(k)].data(), parts[size_t(k)].size()); d += parts[size_t(k)].size();
            }
        }
    });
    for (auto & t : pool) t.join();
    pe.map = host; pe.tiled = false; return pe;
}
