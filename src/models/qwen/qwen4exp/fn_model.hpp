#pragma once
#include <chrono>
// Flash-Next model: DecodeModel (decode, layer-major prefill, MTP drafting) and the packed expert file.
#include "fn_layers.hpp"
#include "expert_parallel.hpp"
#include "decode_rules.hpp"
#include "fn_cpu_experts.hpp"
#include "../../../core/tp_comm.hpp"
#include "cpu_pool.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <set>
#include <thread>
struct PinnedInts { int * p = nullptr; size_t n = 0; void need(size_t k) { if (k > n) { if (p) cudaFreeHost(p); ck(cudaMallocHost(reinterpret_cast<void **>(&p), k * sizeof(int))); n = k; } } };
inline double g_host_ms[3];  // decode host time blocked in: routed ids D2H, cache prepare, prefetch issue
void release_decode_graphs(uint64_t owner);  // fn_layer_forward.cpp: a finished model's decode graphs
size_t decode_graph_instantiations();         // fn_layer_forward.cpp: graph execs built so far (TQLLM_DEBUG=steptrace)
struct DecodeModel {
    const Checkpoint & cp; const Fixtures & fx; const Qwen4ExpConfig & c; int h, width;
    MixedExpertCache & experts; MixedExpertCache * mtp_experts = nullptr;
    std::vector<RecurrentState> rec; std::vector<AttentionState> att; std::vector<std::unique_ptr<Device<float>>> ple_past;
    PleTokenHistory history; int position = 0;
    bool prefetch_on = !tqllm::debug_on("no_guess"); cudaStream_t copy_stream = nullptr, miss_stream = nullptr;  // TQLLM_DEBUG=no_guess: no next-layer guess copies (A/B)
    // speculative decoding: per-token rollback state, final wide residual rows of the last step, MTP head state
    bool spec = false; int spec_pos0 = 0, spec_n = 0; size_t spec_hist0 = 0;
    std::vector<std::unique_ptr<Device<float>>> ple_saved, ple_old;
    std::unique_ptr<Device<float>> pingpong[2];  // prefill residual between layers (layer scratch is reset)
    std::unique_ptr<Device<float>> res_keep;
    std::unique_ptr<Device<__nv_bfloat16>> res_keep_b;  // the layer-major prefill's BF16 wide rows (prefill_wide_bf16)  // [n][width] of the last step, persistent (grows with the prefill chunk)
    AttentionState mtp_att; std::unique_ptr<Device<float>> mtp_h;  // MTP block KV and its last wide output
    float draft_conf = 1.f;  // top probability of the last MTP draft (FnConfig::draft_tau)
    // After a long prompt: the trunk's K/V pages wholly below len - keep move to pinned host memory (same addresses);
    // returns the device bytes freed. TQFN_KV_HOST_KEEP: positions kept on the device (32768); TQFN_KV_HOST=0: off.
    size_t kv_to_host() {
        static const int keep = std::getenv("TQFN_KV_HOST_KEEP") ? std::atoi(std::getenv("TQFN_KV_HOST_KEEP")) : 32768;
        size_t freed = 0;
        for (size_t l = 0; l <= att.size(); ++l) {  // the trunk's attention layers, then the MTP layer (index L, its own gather)
            auto & a = l < att.size() ? att[l] : mtp_att;
            if (!a.k || !a.k->vmm || a.len <= 2 * keep) continue;
            const size_t row = a.k->bytes / size_t(kv_cap(a, cp.rt.kv_capacity));
            const size_t cut = std::min(a.k->on_host.size(), tqllm::flashnext::kv_host_front(size_t(a.len), size_t(keep), a.k->page_rows(row)));
            for (size_t i = 0; i < cut; ++i) for (KvBuf * b : {a.k.get(), a.v.get()})
                if (!b->on_host[i] && b->move(i, true)) freed += b->page;
            if (a.k->any_host()) make_host_gather(cp, int(l));  // decode reads these rows through it: made before the bank refill
        }
        return freed;
    }
    // Before a long prefill writes and re-reads its context: every host page back on the device (with its data, for a
    // reused prefix); returns the bytes moved. Pages that do not fit stay on the host (read over PCIe).
    size_t kv_to_device() {
        size_t moved = 0;
        for (size_t l = 0; l <= att.size(); ++l)
            if (auto & a = l < att.size() ? att[l] : mtp_att; a.k && a.k->vmm)
            for (KvBuf * b : {a.k.get(), a.v.get()}) for (size_t i = 0; i < b->on_host.size(); ++i)
                if (b->on_host[i] && b->move(i, false)) moved += b->page;
        return moved;
    }
    DecodeModel(const Checkpoint & k, const Fixtures & f, MixedExpertCache & e, size_t capacity)
        : cp(k), fx(f), c(k.config), h(c.hidden), width(c.hidden * c.hc), experts(e), rec(c.layers), att(c.layers),
          ple_past(c.layers), history(c.ple, capacity), ple_saved(c.layers), ple_old(c.layers) {
        ck(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        ck(cudaStreamCreateWithFlags(&miss_stream, cudaStreamNonBlocking)); experts.set_miss_stream(miss_stream);
        // TQLLM_DEBUG=graphs: models built at a finished model's address (address-keyed graphs were inherited there before)
        if (tqllm::debug_on("graphs") && finished_models().count(this)) {
            static int said = 0; if (said++ < 8) std::fprintf(stderr, "tqllm: DecodeModel %p reuses a finished model's address (owner %llu)\n",
                                                          static_cast<void *>(this), (unsigned long long) graph_owner);
        }
        PersistentAlloc keep; res_keep = std::make_unique<Device<float>>(size_t(64) * width); mtp_h = std::make_unique<Device<float>>(width);
        // K/V and QSA state sized by this sequence's capacity (its request, or the slot context with the prompt cache), not
        // the slot context of every sequence: np 2 on a 3090 mapped full-context K/V for each slot (11-03)
        kv_capacity = int((capacity + 1023) / 1024 * 1024);
        for (auto & a : att) a.cap = kv_capacity;
        mtp_att.cap = kv_capacity;
        // the decode norm / RoPE position tables cover this sequence (attention and indexer rotary), built here outside any capture
        tqllm_cuda::fn_rope_positions(int64_t(kv_capacity) + 64, k.rotary, k.theta, nullptr);
        tqllm_cuda::fn_rope_positions(int64_t(kv_capacity) + 64, std::min(k.rotary, 128), k.theta, nullptr);
    }
    int kv_capacity = 0;
    // DFlash2 feature capture (null: off): after each layer in cap_layers, every row's stream mean goes to
    // cap_out + row * cap_stride + (index in cap_layers) * hidden (BF16), for the drafter's inject
    const std::vector<int> * cap_layers = nullptr; __nv_bfloat16 * cap_out = nullptr; int64_t cap_stride = 0;
    void capture(int layer, const float * res, int n);
    void capture(int layer, const __nv_bfloat16 * res, int n);
    // decode graphs are this model's own (decode_rules.hpp GraphWarmup): a model built at a finished one's address starts afresh
    const uint64_t graph_owner = tqllm::flashnext::next_graph_owner();
    static std::set<const void *> & finished_models() { static std::set<const void *> s; return s; }
    ~DecodeModel() { release_decode_graphs(graph_owner); if (tqllm::debug_on("graphs")) finished_models().insert(this);
                     if (copy_stream) { cudaStreamSynchronize(copy_stream); cudaStreamDestroy(copy_stream); }
                     if (miss_stream) { experts.release_miss_stream(miss_stream); if (mtp_experts) mtp_experts->release_miss_stream(miss_stream);
                                        cudaStreamSynchronize(miss_stream); cudaStreamDestroy(miss_stream); } }
    void set_spec(bool on) { spec = on; for (auto & r : rec) r.spec = on; }

    // One block (trunk layer 0..L-1, or the MTP block L) on n tokens; residual [n][width] is replaced.
    // a batched step over several sequences: rows [off, off + n) belong to model m (its PLE history, KV, recurrent state)
    struct Seg { DecodeModel * m; int off, n; };
    // Layer-major prefill: the dense part of a layer runs on sub-chunks and stops before the routed experts, leaving its
    // rows at `off` of these whole-chunk buffers (combined residual in place of the input rows, FFN inject, FP16 FFN
    // input, routing, shared expert output); the experts then run once for the whole chunk (prefill_layer)
    // wide_b: the wide rows in BF16 storage (wide null then; layer_forward_step)
    struct PrefillSink { float * wide; float * inject; __half * xr; int * routed; float * weights, * shared_weights, * sd; int off;
                         __nv_bfloat16 * wide_b = nullptr; };
    int hc_path(const std::string & prefix, int n) const;
    bool prefill_wide_bf16(int sub) const;
    // --split layer: every rank must run the same forwards (same rows); a rank whose host decisions differed (draft count,
// prefill chunk) would wait for a hop that never comes, so a mismatch fails the step with the reason instead
// TQLLM_DEBUG=split: every collective of the split in order (to find where two ranks' sequences part)
static void split_trace(const char * op, uint64_t v) {
    static const bool on = tqllm::debug_on("split"); static uint64_t seq = 0;
    if (on) { std::fprintf(stderr, "SPLIT %llu %s %llu\n", (unsigned long long) seq++, op, (unsigned long long) v); std::fflush(stderr); }
}
// Pipelined prefill (head_only, chunks of kPipeRows+ rows, no DFlash2 capture): a rank sends its chunk's trunk rows from a
// staging copy on this second stream, so it starts the next chunk while the link and the next rank work on this one (2x 3090
// on 1 GbE: rank 0 waited 65 % of a 32K prefill). Every other split collective first waits for the pending send, so the
// ranks' NCCL calls never overlap.
static constexpr int kPipeRows = 256;
// `done`: after the last op on the stream; `sent`: after the last send (the staging copy waits for it); `ready`: the compute
// stream's work the next stream op must follow
struct SplitComm { cudaStream_t stream = nullptr; cudaEvent_t ready = nullptr, sent = nullptr, done = nullptr; bool pending = false, sending = false;
                   std::unique_ptr<Device<float>> staging; };
static SplitComm & split_comm() {
    static SplitComm c;
    if (!c.stream) {
        ck(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
        for (cudaEvent_t * e : {&c.ready, &c.sent, &c.done}) ck(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    }
    return c;
}
// the compute stream waits for every op on the split stream (before any NCCL call on the compute stream)
static void split_join() {
    auto & c = split_comm();
    if (c.pending) { ck(cudaStreamWaitEvent(nullptr, c.done, 0)); c.pending = false; }
}
// after a pipelined prefill: its staging buffer goes back to the banks (2x 3090, 128K: 316 MB held through decode)
static void split_release_staging() {
    auto & c = split_comm();
    if (!c.staging) return;
    split_join(); ck(cudaStreamSynchronize(c.stream)); c.sending = false;
    c.staging.reset();
}
static void split_check(tqllm::TpComm & tp, uint64_t v) {
    split_join();
    split_trace("check", v);
    static uint64_t * d = nullptr; if (!d) ck(cudaMalloc(&d, sizeof(uint64_t) * (1 + 8)));
    ck(cudaMemcpy(d, &v, sizeof v, cudaMemcpyHostToDevice));
    tp.all_gather(d, d + 1, sizeof v, nullptr);
    uint64_t all[8] = {}; ck(cudaMemcpy(all, d + 1, sizeof(uint64_t) * size_t(tp.world()), cudaMemcpyDeviceToHost));
    for (int i = 0; i < tp.world(); ++i)
        if (all[i] != v) throw std::runtime_error("--split layer: the ranks diverged (forward of " + std::to_string(v) + " bytes here, " +
                                                  std::to_string(all[i]) + " on rank " + std::to_string(i) + ")");
}
// the smallest value over the ranks, on every rank
static void split_min(tqllm::TpComm & tp, int * v) {
    split_join();
    split_trace("min", uint64_t(*v));
    static int * d = nullptr; if (!d) ck(cudaMalloc(&d, sizeof(int) * 9));
    ck(cudaMemcpy(d, v, sizeof(int), cudaMemcpyHostToDevice));
    tp.all_gather(d, d + 1, sizeof(int), nullptr);
    int all[8] = {}; ck(cudaMemcpy(all, d + 1, sizeof(int) * size_t(tp.world()), cudaMemcpyDeviceToHost));
    for (int i = 0; i < tp.world(); ++i) *v = std::min(*v, all[i]);
}
// the last rank's values on every rank (head_only: its draft block, picked tokens); n <= 64
static void split_from_last(tqllm::TpComm & tp, int * v, int n) {
    split_join();
    split_trace("from_last", uint64_t(v[0]));
    static int * d = nullptr; if (!d) ck(cudaMalloc(&d, sizeof(int) * 64));
    ck(cudaMemcpy(d, v, sizeof(int) * size_t(n), cudaMemcpyHostToDevice));
    tp.broadcast(d, sizeof(int) * size_t(n), tp.world() - 1, nullptr);
    ck(cudaMemcpy(v, d, sizeof(int) * size_t(n), cudaMemcpyDeviceToHost));
}
// rank 0's values on every rank (decisions that depend on each card's free memory, e.g. the prefill chunk)
static void split_agree(tqllm::TpComm & tp, int * v, int n) {
    split_join();
    split_trace("agree", uint64_t(v[0]));
    static int * d = nullptr; if (!d) ck(cudaMalloc(&d, sizeof(int) * 16));
    ck(cudaMemcpy(d, v, sizeof(int) * size_t(n), cudaMemcpyHostToDevice));
    tp.broadcast(d, sizeof(int) * size_t(n), 0, nullptr);
    ck(cudaMemcpy(v, d, sizeof(int) * size_t(n), cudaMemcpyDeviceToHost));
}
// The trunk layers of one forward: all of them, or (--split layer) this rank's [lo, hi) between the boundary hops of the
    // residual rows (`rows()` returns the buffer the next layer reads, `bytes` its size); every rank ends with the trunk output
    // (head_only: the last rank; the ops are tqllm::trunk_ops, tests/layer_split.cpp)
    template <class Layer, class Rows> void trunk_layers(Layer && layer, Rows && rows, size_t bytes, int nrows) {
        const auto & r = cp.run;
        if (!r.split) { for (int l = 0; l < c.layers; ++l) layer(l); return; }
        tqllm::TpComm & tp = *r.split;
        const bool to_all = !r.head_only || cp.rt.rows_to_all;
        // every rank takes the same branch: the same rows, flags and capture on all of them
        const bool pipelined = !to_all && nrows >= kPipeRows && !(cap_layers && cap_out);
        if (!pipelined) split_check(tp, uint64_t(bytes));
        auto & sc = split_comm();
        if (pipelined) {  // the split stream's ops follow what the compute stream did so far (the rows buffer's last writes)
            ck(cudaEventRecord(sc.ready, nullptr)); ck(cudaStreamWaitEvent(sc.stream, sc.ready, 0));
        }
        static const std::vector<int> no_features;
        const auto ops = tqllm::trunk_ops(r.ranges, tp.rank(), to_all, cap_layers && cap_out ? *cap_layers : no_features);
        // TQLLM_DEBUG=splittime: per forward of 256+ rows (prefill chunks), this rank's wait for the rows from the rank
        // before it, its own layers, and the wait for the last rank's output (syncs the device: a profile, not a run)
        static const bool timed = tqllm::debug_on("splittime");
        const bool tm = timed && nrows >= 256;
        using clk = std::chrono::steady_clock;
        clk::time_point t0, t1, t2;
        if (tm) { ck(cudaDeviceSynchronize()); t0 = t1 = clk::now(); }
        // DFlash2 feature rows: [nrows][h] of a target layer, strided in cap_out, staged contiguous for the link
        static std::unique_ptr<Device<__nv_bfloat16>> tmp;
        const size_t need = size_t(nrows) * size_t(h), fbytes = need * sizeof(__nv_bfloat16);
        const size_t pitch = size_t(cap_stride) * sizeof(__nv_bfloat16), frow = size_t(h) * sizeof(__nv_bfloat16);
        auto feat_slot = [&](int i) { return cap_out + size_t(i) * size_t(h); };
        auto stage_out = [&](int i) { ck(cudaMemcpy2DAsync(tmp->p, frow, feat_slot(i), pitch, frow, size_t(nrows), cudaMemcpyDeviceToDevice, nullptr)); };
        auto stage_in = [&](int i) { ck(cudaMemcpy2DAsync(feat_slot(i), pitch, tmp->p, frow, frow, size_t(nrows), cudaMemcpyDeviceToDevice, nullptr)); };
        for (const auto & op : ops) {
            if ((op.kind == tqllm::TrunkOp::send_feat || op.kind == tqllm::TrunkOp::recv_feat || op.kind == tqllm::TrunkOp::bcast_feat) &&
                (!tmp || tmp->count < need)) { PersistentAlloc keep; tmp = std::make_unique<Device<__nv_bfloat16>>(need); }
            switch (op.kind) {
            case tqllm::TrunkOp::recv_rows:
                if (!pipelined) { tp.recv(rows(), bytes, op.peer, nullptr); break; }
                tp.recv(rows(), bytes, op.peer, sc.stream);
                ck(cudaEventRecord(sc.done, sc.stream)); sc.pending = true;
                ck(cudaStreamWaitEvent(nullptr, sc.done, 0));
                break;
            case tqllm::TrunkOp::layers:
                if (tm) { ck(cudaDeviceSynchronize()); t1 = clk::now(); }
                for (uint32_t l = r.lr.lo; l < r.lr.hi; ++l) layer(int(l));
                if (tm) { ck(cudaDeviceSynchronize()); t2 = clk::now(); }
                break;
            case tqllm::TrunkOp::send_rows: {
                const size_t floats = (bytes + 3) / 4;
                if (pipelined && (!sc.staging || sc.staging->count < floats)) {
                    try { PersistentAlloc keep; sc.staging.reset(); sc.staging = std::make_unique<Device<float>>(floats); }
                    catch (const std::exception &) { cudaGetLastError(); sc.staging.reset(); }
                }
                if (!pipelined || !sc.staging) { split_join(); tp.send(rows(), bytes, op.peer, nullptr); break; }
                if (sc.sending) ck(cudaStreamWaitEvent(nullptr, sc.sent, 0));  // the previous chunk left the staging buffer
                ck(cudaMemcpyAsync(sc.staging->p, rows(), bytes, cudaMemcpyDeviceToDevice, nullptr));
                ck(cudaEventRecord(sc.ready, nullptr));
                ck(cudaStreamWaitEvent(sc.stream, sc.ready, 0));
                tp.send(sc.staging->p, bytes, op.peer, sc.stream);
                ck(cudaEventRecord(sc.sent, sc.stream)); ck(cudaEventRecord(sc.done, sc.stream));
                sc.pending = sc.sending = true;
                break;
            }
            case tqllm::TrunkOp::bcast_rows: tp.broadcast(rows(), bytes, op.peer, nullptr); break;
            case tqllm::TrunkOp::send_feat: stage_out(op.feat); tp.send(tmp->p, fbytes, op.peer, nullptr); break;
            case tqllm::TrunkOp::recv_feat: tp.recv(tmp->p, fbytes, op.peer, nullptr); stage_in(op.feat); break;
            case tqllm::TrunkOp::bcast_feat:
                if (op.peer == tp.rank()) stage_out(op.feat);
                tp.broadcast(tmp->p, fbytes, op.peer, nullptr);
                if (op.peer != tp.rank()) stage_in(op.feat);
                break;
            }
        }
        if (tm) {
            ck(cudaDeviceSynchronize());
            const auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            std::fprintf(stderr, "SPLITTIME rank %d rows %d wait_in %.1f layers %.1f wait_out %.1f ms\n", tp.rank(), nrows, ms(t0, t1), ms(t1, t2),
                         ms(t2, clk::now()));
        }
    }
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
        // experts read from the GGUF on demand: the routed experts the host tier lacks are read into this pinned ring (one
        // half per staging half), so a prompt's stream never evicts the tier
        char * ring[2] = {nullptr, nullptr}; size_t ring_bytes = 0;
        int half0 = 0;  // layer-outer prefill: one group per layer, the layers alternate halves (layer & 1)
        // layer-outer prefill: the next layer's experts the tier lacks are read into its ring half on a thread while this
        // layer computes (pre_read[e]: expert e is in the ring)
        std::future<void> pre; int pre_layer = -1; std::vector<char> pre_read;
        // layer-outer (step_outer): ring half layer & 1 holds every streamed expert of ring_layer read so far (ring_have, by
        // expert id), so later chunks of the layer upload them again from RAM instead of the drive
        bool outer = false; int ring_layer = -1; std::vector<char> ring_have; cudaEvent_t ring_ev[2] = {};
        char * bounce[2] = {nullptr, nullptr}; size_t bounce_bytes = 0;  // pinned host blocks per half: experts read on demand (rt.expert_blocks)
        const std::function<void(int, const int *, size_t, char *, size_t)> * blocks = nullptr;  // rt.expert_blocks of the running prefill
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
    static void stage_upload(PrefillStage & st, MixedExpertCache & cache, int layer, int g);
    // start the copies of `layer` (plan + first two groups) unless they are already queued
    static void stage_begin(PrefillStage & st, MixedExpertCache & cache, int layer);
    // layer-outer prefill with a host tier: start reading `layer`'s missing experts into its ring half now
    static void stage_prefetch(PrefillStage & st, MixedExpertCache & cache, int layer);
    // layer-outer, a chunk after the layer's first: its groups staged again (from the tier and the ring) when the layer
    // has more than one group
    static void stage_restart(PrefillStage & st, MixedExpertCache & cache, int layer);
    // Hands back what only layer-major prefill uses (its buffers, both staging banks, the pool's free blocks): the MTP
    // catch-up and decode that follow then have that memory, e.g. for their expert banks (plain cudaMalloc).
    void release_prefill();
    // One trunk layer for a whole prefill chunk (rows of `wide`, replaced in place): the dense part in sub-chunks of
    // --prefill-sub tokens (KV, recurrent and PLE state carried in order), then the routed experts once for every row:
    // the ones in the layer's cache bank run from it (nothing is evicted), the others from the staging banks.
    // wide_b: BF16 wide rows instead of `wide` (null then)
    void prefill_layer(int layer, float * wide, int n, const float * ple_rows, MixedExpertCache & cache, AttentionState * att_override = nullptr,
                       __nv_bfloat16 * wide_b = nullptr);

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
    // Layer-outer prefill of consecutive chunks (each of 256+ tokens, in prompt order; experts read from the GGUF on
    // demand): every layer runs over all chunks before the next layer starts, so a layer's streamed experts are read once
    // per prompt instead of once per chunk (a 16K chunk streamed ~24 GB from a SATA drive). The same (layer, chunk) calls
    // as step() per chunk, with the same rows, positions and carried state: the same output. Each chunk's wide rows stay
    // on the device (BF16 when the layers take them) between layers. done(c, head_rows): after the last layer, chunk c's
    // final rows are in res_keep and head_rows its [n][hidden] head input, in chunk order.
    void step_outer(const std::vector<std::vector<int>> & chunks, const std::function<void(size_t, const Device<float> &)> & done);
    bool pf_hold_next = false;  // step_outer: a layer's staged experts stay for the next chunk (no next-layer prefetch yet)

    // One verify pass over several sequences (this model first): row-independent work (projections, router, experts)
    // runs once for all rows, PLE history and the attention / recurrent cores run per sequence. Returns the head-input
    // rows of every sequence, concatenated in the order given; each model keeps its wide rows in res_keep.
    std::unique_ptr<Device<float>> step_multi(const std::vector<DecodeModel *> & ms, const std::vector<std::vector<int>> & toks);

    // Keep the first m of the last step's n tokens (speculative verify): restore every carried state to token m-1.
    void accept(int m);
    // accept for several sequences of one batched verify: every row copy in one launch and, per GDN layer, one replay for
    // all sequences that rolled back (the same bits as accept per sequence)
    static void accept_multi(const std::vector<std::pair<DecodeModel *, int>> & accepts);

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
// --split experts: Flash-Next's expert shard for the model-independent ExpertParallel (src/experts/expert_parallel.hpp)
tqllm::ExpertShard flashnext_expert_shard(const Checkpoint & cp, MixedExpertCache & cache);

// The packed expert file (TQFNEXP1 / TQFNEXPT), its reader and the per-expert repack are shared components now
// (src/experts/packed_experts.hpp): GLM-5.3-Flash writes and reads the same file. These are the Checkpoint-shaped
// wrappers the Flash-Next engine and tqfn-stage call.
inline std::array<int, 3> expert_u4h_ids(const Checkpoint & checkpoint, int layer) {
    return tqllm::expert_u4h_ids(checkpoint.expert_payload(layer, 0), checkpoint.config.hidden, checkpoint.config.expert_ff);
}
inline std::array<int, 3> expert_tile_bpbs(const Checkpoint & checkpoint, int layer) {
    return tqllm::expert_tile_bpbs(checkpoint.expert_payload(layer, 0), checkpoint.config.hidden, checkpoint.config.expert_ff);
}
inline std::array<std::vector<char>, 6> repack_expert(const Checkpoint & checkpoint, int layer, int id, bool tiled) {
    return tqllm::repack_expert(checkpoint.expert_payload(layer, id), checkpoint.config.hidden, checkpoint.config.expert_ff,
                                tiled);
}
// The packed experts built in pinned host memory from the GGUF: the bytes tqfn-pack writes (untiled), without the file.
// Only layers [lo, hi) are held (--split layer: this rank's trunk layers; each rank held all 48 layers, 2 x 51 GB on a
// 124 GB 2x RTX 4090 host, and the per-layer token embedding pages were pushed out: decode 15-73 tok/s, a rank OOM-killed).
inline PackedExperts build_packed(const Checkpoint & checkpoint, int threads, int lo = 0, int hi = 1 << 30) {
    const auto & c = checkpoint.config; PackedExperts pe; pe.layers = c.layers; pe.experts = c.experts; pe.sizes.resize(size_t(c.layers));
    pe.lo = std::max(0, lo); pe.hi = std::min(c.layers, hi);
    uint64_t off = 0;
    for (int l = 0; l < c.layers; ++l) {
        const auto parts = repack_expert(checkpoint, l, 0, false); uint64_t b = 0;
        for (int k = 0; k < 6; ++k) { pe.sizes[size_t(l)][size_t(k)] = parts[size_t(k)].size(); b += parts[size_t(k)].size(); }
        pe.base.push_back(off); pe.block.push_back(b); off += b * uint64_t(c.experts);
    }
    const auto span = packed_span(pe.base, pe.block, c.experts, pe.lo, pe.hi);
    pe.bytes = size_t(span.second - span.first); char * host = nullptr; ck(cudaMallocHost(reinterpret_cast<void **>(&host), pe.bytes));
    char * origin = host - span.first;  // map + base[l] for the held layers
    std::atomic<int64_t> next{0}; const int64_t total = int64_t(pe.hi - pe.lo) * c.experts; std::vector<std::thread> pool;
    for (int t = 0; t < std::max(1, threads); ++t) pool.emplace_back([&] {
        for (int64_t i; (i = next++) < total;) {
            const int l = pe.lo + int(i / c.experts), e = int(i % c.experts);
            const auto parts = repack_expert(checkpoint, l, e, false); char * d = origin + pe.base[size_t(l)] + uint64_t(e) * pe.block[size_t(l)];
            for (int k = 0; k < 6; ++k) {
                require(parts[size_t(k)].size() == pe.sizes[size_t(l)][size_t(k)], "expert layout varies within a layer");
                std::memcpy(d, parts[size_t(k)].data(), parts[size_t(k)].size()); d += parts[size_t(k)].size();
            }
        }
    });
    for (auto & t : pool) t.join();
    pe.map = origin; pe.tiled = false; return pe;
}
