#include <chrono>
// Flash-Next layer forward (decode, verify and short prefill chunks) and the router.
#include "debug.hpp"
#include "fn_model.hpp"
#include "decode_rules.hpp"
#include <atomic>
#include "fn_graphs.hpp"
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
#include "hot_cold_policy.hpp"
#include "core/tp_comm.hpp"
#include "experts/moe_kernels.cuh"

namespace {
// --split experts, rank 0: every trunk layer of a decode forward that takes the full_slots path exchanges with rank 1. The
// forward is announced on the control channel at layer 0 (rows n); rank 1 then runs exactly c.layers exchanges
// or declines it (decode_rules.hpp ep_before_layer / ep_after_layer keep the two ranks in step or turn it off).
tqllm::flashnext::EpForwardState g_ep_fwd;
// Control channel (TCP, ordered). Rank 1 grants a lease while it has no request of its own (kEpLease); rank 0 then sends
// each decode forward (kEpForward n) without waiting for an answer (the link's round trip is 0.5-1.6 ms, ~3 % of a 1-user
// step). Rank 1 takes the lease back for a request of its own (kEpRevoke) and waits for kEpLeaseEnd: rank 0 sends it in
// order after its last forward, so rank 1 has finished every forward sent to it before it serves.
enum : int32_t { kEpForward = 1, kEpStop = 2, kEpLeaseEnd = 3, kEpLease = 21, kEpRevoke = 22 };
// rank 0: the lease, set by a reader thread (it answers a revoke even while this box is idle); sends go out under `m`
struct EpLease { std::mutex m; bool leased = false; };
EpLease g_ep_lease;
// a layer's expert formats have the dedup kernels the split launch uses (rank 1 checks every layer at start and refuses
// otherwise, so on rank 0 a full bank's own layer is enough)
bool ep_formats_ok(const MixedExpertCache & cache, int layer) {
    for (int p = 0; p < 3; ++p) if (!cache.bank_mat(layer, p).fmt->has_expert_dedup()) return false;
    return true;
}
}

namespace { tqllm::flashnext::DecodeGraphs g_decode_graphs; }  // fn_graphs.hpp
void release_decode_graphs(uint64_t owner) { g_decode_graphs.release(owner); }

// A step whose captured part fails (an operation not allowed in a capture: a weight uploaded on its first use, a handle
// created) runs again without a graph; nothing in an open capture has executed, so the plain run starts from the same
// state. Its layer key then takes two plain calls again before the next capture.
void DecodeModel::layer_forward(int layer, std::unique_ptr<Device<float>> & residual, int n, const float * ple_rows,
                   AttentionState * attn, MixedExpertCache & cache, const std::vector<Seg> * segs,
                   const PrefillSink * sink) {
    try { layer_forward_step(layer, residual, n, ple_rows, attn, cache, segs, sink, false); }
    catch (const std::exception & e) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) != cudaSuccess || cs == cudaStreamCaptureStatusNone) throw;
        cudaGraph_t cg = nullptr; (void) cudaStreamEndCapture(cudaStreamPerThread, &cg);
        if (cg) (void) cudaGraphDestroy(cg);
        (void) cudaGetLastError();
        static int logged = 0;
        if (logged++ < 4) std::fprintf(stderr, "tqllm: layer %d graph capture failed (%s); the step runs without a graph\n", layer, e.what());
        layer_forward_step(layer, residual, n, ple_rows, attn, cache, segs, sink, true);
    }
}

// the layer's hyper-connection path for a block of n rows: 2 row scales + FP16 image (mixers through FP16 weight copies,
// hc_row_scales_h), 1 row scales + BF16 image (BF16 mixers), 0 the plain normalized rows
int DecodeModel::hc_path(const std::string & prefix, int n) const {
    bool copies = true;
    for (const char * w : {"hc_attn_down.weight", "hc_ffn_down.weight", "hc_attn_up.weight", "hc_ffn_up.weight", "hc_attn_inject.weight", "hc_ffn_inject.weight"})
        copies = copies && cp.half_copy_ok(prefix + w);
    if (tqllm::hc_row_scales_h(n, copies, cp.resident && !cp.cpu_q8_projection && !exact_f32() && !bf16_act() &&
                                             !Checkpoint::dense_int8_mode() && !cp.f32_q6_projection)) return 2;
    return tqllm::hc_row_scales_only(n, cp.is_bf16(prefix + "hc_attn_down.weight") && cp.is_bf16(prefix + "hc_ffn_down.weight"),
                                     cp.is_bf16(prefix + "hc_attn_inject.weight") && cp.is_bf16(prefix + "hc_ffn_inject.weight")) ? 1 : 0;
}
// the layer-major prefill keeps its wide rows in BF16 when every trunk layer's sub-chunks take the FP16 row-scale path
// (the kernels of that path read and write BF16 rows; fp32 math in registers)
bool DecodeModel::prefill_wide_bf16(int sub) const {
    for (int l = 0; l < c.layers; ++l) if (hc_path("blk." + std::to_string(l) + ".", sub) != 2) return false;
    return true;
}
void DecodeModel::layer_forward_step(int layer, std::unique_ptr<Device<float>> & residual, int n, const float * ple_rows,
                   AttentionState * attn, MixedExpertCache & cache, const std::vector<Seg> * segs,
                   const PrefillSink * sink, bool plain) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    // The part of a decode / verify step before the routing readback (PLE, norms, mixer, attention or GDN core, routers,
    // top-k) is captured into a CUDA graph each step, updated in place and launched (layer_graphs: unified memory and PCIe
    // cards before Blackwell; text identical either way); with every expert resident the whole layer, for one or several sequences.
    static const bool graph_on = [] { int dev = 0, on = 0, major = 0; cudaGetDevice(&dev);
        const bool unified = cudaDeviceGetAttribute(&on, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) == cudaSuccess && on;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return tqllm::flashnext::layer_graphs(unified, major); }();  // decode_rules.hpp
    // Every step updates all kernel parameters (weights included), so layers of one kind share an exec, keyed by its
    // topology: the node count changes with the position in attention layers and with the draft in the MTP layer.
    // One exec per layer and variant (~11 x 49) held ~1.3 GB the expert banks had taken (out of memory on a 3090).
    // A decode call into a bank that holds every expert (GB10: all resident) builds its slot and dedup maps on the device
    // from the routed ids: no readback, no host map build, and the layer's graph runs to the end of the layer (same
    // kernels and maps as the split path with every assignment a hit: test-dedup-maps).
    const int * full_slots = nullptr;
    if (!sink && !cp.rt.hot_map && !cp.run.small_card && (spec || n == 1) && n * c.topk <= std::min(1024, cache.slots(layer))) {
        struct Table { std::vector<int> host; std::unique_ptr<Device<int>> dev; };
        static std::map<std::pair<const MixedExpertCache *, int>, Table> tables;
        static std::vector<int> slot_of;
        if (cache.full_bank(layer, slot_of)) {
            const Mat mg = cache.bank_mat(layer, 0), mu = cache.bank_mat(layer, 1), md = cache.bank_mat(layer, 2);
            if (mg.fmt->has_expert_matvec() && mu.fmt->has_expert_matvec() && md.fmt->has_expert_matvec()) {
                auto & t = tables[{&cache, layer}];
                if (t.host != slot_of) { PersistentAlloc keep; t.dev = std::make_unique<Device<int>>(slot_of.size()); t.dev->put(slot_of); t.host = slot_of; }
                full_slots = t.dev->p;
                static bool said = false;
                if (!said) { said = true; std::fprintf(stderr, "tqllm: every expert resident: slot and dedup maps built on the device\n"); }
            }
        }
    }
    // --split experts, rank 0: each forward is offered to rank 1 at layer 0 (a busy rank 1 declines); a rank 1 that went
    // away (or a forward left open) turns expert parallelism off. The local branch below computes every expert (all
    // resident), so the output stays the same either way.
    const bool ep_rank0 = cp.run.ep && cp.run.ep->rank() == 0 && layer < c.layers;
    bool ep_on = false;
    if (ep_rank0) {
        const bool was_off = g_ep_fwd.off;
        tqllm::TpComm & ep = *cp.run.ep;
        auto announce = [&](int rows) {  // host only, before this layer's graph capture begins; no answer awaited
            std::lock_guard<std::mutex> g(g_ep_lease.m);
            if (!g_ep_lease.leased) return false;  // rank 1 serves its own users
            const int32_t hdr[2] = {kEpForward, rows}; ep.ctl_send(hdr, sizeof hdr);
            return true;
        };
        const auto call = tqllm::flashnext::ep_before_layer(g_ep_fwd, layer, n, full_slots && ep_formats_ok(cache, layer), ep.alive(), announce);
        if (g_ep_fwd.off && !was_off) std::fprintf(stderr, "tqllm: --split experts off: %s; every layer runs locally\n", cp.run.ep->alive() ? "a forward was left open" : "rank 1 went away");
        if (call == tqllm::flashnext::EpCall::Fail) throw std::runtime_error("--split experts: the forward lost its exchanges with rank 1 at layer " + std::to_string(layer));
        ep_on = call == tqllm::flashnext::EpCall::Exchange;
    }
    bool capturing = false;
    // several sequences (segs) only when the maps are built on the device: their split path waits on the routed ids per layer
    // up to 8 rows; with the maps on the device (no host wait in the layer) up to 32 (np 8 x 4-row blocks)
    if (graph_on && spec && (!segs || full_slots) && !sink && n <= (full_slots ? 32 : 8) && g_arena && !g_persistent) {
        static int epoch = -1; if (epoch != cp.rt.prefill_epoch) { epoch = cp.rt.prefill_epoch; g_decode_graphs.reset_warmup(); }  // lazy scratch is rebuilt by plain calls
        if (g_decode_graphs.capture(graph_owner, layer, n, plain)) { ck(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); capturing = true; }  // calls 1-2 run plain: lazy allocations happen outside capture
    }
    const bool trunk = layer < c.layers, attention = !trunk || c.compress_ratios[layer];
    Device<float> norm(size_t(width) * n), mixed(size_t(h) * n), inject(size_t(c.hc) * n), block(size_t(h) * n), combined(size_t(width) * n);
    // layer-major prefill with BF16 wide rows (sink->wide_b): the row-scale path reads and writes them directly; a PLE layer or
    // a short sub-chunk on another path works on an fp32 copy of its rows and packs its combined rows back
    __nv_bfloat16 * wb = sink && sink->wide_b ? sink->wide_b + size_t(sink->off) * width : nullptr;
    const int hp = hc_path(prefix, n);
    const bool ple_layer = trunk && ple_rows && std::find(c.ple_layers.begin(), c.ple_layers.end(), layer) != c.ple_layers.end();
    if (wb && (ple_layer || hp != 2)) { residual = std::make_unique<Device<float>>(size_t(width) * n); tqllm_cuda::wide_unpack(wb, residual->p, int64_t(width) * n, nullptr); }
    const __nv_bfloat16 * res_b = wb && !ple_layer && hp == 2 ? wb : nullptr;
    if (ple_layer) {
        Device<float> key(size_t(width) * n), value(size_t(h) * n), kn(key.count), qn(key.count), gates(size_t(c.hc) * n), gated(key.count);
        {
            const Checkpoint::SharedInput shared_in(cp, ple_rows, size_t(n) * h);
            cp.project(prefix + "ple_key.weight", ple_rows, key.p, h, width, n);
            cp.project(prefix + "ple_value.weight", ple_rows, value.p, h, h, n);
        }
        tqllm_cuda::hc_norm(key.p, cp.dense_device(prefix + "ple_norm_key.weight"), kn.p, h, c.hc, n, float(cp.epsilon), nullptr);
        tqllm_cuda::hc_norm(residual->p, cp.dense_device(prefix + "ple_norm_query.weight"), qn.p, h, c.hc, n, float(cp.epsilon), nullptr);
        tqllm_cuda::ple_gate(kn.p, qn.p, gates.p, h, c.hc, n, nullptr);
        tqllm_cuda::ple_broadcast(value.p, gates.p, gated.p, h, c.hc, n, nullptr);
        const int hist = (c.ple_kernel - 1) * c.ple.ngram;
        Device<float> pn(key.count), conv(key.count);
        auto out = std::make_unique<Device<float>>(key.count);
        tqllm_cuda::hc_norm(gated.p, cp.dense_device(prefix + "ple_norm_conv.weight"), pn.p, h, c.hc, n, float(cp.epsilon), nullptr);
        auto ple_state = [&](DecodeModel & m, int off, int nn) {  // the conv history is per sequence
            if (!m.ple_past[layer]) { PersistentAlloc keep; m.ple_past[layer] = std::make_unique<Device<float>>(size_t(hist) * width); ck_zero(*m.ple_past[layer]); }
            Device<float> next(m.ple_past[layer]->count);
            const size_t rows = size_t(nn) * width; const float * pseg = pn.p + size_t(off) * width;
            if (m.spec) {
                PersistentAlloc keep;
                if (!m.ple_old[layer]) m.ple_old[layer] = std::make_unique<Device<float>>(m.ple_past[layer]->count);
                if (!m.ple_saved[layer] || m.ple_saved[layer]->count < rows) m.ple_saved[layer] = std::make_unique<Device<float>>(rows);
                ck_copy_n(*m.ple_old[layer], m.ple_past[layer]->p, m.ple_past[layer]->count);
                ck_copy_n(*m.ple_saved[layer], pseg, rows);
            }
            tqllm_cuda::ple_conv(pseg, m.ple_past[layer]->p, cp.dense_device(prefix + "ple_conv1d.weight"), conv.p + size_t(off) * width, next.p,
                                 width, nn, c.ple_kernel, c.ple.ngram, nullptr);
            ck_copy(*m.ple_past[layer], next);
        };
        if (segs) for (const auto & sg : *segs) ple_state(*sg.m, sg.off, sg.n);
        else ple_state(*this, 0, n);
        tqllm_cuda::ple_apply(residual->p, gated.p, conv.p, out->p, int(out->count), nullptr); residual = std::move(out);
    }
    // the norms write their BF16 image for the mixer's BF16 projections (no f32 -> bf16 passes); prefill sub-chunks of more
    // than 64 tokens with a BF16 mixer keep no fp32 normalized rows at all (row scales only, the mix recomputes them; exact)
    auto normb = std::make_unique<Device<__nv_bfloat16>>(size_t(width) * n);
    const bool hc_rs = hp == 1, hc_rsh = hp == 2;
    std::unique_ptr<Device<float>> rs; if (hc_rs || hc_rsh) rs = std::make_unique<Device<float>>(size_t(c.hc) * n);
    std::unique_ptr<Device<__half>> normh; if (hc_rsh) normh = std::make_unique<Device<__half>>(size_t(width) * n);
    __nv_bfloat16 * inj_img = nullptr;  // BF16 inject weights read the FP16 image through their FP16 copy (mix_device_h)
    const float * attn_gamma = cp.dense_device(prefix + "hc_attn_norm.weight");
    if (hc_rs) tqllm_cuda::hc_norm_rs_bf16(residual->p, attn_gamma, rs->p, normb->p, h, c.hc, n, float(cp.epsilon), nullptr);
    else if (hc_rsh && res_b) tqllm_cuda::hc_norm_rs_h(res_b, attn_gamma, rs->p, normh->p, h, c.hc, n, float(cp.epsilon), nullptr, inj_img);
    else if (hc_rsh) tqllm_cuda::hc_norm_rs_h(residual->p, attn_gamma, rs->p, normh->p, h, c.hc, n, float(cp.epsilon), nullptr, inj_img);
    else tqllm_cuda::hc_norm_bf16(residual->p, attn_gamma, norm.p, normb->p, h, c.hc, n, float(cp.epsilon), nullptr);
    // the attention / GDN projections read the mixer's FP16 image of its output (the conversion they did gives the same rows)
    // (and its BF16 image for the BF16 GDN alpha / beta and indexer k / q projections)
    std::unique_ptr<Device<__half>> mixh; std::unique_ptr<Device<__nv_bfloat16>> mixb;
    if (hc_rsh && !segs && n > 64) { mixh = std::make_unique<Device<__half>>(size_t(n) * h); mixb = std::make_unique<Device<__nv_bfloat16>>(size_t(n) * h); }
    if (hc_rsh) mix_device_h(cp, prefix + "hc_attn_", normh->p, inj_img, n, mixed.p, inject.p, res_b ? nullptr : residual->p, rs->p, attn_gamma,
                             mixh ? mixh->p : nullptr, mixb ? mixb->p : nullptr, res_b);
    else mix_device(cp, prefix + "hc_attn_", norm.p, n, mixed.p, inject.p, normb->p,
                    hc_rs ? residual->p : nullptr, hc_rs ? rs->p : nullptr, hc_rs ? attn_gamma : nullptr);
    if (segs) {  // row-wise projections once for every sequence of the batch
        // row-wise projections once for all sequences; only the stateful middle (conv + recurrence, KV + attention) per sequence
        const std::string pf = "blk." + std::to_string(layer) + ".";
        if (attention) {
            const int heads = cp.attention_heads, kvh = cp.attention_kvheads, ad = cp.attention_dim;
            Device<float> qgB(size_t(n) * heads * ad * 2), kB(size_t(n) * kvh * ad), vB(kB.count), aoB(size_t(n) * heads * ad);
            {
                const Checkpoint::SharedInput shared_in(cp, mixed.p, size_t(n) * h);
                cp.project(pf + "attn_q.weight", mixed.p, qgB.p, h, heads * ad * 2, n);
                cp.project(pf + "attn_k.weight", mixed.p, kB.p, h, kvh * ad, n);
                cp.project(pf + "attn_v.weight", mixed.p, vB.p, h, kvh * ad, n);
            }
            for (const auto & sg : *segs) {
                const AttnPre pre{qgB.p + size_t(sg.off) * heads * ad * 2, kB.p + size_t(sg.off) * kvh * ad, vB.p + size_t(sg.off) * kvh * ad};
                attention_core(cp, mixed.p + size_t(sg.off) * h, nullptr, sg.n, layer, nullptr, trunk ? &sg.m->att[layer] : &sg.m->mtp_att,
                               &pre, aoB.p + size_t(sg.off) * heads * ad);
            }
            tqllm_cuda::fn_attention_gate(aoB.p, qgB.p, heads, ad, n, nullptr);
            cp.project(pf + "attn_output.weight", aoB.p, block.p, heads * ad, h, n);
        } else {
            const int kh = cp.key_heads, vh = cp.value_heads, sd = cp.state_dim, channels = (2 * kh + vh) * sd;
            Device<float> qkvB(size_t(n) * channels), zB(size_t(n) * vh * sd), aB(size_t(n) * vh), bB(aB.count), roB(zB.count), gB(zB.count);
            {
                const Checkpoint::SharedInput shared_in(cp, mixed.p, size_t(n) * h);
                cp.project(pf + "attn_qkv.weight", mixed.p, qkvB.p, h, channels, n);
                cp.project(pf + "attn_gate.weight", mixed.p, zB.p, h, vh * sd, n);
                cp.project(pf + "ssm_alpha.weight", mixed.p, aB.p, h, vh, n);
                cp.project(pf + "ssm_beta.weight", mixed.p, bB.p, h, vh, n);
            }
            for (const auto & sg : *segs) {
                const GdnPre pre{qkvB.p + size_t(sg.off) * channels, zB.p + size_t(sg.off) * vh * sd, aB.p + size_t(sg.off) * vh, bB.p + size_t(sg.off) * vh};
                recurrent_core(cp, nullptr, nullptr, sg.n, layer, nullptr, &sg.m->rec[layer], &pre, roB.p + size_t(sg.off) * vh * sd);
            }
            tqllm_cuda::fn_gated_rms(roB.p, zB.p, cp.dense_device(pf + "ssm_norm.weight"), gB.p, vh, sd, n, float(cp.epsilon), nullptr);
            cp.project(pf + "ssm_out.weight", gB.p, block.p, vh * sd, h, n);
        }
    } else if (segs) {
        for (const auto & sg : *segs) {
            const size_t o = size_t(sg.off) * h;
            if (attention) attention_core(cp, mixed.p + o, block.p + o, sg.n, layer, nullptr, trunk ? &sg.m->att[layer] : &sg.m->mtp_att);
            else recurrent_core(cp, mixed.p + o, block.p + o, sg.n, layer, nullptr, &sg.m->rec[layer]);
        }
    } else if (attention) attention_core(cp, mixed.p, block.p, n, layer, nullptr, attn ? attn : &att[layer], nullptr, nullptr, mixh ? mixh->p : nullptr, mixb ? mixb->p : nullptr);
    else recurrent_core(cp, mixed.p, block.p, n, layer, nullptr, &rec[layer], nullptr, nullptr, mixh ? mixh->p : nullptr, mixb ? mixb->p : nullptr);
    // a prefill sub-chunk writes its combined rows straight into the layer's wide rows (combine is elementwise, so in place
    // over the residual view is safe)
    float * comb = sink && !wb ? sink->wide + size_t(sink->off) * width : combined.p;
    __nv_bfloat16 * comb_b = wb && hc_rsh ? wb : nullptr;  // in place over the BF16 rows (elementwise per row)
    if (hc_rs) tqllm_cuda::hc_combine_norm_rs_bf16(residual->p, block.p, inject.p, comb, cp.dense_device(prefix + "hc_ffn_norm.weight"), rs->p, normb->p, h, c.hc, n, float(cp.epsilon), nullptr);
    else if (hc_rsh && comb_b && res_b) tqllm_cuda::hc_combine_norm_rs_h(res_b, block.p, inject.p, comb_b, cp.dense_device(prefix + "hc_ffn_norm.weight"), rs->p, normh->p, h, c.hc, n, float(cp.epsilon), nullptr, inj_img);
    else if (hc_rsh && comb_b) tqllm_cuda::hc_combine_norm_rs_h(residual->p, block.p, inject.p, comb_b, cp.dense_device(prefix + "hc_ffn_norm.weight"), rs->p, normh->p, h, c.hc, n, float(cp.epsilon), nullptr, inj_img);
    else if (hc_rsh) tqllm_cuda::hc_combine_norm_rs_h(residual->p, block.p, inject.p, comb, cp.dense_device(prefix + "hc_ffn_norm.weight"), rs->p, normh->p, h, c.hc, n, float(cp.epsilon), nullptr, inj_img);
    else {
        tqllm_cuda::hc_combine(residual->p, block.p, inject.p, comb, h, c.hc, n, nullptr);
        tqllm_cuda::hc_norm_bf16(comb, cp.dense_device(prefix + "hc_ffn_norm.weight"), norm.p, normb->p, h, c.hc, n, float(cp.epsilon), nullptr);
    }
    // a prefill sub-chunk's FFN mixer writes the FP16 rows the experts and the shared expert read straight into the sink
    __half * ffn_h = hc_rsh && sink ? sink->xr + size_t(sink->off) * h : nullptr;
    if (hc_rsh) mix_device_h(cp, prefix + "hc_ffn_", normh->p, inj_img, n, mixed.p, inject.p, comb_b ? nullptr : comb, rs->p,
                             cp.dense_device(prefix + "hc_ffn_norm.weight"), ffn_h, nullptr, comb_b);
    else mix_device(cp, prefix + "hc_ffn_", norm.p, n, mixed.p, inject.p, normb->p,
                    hc_rs ? comb : nullptr, hc_rs ? rs->p : nullptr, hc_rs ? cp.dense_device(prefix + "hc_ffn_norm.weight") : nullptr);
    Device<float> logits(size_t(n) * (c.experts + 1)), weights(size_t(n) * c.topk), shared_weights(n);
    Device<int> routed(weights.count);
    // cp.run.predict_n (4; 6 with adaptive drafts): next-layer prediction (prefetch) for verify blocks of up to N tokens; above 4 the routers
    // run unfused (placement only: the output does not change)
    const int predict_n = cp.run.predict_n;
    const bool predict = !full_slots && (!cp.rt.hot_map || bool(cp.cold_prefetch)) && prefetch_on && tqllm::flashnext::predicts(n, predict_n, cp.rt.block_seqs) && trunk && layer + 1 < c.layers;
    const std::string np = "blk." + std::to_string(layer + 1) + ".";
    std::unique_ptr<Device<float>> nl;
    if (predict) nl = std::make_unique<Device<float>>(size_t(n) * (c.experts + 1));
    // Both routers + shared gates in one segmented launch and both top-k in one launch (same per-row and per-token
    // arithmetic: test-gemv-seg, test-topk2; +2.7 % 3090 5-chat at --slots 168)
    auto is_bf16 = [&](const std::string & pre) { return cp.file(pre + "ffn_gate_inp.weight").file().tensors.at(pre + "ffn_gate_inp.weight").type == GGML_BF16; };
    constexpr bool f32_router = false;
    const bool fused = !f32_router && n <= 8 && c.experts <= 1024 && h >= 1024 && is_bf16(prefix) && (!predict || is_bf16(np));
    if (fused) {
        tqllm_cuda::GemvSegs g; int rows = 0;
        auto add = [&](const std::string & name, int r, float * y) {
            g.w[g.n] = cp.bf16_device(name); g.y[g.n] = y; g.ld[g.n] = c.experts + 1; g.start[g.n] = rows; rows += r; g.start[++g.n] = rows;
        };
        add(prefix + "ffn_gate_inp.weight", c.experts, logits.p); add(prefix + "ffn_gate_inp_shexp.weight", 1, logits.p + c.experts);
        if (predict) { add(np + "ffn_gate_inp.weight", c.experts, nl->p); add(np + "ffn_gate_inp_shexp.weight", 1, nl->p + c.experts); }
        tqllm_cuda::fn_gemv_bf16_segments(g, mixed.p, h, n, nullptr);
    } else {
        router(prefix, mixed.p, logits.p, n);
        if (predict) router(np, mixed.p, nl->p, n);
    }
    if (cp.rt.keep_mask && trunk) {  // TQLLM_DEBUG=keep_map: pruned experts never win the top-k
        tqllm_cuda::logits_mask(logits.p, cp.rt.keep_mask + size_t(layer) * c.experts, n, c.experts + 1, c.experts, nullptr);
        if (predict) tqllm_cuda::logits_mask(nl->p, cp.rt.keep_mask + size_t(layer + 1) * c.experts, n, c.experts + 1, c.experts, nullptr);
    }
    std::unique_ptr<Device<int>> next_ids;
    if (predict) {
        Device<float> nw(size_t(n) * c.topk), nsw(n); next_ids = std::make_unique<Device<int>>(nw.count);
        if (fused) tqllm_cuda::moe_topk2(logits.p, routed.p, weights.p, shared_weights.p, nl->p, next_ids->p, nw.p, nsw.p, c.experts, c.topk, n, 1.f, nullptr);
        else {
            tqllm_cuda::moe_topk(logits.p, routed.p, weights.p, shared_weights.p, c.experts, c.topk, n, 1.f, nullptr);
            tqllm_cuda::moe_topk(nl->p, next_ids->p, nw.p, nsw.p, c.experts, c.topk, n, 1.f, nullptr);
        }
    } else tqllm_cuda::moe_topk(logits.p, routed.p, weights.p, shared_weights.p, c.experts, c.topk, n, 1.f, nullptr);
    auto end_capture = [&] {
        if (!capturing) return;
        capturing = false;
        cudaGraph_t cg = nullptr;
        const cudaError_t e = cudaStreamEndCapture(cudaStreamPerThread, &cg);
        if (e != cudaSuccess || !cg) throw std::runtime_error(std::string("CUDA graph: capture of layer ") + std::to_string(layer) + " failed: " + cudaGetErrorString(e));
        const bool ple = trunk && ple_rows && std::find(c.ple_layers.begin(), c.ple_layers.end(), layer) != c.ple_layers.end();
        const int kind = !trunk ? -1 : (attention ? 2 * c.compress_ratios[layer] : 0) + ple;
        g_decode_graphs.launch(graph_owner, kind, n, cg, cudaStreamPerThread);
        };
    if (!full_slots) end_capture();
    using clk = std::chrono::steady_clock; auto h0 = clk::now();
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    Device<float> sg(size_t(n) * c.shared_ff), su(sg.count), sa(sg.count), sd(size_t(n) * h), fo(size_t(n) * h);
    auto shared_expert = [&](__half * have = nullptr, float * out = nullptr) {
        const Checkpoint::SharedInput shared_in(cp, mixed.p, size_t(n) * h, have);
        cp.project(prefix + "ffn_gate_shexp.weight", mixed.p, sg.p, h, c.shared_ff, n);
        cp.project(prefix + "ffn_up_shexp.weight", mixed.p, su.p, h, c.shared_ff, n);
        tqllm_cuda::fn_swiglu_f32(sg.p, su.p, sa.p, int64_t(sg.count), nullptr);
        cp.project(prefix + "ffn_down_shexp.weight", sa.p, out ? out : sd.p, c.shared_ff, h, n);
    };
    if (sink) {
        const size_t o = size_t(sink->off);
        if (wb) { if (!comb_b) tqllm_cuda::wide_pack(comb, wb, int64_t(n) * width, nullptr); }
        else if (comb != sink->wide + o * width) ck(cudaMemcpyAsync(sink->wide + o * width, combined.p, size_t(n) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpyAsync(sink->inject + o * c.hc, inject.p, size_t(n) * c.hc * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        if (!ffn_h) tqllm_cuda::vis_f32_to_h(mixed.p, sink->xr + o * h, int64_t(n) * h, nullptr);
        ck(cudaMemcpyAsync(sink->routed + o * c.topk, routed.p, size_t(n) * c.topk * sizeof(int), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpyAsync(sink->weights + o * c.topk, weights.p, size_t(n) * c.topk * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpyAsync(sink->shared_weights + o, shared_weights.p, size_t(n) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        shared_expert(sink->xr + o * h, sink->sd + o * h);  // the same FP16 rows (no second conversion)
        return;
    }
    // TQFN_PREFILL_WAVES (default on): a prefill chunk that routes to more experts than the cache holds runs its MoE in
    // waves of at most `slots` experts; each expert is loaded once per layer per chunk (not once per small chunk).
    std::vector<int> wave_ids;
    // prefill (spec off) always takes waves, decode verify (spec on) the split path when it fits: the two paths sum the
    // down projection in a different order (1 ulp in a few values), so a slot-dependent choice changed greedy text
    // with the slot count (F2: a 23-token prompt ran waves at 140 slots and split at 150 in layer 1)
    // F39 rule (decode_rules.hpp): a verify block takes the split path whenever its distinct experts fit
    constexpr bool old_rule = false;  // F39 rule: np > 1 text equals np 1
    if (trunk && (cp.rt.hot_map || n * c.topk > cache.slots(layer) || (!spec && n > 1))) {  // incl. the short last chunk of a prompt
        wave_ids = routed.get();
        if(predict && cp.rt.hot_map && cp.cold_prefetch)cp.cold_prefetch(layer+1,next_ids->get());
        if (cp.rt.hot_map) for (int id : wave_ids) if (id < 0 || id >= c.experts) {
            const auto ls = logits.get(); const auto xs = mixed.get(); size_t bad_l = 0, bad_x = 0;
            for (float v : ls) bad_l += !std::isfinite(v);
            for (float v : xs) bad_x += !std::isfinite(v);
            throw std::runtime_error("hot/cold invalid route: layer=" + std::to_string(layer) + " id=" + std::to_string(id) +
                    " nonfinite_logits=" + std::to_string(bad_l) + " nonfinite_input=" + std::to_string(bad_x));
        }
        // one wave when the unique experts fit: the split path's miss accounting assumes small steps
        if (!cp.rt.hot_map && !tqllm::flashnext::layer_takes_waves(wave_ids, n, c.topk, cache.slots(layer), spec, old_rule)) wave_ids.clear();
    }
    if (trunk) cp.rt.trace_min_slots = std::min(cp.rt.trace_min_slots, cache.slots(layer));
    if (!wave_ids.empty()) {
        ++cp.rt.trace_waves;
        const int A = n * c.topk; constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
        std::vector<int> u(wave_ids); std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end()), u.end());
        Device<__half> xr(mixed.count); tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr);
        Device<__nv_bfloat16> eg(size_t(A) * c.expert_ff), eu(eg.count), ed(size_t(A) * h);
        Device<__half> ax(eg.count);
        bool shared_queued = false;
        // Each assignment is written by exactly one bank; weights and original router ids are untouched.
        for (int bank_index = 0; bank_index < (cp.rt.hot_map && trunk ? 2 : 1); ++bank_index) {
            // Keep independent shared-expert work on the GPU while the CPU prepares cold demand reads.
            if (cp.cold_overlap >= 2 && bank_index == 1) { shared_expert(xr.p); shared_queued = true; }
            MixedExpertCache & selected = bank_index ? *cp.rt.cold_cache : cache;
            std::vector<int> bank_ids;
            for (int id : u) if (!cp.rt.hot_map || !trunk || expert_is_hot(*cp.rt.hot_map, layer, c.experts, id) == (bank_index == 0)) bank_ids.push_back(id);
            const int W = selected.plan_slots(layer);
            // many tokens per expert (GB10 chunk 1024: ~20): grouped GEMM, weights read once per 64 tokens instead of
            // once per 4; few (3090 chunk 128: ~3) keep the dedup kernel; gemm_avg is the switch-over average.
            // Decided per layer from slot-independent counts, and only when every slot count runs waves (A > experts):
            // the GEMM rounds differently from the dedup kernel the split path uses, so a slot-dependent switch changed
            // greedy text with the slot count (a 23-token prompt: 230 rows, ~8 per expert).
            constexpr int gemm_avg = 8;  // waves use the grouped GEMM from 8 tokens per expert on average
            const bool wave_gemm = gemm_avg > 0 && A > c.experts && A >= gemm_avg * int(u.size());
            for (size_t w0 = 0; w0 < bank_ids.size(); w0 += size_t(W)) {
                const std::vector<int> wave(bank_ids.begin() + long(w0), bank_ids.begin() + long(std::min(bank_ids.size(), w0 + size_t(W))));
                if (bank_index) selected.set_miss_stream(miss_stream);
                const auto wslots = selected.prepare(layer, wave);
                if (selected.split()) selected.wait_misses();
                std::vector<int> slot_of(size_t(c.experts), -1);
                for (size_t i = 0; i < wave.size(); ++i) slot_of[size_t(wave[i])] = wslots[i];
                std::vector<int> hm(size_t(2) * A * DS); int groups[2] = {0, 0};
                for (int part = 0; part < 2; ++part) {  // 0: gate/up (token input), 1: down (assignment input)
                    int * out = hm.data() + size_t(part) * A * DS; std::vector<int> open_at(size_t(W) + 1, -1); int g = 0;
                    for (int a = 0; a < A; ++a) {
                        const int sl = slot_of[size_t(wave_ids[size_t(a)])];
                        if (sl < 0) continue;
                        int & at = open_at[size_t(sl)];
                        if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = sl; out[at * DS + 1] = 0; }
                        int * e = out + at * DS; const int col = e[1]++;
                        e[2 + col] = part ? a : a / c.topk; e[2 + DC + col] = a;
                    }
                    groups[part] = g;
                }
                Device<int> dmap(hm.size()); dmap.put(hm);
                // formats outside the u4h family (GSQ Q2_0: its grouped kernel stages a wave's tokens) take the grouped GEMM path
                std::vector<int> cnt(size_t(W), 0); int wcap = 1;
                for (int a = 0; a < A; ++a) { const int sl = slot_of[size_t(wave_ids[size_t(a)])]; if (sl >= 0) wcap = std::max(wcap, ++cnt[size_t(sl)]); }
                std::vector<int> grp(size_t(W) * wcap, 0), fill(size_t(W), 0);
                for (int a = 0; a < A; ++a) { const int sl = slot_of[size_t(wave_ids[size_t(a)])]; if (sl >= 0) grp[size_t(sl) * wcap + size_t(fill[size_t(sl)]++)] = a; }
                Device<int> dcnt(cnt.size()), dgrp(grp.size()); dcnt.put(cnt); dgrp.put(grp);
                auto gemm_fmt = [](const Mat & mm) {
                    const int id = mm.fmt->u4h_id();
                    return (id == 1 || id == 4 || id == 5 || id == 6 || id == 7 || id == 10 || id == 13 || id == 14) && mm.cols % (id == 14 ? 128 : 256) == 0;
                };
                auto proj = [&](const Mat & mm, const __half * x, __nv_bfloat16 * y, int rows, bool per_expert) {
                    if (mm.fmt->u4h_id() >= 0 && !(wave_gemm && gemm_fmt(mm))) mm.fmt->expert_matvec_dedup(mm, x, y, rows, dmap.p + (per_expert ? size_t(A) * DS : 0), groups[per_expert], nullptr);
                    else require(mm.fmt->expert_gemm(mm, x, y, rows, dcnt.p, dgrp.p, wcap, W, c.topk, per_expert, nullptr),
                                 std::string("prefill waves: no grouped kernel for ") + mm.fmt->name());
                };
                auto m = selected.gate(), u2 = selected.up();
                if (m.fmt == u2.fmt && m.fmt->u4h_id() >= 0 && !(wave_gemm && gemm_fmt(m))) m.fmt->expert_matvec_dedup2(m, u2, xr.p, eg.p, eu.p, c.expert_ff, dmap.p, groups[0], nullptr);
                else { proj(m, xr.p, eg.p, c.expert_ff, false); proj(u2, xr.p, eu.p, c.expert_ff, false); }
                tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
                proj(selected.down(), ax.p, ed.p, h, true);
                // Same-bank waves still wait before reusing slots. The separate cold bank may prepare
                // its demand reads and DMA while the final hot wave remains in flight.
                if (!cp.cold_overlap || !cp.rt.hot_map || !trunk || bank_index != 0 || w0 + size_t(W) < bank_ids.size())
                    ck(cudaStreamSynchronize(nullptr));
        }
        }
        if (!shared_queued) shared_expert();
        tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
        if (cp.rt.hot_map && trunk) {
            const auto out = fo.get();
            if (std::any_of(out.begin(), out.end(), [](float v) { return !std::isfinite(v); })) {
                const auto a = ed.get(), b = eg.get(), d = eu.get(); size_t ea = 0, eb = 0, eu_bad = 0;
                for (auto v : a) ea += !std::isfinite(float(v));
                for (auto v : b) eb += !std::isfinite(float(v));
                for (auto v : d) eu_bad += !std::isfinite(float(v));
                throw std::runtime_error("hot/cold nonfinite MoE: layer=" + std::to_string(layer) + " down=" + std::to_string(ea) +
                        " gate=" + std::to_string(eb) + " up=" + std::to_string(eu_bad));
            }
        }
    } else if (ep_on) {
        // --split experts, rank 0: the input rows and routed ids go to rank 1, this rank computes the experts e % 2 == 0 (and
        // the assignments past the result cap), rank 1 sends back its results; the combine runs in the one-box order
        // (test ep_split: bit-identical to the branch below)
        const int A = n * c.topk, cap = tqllm::flashnext::ep_result_cap(A); constexpr int DS = tqllm_cuda::kDedupStride;
        tqllm::TpComm & ep = *cp.run.ep;
        Device<__half> xr(mixed.count);
        tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr); shared_expert(xr.p);
        end_capture();  // the exchanges run eagerly between this layer's graph and the next
        const cudaStream_t s = cudaStreamPerThread;
        ep.group_begin(); ep.send(xr.p, size_t(n) * h * sizeof(__half), 1, s); ep.send(routed.p, size_t(A) * sizeof(int), 1, s); ep.group_end();
        const Mat mg = cache.bank_mat(layer, 0), mu = cache.bank_mat(layer, 1), md = cache.bank_mat(layer, 2);
        Device<int> local(A), maps(size_t(A) * (1 + 2 * DS));
        int * slots_p = maps.p, * gu = maps.p + A, * dn = gu + size_t(A) * DS;
        tqllm_cuda::ep_local_ids(routed.p, A, nullptr, 0, 0, cap, local.p, s);
        tqllm_cuda::moe_dedup_maps(local.p, full_slots, A, c.topk, slots_p, gu, dn, A, s);
        Device<__nv_bfloat16> eg(size_t(A) * c.expert_ff), eu(eg.count), ed(size_t(A) * h), rb(size_t(std::max(cap, 1)) * h);
        Device<__half> ax(eg.count);
        if (mg.fmt == mu.fmt) mg.fmt->expert_matvec_dedup2(mg, mu, xr.p, eg.p, eu.p, c.expert_ff, gu, A, s);
        else { mg.fmt->expert_matvec_dedup(mg, xr.p, eg.p, c.expert_ff, gu, A, s); mu.fmt->expert_matvec_dedup(mu, xr.p, eu.p, c.expert_ff, gu, A, s); }
        tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), s);
        md.fmt->expert_matvec_dedup(md, ax.p, ed.p, h, dn, A, s);
        if (cap > 0) { ep.recv(rb.p, size_t(cap) * h * sizeof(__nv_bfloat16), 1, s); tqllm_cuda::ep_scatter(rb.p, routed.p, A, 0, cap, ed.p, h, s); }
        if (layer + 1 == c.layers) {  // the forward's last exchange: a rank 1 that went away during it fails the request
            ck(cudaStreamSynchronize(s));
            if (!ep.alive()) throw std::runtime_error("--split experts: rank 1 went away during a forward");
        }
        tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, s);
        cache.add_device_counters(uint64_t(A), 0, 0, 0);
    } else if (full_slots) {
        const int A = n * c.topk; constexpr int DS = tqllm_cuda::kDedupStride;
        Device<__half> xr(mixed.count);
        tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr); shared_expert(xr.p);
        const Mat mg = cache.bank_mat(layer, 0), mu = cache.bank_mat(layer, 1), md = cache.bank_mat(layer, 2);
        const bool dedup_gu = n > 1 && mg.fmt->has_expert_dedup() && mu.fmt->has_expert_dedup(), dedup_dn = n > 1 && md.fmt->has_expert_dedup();
        Device<int> maps(size_t(A) * (1 + 2 * DS));
        int * slots_p = maps.p, * gu = maps.p + A, * dn = gu + size_t(A) * DS;
        tqllm_cuda::moe_dedup_maps(routed.p, full_slots, A, c.topk, slots_p, dedup_gu ? gu : nullptr, dedup_dn ? dn : nullptr, A, nullptr);
        Device<__nv_bfloat16> eg(size_t(A) * c.expert_ff), eu(eg.count), ed(size_t(A) * h);
        Device<__half> ax(eg.count);
        if (dedup_gu && mg.fmt == mu.fmt) mg.fmt->expert_matvec_dedup2(mg, mu, xr.p, eg.p, eu.p, c.expert_ff, gu, A, nullptr);
        else if (dedup_gu) { mg.fmt->expert_matvec_dedup(mg, xr.p, eg.p, c.expert_ff, gu, A, nullptr); mu.fmt->expert_matvec_dedup(mu, xr.p, eu.p, c.expert_ff, gu, A, nullptr); }
        else { mg.fmt->expert_matvec(mg, xr.p, eg.p, c.expert_ff, slots_p, n, c.topk, false, nullptr); mu.fmt->expert_matvec(mu, xr.p, eu.p, c.expert_ff, slots_p, n, c.topk, false, nullptr); }
        tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
        if (dedup_dn) md.fmt->expert_matvec_dedup(md, ax.p, ed.p, h, dn, A, nullptr);
        else md.fmt->expert_matvec(md, ax.p, ed.p, h, slots_p, n, c.topk, true, nullptr);
        tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
        cache.add_device_counters(uint64_t(A), 0, 0, 0);
    } else if (cache.split()) {
        // Shared expert and hit experts run while the host prepares the cache and misses land on the miss stream.
        const int A = n * c.topk;
        static PinnedInts ids_host; ids_host.need(size_t(2) * A);
        // The routed ids go to the host on their own stream, so the default stream runs the FP16 input and the shared
        // expert during the copy instead of waiting on a copy-engine -> SM dependency.
        static cudaStream_t ids_stream = [] { cudaStream_t q = nullptr; ck(cudaStreamCreateWithFlags(&q, cudaStreamNonBlocking)); return q; }();
        static cudaEvent_t routed_done = [] { cudaEvent_t e; ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e; }();
        ck(cudaEventRecord(routed_done, nullptr)); ck(cudaStreamWaitEvent(ids_stream, routed_done, 0));
        const cudaStream_t cs = ids_stream;
        ck(cudaMemcpyAsync(ids_host.p, routed.p, size_t(A) * 4, cudaMemcpyDeviceToHost, cs));
        if (predict) ck(cudaMemcpyAsync(ids_host.p + A, next_ids->p, size_t(A) * 4, cudaMemcpyDeviceToHost, cs));
        static cudaEvent_t ids_ready = [] { cudaEvent_t e; ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e; }();
        static const int cpu_miss = tqllm::flashnext::cpu_miss_threads(std::getenv("TQFN_CPU_MISS") ? std::atoi(std::getenv("TQFN_CPU_MISS")) : -1,
                                                                       cp.run.exact, int(std::thread::hardware_concurrency()));  // decode_rules.hpp
        static std::unique_ptr<CpuMissPool> cpu_pool = cpu_miss > 0 ? std::make_unique<CpuMissPool>(cpu_miss, h, c.expert_ff) : nullptr;
        const bool miss_driven = cpu_pool && !cp.run.small_card;  // decode_rules.hpp runs_on_cpu
        static int cpu_budget = tqllm::flashnext::kCpuMissBudget, calls_at_zero = 0;  // decode_rules.hpp next_cpu_budget
        cp.rt.cpu_owned = cpu_pool && !miss_driven;
        static float * x_host = nullptr; static __nv_bfloat16 * cpu_out = nullptr;
        if (miss_driven && trunk && cpu_budget <= 0) { ++calls_at_zero; if ((cpu_budget = tqllm::flashnext::next_cpu_budget(0, false, calls_at_zero)) > 0) calls_at_zero = 0; }
        const int pool_rows = cp.run.small_card ? tqllm::flashnext::kCpuPoolRows : tqllm::flashnext::kCpuPoolMaxRows;
        const bool use_cpu = cpu_pool && trunk && n <= pool_rows && (!miss_driven || cpu_budget > 0);
        if (use_cpu) {
            if (!x_host) { ck(cudaMallocHost(reinterpret_cast<void **>(&x_host), size_t(pool_rows) * h * 4));
                           ck(cudaMallocHost(reinterpret_cast<void **>(&cpu_out), size_t(pool_rows) * c.topk * h * 2)); }
            ck(cudaMemcpyAsync(x_host, mixed.p, size_t(n) * h * 4, cudaMemcpyDeviceToHost, cs));
        }
        ck(cudaEventRecord(ids_ready, cs));
        Device<__half> xr(mixed.count);  // one FP16 copy of the FFN input for the shared and routed experts
        tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr); shared_expert(xr.p);
        ck(cudaEventSynchronize(ids_ready)); auto h1 = clk::now();
        std::vector<int> routed_ids(ids_host.p, ids_host.p + A);
        // TQFN_TAIL_SKIP=R (APPROXIMATE, changes the output; idea: Strata v0.1.42 route-tail-skip): an expert that is
        // not resident and that every token of this call ranks R-th or lower is neither copied nor computed. Its
        // assignments get weight 0 (other weights unchanged) and point at the token's first choice (computed anyway).
        static const int tail_skip = std::getenv("TQFN_TAIL_SKIP") ? std::atoi(std::getenv("TQFN_TAIL_SKIP")) : 0;
        if (tail_skip > 0 && trunk) {  // decode_rules.hpp tail_skip_targets: any verify block size
            const auto to = tqllm::flashnext::tail_skip_targets(routed_ids, c.topk, tail_skip, [&](int e) { return cache.resident(layer, e); });
            static uint64_t skipped = 0;
            static struct Report { ~Report() { std::printf("TAIL_SKIP assignments skipped %llu\n", (unsigned long long) skipped); } } report;
            for (size_t a = 0; a < to.size(); ++a) if (to[a] >= 0) {
                routed_ids[a] = to[a]; ck(cudaMemsetAsync(weights.p + a, 0, sizeof(float), nullptr)); ++skipped;
            }
        }
        if (static FILE * dump = tqllm::debug_cstr("route_dump") ? std::fopen(tqllm::debug_cstr("route_dump"), "ab") : nullptr; dump && trunk) {
            const int32_t hdr[2] = {layer, n}; std::fwrite(hdr, 4, 2, dump); std::fwrite(routed_ids.data(), 4, routed_ids.size(), dump);
        }
        // Routed and predicted next-layer ids per trunk layer call, for offline cache simulation.
        if (static FILE * dump = tqllm::debug_cstr("pred_dump") ? std::fopen(tqllm::debug_cstr("pred_dump"), "ab") : nullptr; dump && trunk) {
            const int32_t hdr[3] = {layer, n, predict ? A : 0}; std::fwrite(hdr, 4, 3, dump);
            std::fwrite(routed_ids.data(), 4, routed_ids.size(), dump);
            if (predict) std::fwrite(ids_host.p + A, 4, size_t(A), dump);
        }
        // TQFN_CPU_MISS: a non-resident expert goes to the CPU pool; its GPU assignment points at a resident expert of the
        // same token (output row overwritten before the combine), so the cache neither copies nor inserts it.
        std::vector<std::pair<int, int>> cpu_rows;  // (assignment, row in cpu_out)
        std::vector<int> cpu_experts;  // cpu_refill (off): would copy them into the cache after this layer
        if (use_cpu) {
            std::vector<CpuMissPool::Job> jobs; std::map<int, int> job_of;
            for (int a = 0; a < A; ++a) {
                const int e = routed_ids[a], t = a / c.topk;
                if (!tqllm::flashnext::runs_on_cpu(layer, e, cache.resident(layer, e), true, miss_driven)) continue;
                int sub = -1;
                for (int q = 0; q < c.topk && sub < 0; ++q) if (cache.resident(layer, routed_ids[t * c.topk + q])) sub = routed_ids[t * c.topk + q];
                // nothing resident for this token: any resident expert of the layer is the placeholder (its output row is
                // overwritten by the CPU result before the combine), so the miss is not copied over PCIe
                if (sub < 0) { static std::vector<int> any_res(c.layers + 1, -1); int & r = any_res[layer];
                               if (r < 0 || !cache.resident(layer, r)) { r = -1; for (int x = 0; x < c.experts && r < 0; ++x) if (cache.resident(layer, x)) r = x; }
                               sub = r; }
                if (sub < 0) sub = e;  // nothing resident in the whole layer: the expert loads as a miss (its CPU row still overwrites it)
                // at most cpu_max distinct experts per layer call on the CPU, the rest are GPU misses
                const int cpu_max = cp.run.small_card ? 5 : cpu_budget;  // distinct experts per layer call on the CPU pool (miss-driven: adaptive)
                if (!job_of.count(e) && int(jobs.size()) >= cpu_max) continue;
                auto [it, fresh] = job_of.try_emplace(e, int(jobs.size()));
                if (fresh) {
                    CpuMissPool::Job j; const auto pl = cp.expert_payload(layer, e);
                    const int rows[3] = {c.expert_ff, c.expert_ff, h};
                    for (int m = 0; m < 3; ++m) { j.w[m] = pl[m].bytes.data; j.type[m] = pl[m].type; j.row_bytes[m] = pl[m].bytes.size / size_t(rows[m]); }
                    constexpr bool gguf_rows = false;  // rows are rebuilt from the packed copy
                    if (const auto * pv = g_cpu_packed; pv && layer < pv->layers && !gguf_rows) {
                        const char * blk = pv->map + pv->base[layer] + uint64_t(e) * pv->block[layer]; uint64_t off = 0;
                        for (int m = 0; m < 3; ++m) {
                            j.dev[m] = blk + off; j.aux[m] = blk + off + pv->sizes[layer][2 * m]; j.fmt[m] = pv->fmt[layer][m];
                            off += pv->sizes[layer][2 * m] + pv->sizes[layer][2 * m + 1];
                        }
                        j.tiled = pv->tiled;
                    }
                    jobs.push_back(j);
                }
                if (fresh) cpu_experts.push_back(e);
                auto & j = jobs[it->second]; j.cols_t[j.ncols++] = t;
                cpu_rows.push_back({a, it->second * 1000 + j.ncols - 1});
                routed_ids[a] = sub;
            }
            std::vector<int> base(jobs.size(), 0);
            for (size_t q = 1; q < jobs.size(); ++q) base[q] = base[q - 1] + jobs[q - 1].ncols;
            for (auto & [a, r] : cpu_rows) r = base[r / 1000] + r % 1000;
            static uint64_t cpu_assign = 0, all_assign = 0; cpu_assign += cpu_rows.size(); all_assign += A;
            static struct Report { ~Report() { std::printf("CPU_MISS assignments on the CPU %llu of %llu\n", (unsigned long long) cpu_assign, (unsigned long long) all_assign); } } report;
            cpu_pool->start(x_host, n, std::move(jobs), cpu_out);
        }
        const auto slots = cache.prepare(layer, routed_ids); const auto & missed = cache.last_missed();
        require(cache.gate().fmt->has_expert_matvec(), "split expert path needs a grouped expert kernel");
        static std::vector<PinnedInts> slot_host(c.layers + 1); auto & sh = slot_host[layer]; sh.need(size_t(3) * A);
        int hit_n = 0, miss_n = 0;
        for (int a = 0; a < A; ++a) {
            sh.p[a] = slots[a];
            sh.p[A + a] = missed[a] ? -1 : slots[a]; hit_n += !missed[a];
            sh.p[2 * A + a] = missed[a] ? slots[a] : -1; miss_n += missed[a];
        }
        int * slots_p = nullptr, * hit_p = nullptr, * miss_p = nullptr;  // set from the parameter maps below
        auto h2 = clk::now();
        auto issue_prefetch = [&] {
            if (!predict) return;
            // prefetch only each token's pf_k best predicted experts (ids are in descending score order)
            const int pf_k = tqllm::flashnext::prefetch_ranks(cp.rt.block_seqs).last;  // decode_rules.hpp prefetch_ranks
            std::vector<int> want; want.reserve(A);
            const int high = tqllm::flashnext::prefetch_ranks(cp.rt.block_seqs).high;  // ranks 1..high first, the rest only over stale or rarely used victims
            for (int t = 0; t < n; ++t) for (int j = 0; j < std::min(high, c.topk); ++j) if (const int id = ids_host.p[A + t * c.topk + j]; !(cp.rt.cpu_owned && tqllm::flashnext::cpu_owned(layer + 1, id))) want.push_back(id);
            cache.set_prefetch_strict(int(want.size()));
            for (int t = 0; t < n; ++t) for (int j = high; j < std::min(pf_k, c.topk); ++j) if (const int id = ids_host.p[A + t * c.topk + j]; !(cp.rt.cpu_owned && tqllm::flashnext::cpu_owned(layer + 1, id))) want.push_back(id);
            cache.prefetch(layer + 1, want, copy_stream);
        };
        const auto h3 = clk::now();
        g_host_ms[0] += ms(h0, h1); g_host_ms[1] += ms(h1, h2); g_host_ms[2] += ms(h2, h3);
        Device<__nv_bfloat16> eg(size_t(n) * c.topk * c.expert_ff), eu(eg.count), ed(size_t(n) * c.topk * h);
        Device<__half> ax(eg.count);
        // Dedup: one block per unique expert for all verify tokens that route to it (weights read once).
        const bool dedup_gu = n > 1 && cache.gate().fmt->has_expert_dedup() && cache.up().fmt->has_expert_dedup();
        const bool dedup_dn = n > 1 && cache.down().fmt->has_expert_dedup();
        constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
        static std::vector<PinnedInts> dedup_host(c.layers + 1); auto & dh = dedup_host[layer]; dh.need(size_t(5) * A * DS);
        // The hits' SwiGLU and down projection run before the wait for the demand misses (each output row is one
        // assignment's own sum, so splitting the down launch into hit and miss groups is exact; +1.8 % 3090 5-chat)
        const bool split_down = dedup_dn && hit_n && miss_n;
        int groups[5] = {0, 0, 0, 0, 0};  // hit, miss, all (down), hit (down), miss (down)
        if (dedup_gu || dedup_dn) {
            auto build = [&](int which, int * out, bool per_expert) {
                int g = 0;
                std::vector<int> open_at(c.experts + 1, -1);
                for (int a = 0; a < A; ++a) {
                    if ((which == 0 && missed[a]) || (which == 1 && !missed[a])) continue;
                    const int slot = slots[a]; int & at = open_at[slot];
                    if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = slot; out[at * DS + 1] = 0; }
                    int * e = out + at * DS; const int col = e[1]++;
                    e[2 + col] = per_expert ? a : a / c.topk; e[2 + DC + col] = a;
                }
                return g;
            };
            if (dedup_gu) { groups[0] = build(0, dh.p, false); groups[1] = build(1, dh.p + A * DS, false); }
            if (dedup_dn && !split_down) groups[2] = build(2, dh.p + 2 * A * DS, true);
            if (split_down) { groups[3] = build(0, dh.p + 3 * A * DS, true); groups[4] = build(1, dh.p + 4 * A * DS, true); }
        }
        int * dmap[5] = {};
        const int A4 = (A + 3) / 4 * 4;
        // slot and dedup maps go up in one async copy from this layer's own pinned buffer (F25: sending them in a kernel's
        // parameters, param_copy, faulted the 3090's front end every ~15-25 min of long decode); only the used groups of each
        // dedup map are sent. The buffer is rewritten one step later, after this layer's routed-id sync has passed the copy.
        int dpos[5] = {}, used = 3 * A4;
        for (int q = 0; q < 5; ++q) { dpos[q] = used; used += groups[q] * DS; }
        Device<int> maps(size_t(used) + 4);
        {
            static std::vector<PinnedInts> map_host(c.layers + 1); auto & mh = map_host[layer];
            static std::vector<int> blob; blob.assign(size_t(used), -1);
            for (int q = 0; q < 3; ++q) std::memcpy(blob.data() + size_t(q) * A4, sh.p + size_t(q) * A, size_t(A) * 4);
            for (int q = 0; q < 5; ++q) if (groups[q]) std::memcpy(blob.data() + dpos[q], dh.p + size_t(q) * A * DS, size_t(groups[q]) * DS * 4);
            blob.resize((blob.size() + 3) / 4 * 4, -1);
            mh.need(blob.size()); std::memcpy(mh.p, blob.data(), blob.size() * 4);
            ck(cudaMemcpyAsync(maps.p, mh.p, blob.size() * 4, cudaMemcpyHostToDevice, nullptr));
            slots_p = maps.p; hit_p = maps.p + A4; miss_p = maps.p + 2 * A4;
            for (int q = 0; q < 5; ++q) dmap[q] = maps.p + dpos[q];
        }
        auto gate_up = [&](const int * which, int part) {
            auto m = cache.gate(), u = cache.up();
            if (dedup_gu) {
                if (!groups[part]) return;
                if (m.fmt == u.fmt) { m.fmt->expert_matvec_dedup2(m, u, xr.p, eg.p, eu.p, c.expert_ff, dmap[part], groups[part], nullptr); return; }  // gate + up in one launch
                m.fmt->expert_matvec_dedup(m, xr.p, eg.p, c.expert_ff, dmap[part], groups[part], nullptr);
                u.fmt->expert_matvec_dedup(u, xr.p, eu.p, c.expert_ff, dmap[part], groups[part], nullptr);
                return;
            }
            m.fmt->expert_matvec(m, xr.p, eg.p, c.expert_ff, which, n, c.topk, false, nullptr);
            u.fmt->expert_matvec(u, xr.p, eu.p, c.expert_ff, which, n, c.topk, false, nullptr);
        };
        if (hit_n) gate_up(hit_p, 0);
        issue_prefetch();  // after the hit launches: the GPU has work while the host issues copies
        auto m = cache.down();
        if (split_down) {  // miss rows of ax are rewritten below before their down projection reads them
            tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
            m.fmt->expert_matvec_dedup(m, ax.p, ed.p, h, dmap[3], groups[3], nullptr);
        }
        cache.wait_misses();
        if (miss_n) gate_up(miss_p, 1);
        tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
        if (split_down) m.fmt->expert_matvec_dedup(m, ax.p, ed.p, h, dmap[4], groups[4], nullptr);
        else if (dedup_dn) m.fmt->expert_matvec_dedup(m, ax.p, ed.p, h, dmap[2], groups[2], nullptr);
        else m.fmt->expert_matvec(m, ax.p, ed.p, h, slots_p, n, c.topk, true, nullptr);
        if (!cpu_rows.empty()) {
            static cudaEvent_t gpu_done = [] { cudaEvent_t e; ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e; }();
            ck(cudaEventRecord(gpu_done, nullptr));  // after the hits and the miss copies' kernels are queued
            cpu_pool->wait();
            if (miss_driven) {
                const cudaError_t q = cudaEventQuery(gpu_done);
                const bool gpu_busy = q == cudaErrorNotReady;
                if (q != cudaSuccess && !gpu_busy) ck(q);
                cpu_budget = tqllm::flashnext::next_cpu_budget(cpu_budget, gpu_busy, 0, tqllm::flashnext::kCpuMissBudget * std::max(1, cp.rt.block_seqs));  // a batched step has more GPU work to hide the CPU behind
            }
            std::vector<tqllm_cuda::ScatterItem> items;
            for (auto [a, r] : cpu_rows) items.push_back({reinterpret_cast<const char *>(cpu_out + size_t(r) * h), reinterpret_cast<char *>(ed.p + size_t(a) * h), size_t(h) * 2});
            tqllm_cuda::scatter_copy(items.data(), int(items.size()), nullptr, 1);
        }
        tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
        constexpr bool cpu_refill = false;
        if (cpu_refill && !cpu_experts.empty()) {  // the cache still learns the experts the CPU served (copies after this layer)
            cache.set_prefetch_strict(1 << 30); cache.prefetch(layer, cpu_experts, copy_stream);
        }
    } else {
        const auto routed_ids = routed.get(); auto h1 = clk::now();
        if (static FILE * dump = tqllm::debug_cstr("route_dump") ? std::fopen(tqllm::debug_cstr("route_dump"), "ab") : nullptr; dump && trunk) {
            const int32_t hdr[2] = {layer, n}; std::fwrite(hdr, 4, 2, dump); std::fwrite(routed_ids.data(), 4, routed_ids.size(), dump);
        }
        const auto slots = cache.prepare(layer, routed_ids); routed.put(slots); auto h2 = clk::now();
        if (predict) cache.prefetch(layer + 1, next_ids->get(), copy_stream);
        const auto h3 = clk::now();
        g_host_ms[0] += ms(h0, h1); g_host_ms[1] += ms(h1, h2); g_host_ms[2] += ms(h2, h3);
        Device<__half> xr(mixed.count); tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr);
        Device<__nv_bfloat16> eg(size_t(n) * c.topk * c.expert_ff), eu(eg.count), ed(size_t(n) * c.topk * h);
        Device<__half> ax(eg.count);
        // Formats without a grouped expert kernel (MTP's Q8_0 -> I8H): one row-sliced matvec per assignment.
        auto sliced = [&](Mat mat, const __half * x, __nv_bfloat16 * y, int rows, int cols, bool per_assignment) {
            const size_t data = size_t(rows) * mat.fmt->row_bytes(cols), aux = size_t(rows) * mat.fmt->aux_row_bytes(cols);
            for (int a = 0; a < n * c.topk; ++a) {
                Mat v = mat; v.rows = rows;
                v.p = static_cast<const char *>(mat.p) + size_t(slots[a]) * data;
                if (mat.aux) v.aux = static_cast<const char *>(mat.aux) + size_t(slots[a]) * aux;
                v.fmt->matvec(v, x + size_t(per_assignment ? a : a / c.topk) * cols, y + size_t(a) * rows, 1, nullptr);
            }
        };
        auto m = cache.gate();
        // TQFN_NS_DEDUP=0: one block per assignment. Default: one block per unique expert (weights read once per step)
        const bool dd = n > 1 && m.fmt->has_expert_dedup() && cache.up().fmt->has_expert_dedup() && cache.down().fmt->has_expert_dedup();
        // chunks of egemm_min+ tokens: grouped GEMM per cache slot, each expert's
        // weights read once per 64 of its tokens (the dedup kernel reads them once per 4). Prefill only: decode and the batched
        // verify (<= 64 rows) keep the exact dedup path.
        constexpr int egemm_min = 65;  // grouped expert GEMM from 65 rows: above the 64-row batched verify
        auto gemm_ok = [](const Mat & mm) {
            const int id = mm.fmt->u4h_id();
            if (std::string(mm.fmt->name()) == "GSQ_Q2_0") return mm.cols % 64 == 0;
            return (id == 1 || id == 4 || id == 5 || id == 6 || id == 7 || id == 10 || id == 13 || id == 14) && mm.cols % (id == 14 ? 128 : 256) == 0;
        };
        (void) dd;
        // per projection: 0 grouped GEMM, 1 dedup, 2 one block per assignment, 3 row-sliced matvecs (a format without a
        // grouped kernel no longer pulls the layer's other projections onto the slow path)
        // per-slot token lists built on the host with stride = the largest count (not n): the grouped kernels' grids
        // then cover only real tokens (n = 1024 routes ~20 tokens to an expert, not 1024)
        const bool gav = n >= egemm_min;  // the grouped loader reads TQFN_TILED banks too (tests/expert_gemm_tiled.cu)
        int gcap = 1; std::vector<int> hcnt, hgrp;
        if (gav) {
            hcnt.assign(size_t(cache.slots()), 0);
            for (int a = 0; a < n * c.topk; ++a) gcap = std::max(gcap, ++hcnt[size_t(slots[a])]);
            hgrp.assign(size_t(cache.slots()) * gcap, 0); std::vector<int> fill(size_t(cache.slots()), 0);
            for (int a = 0; a < n * c.topk; ++a) hgrp[size_t(slots[a]) * gcap + size_t(fill[size_t(slots[a])]++)] = a;
        }
        Device<int> gcounts(gav ? hcnt.size() : 1), ggroups(gav ? hgrp.size() : 1);
        if (gav) { gcounts.put(hcnt); ggroups.put(hgrp); }
        const Mat mg = cache.gate(), mu = cache.up(), md = cache.down();
        auto mode = [&](const Mat & mm) {
            if (gav && gemm_ok(mm)) return 0;
            if (n > 1 && mm.fmt->has_expert_dedup()) return 1;
            return mm.fmt->has_expert_matvec() ? 2 : 3;
        };
        const int kg = mode(mg), ku = mode(mu), kd = mode(md);

        constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
        const int A = n * c.topk; std::vector<int> hm; int groups[2] = {0, 0};
        if (kg == 1 || ku == 1 || kd == 1) {
            hm.assign(size_t(2) * A * DS, 0);
            for (int part = 0; part < 2; ++part) {  // 0: gate/up (token input), 1: down (assignment input)
                int * out = hm.data() + size_t(part) * A * DS; std::vector<int> open_at(size_t(cache.slots()) + 1, -1); int g = 0;
                for (int a = 0; a < A; ++a) {
                    int & at = open_at[slots[a]];
                    if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = slots[a]; out[at * DS + 1] = 0; }
                    int * e = out + at * DS; const int col = e[1]++;
                    e[2 + col] = part ? a : a / c.topk; e[2 + DC + col] = a;
                }
                groups[part] = g;
            }
        }
        Device<int> dmap(hm.empty() ? 1 : hm.size()); if (!hm.empty()) dmap.put(hm);
        auto run = [&](const Mat & mm, int k, const __half * x, __nv_bfloat16 * y, int rows, int cols, bool per_assignment) {
            if (k == 0) require(mm.fmt->expert_gemm(mm, x, y, rows, gcounts.p, ggroups.p, gcap, cache.slots(), c.topk, per_assignment, nullptr),
                               "expert_gemm unsupported");
            else if (k == 1) mm.fmt->expert_matvec_dedup(mm, x, y, rows, dmap.p + (per_assignment ? size_t(A) * DS : 0), groups[per_assignment], nullptr);
            else if (k == 2) mm.fmt->expert_matvec(mm, x, y, rows, routed.p, n, c.topk, per_assignment, nullptr);
            else sliced(mm, x, y, rows, cols, per_assignment);
        };
        if (kg == 1 && ku == 1 && mg.fmt == mu.fmt) mg.fmt->expert_matvec_dedup2(mg, mu, xr.p, eg.p, eu.p, c.expert_ff, dmap.p, groups[0], nullptr);
        else { run(mg, kg, xr.p, eg.p, c.expert_ff, h, false); run(mu, ku, xr.p, eu.p, c.expert_ff, h, false); }
        tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
        run(md, kd, ax.p, ed.p, h, c.expert_ff, true);
        shared_expert();
        tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
    }
    auto updated = std::make_unique<Device<float>>(size_t(width) * n);
    tqllm_cuda::hc_combine(combined.p, fo.p, inject.p, updated->p, h, c.hc, n, nullptr); residual = std::move(updated);
    end_capture();
    if (ep_rank0) tqllm::flashnext::ep_after_layer(g_ep_fwd, layer, c.layers);
}

void DecodeModel::router(const std::string & prefix, const float * x, float * logits, int n) {
    const bool bf16 = cp.file(prefix + "ffn_gate_inp.weight").file().tensors.at(prefix + "ffn_gate_inp.weight").type == GGML_BF16;
    // prefill chunks: the routed logits as one BF16 tensor-core GEMM (fp32 sums in another order than the per-token kernel:
    // near-tie top-k picks can differ from decode's); the shared gate (one row) stays on the tiled GEMV
    if (bf16 && n > 64) {
        Device<__nv_bfloat16> xb(size_t(n) * h);  // x rounded once (the GEMV kernels round it per product anyway)
        tqllm_cuda::fn_round_bf16(x, xb.p, int64_t(n) * h, nullptr);
        // into packed [n][experts] rows first: the [n][experts + 1] stride leaves cuBLAS an align-1 kernel (0.8 s at 55K)
        const float one = 1.f, zero = 0.f; Device<float> packed(size_t(n) * c.experts);
        require(cublasGemmEx(cp.blas(), CUBLAS_OP_T, CUBLAS_OP_N, c.experts, n, h, &one, cp.bf16_device(prefix + "ffn_gate_inp.weight"), CUDA_R_16BF, h,
                             xb.p, CUDA_R_16BF, h, &zero, packed.p, CUDA_R_32F, c.experts, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS,
                "bf16 router GEMM failed: " + prefix);
        ck(cudaMemcpy2DAsync(logits, size_t(c.experts + 1) * sizeof(float), packed.p, size_t(c.experts) * sizeof(float), size_t(c.experts) * sizeof(float), n,
                             cudaMemcpyDeviceToDevice, nullptr));
        tqllm_cuda::fn_gemv_bf16_chunk_bx(cp.bf16_device(prefix + "ffn_gate_inp_shexp.weight"), xb.p, logits + c.experts, 1, h, n, c.experts + 1, nullptr);
    } else if (bf16) {
        // straight into the [n][experts + 1] logits (no staging copies)
        tqllm_cuda::fn_gemv_bf16_tokens_ld(cp.bf16_device(prefix + "ffn_gate_inp.weight"), x, logits, c.experts, h, n, c.experts + 1, nullptr);
        tqllm_cuda::fn_gemv_bf16_tokens_ld(cp.bf16_device(prefix + "ffn_gate_inp_shexp.weight"), x, logits + c.experts, 1, h, n, c.experts + 1, nullptr);
    } else if (n > 64) {
        // f32 router (the MTP layer): one cuBLAS fp32 GEMM for a prefill chunk instead of the per-token kernel (127 ms per 32K
        // tokens in the catch-up); fp32 sums in another order, and the MTP's routing only shapes the drafts, never the output
        // trunk F32 routers (UD files) from --dense-min tokens: the FP16 copy and x's FP16 image on the tensor cores into packed
        // [n][experts] rows (numeric: near-tie top-k picks can move)
        const float one = 1.f, zero = 0.f; const int ld = c.experts + 1;
        if (cp.half_copy_ok(prefix + "ffn_gate_inp.weight") && tqllm::flashnext::dense_pool_for(1, cp.run.tune.dense_min, n)) {
            Device<__half> xh(size_t(n) * h); tqllm_cuda::vis_f32_to_h(x, xh.p, int64_t(xh.count), nullptr);
            std::unique_ptr<Device<__half>> wh; const __half * w = cp.half_weight(prefix + "ffn_gate_inp.weight", c.experts, h, n, wh);
            Device<float> packed(size_t(n) * c.experts);
            require(cublasGemmEx(cp.blas(), CUBLAS_OP_T, CUBLAS_OP_N, c.experts, n, h, &one, w, CUDA_R_16F, h, xh.p, CUDA_R_16F, h, &zero,
                                 packed.p, CUDA_R_32F, c.experts, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS, "fp16 router GEMM failed: " + prefix);
            ck(cudaMemcpy2DAsync(logits, size_t(ld) * sizeof(float), packed.p, size_t(c.experts) * sizeof(float), size_t(c.experts) * sizeof(float), n,
                                 cudaMemcpyDeviceToDevice, nullptr));
        } else
        require(cublasSgemm(cp.blas(), CUBLAS_OP_T, CUBLAS_OP_N, c.experts, n, h, &one, cp.dense_device(prefix + "ffn_gate_inp.weight"), h,
                            x, h, &zero, logits, ld) == CUBLAS_STATUS_SUCCESS, "f32 router GEMM failed: " + prefix);
        require(cublasSgemm(cp.blas(), CUBLAS_OP_T, CUBLAS_OP_N, 1, n, h, &one, cp.dense_device(prefix + "ffn_gate_inp_shexp.weight"), h,
                            x, h, &zero, logits + c.experts, ld) == CUBLAS_STATUS_SUCCESS, "f32 shared-gate GEMM failed: " + prefix);
    } else {  // decode / verify: the F32 router and shared gate read once per call, all tokens in registers
        tqllm_cuda::fn_gemv_f32_tokens_ld(cp.dense_device(prefix + "ffn_gate_inp.weight"), x, logits, c.experts, h, n, c.experts + 1, nullptr);
        tqllm_cuda::fn_gemv_f32_tokens_ld(cp.dense_device(prefix + "ffn_gate_inp_shexp.weight"), x, logits + c.experts, 1, h, n, c.experts + 1, nullptr);
    }
}

void ep_stop(const Checkpoint & cp) {
    if (cp.run.ep->rank() != 0) return;
    std::lock_guard<std::mutex> g(g_ep_lease.m);
    const int32_t hdr[2] = {kEpStop, 0}; cp.run.ep->ctl_send(hdr, sizeof hdr);
}
void ep_start_reader(const Checkpoint & cp) {
    const std::shared_ptr<tqllm::TpComm> link = cp.run.ep;
    if (!link || link->rank() != 0) return;
    std::thread([link] {  // detached: lives as long as the link
        for (;;) {
            int32_t msg[2] = {0, 0};
            try { link->ctl_recv(msg, sizeof msg); }
            catch (const std::exception &) { std::lock_guard<std::mutex> g(g_ep_lease.m); g_ep_lease.leased = false; return; }  // rank 1 went away
            std::lock_guard<std::mutex> g(g_ep_lease.m);
            if (msg[0] == kEpLease) g_ep_lease.leased = true;
            else if (msg[0] == kEpRevoke) {  // after every forward already sent
                g_ep_lease.leased = false;
                const int32_t end[2] = {kEpLeaseEnd, 0};
                try { link->ctl_send(end, sizeof end); } catch (const std::exception &) { return; }
            }
        }
    }).detach();
}
void ep_grant(const Checkpoint & cp, EpHelperGate & gate) {
    if (!cp.run.ep || cp.run.ep->rank() != 1 || gate.stop || !gate.idle() || !gate.refresh) return;
    { std::lock_guard<std::mutex> l(gate.lm); if (gate.lease_out) return; }
    if (gate.served != gate.banks_checked) {  // banks full, tables current
        if (!gate.refresh()) {  // a request of this box left a bank not full (a long prompt borrowed it): no lease until a later one ends
            static bool said = false;
            if (!said) { said = true; std::fprintf(stderr, "tqllm: --split experts rank 1: a bank is not full after this box's request; rank 0 runs alone until one is\n"); }
            return;
        }
        gate.banks_checked = gate.served;
    }
    { std::lock_guard<std::mutex> l(gate.lm); gate.lease_out = true; }
    try { const int32_t msg[2] = {kEpLease, 1}; std::lock_guard<std::mutex> w(gate.wmu); cp.run.ep->ctl_send(msg, sizeof msg); }
    catch (const std::exception &) { std::lock_guard<std::mutex> l(gate.lm); gate.lease_out = false; }
}
void ep_reclaim(const Checkpoint & cp, EpHelperGate & gate) {
    if (!cp.run.ep || cp.run.ep->rank() != 1) return;
    std::unique_lock<std::mutex> l(gate.lm);
    if (!gate.lease_out) return;
    try { const int32_t msg[2] = {kEpRevoke, 0}; std::lock_guard<std::mutex> w(gate.wmu); cp.run.ep->ctl_send(msg, sizeof msg); }
    catch (const std::exception &) { gate.lease_out = false; return; }
    gate.lcv.wait(l, [&] { return !gate.lease_out; });  // the helper sees kEpLeaseEnd after rank 0's last forward
}
void ep_help(const Checkpoint & cp, MixedExpertCache & cache, std::shared_ptr<EpHelperGate> gate) {
    const std::shared_ptr<tqllm::TpComm> link = cp.run.ep;  // kept alive while this thread waits on it
    if (!link || link->rank() != 1) throw std::runtime_error("--split experts: the helper is rank 1");
    tqllm::TpComm & ep = *link;
    const auto & c = cp.config; const int h = c.hidden, ff = c.expert_ff, layers = c.layers; constexpr int DS = tqllm_cuda::kDedupStride;
    for (int l = 0; l < layers; ++l)
        if (!ep_formats_ok(cache, l)) throw std::runtime_error("--split experts: layer " + std::to_string(l) + " has an expert format without dedup kernels");
    // own stream and buffers sized for the largest forward (rows x topk <= 1024): nothing shared with the serving thread's
    // scratch arena, so this box can serve its own users between the forwards it computes for rank 0
    cudaStream_t s = nullptr; ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const int amax = 1024, nmax = amax / c.topk;
    auto alloc = [](size_t bytes) { void * p = nullptr; ck(cudaMalloc(&p, std::max<size_t>(bytes, 256))); return p; };
    auto * x = static_cast<__half *>(alloc(size_t(nmax) * h * 2));
    auto * ids = static_cast<int *>(alloc(size_t(amax) * 4)), * local = static_cast<int *>(alloc(size_t(amax) * 4));
    auto * maps = static_cast<int *>(alloc(size_t(amax) * (1 + 2 * DS) * 4));
    auto * eg = static_cast<__nv_bfloat16 *>(alloc(size_t(amax) * ff * 2)), * eu = static_cast<__nv_bfloat16 *>(alloc(size_t(amax) * ff * 2));
    auto * ax = static_cast<__half *>(alloc(size_t(amax) * ff * 2));
    auto * ed = static_cast<__nv_bfloat16 *>(alloc(size_t(amax) * h * 2)), * sb = static_cast<__nv_bfloat16 *>(alloc(size_t(amax) * h * 2));
    std::vector<std::vector<int>> host_tables(static_cast<size_t>(layers)); std::vector<int *> tables(static_cast<size_t>(layers), nullptr);
    // the banks this box holds now: every one full (its own prefills may have reshaped them), device slot tables current
    auto banks_full = [&] {
        std::vector<int> slot_of;
        for (int l = 0; l < layers; ++l) {
            if (!cache.full_bank(l, slot_of)) return false;
            if (!tables[size_t(l)]) tables[size_t(l)] = static_cast<int *>(alloc(slot_of.size() * 4));
            if (host_tables[size_t(l)] != slot_of) { host_tables[size_t(l)] = slot_of; ck(cudaMemcpyAsync(tables[size_t(l)], slot_of.data(), slot_of.size() * 4, cudaMemcpyHostToDevice, s)); ck(cudaStreamSynchronize(s)); }
        }
        return true;
    };
    std::fprintf(stderr, "tqllm: --split experts rank 1: computing experts e %% 2 == 1 for rank 0 when this box is idle\n");
    { std::lock_guard<std::recursive_mutex> lk(gate->mu); gate->refresh = banks_full; ep_grant(cp, *gate); }
    auto lease_over = [&] { std::lock_guard<std::mutex> l(gate->lm); gate->lease_out = false; gate->lcv.notify_all(); };
    uint64_t forwards = 0;
    for (;;) {
        int32_t hdr[2] = {0, 0};
        try { ep.ctl_recv(hdr, sizeof hdr); } catch (const std::exception &) { break; }  // rank 0 went away
        if (hdr[0] == kEpStop) break;
        if (hdr[0] == kEpLeaseEnd) { lease_over(); continue; }
        if (hdr[0] != kEpForward || hdr[1] <= 0 || hdr[1] > nmax) { std::fprintf(stderr, "tqllm: --split experts rank 1: bad forward header\n"); break; }
        const int n = hdr[1], A = n * c.topk, cap = tqllm::flashnext::ep_result_cap(A);
        // a forward under the lease: this box has no request of its own (its next one waits in ep_reclaim for the lease end)
        std::lock_guard<std::recursive_mutex> lk(gate->mu);
        for (int layer = 0; layer < layers; ++layer) {
            ep.group_begin(); ep.recv(x, size_t(n) * h * sizeof(__half), 0, s); ep.recv(ids, size_t(A) * sizeof(int), 0, s); ep.group_end();
            const Mat mg = cache.bank_mat(layer, 0), mu = cache.bank_mat(layer, 1), md = cache.bank_mat(layer, 2);
            int * slots_p = maps, * gu = maps + A, * dn = gu + size_t(A) * DS;
            tqllm_cuda::ep_local_ids(nullptr, 0, ids, A, 1, cap, local, s);
            tqllm_cuda::moe_dedup_maps(local, tables[size_t(layer)], A, c.topk, slots_p, gu, dn, A, s);
            if (mg.fmt == mu.fmt) mg.fmt->expert_matvec_dedup2(mg, mu, x, eg, eu, ff, gu, A, s);
            else { mg.fmt->expert_matvec_dedup(mg, x, eg, ff, gu, A, s); mu.fmt->expert_matvec_dedup(mu, x, eu, ff, gu, A, s); }
            tqllm_cuda::fn_swiglu_bf16_h(eg, eu, ax, int64_t(A) * ff, s);
            md.fmt->expert_matvec_dedup(md, ax, ed, h, dn, A, s);
            if (cap > 0) { tqllm_cuda::ep_gather(ed, 0, ids, A, 1, cap, sb, h, s); ep.send(sb, size_t(cap) * h * sizeof(__nv_bfloat16), 0, s); }
        }
        ck(cudaStreamSynchronize(s));  // the forward's work is done before the serving thread may run
        if (++forwards % 1000 == 0) std::fprintf(stderr, "tqllm: --split experts rank 1: %llu forwards computed\n", (unsigned long long) forwards);
    }
    { std::lock_guard<std::recursive_mutex> lk(gate->mu); gate->refresh = nullptr; }  // its tables live on this stack
    lease_over();  // a request of this box waiting for the lease goes on
    std::fprintf(stderr, "tqllm: --split experts rank 1: rank 0 stopped after %llu forwards computed\n", (unsigned long long) forwards);
}
