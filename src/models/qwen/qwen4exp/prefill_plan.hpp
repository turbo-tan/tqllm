#pragma once
// Flash-Next layer-major prefill: the memory plan.
//
// PrefillPlan is a pure function of the request and the engine parameters: how many device bytes each prefill buffer
// takes, what the chunk's peak is, and how many expert banks the request must borrow (release) before its first kernel.
// The byte-to-slot arithmetic and the ledger that owns the borrow are shared with every engine that keeps expert banks and
// live in src/experts/expert_bank_plan.hpp; this file keeps what is Flash-Next's own buffer layout. CUDA-free, so
// tests/prefill_plan.cpp can check it on any machine.
#include "expert_bank_plan.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {

struct PrefillShape {
    int hidden = 0, hc = 0, expert_ff = 0, topk = 0, experts = 0, layers = 0;
    size_t expert_block_bytes = 0;  // one staged expert (gate + up + down, data + aux) in the bank layout
    bool wide_bf16 = false;         // DecodeModel::prefill_wide_bf16: BF16 wide rows beside res_keep
};

struct PrefillTuning {
    int chunk = 8192, sub = 2048, stage = 192;
    size_t arena_bytes = 0, dense_cache_bytes = 0;  // --prefill-arena-mb, --dense-cache-mb
    size_t workspace_bytes = size_t(1024) << 20;    // attention / GDN / QSA scratch the kernels allocate on first use
    size_t mtp_bank_bytes = 0;                      // the MTP layer's expert bank, built in the catch-up after the trunk
    size_t context_bytes = 0;                       // prefill_context_bytes(): the attention state the request allocates at first use
    size_t context_bytes_per_token = 0;             // extra per-token headroom on top of it (0: none)         // attention / indexer state that grows with the prompt, not the chunk (the retired
                                                    // heuristic reserved 60 MB per 1K tokens for the same reason; 128K failed at 4 and 16 KB/token;
                                                    // 124 MiB allocation failed with 87 MiB free on 10-17 when the plan had no such term)
};

// Device bytes of one prefill chunk of n tokens (the per-chunk buffers of DecodeModel::prefill_layer and its staging).
struct PrefillChunkBytes {
    size_t tokens = 0, activations = 0, expert_io = 0, staging = 0, residual = 0, arena = 0, dense_cache = 0, workspace = 0;
    size_t total() const { return activations + expert_io + staging + residual + arena + dense_cache + workspace; }
};

inline size_t align256(size_t b) { return (b + 255) / 256 * 256; }
// Bytes one layer-major staging half needs for K experts of a bank whose six pieces (gate/up/down data + aux) are `piece`
// (each piece's K copies 256-aligned, as DecodeModel::stage_offsets lays them out). The MTP layer's experts can be larger
// than the trunk's (IQ4 MTP beside a UD-IQ1_M trunk), so every upload checks this, not only the routed path (11-06).
inline size_t stage_need(const size_t piece[6], int K) { size_t o = 0; for (int i = 0; i < 6; ++i) o += align256(size_t(K) * piece[i]); return o; }
// Hyper-connection norm of a chunk: more than 64 tokens keep only row scales and a BF16 image (the mix recomputes the
// normalized rows), valid only when the mixer projections reading them (down, inject) are BF16 and so take that image;
// a quantized down / inject reads the fp32 rows.
inline bool hc_row_scales_only(int tokens, bool down_bf16, bool inject_bf16) { return tokens > 64 && down_bf16 && inject_bf16; }
// the FP16 row-scale path: the norm writes an FP16 image the mixer's projections read through FP16 copies of their weights
// (quantised, BF16 or F32 mixers alike: one stacked down + inject GEMM, an FP16 up); resident GPU projections only. It
// takes BF16 mixers too (before the BF16 row-scale path above) and F32 inject (UD files)
inline bool hc_row_scales_h(int tokens, bool mixers_have_fp16_copies, bool gpu_projections) {
    return tokens > 64 && mixers_have_fp16_copies && gpu_projections;
}
// The layer-major expert passes' assignment lists: pass p holds W slots (slot_of[e] = the slot of expert e, -1 when the
// expert is not in that pass); at out + at[p]: W counts, then W lists of up to cap assignment indices, each ascending.
// One sweep over the A routed ids for all passes (an expert sits in at most one pass).
struct PassSlots { int W, cap; size_t at; const int * slot_of; };
inline void fill_pass_lists(const int * ids, int A, int E, const std::vector<PassSlots> & passes, int * out) {
    std::vector<int> pass_of(size_t(E), -1);
    for (size_t p = 0; p < passes.size(); ++p) {
        const auto & ps = passes[p]; std::fill(out + ps.at, out + ps.at + ps.W, 0);
        for (int e = 0; e < E; ++e) if (ps.slot_of[e] >= 0) pass_of[size_t(e)] = int(p);
    }
    for (int a = 0; a < A; ++a) {
        const int e = ids[a], p = pass_of[size_t(e)]; if (p < 0) continue;
        const auto & ps = passes[size_t(p)]; int * cn = out + ps.at, * gr = cn + ps.W; const int sl = ps.slot_of[e];
        const int k = cn[sl]++; gr[size_t(sl) * ps.cap + size_t(k)] = a;
    }
}
// the prefill scratch arena a chunk needs (per-layer temporaries of the dense sub-chunks and attention, without the QSA
// block scores the caller adds; measured 10-17: 2880 MB high-water at 16384 tokens with 2048-query blocks, ~2340 MB
// without their scores; 3519+ MB short at 32768 with 3800). 0.15 MB per token + 128 MB; an overflow falls back to the pool.
inline size_t prefill_arena_bytes(int chunk) { return (size_t(std::max(0, chunk)) * 154 / 1024 + 128) << 20; }
// The attention state a request allocates lazily during its first chunk, all sized by the context: per attention layer
// the bf16 K and V rows, the indexer's raw keys (fp32) and pooled scores; plus one QSA batch scratch shared by the layers
// (sort keys and sorted keys for sub-chunk queries x context/ratio blocks, CUB temp of the same order, selected positions).
inline size_t prefill_context_bytes(int context, int sub, int attention_layers, int kv_heads, int head_dim, int index_dim, int ratio = 4, int top_k = 2048) {
    const size_t ctx = size_t(std::max(0, context)), blocks = ctx / size_t(std::max(1, ratio));
    const size_t per_layer = 2 * ctx * size_t(kv_heads) * size_t(head_dim) * 2 + ctx * size_t(index_dim) * 4 + blocks * 4;
    const size_t items = size_t(std::max(16, sub)) * blocks, width = size_t(std::min<int64_t>(int64_t(ctx), int64_t(top_k) + ratio - 1));
    // lean scratch (QsaBatchScratch::lean: top-k only): 32-bit score bits + 1024 sorted keys per query; else sort keys,
    // sorted keys and CUB temp of the same order
    const bool lean = (width - 1) / size_t(std::max(1, ratio)) + 1 <= 1024 && blocks >= 1024;
    const size_t sel = lean ? items * 4 + size_t(std::max(16, sub)) * 1024 * 8 : items * 8 * 2 + items * 16;
    const size_t scratch = sel + size_t(std::max(16, sub)) * (width * 5 + 4 * 128 * 4);
    return size_t(std::max(0, attention_layers)) * per_layer + scratch;
}

inline PrefillChunkBytes prefill_chunk_bytes(const PrefillShape & s, const PrefillTuning & t, int prompt_tokens) {
    if (s.hidden <= 0 || s.hc <= 0 || s.expert_ff <= 0 || s.topk <= 0 || t.chunk < 256 || t.stage < 1)
        throw std::invalid_argument("prefill_chunk_bytes: invalid shape or tuning");
    PrefillChunkBytes b;
    const size_t n = size_t(std::min(prompt_tokens, t.chunk)), A = n * size_t(s.topk), h = size_t(s.hidden), ff = size_t(s.expert_ff);
    b.tokens = n;
    // pf_inject f32 [n hc], pf_weights f32 [A], pf_sw f32 [n], pf_xr half [n h], pf_sd f32 [n h], pf_routed int [A],
    // pf_idx int (pass tables, bounded by 8 A), q8_1 tiles of the gate/up input (144 bytes per 128 values)
    b.activations = n * size_t(s.hc) * 4 + A * 4 + n * 4 + n * h * 2 + n * h * 4 + A * 4 + 8 * A * 4 + n * ((h + 127) / 128) * 144;
    // pf_eg, pf_eu bf16 [A ff], pf_ax half [A ff], pf_ed bf16 [A h]
    b.expert_io = 3 * A * ff * 2 + A * h * 2;
    b.staging = 2 * align256(size_t(t.stage) * s.expert_block_bytes);
    b.residual = n * h * size_t(s.hc) * (s.wide_bf16 ? 6 : 4);  // res_keep: the chunk's wide residual rows (+ their BF16 copy)
    b.arena = std::max(t.arena_bytes, prefill_arena_bytes(t.chunk)); b.dense_cache = t.dense_cache_bytes; b.workspace = t.workspace_bytes;
    return b;
}

// The device room a prefill plan may use. `budget` is what the engine measured once at start with every buffer in place and
// the allocator's pool trimmed (free + expert bank bytes, rounded down to 256 MB); it is the plan's input on purpose: with
// the free memory of the moment the same prompt length took one chunk of 25600 or two of 21504 depending on what the
// previous request left behind (+-1 s of TTFT, 11-12), and since the chunk boundary decides the tokens, the text followed
// it. `have` is what the session offers now: free memory + the banks that are still resident + the cache's loader staging
// + the request's own terms (the MTP bank a multi-chunk prompt drops before the trunk, the LM heads a parked prompt frees).
// When the session cannot cover the budget at all (a leftover allocation no bank release can give back) the plan uses
// `have` and `from_budget` is false, so the engine can say so in its log.
struct PlanRoom { size_t room = 0, have = 0; bool from_budget = false; };
inline PlanRoom plan_room(size_t budget, size_t free_now, size_t banks, size_t staging, size_t mtp = 0, size_t heads = 0) {
    PlanRoom r; r.have = free_now + banks + staging + mtp + heads;
    const size_t want = budget + mtp + heads;
    if (budget && r.have >= want) { r.room = want; r.from_budget = true; } else r.room = r.have;
    return r;
}

// The scratch arena a chunk and its selection block need: the measured per-token rule (prefill_arena_bytes) plus one
// block's QSA score scratch and headroom.
inline size_t prefill_arena_req(size_t arena_prefill, int chunk, int block, int attention_heads, size_t headroom = size_t(64) << 20) {
    return std::min<size_t>(arena_prefill, prefill_arena_bytes(chunk)) + size_t(std::max(0, block)) * size_t(std::max(0, attention_heads)) * 2052 * 8 + headroom;
}

// The whole request: the trunk's chunk peak, the MTP catch-up that follows (trunk buffers released first), and how many
// bank bytes must be released before the trunk starts.
struct PrefillPlan {
    PrefillChunkBytes chunk;
    int chunks = 0;
    size_t trunk_peak = 0, catchup_peak = 0;
    size_t need = 0;         // device bytes the request must have free before its first kernel (peak + the prompt-length term)
    size_t release = 0;      // bank bytes to release (0: fits in the free memory as is)
    size_t host_copies = 0;  // expert bytes streamed from host over the whole prompt (all experts once per layer-chunk)
};

inline PrefillPlan prefill_plan(const PrefillShape & s, const PrefillTuning & t, int prompt_tokens, size_t free_bytes) {
    if (prompt_tokens <= 0) throw std::invalid_argument("prefill_plan: no tokens");
    PrefillPlan p;
    p.chunk = prefill_chunk_bytes(s, t, prompt_tokens);
    p.chunks = (prompt_tokens + t.chunk - 1) / t.chunk;
    p.trunk_peak = p.chunk.total();
    // the catch-up runs the MTP layer over the prompt in 1024-position pieces: its bank, one piece of residual rows, staging
    p.catchup_peak = t.mtp_bank_bytes + size_t(std::min(prompt_tokens, 1024)) * size_t(s.hidden) * size_t(s.hc) * 4 +
                     p.chunk.staging + p.chunk.arena + t.workspace_bytes;
    p.need = std::max(p.trunk_peak, p.catchup_peak) + t.context_bytes + size_t(prompt_tokens) * t.context_bytes_per_token;
    p.release = p.need > free_bytes ? p.need - free_bytes : 0;
    p.host_copies = size_t(p.chunks) * size_t(s.layers) * size_t(s.experts) * s.expert_block_bytes;
    return p;
}

// The chunk and selection block a prompt gets, and whether parking the LM heads is what would buy the whole prompt in one
// chunk: a pure function of the prompt length, the plan's room and the sizes fixed at start. The engine halves the
// selection block first (2048 -> 1024 -> 512, ~0.5 s at 32K) and then the chunk by 7/8 (every chunk streams every expert
// once, so the largest chunk that fits is the cheapest). The park probe runs once the chunk has halved below the prompt: a
// pinned 634 MB head copy costs ~1.1 s of TTFT, so it pays only when the heads' room makes the whole prompt one chunk.
// A plan whose need is above `room` for every candidate is returned at the smallest chunk, where the caller's plan reports
// it (nothing fits: the request fails as it did before).
struct PrefillChoice { int chunk = 0, block = 0; bool park_heads = false; size_t need = 0; };
using PrefillTuningFn = std::function<PrefillTuning(int chunk, int block)>;
inline PrefillChoice prefill_choice(const PrefillShape & s, const PrefillTuningFn & tune_for, int prompt_tokens, size_t room,
                                    size_t mtp_bytes, size_t heads, int chunk0, int block0, int min_chunk = 2048) {
    if (prompt_tokens <= 0 || chunk0 < min_chunk) throw std::invalid_argument("prefill_choice: no tokens or no chunk");
    PrefillChoice c; c.chunk = chunk0; c.block = block0;
    const auto need_of = [&](int chunk, int block) {
        try { return prefill_plan(s, tune_for(chunk, block), prompt_tokens, 0).need; } catch (const std::exception &) { return size_t(0); }
    };
    std::vector<int> blocks{block0};
    if (block0 > 1024) blocks.push_back(1024);
    if (block0 > 512) blocks.push_back(512);
    bool probed = heads == 0;  // no resident LM heads to park: no probe
    for (int chunk = chunk0;;) {
        if (!probed && prompt_tokens > chunk) {
            probed = true;
            const size_t one = need_of(chunk0, block0);
            if (one && one <= room + heads) { c.park_heads = true; c.chunk = chunk0; c.block = block0; c.need = one; return c; }
        }
        for (int block : blocks) {
            const size_t need = need_of(chunk, block);
            if (need && need <= room + (prompt_tokens > chunk ? mtp_bytes : 0)) { c.chunk = chunk; c.block = block; c.need = need; return c; }
        }
        if (chunk <= min_chunk) break;
        chunk = std::max(min_chunk, int(size_t(chunk) * 7 / 8 / 1024 * 1024));
    }
    return c;
}

// the plan's own two numbers, as the shared bank arithmetic names them
inline int banks_to_borrow(const PrefillPlan & p, const std::vector<size_t> & bank_bytes_lru, BorrowMode mode, int count = 0) {
    return banks_to_borrow(p.release, bank_bytes_lru, mode, count);
}
inline std::vector<int> borrow_banks(BankLedger & ledger, const PrefillPlan & p, BorrowMode mode, int count = 0,
                                     const std::vector<int> & bankless = {}) {
    return ledger.borrow(p.release, mode, count, p.need, bankless);
}

}  // namespace tqllm
