#pragma once
// Flash-Next start-up memory plan as pure functions (no CUDA): what a full context keeps beside the expert banks, the
// slots=auto expert slot count and its tiers, the draft count once the slots are known, and the full-context fit check.
// fn_engine_load.cpp measures free memory and the one-sequence state on the device and calls these; tests/memory_plan.cpp
// pins them with the numbers of real starts (3090 128K, 3090 --np 8, GB10).
#include "models/qwen/qwen4exp/decode_rules.hpp"
#include "models/qwen/qwen4exp/prefill_plan.hpp"
#include "experts/expert_bank_plan.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tqllm::flashnext {

constexpr size_t kMiB = size_t(1) << 20;

struct MemShape {
    int attention_layers = 0;  // layers with QSA indexer keys
    int kv_heads = 0, head_dim = 0, index_dim = 0;
    int experts = 0, topk = 0;
};

struct MemRun {
    int context = 0;             // tokens per slot
    bool kv_host = false;        // K/V of positions older than kv_host_keep move to host memory in decode
    int kv_host_keep = 32768;
    size_t sequences = 1;        // --np
    bool small_card = false;     // under 16 GB free with the CPU expert pool
    bool caching = false;        // --prompt-cache: every slot keeps its full context
    int manifest_drafts = 3;
};

struct ContextKeep { size_t state = 0, reserve = 0; };

// The device memory a full context keeps beside the banks: its attention state (less the K/V that lives in host memory
// in decode) and the decode reserve. 2 GB reserve past 64K (the 245K cells at 64 / 72 slots failed with 1 GB; decode
// scratch, the MTP bank rebuilt after a multi-chunk prompt, prefill staging and refill staging). Up to 64K: 1 GB for one
// sequence (3090, 1.4.3 at --context 65536: 256 MB left 537 MB free after warm-up and 31 of 200 GSM8K decodes of up to
// 4096 tokens failed "out of memory" with ~90 MB free), 256 MB for several (their keep already holds a measured state per
// extra sequence; 11-03 np 2 chat decode kept 1.05-1.15 GB free; np 8 at 8K per slot keeps 145 slots / layer).
inline ContextKeep context_keep(const MemShape & s, const MemRun & r) {
    const size_t kv_row = size_t(2) * size_t(s.kv_heads) * size_t(s.head_dim) * 2;  // K + V bf16 per position
    const size_t moved = r.kv_host && r.context > 2 * r.kv_host_keep
        ? size_t(s.attention_layers) * size_t(r.context - 2 * r.kv_host_keep) * kv_row : 0;
    const size_t full = tqllm::prefill_context_bytes(r.context, 16, s.attention_layers, s.kv_heads, s.head_dim, s.index_dim);
    return {full > moved ? full - moved : 0, (r.context > 65536 ? size_t(2048) : r.sequences > 1 ? size_t(256) : size_t(1024)) * kMiB};
}

// One sequence's attention K/V at `context` positions: per attention layer the bf16 K and V rows, the indexer's raw keys
// (fp32) and its pooled block scores (the per-layer part of prefill_context_bytes, without the shared QSA scratch).
inline size_t sequence_kv_bytes(const MemShape & s, int context, int ratio = 4) {
    const size_t ctx = size_t(std::max(0, context));
    return size_t(std::max(0, s.attention_layers)) *
           (2 * ctx * size_t(s.kv_heads) * size_t(s.head_dim) * 2 + ctx * size_t(s.index_dim) * 4 + ctx / size_t(std::max(1, ratio)) * 4);
}

// The state counted per sequence in the slots=auto keep: several sequences without the prompt cache keep what one short
// sequence measured (a request's K/V is sized by the request: a request that starts short of room shrinks the banks
// first, request_shrink_bytes); otherwise the full-context estimate, or the measurement when it is larger.
inline size_t keep_state(const MemShape & s, const MemRun & r, size_t state_estimate, size_t measured_state) {
    (void) s;
    return r.sequences > 1 && measured_state && !r.caching ? measured_state : std::max(state_estimate, measured_state);
}

// Several sequences: the bytes the banks must give back before a request of `tokens` positions (prompt + max_tokens)
// starts, so its K/V and `state` (what a new sequence holds beside its K/V: PLE histories, GDN state, MTP; 0 when the
// slot keeps its sequence) fit beside the reserve; 0 when free memory already holds it.
inline size_t request_shrink_bytes(const MemShape & s, size_t tokens, size_t free, size_t reserve, size_t state = 0) {
    const size_t need = sequence_kv_bytes(s, int(tokens)) + state + reserve;
    return free >= need ? 0 : need - free;
}
struct SlotPlan {
    size_t keep = 0;      // bytes kept free beside the banks (incl. the measured rebuild extra)
    int slots = 0;        // per layer
    bool tiers = false;   // banks sized to what is free after a long prompt (one sequence)
    int tier_a = 0;       // slots with a full context's state kept (reported)
    size_t tier_keep = 0; // what a refill leaves free
    int verify_min = 0;   // floor per layer: every distinct expert of the largest verify block
    int drafts = 0;       // one sequence: drafts once the slot count is known
    bool adaptive = false;
};

// slots=auto: free = device memory free with no bank, extra = what rebuilding the banks took beyond the slots (measured
// by the engine's rounds), slot_bytes = one slot in every layer. Drafts: one sequence drafts adaptively (up to 5, or 7
// when every expert is resident), small cards keep the CPU pool's block; several sequences keep the manifest's count
// (batch_drafts trims them per step).
inline SlotPlan auto_slot_plan(const MemShape & s, const MemRun & r, size_t free, size_t slot_bytes, size_t measured_state,
                               size_t extra = 0) {
    SlotPlan p;
    const ContextKeep ck = context_keep(s, r);
    const size_t state = keep_state(s, r, ck.state, measured_state);
    p.keep = (r.small_card ? 512 * kMiB : auto_keep_bytes(r.sequences, state, ck.reserve)) + extra;
    p.slots = tqllm::auto_expert_slots(free, p.keep, slot_bytes, s.experts);
    p.drafts = r.manifest_drafts;
    if (r.sequences == 1) {
        const DraftPlan d = one_sequence_drafts(r.manifest_drafts, r.small_card, false);
        p.drafts = auto_slot_drafts(r.manifest_drafts, d.drafts, p.slots, s.experts, r.small_card);
        p.adaptive = d.adaptive;
    }
    p.tiers = slot_tiers(r.sequences, false);
    if (p.tiers) {
        p.tier_a = std::min(p.slots, tqllm::auto_expert_slots(free, state + ck.reserve + 256 * kMiB + extra, slot_bytes, s.experts));
        p.tier_keep = (r.small_card ? 512 : 256) * kMiB;
        p.verify_min = verify_slot_min(p.drafts, s.topk);
    }
    return p;
}

// After the banks are built: a full context must fit beside them. slots=auto (and small cards) keep only the reserve on
// purpose and let a long prompt borrow banks; a fixed slot count that cannot hold the context is refused with the most
// slots that would. Returns -1 when it fits or borrowing covers it, else the largest slot count to suggest.
inline int fit_max_slots(const ContextKeep & k, size_t free_after_banks, bool slots_auto, bool small_card, int slots,
                         size_t bank_bytes) {
    if (free_after_banks >= k.state + k.reserve || slots_auto || small_card) return -1;
    const size_t per_slot = bank_bytes / size_t(std::max(1, slots));
    const int drop = per_slot ? int((k.state + k.reserve - free_after_banks + per_slot - 1) / per_slot) : slots;
    return std::max(0, slots - drop);
}

}  // namespace tqllm::flashnext
