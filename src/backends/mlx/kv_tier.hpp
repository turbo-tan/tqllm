#pragma once
#include <algorithm>
// KV cache format per request, chosen from memory: BF16 (exact reference, the default) while weights + the request's KV
// + the prefill's transient fit the device's recommended working set with a margin, else int8 (per (kv head, key) FP16
// scales; Splash's default), else trellis (tcq3 K / tcq2 V, kv_tcq.hpp). One format is stored, no shadow copy. The
// numbers are the request's: rows (prompt + new tokens + a verify block, Qwen35::reserve_kv), the engine's attention
// layers / kv heads / head dim, the memory in use now (weights, drafter) and the device's recommended working set.
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace tqllm::mlx_backend {
enum class KvTier {Bf16 = 0, Int8 = 1, Trellis = 2};
inline const char* kv_tier_name(KvTier t) {return t == KvTier::Bf16 ? "bf16" : t == KvTier::Int8 ? "int8" : "trellis";}
// bytes per (row, kv head) for K + V
inline size_t kv_row_bytes(KvTier t, int head_dim) {
    if (t == KvTier::Bf16) return size_t(head_dim) * 2 * 2;
    if (t == KvTier::Int8) return size_t(head_dim) * 2 + 2 * 2;   // int8 K + V, FP16 scale each
    return 104 + 72;                                              // trellis records (head_dim 256)
}
struct KvBudget {size_t in_use = 0, working_set = 0, transient = 0;};
// the working set a request can really use: the GPU's recommended size, but no more than what this process holds plus
// what the system has free (free, inactive, purgeable, speculative pages). Other apps' memory is not ours (j1164: BF16
// 121K fit the recommended 40.2 GB on paper and paged, 21.0 tok/s vs int8 32-34)
inline size_t usable_working_set(size_t recommended, size_t ours, size_t available) {
    return available == 0 ? recommended : (ours + available < recommended ? ours + available : recommended);
}
// what the system can hand us without swapping, in pages: free, speculative and purgeable pages, and the inactive pages
// that are file backed (clean cache, dropped for free). Other apps' anonymous inactive pages are not ours: counting them
// (j1164 / j1166 / j1168: 16-20 GB "inactive" on a Mac holding 23 GB in the compressor) let BF16 121K in, and it paged
// at 21-26 tok/s against int8's 32
inline size_t available_memory(size_t free, size_t inactive, size_t speculative, size_t purgeable, size_t file_backed) {
    return free + speculative + purgeable + (inactive < file_backed ? inactive : file_backed);
}
// transient: what the prefill adds besides the cache (chunk activations, the buffer cache, one layer's FP16 prefix copy)
inline bool kv_fits(KvTier t, int64_t rows, int attn_layers, int kvheads, int head_dim, const KvBudget& b) {
    const size_t kv = size_t(rows) * size_t(attn_layers) * size_t(kvheads) * kv_row_bytes(t, head_dim);
    const size_t prefix = size_t(rows) * size_t(kvheads) * size_t(head_dim) * 2 * 2;  // one layer's FP16 K + V for prefill
    return b.in_use + kv + (t == KvTier::Bf16 ? 0 : prefix) + b.transient <= b.working_set;
}
// no margin on top: the working set already stops at what the system can hand us (available_memory); the margin
// (j1137) covered the over-counted "available" that let other apps' pages in
inline KvTier choose_kv_tier(int64_t rows, int attn_layers, int kvheads, int head_dim, const KvBudget& b,
                             KvTier best = KvTier::Bf16) {
    for (KvTier t : {KvTier::Bf16, KvTier::Int8, KvTier::Trellis}) {
        if (int(t) < int(best)) continue;
        if (kv_fits(t, rows, attn_layers, kvheads, head_dim, b)) return t;
    }
    return KvTier::Trellis;
}
// the first format a request tries for speed, memory still decides below it: int8 from 32K keys up (its verify attention
// reads half the bytes: step -2 ms at 32K, -6 ms at 64K, 128K +10 %, 256K +28 %, j1224 / j1225 / j1231), BF16 under that
// (int8 costs ~1 ms a step at short prompts; at 8K / 16K the step is unchanged and only the texts move: doc8k -7 %, 8K +3 %,
// 16K +17 %, one prompt each)
inline KvTier speed_kv_tier(int64_t rows) {return rows >= 32768 ? KvTier::Int8 : KvTier::Bf16;}
// a request's first format follows its prompt plus at most 1K answer rows: max_tokens is a cap (server default 16384),
// counting all of it would send every 16K+ chat that omits it to int8
inline KvTier request_kv_first(int64_t prompt_rows, int64_t answer_rows) {return speed_kv_tier(prompt_rows + std::min<int64_t>(answer_rows, 1024));}
// the drafter's far-context caches (far_row_bytes per row) are worth less than the KV format they would cost: when they
// would push the request from BF16 / int8 down to trellis, the request runs without them (j1158 248K: trellis with them
// 13.7 tok/s, j1154 int8 without them 21.0)
struct KvPlan {KvTier tier = KvTier::Bf16; bool far = false, lean = false;};
inline KvPlan plan_kv_with(int64_t rows, int attn_layers, int kvheads, int head_dim, KvBudget b, size_t far_row_bytes,
                           KvTier best = KvTier::Bf16) {
    const KvTier alone = choose_kv_tier(rows, attn_layers, kvheads, head_dim, b, best);
    b.in_use += size_t(rows) * far_row_bytes;
    const KvTier with = choose_kv_tier(rows, attn_layers, kvheads, head_dim, b, best);
    if (far_row_bytes == 0) return {alone, false};
    // the caches go when they cost a format, or when not even trellis fits with them
    if (with == KvTier::Trellis && (alone != KvTier::Trellis || !kv_fits(with, rows, attn_layers, kvheads, head_dim, b))) return {alone, false};
    return {with, true};
}
inline bool plan_fits(const KvPlan& p, int64_t rows, int attn_layers, int kvheads, int head_dim, KvBudget b, size_t far_row_bytes) {
    if (p.far) b.in_use += size_t(rows) * far_row_bytes;
    return kv_fits(p.tier, rows, attn_layers, kvheads, head_dim, b);
}
// lean_transient: the prefill transient with a small buffer cache (slower prefill, j786 ~-10 % at 8K). Used only when it
// buys int8 over trellis or keeps the far caches (j1165 248K: trellis 11.7 tok/s with the 6 GB transient). It never buys
// BF16 over int8: BF16's 2x KV bytes on a machine that tight page (j1166 121K: BF16 lean 25.9 tok/s vs int8 32.2)
inline KvPlan plan_kv(int64_t rows, int attn_layers, int kvheads, int head_dim, const KvBudget& b, size_t far_row_bytes,
                      size_t lean_transient = 0, KvTier first = KvTier::Bf16) {
    const KvPlan p = plan_kv_with(rows, attn_layers, kvheads, head_dim, b, far_row_bytes, first);
    if (p.tier == first && (p.far || far_row_bytes == 0)) return p;
    if (lean_transient == 0 || lean_transient >= b.transient) return p;
    KvBudget lb = b; lb.transient = lean_transient;
    const KvTier best = p.tier == KvTier::Bf16 ? KvTier::Bf16 : KvTier::Int8;
    KvPlan q = plan_kv_with(rows, attn_layers, kvheads, head_dim, lb, far_row_bytes, best);
    const bool p_fits = plan_fits(p, rows, attn_layers, kvheads, head_dim, b, far_row_bytes);
    const bool q_fits = plan_fits(q, rows, attn_layers, kvheads, head_dim, lb, far_row_bytes);
    if (int(q.tier) < int(p.tier) || (q.tier == p.tier && q.far && !p.far) || (!p_fits && q_fits)) {q.lean = true; return q;}
    return p;
}
}  // namespace tqllm::mlx_backend
