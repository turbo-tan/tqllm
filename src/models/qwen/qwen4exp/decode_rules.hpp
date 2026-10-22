#pragma once
// Flash-Next decode decisions that change exactness or memory, as pure functions (no CUDA) so tests/decode_rules.cpp can
// check them on any machine.
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace tqllm::flashnext {

// Expert path of one trunk layer call: true = waves (several prepares, grouped per wave), false = one split prepare.
// A prompt chunk (spec off, more than one token) always takes waves. A verify block takes the split path whenever its
// distinct experts fit the layer's slots: the waves path sums the down projection in another order, so choosing it by
// the assignment count made a multi-sequence block (np 4: 4 x 4 rows = 160 assignments) differ from np 1 (F39).
// old_rule = the pre-F39 choice (assignments vs slots), kept behind TQFN_VERIFY_WAVES for A/B.
inline bool layer_takes_waves(const std::vector<int> & routed, int n, int topk, int slots, bool spec, bool old_rule = false) {
    if (!(n * topk > slots || (!spec && n > 1))) return false;
    if (!spec || n <= 1 || old_rule) return true;
    std::vector<int> u(routed); std::sort(u.begin(), u.end());
    return std::unique(u.begin(), u.end()) - u.begin() > slots;
}

// Device bytes slots=auto keeps free beside the expert banks. One sequence keeps only the decode reserve and 256 MB of
// working room: a long prompt borrows banks for its attention state and the refill after it stops short of the last
// 512 MB (keeping its full-context state free cost 3090 chats ~16 slots per layer, 85.9 vs 95.7 tok/s at 140 vs 157).
// Several sequences (--np) each keep their own full-context attention / recurrent / MTP state and working room: sized
// for one, np 4 at 32K ran out of memory (F38).
inline size_t auto_keep_bytes(size_t sequences, size_t state, size_t reserve) {
    if (sequences <= 1) return reserve + (size_t(256) << 20);
    return sequences * state + reserve + sequences * (size_t(256) << 20);
}
// The fewest slots a bank keeps: every distinct expert of the largest verify block (drafts + 1 rows of topk), rounded up
// to 8 (5 drafts x top-10: 64; a small card's 3 drafts: 40), so layer_takes_waves keeps the split path at every bank size.
inline int verify_slot_min(int drafts, int topk) { return ((std::max(drafts, 0) + 1) * topk + 7) / 8 * 8; }

// A layer's slot floor (tier A) from its target: scaled like the slot total (tier_a of tier_b per layer on average), and
// never below min_slots (verify_slot_min, F39), so the text does not depend on how much a long prompt took.
inline int slot_floor(int target, int tier_a, int tier_b, int min_slots = 64) {
    if (tier_b <= 0 || target <= 0) return target;
    return std::min(target, std::max(min_slots, int(int64_t(target) * tier_a / tier_b)));
}

// Slot tiers (a bank sized to what is free, the free memory shared by every layer a refill rebuilds) for one sequence,
// small cards included: without them a 12 GB refill after a 16K prompt rebuilt full banks down to its keep and left 4 of
// 48 layers without one (every expert of those layers a PCIe miss: decode 9.5 tok/s instead of ~48).
inline bool slot_tiers(size_t sequences, bool keep_override) { return sequences == 1 && !keep_override; }

// What regrow() at a request's start keeps free: the base room plus the request's own state, scaled from one sequence's
// full-context state (a long request after a long request keeps the smaller banks instead of rebuilding them unchanged).
inline size_t regrow_keep_bytes(size_t base, size_t state_full, int state_tokens, size_t tokens) {
    if (state_tokens <= 0) return base;
    return base + size_t(double(state_full) * double(std::min(tokens, size_t(state_tokens))) / double(state_tokens));
}

// Adaptive draft length: draft again only while the last draft's top probability reaches tau (tau <= 0: fixed count).
inline bool keep_drafting(float tau, float last_draft_probability) { return !(tau > 0.f && last_draft_probability < tau); }

// Rows a decode layer call may have for the CPU expert pool to serve its misses (its pinned input / output buffers).
constexpr int kCpuPoolRows = 4;

// One sequence's draft count. Without a CPU expert pool: adaptive length, up to 5 drafts (3090 chat +0.6 %, 32K +3 %).
// With one (small cards): a verify block must fit the pool's rows, or every miss of the step goes over PCIe instead
// (12 GB 3090: 5 drafts 44.8 / 44.6 tok/s, 3 drafts 49-50).
struct DraftPlan { int drafts; bool adaptive; };
inline DraftPlan one_sequence_drafts(int manifest_drafts, bool cpu_pool) {
    if (manifest_drafts < 1) return {manifest_drafts, false};
    if (cpu_pool) return {std::min(manifest_drafts, kCpuPoolRows - 1), false};
    return {std::max(manifest_drafts, 5), true};
}

// CPU-owned experts (small cards, F26): a fixed eighth of each layer's experts, a pure function of (layer, expert), runs
// on the CPU expert pool and is kept out of the GPU banks; every other expert runs on the GPU, resident or not. Which
// device computes an expert (AVX2 fp32 dots vs the GPU kernels: last-bit differences that flip a near-tie) then does not
// depend on what is resident, and neither does the greedy text (12 GB: the pool computing whatever missed made the same
// 16K prompt twice differ). Owned experts in the banks only took slots decode never used (ec0cf00: -2 tok/s).
inline bool cpu_owned(int layer, int expert) { return (uint32_t(expert) * 2654435761u + uint32_t(layer) * 40503u) >> 29 == 0; }
inline bool runs_on_cpu(int layer, int expert, bool resident, bool pool_serves_call) {
    (void) resident;
    return pool_serves_call && cpu_owned(layer, expert);
}
// The experts a layer's bank is built with: the most routed first (ties by id), at most cap, owned experts left out when
// a CPU pool computes them.
inline std::vector<int> bank_experts(const std::vector<int> & routed_count, int layer, size_t cap, bool cpu_pool) {
    std::vector<int> top;
    for (int e = 0; e < int(routed_count.size()); ++e) if (routed_count[size_t(e)] && !(cpu_pool && cpu_owned(layer, e))) top.push_back(e);
    std::stable_sort(top.begin(), top.end(), [&](int a, int b) { return routed_count[size_t(a)] > routed_count[size_t(b)]; });
    if (top.size() > cap) top.resize(cap);
    return top;
}

// Packed experts (one contiguous block per expert in the decode cache's row layout, so a cache miss is one copy): read
// from the manifest's experts file when it names one; otherwise built in pinned memory from the GGUF at start-up on a
// discrete GPU, so the released GGUF is the only expert source to download. A GPU that shares system memory (GB10)
// keeps every expert resident and needs no pack.
inline bool build_pack_in_memory(bool has_pack_file, bool integrated_gpu) { return !has_pack_file && !integrated_gpu; }
// Pages that may be dropped once the experts sit in pinned memory: the trunk expert tensors only. The per-layer token
// embedding (Flash-Next: 28.8 GB, one 90-byte row per token and layer) is read from the file at every token; with its
// pages dropped each new token waited on the disk (3090: chats 79-88 instead of 94-103 tok/s, 32K first token +2.7 s).
inline bool drop_after_pack(const std::string & tensor) {
    const auto exps = tensor.find("_exps.weight");
    return tensor.rfind("blk.", 0) == 0 && exps != std::string::npos && exps + 12 == tensor.size();
}
// Read at start-up so the first requests do not fault it in from the disk
inline bool warm_at_start(const std::string & tensor) { return tensor == "per_layer_token_embd.weight"; }
// Prefill chunk: the engine's --prefill-chunk setting (default 16384), whether or not the manifest names one; the
// manifest value counts only when that setting is off. A manifest without chunk= once prefilled in chunks of 8 (32K: 263 s).
inline int prefill_chunk(int tune_chunk, int manifest_chunk) { return tune_chunk > 0 ? tune_chunk : manifest_chunk; }
// Byte cap of the expert banks. slots=auto sizes the banks from free device memory itself, so its cap is the device; a
// fixed budget_gb (taken from one card) capped a GB10's 512 resident slots at 18 GB and evicted layers mid-prompt.
inline size_t bank_budget_bytes(bool slots_auto, double budget_gb, size_t device_total) {
    return slots_auto ? device_total : size_t(budget_gb * 1024.0 * 1024.0 * 1024.0);
}
// Per-layer expert slots from a plan (counts per layer, any scale): scaled to slots x layers, each in [1, experts].
// When every expert fits (slots >= experts, e.g. a GB10) the plan is a cache plan for smaller cards: every layer keeps all.
inline std::vector<int> scale_layer_slots(std::vector<int> caps, int slots, int experts) {
    if (slots >= experts) return std::vector<int>(caps.size(), experts);
    long total = 0; for (int c : caps) total += c;
    const long want = long(slots) * long(caps.size());
    if (total != want && total > 0)
        for (int & c : caps) c = std::max(1, std::min(experts, int(double(c) * double(want) / double(total) + 0.5)));
    return caps;
}

}  // namespace tqllm::flashnext
