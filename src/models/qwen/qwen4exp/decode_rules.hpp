#pragma once
// Flash-Next decode decisions that change exactness or memory, as pure functions (no CUDA) so tests/decode_rules.cpp can
// check them on any machine.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <tuple>
#include <cstdlib>
#include <cstddef>
#include <vector>
#include <string>

namespace tqllm::flashnext {

// Decode graphs belong to one DecodeModel (a request's sequence): each model gets a fresh owner id, and its graphs are keyed
// by it and released when the model ends. Keyed by the model's address, a model built where a finished one had lived
// inherited its warm-up calls (its first call captured instead of running plain) and its graph execs, and every finished
// model's execs stayed (G2, the 12 GB np 4 fault hunt).
inline uint64_t next_graph_owner() { static std::atomic<uint64_t> n{0}; return ++n; }
// calls 1-2 of each (owner, layer, rows) run plain, later ones are captured; a plain call (a failed capture) starts over
class GraphWarmup {
    std::map<std::tuple<uint64_t, int, int>, int> calls_;
public:
    bool capture(uint64_t owner, int layer, int n, bool plain) {
        int & c = calls_[{owner, layer, n}];
        if (plain) c = 0;
        return ++c > 2;
    }
    void forget(uint64_t owner) {
        for (auto it = calls_.lower_bound({owner, -(1 << 30), -(1 << 30)}); it != calls_.end() && std::get<0>(it->first) == owner;) it = calls_.erase(it);
    }
    void clear() { calls_.clear(); }
    size_t size() const { return calls_.size(); }
};

// Expert path of one trunk layer call: true = waves (several prepares, grouped per wave), false = one split prepare.
// A prompt chunk (spec off, more than one token) always takes waves. A verify block takes the split path whenever its
// distinct experts fit the layer's slots: the waves path sums the down projection in another order, so choosing it by
// the assignment count made a multi-sequence block (np 4: 4 x 4 rows = 160 assignments) differ from np 1 (F39).
// old_rule = the pre-F39 choice (assignments vs slots), kept for its unit test.
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
// for one, np 4 at 32K ran out of memory (F38). The banks are sized while the warm-up model holds a sequence's buffers,
// so that one is in use already: counting it again left 3.35 GB idle at np 2 with both sequences decoding (3090, 11-02:
// 131 slots / layer vs 160 at np 1, 77-79 % hits vs 91 %, aggregate 60-75 vs 80-95 tok/s).
// Mapping page of a VMM K/V buffer of n bytes: 32 MB pages for long buffers (fewer host moves), but never more than the
// buffer itself rounded to the granule: with a fixed 32 MB page every sequence mapped 26 x 32 MB = 832 MB of K/V however
// short its request (3090 np 2, 11-03: the start-up probe measured 1212 MB per sequence and the keep starved the banks).
inline size_t kv_page_bytes(size_t n, size_t granule) {
    if (!granule) return n;
    const size_t big = std::max(granule, (size_t(32) << 20) / granule * granule);
    return std::min(big, (std::max<size_t>(n, 1) + granule - 1) / granule * granule);
}
// Next-layer prediction (expert prefetch) for a decode block: up to predict_n rows per sequence. Counting the rows of the
// whole block switched prefetch off for every batched step (np 2: 2 x 4 rows > 4): 3090 np 2, 11-03, demand misses
// 20-26 -> 75-105 GB per request, prefetch 14-19 -> 0-4.5 GB, aggregate below np 1. Placement only.
// Predicted experts prefetched per token: ranks 1-3 first, then up to rank 6 over stale victims (one sequence, +3.3 % vs
// all ten, 10-11). Several sequences in a step: ranks 1-3 only - the link is busy 67-78 % at np 2-4 and ~12 % of np 4's
// copies were prefetches evicted unused (3090, 11-03).
struct PrefetchRanks { int high, last; };
// (2 best at np >= 4 measured slower: np 4 86 / 82 vs 89 / 84 tok/s, 067fa3d3)
inline PrefetchRanks prefetch_ranks(int sequences) { return sequences > 1 ? PrefetchRanks{3, 3} : PrefetchRanks{3, 6}; }
inline bool predicts(int rows, int predict_n, int sequences) { return rows <= predict_n * std::max(1, sequences); }
// Per-layer slot caps (manifest layer_slots, scaled to the slot total) with every layer at least min_slots, the slots taken
// from the largest layers so the total stays the same. Several sequences need min_slots = sequences x topk: a batched step
// whose rows route more experts than the smallest layer holds steps one slot at a time (batch_drafts -1); 3090 np 10 / 12
// with pinned layer slots (smallest layers under 100): 37-63 % of steps one slot at a time, 52-58 tok/s against 70 at np 8.
// min_slots above the average cannot hold everywhere: then every layer gets the average (the old uniform split).
inline std::vector<int> lift_layer_floor(std::vector<int> caps, int min_slots, int experts) {
    if (caps.empty() || min_slots <= 0) return caps;
    long total = 0; for (int c : caps) total += c;
    const int L = int(caps.size()), avg = int(total / L);
    const int floor = std::min({min_slots, avg, experts});
    long need = 0; for (int & c : caps) if (c < floor) { need += floor - c; c = floor; }
    while (need > 0) {  // one slot at a time from the currently largest layer (ties: lowest index), never below the floor
        int big = -1; for (int l = 0; l < L; ++l) if (caps[size_t(l)] > floor && (big < 0 || caps[size_t(l)] > caps[size_t(big)])) big = l;
        if (big < 0) break;
        --caps[size_t(big)]; --need;
    }
    return caps;
}
// Drafts per slot for one batched verify step of `active` sequences: the most (up to `drafts`) whose rows fit the batch
// (rows <= max_rows; rows x topk routed experts <= the fewest bank slots of any layer unless every expert is resident:
// no layer may take the waves path, or the text differs from np 1). -1: not even one
// token per slot fits, the slots step one at a time. With the full drafts only, np 8 on a 3090 (51 slots / layer)
// stepped one slot at a time for 259 of 260 steps: 27 tok/s together against 84-100 at np 1 (11-03).
// 4+ sequences verify at most 2 drafts each: GB10 np 8, 8 users, 11-11 (one box, both Flash-Next files, texts = np 1):
// 2 drafts 119 / 124-128 tok/s vs 3 drafts 111-114 / 123-127, 4 drafts 85-94 / 111-113; at 4 users 2 and 3 were level
constexpr int kManyDraftCap = 2, kManyActive = 4;
inline int batch_drafts(int active, int drafts, int topk, int slots, int experts, int max_rows = 64) {
    if (active < 2) return drafts;
    if (active >= kManyActive) drafts = std::min(drafts, kManyDraftCap);
    for (int d = std::max(drafts, 0); d >= 0; --d) {
        const int rows = active * (d + 1);
        if (rows <= max_rows && (slots >= experts || rows * topk <= slots)) return d;
    }
    return -1;
}
// The measured state already holds a sequence's working buffers; the 256 MB working room is one shared margin, as at
// np 1: a working room per sequence left 2.9 GB idle at np 8 with all 8 decoding (3090, 11-03: 120 slots / layer).
inline size_t auto_keep_bytes(size_t sequences, size_t state, size_t reserve) {
    return (std::max<size_t>(sequences, 1) - 1) * state + reserve + (size_t(256) << 20);
}
// The fewest slots a bank keeps: every distinct expert of the largest verify block (drafts + 1 rows of topk), rounded up
// to 8 (5 drafts x top-10: 64; a small card's 3 drafts: 40), so layer_takes_waves keeps the split path at every bank size.
inline int verify_slot_min(int drafts, int topk) { return ((std::max(drafts, 0) + 1) * topk + 7) / 8 * 8; }
// --split experts: result rows per layer exchanged for a forward of `assignments` (rows x topk). About half the assignments
// route to the other rank (stage 0: p50 ~0.5, max ~0.66); 5/8 of them fits nearly every layer, and the rest are computed
// locally (both ranks keep every expert), so the exchange has one fixed size per row count, known on both hosts.
// --split experts, rank 0: the forward being exchanged with rank 1. Each forward is announced at layer 0 and rank 1
// accepts (it then runs exactly `layers` exchanges) or declines (busy with its own users: every layer runs locally). A
// forward that cannot finish its accepted exchanges leaves rank 1 out of step, so expert parallelism then stays off for the
// process (off) and every layer runs locally (every expert resident: same output).
struct EpForwardState { bool open = false, accepted = false, off = false; int next = 0, n = 0; };
enum class EpCall { Local, Exchange, Fail };
// Before a trunk layer (can = this layer may exchange: full bank, kernels for the split; alive = rank 1 is there).
// announce(n) sends layer 0's header and returns rank 1's answer; a plain re-run of a layer 0 whose capture failed is
// not announced again. Fail: an accepted forward lost its exchanges (the request fails).
template<class Announce>
inline EpCall ep_before_layer(EpForwardState & s, int layer, int n, bool can, bool alive, Announce && announce) {
    if (s.off) return EpCall::Local;
    if (!alive) {
        const bool fail = s.open && s.accepted && layer == s.next;
        s = {}; s.off = true; return fail ? EpCall::Fail : EpCall::Local;
    }
    if (layer == 0) {
        if (s.open && s.next == 0 && s.n == n) return s.accepted ? EpCall::Exchange : EpCall::Local;  // the re-run of layer 0
        if (s.open && s.accepted) { s = {}; s.off = true; return EpCall::Local; }  // an accepted forward was left open
        s = {};
        if (!can) return EpCall::Local;
        s.open = true; s.n = n; s.accepted = announce(n);
        return s.accepted ? EpCall::Exchange : EpCall::Local;
    }
    if (!s.open) return EpCall::Local;
    if (layer != s.next || n != s.n || (s.accepted && !can)) {
        const bool fail = s.accepted; s = {}; s.off = fail; return fail ? EpCall::Fail : EpCall::Local;
    }
    return s.accepted ? EpCall::Exchange : EpCall::Local;
}
// After a trunk layer's work (exchanged or local): the open forward moves on; past the last layer it closes
inline void ep_after_layer(EpForwardState & s, int layer, int layers) {
    if (!s.open || layer != s.next) return;
    if (++s.next == layers) { s.open = false; s.accepted = false; s.next = 0; }
}
inline int ep_result_cap(int assignments) { return std::min(std::max(assignments, 0), (5 * std::max(assignments, 0) + 7) / 8); }

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

// Per-layer decode graphs (re-captured every step): they remove launch gaps on unified memory (GB10 +5-6 %) and on PCIe
// cards before Blackwell (3090: chat 86.5 vs 83.9 tok/s without), but on a Blackwell PCIe card the re-capture costs more host
// time than it saves (5090 + EPYC 7543: chat 154 without vs 128 with)
inline bool layer_graphs(bool unified_memory, int cc_major) { return unified_memory || cc_major < 12; }

// Between requests the engine-wide prefill buffers (staging halves, dense fp16 pool, QSA / GDN scratch) and the larger
// prefill arena are handed back so a discrete card's expert banks can use the memory. On unified memory
// (GB10: every expert resident, ~25 GB free in prefill) they stay: freeing and re-allocating them cost each request
// ~0.1-0.3 s (the 3.4K request's arena resize alone 0.12 s). Only while 20 GB stay free (the GB10 margin: a .44 brain at
// np 4 x 131K had 16.3 GB free with them kept after a 24K prompt; below the margin they are handed back as on a 3090).
inline bool keep_prefill_buffers(bool unified_memory, size_t free_bytes) { return unified_memory && free_bytes >= (size_t(20) << 30); }
// the dense FP16 weight pool's size for a block of `tokens` rows: the configured pool from --dense-min tokens, 0 below it
// (the copy then lives in the call's scratch)
inline size_t dense_pool_for(size_t pool_bytes, int dense_min, int tokens) { return dense_min > 0 && tokens >= dense_min ? pool_bytes : 0; }

// A layer whose bank holds every expert (a GB10: 512 slots for 512 experts) loads them all at start-up: filled on first use
// instead, every expert a prompt touched for the first time was a miss inside that prompt's prefill (11-02, .55: the first
// requests paid ~1 s of loads per layer, later ones still up to 85 loads in a layer). A bank smaller than the layer keeps
// its first-use fill (which experts to hold is the routing's call).
inline bool preload_all_experts(int bank_slots, int experts) { return experts > 0 && bank_slots >= experts; }

// QSA raw indexer keys: a ring of the newest cells plus a pinned host mirror on discrete GPUs (~0.9 GB of VRAM back at
// 128K); on unified memory (GB10) pinned host memory is the same memory, so the full array stays (no mirror, no D2H per
// append). Returns the ring's cells (0: the default ring).
inline int qsa_key_ring_cells(bool unified_memory, int max_cells) { return unified_memory ? max_cells : 0; }

// --prompt-cache: prompt tokens a slot reuses. Its last prompt `seq` left its state at `snap` (kTail before that prompt's
// end); a new prompt reuses [0, snap) when it is longer than snap and equals seq on [0, snap] (the token at snap pairs with
// residual snap - 1 in the MTP KV). Anything else prefills from 0.
template<class A, class B> inline int reusable_prefix(const A & seq, int snap, const B & prompt) {
    if (snap <= 0 || prompt.size() <= size_t(snap) || seq.size() <= size_t(snap)) return 0;
    for (size_t i = 0; i <= size_t(snap); ++i) if (int64_t(seq[i]) != int64_t(prompt[i])) return 0;
    return snap;
}
// where the next prompt's state is kept: kTail before this prompt's end, when the new part is at least 2 x kTail
inline int snapshot_at(int prompt, int reused, int tail) { return prompt >= reused + 2 * tail ? prompt - tail : 0; }

// The pinned residual buffer of a multi-chunk prefill (prompt x width floats) is pinned once at start-up when the host
// easily holds it: four times its size available and at most a tenth of RAM. Pinning it per request cost a 32K prompt
// 0.72 s of time to first token (cudaMallocHost of 1.6 GB beside an 80 GB pinned expert pack), more at 128K. Never on
// unified memory (GB10): pinned host memory is the GPU's memory there (6.6 GB left 759 MB after warm-up).
inline bool pin_residual_at_start(size_t bytes, size_t host_ram, size_t available, bool unified_memory) {
    return !unified_memory && bytes > 0 && bytes <= available / 4 && bytes <= host_ram / 10;
}

// Host K/V for positions older than TQFN_KV_HOST_KEEP (32768): on by default for discrete GPUs with VMM pages (RTX 3090,
// 256K: 52-56 -> 65-71 tok/s; contexts up to 2 x 32K move nothing), off on unified memory (GB10: host and device memory are
// the same). env (TQFN_KV_HOST) "0" / "1" forces it either way.
inline bool kv_host_on(const char * env, bool vmm, bool unified) {
    if (env) return std::atoi(env) != 0 && vmm;
    return vmm && !unified;
}

// Approximate tail skip (opt-in; Strata's default route-tail-skip): per assignment of routed (tokens x topk ids), the
// replacement expert or -1 to keep. An expert whose best rank over the block is R-th or lower and that is not resident goes
// to its token's first choice (rank 0, always computed) with weight 0. Any block size (the first version stopped at 4 tokens,
// so 5-draft verify blocks never skipped).
template <class Resident>
inline std::vector<int> tail_skip_targets(const std::vector<int> & routed, int topk, int r, Resident resident) {
    std::vector<int> out(routed.size(), -1);
    if (r <= 0 || topk <= 0) return out;
    auto best = [&](int e) { int b = topk; for (size_t a = 0; a < routed.size(); ++a) if (routed[a] == e) b = std::min(b, int(a % size_t(topk))); return b; };
    for (size_t a = 0; a < routed.size(); ++a) {
        const int e = routed[a];
        if (best(e) < r - 1 || resident(e)) continue;
        out[a] = routed[a / size_t(topk) * size_t(topk)];
    }
    return out;
}

// Adaptive draft length: draft again only while the last draft's top probability reaches tau (tau <= 0: fixed count).
inline bool keep_drafting(float tau, float last_draft_probability) { return !(tau > 0.f && last_draft_probability < tau); }

// Copy drafts: the continuation (up to max_len tokens) after the latest earlier occurrence of the context's last n
// tokens, searched in the context itself, then in earlier requests (latest first); empty when none. Exact: the verify
// keeps only what the model itself emits, so only the tokens per step change.
inline std::vector<int> copy_draft(const std::vector<int> & ctx, const std::vector<const std::vector<int> *> & history, int n, int max_len) {
    if (n <= 0 || max_len <= 0 || int(ctx.size()) < n) return {};
    const int * tail = ctx.data() + ctx.size() - size_t(n);
    auto search = [&](const std::vector<int> & s, int last_start) -> std::vector<int> {
        for (int i = last_start; i >= 0; --i) {
            if (!std::equal(tail, tail + n, s.begin() + i)) continue;
            const int b = i + n, e = std::min(int(s.size()), b + max_len);
            if (b < e) return std::vector<int>(s.begin() + b, s.begin() + e);
        }
        return {};
    };
    if (auto d = search(ctx, int(ctx.size()) - n - 1); !d.empty()) return d;
    for (const auto * h : history) if (h && int(h->size()) > n) if (auto d = search(*h, int(h->size()) - n - 1); !d.empty()) return d;
    return {};
}

// Rows a decode layer call may have for the CPU expert pool to serve its misses (its pinned input / output buffers).
constexpr int kCpuPoolRows = 4;
// Discrete cards (--exact off, miss-driven pool): any decode block up to batch_drafts' max_rows, so batched np > 1 steps
// use the pool too (with the small-card limit of 4 rows np 2+ never did: every miss went over PCIe).
constexpr int kCpuPoolMaxRows = 64;
// The pool's row budget for a decode layer call, 0 when no pool runs (decode_rules.hpp cpu_miss_threads). The miss-driven
// pool on a small card keeps the 4-row block it was measured with; a discrete card takes any block (layout only, exact).
inline int cpu_pool_rows(int pool_threads, bool small_card) { return pool_threads <= 0 ? 0 : (small_card ? kCpuPoolRows : kCpuPoolMaxRows); }

// One sequence's draft count. Without a CPU expert pool: adaptive length, up to 5 drafts (3090 chat +0.6 %, 32K +3 %).
// With one (small cards): the verify block must fit the pool's rows, or every miss of the step goes over PCIe instead
// (12 GB 3090: 5 drafts 44.8 / 44.6 tok/s, 3 drafts 49-50). pool_rows = the pool's row budget (0: no pool).
// Every expert resident (GB10): a longer block costs no copies, so up to 7 drafts (an 8-row block still runs in the layer
// graphs); GB10 np 1 at 5 drafts ended 27 of 123 steps with every draft kept.
struct DraftPlan { int drafts; bool adaptive; };
inline DraftPlan one_sequence_drafts(int manifest_drafts, int pool_rows, bool every_expert_resident = false) {
    if (manifest_drafts < 1) return {manifest_drafts, false};
    if (pool_rows > 0) return {std::min(manifest_drafts, std::max(pool_rows - 1, 1)), false};
    return {std::max(manifest_drafts, every_expert_resident ? 7 : 5), true};
}
// One sequence after slots=auto sized the banks: every expert resident (GB10: 512 of 512) drafts as a pinned slots=512
// manifest does; the count chosen at load came before the slots were known (GB10 decode 104 instead of 122 tok/s).
inline int auto_slot_drafts(int manifest_drafts, int chosen, int slots, int experts, int pool_rows) {
    if (pool_rows > 0 || slots < experts) return chosen;
    return std::max(chosen, one_sequence_drafts(manifest_drafts, 0, true).drafts);
}
// The plan a single sequence's request actually gets: the pool's row limit applies only while the pool serves the misses
// (cpu_miss_threads: --exact off, or TQFN_CPU_MISS above 0). A 16 GB card that must run exact (107 of 512 experts
// resident) keeps the adaptive block of one_sequence_drafts (11-11: 1.2K / 8.5K / 25.3K decode 78.6 / 77.0 / 66.6 ->
// 89.3 / 87.2 / 72.0 tok/s, the same 192 greedy tokens; the pool's 3-draft cap was what held it to 4 rows).
inline DraftPlan sequence_drafts(int manifest_drafts, bool exact, int env_threads, int cores, bool every_expert_resident = false);

// CPU-owned experts (small cards, F26): a fixed eighth of each layer's experts, a pure function of (layer, expert), runs
// on the CPU expert pool and is kept out of the GPU banks; every other expert runs on the GPU, resident or not. Which
// device computes an expert (AVX2 fp32 dots vs the GPU kernels: last-bit differences that flip a near-tie) then does not
// depend on what is resident, and neither does the greedy text (12 GB: the pool computing whatever missed made the same
// 16K prompt twice differ). Owned experts in the banks only took slots decode never used (ec0cf00: -2 tok/s).
inline bool cpu_owned(int layer, int expert) { return (uint32_t(expert) * 2654435761u + uint32_t(layer) * 40503u) >> 29 == 0; }
// Miss-driven (discrete cards, --exact off): whatever is not resident runs on the CPU pool, up to cpu_miss_budget distinct
// experts per layer call, so its bytes never cross PCIe (Strata: ~93 % of misses on the CPU). The static partition above
// made the pool compute a fixed eighth of all experts, hot ones included, while the other misses still went over PCIe:
// 3090 np 1, 14 threads, 11-03: 71-86 vs 89-105 tok/s without the pool, 40 vs 45 GB of copies per 384 tokens. Which
// device computes an expert then depends on residency: not reproducible run to run (F26), acceptable only off exact.
inline bool runs_on_cpu(int layer, int expert, bool resident, bool pool_serves_call, bool miss_driven = false) {
    if (!pool_serves_call) return false;
    return miss_driven ? !resident : cpu_owned(layer, expert);
}
// Distinct missed experts per decode layer call the pool takes (the rest are PCIe misses): what it finishes while the GPU
// runs the hits. 5950X, 14 threads: ~0.1 ms per expert (tests/cpu_miss_perf); a 3090 decode layer call ~0.55 ms (np 1,
// 4-row verify, ~26 ms per step over 48 layers): 4 experts.
// CPU miss pool threads: TQFN_CPU_MISS (env_threads >= 0) wins; else --exact off runs the pool on every card with
// min(15, cores / 2 - 1) threads, --exact on runs none (11-12: running the owned set under --exact on a 16 GB card cut the
// 25.3K decode 72.0 -> 63.6 tok/s, so it stays off; see REPORT.md round 3)
inline int cpu_miss_threads(int env_threads, bool exact, int cores) {
    if (env_threads >= 0) return env_threads;
    return exact ? 0 : std::max(1, std::min(15, cores / 2 - 1));
}
inline DraftPlan sequence_drafts(int manifest_drafts, bool exact, int env_threads, int cores, bool every_expert_resident) {
    // the 4-row block is the measured rule for one sequence's pool on any card (a 12 GB 3090: 5 drafts 44.8 / 44.6 tok/s,
    // 3 drafts 49-50); the layer call keeps its own budget (kCpuPoolMaxRows on a discrete card)
    const int threads = cpu_miss_threads(env_threads, exact, cores);
    return one_sequence_drafts(manifest_drafts, cpu_pool_rows(threads, true), every_expert_resident);
}
// the pool reads missed experts from the pinned packed copy whenever it runs (11-06: the 3090 np 8 --exact off rows fell from
// 76 / 77 to 68 / 70 tok/s when the pool came from --exact off alone and read the GGUF mapping instead)
inline bool cpu_pool_reads_packed(int env_threads, bool exact, int cores) { return cpu_miss_threads(env_threads, exact, cores) > 0; }
// Per-layer expert-slot profile of Qwen3.8-Flash-Next (48 layers, 512 experts), relative: scale_layer_slots fits it to the
// card's slot budget. Measured routing popularity (the brain manifests' layer_slots, 10-27). Used when a manifest gives no
// layer_slots (11-06: Unsloth UD-IQ3_XXS on a 3090 with uniform slots 130.6 / 85.4 tok/s, with this profile 143.3 / 92.4;
// prompt 8K-32K +1 to +3 %): the routing belongs to the model, not to the file.
inline const char * flashnext_layer_profile(int layers, int experts) {
    return layers == 48 && experts == 512 ? "314,291,269,160,189,219,248,239,208,135,212,160,227,227,172,128,128,145,227,169,164,197,236,141,152,134,182,137,189,167,167,128,128,135,213,143,153,170,226,128,129,140,161,131,140,140,176,165" : "";
}
constexpr int kCpuMissBudget = 4;
// The next layer call's CPU budget from whether the GPU was still busy (its hits and the miss copies) when the pool
// finished: the CPU is worth its misses while it finishes inside the GPU's own work for the call. One expert more after a
// call the CPU finished first, one less after a call the GPU had to wait for; 0 sends every miss over PCIe until a probe
// call (every probe_every calls) tries one expert again. The 11-03 rule compared the pool's finish with the host's time
// to queue the GPU work (microseconds), so the share collapsed: 3090 np 8 --exact off 1.8 experts per call, 65-71 tok/s;
// this rule 7.5 experts per call, PCIe -31 %, 76-78 tok/s (Strata 0.2.11: 71-74).
inline int next_cpu_budget(int budget, bool gpu_still_busy, int calls_at_zero, int max_budget = kCpuMissBudget,
                           int probe_every = 64) {
    if (budget <= 0) return calls_at_zero >= probe_every ? 1 : 0;
    if (!gpu_still_busy) return budget - 1;
    return std::min(max_budget, budget + 1);
}
// Staging slices (experts per double-buffered half) for one request on a small card: 64 instead of 128 for a prompt the
// engine's own chunk already covers. The smaller slices are 325 MB less plan room, and on a one-chunk prompt that room goes
// to expert banks that then stay resident (fewer experts streamed: 1.2K TTFT 2.22 -> 1.91 s, 8.5K 4.37 -> 4.29 s). A
// prompt longer than the engine chunk keeps the 128-expert slices: its plan is the measured one, and a smaller plan there
// lets a bigger QSA block fit, which changes the tokens (the prefill's chunk and block size are not text invariant).
inline int small_card_stage(bool small_card, bool stage_set, int stage, int prompt_tokens, int engine_chunk, int cap = 64) {
    if (!small_card || stage_set) return stage;
    return prompt_tokens <= engine_chunk ? std::min(stage, cap) : std::min(stage, 128);
}
// Positions per MTP catch-up piece (one-chunk prompts): the MTP layer's bank holds 64 of 512 experts, so a piece streams the
// MTP experts it routes to that the bank does not hold (one 1.9 MB expert each), and every piece routes to nearly all of
// them: the traffic is the MTP expert set once per piece. 1024-position pieces stream it 25 times for a 25K prompt (1.5 s of
// its TTFT), 4096-position pieces 7 times; the piece's buffers (mtp_prefill's concatenated rows, ~82 KB a position) set the
// ceiling. The MTP's routing shapes the drafts only, never the emitted tokens (verify is exact), so this is a speed knob.
inline int mtp_piece_tokens(int tokens, int cap = 4096) { return std::max(1, std::min(tokens, cap)); }
// Parking the LM heads (pinning a 634 MB copy, ~0.9 s with cudaMallocHost, and the unpark another ~0.2 s) for a prompt longer
// than the engine chunk is a probe: the room it frees can turn the prompt into one chunk and halve the expert bytes. Worth
// it only when that room can actually reach the one-chunk plan: at 16 GB a 25.3K prompt that needs 12.4 GB with 11.1 GB
// free paid 1.1 s of TTFT for a probe that could not succeed (11-12). plan_need = the one-chunk plan's need, room = what
// the plan sees free, heads = the bytes parking hands back, cached = what a trim of the async pool would add.
inline bool park_heads_for_one_chunk(size_t plan_need, size_t room, size_t heads, size_t cached) {
    if (!heads || !plan_need) return true;  // nothing known (a non-resident or composite head): keep the probe
    return plan_need <= room + heads + cached;
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
// Build the packed experts in pinned memory only on a discrete GPU without a pack file, outside hot/cold (its tiers manage
// expert memory) and when the pack fits in 60 % of the machine's RAM; otherwise the cache reads experts from the GGUF on
// demand. Hot/cold on .231 (23 GiB RAM) died pinning 51 GB.
inline bool build_pack_in_memory(bool has_pack_file, bool integrated_gpu, bool hot_cold, double pack_bytes, double host_ram_bytes) {
    return !has_pack_file && !integrated_gpu && !hot_cold && pack_bytes <= 0.6 * host_ram_bytes;
}
// Pages that may be dropped once the experts sit in pinned memory: the trunk expert tensors only. The per-layer token
// embedding (Flash-Next: 28.8 GB, one 90-byte row per token and layer) is read from the file at every token; with its
// pages dropped each new token waited on the disk (3090: chats 79-88 instead of 94-103 tok/s, 32K first token +2.7 s).
inline bool drop_after_pack(const std::string & tensor) {
    const auto exps = tensor.find("_exps.weight");
    return tensor.rfind("blk.", 0) == 0 && exps != std::string::npos && exps + 12 == tensor.size();
}
// Read at start-up so the first requests do not fault it in from the disk, when it fits half the RAM: on a small host
// (.231: 23 GiB, 28.8 GB embedding) the read only cycles the page cache and evicts the cold experts.
inline bool warm_at_start(const std::string & tensor, double tensor_bytes, double host_ram_bytes) {
    return tensor == "per_layer_token_embd.weight" && tensor_bytes <= 0.5 * host_ram_bytes;
}
// Prefill chunk: the engine's --prefill-chunk setting (default 16384), whether or not the manifest names one; the
// manifest value counts only when that setting is off. A manifest without chunk= once prefilled in chunks of 8 (32K: 263 s).
inline int prefill_chunk(int tune_chunk, int manifest_chunk) { return tune_chunk > 0 ? tune_chunk : manifest_chunk; }
// A small card's prefill chunk (every chunk streams every expert once, so the plan takes the largest that fits): the
// prompt's own length, rounded up to 1 KiB, never above max_chunk. A prompt shorter than the engine's chunk gets its own
// length: the experts stream either way, and the 16K chunk's activation and staging buffers (6.5 GB at 1.2K tokens) are
// only taken from the expert banks, which then have to be refilled after the prompt (11-11: 1.2K TTFT 2.41 -> 2.20 s).
inline int small_card_chunk(int prompt_tokens, int max_chunk) {
    if (prompt_tokens <= 0) return 0;
    const int up = (prompt_tokens + 1023) / 1024 * 1024;
    return std::min(max_chunk, std::max(up, 1024));
}
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
