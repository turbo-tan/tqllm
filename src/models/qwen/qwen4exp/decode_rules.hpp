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

// Layer-major prefill without a prepacked pack: the routed experts of a layer go through its bank (prepare loads the
// missing ones) only while they fit in it. Experts read from the GGUF on demand (host_blocks) stream the rest through the
// prefill staging like a pack does: a bank cannot take more experts than its slots (a 28K prompt routes to all 512
// experts of a layer with ~200 slots: "mixed cache: active set exceeds slots"), and a borrowed layer has no bank at all.
// Without either (GB10, every expert resident) the bank holds them all. With a pack only the resident ones use the bank.
inline bool prefill_through_bank(bool prepacked, bool host_blocks, bool bank_ready, int routed, int slots) {
    if (prepacked) return false;
    if (!host_blocks) return true;
    return bank_ready && routed <= slots;
}
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
// Mapping page of a VMM K/V buffer of n bytes: 2 MB (the 3090's granule, 2K positions of one layer's K or V), never more
// than the buffer itself rounded to the granule. 32 MB pages (11-03 .. 11-19) mapped every long buffer up to the next 32K
// positions and kept 32-64K positions on the device after a long prompt: 26 x 64 MB = 1664 MB of K/V for a 32K request
// that needs 936 MB, and 1792 MB at 128K for a 32K window (3090, VRAM account 11-19), memory the expert banks refill into.
// (A fixed 32 MB page also mapped 832 MB per short sequence: 3090 np 2, 11-03.)
constexpr size_t kKvPageBytes = size_t(2) << 20;
inline size_t kv_page_bytes(size_t n, size_t granule) {
    if (!granule) return n;
    const size_t page = std::max(granule, kKvPageBytes / granule * granule);
    return std::min(page, (std::max<size_t>(n, 1) + granule - 1) / granule * granule);
}
// After a long prompt the pages wholly older than the newest `keep` rows go to host memory: [0, kv_host_front) of a buffer
// with rows_per_page rows per page (decode reads them through the gather); the device keeps keep .. keep + one page of rows.
inline size_t kv_host_front(size_t len, size_t keep, size_t rows_per_page) {
    return rows_per_page && len > keep ? (len - keep) / rows_per_page : 0;
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
// Rows of one batched decode / verify step, the one limit the drafts rule, the decode arena (fn_engine_load) and the layer
// graphs (layer_graph_rows) share: the arena was sized for 102 rows the rule never makes (np 32 x 3: ~1.2 GB unused)
constexpr int kBatchRowsMax = 64;
inline int batch_drafts(int active, int drafts, int topk, int slots, int experts) {
    const int max_rows = kBatchRowsMax;
    if (active < 2) return drafts;
    if (active >= kManyActive) drafts = std::min(drafts, kManyDraftCap);
    for (int d = std::max(drafts, 0); d >= 0; --d) {
        const int rows = active * (d + 1);
        if (rows <= max_rows && (slots >= experts || rows * topk <= slots)) return d;
    }
    return -1;
}
// One sequence with both drafters: DFlash2's full chain wins code (2x4090 split 11-14: json 224 vs MTP 181 tok/s), MTP
// wins prose (chat 120 vs 112, 112 vs 98). Per request, the drafter with the higher recent tokens per second drafts; the
// other is tried for kPickProbe steps every kPickEvery steps so its rate stays current. Both keep their state fed.
// Measured 11-19 (h2h, tok/s json / python / btree / chat1 / chat2 / 32K), this pick vs forced DFlash2 vs no drafter: one GB10
// 87.7 / 89.1 / 69.4 / 43.5 / 43.5 / 50.5 vs 92.1 / 96.8 / 64.8 / 33.9 / 35.5 / 56.7 vs 85.9 / 84.5 / 72.3 / 46.9 / 45.7 / 47.3;
// one RTX 3090 the pick loses to no drafter everywhere (114 / 103 / 108 / 90 / 86 / 80 vs 120 / 107 / 113 / 96 / 94 / 88). A rule
// on decayed sums with rare probes for a far-behind drafter gained 1-4 % on the 3090 but lost python / 32K on the GB10 (82.7 /
// 48.2): reverted. Open: pick by accept length x each drafter's own step cost (instrumented), not wall time.
struct DrafterPick { double rate[2] = {0, 0}; int seen[2] = {0, 0}; int step = 0, cur = 0; };
constexpr int kPickWarm = 4, kPickEvery = 32, kPickProbe = 2;
constexpr double kPickAlpha = 0.25;
inline int pick_drafter(const DrafterPick & p) {  // 0 = DFlash2, 1 = MTP
    if (p.step < kPickWarm) return 0;
    if (p.step < 2 * kPickWarm) return 1;
    const int best = p.rate[1] > p.rate[0] ? 1 : 0;
    return (p.step % kPickEvery) >= kPickEvery - kPickProbe ? 1 - best : best;
}
inline void drafter_done(DrafterPick & p, int which, int tokens, double seconds) {
    const double r = tokens / std::max(seconds, 1e-6);
    p.rate[which] = p.seen[which]++ ? (1 - kPickAlpha) * p.rate[which] + kPickAlpha * r : r;
    p.cur = which; ++p.step;
}
// The measured state already holds a sequence's working buffers; the 256 MB working room is one shared margin, as at
// np 1: a working room per sequence left 2.9 GB idle at np 8 with all 8 decoding (3090, 11-03: 120 slots / layer).
inline size_t auto_keep_bytes(size_t sequences, size_t state, size_t reserve) {
    return (std::max<size_t>(sequences, 1) - 1) * state + reserve + (size_t(256) << 20);
}
// The fewest slots a bank keeps: every distinct expert of the largest verify block (drafts + 1 rows of topk), rounded up
// to 8 (5 drafts x top-10: 64; a small card's 3 drafts: 40), so layer_takes_waves keeps the split path at every bank size.
inline int verify_slot_min(int drafts, int topk) { return ((std::max(drafts, 0) + 1) * topk + 7) / 8 * 8; }
// rows of a decode / verify step whose maps the device can build with every expert resident: the device dedup maps take at
// most 1024 assignments (moe_dedup_maps)
inline int graph_rows_max(int topk) { return 1024 / std::max(topk, 1); }
// rows a decode / verify layer runs as a graph: every row of a batch (kBatchRowsMax, within graph_rows_max; the decode arena
// is sized for as many, so nothing in a captured layer falls back to a pool allocation) with every expert resident (device
// maps), else 8 (the split path waits on the routed ids on the host); more rows run eager
inline int layer_graph_rows(bool full_slots, int topk) { return full_slots ? std::min(kBatchRowsMax, graph_rows_max(topk)) : 8; }
// --np limit: 16 (each layer keeps np x topk slots); 32 when the manifest pins every expert resident (slots >= the expert
// count, 512 for Flash-Next): no layer runs short of slots, and 32 is the batched kernels' sequence limit
inline int np_max(bool slots_auto, int slots, int experts) { return !slots_auto && slots >= experts ? 32 : 16; }
// The batched attention's split workspace (rows x heads x splits of 128 positions x (dim + 2) floats, fn_dense_attention_
// multi_workspace) stays within the 64 MB the one-sequence path blocks its queries to; above it the sequences take that
// path. Flash-Next's attention layers all carry a QSA indexer, which keeps the batch below 2,048 positions; a file
// without one would reach ~25 MB a row at 128K inside a captured step.
constexpr size_t kAttnWorkspaceMax = size_t(64) << 20;
inline bool attention_multi_fits(int rows, int heads, int dim, int longest) {
    const size_t splits = size_t((std::max(longest, 1) + 127) / 128);
    return size_t(std::max(rows, 0)) * size_t(std::max(heads, 0)) * splits * size_t(dim + 2) * sizeof(float) <= kAttnWorkspaceMax;
}
// A decode / verify call builds its slot and dedup maps on the device (no routed-id readback, the layer's graph runs to
// its end) when its bank holds every expert: rows up to graph_rows_max, whatever the slot count. Bounding the assignments
// by the bank's slots as well stopped a 512-slot top-10 bank at 51 rows (np 18-21 at 3 rows, np 26-32 at 2 ran eager
// with a readback per layer).
inline bool device_maps(int n, int topk, int bank_slots, int experts, bool spec) {
    return bank_slots >= experts && (spec || n == 1) && n <= graph_rows_max(topk);
}
// A one-sequence decode / verify layer whose maps are built on the device (every expert resident, no host wait in the layer)
// runs the work that only reads the layer's inputs on a parallel branch (fn_runtime.hpp SideBranch): the shared expert beside
// the router and the routed experts, and in GDN layers the gate / alpha / beta projections beside the qkv projection and the
// conv. Same kernels on the same inputs. Not with several sequences, a prefill sink or --split experts (their own orders).
inline bool side_branches(bool device_maps_on, bool several_seqs, bool prefill_sink, bool expert_exchange) {
    return device_maps_on && !several_seqs && !prefill_sink && !expert_exchange;
}
// The MTP layer's catch-up after each decode step feeds only the next step's drafts: a server that never drafts (drafts=0
// and no DFlash2 drafter) skips it (the plain step then runs the trunk and the head only).
inline bool mtp_after_step(int drafts, bool dflash) { return drafts > 0 || dflash; }
// A trunk call reads the routed ids back for the waves rule (layer_takes_waves) when a block's assignments outnumber the
// bank's slots or a prompt chunk runs; a bank holding every expert never takes waves, so it never reads them.
inline bool reads_routes_for_waves(bool hot_map, bool device_maps_on, int n, int topk, int slots, bool spec) {
    return hot_map || (!device_maps_on && (n * topk > slots || (!spec && n > 1)));
}
// Host state a decode layer step changes before its captured kernels run: the GDN history / state double buffers swap,
// buffers are made on first use (their zero fill is captured too), the attention lengths advance (the QSA keys follow
// them: the next call truncates to len). A captured part that fails has executed nothing, so its plain rerun
// (layer_forward) starts from what this saved; rerunning from the swapped buffers gave silently wrong state.
template<class Ptr> struct StepUndo {
    struct Pair { Ptr * a, * b; const void * a0, * b0; };
    std::vector<Pair> pairs; std::vector<std::pair<Ptr *, bool>> made; std::vector<std::pair<int *, int>> ints;
    void swapped(Ptr & a, Ptr & b) { pairs.push_back({&a, &b, a.get(), b.get()}); }
    void first_use(Ptr & p) { made.push_back({&p, bool(p)}); }
    void value(int & v) { ints.push_back({&v, v}); }
    void restore() {
        for (auto & p : pairs) {
            if (p.a->get() != p.a0 && p.b->get() == p.a0) std::swap(*p.a, *p.b);
            if (!p.b0) p.b->reset();
            if (!p.a0) p.a->reset();
        }
        for (auto & [p, had] : made) if (!had) p->reset();
        for (auto & [p, v] : ints) *p = v;
    }
};

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
// Experts a bank lent to a prompt loads again at the refill before the first decode step: the bank is rebuilt at its
// slot count (the same memory as before, so the next prompt's plan does not change), but a discrete card loads one of
// its held experts and decode's misses fill the rest. Loading every held expert first cost the 32K answer 0.72 s
// before its second token (3090, 25 banks, 7.9 GB) and a --split layer rank's refill held the other rank's first step.
// A bank that holds every expert (GB10) loads them all: its decode maps need every one resident.
inline size_t refill_load_count(size_t held, int bank_slots, int experts) {
    return preload_all_experts(bank_slots, experts) ? held : std::min<size_t>(held, 1);
}

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
// --prompt-cache with one sequence: a finished request leaves its state at `live` = prompt + output - 1 (every output token
// but the last was fed); `seq` = prompt ++ output. A chat's next turn repeats both (the shipped template keeps the empty
// think block), so it reuses [0, live) and reads only the new message; anything else falls back to the snapshot.
template<class A, class B> inline int reusable_from(const A & seq, int live, int snap, const B & prompt) {
    const int n = reusable_prefix(seq, live, prompt);
    return n ? n : reusable_prefix(seq, snap, prompt);
}
// the rows a decode step keeps: the accepted ones, cut where max_tokens ends the request (the state then ends with the
// output; kept beyond it, a next turn could not reuse it)
inline int committed_rows(int accepted, int room) { return std::max(1, std::min(accepted, room)); }
// the lent layers the refill rebuilds from scratch: those that got a bank again during the prompt (built at mid-prefill size)
template <class Ready> inline std::vector<int> refill_rebuilds(const std::vector<int> & lent, Ready ready) {
    std::vector<int> r; for (int l : lent) if (ready(l)) r.push_back(l); return r;
}
// rows per step when a reused prefix's new part is read without a prefill plan (under 256 tokens, no bank borrowed):
// decode-size blocks, whose attention scratch beside full banks decode itself proves; one step of every new row at a long
// context did not fit (RTX 3090, 128K chat turn 3, 29 rows: the QSA gather's buffers out of memory with 237 MB free)
// A snapshot restore reads its tail with the step size the read that took the snapshot used (`tail_rows`, 0: the chunk):
// the same prompt again then gives the same text (8-row steps after a one-step first read changed it, and its copy drafts
// with it: 128K again 119 -> 93 tok/s), and a tail first read in 8-row steps never runs as one step beside full banks
// (128K turn 3 again, 29 rows in one step: out of memory).
inline int reused_read_rows(bool from_reply_state, int tail_rows, int new_tokens, int chunk) {
    if (from_reply_state && new_tokens < 256) return std::min(chunk, 8);
    return tail_rows > 0 && !from_reply_state ? std::min(chunk, tail_rows) : chunk;
}
// --prompt-cache with one sequence: its kept K/V is sized for this request plus room for the next turn (the reply, up to
// max_tokens, and a new message), not for the slot context (a full context's state costs the banks 27 slots per layer on a
// 24 GB card). A turn that outgrows it prefills afresh at its own size.
inline size_t cached_sequence_rows(size_t prompt, size_t max_tokens, size_t slot_context) {
    const size_t need = prompt + max_tokens + 16;
    return std::min(slot_context, need + std::max<size_t>(4096, max_tokens + 2048));
}
// does the kept sequence (K/V capacity `kept_rows`) hold the next request?
inline bool kept_sequence_fits(size_t prompt, size_t max_tokens, size_t kept_rows) { return prompt + max_tokens + 16 <= kept_rows; }
// the K/V rows the kept sequence has for the next request: kept_rows when it holds it, else it grows (its rows copied, or
// more pages mapped) to the request plus room for the next turn, by a quarter at least so a long chat does not grow every
// turn, never past the slot context; 0: the slot does not hold the request. Reading the whole prompt again instead cost a
// 128K chat 64 s to the first token at the turn that outgrew its first turn's K/V (GB10 rc170d, reused: 0.3 s).
inline size_t kept_sequence_rows(size_t prompt, size_t max_tokens, size_t kept_rows, size_t slot_context) {
    if (kept_sequence_fits(prompt, max_tokens, kept_rows)) return kept_rows;
    if (prompt + max_tokens + 16 > slot_context) return 0;
    return std::min(slot_context, std::max(cached_sequence_rows(prompt, max_tokens, slot_context), kept_rows + kept_rows / 4));
}
// --prompt-cache keeps a sequence's state after its request for any slot count (it was several slots only: a chat on one
// slot read its whole conversation again every turn, 32K 11.3 s / 128K 46.7 s to the first token on an RTX 3090)
inline bool prompt_cache_keeps(bool prompt_cache, size_t slots) { return prompt_cache && slots >= 1; }
// Host K/V before a prompt's new part is read: a layer-major chunk (256 new tokens or more) re-reads its whole context, so
// the old pages come back to the device, into the room its plan borrowed from the banks. A shorter new part (the same
// prompt again: kTail tokens) runs as decode steps, which read host rows through the gather, and no plan made room for
// it: the pages stay on the host (RTX 3090, a 123K prompt again with one slot: 224 MB came back into the last free
// memory and the first step failed for memory).
inline bool kv_back_for_prompt(int new_tokens, int chunk) { return new_tokens >= 256 && chunk >= 256; }
// where the next prompt's state is kept: kTail before this prompt's end, when the new part is at least 2 x kTail
inline int snapshot_at(int prompt, int reused, int tail) { return prompt >= reused + 2 * tail ? prompt - tail : 0; }

// The pinned residual buffer of a multi-chunk prefill (prompt x width floats) is pinned once at start-up when the host
// easily holds it: four times its size available and at most a tenth of RAM. Pinning it per request cost a 32K prompt
// 0.72 s of time to first token (cudaMallocHost of 1.6 GB beside an 80 GB pinned expert pack), more at 128K. On unified
// memory (GB10) pinned host memory is the GPU's memory, so there it also has to leave kUnifiedFloor available after it
// (cudaMemGetInfo reads ~1 GB free after warm-up there; the page cache is what gives way). GB10 Flash-Next 128K, 11-18:
// pinning 4.3 GB inside the first long request took 2.2 s of its time to first token, beside 60 GB available.
constexpr size_t kUnifiedFloor = size_t(20) << 30;
// --prompt-cache without the option (owner 11-18: Strata keeps the conversation by default, so do we): on whenever the
// memory fits it, --exact on too (a reused turn equals a fresh read up to the reuse point). Discrete cards: on; the kept
// sequence is the slot's own K/V, sized for the request and the next turn (cached_sequence_rows), read afresh when
// outgrown, and parked states have their own byte budget. Unified memory (GB10): on when a kept conversation's K/V
// (kept_bytes, the full context) per slot still leaves kUnifiedFloor available. GB10 Flash-Next, 11-18: the same 32K
// prompt again 16.1 -> 0.21 s to the first token, 128K 65.5 -> 0.25 s (Strata 0.2.15: 0.49 / 1.68 s).
inline bool prompt_cache_by_fit(bool asked, bool asked_value, bool unified_memory, size_t slots, size_t available, size_t kept_bytes) {
    if (asked) return asked_value;
    return !unified_memory || available >= kept_bytes * std::max<size_t>(1, slots) + kUnifiedFloor;
}
// Whether a multi-chunk prefill copies its wide residual rows (hc x hidden fp32, 40 KB per token) to the pinned host
// buffer for an MTP catch-up after the trunk. The rank that runs the head writes the MTP layer's keys / values with each
// chunk from the rows still on the device (kv_inline in fn_engine_prefill.cpp) whenever the chunk is layer-major (256
// tokens or more, which prefill_plan.hpp requires), and the other ranks run no MTP layer: so the buffer is only filled
// for a head with chunks below 256. Neither the start-up pin nor a request allocates it otherwise (pinned at start, by
// pin_residual_at_start, it held 1.3 GB at a 32K context, 2.7 GB at 64K, 5.4 GB at 131K that nothing read).
constexpr int kLayerMajorChunkMin = 256;  // the smallest chunk of a multi-chunk prefill (prefill_chunk_bytes refuses less)
inline bool prefill_residuals_on_host(bool head_here, int chunk) { return head_here && chunk < kLayerMajorChunkMin; }
inline bool pin_residual_at_start(size_t bytes, size_t host_ram, size_t available, bool unified_memory) {
    if (!bytes || bytes > available / 4 || bytes > host_ram / 10) return false;
    return !unified_memory || available >= bytes + kUnifiedFloor;
}

// Host K/V for positions older than TQFN_KV_HOST_KEEP (32768): on by default for discrete GPUs with VMM pages (RTX 3090,
// 256K: 52-56 -> 65-71 tok/s; contexts up to 2 x 32K move nothing), off on unified memory (GB10: host and device memory are
// the same). env (TQFN_KV_HOST) "0" / "1" forces it either way.
inline bool kv_host_on(const char * env, bool vmm, bool unified) {
    if (env) return std::atoi(env) != 0 && vmm;
    return vmm && !unified;
}

// Tail skip runs in --exact off (Strata's default route-tail-skip; --exact on never skips): rank R = 8 of the top-k. 3090, 11-19
// (11-19): KLD vs exact 0.022 (R 9 0.018, R 7 0.026), Q8_0 ruler 0.0787 -> 0.0818,
// GSM8K-200 0.965 -> 0.975, decode json +21 %, chat +5 %, 32K +25 %, 128K +30 % over exact (skip alone, no CPU pool).
constexpr int kTailSkipRank = 8;
inline int tail_skip_rank(bool exact) { return exact ? 0 : kTailSkipRank; }
// A skipped assignment (weight 0) never becomes a CPU miss pool column: its output is multiplied by 0, and its target, the
// token's first choice, is already the token's own column when it is missed (a second column of the same (token, expert)
// is wasted work, and n rows x topk duplicates overflow the pool job's 64 columns). When that target goes to the CPU the
// skipped assignment takes the same resident placeholder.
inline bool cpu_column(bool skipped, bool runs_on_cpu) { return runs_on_cpu && !skipped; }
// Approximate tail skip (tail_skip_rank; Strata's default route-tail-skip): per assignment of routed (tokens x topk ids), the
// replacement expert or -1 to keep. An expert whose best rank over the block is R-th or lower and that is not resident goes
// to its token's first choice (rank 0, always computed) with weight 0. Any block size (the first version stopped at 4 tokens,
// so 5-draft verify blocks never skipped).
// Skipped assignments' weight: false = 0 and the token's other weights unchanged (Strata's route-tail-skip); true = the
// other weights scaled so the token's sum stays 1 (the router's top-k weights are renormalised to sum 1).
constexpr bool kTailSkipRenorm = false;
// With a host tier (experts read from the GGUF on demand) tail skip keeps what the tier holds and skips only what would be a
// drive read (a tier copy is ~0.2 ms, a SATA read ~5 ms). Replay of the .77 routing (GPU 177 slots/layer + 8441 tier
// blocks, R 8): the same drive reads per step as skipping every expert off the card (json 104 / 104, chat1 10 / 10), with
// 1/3 to 1/2 of the skipped assignments (chat1 3.4 -> 1.1 %, 32K 5.0 -> 1.9 %), so most of the approximation lands on the
// slowest misses only.
inline bool tail_skip_keeps(bool on_card, bool in_host_tier) { return on_card || in_host_tier; }
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
// DFlash2 chain cut (Dflash2Drafter::set_tau: a chain stops at its first draft below this probability among its position's 16
// best). With every expert resident (GB10) each further verify row reads more distinct experts, so unsure drafts cost more
// than they keep: 0.35 (Strata's CUDA --draft-conf). Discrete cards keep the full chain (0.9 lost 10 % on 2x 4090, 11-14).
constexpr float kDflashTauResident = 0.35f;
inline float dflash_tau(bool every_expert_resident) { return every_expert_resident ? kDflashTauResident : 0.f; }

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
// A discrete card's pool takes blocks of up to kCpuPoolMaxRows rows, which no draft count reaches: it keeps the adaptive
// block (11-19: the 4-row cap held a discrete 3090 with the pool to 3.4 tokens per step on json instead of 4.4).
inline DraftPlan one_sequence_drafts(int manifest_drafts, int pool_rows, bool every_expert_resident = false) {
    if (manifest_drafts < 1) return {manifest_drafts, false};
    const int adaptive = std::max(manifest_drafts, every_expert_resident ? 7 : 5);
    if (pool_rows > 0 && pool_rows < adaptive + 1) return {std::min(manifest_drafts, std::max(pool_rows - 1, 1)), false};
    return {adaptive, true};
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
inline DraftPlan sequence_drafts(int manifest_drafts, bool exact, int env_threads, int cores, bool every_expert_resident = false,
                                 bool small_card = true, bool host_tier = false);

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
// A small card serving several requests at once (--np > 1) runs no pool (G2, 11-13): the pool's CPU threads saturating host
// memory beside the expert cache's copy-engine DMA faulted the GPU on a 12 GB 3090 (Xid 31 / 32; reproduced without tqLLM:
// docs ISSUE-12gb-np4-g2), and the pool did not pay at np 4 there. One request keeps it.
// TQFN_CPU_MISS as cpu_miss_threads' env_threads: its value, -1 when unset
inline int cpu_miss_env() { const char * v = std::getenv("TQFN_CPU_MISS"); return v ? std::atoi(v) : -1; }
// The pool's kernels are AVX2 (fn_cpu_experts.hpp): elsewhere (GB10 / DGX Spark, arm64) --exact off runs none; only an
// explicit TQFN_CPU_MISS asks for it there, and the pool then refuses to start (11-18: --exact off on a GB10 failed every
// request with "TQFN_CPU_MISS needs an x86 CPU with AVX2")
#if defined(__x86_64__)
constexpr bool kCpuMissPoolBuilt = true;
#else
constexpr bool kCpuMissPoolBuilt = false;
#endif
// Experts read from the GGUF on demand (a host tier is their source, 31 GB .77): no pool unless asked. Its jobs would read
// the experts' GGUF mapping (there is no pinned packed copy), a SATA drive read of ~5 ms per expert on the host thread
// where a tier copy is ~0.2 ms; the same misses go through the tier instead.
// One sequence on a card that is not small (the 24 GB class, `big_one_sequence`): no pool unless asked. On .77 (3090, PCIe 3)
// skip 8 with the pool ran 75-118 tok/s, without it 77-119 (11-20, 2 rounds each), and on .188 the pool beside the copy
// engine's DMA is the platform fault's trigger (20261120_skip3). Small cards keep it (their CPU-owned experts, F26).
inline int cpu_miss_threads(int env_threads, bool exact, int cores, bool small_card_parallel = false, bool pool_built = kCpuMissPoolBuilt,
                            bool host_tier = false, bool big_one_sequence = false) {
    if (env_threads >= 0) return env_threads;
    if (small_card_parallel || !pool_built || host_tier || big_one_sequence) return 0;
    return exact ? 0 : std::max(1, std::min(15, cores / 2 - 1));
}
inline DraftPlan sequence_drafts(int manifest_drafts, bool exact, int env_threads, int cores, bool every_expert_resident, bool small_card, bool host_tier) {
    // the 4-row block is the measured rule for one sequence's pool on any card (a 12 GB 3090: 5 drafts 44.8 / 44.6 tok/s,
    // 3 drafts 49-50); the layer call keeps its own budget (kCpuPoolMaxRows on a discrete card)
    const int threads = cpu_miss_threads(env_threads, exact, cores, false, kCpuMissPoolBuilt, host_tier, !small_card);
    return one_sequence_drafts(manifest_drafts, cpu_pool_rows(threads, small_card), every_expert_resident);
}
// the pool reads missed experts from the pinned packed copy whenever it runs (11-06: the 3090 np 8 --exact off rows fell from
// 76 / 77 to 68 / 70 tok/s when the pool came from --exact off alone and read the GGUF mapping instead)
inline bool cpu_pool_reads_packed(int env_threads, bool exact, int cores, bool small_card_parallel = false, bool host_tier = false,
                                  bool big_one_sequence = false) {
    return cpu_miss_threads(env_threads, exact, cores, small_card_parallel, kCpuMissPoolBuilt, host_tier, big_one_sequence) > 0;
}
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

// Layer-major prefill: the routed experts a layer computes from its bank, in passes the bank holds. One pass when it holds
// them all (with a pack file only resident experts go through the bank); experts read from the GGUF on demand (no pack: a
// 31 GB RAM box) send every routed expert through it, more than it holds for any chunk of 256+ tokens (.77 rc170d: "mixed
// cache: active set exceeds slots" on every such prompt): resident ones first (no load), then cap at a time.
template <class Resident> inline std::vector<std::vector<int>> bank_passes(const std::vector<int> & experts, size_t cap, Resident resident) {
    if (experts.size() <= cap || cap == 0) return {experts};
    std::vector<int> order(experts);
    std::stable_partition(order.begin(), order.end(), [&](int e) { return bool(resident(e)); });
    std::vector<std::vector<int>> passes;
    for (size_t i = 0; i < order.size(); i += cap) passes.emplace_back(order.begin() + long(i), order.begin() + long(std::min(order.size(), i + cap)));
    return passes;
}

// Packed experts (one contiguous block per expert in the decode cache's row layout, so a cache miss is one copy): read
// from the manifest's experts file when it names one; otherwise built in pinned memory from the GGUF at start-up on a
// discrete GPU, so the released GGUF is the only expert source to download. A GPU that shares system memory (GB10)
// keeps every expert resident and needs no pack.
// Build the packed experts in pinned memory only on a discrete GPU without a pack file, outside hot/cold (its tiers manage
// expert memory) and when the pack fits in 60 % of the machine's RAM; otherwise the cache reads experts from the GGUF on
// demand. Hot/cold on .231 (23 GiB RAM) died pinning 51 GB.
// Also when the RAM available now (cgroup-aware) holds the pack beside `beside` (a full prompt's host buffers, the MTP pack
// and kHostTierReserve, what the host tier keeps free too): .188 in a 64 GB scope refused its 51 GB pack (60 % = 38 GB),
// took the host tier, pinned 50.9 GB there anyway and ran 0.78-0.85x of the pack (B4 / B5 vs F, exact, same hashes).
inline bool build_pack_in_memory(bool has_pack_file, bool integrated_gpu, bool hot_cold, double pack_bytes, double host_ram_bytes,
                                 double available_bytes = 0, double beside_bytes = 0) {
    if (has_pack_file || integrated_gpu || hot_cold) return false;
    return pack_bytes <= 0.6 * host_ram_bytes || (available_bytes > 0 && pack_bytes + beside_bytes <= available_bytes);
}
// Experts read from the GGUF on demand (a box whose RAM cannot pin the pack, .77: 31 GB): a pinned host tier of expert
// blocks under the GPU banks (host_expert_tier.hpp). Its blocks: what the box has available now, less a full-context
// prompt's host buffers (`prompt_host`: residual rows, embeddings, the K/V tail), the prefill stream ring and `reserve` for
// the box and the page cache the per-token reads need (embedding rows, dense weights); at most the experts the banks do
// not hold; none when fewer than `min_blocks` fit (a tier smaller than one layer call's misses only adds copies).
// Layer-outer prefill (experts read from the GGUF on demand: every layer over all chunks, so a layer's streamed experts
// are read once per prompt): how many leading chunks [b, e) run that way. Every chunk up to the prompt-cache snapshot
// boundary (the snapshot is taken once they are done; the tail after it runs as before), each of 256+ tokens (shorter
// chunks are not layer-major); 0 unless at least two (one chunk is layer-major already).
inline int layer_outer_chunks(const std::vector<std::pair<int, int>> & spans, int snap) {
    int n = 0;
    for (const auto & [b, e] : spans) {
        if (e - b < 256 || (snap > 0 && e > snap)) break;
        ++n;
    }
    return n >= 2 ? n : 0;
}
constexpr uint64_t kHostTierReserve = uint64_t(5) << 30;
// The host tier is filled at start (host_expert_tier.hpp warm_tier) only when it holds nearly every expert: then the fill
// is the working set (.188 in 64 GB: 24,569 of 24,576, first prompts 24-60 -> 97-112 tok/s). A tier of a third of them
// (.77, 31 GB) would read ~17 GB of experts picked by id from a SATA drive at start and keep few of the ones prompts route to.
inline bool warm_host_tier(uint64_t tier_blocks, uint64_t experts_total) { return tier_blocks * 10 >= experts_total * 9; }
// The tier's elastic chunks (host_expert_tier.hpp attach / detach): what RAM a prompt of several chunks needs beside the base
// tier (its host buffers: rows, a K/V tail past kv_keep positions; sized like the base tier, with kHostTierReserve) is held
// by experts while it is not needed, with kHostTierDecodeReserve left available in decode and for one-chunk prompts (31 GB
// .77, F4: the 3.8 GB of a 128K prompt's buffers and 2 GB of the reserve were free in every short row and at 32K; replay of
// the .77 routing, 8441 -> 11000 tier blocks: drive reads per step btree 25 -> 15, chat 14-16 -> 9-11, 32K 45 -> 26).
constexpr uint64_t kHostTierDecodeReserve = uint64_t(3) << 30;
constexpr uint64_t kHostTierChunk = uint64_t(512) << 20;
inline uint64_t prompt_host_bytes(uint64_t tokens, uint64_t rows_per_token, uint64_t kv_per_token, uint64_t kv_keep) {
    return tokens * rows_per_token + (kv_keep && tokens > kv_keep ? (tokens - kv_keep) * kv_per_token : 0);
}
inline int host_tier_chunks(uint64_t full_prompt_host) {
    return int((full_prompt_host + kHostTierReserve - kHostTierDecodeReserve) / kHostTierChunk);
}
// RAM the elastic chunks leave available for what comes next
inline uint64_t host_tier_keep(bool multi_chunk_prompt, uint64_t prompt_host) {
    return multi_chunk_prompt ? prompt_host + kHostTierReserve : kHostTierDecodeReserve;
}
inline int host_tier_blocks(uint64_t available, uint64_t prompt_host, uint64_t ring, uint64_t reserve, uint64_t block,
                            uint64_t off_card_experts, int min_blocks) {
    if (!block) return 0;
    const uint64_t taken = prompt_host + ring + reserve;
    const uint64_t n = std::min<uint64_t>(available > taken ? (available - taken) / block : 0, off_card_experts);
    return n < uint64_t(std::max(0, min_blocks)) ? 0 : int(std::min<uint64_t>(n, uint64_t(1) << 30));
}
// --split layer, discrete card: the most expert-pack bytes a rank may take so its box keeps `reserve` available. From what
// is available (per rank on the box) come the reserve, the head's prompt buffers (the last rank with one sequence, every rank
// with several: the residual rows and embeddings of a full-context prompt for the MTP catch-up) and, per pack byte, the pack
// itself (x1.02) and the host K/V tail of the same layers (`kv_per_pack`: tail bytes per pack byte); never above the pinned
// pack rule (0.6 x RAM). 2x 3090 at 128K: .77 (31 GB) took 16 layers by the pack rule alone and fell to 1.95 GB available.
inline uint64_t split_ram_cap(uint64_t available, uint64_t host_ram, uint64_t reserve, uint64_t head_bytes, double kv_per_pack,
                              uint64_t fixed_bytes = 0) {
    const uint64_t pinned_rule = uint64_t(0.6 * double(host_ram) / 1.02);
    const uint64_t taken = reserve + head_bytes + fixed_bytes;
    const uint64_t room = available > taken ? available - taken : 0;
    return std::min(pinned_rule, uint64_t(double(room) / (1.02 + std::max(0.0, kv_per_pack))));
}
constexpr uint64_t kSplitHostReserve = uint64_t(20) << 30;  // the 20 GB a box keeps for its own services
// a rank's host memory besides its experts, K/V tail and prompt buffers: CUDA / NCCL host buffers and pinned staging (2x 3090
// 11-19: .77 holding one layer at 128K used 13.4 GB, 5.3 GB of it neither pack nor prompt buffers, 1.5 of that the MTP pack)
constexpr uint64_t kSplitHostFixed = uint64_t(4) << 30;
// The load's 512-token warm-up chunk runs where this rank's first layer reads packed experts. --split layer: only when every
// rank's does (a rank on a box whose RAM cannot pin its layers reads the GGUF on demand; 2x 3090 .188 + .77 11-19: rank 0
// ran the chunk, rank 1 did not, "the ranks diverged" at load). `packed_on_rank`: each rank's answer
inline bool warm_prefill_chunk(const std::vector<int> & packed_on_rank) {
    for (int p : packed_on_rank) if (!p) return false;
    return !packed_on_rank.empty();
}
// Pages that may be dropped once the experts sit in pinned memory: the trunk expert tensors only. The per-layer token
// embedding (Flash-Next: 28.8 GB, one 90-byte row per token and layer) is read from the file at every token; with its
// pages dropped each new token waited on the disk (3090: chats 79-88 instead of 94-103 tok/s, 32K first token +2.7 s).
inline bool drop_after_pack(const std::string & tensor) {
    const auto exps = tensor.find("_exps.weight");
    return tensor.rfind("blk.", 0) == 0 && exps != std::string::npos && exps + 12 == tensor.size();
}
inline bool drop_layer_after_pack(const std::string & tensor, int layer) {
    return tensor.rfind("blk." + std::to_string(layer) + ".", 0) == 0 && drop_after_pack(tensor);
}
// Read at start-up so the first requests do not fault it in from the disk, when it fits half the RAM: on a small host
// (.231: 23 GiB, 28.8 GB embedding) the read only cycles the page cache and evicts the cold experts.
// `available`: the RAM still available after the pinned pack (cgroup-aware; 0: not known): the table must stay cached
// beside it with 2 GB to spare (.188 in a 64 GB scope: 28.8 GB read at start beside a 51 GB pinned pack, about half of
// it cached afterwards: 12 major faults per decoded token, 0.86-0.93x of the same build with all RAM)
inline bool warm_at_start(const std::string & tensor, double tensor_bytes, double host_ram_bytes, double available = 0) {
    return tensor == "per_layer_token_embd.weight" && tensor_bytes <= 0.5 * host_ram_bytes &&
           (available <= 0 || tensor_bytes + 2e9 <= available);
}
// The per-layer token embedding when it is not read at start (larger than half the RAM): its rows are read one by one
// (90 bytes per token and layer), so a fault must not pull the default 128 KB of read-ahead with it (.77, SATA: a 16K
// chunk's 262K rows read ~33 GB around them, 20 s per chunk beside the expert reads; decode 48 faults per token)
// The same rows are asked for together (MADV_WILLNEED per row, MappedGguf::gather_rows), so a step's misses are read in
// parallel instead of one fault after another.
inline bool read_rows_randomly(const std::string & tensor, double tensor_bytes, double host_ram_bytes, double available = 0) {
    return tensor == "per_layer_token_embd.weight" && !warm_at_start(tensor, tensor_bytes, host_ram_bytes, available);
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
// A 24 GB card's prefill chunk for a prompt shorter than the engine chunk: the prompt's length rounded up to 1 KiB (at
// least min_chunk, prefill_choice's floor), so its plan sizes the scratch arena for the prompt and not for a 16K chunk.
// The prompt is one chunk either way (same chunk boundaries, same selection block), so the text does not change; the plan
// borrows fewer expert banks (3090, 59 GiB host, IQ3_XXS file: a 300-token prompt released 19 banks / 6.5 GB and refilled
// them in 350 ms every request). Prompts longer than the engine chunk keep it.
inline int short_prompt_chunk(int prompt_tokens, int engine_chunk, int min_chunk = 2048) {
    if (prompt_tokens <= 0 || prompt_tokens > engine_chunk) return engine_chunk;
    return std::min(engine_chunk, std::max(min_chunk, (prompt_tokens + 1023) / 1024 * 1024));
}
// Byte cap of the expert banks. slots=auto sizes the banks from free device memory itself, so its cap is the device; a
// fixed budget_gb (taken from one card) capped a GB10's 512 resident slots at 18 GB and evicted layers mid-prompt.
inline size_t bank_budget_bytes(bool slots_auto, double budget_gb, size_t device_total) {
    return slots_auto ? device_total : size_t(budget_gb * 1024.0 * 1024.0 * 1024.0);
}
// Per-layer expert slots from a plan (counts per layer, any scale): scaled to slots x layers, each in [1, experts].
// When every expert fits (slots >= experts, e.g. a GB10) the plan is a cache plan for smaller cards: every layer keeps all.
// A layer the scale would put above `experts` keeps all its experts and the rest of the budget goes to the other layers
// (water-filling): clamping alone threw those slots away (2x RTX 4090 --split layer, 11-14: 5 layers at 512, rank 0 over
// its budget and rank 1 ~1,150 slot-layers = 2.3 GB of VRAM idle).
inline std::vector<int> scale_layer_slots(std::vector<int> caps, int slots, int experts) {
    if (slots >= experts) return std::vector<int>(caps.size(), experts);
    long total = 0; for (int c : caps) total += c;
    long want = long(slots) * long(caps.size());
    if (total == want || total <= 0) return caps;
    const std::vector<int> plan = caps;
    std::vector<char> full(caps.size(), 0);
    for (bool clamped = true; clamped;) {
        clamped = false; long t = 0;
        for (size_t i = 0; i < plan.size(); ++i) if (!full[i]) t += plan[i];
        if (t <= 0) break;
        for (size_t i = 0; i < plan.size(); ++i)
            if (!full[i] && double(plan[i]) * double(want) / double(t) >= double(experts)) {
                full[i] = 1; caps[i] = experts; want -= experts; clamped = true;
            }
        if (!clamped)
            for (size_t i = 0; i < plan.size(); ++i)
                if (!full[i]) caps[i] = std::max(1, std::min(experts, int(double(plan[i]) * double(want) / double(t) + 0.5)));
    }
    return caps;
}
// The same over the layers this process holds banks for (`owned`; --split layer: this rank's trunk layers): the budget is
// slots x owned layers, spread by the plan's shape over those layers only. Layers not owned keep `slots` (no bank there).
inline std::vector<int> scale_owned_layer_slots(const std::vector<int> & caps, const std::vector<bool> & owned, int slots,
                                                int experts, int min_slots = 0) {
    if (owned.size() != caps.size()) return lift_layer_floor(scale_layer_slots(caps, slots, experts), min_slots, experts);
    std::vector<int> mine;
    for (size_t i = 0; i < caps.size(); ++i) if (owned[i]) mine.push_back(caps[i]);
    if (mine.empty()) return caps;
    mine = lift_layer_floor(scale_layer_slots(mine, slots, experts), min_slots, experts);
    std::vector<int> out(caps.size(), std::min(slots, experts));
    for (size_t i = 0, j = 0; i < caps.size(); ++i) if (owned[i]) out[i] = mine[j++];
    return out;
}

}  // namespace tqllm::flashnext
