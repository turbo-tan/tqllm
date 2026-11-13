#pragma once
// Multi-box request routing (tqserve --peer URL): every box runs a full tqserve; the front door sends each chat request
// to the box where it is expected to finish soonest, counting the K/V a box already holds for the prompt (prefix reuse).
// Pure, header-only: tests/serve_router.cpp runs it with just g++.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace tqserve {

struct BoxStatus {
    bool ok = false;           // answered the status query in time
    int free = 0, busy = 0, queued = 0;
    int inflight = 0;          // requests the front door sent here that have not finished (counts them before the box's own status does)
    double reuse = 0;          // prompt tokens a free slot already holds (its last conversation's prefix)
    double prefill_tps = 0;    // measured prompt tokens / s (0: unknown)
    double decode_tps = 0;     // measured tokens / s of one sequence alone (0: unknown)
    double hold_s = 0;         // mean wall time of a request on this box (0: unknown)
    bool helped = false;       // the front door with --split experts: its lone requests get the other box's experts
};

// Decode slows with the sequences a box batches: per-sequence rate r1 / (1 + k (n - 1)). GB10 Flash-Next IQ3_S,
// 11-07: 100 / 28 / 17 tok/s per user at 1 / 4 / 8 users -> k ~ 0.68.
constexpr double kBatchSlowdown = 0.68;

// Requests on the box: its own count, or the front door's when the box has not admitted them yet.
inline int route_load(const BoxStatus & b) { return std::max(b.busy + b.queued, b.inflight); }

// Expected seconds until this request is answered on box b: wait for a slot, prefill what the box does not hold,
// decode `gen` tokens beside the box's other sequences, plus the decode time the prefill takes from those sequences.
inline double route_cost(const BoxStatus & b, double prompt, double gen) {
    const double prefill = b.prefill_tps > 0 ? b.prefill_tps : 1500.0;
    const double r1 = b.decode_tps > 0 ? b.decode_tps : 100.0;
    const double hold = b.hold_s > 0 ? b.hold_s : 20.0;
    const int capacity = std::max(1, b.busy + b.free);
    const int load = route_load(b);
    const bool slot = load < capacity;
    const double wait = slot ? 0.0 : (load - capacity + 1) * hold / capacity;
    const double fresh = std::max(0.0, prompt - (slot ? b.reuse : 0.0)) / prefill;
    const int running = std::min(load, capacity);                   // sequences decoding beside this one
    const double decode = gen / (r1 / (1.0 + kBatchSlowdown * running));
    const double stall = fresh * running / (running + 1.0);
    return wait + fresh + decode + stall;
}

// A peer whose status query was late is busy, not gone (16 requests at once: a box that is prefilling answered after
// the 100 ms timeout, was skipped, and its share queued on the full front door, 11-09 NVFP4 16 users 217 vs 2 x 136):
// keep using its last answer with the front door's own in-flight count; skip it only after `kMissesDown` misses in a row.
constexpr int kMissesDown = 3;
struct PeerMemory { BoxStatus last; int misses = 0; };
inline BoxStatus remembered_status(const BoxStatus & now, PeerMemory & m) {
    if (now.ok) { m.last = now; m.misses = 0; return now; }
    if (++m.misses >= kMissesDown || !m.last.ok) return now;
    BoxStatus b = m.last;
    b.reuse = 0;                // which slot holds what is not known now
    b.inflight = now.inflight;  // the front door's count is current
    return b;
}
inline bool route_has_slot(const BoxStatus & b) { return route_load(b) < std::max(1, b.busy + b.free); }
// Index of the box to serve the request; box 0 is the front door (no network hop). Boxes that did not answer are
// skipped; a box with a free slot wins over a full one (a faster-measured box otherwise took 10 of 16 requests and queued
// two beside the other box's free slot, 11-11); near-equal costs (within 2 %) go to the less loaded box, then the front door.
// A lone request (every box idle) goes to a box with expert help (--split experts: the helper computes half of its experts
// only while it has no request of its own), unless another box holds kLoneReuse more of its prompt's K/V. Decode rates are
// not compared: the helped box's measured rate mixes helped and unhelped requests, and a peer not yet measured reads 100.
constexpr double kLoneReuse = 1024;
inline int pick_box_lone(const std::vector<BoxStatus> & boxes, double prompt, double gen) {
    int helped = -1;
    for (int i = 0; i < int(boxes.size()); ++i) {
        const auto & b = boxes[size_t(i)];
        if (b.ok && route_load(b) > 0) return -1;
        if (b.ok && b.helped && helped < 0) helped = i;
    }
    if (helped < 0) return -1;
    (void) prompt; (void) gen;
    for (const auto & b : boxes) if (b.ok && b.reuse >= boxes[size_t(helped)].reuse + kLoneReuse) return -1;
    return helped;
}
// A box not measured yet is costed at the measured boxes' rates (the 100 / 1500 tok/s defaults read faster than a GB10:
// 11-12, after a run of lone requests that all went to the helped front door, the first 4-user burst put 3 on the
// never-measured peer, 132 vs 162 tok/s).
inline std::vector<BoxStatus> with_measured_rates(std::vector<BoxStatus> boxes) {
    double d = 0, p = 0;
    for (const auto & b : boxes) if (b.ok) { d = std::max(d, b.decode_tps); p = std::max(p, b.prefill_tps); }
    for (auto & b : boxes) { if (b.decode_tps <= 0) b.decode_tps = d; if (b.prefill_tps <= 0) b.prefill_tps = p; }
    return boxes;
}
inline int pick_box(const std::vector<BoxStatus> & raw, double prompt, double gen) {
    if (const int h = pick_box_lone(raw, prompt, gen); h >= 0) return h;
    const auto boxes = with_measured_rates(raw);
    bool room = false;
    for (const auto & b : boxes) room = room || (b.ok && route_has_slot(b));
    int best = -1;
    double best_cost = 0;
    for (int i = 0; i < int(boxes.size()); ++i) {
        const auto & b = boxes[size_t(i)];
        if (!b.ok || (room && !route_has_slot(b))) continue;
        const double c = route_cost(b, prompt, gen);
        if (best < 0) { best = i; best_cost = c; continue; }
        const auto & o = boxes[size_t(best)];
        const bool tie = std::fabs(c - best_cost) <= 0.02 * std::max(c, best_cost);
        const int load = route_load(b), oload = route_load(o);
        if ((!tie && c < best_cost) || (tie && load < oload)) { best = i; best_cost = c; }
    }
    return best < 0 ? 0 : best;
}

// Prompt tokens a slot's last conversation (its rendered prompt + answer, `held`) shares with a new rendered prompt,
// estimated from the common text prefix and the box's characters per token.
inline double reuse_tokens(const std::string & held, const std::string & prompt, double chars_per_token) {
    size_t n = 0;
    const size_t m = std::min(held.size(), prompt.size());
    while (n < m && held[n] == prompt[n]) ++n;
    if (n < 1024) return 0;  // a shared system prompt / template header is not a held conversation (and saves < 0.2 s)
    return double(n) / (chars_per_token > 0 ? chars_per_token : 4.0);
}

struct Ema {
    double v = 0;
    bool set = false;
    void add(double x, double a = 0.2) { v = set ? (1 - a) * v + a * x : x; set = true; }
};

} // namespace tqserve
