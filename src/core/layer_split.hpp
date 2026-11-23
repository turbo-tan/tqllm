#pragma once
// Layer split across GPUs (one rank per GPU, NCCL between ranks; docs plan multi-gpu-20261109): rank r owns the
// contiguous layers [lo, hi). Pure arithmetic, no CUDA: tests/layer_split.cpp.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {
struct LayerRange {
    uint32_t lo = 0, hi = 0;
    bool owns(uint32_t l) const { return l >= lo && l < hi; }
    bool first() const { return lo == 0; }
};
constexpr int kMaxSplitRanks = 8;

// every rank's range: `bounds` are the first layers of ranks 1 .. world-1 (strictly increasing, within 1 .. layers - 1);
// empty = an even split (rank r: layers * r / world ..)
inline std::vector<LayerRange> split_ranges(uint32_t layers, int world, const std::vector<uint32_t> & bounds) {
    if (world < 1 || world > kMaxSplitRanks || layers < uint32_t(world))
        throw std::invalid_argument("layer split: " + std::to_string(world) + " ranks for " + std::to_string(layers) +
                                    " layers (1 .. " + std::to_string(kMaxSplitRanks) + " ranks, at least one layer each)");
    std::vector<LayerRange> r(static_cast<size_t>(world));
    if (bounds.empty()) {
        for (int i = 0; i < world; ++i)
            r[size_t(i)] = {uint32_t(uint64_t(layers) * i / world), uint32_t(uint64_t(layers) * (i + 1) / world)};
        return r;
    }
    if (bounds.size() != size_t(world - 1))
        throw std::invalid_argument("--split-at: " + std::to_string(world - 1) + " first layer(s) for " + std::to_string(world) +
                                    " ranks, got " + std::to_string(bounds.size()));
    uint32_t lo = 0;
    for (int i = 0; i < world; ++i) {
        const uint32_t hi = i + 1 < world ? bounds[size_t(i)] : layers;
        if (hi <= lo || hi > layers || (i + 1 < world && hi >= layers))
            throw std::invalid_argument("--split-at must be increasing first layers within 1 .. " + std::to_string(layers - 1));
        r[size_t(i)] = {lo, hi};
        lo = hi;
    }
    return r;
}
inline LayerRange layer_range(uint32_t layers, int rank, int world, const std::vector<uint32_t> & bounds) {
    if (rank < 0 || rank >= world) throw std::invalid_argument("layer split: bad rank / world");
    return split_ranges(layers, world, bounds)[size_t(rank)];
}
inline std::vector<uint32_t> range_bounds(const std::vector<LayerRange> & r) {
    std::vector<uint32_t> b;
    for (size_t i = 1; i < r.size(); ++i) b.push_back(r[i].lo);
    return b;
}
// the rank that runs layer l (-1: none)
inline int layer_owner(const std::vector<LayerRange> & r, uint32_t l) {
    for (size_t i = 0; i < r.size(); ++i) if (r[i].owns(l)) return int(i);
    return -1;
}
// "24,40" -> {24, 40}
inline std::vector<uint32_t> parse_split_at(const std::string & s) {
    std::vector<uint32_t> v;
    size_t at = 0;
    while (at <= s.size()) {
        const size_t comma = s.find(',', at);
        const std::string t = s.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        size_t used = 0; long x = -1;
        try { x = std::stol(t, &used); } catch (...) { used = 0; }
        if (t.empty() || used != t.size() || x < 1) throw std::invalid_argument("--split-at: comma-separated first layers of ranks 1, 2, ... (got '" + s + "')");
        v.push_back(uint32_t(x));
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return v;
}
// Bounds that fill every rank to about the same fraction of its budget: rank r ends where the layers' running cost
// reaches its share of the total (budget[0..r] / sum of budgets), at least one layer per rank. `cost[l]`: layer l's
// device bytes (weights + its K/V or state), `budget[r]`: rank r's free device bytes left after what every rank holds
// whatever its layers (embedding, head, buffers). `cap[r]` (optional, 0: none): the most layer cost rank r may take
// (Flash-Next: the experts its box can pin in RAM); a capped rank's excess goes to the others by their budgets, and when
// the caps cannot hold the model the shares follow the caps. Every rank computes the same bounds from the same inputs.
inline std::vector<uint32_t> bounds_by_budget(const std::vector<uint64_t> & cost, const std::vector<uint64_t> & budget,
                                              const std::vector<uint64_t> & cap = {}) {
    const size_t L = cost.size(), W = budget.size();
    if (W < 1 || W > size_t(kMaxSplitRanks) || L < W) throw std::invalid_argument("layer split: bad layer / rank count");
    if (!cap.empty() && cap.size() != W) throw std::invalid_argument("layer split: one cap per rank");
    long double total_budget = 0, total_cost = 0;
    for (uint64_t b : budget) total_budget += (long double) b;
    for (uint64_t c : cost) total_cost += (long double) c;
    if (total_budget <= 0) throw std::invalid_argument("layer split: no rank has free memory for layers");
    // each rank's share of the total cost: by budget, then water-filled under the caps
    std::vector<long double> share(W, 0);
    std::vector<bool> fixed(W, false);
    long double left = total_cost;
    for (bool again = true; again;) {
        again = false;
        long double b = 0;
        for (size_t r = 0; r < W; ++r) if (!fixed[r]) b += (long double) budget[r];
        if (b <= 0) break;
        for (size_t r = 0; r < W; ++r) if (!fixed[r]) share[r] = left * (long double) budget[r] / b;
        for (size_t r = 0; r < W; ++r)
            if (!fixed[r] && !cap.empty() && cap[r] && share[r] > (long double) cap[r]) {
                share[r] = (long double) cap[r]; fixed[r] = true; left -= share[r]; again = true;
            }
    }
    long double sum = 0;
    for (long double x : share) sum += x;
    const bool caps_hold = sum + 0.5L >= total_cost;
    if (!caps_hold) {  // every rank capped and the caps do not hold the model: shares by cap
        long double caps = 0;
        for (uint64_t c : cap) caps += (long double) c;
        for (size_t r = 0; r < W; ++r) share[r] = caps > 0 ? total_cost * (long double) cap[r] / caps : total_cost / W;
    }
    std::vector<long double> prefix(L + 1, 0);
    for (size_t l = 0; l < L; ++l) prefix[l + 1] = prefix[l] + (long double) cost[l];
    std::vector<uint32_t> bounds;
    long double target = 0;
    size_t lo = 0;
    for (size_t r = 0; r + 1 < W; ++r) {
        target += share[r];
        // the end j (lo < j <= L - (W - 1 - r)) whose running cost is nearest the target; the earlier one on a tie
        size_t best = lo + 1;
        const size_t last = L - (W - 1 - r);
        for (size_t j = lo + 1; j <= last; ++j) {
            const long double d = prefix[j] > target ? prefix[j] - target : target - prefix[j];
            const long double db = prefix[best] > target ? prefix[best] - target : target - prefix[best];
            if (d < db) best = j;
        }
        // a capped rank never takes more than its cap (a pack one layer over the RAM falls back to reading the GGUF)
        if (caps_hold && !cap.empty() && cap[r])
            while (best > lo + 1 && prefix[best] - prefix[lo] > (long double) cap[r]) --best;
        bounds.push_back(uint32_t(best));
        lo = best;
    }
    if (caps_hold && W > 1 && !cap.empty() && cap[W - 1]) {  // the last rank: its first layer moves up while it is over its cap and
        size_t b = lo;                            // the rank before it stays under its own
        const size_t prev = W > 2 ? size_t(bounds[W - 3]) : 0;
        while (b + 1 < L && prefix[L] - prefix[b] > (long double) cap[W - 1] &&
               (cap[W - 2] == 0 || prefix[b + 1] - prefix[prev] <= (long double) cap[W - 2])) ++b;
        bounds[W - 2] = uint32_t(b);
    }
    return bounds;
}

// One forward's trunk on a rank (Flash-Next DecodeModel::trunk_layers): the rank whose residual rows it receives before its
// first layer (-1: none), its layers [lo, hi), the rank it sends them to after its last layer (-1: none), and the rank
// whose trunk output is broadcast (the last), so every rank runs the head, the MTP layer and sampling on the same rows
struct TrunkPlan { int recv_from = -1, send_to = -1, root = 0; uint32_t lo = 0, hi = 0; };
inline TrunkPlan trunk_plan(const LayerRange & lr, int rank, int world) {
    return {rank > 0 ? rank - 1 : -1, rank + 1 < world ? rank + 1 : -1, world - 1, lr.lo, lr.hi};
}
// Host RAM per rank (Flash-Next on discrete cards: `limited`, `cap` = the expert bytes its box can pin and keep its reserve):
// the ranks that cannot hold even the cheapest layer (refused at start), and whether the last rank, which also holds the
// head's prompt buffers, has the least room while another rank has more (then the box with the most RAM should be last)
struct RoomCheck { std::vector<int> refused; bool move_head = false; };
inline RoomCheck room_check(const std::vector<bool> & limited, const std::vector<uint64_t> & cap, uint64_t cheapest_layer) {
    RoomCheck out;
    const size_t w = cap.size();
    for (size_t r = 0; r < w; ++r) if (limited[r] && cap[r] < cheapest_layer) out.refused.push_back(int(r));
    if (w > 1 && limited[w - 1])
        for (size_t r = 0; r + 1 < w; ++r) if (!limited[r] || cap[r] > cap[w - 1]) { out.move_head = true; break; }
    return out;
}

// The cross-rank operations of one trunk forward on `rank`, in order (Flash-Next DecodeModel::trunk_layers runs exactly
// these): receive the residual rows from the rank before, run the own layers, send them to the next rank; then either the
// last rank's output to every rank (to_all) or not (head_only: only the last rank reads it), and for each DFlash2 target
// layer its captured rows from the owner to every rank (to_all) or to the last rank only. `feat_layers`: target layers.
struct TrunkOp {
    enum Kind { recv_rows, layers, send_rows, bcast_rows, send_feat, recv_feat, bcast_feat } kind;
    int peer = -1;  // recv / send: the other rank; bcast: the root
    int feat = -1;  // feature ops: index into feat_layers
};
inline std::vector<TrunkOp> trunk_ops(const std::vector<LayerRange> & ranges, int rank, bool to_all, const std::vector<int> & feat_layers) {
    const int world = int(ranges.size()), last = world - 1;
    std::vector<TrunkOp> ops;
    if (rank > 0) ops.push_back({TrunkOp::recv_rows, rank - 1});
    ops.push_back({TrunkOp::layers});
    if (rank < last) ops.push_back({TrunkOp::send_rows, rank + 1});
    if (to_all) ops.push_back({TrunkOp::bcast_rows, last});
    for (size_t i = 0; i < feat_layers.size(); ++i) {
        const int owner = layer_owner(ranges, uint32_t(feat_layers[i]));
        if (owner < 0) continue;
        if (to_all) ops.push_back({TrunkOp::bcast_feat, owner, int(i)});
        else if (owner != last && rank == owner) ops.push_back({TrunkOp::send_feat, last, int(i)});
        else if (owner != last && rank == last) ops.push_back({TrunkOp::recv_feat, owner, int(i)});
    }
    return ops;
}
}  // namespace tqllm
