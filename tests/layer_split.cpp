// layer ranges of the multi-GPU layer split (src/core/layer_split.hpp)
#include "../src/core/layer_split.hpp"
#include <cstdio>
static int failures = 0;
static void check(bool ok, const char * what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; } }
template <class F> static bool throws(F && f) { try { f(); } catch (const std::invalid_argument &) { return true; } return false; }
static bool covers_once(const std::vector<tqllm::LayerRange> & r, uint32_t layers) {
    uint32_t at = 0;
    for (const auto & x : r) { if (x.lo != at || x.hi <= x.lo) return false; at = x.hi; }
    return at == layers;
}
int main() {
    using tqllm::layer_range; using tqllm::split_ranges; using tqllm::bounds_by_budget;
    const std::vector<uint32_t> even;
    const auto a = layer_range(64, 0, 2, even), b = layer_range(64, 1, 2, even);
    check(a.lo == 0 && a.hi == 32 && b.lo == 32 && b.hi == 64, "64 layers on two GPUs: 32 + 32");
    const auto c = layer_range(65, 0, 2, even), d = layer_range(65, 1, 2, even);
    check(c.hi == d.lo && c.lo == 0 && d.hi == 65, "odd count: contiguous, covers every layer");
    const auto e = layer_range(64, 0, 2, {40}), f = layer_range(64, 1, 2, {40});
    check(e.hi == 40 && f.lo == 40 && f.hi == 64, "--split-at 40");
    check(layer_range(64, 0, 1, even).hi == 64, "one GPU: every layer");
    check(a.owns(31) && !a.owns(32) && b.owns(32) && a.first() && !b.first(), "ownership");
    check(throws([] { layer_range(64, 0, 2, {64}); }), "--split-at out of range refused");

    // N ranks
    const auto r3 = split_ranges(64, 3, even);
    check(covers_once(r3, 64) && r3[0].hi == 21 && r3[1].hi == 42, "64 layers on three GPUs: 21 + 21 + 22");
    const auto m3 = split_ranges(64, 3, {20, 44});
    check(covers_once(m3, 64) && m3[0].hi == 20 && m3[1].lo == 20 && m3[1].hi == 44 && m3[2].lo == 44, "--split-at 20,44 on three ranks");
    check(throws([] { split_ranges(64, 3, {40}); }), "three ranks need two first layers");
    check(throws([] { split_ranges(64, 3, {40, 30}); }), "first layers must increase");
    check(throws([] { split_ranges(64, 3, {20, 20}); }), "an empty rank is refused");
    check(throws([] { split_ranges(64, 2, {0}); }), "rank 1 cannot start at layer 0");
    check(throws([&] { split_ranges(4, 5, even); }), "more ranks than layers refused");
    check(throws([&] { split_ranges(64, 9, even); }), "more than kMaxSplitRanks refused");
    check(tqllm::layer_owner(m3, 0) == 0 && tqllm::layer_owner(m3, 20) == 1 && tqllm::layer_owner(m3, 63) == 2 &&
          tqllm::layer_owner(m3, 64) == -1, "layer owner");
    check(tqllm::range_bounds(m3) == std::vector<uint32_t>({20, 44}), "bounds round trip");
    check(tqllm::parse_split_at("24") == std::vector<uint32_t>({24}) && tqllm::parse_split_at("20,44") == std::vector<uint32_t>({20, 44}),
          "--split-at list parses");
    check(throws([] { tqllm::parse_split_at("20,,44"); }) && throws([] { tqllm::parse_split_at("x"); }) &&
          throws([] { tqllm::parse_split_at("0"); }) && throws([] { tqllm::parse_split_at("20,"); }), "bad --split-at refused");

    {   // split by free memory: equal costs, budgets 24 / 24 / 16 GB -> 24 / 24 / 16 of 64 layers
        const std::vector<uint64_t> cost(64, 400);
        const auto bb = bounds_by_budget(cost, {24, 24, 16});
        check(bb == std::vector<uint32_t>({24, 48}), "budgets 24/24/16: layers 24/24/16");
        check(bounds_by_budget(cost, {10, 10}) == std::vector<uint32_t>({32}), "equal budgets: even split");
        check(bounds_by_budget(cost, {16, 24, 24}) == std::vector<uint32_t>({16, 40}), "small card first: 16/24/24");
        // a tiny budget still gets one layer, and every other rank at least one
        const auto tiny = bounds_by_budget(std::vector<uint64_t>(8, 1), {1, 1000, 1});
        const auto rt = split_ranges(8, 3, tiny);
        check(covers_once(rt, 8) && rt[0].hi - rt[0].lo == 1 && rt[2].hi - rt[2].lo == 1, "at least one layer per rank");
        // uneven layer costs: the attention layers (every 4th) cost 3x; rank 0 has twice the budget of rank 1
        std::vector<uint64_t> mixed(48, 1);
        for (size_t l = 3; l < 48; l += 4) mixed[l] = 3;  // 72 units in all
        const auto bm = bounds_by_budget(mixed, {2, 1});
        uint64_t c0 = 0; for (uint32_t l = 0; l < bm[0]; ++l) c0 += mixed[l];
        check(c0 >= 46 && c0 <= 50, "uneven costs: rank 0 gets about 2/3 of the cost");
        check(throws([] { bounds_by_budget(std::vector<uint64_t>(4, 1), {0, 0}); }), "no budget anywhere refused");
        // RAM caps (Flash-Next: the experts a box can pin): .188 (125 GB RAM) + .77 (31 GB), 48 layers of 1.1 GB, budgets
        // 21.6 / 23.1 GB: by budget alone .77 got 24 layers (26.5 GB, more than 0.6 x 31 GB); capped at 18.6 GB it takes 16
        const std::vector<uint64_t> fl(48, 1100);
        check(bounds_by_budget(fl, {216, 231}) == std::vector<uint32_t>({23}), "Flash-Next 2 ranks by budget: 23 / 25");
        const auto capped = bounds_by_budget(fl, {216, 231}, {0, 18600});
        check(capped == std::vector<uint32_t>({32}), "a capped rank takes what its RAM pins (16 layers, never over the cap), the rest goes to rank 0");
        check(bounds_by_budget(fl, {231, 216}, {18600, 0}) == std::vector<uint32_t>({16}), "a capped first rank: 16 layers");
        check(bounds_by_budget(fl, {216, 231, 145}, {0, 18600, 9000}) == std::vector<uint32_t>({23, 39}), "two capped ranks: 16 and 8 layers");
        check(bounds_by_budget(fl, {216, 231, 145}, {0, 25000, 25000}) == bounds_by_budget(fl, {216, 231, 145}),
              "caps that do not bind change nothing");
        const auto all_capped = bounds_by_budget(fl, {10, 10}, {11000, 33000});
        check(all_capped == std::vector<uint32_t>({12}), "caps that cannot hold the model: shares follow the caps (1 : 3)");
        // the result is a valid --split-at for split_ranges
        check(covers_once(split_ranges(64, 3, bb), 64), "bounds_by_budget feeds split_ranges");
    }
    {   // Flash-Next trunk on two ranks (48 layers): rank 0 runs 0..23 and sends, rank 1 receives and runs 24..47, rank 1's
        // output is broadcast; every layer runs exactly once over the ranks
        using tqllm::trunk_plan;
        const auto r0 = trunk_plan(layer_range(48, 0, 2, even), 0, 2), r1 = trunk_plan(layer_range(48, 1, 2, even), 1, 2);
        check(r0.recv_from == -1 && r0.send_to == 1 && r0.lo == 0 && r0.hi == 24, "rank 0: no receive, layers 0..23, sends to rank 1");
        check(r1.recv_from == 0 && r1.send_to == -1 && r1.lo == 24 && r1.hi == 48, "rank 1: receives from rank 0, layers 24..47, sends nothing");
        check(r0.root == 1 && r1.root == 1, "the last rank's trunk output goes to every rank");
        int runs[48] = {};
        for (const auto & p : {r0, r1}) for (uint32_t l = p.lo; l < p.hi; ++l) ++runs[l];
        bool once = true; for (int n : runs) once = once && n == 1;
        check(once, "every trunk layer runs on exactly one rank");
        const auto one = trunk_plan(layer_range(48, 0, 1, even), 0, 1);
        check(one.recv_from == -1 && one.send_to == -1 && one.lo == 0 && one.hi == 48, "one GPU: the whole trunk, no hops");
        // three ranks: a chain 0 -> 1 -> 2, the last broadcasts
        const auto t = split_ranges(48, 3, {20, 36});
        const auto p1 = trunk_plan(t[1], 1, 3), p2 = trunk_plan(t[2], 2, 3);
        check(p1.recv_from == 0 && p1.send_to == 2 && p1.lo == 20 && p1.hi == 36, "middle rank: receives from 0, sends to 2");
        check(p2.recv_from == 1 && p2.send_to == -1 && p2.root == 2, "last of three: receives from 1, its rows are broadcast");
    }
    {   // trunk_ops: on every pair of ranks the sends and receives match in order, and every rank joins the same broadcasts
        // in the same order (a mismatch hangs NCCL); DFlash2 target layers 5, 15, 25, 35, 45 over 48 layers
        using tqllm::TrunkOp;
        const std::vector<int> feats = {5, 15, 25, 35, 45};
        auto consistent = [&](const std::vector<tqllm::LayerRange> & rr, bool to_all, const std::vector<int> & fl) {
            const int w = int(rr.size());
            std::vector<std::vector<TrunkOp>> ops;
            for (int k = 0; k < w; ++k) ops.push_back(tqllm::trunk_ops(rr, k, to_all, fl));
            for (int a = 0; a < w; ++a) for (int b = 0; b < w; ++b) if (a != b) {
                std::vector<int> sends, recvs;  // what a sends to b, what b receives from a (kind and feature, in order)
                for (const auto & o : ops[size_t(a)]) if ((o.kind == TrunkOp::send_rows || o.kind == TrunkOp::send_feat) && o.peer == b) sends.push_back(o.feat);
                for (const auto & o : ops[size_t(b)]) if ((o.kind == TrunkOp::recv_rows || o.kind == TrunkOp::recv_feat) && o.peer == a) recvs.push_back(o.feat);
                if (sends != recvs) return false;
            }
            std::vector<std::vector<std::pair<int, int>>> bc(static_cast<size_t>(w));
            for (int k = 0; k < w; ++k) for (const auto & o : ops[size_t(k)]) if (o.kind == TrunkOp::bcast_rows || o.kind == TrunkOp::bcast_feat) bc[size_t(k)].push_back({o.peer, o.feat});
            for (int k = 1; k < w; ++k) if (bc[size_t(k)] != bc[0]) return false;
            for (int k = 0; k < w; ++k) {  // every rank runs its layers once, after its receive of rows
                int layers = 0, seen_recv = k == 0;
                for (const auto & o : ops[size_t(k)]) { if (o.kind == TrunkOp::recv_rows) seen_recv = 1; if (o.kind == TrunkOp::layers) { ++layers; if (!seen_recv) return false; } }
                if (layers != 1) return false;
            }
            return true;
        };
        for (int w = 1; w <= 4; ++w) for (bool all : {true, false}) {
            const auto rr = split_ranges(48, w, even);
            check(consistent(rr, all, feats) && consistent(rr, all, {}), "trunk ops match on every pair (even split)");
        }
        check(consistent(split_ranges(48, 3, {40, 46}), false, feats), "trunk ops match: features owned by every rank, head on the last");
        // head_only: the last rank gets each feature from its owner and no rows go back; to_all: everyone gets everything
        const auto r3 = split_ranges(48, 3, {16, 32});
        const auto last = tqllm::trunk_ops(r3, 2, false, feats), first = tqllm::trunk_ops(r3, 0, false, feats);
        int recv_feat = 0, bcast = 0; for (const auto & o : last) { recv_feat += o.kind == TrunkOp::recv_feat; bcast += o.kind == TrunkOp::bcast_rows; }
        check(recv_feat == 3 && bcast == 0, "head_only: the last rank receives the 3 features it does not own (5, 15, 25), no rows broadcast");
        int sends = 0; for (const auto & o : first) sends += o.kind == TrunkOp::send_feat;
        check(sends == 2, "head_only: rank 0 sends its two target layers (5, 15) to the last rank");
        const auto mid = tqllm::trunk_ops(r3, 1, true, feats);
        int bc = 0; for (const auto & o : mid) bc += o.kind == TrunkOp::bcast_rows || o.kind == TrunkOp::bcast_feat;
        check(bc == 6, "to_all: rows and all 5 features broadcast");
    }
    {   // host RAM room: refuse a rank that cannot hold its cheapest layer; ask for the roomiest box last
        const auto none = tqllm::room_check({false, false}, {0, 0}, 1000);
        check(none.refused.empty() && !none.move_head, "no RAM limits: nothing refused, nothing to move");
        const auto small_last = tqllm::room_check({true, true}, {78000, 2600}, 1100);
        check(small_last.refused.empty() && small_last.move_head, ".77 last with 2.6 GB of room vs .188 78 GB: move the head");
        const auto too_small = tqllm::room_check({true, true}, {78000, 900}, 1100);
        check(too_small.refused == std::vector<int>({1}), "a rank below one layer is refused");
        const auto big_last = tqllm::room_check({true, true}, {7600, 78000}, 1100);
        check(big_last.refused.empty() && !big_last.move_head, "the roomiest box already last: no hint");
    }
    if (failures) return 1;
    std::puts("layer_split: ok");
    return 0;
}
