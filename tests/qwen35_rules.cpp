// qwen35_rules.hpp: the batched-verify column budget per device class; the host K/V tail graph kinds; DFlash2 drafter shapes.
#include "qwen35_rules.hpp"
#include <array>
#include <cstdio>
#include <vector>
int main() {
    using tqllm::qwen35::verify_column_budget;
    using tqllm::qwen35::stage_graph_kind;
    using tqllm::qwen35::dflash2_shape_ok;
    using tqllm::qwen35::dflash_mix_cols;
    using tqllm::qwen35::dflash_attn_rows;
    int fails = 0;
    auto expect = [&](bool ok, const char * what) { if (!ok) { std::printf("FAIL %s\n", what); ++fails; } };
    expect(verify_column_budget(8, false, 64) == 16, "3090 (sm_86): 16 columns");
    expect(verify_column_budget(12, true, 64) == 24, "GB10 (integrated Blackwell): 24 columns (np 8 104.5 -> 135.1 tok/s)");
    expect(verify_column_budget(12, false, 64) == 64, "5060 Ti (discrete Blackwell): the full width");
    expect(verify_column_budget(12, true, 8) == 8 && verify_column_budget(8, false, 8) == 8, "never above the full width");
    // 11-17: a request with nothing in host memory captured placeholder copies; the next request that spilled updated them
    // to the mapped K/V and failed ("generation failed: invalid argument", 27B q8_0 K/V + -fit on, 16 GB)
    expect(stage_graph_kind({0, 0, 0, 0}) != stage_graph_kind({1 << 20, 1 << 20, 0, 0}), "nothing staged vs a host tail: two graphs");
    expect(stage_graph_kind({1 << 20, 0, 0, 0}) != stage_graph_kind({1 << 20, 1 << 20, 0, 0}), "K staged, V not yet: its own graph");
    expect(stage_graph_kind({1 << 20, 1 << 20, 0, 0}) == stage_graph_kind({3 << 20, 2 << 20, 0, 0}), "a growing tail updates in place");
    expect(stage_graph_kind({1 << 20, 1 << 20, 1 << 20, 1 << 20}) == stage_graph_kind({0, 0, 0, 0}), "all compacted: placeholders");
    // 11-17: Qwen3.6-35B-A3B (MoE, 512-wide experts) + its DFlash2 drafter (6144 FFN) was refused as "unsupported DFlash2
    // drafter shape": the check compared the drafter's FFN with the target's, true only for the dense 27B
    expect(dflash2_shape_ok(128, 16, 2, 17408), "27B DFlash2 drafter");
    expect(dflash2_shape_ok(128, 16, 2, 6144), "35B-A3B DFlash2 drafter (FFN 6144 beside 512-wide experts)");
    expect(!dflash2_shape_ok(256, 16, 2, 6144) && !dflash2_shape_ok(128, 8, 2, 6144) && !dflash2_shape_ok(128, 16, 3, 6144) &&
               !dflash2_shape_ok(128, 16, 2, 0), "head_dim / top-k / conv kernel / FFN the kernels do not serve");
    // the drafter's gate|up rows (2 x 6144) are 24x wider than a 512-wide expert's FFN scratch
    expect(dflash_mix_cols(4096 + 2 * 1024, 6144) == 12288, "35B drafter: gate|up wider than q|k|v");
    expect(dflash_mix_cols(4096 + 2 * 1024, 1024) == 6144, "a narrow drafter FFN: q|k|v sets the width");
    expect(dflash_attn_rows(2240, 0, 32768) == 2240 && dflash_attn_rows(0, 0, 32768) == 32768 &&
               dflash_attn_rows(2240, 4096, 32768) == 4096, "attention partials: ring, context, retrieval scratch");
    {   // --split layer forwards (split_step_ops): on every pair of ranks the sends and receives match in order, every rank
        // joins the same broadcasts (block / picks from the last rank, rows, features) in the same order (a mismatch hangs
        // NCCL), and with one sequence only the last rank runs the head; 27B: 64 layers, DFlash2 target layers 1 16 31 46 61
        using tqllm::TrunkOp;
        using tqllm::qwen35::SplitStep;
        using tqllm::qwen35::SplitStepOp;
        using tqllm::qwen35::split_step_ops;
        const std::vector<int> feats = {1, 16, 31, 46, 61};
        const SplitStep steps[] = {SplitStep::chunk, SplitStep::last_chunk, SplitStep::plain, SplitStep::chain,
                                   SplitStep::tree, SplitStep::mtp, SplitStep::to_all};
        auto p2p = [](const SplitStepOp & o) { return o.kind == SplitStepOp::trunk || o.kind == SplitStepOp::late; };
        auto is_send = [&](const SplitStepOp & o) { return p2p(o) && (o.op.kind == TrunkOp::send_rows || o.op.kind == TrunkOp::send_feat); };
        auto is_recv = [&](const SplitStepOp & o) { return p2p(o) && (o.op.kind == TrunkOp::recv_rows || o.op.kind == TrunkOp::recv_feat); };
        auto bcast = [](const SplitStepOp & o) {  // {what, root, feature}; what -1: none
            if (o.kind == SplitStepOp::block) return std::array<int, 3>{0, -2, -1};
            if (o.kind == SplitStepOp::picks) return std::array<int, 3>{1, -2, -1};
            if (o.kind == SplitStepOp::trunk && o.op.kind == TrunkOp::bcast_rows) return std::array<int, 3>{2, o.op.peer, -1};
            if (o.kind == SplitStepOp::trunk && o.op.kind == TrunkOp::bcast_feat) return std::array<int, 3>{3, o.op.peer, o.op.feat};
            return std::array<int, 3>{-1, 0, 0};
        };
        auto pairs_up = [&](const std::vector<tqllm::LayerRange> & rr, SplitStep st, bool head_only, const std::vector<int> & fl) {
            const int w = int(rr.size());
            std::vector<std::vector<SplitStepOp>> ops;
            for (int k = 0; k < w; ++k) ops.push_back(split_step_ops(st, rr, k, head_only, fl));
            for (int a = 0; a < w; ++a) for (int b = 0; b < w; ++b) if (a != b) {
                std::vector<int> sends, recvs;
                for (const auto & o : ops[size_t(a)]) if (is_send(o) && o.op.peer == b) sends.push_back(o.op.feat * 2 + (o.kind == SplitStepOp::late));
                for (const auto & o : ops[size_t(b)]) if (is_recv(o) && o.op.peer == a) recvs.push_back(o.op.feat * 2 + (o.kind == SplitStepOp::late));
                if (sends != recvs) return false;
            }
            std::vector<std::vector<std::array<int, 3>>> bc(static_cast<size_t>(w));
            for (int k = 0; k < w; ++k) for (const auto & o : ops[size_t(k)]) if (bcast(o)[0] >= 0) bc[size_t(k)].push_back(bcast(o));
            for (int k = 1; k < w; ++k) if (bc[size_t(k)] != bc[0]) return false;
            for (int k = 0; k < w; ++k) {  // the block comes before the trunk, the picks after the head, late features last
                int seen_trunk = 0, seen_head = 0, seen_picks = 0, heads = 0;
                for (const auto & o : ops[size_t(k)]) {
                    if (o.kind == SplitStepOp::block && seen_trunk) return false;
                    if (o.kind == SplitStepOp::trunk) seen_trunk = 1;
                    if (o.kind == SplitStepOp::head) seen_head = 1, ++heads;
                    if (o.kind == SplitStepOp::picks && k == w - 1 && !seen_head) return false;
                    if (o.kind == SplitStepOp::picks) seen_picks = 1;
                    if (o.kind == SplitStepOp::late && !seen_picks) return false;
                }
                const bool want_head = st != SplitStep::chunk && (!head_only || st == SplitStep::to_all || w == 1 || k == w - 1);
                if (heads != int(want_head)) return false;
            }
            return true;
        };
        for (int w = 1; w <= 4; ++w) for (bool ho : {false, true}) for (SplitStep st : steps) {
            const auto rr = tqllm::split_ranges(64, w, {});
            expect(pairs_up(rr, st, ho, feats) && pairs_up(rr, st, ho, {}), "split step ops pair up (even split)");
        }
        expect(pairs_up(tqllm::split_ranges(64, 3, {50, 62}), SplitStep::tree, true, feats), "split step ops pair up: features on every rank");
        // one sequence on 2 ranks (31 + 33 layers): no rows go back, rank 0 sends its two target layers (1, 16) to the last
        // rank (a prefill chunk with the forward, a decode step after the picks), the block and the picks come from the last
        // rank; a to_all forward broadcasts the rows and all 5 features
        const auto r2 = tqllm::split_ranges(64, 2, {31});
        auto count = [](const std::vector<SplitStepOp> & v, auto pred) { int c = 0; for (const auto & o : v) c += pred(o); return c; };
        const auto r0 = split_step_ops(SplitStep::tree, r2, 0, true, feats), r1 = split_step_ops(SplitStep::tree, r2, 1, true, feats);
        expect(count(r0, [](const SplitStepOp & o) { return o.kind == SplitStepOp::trunk && o.op.kind == TrunkOp::bcast_rows; }) == 0,
               "head_only: no rows back to rank 0");
        expect(count(r0, [](const SplitStepOp & o) { return o.kind == SplitStepOp::trunk && o.op.kind == TrunkOp::send_feat; }) == 0 &&
                   count(r0, [](const SplitStepOp & o) { return o.kind == SplitStepOp::late && o.op.kind == TrunkOp::send_feat; }) == 2 &&
                   count(r1, [](const SplitStepOp & o) { return o.kind == SplitStepOp::late && o.op.kind == TrunkOp::recv_feat; }) == 2,
               "head_only decode: rank 0's two target layers go to the last rank after the picks");
        const auto c0 = split_step_ops(SplitStep::chunk, r2, 0, true, feats);
        expect(count(c0, [](const SplitStepOp & o) { return o.kind == SplitStepOp::trunk && o.op.kind == TrunkOp::send_feat; }) == 2,
               "head_only prefill chunk: the features with the forward");
        const auto m0 = split_step_ops(SplitStep::mtp, r2, 0, true, feats);
        expect(count(m0, [](const SplitStepOp & o) { return o.op.kind == TrunkOp::send_feat; }) == 0, "head_only MTP step: no features");
        expect(count(r0, [](const SplitStepOp & o) { return o.kind == SplitStepOp::head; }) == 0 &&
                   count(r1, [](const SplitStepOp & o) { return o.kind == SplitStepOp::head; }) == 1, "head_only: the head on the last rank");
        expect(count(r0, [](const SplitStepOp & o) { return o.kind == SplitStepOp::block || o.kind == SplitStepOp::picks; }) == 2,
               "head_only tree step: the block and the picks");
        const auto pl = split_step_ops(SplitStep::plain, r2, 0, true, feats), ch = split_step_ops(SplitStep::chunk, r2, 0, true, feats);
        expect(count(pl, [](const SplitStepOp & o) { return o.kind == SplitStepOp::block; }) == 0 &&
                   count(pl, [](const SplitStepOp & o) { return o.kind == SplitStepOp::picks; }) == 1, "plain step: no block, the picks");
        expect(count(ch, [](const SplitStepOp & o) { return o.kind != SplitStepOp::trunk; }) == 0, "a prefill chunk: the trunk only");
        const auto ta = split_step_ops(SplitStep::to_all, r2, 0, true, feats);
        expect(count(ta, [](const SplitStepOp & o) { return o.kind == SplitStepOp::trunk && (o.op.kind == TrunkOp::bcast_rows || o.op.kind == TrunkOp::bcast_feat); }) == 6 &&
                   count(ta, [](const SplitStepOp & o) { return o.kind == SplitStepOp::head; }) == 1, "to_all: rows and 5 features to every rank, the head everywhere");
        const auto np2 = split_step_ops(SplitStep::tree, r2, 0, false, feats);
        expect(count(np2, [](const SplitStepOp & o) { return o.kind == SplitStepOp::block || o.kind == SplitStepOp::picks; }) == 0 &&
                   count(np2, [](const SplitStepOp & o) { return o.kind == SplitStepOp::head; }) == 1, "--np > 1: every rank on the broadcast rows");
        expect(tqllm::qwen35::split_msg_ints(16) == 49, "message: a 16-column tree's n, tokens, parents, depths");
    }
    std::printf("qwen35 rules: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
