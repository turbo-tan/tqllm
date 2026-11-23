#pragma once
// Pure rules of the qwen35 engine (no CUDA), tested in tests/qwen35_rules.cpp.
#include <array>
#include <cstdint>
#include <vector>

#include "layer_split.hpp"
namespace tqllm::qwen35 {
// Host K/V tail: which of K / V a verify step copies back (stage plan [K bytes, V bytes, K offset, V offset]). With
// nothing to copy the captured node is a 16-byte placeholder between ordinary allocations, and CUDA refuses to update it
// to the mapped K/V (cudaGraphExecMemcpyNodeSetParams1D: invalid argument), so each kind needs its own graph.
inline uint64_t stage_graph_kind(const std::array<int64_t, 4> & s) {
    return uint64_t(s[0] > s[2]) | uint64_t(s[1] > s[3]) << 1;
}
// Columns one batched verify step may hold across slots (drafts per slot shrink as slots grow; the greedy output does not
// depend on the draft length). GA10x GeForce: 16 (3090 c=4: 16 -> 283 / 198, 24 -> 281 / 173, 32 -> 249 / 138 tok/s).
// GB10 (integrated Blackwell): 24 (11-06, TQ4_SX + DFlash2-TQ3, 8 requests total: full width 104.5, 24 -> 135.1, 16 -> 128.0,
// 12 -> 127.0 tok/s; 4 requests 116.7 / 127.9 / 128.9 / 115.6; 1-2 requests unchanged): the verify step grew ~4 ms per column.
// Other Blackwell cards (5060 Ti): the full width, as measured there.
inline int verify_column_budget(int major, bool integrated, int full) {
    if (major < 10) return full < 16 ? full : 16;
    if (integrated) return full < 24 ? full : 24;
    return full;
}
// DFlash2 drafter shapes the draft kernels serve: head_dim 128 (attention_gen128), the top-16 selector lattice and the
// kernel-2 block conv. The FFN width is the drafter's own and is not compared with the target's: the MoE
// Qwen3.6-35B-A3B (512 per expert) pairs with a 6144-wide drafter.
inline bool dflash2_shape_ok(int head_dim, int top_k, int conv_kernel, int ffn) {
    return head_dim == 128 && top_k == 16 && conv_kernel == 2 && ffn > 0;
}
// Columns per row of the drafter's projection scratch: q|k|v, then gate|up of its FFN.
inline uint64_t dflash_mix_cols(int qkv_rows, int ffn) {
    return uint64_t(qkv_rows > 2 * ffn ? qkv_rows : 2 * ffn);
}
// Key rows the drafter's attention scans at most: the K/V ring (or the retrieval scratch beside it), else the context.
inline int dflash_attn_rows(int ring, int retr_rows, int context) {
    if (!ring) return context;
    return ring > retr_rows ? ring : retr_rows;
}
// --split layer: who runs the head, and which of the last rank's decisions travel, for one forward of the qwen35 engine.
// With one sequence (head_only, --np 1) the head, the MTP layer, the DFlash2 drafter and sampling run on the last rank
// only: it sends the block to verify before the forward (block: drafts, tree columns) and the picked tokens after it
// (picks: the first token, the accepted path and the next token), a few ints; the trunk output never travels back. Prefill
// chunks send the DFlash2 feature rows owner -> last rank with the forward; a decode step sends only the accepted rows',
// after the picks (feats_late; MTP steps none). --np > 1 and to_all forwards (Engine::verify, step, drafter_features) keep
// every rank on the broadcast rows and run the head everywhere, as before.
enum class SplitStep {
    chunk,       // a prefill chunk that is not the prompt's last: no head
    last_chunk,  // the prompt's last chunk: the first token
    plain,       // one token, no drafts
    chain,       // a DFlash2 chain
    tree,        // a DFlash2 token tree (and copy / lookup chains)
    mtp,         // MTP drafts
    to_all,      // every rank reads the head output
};
struct HeadPlan {
    bool to_all = true;      // the trunk output and the feature rows go to every rank
    bool head_here = true;   // this rank runs the head (and drafts, samples)
    bool block = false;      // the last rank's block to verify is broadcast before the forward
    bool picks = false;      // the last rank's picked tokens are broadcast after it
    bool feats = true;       // the feature rows travel with the forward
    bool feats_late = false; // the accepted rows' features go owner -> last rank after the picks
};
inline HeadPlan head_plan(SplitStep s, int rank, int world, bool head_only) {
    HeadPlan p;
    if (world < 2 || !head_only || s == SplitStep::to_all) return p;
    p.to_all = false;
    p.head_here = rank == world - 1;
    p.block = s == SplitStep::chain || s == SplitStep::tree || s == SplitStep::mtp;
    p.picks = s != SplitStep::chunk;
    p.feats = s == SplitStep::chunk || s == SplitStep::last_chunk;
    p.feats_late = s == SplitStep::chain || s == SplitStep::tree;
    return p;
}
// One forward's cross-rank operations on `rank`, in order: the block broadcast, the trunk (tqllm::trunk_ops), the head
// (local), the picks broadcast, the accepted rows' features (late). tests/qwen35_rules.cpp: on every pair of ranks the
// sends and receives match, and every rank joins the same broadcasts in the same order.
struct SplitStepOp {
    enum Kind { block, trunk, head, picks, late } kind;
    TrunkOp op{TrunkOp::layers};  // trunk / late: the operation
};
// the feature sends / receives of a head_only forward (owner -> last rank), run after the picks
inline std::vector<TrunkOp> late_feat_ops(const std::vector<LayerRange> & ranges, int rank, const std::vector<int> & feat_layers) {
    std::vector<TrunkOp> out;
    for (const auto & o : trunk_ops(ranges, rank, false, feat_layers))
        if (o.kind == TrunkOp::send_feat || o.kind == TrunkOp::recv_feat) out.push_back(o);
    return out;
}
inline std::vector<SplitStepOp> split_step_ops(SplitStep s, const std::vector<LayerRange> & ranges, int rank, bool head_only,
                                               const std::vector<int> & feat_layers) {
    const int world = int(ranges.size());
    const HeadPlan p = head_plan(s, rank, world, head_only);
    std::vector<SplitStepOp> ops;
    if (p.block) ops.push_back({SplitStepOp::block});
    static const std::vector<int> none;
    if (world > 1)
        for (const auto & o : trunk_ops(ranges, rank, p.to_all, p.to_all || p.feats ? feat_layers : none)) ops.push_back({SplitStepOp::trunk, o});
    if (p.head_here && s != SplitStep::chunk) ops.push_back({SplitStepOp::head});
    if (p.picks) ops.push_back({SplitStepOp::picks});
    if (p.feats_late)
        for (const auto & o : late_feat_ops(ranges, rank, feat_layers)) ops.push_back({SplitStepOp::late, o});
    return ops;
}
// Ints in one block / picks message: a tree's n, tokens, parents and depths (3 x the verify columns), or an accepted path
// of at most `cols` columns plus the next token.
constexpr int split_msg_ints(int cols) { return 1 + 3 * cols; }
}  // namespace tqllm::qwen35
