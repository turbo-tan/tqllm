#pragma once
// Draft-ahead speculative decoding (greedy): each step submits verify(k) and, chained behind it on the GPU, the
// drafter's inject of verify(k)'s accepted feature rows and draft(k+1), all from device values (accepted count, next
// anchor, next position), then waits only for verify(k)'s choices. The host's accept / output / graph building for
// step k+1 runs while the GPU drafts, so the GPU does not idle between verify and the next draft. Exact: the same
// verify and accept as the synchronous loop; only the draft's timing moves.
#include "attention_decode.hpp"
#include "attention_exact.hpp"
#include "dflash.hpp"
#include "qwen35.hpp"
#include "sampling_topk.hpp"
#include "tree_device.hpp"
#include "tree_walk.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace tqllm::mlx_backend {
class DraftAhead {
public:
    DraftAhead(Qwen35& engine, DFlash2& drafter, int width) : engine_(engine), drafter_(drafter), width_(width) {
        if (width < 1 || width > 15) throw std::invalid_argument("draft-ahead width must be 1..15");
        drafter_.reset_inject_block();
    }
    // after the prefill and its host inject: the first draft (`width` tokens, 1..width) from the pending token at the
    // engine's position
    void start(uint32_t pending, int width = 0) {
        anchor_ = mlx::core::array(pending);
        position_ = mlx::core::array(engine_.position(), mlx::core::int32);
        drafted_width_ = clamp_width(width);
        drafted_ = drafter_.draft_device_at(*anchor_, *position_, drafted_width_);
        mlx::core::async_eval(*drafted_);
    }
    bool active() const {return drafted_.has_value();}
    // tokens in the draft waiting for the next step (the verify width cannot exceed it)
    int drafted_width() const {return drafted_width_;}
    void stop() {anchor_.reset(); position_.reset(); drafted_.reset();}
    struct Step {
        std::vector<uint32_t> proposed;  // the verified drafted tokens
        std::vector<uint32_t> choices;   // the target's argmax per verify row (verified + 1)
        int accepted = 0;                // drafted tokens equal to the target's choice, from the first
        // host phases of this step (ms): graph build of verify(k), submit (encode of verify + next draft), wait for
        // verify(k), post (readback, accept, settle). GPU idle shows as build + submit beyond the previous draft's run
        double build_ms = 0, submit_ms = 0, wait_ms = 0, post_ms = 0;
    };
    // verify the first `verify` (1..drafted_width) drafted tokens; the next draft is `next_width` tokens (1..width,
    // 0: width): a 7-token draft runs the drafter's native 8-row block, which drafts better than the head of a 16-draft
    Step step(int verify, int next_width = 0) {
        namespace mx = mlx::core;
        if (!active()) throw std::logic_error("draft-ahead step before start");
        if (verify < 1 || verify > drafted_width_) throw std::invalid_argument("draft-ahead verify width must be 1..drafted width");
        next_width = clamp_width(next_width);
        using Clock = std::chrono::steady_clock;
        auto ms_since = [](Clock::time_point t) {return std::chrono::duration<double, std::milli>(Clock::now() - t).count();};
        const auto t0 = Clock::now();
        const int pos = engine_.position();
        auto drafted = verify == drafted_width_ ? *drafted_ : mx::slice(*drafted_, {0}, {verify});
        auto block = mx::concatenate({mx::reshape(*anchor_, {1}), drafted}, 0);
        Qwen35::Verification ticket;
        engine_.defer_eval(true);
        mx::array verified = mx::array(0.0f);
        try {verified = engine_.verify_device(block, ticket);} catch (...) {engine_.defer_eval(false); throw;}
        engine_.defer_eval(false);
        auto choices = mx::astype(mx::argmax(verified, -1), mx::uint32);
        auto equal = mx::astype(mx::equal(mx::slice(choices, {0}, {verify}), drafted), mx::int32);
        auto accepted = mx::sum(mx::cumprod(equal, 0), false);
        auto next_anchor = mx::take(choices, accepted);
        auto rows = mx::add(accepted, mx::array(1, mx::int32));
        // two submissions: the wait below covers verify(k) only. One async_eval of {choices, next} signals once, after
        // the next draft too, so the host would wait for the draft and build verify(k+1) with the GPU idle (j771: +2.7-3.8 ms)
        const auto t1 = Clock::now();
        mx::async_eval({choices, accepted, next_anchor, rows});
        drafter_.inject_device(engine_.feature_states(), rows, *position_);
        auto next_position = mx::add(*position_, rows);
        auto next = drafter_.draft_device_at(next_anchor, next_position, next_width);
        mx::async_eval(next);
        const auto t2 = Clock::now();
        mx::eval({choices, accepted, drafted});
        const auto t3 = Clock::now();
        Step out;
        out.build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        out.submit_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        out.wait_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        out.accepted = accepted.item<int32_t>();
        out.proposed.assign(drafted.data<uint32_t>(), drafted.data<uint32_t>() + verify);
        out.choices.assign(choices.data<uint32_t>(), choices.data<uint32_t>() + verify + 1);
        if (engine_.position() != pos + verify + 1) throw std::logic_error("draft-ahead: verify moved the position unexpectedly");
        engine_.accept_prefix(ticket, size_t(out.accepted + 1));
        drafter_.settle_device_inject(out.accepted + 1);
        anchor_ = next_anchor; position_ = next_position; drafted_ = next; drafted_width_ = next_width;
        out.post_ms = ms_since(t3);
        return out;
    }
private:
    int clamp_width(int w) const {return w < 1 || w > width_ ? width_ : w;}
    Qwen35& engine_;
    DFlash2& drafter_;
    int width_, drafted_width_ = 0;
    std::optional<mlx::core::array> anchor_, position_, drafted_;
};
// Draft-ahead for token trees: the DFlash2 lattice of step k + 1 becomes a tree on the device (tree_device.hpp) and is
// chained behind verify(k) on the GPU like the chain's draft, from device values (the walked path's rows, the next
// anchor and position). The host reads back only the verify's choices, the tree and the walked path; it commits the path
// (accept_path) while the GPU drafts. Exact: the verify, the walk and the commit are the synchronous tree step's.
// Sampled requests (sample()): each verified row draws its token on the device (sample_rows) in place of the argmax, so
// the walk and the next draft stay on the GPU.
class TreeAhead {
public:
    static constexpr int kPositions = 15;  // the drafter's lattice positions below the anchor (DFlash2: 16 rows)
    TreeAhead(Qwen35& engine, DFlash2& drafter, const DeviceTreeOptions& options = {})
        : engine_(engine), drafter_(drafter), options_(options) {
        // the tree may verify more rows than the drafter's block: its accepted path then needs a two-block inject
        drafter_.inject_block(std::min(std::max(options_.nodes + 1, kPositions + 1), 2 * (kPositions + 1)));
    }
    void sample(float temperature, int top_k, float top_p, std::mt19937_64* rng) {
        temperature_ = temperature; top_k_ = top_k; top_p_ = top_p; rng_ = rng;
    }
    void start(uint32_t pending) {
        anchor_ = mlx::core::array(pending);
        position_ = mlx::core::array(engine_.position(), mlx::core::int32);
        tree_ = draft(*anchor_, *position_, chain_at(engine_.position(), engine_.position()));
    }
    bool active() const {return tree_.has_value();}
    void stop() {anchor_.reset(); position_.reset(); tree_.reset();}
    struct Step {
        std::vector<uint32_t> ids;       // the verified rows' tokens (row 0 the anchor)
        std::vector<int32_t> path;       // the accepted path's rows (row 0 first)
        std::vector<uint32_t> choices;   // the target's token per verified row (argmax, or its draw when sampling)
        int accepted = 0;                // drafted tokens kept (path rows below the anchor)
        // host phases of this step (ms): graph build of verify(k), submit (encode of verify + next draft), wait for
        // verify(k), post (readback, accept, settle). GPU idle shows as build + submit beyond the previous draft's run
        double build_ms = 0, submit_ms = 0, wait_ms = 0, post_ms = 0;
        double inject_ms = 0, encode_ms = 0;  // inside submit: the inject, then the next tree's encode
        double vsubmit_ms = 0;                // inside build: the verify's submission, i.e. what the GPU had left
    };
    // max_rows: the rows (path + bonus) the request can still emit; a longer accepted path is cut on the device before the
    // commit and the drafter's inject, so the target and drafter hold exactly the emitted text
    // copy: tokens that followed an earlier repeat of the committed text (src/spec/suffix_copy.hpp); when given, this pass
    // verifies anchor + the copy as a chain (rows 1 .. c) beside the drafted tree's best first draft (the last row), in the
    // same 16 rows. The walk is unchanged, so the text is the target's whatever was proposed; the chain comes first, so a
    // first draft equal to the copy's first token follows the copy.
    Step step(int max_rows = 1 << 30, const std::vector<uint32_t>* copy = nullptr) {
        namespace mx = mlx::core;
        if (!active()) throw std::logic_error("tree draft-ahead step before start");
        using Clock = std::chrono::steady_clock;
        auto ms = [](Clock::time_point a, Clock::time_point b) {return std::chrono::duration<double, std::milli>(b - a).count();};
        const auto t0 = Clock::now();
        const int pos = engine_.position();
        const DeviceTree tree = copy && !copy->empty() ? copy_tree(*tree_, *copy, pos) : *tree_;
        Qwen35::Verification ticket;
        engine_.defer_eval(true);
        mx::array verified = mx::array(0.0f);
        try {verified = engine_.verify_tree_device(tree.ids, tree.parents, tree.depth, tree.visible, ticket);}
        catch (...) {engine_.defer_eval(false); throw;}
        engine_.defer_eval(false);
        const double vsubmit_ms = engine_.defer_submit_ms();
        auto choices = temperature_ > 0.f ? sample_rows(mx::reshape(verified, {tree.rows, -1}), temperature_, top_k_, top_p_, (*rng_)())
                                          : mx::astype(mx::argmax(verified, -1), mx::uint32);
        auto walk = tree_path(tree, choices);
        if (max_rows < tree.rows) {
            walk.accepted = mx::minimum(walk.accepted, mx::array(std::max(max_rows, 1) - 1, mx::int32));
            walk.next_anchor = mx::astype(mx::take(choices, mx::take(walk.rows, walk.accepted)), mx::uint32);
        }
        auto rows = mx::add(walk.accepted, mx::array(1, mx::int32));
        const auto t1 = Clock::now();
        // two submissions, as DraftAhead: the wait below covers verify(k) and its walk only
        mx::async_eval({choices, walk.accepted, walk.rows, walk.next_anchor, rows, tree.ids, tree.parents});
        // the drafter takes the path's feature rows (the path's rows first, `rows` of them)
        drafter_.inject_device(mx::take(engine_.feature_states(), walk.rows, 0), rows, *position_);
        const auto t_inject = Clock::now();
        auto next_position = mx::add(*position_, rows);
        auto next = draft(walk.next_anchor, next_position, chain_at(pos + 1, pos + tree.rows));
        const auto t2 = Clock::now();
        mx::eval({choices, walk.accepted, walk.rows, tree.ids, tree.parents});
        const auto t3 = Clock::now();
        const int n = tree.rows;
        Step out;
        out.accepted = walk.accepted.item<int32_t>();
        out.ids.assign(tree.ids.data<uint32_t>(), tree.ids.data<uint32_t>() + n);
        out.choices.assign(choices.data<uint32_t>(), choices.data<uint32_t>() + n);
        const int32_t* path_rows = walk.rows.data<int32_t>();
        out.path.assign(path_rows, path_rows + out.accepted + 1);
        const std::vector<int32_t> parents(tree.parents.data<int32_t>(), tree.parents.data<int32_t>() + n);
        // the host walk must agree with the device's: a disagreement would have fed the drafter the wrong rows
        if (const char* e = tqllm::tree_walk_error(parents, out.ids, out.choices, out.path, out.accepted + 1 >= max_rows))
            throw std::logic_error(e);
        if (engine_.position() != pos + n) throw std::logic_error("tree draft-ahead: verify moved the position unexpectedly");
        engine_.accept_path(ticket, out.path, &parents);
        drafter_.settle_device_inject(out.accepted + 1);
        anchor_ = walk.next_anchor; position_ = next_position; tree_ = next;
        out.build_ms = ms(t0, t1); out.submit_ms = ms(t1, t2); out.wait_ms = ms(t2, t3); out.post_ms = ms(t3, Clock::now());
        out.inject_ms = ms(t1, t_inject); out.encode_ms = ms(t_inject, t2); out.vsubmit_ms = vsubmit_ms;
        return out;
    }
private:
    DeviceTree copy_tree(const DeviceTree& drafted, const std::vector<uint32_t>& copy, int pos) const {
        namespace mx = mlx::core;
        const int c = std::min(int(copy.size()), rows() - 2);
        const bool sibling = !chain_at(pos, pos + c + 2);  // exact chains (no matrix unit, or the segment boundary): copy only
        const int n = 1 + c + (sibling ? 1 : 0);
        std::vector<int32_t> parents(static_cast<size_t>(n));
        parents[0] = -1;
        for (int r = 1; r <= c; ++r) parents[size_t(r)] = r - 1;
        if (sibling) parents[size_t(n - 1)] = 0;
        std::vector<mx::array> parts = {mx::slice(drafted.ids, {0}, {1}), mx::array(copy.data(), {c}, mx::uint32)};
        if (sibling) {  // two equal children of the anchor would both be on the device walk's path: a draft equal to the
                        // copy's first token is replaced by another valid id (the copy's row is the one that may match)
            const uint32_t first = copy[0], other = first > 0 ? first - 1 : 1;
            auto draft = mx::slice(drafted.ids, {1}, {2});
            parts.push_back(mx::where(mx::equal(draft, mx::array(first)), mx::array(other), draft));
        }
        DeviceTree t{mx::concatenate(parts, 0), mx::array(parents.data(), {n}, mx::int32), mx::array(0), mx::array(false), n};
        auto [visible, depth] = tree_ancestors(t.parents);
        t.visible = visible; t.depth = depth;
        return t;
    }
    // A block that crosses the first attention segment boundary verifies some rows on the folded path and some on the
    // segmented one in their AR steps; a chain keeps every row's keys where AR has them, so that step drafts a chain.
    // The next tree is drafted before its anchor position is read back, so every anchor the verify could produce counts.
    int rows() const {return options_.nodes + 1;}
    // With the M5 attention kernel (attention_exact.hpp) every tree row is its AR step bit for bit. Without it a side row
    // matches its AR step only to rounding (and across the first segment boundary the rows change kernels), so greedy
    // --exact on requests draft chains there, as does every request at that boundary.
    bool chain_at(int anchor_lo, int anchor_hi) const {
        if (options_.children <= 1) return true;
        if (matrix_unit_available()) return false;
        return (temperature_ <= 0.f && engine_.exact()) || (anchor_lo < kAttentionSegment && anchor_hi + rows() > kAttentionSegment);
    }
    DeviceTree draft(const mlx::core::array& anchor, const mlx::core::array& position, bool chain) {
        auto lattice = drafter_.draft_lattice_device(anchor, position, kPositions);
        DeviceTreeOptions shape = options_;
        if (chain) shape.children = 1;
        auto tree = device_tree(lattice.mapped, lattice.unary, lattice.gate, lattice.successors, drafter_.selector_predecessor(),
                                anchor, kPositions, shape, drafter_.dspine()?&lattice.gate:nullptr);
        mlx::core::async_eval({tree.ids, tree.parents, tree.depth, tree.visible});
        return tree;
    }
    Qwen35& engine_;
    DFlash2& drafter_;
    DeviceTreeOptions options_;
    float temperature_ = 0.f, top_p_ = 1.f;
    int top_k_ = 0;
    std::mt19937_64* rng_ = nullptr;
    std::optional<mlx::core::array> anchor_, position_;
    std::optional<DeviceTree> tree_;
};
}  // namespace tqllm::mlx_backend
