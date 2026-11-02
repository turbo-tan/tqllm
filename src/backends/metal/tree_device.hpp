#pragma once
// Token trees built and walked on the device, so a tree step needs no host round trip between the draft, the verify and
// the next draft (the draft-ahead pipeline of spec_pipeline.hpp, TreeAhead). Plain MLX ops only: the same code runs on
// the CPU backend, where tests/mlx_tree_device.cpp checks it against the host selector (src/spec/draft_tree.cpp).
//
// device_tree: the best-first tree of src/spec/draft_tree.cpp (same scores: weighted unary logit plus the selector's
// predecessor/successor edge, over tau, log-softmax per expansion; the best `children` candidates per expansion; `nodes`
// rows). A child's cumulative score never exceeds its parent's (log-probabilities), so the best-first tree is the top
// `nodes` nodes by cumulative score of the implicit tree in which every node has its best `children` candidates as
// children; level by level only the best `nodes` of a depth can belong to it, so each level keeps at most `nodes`.
// Rows come out in depth-first order (siblings best first, as the host builds them), row 0 the anchor.
//
// tree_path: the target's greedy walk of a verified tree as array ops: a row is on the accepted path iff it and all its
// ancestors equal the target's choice after their parent. Sibling tokens are distinct (top-16 candidates), so the
// accepted rows form one path; in depth-first order its rows ascend.
#include <mlx/mlx.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tqllm::mlx_backend {
struct DeviceTreeOptions {
    int nodes = 15;       // tree rows below the anchor
    int children = 4;     // candidates kept per expansion
    float tau = 1.5f;     // score temperature
    float edge = 0.6f;    // weight of the predecessor/successor edge
    float unary_weight = 1.0f;
};

struct DeviceTree {
    mlx::core::array ids;      // uint32 [n]: row 0 the anchor, then the tree rows in depth-first order
    mlx::core::array parents;  // int32 [n]: each row's parent row, -1 for row 0
    mlx::core::array depth;    // int32 [n]: rows below the anchor (0 for row 0)
    mlx::core::array visible;  // bool [n, n]: row t sees row a (a is t or one of its ancestors)
    int rows = 0;              // n = nodes + 1 (static: the graph never waits for the tree's size)
};

// visible and depth from a parent array (int32 [n], parent < row, -1 for row 0): ancestor closure by repeated squaring
inline std::pair<mlx::core::array, mlx::core::array> tree_ancestors(const mlx::core::array& parents) {
    namespace mx = mlx::core;
    const int n = parents.shape(0);
    auto one = mx::astype(mx::equal(mx::reshape(parents, {n, 1}), mx::reshape(mx::arange(n, mx::int32), {1, n})), mx::float32);
    auto reach = mx::add(mx::eye(n, mx::float32), one);
    for (int span = 1; span < n; span *= 2) reach = mx::minimum(mx::add(reach, mx::matmul(reach, reach)), mx::array(1.0f));
    auto visible = mx::greater(reach, mx::array(0.5f));
    auto depth = mx::subtract(mx::sum(mx::astype(visible, mx::int32), 1), mx::array(1, mx::int32));
    return {visible, depth};
}

// lattice: the drafter's block rows (row 0 the anchor's, rows 1 .. positions the drafted positions): mapped [rows, 16]
// candidate ids, unary [rows, 16] logits, gate [rows, rank], successors [rows, 16, rank]; predecessor [vocab, rank] is
// the selector's table (read only for the anchor and the candidates); anchor uint32 scalar
inline DeviceTree device_tree(const mlx::core::array& mapped, const mlx::core::array& unary, const mlx::core::array& gate,
                              const mlx::core::array& successors, const mlx::core::array& predecessor,
                              const mlx::core::array& anchor, int positions, const DeviceTreeOptions& o = {}) {
    namespace mx = mlx::core;
    const int topk = mapped.shape(1), rank = gate.shape(1);
    if (positions < 1 || mapped.shape(0) < positions + 1 || unary.shape(0) < positions + 1 || topk > 16 ||
        successors.shape(1) != topk || successors.shape(2) != rank || predecessor.shape(1) != rank)
        throw std::invalid_argument("device tree: invalid lattice");
    if (o.nodes < 1 || o.nodes > 63 || o.children < 1 || o.children > topk || !(o.tau > 0))
        throw std::invalid_argument("device tree: invalid options");
    const int P = positions, k = o.children, N = o.nodes, n = N + 1;
    auto tok = mx::astype(mx::slice(mapped, {1, 0}, {P + 1, topk}), mx::uint32);
    auto un = mx::astype(mx::slice(unary, {1, 0}, {P + 1, topk}), mx::float32);
    auto ga = mx::astype(mx::slice(gate, {1, 0}, {P + 1, rank}), mx::float32);
    auto weighted = mx::multiply(mx::astype(mx::slice(successors, {1, 0, 0}, {P + 1, topk, rank}), mx::float32),
                                 mx::reshape(ga, {P, 1, rank}));  // [P, topk, rank]: gate x successor
    // edge scores: position 0 against the anchor's predecessor row, position d >= 1 against each candidate of d - 1
    auto anchor_row = mx::astype(mx::take(predecessor, mx::reshape(mx::astype(anchor, mx::uint32), {1}), 0), mx::float32);
    auto edge0 = mx::reshape(mx::matmul(anchor_row, mx::transpose(mx::reshape(mx::slice(weighted, {0, 0, 0}, {1, topk, rank}),
                                                                             {topk, rank}))), {topk});
    auto scaled = [&](const mx::array& u, const mx::array& e) {
        auto s = mx::divide(mx::add(mx::multiply(u, mx::array(o.unary_weight)), mx::multiply(e, mx::array(o.edge))), mx::array(o.tau));
        // log-softmax per expansion; a child is kept strictly below its parent so the top nodes stay ancestor-closed
        return mx::minimum(mx::subtract(s, mx::logsumexp(s, -1, true)), mx::array(-1e-6f));
    };
    auto first = scaled(mx::reshape(mx::slice(un, {0, 0}, {1, topk}), {topk}), edge0);  // [topk]
    std::optional<mx::array> later;  // [P - 1, topk parents, topk children]
    if (P > 1) {
        auto parent_rows = mx::astype(mx::take(predecessor, mx::reshape(mx::slice(tok, {0, 0}, {P - 1, topk}), {-1}), 0), mx::float32);
        auto edges = mx::matmul(mx::reshape(parent_rows, {P - 1, topk, rank}),
                                mx::transpose(mx::slice(weighted, {1, 0, 0}, {P, topk, rank}), {0, 2, 1}));
        later = scaled(mx::reshape(mx::slice(un, {1, 0}, {P, topk}), {P - 1, 1, topk}), edges);
    }
    // keys: a node's path of child ranks in base (k + 1), the root's digit first, so sorting keys is a depth-first walk
    std::vector<int64_t> place(size_t(P), 1);
    for (int d = P - 2; d >= 0; --d) place[size_t(d)] = place[size_t(d) + 1] * int64_t(k + 1);
    auto digits = mx::reshape(mx::astype(mx::arange(1, k + 1), mx::int64), {1, k});
    std::vector<mx::array> all_score, all_key, all_parent, all_token;
    // level 0: the anchor's best k children
    auto order = mx::slice(mx::argsort(mx::negative(first)), {0}, {k});
    auto score = mx::take(first, order);
    auto cand = order;
    auto key = mx::multiply(mx::reshape(digits, {k}), mx::array(place[0], mx::int64));
    auto parent_key = mx::zeros({k}, mx::int64);
    int width = k;
    for (int d = 0;; ++d) {
        all_score.push_back(score);all_key.push_back(key);all_parent.push_back(parent_key);
        all_token.push_back(mx::take(mx::reshape(mx::slice(tok, {d, 0}, {d + 1, topk}), {topk}), cand));
        if (d + 1 >= P) break;
        // children of this level's nodes: the next position's scores against each node's candidate
        auto rows = mx::take(mx::reshape(mx::slice(*later, {d, 0, 0}, {d + 1, topk, topk}), {topk, topk}), cand, 0);  // [width, topk]
        auto best = mx::slice(mx::argsort(mx::negative(rows), 1), {0, 0}, {width, k});
        auto child = mx::add(mx::reshape(score, {width, 1}), mx::take_along_axis(rows, best, 1));
        auto child_key = mx::add(mx::reshape(key, {width, 1}), mx::multiply(digits, mx::array(place[size_t(d) + 1], mx::int64)));
        auto child_parent = mx::broadcast_to(mx::reshape(key, {width, 1}), {width, k});
        const int next = std::min(N, width * k);
        auto keep = mx::slice(mx::argsort(mx::negative(mx::reshape(child, {-1}))), {0}, {next});
        score = mx::take(mx::reshape(child, {-1}), keep);
        key = mx::take(mx::reshape(child_key, {-1}), keep);
        parent_key = mx::take(mx::reshape(child_parent, {-1}), keep);
        cand = mx::take(mx::reshape(best, {-1}), keep);
        width = next;
    }
    auto scores = mx::concatenate(all_score, 0);
    const int total = scores.shape(0);
    if (total < N) throw std::invalid_argument("device tree: the lattice has fewer nodes than the tree");
    auto top = mx::slice(mx::argsort(mx::negative(scores)), {0}, {N});
    auto keys = mx::take(mx::concatenate(all_key, 0), top);
    auto dfs = mx::argsort(keys);  // depth-first: a parent's key is a prefix (smaller), siblings by rank
    keys = mx::take(keys, dfs);
    auto parent_keys = mx::take(mx::take(mx::concatenate(all_parent, 0), top), dfs);
    auto tokens = mx::take(mx::take(mx::concatenate(all_token, 0), top), dfs);
    auto row_keys = mx::concatenate({mx::zeros({1}, mx::int64), keys}, 0);  // row 0: the anchor (key 0)
    auto parent_rows = mx::astype(mx::argmax(mx::equal(mx::reshape(parent_keys, {N, 1}), mx::reshape(row_keys, {1, n})), 1), mx::int32);
    DeviceTree out{mx::concatenate({mx::reshape(mx::astype(anchor, mx::uint32), {1}), tokens}, 0),
                   mx::concatenate({mx::array({-1}, mx::int32), parent_rows}, 0), mx::array(0), mx::array(false), n};
    auto [visible, depth] = tree_ancestors(out.parents);
    out.visible = visible;out.depth = depth;
    return out;
}

struct DevicePath {
    mlx::core::array accepted;     // int32 scalar: drafted rows kept (the path's rows below the anchor)
    mlx::core::array rows;         // int32 [n]: the path's rows first (row 0, then ascending), then the rest
    mlx::core::array next_anchor;  // uint32 scalar: the target's choice after the path's last row
};
// choices: the target's argmax per verify row (uint32 [n])
inline DevicePath tree_path(const DeviceTree& tree, const mlx::core::array& choices) {
    namespace mx = mlx::core;
    const int n = tree.rows;
    auto index = mx::arange(n, mx::int32);
    auto after_parent = mx::take(choices, mx::maximum(tree.parents, mx::array(0, mx::int32)));
    auto match = mx::logical_or(mx::equal(tree.ids, after_parent), mx::equal(index, mx::array(0, mx::int32)));
    auto misses = mx::matmul(mx::astype(tree.visible, mx::float32), mx::reshape(mx::astype(mx::logical_not(match), mx::float32), {n, 1}));
    auto on_path = mx::equal(mx::reshape(misses, {n}), mx::array(0.0f));
    auto accepted = mx::subtract(mx::sum(mx::astype(on_path, mx::int32)), mx::array(1, mx::int32));
    auto rows = mx::astype(mx::argsort(mx::where(on_path, index, mx::add(index, mx::array(n, mx::int32)))), mx::int32);
    auto last = mx::take(rows, accepted);
    return {accepted, rows, mx::astype(mx::take(choices, last), mx::uint32)};
}
}  // namespace tqllm::mlx_backend
