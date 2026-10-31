#include "draft_tree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tqllm {
namespace {
constexpr int kMaxTopK = 16;  // the drafter's top-16 head
constexpr int kMaxNodes = 63; // TensorFold's own ceiling for one tree

void check(const DraftLattice& lattice, const DraftTreeOptions& options) {
    if (lattice.positions < 1 || lattice.topk < 1 || lattice.topk > kMaxTopK || lattice.rank < 1 || !lattice.tokens ||
        !lattice.unary || !lattice.gate || !lattice.successor || !lattice.predecessor)
        throw std::invalid_argument("invalid draft lattice");
    if (options.nodes < 1 || options.nodes > kMaxNodes || options.children < 1 || options.children > lattice.topk ||
        !std::isfinite(options.tau) || options.tau <= 0 || !std::isfinite(options.edge) ||
        !std::isfinite(options.unary_weight))
        throw std::invalid_argument("invalid draft tree options");
}

// A candidate's selector score before its position's normalisation: the weighted unary logit plus the weighted
// predecessor/successor edge (one predecessor row serves the whole expansion).
float candidate_score(const DraftTreeOptions& options, const float* predecessor, const float* gate, const float* successor,
                      int rank, float unary) {
    double edge = 0;
    for (int r = 0; r < rank; ++r) edge += double(predecessor[r]) * double(gate[r]) * double(successor[r]);
    return (options.unary_weight * unary + options.edge * float(edge)) / options.tau;
}

struct FrontierNode {
    float score;
    int parent;
    uint32_t token;
    int depth;
};

// best first: the higher cumulative score, then the lower parent row, then the lower token
bool better(const FrontierNode& a, const FrontierNode& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.parent != b.parent) return a.parent < b.parent;
    return a.token < b.token;
}
}  // namespace

DraftTree best_first_tree(const DraftLattice& lattice, uint32_t anchor, const DraftTreeOptions& options) {
    check(lattice, options);
    DraftTree tree;
    tree.tokens.reserve(size_t(options.nodes));
    tree.parents.reserve(size_t(options.nodes));
    tree.scores.reserve(size_t(options.nodes));
    std::vector<FrontierNode> frontier;
    frontier.reserve(size_t(kMaxTopK + options.nodes * options.children));
    std::array<float, kMaxTopK> score{};
    std::array<int, kMaxTopK> order{};

    // Score the position's candidates against one parent token (log-softmax, so a child's cumulative score is the log
    // probability of its whole path), then offer the best `children` of them to the frontier.
    auto expand = [&](uint32_t parent_token, int parent, int depth, float cumulative) {
        if (depth >= lattice.positions) return;
        const float* predecessor = lattice.predecessor(lattice.context, parent_token);
        const float* gate = lattice.gate + size_t(depth) * lattice.rank;
        const float* unary = lattice.unary + size_t(depth) * lattice.topk;
        const float* successor = lattice.successor + size_t(depth) * lattice.topk * lattice.rank;
        float maximum = -INFINITY;
        for (int c = 0; c < lattice.topk; ++c) {
            score[c] = candidate_score(options, predecessor, gate, successor + size_t(c) * lattice.rank, lattice.rank,
                                       unary[c]);
            maximum = std::max(maximum, score[c]);
        }
        double total = 0;
        for (int c = 0; c < lattice.topk; ++c) total += std::exp(double(score[c]) - double(maximum));
        const float normalizer = maximum + float(std::log(total));
        for (int c = 0; c < lattice.topk; ++c) {
            score[c] -= normalizer;
            order[c] = c;
        }
        std::sort(order.begin(), order.begin() + lattice.topk, [&](int a, int b) {
            return score[a] > score[b] || (score[a] == score[b] && a < b);
        });
        for (int i = 0; i < std::min(options.children, lattice.topk); ++i) {
            const int candidate = order[i];
            frontier.push_back({cumulative + score[candidate], parent,
                                lattice.tokens[size_t(depth) * lattice.topk + candidate], depth});
        }
    };

    expand(anchor, -1, 0, 0.0f);
    while (!frontier.empty() && int(tree.tokens.size()) < options.nodes) {
        const auto best = std::min_element(frontier.begin(), frontier.end(), better);
        const FrontierNode node = *best;
        frontier.erase(best);
        const int row = int(tree.tokens.size());
        tree.tokens.push_back(node.token);
        tree.parents.push_back(node.parent);
        tree.scores.push_back(node.score);
        expand(node.token, row, node.depth + 1, node.score);
    }
    return tree;
}

DraftTree chain_tree(const DraftLattice& lattice, uint32_t anchor, const DraftTreeOptions& options) {
    DraftTreeOptions chain = options;
    chain.children = 1;
    chain.nodes = lattice.positions;  // no dead node: one token per position
    return best_first_tree(lattice, anchor, chain);
}

DraftTreeSelector draft_tree_selector(std::string_view name) {
    static constexpr struct {
        std::string_view name;
        DraftTreeSelector selector;
    } kSelectors[] = {{"chain", chain_tree}, {"best_first", best_first_tree}};
    for (const auto& entry : kSelectors)
        if (entry.name == name) return entry.selector;
    return nullptr;
}
}  // namespace tqllm