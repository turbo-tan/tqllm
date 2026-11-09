// Unit tests for the drafter's tree selectors (src/spec/draft_tree.*): the chain the engine drafts today and the
// TensorFold-style best-first tree, with the score knobs the tree sweeps use. No CUDA, no MLX.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "draft_tree.hpp"

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}
void expect_tree(const tqllm::DraftTree& tree, const std::vector<uint32_t>& tokens, const std::vector<int32_t>& parents,
                 const std::vector<float>& scores, const char* what) {
    if (tree.tokens != tokens || tree.parents != parents || tree.scores.size() != scores.size()) {
        std::printf("FAIL %s: tree shape\n", what);
        ++failures;
        return;
    }
    for (size_t i = 0; i < scores.size(); ++i)
        if (std::fabs(tree.scores[i] - scores[i]) > 1e-5f) {
            std::printf("FAIL %s: score %zu got %.6f want %.6f\n", what, i, tree.scores[i], scores[i]);
            ++failures;
            return;
        }
}

struct Fixture {
    int positions = 0;
    int topk = 0;
    int rank = 1;
    int vocab = 1;
    std::vector<uint32_t> tokens;   // [positions * topk]
    std::vector<float> unary;       // [positions * topk]
    std::vector<float> gate;        // [positions * rank]
    std::vector<float> successor;   // [positions * topk * rank]
    std::vector<float> predecessor; // [vocab * rank]
    int predecessor_calls = 0;

    Fixture(int positions, int topk, int rank, int vocab) : positions(positions), topk(topk), rank(rank), vocab(vocab) {
        tokens.resize(size_t(positions) * topk);
        unary.assign(size_t(positions) * topk, 0.0f);
        gate.assign(size_t(positions) * rank, 0.0f);
        successor.assign(size_t(positions) * topk * rank, 0.0f);
        predecessor.assign(size_t(vocab) * rank, 0.0f);
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = uint32_t(i);
    }
    void token_ids(std::vector<uint32_t> ids) { tokens = std::move(ids); }
    tqllm::DraftLattice lattice() {
        tqllm::DraftLattice l;
        l.positions = positions;
        l.topk = topk;
        l.rank = rank;
        l.tokens = tokens.data();
        l.unary = unary.data();
        l.gate = gate.data();
        l.successor = successor.data();
        l.predecessor = &Fixture::predecessor_row;
        l.context = this;
        return l;
    }
    static const float* predecessor_row(void* context, uint32_t token) {
        auto& fixture = *static_cast<Fixture*>(context);
        ++fixture.predecessor_calls;
        return fixture.predecessor.data() + size_t(token) * fixture.rank;
    }
};

// The chain keeps the target head's best candidate at every position: unary 2,0 then 1,0 then 3,0 with no edge.
void test_chain_follows_the_unary_argmax() {
    Fixture f(3, 2, 1, 2);
    f.unary = {2, 0, 1, 0, 3, 0};
    tqllm::DraftTreeOptions o;
    o.edge = 0;
    o.tau = 1;
    const auto tree = tqllm::chain_tree(f.lattice(), /*anchor=*/0, o);
    expect_tree(tree, {0, 2, 4}, {-1, 0, 1}, {-0.126928f, -0.440190f, -0.488777f}, "chain follows the unary argmax");
}

// Two positions, two kept candidates, no edge: the root offers both candidates, the better one expands, and the two
// dead-end siblings fill the budget in score order.
void test_best_first_builds_the_expected_tree() {
    Fixture f(2, 2, 1, 4);
    f.unary = {2, 0, 3, 0};
    f.tokens = {10, 11, 20, 21};
    tqllm::DraftTreeOptions o;
    o.nodes = 4;
    o.children = 2;
    o.edge = 0;
    o.tau = 1;
    const auto tree = tqllm::best_first_tree(f.lattice(), /*anchor=*/0, o);
    expect_tree(tree, {10, 20, 11, 20}, {-1, 0, -1, 2}, {-0.126928f, -0.175515f, -2.126928f, -2.175515f},
                "best_first builds the expected tree");
}

// With one child and one node per position the search is the chain, bit for bit.
void test_best_first_with_one_child_is_the_chain() {
    Fixture f(3, 3, 2, 6);
    f.unary = {0.5f, 1.5f, -0.5f, 2.0f, 0.1f, 0.3f, -1.0f, 0.0f, 1.0f};
    for (int p = 0; p < 3; ++p) {
        f.gate[size_t(p) * 2] = 1.0f;
        f.gate[size_t(p) * 2 + 1] = 0.5f;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 2; ++r) f.successor[(size_t(p) * 3 + c) * 2 + r] = 0.25f * float(p + c + r);
    }
    for (int t = 0; t < 6; ++t)
        for (int r = 0; r < 2; ++r) f.predecessor[size_t(t) * 2 + r] = 0.1f * float(t + r);
    tqllm::DraftTreeOptions o;
    o.edge = 0.7f;
    const auto chain = tqllm::chain_tree(f.lattice(), 1, o);
    o.children = 1;
    o.nodes = 3;
    const auto best_first = tqllm::best_first_tree(f.lattice(), 1, o);
    expect(chain.tokens == best_first.tokens && chain.parents == best_first.parents &&
               chain.scores.size() == best_first.scores.size(),
           "one child is the chain");
    bool scores_equal = chain.scores.size() == best_first.scores.size();
    for (size_t i = 0; scores_equal && i < chain.scores.size(); ++i)
        scores_equal = std::fabs(chain.scores[i] - best_first.scores[i]) <= 1e-6f;
    expect(scores_equal, "one child keeps the chain's scores");
}

// Equal scores: the lower token first, then its sibling, and the rest in the same order.
void test_ties_break_by_token() {
    Fixture f(1, 2, 1, 4);
    f.tokens = {7, 3};
    tqllm::DraftTreeOptions o;
    o.nodes = 2;
    o.children = 2;
    o.edge = 0;
    const auto tree = tqllm::best_first_tree(f.lattice(), 0, o);
    expect_tree(tree, {3, 7}, {-1, -1}, {-0.693147f, -0.693147f}, "ties break by token");
}

// unary_weight scales the unary logit against the edge: 0.44 turns a unary pick into an edge pick.
void test_unary_weight_and_edge_move_the_pick() {
    Fixture f(1, 2, 1, 4);
    f.unary = {10.0f, 0.0f};
    f.gate = {1.0f};
    f.successor = {0.0f, 1.0f};  // candidate 0 has no edge, candidate 1 has edge 1
    f.predecessor = {1.0f, 1.0f, 1.0f, 1.0f};
    tqllm::DraftTreeOptions o;
    o.nodes = 1;
    o.children = 2;
    o.edge = 1.2f;
    o.tau = 1;
    const auto anchored = tqllm::best_first_tree(f.lattice(), 0, o);
    expect_tree(anchored, {f.tokens[0]}, {-1}, {-0.000151f}, "the unary logit keeps the pick");
    o.unary_weight = 0.1f;
    const auto edge_first = tqllm::best_first_tree(f.lattice(), 0, o);
    expect_tree(edge_first, {f.tokens[1]}, {-1}, {-0.598139f}, "unary_weight 0.1 hands the pick to the edge");
}

// Structural invariants on a pseudo-random lattice: parent-closed rows, at most `children` siblings, depth within the
// lattice, at most `nodes` rows, deterministic, and the predecessor table is read once per expansion, not copied.
void test_structure_on_a_random_lattice() {
    Fixture f(10, 16, 8, 50);
    uint32_t state = 12345;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return float((state >> 8) % 2000) / 1000.0f - 1.0f;
    };
    for (auto& value : f.unary) value = 3.0f * next();
    for (auto& value : f.gate) value = next();
    for (auto& value : f.successor) value = next();
    for (auto& value : f.predecessor) value = next();
    for (auto& token : f.tokens) {
        state = state * 1664525u + 1013904223u;
        token = (state >> 8) % 50;
    }
    tqllm::DraftTreeOptions o;
    o.nodes = 15;
    o.children = 4;
    o.edge = 0.6f;
    o.tau = 1.5f;
    f.predecessor_calls = 0;
    const auto tree = tqllm::best_first_tree(f.lattice(), 3, o);
    const int expansions = f.predecessor_calls;
    expect(int(tree.tokens.size()) <= o.nodes, "random tree within the node budget");
    expect(!tree.tokens.empty(), "random tree is not empty");
    std::vector<int> depth(tree.tokens.size(), 0);
    std::vector<int> child_count(tree.tokens.size() + 1, 0);  // the last slot counts the root's children
    for (size_t row = 0; row < tree.tokens.size(); ++row) {
        const int parent = tree.parents[row];
        expect(parent >= -1 && parent < int(row), "random tree is parent-closed");
        if (parent < 0) {
            ++child_count[tree.tokens.size()];
            depth[row] = 0;
        } else {
            ++child_count[size_t(parent)];
            depth[row] = depth[size_t(parent)] + 1;
        }
        expect(depth[row] < f.positions, "random tree depth within the lattice");
    }
    for (int count : child_count) expect(count <= o.children, "random tree siblings within the budget");
    expect(expansions <= int(tree.tokens.size()) + 1, "one predecessor row per expansion");
    const auto again = tqllm::best_first_tree(f.lattice(), 3, o);
    expect(again.tokens == tree.tokens && again.parents == tree.parents, "random tree is deterministic");
}

// A budget past the frontier's end stops at the frontier, and the root anchor feeds every expansion.
void test_budget_past_the_frontier_stops() {
    Fixture f(2, 2, 1, 4);
    f.unary = {0.0f, 0.0f, 0.0f, 0.0f};
    tqllm::DraftTreeOptions o;
    o.nodes = 63;
    o.children = 2;
    o.edge = 0;
    const auto tree = tqllm::best_first_tree(f.lattice(), 0, o);
    expect(tree.tokens.size() == 6, "budget past the frontier stops at the reachable rows");
}

void test_invalid_arguments_are_rejected() {
    Fixture f(2, 2, 1, 4);
    tqllm::DraftTreeOptions o;
    auto thrown = [](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    expect(thrown([&] { auto l = f.lattice(); l.positions = 0; tqllm::best_first_tree(l, 0, o); }),
           "zero positions rejected");
    expect(thrown([&] { auto l = f.lattice(); l.topk = 17; tqllm::best_first_tree(l, 0, o); }), "topk past 16 rejected");
    expect(thrown([&] { auto l = f.lattice(); l.predecessor = nullptr; tqllm::best_first_tree(l, 0, o); }),
           "missing predecessor rejected");
    expect(thrown([&] { auto l = f.lattice(); tqllm::best_first_tree(l, 0, {.nodes = 0}); }), "zero nodes rejected");
    expect(thrown([&] { auto l = f.lattice(); tqllm::best_first_tree(l, 0, {.children = 0}); }),
           "zero children rejected");
    expect(thrown([&] { auto l = f.lattice(); tqllm::best_first_tree(l, 0, {.tau = 0.0f}); }), "zero tau rejected");
    expect(thrown([&] { auto l = f.lattice(); tqllm::best_first_tree(l, 0, {.children = 3}); }),
           "more children than candidates rejected");
}

void test_selector_registry() {
    expect(tqllm::draft_tree_selector("chain") == &tqllm::chain_tree, "chain is registered");
    expect(tqllm::draft_tree_selector("best_first") == &tqllm::best_first_tree, "best_first is registered");
    expect(tqllm::draft_tree_selector("nope") == nullptr, "unknown selector is null");
}
}  // namespace

void test_dspine_dense_lattice() {
    const uint32_t tokens[]={10,11,20,21};
    const float scores[]={5,-5,5,-5,0,8,7,0};
    tqllm::DraftLattice l;l.positions=2;l.topk=2;l.tokens=tokens;l.dense_scores=scores;
    tqllm::DraftTreeOptions o;o.nodes=3;o.children=2;o.edge=0;o.unary_weight=0;
    const auto chain=tqllm::chain_tree(l,99,o);
    expect(chain.tokens==std::vector<uint32_t>{10,21},"DSpine chain follows conditional predecessor scores");
    const auto tree=tqllm::best_first_tree(l,99,o);
    expect(tree.tokens==std::vector<uint32_t>{10,21,20},"DSpine tree ranks complete conditional logits");
    expect(tree.parents==std::vector<int32_t>{-1,0,0},"DSpine tree preserves parent links");
}

int main() {
    test_dspine_dense_lattice();
    test_chain_follows_the_unary_argmax();
    test_best_first_builds_the_expected_tree();
    test_best_first_with_one_child_is_the_chain();
    test_ties_break_by_token();
    test_unary_weight_and_edge_move_the_pick();
    test_structure_on_a_random_lattice();
    test_budget_past_the_frontier_stops();
    test_invalid_arguments_are_rejected();
    test_selector_registry();
    if (failures) {
        std::printf("draft_tree: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("draft_tree: ok");
    return 0;
}