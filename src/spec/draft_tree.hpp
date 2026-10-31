#pragma once
#include <cstdint>
#include <string_view>
#include <vector>

namespace tqllm {
// One draft's selector lattice: the target head's top-k candidates per position with their unary logits, and the
// selector's edge factors. A tree selector turns the lattice into the token tree the target verifies in one pass.
struct DraftLattice {
    int positions = 0;                 // drafted tokens asked for (at most the drafter's block minus one)
    int topk = 0;                      // candidate tokens per position (16 in DFlash2)
    int rank = 0;                      // selector rank (256 in DFlash2)
    const uint32_t* tokens = nullptr;  // [positions * topk] candidate token ids
    const float* unary = nullptr;      // [positions * topk] their logits
    const float* gate = nullptr;       // [positions * rank] the selector's hidden projection row
    const float* successor = nullptr;  // [positions * topk * rank] the codebook row of each candidate
    // The predecessor codebook row of a token (rank values). The drafter reads it on demand, so the whole table (127 MB
    // bf16 for DFlash2) never comes to the host.
    const float* (*predecessor)(void* context, uint32_t token) = nullptr;
    void* context = nullptr;
};

// A draft tree: row i is the token tokens[i] whose parent row is parents[i] (-1 for the anchor root row, which is not
// listed). Every parent is an earlier row, so the whole tree verifies in one target pass under a parent mask.
struct DraftTree {
    std::vector<uint32_t> tokens;
    std::vector<int32_t> parents;
    std::vector<float> scores;  // cumulative selector score of each row (log-softmax per position, added down the path)
};

struct DraftTreeOptions {
    int nodes = 15;      // tree rows, the root anchor excluded
    int children = 4;    // candidates kept per expansion (1 = one chain)
    float tau = 1.5f;    // temperature of a candidate's score, so deeper rows' cumulative scores stay comparable
    float edge = 0.6f;   // weight of the selector's predecessor/successor edge
    float unary_weight = 1.0f;  // weight of the unary logit (1: the TensorFold DFlash2 scoring)
};

// A selector plugs in by name: "chain" is the drafter's single path, "best_first" is the TensorFold DFlash2 tree search.
using DraftTreeSelector = DraftTree (*)(const DraftLattice&, uint32_t anchor, const DraftTreeOptions&);
DraftTreeSelector draft_tree_selector(std::string_view name);  // nullptr when the name is unknown

DraftTree chain_tree(const DraftLattice&, uint32_t anchor, const DraftTreeOptions&);
DraftTree best_first_tree(const DraftLattice&, uint32_t anchor, const DraftTreeOptions&);
}  // namespace tqllm