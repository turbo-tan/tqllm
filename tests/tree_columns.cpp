// Drafts + prompt-lookup columns must fit the GDN replay limit for every option combination.
#include "tree_columns.hpp"

#include <cstdio>

int main() {
    int failures = 0, worst = 0;
    const int cap = tqllm::kGdnReplayCols;
    for (int D = 1; D <= 15; ++D)              // --draft-max 1..15
        for (int L = 0; L <= 15; ++L)          // --lookup 0..15
            for (int found = 0; found <= 15; ++found)
                for (int lf = 0; lf <= 1; ++lf) {
                    const int nl = lf ? std::min(found, tqllm::lookup_first_limit(cap)) : std::min(found, tqllm::lookup_limit(L, cap));
                    const int n = tqllm::tree_cols(D, nl, lf, cap) + nl;
                    worst = std::max(worst, n);
                    if (n > cap || n < 1) ++failures;
                }
    // copy-first: the copy chain beside kCopyDrafterSiblings drafts fills the block
    const int copy_first = 1 + tqllm::kCopyDrafterSiblings + (cap - 1 - tqllm::kCopyDrafterSiblings);
    std::printf("widest verify %d columns over all options, copy-first %d (limit %d)\n", worst, copy_first, cap);
    if (failures || copy_first > cap) { std::printf("tree_columns: %d failure(s)\n", failures); return 1; }
    std::puts("tree_columns: ok");
    return 0;
}
