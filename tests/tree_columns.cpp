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
    // the old sizing (compile-time 16 columns) for the cases that failed: lookup 8 on a 7-draft tree, draft-max 10
    const int old_lookup = 1 + std::min(7, 16 - 1 - 8) + 8, old_dmax = 1 + std::min(10, 16 - 1);
    std::printf("old sizing: lookup 8 + 7 drafts = %d columns, draft-max 10 = %d columns (limit %d)\n", old_lookup, old_dmax, cap);
    std::printf("new sizing: widest verify %d columns over all options\n", worst);
    if (failures || old_lookup <= cap || old_dmax <= cap) { std::printf("tree_columns: %d failure(s)\n", failures); return 1; }
    std::puts("tree_columns: ok");
    return 0;
}
