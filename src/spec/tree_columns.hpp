#pragma once
#include <algorithm>

namespace tqllm {
// Verify columns of one token-tree step: nt tree columns (root + drafts) and nl prompt-lookup / copy columns. The hybrid
// models' GDN layers replay at most kGdnReplayCols columns per verify, so drafts and copies share that budget.
inline constexpr int kGdnReplayCols = 16;
// longest lookup chain to ask for: leaves the root and at least one draft
inline int lookup_limit(int requested, int cap) { return std::max(0, std::min(requested, cap - 2)); }
// lookup-first takes the whole block except the root
inline int lookup_first_limit(int cap) { return cap - 1; }
// tree columns (root included) for D drafts next to nl lookup columns (lf: lookup-first took the block)
inline int tree_cols(int D, int nl, bool lf, int cap) { return lf ? 1 : 1 + std::max(0, std::min(D, cap - 1 - nl)); }
// Copy drafts beside the DFlash2 tree (no --lookup): a repeat of the context (prompt + output) at least kCopyFirstMatch
// tokens long takes the verify block, the drafter keeps kCopyDrafterSiblings drafts beside it. 27B, YASAKANI's 15 tasks,
// greedy, offline on accept traces (11-13), tokens per pass at 16 rows: today / copy-first at 16 / 24 / 32 tokens
// code edit 5.15 / 15.4 / 15.3 / 15.3, JSONL edit 6.76 / 15.9 / 15.8 / 15.8, structured 7.04 / 8.4 / 9.5 / 9.8, long
// JSONL 6.84 / 8.2 / 8.4 / 8.5; prose and code unchanged. A 4-token threshold (--lookup-first 4) took passes the drafter
// would have won (JSONL edit 277 -> 207 tok/s on a 3090).
inline constexpr int kCopyFirstMatch = 32;
inline constexpr int kCopyDrafterSiblings = 1;
// A copy tail under the drafter's chain (rows 9-16 below kCopyFirstMatch) measured a loss on the 3090: its host sync and
// rows past 8 (~1.2 ms each: 24 ms for 8, 34 for 16) cost more than it added (long JSONL 282 -> 252 tok/s at 6.84 -> 6.94
// tokens per pass, structured 298 -> 291 at 7.04 -> 7.77).
}  // namespace tqllm
