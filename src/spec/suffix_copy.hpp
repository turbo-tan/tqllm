#pragma once
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tqllm {
// Copy drafts from the context: when the last tokens of the committed text repeat an earlier stretch of it, the tokens
// that followed that earlier stretch are proposed as the draft. Long repeats (documents quoted back, code edited in place,
// tool output read twice) verify many tokens per pass where a neural drafter tops out at its block.
//
// Only committed tokens are indexed (append), never drafts, so a rejected draft cannot be proposed back. propose() reads
// the index and never changes it. Host only; a lookup is O(candidates * match cap), both bounded by the options.
struct SuffixCopyOptions {
    int key = 4;             // tokens hashed per index entry (a match shorter than this is never found)
    int min_match = 20;      // matching tokens a copy needs (20 beside a neural drafter, ~5 without one)
    int max_match = 64;      // the match length is counted up to this (enough to rank candidates)
    int max_candidates = 64; // most recent earlier occurrences of the key examined per lookup
};

class SuffixCopy {
public:
    explicit SuffixCopy(const SuffixCopyOptions& options = {});

    void append(uint32_t token);                    // one committed token
    void append(const uint32_t* tokens, size_t n);  // several committed tokens, in order
    void reset();                                    // a new sequence

    // The copy draft for the current context: up to `max_tokens` tokens that followed the longest earlier repeat of the
    // context's suffix (the most recent one among equally long repeats). Empty when no repeat reaches min_match.
    std::vector<uint32_t> propose(int max_tokens) const;
    int last_match() const { return last_match_; }  // the matched length of the last propose() (0: none)
    size_t size() const { return context_.size(); }

private:
    uint64_t key_hash(size_t end) const;  // hash of context_[end - key, end)

    SuffixCopyOptions options_;
    std::vector<uint32_t> context_;
    std::unordered_map<uint64_t, std::vector<uint32_t>> index_;  // key hash -> end positions, oldest first
    mutable int last_match_ = 0;
};
}  // namespace tqllm
