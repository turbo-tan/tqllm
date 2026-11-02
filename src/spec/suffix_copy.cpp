#include "suffix_copy.hpp"

#include <stdexcept>

namespace tqllm {
SuffixCopy::SuffixCopy(const SuffixCopyOptions& options) : options_(options) {
    if (options_.key < 1 || options_.min_match < options_.key || options_.max_match < options_.min_match ||
        options_.max_candidates < 1)
        throw std::invalid_argument("invalid suffix copy options");
}

uint64_t SuffixCopy::key_hash(size_t end) const {
    uint64_t h = 1469598103934665603ull;  // FNV-1a over the key's token ids
    for (size_t i = end - size_t(options_.key); i < end; ++i) {
        uint32_t t = context_[i];
        for (int b = 0; b < 4; ++b, t >>= 8) h = (h ^ (t & 0xffu)) * 1099511628211ull;
    }
    return h;
}

void SuffixCopy::append(uint32_t token) {
    context_.push_back(token);
    if (context_.size() >= size_t(options_.key)) index_[key_hash(context_.size())].push_back(uint32_t(context_.size()));
}

void SuffixCopy::append(const uint32_t* tokens, size_t n) {
    for (size_t i = 0; i < n; ++i) append(tokens[i]);
}

void SuffixCopy::reset() {
    context_.clear();
    index_.clear();
    last_match_ = 0;
}

std::vector<uint32_t> SuffixCopy::propose(int max_tokens) const {
    last_match_ = 0;
    const size_t n = context_.size();
    if (max_tokens < 1 || n < size_t(options_.min_match) + 1) return {};
    const auto it = index_.find(key_hash(n));
    if (it == index_.end()) return {};
    const auto& ends = it->second;
    size_t best_end = 0;
    int best = 0;
    int examined = 0;
    // newest first, so an equally long older repeat never replaces a newer one
    for (size_t k = ends.size(); k-- > 0 && examined < options_.max_candidates;) {
        const size_t end = ends[k];
        if (end >= n) continue;  // the suffix itself
        ++examined;
        int match = 0;
        while (match < options_.max_match && size_t(match) < end && context_[end - 1 - match] == context_[n - 1 - match])
            ++match;
        if (match > best) best = match, best_end = end;
    }
    if (best < options_.min_match) return {};
    last_match_ = best;
    // the tokens after the repeat; a repeat that runs into the suffix continues with its own period (n - best_end)
    std::vector<uint32_t> draft;
    draft.reserve(size_t(max_tokens));
    const size_t period = n - best_end;
    for (int j = 0; j < max_tokens; ++j) draft.push_back(context_[best_end + size_t(j) % period]);
    return draft;
}
}  // namespace tqllm
