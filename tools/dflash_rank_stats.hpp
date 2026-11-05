#pragma once
// Drafter rank statistics at the first rejection of each speculative step: where the target's token and the chosen
// draft token sat among the drafter's 16 candidates, and at which draft position the step was rejected. Sized from the
// draft width (a rejection happens at positions 0 .. width - 1).
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace tqllm {
struct DraftRankStats {
    static constexpr int kCandidates = 16;
    std::vector<long> target_rank, chosen_rank, rejection_at;  // rank kCandidates = not among the candidates
    long rejections = 0;
    explicit DraftRankStats(int width) : target_rank(kCandidates + 1), chosen_rank(kCandidates + 1), rejection_at(size_t(width)) {
        if (width < 1) throw std::invalid_argument("draft width must be positive");
    }
    // candidates[p] holds the drafter's candidate ids at draft position p; the step accepted `accepted` tokens
    void record(const std::vector<std::vector<uint32_t>>& candidates, int accepted, uint32_t target, uint32_t proposed) {
        if (accepted < 0 || size_t(accepted) >= rejection_at.size())
            throw std::out_of_range("rejection position outside the draft width");
        int rank = kCandidates, chosen = kCandidates;
        if (size_t(accepted) < candidates.size()) {
            const auto& c = candidates[size_t(accepted)];
            for (int j = 0; j < kCandidates && j < int(c.size()); ++j) {
                if (rank == kCandidates && c[size_t(j)] == target) rank = j;
                if (chosen == kCandidates && c[size_t(j)] == proposed) chosen = j;
            }
        }
        ++target_rank[size_t(rank)]; ++chosen_rank[size_t(chosen)]; ++rejection_at[size_t(accepted)]; ++rejections;
    }
    void write_json(std::ostream& out) const {
        auto list = [&](const std::vector<long>& v) {for (size_t j = 0; j < v.size(); ++j) out << (j ? "," : "") << v[j];};
        out << "{\"kind\":\"dflash_rank_stats\",\"rejections\":" << rejections << ",\"target_rank\":[";
        list(target_rank); out << "],\"chosen_rank\":["; list(chosen_rank); out << "],\"rejection_position\":["; list(rejection_at);
        out << "]}";
    }
};
}  // namespace tqllm
