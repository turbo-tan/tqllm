// DraftRankStats: a rejection at the last position of a width-15 draft is counted in its own slot (the inline
// histogram in tqmlx-dflash had 8 slots and wrote past them once a step accepted 8 or more tokens), ranks are the first
// matching candidate, a token outside the candidates counts as rank 16, and positions outside the width are refused.
#include "dflash_rank_stats.hpp"
#include <cstdio>
#include <sstream>
#include <string>
#include <stdexcept>

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {if (!ok) {std::printf("FAIL %s\n", what); ++failures;}};
    const int width = 15;
    std::vector<std::vector<uint32_t>> candidates(width);
    for (int p = 0; p < width; ++p) for (uint32_t j = 0; j < 16; ++j) candidates[size_t(p)].push_back(1000u * uint32_t(p) + j);
    tqllm::DraftRankStats stats(width);
    check(stats.rejection_at.size() == size_t(width), "one rejection slot per draft position");
    stats.record(candidates, 14, 14000u + 2, 14000u + 0);   // rejected at the last position: target rank 2, chosen rank 0
    stats.record(candidates, 8, 999999u, 8000u + 5);        // target not among the candidates
    stats.record(candidates, 0, 0u + 1, 0u + 1);
    check(stats.rejection_at[14] == 1 && stats.rejection_at[8] == 1 && stats.rejection_at[0] == 1, "positions 0 / 8 / 14 counted");
    check(stats.target_rank[2] == 1 && stats.target_rank[16] == 1 && stats.target_rank[1] == 1, "target ranks 2 / not found / 1");
    check(stats.chosen_rank[0] == 1 && stats.chosen_rank[5] == 1 && stats.chosen_rank[1] == 1, "chosen ranks 0 / 5 / 1");
    check(stats.rejections == 3, "three rejections");
    bool refused = false;
    try {stats.record(candidates, width, 0, 0);} catch (const std::out_of_range&) {refused = true;}
    check(refused, "position == width refused");
    std::ostringstream json;
    stats.write_json(json);
    const std::string text = json.str();
    const auto at = text.find("\"rejection_position\":[");
    size_t commas = 0;
    for (size_t i = at; i < text.size() && text[i] != ']'; ++i) commas += text[i] == ',';
    check(at != std::string::npos && commas + 1 == size_t(width), "json lists every position");
    std::puts(failures ? "DFLASH_RANK_STATS_FAIL" : "DFLASH_RANK_STATS_PASS");
    return failures ? 1 : 0;
}
