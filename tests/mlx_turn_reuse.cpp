// Turn reuse rules (turn_reuse.hpp): a prefill's chunks are a property of the prompt's prefix, so a prefill resumed at a
// state the previous turn kept runs exactly the chunks a cold prefill of the same prompt runs from there on.
#include "turn_reuse.hpp"
#include <cstdio>
#include <vector>

using namespace tqllm::mlx_backend;
namespace {
int failures = 0;
void expect(bool ok, const char* what) {if (!ok) {std::fprintf(stderr, "FAIL %s\n", what); ++failures;}}
constexpr uint32_t IM = 7, AS = 8, US = 9;
// messages: (assistant?, body tokens); every message is IM, role, body
std::vector<uint32_t> chat(const std::vector<std::pair<bool, int>>& messages) {
    std::vector<uint32_t> p;
    for (const auto& [assistant, body] : messages) {
        p.push_back(IM); p.push_back(assistant ? AS : US);
        for (int i = 0; i < body; ++i) p.push_back(100 + uint32_t(p.size() % 50));
    }
    return p;
}
std::vector<uint32_t> chunks(const std::vector<uint32_t>& prompt, uint32_t from, uint32_t chunk) {
    const auto starts = message_starts(prompt, IM);
    std::vector<uint32_t> ends;
    for (uint32_t t = from; t < prompt.size();) {t = turn_chunk_end(starts, t, uint32_t(prompt.size()), chunk); ends.push_back(t);}
    return ends;
}
}  // namespace

int main() {
    // short prompts: one pass, nothing kept (the first token of a short chat pays no extra forward)
    {
        const auto p = chat({{false, 300}, {true, 1}});
        expect(message_starts(p, IM).empty(), "no start below kTurnMin");
        expect(chunks(p, 0, 8192) == std::vector<uint32_t>{uint32_t(p.size())}, "short prompt: one chunk");
        expect(turn_kept_at(message_starts(p, IM), 0) == 0, "short prompt keeps nothing");
    }
    // without the template's tokens the grid is the plain one
    {
        const auto p = chat({{false, 20000}, {true, 1}});
        expect(message_starts(p, ~0u).empty(), "no template token: no starts");
        const auto starts = std::vector<uint32_t>{};
        expect(turn_chunk_end(starts, 0, 20005, 8192) == 8192 && turn_chunk_end(starts, 16384, 20005, 8192) == 20005, "plain grid");
    }
    for (uint32_t chunk : {8192u, 2048u, 128u}) {
        // turn 1: a long user message, then the generation prompt; turn 2 re-renders the answer and adds a user message;
        // turn 3 again. The bodies differ after each generation prompt (the reasoning is gone), as a client sends them
        const auto t1 = chat({{false, 23000}, {true, 1}});
        const auto t2 = chat({{false, 23000}, {true, 700}, {false, 60}, {true, 1}});
        const auto t3 = chat({{false, 23000}, {true, 700}, {false, 60}, {true, 9000}, {false, 5}, {true, 1}});
        const uint32_t k1 = turn_kept_at(message_starts(t1, IM), 0);
        expect(k1 == 23002, "turn 1 keeps its state at the generation prompt");
        expect(chunks(t1, 0, chunk).back() == t1.size() && chunks(t1, 0, chunk)[chunks(t1, 0, chunk).size() - 2] == k1, "turn 1 has a chunk boundary there");
        const uint32_t r2 = turn_shared(std::vector<uint32_t>(t1.begin(), t1.begin() + k1 + 2), t2) ? k1 : 0;
        expect(r2 == k1, "turn 2 shares turn 1's prompt up to the kept state");
        // the resumed prefill's chunks are the cold prefill's chunks from there on
        auto cold = chunks(t2, 0, chunk);
        const auto resumed = chunks(t2, r2, chunk);
        const auto at = std::find(cold.begin(), cold.end(), r2);
        expect(at != cold.end() && std::vector<uint32_t>(at + 1, cold.end()) == resumed, "turn 2: resumed chunks == cold chunks past the kept state");
        const uint32_t k2 = turn_kept_at(message_starts(t2, IM), r2);
        expect(k2 == uint32_t(t2.size()) - 3 && k2 > k1, "turn 2 keeps its own generation prompt");
        const uint32_t r3 = turn_shared(std::vector<uint32_t>(t2.begin(), t2.begin() + k2 + 2), t3) ? k2 : 0;
        cold = chunks(t3, 0, chunk);
        const auto resumed3 = chunks(t3, r3, chunk);
        const auto at3 = std::find(cold.begin(), cold.end(), r3);
        expect(r3 == k2 && at3 != cold.end() && std::vector<uint32_t>(at3 + 1, cold.end()) == resumed3, "turn 3: resumed chunks == cold chunks");
        // the cold chunks of a prompt's prefix do not depend on what follows it
        const auto c2 = chunks(t2, 0, chunk), c3 = chunks(t3, 0, chunk);
        size_t same = 0;
        while (same < c2.size() && c2[same] <= k2 && c3[same] == c2[same]) ++same;
        expect(same > 0 && c2[same - 1] == k2, "prefix chunks are the same in the longer prompt");
        // the same prompt again (regenerate): resumes, keeps nothing new
        expect(turn_shared(std::vector<uint32_t>(t2.begin(), t2.begin() + k2 + 2), t2) && turn_kept_at(message_starts(t2, IM), k2) == 0, "regenerate");
        // an edited earlier message, or a prompt that ends inside the kept part: cold
        auto edited = t2; edited[500] += 1;
        expect(!turn_shared(std::vector<uint32_t>(t1.begin(), t1.begin() + k1 + 2), edited), "edited history is not shared");
        expect(!turn_shared(std::vector<uint32_t>(t1.begin(), t1.begin() + k1 + 2), std::vector<uint32_t>(t1.begin(), t1.begin() + k1 + 1)), "shorter prompt is not shared");
        auto other_role = t2; other_role[k1 + 1] = US;
        expect(!turn_shared(std::vector<uint32_t>(t1.begin(), t1.begin() + k1 + 2), other_role), "the role token at the kept state must match");
        // the answer prefilled ahead: turn 1's prompt + the re-rendered answer message ends where turn 2's user message
        // starts; a prefill of it from turn 1's kept state runs the chunks turn 2's cold prefill runs there, and turn 2
        // resumed at its end runs the rest
        const uint32_t e = k1 + 2 + 700;
        expect(t2[e] == IM && t2[e + 1] == US, "the answer's message ends at the next user message");
        std::vector<uint32_t> ahead(t2.begin(), t2.begin() + e);
        cold = chunks(t2, 0, chunk);
        const auto a0 = std::find(cold.begin(), cold.end(), k1), a1 = std::find(cold.begin(), cold.end(), e);
        expect(a0 != cold.end() && a1 != cold.end() && std::vector<uint32_t>(a0 + 1, a1 + 1) == chunks(ahead, k1, chunk), "ahead chunks == cold chunks of the answer");
        ahead.push_back(IM);
        expect(turn_shared(ahead, t2) && std::vector<uint32_t>(a1 + 1, cold.end()) == chunks(t2, e, chunk), "turn 2 resumed after the answer == cold chunks");
        auto other_answer = t2; other_answer[k1 + 50] += 1;
        expect(!turn_shared(ahead, other_answer) && turn_shared(std::vector<uint32_t>(t1.begin(), t1.begin() + k1 + 2), other_answer),
               "another answer text misses the prefilled answer and still shares turn 1's kept state");
    }
    if (failures) return 1;
    std::printf("PASS turn reuse rules\n");
    return 0;
}
