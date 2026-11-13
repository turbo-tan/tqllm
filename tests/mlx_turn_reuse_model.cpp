// Turn reuse on the Metal engine with a model (run by hand): a three-turn chat whose client re-renders each answer
// without its reasoning. Turns 2 and 3 resume at the state the turn before kept (turn 3 at a state kept inside a resumed
// prefill); their text, steps, drafted and accepted tokens must equal those of a prefill of the same prompt from 0.
// usage: test-mlx-turn-reuse-model model.gguf drafter.gguf [history tokens, default 3000] [qsa-from keys]
#include "engine_mlx.hpp"
#include "tqllm/options.hpp"
#include "tqllm/tokenizer.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
struct Run {std::vector<uint32_t> out; std::array<uint64_t, 5> stats{};};
Run run(tqllm::Engine& e, const std::vector<uint32_t>& prompt, uint32_t count) {
    Run r;
    e.spec_begin(prompt, count, 0);
    while (e.spec_advance()) {}
    r.out = e.spec_finish(r.stats.data());
    return r;
}
int failures = 0;
void same(const char* what, const Run& resumed, const Run& cold) {
    const bool text = resumed.out == cold.out;
    const bool counts = resumed.stats[0] == cold.stats[0] && resumed.stats[1] == cold.stats[1] && resumed.stats[2] == cold.stats[2];
    const bool faster = resumed.stats[4] * 2 < cold.stats[4];
    std::printf("%s: text %s, steps %llu / %llu drafted %llu / %llu accepted %llu / %llu, prefill %.2f s resumed, %.2f s from 0\n", what,
                text ? "equal" : "DIFFERS", (unsigned long long) resumed.stats[0], (unsigned long long) cold.stats[0],
                (unsigned long long) resumed.stats[1], (unsigned long long) cold.stats[1], (unsigned long long) resumed.stats[2],
                (unsigned long long) cold.stats[2], resumed.stats[4] / 1e6, cold.stats[4] / 1e6);
    if (!text || !counts || !faster) {std::fprintf(stderr, "FAIL %s%s%s%s\n", what, text ? "" : " text", counts ? "" : " counts", faster ? "" : " not resumed"); ++failures;}
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {std::fprintf(stderr, "usage: test-mlx-turn-reuse-model model.gguf drafter.gguf [history tokens]\n"); return 2;}
    const int history = argc > 3 ? std::atoi(argv[3]) : 3000;
    try {
        tqllm::Tokenizer tok; tok.load(argv[1]);
        std::string doc;
        for (int n = 1; int(tok.encode(doc).size()) < history; n += 40)
            for (int k = n; k < n + 40; ++k)
                doc += "Record " + std::to_string(k) + ": station " + std::to_string(k * 7 % 113) + " logged " + std::to_string(k * 13 % 97) +
                       " mm of rain and a wind of " + std::to_string(k * 5 % 31) + " knots.\n";
        const std::string gen = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        auto user = [](const std::string& text) {return "<|im_start|>user\n" + text + "<|im_end|>\n";};
        auto answer = [&](const Run& r) {
            auto ids = r.out;
            while (!ids.empty() && tok.is_eog(ids.back())) ids.pop_back();
            return "<|im_start|>assistant\n" + tok.decode(ids) + "<|im_end|>\n";  // as a client sends it back: no reasoning
        };
        const uint32_t count = 64;
        tqllm::EngineOptions options; options.draft_model = argv[2]; options.prompt_cache = true;  // turn reuse is --prompt-cache on
        if (argc > 4) options.qsa_from = std::atoi(argv[4]);  // QSA sparse attention from this many keys (below the history: the resumed turns attend sparsely)
        auto engine = tqllm::make_mlx_qwen35_engine(argv[1], "", uint32_t(history) * 2 + 4096, 1, options);
        const std::string h1 = user(doc + "\nWhich station logged the most rain? Answer in two sentences.");
        const auto p1 = tok.encode(h1 + gen);
        const Run a1 = run(*engine, p1, count);
        const std::string h2 = h1 + answer(a1) + user("And which record has the strongest wind? Two sentences.");
        const auto p2 = tok.encode(h2 + gen);
        const Run r2 = run(*engine, p2, count);
        const std::string h3 = h2 + answer(r2) + user("Summarise both answers in one sentence.");
        const auto p3 = tok.encode(h3 + gen);
        const Run r3 = run(*engine, p3, count);
        const Run r3again = run(*engine, p3, count);  // the same prompt once more resumes at the same state
        const auto other = tok.encode(user("Say hello.") + gen);
        run(*engine, other, 4);  // an unrelated request: nothing of the chat stays
        const Run c2 = run(*engine, p2, count);
        run(*engine, other, 4);
        const Run c3 = run(*engine, p3, count);
        std::printf("prompts %zu / %zu / %zu tokens\n", p1.size(), p2.size(), p3.size());
        same("turn 2", r2, c2);
        same("turn 3", r3, c3);
        same("turn 3 again", r3again, c3);
        // the answer prefilled while idle (Engine::prefill_ahead with the history as the client returns it): turn 2 then
        // prefills only its new user message; turn 3 follows a hit; another turn 2 (the client changed the answer) misses
        // and falls back to the state kept at turn 1's generation prompt. All equal prefills from 0.
        run(*engine, other, 4);
        const Run a1b = run(*engine, p1, count);
        if (a1b.out != a1.out) {std::fprintf(stderr, "FAIL turn 1 again differs\n"); ++failures;}
        if (!engine->prefill_ahead(0, tok.encode(h1 + answer(a1)))) {std::fprintf(stderr, "FAIL prefill_ahead refused turn 1's answer\n"); ++failures;}
        const Run h2r = run(*engine, p2, count);
        if (!engine->prefill_ahead(0, tok.encode(h2 + answer(r2)))) {std::fprintf(stderr, "FAIL prefill_ahead refused turn 2's answer\n"); ++failures;}
        const Run h3r = run(*engine, p3, count);
        same("turn 2 after its answer was prefilled", h2r, c2);
        same("turn 3 after a prefilled turn 2", h3r, c3);
        if (h2r.stats[4] >= r2.stats[4]) std::printf("NOTE turn 2: prefill %.2f s with the answer prefilled, %.2f s without\n", h2r.stats[4] / 1e6, r2.stats[4] / 1e6);
        const std::string h2x = h1 + "<|im_start|>assistant\nStation 7, I think.<|im_end|>\n" + user("And which record has the strongest wind? Two sentences.");
        const auto p2x = tok.encode(h2x + gen);
        run(*engine, other, 4);
        const Run cx = run(*engine, p2x, count);
        run(*engine, other, 4);
        run(*engine, p1, count);
        engine->prefill_ahead(0, tok.encode(h1 + answer(a1)));
        const Run mx_ = run(*engine, p2x, count);
        same("turn 2 with another answer (a miss)", mx_, cx);
        std::printf("prefill s: turn 2 resumed %.2f, with its answer prefilled %.2f; turn 3 resumed %.2f, prefilled %.2f\n",
                    r2.stats[4] / 1e6, h2r.stats[4] / 1e6, r3.stats[4] / 1e6, h3r.stats[4] / 1e6);
    } catch (const std::exception& e) {std::fprintf(stderr, "FAIL %s\n", e.what()); return 1;}
    if (failures) return 1;
    std::printf("PASS turn reuse: resumed turns equal prefills from 0\n");
    return 0;
}
