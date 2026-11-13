// The Metal engine serves requests from different threads (tqserve without the scheduler: one connection thread per
// request). A request that extends the previous request's whole text resumes on its kept state (--exact off: the live
// decode state; --exact on --prompt-cache on: the kept prefill states; --exact off --prompt-cache on: both). Each request here runs on its own thread; the
// later requests must neither fail (MLX 0.32: "There is no Stream(gpu, N) in current thread", j2196) nor leave the
// engine unusable for the next. Run by hand: test-mlx-cross-thread-model model.gguf drafter.gguf
#include "engine_mlx.hpp"
#include "tqllm/options.hpp"
#include "tqllm/tokenizer.hpp"
#include <array>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {
struct Run {std::vector<uint32_t> out; std::string error;};
Run on_thread(tqllm::Engine& e, const std::vector<uint32_t>& prompt, uint32_t count) {
    Run r;
    std::thread t([&] {
        try {
            e.spec_begin(prompt, count, 0);
            while (e.spec_advance()) {}
            std::array<uint64_t, 5> st{};
            r.out = e.spec_finish(st.data());
        } catch (const std::exception& x) {r.error = x.what();}
    });
    t.join();
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {std::fprintf(stderr, "usage: test-mlx-cross-thread-model model.gguf drafter.gguf\n"); return 2;}
    int failures = 0;
    try {
        tqllm::Tokenizer tok; tok.load(argv[1]);
        std::string doc;
        for (int k = 1; k < 120; ++k) doc += "Record " + std::to_string(k) + ": station " + std::to_string(k * 7 % 113) + " logged " + std::to_string(k * 13 % 97) + " mm of rain.\n";
        for (const std::string mode : {"exact off", "exact on, prompt cache on", "exact off, prompt cache on"}) {
            tqllm::EngineOptions options; options.draft_model = argv[2];
            options.exact = mode.rfind("exact on", 0) == 0;
            options.prompt_cache = mode.find("prompt cache on") != std::string::npos;
            auto engine = tqllm::make_mlx_qwen35_engine(argv[1], "", 16384, 1, options);
            auto prompt = tok.encode("<|im_start|>user\n" + doc + "Which station logged the most rain?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            for (int turn = 1; turn <= 4; ++turn) {
                const Run r = on_thread(*engine, prompt, 48);
                std::printf("%s, turn %d on its own thread: prompt %zu, %s\n", mode.c_str(), turn, prompt.size(),
                            r.error.empty() ? ("ok, " + std::to_string(r.out.size()) + " tokens").c_str() : ("FAILED: " + r.error).c_str());
                if (!r.error.empty()) {++failures; continue;}
                // the next prompt extends this one's whole text (as a client that returns the answer as generated)
                auto out = r.out;
                while (!out.empty() && tok.is_eog(out.back())) out.pop_back();
                prompt.insert(prompt.end(), out.begin(), out.end());
                const auto more = tok.encode("<|im_end|>\n<|im_start|>user\nAnd the least?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
                prompt.insert(prompt.end(), more.begin(), more.end());
            }
        }
    } catch (const std::exception& e) {std::fprintf(stderr, "FAIL %s\n", e.what()); return 1;}
    std::printf(failures ? "FAIL cross thread: %d requests failed\n" : "PASS cross thread\n", failures);
    return failures ? 1 : 0;
}
