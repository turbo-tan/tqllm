// tqverify: exactness + cost of the multi-token verify step against sequential greedy decode.
// Usage: tqverify --model-dir DIR --weights FILE.gguf --text PROMPT [--n 5] [--chain 40] [--context 4096]
#include "tqllm/options.hpp"
#include "tqllm/model.hpp"
#include "tqllm/tokenizer.hpp"

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqverify");
    try {
        std::string model_dir, weights, text;
        int n = 5, chain = 40, keep = 0, iters = 40;  // keep: tokens committed per verify (0 = all n)
        bool garbage = false;             // replace uncommitted columns by junk tokens (tests column independence)
        uint32_t context = 4096;
        uint32_t depth = 0;
        tqllm::EngineOptions engine;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--model-dir") model_dir = value();
            else if (arg == "--weights") weights = value();
            else if (arg == "--text") text = value();
            else if (arg == "--n") n = std::stoi(value());
            else if (arg == "--chain") chain = std::stoi(value());
            else if (arg == "--commit") keep = std::stoi(value());
            else if (arg == "--garbage") garbage = true;
            else if (arg == "--context") context = std::stoul(value());
            else if (arg == "--depth") depth = std::stoul(value());
            else if (arg == "--iters") iters = std::stoi(value());  // timed verify calls (default 40)
            else if (tqllm::parse_engine_option(argc, argv, i, engine)) continue;
            else throw std::invalid_argument("unknown argument " + arg);
        }
        const std::filesystem::path dir(model_dir);
        tqllm::Tokenizer tok;
        tok.load((dir / "tokenizer.tqtok").string());
        const auto prompt = tok.encode(text);
        const auto P = uint32_t(prompt.size());
        tqllm::Model model(weights, (dir / "tokenizer.tqtok").string(), context, 1, engine);
        using clk = std::chrono::steady_clock;
        if (depth > 0) {  // timing only: verify(n) + commit(n) at position depth (KV contents do not affect cost)
            std::vector<uint32_t> in(n, prompt[0]), next;
            std::vector<double> times;
            // timed calls after as many untimed warm-up calls (clocks ramp up)
            for (int i = 0; i < 2 * iters; ++i) {
                const auto v0 = clk::now();
                model.verify(in, depth + (i % iters) * n, next);
                model.commit(n);
                if (i >= iters) times.push_back(std::chrono::duration<double, std::milli>(clk::now() - v0).count());
            }
            std::sort(times.begin(), times.end());
            std::printf("depth %u verify(%d)+commit median %.2f ms\n", depth, n, times[times.size() / 2]);
            return 0;
        }

        // A: sequential greedy chain
        for (uint32_t i = 0; i + 1 < P; ++i) model.forward(prompt[i], i, false);
        std::vector<uint32_t> seq;
        uint32_t t = model.forward(prompt[P - 1], P - 1, true);
        const auto a0 = clk::now();
        for (int i = 0; i < chain; ++i) {
            seq.push_back(t);
            t = model.forward(t, P + i, true);
        }
        const double single_ms = std::chrono::duration<double, std::milli>(clk::now() - a0).count() / chain;

        // B: same chain through verify(n) + commit(n) (teacher forcing on the sequential chain)
        for (uint32_t i = 0; i + 1 < P; ++i) model.forward(prompt[i], i, false);
        int agree = 0, total = 0, first_diff = -1;
        uint32_t pos = P - 1;
        uint32_t last = prompt[P - 1];
        double verify_ms = 0;
        int calls = 0;
        std::vector<double> times;
        std::vector<uint32_t> next;
        const int k = keep > 0 ? keep : n;
        for (int c = 0; c + n <= chain; c += k) {
            std::vector<uint32_t> in{last};
            for (int i = 0; i < n - 1; ++i) in.push_back(seq[c + i]);
            if (garbage)  // columns >= k (not committed) get unrelated tokens; columns < k must be unaffected
                for (int i = (keep > 0 ? keep : n); i < n; ++i) in[i] = 1000 + 37 * i;
            const auto v0 = clk::now();
            model.verify(in, pos, next);
            times.push_back(std::chrono::duration<double, std::milli>(clk::now() - v0).count());
            verify_ms += times.back();
            ++calls;
            const int check = garbage ? (keep > 0 ? keep : n) : n;  // only columns with a true prefix
            for (int i = 0; i < check; ++i) {
                ++total;
                if (next[i] == seq[c + i]) ++agree;
                else if (first_diff < 0) first_diff = c + i;
            }
            model.commit(k);  // k < n exercises the replay rollback of the recurrent layers
            pos += k;
            last = seq[c + k - 1];
        }
        std::printf("verify n=%d: %d/%d argmax agree with sequential greedy (first diff at %d)\n", n, agree, total,
                    first_diff);
        std::sort(times.begin(), times.end());
        const double med = times[times.size() / 2];
        std::printf("single step %.2f ms | verify(%d) median %.2f ms (mean %.2f, first calls capture graphs) = %.2fx | "
                    "tokens/s if all accepted: %.1f\n",
                    single_ms, n, med, verify_ms / calls, med / single_ms, 1000.0 * n / med);
        std::printf("sequential: %s\n", tok.decode(seq).c_str());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
