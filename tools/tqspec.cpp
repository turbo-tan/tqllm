// tqspec: MTP self-speculative greedy generation vs plain greedy (same engine): exactness and tok/s.
// Usage: tqspec --model-dir DIR --weights FILE.gguf --text PROMPT [--draft 3] [--gen 256] [--context 8192]
#include "tqllm/options.hpp"
#include "tqllm/model.hpp"
#include "tqllm/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqspec");
    try {
        std::string model_dir, weights, text, token_ids_file;
        int draft = 3;
        uint32_t gen = 256, context = 8192;
        bool compare = true, warmup = true, show_diff = false;
        std::string save;
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
            else if (arg == "--text-file") {
                std::ifstream in(value());
                text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            }
            else if (arg == "--token-ids-file") token_ids_file = value();
            else if (arg == "--draft") draft = std::stoi(value());
            else if (arg == "--gen") gen = std::stoul(value());
            else if (arg == "--context") context = std::stoul(value());
            else if (arg == "--no-compare") compare = false;
            else if (arg == "--no-warmup") warmup = false;  // long prompts: one prefill in total
            else if (arg == "--save") save = value();      // generated text (quality gates re-feed it)
            else if (arg == "--show-diff") show_diff = true;
            else if (tqllm::parse_engine_option(argc, argv, i, engine)) continue;
            else throw std::invalid_argument("unknown argument " + arg);
        }
        const std::filesystem::path dir(model_dir);
        tqllm::Tokenizer tok;
        tok.load((dir / "tokenizer.tqtok").string());
        std::vector<uint32_t> prompt;
        if (!token_ids_file.empty()) {
            std::ifstream in(token_ids_file);
            uint32_t id;
            while (in >> id) prompt.push_back(id);
            if (prompt.empty()) throw std::runtime_error("token ID file is empty or unreadable");
        } else {
            prompt = tok.encode(text);
        }
        tqllm::Model model(weights, (dir / "tokenizer.tqtok").string(), context, 1, engine);
        using clk = std::chrono::steady_clock;
        std::vector<uint32_t> base;
        double base_tps = 0;
        if (compare) {
            const auto t0 = clk::now();
            base = model.generate(prompt, gen);
            base_tps = base.size() / std::chrono::duration<double>(clk::now() - t0).count();
        }
        if (warmup)  // without: the timed prefill run below captures the graphs too
            model.generate_spec(prompt, 16, draft);  // warm-up: captures the verify graphs
        const bool quick = !warmup;
        double prefill_s = 0;
        if (!quick) {
            const auto tp = clk::now();
            model.generate_spec(prompt, 1, draft);  // prefill + first token only
            prefill_s = std::chrono::duration<double>(clk::now() - tp).count();
        }
        uint64_t st[5] = {0, 0, 0, 0, 0};
        const auto t1 = clk::now();
        const auto spec = model.generate_spec(prompt, gen, draft, st);
        if (!save.empty()) {
            std::vector<uint32_t> body;
            for (const uint32_t t : spec)
                if (t != tok.eos_id()) body.push_back(t);
            if (FILE * f = std::fopen(save.c_str(), "w")) {
                std::fputs(tok.decode(body).c_str(), f);
                std::fclose(f);
            }
        }
        const double secs = std::chrono::duration<double>(clk::now() - t1).count();
        // (includes prefill; tokens/s below counts generated tokens only)
        int same = 0;
        for (size_t i = 0; i < std::min(base.size(), spec.size()); ++i) {
            if (base[i] != spec[i]) break;
            ++same;
        }
        const double decode_s = st[3] > 0 ? double(st[3]) / 1e6 : std::max(1e-9, secs - prefill_s);
        if (quick) prefill_s = secs - decode_s;  // prefill + first token of the same run
        std::printf("draft %d: %zu tokens, decode %.1f tok/s (%.2f s decode loop; prefill of %zu tokens %.2f s) | steps %llu, accepted "
                    "%llu/%llu (%.1f%%), %.2f tokens/step\n",
                    draft, spec.size(), (spec.size() - 1) / decode_s, decode_s, prompt.size(), prefill_s,
                    (unsigned long long) st[0],
                    (unsigned long long) st[2], (unsigned long long) st[1], 100.0 * st[2] / std::max<uint64_t>(1, st[1]),
                    double(spec.size()) / std::max<uint64_t>(1, st[0]));
        if (compare) {
            std::printf("plain greedy %.1f tok/s | spec matches plain greedy for the first %d of %zu tokens\n", base_tps,
                        same, spec.size());
            if (same < int(spec.size()) && show_diff) {
                const int a = std::max(0, same - 4), b = std::min<int>(int(spec.size()), same + 6);
                std::printf("plain:");
                for (int i = a; i < b; ++i) std::printf(" %u", base[i]);
                std::printf("\nspec: ");
                for (int i = a; i < b; ++i) std::printf(" %u", spec[i]);
                std::printf("\n");
            }
        }
        std::printf("--- %s\n", tok.decode(spec).c_str());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
