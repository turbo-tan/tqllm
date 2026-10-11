// tqgolden: validation dump for the golden-fixture gates (Q4).
// Usage: tqgolden --model-dir DIR (--prompt TEXT | --text FORMATTED) --out PREFIX [--greedy N] [--context N]
// --prompt wraps TEXT in the Qwen2.5 chat template; --text tokenizes the given string as-is.
// Writes PREFIX.ids (chat-formatted token ids, one line), PREFIX.hidden.f32 ([layers][hidden]),
// PREFIX.logits.f32 ([vocab]) and PREFIX.greedy (generated ids, one line).
#include "tqllm/options.hpp"
#include "tqllm/model.hpp"
#include "tqllm/tokenizer.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void write_ids(const std::string & path, const std::vector<uint32_t> & ids) {
    std::ofstream o(path);
    for (size_t i = 0; i < ids.size(); ++i) o << (i ? " " : "") << ids[i];
    o << '\n';
}

void write_f32(const std::string & path, const std::vector<float> & v) {
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(float)));
}
} // namespace

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqgolden");
    try {
        std::string weights, model_dir, prompt, text, out;
        uint32_t greedy = 32, context = 2048;
        tqllm::EngineOptions engine;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--model-dir") model_dir = value();
            else if (arg == "--weights") weights = value();  // e.g. a .gguf; tokenizer still from --model-dir
            else if (arg == "--prompt") prompt = value();
            else if (arg == "--text") text = value();
            else if (arg == "--out") out = value();
            else if (arg == "--greedy") greedy = std::stoul(value());
            else if (arg == "--context") context = std::stoul(value());
            else if (tqllm::parse_engine_option(argc, argv, i, engine)) continue;
            else throw std::invalid_argument("unknown argument " + arg);
        }
        if (model_dir.empty() || (prompt.empty() == text.empty()) || out.empty())
            throw std::invalid_argument("need --model-dir, exactly one of --prompt/--text, --out");
        const std::filesystem::path dir(model_dir);
        tqllm::Tokenizer tokenizer;
        tokenizer.load((dir / "tokenizer.tqtok").string());
        constexpr const char * system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
        const auto ids = !text.empty() ? tokenizer.encode(text)
                                       : tokenizer.encode(std::string("<|im_start|>system\n") + system +
                                                          "<|im_end|>\n<|im_start|>user\n" + prompt +
                                                          "<|im_end|>\n<|im_start|>assistant\n");
        write_ids(out + ".ids", ids);
        std::vector<float> hidden, logits;
        {
            tqllm::Model model((weights.empty() ? (dir / "weights.tqbin").string() : weights), (dir / "tokenizer.tqtok").string(), context, 1, engine);
            model.capture(ids, hidden, logits);
        }
        write_f32(out + ".hidden.f32", hidden);
        write_f32(out + ".logits.f32", logits);
        tqllm::Model model((weights.empty() ? (dir / "weights.tqbin").string() : weights), (dir / "tokenizer.tqtok").string(), context, 1, engine);
        write_ids(out + ".greedy", model.generate(ids, greedy));
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
