#include "tqllm/options.hpp"
#include "tqllm/runtime.hpp"
#ifdef TQLLM_HAS_CUDA
#include "tqllm/model.hpp"
#endif
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void usage() {
    std::cerr << "Usage:\n"
              << "  tqrun --plugin /path/to/libtqexample.so\n"
              << "  tqrun --model-dir DIR --prompt TEXT [--max-tokens N] [--context N]\n";
}
}

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqrun");
    try {
        std::string plugin, model_dir, prompt;
        uint32_t max_tokens = 64, context = 2048;
        tqllm::EngineOptions engine;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--plugin") plugin = value();
            else if (arg == "--model-dir") model_dir = value();
            else if (arg == "--prompt") prompt = value();
            else if (arg == "--max-tokens") max_tokens = std::stoul(value());
            else if (arg == "--context") context = std::stoul(value());
            else if (tqllm::parse_engine_option(argc, argv, i, engine)) continue;
            else { usage(); return 2; }
        }
        if (!plugin.empty()) {
            tqllm::Runtime runtime;
            runtime.load(plugin);
            for (const auto & id : runtime.kernels()) std::cout << id << '\n';
            auto plan = runtime.resolve("example.host.add.f32.v1", 4);
            float a[]{1, 2, 3, 4}, b[]{2, 4, 6, 8}, out[4]{};
            plan.run(a, b, out);
            for (int i = 0; i < 4; ++i) if (out[i] != a[i] + b[i]) return 1;
            std::cout << "plugin proof: PASS\n";
            return 0;
        }
#ifdef TQLLM_HAS_CUDA
        if (model_dir.empty() || prompt.empty()) { usage(); return 2; }
        const std::filesystem::path dir(model_dir);
        tqllm::Tokenizer tokenizer;
        tokenizer.load((dir / "tokenizer.tqtok").string());
        constexpr const char * system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
        const auto formatted = std::string("<|im_start|>system\n") + system + "<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        const auto input = tokenizer.encode(formatted);
        tqllm::Model model((dir / "weights.tqbin").string(), (dir / "tokenizer.tqtok").string(), context, 1, engine);
        const auto output = model.generate(input, max_tokens);
        std::cout << tokenizer.decode(output) << std::flush;
        return 0;
#else
        (void) model_dir; (void) prompt; (void) max_tokens; (void) context;
        throw std::runtime_error("model inference requires a CUDA-enabled build");
#endif
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
