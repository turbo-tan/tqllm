// tqbench: decode throughput vs context depth, JSON lines on stdout.
// Usage: tqbench --model-dir DIR [--depths 512,2048,8192,32000] [--gen 128] [--reps 3] [--warmup 16]
// Context is filled with synthetic token ids (timing only). Prefill is currently token-by-token (Q5 adds batched
// prefill), reported separately. Decode: `gen` tokens per rep from the given depth, median/min/max over `reps`.
#include "tqllm/options.hpp"
#include "tqllm/model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;
double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }
uint32_t synth(uint32_t i) { return 100 + (i * 7919u) % 150000u; }
} // namespace

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqbench");
    try {
        std::string weights, model_dir, depths_arg = "512,2048,8192,32000";
        uint32_t gen = 128, reps = 3, warmup = 16;
        tqllm::EngineOptions engine;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--model-dir") model_dir = value();
            else if (arg == "--weights") weights = value();  // e.g. a .gguf; tokenizer still from --model-dir
            else if (arg == "--depths") depths_arg = value();
            else if (arg == "--gen") gen = std::stoul(value());
            else if (arg == "--reps") reps = std::stoul(value());
            else if (arg == "--warmup") warmup = std::stoul(value());
            else if (tqllm::parse_engine_option(argc, argv, i, engine)) continue;
            else throw std::invalid_argument("unknown argument " + arg);
        }
        if (model_dir.empty()) throw std::invalid_argument("need --model-dir");
        std::vector<uint32_t> depths;
        std::stringstream ss(depths_arg);
        for (std::string t; std::getline(ss, t, ',');) depths.push_back(std::stoul(t));
        std::sort(depths.begin(), depths.end());
        const uint32_t context = depths.back() + gen + warmup;

        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        int driver = 0, runtime = 0;
        cudaDriverGetVersion(&driver);
        cudaRuntimeGetVersion(&runtime);
        size_t free0 = 0, total = 0;
        cudaMemGetInfo(&free0, &total);

        const std::filesystem::path dir(model_dir);
        tqllm::Model model((weights.empty() ? (dir / "weights.tqbin").string() : weights), (dir / "tokenizer.tqtok").string(), context, 1, engine);
        size_t free1 = 0;
        cudaMemGetInfo(&free1, &total);

        uint32_t filled = 0;
        double prefill_s = 0;
        for (const auto depth : depths) {
            const auto t0 = clk::now();
            for (; filled < depth; ++filled) model.forward(synth(filled), filled, false);
            model.synchronize();
            prefill_s += secs(t0, clk::now());

            uint32_t tok = synth(depth);
            for (uint32_t i = 0; i < warmup; ++i) tok = model.forward(tok, depth + i, true);
            std::vector<double> rates;
            for (uint32_t r = 0; r < reps; ++r) {
                tok = synth(depth);
                const auto t1 = clk::now();
                for (uint32_t i = 0; i < gen; ++i) tok = model.forward(tok, depth + i, true);
                rates.push_back(gen / secs(t1, clk::now()));
            }
            std::sort(rates.begin(), rates.end());
            std::printf("{\"engine\":\"tqllm\",\"model\":\"%s\",\"gpu\":\"%s\",\"sm\":%d%d,\"cuda_runtime\":%d,\"driver\":%d,"
                        "\"depth\":%u,\"gen\":%u,\"reps\":%u,\"decode_tps_median\":%.2f,\"decode_tps_min\":%.2f,"
                        "\"decode_tps_max\":%.2f,\"prefill_tps_tokenwise\":%.1f,\"model_mem_mib\":%.0f}\n",
                        dir.parent_path().filename().string().c_str(), prop.name, prop.major, prop.minor, runtime, driver,
                        depth, gen, reps, rates[rates.size() / 2], rates.front(), rates.back(), filled / prefill_s,
                        double(free0 - free1) / (1 << 20));
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
