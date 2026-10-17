#include "qwen35.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/memory.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
using tqllm::mlx_backend::Qwen35;
int main(int argc, char** argv) {
    try {
        const bool strict = argc == 3 && std::string(argv[2]) == "--strict";
        if (argc != 2 && !strict) throw std::invalid_argument("usage: tqmlx-layer-audit model.gguf [--strict]");
        auto file = tqllm::gguf_open(argv[1]);
        const int layers = int(file.num("qwen35.block_count")) - int(file.num("qwen35.nextn_predict_layers",0));
        const int hidden = int(file.num("qwen35.embedding_length"));
        if (layers <= 0 || hidden <= 0) throw std::invalid_argument("invalid layer audit dimensions");
        tqllm::Tokenizer tokenizer; tokenizer.load(argv[1]);
        auto prompt = tokenizer.encode("Write a Python function that merges two sorted lists, then explain its time complexity.");
        auto tokens = tokenizer.encode("The function compares the first elements of each list before appending the remaining elements, and returns the merged list.");
        if (tokens.size() < 16) throw std::runtime_error("layer audit fixture needs sixteen tokens");
        mx::set_cache_limit(512*1024*1024);
        Qwen35 model(argv[1],256);
        model.gdn_mode(Qwen35::GdnMode::scan); model.verify_tiles(true);
        std::vector<int> selected(layers); std::iota(selected.begin(),selected.end(),0);
        model.feature_layers(selected); model.forward(prompt);
        auto start = model.checkpoint();
        for (int width : {1,2,4,8,16}) {
            model.restore(start);
            std::vector<std::vector<float>> reference(width);
            for (int t = 0; t < width; ++t) {
                model.forward({tokens.data()+t,1},true);
                auto features = model.feature_states(), last = model.hidden_states();
                mx::eval(features,last);
                reference[t].assign(features.data<float>(),features.data<float>()+features.size());
                reference[t].insert(reference[t].end(),last.data<float>(),last.data<float>()+last.size());
            }
            model.restore(start); Qwen35::Verification ticket;
            const char* capture = width == 8 ? std::getenv("TQLLM_MLX_CAPTURE_VERIFY") : nullptr;
            if (capture) {
                auto warm = model.verify({tokens.data(),size_t(width)},ticket);
                mx::eval(warm);model.restore(start);
                mx::metal::start_capture(capture);
            }
            const auto began = std::chrono::steady_clock::now();
            try {
                auto logits = model.verify({tokens.data(),size_t(width)},ticket);
                mx::eval(logits);
            } catch (...) {
                if (capture) mx::metal::stop_capture();
                throw;
            }
            if (capture) mx::metal::stop_capture();
            std::cout << "LAYER_VERIFY_MS width=" << width << " ms="
                      << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count() << std::endl;
            auto features = model.feature_states(), last = model.hidden_states(); mx::eval(features,last);
            int first = -1;
            for (int l = 0; l <= layers; ++l) {
                size_t changed = 0; float maximum = 0; double squared = 0, total = 0;
                for (int t = 0; t < width; ++t) for (int h = 0; h < hidden; ++h) {
                    const float a = reference[t][size_t(l)*hidden+h];
                    const float b = l == layers ? last.data<float>()[size_t(t)*hidden+h] :
                        features.data<float>()[size_t(t)*layers*hidden+size_t(l)*hidden+h];
                    if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("nonfinite hidden state");
                    changed += std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b);
                    maximum = std::max(maximum,std::abs(a-b));
                    squared += double(a-b)*(a-b); total += double(a)*a;
                }
                if (changed && first < 0) first = l;
                if (width == 1 && changed) throw std::runtime_error("width-one audit changed hidden state");
                std::cout << "{\"kind\":\"layer_column_audit\",\"width\":" << width << ",\"layer_input\":" << l
                          << ",\"changed\":" << changed << ",\"total\":" << size_t(width)*hidden
                          << ",\"max_abs\":" << maximum << ",\"rel_l2\":" << std::sqrt(squared/std::max(total,1e-20)) << "}" << std::endl;
            }
            std::cout << "LAYER_FIRST_DIFFERENCE width=" << width << " layer_input=" << first << std::endl;
            if (strict && first >= 0) throw std::runtime_error("batch hidden states differ from sequential forward");
        }
        std::cout << "LAYER_AUDIT_COMPLETE diagnostic_only=1" << std::endl;
    } catch (const std::exception& e) { std::cerr << "LAYER_AUDIT_ERROR " << e.what() << '\n'; return 1; }
}
