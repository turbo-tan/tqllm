// Plain greedy decode on the Metal target, one token per forward (no drafter): the reference text --exact on promises.
// usage: tqmlx-ar model.gguf prompt.txt count [prompt-cache]   -> prints the generated token ids, space separated, on one
// line. prompt-cache: prefill in the chunks tqserve --prompt-cache on uses.
// The prompt file's text is tokenized as the server does (Tokenizer::encode).
#include "qwen35.hpp"
#include "turn_reuse.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {std::fprintf(stderr, "usage: tqmlx-ar model.gguf prompt.txt count [prompt-cache]\n"); return 2;}
    std::ifstream in(argv[2]); std::stringstream ss; ss << in.rdbuf();
    tqllm::Tokenizer tok; tok.load(argv[1]);
    const auto prompt = tok.encode(ss.str());
    const int count = std::atoi(argv[3]);
    const uint32_t eos = tok.eos_id();
    Qwen35 engine(argv[1], int(prompt.size()) + count + 64);
    engine.set_exact(true);
    engine.reset(); engine.reserve_kv(int(prompt.size()) + count + 16); engine.kv_format(KvTier::Bf16);
    // the prompt in the server's prefill chunks (turn_reuse.hpp: they restart at message starts)
    const int chunk = std::getenv("TQLLM_MLX_PREFILL_CHUNK") ? std::max(128, std::atoi(std::getenv("TQLLM_MLX_PREFILL_CHUNK"))) : 8192;
    // --prompt-cache on (5th argument "prompt-cache"): the server's chunks restart at message starts; off: plain chunks
    const bool turn_chunks = argc > 4 && std::string(argv[4]) == "prompt-cache";
    const auto starts = turn_chunks ? message_starts(prompt, tok.special_id("<|im_start|>")) : std::vector<uint32_t>{};
    auto logits = mx::array(0.0f);
    for (size_t off = 0; off < prompt.size();) {
        const size_t end = turn_chunk_end(starts, uint32_t(off), uint32_t(prompt.size()), uint32_t(chunk));
        logits = engine.forward(std::span<const uint32_t>(prompt.data() + off, end - off));
        off = end;
    }
    uint32_t next = mx::argmax(logits).item<uint32_t>();
    std::vector<uint32_t> ids;
    while (int(ids.size()) < count) {
        if (next == eos) break;  // the server's text ends before its stop token
        ids.push_back(next);
        logits = engine.forward(std::vector<uint32_t>{next});
        next = mx::argmax(logits).item<uint32_t>();
    }
    for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%u", i ? " " : "", ids[i]);
    std::printf("\n");
    std::fprintf(stderr, "AR prompt %zu out %zu\n", prompt.size(), ids.size());
    return 0;
}
