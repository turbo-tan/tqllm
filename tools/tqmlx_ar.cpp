// Plain greedy decode on the Metal target, one token per forward (no drafter): the reference text --exact on promises.
// usage: tqmlx-ar model.gguf prompt.txt count   -> prints the generated token ids, space separated, on one line.
// The prompt file's text is tokenized as the server does (Tokenizer::encode).
#include "qwen35.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/mlx.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

int main(int argc, char** argv) {
    if (argc != 4) {std::fprintf(stderr, "usage: tqmlx-ar model.gguf prompt.txt count\n"); return 2;}
    std::ifstream in(argv[2]); std::stringstream ss; ss << in.rdbuf();
    tqllm::Tokenizer tok; tok.load(argv[1]);
    const auto prompt = tok.encode(ss.str());
    const int count = std::atoi(argv[3]);
    const uint32_t eos = tok.eos_id();
    Qwen35 engine(argv[1], int(prompt.size()) + count + 64);
    engine.set_exact(true);
    engine.reset(); engine.reserve_kv(int(prompt.size()) + count + 16); engine.kv_format(KvTier::Bf16);
    auto logits = engine.forward(prompt);
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
