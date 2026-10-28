#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tqllm::android_backend {
// exact keeps FP32 activations in every projection; the default quantizes them to int8 per 256.
// head_rows > 0 limits the output projection to the first head_rows tokens plus EOS; the other logits are -inf.
// Meant for a speculative draft model, whose guesses only need the common vocabulary.
struct Options { int context = 4096; int threads = 4; bool exact = false; int head_rows = 0; };
// Native Arm CPU baseline. Standard GGUF weights stay packed and memory-mapped.
class Model {
public:
    Model(const std::string & path, Options options = {});
    ~Model();
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    void reset();
    // Input must be consecutive. The returned logits belong to this instance.
    const std::vector<float> & forward(uint32_t token, bool head = true);
    // Consecutive tokens in one pass over the weights: logits for every token, or only the last one
    // (prefill). Bit-identical to calling forward() once per token.
    const std::vector<std::vector<float>> & forward_batch(const std::vector<uint32_t> & tokens, bool every = true);
    // Speculative decoding: checkpoint(), forward any tokens, then rollback(keep) leaves the model as if only the
    // first `keep` of them had been forwarded. Attention KV past the position is simply overwritten later.
    void checkpoint();
    void rollback(int keep);
    int position() const;
    int context() const;
    int vocabulary() const;
    uint32_t eos() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
