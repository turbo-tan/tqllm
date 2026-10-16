#pragma once
#include "tqllm/options.hpp"
#include "tqllm/tokenizer.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace tqllm {
struct WeightInfo { std::string name; uint8_t dtype; std::vector<uint64_t> shape; uint64_t offset; uint64_t bytes; };
class Model {
public:
    // `slots` > 1: independent sequences sharing the weights (qwen35 GGUF + DFlash drafter), `context` positions each.
    Model(const std::string & weights, const std::string & tokenizer, uint32_t context, int slots = 1,
          const EngineOptions & options = {});
    ~Model();
    Model(Model &&) noexcept;
    Model & operator=(Model &&) noexcept;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    std::vector<uint32_t> generate(const std::vector<uint32_t> & prompt, uint32_t count);
    // Validation hook: runs the prompt and captures, for its last token, the hidden state after every layer
    // ([layers][hidden]; the last entry has the final norm applied, matching Hugging Face) and the logits.
    void capture(const std::vector<uint32_t> & prompt, std::vector<float> & hidden, std::vector<float> & logits);
    // One token at position `pos`; returns the greedy next token when `sample`, else 0 (no host sync).
    uint32_t forward(uint32_t token, uint32_t pos, bool sample);
    // Speculative decoding primitives (see Engine::verify / commit).
    void verify(const std::vector<uint32_t> & tokens, uint32_t pos, std::vector<uint32_t> & next);
    void commit(int accepted);
    // Device pointer to the last verify's logits, [n][vocab] FP32 (complete only after set_full_logits(true))
    const float * verify_logits() const;
    // Full int8-head logits; default off = empirical 4-bit shortlist + int8 candidate re-score.
    // Set before the first forward/verify call on a fresh model (captured graphs retain their head path).
    void set_full_logits(bool on);
    // Greedy generation with MTP self-speculation; stats (optional, 5 values): steps, drafted, accepted, decode us, prefill us.
    std::vector<uint32_t> generate_spec(const std::vector<uint32_t> & prompt, uint32_t count, int draft,
                                        uint64_t * stats = nullptr);
    // Sampling for generate_spec: temperature 0 = greedy; top_k <= 16, top_p, seed.
    void set_sampling(float temperature, float top_p = 1.f, int top_k = 0, uint64_t seed = 0, float presence = 0.f);
    // Multi-slot: select_slot makes a slot active for the calls that follow (set_sampling included).
    int slots() const;
    uint32_t context() const;  // positions per slot (the memory fit may have halved the requested context)
    // TqCache compaction (--kv-compact): work waiting for an idle moment; one step compacts one old block (~0.3 s)
    bool kv_idle_pending() const;
    bool kv_idle_step();
    bool has_drafter() const;  // a DFlash2 drafter is loaded (-md)
    bool native_mtp() const;   // --mtp: the model's own MTP head drafts
    std::string fit_note() const;  // what the memory fit chose instead of the defaults (KV, context, buffers), or empty
    bool incremental() const;  // spec_begin / spec_advance / spec_finish available (chunked prefill), with or without a drafter
    void select_slot(int slot);
    // Image embeddings for the next spec_begin on the active slot: prompt tokens [at, at + nx * ny) are placeholders;
    // emb: device FP32 [nx * ny][hidden_size()], complete when called (vision models, mmproj encoders).
    struct PromptImage {
        uint32_t at;
        int nx, ny;
        const float * emb;
    };
    void set_prompt_images(const std::vector<PromptImage> & images);
    void spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int draft);
    bool spec_advance();  // one draft + verify step; false once done
    // Batched step: one verify pass over all listed slots (weights read once); done[i] = slot slots[i] finished.
    void spec_step(const std::vector<int> & slots, std::vector<char> & done);
    std::vector<uint32_t> spec_finish(uint64_t * stats = nullptr);
    std::vector<uint32_t> spec_tokens(int slot = 0) const;  // ids generated so far by a begun slot (streaming)
    // Drafter target features [P][nf * hidden] (BF16 bits) of every position of `tokens` (prefilled from 0); returns nf.
    int drafter_features(const std::vector<uint32_t> & tokens, std::vector<uint16_t> & feats);
    // Prefill tokens at positions [pos, pos + n) without the LM head (KV and recurrent state only); false: unsupported.
    bool prefill_state(const std::vector<uint32_t> & tokens, uint32_t pos);
    void synchronize();
    uint32_t layers() const;
    uint32_t hidden_size() const;
    uint32_t vocab_size() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
