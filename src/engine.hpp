#pragma once
// Internal per-architecture decode engine behind tqllm::Model.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tqllm/model.hpp"

namespace tqllm {

struct Engine {
    virtual ~Engine() = default;
    // One token at `pos`. With `sample`, runs the LM head and returns the greedy next token (one host sync);
    // otherwise returns 0 without syncing. `capture` (validation only) receives the last-token hidden state after
    // every layer, [layers][hidden], the last entry after the final norm.
    virtual uint32_t step(uint32_t token, uint32_t pos, bool sample, float * capture) = 0;
    // Speculative verify: runs `n` tokens (n <= 8) at positions pos, pos+1, ... in one pass and writes, for each, the
    // greedy next token to `next` (host). KV is written for all n; the recurrent state after each token is kept.
    // commit(k) then keeps the state after the first k tokens (1 <= k <= n); rejected KV entries are overwritten later.
    virtual void verify(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) = 0;
    virtual void commit(int k) = 0;
    // Greedy generation with MTP self-speculation (D draft tokens per step, D <= 7). Output equals the greedy decode
    // of the verify path. stats (optional): [0] = steps, [1] = drafted, [2] = accepted drafts, [3] = decode microseconds.
    virtual std::vector<uint32_t> generate_spec(const std::vector<uint32_t> &, uint32_t, int, uint64_t *) {
        throw std::runtime_error("speculative decoding not supported for this model");
    }
    // Sampling for generate_spec (temperature 0 = greedy): top-k <= 16 of the exact head logits, top-p, seed.
    // presence: subtracted from the logit of every candidate seen in the last kPresenceLastN output tokens
    // (OpenAI presence_penalty; breaks the phrase-level loops of long sampled thinking)
    virtual void set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence = 0.f) {
        if (temperature > 0.f) throw std::runtime_error("sampling not supported for this model");
        (void) top_p, (void) top_k, (void) seed, (void) presence;
    }
    // Multi-slot serving: each slot holds one sequence (KV, recurrent state, drafter K/V, sampling). select_slot makes
    // a slot active for the calls that follow; spec_begin prefills it, spec_advance runs one draft + verify step
    // (false once done) and spec_finish returns the generated ids. generate_spec = begin + advance* + finish.
    virtual int slots() const { return 1; }
    virtual bool has_drafter() const { return false; }
    virtual bool native_mtp() const { return false; }  // --mtp drafts with the model's own nextn head
    virtual bool incremental() const { return false; }  // spec_begin / spec_advance work (chunked prefill), drafter or not
    virtual void select_slot(int s) {
        if (s != 0) throw std::runtime_error("this model has a single slot");
    }
    virtual void spec_begin(const std::vector<uint32_t> &, uint32_t, int) {
        throw std::runtime_error("incremental speculative decoding not supported for this model");
    }
    virtual bool spec_advance() { return false; }
    // Image embeddings for the next spec_begin on the active slot: prompt tokens [at, at + nx * ny) are the image's
    // placeholders; emb: device FP32 [nx * ny][hidden], row-major over the nx x ny grid, complete when called.
    using PromptImage = Model::PromptImage;
    virtual void set_prompt_images(const std::vector<PromptImage> &) {
        throw std::runtime_error("this model does not take images");
    }
    // Batched step over n slots (each begun with spec_begin): one verify pass for all of them; done[i] = finished.
    virtual void spec_step(const int *, int, bool *) {
        throw std::runtime_error("batched speculative decoding not supported for this model");
    }
    virtual std::vector<uint32_t> spec_finish(uint64_t *) { return {}; }
    virtual std::vector<uint32_t> spec_tokens(int) const { return {}; }  // slot's ids so far (streaming)
    // DFlash drafter training data: prefill `tokens` from position 0 and return the drafter's target features
    // [P][nf * hidden] (raw BF16 bits) of every position; returns nf (0: no drafter loaded)
    virtual int drafter_features(const std::vector<uint32_t> &, std::vector<uint16_t> &) { return 0; }
    // Chunked prefill of tokens at positions [pos, pos + n) with no LM head (KV and recurrent state only); false: unsupported.
    virtual bool prefill_state(const std::vector<uint32_t> &, uint32_t) { return false; }
    virtual const float * logits() const = 0;  // device, [vocab] FP32, valid after a sampling step
    virtual const float * verify_logits() const { return nullptr; }  // device, [n][vocab] FP32 of the last verify
    virtual void set_full_logits(bool) {}  // verify computes every logit (int8 head) instead of the two-stage top-16
    virtual void synchronize() = 0;
    virtual uint32_t layers() const = 0;
    virtual uint32_t hidden() const = 0;
    virtual uint32_t vocab() const = 0;
    virtual uint32_t eos() const = 0;
    virtual uint32_t context() const = 0;
    virtual size_t kv_token_bytes() const { return 0; }  // KV bytes of one position, all slots (0: not a KV-store engine)
    virtual size_t kv_host_bytes() const { return 0; }   // KV bytes in host memory (the --kv-host tail)
    virtual size_t kv_device_bytes() const { return 0; } // KV bytes in device memory (pool pages incl. granule rounding)
    virtual size_t kv_round_bytes() const { return 0; }  // most device memory a host tail rounds up (mapping granules)
    virtual bool kv_idle_pending() const { return false; }  // TqCache compaction work is waiting (call kv_idle_step when idle)
    virtual bool kv_idle_step() { return false; }           // compacts one old block; true while more is pending
};

// Qwen3.5 dense hybrid (Gated DeltaNet + gated attention), BF16 weights read from safetensors in `dir`.
// `slots` sequences share the weights; each gets `context` positions.
// Flash-Next (qwen4exp) engine: compiled into tqserve with tools/tqfn_stage.cpp (TQFN_SERVE_ENGINE), which registers it;
// a model path ending in .tqfn (a manifest of shard / expert / MTP files) selects it.
using FlashNextFactory = std::unique_ptr<Engine> (*)(const std::string & manifest, const std::string & tokenizer, uint32_t context,
                                                     int slots, const EngineOptions & options);
void register_flashnext_engine(FlashNextFactory factory);
std::unique_ptr<Engine> make_qwen35_engine(const std::string & dir, const std::string & tokenizer, uint32_t context,
                                           int slots = 1, const EngineOptions & options = {});

} // namespace tqllm
