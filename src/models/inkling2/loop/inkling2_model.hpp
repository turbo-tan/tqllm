#pragma once
// Inkling-2-Mini text model on one GPU with every weight resident (a GB10 holds the 18.9 GB), the reference path for the
// port: chunks of tokens through the 48 layer visits (prelude 0..5, the core 6..17 three times with cross-pass messages,
// coda 18..23), each visit with its own KV cache and four convolution states. No sampling or serving here.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tqllm {

// fast dense format for --exact off: INT8 W8A16 rows (half the BF16 bytes; the gemv decodes int8 in registers on any
// sm_80+ card, gated by KLD / PPL against the BF16 path)
// default prompt chunk for serving: 8192 on cards with >= 64 GB (GB10), else 2048 (24 GB cards keep their scratch small)
int inkling2_prompt_chunk();
inline bool inkling2_int8_dense(bool exact, int sm_major, int sm_minor) { (void)sm_minor; return !exact && sm_major >= 8; }

class Inkling2Model {
public:
    // path: the published checkpoint folder (or a GGUF with the inkling2-loop names); context: cache rows per visit;
    // chunk: the most tokens one forward() call takes; hf_attention: attention in the transformers reference's eager BF16
    // arithmetic (to gate against its golden), else F32 scores and softmax
    // expert_budget: device bytes for the routed experts (default: the free memory after everything else); experts that
    // do not fit stay in mapped host memory and stream over the bus
    // exact (default): BF16 dense weights as the checkpoint; --exact off: the device's fast dense format
    // (inkling2_int8_dense), gated by KLD / PPL against the BF16 path
    Inkling2Model(const std::string & path, int context, int chunk = 64, bool hf_attention = false, size_t expert_budget = SIZE_MAX,
                  bool exact = true);
    ~Inkling2Model();
    Inkling2Model(const Inkling2Model &) = delete;
    Inkling2Model & operator=(const Inkling2Model &) = delete;
    // tokens at positions length() .. length() + n - 1 (n <= chunk); logits (softcapped, F32) of the last `want` of them,
    // row-major [want][vocab()]
    void forward(const int * tokens, int n, int want, std::vector<float> & logits);
    void reset();  // empty caches and states
    // per-visit error against the reference's own hidden states (x0, visit outputs, final), one step each; n <= chunk
    void debug_visits(const std::vector<float> & hs, int n);
    // record the routed expert ids (topk per token) of every visit call, in call order; nullptr stops
    void log_routes(std::vector<std::vector<int>> * log);
    int length() const;
    int chunk() const;  // the most tokens one forward() call takes
    // partial residency: move every layer's most-routed experts (counted since load) onto the device; returns the experts
    // moved (0 when all are resident). The decode graph stays valid (its kernels read the pointer tables).
    int replan_experts();
    int resident_experts() const;  // experts per layer on the device (of 128)
    // MTP (mtp.safetensors in dir; before the first forward): every committed token's target taps become draft context.
    // draft(anchor, steps): the first `steps` (<= 7) greedy draft tokens of the block after `anchor` at position length(). verify(tokens, n <= 8): the target over
    // the block (all n logits), then commit(keep): keep its first keep tokens (conv states rolled back, the rest dropped).
    void enable_mtp(const std::string & dir);
    void draft(int anchor, std::vector<int> & tokens, int steps = 7);
    void verify(const int * tokens, int n, std::vector<float> & logits);
    void commit(int keep);
    int vocab() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace tqllm
