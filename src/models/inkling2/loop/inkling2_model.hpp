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
                  bool exact = true, size_t drafter_bytes = 0, int expert_kernels = -1);
    ~Inkling2Model();
    Inkling2Model(const Inkling2Model &) = delete;
    Inkling2Model & operator=(const Inkling2Model &) = delete;
    // tokens at positions length() .. length() + n - 1 (n <= chunk); logits (softcapped, F32) of the last `want` of them,
    // row-major [want][vocab()]
    void forward(const int * tokens, int n, int want, std::vector<float> & logits);
    void reset();  // empty caches and states
    // one decode token at position length(): returns the greedy next token (argmax of the softcapped logits, lowest index
    // on ties, == forward()'s logits' argmax) without producing the logits; exact mode reads an INT8 copy of the LM head
    // and rescores in BF16 only the rows that can still win
    int forward_greedy(int token);
    // conversation reuse: snapshot() keeps the conv states, the position and the sliding-window rings' last window of
    // K / V rows (later tokens may overwrite them; the full-attention rows stay in place); restore() goes back to exactly
    // that state (false only without a snapshot or inside a verify block). Prefilling the same tokens from there in the
    // same chunks gives the same results bitwise as prefilling from 0.
    void snapshot();
    bool restore();
    // per-visit error against the reference's own hidden states (x0, visit outputs, final), one step each; n <= chunk
    void debug_visits(const std::vector<float> & hs, int n);
    // record the routed expert ids (topk per token) of every visit call, in call order; nullptr stops
    void log_routes(std::vector<std::vector<int>> * log);
    int length() const;
    int chunk() const;  // the most tokens one forward() call takes
    // partial residency: move every layer's most-routed experts (counted since load) onto the device; returns the experts
    // moved (0 when all are resident). The decode graph stays valid (its kernels read the pointer tables).
    int replan_experts();
    // true once after a step that also re-planned the resident experts (the spec policy leaves that step's time out)
    bool take_replanned();
    int resident_experts() const;  // experts per layer on the device (of 128)
    // MTP (mtp.safetensors in dir; before the first forward): every committed token's target taps become draft context.
    // draft(anchor, steps): the first `steps` (<= 7) greedy draft tokens of the block after `anchor` at position length(). verify(tokens, n <= 8): the target over
    // the block (all n logits), then commit(keep): keep its first keep tokens (conv states rolled back, the rest dropped).
    void enable_mtp(const std::string & dir);  // or a DFlash2 drafter folder (incoai/Inkling-2-Mini-DFlash2)
    // device bytes enable_mtp(dir) will take (weights, conversion peak, caches, scratch): pass it as drafter_bytes to the
    // constructor so the expert plan leaves it free
    static size_t drafter_bytes(const std::string & dir);
    // the decode / verify kernel families the load picked, e.g. "experts tensor cores (auto), dense tensor cores"
    std::string kernels() const;
    // DFlash2 reference check (tools/inkling2/dflash2_reference.py): the drafter alone over `tokens`' input rows as taps
    void draft_trace(const std::vector<int> & tokens, int anchor, std::vector<float> & hidden, std::vector<int> & ids, std::vector<int> & path);
    void draft(int anchor, std::vector<int> & tokens, int steps = 7);
    // DFlash2: the same draft, plus conf[i] = the softmax weight of tokens[i] among its row's 16 head candidates (empty for
    // the MTP drafter); read in the draft's own device-to-host copy, so it costs no extra sync
    void draft(int anchor, std::vector<int> & tokens, int steps, std::vector<float> & conf);
    bool dflash_drafter() const;  // the drafter is DFlash2 (one pass drafts the whole block)
    void verify(const int * tokens, int n, std::vector<float> & logits);
    void commit(int keep);
    // verify() without logits: next[i] = the greedy token after tokens[0 .. i] (exact, as argmax of verify()'s rows)
    void verify_greedy(const int * tokens, int n, int * next);
    int vocab() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace tqllm
