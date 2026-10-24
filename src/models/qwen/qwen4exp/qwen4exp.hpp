#pragma once
#include "gguf.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tqllm {

struct PleHashConfig {
    int ngram = 0, heads_per_ngram = 0, eos = 0, image = 0;
    std::vector<uint64_t> multipliers, offsets, vocab_sizes;
    void validate() const;
    bool operator==(const PleHashConfig &) const = default;
};

struct Qwen4ExpConfig {
    int layers = 0, hidden = 0, experts = 0, topk = 0, expert_ff = 0, shared_ff = 0, dense_ff = 0;  // experts = 0: dense FFN
    int hc = 0, hc_low_rank = 0, ple_kernel = 0, ple_dim = 0;
    int index_heads = 0, index_dim = 0, index_topk = 0;
    bool gdn_silu_gate = false;  // ssm.gate_activation = "silu": GDN output gate z * sigmoid(z) instead of sigmoid(z)
    std::vector<int> ple_layers, compress_ratios;
    PleHashConfig ple;
    bool operator==(const Qwen4ExpConfig &) const = default;
};
Qwen4ExpConfig qwen4exp_config(const GgufFile &);
std::string qwen4exp_prefix(const GgufFile &);  // "qwen4exp." or "qwen4."

// One instance per sequence. Rejected speculative tokens are removed before the next append.
class PleTokenHistory {
    PleHashConfig config_;
    std::vector<int32_t> tokens_;
    size_t capacity_;
public:
    explicit PleTokenHistory(PleHashConfig, size_t capacity);
    size_t size() const { return tokens_.size(); }
    void reset() { tokens_.clear(); }
    void rollback(size_t accepted_size);
    std::vector<int32_t> append(int32_t token);
    std::vector<int32_t> append_image();
    std::vector<int32_t> append_batch(const std::vector<int32_t> &);
};

struct QsaSelection {
    std::vector<int32_t> positions;
    std::vector<uint8_t> valid;
};

// CPU oracle for one contiguous text sequence. Scores are for complete ratio-sized blocks.
QsaSelection qsa_select_visible(const std::vector<float> & block_scores, int cells, int query_position,
                                int ratio, int top_k);

} // namespace tqllm
