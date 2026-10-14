#include "qwen4exp.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace tqllm {
namespace {
uint64_t exact_integer(double x, const std::string & key, bool positive = false) {
    if (!std::isfinite(x) || x < (positive ? 1. : 0.) || x > 9007199254740991. || std::floor(x) != x)
        throw std::runtime_error("qwen4exp: invalid integer for " + key);
    return uint64_t(x);
}
int scalar(const GgufFile & f, const std::string & key, bool positive = true, double fallback = -1.) {
    const double value = fallback >= 0. ? f.num("qwen4exp." + key, fallback) : f.num("qwen4exp." + key);
    const uint64_t x = exact_integer(value, key, positive);
    if (x > uint64_t(std::numeric_limits<int>::max())) throw std::runtime_error("qwen4exp: integer too large for " + key);
    return int(x);
}
std::vector<uint64_t> array(const GgufFile & f, const std::string & key, size_t minimum) {
    const auto it = f.kv.find("qwen4exp." + key);
    if (it == f.kv.end() || it->second.arr.size() < minimum)
        throw std::runtime_error("qwen4exp: missing or short array " + key);
    std::vector<uint64_t> result;
    for (double x : it->second.arr) result.push_back(exact_integer(x, key));
    return result;
}
std::vector<int> int_array(const std::vector<uint64_t> & values, const char * name) {
    std::vector<int> result;
    for (auto x : values) {
        if (x > uint64_t(std::numeric_limits<int>::max())) throw std::runtime_error(std::string("qwen4exp: oversized ") + name);
        result.push_back(int(x));
    }
    return result;
}
} // namespace

void PleHashConfig::validate() const {
    if (ngram < 2 || ngram > 8 || heads_per_ngram <= 0 || int64_t(ngram - 1) * heads_per_ngram > 64 ||
        eos < 0 || image < 0 || multipliers.size() != size_t(ngram))
        throw std::runtime_error("qwen4exp: invalid PLE hash configuration");
    const size_t heads = size_t(ngram - 1) * heads_per_ngram;
    if (offsets.size() != heads || vocab_sizes.size() != heads)
        throw std::runtime_error("qwen4exp: PLE head arrays have wrong length");
    for (size_t h = 0; h < heads; ++h) {
        const auto limit = uint64_t(std::numeric_limits<int32_t>::max());
        if (!vocab_sizes[h] || offsets[h] > limit || vocab_sizes[h] > limit || offsets[h] > limit - vocab_sizes[h])
            throw std::runtime_error("qwen4exp: invalid PLE row range");
    }
}

Qwen4ExpConfig qwen4exp_config(const GgufFile & f) {
    if (f.str("general.architecture") != "qwen4exp") throw std::runtime_error("qwen4exp: wrong architecture");
    Qwen4ExpConfig c;
    c.layers = scalar(f, "block_count"); c.hidden = scalar(f, "embedding_length");
    c.experts = scalar(f, "expert_count"); c.topk = scalar(f, "expert_used_count");
    c.expert_ff = scalar(f, "expert_feed_forward_length"); c.shared_ff = scalar(f, "expert_shared_feed_forward_length");
    c.hc = scalar(f, "hyper_connection.count"); c.hc_low_rank = scalar(f, "hyper_connection.low_rank");
    if (c.topk > c.experts || int64_t(c.hc) * c.hidden > std::numeric_limits<int>::max())
        throw std::runtime_error("qwen4exp: invalid expert or hyperconnection dimensions");
    if (f.has("qwen4exp.attention.compress_ratios")) {
        c.compress_ratios = int_array(array(f, "attention.compress_ratios", c.layers), "compression ratios");
        if (c.compress_ratios.size() != size_t(c.layers)) throw std::runtime_error("qwen4exp: compression array length mismatch");
    } else c.compress_ratios.assign(c.layers, 0);
    if (std::any_of(c.compress_ratios.begin(), c.compress_ratios.end(), [](int ratio) { return ratio > 64; }))
        throw std::runtime_error("qwen4exp: compression ratio exceeds QSA limit");
    if (std::any_of(c.compress_ratios.begin(), c.compress_ratios.end(), [](int ratio) { return ratio > 0; })) {
        c.index_heads = scalar(f, "attention.indexer.head_count");
        c.index_dim = scalar(f, "attention.indexer.key_length");
        c.index_topk = scalar(f, "attention.indexer.top_k");
        if (int64_t(c.index_heads) * c.index_dim > std::numeric_limits<int>::max())
            throw std::runtime_error("qwen4exp: oversized indexer query");
    }
    if (!f.has("qwen4exp.ple.layers") || f.kv.at("qwen4exp.ple.layers").arr.empty()) return c;
    c.ple_layers = int_array(array(f, "ple.layers", 1), "PLE layers");
    if (c.ple_layers.size() != 1 || c.ple_layers[0] >= c.layers)
        throw std::runtime_error("qwen4exp: only one in-range PLE layer is supported");
    c.ple_kernel = scalar(f, "ple.conv_kernel"); c.ple_dim = scalar(f, "embedding_length_per_layer_input");
    c.ple.ngram = scalar(f, "ple.ngram_size"); c.ple.heads_per_ngram = scalar(f, "ple.heads_per_ngram");
    c.ple.eos = scalar(f, "ple.eos_token_id", false);
    c.ple.image = scalar(f, "ple.image_token_id", false, 0.);
    const int64_t heads = int64_t(c.ple.ngram - 1) * c.ple.heads_per_ngram;
    if (c.ple.ngram > 8 || heads < 1 || heads > 64) throw std::runtime_error("qwen4exp: invalid PLE head count");
    c.ple.multipliers = array(f, "ple.layer_multipliers", c.ple.ngram);
    c.ple.offsets = array(f, "ple.head_offsets", heads);
    c.ple.vocab_sizes = array(f, "ple.head_vocab_sizes", heads);
    c.ple.validate();
    const auto & mask_key = f.kv.find("qwen4exp.attention.recurrent_layers");
    if (mask_key != f.kv.end()) {
        const auto mask = array(f, "attention.recurrent_layers", c.layers);
        if (mask.size() != size_t(c.layers) || mask[c.ple_layers[0]] != 1)
            throw std::runtime_error("qwen4exp: PLE requires a recurrent layer");
    } else {
        const int interval = f.has("qwen4exp.full_attention_interval") ? scalar(f, "full_attention_interval") : 4;
        if ((c.ple_layers[0] + 1) % interval == 0) throw std::runtime_error("qwen4exp: PLE requires a recurrent layer");
    }
    return c;
}

PleTokenHistory::PleTokenHistory(PleHashConfig config, size_t capacity)
    : config_(std::move(config)), capacity_(capacity) {
    config_.validate();
    if (!capacity_) throw std::runtime_error("qwen4exp: empty PLE history capacity");
}
void PleTokenHistory::rollback(size_t accepted_size) {
    if (accepted_size > tokens_.size()) throw std::runtime_error("qwen4exp: PLE rollback beyond history");
    tokens_.resize(accepted_size);
}
std::vector<int32_t> PleTokenHistory::append(int32_t token) {
    if (token < 0 || tokens_.size() >= capacity_) throw std::runtime_error("qwen4exp: invalid token or full PLE history");
    std::vector<uint64_t> context(config_.ngram, uint64_t(config_.eos));
    context[0] = uint64_t(token);
    for (int back = 1; back < config_.ngram; ++back) {
        if (size_t(back) > tokens_.size()) break;
        const auto prior = tokens_[tokens_.size() - back];
        if (prior == config_.eos) break;
        context[back] = uint64_t(prior);
    }
    std::vector<int32_t> rows(config_.offsets.size());
    uint64_t hash = context[0] * config_.multipliers[0];
    for (int length = 2; length <= config_.ngram; ++length) {
        hash ^= context[length - 1] * config_.multipliers[length - 1];
        for (int g = 0; g < config_.heads_per_ngram; ++g) {
            const int head = (length - 2) * config_.heads_per_ngram + g;
            rows[head] = int32_t(hash % config_.vocab_sizes[head] + config_.offsets[head]);
        }
    }
    tokens_.push_back(token);
    return rows;
}
std::vector<int32_t> PleTokenHistory::append_image() {
    return append(config_.image ? config_.image : config_.eos);
}
std::vector<int32_t> PleTokenHistory::append_batch(const std::vector<int32_t> & tokens) {
    if (tokens.size() > capacity_ - tokens_.size() ||
        std::any_of(tokens.begin(), tokens.end(), [](auto t) { return t < 0; }))
        throw std::runtime_error("qwen4exp: invalid PLE token batch");
    std::vector<int32_t> result;
    result.reserve(tokens.size() * config_.offsets.size());
    for (auto token : tokens) {
        const auto rows = append(token);
        result.insert(result.end(), rows.begin(), rows.end());
    }
    return result;
}

QsaSelection qsa_select_visible(const std::vector<float> & block_scores, int cells, int query_position,
                                int ratio, int top_k) {
    if (cells <= 0 || query_position < 0 || query_position >= cells || ratio <= 0 || ratio > 64 || top_k <= 0)
        throw std::runtime_error("qwen4exp: invalid QSA selection dimensions");
    const int complete = (query_position + 1) / ratio;
    if (block_scores.size() < size_t(complete)) throw std::runtime_error("qwen4exp: missing QSA block scores");
    std::vector<int32_t> blocks(complete);
    for (int b = 0; b < complete; ++b) {
        if (!std::isfinite(block_scores[b]) || block_scores[b] < 0.f)
            throw std::runtime_error("qwen4exp: invalid QSA block score");
        blocks[b] = b;
    }
    std::sort(blocks.begin(), blocks.end(), [&](int32_t a, int32_t b) {
        return block_scores[a] > block_scores[b] ||
               (block_scores[a] == block_scores[b] && a < b);
    });
    const int width = int(std::min<int64_t>(cells, int64_t(top_k) + ratio - 1));
    QsaSelection selected;
    selected.positions.reserve(width); selected.valid.reserve(width);
    auto append = [&](int32_t position, bool valid) {
        if (int(selected.positions.size()) < width) {
            selected.positions.push_back(position);
            selected.valid.push_back(valid ? 1 : 0);
        }
    };
    for (int p = complete * ratio; p <= query_position; ++p) append(p, true);
    for (int32_t block : blocks) {
        for (int p = block * ratio; p < (block + 1) * ratio; ++p) append(p, true);
    }
    for (int p = query_position + 1; int(selected.positions.size()) < width && p < cells; ++p) append(p, false);
    if (int(selected.positions.size()) != width) throw std::logic_error("qwen4exp: QSA selection width differs");
    return selected;
}

} // namespace tqllm
