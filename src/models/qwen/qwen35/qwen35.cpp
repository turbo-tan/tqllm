// qwen35 engine: free helpers and the factory; the engine class is in qwen35_engine.hpp.
#include "qwen35_engine.hpp"

#include <cstdio>
#include <filesystem>

namespace tqllm {

// The grouped-expert guard used by load_moe: a loaded expert matrix must have a format with a
// grouped-expert projection. The forward path calls the format's own expert_matvec / expert_gemm,
// which handle any basis, so the basis is not checked here. (The old guard also required
// Basis::Plain, which rejected our own TQ3_4S MoE files.)
void check_grouped_expert_format(const Mat & m, const std::string & prefix) {
    if (m.p && !m.fmt->has_expert_matvec())
        throw std::runtime_error(prefix + "unsupported grouped expert format: " + m.fmt->name());
}

MixStaging mix_staging(Basis basis, bool bf16_ba, bool quantized_model) {
    if (!quantized_model) return {};
    return {basis != Basis::Tq3Rot32 || bf16_ba, bf16_ba && bf16_ba_unrotated(basis)};
}

using qwen35_detail::Qwen35Engine;

std::unique_ptr<Engine> make_qwen35_engine(const std::string & path, const std::string & tokenizer, uint32_t context,
                                           int slots, const EngineOptions & options) {
    return std::make_unique<Qwen35Engine>(path, tokenizer, context, slots, options);
}

std::unique_ptr<Engine> make_qwen35_split_engine(const std::string & path, const std::string & tokenizer, uint32_t context,
                                                 int slots, const EngineOptions & options, std::shared_ptr<TpComm> tp) {
    return std::make_unique<Qwen35Engine>(path, tokenizer, context, slots, options, std::move(tp));
}

void qwen35_split_costs(const std::string & path, uint32_t context, int slots, const EngineOptions & o, std::vector<uint64_t> & cost,
                        uint64_t & fixed) {
    const GgufFile f = gguf_open(path);
    const qwen35_detail::Config c = qwen35_detail::config_from_gguf(f);
    cost.assign(c.layers, 0);
    fixed = 0;
    for (const auto & [name, t] : f.tensors) {
        unsigned l = 0; int used = 0;
        if (std::sscanf(name.c_str(), "blk.%u.%n", &l, &used) == 1 && used > 0 && l < c.layers) cost[l] += t.bytes;
        else fixed += t.bytes;
    }
    // K/V of a full-attention layer (BF16 unless -ctk / --kv: an upper bound) and the recurrent state of a Gated DeltaNet one
    const uint64_t kv = uint64_t(context) * uint64_t(std::max(slots, 1)) * 2 * c.kvheads * c.head_dim * 2;
    const uint64_t rec = uint64_t(std::max(slots, 1)) * 2 * (uint64_t(c.lin_vheads) * c.lin_dk * c.lin_dv * 4);
    for (uint32_t l = 0; l < c.layers; ++l) cost[l] += c.full[l] ? kv : rec;
    std::error_code ec;
    if (!o.draft_model.empty()) fixed += std::filesystem::file_size(o.draft_model, ec);
    fixed += uint64_t(1536) << 20;  // activations, prefill and cuBLAS buffers
}

} // namespace tqllm
