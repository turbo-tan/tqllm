// qwen35 engine: free helpers and the factory; the engine class is in qwen35_engine.hpp.
#include "qwen35_engine.hpp"

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

} // namespace tqllm
