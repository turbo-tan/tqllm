#pragma once
// Contextual embeddings of pplx-embed-v2-context (a Qwen3.5 text tower with bidirectional attention, chunk mean-pool,
// a linear projection): GGUF from tools/convert/pplx/pplx_context_to_gguf.py. One document (or query) is one forward
// pass in FP32 activations over FP32 copies of the BF16 weights; the LM head is never run.
#include "tqllm/embed_inputs.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tqllm {

class PplxEmbedder {
public:
    // exact: FP32 activations and products (the reference). bf16_products: exact with BF16-rounded matmul inputs and
    // weights (a test mode). fast: the BF16 forward of a BF16 Hugging Face run (BF16 residual stream, tensor-core GEMMs with
    // FP32 accumulate, the qwen35 prefill kernels, bidirectional flash attention); pooling and projection stay FP32.
    enum class Mode { exact, bf16_products, fast };
    static std::unique_ptr<PplxEmbedder> load(const std::string & gguf_path, int device = 0, Mode mode = Mode::exact);
    virtual ~PplxEmbedder() = default;
    // true when the GGUF is a contextual embedding model (has the pplx.* pooling metadata)
    static bool is_embedding_gguf(const std::string & gguf_path);

    virtual const EmbedPrompt & prompt() const = 0;
    virtual int dim() const = 0;
    virtual size_t max_tokens() const = 0;
    // one vector of dim() floats per span of `in` (the projection output, before any normalisation or int8 rounding);
    // `layers`, when given, receives the hidden state of every token after each layer (then after the final norm)
    virtual std::vector<float> embed(const EmbedInput & in, std::vector<std::vector<float>> * layers = nullptr) = 0;
    // several independent inputs (documents or queries); the fast mode runs them as segments of one forward pass
    virtual std::vector<std::vector<float>> embed_many(const std::vector<EmbedInput> & in) {
        std::vector<std::vector<float>> out;
        for (const auto & x : in) out.push_back(embed(x));
        return out;
    }
};

// TQ4 (GGUF 300) groups of 128 -> BF16 in the original basis (device pointers; the fast path's decoder, for its test)
void pplx_tq4_to_bf16(const void * w, size_t groups, void * out);

}  // namespace tqllm
