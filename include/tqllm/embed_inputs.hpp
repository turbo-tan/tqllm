#pragma once
// Token input of a contextual embedding model (pplx-embed-v2-context): a document is one sequence
// prefix + chunk0 + marker + chunk1 + ..., with one token span per chunk; a query is prefix_id + its tokens.
#include "tqllm/tokenizer.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tqllm {

struct EmbedInput {
    std::vector<uint32_t> ids;
    std::vector<std::pair<uint32_t, uint32_t>> spans;  // token [begin, end) pooled for each output vector
};

struct EmbedPrompt {
    std::string document_prefix, boundary_marker;
    uint32_t query_prefix_id = ~0u;
};

EmbedInput embed_document_input(const Tokenizer & tok, const EmbedPrompt & p, const std::vector<std::string> & chunks);
EmbedInput embed_query_input(const Tokenizer & tok, const EmbedPrompt & p, const std::string & text);

}  // namespace tqllm
