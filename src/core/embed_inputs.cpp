#include "tqllm/embed_inputs.hpp"

#include <stdexcept>

namespace tqllm {

// Control tokens written in the text are text here (HF split_special_tokens=True); a chunk pools every token
// whose bytes overlap it, so a token straddling the chunk edge counts for that chunk.
EmbedInput embed_document_input(const Tokenizer & tok, const EmbedPrompt & p, const std::vector<std::string> & chunks) {
    if (chunks.empty()) throw std::invalid_argument("embeddings: a document needs at least one chunk");
    std::string text = p.document_prefix;
    std::vector<std::pair<size_t, size_t>> bytes;
    for (size_t i = 0; i < chunks.size(); ++i) {
        if (chunks[i].empty()) throw std::invalid_argument("embeddings: chunk " + std::to_string(i) + " is empty");
        if (i) text += p.boundary_marker;
        bytes.emplace_back(text.size(), text.size() + chunks[i].size());
        text += chunks[i];
    }
    EmbedInput in;
    std::vector<std::pair<uint32_t, uint32_t>> off;
    in.ids = tok.encode(text, Tokenizer::Specials::user_only, &off);
    size_t t = 0;
    for (size_t c = 0; c < bytes.size(); ++c) {
        const auto [b, e] = bytes[c];
        while (t < off.size() && off[t].second <= b) ++t;
        size_t u = t;
        while (u < off.size() && off[u].first < e) ++u;
        if (u == t) throw std::invalid_argument("embeddings: chunk " + std::to_string(c) + " produced no tokens");
        in.spans.emplace_back(uint32_t(t), uint32_t(u));
    }
    return in;
}

EmbedInput embed_query_input(const Tokenizer & tok, const EmbedPrompt & p, const std::string & text) {
    if (p.query_prefix_id == ~0u) throw std::invalid_argument("embeddings: the model has no query prefix token");
    EmbedInput in;
    in.ids.push_back(p.query_prefix_id);
    const auto ids = tok.encode(text, Tokenizer::Specials::user_only);
    in.ids.insert(in.ids.end(), ids.begin(), ids.end());
    in.spans.emplace_back(0u, uint32_t(in.ids.size()));
    return in;
}

}  // namespace tqllm
