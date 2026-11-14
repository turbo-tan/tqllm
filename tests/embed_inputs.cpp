// embed_document_input / embed_query_input give the ids and chunk token spans of the HF reference
// (PplxContextualModel.prepare_inputs). TQLLM_PPLX_GGUF = converted model, TQLLM_PPLX_TOKENS = HF dump
// ({"docs": [{"chunks", "ids", "spans"}], "queries": [{"text", "ids"}]}).
#include "tqllm/embed_inputs.hpp"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>

namespace {
int failures = 0;
void check(bool ok, const std::string & what) {
    if (!ok && failures++ < 10) std::cerr << "FAIL: " << what << "\n";
}
std::vector<uint32_t> u32s(const tqllm::Json & a) { std::vector<uint32_t> v; for (const auto & x : a.arr) v.push_back(uint32_t(x.num)); return v; }
}  // namespace

int main() {
    const char * gguf = std::getenv("TQLLM_PPLX_GGUF");
    const char * fixture = std::getenv("TQLLM_PPLX_TOKENS");
    if (!gguf || !fixture) { std::cout << "embed inputs skipped (set TQLLM_PPLX_GGUF and TQLLM_PPLX_TOKENS)\n"; return 77; }
    tqllm::Tokenizer tok;
    tok.load(gguf);
    if (std::getenv("TQLLM_TOK_WARM")) {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t n = tok.warm_cache();
        std::cout << "piece cache warmed: " << n << " pieces in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
    }
    const tqllm::EmbedPrompt p{"[D] ", "<|chunk_sep|>", tok.special_id("[Q] ")};
    std::stringstream ss; ss << std::ifstream(fixture).rdbuf();
    const std::string text = ss.str();
    const auto j = tqllm::Parser(text).parse();
    const auto * docs0 = j.get("docs");
    if (!docs0 || docs0->arr.empty() || !docs0->arr[0].get("ids") || !docs0->arr[0].get("spans")) {
        std::cerr << "FAIL: " << fixture << " is not an HF token dump (docs need ids and spans)\n";
        return 1;
    }
    size_t nd = 0, nq = 0, nt = 0;
    double tok_s = 0;
    for (const auto & d : j.get("docs")->arr) {
        std::vector<std::string> chunks;
        for (const auto & c : d.get("chunks")->arr) chunks.push_back(c.str);
        const auto t0 = std::chrono::steady_clock::now();
        const auto in = tqllm::embed_document_input(tok, p, chunks);
        tok_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const auto ids = u32s(*d.get("ids"));
        std::vector<std::pair<uint32_t, uint32_t>> spans;
        for (const auto & s : d.get("spans")->arr) spans.emplace_back(uint32_t(s.arr[0].num), uint32_t(s.arr[1].num));
        check(in.ids == ids, "doc " + std::to_string(nd) + " ids differ (" + std::to_string(in.ids.size()) + " vs " + std::to_string(ids.size()) + ")");
        check(in.spans == spans, "doc " + std::to_string(nd) + " chunk spans differ");
        ++nd; nt += ids.size();
    }
    for (const auto & q : j.get("queries")->arr) {
        const auto in = tqllm::embed_query_input(tok, p, q.get("text")->str);
        const auto ids = u32s(*q.get("ids"));
        check(in.ids == ids, "query " + std::to_string(nq) + " ids differ");
        check(in.spans.size() == 1 && in.spans[0] == std::make_pair(0u, uint32_t(ids.size())), "query " + std::to_string(nq) + " pools every token");
        ++nq; nt += ids.size();
    }
    std::cout << "documents tokenized in " << tok_s << " s\n";
    std::cout << "embed inputs: " << nd << " docs, " << nq << " queries, " << nt << " tokens: " << (failures ? "FAILED" : "OK") << "\n";
    return failures ? 1 : 0;
}
