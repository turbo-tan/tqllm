// /v1/embeddings request parsing and response shape (tools/serve/embeddings.hpp)
#include "serve/embeddings.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
int failures = 0;
void check(bool ok, const std::string & what) { if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; } }
}  // namespace

int main() {
    using tqserve::parse_embed_request;
    auto a = parse_embed_request(R"({"model":"m","input":"hello"})");
    check(a.error.empty() && a.docs.size() == 1 && a.docs[0] == std::vector<std::string>{"hello"} && !a.query && !a.nested, "string input");
    auto b = parse_embed_request(R"({"input":["x","y"],"task":"query"})");
    check(b.error.empty() && b.docs.size() == 2 && b.docs[1][0] == "y" && b.query, "list of queries: one entry each");
    auto c = parse_embed_request(R"({"input":[["a","b"],["c"]],"encoding_format":"int8"})");
    check(c.error.empty() && c.nested && c.int8 && c.docs.size() == 2 && c.docs[0].size() == 2, "documents of chunks");
    for (const char * bad : {R"({"input":[["a"]],"task":"query"})", R"({"input":["a",["b"]]})", R"({"input":[]})", R"({"input":[[]]})",
                             R"({"input":""})", R"({"task":"doc","input":"a"})", R"({"input":"a","encoding_format":"binary"})", R"({})", "[1]",
                             R"({"input":"a","dimensions":"8"})"})
        check(!parse_embed_request(bad).error.empty(), std::string("rejected: ") + bad);
    const std::vector<std::vector<float>> v = {{3, 4, 0, 0, 1, 0}, {0, 0, 2}};
    const auto body = tqserve::embed_response(c, v, 3, "m", 7);
    const auto j = tqllm::parse_json(body);
    const auto & data = j.get("data")->arr;
    check(data.size() == 3, "one item per chunk");
    check(data[1].get("index")->num == 1 && data[1].get("document")->num == 0 && data[1].get("chunk")->num == 1, "item 1 = doc 0 chunk 1");
    check(data[2].get("document")->num == 1 && data[2].get("chunk")->num == 0, "item 2 = doc 1 chunk 0");
    check(data[0].get("embedding")->arr[0].num == 126 && data[0].get("embedding")->arr[2].num == 0, "int8 = round(tanh(x) * 127)");
    check(j.get("usage")->get("prompt_tokens")->num == 7, "usage");
    const auto f = tqllm::parse_json(tqserve::embed_response(a, {{3, 4, 0}}, 3, "m", 2));
    const auto & e = f.get("data")->arr[0].get("embedding")->arr;
    check(std::abs(e[0].num - 0.6) < 1e-7 && std::abs(e[1].num - 0.8) < 1e-7 && !f.get("data")->arr[0].get("document"), "float = L2-normalised, flat items");

    // limits: entries, chunks and text bytes per request, each named in the 400
    auto many = [](size_t n, const char * item) { std::string b = "{\"input\":["; for (size_t i = 0; i < n; ++i) b += (i ? "," : "") + std::string(item); return b + "]}"; };
    const auto lim_in = parse_embed_request(many(tqserve::kEmbedMaxInputs + 1, "\"x\""));
    check(lim_in.error.find("inputs") != std::string::npos && lim_in.docs.empty(), "too many inputs: 400 naming the limit (" + lim_in.error + ")");
    check(parse_embed_request(many(tqserve::kEmbedMaxInputs, "\"x\"")).error.empty(), "the input limit itself is accepted");
    std::string doc = "["; for (size_t i = 0; i <= tqserve::kEmbedMaxChunks / 2; ++i) doc += (i ? "," : "") + std::string("\"c\""); doc += "]";
    const auto lim_ch = parse_embed_request(many(2, doc.c_str()));
    check(lim_ch.error.find("chunks") != std::string::npos, "too many chunks in total: 400 naming the limit (" + lim_ch.error + ")");
    const std::string big(tqserve::kEmbedMaxBytes / 2 + 1, 'a');
    const auto lim_b = parse_embed_request("{\"input\":[\"" + big + "\",\"" + big + "\"]}");
    check(lim_b.error.find("bytes") != std::string::npos, "too much text: 400 naming the limit (" + lim_b.error + ")");

    // untrusted JSON: nesting deeper than the parser's limit is an error, not a stack overflow
    const std::string deep = std::string(200000, '[') + std::string(200000, ']');
    check(!parse_embed_request("{\"input\":" + deep + "}").error.empty(), "deeply nested body rejected");
    bool threw = false;
    try { tqllm::parse_json(deep); } catch (const std::exception &) { threw = true; }
    check(threw, "parse_json refuses deep nesting");
    check(tqllm::parse_json(std::string(64, '[') + std::string(64, ']')).kind == tqllm::Json::Array, "ordinary nesting still parses");

    // OpenAI clients: base64 (little-endian float32 of the normalised vector; the openai SDK asks for it by default),
    // dimensions only when it equals the model's
    const auto b64 = parse_embed_request(R"({"input":"hello","encoding_format":"base64"})");
    check(b64.error.empty() && b64.base64, "base64 accepted");
    const auto bj = tqllm::parse_json(tqserve::embed_response(b64, {{3, 4, 0}}, 3, "m", 2));
    const auto * emb = bj.get("data")->arr[0].get("embedding");
    check(emb && emb->kind == tqllm::Json::String, "base64 embedding is a string");
    if (emb && emb->kind == tqllm::Json::String) {
        const auto raw = tqserve::base64_decode(emb->str);
        float f3[3] = {};
        check(raw.size() == 12, "base64 holds 3 float32");
        if (raw.size() == 12) std::memcpy(f3, raw.data(), 12);
        check(std::abs(f3[0] - 0.6f) < 1e-7f && std::abs(f3[1] - 0.8f) < 1e-7f && f3[2] == 0, "base64 = normalised float32, little-endian");
    }
    check(tqserve::base64_decode("AACAPw==").size() == 4, "base64 decoder (1.0f)");
    check(parse_embed_request(R"({"input":"a","dimensions":3})").dimensions == 3, "dimensions read");
    check(!tqserve::check_dimensions(parse_embed_request(R"({"input":"a","dimensions":512})"), 1024).empty(), "dimensions != model dim: 400");
    check(tqserve::check_dimensions(parse_embed_request(R"({"input":"a","dimensions":1024})"), 1024).empty(), "dimensions == model dim: ok");

    // non-finite projections: an error, never "nan" in the JSON or 127 in int8
    bool nonfinite = false;
    try { tqserve::embed_response(a, {{std::nanf(""), 1, 0}}, 3, "m", 1); } catch (const std::runtime_error &) { nonfinite = true; }
    check(nonfinite, "NaN output refused (float)");
    nonfinite = false;
    try { tqserve::embed_response(c, {{1, std::numeric_limits<float>::infinity(), 0}}, 3, "m", 1); } catch (const std::runtime_error &) { nonfinite = true; }
    check(nonfinite, "inf output refused (int8)");
    std::cout << (failures ? "serve embeddings: FAILED\n" : "serve embeddings: OK\n");
    return failures ? 1 : 0;
}
