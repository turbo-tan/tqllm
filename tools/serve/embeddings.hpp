#pragma once
// POST /v1/embeddings for contextual embedding models (no CUDA, no engine).
// OpenAI shape: "input" is a string or a list of strings, each one chunk (its own document, or a query).
// Extension: "input" a list of documents, each a list of chunks -> one vector per chunk, in order; every item also
// carries "document" and "chunk". "task": "document" (default) | "query" (a query is one string).
// "encoding_format": "float" (default, L2-normalised) | "base64" (the same floats as little-endian float32, what the
// openai SDK asks for by default) | "int8" (round(tanh(x) * 127), the model's encode()). "dimensions" must equal the
// model's (no truncation). Limits per request: kEmbedMaxInputs entries, kEmbedMaxChunks chunks, kEmbedMaxBytes of text.
#include "serve/json_out.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqserve {

inline constexpr size_t kEmbedMaxInputs = 4096;            // entries (documents or queries) per request
inline constexpr size_t kEmbedMaxChunks = 32768;            // chunks over all documents of a request
inline constexpr size_t kEmbedMaxBytes = size_t(16) << 20;  // text bytes per request, before tokenizing
inline constexpr size_t kEmbedMaxTokens = size_t(1) << 20;  // tokens per request, after tokenizing (checked by the server)

struct EmbedRequest {
    std::vector<std::vector<std::string>> docs;  // one entry per forward pass; a query is a one-chunk entry
    bool query = false, nested = false, int8 = false, base64 = false;
    long dimensions = 0;  // 0 = not given
    std::string error;
};

inline EmbedRequest parse_embed_request(const std::string & body) {
    EmbedRequest r;
    tqllm::Json in;
    try { in = tqllm::parse_json(body); } catch (const std::exception & e) { r.error = e.what(); return r; }
    if (in.kind != tqllm::Json::Object) { r.error = "the request body must be a JSON object"; return r; }
    if (const auto * t = in.get("task")) {
        if (t->kind != tqllm::Json::String || (t->str != "query" && t->str != "document")) { r.error = "\"task\" must be \"query\" or \"document\""; return r; }
        r.query = t->str == "query";
    }
    if (const auto * f = in.get("encoding_format")) {
        if (f->kind != tqllm::Json::String || (f->str != "float" && f->str != "base64" && f->str != "int8")) {
            r.error = "\"encoding_format\" must be \"float\", \"base64\" or \"int8\""; return r;
        }
        r.int8 = f->str == "int8";
        r.base64 = f->str == "base64";
    }
    if (const auto * dm = in.get("dimensions")) {
        if (dm->kind != tqllm::Json::Number || dm->num < 1 || dm->num != std::floor(dm->num)) { r.error = "\"dimensions\" must be a positive integer"; return r; }
        r.dimensions = long(dm->num);
    }
    const auto * x = in.get("input");
    if (!x) { r.error = "missing \"input\""; return r; }
    size_t bytes = 0, chunks_total = 0;
    auto text = [&](const tqllm::Json & v, std::string & out) {
        if (v.kind != tqllm::Json::String) { r.error = "\"input\" items must be strings"; return false; }
        if (v.str.empty()) { r.error = "\"input\" has an empty string"; return false; }
        if ((bytes += v.str.size()) > kEmbedMaxBytes) { r.error = "\"input\" has more than " + std::to_string(kEmbedMaxBytes) + " bytes of text (limit per request)"; return false; }
        if (++chunks_total > kEmbedMaxChunks) { r.error = "\"input\" has more than " + std::to_string(kEmbedMaxChunks) + " chunks (limit per request)"; return false; }
        out = v.str;
        return true;
    };
    if (x->kind == tqllm::Json::String) {
        r.docs.push_back({{}});
        if (!text(*x, r.docs[0][0])) r.docs.clear();
        return r;
    }
    if (x->kind != tqllm::Json::Array || x->arr.empty()) { r.error = "\"input\" must be a string or a non-empty list"; return r; }
    r.nested = x->arr[0].kind == tqllm::Json::Array;
    if (r.nested && r.query) { r.error = "a query is one string: send \"input\" as a string or a list of strings"; return r; }
    if (x->arr.size() > kEmbedMaxInputs) { r.error = "\"input\" has more than " + std::to_string(kEmbedMaxInputs) + " inputs (limit per request)"; return r; }
    for (const auto & d : x->arr) {
        if ((d.kind == tqllm::Json::Array) != r.nested) { r.error = "\"input\" mixes strings and lists"; r.docs.clear(); return r; }
        std::vector<std::string> chunks;
        for (const auto & c : r.nested ? d.arr : std::vector<tqllm::Json>{d}) {
            chunks.emplace_back();
            if (!text(c, chunks.back())) { r.docs.clear(); return r; }
        }
        if (chunks.empty()) { r.error = "a document has no chunks"; r.docs.clear(); return r; }
        r.docs.push_back(std::move(chunks));
    }
    return r;
}

// "" when the request's dimensions (if any) is the model's; the 400 text otherwise
inline std::string check_dimensions(const EmbedRequest & r, int dim) {
    if (r.dimensions == 0 || r.dimensions == dim) return {};
    return "\"dimensions\": this model returns " + std::to_string(dim) + "-dimensional vectors (no truncation)";
}

inline void base64_append(std::string & out, const uint8_t * p, size_t n) {
    static const char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t a = p[i], b = i + 1 < n ? p[i + 1] : 0, c = i + 2 < n ? p[i + 2] : 0, t = a << 16 | b << 8 | c;
        out += kB64[t >> 18 & 63];
        out += kB64[t >> 12 & 63];
        out += i + 1 < n ? kB64[t >> 6 & 63] : '=';
        out += i + 2 < n ? kB64[t & 63] : '=';
    }
}

inline std::string base64_decode(const std::string & s) {
    std::string out;
    uint32_t acc = 0;
    int bits = 0;
    for (char ch : s) {
        int v = ch >= 'A' && ch <= 'Z' ? ch - 'A' : ch >= 'a' && ch <= 'z' ? ch - 'a' + 26 : ch >= '0' && ch <= '9' ? ch - '0' + 52
              : ch == '+' ? 62 : ch == '/' ? 63 : -1;
        if (v < 0) continue;
        acc = acc << 6 | uint32_t(v);
        if ((bits += 6) >= 8) { bits -= 8; out += char(acc >> bits & 0xff); }
    }
    return out;
}

// vectors[d]: the raw projections of entry d's chunks, dim floats each. A non-finite projection throws
// std::runtime_error (the server answers 500): JSON has no NaN / inf, and int8 would turn NaN into 127.
inline std::string embed_response(const EmbedRequest & r, const std::vector<std::vector<float>> & vectors, int dim,
                                  const std::string & model, size_t tokens) {
    std::string out = "{\"object\":\"list\",\"data\":[";
    size_t total = 0;
    for (const auto & v : vectors) total += v.size();
    for (const auto & v : vectors)
        for (float x : v)
            if (!std::isfinite(x)) throw std::runtime_error("the model produced a non-finite value (NaN or inf); no vectors returned");
    out.reserve(total * (r.int8 ? 5 : r.base64 ? 6 : 16) + vectors.size() * 96 + 128);
    size_t index = 0;
    char buf[32];
    for (size_t d = 0; d < vectors.size(); ++d) {
        for (size_t c = 0; c * dim < vectors[d].size(); ++c, ++index) {
            const float * v = vectors[d].data() + c * dim;
            if (index) out += ',';
            out += "{\"object\":\"embedding\",\"index\":" + std::to_string(index);
            if (r.nested) out += ",\"document\":" + std::to_string(d) + ",\"chunk\":" + std::to_string(c);
            double norm = 0;
            for (int i = 0; i < dim; ++i) norm += double(v[i]) * v[i];
            const double inv = norm > 0 ? 1 / std::sqrt(norm) : 0;
            if (r.base64) {
                std::vector<uint8_t> raw(size_t(dim) * 4);
                for (int i = 0; i < dim; ++i) {
                    const float f = float(double(v[i]) * inv);
                    uint32_t u;
                    std::memcpy(&u, &f, 4);
                    for (int b = 0; b < 4; ++b) raw[size_t(i) * 4 + b] = uint8_t(u >> (8 * b));  // little-endian
                }
                out += ",\"embedding\":\"";
                base64_append(out, raw.data(), raw.size());
                out += "\"}";
                continue;
            }
            out += ",\"embedding\":[";
            for (int i = 0; i < dim; ++i) {
                if (i) out += ',';
                // std::to_chars: shortest round-trip text, no locale (snprintf / to_string were ~40 ms per 1.5M values)
                const auto res = r.int8 ? std::to_chars(buf, buf + sizeof buf, int(std::lround(std::fmax(-128.0, std::fmin(127.0, std::tanh(double(v[i])) * 127)))))
                                        : std::to_chars(buf, buf + sizeof buf, float(double(v[i]) * inv));
                out.append(buf, res.ptr);
            }
            out += "]}";
        }
    }
    out += "],\"model\":";
    jstr(out, model);
    out += ",\"usage\":{\"prompt_tokens\":" + std::to_string(tokens) + ",\"total_tokens\":" + std::to_string(tokens) + "}}";
    return out;
}

}  // namespace tqserve
