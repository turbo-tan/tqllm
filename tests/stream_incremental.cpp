// Streaming deltas: StreamWriter::update() (incremental, O(n) per stream) sends exactly the bytes update_reference()
// (finish() on the whole prefix per token, O(n^2)) sends, on fuzzed byte-level token streams fed in spec-sized steps:
// thinking on / off, tools on / off, stop strings, EOS, multi-byte UTF-8 split across tokens, tool markers. Then the
// per-token cost of a 20K-token stream vs a 2K one (linear: about flat per token).
#include "session.hpp"
#include "tqllm/tokenizer.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
struct GgufWriter {
    std::string body; uint64_t n = 0;
    template <class T> static void put(std::string & s, T v) { s.append(reinterpret_cast<const char *>(&v), sizeof(v)); }
    static void str(std::string & s, const std::string & x) { put<uint64_t>(s, x.size()); s += x; }
    void kv_str(const std::string & k, const std::string & v) { str(body, k); put<uint32_t>(body, 8); str(body, v); ++n; }
    void kv_u32(const std::string & k, uint32_t v) { str(body, k); put<uint32_t>(body, 4); put<uint32_t>(body, v); ++n; }
    void kv_strs(const std::string & k, const std::vector<std::string> & v) {
        str(body, k); put<uint32_t>(body, 9); put<uint32_t>(body, 8); put<uint64_t>(body, v.size());
        for (const auto & x : v) str(body, x);
        ++n;
    }
    void kv_i32s(const std::string & k, const std::vector<int32_t> & v) {
        str(body, k); put<uint32_t>(body, 9); put<uint32_t>(body, 5); put<uint64_t>(body, v.size());
        for (auto x : v) put<int32_t>(body, x);
        ++n;
    }
    void write(const std::string & path) const {
        std::string h = "GGUF"; put<uint32_t>(h, 3); put<uint64_t>(h, 0); put<uint64_t>(h, n);
        std::ofstream(path, std::ios::binary) << h << body;
    }
};
std::string utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += char(cp);
    else if (cp < 0x800) {s += char(0xC0 | (cp >> 6)); s += char(0x80 | (cp & 63));}
    else {s += char(0xE0 | (cp >> 12)); s += char(0x80 | ((cp >> 6) & 63)); s += char(0x80 | (cp & 63));}
    return s;
}
// byte-level vocabulary: ids 0..255 are the 256 bytes (GPT-2 byte-to-unicode pieces), then specials
std::string mock(const std::string & dir) {
    std::vector<int> bs;
    for (int b = 33; b < 127; ++b) bs.push_back(b);
    for (int b = 161; b < 173; ++b) bs.push_back(b);
    for (int b = 174; b < 256; ++b) bs.push_back(b);
    std::vector<uint32_t> code(256);
    for (int b : bs) code[b] = uint32_t(b);
    int extra = 0;
    for (int b = 0; b < 256; ++b) if (std::find(bs.begin(), bs.end(), b) == bs.end()) code[b] = 256 + extra++;
    std::vector<std::string> tokens;
    std::vector<int32_t> types;
    for (int b = 0; b < 256; ++b) {tokens.push_back(utf8(code[b])); types.push_back(1);}
    tokens.push_back("<|im_end|>"); types.push_back(3);   // 256: EOS
    tokens.push_back("</think>"); types.push_back(3);     // 257
    GgufWriter w;
    w.kv_str("general.architecture", "qwen35");
    w.kv_str("tokenizer.ggml.model", "gpt2");
    w.kv_str("tokenizer.ggml.pre", "qwen35");
    w.kv_strs("tokenizer.ggml.tokens", tokens);
    w.kv_i32s("tokenizer.ggml.token_type", types);
    w.kv_strs("tokenizer.ggml.merges", {});
    w.kv_u32("tokenizer.ggml.eos_token_id", 256);
    const auto path = dir + "/mock_stream.gguf";
    w.write(path);
    return path;
}
std::vector<uint32_t> bytes(const std::string & s) {std::vector<uint32_t> v; for (unsigned char c : s) v.push_back(c); return v;}
}  // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() / ("tqllm_stream_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    tqllm::Tokenizer tok;
    tok.load(mock(dir.string()));
    std::mt19937 rng(23);
    const std::vector<std::string> pieces = {"hello ", "world", "\n", "\n\n", "  ", "\t", "STOP", "ST", "OP", "<tool_call>", "<tool_", "call>",
                                             "<function=", "x", "{\"a\": 1}", "\xc3\xa9", "\xe6\x9d\xb1", "\xf0\x9f\x98\x8a", "end", "."};
    int failures = 0, cases = 0, final_failures = 0;
    for (int trial = 0; trial < 600; ++trial) {
        tqserve::ChatRequest req;
        req.chat = trial % 5 != 0;
        req.thinking = trial % 2 == 0;
        req.tools = trial % 3 == 0;
        if (trial % 4 == 1) req.stop = {"STOP"};
        if (trial % 4 == 2) req.stop = {"\n\n", "end."};
        std::vector<uint32_t> gen;
        const int parts = 5 + int(rng() % 60);
        for (int i = 0; i < parts; ++i) {
            const uint32_t r = rng() % 100;
            if (r < 3 && req.thinking) gen.push_back(257);
            else if (r < 4 && trial % 7 == 0) gen.push_back(256);
            else for (auto b : bytes(pieces[rng() % pieces.size()])) gen.push_back(b);
        }
        std::string out_new, out_ref;
        tqserve::StreamWriter a(tok, req, "m", [&](const std::string & s) {out_new += s; return true;});
        tqserve::StreamWriter b(tok, req, "m", [&](const std::string & s) {out_ref += s; return true;});
        std::vector<uint32_t> fed;
        for (size_t at = 0; at < gen.size();) {
            const size_t step = 1 + rng() % 16;  // a spec step accepts 1..16 tokens
            for (size_t j = 0; j < step && at < gen.size(); ++j) fed.push_back(gen[at++]);
            a.update(fed); b.update_reference(fed);
        }
        // the deltas of a whole stream (update() then close()) concatenate to finish()'s non-stream answer
        std::string content;
        tqserve::StreamWriter c(tok, req, "m", [&](const std::string & s) {
            if (s.rfind("data: {", 0) != 0) return true;
            const auto j = tqllm::parse_json(s.substr(6));
            const auto * ch = j.get("choices");
            if (!ch || ch->arr.empty()) return true;
            const auto * d = req.chat ? ch->arr[0].get("delta") : &ch->arr[0];
            if (const auto * t = d ? d->get(req.chat ? "content" : "text") : nullptr; t && t->kind == tqllm::Json::String) content += t->str;
            return true;
        });
        std::vector<uint32_t> fed2;
        for (size_t at = 0; at < gen.size();) {
            const size_t step = 1 + rng() % 16;
            for (size_t j = 0; j < step && at < gen.size(); ++j) fed2.push_back(gen[at++]);
            c.update(fed2);
        }
        tqserve::DecodeResult fin;
        tqserve::finish(tok, req, gen, fin);
        c.close(fin);
        auto esc = [](const std::string & x) { std::string o; for (char ch : x.substr(0, 160)) { if (ch == '\n') o += "\\n"; else if (ch == '\t') o += "\\t"; else o += ch; } return o; };
        if (req.chat && content != fin.text && ++final_failures <= 12)  // completions stream the reasoning as text too
            std::printf("STREAM != NON-STREAM trial %d thinking %d tools %d stops %zu\n  stream %s\n  final  %s\n", trial, int(req.thinking),
                        int(req.tools), req.stop.size(), esc(content).c_str(), esc(fin.text).c_str());
        ++cases;
        if (out_new != out_ref) {
            if (++failures <= 3) std::printf("MISMATCH trial %d thinking %d tools %d stops %zu\n  new %s\n  ref %s\n", trial, int(req.thinking), int(req.tools),
                                             req.stop.size(), out_new.substr(0, 400).c_str(), out_ref.substr(0, 400).c_str());
        }
    }
    std::printf("STREAM_INCREMENTAL %d / %d streams byte-identical to the whole-prefix reference\n", cases - failures, cases);
    std::printf("STREAM_FINAL %d / %d streams concatenate to the non-stream answer\n", cases - final_failures, cases);
    // per-token cost: linear in the stream length
    auto per_token = [&](size_t n, bool reference) {
        tqserve::ChatRequest req; req.thinking = true; req.stop = {"STOP"};
        size_t sink = 0;
        tqserve::StreamWriter w(tok, req, "m", [&](const std::string & s) {sink += s.size(); return true;});
        std::vector<uint32_t> fed;
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n; ++i) {
            fed.push_back(uint32_t('a' + i % 26));
            if (i % 6 == 5) {if (reference) w.update_reference(fed); else w.update(fed);}
        }
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / double(n);
    };
    const double new2k = per_token(2000, false), new20k = per_token(20000, false), ref2k = per_token(2000, true), ref8k = per_token(8000, true);
    std::printf("STREAM_COST us per token: incremental 2K %.2f, 20K %.2f | reference 2K %.2f, 8K %.2f\n", new2k, new20k, ref2k, ref8k);
    const bool linear = new20k < 3.0 * new2k + 1.0;
    std::filesystem::remove_all(dir);
    const bool ok = failures == 0 && final_failures == 0 && linear;
    std::puts(ok ? "STREAM_INCREMENTAL_PASS" : "STREAM_INCREMENTAL_FAIL");
    return ok ? 0 : 1;
}
