// Inkling-2 reasoning split: the model writes a short thinking message even with enable_thinking false and
// reasoning_effort "none" (<|message_model|><|content_thinking|>...<|end_message|><|message_model|><|content_text|>answer);
// the detokenizer drops those tokens, so without the split the thinking landed in content with nothing to cut it at.
// A recorded-shape token sequence through finish() (non-stream) and StreamWriter (stream), thinking on and off; a reply
// without a thinking message stays content; reasoning_effort "none" parses.
#include "session.hpp"
#include "tqllm/tokenizer.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
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
    // Inkling-2's message tokens: 256 end of sampling (EOS), 257 model message, 258 thinking content, 259 end of message,
    // 260 text content
    for (const char * t : {"<|content_model_end_sampling|>", "<|message_model|>", "<|content_thinking|>", "<|end_message|>", "<|content_text|>"}) {
        tokens.push_back(t); types.push_back(3);
    }
    GgufWriter w;
    w.kv_str("general.architecture", "qwen35");
    w.kv_str("tokenizer.ggml.model", "gpt2");
    w.kv_str("tokenizer.ggml.pre", "qwen35");
    w.kv_strs("tokenizer.ggml.tokens", tokens);
    w.kv_i32s("tokenizer.ggml.token_type", types);
    w.kv_strs("tokenizer.ggml.merges", {});
    w.kv_u32("tokenizer.ggml.eos_token_id", 256);
    const auto path = dir + "/mock_inkling.gguf";
    w.write(path);
    return path;
}
std::vector<uint32_t> bytes(const std::string & s) {std::vector<uint32_t> v; for (unsigned char c : s) v.push_back(c); return v;}
}  // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() / ("tqllm_inkling_split_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    tqllm::Tokenizer tok;
    tok.load(mock(dir.string()));
    auto & so = tqserve::serve_options();  // as tqserve sets them for Inkling-2
    so.think_open = "<|content_thinking|>"; so.think_close = "<|end_message|>"; so.think_off.clear(); so.keep_high = true;
    const std::string think = "The user wants exactly 3 bullet points. Need produce final only.", answer = "- one\n- two\n- three";
    std::vector<uint32_t> gen = {257, 258};
    for (auto b : bytes(think)) gen.push_back(b);
    gen.push_back(259); gen.push_back(257); gen.push_back(260);
    for (auto b : bytes(answer)) gen.push_back(b);
    gen.push_back(259); gen.push_back(256);
    std::vector<uint32_t> plain = {257, 260};
    for (auto b : bytes(answer)) plain.push_back(b);
    plain.push_back(256);
    int fail = 0;
    auto check = [&](bool c, const char * what) { if (!c) { std::printf("FAIL: %s\n", what); ++fail; } };
    auto stream = [&](const tqserve::ChatRequest & req, const std::vector<uint32_t> & g, std::string & reasoning, std::string & content) {
        tqserve::StreamWriter w(tok, req, "m", [&](const std::string & s) {
            if (s.rfind("data: {", 0) != 0) return true;
            const auto j = tqllm::parse_json(s.substr(6));
            const auto * ch = j.get("choices");
            if (!ch || ch->arr.empty()) return true;
            const auto * d = ch->arr[0].get("delta");
            if (const auto * t = d ? d->get("content") : nullptr; t && t->kind == tqllm::Json::String) content += t->str;
            if (const auto * t = d ? d->get("reasoning_content") : nullptr; t && t->kind == tqllm::Json::String) reasoning += t->str;
            return true;
        });
        std::vector<uint32_t> fed;
        for (const auto t : g) { fed.push_back(t); w.update(fed); }
        tqserve::DecodeResult fin;
        tqserve::finish(tok, req, g, fin);
        w.close(fin);
    };
    for (const bool thinking : {false, true}) {
        tqserve::ChatRequest req;
        req.chat = true;
        req.thinking = thinking;
        tqserve::DecodeResult r;
        tqserve::finish(tok, req, gen, r);
        std::printf("thinking %d: reasoning \"%s\" | content \"%s\"\n", int(thinking), r.reasoning.c_str(), r.text.c_str());
        check(r.reasoning == think, "non-stream: the thinking message goes to reasoning_content");
        check(r.text == answer, "non-stream: the answer alone is content");
        std::string sr, sc;
        stream(req, gen, sr, sc);
        check(sr == think, "stream: reasoning_content deltas = the thinking message");
        check(sc == answer, "stream: content deltas = the answer");
    }
    {   // a reply without a thinking message (thinking off): all content, nothing in reasoning
        tqserve::ChatRequest req;
        req.chat = true;
        tqserve::DecodeResult r;
        tqserve::finish(tok, req, plain, r);
        check(r.reasoning.empty() && r.text == answer, "no thinking message: content only");
        std::string sr, sc;
        stream(req, plain, sr, sc);
        check(sr.empty() && sc == answer, "stream, no thinking message: content only");
    }
    {   // reasoning_effort "none" with thinking off (Tech2Wild's TensorFold bench): accepted, passed to the template verbatim
        const auto req = tqserve::parse_request(
            R"({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"none"}})", 64, "", true, true,
            "high");
        check(req.ok && req.error.empty(), "reasoning_effort none parses");
        check(!req.thinking, "enable_thinking false is kept");
    }
    {   // thinking off, rendered with a Jinja template: the prompt opens the answer's text message (no thinking message);
        // with tools, or with thinking on, the turn is left to the model
        so.think_off = "<|message_model|><|content_text|>"; so.think_off_without_tools = true;
        const tqserve::PromptRenderer render = [](const tqllm::Json &, const tqllm::Json *, const tqllm::Json &) { return std::string("P"); };
        const auto off = tqserve::parse_request(
            R"({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"none"}})", 64, "", true, true,
            "high", &render);
        check(off.ok && off.prompt_text == "P<|message_model|><|content_text|>", "thinking off: the answer's text message is opened");
        const auto on = tqserve::parse_request(R"({"messages":[{"role":"user","content":"hi"}]})", 64, "", true, true, "high", &render);
        check(on.ok && on.prompt_text == "P", "thinking on: the model opens its turn");
        const auto tools = tqserve::parse_request(
            R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{}}}],"chat_template_kwargs":{"enable_thinking":false}})",
            64, "", true, true, "high", &render);
        check(tools.ok && tools.prompt_text == "P", "thinking off with tools: the model opens its turn (a call may come first)");
        so.think_off.clear(); so.think_off_without_tools = false;
    }
    {   // thinking off with the answer message opened (Mac + 3090, 1.8.1: "What is 17 times 23? Answer with the number
        // only."): the answer's <|end_message|> ends the reply; the model went on with a thinking message and a second
        // answer, and content read "391The user asks ... 391"
        so.think_off = "<|message_model|><|content_text|>"; so.think_off_without_tools = true; so.think_off_ends_at_close = true;
        const tqserve::PromptRenderer render = [](const tqllm::Json &, const tqllm::Json *, const tqllm::Json &) { return std::string("P"); };
        const auto req = tqserve::parse_request(
            R"({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false}})", 64, "", true, true,
            "high", &render);
        check(req.answer_open, "thinking off: the request has the answer message open");
        std::vector<uint32_t> g = bytes("391");
        g.push_back(259); g.push_back(257); g.push_back(258);
        for (auto b : bytes("The user asks for 17 times 23.")) g.push_back(b);
        g.push_back(259); g.push_back(257); g.push_back(260);
        for (auto b : bytes("391")) g.push_back(b);
        g.push_back(259); g.push_back(256);
        tqserve::DecodeResult r;
        tqserve::finish(tok, req, g, r);
        std::printf("answer open: reasoning \"%s\" | content \"%s\" | %s\n", r.reasoning.c_str(), r.text.c_str(), r.finish_reason.c_str());
        check(r.text == "391" && r.reasoning.empty() && r.finish_reason == "stop", "non-stream: the reply ends with the answer message");
        std::string sr, sc;
        stream(req, g, sr, sc);
        check(sc == "391" && sr.empty(), "stream: the reply ends with the answer message");
        tqserve::AnswerEnd end{tqserve::answer_end_id(tok, req)};
        check(!end.reached(tok, bytes("39")), "spec loop: no end before the answer's end_message");
        check(end.reached(tok, std::vector<uint32_t>(g.begin(), g.begin() + 5)), "spec loop: stops at the answer's end_message");
        auto thinks = req;  // the reply opened a thinking message after all: its end_message is the thinking's end
        tqserve::DecodeResult t;
        tqserve::finish(tok, thinks, gen, t);
        check(t.reasoning == think && t.text == answer, "answer open, reply opens a thinking message: split as before");
        tqserve::AnswerEnd end2{tqserve::answer_end_id(tok, thinks)};
        check(!end2.reached(tok, gen), "spec loop: a thinking message's end_message does not stop the reply");
        const auto on = tqserve::parse_request(R"({"messages":[{"role":"user","content":"hi"}]})", 64, "", true, true, "high", &render);
        check(!on.answer_open, "thinking on: no answer end rule");
        so.think_off.clear(); so.think_off_without_tools = false; so.think_off_ends_at_close = false;
    }
    std::filesystem::remove_all(dir);
    std::printf(fail ? "inkling2-reasoning-split: FAIL\n" : "inkling2-reasoning-split: PASS\n");
    return fail ? 1 : 0;
}
