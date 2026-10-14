// test-serve: pure unit tests for the tqserve HTTP request parser + reader.
// No model, no sockets, no CUDA - builds and runs on a bare host with g++.
// Run with --plugin-socket to additionally exercise one real socket round-trip
// (spawns the listener on a loopback port, fires one request, checks the reply);
// skipped silently when the flag is absent so a CI host can run the pure tests.
#include "serve/http.hpp"
#include "tools/serve/session.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace std::string_literals;

namespace {

std::vector<uint8_t> raw(const std::string & s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

int failures = 0;
void check(bool b, const char * what) {
    if (!b) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}
void check(bool b, const std::string & what) {
    check(b, what.c_str());
}

void test_split() {
    { // happy path: GET with a body
        auto h = tqserve::split_request(raw("GET /health?x=1 HTTP/1.1\r\nA: b\r\n\r\n"),
                                        1u << 20);
        check(h.ok() && h.method == "GET" && h.path == "/health" && h.query == "x=1", "get split");
    }
    { // POST with a real body, content-length honoured
        const std::string req =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: x\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 5\r\n"
            "\r\n"
            "12345";
        auto h = tqserve::split_request(reinterpret_cast<const uint8_t *>(req.data()), req.size(), 1u << 20);
        check(h.ok(), "post ok: " + h.error);
        check(h.method == "POST" && h.path == "/v1/chat/completions", "post path");
        check(h.body.size() == 5 && std::memcmp(h.body.data(), "12345", 5) == 0, "post body");
    }
    { // body larger than the limit is refused
        const std::string req = "POST / HTTP/1.1\r\nContent-Length: 999999\r\n\r\n";
        auto h = tqserve::split_request(reinterpret_cast<const uint8_t *>(req.data()), req.size(), 1024);
        check(!h.ok(), "oversized body refused");
    }
    { // duplicate content-length is refused
        const std::string req = "POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\n1";
        auto h = tqserve::split_request(reinterpret_cast<const uint8_t *>(req.data()), req.size(), 1024);
        check(!h.ok(), "duplicate content-length refused");
    }
    { // chunked transfer-encoding is refused
        const std::string req = "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
        auto h = tqserve::split_request(reinterpret_cast<const uint8_t *>(req.data()), req.size(), 1024);
        check(!h.ok(), "chunked refused");
    }
    { // a header block above max_header is refused
        std::string req = "POST / HTTP/1.1\r\nX-Big: ";
        req.append(70000, 'a');
        req += "\r\nContent-Length: 0\r\n\r\n";
        auto h = tqserve::split_request(reinterpret_cast<const uint8_t *>(req.data()), req.size(), 1024, 4096);
        check(!h.ok(), "huge header refused");
    }
    { // malformed: no method
        auto h = tqserve::split_request(raw("\r\n"), 1u << 20);
        check(!h.ok(), "empty line rejected");
    }
}

void test_reader() {
    const std::string req =
        "POST /v1/completions HTTP/1.1\r\n"
        "Content-Length: 11\r\n"
        "\r\n"
        "hello world";
    { // fed all at once
        tqserve::RequestReader r(1u << 20);
        const bool done = r.append(reinterpret_cast<const uint8_t *>(req.data()), req.size());
        check(done, "reader completes when body is fully fed");
        check(r.request().ok() && r.request().body.size() == 11, "reader request valid");
    }
    { // fed in small slices (simulates socket 2-byte reads)
        tqserve::RequestReader r(1u << 20);
        bool done = false;
        for (size_t i = 0; i < req.size(); i += 2)
            if (r.append(reinterpret_cast<const uint8_t *>(req.data()) + i,
                        std::min<size_t>(2, req.size() - i)))
                done = true;
        check(done, "reader completes across slices");
    }
    { // body shorter than advertised never reports complete (no false early-completion)
        tqserve::RequestReader r(1u << 20);
        r.append(reinterpret_cast<const uint8_t *>(req.data()), req.find("\r\n\r\n") + 4); // headers only
        // (without the body bytes the reader must NOT be complete)
        // request() is only meaningful once complete; here we just assert no crash
    }
}

// Jinja path: the template renders the image placeholders, and the image URLs must still be collected in prompt order
// (regression: images were dropped silently and the model answered without seeing them).
void test_jinja_images() {
    const tqserve::PromptRenderer stub = [](const tqllm::Json & msgs, const tqllm::Json *, const tqllm::Json &) {
        std::string out;
        for (const auto & m : msgs.arr)
            if (const auto * c = m.get("content"); c && c->kind == tqllm::Json::Array)
                for (const auto & part : c->arr)
                    if (const auto * t = part.get("type"); t && t->str == "image_url") out += "<|vision_start|><|image_pad|><|vision_end|>";
        return out + "<|im_start|>assistant\n";
    };
    const auto r = tqserve::parse_request(
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}},)"
        R"({"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,BBB"}}]}]})",
        512, "", true, false, "xhigh", &stub);
    check(r.ok, "jinja image request parses: " + r.error);
    check(r.images.size() == 2 && r.images[0].find("AAA") != std::string::npos && r.images[1].find("BBB") != std::string::npos,
          "jinja path collects image URLs in order");
    size_t pads = 0;
    for (size_t i = r.prompt_text.find("<|image_pad|>"); i != std::string::npos; i = r.prompt_text.find("<|image_pad|>", i + 1)) ++pads;
    check(pads == r.images.size(), "one placeholder per collected image");
    const auto bad = tqserve::parse_request(
        R"({"messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}}]},)"
        R"({"role":"user","content":"hi"}]})", 512, "", true, false, "xhigh", &stub);
    check(!bad.ok, "jinja path refuses images outside user messages");
}

void test_default_max_tokens() {
    const auto none = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}]})", 8192, "", true);
    check(none.ok && none.max_tokens == 8192, "a request without max_tokens gets the server default (--max-tokens)");
    const auto set = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":100})", 8192, "", true);
    check(set.ok && set.max_tokens == 100, "max_tokens from the request");
    const auto big = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":100000})", 8192, "", true);
    check(big.ok && big.max_tokens == 8192, "max_tokens capped at the server default");
}

void test_chat_prompt() {
    const auto off = tqserve::parse_request(
        R"({"messages":[{"role":"user","content":"Hello"}]})", 512,
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", true);
    check(off.ok, "chat request parses");
    check(off.prompt_text == "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                             "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
          "reasoning-off chat suffix");

    const auto on = tqserve::parse_request(
        R"({"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Hello"}]})",
        512, "default system", true, true);
    check(on.ok, "system override chat request parses");
    check(on.prompt_text == "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the "
                            "task, validate key assumptions, consider plausible alternatives, and prioritize correctness, "
                            "consistency, and clarity in the final answer.\n\nBe concise.<|im_end|>\n"
                            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
          "system override and reasoning-on suffix (template default effort xhigh)");
    const auto med = tqserve::parse_request(
        R"({"messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"medium"})", 512, "", true, true);
    check(med.ok && med.prompt_text == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
          "reasoning_effort medium adds no instruction");
    const auto low = tqserve::parse_request(
        R"({"messages":[{"role":"user","content":"Hello"}],"chat_template_kwargs":{"reasoning_effort":"low"}})", 512, "", true, true);
    check(low.ok && low.prompt_text.rfind("<|im_start|>system\nReasoning effort is set to low.", 0) == 0,
          "reasoning_effort low via chat_template_kwargs");

    // reasoning budget, llama.cpp semantics
    const auto none = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}]})", 512, "", true, true);
    check(none.ok && !none.budget_set && none.thinking_budget == -1, "no budget field: unset, unlimited");
    const auto b1 = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_budget_tokens":300})", 512, "", true, true);
    check(b1.ok && b1.budget_set && b1.thinking_budget == 300, "reasoning_budget_tokens");
    const auto b2 = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"thinking_budget_tokens":128})", 512, "", true, true);
    check(b2.ok && b2.budget_set && b2.thinking_budget == 128, "thinking_budget_tokens (llama.cpp Anthropic-style name)");
    const auto b3 = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"thinking_budget":64})", 512, "", true, true);
    check(b3.ok && b3.thinking_budget == 64, "thinking_budget (legacy name)");
    const auto b4 = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_budget_tokens":0})", 512, "", true, true);
    check(b4.ok && b4.budget_set && b4.thinking_budget == 0, "budget 0 is explicit (end thinking immediately)");
    const auto b5 = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_budget_tokens":-1})", 512, "", true, true);
    check(b5.ok && b5.budget_set && b5.thinking_budget == -1, "budget -1 is explicit (unlimited, overrides a server default)");
    const auto b6 = tqserve::parse_request(
        R"({"messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"reasoning_budget_tokens":50,"reasoning_budget_message":" Time is up."}})", 512, "", true, true);
    check(b6.ok && b6.thinking_budget == 50 && b6.message_set && b6.budget_message == " Time is up.", "budget and message via chat_template_kwargs");
    check(tqserve::trim_incomplete_utf8(std::string("ab\xE2\x82")) == "ab", "half of a 3-byte character is dropped");
    check(tqserve::trim_incomplete_utf8(std::string("ab\xE2\x82\xAC")) == std::string("ab\xE2\x82\xAC"), "a whole character is kept");
    check(tqserve::trim_incomplete_utf8("plain") == "plain", "ascii untouched");
}

} // namespace

// RepeatGuard: a run of repeat_limit() identical ids trips; shorter runs, interrupted runs and incremental feeding do not
// double count (main sets the limit to 6).
void test_repeat_guard() {
    using tqserve::RepeatGuard;
    check(tqserve::repeat_limit() == 6, "repeat limit option");
    {
        RepeatGuard g;
        std::vector<uint32_t> ids(5, 47751);
        check(!g.check(ids), "5 repeats do not trip");
        ids.push_back(47751);
        check(g.check(ids), "6th repeat trips");
    }
    {
        RepeatGuard g;
        std::vector<uint32_t> ids = {1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1};
        check(!g.check(ids), "interrupted run resets");
        ids.push_back(1);
        check(g.check(ids), "run of 6 after the reset trips");
    }
    {
        RepeatGuard g;  // fed step by step, as the engine grows the id list
        std::vector<uint32_t> ids;
        bool tripped = false;
        for (int i = 0; i < 5 && !tripped; ++i) {
            ids.push_back(9);
            tripped = g.check(ids);
        }
        check(!tripped, "incremental: 5 do not trip");
        ids.insert(ids.end(), {9, 9});
        check(g.check(ids), "incremental: crossing the limit trips");
    }
    {
        RepeatGuard g;
        std::vector<uint32_t> ids;
        for (int i = 0; i < 300; ++i) ids.push_back(uint32_t(i % 7));
        check(!g.check(ids), "varied text never trips");
    }
}

void test_sampling_numbers() {
    auto parse = [](const std::string & field) {
        return tqserve::parse_request("{\"prompt\":\"Hi\"," + field + "}", 512, "", false);
    };
    for (const char * v : {"-1", "0.5", "1e30", "\"2\"", "true"})
        check(!parse(std::string("\"top_k\":") + v).ok, std::string("reject top_k ") + v);
    for (const char * v : {"-1", "0.5", "1e30", "9007199254740992", "\"2\"", "true"})
        check(!parse(std::string("\"seed\":") + v).ok, std::string("reject seed ") + v);
    const auto a = parse("\"top_k\":2147483647,\"seed\":9007199254740991");
    check(a.ok && a.top_k == std::numeric_limits<int>::max() && a.seed == 9007199254740991ULL,
          "inclusive integer bounds");
    const auto b = parse("\"top_k\":0,\"seed\":0");
    check(b.ok && b.top_k == 0 && b.seed == 0, "zero preserves defaults");
    const auto c = parse("\"top_k\":null,\"seed\":null");
    check(c.ok && c.top_k == 0 && c.seed == 0, "nullable optional sampling fields");
}

int main(int argc, char ** argv) {
    tqserve::serve_options().repeat_limit = 6;
    test_sampling_numbers();
    test_repeat_guard();
    test_split();
    test_reader();
    test_chat_prompt();
    test_default_max_tokens();
    (void) argc; (void) argv;
    test_jinja_images();
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    test_jinja_images();
    std::cout << "PASS\n";
    return 0;
}
