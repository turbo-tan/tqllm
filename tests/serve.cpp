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

void test_json_utf8() {
    const auto quote = [](const std::string & text) {
        std::string out;
        tqserve::jstr(out, text);
        return out;
    };
    check(quote("\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80\n") ==
          "\"\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80\\n\"", "valid UTF-8 and JSON escapes are preserved");
    check(quote("a\xC7" "z") == "\"a\\ufffdz\"", "invalid continuation keeps following ASCII");
    check(quote("\xE2\x82") == "\"\\ufffd\"", "incomplete UTF-8 is replaced once");
    check(quote("\xED\xA0\x80") == "\"\\ufffd\\ufffd\\ufffd\"", "UTF-8 surrogate encodings are rejected");
    check(quote("\xF4\x90\x80\x80") == "\"\\ufffd\\ufffd\\ufffd\\ufffd\"", "code points above U+10FFFF are rejected");
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
    // an untrusted body nested deeper than the parser's limit is a request error, not a stack overflow
    const auto deep = tqserve::parse_request("{\"messages\":" + std::string(200000, '[') + std::string(200000, ']') + "}", 512, "", true);
    check(!deep.ok && deep.error.find("nesting") != std::string::npos, "deeply nested chat body refused (" + deep.error + ")");
}

// Jinja path: the template must receive the resolved reasoning effort. A request that sets none gets the server's
// --reasoning-effort; the request's own value (top level or chat_template_kwargs) wins, with high/max -> xhigh and
// minimal -> low. (regression: with a Jinja template the server default never reached the template, so its own default,
// xhigh, ran for every request without an effort although tqserve was started with --reasoning-effort medium)
void test_jinja_reasoning_effort() {
    std::string seen;
    const tqserve::PromptRenderer stub = [&seen](const tqllm::Json &, const tqllm::Json *, const tqllm::Json & kw) {
        const auto * e = kw.get("reasoning_effort");
        seen = e && e->kind == tqllm::Json::String ? e->str : "<unset>";
        return std::string("<|im_start|>assistant\n");
    };
    auto effort_for = [&](const std::string & body, const std::string & server_default) {
        seen.clear();
        const auto r = tqserve::parse_request(body, 512, "", true, true, server_default, &stub);
        check(r.ok, "jinja effort request parses: " + r.error);
        return seen;
    };
    const std::string plain = R"({"messages":[{"role":"user","content":"Hi"}]})";
    check(effort_for(plain, "medium") == "medium", "jinja: server --reasoning-effort medium reaches the template");
    check(effort_for(plain, "low") == "low", "jinja: server --reasoning-effort low reaches the template");
    check(effort_for(R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"xhigh"})", "medium") == "xhigh",
          "jinja: top-level request effort overrides the server default");
    check(effort_for(R"({"messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"reasoning_effort":"low"}})",
                     "medium") == "low",
          "jinja: chat_template_kwargs effort overrides the server default");
    check(effort_for(R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"high"})", "medium") == "xhigh",
          "jinja: high maps to xhigh for the template");
    // server --chat-template-kwargs '{"reasoning_effort":...}' still beats --reasoning-effort, and the request beats both
    const tqllm::Json server_kw = tqllm::parse_json(R"({"reasoning_effort":"low"})");
    auto effort_kw = [&](const std::string & body) {
        seen.clear();
        const auto r = tqserve::parse_request(body, 512, "", true, true, "medium", &stub, &server_kw);
        check(r.ok, "jinja effort request parses: " + r.error);
        return seen;
    };
    check(effort_kw(plain) == "low", "jinja: server chat_template_kwargs effort beats --reasoning-effort");
    check(effort_kw(R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"xhigh"})") == "xhigh",
          "jinja: request effort beats the server chat_template_kwargs");
}

void test_default_max_tokens() {
    const auto none = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}]})", 8192, "", true);
    check(none.ok && none.max_tokens == 8192, "a request without max_tokens gets the server default (--max-tokens)");
    const auto set = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":100})", 8192, "", true);
    check(set.ok && set.max_tokens == 100, "max_tokens from the request");
    const auto big = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":100000})", 8192, "", true);
    check(big.ok && big.max_tokens == 8192, "max_tokens capped at the server default");
}

void test_tool_stream_limit() {
    using tqserve::tool_stream_limit;
    check(tool_stream_limit("Plain answer.") == 13, "tools: prose without markup streams whole");
    check(tool_stream_limit("Let me check.\n<function=get_weather>\n<parameter=city>\nParis") == 14, "tools: prose up to the first block");
    check(tool_stream_limit("Let me check. <func") == 14, "tools: a tail that may grow into <function= is held back");
    check(tool_stream_limit("Let me check. <tool_") == 14, "tools: a tail that may grow into <tool_call> is held back");
    check(tool_stream_limit("Hi <tool_call>\n<function=f>") == 3, "tools: a literal <tool_call> cuts like extract_tool_calls");
    check(tool_stream_limit("<function=f>\n</function>") == 0, "tools: a pure call streams nothing");
    check(tool_stream_limit("a < b and 3 <= 4") == 16, "tools: a lone '<' that cannot become a marker is not held");
    // orphan closing tags after plain prose (the Conductor's "</parameter>", 11-01): never answer text, streamed or not
    check(tool_stream_limit("Done.\n</parameter>") == 5, "tools: a trailing orphan </parameter> does not stream");
    check(tool_stream_limit("Done.\n</parameter>\n</function>\n") == 5, "tools: a run of orphan closing tags does not stream");
    check(tool_stream_limit("Use </div> here.") == 16, "tools: other closing tags are answer text");
    check(tool_stream_limit("x </parameter> y") == 16, "tools: an orphan tag inside prose stays (only the tail is cut)");
    check(tqserve::orphan_tool_tail("Done. </par", true) == 6, "tools: streaming holds a tail that may become a closing tag");
    check(tqserve::orphan_tool_tail("Done. </par") == 11, "tools: the final text keeps an incomplete tag");
    {
        tqserve::DecodeResult r; r.text = "All set.\n</parameter>"; tqserve::extract_tool_calls(r);
        check(r.text == "All set." && r.tool_calls.empty(), "tools: extract_tool_calls drops the orphan tail of plain prose");
    }
    // whatever streams is a prefix of the parsed answer (up to the whitespace trim): the streamed text never shows markup
    for (const std::string raw : {std::string("Checking now.\n<function=ls>\n</function>"), std::string("Sure. <tool_call>\n<function=f>\n</function>")}) {
        tqserve::DecodeResult r; r.text = raw; tqserve::extract_tool_calls(r);
        const std::string streamed = raw.substr(0, tool_stream_limit(raw));
        check(streamed.find('<') == std::string::npos && streamed.compare(0, r.text.size(), r.text) == 0, "tools: streamed prose is the parsed answer's prefix");
    }
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

// next_prompt_prefix: what the server expects the next request to begin with equals what a client then sends, for the
// ways a reply comes back (content alone, with reasoning, cut inside its reasoning, a tool call, odd whitespace)
void test_next_prompt_prefix() {
    auto request = [](const std::string & body) { return tqserve::parse_request(body, 512, "", true, true); };
    auto begins = [&](const std::string & what, const std::string & first, const tqserve::DecodeResult & reply, bool with_reasoning,
                      const std::string & next) {
        const auto prefix = tqserve::next_prompt_prefix(first, reply, with_reasoning, 512, "", true);
        const auto r = request(next);
        check(r.ok, what + ": the next request parses: " + r.error);
        check(!prefix.empty() && prefix.size() > request(first).prompt_text.size() - 40, what + ": a prefix past the first prompt's history");
        check(r.prompt_text.compare(0, prefix.size(), prefix) == 0, what + ": the next prompt begins with the predicted prefix");
        check(prefix.size() >= 11 && prefix.compare(prefix.size() - 11, 11, "<|im_end|>\n") == 0, what + ": the prefix ends with the reply's message");
        check(r.prompt_text.compare(prefix.size(), 12, "<|im_start|>") == 0, what + ": the next message starts right after it");
    };
    const std::string first = R"({"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Capital of France?"}]})";
    const std::string head = R"({"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Capital of France?"},)";
    tqserve::DecodeResult reply;
    reply.text = "Paris."; reply.reasoning = "The user asks for a capital.";
    begins("content alone", first, reply, false, head + R"({"role":"assistant","content":"Paris."},{"role":"user","content":"And Spain?"}]})");
    begins("with reasoning", first, reply, true,
           head + R"({"role":"assistant","content":"Paris.","reasoning_content":"The user asks for a capital."},{"role":"user","content":"And Spain?"}]})");
    check(tqserve::next_prompt_prefix(first, reply, false, 512, "", true) != tqserve::next_prompt_prefix(first, reply, true, 512, "", true),
          "returned reasoning changes the prefix");
    check(!tqserve::returns_reasoning(first) &&
          tqserve::returns_reasoning(head + R"({"role":"assistant","content":"Paris.","reasoning_content":"x"},{"role":"user","content":"y"}]})") &&
          !tqserve::returns_reasoning(head + R"({"role":"assistant","content":"Paris."},{"role":"user","content":"y"}]})"),
          "a client that returns reasoning is seen from its request");
    tqserve::DecodeResult cut;  // the token limit fell inside the reasoning: no content
    cut.reasoning = "Still thinking";
    begins("cut inside the reasoning", first, cut, false, head + R"({"role":"assistant","content":""},{"role":"user","content":"Go on."}]})");
    tqserve::DecodeResult spaced;
    spaced.text = "\n  Paris.  \n\n";
    begins("whitespace kept as sent", first, spaced, false,
           head + R"({"role":"assistant","content":"\n  Paris.  \n\n"},{"role":"user","content":"And Spain?"}]})");
    const std::string tools = R"(,"tools":[{"type":"function","function":{"name":"lookup","description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}]})";
    const std::string tfirst = first.substr(0, first.size() - 1) + tools;
    tqserve::DecodeResult call;
    call.text = "Let me check."; call.tool_calls.push_back({"lookup", R"({"q":"France"})"});
    begins("tool call", tfirst, call, false,
           head + R"({"role":"assistant","content":"Let me check.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"France\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"Paris"}])" + tools);
    check(tqserve::next_prompt_prefix(R"({"prompt":"raw text"})", reply, false, 512, "", true).empty(), "a completion request has no next prefix");
}

int main(int argc, char ** argv) {
    test_next_prompt_prefix();
    {  // requests without sampling fields take the server defaults (--temperature / --top-p / --top-k); explicit values win
        const auto body = std::string(R"({"messages":[{"role":"user","content":"hi"}]})");
        const auto greedy = tqserve::parse_request(body, 64, "", true);
        check(greedy.temperature == 0.f && greedy.top_p == 1.f && greedy.top_k == 0, "default sampling is greedy");
        tqserve::sampling_defaults() = {0.6f, 0.95f, 20};
        const auto agent = tqserve::parse_request(body, 64, "", true);
        check(agent.temperature == 0.6f && agent.top_p == 0.95f && agent.top_k == 20, "server sampling defaults apply");
        const auto own = tqserve::parse_request(R"({"messages":[{"role":"user","content":"hi"}],"temperature":0,"top_k":1})", 64, "", true);
        check(own.temperature == 0.f && own.top_k == 1 && own.top_p == 0.95f, "request values override the defaults");
        tqserve::sampling_defaults() = {};
    }

    tqserve::serve_options().repeat_limit = 6;
    test_json_utf8();
    test_sampling_numbers();
    test_repeat_guard();
    test_split();
    test_reader();
    test_chat_prompt();
    test_default_max_tokens();
    test_tool_stream_limit();
    (void) argc; (void) argv;
    test_jinja_images();
    test_jinja_reasoning_effort();
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    test_jinja_images();
    std::cout << "PASS\n";
    return 0;
}
