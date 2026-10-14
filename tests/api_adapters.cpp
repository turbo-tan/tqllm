// /v1/responses and /v1/messages adapters: request -> chat-completions body, chat answer / chat SSE -> the other API.
#include "tools/serve/api_adapters.hpp"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

using tqllm::Json;
static const Json * at(const Json & j, const char * k) { return j.kind == Json::Object ? j.get(k) : nullptr; }
static std::string s(const Json * j) { return j && j->kind == Json::String ? j->str : std::string("<none>"); }
static bool has(const std::string & hay, const std::string & needle) { return hay.find(needle) != std::string::npos; }

// what StreamWriter sends for an answer: role + content deltas, then the final chunk with tool calls / finish / usage
static std::vector<std::string> chat_chunks(bool tool) {
    std::vector<std::string> v = {
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"think\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n"};
    if (tool)
        v.push_back("data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\","
                    "\"function\":{\"name\":\"exec_command\",\"arguments\":\"{\\\"cmd\\\": \\\"python3 hello.py\\\"}\"}}]},"
                    "\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}\n\n");
    else
        v.push_back("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],"
                    "\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,\"total_tokens\":15}}\n\n");
    v.push_back("data: [DONE]\n\n");
    return v;
}

static void test_paths() {
    CHECK(tqserve::api_for_path("/v1/responses") == tqserve::Api::Responses);
    CHECK(tqserve::api_for_path("/v1/messages") == tqserve::Api::Messages);
    CHECK(tqserve::api_for_path("/v1/chat/completions") == tqserve::Api::Chat);
    CHECK(tqserve::api_for_path("/v1/nope") == tqserve::Api::None);
}

static void test_responses_request() {
    // a Codex-shaped request: developer + user messages, a tool call and its output, function + unsupported tools
    const std::string body = R"({"model":"tqllm","stream":true,"instructions":"be brief",
      "input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"rules"}]},
               {"type":"message","role":"user","content":[{"type":"input_text","text":"make hello.py"}]},
               {"type":"reasoning","id":"rs_1","summary":[]},
               {"type":"function_call","call_id":"call_0","name":"exec_command","arguments":"{\"cmd\":\"ls\"}"},
               {"type":"function_call_output","call_id":"call_0","output":"hello.py"}],
      "tools":[{"type":"function","name":"exec_command","description":"run","parameters":{"type":"object"}},
               {"type":"web_search"},{"type":"namespace","name":"mcp__x","tools":[]}],
      "tool_choice":"auto","max_output_tokens":256,"reasoning":{"effort":"high"}})";
    std::string err;
    const std::string chat = tqserve::responses_to_chat(body, err);
    CHECK(err.empty());
    const Json c = tqllm::parse_json(chat);
    const auto * m = at(c, "messages");
    CHECK(m && m->arr.size() == 4);  // system (instructions + developer), user, assistant tool call, tool result
    if (m && m->arr.size() == 4) {
        CHECK(s(at(m->arr[0], "role")) == "system" && s(at(m->arr[0], "content")) == "be brief\n\nrules");
        CHECK(s(at(m->arr[1], "role")) == "user" && s(at(m->arr[1], "content")) == "make hello.py");
        CHECK(s(at(m->arr[2], "role")) == "assistant");
        const auto * tc = at(m->arr[2], "tool_calls");
        CHECK(tc && tc->arr.size() == 1 && s(at(tc->arr[0], "id")) == "call_0" && s(at(*at(tc->arr[0], "function"), "name")) == "exec_command");
        CHECK(s(at(m->arr[3], "role")) == "tool" && s(at(m->arr[3], "tool_call_id")) == "call_0" && s(at(m->arr[3], "content")) == "hello.py");
    }
    const auto * tools = at(c, "tools");
    CHECK(tools && tools->arr.size() == 1);  // web_search and namespace tools left out
    CHECK(at(c, "max_tokens") && at(c, "max_tokens")->num == 256);
    CHECK(s(at(c, "reasoning_effort")) == "xhigh");
    CHECK(at(c, "stream") && at(c, "stream")->num != 0);
    CHECK(tqserve::responses_to_chat("{\"input\":\"hi\"}", err).find("\"user\"") != std::string::npos);
}

static void test_responses_answer() {
    const std::string chat = R"({"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello",
      "tool_calls":[{"id":"call_0","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"ls\"}"}}]},
      "finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}})";
    const Json r = tqllm::parse_json(tqserve::chat_to_responses(chat, "tqllm"));
    CHECK(s(at(r, "object")) == "response" && s(at(r, "status")) == "completed");
    const auto * out = at(r, "output");
    CHECK(out && out->arr.size() == 2);
    if (out && out->arr.size() == 2) {
        CHECK(s(at(out->arr[0], "type")) == "message" && s(at(at(out->arr[0], "content")->arr[0], "text")) == "Hello");
        CHECK(s(at(out->arr[1], "type")) == "function_call" && s(at(out->arr[1], "call_id")).rfind("call_", 0) == 0 &&
              s(at(out->arr[1], "call_id")) != "call_0" &&  // unique across the conversation, not per answer
              s(at(out->arr[1], "arguments")) == "{\"cmd\":\"ls\"}");
    }
    CHECK(at(*at(r, "usage"), "input_tokens")->num == 5 && at(*at(r, "usage"), "output_tokens")->num == 2);
}

static void test_responses_stream(bool tool) {
    std::string sent;
    tqserve::ResponsesSse sse("tqllm", [&](const std::string & x) { sent += x; return true; });
    CHECK(sse.begin());
    for (const auto & c : chat_chunks(tool)) CHECK(sse.feed(c));
    for (const char * e : {"event: response.created", "event: response.in_progress", "event: response.output_item.added",
                           "event: response.content_part.added", "event: response.output_text.delta", "event: response.output_text.done",
                           "event: response.output_item.done", "event: response.reasoning_summary_text.delta", "event: response.completed"})
        CHECK(has(sent, e));
    CHECK(has(sent, "\"delta\": \"Hel\"") && has(sent, "\"text\": \"Hello\""));
    CHECK(sent.find("event: response.completed") > sent.find("event: response.output_text.done"));
    CHECK(has(sent, "\"input_tokens\": 12"));
    if (tool) {
        CHECK(has(sent, "event: response.function_call_arguments.delta") && has(sent, "event: response.function_call_arguments.done"));
        CHECK(!has(sent, "\"call_id\": \"call_0\"") && has(sent, "\"arguments\": \"{\\\"cmd\\\": \\\"python3 hello.py\\\"}\""));
        // the completed response lists reasoning, message and the call, in order
        const auto done = sent.substr(sent.find("event: response.completed"));
        CHECK(done.find("\"type\": \"reasoning\"") < done.find("\"type\": \"message\"") &&
              done.find("\"type\": \"message\"") < done.find("\"type\": \"function_call\""));
    }
}

static void test_messages() {
    const std::string body = R"({"model":"tqllm","max_tokens":100,"system":[{"type":"text","text":"sys"}],"stream":false,
      "messages":[{"role":"user","content":"hi"},{"role":"system","content":"ctx"},
                  {"role":"assistant","content":[{"type":"text","text":"checking"},{"type":"tool_use","id":"toolu_call_0","name":"ls","input":{"p":"."}}]},
                  {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_call_0","content":"a.txt"}]}],
      "tools":[{"name":"ls","description":"list","input_schema":{"type":"object"}}],"tool_choice":{"type":"any"}})";
    std::string err;
    const Json c = tqllm::parse_json(tqserve::messages_to_chat(body, err));
    CHECK(err.empty());
    const auto * m = at(c, "messages");
    CHECK(m && m->arr.size() == 4);
    if (m && m->arr.size() == 4) {
        CHECK(s(at(m->arr[0], "role")) == "system" && s(at(m->arr[0], "content")) == "sys\n\nctx");  // inline system folded in
        CHECK(s(at(m->arr[1], "role")) == "user");
        const auto * tc = at(m->arr[2], "tool_calls");
        CHECK(tc && s(at(tc->arr[0], "id")) == "call_0");
        CHECK(s(at(m->arr[3], "role")) == "tool" && s(at(m->arr[3], "tool_call_id")) == "call_0" && s(at(m->arr[3], "content")) == "a.txt");
    }
    CHECK(s(at(c, "tool_choice")) == "required" && at(c, "tools")->arr.size() == 1);
    const std::string chat = R"({"choices":[{"index":0,"message":{"role":"assistant","content":"Hi","tool_calls":[{"id":"call_0","type":"function",
      "function":{"name":"ls","arguments":"{\"p\":\".\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2}})";
    const Json a = tqllm::parse_json(tqserve::chat_to_messages(chat, "tqllm"));
    CHECK(s(at(a, "stop_reason")) == "tool_use" && at(a, "content")->arr.size() == 2);
    const std::string id1 = s(at(at(a, "content")->arr[1], "id"));
    CHECK(id1.rfind("toolu_call_", 0) == 0 && id1 != "toolu_call_0" && s(at(*at(at(a, "content")->arr[1], "input"), "p")) == ".");
    // two answers that both number their call call_0 must not repeat an id
    CHECK(s(at(at(tqllm::parse_json(tqserve::chat_to_messages(chat, "tqllm")), "content")->arr[1], "id")) != id1);
    std::string sent;
    tqserve::MessagesSse sse("tqllm", [&](const std::string & x) { sent += x; return true; });
    CHECK(sse.begin());
    for (const auto & ch : chat_chunks(true)) CHECK(sse.feed(ch));
    for (const char * e : {"event: message_start", "event: content_block_start", "\"text_delta\"",
                           "\"input_json_delta\"", "event: content_block_stop", "\"stop_reason\": \"tool_use\"", "event: message_stop"})
        CHECK(has(sent, e));
    CHECK(!has(sent, "thinking"));  // unsigned thinking blocks stall Anthropic clients
}

int main() {
    test_paths();
    test_responses_request();
    test_responses_answer();
    test_responses_stream(false);
    test_responses_stream(true);
    test_messages();
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("api_adapters: PASS\n");
    return 0;
}
