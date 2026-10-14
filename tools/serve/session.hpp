#pragma once
#include "tqllm/model.hpp"
#include "tqllm/tokenizer.hpp"
#include <src/json.hpp>
#include <src/vision.hpp>
#include "json_out.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tqserve {

struct ServeOptions {
    int draft_max = 0;            // --draft-max N: drafts per step (1..15); 0: the drafter's default (DFlash2 7, DSpark 3)
    uint32_t repeat_limit = 256;  // --repeat-limit N: one token N times in a row ends the request (0 = off)
};
inline ServeOptions & serve_options() {  // set by tqserve's main before the first request
    static ServeOptions o;
    return o;
}
inline int draft_length(bool native_mtp = false) {
    const int d = serve_options().draft_max;
    return d ? d : native_mtp ? 4 : 0;  // --mtp 4: best of depths 1-5 and 7 at 1.6K-51K (.77 3090, 10-13); 0: the drafter's default
}
inline uint32_t repeat_limit() { return serve_options().repeat_limit; }

// Degenerate output guard. Under memory pressure the sampler can pin on one token (">" ...) until max_tokens, which
// looks like a hang from outside. A run of repeat_limit() identical ids ends the request as "length" and is logged.
// device free memory in MB for the repeat-guard log (tqserve installs a CUDA probe; unset in host-only builds)
inline std::function<size_t()> & free_mb_probe() {
    static std::function<size_t()> f;
    return f;
}

struct RepeatGuard {
    uint32_t tok = 0, run = 0;
    size_t seen = 0;
    bool check(const std::vector<uint32_t> & ids) {  // ids: the generated ids so far, growing between calls
        const uint32_t limit = repeat_limit();
        if (!limit) return false;
        for (; seen < ids.size(); ++seen) {
            if (seen && ids[seen] == tok) ++run;
            else tok = ids[seen], run = 1;
            if (run >= limit) {
                const auto & probe = free_mb_probe();
                fprintf(stderr, "tqserve: repeat guard: token %u repeated %u times after %zu generated (device free %zu MB)\n",
                        tok, run, ids.size(), probe ? probe() : size_t(0));
                return true;
            }
        }
        return false;
    }
};

inline std::string trim(const std::string & x) {
    const auto a = x.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return x.substr(a, x.find_last_not_of(" \t\r\n") - a + 1);
}

inline std::string json_error(const std::string & message, const char * type = "invalid_request_error") {
    std::string out = "{\"error\":{\"message\":";
    jstr(out, message);
    out += ",\"type\":\"" + std::string(type) + "\"}}";
    return out;
}

struct ChatRequest {
    std::string prompt_text;
    std::vector<std::string> stop;
    uint32_t max_tokens = 64;
    bool chat = true;
    bool tools = false;  // tool schemas were rendered: parse <tool_call> blocks in the output
    bool thinking = false;  // prompt opens a <think> block: split the output into reasoning_content / content
    // reasoning budget, llama.cpp semantics: -1 unlimited, 0 end thinking immediately, N > 0 at most N thinking tokens, then the
    // message + end-of-thinking tag are forced and the answer continues. budget_set / message_set: the request named them
    // (otherwise the server defaults apply).
    int32_t thinking_budget = -1;
    bool budget_set = false;
    std::string budget_message;
    bool message_set = false;
    float temperature = 0.f, top_p = 1.f;  // temperature 0 = greedy (TQLLM samples from the exact top-16 logits)
    int top_k = 0;
    uint64_t seed = 0;
    bool stream = false;  // SSE (chat.completion.chunk / text_completion chunks, final usage + timings, [DONE])
    std::vector<std::string> images;  // image_url parts in prompt order (data: URIs)
    bool return_token_ids = false;  // add the generated token ids to the response (choices[0].token_ids)
    bool ok = false;
    std::string error;
};

inline bool read_token_count(const tqllm::Json * value, uint32_t & out) {
    if (!value || value->kind != tqllm::Json::Number || !std::isfinite(value->num) || value->num < 0 ||
        value->num > std::numeric_limits<uint32_t>::max()) return false;
    out = static_cast<uint32_t>(value->num);
    return true;
}

// Text of a message; image_url parts render as the chat template's placeholder and their URLs go to `images`
// (null: images are not allowed here).
inline bool read_content(const tqllm::Json & content, std::string & out, std::vector<std::string> * images = nullptr) {
    if (content.kind == tqllm::Json::String) {
        out = content.str;
        return true;
    }
    if (content.kind != tqllm::Json::Array) return false;
    for (const auto & part : content.arr) {
        if (part.kind != tqllm::Json::Object) continue;
        const auto * type = part.get("type");
        const auto * text = part.get("text");
        if (type && type->kind == tqllm::Json::String && type->str == "text" && text && text->kind == tqllm::Json::String)
            out += text->str;
        const auto * iu = part.get("image_url");
        if (type && type->kind == tqllm::Json::String && type->str == "image_url" && iu) {
            const auto * url = iu->kind == tqllm::Json::Object ? iu->get("url") : iu;
            if (!images || !url || url->kind != tqllm::Json::String) return false;
            images->push_back(url->str);
            out += "<|vision_start|><|image_pad|><|vision_end|>";
        }
    }
    return true;
}

// Drops an incomplete UTF-8 sequence at the end of a string (a budget can cut a multi-byte character in half).
inline std::string trim_incomplete_utf8(std::string s) {
    size_t n = s.size(), i = n;
    int back = 0;
    while (i > 0 && back < 4 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) { --i; ++back; }
    if (i > 0) {
        const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
        const int need = lead >= 0xF0 ? 3 : lead >= 0xE0 ? 2 : lead >= 0xC0 ? 1 : 0;
        if (need > back) s.resize(i - 1);  // lead byte without all its continuation bytes
    }
    return s;
}

// Optional prompt renderer (llama.cpp --jinja style): renders messages / tools / template kwargs with a Jinja chat template
// instead of the native Qwen layout. Set by tqserve when --jinja or --chat-template-file is given.
using PromptRenderer = std::function<std::string(const tqllm::Json & messages, const tqllm::Json * tools, const tqllm::Json & kwargs)>;

inline ChatRequest parse_request(const std::string & body, uint32_t default_max_tokens, const std::string & system,
                                bool chat_endpoint, bool reasoning = false, const std::string & default_effort = "xhigh",
                                const PromptRenderer * renderer = nullptr, const tqllm::Json * default_kwargs = nullptr) {
    ChatRequest req;
    req.chat = chat_endpoint;
    try {
        const auto root = tqllm::parse_json(body);
        if (root.kind != tqllm::Json::Object) {
            req.error = "request body must be a JSON object";
            return req;
        }

        const auto * stream = root.get("stream");
        req.stream = stream && stream->kind == tqllm::Json::Bool && stream->num != 0;
        const auto * rti = root.get("return_token_ids");
        req.return_token_ids = rti && rti->kind == tqllm::Json::Bool && rti->num != 0;

        auto num = [&](const char * key, double & v) {
            if (const auto * x = root.get(key); x && x->kind == tqllm::Json::Number) { v = x->num; return true; }
            return false;
        };
        double tv = 0;
        if (num("temperature", tv)) req.temperature = float(tv);
        if (num("top_p", tv)) req.top_p = float(tv);
        if (const auto * x = root.get("top_k"); x && x->kind != tqllm::Json::Null) {
            if (x->kind != tqllm::Json::Number || !std::isfinite(x->num) || x->num < 0 ||
                x->num > std::numeric_limits<int>::max() || std::floor(x->num) != x->num) {
                req.error = "top_k must be a non-negative integer no greater than INT_MAX";
                return req;
            }
            req.top_k = int(x->num);
        }
        if (const auto * x = root.get("seed"); x && x->kind != tqllm::Json::Null) {
            // JSON numbers are binary64: accept only the range with exact integer representation.
            constexpr double max_seed = 9007199254740991.0;
            if (x->kind != tqllm::Json::Number || !std::isfinite(x->num) || x->num < 0 ||
                x->num > max_seed || std::floor(x->num) != x->num) {
                req.error = "seed must be an integer from 0 to 9007199254740991";
                return req;
            }
            req.seed = uint64_t(x->num);
        }
        const auto * max_tokens = root.get("max_tokens");
        if (!max_tokens) max_tokens = root.get("max_completion_tokens");
        if (max_tokens && !read_token_count(max_tokens, req.max_tokens)) {
            req.error = "max_tokens must be a non-negative integer";
            return req;
        }
        if (!max_tokens || req.max_tokens == 0) req.max_tokens = default_max_tokens;  // omitted (Codex never sends it): server default
        req.max_tokens = std::min(req.max_tokens, default_max_tokens);

        if (const auto * stop = root.get("stop")) {
            if (stop->kind == tqllm::Json::String) req.stop.push_back(stop->str);
            else if (stop->kind == tqllm::Json::Array) {
                for (const auto & item : stop->arr) {
                    if (item.kind != tqllm::Json::String) {
                        req.error = "stop array entries must be strings";
                        return req;
                    }
                    req.stop.push_back(item.str);
                }
            } else {
                req.error = "stop must be a string or an array of strings";
                return req;
            }
        }

        if (chat_endpoint) {
            const auto * messages = root.get("messages");
            if (!messages || messages->kind != tqllm::Json::Array || messages->arr.empty()) {
                req.error = "messages must be a non-empty array";
                return req;
            }
            std::string system_text = system;
            size_t first_message = 0;
            if (messages->arr.front().kind == tqllm::Json::Object) {
                const auto * role = messages->arr.front().get("role");
                const auto * content = messages->arr.front().get("content");
                std::string text;
                if (role && role->kind == tqllm::Json::String && role->str == "system" && content && read_content(*content, text)) {
                    system_text = text;
                    first_message = 1;
                }
            }
            // Thinking and reasoning effort, as the Qwen3.8 chat template: enable_thinking (chat_template_kwargs) overrides
            // the server default; reasoning_effort (top level or chat_template_kwargs) defaults to the server's
            // --reasoning-effort (xhigh) when thinking
            req.thinking = reasoning;
            for (const auto * src : {&root, root.get("chat_template_kwargs")}) {  // budget fields (top level or kwargs)
                if (!src || src->kind != tqllm::Json::Object) continue;
                // llama.cpp names first (reasoning_budget_tokens; thinking_budget_tokens is its Anthropic-style name), then ours
                for (const char * key : {"reasoning_budget_tokens", "thinking_budget_tokens", "thinking_budget", "reasoning_budget"})
                    if (const auto * b = src->get(key); b && b->kind == tqllm::Json::Number && b->num >= -1) {
                        req.thinking_budget = static_cast<int32_t>(b->num);
                        req.budget_set = true;
                        break;
                    }
                if (const auto * m = src->get("reasoning_budget_message"); m && m->kind == tqllm::Json::String) {
                    req.budget_message = m->str;
                    req.message_set = true;
                }
            }
            std::string effort = default_effort;
            if (const auto * e = root.get("reasoning_effort"); e && e->kind == tqllm::Json::String) effort = e->str;
            if (const auto * kw = root.get("chat_template_kwargs"); kw && kw->kind == tqllm::Json::Object) {
                if (const auto * et = kw->get("enable_thinking"); et && et->kind == tqllm::Json::Bool) req.thinking = et->num != 0;
                if (const auto * e = kw->get("reasoning_effort"); e && e->kind == tqllm::Json::String) effort = e->str;
            }
            if (effort == "minimal") effort = "low";
            if (effort == "high" || effort == "max") effort = "xhigh";
            std::string effort_text;
            if (req.thinking) {
                if (effort == "xhigh")
                    effort_text = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                                  "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
                                  "clarity in the final answer.";
                else if (effort == "low")
                    effort_text = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
                                  "the conclusion without unnecessary elaboration.";
                else if (effort != "medium") {
                    req.error = "reasoning_effort must be low, medium or xhigh";
                    return req;
                }
            }
            const auto * tools = root.get("tools");
            if (renderer && *renderer) {  // Jinja chat template: server default kwargs, then the request's, then enable_thinking
                tqllm::Json kw;
                kw.kind = tqllm::Json::Object;
                auto put = [&](const std::string & k, const tqllm::Json & v) {
                    for (auto & [key, val] : kw.obj)
                        if (key == k) { val = v; return; }
                    kw.obj.push_back({k, v});
                };
                if (default_kwargs && default_kwargs->kind == tqllm::Json::Object)
                    for (const auto & [k, v] : default_kwargs->obj) put(k, v);
                if (const auto * rk = root.get("chat_template_kwargs"); rk && rk->kind == tqllm::Json::Object)
                    for (const auto & [k, v] : rk->obj) put(k, v);
                if (const auto * e = root.get("reasoning_effort"); e && e->kind == tqllm::Json::String) put("reasoning_effort", *e);
                tqllm::Json th;
                th.kind = tqllm::Json::Bool;
                th.num = req.thinking ? 1 : 0;
                put("enable_thinking", th);
                tqllm::Json msgs = *messages;
                if (!system.empty() && !(msgs.arr.front().kind == tqllm::Json::Object && msgs.arr.front().get("role") &&
                                         msgs.arr.front().get("role")->str == "system")) {
                    tqllm::Json sm;
                    sm.kind = tqllm::Json::Object;
                    tqllm::Json r, c;
                    r.kind = c.kind = tqllm::Json::String;
                    r.str = "system";
                    c.str = system;
                    sm.obj = {{"role", r}, {"content", c}};
                    msgs.arr.insert(msgs.arr.begin(), sm);
                }
                // the template renders the image placeholders; the URLs must still reach the image path, in prompt order
                for (const auto & m : msgs.arr) {
                    const auto * role = m.kind == tqllm::Json::Object ? m.get("role") : nullptr;
                    const auto * content = m.kind == tqllm::Json::Object ? m.get("content") : nullptr;
                    if (!content || content->kind != tqllm::Json::Array) continue;
                    std::string unused;
                    const bool user = role && role->kind == tqllm::Json::String && role->str == "user";
                    if (!read_content(*content, unused, user ? &req.images : nullptr)) {
                        req.error = "image_url parts are allowed in user messages only";
                        return req;
                    }
                }
                req.tools = tools && tools->kind == tqllm::Json::Array && !tools->arr.empty();
                req.prompt_text = (*renderer)(msgs, req.tools ? tools : nullptr, kw);
                req.ok = true;
                return req;
            }
            if (tools && tools->kind == tqllm::Json::Array && !tools->arr.empty()) {  // Qwen3.5 chat template layout
                req.tools = true;
                req.prompt_text = "<|im_start|>system\n" + (effort_text.empty() ? std::string() : effort_text + "\n\n") +
                                  "# Tools\n\nYou have access to the following functions:\n\n<tools>";
                for (const auto & tool : tools->arr) req.prompt_text += "\n" + dump(tool);
                req.prompt_text +=
                    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                    "<think>\nBrief explanation of tool call\n</think>\n"
                    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
                    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
                    "- You can use the <think></think> block to plan your next tool call OR to synthesize data and formulate your "
                    "final response to the user.\n"
                    "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.\n"
                    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
                    "within <tool_call></tool_call> XML tags.\n"
                    "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after thinking, with NO "
                    "conversational text before it.\n"
                    "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO spaces or "
                    "indentation before them.\n"
                    "- To call multiple functions, output a separate, completely closed <tool_call></tool_call> block for EACH "
                    "function. Do NOT nest <tool_call> blocks.\n"
                    "- If you have all necessary data, provide your final answer directly to the user without any tool call.\n"
                    "</IMPORTANT>";
                if (!trim(system_text).empty()) req.prompt_text += "\n\n" + trim(system_text);
                req.prompt_text += "<|im_end|>\n";
            } else if (!trim(system_text).empty()) {
                req.prompt_text = "<|im_start|>system\n" + (effort_text.empty() ? std::string() : effort_text + "\n\n") +
                                  trim(system_text) + "<|im_end|>\n";
            } else if (!effort_text.empty()) {
                req.prompt_text = "<|im_start|>system\n" + effort_text + "<|im_end|>\n";
            }
            // Chat-template parity with froggeric v22 (TQ3_4S-v2's embedded template): past reasoning is kept unless
            // preserve_thinking / preserve_reasoning is false, then only after the last real user query
            bool preserve = true;
            if (const auto * kw = root.get("chat_template_kwargs"); kw && kw->kind == tqllm::Json::Object) {
                for (const char * key : {"preserve_thinking", "preserve_reasoning"})
                    if (const auto * v = kw->get(key); v && v->kind == tqllm::Json::Bool) { preserve = v->num != 0; break; }
            }
            const size_t n_msgs = messages->arr.size() - first_message;
            size_t last_query = n_msgs > 51 ? n_msgs - 1 : 0;
            for (size_t i = messages->arr.size(); i-- > first_message;) {
                const auto & m = messages->arr[i];
                const auto * r = m.get("role");
                const auto * c = m.get("content");
                std::string t;
                std::vector<std::string> ignored;
                if (!r || r->kind != tqllm::Json::String || r->str != "user" || !c || !read_content(*c, t, &ignored)) continue;
                t = trim(t);
                const bool tool_resp = t.rfind("<tool_response>", 0) == 0 && t.size() >= 16 &&
                                       t.compare(t.size() - 16, 16, "</tool_response>") == 0;
                if (!tool_resp) { last_query = i - first_message; break; }
            }
            int tool_failures = 0;
            for (size_t i = first_message; i < messages->arr.size(); ++i) {
                const auto & message = messages->arr[i];
                if (message.kind != tqllm::Json::Object) {
                    req.error = "each message must be an object";
                    return req;
                }
                const auto * role = message.get("role");
                const auto * content = message.get("content");
                const auto * calls = message.get("tool_calls");
                std::string text;
                const bool has_calls = calls && calls->kind == tqllm::Json::Array && !calls->arr.empty();
                const bool null_content = !content || content->kind == tqllm::Json::Null;
                const bool user = role && role->kind == tqllm::Json::String && role->str == "user";
                if (!role || role->kind != tqllm::Json::String || role->str.empty() ||
                    role->str.find_first_of("\r\n") != std::string::npos ||
                    (!(null_content && has_calls) && (!content || !read_content(*content, text, user ? &req.images : nullptr)))) {
                    req.error = "each message needs a role and string or text-array content (images: user messages only)";
                    return req;
                }
                text = trim(text);
                if (role->str == "tool") {  // consecutive tool results share one user turn
                    const auto * prev = i > first_message ? messages->arr[i - 1].get("role") : nullptr;
                    const auto * next = i + 1 < messages->arr.size() ? messages->arr[i + 1].get("role") : nullptr;
                    std::string lower = text;
                    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return std::tolower(ch); });
                    const std::string head = lower.substr(0, 80);
                    bool failed = text.size() < 500 && text.find("$ ") == std::string::npos && lower.find("took ") == std::string::npos;
                    if (failed) {
                        failed = false;
                        for (const char * e : {"\"error\":", "error:", "err!", "fatal:", "exception:", "traceback", "command not found",
                                               "invalid syntax", "failed to"})
                            if (head.find(e) != std::string::npos) { failed = true; break; }
                    }
                    tool_failures = failed ? tool_failures + 1 : 0;
                    if (!prev || prev->kind != tqllm::Json::String || prev->str != "tool") req.prompt_text += "<|im_start|>user";
                    req.prompt_text += "\n<tool_response>\n" + text;
                    if (tool_failures >= 2)
                        req.prompt_text += "\n\n\xe2\x9a\xa0\xef\xb8\x8f SYSTEM WARNING: " + std::to_string(tool_failures) +
                                           " consecutive tool errors detected. Your previous approach is incorrect. You MUST use a "
                                           "fundamentally different approach or corrected arguments.";
                    else if (tool_failures == 1)
                        req.prompt_text += "\n\n\xe2\x9a\xa0\xef\xb8\x8f SYSTEM WARNING: The previous tool call returned an error. "
                                           "Diagnose the failure and retry with completely corrected arguments.";
                    req.prompt_text += "\n</tool_response>";
                    if (!next || next->kind != tqllm::Json::String || next->str != "tool") req.prompt_text += "<|im_end|>\n";
                    continue;
                }
                if (role->str == "assistant") {
                    std::string reasoning;
                    for (const char * key : {"reasoning_content", "thinking"})
                        if (const auto * rc = message.get(key); rc && rc->kind == tqllm::Json::String) { reasoning = rc->str; break; }
                    if (const auto end = text.find("</think>"); end != std::string::npos) {
                        if (reasoning.empty()) {
                            reasoning = text.substr(0, end);
                            if (const auto start = reasoning.rfind("<think>"); start != std::string::npos) reasoning = reasoning.substr(start + 7);
                        }
                        text = trim(text.substr(end + 8));
                    }
                    reasoning = trim(reasoning);
                    if ((preserve || i - first_message > last_query) && !reasoning.empty())
                        req.prompt_text += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + text;
                    else
                        req.prompt_text += "<|im_start|>assistant\n" + text;
                    if (has_calls) {
                        for (size_t c = 0; c < calls->arr.size(); ++c) {
                            const tqllm::Json * fn = calls->arr[c].get("function");
                            if (!fn) fn = &calls->arr[c];
                            const auto * name = fn->get("name");
                            const auto * args = fn->get("arguments");
                            req.prompt_text += (c || !text.empty()) ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=";
                            req.prompt_text += (name && name->kind == tqllm::Json::String ? name->str : "") + ">\n";
                            tqllm::Json parsed;
                            if (args && args->kind == tqllm::Json::String) {
                                try { parsed = tqllm::parse_json(args->str); } catch (const std::exception &) {}
                            } else if (args) parsed = *args;
                            for (const auto & [k, v] : parsed.obj)
                                req.prompt_text += "<parameter=" + k + ">\n" +
                                                   (v.kind == tqllm::Json::String ? v.str : dump(v)) + "\n</parameter>\n";
                            req.prompt_text += "</function>\n</tool_call>";
                        }
                    }
                    req.prompt_text += "<|im_end|>\n";
                    continue;
                }
                if (role->str == "user") tool_failures = 0;
                req.prompt_text += "<|im_start|>" + role->str + "\n" + text + "<|im_end|>\n";
            }
            req.prompt_text += "<|im_start|>assistant\n<think>\n";
            if (!req.thinking) req.prompt_text += "\n</think>\n\n";
        } else {
            const auto * prompt = root.get("prompt");
            if (!prompt || prompt->kind != tqllm::Json::String || prompt->str.empty()) {
                req.error = "prompt must be a non-empty string";
                return req;
            }
            req.prompt_text = prompt->str;
        }
        req.ok = true;
    } catch (const std::exception & e) {
        req.error = e.what();
    }
    return req;
}

struct ToolCall {
    std::string name, arguments;  // arguments: JSON object text
};

struct DecodeResult {
    std::string text;
    std::string reasoning;  // thinking text (before </think>) when the request thinks
    std::vector<ToolCall> tool_calls;
    uint32_t prompt_tokens = 0;
    uint32_t completion_tokens = 0;
    std::vector<uint32_t> token_ids;  // generated ids before EOS, when the request set return_token_ids
    // engine timings (llama.cpp `timings` fields): prefill / decode wall time, speculative drafts proposed / accepted
    double prefill_ms = 0, decode_ms = 0;
    uint64_t draft_n = 0, draft_accepted = 0;
    std::string finish_reason = "length";
    std::string error;
};

inline void extract_tool_calls(DecodeResult & r);

// stats from generate_spec / spec_finish: steps, drafted, accepted, decode us, prefill us
inline void set_timings(DecodeResult & r, const uint64_t * st) {
    r.draft_n = st[1];
    r.draft_accepted = st[2];
    r.decode_ms = double(st[3]) / 1000.0;
    r.prefill_ms = double(st[4]) / 1000.0;
}

// Server-wide vision encoder (--mmproj); requests encode under the mutex (the encoder reuses its buffers).
struct Vision {
    std::unique_ptr<tqllm::VisionEncoder> enc;
    std::mutex mu;
    cudaStream_t stream = nullptr;
    uint32_t pad_id = 0;  // <|image_pad|>
};
inline Vision *& vision() {
    static Vision * v = nullptr;
    return v;
}

// Image embeddings of one request: device memory owned here until the job ends.
struct PreparedImages {
    std::vector<tqllm::Model::PromptImage> spans;
    std::shared_ptr<void> mem;
};

inline bool base64_decode(const std::string & in, size_t from, std::vector<uint8_t> & out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+' || c == '-') return 62;
        if (c == '/' || c == '_') return 63;
        return -1;
    };
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = from; i < in.size(); ++i) {
        const char c = in[i];
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        const int v = val(c);
        if (v < 0) return false;
        acc = (acc << 6) | uint32_t(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t(acc >> bits));
        }
    }
    return true;
}

// Each <|image_pad|> of the prompt becomes the image's nx * ny placeholders; embeddings land in `images`.
inline std::string expand_images(const ChatRequest & req, std::vector<uint32_t> & prompt, PreparedImages & images) {
    auto * v = vision();
    if (!v) return "this server has no vision encoder (start tqserve with --mmproj)";
    std::vector<size_t> pads;
    for (size_t i = 0; i < prompt.size(); ++i)
        if (prompt[i] == v->pad_id) pads.push_back(i);
    if (pads.size() != req.images.size()) return "image placeholders in the text do not match the images";
    struct Enc { int nx, ny; float * d; };
    std::vector<Enc> enc;
    auto owned = std::make_shared<std::vector<float *>>();
    images.mem = std::shared_ptr<void>(owned.get(), [owned](void *) {
        for (float * p : *owned) cudaFree(p);
    });
    {
        std::lock_guard<std::mutex> lock(v->mu);
        for (const auto & url : req.images) {
            const auto comma = url.find(',');
            if (url.rfind("data:", 0) != 0 || comma == std::string::npos || url.find(";base64") > comma)
                return "image_url must be a base64 data: URI";
            std::vector<uint8_t> bytes, rgb;
            int w = 0, h = 0, nx = 0, ny = 0;
            if (!base64_decode(url, comma + 1, bytes) || !tqllm::decode_image(bytes.data(), bytes.size(), rgb, w, h))
                return "cannot decode image";
            const float * e = v->enc->encode(v->enc->preprocess(rgb.data(), w, h), nx, ny);
            const size_t n = size_t(nx) * size_t(ny) * size_t(v->enc->embd());
            float * d = nullptr;
            if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return "out of device memory for image embeddings";
            owned->push_back(d);
            cudaMemcpyAsync(d, e, n * sizeof(float), cudaMemcpyDeviceToDevice, v->stream);
            if (cudaStreamSynchronize(v->stream) != cudaSuccess) return "vision encoder failed";
            enc.push_back({nx, ny, d});
        }
    }
    std::vector<uint32_t> out;
    out.reserve(prompt.size());
    size_t k = 0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (k < pads.size() && i == pads[k]) {
            const auto & e = enc[k++];
            images.spans.push_back({uint32_t(out.size()), e.nx, e.ny, e.d});
            out.insert(out.end(), size_t(e.nx) * size_t(e.ny), v->pad_id);
        } else {
            out.push_back(prompt[i]);
        }
    }
    prompt = std::move(out);
    return {};
}

// Tokenizes the prompt and sets the generation budget; result.error is set when the request cannot run.
inline std::vector<uint32_t> prepare(const tqllm::Tokenizer & tokenizer, const ChatRequest & req, uint32_t context,
                                     DecodeResult & result, uint32_t & count, PreparedImages * images = nullptr) {
    auto prompt = tokenizer.encode(req.prompt_text);
    if (!req.images.empty()) {
        result.error = images ? expand_images(req, prompt, *images) : "this path does not take images";
        if (!result.error.empty()) return prompt;
    }
    result.prompt_tokens = static_cast<uint32_t>(prompt.size());
    if (prompt.empty()) result.error = "prompt tokenized to an empty sequence";
    // max_tokens is clamped to the room left, minus what a speculative step may write past the last token
    constexpr uint32_t reserve = 64;
    if (!result.error.empty()) return prompt;
    if (prompt.size() + reserve >= context) result.error = "prompt leaves no room for generation in the configured context";
    else count = std::min<uint32_t>(req.max_tokens, context - reserve - static_cast<uint32_t>(prompt.size()));
    return prompt;
}

// Generated ids of the speculative path -> text, finish reason, stop strings, reasoning split, tool calls.
inline void finish(const tqllm::Tokenizer & tokenizer, const ChatRequest & req, const std::vector<uint32_t> & gen,
                   DecodeResult & result) {
    {
        std::vector<uint32_t> output;
        result.finish_reason = "length";
        for (const uint32_t tok : gen) {
            ++result.completion_tokens;
            if (tok == tokenizer.eos_id()) {
                result.finish_reason = "stop";
                break;
            }
            output.push_back(tok);
        }
        if (req.return_token_ids) result.token_ids = output;
        result.text = tokenizer.decode(output);
        for (const auto & stop : req.stop) {
            if (stop.empty()) continue;
            if (const auto at = result.text.find(stop); at != std::string::npos) {
                result.text.resize(at);
                result.finish_reason = "stop";
            }
        }
        if (req.thinking) {  // like vLLM's reasoning parser: answer after </think>; truncated thinking -> empty answer
            // </think> is a special token (decode() drops it), so split on its id
            static const auto think_end = tokenizer.encode("</think>");
            const auto at = think_end.size() == 1 ? std::find(output.begin(), output.end(), think_end[0]) : output.end();
            result.reasoning = trim(tokenizer.decode({output.begin(), at}));
            result.text = at == output.end() ? std::string() : trim(tokenizer.decode({at + 1, output.end()}));
            for (const auto & stop : req.stop)
                if (const auto p = stop.empty() ? std::string::npos : result.text.find(stop); p != std::string::npos) result.text.resize(p);
        }
        if (req.tools) extract_tool_calls(result);
    }
}

inline DecodeResult decode(const tqllm::Tokenizer & tokenizer, tqllm::Model & model, const ChatRequest & req,
                           uint32_t context) {
    DecodeResult result;
    uint32_t count = 0;
    const bool incremental = model.has_drafter() || model.incremental();
    PreparedImages images;
    const auto prompt = prepare(tokenizer, req, context, result, count, incremental ? &images : nullptr);
    if (!result.error.empty()) return result;
    // verify path (quantized qwen35 GGUF): chunked prefill, then DFlash2 drafts or one verified token per step.
    // The forward() loop below prefills token by token (one full weight pass per prompt token).
    if (incremental) {
        model.set_sampling(req.temperature, req.top_p, req.top_k, req.seed);
        if (!images.spans.empty()) model.set_prompt_images(images.spans);
        model.spec_begin(prompt, count, draft_length(model.native_mtp()));
        RepeatGuard guard;
        while (model.spec_advance())
            if (guard.check(model.spec_tokens(0))) break;
        uint64_t st[5] = {};
        finish(tokenizer, req, model.spec_finish(st), result);
        set_timings(result, st);
        return result;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i + 1 < prompt.size(); ++i) model.forward(prompt[i], i, false);
    uint32_t next = model.forward(prompt.back(), static_cast<uint32_t>(prompt.size() - 1), true);
    const auto t1 = std::chrono::steady_clock::now();
    result.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    struct DecodeClock {  // decode time ends when the loop exits, whichever way
        DecodeResult & r;
        std::chrono::steady_clock::time_point t;
        ~DecodeClock() { r.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }
    } clock{result, t1};
    std::vector<uint32_t> output;
    output.reserve(count);
    RepeatGuard guard;
    for (uint32_t i = 0; i < count; ++i) {
        ++result.completion_tokens;
        if (next == tokenizer.eos_id()) {
            result.finish_reason = "stop";
            break;
        }
        output.push_back(next);
        if (guard.check(output)) break;
        result.text = tokenizer.decode(output);
        bool stopped = false;
        for (const auto & stop : req.stop) {
            if (stop.empty()) continue;
            const auto at = result.text.find(stop);
            if (at != std::string::npos) {
                result.text.resize(at);
                stopped = true;
                break;
            }
        }
        if (stopped) {
            result.finish_reason = "stop";
            break;
        }
        if (i + 1 < count) next = model.forward(next, static_cast<uint32_t>(prompt.size()) + i, true);
    }
    if (req.return_token_ids) result.token_ids = output;
    if (req.tools) extract_tool_calls(result);
    return result;
}

// [<tool_call>]\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>\n</tool_call>: values that parse as
// JSON (numbers, booleans, objects, arrays) are kept as JSON, anything else becomes a string.
inline void extract_tool_calls(DecodeResult & r) {
    // <tool_call> / </tool_call> are special tokens the detokenizer drops, so blocks are found by <function=...>.
    const auto first_fn = r.text.find("<function=");
    if (first_fn == std::string::npos) return;
    const auto first_tc = r.text.find("<tool_call>");
    const size_t first = std::min(first_fn, first_tc);
    size_t at = first_fn;
    while ((at = r.text.find("<function=", at)) != std::string::npos) {
        const auto end = r.text.find("</function>", at);
        const std::string block = r.text.substr(at, end == std::string::npos ? std::string::npos : end - at);
        at = end == std::string::npos ? r.text.size() : end + 11;
        const auto fe = block.find('>');
        if (fe == std::string::npos) continue;
        ToolCall call;
        call.name = trim(block.substr(10, fe - 10));
        std::string args = "{";
        size_t p = fe;
        while ((p = block.find("<parameter=", p)) != std::string::npos) {
            const auto pe = block.find('>', p);
            if (pe == std::string::npos) break;
            const std::string key = trim(block.substr(p + 11, pe - p - 11));
            const auto ve = block.find("</parameter>", pe);
            std::string value = block.substr(pe + 1, ve == std::string::npos ? std::string::npos : ve - pe - 1);
            if (!value.empty() && value.front() == '\n') value.erase(0, 1);
            if (!value.empty() && value.back() == '\n') value.pop_back();
            if (args.size() > 1) args += ", ";
            jstr(args, key);
            args += ": ";
            bool as_json = false;
            try {
                const auto j = tqllm::parse_json(value);
                as_json = j.kind != tqllm::Json::String;
                if (as_json) args += dump(j);
            } catch (const std::exception &) {}
            if (!as_json) jstr(args, value);
            p = ve == std::string::npos ? block.size() : ve;
        }
        call.arguments = args + "}";
        r.tool_calls.push_back(std::move(call));
    }
    if (!r.tool_calls.empty()) {
        r.text = trim(r.text.substr(0, first));
        r.finish_reason = "tool_calls";
    }
}

inline void usage_timings(std::string & out, const DecodeResult & result);

inline std::string completion_json(const std::string & model, const DecodeResult & result, bool chat) {
    std::string out = "{\"id\":";
    jstr(out, chat ? "chatcmpl-tqllm" : "cmpl-tqllm");
    out += chat ? ",\"object\":\"chat.completion\",\"model\":" : ",\"object\":\"text_completion\",\"model\":";
    jstr(out, model);
    out += ",\"choices\":[{\"index\":0,";
    if (chat) {
        out += "\"message\":{\"role\":\"assistant\",\"content\":";
        if (result.tool_calls.empty() || !result.text.empty()) jstr(out, result.text);
        else out += "null";
        if (!result.reasoning.empty()) {
            out += ",\"reasoning_content\":";
            jstr(out, result.reasoning);
        }
        if (!result.tool_calls.empty()) {
            out += ",\"tool_calls\":[";
            for (size_t i = 0; i < result.tool_calls.size(); ++i) {
                if (i) out.push_back(',');
                out += "{\"id\":";
                jstr(out, "call_" + std::to_string(i));
                out += ",\"type\":\"function\",\"function\":{\"name\":";
                jstr(out, result.tool_calls[i].name);
                out += ",\"arguments\":";
                jstr(out, result.tool_calls[i].arguments);
                out += "}}";
            }
            out.push_back(']');
        }
        out += "},";
    } else {
        out += "\"text\":";
        jstr(out, result.text);
        out += ',';
    }
    if (!result.token_ids.empty()) {
        out += "\"token_ids\":[";
        for (size_t i = 0; i < result.token_ids.size(); ++i) out += (i ? "," : "") + std::to_string(result.token_ids[i]);
        out += "],";
    }
    out += "\"finish_reason\":";
    jstr(out, result.finish_reason);
    out += "}]";
    usage_timings(out, result);
    out += "}";
    return out;
}

// ,"usage":{...},"timings":{...} (llama.cpp-style engine timings)
inline void usage_timings(std::string & out, const DecodeResult & result) {
    out += ",\"usage\":{\"prompt_tokens\":" + std::to_string(result.prompt_tokens);
    out += ",\"completion_tokens\":" + std::to_string(result.completion_tokens);
    out += ",\"total_tokens\":" + std::to_string(result.prompt_tokens + result.completion_tokens) + "}";
    // llama.cpp-style engine timings; the first token comes from prefill, so decode rate = (n - 1) / decode time
    const uint32_t n = result.completion_tokens;
    auto num = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%.3f", v); return std::string(b); };
    out += ",\"timings\":{\"prompt_n\":" + std::to_string(result.prompt_tokens) + ",\"prompt_ms\":" + num(result.prefill_ms);
    out += ",\"prompt_per_second\":" + num(result.prefill_ms > 0 ? result.prompt_tokens * 1000.0 / result.prefill_ms : 0);
    out += ",\"predicted_n\":" + std::to_string(n) + ",\"predicted_ms\":" + num(result.decode_ms);
    out += ",\"predicted_per_second\":" + num(result.decode_ms > 0 && n > 1 ? (n - 1) * 1000.0 / result.decode_ms : 0);
    out += ",\"draft_n\":" + std::to_string(result.draft_n) + ",\"draft_n_accepted\":" + std::to_string(result.draft_accepted) + "}";
}

// SSE output of one request. update() sends what the ids generated so far add to the reasoning / content texts that
// finish() would return; text that can still change (a partial UTF-8 sequence, a possible stop-string prefix) is held
// back. With tools, content goes out at the end with the parsed tool calls. close() ends with usage + timings, [DONE].
class StreamWriter {
public:
    using Send = std::function<bool(const std::string &)>;  // false: the client went away
    StreamWriter(const tqllm::Tokenizer & tok, const ChatRequest & req, std::string model, Send send)
        : tok_(tok), req_(req), plain_(req), model_(std::move(model)), send_(std::move(send)),
          created_(std::to_string(std::time(nullptr))) {
        plain_.tools = false;
        for (const auto & st : req.stop) hold_ = std::max(hold_, st.empty() ? size_t(0) : st.size() - 1);
    }
    bool update(const std::vector<uint32_t> & gen) {  // one delta per new token (clients count deltas as tokens)
        for (size_t k = std::min(seen_ + 1, gen.size()); k <= gen.size() && k > seen_; ++k) {
            DecodeResult r;
            finish(tok_, plain_, std::vector<uint32_t>(gen.begin(), gen.begin() + std::ptrdiff_t(k)), r);
            if (!emit(r.reasoning, sr_, true, false) || !(req_.tools || emit(r.text, sc_, false, false))) return false;
        }
        seen_ = std::max(seen_, gen.size());
        return true;
    }
    bool close(const DecodeResult & result) {
        if (!emit(result.reasoning, sr_, true, true) || !emit(result.text, sc_, false, true)) return false;
        std::string out = head() + "\"choices\":[{\"index\":0,";
        if (req_.chat) {
            out += "\"delta\":{";
            if (!result.tool_calls.empty()) {
                out += "\"tool_calls\":[";
                for (size_t i = 0; i < result.tool_calls.size(); ++i) {
                    if (i) out.push_back(',');
                    out += "{\"index\":" + std::to_string(i) + ",\"id\":";
                    jstr(out, "call_" + std::to_string(i));
                    out += ",\"type\":\"function\",\"function\":{\"name\":";
                    jstr(out, result.tool_calls[i].name);
                    out += ",\"arguments\":";
                    jstr(out, result.tool_calls[i].arguments);
                    out += "}}";
                }
                out.push_back(']');
            }
            out += "},";
        } else {
            out += "\"text\":\"\",";
        }
        out += "\"finish_reason\":";
        jstr(out, result.tool_calls.empty() ? result.finish_reason : std::string("tool_calls"));
        out += "}]";
        usage_timings(out, result);
        out += "}";
        return send_("data: " + out + "\n\n") && send_("data: [DONE]\n\n");
    }
    bool error(const std::string & message) { return send_("data: " + json_error(message) + "\n\n"); }
    // thinking budget: send what is left of a phase's deltas without the final chunk; a phase-2 writer skips the role
    bool flush(const DecodeResult & r) { return emit(r.reasoning, sr_, true, true) && (req_.tools || emit(r.text, sc_, false, true)); }
    void mark_role_sent() { role_sent_ = true; }

private:
    std::string head() const {
        std::string out = "{\"id\":";
        jstr(out, req_.chat ? "chatcmpl-tqllm" : "cmpl-tqllm");
        out += req_.chat ? ",\"object\":\"chat.completion.chunk\"" : ",\"object\":\"text_completion\"";
        out += ",\"created\":" + created_ + ",\"model\":";
        jstr(out, model_);
        out += ',';
        return out;
    }
    // bytes at the end of `t` that may still change: an incomplete UTF-8 sequence or a replacement character
    static size_t unstable_tail(const std::string & t) {
        if (t.size() >= 3 && t.compare(t.size() - 3, 3, "\xEF\xBF\xBD") == 0) return 3;
        for (size_t k = 1; k <= 3 && k <= t.size(); ++k) {
            const auto c = static_cast<unsigned char>(t[t.size() - k]);
            if ((c & 0xC0) == 0x80) continue;                          // continuation byte
            const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            return need > k ? k : 0;
        }
        return 0;
    }
    bool emit(const std::string & full, std::string & sent, bool reasoning, bool final) {
        if (full.size() <= sent.size() || full.compare(0, sent.size(), sent) != 0) return true;  // nothing new / not a prefix
        size_t end = full.size();
        if (!final) {
            end -= std::min(end, unstable_tail(full));
            if (!reasoning) end = end > hold_ ? end - hold_ : 0;
        }
        if (end <= sent.size()) return true;
        const std::string piece = full.substr(sent.size(), end - sent.size());
        sent += piece;
        std::string out = head() + "\"choices\":[{\"index\":0,";
        if (req_.chat) {
            out += "\"delta\":{";
            if (!role_sent_) out += "\"role\":\"assistant\",";
            role_sent_ = true;
            out += reasoning ? "\"reasoning_content\":" : "\"content\":";
            jstr(out, piece);
            out += "},";
        } else {
            out += "\"text\":";
            jstr(out, piece);
            out += ',';
        }
        out += "\"finish_reason\":null}]}";
        return send_("data: " + out + "\n\n");
    }
    const tqllm::Tokenizer & tok_;
    const ChatRequest & req_;
    ChatRequest plain_;  // req without tool parsing: intermediate texts
    std::string model_;
    Send send_;
    std::string created_, sr_, sc_;
    size_t hold_ = 0, seen_ = 0;
    bool role_sent_ = false;
};

// np = 1 streaming: begin / advance / finish on the model, `on_tokens` after every step (false: client gone, stop).
// Models without the verify path (no chunked prefill) get the whole answer from decode(), sent at the end.
inline DecodeResult decode_stream(const tqllm::Tokenizer & tokenizer, tqllm::Model & model, const ChatRequest & req,
                                  uint32_t context, const std::function<bool(const std::vector<uint32_t> &)> & on_tokens) {
    if (!model.has_drafter() && !model.incremental()) return decode(tokenizer, model, req, context);
    DecodeResult result;
    uint32_t count = 0;
    PreparedImages images;
    const auto prompt = prepare(tokenizer, req, context, result, count, &images);
    if (!result.error.empty()) return result;
    model.set_sampling(req.temperature, req.top_p, req.top_k, req.seed);
    if (!images.spans.empty()) model.set_prompt_images(images.spans);
    model.spec_begin(prompt, count, draft_length(model.native_mtp()));
    RepeatGuard guard;
    while (model.spec_advance()) {
        const auto ids = model.spec_tokens(0);
        if (guard.check(ids) || !on_tokens(ids)) break;
    }
    uint64_t st[5] = {};
    finish(tokenizer, req, model.spec_finish(st), result);
    set_timings(result, st);
    return result;
}

} // namespace tqserve
