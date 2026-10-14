#pragma once
// OpenAI Responses (/v1/responses) and Anthropic Messages (/v1/messages) on top of the chat-completions path: the
// request is rewritten into a chat-completions body (same template, tool parsing, reasoning options), and the
// chat-completions answer (JSON, or the SSE chunks StreamWriter sends) is rewritten into the other API's shape.
// No engine and no CUDA here, so the unit test runs in the CPU build.
#include "json_out.hpp"

#include <atomic>
#include <ctime>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace tqserve {

enum class Api { Chat, Completions, Responses, Messages, None };
// which API a POST path speaks (the server answers every API from the chat-completions path)
inline Api api_for_path(const std::string & path) {
    if (path == "/v1/chat/completions") return Api::Chat;
    if (path == "/v1/completions") return Api::Completions;
    if (path == "/v1/responses") return Api::Responses;
    if (path == "/v1/messages") return Api::Messages;
    return Api::None;
}

namespace adapt {

using tqllm::Json;

inline Json S(const std::string & s) { Json j; j.kind = Json::String; j.str = s; return j; }
inline Json N(double v) { Json j; j.kind = Json::Number; j.num = v; return j; }
inline Json B(bool v) { Json j; j.kind = Json::Bool; j.num = v ? 1 : 0; return j; }
inline Json Nul() { return Json{}; }
inline Json O(std::vector<std::pair<std::string, Json>> kv) { Json j; j.kind = Json::Object; j.obj = std::move(kv); return j; }
inline Json A(std::vector<Json> v = {}) { Json j; j.kind = Json::Array; j.arr = std::move(v); return j; }

inline const Json * field(const Json & o, const char * k) { return o.kind == Json::Object ? o.get(k) : nullptr; }
inline std::string str_of(const Json * v) { return v && v->kind == Json::String ? v->str : std::string(); }
inline bool is_str(const Json * v, const char * s) { return v && v->kind == Json::String && v->str == s; }

// text of a content value: a string, or the text parts of an array (input_text / output_text / text)
inline std::string text_parts(const Json * c) {
    if (!c) return {};
    if (c->kind == Json::String) return c->str;
    std::string out;
    if (c->kind == Json::Array)
        for (const auto & p : c->arr) {
            const auto * t = field(p, "type");
            if (is_str(t, "input_text") || is_str(t, "output_text") || is_str(t, "text")) out += str_of(field(p, "text"));
        }
    return out;
}

// chat content for a user message: text, plus images as chat image_url parts
inline Json chat_content(const Json * c) {
    if (!c || c->kind != Json::Array) return S(text_parts(c));
    std::vector<Json> parts;
    bool image = false;
    for (const auto & p : c->arr) {
        const auto * t = field(p, "type");
        if (is_str(t, "input_image")) {
            const auto * u = field(p, "image_url");
            const std::string url = u && u->kind == Json::Object ? str_of(field(*u, "url")) : str_of(u);
            if (!url.empty()) { parts.push_back(O({{"type", S("image_url")}, {"image_url", O({{"url", S(url)}})}})); image = true; }
        } else if (is_str(t, "image")) {  // Anthropic: {"type":"image","source":{"type":"base64","media_type":..,"data":..}}
            const auto * src = field(p, "source");
            if (src && is_str(field(*src, "type"), "base64")) {
                parts.push_back(O({{"type", S("image_url")}, {"image_url", O({{"url", S("data:" + str_of(field(*src, "media_type")) +
                                                                                       ";base64," + str_of(field(*src, "data")))}})}}));
                image = true;
            } else if (src && is_str(field(*src, "type"), "url")) {
                parts.push_back(O({{"type", S("image_url")}, {"image_url", O({{"url", S(str_of(field(*src, "url")))}})}}));
                image = true;
            }
        } else {
            const auto * tx = field(p, "text");
            if (tx && tx->kind == Json::String) parts.push_back(O({{"type", S("text")}, {"text", *tx}}));
        }
    }
    if (!image) return S(text_parts(c));
    return A(std::move(parts));
}

inline void copy_if(Json & out, const Json & in, const char * from, const char * to) {
    if (const auto * v = field(in, from)) out.obj.emplace_back(to, *v);
}

// Codex / OpenAI efforts onto the server's (low, medium, xhigh; minimal / none = no thinking)
inline void put_effort(Json & out, const std::string & effort) {
    if (effort.empty()) return;
    if (effort == "minimal" || effort == "none") {
        out.obj.emplace_back("chat_template_kwargs", O({{"enable_thinking", B(false)}}));
        return;
    }
    out.obj.emplace_back("reasoning_effort", S(effort == "low" ? "low" : effort == "medium" ? "medium" : "xhigh"));
}

} // namespace adapt

// ---- OpenAI Responses -> chat completions --------------------------------------------------------------------------

// Rewrites a /v1/responses body into a /v1/chat/completions body. Unsupported tool types (web_search, namespaces of
// remote tools, ...) are left out; reasoning items in the input are dropped (the model re-reads the conversation).
inline std::string responses_to_chat(const std::string & body, std::string & error) {
    using namespace adapt;
    Json in;
    try { in = tqllm::parse_json(body); } catch (const std::exception & e) { error = e.what(); return {}; }
    if (in.kind != Json::Object) { error = "body must be a JSON object"; return {}; }
    std::vector<Json> msgs;
    if (const auto * ins = field(in, "instructions"); ins && ins->kind == Json::String && !ins->str.empty())
        msgs.push_back(O({{"role", S("system")}, {"content", *ins}}));
    const auto * input = field(in, "input");
    if (input && input->kind == Json::String) msgs.push_back(O({{"role", S("user")}, {"content", *input}}));
    else if (input && input->kind == Json::Array) {
        for (const auto & it : input->arr) {
            const auto * type = field(it, "type");
            const std::string role = str_of(field(it, "role"));
            if ((!type && !role.empty()) || is_str(type, "message")) {
                const std::string r = role == "developer" || role == "system" ? "system" : role == "assistant" ? "assistant" : "user";
                if (r == "system" && !msgs.empty() && is_str(field(msgs.front(), "role"), "system")) {
                    msgs.front().obj[1].second.str += "\n\n" + text_parts(field(it, "content"));  // one system turn
                    continue;
                }
                if (r == "system") {  // a later developer / system message joins the system prompt (templates want it first)
                    msgs.insert(msgs.begin(), O({{"role", S("system")}, {"content", S(text_parts(field(it, "content")))}}));
                    continue;
                }
                msgs.push_back(O({{"role", S(r)}, {"content", r == "user" ? chat_content(field(it, "content")) : S(text_parts(field(it, "content")))}}));
            } else if (is_str(type, "function_call")) {
                Json call = O({{"id", S(str_of(field(it, "call_id")))}, {"type", S("function")},
                               {"function", O({{"name", S(str_of(field(it, "name")))}, {"arguments", S(str_of(field(it, "arguments")))}})}});
                Json * prev = msgs.empty() ? nullptr : &msgs.back();
                if (prev && is_str(field(*prev, "role"), "assistant")) {
                    Json * calls = nullptr;
                    for (auto & [k, v] : prev->obj) if (k == "tool_calls") calls = &v;
                    if (!calls) { prev->obj.emplace_back("tool_calls", A()); calls = &prev->obj.back().second; }
                    calls->arr.push_back(std::move(call));
                } else {
                    msgs.push_back(O({{"role", S("assistant")}, {"content", S("")}, {"tool_calls", A({std::move(call)})}}));
                }
            } else if (is_str(type, "function_call_output")) {
                msgs.push_back(O({{"role", S("tool")}, {"tool_call_id", S(str_of(field(it, "call_id")))},
                                  {"content", S(text_parts(field(it, "output")))}}));
            }
        }
    }
    if (msgs.empty()) { error = "input must be a string or a list of items"; return {}; }
    Json out = O({{"model", S(str_of(field(in, "model")))}, {"messages", A(std::move(msgs))}});
    if (const auto * t = field(in, "tools"); t && t->kind == Json::Array) {
        std::vector<Json> tools;
        for (const auto & tool : t->arr)
            if (is_str(field(tool, "type"), "function")) {
                Json fn = O({{"name", S(str_of(field(tool, "name")))}, {"description", S(str_of(field(tool, "description")))}});
                if (const auto * p = field(tool, "parameters")) fn.obj.emplace_back("parameters", *p);
                tools.push_back(O({{"type", S("function")}, {"function", std::move(fn)}}));
            }
        if (!tools.empty()) out.obj.emplace_back("tools", A(std::move(tools)));
    }
    if (const auto * tc = field(in, "tool_choice")) {
        if (tc->kind == Json::String) out.obj.emplace_back("tool_choice", *tc);
        else if (is_str(field(*tc, "type"), "function"))
            out.obj.emplace_back("tool_choice", O({{"type", S("function")}, {"function", O({{"name", S(str_of(field(*tc, "name")))}})}}));
    }
    copy_if(out, in, "max_output_tokens", "max_tokens");
    copy_if(out, in, "temperature", "temperature");
    copy_if(out, in, "top_p", "top_p");
    copy_if(out, in, "stream", "stream");
    if (const auto * r = field(in, "reasoning")) put_effort(out, str_of(field(*r, "effort")));
    return dump(out);
}

// ---- chat completions -> OpenAI Responses --------------------------------------------------------------------------

namespace adapt {
inline std::string new_id(const char * prefix) {
    static std::atomic<unsigned long long> n{0};
    char b[64];
    std::snprintf(b, sizeof b, "%s_%llx%06llx", prefix, static_cast<unsigned long long>(std::time(nullptr)), ++n);
    return b;
}
// chat answers number tool calls call_0, call_1 per answer; agent clients need ids unique across the conversation
inline std::string unique_call(const std::string & chat_id) { return new_id("call") + "_" + chat_id; }
inline Json resp_usage(const Json * u) {
    const double in = u ? field(*u, "prompt_tokens") ? field(*u, "prompt_tokens")->num : 0 : 0;
    const double outn = u ? field(*u, "completion_tokens") ? field(*u, "completion_tokens")->num : 0 : 0;
    return O({{"input_tokens", N(in)}, {"input_tokens_details", O({{"cached_tokens", N(0)}})}, {"output_tokens", N(outn)},
              {"output_tokens_details", O({{"reasoning_tokens", N(0)}})}, {"total_tokens", N(in + outn)}});
}
inline Json resp_object(const std::string & id, const std::string & model, const std::string & status, std::vector<Json> output,
                        const Json * usage) {
    Json r = O({{"id", S(id)}, {"object", S("response")}, {"created_at", N(double(std::time(nullptr)))}, {"status", S(status)},
                {"model", S(model)}, {"output", A(std::move(output))}});
    if (status == "incomplete") r.obj.emplace_back("incomplete_details", O({{"reason", S("max_output_tokens")}}));
    r.obj.emplace_back("usage", usage ? resp_usage(usage) : Nul());
    return r;
}
inline Json message_item(const std::string & id, const std::string & text, const char * status) {
    return O({{"type", S("message")}, {"id", S(id)}, {"status", S(status)}, {"role", S("assistant")},
              {"content", A({O({{"type", S("output_text")}, {"text", S(text)}, {"annotations", A()}})})}});
}
inline Json reasoning_item(const std::string & id, const std::string & text) {
    return O({{"type", S("reasoning")}, {"id", S(id)}, {"summary", A({O({{"type", S("summary_text")}, {"text", S(text)}})})}});
}
inline Json call_item(const std::string & id, const std::string & call_id, const std::string & name, const std::string & args,
                      const char * status) {
    return O({{"type", S("function_call")}, {"id", S(id)}, {"call_id", S(call_id)}, {"name", S(name)}, {"arguments", S(args)},
              {"status", S(status)}});
}
} // namespace adapt

// A /v1/chat/completions JSON answer as a /v1/responses one.
inline std::string chat_to_responses(const std::string & chat_json, const std::string & model) {
    using namespace adapt;
    const Json c = tqllm::parse_json(chat_json);
    std::vector<Json> output;
    const auto * ch = field(c, "choices");
    const Json * choice = ch && ch->kind == Json::Array && !ch->arr.empty() ? &ch->arr[0] : nullptr;
    const auto * msg = choice ? field(*choice, "message") : nullptr;
    if (msg) {
        if (const auto r = str_of(field(*msg, "reasoning_content")); !r.empty()) output.push_back(reasoning_item(new_id("rs"), r));
        if (const auto t = str_of(field(*msg, "content")); !t.empty()) output.push_back(message_item(new_id("msg"), t, "completed"));
        if (const auto * calls = field(*msg, "tool_calls"); calls && calls->kind == Json::Array)
            for (const auto & tc : calls->arr) {
                const auto * fn = field(tc, "function");
                output.push_back(call_item(new_id("fc"), unique_call(str_of(field(tc, "id"))), fn ? str_of(field(*fn, "name")) : "",
                                           fn ? str_of(field(*fn, "arguments")) : "", "completed"));
            }
    }
    const bool cut = choice && is_str(field(*choice, "finish_reason"), "length");
    return dump(resp_object(new_id("resp"), model, cut ? "incomplete" : "completed", std::move(output), field(c, "usage")));
}

// Turns the chat SSE chunks of one request into Responses events (event: <type> / data: {...}). Feed every string the
// chat StreamWriter sends; begin() sends response.created first.
class ResponsesSse {
public:
    using Send = std::function<bool(const std::string &)>;
    ResponsesSse(std::string model, Send send) : model_(std::move(model)), send_(std::move(send)), id_(adapt::new_id("resp")) {}
    bool begin() {
        using namespace adapt;
        return ev("response.created", O({{"response", resp_object(id_, model_, "in_progress", {}, nullptr)}})) &&
               ev("response.in_progress", O({{"response", resp_object(id_, model_, "in_progress", {}, nullptr)}}));
    }
    bool feed(const std::string & sse) {
        size_t pos = 0;
        while (pos < sse.size()) {
            size_t end = sse.find("\n\n", pos);
            if (end == std::string::npos) end = sse.size();
            std::string line = sse.substr(pos, end - pos);
            pos = end + 2;
            if (line.rfind("data: ", 0) != 0) continue;
            line = line.substr(6);
            if (line == "[DONE]") continue;
            tqllm::Json d;
            try { d = tqllm::parse_json(line); } catch (...) { continue; }
            if (!chunk(d)) return false;
        }
        return true;
    }

private:
    bool ev(const char * type, tqllm::Json data) {
        data.obj.insert(data.obj.begin(), {"type", adapt::S(type)});
        data.obj.emplace_back("sequence_number", adapt::N(double(seq_++)));
        return send_(std::string("event: ") + type + "\ndata: " + dump(data) + "\n\n");
    }
    bool open_text() {
        using namespace adapt;
        if (text_open_) return true;
        if (!close_reasoning()) return false;
        text_open_ = true;
        msg_id_ = new_id("msg");
        msg_index_ = int(output_.size());
        output_.push_back(Nul());  // filled when the message closes
        return ev("response.output_item.added", O({{"output_index", N(msg_index_)}, {"item", O({{"type", S("message")}, {"id", S(msg_id_)},
                      {"status", S("in_progress")}, {"role", S("assistant")}, {"content", A()}})}})) &&
               ev("response.content_part.added", O({{"item_id", S(msg_id_)}, {"output_index", N(msg_index_)}, {"content_index", N(0)},
                      {"part", O({{"type", S("output_text")}, {"text", S("")}, {"annotations", A()}})}}));
    }
    bool close_text() {
        using namespace adapt;
        if (!text_open_) return true;
        text_open_ = false;
        output_[size_t(msg_index_)] = message_item(msg_id_, text_, "completed");
        return ev("response.output_text.done", O({{"item_id", S(msg_id_)}, {"output_index", N(msg_index_)}, {"content_index", N(0)},
                      {"text", S(text_)}})) &&
               ev("response.content_part.done", O({{"item_id", S(msg_id_)}, {"output_index", N(msg_index_)}, {"content_index", N(0)},
                      {"part", O({{"type", S("output_text")}, {"text", S(text_)}, {"annotations", A()}})}})) &&
               ev("response.output_item.done", O({{"output_index", N(msg_index_)}, {"item", output_[size_t(msg_index_)]}}));
    }
    bool close_reasoning() {
        using namespace adapt;
        if (!reasoning_open_) return true;
        reasoning_open_ = false;
        output_[size_t(rs_index_)] = reasoning_item(rs_id_, reasoning_);
        return ev("response.reasoning_summary_text.done", O({{"item_id", S(rs_id_)}, {"output_index", N(rs_index_)},
                      {"summary_index", N(0)}, {"text", S(reasoning_)}})) &&
               ev("response.output_item.done", O({{"output_index", N(rs_index_)}, {"item", output_[size_t(rs_index_)]}}));
    }
    bool chunk(const tqllm::Json & d) {
        using namespace adapt;
        if (const auto * err = field(d, "error")) {
            failed_ = true;
            return ev("response.failed", O({{"response", O({{"id", S(id_)}, {"object", S("response")}, {"status", S("failed")},
                          {"model", S(model_)}, {"error", O({{"code", S("server_error")}, {"message", S(str_of(field(*err, "message")))}})}})}}));
        }
        const auto * chs = field(d, "choices");
        const Json * ch = chs && chs->kind == Json::Array && !chs->arr.empty() ? &chs->arr[0] : nullptr;
        if (ch) {
            const auto * dl = field(*ch, "delta");
            if (dl) {
                if (const auto r = str_of(field(*dl, "reasoning_content")); !r.empty()) {
                    if (!reasoning_open_) {
                        reasoning_open_ = true;
                        rs_id_ = new_id("rs");
                        rs_index_ = int(output_.size());
                        output_.push_back(Nul());
                        if (!ev("response.output_item.added", O({{"output_index", N(rs_index_)}, {"item", O({{"type", S("reasoning")},
                                    {"id", S(rs_id_)}, {"summary", A()}})}})) ||
                            !ev("response.reasoning_summary_part.added", O({{"item_id", S(rs_id_)}, {"output_index", N(rs_index_)},
                                    {"summary_index", N(0)}, {"part", O({{"type", S("summary_text")}, {"text", S("")}})}})))
                            return false;
                    }
                    reasoning_ += r;
                    if (!ev("response.reasoning_summary_text.delta", O({{"item_id", S(rs_id_)}, {"output_index", N(rs_index_)},
                                {"summary_index", N(0)}, {"delta", S(r)}})))
                        return false;
                }
                if (const auto t = str_of(field(*dl, "content")); !t.empty()) {
                    if (!open_text()) return false;
                    text_ += t;
                    if (!ev("response.output_text.delta", O({{"item_id", S(msg_id_)}, {"output_index", N(msg_index_)},
                                {"content_index", N(0)}, {"delta", S(t)}})))
                        return false;
                }
                if (const auto * calls = field(*dl, "tool_calls"); calls && calls->kind == Json::Array) {
                    if (!close_reasoning() || !close_text()) return false;
                    for (const auto & tc : calls->arr) {
                        const auto * fn = field(tc, "function");
                        const std::string fid = new_id("fc"), call_id = unique_call(str_of(field(tc, "id"))),
                                          name = fn ? str_of(field(*fn, "name")) : "", args = fn ? str_of(field(*fn, "arguments")) : "";
                        const int idx = int(output_.size());
                        output_.push_back(call_item(fid, call_id, name, args, "completed"));
                        if (!ev("response.output_item.added", O({{"output_index", N(idx)}, {"item", call_item(fid, call_id, name, "", "in_progress")}})) ||
                            !ev("response.function_call_arguments.delta", O({{"item_id", S(fid)}, {"output_index", N(idx)}, {"delta", S(args)}})) ||
                            !ev("response.function_call_arguments.done", O({{"item_id", S(fid)}, {"output_index", N(idx)}, {"arguments", S(args)}})) ||
                            !ev("response.output_item.done", O({{"output_index", N(idx)}, {"item", output_.back()}})))
                            return false;
                    }
                }
            }
            if (const auto * fr = field(*ch, "finish_reason"); fr && fr->kind == Json::String) {
                if (!close_reasoning() || !close_text()) return false;
                std::vector<Json> out;
                for (auto & o : output_) if (o.kind != Json::Null) out.push_back(o);
                const bool cut = fr->str == "length";
                return ev(cut ? "response.incomplete" : "response.completed",
                          O({{"response", resp_object(id_, model_, cut ? "incomplete" : "completed", std::move(out), field(d, "usage"))}}));
            }
        }
        return true;
    }
    std::string model_;
    Send send_;
    std::string id_, msg_id_, rs_id_, text_, reasoning_;
    std::vector<tqllm::Json> output_;
    int msg_index_ = -1, rs_index_ = -1;
    unsigned long long seq_ = 0;
    bool text_open_ = false, reasoning_open_ = false, failed_ = false;
};

// ---- Anthropic Messages -> chat completions ----

// tool_use ids: Anthropic clients send our ids back as tool_use_id; strip the prefix chat_to_messages adds
inline std::string messages_tool_id(const std::string & id) { return id.rfind("toolu_", 0) == 0 ? id.substr(6) : id; }

// ---- ------------------------------------------------------------------------

inline std::string messages_to_chat(const std::string & body, std::string & error) {
    using namespace adapt;
    Json in;
    try { in = tqllm::parse_json(body); } catch (const std::exception & e) { error = e.what(); return {}; }
    if (in.kind != Json::Object) { error = "body must be a JSON object"; return {}; }
    std::vector<Json> msgs;
    if (const auto * sys = field(in, "system")) {
        const std::string s = text_parts(sys);
        if (!s.empty()) msgs.push_back(O({{"role", S("system")}, {"content", S(s)}}));
    }
    if (const auto * ms = field(in, "messages"); ms && ms->kind == Json::Array)
        for (const auto & m : ms->arr) {
            const std::string role = str_of(field(m, "role"));
            const auto * c = field(m, "content");
            if (role == "system") {  // Claude Code sends context as system messages between turns: fold into the system prompt
                const std::string t = text_parts(c);
                if (t.empty()) continue;
                if (!msgs.empty() && is_str(field(msgs.front(), "role"), "system")) msgs.front().obj[1].second.str += "\n\n" + t;
                else msgs.insert(msgs.begin(), O({{"role", S("system")}, {"content", S(t)}}));
                continue;
            }
            if (c && c->kind == Json::String) { msgs.push_back(O({{"role", S(role == "assistant" ? "assistant" : "user")}, {"content", *c}})); continue; }
            if (!c || c->kind != Json::Array) continue;
            if (role == "assistant") {
                std::vector<Json> calls;
                for (const auto & b : c->arr)
                    if (is_str(field(b, "type"), "tool_use")) {
                        const auto * inp = field(b, "input");
                        calls.push_back(O({{"id", S(messages_tool_id(str_of(field(b, "id"))))}, {"type", S("function")},
                                           {"function", O({{"name", S(str_of(field(b, "name")))}, {"arguments", S(inp ? dump(*inp) : "{}")}})}}));
                    }
                Json msg = O({{"role", S("assistant")}, {"content", S(text_parts(c))}});
                if (!calls.empty()) msg.obj.emplace_back("tool_calls", A(std::move(calls)));
                msgs.push_back(std::move(msg));
            } else {
                for (const auto & b : c->arr)
                    if (is_str(field(b, "type"), "tool_result"))
                        msgs.push_back(O({{"role", S("tool")}, {"tool_call_id", S(messages_tool_id(str_of(field(b, "tool_use_id"))))},
                                          {"content", S(text_parts(field(b, "content")))}}));
                bool any = false;
                for (const auto & b : c->arr) {
                    const auto * t = field(b, "type");
                    if (is_str(t, "text") || is_str(t, "image")) any = true;
                }
                if (any) msgs.push_back(O({{"role", S("user")}, {"content", chat_content(c)}}));
            }
        }
    if (msgs.empty()) { error = "messages must not be empty"; return {}; }
    Json out = O({{"model", S(str_of(field(in, "model")))}, {"messages", A(std::move(msgs))}});
    copy_if(out, in, "max_tokens", "max_tokens");
    copy_if(out, in, "temperature", "temperature");
    copy_if(out, in, "top_p", "top_p");
    copy_if(out, in, "top_k", "top_k");
    copy_if(out, in, "stop_sequences", "stop");
    copy_if(out, in, "stream", "stream");
    if (const auto * t = field(in, "tools"); t && t->kind == Json::Array) {
        std::vector<Json> tools;
        for (const auto & tool : t->arr) {
            if (!field(tool, "name") || (field(tool, "type") && !is_str(field(tool, "type"), "custom") && !field(tool, "input_schema"))) continue;
            Json fn = O({{"name", S(str_of(field(tool, "name")))}, {"description", S(str_of(field(tool, "description")))}});
            if (const auto * p = field(tool, "input_schema")) fn.obj.emplace_back("parameters", *p);
            tools.push_back(O({{"type", S("function")}, {"function", std::move(fn)}}));
        }
        if (!tools.empty()) out.obj.emplace_back("tools", A(std::move(tools)));
    }
    if (const auto * tc = field(in, "tool_choice")) {
        const std::string type = str_of(field(*tc, "type"));
        if (type == "any") out.obj.emplace_back("tool_choice", S("required"));
        else if (type == "none") out.obj.emplace_back("tool_choice", S("none"));
        else if (type == "tool")
            out.obj.emplace_back("tool_choice", O({{"type", S("function")}, {"function", O({{"name", S(str_of(field(*tc, "name")))}})}}));
    }
    if (const auto * th = field(in, "thinking"))
        if (is_str(field(*th, "type"), "disabled")) out.obj.emplace_back("chat_template_kwargs", O({{"enable_thinking", B(false)}}));
    return dump(out);
}

// ---- chat completions -> Anthropic Messages ------------------------------------------------------------------------

namespace adapt {
inline std::string stop_reason(const std::string & finish, bool tools) {
    if (tools || finish == "tool_calls") return "tool_use";
    return finish == "length" ? "max_tokens" : "end_turn";
}
inline Json anthropic_usage(const Json * u) {
    return O({{"input_tokens", N(u && field(*u, "prompt_tokens") ? field(*u, "prompt_tokens")->num : 0)},
              {"output_tokens", N(u && field(*u, "completion_tokens") ? field(*u, "completion_tokens")->num : 0)}});
}
inline Json tool_input(const std::string & args) {
    try {
        Json j = tqllm::parse_json(args.empty() ? "{}" : args);
        if (j.kind == Json::Object) return j;
    } catch (...) {}
    return O({{"_raw", S(args)}});
}
} // namespace adapt

// {"error":{"message":..,"type":..}} (json_error) in the Anthropic error shape
inline std::string messages_error(const std::string & openai_error) {
    using namespace adapt;
    std::string message = "error", type = "api_error";
    try {
        const Json e = tqllm::parse_json(openai_error);
        if (const auto * er = field(e, "error")) {
            message = str_of(field(*er, "message"));
            if (str_of(field(*er, "type")) == "invalid_request_error") type = "invalid_request_error";
        }
    } catch (...) {}
    return dump(O({{"type", S("error")}, {"error", O({{"type", S(type)}, {"message", S(message)}})}}));
}

inline std::string chat_to_messages(const std::string & chat_json, const std::string & model) {
    using namespace adapt;
    const Json c = tqllm::parse_json(chat_json);
    std::vector<Json> content;
    const auto * ch = field(c, "choices");
    const Json * choice = ch && ch->kind == Json::Array && !ch->arr.empty() ? &ch->arr[0] : nullptr;
    const auto * msg = choice ? field(*choice, "message") : nullptr;
    bool tools = false;
    if (msg) {
        // reasoning is not returned: Anthropic clients expect signed thinking blocks
        if (const auto t = str_of(field(*msg, "content")); !t.empty()) content.push_back(O({{"type", S("text")}, {"text", S(t)}}));
        if (const auto * calls = field(*msg, "tool_calls"); calls && calls->kind == Json::Array)
            for (const auto & tc : calls->arr) {
                const auto * fn = field(tc, "function");
                tools = true;
                content.push_back(O({{"type", S("tool_use")}, {"id", S("toolu_" + unique_call(str_of(field(tc, "id"))))},
                                     {"name", S(fn ? str_of(field(*fn, "name")) : "")}, {"input", tool_input(fn ? str_of(field(*fn, "arguments")) : "")}}));
            }
    }
    return dump(O({{"id", S(new_id("msg"))}, {"type", S("message")}, {"role", S("assistant")}, {"model", S(model)},
                   {"content", A(std::move(content))},
                   {"stop_reason", S(stop_reason(choice ? str_of(field(*choice, "finish_reason")) : "", tools))},
                   {"stop_sequence", Nul()}, {"usage", anthropic_usage(field(c, "usage"))}}));
}

// Turns the chat SSE chunks into Anthropic Messages events.
class MessagesSse {
public:
    using Send = std::function<bool(const std::string &)>;
    MessagesSse(std::string model, Send send) : model_(std::move(model)), send_(std::move(send)) {}
    bool begin() {
        using namespace adapt;
        return ev("message_start", O({{"message", O({{"id", S(new_id("msg"))}, {"type", S("message")}, {"role", S("assistant")},
                  {"model", S(model_)}, {"content", A()}, {"stop_reason", Nul()}, {"stop_sequence", Nul()},
                  {"usage", O({{"input_tokens", N(0)}, {"output_tokens", N(0)}})}})}}));
    }
    bool feed(const std::string & sse) {
        size_t pos = 0;
        while (pos < sse.size()) {
            size_t end = sse.find("\n\n", pos);
            if (end == std::string::npos) end = sse.size();
            std::string line = sse.substr(pos, end - pos);
            pos = end + 2;
            if (line.rfind("data: ", 0) != 0) continue;
            line = line.substr(6);
            if (line == "[DONE]") continue;
            tqllm::Json d;
            try { d = tqllm::parse_json(line); } catch (...) { continue; }
            if (!chunk(d)) return false;
        }
        return true;
    }

private:
    bool ev(const char * type, tqllm::Json data) {
        data.obj.insert(data.obj.begin(), {"type", adapt::S(type)});
        return send_(std::string("event: ") + type + "\ndata: " + dump(data) + "\n\n");
    }
    bool close_block() {
        using namespace adapt;
        if (open_.empty()) return true;
        open_.clear();
        return ev("content_block_stop", O({{"index", N(idx_)}}));
    }
    bool open_block(const std::string & kind, tqllm::Json block) {
        using namespace adapt;
        if (open_ == kind) return true;
        if (!close_block()) return false;
        open_ = kind;
        ++idx_;
        return ev("content_block_start", O({{"index", N(idx_)}, {"content_block", std::move(block)}}));
    }
    bool chunk(const tqllm::Json & d) {
        using namespace adapt;
        if (const auto * err = field(d, "error"))
            return ev("error", O({{"error", O({{"type", S("api_error")}, {"message", S(str_of(field(*err, "message")))}})}}));
        const auto * chs = field(d, "choices");
        const Json * ch = chs && chs->kind == Json::Array && !chs->arr.empty() ? &chs->arr[0] : nullptr;
        if (!ch) return true;
        if (const auto * dl = field(*ch, "delta")) {
            if (const auto t = str_of(field(*dl, "content")); !t.empty()) {
                if (!open_block("text", O({{"type", S("text")}, {"text", S("")}})) ||
                    !ev("content_block_delta", O({{"index", N(idx_)}, {"delta", O({{"type", S("text_delta")}, {"text", S(t)}})}})))
                    return false;
            }
            if (const auto * calls = field(*dl, "tool_calls"); calls && calls->kind == Json::Array)
                for (const auto & tc : calls->arr) {
                    const auto * fn = field(tc, "function");
                    tools_ = true;
                    if (!close_block()) return false;
                    open_ = "tool";
                    ++idx_;
                    if (!ev("content_block_start", O({{"index", N(idx_)}, {"content_block", O({{"type", S("tool_use")},
                                {"id", S("toolu_" + unique_call(str_of(field(tc, "id"))))}, {"name", S(fn ? str_of(field(*fn, "name")) : "")}, {"input", O({})}})}})) ||
                        !ev("content_block_delta", O({{"index", N(idx_)}, {"delta", O({{"type", S("input_json_delta")},
                                {"partial_json", S(fn ? str_of(field(*fn, "arguments")) : "")}})}})))
                        return false;
                }
        }
        if (const auto * fr = field(*ch, "finish_reason"); fr && fr->kind == Json::String) {
            if (!close_block()) return false;
            const auto * u = field(d, "usage");
            return ev("message_delta", O({{"delta", O({{"stop_reason", S(stop_reason(fr->str, tools_))}, {"stop_sequence", Nul()}})},
                          {"usage", O({{"output_tokens", N(u && field(*u, "completion_tokens") ? field(*u, "completion_tokens")->num : 0)}})}})) &&
                   ev("message_stop", O({}));
        }
        return true;
    }
    std::string model_;
    Send send_;
    std::string open_;
    int idx_ = -1;
    bool tools_ = false;
};

} // namespace tqserve
