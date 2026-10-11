#include "jinja_template.hpp"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <cmath>
#include <stdexcept>

namespace tqserve {

struct JinjaTemplate::Impl {
    std::string source;
    jinja::program prog;
};

JinjaTemplate::JinjaTemplate(const std::string & source) : impl_(std::make_shared<Impl>()) {
    impl_->source = source;
    jinja::lexer lexer;
    const auto lexed = lexer.tokenize(source);
    impl_->prog = jinja::parse_from_tokens(lexed);
}

static jinja::value to_value(const tqllm::Json & j, bool mark_input) {
    using K = tqllm::Json::Kind;
    switch (j.kind) {
        case K::Null: return jinja::mk_val<jinja::value_none>();
        case K::Bool: return jinja::mk_val<jinja::value_bool>(j.num != 0);
        case K::Number:
            if (std::floor(j.num) == j.num && std::fabs(j.num) < 9.0e15) return jinja::mk_val<jinja::value_int>(static_cast<int64_t>(j.num));
            return jinja::mk_val<jinja::value_float>(j.num);
        case K::String: {
            auto s = jinja::mk_val<jinja::value_string>(j.str);
            if (mark_input) s->mark_input();
            return s;
        }
        case K::Array: {
            auto a = jinja::mk_val<jinja::value_array>();
            for (const auto & item : j.arr) a->push_back(to_value(item, mark_input));
            return a;
        }
        case K::Object: {
            auto o = jinja::mk_val<jinja::value_object>();
            for (const auto & [k, v] : j.obj) o->insert(k, to_value(v, mark_input));
            return o;
        }
    }
    throw std::runtime_error("unsupported JSON value");
}

// OpenAI clients send tool_calls[].function.arguments as a JSON string; templates iterate it as a mapping (arguments|items).
// llama.cpp parses it before rendering, so do the same (a string that is not valid JSON is kept as is).
static tqllm::Json normalise_messages(const tqllm::Json & messages) {
    tqllm::Json out = messages;
    if (out.kind != tqllm::Json::Array) return out;
    for (auto & m : out.arr) {
        if (m.kind != tqllm::Json::Object) continue;
        for (auto & [k, v] : m.obj) {
            if (k != "tool_calls" || v.kind != tqllm::Json::Array) continue;
            for (auto & tc : v.arr) {
                if (tc.kind != tqllm::Json::Object) continue;
                for (auto & [fk, fv] : tc.obj) {
                    if (fk != "function" || fv.kind != tqllm::Json::Object) continue;
                    for (auto & [ak, av] : fv.obj) {
                        if (ak != "arguments" || av.kind != tqllm::Json::String) continue;
                        try {
                            tqllm::Json parsed = tqllm::Parser(av.str).parse();
                            if (parsed.kind == tqllm::Json::Object) av = std::move(parsed);
                        } catch (const std::exception &) {
                        }
                    }
                }
            }
        }
    }
    return out;
}

std::string JinjaTemplate::render(const tqllm::Json & messages, const tqllm::Json * tools, bool add_generation_prompt,
                                  const tqllm::Json * kwargs, const std::string & bos_token, const std::string & eos_token,
                                  bool mark_input) const {
    jinja::context ctx(impl_->source);
    ctx.set_val("messages", to_value(normalise_messages(messages), mark_input));
    ctx.set_val("bos_token", jinja::mk_val<jinja::value_string>(bos_token));
    ctx.set_val("eos_token", jinja::mk_val<jinja::value_string>(eos_token));
    if (tools && tools->kind == tqllm::Json::Array && !tools->arr.empty()) ctx.set_val("tools", to_value(*tools, false));
    if (add_generation_prompt) ctx.set_val("add_generation_prompt", jinja::mk_val<jinja::value_bool>(true));
    if (kwargs && kwargs->kind == tqllm::Json::Object) {
        for (const auto & [k, v] : kwargs->obj) ctx.set_val(k, to_value(v, false));
        // llama.cpp aliases: templates name the same switches differently
        if (const auto * e = kwargs->get("reasoning_effort"); e && e->kind == tqllm::Json::String) ctx.set_val("reasoning_strength", to_value(*e, false));
        if (const auto * p = kwargs->get("preserve_thinking"); p && p->kind == tqllm::Json::Bool) {
            const bool on = p->num != 0;
            ctx.set_val("clear_thinking", jinja::mk_val<jinja::value_bool>(!on));
            ctx.set_val("truncate_history_thinking", jinja::mk_val<jinja::value_bool>(!on));
            ctx.set_val("drop_thinking", jinja::mk_val<jinja::value_bool>(!on));
        }
    }
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(impl_->prog);
    auto parts = jinja::runtime::gather_string_parts(results);
    return parts->as_string().str();
}

}  // namespace tqserve
