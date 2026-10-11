#pragma once
// Chat template rendering with llama.cpp's Jinja engine (vendored in third_party/jinja, MIT, (c) the ggml authors).
// The native Qwen renderer in session.hpp stays the default; this path renders any template from the GGUF or a file,
// like llama.cpp --jinja, and is the reference the native renderer is tested against.
#include "src/json.hpp"

#include <memory>
#include <string>

namespace tqserve {

class JinjaTemplate {
public:
    explicit JinjaTemplate(const std::string & source);  // parses; throws std::runtime_error on a syntax error
    // messages / tools: JSON arrays (tools may be null); kwargs: extra template variables (chat_template_kwargs, e.g.
    // enable_thinking, reasoning_effort, preserve_thinking). mark_input marks user strings so special tokens in them are not parsed.
    std::string render(const tqllm::Json & messages, const tqllm::Json * tools, bool add_generation_prompt,
                       const tqllm::Json * kwargs, const std::string & bos_token = "", const std::string & eos_token = "",
                       bool mark_input = false) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace tqserve
