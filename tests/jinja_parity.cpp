// test-jinja: renders the shared chat-template cases (tests/template/cases.json) through llama.cpp's Jinja engine (vendored) with
// the Qwen3.8 chat template and compares with tqserve's native renderer (tqserve::parse_request). Host-only, no CUDA.
// usage: test-jinja [template.jinja] [cases.json] [dump_dir]   (dump_dir: also writes each Jinja render, for tests/template/compare_jinja.py)
#include "tools/serve/jinja_template.hpp"
#include "tools/serve/session.hpp"

#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

static std::string slurp(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static size_t first_diff(const std::string & a, const std::string & b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
    return i;
}

// The native renderer follows the community "froggeric v22" Qwen template for tool prompts and inline <think> in assistant turns;
// the model's own shipped template (what llama.cpp renders) does not. These cases are expected to differ and are reported only.
static bool known_template_difference(const std::string & name) {
    return name == "D_tools" || name == "E_tool_multiturn" || name == "E2_tool_dict_args_2calls" || name == "H_inline_think";
}

int main(int argc, char ** argv) {
    const std::string tpl_path = argc > 1 ? argv[1] : "tests/template/qwen38_chat_template.jinja";
    const std::string cases_path = argc > 2 ? argv[2] : "tests/template/cases.json";
    tqserve::JinjaTemplate tpl(slurp(tpl_path));
    const std::string cases_text = slurp(cases_path);
    tqllm::Json cases = tqllm::Parser(cases_text).parse();
    int bad = 0;
    for (const auto & [name, body_json] : cases.obj) {
        // re-serialise the case body for parse_request
        std::string body;
        {
            struct W {
                static void w(std::string & o, const tqllm::Json & j) {
                    switch (j.kind) {
                        case tqllm::Json::Null: o += "null"; break;
                        case tqllm::Json::Bool: o += j.num != 0 ? "true" : "false"; break;
                        case tqllm::Json::Number: { std::ostringstream n; n << j.num; o += n.str(); break; }
                        case tqllm::Json::String: tqserve::jstr(o, j.str); break;
                        case tqllm::Json::Array: o += '['; for (size_t i = 0; i < j.arr.size(); ++i) { if (i) o += ','; w(o, j.arr[i]); } o += ']'; break;
                        case tqllm::Json::Object: o += '{'; for (size_t i = 0; i < j.obj.size(); ++i) { if (i) o += ','; tqserve::jstr(o, j.obj[i].first); o += ':'; w(o, j.obj[i].second); } o += '}'; break;
                    }
                }
            };
            W::w(body, body_json);
        }
        const auto native = tqserve::parse_request(body, 512, "", true, true);
        if (!native.ok) { std::cout << name << ": native parse failed: " << native.error << "\n"; ++bad; continue; }
        const tqllm::Json * messages = body_json.get("messages");
        const tqllm::Json * tools = body_json.get("tools");
        tqllm::Json kwargs;
        kwargs.kind = tqllm::Json::Object;
        if (const auto * kw = body_json.get("chat_template_kwargs")) kwargs = *kw;
        if (const auto * e = body_json.get("reasoning_effort")) kwargs.obj.push_back({"reasoning_effort", *e});
        std::string rendered;
        try {
            rendered = tpl.render(*messages, tools, true, &kwargs);
        } catch (const std::exception & e) {
            std::cout << name << ": jinja render failed: " << e.what() << "\n";
            ++bad;
            continue;
        }
        if (argc > 3) { std::ofstream(std::string(argv[3]) + "/" + name + ".txt", std::ios::binary) << rendered; }
        if (rendered == native.prompt_text) {
            std::cout << name << ": identical (" << rendered.size() << " bytes)\n";
        } else {
            const size_t d = first_diff(rendered, native.prompt_text);
            const bool known = known_template_difference(name);
            std::cout << name << (known ? ": differs (known: native follows froggeric v22)" : ": DIFFERENT") << " at byte " << d << "\n  jinja : ..." << rendered.substr(d > 40 ? d - 40 : 0, 120)
                      << "\n  native: ..." << native.prompt_text.substr(d > 40 ? d - 40 : 0, 120) << "\n";
            if (!known) ++bad;
        }
    }
    {  // parse_request with a renderer returns the Jinja render (what tqserve --jinja does), with enable_thinking from the request
        tqserve::PromptRenderer renderer = [&](const tqllm::Json & messages, const tqllm::Json * tools, const tqllm::Json & kwargs) {
            return tpl.render(messages, tools, true, &kwargs);
        };
        const std::string body = R"({"messages":[{"role":"user","content":"What is 2+2?"}],"chat_template_kwargs":{"enable_thinking":false}})";
        const auto viaparse = tqserve::parse_request(body, 512, "", true, true, "xhigh", &renderer, nullptr);
        tqllm::Json kw;
        kw.kind = tqllm::Json::Object;
        tqllm::Json f;
        f.kind = tqllm::Json::Bool;
        f.num = 0;
        kw.obj.push_back({"enable_thinking", f});
        const auto parsed_root = tqllm::parse_json(body);
        const std::string expect = tpl.render(*parsed_root.get("messages"), nullptr, true, &kw);
        const bool ok = viaparse.ok && viaparse.prompt_text == expect && !viaparse.thinking;
        std::cout << "parse_request with a Jinja renderer: " << (ok ? "identical" : "DIFFERENT") << "\n";
        bad += !ok;
        const auto sysparse = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}]})", 512, "Be brief.", true, true, "xhigh", &renderer, nullptr);
        const bool sys_ok = sysparse.ok && sysparse.prompt_text.find("Be brief.") != std::string::npos;
        std::cout << "server --system reaches the Jinja template: " << (sys_ok ? "yes" : "NO") << "\n";
        bad += !sys_ok;
    }
    std::cout << (bad ? "FAIL" : "PASS") << ": " << cases.obj.size() << " cases, " << bad << " differ\n";
    return bad ? 1 : 0;
}
