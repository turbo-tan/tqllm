// Inkling-2-Mini prompt path as tqserve runs it: the checkpoint's chat_template.jinja rendered by the vendored Jinja engine
// (tqserve --jinja), tokenized by tqllm::Tokenizer from the Hugging Face folder, must give the input_ids the transformers
// reference built for the 5 golden greedy prompts (golden/greedy.json; reasoning_effort as recorded, the template's
// default when none). Needs INKLING2_DIR (the checkpoint folder with golden/); skipped otherwise.
#include "jinja_template.hpp"
#include "nlohmann/json.hpp"
#include "tqllm/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string slurp(const std::string & p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
std::string quote(const std::string & s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
        else if (c == '\n') o += "\\n";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += char(c);
    }
    return o + "\"";
}
}  // namespace

int main() try {
    const char * dir = std::getenv("INKLING2_DIR");
    if (!dir) { std::printf("INKLING2_DIR not set: skipped\n"); return 77; }
    const std::string d = dir;
    tqserve::JinjaTemplate tpl(slurp(d + "/chat_template.jinja"));
    tqllm::Tokenizer tok;
    tok.load(d);
    const tqllm::Json golden = tqllm::parse_json(slurp(d + "/golden/greedy.json"));
    int bad = 0, n = 0;
    for (const auto & g : golden.arr) {
        const tqllm::Json messages = tqllm::parse_json("[{\"role\":\"user\",\"content\":" + quote(g.get("prompt")->str) + "}]");
        tqllm::Json kwargs = tqllm::parse_json("{}");
        const tqllm::Json * eff = g.get("reasoning_effort");
        if (eff && eff->kind == tqllm::Json::String) kwargs = tqllm::parse_json("{\"reasoning_effort\":\"" + eff->str + "\"}");
        const std::string text = tpl.render(messages, nullptr, true, &kwargs);
        const auto ids = tok.encode(text);
        std::vector<uint32_t> ref;
        for (const auto & x : g.get("input_ids")->arr) ref.push_back(uint32_t(x.num));
        size_t first = 0;
        while (first < ids.size() && first < ref.size() && ids[first] == ref[first]) ++first;
        const bool same = ids == ref;
        bad += !same; ++n;
        std::printf("%-12s %s: %zu tokens (reference %zu)", g.get("name")->str.c_str(), same ? "equal" : "DIFFER", ids.size(), ref.size());
        if (!same) std::printf(", first difference at %zu (ours %u, reference %u)", first, first < ids.size() ? ids[first] : 0u,
                               first < ref.size() ? ref[first] : 0u);
        std::printf("\n");
    }
    std::printf(bad ? "inkling2_prompt: FAIL (%d of %d differ)\n" : "inkling2_prompt: PASS (%d of %d differ)\n", bad, n);
    return bad ? 1 : 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "inkling2_prompt: %s\n", e.what());
    return 1;
}
