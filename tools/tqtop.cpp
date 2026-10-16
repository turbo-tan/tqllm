// tqtop: top-k logits at a chosen position, for interpreting a spec-vs-non-spec divergence.
// Renders the model's chat template with the user prompt, appends a generated-prefix string,
// runs one forward and prints the top-k (token, logit) at the last position. Run with
// TQLLM_DEBUG=x3n1x3v to compare the exact x3v n==1 path against the default (sq) path.
//
// Usage: tqtop --model M.gguf --prompt TEXT [--prefix TEXT | --prefix-file F] [--topk 5]
//              [--no-template] [--context N]
#include "tqllm/model.hpp"
#include "tqllm/options.hpp"
#include "tqllm/tokenizer.hpp"

#include "src/gguf.hpp"
#include "src/json.hpp"
#include "jinja_template.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqtop");
    try {
        std::string model, prompt, prefix, prefix_file;
        int topk = 5;
        uint32_t context = 4096;
        bool use_template = true;
        tqllm::EngineOptions engine;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto value = [&]() -> std::string { if (++i >= argc) throw std::invalid_argument("missing value for " + a); return argv[i]; };
            if (a == "--model") model = value();
            else if (a == "--prompt") prompt = value();
            else if (a == "--prefix") prefix = value();
            else if (a == "--prefix-file") { std::ifstream in(value()); prefix.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()); }
            else if (a == "--topk") topk = std::stoi(value());
            else if (a == "--context") context = std::stoul(value());
            else if (a == "--no-template") use_template = false;
            else if (tqllm::parse_engine_option(argc, argv, i, engine)) continue;
            else throw std::invalid_argument("unknown argument " + a);
        }
        if (model.empty()) throw std::invalid_argument("--model is required");

        std::string text = prompt;
        if (use_template) {
            const auto g = tqllm::gguf_open(model);
            const auto it = g.kv.find("tokenizer.chat_template");
            if (it == g.kv.end()) throw std::runtime_error("no tokenizer.chat_template in " + model + " (use --no-template)");
            tqserve::JinjaTemplate tpl(it->second.str);
            tqllm::Json role; role.kind = tqllm::Json::String; role.str = "user";
            tqllm::Json content; content.kind = tqllm::Json::String; content.str = prompt;
            tqllm::Json m; m.kind = tqllm::Json::Object; m.obj = {{"role", role}, {"content", content}};
            tqllm::Json msgs; msgs.kind = tqllm::Json::Array; msgs.arr.push_back(std::move(m));
            text = tpl.render(msgs, nullptr, true, nullptr);
        }
        text += prefix;

        tqllm::Tokenizer tok;
        tok.load(model);
        tqllm::Model mdl(model, model, context, 1, engine);
        mdl.set_full_logits(true);
        const std::vector<uint32_t> ids = tok.encode(text);

        // Logits at the last position via the verify path (capture() needs TQ3_4S/BF16 weights).
        const size_t V = mdl.vocab_size();
        std::vector<float> logits(V);
        const int B = 8;
        for (size_t pos = 0; pos < ids.size(); pos += B) {
            const int n = int(std::min<size_t>(B, ids.size() - pos));
            std::vector<uint32_t> blk(ids.begin() + pos, ids.begin() + pos + n), next;
            mdl.verify(blk, uint32_t(pos), next);
            if (pos + size_t(n) == ids.size() &&
                cudaMemcpy(logits.data(), mdl.verify_logits() + size_t(n - 1) * V, V * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("cudaMemcpy of verify logits failed");
            mdl.commit(n);
        }

        std::vector<int> idx(logits.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = int(i);
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        std::printf("pos=%zu mode=%s top%d:\n", ids.size(), std::getenv("TQLLM_DEBUG") ? std::getenv("TQLLM_DEBUG") : "default", topk);
        for (int i = 0; i < topk; ++i)
            std::printf("  %2d  id=%-7d logit=%+9.4f  %s\n", i + 1, idx[i], logits[idx[i]], tok.decode({uint32_t(idx[i])}).c_str());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "tqtop: %s\n", e.what());
        return 1;
    }
}
