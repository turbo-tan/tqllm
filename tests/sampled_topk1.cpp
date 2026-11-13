// A sampled request at top_k = 1 draws the argmax at every position, so its text must equal the greedy text: DFlash2
// chain, DFlash2 token tree and native MTP. 1.6.0 broke it from the first token: a chat prompt's last turn is a prefill
// chunk of its own (a few tokens), and the sampled first token was drawn from that chunk's first position instead of the
// last prompt position (PR #42). Model test: TQLLM_TEST_MODEL (a qwen35 GGUF) and TQLLM_TEST_DRAFTER (its DFlash2
// drafter); TQLLM_TEST_DRAFT_VOCAB is optional; skipped without the model and drafter.
#include "tqllm/model.hpp"
#include "tqllm/options.hpp"
#include "tqllm/tokenizer.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> run(tqllm::Model & m, const std::vector<uint32_t> & in, uint32_t count) {
    m.spec_begin(in, count, 7);
    while (m.spec_advance()) {}
    uint64_t st[5]{};
    return m.spec_finish(st);
}

int main() {
    const char * model = std::getenv("TQLLM_TEST_MODEL");
    const char * drafter = std::getenv("TQLLM_TEST_DRAFTER");
    const char * vocab = std::getenv("TQLLM_TEST_DRAFT_VOCAB");
    if (!model || !drafter) { std::puts("sampled_topk1: skipped (set TQLLM_TEST_MODEL and TQLLM_TEST_DRAFTER)"); return 77; }
    try {
        tqllm::Tokenizer tok;
        tok.load(model);
        const std::vector<std::string> turns{"Explain how a hash map resolves collisions, in three sentences.",
                                             "Write a Python function that returns the n-th Fibonacci number."};
        std::vector<std::vector<uint32_t>> prompts;
        for (const auto & t : turns)
            prompts.push_back(tok.encode("<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\n" + t +
                                         "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        const uint32_t count = 96;
        const uint32_t context = 8192;
        tqllm::EngineOptions base;
        base.draft_model = drafter;
        if (vocab) base.draft_vocab = vocab;
        base.tree = false;
        std::vector<std::vector<uint32_t>> refs;
        {
            tqllm::Model greedy(model, model, context, 1, base);
            for (const auto & p : prompts) refs.push_back(run(greedy, p, count));
        }
        struct Case { const char * name; bool tree, mtp; };
        int failures = 0, ran = 0;
        for (const Case c : {Case{"DFlash2 chain", false, false}, Case{"DFlash2 tree", true, false}, Case{"native MTP", false, true}}) {
            tqllm::EngineOptions o = base;
            o.tree = c.tree;
            if (c.mtp) { o.draft_model.clear(); o.draft_vocab.clear(); o.native_mtp = true; }
            std::unique_ptr<tqllm::Model> m;
            try {
                m = std::make_unique<tqllm::Model>(model, model, context, 1, o);
            } catch (const std::exception & e) {
                if (!c.mtp) throw;
                std::cout << "SKIP " << c.name << ": " << e.what() << "\n";  // a model without an MTP head
                continue;
            }
            m->set_sampling(0.6f, 1.0f, 1, 7);
            for (size_t i = 0; i < prompts.size(); ++i) {
                const auto got = run(*m, prompts[i], count);
                ++ran;
                if (got == refs[i]) continue;
                const size_t at = size_t(std::mismatch(refs[i].begin(), refs[i].end(), got.begin(), got.end()).first - refs[i].begin());
                std::cout << "FAIL " << c.name << " prompt " << i << ": top_k = 1 differs from greedy at token " << at << " of "
                          << refs[i].size() << "\n";
                ++failures;
            }
            if (!failures) std::cout << "PASS " << c.name << "\n";
        }
        if (!ran) throw std::runtime_error("no case ran");
        std::cout << (failures ? "sampled_topk1: FAIL" : "sampled_topk1: PASS (top_k = 1 == greedy: chain, tree, MTP)") << "\n";
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::cerr << "sampled_topk1: FAIL: " << e.what() << "\n";
        return 1;
    }
}
