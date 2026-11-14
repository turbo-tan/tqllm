// Copy drafts beside the DFlash2 token tree: a reply that repeats a long stretch of its prompt verifies the copy (up to
// 16 rows), and the text stays the greedy text (the chain verify without copies) in fewer passes than the chain's 8
// tokens per pass. Model test: TQLLM_TEST_MODEL (a qwen35 GGUF) and TQLLM_TEST_DRAFTER (its DFlash2 drafter);
// TQLLM_TEST_DRAFT_VOCAB is optional; skipped without them.
#include "tqllm/model.hpp"
#include "tqllm/options.hpp"
#include "tqllm/tokenizer.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> run(tqllm::Model & m, const std::vector<uint32_t> & in, uint32_t count, uint64_t & steps) {
    m.spec_begin(in, count, 7);
    while (m.spec_advance()) {}
    uint64_t st[5]{};
    auto out = m.spec_finish(st);
    steps = st[0];
    return out;
}

int main() {
    const char * model = std::getenv("TQLLM_TEST_MODEL");
    const char * drafter = std::getenv("TQLLM_TEST_DRAFTER");
    const char * vocab = std::getenv("TQLLM_TEST_DRAFT_VOCAB");
    if (!model || !drafter) { std::cout << "copy_first: skipped (set TQLLM_TEST_MODEL and TQLLM_TEST_DRAFTER)\n"; return 77; }
    try {
        tqllm::Tokenizer tok;
        tok.load(model);
        std::string body;
        for (int i = 0; i < 24; ++i)
            body += "def f" + std::to_string(i) + "(x):\n    return x * " + std::to_string(i * 7 + 3) + " + " +
                    std::to_string(i % 5) + "\n\n";
        const auto prompt = tok.encode("<|im_start|>user\nRepeat this code exactly, with no comment before or after it:\n\n" +
                                       body + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        const uint32_t count = 400, context = 8192;
        tqllm::EngineOptions o;
        o.draft_model = drafter;
        if (vocab) o.draft_vocab = vocab;
        o.tree = false;
        uint64_t chain_steps = 0, tree_steps = 0;
        std::vector<uint32_t> ref, got;
        {
            tqllm::Model m(model, model, context, 1, o);
            ref = run(m, prompt, count, chain_steps);
        }
        o.tree = true;
        {
            tqllm::Model m(model, model, context, 1, o);
            got = run(m, prompt, count, tree_steps);
        }
        // sampled at top_k = 1 (the argmax at every position): the same text, and the copy must still save passes when
        // the drafter's sibling holds the copy's first token (two equal children of the root)
        uint64_t sampled_steps = 0;
        std::vector<uint32_t> sgot;
        {
            tqllm::Model m(model, model, context, 1, o);
            m.set_sampling(0.6f, 1.0f, 1, 7);
            sgot = run(m, prompt, count, sampled_steps);
        }
        std::cout << "copy_first: " << ref.size() << " tokens, chain " << chain_steps << " passes, tree + copy " << tree_steps
                  << " passes, sampled top_k 1 " << sampled_steps << " passes\n";
        if (sgot != ref) {
            std::cout << "copy_first: FAIL: sampled top_k = 1 differs from the greedy text\n";
            return 1;
        }
#if defined(__APPLE__)
        // Metal: the chain already fills a verbatim repeat (no fewer passes possible); the copy must still take the block
        if (sampled_steps > tree_steps || double(sgot.size()) / double(std::max<uint64_t>(sampled_steps, 1)) < 10) {
            std::cout << "copy_first: FAIL: sampled, the copy lost the block (more passes than greedy tree + copy, or < 10 per pass)\n";
            return 1;
        }
#else
        if (sampled_steps >= chain_steps) {
            std::cout << "copy_first: FAIL: sampled, the copy saved no passes (the walk took the drafter's equal sibling)\n";
            return 1;
        }
#endif
        if (got != ref) {
            const size_t at = size_t(std::mismatch(ref.begin(), ref.end(), got.begin(), got.end()).first - ref.begin());
            std::cout << "copy_first: FAIL: tree + copy differs from the chain text at token " << at << "\n";
            return 1;
        }
#if defined(__APPLE__)
        // Metal: the chain already verifies up to 15 drafted tokens and fills a verbatim repeat; the copy must not cost passes
        if (tree_steps > chain_steps) {
            std::cout << "copy_first: FAIL: tree + copy took more passes than the chain\n";
            return 1;
        }
#else
        // the chain verifies at most 8 tokens per pass; a copy fills the 16-row block
        if (tree_steps >= chain_steps) {
            std::cout << "copy_first: FAIL: the copy saved no passes\n";
            return 1;
        }
#endif
        // a verbatim repeat fills the 16-row block: a drafter alone stays well below 10 tokens per pass here
        const double per_pass = double(got.size()) / double(std::max<uint64_t>(tree_steps, 1));
        std::cout << "copy_first: " << per_pass << " tokens per tree pass\n";
        if (per_pass < 10) {
            std::cout << "copy_first: FAIL: under 10 tokens per pass on a verbatim repeat (no copy drafts?)\n";
            return 1;
        }
        std::cout << "copy_first: PASS (tree + copy == chain greedy, fewer passes)\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "copy_first: FAIL: " << e.what() << "\n";
        return 1;
    }
}
