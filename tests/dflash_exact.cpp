// DFlash2 drafting (tree and --no-tree) gives the greedy output of plain decoding, token for token.
// Reproduces the 10-13 D1 finding: a 3.7K-token code-review prompt answered differently with the drafter (release
// ea03c5f too). A model without -md produces the reference; models with the drafter (tree, then linear) must match it.
//   test-dflash-exact WEIGHTS TOKENIZER DRAFTER DRAFT_VOCAB CODE_FILE [TOKENS]
#include "tqllm/model.hpp"
#include "tqllm/options.hpp"
#include "tqllm/tokenizer.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> run(tqllm::Model & m, const std::vector<uint32_t> & in, uint32_t count, int depth) {
    m.spec_begin(in, count, depth);
    while (m.spec_advance()) {}
    uint64_t st[5]{};
    return m.spec_finish(st);
}

int main(int argc, char ** argv) {
    try {
        if (argc < 6) throw std::runtime_error("usage: test-dflash-exact WEIGHTS TOKENIZER DRAFTER DRAFT_VOCAB CODE_FILE [TOKENS]");
        const uint32_t count = argc > 6 ? uint32_t(std::stoul(argv[6])) : 512;
        std::ifstream f(argv[5]);
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string code = ss.str().substr(0, 12000);
        tqllm::Tokenizer tok;
        tok.load(argv[2]);
        const auto in = tok.encode("<|im_start|>user\nReview this C++ code: list concrete bugs, risks and improvements, quoting the lines.\n\n```cpp\n" +
                                   code + "\n```<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        tqllm::EngineOptions base;
        base.ubatch = 512;
        std::vector<uint32_t> ref;
        {
            tqllm::Model ar(argv[1], argv[2], 16384, 1, base);
            ref = run(ar, in, count, 7);
        }
        std::cout << "prompt " << in.size() << " tokens, reference " << ref.size() << " tokens\n";
        int failures = 0;
        for (bool tree : {true, false}) {
            tqllm::EngineOptions o = base;
            o.draft_model = argv[3];
            o.draft_vocab = argv[4];
            o.tree = tree;
            tqllm::Model m(argv[1], argv[2], 16384, 1, o);
            const auto got = run(m, in, count, 7);
            const auto mm = std::mismatch(ref.begin(), ref.end(), got.begin(), got.end());
            if (got == ref) {
                std::cout << "PASS " << (tree ? "tree" : "linear") << ": " << got.size() << " tokens identical\n";
                continue;
            }
            ++failures;
            const size_t i = size_t(mm.first - ref.begin());
            std::cout << "FAIL " << (tree ? "tree" : "linear") << ": first diff at output token " << i << " (reference "
                      << (i < ref.size() ? int64_t(ref[i]) : -1) << " '" << (i < ref.size() ? tok.decode({ref[i]}) : "")
                      << "', drafter " << (i < got.size() ? int64_t(got[i]) : -1) << " '" << (i < got.size() ? tok.decode({got[i]}) : "")
                      << "'), text before: '" << tok.decode(std::vector<uint32_t>(ref.begin() + std::max<ptrdiff_t>(0, ptrdiff_t(i) - 12), ref.begin() + ptrdiff_t(i))) << "'\n";
        }
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
