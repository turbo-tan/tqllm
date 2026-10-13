// Native MTP drafting (--mtp, the model's own nextn head) gives the greedy output of plain decoding, token for token:
// a model without --mtp produces the references, then one with --mtp (same KV, prefill buffers, context) must match
// them at draft depths 1 and 3, keep a stable streamed prefix and count drafts consistently.
#include "tqllm/model.hpp"
#include "tqllm/options.hpp"
#include "tqllm/tokenizer.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: test-native-mtp WEIGHTS TOKENIZER");
        tqllm::Tokenizer tok;
        tok.load(argv[2]);
        std::vector<std::string> prompts{
            "Write a Python function that checks whether a number is prime.",
            "Explain how a hash map works.",
            "Two trains travel 300 km toward each other at 60 and 90 km/h. When do they meet?"};
        std::string context;
        for (int i = 0; i < 40; ++i)
            context += "A hash map resolves key collisions with probing and grows when its load factor increases.\n";
        prompts.push_back(context + "Summarise the data structure in two sentences.");
        std::vector<std::vector<uint32_t>> inputs;
        for (const auto & p : prompts)
            inputs.push_back(tok.encode("<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        const std::vector<uint32_t> counts{1u, 97u};
        tqllm::EngineOptions base;
        base.ubatch = 512;
        std::vector<std::vector<uint32_t>> refs;
        {
            tqllm::Model ar(argv[1], argv[2], 8192, 1, base);
            for (const auto & in : inputs)
                for (uint32_t count : counts) {
                    ar.spec_begin(in, count, 3);
                    while (ar.spec_advance()) {}
                    uint64_t st[5]{};
                    refs.push_back(ar.spec_finish(st));
                }
        }
        for (int lookup : {0, 16}) {  // MTP alone, then MTP with prompt lookup (prompt 3 repeats a sentence 40 times)
        tqllm::EngineOptions with = base;
        with.native_mtp = true;
        with.lookup = lookup;
        tqllm::Model model(argv[1], argv[2], 8192, 1, with);
        size_t r = 0;
        for (size_t p = 0; p < inputs.size(); ++p)
            for (uint32_t count : counts) {
                const auto & ref = refs[r++];
                for (int depth : {1, 3}) {
                    model.spec_begin(inputs[p], count, depth);
                    auto prefix = model.spec_tokens();
                    std::vector<char> done;
                    do {
                        model.spec_step({0}, done);
                        const auto next = model.spec_tokens();
                        if (next.size() < prefix.size() || !std::equal(prefix.begin(), prefix.end(), next.begin()))
                            throw std::runtime_error("stream prefix changed");
                        prefix = next;
                    } while (!done[0]);
                    uint64_t stats[5]{};
                    const auto mtp = model.spec_finish(stats);
                    if (mtp != ref) {
                        const auto mm = std::mismatch(ref.begin(), ref.end(), mtp.begin(), mtp.end());
                        std::cerr << "FAIL lookup " << lookup << " prompt " << p << " count " << count << " depth " << depth << " first diff "
                                  << mm.first - ref.begin() << '\n';
                        return 1;
                    }
                    if (prefix != mtp || mtp.empty() || mtp.size() > count || !stats[4])
                        throw std::runtime_error("invalid output or prefill statistics");
                    if (count == 1 && (stats[0] || stats[1] || stats[2]))
                        throw std::runtime_error("one-token request drafted unnecessarily");
                    if ((!lookup && stats[1] != stats[0] * uint64_t(depth)) || stats[2] > stats[1])
                        throw std::runtime_error("invalid acceptance statistics");
                    std::cout << "PASS lookup " << lookup << " prompt " << p << " count " << count << " depth " << depth << " tokens " << mtp.size()
                              << " accepted " << stats[2] << '/' << stats[1] << '\n' << std::flush;
                }
            }
        }
        std::cout << "all native MTP tests passed\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
