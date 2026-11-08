// Native MTP drafting (--mtp, the model's own nextn head) gives the greedy output of plain decoding, token for token:
// a model without --mtp produces the references, then one with --mtp (same KV, prefill buffers, context) must match
// them at draft depths 1 and 3, keep a stable streamed prefix and count drafts consistently.
#include "tqllm/model.hpp"
#include "tqllm/options.hpp"
#include "tqllm/tokenizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
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
        // Sampled requests (temperature > 0) keep MTP drafting: the drafts are argmax tokens, so accept_drafts gives exact
        // samples of the target's distribution. Checks: (a) a sampled request with --mtp drafts (before, sampled requests
        // silently fell back to one token per step); (b) the first token, drawn from the prefill head before any draft,
        // equals plain decoding's for the same seed; (c) at every later position the token distribution over many seeds
        // matches plain sampled decoding: its total-variation distance to plain decoding stays within the distance between
        // two independent plain runs (the sampling-noise floor) plus a small margin. An MTP path that kept the argmax
        // drafts whenever they were proposed would pull positions 1.. toward the mode and fail (c).
        {
            const auto in = tok.encode("<|im_start|>user\nWrite one short sentence about the sea.<|im_end|>\n"
                                       "<|im_start|>assistant\n<think>\n\n</think>\n\n");
            const int seeds = 400;
            const uint32_t n = 6;
            using Hist = std::vector<std::map<uint32_t, int>>;
            auto run = [&](tqllm::Model & m, uint64_t seed, uint64_t * st) {
                m.set_sampling(1.0f, 1.0f, 0, seed);
                m.spec_begin(in, n, 3);
                while (m.spec_advance()) {}
                return m.spec_finish(st);
            };
            auto add = [&](Hist & h, const std::vector<uint32_t> & o) {
                for (size_t i = 0; i < o.size() && i < h.size(); ++i) ++h[i][o[i]];
            };
            auto tv_at = [&](const Hist & a, const Hist & b, size_t i) {
                double d = 0, na = 0, nb = 0;
                for (const auto & [k, c] : a[i]) na += c;
                for (const auto & [k, c] : b[i]) nb += c;
                if (!na || !nb) return 0.0;
                for (const auto & [k, c] : a[i]) d += std::abs(c / na - (b[i].count(k) ? b[i].at(k) / nb : 0.0));
                for (const auto & [k, c] : b[i]) if (!a[i].count(k)) d += c / nb;
                return d / 2;
            };
            Hist plain_a(n), plain_b(n), mtp_h(n);
            std::vector<uint32_t> first_plain;
            {
                tqllm::Model ar(argv[1], argv[2], 8192, 1, base);
                for (int s = 0; s < seeds; ++s) {
                    uint64_t st[5]{};
                    const auto o = run(ar, uint64_t(s + 1), st);
                    first_plain.push_back(o.empty() ? UINT32_MAX : o[0]);
                    add(plain_a, o);
                    add(plain_b, run(ar, uint64_t(seeds + s + 1), st));
                }
            }
            tqllm::EngineOptions with = base;
            with.native_mtp = true;
            tqllm::Model model(argv[1], argv[2], 8192, 1, with);
            uint64_t drafted = 0, accepted = 0, first_diff = 0;
            for (int s = 0; s < seeds; ++s) {
                uint64_t st[5]{};
                const auto o = run(model, uint64_t(s + 1), st);
                if (o.empty() || o.size() > n) throw std::runtime_error("sampled MTP: invalid output length");
                drafted += st[1];
                accepted += st[2];
                first_diff += o[0] != first_plain[size_t(s)];
                add(mtp_h, o);
            }
            std::cout << "sampled MTP: drafted " << drafted << ", accepted " << accepted << ", first-token mismatches "
                      << first_diff << '\n';
            bool dist_ok = true;
            for (size_t i = 1; i < n; ++i) {
                const double noise = tv_at(plain_a, plain_b, i), got = tv_at(plain_a, mtp_h, i);
                std::cout << "  position " << i << ": TV(plain, mtp) " << got << ", TV(plain, plain) " << noise << ", distinct "
                          << plain_a[i].size() << '\n';
                if (got > noise + 0.08) dist_ok = false;
            }
            if (!drafted) throw std::runtime_error("sampled MTP: no drafts (sampled requests fell back to plain decoding)");
            if (first_diff) throw std::runtime_error("sampled MTP: first token differs from plain decoding for the same seed");
            if (!dist_ok) throw std::runtime_error("sampled MTP: token distribution differs from plain sampled decoding");
            std::cout << "PASS sampled MTP\n";
        }
        std::cout << "all native MTP tests passed\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
