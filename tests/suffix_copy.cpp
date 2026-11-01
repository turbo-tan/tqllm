// Copy drafts from the context (src/spec/suffix_copy.hpp): what is proposed, when nothing is, and what is never indexed.
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "suffix_copy.hpp"

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}
using Tokens = std::vector<uint32_t>;
tqllm::SuffixCopy make(int key, int min_match, const Tokens& context) {
    tqllm::SuffixCopyOptions o;
    o.key = key;
    o.min_match = min_match;
    tqllm::SuffixCopy s(o);
    s.append(context.data(), context.size());
    return s;
}
}  // namespace

int main() {
    {   // the tokens after an earlier repeat, capped at max_tokens
        auto s = make(2, 3, {7, 1, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3});
        expect(s.propose(3) == Tokens({4, 5, 6}), "copy after the repeat");
        expect(s.last_match() == 3, "matched length");
        expect(s.propose(1) == Tokens({4}), "cap");
    }
    {   // below min_match: nothing
        auto s = make(2, 4, {7, 1, 2, 3, 4, 5, 6, 8, 9, 1, 2, 3});
        expect(s.propose(3).empty() && s.last_match() == 0, "short repeat rejected");
    }
    {   // the longest repeat wins over a more recent shorter one
        auto s = make(2, 2, {9, 1, 2, 3, 10, 11, 8, 2, 3, 20, 21, 9, 1, 2, 3});
        expect(s.propose(2) == Tokens({10, 11}), "longest repeat");
        expect(s.last_match() == 4, "longest length");
    }
    {   // equally long repeats: the most recent
        auto s = make(2, 2, {5, 1, 2, 30, 6, 1, 2, 40, 7, 1, 2});
        expect(s.propose(1) == Tokens({40}), "most recent of equals");
    }
    {   // a repeat running into the suffix continues with its period
        auto s = make(2, 4, {5, 6, 7, 5, 6, 7, 5, 6});
        expect(s.propose(5) == Tokens({7, 5, 6, 7, 5}), "periodic continuation");
    }
    {   // propose() never indexes its own draft; append is the only way in
        auto s = make(2, 3, {1, 2, 3, 4, 1, 2, 3});
        const size_t before = s.size();
        expect(s.propose(2) == Tokens({4, 1}), "copy");
        expect(s.size() == before, "draft not indexed");
        s.append(9);  // the target rejected the copy and committed 9 instead
        expect(s.propose(2).empty(), "no repeat after the rejection");
    }
    {   // incremental append == one-shot append; reset forgets
        tqllm::SuffixCopyOptions o;
        o.key = 3;
        o.min_match = 5;
        tqllm::SuffixCopy a(o), b(o);
        Tokens text;
        for (int r = 0; r < 3; ++r)
            for (uint32_t t : {11u, 12u, 13u, 14u, 15u, 16u, 17u}) text.push_back(t + uint32_t(r == 1 ? 100 : 0));
        text.insert(text.end(), {11, 12, 13, 14, 15});
        for (uint32_t t : text) a.append(t);
        b.append(text.data(), text.size());
        expect(a.propose(4) == b.propose(4) && a.propose(4) == Tokens({16, 17, 11, 12}), "incremental == batch");
        a.reset();
        expect(a.size() == 0 && a.propose(4).empty(), "reset");
    }
    {   // a context no longer than min_match cannot repeat; bad options are refused
        auto s = make(2, 3, {1, 2, 1});
        expect(s.propose(4).empty() && s.propose(0).empty(), "too short / zero tokens");
        bool threw = false;
        try {
            tqllm::SuffixCopyOptions o;
            o.key = 8;
            o.min_match = 5;
            tqllm::SuffixCopy bad(o);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "min_match below key refused");
    }
    if (failures) return 1;
    std::printf("suffix_copy: ok\n");
    return 0;
}
