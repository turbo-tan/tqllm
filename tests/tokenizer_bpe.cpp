// BPE on untrusted text: the merge result equals the reference algorithm (lowest rank first, leftmost on ties), a
// 300,000-byte piece with no pre-token break tokenizes in seconds (the merge was quadratic), an empty user-defined
// token does not stop encode() from advancing, and an overfull piece cache keeps most of its entries.
#include "tqllm/tokenizer.hpp"

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string & what) { if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; } }
struct GgufWriter {
    std::string body; uint64_t n = 0;
    template <class T> static void put(std::string & s, T v) { s.append(reinterpret_cast<const char *>(&v), sizeof(v)); }
    static void str(std::string & s, const std::string & x) { put<uint64_t>(s, x.size()); s += x; }
    void kv_str(const std::string & k, const std::string & v) { str(body, k); put<uint32_t>(body, 8); str(body, v); ++n; }
    void kv_u32(const std::string & k, uint32_t v) { str(body, k); put<uint32_t>(body, 4); put<uint32_t>(body, v); ++n; }
    void kv_strs(const std::string & k, const std::vector<std::string> & v) {
        str(body, k); put<uint32_t>(body, 9); put<uint32_t>(body, 8); put<uint64_t>(body, v.size());
        for (const auto & x : v) str(body, x);
        ++n;
    }
    void kv_i32s(const std::string & k, const std::vector<int32_t> & v) {
        str(body, k); put<uint32_t>(body, 9); put<uint32_t>(body, 5); put<uint64_t>(body, v.size());
        for (auto x : v) put<int32_t>(body, x);
        ++n;
    }
    void write(const std::string & path) const {
        std::string h = "GGUF"; put<uint32_t>(h, 3); put<uint64_t>(h, 0); put<uint64_t>(h, n);
        std::ofstream(path, std::ios::binary) << h << body;
    }
};

const std::vector<std::string> kTokens = {"a", "b", "ab", "abab", "ba", "aa", "aab", "bb", "", "<|x|>"};
const std::vector<std::string> kMerges = {"a b", "ab ab", "b a", "a a", "aa b", "b b"};  // rank = position

// the reference: repeatedly merge the lowest-ranked adjacent pair, the leftmost among equals
std::vector<uint32_t> reference(const std::string & piece) {
    std::map<std::string, uint32_t> rank, id;
    for (size_t i = 0; i < kMerges.size(); ++i) {
        const auto sp = kMerges[i].find(' ');
        rank[kMerges[i].substr(0, sp) + '\0' + kMerges[i].substr(sp + 1)] = uint32_t(i);
    }
    for (size_t i = 0; i < kTokens.size(); ++i) id[kTokens[i]] = uint32_t(i);
    std::vector<std::string> s;
    for (char c : piece) s.push_back(std::string(1, c));
    while (s.size() > 1) {
        uint32_t best = UINT32_MAX; size_t at = s.size();
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            const auto it = rank.find(s[i] + '\0' + s[i + 1]);
            if (it != rank.end() && it->second < best) { best = it->second; at = i; }
        }
        if (at == s.size()) break;
        s[at] += s[at + 1];
        s.erase(s.begin() + long(at) + 1);
    }
    std::vector<uint32_t> out;
    for (const auto & x : s) out.push_back(id.at(x));
    return out;
}
}  // namespace

int main() {
    alarm(120);  // a hang (the empty-token loop, the quadratic merge) fails the test instead of the ctest timeout
    char tmpl[] = "/tmp/tqllm-tokbpe-XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    GgufWriter w;
    w.kv_str("general.architecture", "qwen35");
    w.kv_str("tokenizer.ggml.model", "gpt2");
    w.kv_str("tokenizer.ggml.pre", "qwen35");
    w.kv_strs("tokenizer.ggml.tokens", kTokens);
    w.kv_i32s("tokenizer.ggml.token_type", {1, 1, 1, 1, 1, 1, 1, 1, 4, 4});  // "" and "<|x|>" user-defined
    w.kv_strs("tokenizer.ggml.merges", kMerges);
    w.kv_u32("tokenizer.ggml.eos_token_id", 9);
    const std::string path = dir + "/mock.gguf";
    w.write(path);
    tqllm::Tokenizer t;
    t.load(path);

    std::mt19937 rng(7);
    int mismatches = 0;
    for (int n = 0; n < 3000; ++n) {
        std::string s(1 + rng() % 120, 'a');
        for (auto & c : s) c = "ab"[rng() % 2];
        if (t.encode(s) != reference(s) && !mismatches++) std::cerr << "first mismatch: " << s << "\n";
    }
    check(mismatches == 0, std::to_string(mismatches) + " of 3000 random pieces differ from the reference merge");

    std::string big(300000, 'a');
    for (auto & c : big) c = "ab"[rng() % 2];
    const auto t0 = std::chrono::steady_clock::now();
    const auto ids = t.encode(big);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("300,000-byte piece: %zu ids in %.2f s\n", ids.size(), secs);
    check(secs < 3, "a 300,000-byte piece tokenizes in under 3 s");
    const std::string head = big.substr(0, 3000);
    check(t.encode(head) == reference(head), "a 3,000-byte piece equals the reference merge");

    check(t.encode("ab<|x|>ba") == std::vector<uint32_t>{2, 9, 4}, "user-defined token matched, the empty one ignored");
    std::vector<std::pair<uint32_t, uint32_t>> off;
    check(t.encode("ab<|x|>ba", tqllm::Tokenizer::Specials::user_only, &off) == std::vector<uint32_t>{2, 9, 4}, "user_only path too");

    // piece cache: more distinct pieces than it holds must not empty it
    const size_t cap = t.piece_cache_capacity();
    for (size_t i = 0; i < cap + cap / 4; ++i) {
        std::string s;
        for (size_t v = i + 1; v; v >>= 1) s += "ab"[v & 1];
        t.encode(s + "a");
    }
    std::printf("piece cache: %zu of %zu after %zu distinct pieces\n", t.piece_cache_size(), cap, cap + cap / 4);
    check(t.piece_cache_size() >= cap / 2, "an overfull piece cache keeps at least half its entries");
    std::remove(path.c_str()); rmdir(dir.c_str());
    std::cout << (failures ? "tokenizer bpe: FAILED\n" : "tokenizer bpe: OK\n");
    return failures ? 1 : 0;
}
