// Tokenizer::encode(text, Specials::user_only, offsets): control tokens written in the text are tokenized as text, the
// user-defined ones still match atomically (HF split_special_tokens=True, how pplx-embed-v2-context builds its document
// input: "[D] " is split, "<|chunk_sep|>" is one token), and every id comes with the byte range of the text it covers.
#include "tqllm/tokenizer.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string & what) {
    if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}
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
std::string join(const std::vector<uint32_t> & v) { std::string s; for (auto x : v) s += std::to_string(x) + " "; return s; }
}  // namespace

int main() {
    char tmpl[] = "/tmp/tqllm-tokspec-XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    GgufWriter w;
    w.kv_str("general.architecture", "qwen35");
    w.kv_str("tokenizer.ggml.model", "gpt2");
    w.kv_str("tokenizer.ggml.pre", "qwen35");
    //            0    1    2    3    4    5    6         7          8       9       10
    w.kv_strs("tokenizer.ggml.tokens", {"a", "b", "[", "]", "D", "\xc4\xa0", "<|sep|>", "[D] ", "\xc4\xa0" "a", "<", "|"});
    w.kv_i32s("tokenizer.ggml.token_type", {1, 1, 1, 1, 1, 1, 4, 3, 1, 1, 1});
    w.kv_strs("tokenizer.ggml.merges", {"\xc4\xa0 a"});
    w.kv_u32("tokenizer.ggml.eos_token_id", 6);
    const std::string path = dir + "/mock.gguf";
    w.write(path);
    tqllm::Tokenizer t;
    t.load(path);
    const std::string text = "[D] a<|sep|>b a";   // "[D] " is a control token, "<|sep|>" user-defined
    check(join(t.encode(text)) == "7 0 6 1 8 ", "encode(): every added token matched: " + join(t.encode(text)));
    std::vector<std::pair<uint32_t, uint32_t>> off;
    const auto ids = t.encode(text, tqllm::Tokenizer::Specials::user_only, &off);
    check(join(ids) == "2 4 3 8 6 1 8 ", "user_only: the control token is text, the user-defined one matches: " + join(ids));
    const std::vector<std::pair<uint32_t, uint32_t>> want = {{0, 1}, {1, 2}, {2, 3}, {3, 5}, {5, 12}, {12, 13}, {13, 15}};
    check(off == want, "user_only: byte ranges per id");
    // Specials::all with offsets runs the offsets implementation (without offsets it is encode() itself)
    std::vector<std::pair<uint32_t, uint32_t>> off_all;
    const auto all_ids = t.encode(text, tqllm::Tokenizer::Specials::all, &off_all);
    check(join(all_ids) == "7 0 6 1 8 ", "Specials::all with offsets: both added tokens matched: " + join(all_ids));
    const std::vector<std::pair<uint32_t, uint32_t>> want_all = {{0, 4}, {4, 5}, {5, 12}, {12, 13}, {13, 15}};
    check(off_all == want_all, "Specials::all: byte ranges per id");
    std::remove(path.c_str()); rmdir(dir.c_str());
    std::cout << (failures ? "tokenizer specials: FAILED\n" : "tokenizer specials: OK\n");
    return failures ? 1 : 0;
}
