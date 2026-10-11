// GGUF-embedded tokenizer: a mock GGUF checks loading, merges, specials and errors; with TQLLM_TOK_GGUF and
// TQLLM_TOK_TQTOK set, a real model's GGUF metadata must give the same ids as its tokenizer.tqtok.
#include "tqllm/tokenizer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

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

std::string mock(const std::string & dir, const std::string & pre) {
    // byte-level pieces: ' ' is U+0120 ("\xc4\xa0")
    GgufWriter w;
    w.kv_str("general.architecture", "qwen35");
    w.kv_str("tokenizer.ggml.model", "gpt2");
    w.kv_str("tokenizer.ggml.pre", pre);
    w.kv_strs("tokenizer.ggml.tokens", {"h", "e", "l", "o", "\xc4\xa0", "he", "ll", "hell", "hello", "<|im_end|>", "[PAD10]", "\xc4\xa0h", "<think>"});
    w.kv_i32s("tokenizer.ggml.token_type", {1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5, 1, 4});
    w.kv_strs("tokenizer.ggml.merges", {"h e", "l l", "he ll", "hell o", "\xc4\xa0 h"});
    w.kv_u32("tokenizer.ggml.eos_token_id", 9);
    const auto path = dir + "/mock_" + pre + ".gguf";
    w.write(path);
    return path;
}

std::string join(const std::vector<uint32_t> & v) {
    std::string s;
    for (auto x : v) s += std::to_string(x) + " ";
    return s;
}

const char * kTexts[] = {
    "Hello, world!", "<|im_start|>user\nWhat is 2 + 2?<|im_end|>", "Caf\xc3\xa9 \xe2\x80\x94 \xe6\x9d\xb1\xe4\xba\xac \xf0\x9f\x98\x8a",
    "I'm testing\n whitespace  and\tpunctuation!!!", "  and", "foo123",
    "\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87 \xe0\xa4\xa6\xe0\xa5\x81\xe0\xa4\xa8\xe0\xa4\xbf\xe0\xa4\xaf\xe0\xa4\xbe",
    "Ti\xe1\xba\xbfng Vi\xe1\xbb\x87t c\xc3\xb3 d\xe1\xba\xa5u", "<|im_start|>assistant\n<think>\n\n</think>\n\nOK<tool_call>{}</tool_call>",
    "x\xcc\x81y", "int f() {\n    if (x) {\n        return 1;\n    }\n}\n", "a   \n\n   b\t\t\tc", "trailing   ",
    "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n",
};
}

int main() {
    const auto dir = std::filesystem::temp_directory_path() / ("tqllm_tok_gguf_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);

    tqllm::Tokenizer t;
    t.load(mock(dir.string(), "qwen35"));
    check(t.eos_id() == 9, "eos from tokenizer.ggml.eos_token_id");
    check(t.encode("hello") == std::vector<uint32_t>{8}, "merges reach 'hello': " + join(t.encode("hello")));
    check(t.encode("hel") == std::vector<uint32_t>{5, 2}, "partial merge 'hel': " + join(t.encode("hel")));
    check(t.encode("hello<|im_end|>") == std::vector<uint32_t>{8, 9}, "control token matched atomically");
    check(t.encode("<think>hello") == std::vector<uint32_t>{12, 8}, "user-defined token matched atomically");
    check(t.encode("hello hello") == std::vector<uint32_t>{8, 4, 8}, "space prefix piece: " + join(t.encode("hello hello")));
    bool pad_special = false;  // the mock vocab has no '[' byte, so a non-special [PAD10] cannot encode at all
    try { pad_special = t.encode("[PAD10]") == std::vector<uint32_t>{10}; } catch (const std::exception &) {}
    check(!pad_special, "unused padding token is not matched as a special");
    check(t.decode({8, 9, 8}) == "hellohello", "decode skips specials: " + t.decode({8, 9, 8}));

    bool threw = false;
    try { tqllm::Tokenizer bad; bad.load(mock(dir.string(), "llama-bpe")); } catch (const std::exception &) { threw = true; }
    check(threw, "unsupported pre-tokenizer is rejected");

    const char * gguf = std::getenv("TQLLM_TOK_GGUF");
    const char * tqtok = std::getenv("TQLLM_TOK_TQTOK");
    if (gguf && tqtok) {
        tqllm::Tokenizer a, b;
        a.load(gguf); b.load(tqtok);
        check(a.eos_id() == b.eos_id(), "real eos ids match");
        size_t n = 0;
        for (const char * text : kTexts) {
            const auto x = a.encode(text), y = b.encode(text);
            check(x == y, std::string("real ids differ on: ") + text + "\n  gguf  " + join(x) + "\n  tqtok " + join(y));
            check(a.decode(x) == b.decode(y), std::string("real decode differs on: ") + text);
            n += x.size();
        }
        std::cout << "real parity: " << std::size(kTexts) << " texts, " << n << " tokens\n";
    } else {
        std::cout << "real parity skipped (set TQLLM_TOK_GGUF and TQLLM_TOK_TQTOK)\n";
    }
    std::filesystem::remove_all(dir);
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "tokenizer_gguf: ok\n";
    return 0;
}
