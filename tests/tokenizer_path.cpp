// tqserve and tqkld pick the tokenizer the same way: --tokenizer, else the GGUF model itself.
#include "../src/tokenizer_path.hpp"

#include <cstdio>

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char * what) { std::printf("%-56s %s\n", what, ok ? "ok" : "FAIL"); failures += !ok; };
    check(tqllm::tokenizer_for("m.gguf", "t.tqtok") == "t.tqtok", "--tokenizer wins");
    check(tqllm::tokenizer_for("/models/Qwen3.8-27B-TQ_XS.gguf", "") == "/models/Qwen3.8-27B-TQ_XS.gguf", "GGUF without --tokenizer: the GGUF");
    check(tqllm::tokenizer_for("model.safetensors", "").empty(), "non-GGUF without --tokenizer: none");
    check(tqllm::tokenizer_for(".gguf", "").empty(), "bare extension is not a model");
    check(tqllm::tokenizer_for("", "").empty(), "nothing given: none");
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::puts("tokenizer_path: ok");
    return 0;
}
