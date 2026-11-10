// tqserve's unified-memory start-up guard (src/core/load_guard.hpp): model sizes from files, manifests and split GGUFs,
// and the refusal rule. .44 (GB10, 11-08) hung when a second model loaded beside one holding most of the memory.
#include "load_guard.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
static int failures = 0;
static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
static void write(const fs::path & p, size_t bytes) { std::ofstream(p, std::ios::binary) << std::string(bytes, 'x'); }

int main() {
    const fs::path d = fs::temp_directory_path() / ("tqllm-load-guard-" + std::to_string(std::rand()));
    fs::create_directories(d);
    write(d / "m-00001-of-00002.gguf", 1000);
    write(d / "m-00002-of-00002.gguf", 500);
    write(d / "other-00001-of-00003.gguf", 7);  // another split model in the same folder: not counted
    write(d / "mtp.gguf", 30);
    write(d / "single.gguf", 42);
    std::ofstream(d / "fn.tqfn") << "# manifest\ngguf=m-00001-of-00002.gguf\ngguf=m-00002-of-00002.gguf\nmtp=mtp.gguf\n"
                                    "draft_vocab=vocab.txt\nslots=512\nbudget_gb=60\n";

    check(tqllm::model_files_bytes((d / "single.gguf").string()) == 42, "plain GGUF: its size");
    check(tqllm::model_files_bytes((d / "m-00001-of-00002.gguf").string()) == 1500, "split GGUF: both shards, not the other model");
    check(tqllm::model_files_bytes((d / "fn.tqfn").string()) == 1530, "manifest: gguf shards + mtp, not the vocab");
    check(tqllm::model_files_bytes((d / "missing.gguf").string()) == 0, "missing file: 0 (no guard)");
    check(tqllm::model_files_bytes("") == 0, "no drafter: 0");

    const uint64_t G = uint64_t(1) << 30;
    check(tqllm::unified_memory_refusal(110 * G, 85 * G).empty(), "idle GB10, Flash-Next: loads");
    check(!tqllm::unified_memory_refusal(60 * G, 85 * G).empty(), "another 53 GB model running: refused");
    check(tqllm::unified_memory_refusal(0, 0).empty(), "unknown size: no guard");
    check(tqllm::unified_memory_refusal(60 * G, 85 * G).find("85 GB") != std::string::npos, "message names the sizes");

    fs::remove_all(d);
    if (failures) return 1;
    std::puts("load_guard: ok");
    return 0;
}
