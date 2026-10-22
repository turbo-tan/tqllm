#include "gguf_mapped.hpp"
#include "qwen4exp.hpp"
#include "dequant.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#ifndef _WIN32
#include <sys/resource.h>
#include <unistd.h>
#endif

using namespace tqllm;
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
template<class T> static void put(std::vector<char> & b, T v) {
    const auto p = reinterpret_cast<const char *>(&v); b.insert(b.end(), p, p + sizeof(v));
}
static void text(std::vector<char> & b, const std::string & s) {
    put<uint64_t>(b, s.size()); b.insert(b.end(), s.begin(), s.end());
}
static void unit_test() {
    std::vector<char> header = {'G', 'G', 'U', 'F'};
    put<uint32_t>(header, 3); put<uint64_t>(header, 2); put<uint64_t>(header, 0);
    text(header, "experts"); put<uint32_t>(header, 3);
    for (uint64_t d : {2u, 3u, 4u}) put<uint64_t>(header, d);
    put<uint32_t>(header, 0); put<uint64_t>(header, 0);
    text(header, "embeddings"); put<uint32_t>(header, 2); put<uint64_t>(header, 32); put<uint64_t>(header, 5);
    put<uint32_t>(header, 8); put<uint64_t>(header, 96);
    header.resize((header.size() + 31) / 32 * 32);
    std::vector<char> payload;
    for (int expert = 0; expert < 4; ++expert) for (int row = 0; row < 3; ++row) for (int col = 0; col < 2; ++col)
        put<float>(payload, float(expert * 100 + row * 10 + col));
    for (int row = 0; row < 5; ++row) {
        put<uint16_t>(payload, 0x3c00); // FP16 scale 1.
        for (int col = 0; col < 32; ++col) put<int8_t>(payload, int8_t(row * 10 + col - 16));
    }
    const auto path = std::filesystem::temp_directory_path() / ("tqllm-map-" + std::to_string(getpid()) + ".gguf");
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove(path); } } cleanup{path};
    { std::ofstream out(path, std::ios::binary); out.write(header.data(), header.size()); out.write(payload.data(), payload.size()); }
    MappedGguf mapped(path.string());
    require(mapped.tensor("experts").size == 96, "mapped tensor size differs");
    for (int expert = 0; expert < 4; ++expert) {
        const auto view = mapped.expert("experts", expert);
        require(view.size == 24, "expert mapping length differs");
        float values[6]; std::memcpy(values, view.data, sizeof(values));
        for (int i = 0; i < 6; ++i) require(values[i] == expert * 100 + (i / 2) * 10 + i % 2, "wrong expert mapped");
    }
    const auto rows = mapped.gather_rows("embeddings", {4, 1, 4, 0});
    for (int i = 0; i < 4; ++i) for (int col = 0; col < 32; ++col)
        require(rows[i * 32 + col] == (i == 0 || i == 2 ? 40 : i == 1 ? 10 : 0) + col - 16,
                "Q8 gather lost row order, signs, scale or duplicates");
    {   // a prefill-sized gather runs on several threads: same rows in the same order
        std::vector<uint64_t> many(5000); for (size_t i = 0; i < many.size(); ++i) many[i] = (i * 7 + 3) % 5;
        const auto big = mapped.gather_rows("embeddings", many);
        for (size_t i = 0; i < many.size(); ++i) for (int col = 0; col < 32; ++col)
            require(big[i * 32 + col] == float(many[i] * 10) + col - 16, "threaded gather lost row order");
    }
    require(mapped.gather_rows("embeddings", {}).empty(), "empty gather not empty");
    bool rejected = false;
    try { mapped.expert("experts", 4); } catch (const std::exception &) { rejected = true; }
    require(rejected, "out-of-range expert accepted");
    rejected = false;
    try { mapped.gather_rows("embeddings", {5}); } catch (const std::exception &) { rejected = true; }
    require(rejected, "out-of-range embedding row accepted");
    rejected = false;
    try { mapped.gather_rows("experts", {0}); } catch (const std::exception &) { rejected = true; }
    require(rejected, "3-D tensor treated as row table");
    std::puts("read-only GGUF mappings, distinct experts, selected Q8 rows, bounds and duplicates: PASS");
}
static void gsq_dialect_test() {
    std::vector<char> header = {'G', 'G', 'U', 'F'};
    put<uint32_t>(header, 3); put<uint64_t>(header, 1); put<uint64_t>(header, 0);
    text(header, "experts"); put<uint32_t>(header, 3);
    for (uint64_t d : {64u, 1u, 2u}) put<uint64_t>(header, d);
    put<uint32_t>(header, 42); put<uint64_t>(header, 0);
    header.resize((header.size() + 31) / 32 * 32);
    std::vector<char> payload;
    for (uint16_t scale : {0x3800u, 0xbc00u}) {
        put<uint16_t>(payload, scale);
        for (int j = 0; j < 16; ++j) put<uint8_t>(payload, 0xe4); // codes 0,1,2,3
    }
    const auto path = std::filesystem::temp_directory_path() / ("tqllm-gsq-" + std::to_string(getpid()) + ".gguf");
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::filesystem::remove(p); } } cleanup{path};
    { std::ofstream out(path, std::ios::binary); out.write(header.data(), header.size()); out.write(payload.data(), payload.size()); }
    bool rejected = false;
    try { MappedGguf native(path.string()); } catch (const std::exception &) { rejected = true; }
    require(rejected, "default dialect silently accepted ambiguous type42");
    MappedGguf gsq(path.string(), GgufDialect::GsqRco);
    require(gsq.file().tensors.at("experts").type == GGML_GSQ_Q2_0, "GSQ dialect did not resolve type42 explicitly");
    { std::ifstream in(path, std::ios::binary); uint32_t disk_type = 0;
      in.seekg(gsq.file().tensors.at("experts").type_header_offset); in.read(reinterpret_cast<char *>(&disk_type), 4);
      require(disk_type == 42, "GSQ header type offset differs or source file was rewritten"); }
    for (int expert = 0; expert < 2; ++expert) {
        const auto raw = gsq.expert("experts", expert); require(raw.size == 18, "Q2_0 slice bytes differ");
        float values[64]; dequantize_host(GGML_GSQ_Q2_0, raw.data, values, 64);
        for (int j = 0; j < 64; ++j)
            require(values[j] == (j % 4 - 1) * (expert ? -1.f : .5f), "Q2_0 code order or signed scale differs");
    }
    uint64_t block = 0, bytes = 0;
    require(!gguf_block_layout(42, block, bytes), "ambiguous external id42 leaked into default layout table");
    require(gguf_block_layout(GGML_GSQ_Q2_0, block, bytes) && block == 64 && bytes == 18, "Q2_0 block geometry differs");
    std::filesystem::resize_file(path, header.size() + payload.size() - 1);
    rejected = false;
    try { MappedGguf truncated(path.string(), GgufDialect::GsqRco); } catch (const std::exception &) { rejected = true; }
    require(rejected, "truncated GSQ payload accepted");
    std::puts("explicit GSQ dialect, type42 fail closed, Q2_0 code/scale/slice/truncation: PASS");
}
static void checkpoint_probe(int argc, char ** argv) {
    std::vector<std::unique_ptr<MappedGguf>> files;
    for (int i = 1; i < argc; ++i) files.push_back(std::make_unique<MappedGguf>(argv[i]));
    const auto config = qwen4exp_config(files[0]->file());
    PleTokenHistory history(config.ple, 128);
    const auto indices = history.append_batch({785, 1273, 1206, 3992});
    std::vector<uint64_t> ids(indices.begin(), indices.end());
    bool found_table = false, found_expert = false;
    for (const auto & file : files) {
        const auto & metadata = file->file();
        if (metadata.tensors.count("per_layer_token_embd.weight")) {
            const auto & t = metadata.tensors.at("per_layer_token_embd.weight");
            const auto gathered = file->gather_rows("per_layer_token_embd.weight", ids);
            for (size_t i = 0; i < ids.size(); ++i) {
                auto row = t;
                row.ne[1] = 1; row.bytes = t.bytes / t.ne[1]; row.offset += ids[i] * row.bytes;
                const auto raw = metadata.read(row);
                std::vector<float> expected(t.ne[0]);
                dequantize_host(t.type, raw.data(), expected.data(), t.ne[0]);
                require(std::memcmp(expected.data(), gathered.data() + i * t.ne[0], expected.size() * sizeof(float)) == 0,
                        "mapped PLE gather differs from independent file reads");
            }
            std::printf("PLE table_bytes=%llu gathered_rows=%zu float_bytes=%zu: bit-exact vs selected file reads\n",
                        static_cast<unsigned long long>(t.bytes), ids.size(), gathered.size() * sizeof(float));
            found_table = true;
        }
        const std::string name = "blk.0.ffn_gate_exps.weight";
        if (metadata.tensors.count(name)) {
            for (uint64_t expert : {0u, 511u}) {
                const auto view = file->expert(name, expert);
                const auto raw = metadata.read(gguf_expert_slice(metadata.tensors.at(name), expert, name));
                require(view.size == raw.size() && std::memcmp(view.data, raw.data(), raw.size()) == 0,
                        "mapped expert differs from file read");
            }
            found_expert = true;
        }
    }
    require(found_table && found_expert, "checkpoint missing probe tensors");
    struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
    std::printf("All shards virtually mapped, max_RSS_kib=%ld (Linux units); no full-table materialization\n", usage.ru_maxrss);
    require(usage.ru_maxrss < 256 * 1024, "selected gathers allocated too much resident memory");
}
int main(int argc, char ** argv) {
    try {
        if (argc > 1) checkpoint_probe(argc, argv); else { unit_test(); gsq_dialect_test(); }
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
