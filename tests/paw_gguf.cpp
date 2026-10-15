// G1: paw-dense GGUF adapter. Writes a tiny synthetic paw-dense file (one K3.5 and one K3 matrix)
// and checks the arch switch, key aliasing, the GGML_X3 virtual shape, that the m3_* parts are gone,
// and that read() returns exactly header + trellis + suh + svh bytes.
#include "gguf.hpp"
#include "formats/x3_record.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace tqllm;

namespace {

struct Spec {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> dims;  // as stored: dims[0] = ne[0]
    std::vector<char> data;
};

void put_u32(std::ofstream & o, uint32_t v) { o.write(reinterpret_cast<const char *>(&v), 4); }
void put_u64(std::ofstream & o, uint64_t v) { o.write(reinterpret_cast<const char *>(&v), 8); }
void put_str(std::ofstream & o, const std::string & s) {
    put_u64(o, s.size());
    o.write(s.data(), std::streamsize(s.size()));
}

std::vector<char> blob_i16(const std::vector<int16_t> & v) {
    return std::vector<char>(reinterpret_cast<const char *>(v.data()),
                             reinterpret_cast<const char *>(v.data()) + v.size() * 2);
}
std::vector<char> blob_f16_pattern(uint64_t n, uint16_t seed) {
    std::vector<char> b;
    for (uint64_t i = 0; i < n; ++i) {
        const uint16_t w = uint16_t(seed + i);
        b.push_back(char(w & 0xff));
        b.push_back(char((w >> 8) & 0xff));
    }
    return b;
}

constexpr uint64_t kAlign = 32;
uint64_t align_up(uint64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

} // namespace

int main() {
    const std::string path = "/tmp/paw_gguf_test.gguf";

    // Two matrices: attn_qkv K3.5 (words 56), ffn_gate K3 (words 48). rows=128, cols=256.
    const uint64_t rows = 128, cols = 256, tiles = (rows / 16) * (cols / 16);
    std::vector<Spec> specs;
    auto add_triple = [&](const std::string & base, uint32_t words, uint16_t seed) {
        std::vector<int16_t> tr(tiles * words);
        for (size_t i = 0; i < tr.size(); ++i) tr[i] = int16_t((i * 7 + seed) & 0x7fff);
        specs.push_back({base + ".m3_trellis", GGML_I16, {words, tiles}, blob_i16(tr)});
        specs.push_back({base + ".m3_suh", GGML_F16, {cols}, blob_f16_pattern(cols, seed)});
        specs.push_back({base + ".m3_svh", GGML_F16, {rows}, blob_f16_pattern(rows, seed)});
    };
    add_triple("blk.0.attn_qkv", 56, 1);
    add_triple("blk.0.ffn_gate", 48, 2);
    specs.push_back({"mach1.ne_tlut", GGML_F16, {2, 512}, blob_f16_pattern(2 * 512, 3)});

    // KV entries (type 4 = uint32, 8 = string).
    struct KV { std::string k; uint32_t t; std::string s; uint32_t u; };
    std::vector<KV> kvs = {
        {"general.architecture", 8, "paw-dense", 0},
        {"general.alignment", 4, "", 32},
        {"paw.format_version", 4, "", 3},
        {"paw.x3_rewrite", 8, "mul1-v1", 0},
        {"paw-dense.block_count", 4, "", 64},
        {"paw-dense.embedding_length", 4, "", 5120},
    };

    std::vector<uint64_t> offsets(specs.size());
    uint64_t off = 0;
    for (size_t i = 0; i < specs.size(); ++i) { offsets[i] = off; off = align_up(off + specs[i].data.size()); }

    {
        std::ofstream o(path, std::ios::binary);
        put_u32(o, 0x46554747u);
        put_u32(o, 3);
        put_u64(o, specs.size());
        put_u64(o, kvs.size());
        for (auto & k : kvs) {
            put_str(o, k.k);
            put_u32(o, k.t);
            if (k.t == 8) put_str(o, k.s); else put_u32(o, k.u);
        }
        for (size_t i = 0; i < specs.size(); ++i) {
            put_str(o, specs[i].name);
            put_u32(o, uint32_t(specs[i].dims.size()));
            for (uint64_t d : specs[i].dims) put_u64(o, d);
            put_u32(o, specs[i].type);
            put_u64(o, offsets[i]);
        }
        const uint64_t here = uint64_t(o.tellp());
        for (uint64_t p = here; p < align_up(here); ++p) o.put('\0');
        for (size_t i = 0; i < specs.size(); ++i) {
            o.write(specs[i].data.data(), std::streamsize(specs[i].data.size()));
            const uint64_t cur = uint64_t(o.tellp()) - align_up(here);
            for (uint64_t p = cur; p < offsets[i] + specs[i].data.size(); ++p) o.put('\0');
        }
        o.flush();
        if (!o) { std::fprintf(stderr, "failed to write %s\n", path.c_str()); return 1; }
    }

    int rc = 0;
    auto check = [&](bool c, const char * what) {
        if (!c) { std::fprintf(stderr, "FAIL: %s\n", what); rc = 1; }
    };

    GgufFile f = gguf_open(path);
    check(f.str("general.architecture") == "qwen35", "arch aliased to qwen35");
    check(f.has("qwen35.block_count") && int(f.num("qwen35.block_count")) == 64, "paw-dense key aliased");
    check(!f.has("paw-dense.block_count"), "paw-dense key removed");

    check(f.tensors.count("blk.0.attn_qkv.weight") == 1, "virtual tensor present");
    check(f.tensors.count("blk.0.ffn_gate.weight") == 1, "second virtual tensor present");
    for (const auto & [n, t] : f.tensors) {
        (void) t;
        check(n.find(".m3_") == std::string::npos, "no m3_* tensor remains");
    }
    check(f.tensors.count("mach1.ne_tlut") == 0, "rt tlut dropped");

    for (auto & [name, words] : std::vector<std::pair<std::string, uint32_t>>{
             {"blk.0.attn_qkv.weight", 56}, {"blk.0.ffn_gate.weight", 48}}) {
        const GgufTensor & vt = f.tensors.at(name);
        check(vt.type == GGML_X3, "virtual type is GGML_X3");
        check(vt.ne.size() == 2 && vt.ne[0] == cols && vt.ne[1] == rows, "virtual ne = {cols, rows}");
        const uint64_t exp = sizeof(X3Record) + tiles * words * 2 + cols * 2 + rows * 2;
        check(vt.bytes == exp, "virtual bytes = header + trellis + suh + svh");
        const std::vector<char> data = f.read(vt);
        check(data.size() == vt.bytes, "read() length matches bytes");
        X3Record rec{};
        std::memcpy(&rec, data.data(), sizeof(rec));
        check(x3_record_valid(rec) && rec.rows == rows && rec.cols == cols && rec.words == words,
              "record header decodes");
        check(rec.codec == X3_CODEC_MUL1, "record codec = mul1");
        // bytes after the header must equal the original trellis + suh + svh blobs.
        const Spec * tr = nullptr, * suh = nullptr, * svh = nullptr;
        for (const Spec & s : specs) {
            if (s.name == name.substr(0, name.size() - 7) + ".m3_trellis") tr = &s;
            if (s.name == name.substr(0, name.size() - 7) + ".m3_suh") suh = &s;
            if (s.name == name.substr(0, name.size() - 7) + ".m3_svh") svh = &s;
        }
        check(tr && suh && svh, "found source triples");
        if (tr && suh && svh) {
            const char * p = data.data() + sizeof(X3Record);
            check(std::memcmp(p, tr->data.data(), tr->data.size()) == 0, "trellis bytes exact");
            p += tr->data.size();
            check(std::memcmp(p, suh->data.data(), suh->data.size()) == 0, "suh bytes exact");
            p += suh->data.size();
            check(std::memcmp(p, svh->data.data(), svh->data.size()) == 0, "svh bytes exact");
        }
    }

    std::remove(path.c_str());
    if (rc == 0) std::printf("paw_gguf: OK\n");
    return rc;
}
