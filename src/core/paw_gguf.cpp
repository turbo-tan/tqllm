#include "paw_gguf.hpp"
#include "gguf.hpp"
#include "x3_record.hpp"

#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tqllm {

namespace {

// A metadata-key prefix rename applied when an arch is adapted (paw-dense.* -> qwen35.*).
struct KeyAlias {
    const char * from;
    const char * to;
};

// Builds the virtual tensors an arch's file exposes to the engine. May throw on a malformed file.
using TensorMapFn = void (*)(GgufFile &);

// One row per PAW checkpoint family: its file arch, the arch the engine sees, the key renames and
// the tensor rewrite. Adding a family (e.g. paw-moe) is a new row plus one map function.
struct PawArch {
    const char * paw_arch;
    const char * tq_arch;
    std::span<const KeyAlias> aliases;
    TensorMapFn map;
};

bool ends_with(const std::string & s, const char * suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

std::string strip_suffix(const std::string & s, const char * suf) {
    return s.substr(0, s.size() - std::strlen(suf));
}

// Rename every metadata key with prefix `from` to prefix `to` (in place).
void prefix_rename(std::map<std::string, GgufValue> & kv, const char * from, const char * to) {
    const std::string f(from), t(to);
    std::vector<std::pair<std::string, GgufValue>> moved;
    for (auto it = kv.begin(); it != kv.end();) {
        if (it->first.rfind(f, 0) == 0) {
            moved.emplace_back(t + it->first.substr(f.size()), it->second);
            it = kv.erase(it);
        } else {
            ++it;
        }
    }
    for (auto & [k, v] : moved) kv[k] = std::move(v);
}

// paw-dense map: an `m3_trellis` / `m3_suh` / `m3_svh` triple becomes one GGML_X3 virtual tensor.
void map_m3_x3(GgufFile & f) {
    // Refuse files the x3 codec cannot read.
    for (const auto & [name, t] : f.tensors) {
        (void) t;
        if (ends_with(name, ".rt_trellis"))
            throw std::runtime_error("paw_adapt: " + f.path +
                                     " uses the legacy rt codec (rt_trellis); only x3 is supported");
    }

    if (int(f.num("paw.format_version", 0)) != 3)
        throw std::runtime_error("paw_adapt: " + f.path + " is not paw.format_version 3");
    const std::string rewrite = f.str("paw.x3_rewrite");
    uint16_t codec = 0;
    if (rewrite == "mul1-v1") codec = X3_CODEC_MUL1;
    else throw std::runtime_error("paw_adapt: " + f.path + " has unknown paw.x3_rewrite '" + rewrite + "'");

    // Build one GGML_X3 virtual tensor per m3 triple, drop the parts, drop the unused rt tlut.
    std::map<std::string, GgufTensor> added;
    std::vector<std::string> drop;
    for (const auto & [name, tr] : f.tensors) {
        if (!ends_with(name, ".m3_trellis")) continue;
        const std::string base = strip_suffix(name, ".m3_trellis");
        const auto suh_it = f.tensors.find(base + ".m3_suh");
        const auto svh_it = f.tensors.find(base + ".m3_svh");
        if (suh_it == f.tensors.end() || svh_it == f.tensors.end())
            throw std::runtime_error("paw_adapt: incomplete x3 triple at " + base);
        const GgufTensor & suh = suh_it->second;
        const GgufTensor & svh = svh_it->second;
        if (tr.ne.size() != 2 || suh.ne.size() != 1 || svh.ne.size() != 1)
            throw std::runtime_error("paw_adapt: bad x3 triple rank at " + base);
        const uint64_t words = tr.ne[0];      // 16*K uint16 per tile
        const uint64_t cols = suh.ne[0];       // input dim
        const uint64_t rows = svh.ne[0];       // output dim
        // words = 16*K; K may be half-integer (56 = K3.5), so words is a multiple of 8, not 16.
        if (words % 8 || words < 16)
            throw std::runtime_error("paw_adapt: x3 tile width is not a valid 16*K at " + base);
        if (cols % 128 || rows % 128)
            throw std::runtime_error("paw_adapt: x3 dims not multiples of 128 at " + base);
        if (tr.ne[1] != (rows / 16) * (cols / 16))
            throw std::runtime_error("paw_adapt: x3 trellis tile count mismatch at " + base);
        if (tr.bytes != (rows / 16) * (cols / 16) * words * 2 ||
            suh.bytes != cols * 2 || svh.bytes != rows * 2)
            throw std::runtime_error("paw_adapt: x3 part byte count mismatch at " + base);

        GgufTensor vt;
        vt.type = GGML_X3;
        vt.ne = {cols, rows};
        vt.offset = 0;  // virtual: read() walks `parts`
        vt.bytes = sizeof(X3Record) + tr.bytes + suh.bytes + svh.bytes;
        const X3Record rec = x3_record_make(uint32_t(rows), uint32_t(cols), uint32_t(words), codec);
        vt.header.assign(reinterpret_cast<const char *>(&rec),
                         reinterpret_cast<const char *>(&rec) + sizeof(rec));
        vt.parts = {tr, suh, svh};
        added[base + ".weight"] = std::move(vt);
        drop.push_back(base + ".m3_trellis");
        drop.push_back(base + ".m3_suh");
        drop.push_back(base + ".m3_svh");
    }
    for (const std::string & n : drop) f.tensors.erase(n);
    f.tensors.erase("mach1.ne_tlut");  // legacy rt codec, unused by x3
    for (auto & [n, t] : added) f.tensors.emplace(n, std::move(t));
}

constexpr KeyAlias kPawDenseAliases[] = {{"paw-dense.", "qwen35."}};

constexpr PawArch kPawArches[] = {
    {"paw-dense", "qwen35", kPawDenseAliases, map_m3_x3},
};

} // namespace

void paw_adapt(GgufFile & f) {
    if (!f.has("general.architecture")) return;
    const std::string arch = f.str("general.architecture");
    const PawArch * pa = nullptr;
    for (const PawArch & p : kPawArches)
        if (arch == p.paw_arch) { pa = &p; break; }
    if (!pa) return;

    for (const KeyAlias & al : pa->aliases) prefix_rename(f.kv, al.from, al.to);
    f.kv["general.architecture"].str = pa->tq_arch;
    f.kv["general.architecture"].num = 0;
    pa->map(f);
}

} // namespace tqllm
