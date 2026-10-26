#include "gguf.hpp"
#include "paw_gguf.hpp"
#include "hf_folder.hpp"

#include <cstdio>
#include <fstream>
#include <regex>
#include <limits>
#include <stdexcept>

namespace tqllm {
namespace {

struct Reader {
    std::ifstream in;
    template <class T> T get() {
        T v{};
        in.read(reinterpret_cast<char *>(&v), sizeof(v));
        if (!in) throw std::runtime_error("truncated GGUF header");
        return v;
    }
    std::string str() {
        const auto n = get<uint64_t>();
        if (n > (1u << 30)) throw std::runtime_error("bad GGUF string length");
        std::string s(n, '\0');
        in.read(s.data(), std::streamsize(n));
        if (!in) throw std::runtime_error("truncated GGUF string");
        return s;
    }
    // Reads a value of GGUF type `t`; numbers go to v.num (or v.arr), strings to v.str (or v.strs).
    void value(uint32_t t, GgufValue & v, bool in_array = false) {
        auto push = [&](double x) { in_array ? v.arr.push_back(x) : void(v.num = x); };
        switch (t) {
            case 0: push(get<uint8_t>()); break;
            case 1: push(get<int8_t>()); break;
            case 2: push(get<uint16_t>()); break;
            case 3: push(get<int16_t>()); break;
            case 4: push(get<uint32_t>()); break;
            case 5: push(get<int32_t>()); break;
            case 6: push(get<float>()); break;
            case 7: push(get<uint8_t>()); break;
            case 8: {
                auto s = str();
                if (in_array) v.strs.push_back(std::move(s));
                else v.str = std::move(s);
                break;
            }
            case 9: {
                const auto et = get<uint32_t>();
                const auto n = get<uint64_t>();
                for (uint64_t i = 0; i < n; ++i) value(et, v, true);
                break;
            }
            case 10: push(double(get<uint64_t>())); break;
            case 11: push(double(get<int64_t>())); break;
            case 12: push(get<double>()); break;
            default: throw std::runtime_error("unknown GGUF value type " + std::to_string(t));
        }
    }
};

uint64_t checked_mul(uint64_t a, uint64_t b, const std::string & name) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::runtime_error("GGUF tensor size overflow: " + name);
    return a * b;
}

uint64_t tensor_bytes(uint32_t type, uint64_t n, const std::string & name, const std::vector<uint64_t> & ne = {}) {
    switch (type) {
        case GGML_TQMLX_TILED3: case GGML_TQMLX_TILED4: {  // planes then FP16 scales, rows padded to 32
            if (ne.size() != 2 || ne[0] % 128) break;
            const uint64_t tiles = (ne[1] + 31) / 32, bits = type == GGML_TQMLX_TILED3 ? 3 : 4;
            return checked_mul(tiles, ne[0] * (bits * 4) + ne[0] / 4 * 2, name);
        }
        case GGML_TQMLX_MMA8_3: case GGML_TQMLX_MMA8_4: {  // nibble words then FP16 scales, rows padded to 64
            if (ne.size() != 2 || ne[0] % 128) break;
            const uint64_t rp = (ne[1] + 63) / 64 * 64;
            return checked_mul(rp, ne[0] / 2 + ne[0] / 64, name);
        }
        case GGML_NAX_A4_G128: case GGML_NAX_A6_G64: case GGML_NAX_A4_G64: {  // codes, high bits, FP16 scales, biases
            const uint64_t bits = type == GGML_NAX_A6_G64 ? 6 : 4, group = type == GGML_NAX_A4_G128 ? 128 : 64;
            if (ne.size() != 2 || ne[0] % group || ne[1] % 128) break;
            return checked_mul(ne[1], ne[0] * bits / 8 + ne[0] / group * 4, name);
        }
        case GGML_AFF8_G64: {  // row-major 8-bit codes, then FP16 scales and biases
            if (ne.size() != 2 || ne[0] % 64) break;
            return checked_mul(ne[1], ne[0] + ne[0] / 16, name);
        }
        default: break;
    }
    uint64_t blk, bpb;
    if (!gguf_block_layout(type, blk, bpb))
        throw std::runtime_error("unsupported GGUF tensor type " + std::to_string(type) + " for " + name);
    if (n % blk)
        throw std::runtime_error("GGUF tensor " + name + ": " + std::to_string(n) +
                                 " elements is not a multiple of the " + std::to_string(blk) +
                                 "-element block of type " + std::to_string(type));
    return checked_mul(n / blk, bpb, name);
}

} // namespace

bool gguf_block_layout(uint32_t type, uint64_t & elems, uint64_t & bytes) {
    // elems = weights per block, bytes = encoded size of one block.
    // The 2-bit K-quants (Q2_K, IQ2_S) carry MoE expert tensors - GLM-5.3-Flash uses
    // IQ2_S for ffn_gate_exps / ffn_up_exps and Q2_K for ffn_down_exps. Their block sizes
    // were verified 2026-09-29 against the real 4-shard GGUF on .55, not assumed: all 1412
    // tensors divide evenly, none is out of bounds, and per-shard byte sums match the file
    // payload to within alignment padding (560/480/480/280 B, 0.0000%). Table in
    // docs/MOE_PLAN.md 3b.
    switch (type) {
        case GGML_F32:    elems = 1;   bytes = 4;   return true;
        case GGML_F16:
        case GGML_BF16:
        case GGML_I16:    elems = 1;   bytes = 2;   return true;
        case GGML_Q4_0:   elems = 32;  bytes = 18;  return true;
        case GGML_Q8_0:   elems = 32;  bytes = 34;  return true;
        case GGML_MXFP8:  elems = 32;  bytes = 33;  return true;  // per row: the E4M3 bytes, then the row's scales
        case GGML_GSQ_Q2_0: elems = 64; bytes = 18; return true;
        case GGML_TQ3_4S: elems = 32;  bytes = 16;  return true;
        case GGML_Q2_K:   elems = 256; bytes = 84;  return true;
        case GGML_Q3_K:   elems = 256; bytes = 110; return true;
        case GGML_Q4_K:   elems = 256; bytes = 144; return true;
        case GGML_Q5_K:   elems = 256; bytes = 176; return true;
        case GGML_Q6_K:   elems = 256; bytes = 210; return true;
        case GGML_IQ2_XXS:elems = 256; bytes = 66;  return true;
        case GGML_IQ2_XS: elems = 256; bytes = 74;  return true;
        case GGML_IQ1_S:  elems = 256; bytes = 50;  return true;
        case GGML_IQ4_NL: elems = 32;  bytes = 18;  return true;
        case GGML_IQ1_M:  elems = 256; bytes = 56;  return true;
        case GGML_IQ3_XXS:elems = 256; bytes = 98;  return true;
        case GGML_IQ3_S:  elems = 256; bytes = 110; return true;
        case GGML_IQ2_S:  elems = 256; bytes = 82;  return true;
        case GGML_IQ4_XS: elems = 256; bytes = 136; return true;
        case GGML_TQ4:    elems = 128; bytes = 66;  return true;  // TQLLM: FP16 scale + 64 B of 4-bit codes
        case GGML_TQ3:    elems = 128; bytes = 50;  return true;  // TQLLM: FP16 scale + 48 B of 3-bit planes
        default: return false;
    }
}

uint64_t gguf_expert_count(const GgufTensor & t) {
    return t.ne.size() == 3 ? t.ne[2] : 0;
}

std::string gguf_shape_string(const GgufTensor & t) {
    std::string s = "[";
    for (size_t i = 0; i < t.ne.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(t.ne[i]);
    }
    return s + "]";
}

GgufTensor gguf_expert_slice(const GgufTensor & t, uint64_t idx, const std::string & name) {
    if (t.ne.size() != 3)
        throw std::runtime_error(name + ": not a 3-D expert tensor (ne.size()=" +
                                 std::to_string(t.ne.size()) + ")");
    const uint64_t n_experts = t.ne[2];
    if (n_experts == 0)
        throw std::runtime_error(name + ": expert tensor with zero experts");
    if (idx >= n_experts)
        throw std::runtime_error(name + ": expert index " + std::to_string(idx) +
                                 " out of range (" + std::to_string(n_experts) + " experts)");

    uint64_t blk, bpb;
    if (!gguf_block_layout(t.type, blk, bpb))
        throw std::runtime_error(name + ": unsupported expert tensor type " + std::to_string(t.type));

    // One expert is the [cols, rows] matrix t.ne[0] x t.ne[1]. Its element count must divide
    // into whole quant blocks, otherwise the expert boundary would fall inside a block and no
    // slice could address it independently.
    const uint64_t per_expert_elems = t.ne[0] * t.ne[1];
    if (per_expert_elems == 0 || per_expert_elems % blk)
        throw std::runtime_error(name + ": per-expert element count " +
                                 std::to_string(per_expert_elems) + " is not a multiple of the " +
                                 std::to_string(blk) + "-element block of type " +
                                 std::to_string(t.type));
    const uint64_t per_expert_bytes = per_expert_elems / blk * bpb;

    // Cross-check against the whole-tensor byte count computed at open time. If these disagree
    // the layout assumption is wrong and every expert would be read from the wrong place.
    if (per_expert_bytes * n_experts != t.bytes)
        throw std::runtime_error(name + ": expert slice bytes (" + std::to_string(per_expert_bytes) +
                                 " x " + std::to_string(n_experts) + ") != tensor bytes (" +
                                 std::to_string(t.bytes) + ")");

    GgufTensor s;
    s.type = t.type;
    s.ne = {t.ne[0], t.ne[1]};
    s.offset = t.offset + idx * per_expert_bytes;
    s.bytes = per_expert_bytes;
    return s;
}

uint64_t gguf_tensor_bytes(uint32_t type, const std::vector<uint64_t> & ne, const std::string & name) {
    uint64_t n = 1;
    for (auto e : ne) n = checked_mul(n, e, name);
    return tensor_bytes(type, n, name, ne);
}

double GgufFile::num(const std::string & key) const {
    const auto it = kv.find(key);
    if (it == kv.end()) throw std::runtime_error("GGUF missing key " + key);
    return it->second.num;
}

double GgufFile::num(const std::string & key, double fallback) const {
    const auto it = kv.find(key);
    return it == kv.end() ? fallback : it->second.num;
}

// String KV. `value()` stores strings in GgufValue::str and leaves it empty for numeric values, so
// an empty result for a PRESENT key means the key is not a string - report that distinctly rather
// than returning "" and letting a caller branch on an empty arch as if the file declared none.
std::string GgufFile::str(const std::string & key) const {
    const auto it = kv.find(key);
    if (it == kv.end()) throw std::runtime_error("GGUF missing key " + key);
    if (it->second.str.empty())
        throw std::runtime_error("GGUF key " + key + " is present but holds no string value");
    return it->second.str;
}

bool GgufFile::has(const std::string & key) const { return kv.count(key) != 0; }

std::vector<char> GgufFile::read(const GgufTensor & t) const {
    if (t.make) {
        auto buf = t.make();
        if (buf.size() != t.bytes) throw std::runtime_error("built GGUF tensor byte-count mismatch");
        return buf;
    }
    if (!t.parts.empty() || !t.header.empty()) {  // virtual tensor: header, then each part (its header, its bytes)
        std::vector<char> buf(t.header);
        buf.reserve(size_t(t.bytes));
        std::string open_file;
        std::ifstream pin;
        for (const GgufTensor & p : t.parts) {
            buf.insert(buf.end(), p.header.begin(), p.header.end());
            if (!p.bytes) continue;
            const std::string & pf = p.file.empty() ? (t.file.empty() ? path : t.file) : p.file;
            if (pf != open_file) { pin = std::ifstream(pf, std::ios::binary); open_file = pf; }
            const size_t at = buf.size();
            buf.resize(at + p.bytes);
            pin.seekg(std::streamoff(p.offset));
            pin.read(buf.data() + at, std::streamsize(p.bytes));
            if (!pin) throw std::runtime_error("failed reading GGUF tensor part");
        }
        if (buf.size() != t.bytes) throw std::runtime_error("virtual GGUF tensor byte-count mismatch");
        return buf;
    }
    std::ifstream in(t.file.empty() ? path : t.file, std::ios::binary);
    std::vector<char> buf(t.bytes);
    in.seekg(std::streamoff(t.offset));
    in.read(buf.data(), std::streamsize(t.bytes));
    if (!in) throw std::runtime_error("failed reading GGUF tensor data");
    return buf;
}

GgufFile gguf_open_shard(const std::string & path, GgufDialect dialect) {
    GgufFile f;
    f.path = path;
    Reader r{std::ifstream(path, std::ios::binary)};
    if (!r.in) throw std::runtime_error("cannot open " + path);
    if (r.get<uint32_t>() != 0x46554747u) throw std::runtime_error("not a GGUF file: " + path);
    if (r.get<uint32_t>() != 3) throw std::runtime_error("only GGUF v3 is supported");
    const auto n_tensors = r.get<uint64_t>(), n_kv = r.get<uint64_t>();
    for (uint64_t i = 0; i < n_kv; ++i) {
        auto key = r.str();
        GgufValue v;
        const auto type = r.get<uint32_t>();
        if (key == "general.alignment" && type != 4)
            throw std::runtime_error("GGUF alignment must be uint32");
        r.value(type, v);
        f.kv.emplace(std::move(key), std::move(v));
    }
    f.n_kv = n_kv;
    f.kv_end = uint64_t(r.in.tellg());
    std::vector<std::pair<std::string, GgufTensor>> infos;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        auto name = r.str();
        GgufTensor t;
        const auto nd = r.get<uint32_t>();
        if (nd == 0 || nd > 4) throw std::runtime_error("invalid GGUF tensor rank: " + name);
        uint64_t n = 1;
        for (uint32_t d = 0; d < nd; ++d) {
            t.ne.push_back(r.get<uint64_t>());
            if (t.ne.back() == 0 || t.ne.back() > uint64_t(std::numeric_limits<int64_t>::max()))
                throw std::runtime_error("invalid GGUF tensor extent: " + name);
            n = checked_mul(n, t.ne.back(), name);
        }
        t.type_header_offset = uint64_t(r.in.tellg());
        t.type = r.get<uint32_t>();
        if (dialect == GgufDialect::GsqRco && t.type == 42) t.type = GGML_GSQ_Q2_0;
        t.offset = r.get<uint64_t>();
        if (t.type == GGML_X3) {  // self-describing X3Record groups (the K can differ per expert): size from x3.bytes.<name>
            t.bytes = uint64_t(f.num("x3.bytes." + name, 0));
            if (!t.bytes) throw std::runtime_error("GGUF X3 tensor " + name + " has no x3.bytes." + name);
        } else {
            t.bytes = tensor_bytes(t.type, n, name, t.ne);
        }
        infos.emplace_back(std::move(name), std::move(t));
    }
    const uint64_t align = uint64_t(f.num("general.alignment", 32));
    f.alignment = align;
    uint64_t data = uint64_t(r.in.tellg());
    if (align == 0 || (align & (align - 1))) throw std::runtime_error("GGUF alignment must be a power of two");
    if (data > std::numeric_limits<uint64_t>::max() - (align - 1))
        throw std::runtime_error("GGUF data offset overflow");
    data = (data + align - 1) / align * align;
    r.in.seekg(0, std::ios::end);
    const uint64_t size = uint64_t(r.in.tellg());
    for (auto & [name, t] : infos) {
        if (data > size || t.offset > size - data)
            throw std::runtime_error("GGUF tensor out of bounds: " + name);
        t.offset += data;
        if (t.bytes > size - t.offset) throw std::runtime_error("GGUF tensor out of bounds: " + name);
        f.tensors.emplace(name, std::move(t));
    }
    return f;
}

std::string gguf_architecture(const std::string & path) {
    if (const std::string a = hf_folder_arch(path); !a.empty()) return a;
    const GgufFile f = gguf_open_shard(path);
    return f.has("general.architecture") ? f.str("general.architecture") : std::string();
}

GgufFile gguf_open(const std::string & path, GgufDialect dialect) {
    if (hf_folder(path)) return hf_open(path);  // a Hugging Face checkpoint folder, opened in place
    GgufFile f = gguf_open_shard(path, dialect);
    const uint64_t count = uint64_t(f.num("split.count", 1));
    if (count <= 1) { paw_adapt(f); return f; }  // paw-dense -> qwen35 view (no-op for every other arch)
    static const std::regex name(R"((.*)-(\d+)-of-(\d+)\.gguf$)");
    std::smatch m;
    if (!std::regex_match(path, m, name) || std::stoull(m[3].str()) != count)
        throw std::runtime_error("split GGUF " + path + ": expected a name ending -NNNNN-of-" + std::to_string(count) + ".gguf");
    const int width = int(m[2].length());
    auto shard_path = [&](uint64_t i) {
        char num[32];
        std::snprintf(num, sizeof num, "%0*llu", width, static_cast<unsigned long long>(i + 1));
        return m[1].str() + "-" + num + "-of-" + m[3].str() + ".gguf";
    };
    GgufFile all = uint64_t(f.num("split.no", 0)) == 0 ? std::move(f) : gguf_open_shard(shard_path(0), dialect);
    if (uint64_t(all.num("split.no", 0)) != 0 || uint64_t(all.num("split.count", 1)) != count)
        throw std::runtime_error("split GGUF " + shard_path(0) + ": not the first of " + std::to_string(count) + " shards");
    all.path = shard_path(0);
    for (auto & [n, t] : all.tensors) t.file = all.path;
    for (uint64_t i = 1; i < count; ++i) {
        const auto p = shard_path(i);
        GgufFile s = gguf_open_shard(p, dialect);
        if (uint64_t(s.num("split.no", -1)) != i || uint64_t(s.num("split.count", 0)) != count)
            throw std::runtime_error("split GGUF " + p + ": split.no / split.count do not match its name");
        for (auto & [n, t] : s.tensors) {
            t.file = p;
            if (!all.tensors.emplace(n, t).second) throw std::runtime_error("split GGUF: tensor " + n + " is in two shards");
        }
    }
    if (all.has("split.tensors.count") && uint64_t(all.num("split.tensors.count")) != all.tensors.size())
        throw std::runtime_error("split GGUF: " + std::to_string(all.tensors.size()) + " tensors found, split.tensors.count says " +
                                 std::to_string(uint64_t(all.num("split.tensors.count"))));
    paw_adapt(all);
    return all;
}

} // namespace tqllm
