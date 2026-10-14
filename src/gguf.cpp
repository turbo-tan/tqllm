#include "gguf.hpp"

#include <fstream>
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
        case GGML_F32: return checked_mul(n, 4, name);
        case GGML_F16:
        case GGML_BF16: return checked_mul(n, 2, name);
        case GGML_Q6_K:
            if (n % 256) break;
            return checked_mul(n / 256, 210, name);
        case GGML_TQ3_4S:
            if (n % 32) break;
            return checked_mul(n / 32, 16, name);
        case GGML_TQ4:  // TQLLM TQ4: FP16 scale + 64 bytes of 4-bit codes per 128
            if (n % 128) break;
            return checked_mul(n / 128, 66, name);
        case GGML_TQ3:  // TQLLM TQ3: FP16 scale + 48 bytes of 3-bit code planes per 128
            if (n % 128) break;
            return checked_mul(n / 128, 50, name);
        case GGML_Q4_0:
            if (n % 32) break;
            return checked_mul(n / 32, 18, name);
        case GGML_Q8_0:
            if (n % 32) break;
            return checked_mul(n / 32, 34, name);
        case 18:  // IQ3_XXS
            if (n % 256) break;
            return checked_mul(n / 256, 98, name);
        case 21:  // IQ3_S
            if (n % 256) break;
            return checked_mul(n / 256, 110, name);
        case 23:  // IQ4_XS
            if (n % 256) break;
            return checked_mul(n / 256, 136, name);
        case 11:  // Q3_K
            if (n % 256) break;
            return checked_mul(n / 256, 110, name);
        case 12:  // Q4_K
            if (n % 256) break;
            return checked_mul(n / 256, 144, name);
        case 13:  // Q5_K
            if (n % 256) break;
            return checked_mul(n / 256, 176, name);
    }
    throw std::runtime_error("unsupported GGUF tensor type " + std::to_string(type) + " for " + name);
}

} // namespace

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

std::vector<char> GgufFile::read(const GgufTensor & t) const {
    std::ifstream in(path, std::ios::binary);
    std::vector<char> buf(t.bytes);
    in.seekg(std::streamoff(t.offset));
    in.read(buf.data(), std::streamsize(t.bytes));
    if (!in) throw std::runtime_error("failed reading GGUF tensor data");
    return buf;
}

GgufFile gguf_open(const std::string & path) {
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
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        t.bytes = tensor_bytes(t.type, n, name, t.ne);
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

} // namespace tqllm
