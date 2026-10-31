#include "hf_folder.hpp"
#include "hf_arch.hpp"
#include "x3_record.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tqllm {

namespace {

namespace fs = std::filesystem;

std::string slurp(const fs::path & p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("hf: cannot read " + p.string());
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

const Json & need(const Json & o, const std::string & k) {
    const Json * v = o.get(k);
    if (!v) throw std::runtime_error("hf: no '" + k + "'");
    return *v;
}

GgufTensor range(const SafeTensor & s, uint32_t type, std::vector<uint64_t> ne) {
    GgufTensor t;
    t.type = type;
    t.ne = std::move(ne);
    t.file = s.file;
    t.offset = s.begin;
    t.bytes = s.bytes;
    return t;
}

uint32_t type_of(const SafeTensor & s) {
    if (s.dtype == "BF16") return GGML_BF16;
    if (s.dtype == "F16") return GGML_F16;
    if (s.dtype == "F32") return GGML_F32;
    throw std::runtime_error("hf: tensor dtype " + s.dtype + " is stored in an encoding the loader does not read as is");
}

float bf16_value(uint16_t v) { uint32_t u = uint32_t(v) << 16; float f; std::memcpy(&f, &u, 4); return f; }
float f16_value(uint16_t h) {
    const uint32_t s = uint32_t(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) { if (!m) u = s; else { float f = std::ldexp(float(m), -24); std::memcpy(&u, &f, 4); u |= s; } }
    else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}
float e4m3_value(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e == 15 && m == 7 ? NAN : e == 0 ? m / 8.0f * std::ldexp(1.0f, -6) : (1 + m / 8.0f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}

// exl3 (exllamav3, mul1 codebook): trellis int16 [in/16][out/16][16 K] (the X3 tile order), suh [in], svh [out]
void add_trellis(const HfFolder & h, GgufTensor & t, const std::string & m) {
    const auto & tr = h.at(m + ".trellis");
    if (!h.has(m + ".mul1")) throw std::runtime_error("hf: " + m + " is a trellis matrix without the mul1 codebook marker");
    if (tr.dtype != "I16" || tr.shape.size() != 3) throw std::runtime_error("hf: " + m + ".trellis is not I16 [k/16][n/16][16K]");
    const uint32_t cols = uint32_t(tr.shape[0] * 16), rows = uint32_t(tr.shape[1] * 16), words = uint32_t(tr.shape[2]);
    const auto & suh = h.at(m + ".suh");
    const auto & svh = h.at(m + ".svh");
    if (suh.bytes != uint64_t(cols) * 2 || svh.bytes != uint64_t(rows) * 2) throw std::runtime_error("hf: " + m + " suh / svh size");
    GgufTensor rec;
    const X3Record r = x3_record_make(rows, cols, words, X3_CODEC_MUL1);
    rec.header.assign(reinterpret_cast<const char *>(&r), reinterpret_cast<const char *>(&r) + sizeof r);
    t.parts.push_back(std::move(rec));
    for (const SafeTensor * s : {&tr, &suh, &svh}) t.parts.push_back(range(*s, GGML_I16, {s->bytes / 2}));
    t.bytes += sizeof r + tr.bytes + suh.bytes + svh.bytes;
}

// [rows][cols] BF16 of an MXFP8 matrix (E4M3 values, one E8M0 scale per 32 along a row): exact in BF16
GgufTensor mxfp8(const SafeTensor & w, const SafeTensor & sc) {
    const uint64_t rows = w.shape[0], cols = w.shape[1];
    if (sc.bytes != rows * cols / 32) throw std::runtime_error("hf: MXFP8 scale shape");
    GgufTensor t;
    t.type = GGML_BF16;
    t.ne = {cols, rows};
    t.bytes = rows * cols * 2;
    t.make = [w, sc] {
        const auto q = safetensors_read(w), s = safetensors_read(sc);
        std::vector<char> out(q.size() * 2);
        for (size_t i = 0; i < q.size(); ++i) {
            const float f = e4m3_value(uint8_t(q[i])) * std::ldexp(1.0f, int(uint8_t(s[i / 32])) - 127);
            uint32_t u; std::memcpy(&u, &f, 4);
            const uint16_t b = uint16_t(u >> 16);
            std::memcpy(out.data() + 2 * i, &b, 2);
        }
        return out;
    };
    return t;
}

}  // namespace

HfFolder::HfFolder(const std::string & path) : dir_(path) {
    config_text_ = slurp(dir_ / "config.json");
    root_ = Parser(config_text_).parse();
    const Json * tc = root_.get("text_config");
    cfg_ = tc && tc->kind == Json::Object ? tc : &root_;
    index_ = safetensors_index(dir_);
}

std::string HfFolder::architecture() const {
    const Json * a = root_.get("architectures");
    return a && a->kind == Json::Array && !a->arr.empty() ? a->arr[0].str : std::string();
}
double HfFolder::num(const char * key) const { return need(*cfg_, key).num; }
const Json * HfFolder::section(const char * key) const {
    const Json * v = cfg_->get(key);
    return v && v->kind == Json::Object ? v : nullptr;
}
const SafeTensor & HfFolder::at(const std::string & n) const {
    auto it = index_.find(n);
    if (it == index_.end()) throw std::runtime_error("hf: missing tensor " + n + " in " + dir_.string());
    return it->second;
}

GgufTensor HfFolder::matrix(const std::string & m) const {
    if (has(m + ".trellis")) {
        const auto & tr = at(m + ".trellis");
        GgufTensor t;
        t.type = GGML_X3;
        t.ne = {tr.shape[0] * 16, tr.shape[1] * 16};
        add_trellis(*this, t, m);
        return t;
    }
    const auto & w = at(m + ".weight");
    if (w.shape.size() != 2) throw std::runtime_error("hf: " + m + ".weight is not a matrix");
    if (has(m + ".weight_scale_inv") && w.dtype == "F8_E4M3") {
        const auto & sc = at(m + ".weight_scale_inv");
        if (w.shape[1] % 512 || sc.bytes != w.shape[0] * w.shape[1] / 32) return mxfp8(w, sc);  // BF16, exact
        GgufTensor t;  // MXFP8 as stored, in GGML_MXFP8 rows: each row's E4M3 bytes, then its scales (stackable)
        t.type = GGML_MXFP8;
        t.ne = {w.shape[1], w.shape[0]};
        t.bytes = w.bytes + sc.bytes;
        t.make = [w, sc] {
            const auto q = safetensors_read(w), s = safetensors_read(sc);
            const size_t rows = w.shape[0], cols = w.shape[1];
            std::vector<char> out(q.size() + s.size());
            for (size_t r = 0; r < rows; ++r) {
                std::memcpy(out.data() + r * (cols + cols / 32), q.data() + r * cols, cols);
                std::memcpy(out.data() + r * (cols + cols / 32) + cols, s.data() + r * (cols / 32), cols / 32);
            }
            return out;
        };
        return t;
    }
    return range(w, type_of(w), {w.shape[1], w.shape[0]});
}

GgufTensor HfFolder::stack(const std::vector<std::string> & modules) const {
    const auto & tr = at(modules.front() + ".trellis");  // stacks of the published packs are trellis banks
    GgufTensor t;
    t.type = GGML_X3;
    t.ne = {tr.shape[0] * 16, tr.shape[1] * 16, modules.size()};
    for (const auto & m : modules) {
        const auto & p = at(m + ".trellis");
        if (p.shape[0] != tr.shape[0] || p.shape[1] != tr.shape[1]) throw std::runtime_error("hf: expert " + m + " shape differs");
        add_trellis(*this, t, m);
    }
    return t;
}

GgufTensor HfFolder::norm(const std::string & n, bool plus_one) const {
    const SafeTensor s = at(n);
    GgufTensor t;
    t.type = GGML_F32;
    t.ne = {s.shape[0]};
    t.bytes = s.shape[0] * 4;
    t.make = [s, plus_one] {
        const auto raw = safetensors_read(s);
        const size_t n = size_t(s.shape[0]);
        std::vector<char> out(n * 4);
        for (size_t i = 0; i < n; ++i) {
            float f;
            if (s.dtype == "F32") std::memcpy(&f, raw.data() + 4 * i, 4);
            else { uint16_t v; std::memcpy(&v, raw.data() + 2 * i, 2); f = s.dtype == "BF16" ? bf16_value(v) : f16_value(v); }
            if (plus_one) f += 1.0f;
            std::memcpy(out.data() + 4 * i, &f, 4);
        }
        return out;
    };
    return t;
}

GgufTensor HfFolder::plain(const std::string & n) const {
    const auto & s = at(n);
    std::vector<uint64_t> ne(s.shape.rbegin(), s.shape.rend());  // GGUF order: the contiguous dimension first
    return range(s, type_of(s), std::move(ne));
}

void HfFolder::tokenizer(GgufFile & f, const std::string & pre, uint64_t vocab) const {
    const std::string tj = slurp(dir_ / "tokenizer.json"), cj = slurp(dir_ / "tokenizer_config.json");
    const Json t = Parser(tj).parse(), c = Parser(cj).parse();
    std::vector<std::string> tokens(vocab);
    std::vector<double> types(vocab, 5);  // unused
    for (const auto & [piece, id] : need(need(t, "model"), "vocab").obj)
        if (size_t(id.num) < vocab) { tokens[size_t(id.num)] = piece; types[size_t(id.num)] = 1; }
    for (const auto & a : need(t, "added_tokens").arr) {
        const size_t i = size_t(need(a, "id").num);
        if (i >= vocab) continue;
        tokens[i] = need(a, "content").str;
        const Json * sp = a.get("special");
        types[i] = sp && sp->kind == Json::Bool && sp->num != 0 ? 3 : 4;  // control : user-defined
    }
    std::map<std::string, size_t> ids;
    for (size_t i = 0; i < vocab; ++i) {
        if (tokens[i].empty() && types[i] == 5) tokens[i] = "[PAD" + std::to_string(i) + "]";
        ids.emplace(tokens[i], i);
    }
    std::vector<std::string> merges;
    for (const auto & m : need(need(t, "model"), "merges").arr) {
        if (m.kind == Json::String) merges.push_back(m.str);
        else if (m.kind == Json::Array && m.arr.size() == 2) merges.push_back(m.arr[0].str + " " + m.arr[1].str);
    }
    f.kv["tokenizer.ggml.model"].str = "gpt2";
    f.kv["tokenizer.ggml.pre"].str = pre;
    f.kv["tokenizer.ggml.tokens"].strs = std::move(tokens);
    f.kv["tokenizer.ggml.token_type"].arr = std::move(types);
    f.kv["tokenizer.ggml.merges"].strs = std::move(merges);
    auto special = [&](const char * key, const char * gguf) {
        const Json * v = c.get(key);
        if (!v || (v->kind != Json::String && v->kind != Json::Object)) return;
        const std::string & s = v->kind == Json::String ? v->str : need(*v, "content").str;
        if (auto it = ids.find(s); it != ids.end()) f.kv[gguf].num = double(it->second);
    };
    special("eos_token", "tokenizer.ggml.eos_token_id");
    special("bos_token", "tokenizer.ggml.bos_token_id");
    special("pad_token", "tokenizer.ggml.padding_token_id");
    if (fs::exists(dir_ / "chat_template.jinja")) f.kv["tokenizer.chat_template"].str = slurp(dir_ / "chat_template.jinja");
}

std::string hf_folder_arch(const std::string & path) {
    std::error_code ec;
    if (!fs::is_directory(path, ec) || !fs::exists(fs::path(path) / "config.json", ec)) return {};
    try {
        const std::string text = slurp(fs::path(path) / "config.json");
        const Json root = Parser(text).parse();
        const Json * a = root.get("architectures");
        const HfArch * h = a && a->kind == Json::Array && !a->arr.empty() ? hf_arch(a->arr[0].str) : nullptr;
        return h ? h->arch : std::string();
    } catch (const std::exception &) {
        return {};
    }
}

GgufFile hf_open(const std::string & path) {
    const HfFolder h(path);
    const HfArch * a = hf_arch(h.architecture());
    if (!a) throw std::runtime_error("hf: no tensor map for architecture '" + h.architecture() + "' (" + path + ")");
    GgufFile f;
    f.path = path;
    f.kv["general.architecture"].str = a->arch;
    a->map(h, f);
    return f;
}

}  // namespace tqllm
