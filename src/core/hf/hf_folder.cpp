#include "hf_folder.hpp"
#include "hf_arch.hpp"
#include "x3_record.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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

// ---- MLX affine 4-bit (group 32) ----
namespace {

// BF16 bits -> FP16 bits. Exact for every normal FP16 value; a value in FP16's subnormal range rounds to the nearest
// subnormal (TensorFold's Flash-Next folder: 2,501 of 11.2 billion scales / biases, all below 7.3e-6, moved by at most
// 3e-8). Out of FP16 range is an error (the repack would change the weights).
uint16_t bf16_to_f16_exact(uint16_t b, const std::string & what) {
    const uint32_t u = uint32_t(b) << 16, sign = (u >> 16) & 0x8000u;
    const int exp = int((u >> 23) & 0xff) - 127;
    const uint32_t mant = u & 0x7fffffu;  // BF16: the low 16 bits are zero
    if (((u >> 23) & 0xff) == 0 && mant == 0) return uint16_t(sign);
    if (((u >> 23) & 0xff) == 0xff || exp > 15) throw std::runtime_error("hf: " + what + ": a BF16 scale / bias is outside FP16");
    if (exp >= -14) return uint16_t(sign | uint32_t(exp + 15) << 10 | mant >> 13);
    // FP16 subnormal: value = m 2^-24 needs an integer m
    const int shift = -14 - exp;                         // 1 .. : the implicit bit moves into the mantissa
    const uint32_t full = (mant | 0x800000u) >> 13;      // 11 bits: 1.mmmmmmm0000
    if (shift > 11) return uint16_t(sign);  // below half the smallest FP16 subnormal
    uint32_t r = full >> shift; const uint32_t rem = full & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1u))) ++r;
    return uint16_t(sign | r);
}
float bf16_f(uint16_t v) { uint32_t u = uint32_t(v) << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t f_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
uint16_t f_f16(float f) {  // round to nearest even (dequantised values; not on the exact path)
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u; int e = int((u >> 23) & 0xff) - 127; uint32_t m = u & 0x7fffffu;
    if (e > 15) return uint16_t(sign | 0x7c00u);
    if (e < -24) return uint16_t(sign);
    if (e < -14) { m |= 0x800000u; const int sh = -1 - e; uint32_t r = m >> sh; if ((m >> (sh - 1)) & 1u && ((m & ((1u << (sh - 1)) - 1)) || (r & 1u))) ++r; return uint16_t(sign | r); }
    uint32_t r = uint32_t(e + 15) << 10 | m >> 13; const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    return uint16_t(sign | r);
}

struct Affine {  // one MLX affine 4-bit matrix (or a bank: [experts][rows][cols / 8])
    SafeTensor w, s, b; uint64_t experts = 1, rows = 0, cols = 0;
};
Affine affine_of(const HfFolder & h, const std::string & m) {
    Affine a{h.at(m + ".weight"), h.at(m + ".scales"), h.at(m + ".biases")};
    if (a.w.dtype != "U32" || a.s.dtype != "BF16" || a.b.dtype != "BF16") throw std::runtime_error("hf: " + m + " is not MLX affine 4-bit");
    const auto & ws = a.w.shape;
    if (ws.size() == 3) { a.experts = ws[0]; a.rows = ws[1]; a.cols = ws[2] * 8; }
    else if (ws.size() == 2) { a.rows = ws[0]; a.cols = ws[1] * 8; }
    else throw std::runtime_error("hf: " + m + ".weight rank");
    if (a.s.bytes != a.experts * a.rows * (a.cols / 32) * 2 || a.b.bytes != a.s.bytes)
        throw std::runtime_error("hf: " + m + ": only group 32 is read as Q4_1 (scales " + std::to_string(a.s.bytes) + " bytes)");
    return a;
}
constexpr uint64_t kQ41 = 20;  // bytes per 32 weights: d, m (FP16), 16 bytes of nibbles

// rows [r0, r0 + n) of slice `e` (every row: words, scales, biases) from the three byte arrays
void q4_1_rows(const Affine & a, const char * w, const char * s, const char * b, uint64_t r0, uint64_t n, char * out, const std::string & what) {
    const uint64_t words = a.cols / 8, groups = a.cols / 32;
    for (uint64_t r = r0; r < r0 + n; ++r) {
        const auto * wr = reinterpret_cast<const uint32_t *>(w) + r * words;
        const auto * sr = reinterpret_cast<const uint16_t *>(s) + r * groups;
        const auto * br = reinterpret_cast<const uint16_t *>(b) + r * groups;
        char * o = out + (r - r0) * groups * kQ41;
        for (uint64_t g = 0; g < groups; ++g) {
            const uint16_t d = bf16_to_f16_exact(sr[g], what), mn = bf16_to_f16_exact(br[g], what);
            std::memcpy(o + g * kQ41, &d, 2); std::memcpy(o + g * kQ41 + 2, &mn, 2);
            auto q = [&](uint64_t e) { return uint8_t((wr[g * 4 + e / 8] >> (4 * (e % 8))) & 15u); };
            for (int i = 0; i < 16; ++i) o[g * kQ41 + 4 + i] = char(q(i) | q(i + 16) << 4);
        }
    }
}
void affine_rows_f32(const Affine & a, const char * w, const char * s, const char * b, uint64_t r0, uint64_t n, float * out) {
    const uint64_t words = a.cols / 8, groups = a.cols / 32;
    for (uint64_t r = r0; r < r0 + n; ++r) {
        const auto * wr = reinterpret_cast<const uint32_t *>(w) + r * words;
        const auto * sr = reinterpret_cast<const uint16_t *>(s) + r * groups;
        const auto * br = reinterpret_cast<const uint16_t *>(b) + r * groups;
        for (uint64_t c = 0; c < a.cols; ++c)
            out[(r - r0) * a.cols + c] = float((wr[c / 8] >> (4 * (c % 8))) & 15u) * bf16_f(sr[c / 32]) + bf16_f(br[c / 32]);
    }
}

// block j of the result = block order[j] (blocks of `block` units of `unit` bytes, after `offset` units)
std::vector<char> reorder(std::vector<char> src, const HfFolder::BlockOrder & o, uint64_t units, uint64_t unit, uint64_t lines = 1) {
    if (o.order.empty()) return src;
    if ((o.offset + o.order.size() * o.block) > units) throw std::runtime_error("hf: block order exceeds the tensor");
    std::vector<char> out(src);
    const uint64_t line = units * unit, run = o.block * unit;
    for (uint64_t l = 0; l < lines; ++l)
        for (size_t j = 0; j < o.order.size(); ++j)
            std::memcpy(out.data() + l * line + (o.offset + j * o.block) * unit, src.data() + l * line + (o.offset + uint64_t(o.order[j]) * o.block) * unit, run);
    return out;
}

// read-only mappings of shard files, shared by the gather / warm hooks of one tensor
struct ShardMaps {
    std::mutex lock;  // the bank build's workers slice experts concurrently
    std::map<std::string, std::pair<const char *, size_t>> maps;
    const char * at(const SafeTensor & t) {
        std::lock_guard<std::mutex> hold(lock);
        auto it = maps.find(t.file);
        if (it == maps.end()) {
            const int fd = open(t.file.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) throw std::runtime_error("hf: cannot open " + t.file);
            struct stat st{}; fstat(fd, &st);
            void * p = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0); close(fd);
            if (p == MAP_FAILED) throw std::runtime_error("hf: cannot map " + t.file);
            madvise(p, size_t(st.st_size), MADV_RANDOM);
            it = maps.emplace(t.file, std::make_pair(static_cast<const char *>(p), size_t(st.st_size))).first;
        }
        return it->second.first + t.begin;
    }
    ~ShardMaps() { for (auto & [f, m] : maps) munmap(const_cast<char *>(m.first), m.second); }
};

}  // namespace

bool HfFolder::affine4(const std::string & m) const {
    return has(m + ".weight") && has(m + ".scales") && has(m + ".biases") && at(m + ".weight").dtype == "U32";
}

GgufTensor HfFolder::q4(const std::string & m, const BlockOrder & rows, const BlockOrder & cols) const {
    const Affine a = affine_of(*this, m);
    if (a.experts != 1) throw std::runtime_error("hf: " + m + " is a bank (q4_experts)");
    if (!cols.order.empty() && (cols.block % 32 || cols.offset % 32)) throw std::runtime_error("hf: column blocks of " + m + " split a group");
    GgufTensor t; t.type = GGML_Q4_1; t.ne = {a.cols, a.rows}; t.bytes = a.rows * (a.cols / 32) * kQ41;
    t.make = [a, rows, cols, m] {
        const auto w = safetensors_read(a.w), s = safetensors_read(a.s), b = safetensors_read(a.b);
        std::vector<char> out(a.rows * (a.cols / 32) * kQ41);
        q4_1_rows(a, w.data(), s.data(), b.data(), 0, a.rows, out.data(), m);
        BlockOrder c = cols; c.block /= 32; c.offset /= 32;
        out = reorder(std::move(out), c, a.cols / 32, kQ41, a.rows);
        return reorder(std::move(out), rows, a.rows, (a.cols / 32) * kQ41);
    };
    return t;
}

GgufTensor HfFolder::q4_cols(const std::vector<std::string> & ms) const {
    std::vector<Affine> as; for (const auto & m : ms) as.push_back(affine_of(*this, m));
    uint64_t cols = 0; for (const auto & a : as) { if (a.rows != as[0].rows || a.experts != 1) throw std::runtime_error("hf: column concat shapes"); cols += a.cols; }
    GgufTensor t; t.type = GGML_Q4_1; t.ne = {cols, as[0].rows}; t.bytes = as[0].rows * (cols / 32) * kQ41;
    t.make = [as, cols, ms] {
        const uint64_t rows = as[0].rows, line = (cols / 32) * kQ41;
        std::vector<char> out(rows * line); uint64_t at = 0;
        for (size_t i = 0; i < as.size(); ++i) {
            const auto w = safetensors_read(as[i].w), s = safetensors_read(as[i].s), b = safetensors_read(as[i].b);
            const uint64_t part = (as[i].cols / 32) * kQ41; std::vector<char> one(rows * part);
            q4_1_rows(as[i], w.data(), s.data(), b.data(), 0, rows, one.data(), ms[i]);
            for (uint64_t r = 0; r < rows; ++r) std::memcpy(out.data() + r * line + at, one.data() + r * part, part);
            at += part;
        }
        return out;
    };
    return t;
}

GgufTensor HfFolder::q4_experts(const std::string & m) const {
    const Affine a = affine_of(*this, m);
    const uint64_t per = a.rows * (a.cols / 32) * kQ41;
    GgufTensor t; t.type = GGML_Q4_1; t.ne = {a.cols, a.rows, a.experts}; t.bytes = a.experts * per;
    auto maps = std::make_shared<ShardMaps>();
    t.make_slice = [a, per, m, maps](uint64_t e) {
        if (e >= a.experts) throw std::runtime_error("hf: expert index");
        std::vector<char> out(per);
        const uint64_t wb = a.rows * (a.cols / 8) * 4, sb = a.rows * (a.cols / 32) * 2;
        q4_1_rows(a, maps->at(a.w) + e * wb, maps->at(a.s) + e * sb, maps->at(a.b) + e * sb, 0, a.rows, out.data(), m);
        return out;
    };
    t.make = [t] { std::vector<char> out; for (uint64_t e = 0; e < t.ne[2]; ++e) { const auto s = t.make_slice(e); out.insert(out.end(), s.begin(), s.end()); } return out; };
    return t;
}

GgufTensor HfFolder::q4_rows(const std::vector<std::string> & ms) const {
    auto as = std::make_shared<std::vector<Affine>>(); auto starts = std::make_shared<std::vector<uint64_t>>();
    uint64_t rows = 0;
    for (const auto & m : ms) {
        as->push_back(affine_of(*this, m));
        if (as->back().cols != as->front().cols || as->back().experts != 1) throw std::runtime_error("hf: row concat shapes");
        starts->push_back(rows); rows += as->back().rows;
    }
    const uint64_t cols = as->front().cols;
    GgufTensor t; t.type = GGML_Q4_1; t.ne = {cols, rows}; t.bytes = rows * (cols / 32) * kQ41;
    auto maps = std::make_shared<ShardMaps>();
    t.gather = [as, starts, maps, cols](const std::vector<uint64_t> & want, float * out) {
        for (size_t i = 0; i < want.size(); ++i) {
            const size_t k = size_t(std::upper_bound(starts->begin(), starts->end(), want[i]) - starts->begin()) - 1;
            const Affine & a = (*as)[k];
            affine_rows_f32(a, maps->at(a.w), maps->at(a.s), maps->at(a.b), want[i] - (*starts)[k], 1, out + i * cols);
        }
    };
    t.warm = [as](int threads) {
        std::vector<const SafeTensor *> parts; for (const auto & a : *as) for (const SafeTensor * p : {&a.w, &a.s, &a.b}) parts.push_back(p);
        std::atomic<size_t> next{0}, done{0}; std::vector<std::thread> pool;
        for (int i = 0; i < std::max(1, threads); ++i) pool.emplace_back([&] {
            std::vector<char> buf(size_t(1) << 24);
            for (size_t j; (j = next++) < parts.size();) {
                const SafeTensor & p = *parts[j]; const int fd = open(p.file.c_str(), O_RDONLY | O_CLOEXEC); if (fd < 0) continue;
                for (uint64_t at = 0; at < p.bytes; at += buf.size()) {
                    const ssize_t r = pread(fd, buf.data(), size_t(std::min<uint64_t>(buf.size(), p.bytes - at)), off_t(p.begin + at));
                    if (r <= 0) break; done += size_t(r);
                }
                close(fd);
            }
        });
        for (auto & th : pool) th.join();
        return size_t(done);
    };
    t.make = []() -> std::vector<char> { throw std::runtime_error("hf: a row-gathered table is read by row (gather_rows), not whole"); };
    return t;
}

GgufTensor HfFolder::dequant(const std::string & m, uint32_t type, const BlockOrder & rows, uint64_t row0, uint64_t nrows) const {
    const uint64_t es = type == GGML_F32 ? 4 : 2;
    GgufTensor t; t.type = type;
    if (affine4(m)) {
        const Affine a = affine_of(*this, m);
        const uint64_t n = nrows ? nrows : a.rows - row0;
        t.ne = {a.cols, n}; t.bytes = a.cols * n * es;
        t.make = [a, rows, row0, n, type, es] {
            const auto w = safetensors_read(a.w), s = safetensors_read(a.s), b = safetensors_read(a.b);
            std::vector<float> f(a.cols * n); affine_rows_f32(a, w.data(), s.data(), b.data(), row0, n, f.data());
            std::vector<char> out(f.size() * es);
            for (size_t i = 0; i < f.size(); ++i) {
                if (type == GGML_F32) std::memcpy(out.data() + 4 * i, &f[i], 4);
                else { const uint16_t v = type == GGML_BF16 ? f_bf16(f[i]) : f_f16(f[i]); std::memcpy(out.data() + 2 * i, &v, 2); }
            }
            return reorder(std::move(out), rows, n, a.cols * es);
        };
        return t;
    }
    const auto & s = at(m + ".weight");
    if (s.shape.size() != 2) throw std::runtime_error("hf: " + m + ".weight is not a matrix");
    return values(m + ".weight", type, {s.shape[0], s.shape[1]}, nullptr, rows);
}

GgufTensor HfFolder::values(const std::string & n, uint32_t type, std::vector<uint64_t> ne, float (*f)(float), const BlockOrder & rows) const {
    const auto & s = at(n);
    uint64_t count = 1; for (auto d : ne) count *= d;
    const uint64_t in_es = s.dtype == "F32" ? 4 : 2;
    if (s.dtype != "BF16" && s.dtype != "F16" && s.dtype != "F32") throw std::runtime_error("hf: " + n + " dtype " + s.dtype);
    if (count * in_es != s.bytes) throw std::runtime_error("hf: " + n + ": shape does not match the stored size");
    const uint64_t es = type == GGML_F32 ? 4 : 2;
    GgufTensor t; t.type = type; t.ne.assign(ne.rbegin(), ne.rend()); t.bytes = count * es;
    const uint64_t line = ne.size() > 1 ? count / ne[0] : 1, lines = ne.size() > 1 ? ne[0] : count;
    t.make = [s, type, f, rows, count, es, line, lines] {
        const auto raw = safetensors_read(s);
        std::vector<char> out(count * es);
        for (uint64_t i = 0; i < count; ++i) {
            float v;
            if (s.dtype == "F32") std::memcpy(&v, raw.data() + 4 * i, 4);
            else { uint16_t h; std::memcpy(&h, raw.data() + 2 * i, 2); v = s.dtype == "BF16" ? bf16_f(h) : f16_value(h); }
            if (f) v = f(v);
            if (type == GGML_F32) std::memcpy(out.data() + 4 * i, &v, 4);
            else { const uint16_t o = type == GGML_BF16 ? f_bf16(v) : f_f16(v); std::memcpy(out.data() + 2 * i, &o, 2); }
        }
        return reorder(std::move(out), rows, lines, line * es);
    };
    return t;
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
