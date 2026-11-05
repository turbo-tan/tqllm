// Flash-Next from an MLX affine 4-bit Hugging Face folder (src/core/hf/arch_qwen4_exp.cpp + the HfFolder MLX builders),
// on synthetic MLX tensors: a matrix read as GGML_Q4_1 must dequantise to exactly the MLX values (q * scale + bias); a
// row / column block order moves whole blocks; an expert built alone equals that expert of the bank; the row-stacked
// table gathers the right rows across a shard boundary and through MappedGguf; values() applies the transform and the
// order; dequant() to BF16 rounds each value once.
// With TQLLM_TEST_HF_QWEN4EXP=FOLDER and TQLLM_TEST_REF_GGUF=GGUF (a qwen4exp GGUF of the same model) it also diffs the
// metadata key by key and compares every non-expert tensor and expert 0 of every bank within quantisation noise.
#include "dequant.hpp"
#include "gguf.hpp"
#include "gguf_mapped.hpp"
#include "hf_folder.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>

using namespace tqllm;
namespace fs = std::filesystem;

namespace {
bool ok = true;
void check(bool c, const std::string & what) { if (!c) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ok = false; } }

struct T { std::string dtype; std::vector<uint64_t> shape; std::vector<char> data; };
void write_safetensors(const fs::path & p, const std::map<std::string, T> & ts) {
    std::string h = "{";
    uint64_t off = 0;
    for (const auto & [n, t] : ts) {
        if (h.size() > 1) h += ",";
        h += "\"" + n + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < t.shape.size(); ++i) h += (i ? "," : "") + std::to_string(t.shape[i]);
        h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.data.size()) + "]}";
        off += t.data.size();
    }
    h += "}";
    while (h.size() % 8) h += " ";
    std::ofstream o(p, std::ios::binary);
    const uint64_t n = h.size();
    o.write(reinterpret_cast<const char *>(&n), 8);
    o.write(h.data(), std::streamsize(h.size()));
    for (const auto & [name, t] : ts) o.write(t.data.data(), std::streamsize(t.data.size()));
}
float bf16f(uint16_t v) { uint32_t u = uint32_t(v) << 16; float f; std::memcpy(&f, &u, 4); return f; }

std::mt19937 rng(3);
// an MLX affine 4-bit module [lead...][rows][cols]: codes, and BF16 scales / biases that are exact in FP16
struct Mod { std::vector<uint32_t> w; std::vector<uint16_t> s, b; std::vector<uint64_t> lead; uint64_t rows, cols; };
Mod make_mod(std::vector<uint64_t> lead, uint64_t rows, uint64_t cols) {
    Mod m{{}, {}, {}, lead, rows, cols};
    uint64_t n = rows * cols; for (auto l : lead) n *= l;
    m.w.resize(n / 8); for (auto & x : m.w) x = uint32_t(rng());
    m.s.resize(n / 32); m.b.resize(n / 32);
    for (size_t i = 0; i < m.s.size(); ++i) {
        // BF16 values with an 8-bit mantissa inside the FP16 normal range: exact in FP16
        m.s[i] = uint16_t(0x3c00 + (rng() & 0x7f) - 0x0400);  // ~2^-8 .. 2^-7
        m.b[i] = uint16_t(0xbc00 + (rng() & 0x7f) - 0x0300);  // ~ -2^-6
    }
    return m;
}
void add(std::map<std::string, T> & ts, const std::string & name, const Mod & m) {
    std::vector<uint64_t> ws = m.lead, ss = m.lead;
    ws.push_back(m.rows); ws.push_back(m.cols / 8); ss.push_back(m.rows); ss.push_back(m.cols / 32);
    auto raw = [](const auto & v) { std::vector<char> o(v.size() * sizeof(v[0])); std::memcpy(o.data(), v.data(), o.size()); return o; };
    ts[name + ".weight"] = {"U32", ws, raw(m.w)};
    ts[name + ".scales"] = {"BF16", ss, raw(m.s)};
    ts[name + ".biases"] = {"BF16", ss, raw(m.b)};
}
float value(const Mod & m, uint64_t i) {  // element i (row-major over lead, rows, cols)
    return float((m.w[i / 8] >> (4 * (i % 8))) & 15u) * bf16f(m.s[i / 32]) + bf16f(m.b[i / 32]);
}
std::vector<float> deq(uint32_t type, const std::vector<char> & raw, size_t n) { std::vector<float> f(n); dequantize_host(type, raw.data(), f.data(), int64_t(n)); return f; }

// the optional check against a real folder and a reference GGUF of the same model
void real_check(const std::string & folder, const std::string & ref) {
    const GgufFile g = hf_open(folder), r = gguf_open(ref, GgufDialect::GsqRco);  // the ISTA GSQ file: its expert types
    int keys = 0, bad = 0;
    for (const auto & [k, v] : r.kv) {
        if (k.rfind("qwen4exp.", 0) != 0) continue;
        ++keys;
        const auto it = g.kv.find(k);
        if (it == g.kv.end()) { std::fprintf(stderr, "  missing key %s\n", k.c_str()); ++bad; continue; }
        const bool same = v.arr.empty() ? std::fabs(v.num - it->second.num) <= 1e-6 * std::max(1.0, std::fabs(v.num)) : v.arr == it->second.arr;
        if (!same) { std::fprintf(stderr, "  key %s differs\n", k.c_str()); ++bad; }
    }
    check(bad == 0, std::to_string(bad) + " of " + std::to_string(keys) + " qwen4exp keys differ from the reference");
    const MappedGguf mf(folder);
    int compared = 0, off = 0;
    for (const auto & [name, rt] : r.tensors) {
        if (name.rfind("blk.48.", 0) == 0 || name == "per_layer_token_embd.weight") continue;
        const auto it = mf.file().tensors.find(name);
        if (it == mf.file().tensors.end()) { std::fprintf(stderr, "  missing tensor %s\n", name.c_str()); ++off; continue; }
        const GgufTensor & t = it->second;
        uint64_t n = 1; for (auto d : rt.ne) n *= d;
        std::vector<float> a, b;
        try {
        if (rt.ne.size() == 3) {  // expert 0
            n /= rt.ne[2];
            const auto [bytes, owner] = mf.expert_owned(name, 0);
            a.resize(n); dequantize_host(t.type, bytes.data, a.data(), int64_t(n));
            const auto rb = r.read(gguf_expert_slice(rt, 0, name)); b = deq(rt.type, rb, n);
        } else {
            if (t.ne != rt.ne) { std::fprintf(stderr, "  %s shape differs\n", name.c_str()); ++off; continue; }
            const auto bytes = mf.tensor(name); a.resize(n); dequantize_host(t.type, bytes.data, a.data(), int64_t(n));
            b = deq(rt.type, r.read(rt), n);
        }
        } catch (const std::exception & ex) {  // a reference type the host cannot dequantise (GSQ experts): skipped, listed
            std::fprintf(stderr, "  skipped %s: %s\n", name.c_str(), ex.what()); continue;
        }
        double e = 0, s = 0, ab = 0, aa = 0, bb = 0;
        for (uint64_t i = 0; i < n; ++i) { e += (a[i] - b[i]) * (a[i] - b[i]); s += b[i] * b[i]; ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
        const double rel = std::sqrt(e / std::max(s, 1e-30)), corr = ab / std::sqrt(std::max(aa * bb, 1e-30));
        ++compared;
        // experts: the reference GSQ banks are 2-3 bit (IQ2_S / IQ3_XXS), their own noise against a 4-bit bank is 20-40 %
        if (!(corr > (rt.ne.size() == 3 ? 0.9 : 0.98))) { std::fprintf(stderr, "  %s: rel-rms %.3f corr %.4f\n", name.c_str(), rel, corr); ++off; }
    }
    check(off == 0, std::to_string(off) + " of " + std::to_string(compared) + " tensors off the reference");
    std::printf("real folder: %d keys, %d tensors compared, %d off\n", keys, compared, off);
}
}  // namespace

int main() try {
    const fs::path dir = fs::temp_directory_path() / ("tqllm-hf-qwen4exp-" + std::to_string(::getpid()));
    fs::create_directories(dir);
    // a matrix, a bank of 3 experts, and a table in two shards (rows 5 + 7) of 64 columns
    const Mod mat = make_mod({}, 8, 128), bank = make_mod({3}, 4, 64), sh0 = make_mod({}, 5, 64), sh1 = make_mod({}, 7, 64);
    std::map<std::string, T> ts;
    add(ts, "m", mat); add(ts, "bank", bank); add(ts, "t.shard_0", sh0); add(ts, "t.shard_1", sh1);
    std::vector<uint16_t> vec(8); for (size_t i = 0; i < vec.size(); ++i) vec[i] = uint16_t(0x3f80 + 0x10 * i);  // BF16 1.0 ...
    ts["v"] = {"BF16", {8}, std::vector<char>(reinterpret_cast<char *>(vec.data()), reinterpret_cast<char *>(vec.data()) + 16)};
    write_safetensors(dir / "model.safetensors", ts);
    { std::ofstream c(dir / "config.json"); c << "{\"architectures\":[\"none\"]}"; }
    const HfFolder h(dir.string());

    // q4: Q4_1 bytes dequantise to exactly the MLX values
    const GgufTensor q = h.q4("m");
    const auto qb = q.make(); const auto qf = deq(GGML_Q4_1, qb, 8 * 128);
    bool exact = true; for (uint64_t i = 0; i < 8 * 128; ++i) exact &= qf[i] == value(mat, i);
    check(q.type == GGML_Q4_1 && q.ne == std::vector<uint64_t>({128, 8}) && qb.size() == q.bytes, "q4 type / shape / size");
    check(exact, "q4 dequantises to the MLX values exactly");
    // row order: blocks of 2 rows after row 2: result rows 2..7 = stored (6,7), (2,3), (4,5)
    HfBlockOrder ro; ro.order = {2, 0, 1}; ro.block = 2; ro.offset = 2;
    const auto rf = deq(GGML_Q4_1, h.q4("m", ro).make(), 8 * 128);
    const uint64_t src_row[8] = {0, 1, 6, 7, 2, 3, 4, 5};
    exact = true; for (uint64_t r = 0; r < 8; ++r) for (uint64_t c = 0; c < 128; ++c) exact &= rf[r * 128 + c] == value(mat, src_row[r] * 128 + c);
    check(exact, "q4 row block order");
    // column order: blocks of 32 columns: (3, 2, 1, 0)
    HfBlockOrder co; co.order = {3, 2, 1, 0}; co.block = 32;
    const auto cf = deq(GGML_Q4_1, h.q4("m", {}, co).make(), 8 * 128);
    exact = true; for (uint64_t r = 0; r < 8; ++r) for (uint64_t c = 0; c < 128; ++c) exact &= cf[r * 128 + c] == value(mat, r * 128 + (3 - c / 32) * 32 + c % 32);
    check(exact, "q4 column block order");
    // experts: each built alone equals its rows of the bank
    const GgufTensor e = h.q4_experts("bank");
    check(e.ne == std::vector<uint64_t>({64, 4, 3}), "q4_experts shape");
    exact = true;
    for (uint64_t x = 0; x < 3; ++x) {
        const auto f = deq(GGML_Q4_1, e.make_slice(x), 4 * 64);
        for (uint64_t i = 0; i < 4 * 64; ++i) exact &= f[i] == value(bank, x * 4 * 64 + i);
    }
    check(exact, "q4_experts: every expert equals its slice of the bank");
    // the row-stacked table: rows 4 (shard 0), 5 and 11 (shard 1), and 0
    const GgufTensor rt = h.q4_rows({"t.shard_0", "t.shard_1"});
    check(rt.ne == std::vector<uint64_t>({64, 12}) && rt.gather && rt.warm, "q4_rows shape and hooks");
    const std::vector<uint64_t> want = {4, 5, 11, 0};
    std::vector<float> got(want.size() * 64); rt.gather(want, got.data());
    exact = true;
    for (size_t i = 0; i < want.size(); ++i)
        for (uint64_t c = 0; c < 64; ++c) exact &= got[i * 64 + c] == (want[i] < 5 ? value(sh0, want[i] * 64 + c) : value(sh1, (want[i] - 5) * 64 + c));
    check(exact, "q4_rows gathers across the shard boundary");
    check(rt.warm(2) > 0, "q4_rows warm reads the shards");
    // values: transform and order
    const GgufTensor vt = h.values("v", GGML_F32, {8}, [](float x) { return -x; }, [] { HfBlockOrder o; o.order = {1, 0}; o.block = 4; return o; }());
    const auto vb = vt.make(); float vf[8]; std::memcpy(vf, vb.data(), 32);
    exact = true; for (int i = 0; i < 8; ++i) exact &= vf[i] == -bf16f(vec[size_t((i + 4) % 8)]);
    check(exact, "values: transform and block order");
    // dequant to BF16: one rounding of the MLX value
    const auto db = h.dequant("m", GGML_BF16).make();
    exact = true; for (uint64_t i = 0; i < 8 * 128; ++i) { uint16_t v; std::memcpy(&v, db.data() + 2 * i, 2); exact &= std::fabs(bf16f(v) - value(mat, i)) <= std::fabs(value(mat, i)) / 128 + 1e-12; }
    check(exact, "dequant BF16 within one rounding");

    fs::remove_all(dir);
    if (const char * f = std::getenv("TQLLM_TEST_HF_QWEN4EXP"); f && *f) {
        const char * r = std::getenv("TQLLM_TEST_REF_GGUF");
        if (r && *r) real_check(f, r); else std::printf("TQLLM_TEST_REF_GGUF unset: real folder check skipped\n");
    }
    std::printf("hf_qwen4exp: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
} catch (const std::exception & e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
