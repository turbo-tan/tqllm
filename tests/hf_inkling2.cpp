// Inkling-2-Mini experts from the published folder (src/core/hf/arch_inkling2_loop.cpp, HfFolder::nvfp4_experts /
// mxfp8_experts). Synthetic part (always): a tiny folder with a modelopt NVFP4 stack (per-expert amax / denominator) and
// an MXFP8 stack, w13 rows interleaved gate / up; every built slice, decoded by the CPU reference (dequantize_host[_rows]),
// must equal the transformers PR #51050 formula (Inkling2LoopDequantizeExperts + Interleave) after BF16 rounding.
// Real part (when INKLING2_DIR names the checkpoint folder and it holds golden/dequant_ref.{json,bin}):
// hf_open() maps the folder, and 60 expert rows of layers 0, 19 (NVFP4) and 20, 23 (MXFP8) equal the BF16 rows the PR's
// own load ops produced (golden/dequant_ref.py).
#include "dequant.hpp"
#include "gguf.hpp"
#include "hf_folder.hpp"
#include "nlohmann/json.hpp"
#include "models/inkling2/loop/inkling2_model.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
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

// the PR's decoding, written independently of src/formats
const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
    return s ? -v : v;
}
uint16_t bf16(float f) {  // round to nearest even
    uint32_t u; std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
std::vector<char> f32s(const std::vector<float> & v) { std::vector<char> o(v.size() * 4); std::memcpy(o.data(), v.data(), o.size()); return o; }

// decoded rows of a built tensor's expert slice, as BF16. NVFP4G records are decoded in the PR's order, (e2m1 x scale) x
// global, so equality is about the bytes the loader built; the shared CPU reference (s x global first) is held to one
// F32 ulp of it. MXFP8 rows go through dequantize_host_rows (one rounding, exact).
size_t ulp_worst = 0;
std::vector<uint16_t> decode(const GgufTensor & t, uint64_t e) {
    const auto raw = t.make_slice(e);
    const size_t n = size_t(t.ne[0] * t.ne[1]);
    std::vector<float> f(n);
    dequantize_host_rows(t.type, raw.data(), f.data(), int64_t(t.ne[1]), int64_t(t.ne[0]));
    std::vector<uint16_t> out(n);
    if (t.type != GGML_NVFP4G) { for (size_t i = 0; i < n; ++i) out[i] = bf16(f[i]); return out; }
    for (size_t b = 0; b < n / 128; ++b) {
        const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 76;
        float g; std::memcpy(&g, blk, 4);
        for (int j = 0; j < 128; ++j) {
            const uint8_t q = blk[12 + j / 2];
            const float v = (kE2M1[(j & 1) ? q >> 4 : q & 15] * e4m3(blk[4 + j / 16])) * g;
            out[b * 128 + j] = bf16(v);
            uint32_t a, c; std::memcpy(&a, &v, 4); std::memcpy(&c, &f[b * 128 + j], 4);
            const size_t d = a > c ? a - c : c - a;
            if (d > ulp_worst) ulp_worst = d;
        }
    }
    return out;
}

void synthetic() {
    const fs::path dir = fs::temp_directory_path() / ("tqllm-inkling2-" + std::to_string(getpid()));
    fs::create_directories(dir);
    std::mt19937 rng(7);
    auto byte = [&] { return uint8_t(rng()); };
    auto e4m3_byte = [&] { uint8_t b; do b = byte(); while ((b & 0x7f) == 0x7f); return b; };  // no NaN
    const uint64_t E = 3, I = 4, R = 2 * I, K = 256, KD = 128;  // w13 [E][2I][K], w2 [E][K2][I2] small
    std::map<std::string, T> ts;
    const std::string nv = "nv.w13_weight", nd = "nv.w2_weight", mx = "mx.w13_weight";
    T codes{"U8", {E, R, K / 2}, {}}, scales{"F8_E4M3", {E, R, K / 16}, {}};
    for (uint64_t i = 0; i < E * R * K / 2; ++i) codes.data.push_back(char(byte()));
    for (uint64_t i = 0; i < E * R * K / 16; ++i) scales.data.push_back(char(e4m3_byte()));
    std::vector<float> amax = {0.11f, 0.3310525f, 0.4876f};
    ts[nv] = codes; ts[nv + "_scale"] = scales;
    ts[nv + "_amax"] = {"F32", {E}, f32s(amax)};
    ts[nv + "_outer_scale_denominator"] = {"F32", {1}, f32s({2688.0f})};
    T dcodes{"U8", {E, KD, I * 32 / 2}, {}}, dscales{"F8_E4M3", {E, KD, I * 32 / 16}, {}};  // down: in = 128
    for (uint64_t i = 0; i < E * KD * I * 16; ++i) dcodes.data.push_back(char(byte()));
    for (uint64_t i = 0; i < E * KD * I * 2; ++i) dscales.data.push_back(char(e4m3_byte()));
    ts[nd] = dcodes; ts[nd + "_scale"] = dscales;
    ts[nd + "_amax"] = {"F32", {E}, f32s({0.15f, 0.6793f, 0.2f})};
    ts[nd + "_outer_scale_denominator"] = {"F32", {1}, f32s({2688.0f})};
    T mq{"F8_E4M3", {E, R, K}, {}}, ms{"U8", {E, R, K / 32}, {}};
    for (uint64_t i = 0; i < E * R * K; ++i) mq.data.push_back(char(e4m3_byte()));
    for (uint64_t i = 0; i < E * R * K / 32; ++i) ms.data.push_back(char(110 + rng() % 12));
    ms.data[5] = 0;  // E8M0 0 = 2^-127
    ts[mx] = mq; ts[mx + "_scale"] = ms;
    write_safetensors(dir / "model.safetensors", ts);
    std::ofstream(dir / "config.json") << "{\"architectures\":[\"None\"]}";
    const HfFolder h(dir.string());

    // reference rows: [E][rows][cols] BF16 of the PR formula, w13 de-interleaved (gate = even rows, up = odd rows)
    auto nv_ref = [&](const T & c, const T & s, const std::vector<float> & am, uint64_t rows, uint64_t cols, uint64_t e, uint64_t r) {
        std::vector<uint16_t> out(cols);
        for (uint64_t j = 0; j < cols; ++j) {
            const uint8_t q = uint8_t(c.data[(e * rows + r) * (cols / 2) + j / 2]);
            const float v = kE2M1[(j & 1) ? q >> 4 : q & 15] * e4m3(uint8_t(s.data[(e * rows + r) * (cols / 16) + j / 16]));
            out[j] = bf16(v * (am[e] / 2688.0f));
        }
        return out;
    };
    auto mx_ref = [&](uint64_t e, uint64_t r) {
        std::vector<uint16_t> out(K);
        for (uint64_t j = 0; j < K; ++j)
            out[j] = bf16(e4m3(uint8_t(mq.data[(e * R + r) * K + j])) * std::exp2(float(uint8_t(ms.data[(e * R + r) * (K / 32) + j / 32])) - 127.0f));
        return out;
    };
    const std::vector<float> damax = {0.15f, 0.6793f, 0.2f};
    const GgufTensor g = h.nvfp4_experts(nv, 0, 2, I), u = h.nvfp4_experts(nv, 1, 2, I), d = h.nvfp4_experts(nd);
    const GgufTensor mg = h.mxfp8_experts(mx, 0, 2, I), mu = h.mxfp8_experts(mx, 1, 2, I);
    check(g.type == GGML_NVFP4G && g.ne == std::vector<uint64_t>{K, I, E}, "NVFP4 gate tensor type / shape");
    check(d.ne == std::vector<uint64_t>{I * 32, KD, E}, "NVFP4 down shape");
    check(mg.type == GGML_MXFP8 && mg.ne == std::vector<uint64_t>{K, I, E}, "MXFP8 gate tensor type / shape");
    check(g.bytes == E * I * (K / 128) * 76 && mg.bytes == E * I * (K + K / 32), "byte counts");
    size_t rows = 0;
    for (uint64_t e = 0; e < E; ++e) {
        const auto G = decode(g, e), U = decode(u, e), Dn = decode(d, e), MG = decode(mg, e), MU = decode(mu, e);
        for (uint64_t r = 0; r < I; ++r) {
            const auto rg = nv_ref(codes, scales, amax, R, K, e, 2 * r), ru = nv_ref(codes, scales, amax, R, K, e, 2 * r + 1);
            check(std::memcmp(G.data() + r * K, rg.data(), K * 2) == 0, "NVFP4 gate row " + std::to_string(r) + " expert " + std::to_string(e));
            check(std::memcmp(U.data() + r * K, ru.data(), K * 2) == 0, "NVFP4 up row " + std::to_string(r) + " expert " + std::to_string(e));
            const auto mgr = mx_ref(e, 2 * r), mur = mx_ref(e, 2 * r + 1);
            check(std::memcmp(MG.data() + r * K, mgr.data(), K * 2) == 0, "MXFP8 gate row " + std::to_string(r) + " expert " + std::to_string(e));
            check(std::memcmp(MU.data() + r * K, mur.data(), K * 2) == 0, "MXFP8 up row " + std::to_string(r) + " expert " + std::to_string(e));
            rows += 4;
        }
        for (uint64_t r = 0; r < KD; ++r) {
            const auto rd = nv_ref(dcodes, dscales, damax, KD, I * 32, e, r);
            check(std::memcmp(Dn.data() + r * I * 32, rd.data(), I * 32 * 2) == 0, "NVFP4 down row " + std::to_string(r));
            ++rows;
        }
    }
    bool threw = false;
    try { (void)h.nvfp4_experts(nv, 1, 2, I + 1); } catch (const std::exception &) { threw = true; }
    check(threw, "a row selection past the stack is refused");
    check(ulp_worst <= 1, "CPU reference within one F32 ulp of the PR order");
    std::printf("synthetic: %zu rows equal the PR formula (BF16); CPU reference worst %zu F32 ulp\n", rows, ulp_worst);
    fs::remove_all(dir);
}

void real() {
    const char * env = std::getenv("INKLING2_DIR");
    if (!env) { std::printf("real: skipped (INKLING2_DIR not set)\n"); return; }
    const fs::path dir = env;
    const fs::path ref = dir / "golden" / "dequant_ref";
    if (!fs::exists(dir / "config.json") || !fs::exists(ref.string() + ".json")) {
        std::printf("real: skipped (no checkpoint / golden at %s)\n", dir.string().c_str());
        return;
    }
    const GgufFile f = gguf_open(dir.string());
    check(f.kv.at("general.architecture").str == "inkling2-loop", "architecture");
    check(f.tensors.at("output.weight").ne == std::vector<uint64_t>{2304, 200058}, "unembed cut to the unpadded vocab");
    check(f.tensors.at("blk.0.ffn_gate_exps.weight").type == GGML_NVFP4G, "layer 0 experts NVFP4");
    check(f.tensors.at("blk.23.ffn_down_exps.weight").type == GGML_MXFP8, "layer 23 experts MXFP8");
    check(f.tensors.count("blk.5.attn_k_conv_proj.weight") == 1 && f.tensors.count("blk.4.attn_k_conv_proj.weight") == 0, "full layer 5");
    check(f.tensors.count("blk.6.loop_ctrl_q_down.weight") == 1 && f.tensors.count("blk.18.loop_ctrl_q_down.weight") == 0, "loop controllers 6..17");
    check(f.kv.at("inkling2-loop.attention.full_layers").str == "000001000001000001000001", "full-attention layer pattern");
    std::ifstream js(ref.string() + ".json"), bin(ref.string() + ".bin", std::ios::binary);
    std::stringstream ss; ss << js.rdbuf();
    const std::string blob((std::istreambuf_iterator<char>(bin)), {});
    const Json entries = Parser(ss.str()).parse();
    std::map<std::string, std::vector<uint16_t>> cache;
    size_t n = 0;
    for (const Json & x : entries.arr) {
        const int layer = int(x.get("layer")->num), e = int(x.get("expert")->num), r = int(x.get("row")->num), cols = int(x.get("cols")->num);
        const std::string proj = x.get("proj")->str;
        const std::string name = "blk." + std::to_string(layer) + ".ffn_" + proj + "_exps.weight";
        const GgufTensor & t = f.tensors.at(name);
        const std::string k = name + "#" + std::to_string(e);
        if (!cache.count(k)) cache[k] = decode(t, uint64_t(e));
        const auto & got = cache[k];
        check(uint64_t(cols) == t.ne[0], name + " cols");
        const bool same = std::memcmp(got.data() + size_t(r) * cols, blob.data() + size_t(x.get("offset")->num) * 2, size_t(cols) * 2) == 0;
        check(same, name + " expert " + std::to_string(e) + " row " + std::to_string(r) + " equals the PR's load");
        ++n;
    }
    check(ulp_worst <= 1, "CPU reference within one F32 ulp of the PR order");
    std::printf("real: %zu expert rows equal transformers PR #51050's BF16 load; CPU reference worst %zu F32 ulp\n", n, ulp_worst);
}
}  // namespace

// --exact off picks INT8 dense on sm_80+; exact keeps BF16 everywhere
void dense_mode() {
    const struct { bool exact; int ma, mi, want; } c[] = {{true, 8, 6, 0}, {false, 8, 6, 1}, {false, 8, 0, 1}, {false, 8, 7, 1},
                                                          {false, 8, 9, 1}, {false, 12, 1, 1}, {false, 10, 0, 1}, {true, 12, 1, 0}, {false, 7, 5, 0}};
    for (const auto & x : c)
        if (int(inkling2_int8_dense(x.exact, x.ma, x.mi)) != x.want) {
            std::printf("dense mode: exact %d sm_%d%d -> %d, want %d\n", int(x.exact), x.ma, x.mi, 1 - x.want, x.want); ok = false;
        }
}

int main() {
    dense_mode();
    synthetic();
    real();
    std::printf(ok ? "hf_inkling2: OK\n" : "hf_inkling2: FAILED\n");
    return ok ? 0 : 1;
}
