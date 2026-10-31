// MiniMax-M3.1-Flash safetensors (NVFP4 routed experts, MXFP8 rest) -> GGUF with chosen quant types (own dequant;
// ggml's quantizers and GGUF writer). Text model only.
//
//   minimax_convert CKPT OUT.gguf [--layers N] [--experts Q4_K] [--dense Q5_K] [--head Q6_K] [--embd Q8_0]
//                   [--imatrix IM.dat] [--threads 16] [--exl3 PACK]
//
// --exl3 PACK: every matrix the EXL3 pack (exllamav3 mul1 trellis) holds is copied byte for byte as X3 records
// (GGUF type 320: X3Record + trellis + suh + svh per part, size in x3.bytes.<name>); the rest comes from CKPT as usual.
//
// Gemma-style norms (x * (1 + w)) are stored as 1 + w, so every norm in the file is a plain RMSNorm weight.
// build: g++ -O2 -std=c++20 tools/minimax_convert.cpp src/core/safetensors.cpp -Isrc/core -I$GGML/include
//        -L$GGML_LIB -lggml-base -lggml -pthread -Wl,-rpath,$GGML_LIB
#include "safetensors.hpp"
#include "x3_record.hpp"
#include "ggml.h"
#include "gguf.h"

#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <tuple>
#include <mutex>
#include "nlohmann/json.hpp"
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using tqllm::SafeTensor;
namespace fs = std::filesystem;

static float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = e == 15 && m == 7 ? NAN : e == 0 ? m / 8.0f * std::ldexp(1.0f, -6) : (1 + m / 8.0f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}
static const float E2M1[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};

// GSQ_Q2_0 rows, per 64-weight block: fp16 scale s and codes q in {-1, 0, 1, 2} minimising sum w_i (x_i - s q_i)^2
// (w = the imatrix column importance, 1 without one): both signs of the start scale, 5 least-squares refinements.
static void encode_gsq_q2_weighted(const float * x, char * out, int64_t rows, int64_t cols, const float * imp) {
    auto code = [](float r) { return r <= -0.5f ? -1 : r < 0.5f ? 0 : r < 1.5f ? 1 : 2; };
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < cols / 64; ++b) {
            const float * v = x + r * cols + b * 64;
            const float * w = imp ? imp + b * 64 : nullptr;
            float peak = 0;
            for (int i = 0; i < 64; ++i) peak = std::max(peak, std::fabs(v[i]));
            double best = INFINITY; float best_s = 0;
            for (int sign : {-1, 1}) {
                float s = float(sign) * peak / 2;
                for (int it = 0; it < 5 && s != 0; ++it) {
                    double xy = 0, yy = 0;
                    for (int i = 0; i < 64; ++i) { const double wi = w ? w[i] : 1.0; const int q = code(v[i] / s); xy += wi * v[i] * q; yy += wi * q * q; }
                    if (yy > 0) s = float(xy / yy);
                }
                s = float(_Float16(s));
                if (!std::isfinite(s)) continue;
                double e = 0;
                for (int i = 0; i < 64; ++i) { const double wi = w ? w[i] : 1.0; const int q = s ? code(v[i] / s) : 0; const double d = v[i] - double(s) * q; e += wi * d * d; }
                if (e < best) { best = e; best_s = s; }
            }
            char * dst = out + (r * (cols / 64) + b) * 18;
            const _Float16 hs = _Float16(best_s);
            std::memcpy(dst, &hs, 2);
            std::memset(dst + 2, 0, 16);
            for (int i = 0; i < 64; ++i) {
                const int q = best_s ? code(v[i] / best_s) : 0;
                dst[2 + i / 4] = char(uint8_t(dst[2 + i / 4]) | uint8_t((q + 1) << (2 * (i % 4))));
            }
        }
}

struct Ckpt {
    std::map<std::string, SafeTensor> idx;
    bool has(const std::string & n) const { return idx.count(n) != 0; }
    const SafeTensor & at(const std::string & n) const {
        auto it = idx.find(n);
        if (it == idx.end()) throw std::runtime_error("missing tensor " + n);
        return it->second;
    }
    std::vector<char> raw(const std::string & n) const { return tqllm::safetensors_read(at(n)); }
    // rows x cols F32 of module `p` (…proj, …experts.E.w1, …norm)
    std::vector<float> weight(const std::string & p, int64_t & rows, int64_t & cols) const {
        if (has(p + ".weight_packed")) {
            const auto & t = at(p + ".weight_packed");
            rows = int64_t(t.shape[0]); cols = int64_t(t.shape[1]) * 2;
            const auto q = raw(p + ".weight_packed"), s = raw(p + ".weight_scale"), g = raw(p + ".weight_global_scale");
            float gs; std::memcpy(&gs, g.data(), 4);
            if (int64_t(s.size()) != rows * cols / 16) throw std::runtime_error(p + ": nvfp4 scale shape");
            std::vector<float> w(size_t(rows * cols));
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < cols; c += 2) {
                    const uint8_t b = uint8_t(q[size_t(r * cols / 2 + c / 2)]);
                    const float sc = e4m3(uint8_t(s[size_t(r * cols / 16 + c / 16)])) / gs;
                    w[size_t(r * cols + c)] = E2M1[b & 15] * sc;
                    w[size_t(r * cols + c + 1)] = E2M1[b >> 4] * sc;
                }
            return w;
        }
        const auto & t = at(p + ".weight");
        rows = int64_t(t.shape[0]); cols = t.shape.size() > 1 ? int64_t(t.shape[1]) : 1;
        const auto d = raw(p + ".weight");
        std::vector<float> w(size_t(rows * cols));
        if (has(p + ".weight_scale_inv")) {
            const auto s = raw(p + ".weight_scale_inv");
            if (int64_t(s.size()) != rows * cols / 32) throw std::runtime_error(p + ": mxfp8 scale shape");
            for (int64_t i = 0; i < rows * cols; ++i)
                w[size_t(i)] = e4m3(uint8_t(d[size_t(i)])) * std::ldexp(1.0f, int(uint8_t(s[size_t(i / 32)])) - 127);
        } else if (t.dtype == "BF16") {
            for (int64_t i = 0; i < rows * cols; ++i) { uint32_t u = uint32_t(uint16_t(reinterpret_cast<const uint16_t *>(d.data())[i])) << 16; std::memcpy(&w[size_t(i)], &u, 4); }
        } else if (t.dtype == "F32") {
            std::memcpy(w.data(), d.data(), w.size() * 4);
        } else throw std::runtime_error(p + ": dtype " + t.dtype);
        return w;
    }
};

struct Out {
    std::string name;
    ggml_type type;
    std::vector<int64_t> ne;  // ggml order: cols, rows[, experts]
    std::function<std::vector<float>(int64_t part)> rows;  // F32 of part (expert) e, or the whole tensor when parts == 1
    int64_t parts = 1;
    std::vector<std::string> x3;  // --exl3: the pack's module per part; the tensor is copied as X3 records, not quantised
};

// GSQ_Q2_0 has no ggml type: GGML_TYPE_COUNT stands for it until the writer gives it its GGUF id
static ggml_type type_of(const std::string & s) {
    if (s == "GSQ_Q2_0" || s == "gsq_q2_0") return GGML_TYPE_COUNT;
    auto lower = [](std::string x) { for (auto & c : x) c = char(std::tolower(static_cast<unsigned char>(c))); return x; };
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_get_type_traits(ggml_type(t))->type_name;
        if (n && lower(s) == lower(n)) return ggml_type(t);
    }
    throw std::runtime_error("unknown ggml type " + s);
}

// imatrix as written by llama-imatrix (.dat legacy format): n_entries, then per entry: name, ncall, nval, values
static std::map<std::string, std::vector<float>> read_imatrix(const std::string & path) {
    std::map<std::string, std::vector<float>> m;
    if (path.empty()) return m;
    std::ifstream f(path, std::ios::binary);
    int n = 0; f.read(reinterpret_cast<char *>(&n), 4);
    for (int i = 0; i < n && f; ++i) {
        int len = 0; f.read(reinterpret_cast<char *>(&len), 4);
        std::string name(size_t(len), '\0'); f.read(name.data(), len);
        int ncall = 0, nval = 0; f.read(reinterpret_cast<char *>(&ncall), 4); f.read(reinterpret_cast<char *>(&nval), 4);
        std::vector<float> v(static_cast<size_t>(nval)); f.read(reinterpret_cast<char *>(v.data()), nval * 4);
        if (ncall > 0) for (auto & x : v) x /= float(ncall);
        m[name] = std::move(v);
    }
    return m;
}

int main(int argc, char ** argv) try {
    if (argc < 3) { std::fprintf(stderr, "usage: minimax_convert CKPT OUT.gguf [--layers N] [--experts T] [--dense T] [--head T] [--embd T] [--imatrix F] [--threads N]\n"); return 2; }
    const fs::path dir = argv[1]; const std::string out_path = argv[2];
    int layers = -1, threads = 16; std::string te = "Q4_K", td = "Q5_K", th = "Q6_K", tb = "Q8_0", imp, emap;
    bool dspark = false;  // --dspark 1: CKPT is the checkpoint's dspark/ folder; writes the draft model
    std::string keep_path;  // --keep-experts OLD.gguf: the routed expert tensors are copied byte for byte from OLD
    std::string exl3_dir;
    int exl3_max_k = 8;  // --exl3-max-k K: pack matrices above K bits are quantised from CKPT instead (codecs not built yet)
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--layers") layers = std::stoi(v); else if (k == "--experts") te = v; else if (k == "--dense") td = v;
        else if (k == "--head") th = v; else if (k == "--embd") tb = v; else if (k == "--imatrix") imp = v;
        else if (k == "--threads") threads = std::stoi(v); else if (k == "--experts-map") emap = v;
        else if (k == "--dspark") dspark = v == "1";
        else if (k == "--keep-experts") keep_path = v;
        else if (k == "--exl3") exl3_dir = v;
        else if (k == "--exl3-max-k") exl3_max_k = std::stoi(v);
        else throw std::runtime_error("unknown option " + k);
    }
    Ckpt ck; ck.idx = tqllm::safetensors_index(dir);
    Ckpt px; if (!exl3_dir.empty()) px.idx = tqllm::safetensors_index(exl3_dir);
    auto in_pack1 = [&](const std::string & src) {
        return !exl3_dir.empty() && px.has(src + ".trellis") && int(px.at(src + ".trellis").shape.back() / 16) <= exl3_max_k;
    };
    // the engine stacks q / k / v into one matrix of one type: q is copied only when k and v are too
    auto in_pack = [&](const std::string & src) {
        const std::string q = "self_attn.q_proj";
        if (src.size() > q.size() && src.compare(src.size() - q.size(), q.size(), q) == 0) {
            const std::string base = src.substr(0, src.size() - std::string("q_proj").size());
            return in_pack1(src) && in_pack1(base + "k_proj") && in_pack1(base + "v_proj");
        }
        return in_pack1(src);
    };
    std::stringstream cs; cs << std::ifstream(dir / "config.json").rdbuf();
    const std::string ctext = cs.str();
    const tqllm::Json root = tqllm::Parser(ctext).parse();
    const tqllm::Json & cfgj = root.get("text_config") ? *root.get("text_config") : root;
    auto num = [&](const tqllm::Json & o, const std::string & k) {
        const tqllm::Json * v = o.get(k);
        if (!v || v->kind != tqllm::Json::Number) throw std::runtime_error("config: no number " + k);
        return v->num;
    };
    struct Cfg { const tqllm::Json & j; decltype(num) & n; double operator[](const std::string & k) const { return n(j, k); } };
    Cfg cfg{cfgj, num};
    const tqllm::Json none{};
    const tqllm::Json * saj_p = cfgj.get("sparse_attention_config");
    const tqllm::Json & saj = saj_p && saj_p->kind == tqllm::Json::Object ? *saj_p : none;
    Cfg sa{saj, num};
    const int n_layer = layers > 0 ? layers : int(cfg["num_hidden_layers"]);
    const auto imatrix = read_imatrix(imp);
    struct Kept { uint32_t type; size_t offset, bytes; };
    std::map<std::string, Kept> kept;
    if (!keep_path.empty()) {
        gguf_init_params gp{true, nullptr};
        gguf_context * kg = gguf_init_from_file(keep_path.c_str(), gp);
        if (!kg) throw std::runtime_error("cannot read " + keep_path);
        const size_t base = gguf_get_data_offset(kg);
        for (int64_t i = 0; i < gguf_get_n_tensors(kg); ++i) {
            const std::string n = gguf_get_tensor_name(kg, i);
            if (n.find("_exps.weight") == std::string::npos) continue;
            kept[n] = {uint32_t(gguf_get_tensor_type(kg, i)), base + gguf_get_tensor_offset(kg, i), gguf_get_tensor_size(kg, i)};
        }
        gguf_free(kg);
        std::printf("keeping %zu expert tensors from %s\n", kept.size(), keep_path.c_str());
    }
    const ggml_type TE = type_of(te), TD = type_of(td), TH = type_of(th), TB = type_of(tb);
    // --experts-map "A-B:TYPE,C-D:TYPE[:down]": the routed experts of layers A..B take TYPE (others --experts);
    // a third field (gate, up or down) limits the item to that projection, and wins over a whole-layer item
    std::map<std::pair<int, std::string>, ggml_type> layer_type;
    for (size_t at = 0; at < emap.size();) {
        const size_t end = emap.find(',', at), colon = emap.find(':', at);
        const std::string item = emap.substr(at, end == std::string::npos ? std::string::npos : end - at);
        const std::string range = item.substr(0, item.find(':'));
        std::string ty = item.substr(item.find(':') + 1), proj;
        if (ty.find(':') != std::string::npos) { proj = ty.substr(ty.find(':') + 1); ty = ty.substr(0, ty.find(':')); }
        if (!proj.empty() && proj != "gate" && proj != "up" && proj != "down") throw std::runtime_error("--experts-map: projection " + proj);
        const int a = std::stoi(range.substr(0, range.find('-'))), b = range.find('-') == std::string::npos ? a : std::stoi(range.substr(range.find('-') + 1));
        for (int l = a; l <= b; ++l) layer_type[{l, proj}] = type_of(ty);
        (void)colon;
        at = end == std::string::npos ? emap.size() : end + 1;
    }
    const std::string P = "language_model.model.";

    std::vector<Out> outs;
    auto plain = [&](const std::string & name, const std::string & src, ggml_type t, bool gemma) {
        int64_t r, c; ck.weight(src, r, c);  // shape probe (cheap enough for metadata; norms / small)
        Out o{name, t, c == 1 ? std::vector<int64_t>{r} : std::vector<int64_t>{c, r}, nullptr, 1};
        o.rows = [&ck, src, gemma](int64_t) { int64_t a, b; auto w = ck.weight(src, a, b); if (gemma) for (auto & x : w) x += 1.0f; return w; };
        outs.push_back(std::move(o));
    };
    auto shape2 = [&](const std::string & src) {
        const auto & t = ck.has(src + ".weight_packed") ? ck.at(src + ".weight_packed") : ck.at(src + ".weight");
        const int64_t rows = int64_t(t.shape[0]), cols = int64_t(t.shape[1]) * (ck.has(src + ".weight_packed") ? 2 : 1);
        return std::pair<int64_t, int64_t>{rows, cols};
    };
    auto mat = [&](const std::string & name, const std::string & src, ggml_type t) {
        auto [rows, cols] = shape2(src);
        Out o{name, t, {cols, rows}, nullptr, 1};
        o.rows = [&ck, src](int64_t) { int64_t a, b; return ck.weight(src, a, b); };
        if (in_pack(src)) o.x3 = {src};
        outs.push_back(std::move(o));
    };
    if (dspark) {
        // DSpark draft (DFlash-style block drafter): fc over the concatenated target layer states, hidden_norm, dense
        // layers whose attention also reads the fused context, final norm, Markov bias W2 W1[prev token]; the token
        // embedding and lm_head are the target's (not in this file)
        const std::string D = "language_model.model.dspark.";
        mat("dspark.fc.weight", D + "fc", TD);
        plain("dspark.hidden_norm.weight", D + "hidden_norm", GGML_TYPE_F32, true);
        plain("dspark.output_norm.weight", D + "final_norm", GGML_TYPE_F32, true);
        for (int l = 0; l < n_layer; ++l) {
            const std::string s = D + "layers." + std::to_string(l) + ".decoder_layer.", d = "dspark.blk." + std::to_string(l) + ".";
            plain(d + "attn_norm.weight", s + "input_layernorm", GGML_TYPE_F32, true);
            plain(d + "ffn_norm.weight", s + "post_attention_layernorm", GGML_TYPE_F32, true);
            mat(d + "attn_q.weight", s + "self_attn.q_proj", TD);
            mat(d + "attn_k.weight", s + "self_attn.k_proj", TD);
            mat(d + "attn_v.weight", s + "self_attn.v_proj", TD);
            mat(d + "attn_output.weight", s + "self_attn.o_proj", TD);
            plain(d + "attn_q_norm.weight", s + "self_attn.q_norm", GGML_TYPE_F32, true);
            plain(d + "attn_k_norm.weight", s + "self_attn.k_norm", GGML_TYPE_F32, true);
            mat(d + "ffn_gate.weight", s + "mlp.gate_proj", TD);
            mat(d + "ffn_up.weight", s + "mlp.up_proj", TD);
            mat(d + "ffn_down.weight", s + "mlp.down_proj", TD);
        }
        mat("dspark.markov_w1.weight", D + "markov_head.markov_w1", GGML_TYPE_F16);  // [vocab][rank]: a row per token
        mat("dspark.markov_w2.weight", D + "markov_head.markov_w2", TD);             // [vocab][rank]: rank -> vocab bias
        mat("dspark.confidence.weight", D + "confidence_head.proj", GGML_TYPE_F32);
        outs.push_back({"dspark.confidence.bias", GGML_TYPE_F32, {1}, [&ck, D](int64_t) {
            const auto r = ck.raw(D + "confidence_head.proj.bias");
            uint32_t u = uint32_t(uint16_t(reinterpret_cast<const uint16_t *>(r.data())[0])) << 16; float f; std::memcpy(&f, &u, 4);
            return std::vector<float>{f}; }, 1});
    } else {
    outs.push_back({"token_embd.weight", TB, {}, nullptr, 1});
    {
        auto [rows, cols] = shape2(P + "embed_tokens"); outs.back().ne = {cols, rows};
        outs.back().rows = [&ck, P](int64_t) { int64_t a, b; return ck.weight(P + "embed_tokens", a, b); };
    }
    for (int l = 0; l < n_layer; ++l) {
        const std::string s = P + "layers." + std::to_string(l) + ".", d = "blk." + std::to_string(l) + ".";
        plain(d + "attn_norm.weight", s + "input_layernorm", GGML_TYPE_F32, true);
        plain(d + "ffn_norm.weight", s + "post_attention_layernorm", GGML_TYPE_F32, true);
        mat(d + "attn_q.weight", s + "self_attn.q_proj", TD);
        mat(d + "attn_k.weight", s + "self_attn.k_proj", TD);
        mat(d + "attn_v.weight", s + "self_attn.v_proj", TD);
        mat(d + "attn_output.weight", s + "self_attn.o_proj", TD);
        plain(d + "attn_q_norm.weight", s + "self_attn.q_norm", GGML_TYPE_F32, true);
        plain(d + "attn_k_norm.weight", s + "self_attn.k_norm", GGML_TYPE_F32, true);
        if (ck.has(s + "self_attn.index_q_proj.weight")) {
            mat(d + "indexer.attn_q.weight", s + "self_attn.index_q_proj", TD);
            mat(d + "indexer.attn_k.weight", s + "self_attn.index_k_proj", TD);
            plain(d + "indexer.q_norm.weight", s + "self_attn.index_q_norm", GGML_TYPE_F32, true);
            plain(d + "indexer.k_norm.weight", s + "self_attn.index_k_norm", GGML_TYPE_F32, true);
        }
        const std::string m = s + "block_sparse_moe.";
        if (ck.has(m + "gate.weight")) {
            plain(d + "ffn_gate_inp.weight", m + "gate", GGML_TYPE_F32, false);
            outs.push_back({d + "exp_probs_b.bias", GGML_TYPE_F32, {int64_t(ck.at(m + "e_score_correction_bias").shape[0])},
                            [&ck, m](int64_t) { int64_t a, b; auto r = ck.raw(m + "e_score_correction_bias"); std::vector<float> v(r.size() / 4); std::memcpy(v.data(), r.data(), r.size()); (void)a; (void)b; return v; }, 1});
            mat(d + "ffn_gate_shexp.weight", m + "shared_experts.gate_proj", TD);
            mat(d + "ffn_up_shexp.weight", m + "shared_experts.up_proj", TD);
            mat(d + "ffn_down_shexp.weight", m + "shared_experts.down_proj", TD);
            const int64_t E = int64_t(cfg["num_local_experts"]);
            for (auto [gname, w, proj] : {std::tuple<const char *, const char *, const char *>{"ffn_gate_exps", "w1", "gate"},
                                          {"ffn_up_exps", "w3", "up"}, {"ffn_down_exps", "w2", "down"}}) {
                auto [rows, cols] = shape2(m + "experts.0." + w);
                const auto it = layer_type.count({l, proj}) ? layer_type.find({l, proj}) : layer_type.find({l, ""});
                Out o{d + gname + ".weight", it != layer_type.end() ? it->second : TE, {cols, rows, E}, nullptr, E};
                const std::string base = m + "experts.", ws = w;
                o.rows = [&ck, base, ws](int64_t e) { int64_t a, b; return ck.weight(base + std::to_string(e) + "." + ws, a, b); };
                if (in_pack(base + "0." + ws))
                    for (int64_t e = 0; e < E; ++e) o.x3.push_back(base + std::to_string(e) + "." + ws);
                outs.push_back(std::move(o));
            }
        } else {
            mat(d + "ffn_gate.weight", s + "mlp.gate_proj", TD);
            mat(d + "ffn_up.weight", s + "mlp.up_proj", TD);
            mat(d + "ffn_down.weight", s + "mlp.down_proj", TD);
        }
    }
    plain("output_norm.weight", P + "norm", GGML_TYPE_F32, true);
    mat("output.weight", "language_model.lm_head", TH);
    }

    // metadata (own GGUF v3 writer: the expert type GSQ_Q2_0 = 302 is ours, ggml's writer does not know it)
    struct Kv { std::string key; int type; std::string s; uint32_t u = 0; float f = 0; };  // type 4 u32, 6 f32, 8 string
    std::vector<Kv> kvs;
    const std::string A = "minimax-m3";
    auto u32 = [&](const std::string & k, int64_t v) { kvs.push_back({A + "." + k, 4, "", uint32_t(v), 0}); };
    auto f32 = [&](const std::string & k, double v) { kvs.push_back({A + "." + k, 6, "", 0, float(v)}); };
    kvs.push_back({"general.architecture", 8, A});
    kvs.push_back({"general.name", 8, dspark ? "MiniMax-M3.1-Flash DSpark draft" : "MiniMax-M3.1-Flash"});
    if (dspark) {
        u32("dspark.block_size", int64_t(cfg["dspark_block_size"])); u32("dspark.mask_token_id", int64_t(cfg["dspark_noise_token_id"]));
        u32("dspark.markov_rank", int64_t(cfg["dspark_markov_rank"])); u32("dspark.sliding_window", int64_t(cfg["sliding_window"]));
        u32("dspark.feed_forward_length", int64_t(cfg["dspark_ffn_hidden_size"]));
        std::string ids;
        for (const auto & v : cfgj.get("dspark_target_layer_ids")->arr) ids += (ids.empty() ? "" : ",") + std::to_string(int(v.num));
        kvs.push_back({A + ".dspark.target_layers", 8, ids});
    }
    u32("block_count", n_layer); u32("context_length", int64_t(cfg["max_position_embeddings"]));
    u32("embedding_length", int64_t(cfg["hidden_size"])); u32("feed_forward_length", int64_t(cfg["dense_intermediate_size"]));
    u32("expert_feed_forward_length", int64_t(cfg["intermediate_size"]));
    u32("expert_shared_feed_forward_length", int64_t(cfg["shared_intermediate_size"]));
    u32("attention.head_count", int64_t(cfg["num_attention_heads"])); u32("attention.head_count_kv", int64_t(cfg["num_key_value_heads"]));
    u32("attention.key_length", int64_t(cfg["head_dim"])); u32("attention.value_length", int64_t(cfg["head_dim"]));
    f32("attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"]);
    u32("rope.dimension_count", int64_t(cfg["rotary_dim"])); f32("rope.freq_base", cfg["rope_theta"]);
    u32("expert_count", int64_t(cfg["num_local_experts"])); u32("expert_used_count", int64_t(cfg["num_experts_per_tok"]));
    u32("expert_shared_count", int64_t(cfg["n_shared_experts"])); f32("expert_weights_scale", cfg["routed_scaling_factor"]);
    u32("leading_dense_block_count", 3);
    f32("swiglu_alpha", cfg["swiglu_alpha"]); f32("swiglu_limit", cfg["swiglu_limit"]);
    if (saj_p && saj_p->kind == tqllm::Json::Object) {
    u32("indexer.head_count", int64_t(sa["sparse_num_index_heads"])); u32("indexer.key_length", int64_t(sa["sparse_index_dim"]));
    u32("indexer.top_k_blocks", int64_t(sa["sparse_topk_blocks"])); u32("indexer.block_size", int64_t(sa["sparse_block_size"]));
    u32("indexer.local_blocks", int64_t(sa["sparse_local_block"]));
    }
    u32("vocab_size", int64_t(cfg["vocab_size"]));

    constexpr uint32_t kX3 = 320;  // X3 (src/core/x3_record.hpp): per part X3Record "X3R2" + trellis + suh + svh
    // one part: trellis int16 [in/16][out/16][16 K] (EXL3 order, the X3 tile order), suh FP16 [in], svh FP16 [out]
    auto x3_part = [&](const std::string & src) {
        const auto & tr = px.at(src + ".trellis");
        if (tr.shape.size() != 3 || tr.dtype != "I16") throw std::runtime_error(src + ": trellis is not I16 [k/16][n/16][16K]");
        if (!px.has(src + ".mul1")) throw std::runtime_error(src + ": not a mul1 pack (no .mul1)");
        const uint32_t cols = uint32_t(tr.shape[0] * 16), rows = uint32_t(tr.shape[1] * 16), words = uint32_t(tr.shape[2]);
        std::vector<char> out(sizeof(tqllm::X3Record));
        const tqllm::X3Record rec = tqllm::x3_record_make(rows, cols, words, tqllm::X3_CODEC_MUL1);
        std::memcpy(out.data(), &rec, sizeof rec);
        for (const char * part : {".trellis", ".suh", ".svh"}) { const auto b = px.raw(src + part); out.insert(out.end(), b.begin(), b.end()); }
        if (out.size() != sizeof rec + size_t(rows / 16) * (cols / 16) * words * 2 + size_t(cols) * 2 + size_t(rows) * 2)
            throw std::runtime_error(src + ": X3 part size");
        return out;
    };
    auto x3_part_bytes = [&](const std::string & src) {
        const auto & tr = px.at(src + ".trellis");
        return sizeof(tqllm::X3Record) + size_t(tr.shape[0]) * tr.shape[1] * tr.shape[2] * 2 + size_t(tr.shape[0] * 16) * 2 + size_t(tr.shape[1] * 16) * 2;
    };
    constexpr uint32_t kGsqQ2 = 302;  // 64 weights: fp16 scale + 64 two-bit codes on {-1, 0, 1, 2} (src/formats/weight/gsq_q2_host.hpp)
    auto row_size = [&](uint32_t t, int64_t cols) -> size_t {
        return t == kGsqQ2 ? size_t(cols / 64) * 18 : ggml_row_size(ggml_type(t), cols);
    };
    std::vector<uint32_t> types(outs.size());
    std::vector<size_t> offsets(outs.size());
    const size_t align = 32;
    size_t off = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
        auto & o = outs[i];
        uint32_t t = o.type == GGML_TYPE_COUNT ? kGsqQ2 : uint32_t(o.type);
        const int64_t blk = t == kGsqQ2 ? 64 : ggml_blck_size(ggml_type(t));
        if (o.ne[0] % blk != 0) t = GGML_TYPE_F32;  // 1-D norms and odd widths stay F32
        if (const auto k = kept.find(o.name); k != kept.end()) t = k->second.type;
        int64_t rows = 1; for (size_t d = 1; d < o.ne.size(); ++d) rows *= o.ne[d];
        offsets[i] = off;
        if (!o.x3.empty()) {
            t = kX3;
            size_t b = 0;
            for (const auto & src : o.x3) b += x3_part_bytes(src);
            kvs.push_back({"x3.bytes." + o.name, 4, "", uint32_t(b), 0});
            if (b != size_t(uint32_t(b))) throw std::runtime_error(o.name + ": X3 tensor over 4 GB");
            off += b;
        } else {
            off += row_size(t, o.ne[0]) * size_t(rows);
        }
        types[i] = t;
        off += (align - off % align) % align;
    }
    std::FILE * f = std::fopen(out_path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + out_path);
    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    auto w64 = [&](uint64_t v) { std::fwrite(&v, 8, 1, f); };
    auto wstr = [&](const std::string & v) { w64(v.size()); std::fwrite(v.data(), 1, v.size(), f); };
    std::fwrite("GGUF", 1, 4, f); w32(3); w64(outs.size()); w64(kvs.size());
    for (const auto & kv : kvs) {
        wstr(kv.key); w32(uint32_t(kv.type));
        if (kv.type == 4) w32(kv.u); else if (kv.type == 6) std::fwrite(&kv.f, 4, 1, f); else wstr(kv.s);
    }
    for (size_t i = 0; i < outs.size(); ++i) {
        wstr(outs[i].name); w32(uint32_t(outs[i].ne.size()));
        for (int64_t d : outs[i].ne) w64(uint64_t(d));
        w32(types[i]); w64(offsets[i]);
    }
    {
        const long pos = std::ftell(f);
        static const char z[32] = {};
        std::fwrite(z, 1, size_t((align - size_t(pos) % align) % align), f);
    }
    size_t written = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
        auto & o = outs[i];
        const uint32_t t = types[i];
        const int64_t cols = o.ne[0], rows = o.ne.size() > 1 ? o.ne[1] : 1;
        const size_t row_bytes = row_size(t, cols), part_bytes = row_bytes * size_t(rows);
        if (t == kX3) {  // the pack's bytes behind one X3Record per part
            size_t b = 0;
            for (const auto & src : o.x3) { const auto part = x3_part(src); std::fwrite(part.data(), 1, part.size(), f); b += part.size(); }
            written += b;
            const size_t pad = (align - written % align) % align;
            static const char z32[32] = {};
            std::fwrite(z32, 1, pad, f);
            written += pad;
            std::printf("%-36s x3       %s %.3f GB (exl3)\n", o.name.c_str(), o.parts > 1 ? "experts" : "", b / 1e9);
            std::fflush(stdout);
            continue;
        }
        if (const auto k = kept.find(o.name); k != kept.end()) {  // copied, not requantised
            if (k->second.bytes != part_bytes * size_t(o.parts)) throw std::runtime_error(o.name + ": kept tensor size differs");
            std::FILE * src = std::fopen(keep_path.c_str(), "rb");
            if (!src || std::fseek(src, long(k->second.offset), SEEK_SET)) throw std::runtime_error("cannot seek " + keep_path);
            std::vector<char> buf(size_t(64) << 20);
            for (size_t left = k->second.bytes; left;) {
                const size_t n = std::min(left, buf.size());
                if (std::fread(buf.data(), 1, n, src) != n) throw std::runtime_error("short read " + keep_path);
                std::fwrite(buf.data(), 1, n, f);
                left -= n;
            }
            std::fclose(src);
            written += k->second.bytes;
            const size_t pad = (align - written % align) % align;
            static const char z32[32] = {};
            std::fwrite(z32, 1, pad, f);
            written += pad;
            std::printf("%-36s kept     %.3f GB\n", o.name.c_str(), k->second.bytes / 1e9);
            std::fflush(stdout);
            continue;
        }
        std::vector<char> data(part_bytes * size_t(o.parts));
        const float * im = nullptr;
        auto it = imatrix.find(o.name);
        if (it != imatrix.end() && int64_t(it->second.size()) == cols * (o.parts > 1 ? o.parts : 1)) im = it->second.data();
        if (t != kGsqQ2 && t != GGML_TYPE_F32) {
            if (ggml_quantize_requires_imatrix(ggml_type(t)) && !im) throw std::runtime_error(o.name + ": " + ggml_type_name(ggml_type(t)) + " needs an imatrix");
            ggml_quantize_init(ggml_type(t));
        }
        auto encode = [&](const float * src, char * dst, int64_t nrows, const float * imr) {
            if (t == GGML_TYPE_F32) { std::memcpy(dst, src, size_t(nrows * cols) * 4); return; }
            if (t == kGsqQ2) { encode_gsq_q2_weighted(src, dst, nrows, cols, imr); return; }
            ggml_quantize_chunk(ggml_type(t), src, dst, 0, nrows, cols, imr);
        };
        // experts: one task per expert; one matrix: row blocks of 256 rows
        std::vector<float> whole;
        if (o.parts == 1) { whole = o.rows(0); if (int64_t(whole.size()) != cols * rows) throw std::runtime_error(o.name + ": size mismatch"); }
        const int64_t step = 256, tasks = o.parts > 1 ? o.parts : (rows + step - 1) / step;
        std::atomic<int64_t> next{0}; std::exception_ptr err; std::mutex em;
        auto work = [&] {
            for (int64_t p; (p = next++) < tasks;) try {
                if (o.parts > 1) {
                    auto w = o.rows(p);
                    if (int64_t(w.size()) != cols * rows) throw std::runtime_error(o.name + ": size mismatch");
                    encode(w.data(), data.data() + size_t(p) * part_bytes, rows, im ? im + p * cols : nullptr);
                } else {
                    const int64_t r0 = p * step, n = std::min(step, rows - r0);
                    encode(whole.data() + r0 * cols, data.data() + size_t(r0) * row_bytes, n, im);
                }
            } catch (...) { std::lock_guard<std::mutex> lk(em); if (!err) err = std::current_exception(); }
        };
        std::vector<std::thread> pool;
        for (int k = 1; k < std::min<int64_t>(threads, tasks); ++k) pool.emplace_back(work);
        work(); for (auto & th : pool) th.join();
        if (err) std::rethrow_exception(err);
        std::fwrite(data.data(), 1, data.size(), f);
        written += data.size();
        const size_t pad = (align - written % align) % align;
        static const char zeros[32] = {};
        std::fwrite(zeros, 1, pad, f);
        written += pad;
        std::printf("%-36s %-8s %s %.3f GB%s\n", o.name.c_str(), t == kGsqQ2 ? "gsq_q2_0" : ggml_type_name(ggml_type(t)),
                    o.parts > 1 ? "experts" : "", data.size() / 1e9, im ? " (imatrix)" : "");
        std::fflush(stdout);
    }
    std::fclose(f);
    std::printf("wrote %s: %zu tensors, %.2f GB of data\n", out_path.c_str(), outs.size(), written / 1e9);
    return 0;
} catch (const std::exception & e) { std::fprintf(stderr, "minimax_convert: %s\n", e.what()); return 1; }
