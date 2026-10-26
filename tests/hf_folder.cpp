// A Hugging Face checkpoint folder opened in place (src/core/hf): a tiny MiniMax-M3 EXL3 folder written here (one dense
// layer, trellis matrices, BF16 norms and embedding, tokenizer.json) must come back from gguf_open() in the engine's
// names: X3 tensors whose bytes are the X3Record + the shard's trellis / suh / svh bytes, norms as F32 1 + w, the
// embedding as a BF16 range of the shard, the tokenizer keys; a folder of an unmapped architecture is not a checkpoint.
#include "gguf.hpp"
#include "hf_folder.hpp"
#include "x3_record.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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

std::vector<char> bytes(size_t n, uint8_t seed) {
    std::vector<char> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = char(uint8_t(seed + i * 37));
    return v;
}
std::vector<char> bf16s(const std::vector<float> & f) {
    std::vector<char> v(f.size() * 2);
    for (size_t i = 0; i < f.size(); ++i) { uint32_t u; std::memcpy(&u, &f[i], 4); const uint16_t b = uint16_t(u >> 16); std::memcpy(v.data() + 2 * i, &b, 2); }
    return v;
}
// one trellis matrix (K = 2): in x out
void trellis(std::map<std::string, T> & ts, const std::string & m, uint64_t in, uint64_t out, uint8_t seed) {
    ts[m + ".trellis"] = {"I16", {in / 16, out / 16, 32}, bytes(in / 16 * out / 16 * 32 * 2, seed)};
    ts[m + ".suh"] = {"F16", {in}, bytes(in * 2, seed + 1)};
    ts[m + ".svh"] = {"F16", {out}, bytes(out * 2, seed + 2)};
    ts[m + ".mul1"] = {"I32", {}, {'\x2d', '\xd1', '\xdc', '\x83'}};
}
}  // namespace

int main() try {
    const fs::path dir = fs::temp_directory_path() / ("tqllm-hf-folder-" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const uint64_t H = 128, V = 128, hd = 128, ff = 128;
    {
        std::ofstream(dir / "config.json") << R"({"architectures":["MiniMaxM3SparseForConditionalGeneration"],
"quantization_config":{"quant_method":"exl3","version":"0.0.2"},
"text_config":{"num_hidden_layers":1,"num_local_experts":4,"vocab_size":128,"max_position_embeddings":4096,"hidden_size":128,
"dense_intermediate_size":128,"intermediate_size":128,"shared_intermediate_size":128,"num_attention_heads":1,"num_key_value_heads":1,
"head_dim":128,"rms_norm_eps":1e-6,"rotary_dim":64,"rope_theta":10000,"num_experts_per_tok":2,"n_shared_experts":1,
"routed_scaling_factor":1.0,"swiglu_alpha":1.702,"swiglu_limit":7.0}})";
        std::ofstream(dir / "tokenizer.json") << R"({"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]},
"added_tokens":[{"id":3,"content":"<eos>","special":true}]})";
        std::ofstream(dir / "tokenizer_config.json") << R"({"eos_token":"<eos>"})";
        std::map<std::string, T> ts;
        const std::string P = "language_model.model.", s = P + "layers.0.";
        ts[P + "embed_tokens.weight"] = {"BF16", {V, H}, bytes(V * H * 2, 9)};
        ts[P + "norm.weight"] = {"BF16", {H}, bf16s(std::vector<float>(H, 0.25f))};
        for (const char * n : {"input_layernorm", "post_attention_layernorm"}) ts[s + n + ".weight"] = {"BF16", {H}, bf16s(std::vector<float>(H, -0.5f))};
        for (const char * n : {"self_attn.q_norm", "self_attn.k_norm"}) ts[s + n + ".weight"] = {"BF16", {hd}, bf16s(std::vector<float>(hd, 0.f))};
        trellis(ts, "language_model.lm_head", H, V, 11);
        trellis(ts, s + "self_attn.q_proj", H, hd, 21);
        trellis(ts, s + "self_attn.k_proj", H, hd, 31);
        trellis(ts, s + "self_attn.v_proj", H, hd, 41);
        trellis(ts, s + "self_attn.o_proj", hd, H, 51);
        trellis(ts, s + "mlp.gate_proj", H, ff, 61);
        trellis(ts, s + "mlp.up_proj", H, ff, 71);
        trellis(ts, s + "mlp.down_proj", ff, H, 81);
        write_safetensors(dir / "model.safetensors", ts);
    }
    check(hf_folder(dir.string()), "a MiniMax EXL3 folder is a checkpoint folder");
    const GgufFile f = gguf_open(dir.string());
    check(f.str("general.architecture") == "minimax-m3", "engine architecture");
    check(uint32_t(f.num("minimax-m3.block_count")) == 1 && uint32_t(f.num("minimax-m3.expert_count")) == 4, "architecture keys");

    const auto & q = f.tensors.at("blk.0.attn_q.weight");
    check(q.type == GGML_X3 && q.ne == std::vector<uint64_t>{H, hd}, "q_proj is an X3 [in, out] tensor");
    const auto qb = f.read(q);
    X3Record rec;
    std::memcpy(&rec, qb.data(), sizeof rec);
    check(x3_record_valid(rec) && rec.rows == hd && rec.cols == H && rec.words == 32 && rec.codec == X3_CODEC_MUL1, "X3 record");
    const auto tr = bytes(H / 16 * hd / 16 * 32 * 2, 21), suh = bytes(H * 2, 22), svh = bytes(hd * 2, 23);
    check(qb.size() == sizeof rec + tr.size() + suh.size() + svh.size() &&
          std::memcmp(qb.data() + sizeof rec, tr.data(), tr.size()) == 0 &&
          std::memcmp(qb.data() + sizeof rec + tr.size(), suh.data(), suh.size()) == 0 &&
          std::memcmp(qb.data() + sizeof rec + tr.size() + suh.size(), svh.data(), svh.size()) == 0, "X3 bytes are the shard's bytes");

    const auto nb = f.read(f.tensors.at("blk.0.attn_norm.weight"));
    float n0; std::memcpy(&n0, nb.data(), 4);
    check(f.tensors.at("blk.0.attn_norm.weight").type == GGML_F32 && std::fabs(n0 - 0.5f) < 1e-6f, "norm stored as 1 + w in F32");
    const auto & e = f.tensors.at("token_embd.weight");
    check(e.type == GGML_BF16 && e.ne == std::vector<uint64_t>{H, V} && !e.built(), "embedding is a BF16 range of the shard");
    check(f.read(e) == bytes(V * H * 2, 9), "embedding bytes");
    check(f.tensors.count("blk.0.ffn_down.weight") && !f.tensors.count("blk.0.ffn_gate_inp.weight"), "a dense layer maps to the dense FFN");
    check(f.kv.at("tokenizer.ggml.tokens").strs.at(2) == "ab" && f.kv.at("tokenizer.ggml.merges").strs.at(0) == "a b" &&
          uint32_t(f.num("tokenizer.ggml.eos_token_id")) == 3, "tokenizer keys");

    {   // an architecture with no map is not a checkpoint folder
        const fs::path other = dir / "other";
        fs::create_directories(other);
        std::ofstream(other / "config.json") << R"({"architectures":["SomethingElse"]})";
        check(!hf_folder(other.string()), "an unmapped architecture is not opened");
    }
    fs::remove_all(dir);
    std::printf(ok ? "hf_folder: OK\n" : "hf_folder: FAILED\n");
    return ok ? 0 : 1;
} catch (const std::exception & ex) { std::fprintf(stderr, "hf_folder: %s\n", ex.what()); return 1; }
