// Qwen3.8-27B (Qwen3_5ForConditionalGeneration, dense hybrid GDN + full attention) as the qwen35 GGUF engine reads it, from
// a folder as published (exl3 trellis matrices or BF16 / F16 / F32 weights); the vision tower is not read.
// The rules of llama.cpp's Qwen3.5 converter (conversion/qwen.py): every RMSNorm stored as 1 + w except the GDN output norm
// (linear_attn.norm); A_log -> -exp(A_log); conv1d [ch][1][k] -> [ch][k]; and the GDN value heads from grouped (per key
// head) to tiled order: the value rows of in_proj_qkv, the rows of in_proj_z / in_proj_a / in_proj_b, A, dt_bias, the conv
// value channels and the columns of out_proj. Value heads are 128 wide, so an exl3 matrix moves whole Hadamard blocks.
#include "hf_arch.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace tqllm {

namespace {
float neg_exp(float v) { return -std::exp(v); }
}  // namespace

void hf_map_qwen35(const HfFolder & h, GgufFile & f) {
    const std::string A = "qwen35.";
    auto key = [&](const std::string & k, double v) { f.kv[A + k].num = v; };
    const Json & c = h.config();
    const int L = int(h.num("num_hidden_layers"));
    const int mtp = c.get("mtp_num_hidden_layers") ? int(h.num("mtp_num_hidden_layers")) : 0;
    const uint64_t kh = uint64_t(h.num("linear_num_key_heads")), vh = uint64_t(h.num("linear_num_value_heads"));
    const uint64_t vd = uint64_t(h.num("linear_value_head_dim")), kd = uint64_t(h.num("linear_key_head_dim"));
    const uint64_t vocab = uint64_t(h.num("vocab_size"));
    f.kv["general.name"].str = "Qwen3.8-27B";
    key("block_count", L + mtp);
    key("nextn_predict_layers", mtp);
    key("context_length", h.num("max_position_embeddings"));
    key("embedding_length", h.num("hidden_size"));
    key("feed_forward_length", h.num("intermediate_size"));
    key("attention.head_count", h.num("num_attention_heads"));
    key("attention.head_count_kv", h.num("num_key_value_heads"));
    key("attention.key_length", h.num("head_dim"));
    key("attention.value_length", h.num("head_dim"));
    key("attention.layer_norm_rms_epsilon", h.num("rms_norm_eps"));
    key("ssm.conv_kernel", h.num("linear_conv_kernel_dim"));
    key("ssm.state_size", double(vd));
    key("ssm.group_count", double(kh));
    key("ssm.time_step_rank", double(vh));
    key("ssm.inner_size", double(vh * vd));
    key("full_attention_interval", h.num("full_attention_interval"));
    const Json * rope = h.section("rope_parameters");
    if (!rope) throw std::runtime_error("hf: qwen3_5 without rope_parameters");
    key("rope.freq_base", rope->get("rope_theta")->num);
    key("rope.dimension_count", std::floor(h.num("head_dim") * rope->get("partial_rotary_factor")->num));
    std::vector<double> sections;
    for (const auto & v : rope->get("mrope_section")->arr) sections.push_back(v.num);
    while (sections.size() < 4) sections.push_back(0);
    f.kv[A + "rope.dimension_sections"].arr = sections;
    h.tokenizer(f, "qwen35", vocab);

    const Json * types = c.get("layer_types");
    auto full = [&](int l) { return types ? types->arr[size_t(l)].str == "full_attention" : (l + 1) % int(h.num("full_attention_interval")) == 0; };
    HfBlockOrder heads;  // result head j (tiled: j = r * kh + g) is stored head g * (vh / kh) + r
    for (uint64_t j = 0; j < vh; ++j) heads.order.push_back(uint32_t((j % kh) * (vh / kh) + j / kh));
    auto order = [&](uint64_t block, uint64_t offset) { HfBlockOrder o = heads; o.block = block; o.offset = offset; return o; };
    const uint64_t qk = 2 * kh * kd;

    auto & T = f.tensors;
    const std::string P = "model.language_model.";
    auto vec = [&](const std::string & n) { return h.values(n, GGML_F32, {h.at(n).shape[0]}); };
    T["token_embd.weight"] = h.plain(P + "embed_tokens.weight");
    T["output_norm.weight"] = h.norm(P + "norm.weight", true);
    T["output.weight"] = h.matrix("lm_head");
    auto attention = [&](const std::string & s, const std::string & d) {
        T[d + "attn_q.weight"] = h.matrix(s + "self_attn.q_proj");
        T[d + "attn_k.weight"] = h.matrix(s + "self_attn.k_proj");
        T[d + "attn_v.weight"] = h.matrix(s + "self_attn.v_proj");
        T[d + "attn_output.weight"] = h.matrix(s + "self_attn.o_proj");
        T[d + "attn_q_norm.weight"] = h.norm(s + "self_attn.q_norm.weight", true);
        T[d + "attn_k_norm.weight"] = h.norm(s + "self_attn.k_norm.weight", true);
    };
    auto mlp = [&](const std::string & s, const std::string & d) {
        T[d + "attn_norm.weight"] = h.norm(s + "input_layernorm.weight", true);
        T[d + "post_attention_norm.weight"] = h.norm(s + "post_attention_layernorm.weight", true);
        T[d + "ffn_gate.weight"] = h.matrix(s + "mlp.gate_proj");
        T[d + "ffn_up.weight"] = h.matrix(s + "mlp.up_proj");
        T[d + "ffn_down.weight"] = h.matrix(s + "mlp.down_proj");
    };
    for (int l = 0; l < L; ++l) {
        const std::string s = P + "layers." + std::to_string(l) + ".", d = "blk." + std::to_string(l) + ".";
        mlp(s, d);
        if (full(l)) { attention(s, d); continue; }
        const std::string la = s + "linear_attn.";
        T[d + "attn_qkv.weight"] = h.matrix(la + "in_proj_qkv", order(vd, qk), {});
        T[d + "attn_gate.weight"] = h.matrix(la + "in_proj_z", order(vd, 0), {});
        T[d + "ssm_alpha.weight"] = h.matrix(la + "in_proj_a", order(1, 0), {});
        T[d + "ssm_beta.weight"] = h.matrix(la + "in_proj_b", order(1, 0), {});
        T[d + "ssm_out.weight"] = h.matrix(la + "out_proj", {}, order(vd, 0));
        T[d + "ssm_a"] = h.values(la + "A_log", GGML_F32, {vh}, neg_exp, order(1, 0));
        T[d + "ssm_dt.bias"] = h.values(la + "dt_bias", GGML_F32, {vh}, nullptr, order(1, 0));
        const auto & conv = h.at(la + "conv1d.weight");
        T[d + "ssm_conv1d.weight"] = h.values(la + "conv1d.weight", GGML_F32, {conv.shape[0], conv.shape.back()}, nullptr, order(vd, qk));
        T[d + "ssm_norm.weight"] = vec(la + "norm.weight");
    }
    if (mtp && (h.has("mtp.fc.trellis") || h.has("mtp.fc.weight"))) {  // the MTP block, blk.<layers>
        const std::string s = "mtp.layers.0.", d = "blk." + std::to_string(L) + ".";
        mlp(s, d);
        attention(s, d);
        T[d + "nextn.eh_proj.weight"] = h.matrix("mtp.fc");
        T[d + "nextn.enorm.weight"] = h.norm("mtp.pre_fc_norm_embedding.weight", true);
        T[d + "nextn.hnorm.weight"] = h.norm("mtp.pre_fc_norm_hidden.weight", true);
        T[d + "nextn.shared_head_norm.weight"] = h.norm("mtp.norm.weight", true);
    }
}

}  // namespace tqllm
