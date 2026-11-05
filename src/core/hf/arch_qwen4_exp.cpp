// Qwen3.8 Flash-Next (Qwen4ExpForConditionalGeneration), MLX affine 4-bit, group 32 (TensorFold's and mlx-community's
// checkpoints): the qwen4exp names, metadata and tensor types of the GGUF engine, from the folder as published.
// Every 4-bit matrix stays 4-bit (Q4_1, the same values); the tensors the engine reads in BF16 / F32 (hyper-connection
// mixers, routers, norms, the PLE projections, the indexer projections, ssm alpha / beta) are dequantised once on read.
// The GGUF engine holds the gated-delta-net value heads tiled (head r * KH + g); the checkpoint groups them per key
// head (g * (VH / KH) + r): the value rows of qkv / z / alpha / beta / conv / A / dt and the columns of out_proj follow.
#include "hf_arch.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace tqllm {

namespace {
float neg_exp(float v) { return -std::exp(v); }
}  // namespace

void hf_map_qwen4_exp(const HfFolder & h, GgufFile & f) {
    const std::string A = "qwen4exp.";
    auto key = [&](const std::string & k, double v) { f.kv[A + k].num = v; };
    auto arr = [&](const std::string & k, std::vector<double> v) { f.kv[A + k].arr = std::move(v); };
    const Json & c = h.config();
    const int L = int(h.num("num_hidden_layers"));
    const uint64_t kh = uint64_t(h.num("linear_num_key_heads")), vh = uint64_t(h.num("linear_num_value_heads"));
    const uint64_t hd = uint64_t(h.num("linear_value_head_dim")), kd = uint64_t(h.num("linear_key_head_dim"));
    f.kv["general.name"].str = "Qwen3.8-Flash-Next";
    key("block_count", L);
    key("context_length", h.num("max_position_embeddings"));
    key("embedding_length", h.num("hidden_size"));
    key("attention.head_count", h.num("num_attention_heads"));
    key("attention.head_count_kv", h.num("num_key_value_heads"));
    key("attention.key_length", h.num("head_dim"));
    key("attention.value_length", h.num("head_dim"));
    key("attention.layer_norm_rms_epsilon", h.num("rms_norm_eps"));
    key("expert_count", h.num("num_experts"));
    key("expert_used_count", h.num("num_experts_per_tok"));
    key("expert_feed_forward_length", h.num("moe_intermediate_size"));
    key("expert_shared_feed_forward_length", h.num("shared_expert_intermediate_size"));
    key("ssm.conv_kernel", h.num("linear_conv_kernel_dim"));
    key("ssm.state_size", double(hd));
    key("ssm.group_count", double(kh));
    key("ssm.time_step_rank", double(vh));
    key("ssm.inner_size", double(vh * hd));
    key("full_attention_interval", h.num("full_attention_interval"));
    key("hyper_connection.count", h.num("hc_count"));
    key("hyper_connection.low_rank", h.num("hc_lowrank"));
    key("attention.indexer.head_count", h.num("indexer_n_heads"));
    key("attention.indexer.key_length", h.num("indexer_head_dim"));
    key("attention.indexer.top_k", h.num("indexer_budget"));
    key("ple.ngram_size", h.num("ngram_size"));
    key("ple.heads_per_ngram", h.num("heads_per_ngram"));
    key("ple.conv_kernel", h.num("ple_conv_kernel_size"));
    key("ple.eos_token_id", h.num("eos_token_id"));
    const Json * rope = h.section("rope_parameters");
    if (!rope) throw std::runtime_error("hf: qwen4_exp without rope_parameters");
    key("rope.freq_base", rope->get("rope_theta")->num);
    key("rope.dimension_count", std::floor(h.num("head_dim") * rope->get("partial_rotary_factor")->num));
    std::vector<double> sections;
    for (const auto & v : rope->get("mrope_section")->arr) sections.push_back(v.num);
    while (sections.size() < 4) sections.push_back(0);
    arr("rope.dimension_sections", sections);
    const Json * types = c.get("layer_types");
    std::vector<double> ratios;
    std::vector<bool> full = std::vector<bool>(static_cast<size_t>(L));
    for (int l = 0; l < L; ++l) {
        full[size_t(l)] = types && types->arr[size_t(l)].str == "full_attention";
        ratios.push_back(full[size_t(l)] ? h.num("indexer_compress_ratio") : 0);
    }
    arr("attention.compress_ratios", ratios);
    const std::string P = "language_model.model.", M = "language_model.mtp.";
    std::vector<double> ple_layers;
    for (int l = 0; l < L; ++l) if (h.has(P + "layers." + std::to_string(l) + ".ple.key_proj.weight")) ple_layers.push_back(l);
    arr("ple.layers", ple_layers);
    if (!ple_layers.empty()) {
        const std::string e = P + "layers." + std::to_string(int(ple_layers[0])) + ".ple.ple_embedding.";
        auto ints = [&](const std::string & n) {
            const auto raw = safetensors_read(h.at(e + n)); std::vector<double> v(raw.size() / 8);
            for (size_t i = 0; i < v.size(); ++i) { int64_t x; std::memcpy(&x, raw.data() + 8 * i, 8); v[i] = double(x); }
            return v;
        };
        key("embedding_length_per_layer_input", double(h.at(e + "ngram_embedding.shard_0.weight").shape[1] * 8));  // the table's row width
        arr("ple.layer_multipliers", ints("layer_multipliers"));
        arr("ple.head_offsets", ints("ngram_heads_offsets"));
        arr("ple.head_vocab_sizes", ints("ngram_heads_vocab_sizes"));
    }
    if (const Json * img = h.root().get("image_token_id")) key("ple.image_token_id", img->num);

    auto & T = f.tensors;
    T["token_embd.weight"] = h.q4("language_model.model.embed_tokens");
    T["output.weight"] = h.q4("language_model.lm_head");
    T["output_hc_down.weight"] = h.dequant(P + "hyper_connection_mixer.input_mix_weight_down", GGML_BF16);
    T["output_hc_up.weight"] = h.dequant(P + "hyper_connection_mixer.input_mix_weight_up", GGML_BF16);
    T["output_hc_norm.weight"] = h.values(P + "hyper_connection_mixer.hc_norm.weight", GGML_F32, {h.at(P + "hyper_connection_mixer.hc_norm.weight").shape[0]});

    // the GDN value-head order (above); kd * kh is one of q / k
    HfBlockOrder heads; for (uint64_t j = 0; j < vh; ++j) heads.order.push_back(uint32_t((j % kh) * (vh / kh) + j / kh));
    auto order = [&](uint64_t block, uint64_t offset) { HfBlockOrder o = heads; o.block = block; o.offset = offset; return o; };
    const uint64_t qk = 2 * kh * kd;

    // one decoder block: trunk layers (s = layers.N.) and the MTP block (s = mtp.layers.0.), d = blk.N.
    auto block = [&](const std::string & s, const std::string & d, bool attn, bool mtp) {
        const uint32_t router = mtp ? GGML_F32 : GGML_BF16;  // the GGUF engine's MTP router is F32
        auto vec = [&](const std::string & n) { return h.values(n, GGML_F32, {h.at(n).shape[0]}); };
        auto mixer = [&](const std::string & n) { return mtp ? h.q4(n) : h.dequant(n, GGML_BF16); };
        for (const char * part : {"attn", "mlp"}) {
            const std::string hc = s + part + "_hyper_connection.", g = d + "hc_" + (part[0] == 'a' ? "attn" : "ffn") + "_";
            T[g + "down.weight"] = mixer(hc + "input_mix_weight_down");
            T[g + "up.weight"] = mixer(hc + "input_mix_weight_up");
            T[g + "inject.weight"] = mixer(hc + "block_inject_weight");
            T[g + "norm.weight"] = vec(hc + "hc_norm.weight");
        }
        if (attn) {
            T[d + "attn_q.weight"] = h.q4(s + "self_attn.q_proj");
            T[d + "attn_k.weight"] = h.q4(s + "self_attn.k_proj");
            T[d + "attn_v.weight"] = h.q4(s + "self_attn.v_proj");
            T[d + "attn_output.weight"] = h.q4(s + "self_attn.o_proj");
            T[d + "attn_q_norm.weight"] = vec(s + "self_attn.q_norm.weight");
            T[d + "attn_k_norm.weight"] = vec(s + "self_attn.k_norm.weight");
            const std::string ix = s + "self_attn.indexer.";
            const uint64_t iq = uint64_t(h.num("indexer_n_heads") * h.num("indexer_head_dim")), ik = uint64_t(h.num("indexer_head_dim"));
            T[d + "indexer.q_proj.weight"] = h.dequant(ix + "index_qk_proj", GGML_BF16, {}, 0, iq);
            T[d + "indexer.k_proj.weight"] = h.dequant(ix + "index_qk_proj", GGML_BF16, {}, iq, ik);
            T[d + "indexer.q_norm.weight"] = vec(ix + "q_layernorm.weight");
            T[d + "indexer.k_norm.weight"] = vec(ix + "k_layernorm.weight");
        } else {
            const std::string la = s + "linear_attn.";
            T[d + "attn_qkv.weight"] = h.q4(la + "in_proj_qkv", order(hd, qk));
            T[d + "attn_gate.weight"] = h.q4(la + "in_proj_z", order(hd, 0));
            T[d + "ssm_alpha.weight"] = h.dequant(la + "in_proj_a", GGML_BF16, order(1, 0));
            T[d + "ssm_beta.weight"] = h.dequant(la + "in_proj_b", GGML_BF16, order(1, 0));
            T[d + "ssm_out.weight"] = h.q4(la + "out_proj", {}, order(hd, 0));
            T[d + "ssm_a"] = h.values(la + "A_log", GGML_F32, {vh}, neg_exp, order(1, 0));
            T[d + "ssm_dt.bias"] = h.values(la + "dt_bias", GGML_F32, {vh}, nullptr, order(1, 0));
            const auto & conv = h.at(la + "conv1d.weight");
            T[d + "ssm_conv1d.weight"] = h.values(la + "conv1d.weight", GGML_F32, {conv.shape[0], conv.shape[1]}, nullptr, order(hd, qk));
            T[d + "ssm_norm.weight"] = vec(la + "norm.weight");
        }
        T[d + "ffn_gate_inp.weight"] = h.dequant(s + "mlp.gate", router);
        GgufTensor sg = h.dequant(s + "mlp.shared_expert_gate", router); sg.ne = {sg.ne[0]};  // [1][hidden] -> [hidden]
        T[d + "ffn_gate_inp_shexp.weight"] = std::move(sg);
        T[d + "ffn_gate_shexp.weight"] = h.q4(s + "mlp.shared_expert.gate_proj");
        T[d + "ffn_up_shexp.weight"] = h.q4(s + "mlp.shared_expert.up_proj");
        T[d + "ffn_down_shexp.weight"] = h.q4(s + "mlp.shared_expert.down_proj");
        T[d + "ffn_gate_exps.weight"] = h.q4_experts(s + "mlp.switch_mlp.gate_proj");
        T[d + "ffn_up_exps.weight"] = h.q4_experts(s + "mlp.switch_mlp.up_proj");
        T[d + "ffn_down_exps.weight"] = h.q4_experts(s + "mlp.switch_mlp.down_proj");
    };
    for (int l = 0; l < L; ++l) {
        const std::string s = P + "layers." + std::to_string(l) + ".", d = "blk." + std::to_string(l) + ".";
        block(s, d, full[size_t(l)], false);
        if (h.has(s + "ple.key_proj.weight")) {
            const auto & conv = h.at(s + "ple.conv1d.weight");
            T[d + "ple_conv1d.weight"] = h.values(s + "ple.conv1d.weight", GGML_F16, {conv.shape[0], conv.shape[1]});
            T[d + "ple_key.weight"] = h.dequant(s + "ple.key_proj", GGML_BF16);
            T[d + "ple_value.weight"] = h.dequant(s + "ple.value_proj", GGML_BF16);
            for (const char * n : {"conv", "key", "query"}) {
                const std::string src = s + "ple.norm_" + n + ".weight";
                T[d + "ple_norm_" + n + ".weight"] = h.values(src, GGML_F32, {h.at(src).shape[0]});
            }
            std::vector<std::string> shards;
            for (int i = 0;; ++i) {
                const std::string m = s + "ple.ple_embedding.ngram_embedding.shard_" + std::to_string(i);
                if (!h.has(m + ".weight")) break;
                shards.push_back(m);
            }
            T["per_layer_token_embd.weight"] = h.q4_rows(shards);
        }
    }
    if (h.has(M + "fc_hidden.weight")) {  // the MTP block, blk.<layers>
        const std::string d = "blk." + std::to_string(L) + ".";
        block(M + "layers.0.", d, true, true);
        T[d + "nextn.eh_proj.weight"] = h.q4_cols({M + "fc_embedding", M + "fc_hidden"});
        T[d + "nextn.enorm.weight"] = h.values(M + "pre_fc_norm_embedding.weight", GGML_F32, {h.at(M + "pre_fc_norm_embedding.weight").shape[0]});
        T[d + "nextn.hnorm.weight"] = h.values(M + "pre_fc_norm_hidden.weight", GGML_F32, {h.at(M + "pre_fc_norm_hidden.weight").shape[0]});
        T[d + "nextn.hc_head_down.weight"] = h.q4(M + "hyper_connection_mixer.input_mix_weight_down");
        T[d + "nextn.hc_head_up.weight"] = h.q4(M + "hyper_connection_mixer.input_mix_weight_up");
        T[d + "nextn.hc_head_norm.weight"] = h.values(M + "hyper_connection_mixer.hc_norm.weight", GGML_F32, {h.at(M + "hyper_connection_mixer.hc_norm.weight").shape[0]});
    }
}

}  // namespace tqllm
