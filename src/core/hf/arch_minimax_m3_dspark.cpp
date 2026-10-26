// The MiniMax-M3.1-Flash DSpark draft (DSparkMiniMaxDraftModel, the checkpoint's dspark/ folder): the names and keys
// tools/minimax_convert.cpp --dspark writes. Its token embedding and LM head are the target's.
#include "hf_arch.hpp"

#include <string>

namespace tqllm {

void hf_map_minimax_m3_dspark(const HfFolder & h, GgufFile & f) {
    const std::string A = "minimax-m3.";
    auto key = [&](const std::string & k, double v) { f.kv[A + k].num = v; };
    const int L = int(h.num("num_hidden_layers"));
    f.kv["general.name"].str = "MiniMax-M3.1-Flash DSpark draft";
    key("block_count", L);
    key("embedding_length", h.num("hidden_size"));
    key("attention.head_count", h.num("num_attention_heads"));
    key("attention.head_count_kv", h.num("num_key_value_heads"));
    key("attention.key_length", h.num("head_dim"));
    key("attention.value_length", h.num("head_dim"));
    key("attention.layer_norm_rms_epsilon", h.num("rms_norm_eps"));
    key("rope.dimension_count", h.num("rotary_dim"));
    key("rope.freq_base", h.num("rope_theta"));
    key("vocab_size", h.num("vocab_size"));
    key("dspark.block_size", h.num("dspark_block_size"));
    key("dspark.mask_token_id", h.num("dspark_noise_token_id"));
    key("dspark.markov_rank", h.num("dspark_markov_rank"));
    key("dspark.sliding_window", h.num("sliding_window"));
    key("dspark.feed_forward_length", h.num("dspark_ffn_hidden_size"));
    std::string ids;
    const Json * t = h.config().get("dspark_target_layer_ids");
    if (!t || t->kind != Json::Array) throw std::runtime_error("hf: DSpark config has no dspark_target_layer_ids");
    for (const auto & v : t->arr) ids += (ids.empty() ? "" : ",") + std::to_string(int(v.num));
    f.kv[A + "dspark.target_layers"].str = ids;

    const std::string D = "language_model.model.dspark.";
    auto & T = f.tensors;
    T["dspark.fc.weight"] = h.matrix(D + "fc");
    T["dspark.hidden_norm.weight"] = h.norm(D + "hidden_norm.weight", true);
    T["dspark.output_norm.weight"] = h.norm(D + "final_norm.weight", true);
    for (int l = 0; l < L; ++l) {
        const std::string s = D + "layers." + std::to_string(l) + ".decoder_layer.", d = "dspark.blk." + std::to_string(l) + ".";
        T[d + "attn_norm.weight"] = h.norm(s + "input_layernorm.weight", true);
        T[d + "ffn_norm.weight"] = h.norm(s + "post_attention_layernorm.weight", true);
        T[d + "attn_q.weight"] = h.matrix(s + "self_attn.q_proj");
        T[d + "attn_k.weight"] = h.matrix(s + "self_attn.k_proj");
        T[d + "attn_v.weight"] = h.matrix(s + "self_attn.v_proj");
        T[d + "attn_output.weight"] = h.matrix(s + "self_attn.o_proj");
        T[d + "attn_q_norm.weight"] = h.norm(s + "self_attn.q_norm.weight", true);
        T[d + "attn_k_norm.weight"] = h.norm(s + "self_attn.k_norm.weight", true);
        T[d + "ffn_gate.weight"] = h.matrix(s + "mlp.gate_proj");
        T[d + "ffn_up.weight"] = h.matrix(s + "mlp.up_proj");
        T[d + "ffn_down.weight"] = h.matrix(s + "mlp.down_proj");
    }
    T["dspark.markov_w1.weight"] = h.matrix(D + "markov_head.markov_w1");
    T["dspark.markov_w2.weight"] = h.matrix(D + "markov_head.markov_w2");
    T["dspark.confidence.weight"] = h.matrix(D + "confidence_head.proj");
}

}  // namespace tqllm
