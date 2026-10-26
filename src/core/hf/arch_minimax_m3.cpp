// MiniMax-M3.1-Flash (MiniMaxM3SparseForConditionalGeneration): the names and keys tools/minimax_convert.cpp writes.
#include "hf_arch.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tqllm {

void hf_map_minimax_m3(const HfFolder & h, GgufFile & f) {
    const std::string A = "minimax-m3.";
    auto key = [&](const std::string & k, double v) { f.kv[A + k].num = v; };
    const int L = int(h.num("num_hidden_layers"));
    const uint64_t E = uint64_t(h.num("num_local_experts")), vocab = uint64_t(h.num("vocab_size"));
    f.kv["general.name"].str = "MiniMax-M3.1-Flash";
    key("block_count", L);
    key("context_length", h.num("max_position_embeddings"));
    key("embedding_length", h.num("hidden_size"));
    key("feed_forward_length", h.num("dense_intermediate_size"));
    key("expert_feed_forward_length", h.num("intermediate_size"));
    key("expert_shared_feed_forward_length", h.num("shared_intermediate_size"));
    key("attention.head_count", h.num("num_attention_heads"));
    key("attention.head_count_kv", h.num("num_key_value_heads"));
    key("attention.key_length", h.num("head_dim"));
    key("attention.value_length", h.num("head_dim"));
    key("attention.layer_norm_rms_epsilon", h.num("rms_norm_eps"));
    key("rope.dimension_count", h.num("rotary_dim"));
    key("rope.freq_base", h.num("rope_theta"));
    key("expert_count", double(E));
    key("expert_used_count", h.num("num_experts_per_tok"));
    key("expert_shared_count", h.num("n_shared_experts"));
    key("expert_weights_scale", h.num("routed_scaling_factor"));
    key("leading_dense_block_count", 3);
    key("swiglu_alpha", h.num("swiglu_alpha"));
    key("swiglu_limit", h.num("swiglu_limit"));
    key("vocab_size", double(vocab));
    if (const Json * sa = h.section("sparse_attention_config")) {
        auto sk = [&](const char * k) { return sa->get(k) ? sa->get(k)->num : 0.0; };
        key("indexer.head_count", sk("sparse_num_index_heads"));
        key("indexer.key_length", sk("sparse_index_dim"));
        key("indexer.top_k_blocks", sk("sparse_topk_blocks"));
        key("indexer.block_size", sk("sparse_block_size"));
        key("indexer.local_blocks", sk("sparse_local_block"));
    }

    const std::string P = "language_model.model.";
    auto& T = f.tensors;
    T["token_embd.weight"] = h.plain(P + "embed_tokens.weight");
    T["output_norm.weight"] = h.norm(P + "norm.weight", true);
    T["output.weight"] = h.matrix("language_model.lm_head");
    for (int l = 0; l < L; ++l) {
        const std::string s = P + "layers." + std::to_string(l) + ".", d = "blk." + std::to_string(l) + ".";
        T[d + "attn_norm.weight"] = h.norm(s + "input_layernorm.weight", true);
        T[d + "ffn_norm.weight"] = h.norm(s + "post_attention_layernorm.weight", true);
        T[d + "attn_q.weight"] = h.matrix(s + "self_attn.q_proj");
        T[d + "attn_k.weight"] = h.matrix(s + "self_attn.k_proj");
        T[d + "attn_v.weight"] = h.matrix(s + "self_attn.v_proj");
        T[d + "attn_output.weight"] = h.matrix(s + "self_attn.o_proj");
        T[d + "attn_q_norm.weight"] = h.norm(s + "self_attn.q_norm.weight", true);
        T[d + "attn_k_norm.weight"] = h.norm(s + "self_attn.k_norm.weight", true);
        if (h.has(s + "self_attn.index_q_proj.weight")) {
            T[d + "indexer.attn_q.weight"] = h.matrix(s + "self_attn.index_q_proj");
            T[d + "indexer.attn_k.weight"] = h.matrix(s + "self_attn.index_k_proj");
            T[d + "indexer.q_norm.weight"] = h.norm(s + "self_attn.index_q_norm.weight", true);
            T[d + "indexer.k_norm.weight"] = h.norm(s + "self_attn.index_k_norm.weight", true);
        }
        const std::string m = s + "block_sparse_moe.";
        if (h.has(m + "gate.weight")) {
            T[d + "ffn_gate_inp.weight"] = h.plain(m + "gate.weight");
            T[d + "exp_probs_b.bias"] = h.plain(m + "e_score_correction_bias");
            T[d + "ffn_gate_shexp.weight"] = h.matrix(m + "shared_experts.gate_proj");
            T[d + "ffn_up_shexp.weight"] = h.matrix(m + "shared_experts.up_proj");
            T[d + "ffn_down_shexp.weight"] = h.matrix(m + "shared_experts.down_proj");
            for (auto [name, w] : {std::pair<const char *, const char *>{"ffn_gate_exps", "w1"}, {"ffn_up_exps", "w3"}, {"ffn_down_exps", "w2"}}) {
                std::vector<std::string> experts;
                for (uint64_t e = 0; e < E; ++e) experts.push_back(m + "experts." + std::to_string(e) + "." + w);
                T[d + name + ".weight"] = h.stack(experts);
            }
        } else {
            T[d + "ffn_gate.weight"] = h.matrix(s + "mlp.gate_proj");
            T[d + "ffn_up.weight"] = h.matrix(s + "mlp.up_proj");
            T[d + "ffn_down.weight"] = h.matrix(s + "mlp.down_proj");
        }
    }
    h.tokenizer(f, "minimax-m3", vocab);
}

}  // namespace tqllm
