// Inkling-2-Mini (Inkling2LoopForConditionalGeneration, thinkingmachines): the text model of the published folder as is.
// Experts stay in their stored formats: NVFP4 (modelopt, per-expert amax / 2688) as GGML_NVFP4G and MXFP8 as GGML_MXFP8,
// w13 split into gate (even rows) and up (odd rows). Everything else is BF16 / F32 as stored; the unembedding keeps its
// first unpadded_vocab_size rows. Reference: transformers PR #51050 (models/inkling2_loop).
// tqt_experts.gguf in the folder (optional): TQT / TQT2 expert stacks (types 306 / 307, tqllm.tqt.code / tqllm.tqt2.code) and Q8_0 copies of
// dense matrices (embedding, head, attention q / k / v / o) with the engine names below that replace the folder's tensors one by one
// (a partial file keeps the others as stored).
#include "hf_arch.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace tqllm {

void hf_map_inkling2_loop(const HfFolder & h, GgufFile & f) {
    const std::string A = "inkling2-loop.";
    auto key = [&](const std::string & k, double v) { f.kv[A + k].num = v; };
    const int L = int(h.num("num_hidden_layers"));
    const uint64_t vocab = uint64_t(h.num("vocab_size")), out_vocab = uint64_t(h.num("unpadded_vocab_size"));
    const int D = int(h.num("hidden_size")), I = int(h.num("intermediate_size"));
    const Json * loop = h.section("loop_config");
    if (!loop) throw std::runtime_error("inkling2-loop: no loop_config");
    auto lk = [&](const char * k) { const Json * v = loop->get(k); if (!v) throw std::runtime_error(std::string("inkling2-loop: loop_config.") + k); return v->num; };
    const int core0 = int(lk("core_start")), core1 = int(lk("core_end"));
    const double block = h.num("n_layers_per_attn_res_block");
    f.kv["general.name"].str = "Inkling-2-Mini";
    key("block_count", L);
    key("context_length", h.num("model_max_length"));
    key("embedding_length", D);
    key("expert_feed_forward_length", I);
    key("expert_count", h.num("n_routed_experts"));
    key("expert_used_count", h.num("num_experts_per_tok"));
    key("expert_weights_scale", h.num("route_scale"));
    key("expert_gate_softcap", h.num("mlp_gate_softcap"));
    key("expert_up_softcap", h.num("mlp_up_softcap"));
    key("attention.head_count", h.num("num_attention_heads"));
    key("attention.head_count_kv", h.num("num_key_value_heads"));
    key("attention.key_length", h.num("head_dim"));
    key("attention.sliding_window", h.num("sliding_window_size"));
    key("attention.layer_norm_rms_epsilon", h.num("rms_norm_eps"));
    key("attention.qk_clip", h.num("qk_l2_clip_threshold"));
    key("attention.relative_length", h.num("d_rel"));
    key("attention.relative_extent", h.num("rel_extent"));
    key("attention.relative_clip", h.num("rel_l2_clip_threshold"));
    key("attention.log_scale_floor", h.num("log_scaling_n_floor"));
    key("attention.log_scale_alpha", h.num("log_scaling_alpha"));
    key("shortconv.l_cache", h.num("sconv_kernel_size"));
    key("attn_res.block_size", block);
    key("final_logit_softcapping", h.num("final_logit_softcapping"));
    key("loop.core_start", core0);
    key("loop.core_end", core1);
    key("loop.num_passes", lk("num_passes"));
    key("loop.gate_rank", lk("gate_rank"));
    key("vocab_size", double(vocab));
    key("output_vocab_size", double(out_vocab));

    const std::string P = "model.llm.";
    auto & T = f.tensors;
    T["token_embd.weight"] = h.plain(P + "embed.weight");
    T["token_embd_norm.weight"] = h.norm(P + "embed_norm.weight", false);
    T["output_norm.weight"] = h.norm(P + "norm.weight", false);
    GgufTensor out = h.plain(P + "unembed.weight");  // [vocab][D] BF16: the padded rows are dropped
    if (out.ne.size() != 2 || out.ne[1] < out_vocab) throw std::runtime_error("inkling2-loop: unembed shape");
    out.ne[1] = out_vocab;
    out.bytes = out_vocab * out.ne[0] * 2;
    T["output.weight"] = out;
    T["output_res_q.weight"] = h.plain(P + "attn_res.output_query");
    T["loop_msg_q.weight"] = h.plain(P + "middle_cycle_cross_pass_queries");      // rows (q, ffn) per core layer
    T["loop_msg_kv_q.weight"] = h.plain(P + "middle_cycle_cross_pass_kv_queries"); // rows (k, v) per core layer

    std::string full;
    for (int l = 0; l < L; ++l) {
        const std::string s = P + "layers." + std::to_string(l) + ".", d = "blk." + std::to_string(l) + ".";
        const bool sliding = h.has(s + "attn_norm.weight");
        full += sliding ? "0" : "1";
        if (l > 0) T[d + "attn_res_q.weight"] = h.plain(P + "attn_res.attn_queries." + std::to_string(l - 1));
        T[d + "ffn_res_q.weight"] = h.plain(P + "attn_res.mlp_queries." + std::to_string(l));
        if (sliding) {
            T[d + "attn_norm.weight"] = h.norm(s + "attn_norm.weight", false);
            T[d + "attn_k_conv.weight"] = h.plain(s + "attn.k_sconv.weight");  // [1024][1][4], tap 3 = current token
            T[d + "attn_v_conv.weight"] = h.plain(s + "attn.v_sconv.weight");
        } else {
            T[d + "attn_q_in_norm.weight"] = h.norm(s + "q_input_norm.weight", false);
            T[d + "attn_k_in_norm.weight"] = h.norm(s + "k_input_norm.weight", false);
            T[d + "attn_v_in_norm.weight"] = h.norm(s + "v_input_norm.weight", false);
            T[d + "attn_k_conv_static.weight"] = h.plain(s + "attn.k_sconv.static_w");  // [4][1024], row 0 = current token
            T[d + "attn_v_conv_static.weight"] = h.plain(s + "attn.v_sconv.static_w");
            T[d + "attn_k_conv_proj.weight"] = h.plain(s + "attn.k_sconv.weight_proj");  // [kv heads x 4][D]
            T[d + "attn_v_conv_proj.weight"] = h.plain(s + "attn.v_sconv.weight_proj");
        }
        T[d + "attn_q.weight"] = h.matrix(s + "attn.wq_du");
        T[d + "attn_k.weight"] = h.matrix(s + "attn.wk_dv");
        T[d + "attn_v.weight"] = h.matrix(s + "attn.wv_dv");
        T[d + "attn_output.weight"] = h.matrix(s + "attn.wo_ud");
        T[d + "attn_gate.weight"] = h.matrix(s + "attn.gate_attn");
        T[d + "attn_rel.weight"] = h.matrix(s + "attn.wr_du");
        T[d + "attn_rel_proj.weight"] = h.plain(s + "attn.rel_logits_proj.proj");  // [d_rel][extent]
        T[d + "attn_rel_clip.weight"] = h.plain(s + "attn.rel_l2_clip.scale");     // [heads][d_rel]
        T[d + "attn_q_clip.weight"] = h.plain(s + "attn.q_l2_clip.scale");         // [heads][head_dim]
        T[d + "attn_sink.weight"] = h.matrix(s + "attn.attn_sink_proj");
        T[d + "attn_post_conv.weight"] = h.plain(s + "attn_sconv.weight");
        T[d + "ffn_post_conv.weight"] = h.plain(s + "mlp_sconv.weight");
        T[d + "ffn_norm.weight"] = h.norm(s + "mlp_norm.weight", false);
        T[d + "ffn_gate_inp.weight"] = h.plain(s + "mlp.gate.weight");
        T[d + "exp_probs_b.bias"] = h.plain(s + "mlp.gate.bias");
        const std::string w13 = s + "mlp.experts.w13_weight", w2 = s + "mlp.experts.w2_weight";
        const bool nv = h.has(w13 + "_amax");
        const uint64_t rows = uint64_t(I);
        T[d + "ffn_gate_exps.weight"] = nv ? h.nvfp4_experts(w13, 0, 2, rows) : h.mxfp8_experts(w13, 0, 2, rows);
        T[d + "ffn_up_exps.weight"] = nv ? h.nvfp4_experts(w13, 1, 2, rows) : h.mxfp8_experts(w13, 1, 2, rows);
        T[d + "ffn_down_exps.weight"] = nv ? h.nvfp4_experts(w2) : h.mxfp8_experts(w2);
        if (l >= core0 && l < core1) {
            for (auto [c, g] : {std::pair<const char *, const char *>{"attn", "q"}, {"k", "k"}, {"v", "v"}, {"mlp", "ffn"}}) {
                const std::string cs = s + "controllers." + c + ".", cd = d + "loop_ctrl_" + g + "_";
                T[cd + "down.weight"] = h.plain(cs + "down_weight");    // [rank][2 D]
                T[cd + "gain.weight"] = h.plain(cs + "gain_weight");    // [D][rank]
                T[cd + "forget.weight"] = h.plain(cs + "forget_weight");
            }
        }
    }
    const std::filesystem::path overlay = h.dir() / "tqt_experts.gguf";
    if (std::filesystem::exists(overlay)) {
        const GgufFile o = gguf_open(overlay.string());
        for (const auto & [name, t] : o.tensors) {
            auto it = T.find(name);
            const bool experts = name.find("_exps.weight") != std::string::npos;
            if (it == T.end()) throw std::runtime_error("inkling2-loop: tqt_experts.gguf tensor " + name + " is not a tensor of the model");
            if (experts ? (t.type != GGML_TQT && t.type != GGML_TQT2) : t.type != GGML_Q8_0)
                throw std::runtime_error("inkling2-loop: tqt_experts.gguf tensor " + name + " has an unexpected type " + std::to_string(t.type));
            if (t.ne != it->second.ne) throw std::runtime_error("inkling2-loop: tqt_experts.gguf tensor " + name + " has another shape");
            GgufTensor g = t;
            g.file = overlay.string();
            it->second = g;
        }
        for (const char * k : {"tqllm.tqt.code", "tqllm.tqt2.code"})
            if (o.kv.count(k)) f.kv[k] = o.kv.at(k);
    }
    f.kv[A + "attention.full_layers"].str = full;
    h.tokenizer(f, "inkling2", vocab);
}

}  // namespace tqllm
