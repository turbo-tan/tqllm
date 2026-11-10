#include "qwen35_engine.hpp"

namespace tqllm::qwen35_detail {
void Qwen35Engine::load_dspine(const GgufFile& f) {
    const auto& c = dfl.spine;
    const int H = c.hidden, C = c.code, R = c.rank, N = dfl.block * nslots;
    if (!has_head8()) throw std::invalid_argument("DSpine CUDA refinement requires a Q5_K or Q6_K target head");
    auto matrix = [&](const std::string& name, int rows, int cols) {
        return Mat{bf16_table(f, name, size_t(rows) * cols), GGML_BF16, rows, cols};
    };
    dfl.spine_codebook = bf16_table(f, "dspine_codebook.weight", size_t(cfg.vocab) * C);
    dfl.spine_bias = f32_of(f, "dspine_gate.bias", R);
    std::vector<bf16> packed;
    for (const auto* name : {"dspine_gate.weight", "dspine_pred_gate.weight", "dspine_prev.weight"}) {
        const auto& t = gt(f, name);
        const auto raw = f.read(t);
        std::vector<float> values(size_t(R) * C);
        dequantize_host(t.type, raw.data(), values.data(), values.size());
        for (float x : values) packed.push_back(__float2bfloat16_rn(x));
    }
    dfl.spine_packed = {put(packed.data(), packed.size() * 2), GGML_BF16, 3 * R, C};
    dfl.spine_pred = {static_cast<const bf16*>(dfl.spine_packed.p) + size_t(R) * C, GGML_BF16, 2 * R, C};
    for (int l = 0; l < c.layers; ++l) {
        auto& layer = dfl.L[size_t(l)];
        const auto p = "blk." + std::to_string(l) + ".";
        layer.dspine_read = matrix(p + "dspine_read.weight", C, H);
        layer.dspine_write = matrix(p + "dspine_write.weight", H, R);
    }
    sp_norm = Buf(size_t(N) * H * 2); sp_codes = Buf(size_t(N) * C * 2);
    sp_packed = Buf(size_t(N) * 3 * R * 2); sp_message = Buf(size_t(N) * R * 2);
    sp_delta = Buf(size_t(N) * H * 2); sp_raw = Buf(size_t(N) * H * 2); sp_rms = Buf(size_t(N) * 4);
    sp_predcodes = Buf(size_t(N) * 16 * C * 2); sp_predpv = Buf(size_t(N) * 16 * 2 * R * 2);
    sp_refmsg = Buf(size_t(N) * 16 * R * 2); sp_refdelta = Buf(size_t(N) * 16 * H * 2);
    sp_refined = Buf(size_t(N) * 16 * H * 2);
    sp_part = Buf(size_t(tqllm_cuda::attention_gen_partial_floats(dfl.heads, int(context_), dfl.block)) * 4);
    std::fprintf(stderr, "tqllm: DSpine bridge: code %d, rank %d, block %d, %d slots\n", C, R, c.block, nslots);
}

void Qwen35Engine::dspine_write(size_t layer, int rows) {
    const auto& c = dfl.spine;
    const auto& l = dfl.L[layer];
    auto* x = d_x.as<bf16>();
    if (layer + 1 == dfl.L.size())
        ck(cudaMemcpyAsync(sp_raw.p, x, size_t(rows) * c.hidden * 2, cudaMemcpyDeviceToDevice, stream));
    tqllm_cuda::dspine_read_norm(x, sp_norm.as<bf16>(), sp_rms.as<float>(), c.hidden, rows, c.epsilon, stream);
    mmx(l.dspine_read, sp_norm.as<bf16>(), sp_codes.as<bf16>(), rows);
    tqllm_cuda::dspine_codes(sp_codes.as<bf16>(), dfl.spine_codebook, d_tok.as<uint32_t>(),
                            c.code, dfl.block, rows, c.epsilon, stream);
    mmx(dfl.spine_packed, sp_codes.as<bf16>(), sp_packed.as<bf16>(), rows);
    tqllm_cuda::dspine_message(sp_packed.as<bf16>(), dfl.spine_bias, sp_message.as<bf16>(), c.rank, dfl.block, rows, stream);
    mmx(l.dspine_write, sp_message.as<bf16>(), sp_delta.as<bf16>(), rows);
    tqllm_cuda::dspine_add(x, sp_delta.as<bf16>(), sp_rms.as<float>(), c.hidden, dfl.block, rows, stream);
}

void Qwen35Engine::dspine_lattice(int rows) {
    const auto& c = dfl.spine;
    tqllm_cuda::dspine_predecessor_codes(d_idx.as<uint32_t>(), d_tok.as<uint32_t>(), dfl.spine_codebook,
                                        sp_predcodes.as<bf16>(), c.code, dfl.block, rows, stream);
    mmx(dfl.spine_pred, sp_predcodes.as<bf16>(), sp_predpv.as<bf16>(), rows * 16);
    tqllm_cuda::dspine_refine_message(sp_packed.as<bf16>(), sp_predpv.as<bf16>(), dfl.spine_bias,
                                     sp_refmsg.as<bf16>(), c.rank, rows, stream);
    mmx(dfl.L.back().dspine_write, sp_refmsg.as<bf16>(), sp_refdelta.as<bf16>(), rows * 16);
    tqllm_cuda::dspine_refine_norm(sp_raw.as<bf16>(), sp_refdelta.as<bf16>(), sp_rms.as<float>(), dfl.out_norm,
                                  sp_refined.as<bf16>(), c.hidden, rows, cfg.eps, stream);
    tqllm_cuda::dspine_score(w8(), s8(), sp_refined.as<bf16>(), d_idx.as<uint32_t>(), d_score.as<float>(),
                            c.hidden, rows, stream);
}
}
