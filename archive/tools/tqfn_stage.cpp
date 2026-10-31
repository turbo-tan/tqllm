// tqfn-stage: Flash-Next bring-up and measurement tool (stage replays against fixtures, chain KLD, offline decode,
// expert packing). The engine itself is in src/models/qwen/qwen4exp/fn_*.hpp; tqserve serves it through tqserve_flashnext.cpp.
#include "fn_model.hpp"
static void run_recurrent(int argc, char ** argv, bool cpu_q8k = false) {
    require(argc >= 3, "usage: tqfn-stage --recurrent fixture-directory shard1.gguf [more shards]");
    Fixtures fixtures(argv[1]); Checkpoint checkpoint(argc, argv); int layers = 0;
    checkpoint.cpu_q8_projection = cpu_q8k;
    for (int layer = 0; layer < checkpoint.config.layers; ++layer) {
        if (checkpoint.config.compress_ratios[layer]) continue;
        recurrent_forward(checkpoint, fixtures.values("hc_mixed-" + std::to_string(layer), 0), layer, &fixtures);
        ++layers;
    }
    std::cout << "NATIVE_RECURRENT_REPLAY_DONE layers=" << layers << " full_forward=false\n";
}
static void run_attention(int argc, char ** argv, bool cpu_q8 = false) {
    require(argc >= 3, "usage: tqfn-stage --attention fixture-directory shard1.gguf [more shards]");
    Fixtures fixtures(argv[1]); Checkpoint checkpoint(argc, argv); int layers = 0;
    checkpoint.cpu_q8_projection = cpu_q8;
    std::cout << "PROJECTION_ARITHMETIC " << (cpu_q8 ? "cpu-q8-rht-diagnostic" : "native-f32-rht") << '\n';
    for (int layer = 0; layer < checkpoint.config.layers; ++layer) {
        if (!checkpoint.config.compress_ratios[layer]) continue;
        attention_forward(checkpoint, fixtures.values("hc_mixed-" + std::to_string(layer), 0), layer, &fixtures);
        ++layers;
    }
    std::cout << "NATIVE_ATTENTION_REPLAY_DONE layers=" << layers << " QSA=false context_below_budget=true full_forward=false\n";
}
// Complete short-context chain. Fixtures supply only token IDs and final expected logits.
// Temporary transfers/allocations are diagnostic; this is not the production session or speed path.
// --chain-kld: tokens from a llama.cpp KL base instead of the fixture, no trace files, final residual handed back
static const std::vector<int> * g_chain_tokens = nullptr;
static bool g_chain_quiet = false;
static std::vector<float> g_chain_residual;
static bool g_chain_f16_act = false;
static void run_chain(int argc, char ** argv, bool cpu_q8, bool f32_experts = false, bool gsq_q8k = false) {
    require(argc >= 3, "usage: tqfn-stage --chain[-cpu-q8] fixture-dir shard1.gguf [more shards]");
    Fixtures fixtures(argv[1]); Checkpoint checkpoint(argc, argv); checkpoint.cpu_q8_projection = cpu_q8 || gsq_q8k;
    if (g_chain_f16_act) checkpoint.cpu_q8_projection = false;
    const auto & c = checkpoint.config; const int h = c.hidden, width = h * c.hc;
    require(checkpoint.trace_layer < c.layers, "trace layer exceeds model depth");
    const std::string mode = gsq_q8k ? std::string("gsq-q8k") + (checkpoint.f32_expert_output ? "-expert-f32-out" : "")
                                      : std::string(cpu_q8 ? "q8" : "f32") + (f32_experts ? "-expert-f32" : "");
    const auto tokens = g_chain_tokens ? *g_chain_tokens : fixtures.tokens(); const int nt = int(tokens.size());
    require(nt > 0 && nt <= 64, "chain diagnostic currently bounded to 64 tokens");
    auto trace = [&](const std::string & name, const std::vector<float> & data) {
        if (g_chain_quiet) return;
        std::ofstream file(std::filesystem::path(argv[1]) / ("trace-" + mode + "-" + name + ".f32"), std::ios::binary);
        require(bool(file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float))), "cannot write native trace");
    };
    std::vector<uint64_t> token_rows;
    for (int token : tokens) { require(token >= 0, "negative token ID"); token_rows.push_back(uint64_t(token)); }
    auto embedding = checkpoint.file("token_embd.weight").gather_rows("token_embd.weight", token_rows);
    trace("embedding", embedding);
    Device<float> de(embedding.size()), init(size_t(width) * nt); de.put(embedding);
    tqllm_cuda::hc_init(de.p, init.p, h, c.hc, nt, nullptr); auto residual = init.get();
    PleTokenHistory history(c.ple, tokens.size()); const auto ids = history.append_batch(tokens);
    std::vector<uint64_t> ple_rows(ids.begin(), ids.end());
    auto ple = checkpoint.file("per_layer_token_embd.weight").gather_rows("per_layer_token_embd.weight", ple_rows);
    const int slots = std::min(c.experts, nt * c.topk);
    std::unique_ptr<Tq3ExpertCache> tq_cache;
    std::unique_ptr<MixedExpertCache> mixed_cache;
    const bool mixed_formats = checkpoint.expert_payload(0, 0)[0].type != GGML_TQ3_4S;
    std::cout << "CHAIN_EXPERT_LAYOUT " << (mixed_formats ? "native-mixed-plain" : "TQ3-PRMT-rotated") << '\n';
    require(!gsq_q8k || mixed_formats, "Q8_K chain diagnostic requires native mixed formats");
    require(!checkpoint.f32_expert_output || (gsq_q8k && mixed_formats), "F32 expert output needs mixed GSQ chain");
    require(!mixed_formats || (!f32_experts && !cpu_q8), "mixed chain uses native grouped formats, not TQ3 precision modes");
    if (mixed_formats) mixed_cache = std::make_unique<MixedExpertCache>(c.layers, c.experts, h, c.expert_ff, slots,
        size_t(2) * 1024 * 1024 * 1024, [&](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
    else tq_cache = std::make_unique<Tq3ExpertCache>(c.layers, c.experts, h, c.expert_ff, slots,
        size_t(2) * 1024 * 1024 * 1024, [&](int layer, int expert) { return checkpoint.expert(layer, expert); }, nullptr);
    auto gate_matrix = [&] { return mixed_formats ? mixed_cache->gate() : tq_cache->gate(); };
    auto up_matrix = [&] { return mixed_formats ? mixed_cache->up() : tq_cache->up(); };
    auto down_matrix = [&] { return mixed_formats ? mixed_cache->down() : tq_cache->down(); };
    for (int layer = 0; layer < c.layers; ++layer) {
        const std::string suffix = "-" + std::to_string(layer), prefix = "blk." + std::to_string(layer) + ".";
        const bool selected_trace = layer == checkpoint.trace_layer;
        auto trace_selected = [&](const std::string & name, const std::vector<float> & data) {
            trace(layer ? name + suffix : name, data);
        };
        if (std::find(c.ple_layers.begin(), c.ple_layers.end(), layer) != c.ple_layers.end()) {
            require(ple.size() == size_t(h) * nt, "PLE gather width differs");
            Device<float> dp(ple.size()), key(residual.size()), value(size_t(h) * nt), query(residual.size());
            Device<float> kn(key.count), qn(query.count), gamma(width), gates(size_t(c.hc) * nt), gated(residual.size());
            dp.put(ple); query.put(residual);
            checkpoint.project(prefix + "ple_key.weight", dp.p, key.p, h, width, nt);
            checkpoint.project(prefix + "ple_value.weight", dp.p, value.p, h, h, nt);
            gamma.put(checkpoint.dense(prefix + "ple_norm_key.weight"));
            tqllm_cuda::hc_norm(key.p, gamma.p, kn.p, h, c.hc, nt, float(checkpoint.epsilon), nullptr);
            gamma.put(checkpoint.dense(prefix + "ple_norm_query.weight"));
            tqllm_cuda::hc_norm(query.p, gamma.p, qn.p, h, c.hc, nt, float(checkpoint.epsilon), nullptr);
            tqllm_cuda::ple_gate(kn.p, qn.p, gates.p, h, c.hc, nt, nullptr);
            tqllm_cuda::ple_broadcast(value.p, gates.p, gated.p, h, c.hc, nt, nullptr);
            gamma.put(checkpoint.dense(prefix + "ple_norm_conv.weight"));
            Device<float> norm(residual.size()), kernel{checkpoint.dense(prefix + "ple_conv1d.weight").size()};
            kernel.put(checkpoint.dense(prefix + "ple_conv1d.weight"));
            const int hist = (c.ple_kernel - 1) * c.ple.ngram;
            Device<float> past(size_t(hist) * width), next(past.count), conv(residual.size()), out(residual.size());
            past.put(std::vector<float>(past.count, 0.f));
            tqllm_cuda::hc_norm(gated.p, gamma.p, norm.p, h, c.hc, nt, float(checkpoint.epsilon), nullptr);
            tqllm_cuda::ple_conv(norm.p, past.p, kernel.p, conv.p, next.p, width, nt, c.ple_kernel, c.ple.ngram, nullptr);
            tqllm_cuda::ple_apply(query.p, gated.p, conv.p, out.p, int(out.count), nullptr); residual = out.get();
        }
        int n = int(residual.size() / width);
        Device<float> dr(residual.size()), norm(residual.size()), gamma(width); dr.put(residual);
        gamma.put(checkpoint.dense(prefix + "hc_attn_norm.weight"));
        tqllm_cuda::hc_norm(dr.p, gamma.p, norm.p, h, c.hc, n, float(checkpoint.epsilon), nullptr);
        auto attention_mix = replay_mix(fixtures, checkpoint, suffix, 0, norm, false, false);
        if (selected_trace) trace_selected("attention_mixed", attention_mix.mixed);
        auto block = c.compress_ratios[layer] ? attention_forward(checkpoint, attention_mix.mixed, layer)
                                             : recurrent_forward(checkpoint, attention_mix.mixed, layer);
        if (selected_trace) trace_selected("attention_out", block);
        if (layer + 1 == c.layers && !g_chain_quiet) {
            residual.erase(residual.begin(), residual.end() - width);
            block.erase(block.begin(), block.end() - h);
            attention_mix.inject.erase(attention_mix.inject.begin(), attention_mix.inject.end() - c.hc); n = 1;
        }
        Device<float> ar(residual.size()), ao(block.size()), ai(attention_mix.inject.size()), combined(residual.size());
        ar.put(residual); ao.put(block); ai.put(attention_mix.inject);
        tqllm_cuda::hc_combine(ar.p, ao.p, ai.p, combined.p, h, c.hc, n, nullptr); residual = combined.get();
        if (selected_trace) trace_selected("attention_combined", residual);
        Device<float> fr(residual.size()), fn(residual.size()); fr.put(residual);
        gamma.put(checkpoint.dense(prefix + "hc_ffn_norm.weight"));
        tqllm_cuda::hc_norm(fr.p, gamma.p, fn.p, h, c.hc, n, float(checkpoint.epsilon), nullptr);
        const auto mix = replay_mix(fixtures, checkpoint, suffix, 1, fn, false, false);
        if (selected_trace) trace_selected("ffn_mixed", mix.mixed);
        Device<float> plain(mix.mixed.size()), router{checkpoint.dense(prefix + "ffn_gate_inp.weight").size()}, shared_gate(h);
        plain.put(mix.mixed); router.put(checkpoint.dense(prefix + "ffn_gate_inp.weight"));
        shared_gate.put(checkpoint.dense(prefix + "ffn_gate_inp_shexp.weight"));
        Device<float> logits(size_t(n) * (c.experts + 1)), weights(size_t(n) * c.topk), shared_weights(n);
        Device<int> routed_ids(weights.count);
        Device<float> router_input(plain.count);
        ck(cudaMemcpy(router_input.p, plain.p, plain.count * sizeof(float), cudaMemcpyDeviceToDevice));
        if (bf16_act()) round_bf16_device(router_input.p, router_input.count);
        tqllm_cuda::moe_router_f32(router.p, shared_gate.p, router_input.p, logits.p, h, c.experts, n, nullptr);
        tqllm_cuda::moe_topk(logits.p, routed_ids.p, weights.p, shared_weights.p, c.experts, c.topk, n, 1.f, nullptr);
        const auto actual_ids = routed_ids.get();
        const auto slot_ids = mixed_formats ? mixed_cache->prepare(layer, actual_ids) : tq_cache->prepare(layer, actual_ids); routed_ids.put(slot_ids);
        auto mixed_f32_projection = [&](Mat matrix, const __half * x, int rows, bool per_expert) {
            Device<float> output(size_t(n) * c.topk * rows);
            const size_t data_stride = size_t(rows) * matrix.fmt->row_bytes(matrix.cols);
            const size_t aux_stride = size_t(rows) * matrix.fmt->aux_row_bytes(matrix.cols);
            for (int assignment = 0; assignment < n * c.topk; ++assignment) {
                Mat view = matrix;
                view.p = static_cast<const char *>(matrix.p) + size_t(slot_ids[assignment]) * data_stride;
                if (matrix.aux) view.aux = static_cast<const char *>(matrix.aux) + size_t(slot_ids[assignment]) * aux_stride;
                view.rows = rows;
                const int token = per_expert ? assignment : assignment / c.topk;
                view.fmt->matvec_f32(view, x + size_t(token) * matrix.cols,
                                     output.p + size_t(assignment) * rows, 1, nullptr);
            }
            return output.get();
        };
        auto f32_projection = [&](const Mat & matrix, const std::vector<float> & values, int rows, bool per_expert) {
            const int columns = per_expert ? n * c.topk : n;
            Device<float> input(values.size()), rotated(values.size()), output(size_t(n) * c.topk * rows); input.put(values);
            tqllm_cuda::tq3_rotate_f32(input.p, rotated.p, matrix.cols, columns, nullptr);
            if (cpu_q8) { auto rounded = rotated.get(); cpu_q8_round(rounded); rotated.put(rounded); }
            const size_t stride = size_t(rows) * matrix.cols / 2;
            for (int assignment = 0; assignment < n * c.topk; ++assignment)
                tqllm_cuda::tq3_matvec_f32_repacked(static_cast<const char *>(matrix.p) + size_t(slot_ids[assignment]) * stride,
                    rotated.p + size_t(per_expert ? assignment : assignment / c.topk) * matrix.cols,
                    output.p + size_t(assignment) * rows, rows, matrix.cols, 1, nullptr);
            return output.get();
        };
        std::vector<float> gate, up, routed;
        if (f32_experts) {
            gate = f32_projection(gate_matrix(), mix.mixed, c.expert_ff, false);
            up = f32_projection(up_matrix(), mix.mixed, c.expert_ff, false);
        } else {
            Device<__nv_bfloat16> xb(plain.count), eg(size_t(n) * c.topk * c.expert_ff), eu(eg.count);
            Device<__half> xr(plain.count);
            if (gsq_q8k) {
                if (exact_f32()) f32_to_half_host(plain.p, xr.p, plain.count);
                else if (round32_gpu_side("ffn_gate_exps")) tqllm_cuda::gsq_q8_round_h(plain.p, xr.p, plain.count, nullptr);
                else tqllm_cuda::gsq_q8k_round_h(plain.p, xr.p, plain.count, nullptr);
                if (checkpoint.validate_gsq_round) validate_round(xr, mix.mixed, true, prefix + "expert_input");
            } else {
                tqllm_cuda::f32_to_bf16(plain.p, xb.p, plain.count, nullptr);
                if (mixed_formats) tqllm_cuda::bf16_to_h(xb.p, xr.p, int(xb.count), nullptr);
                else tqllm_cuda::tq3_rotate_h(xb.p, xr.p, h, n, nullptr);
            }
            if (checkpoint.f32_expert_output) {
                gate = mixed_f32_projection(gate_matrix(), xr.p, c.expert_ff, false);
                up = mixed_f32_projection(up_matrix(), xr.p, c.expert_ff, false);
            } else {
                auto matrix = gate_matrix(); matrix.fmt->expert_matvec(matrix, xr.p, eg.p, c.expert_ff, routed_ids.p, n, c.topk, false, nullptr);
                matrix = up_matrix(); matrix.fmt->expert_matvec(matrix, xr.p, eu.p, c.expert_ff, routed_ids.p, n, c.topk, false, nullptr);
                gate = bf16_values(eg); up = bf16_values(eu);
            }
        }
        std::vector<float> activation(gate.size());
        for (size_t j = 0; j < activation.size(); ++j) activation[j] = gate[j] / (1.f + std::exp(-gate[j])) * up[j];
        if (selected_trace) { trace_selected("expert_gate", gate); trace_selected("expert_up", up); trace_selected("expert_activation", activation); }
        if (f32_experts) routed = f32_projection(down_matrix(), activation, h, true);
        else {
            Device<float> af(activation.size()); af.put(activation); Device<__nv_bfloat16> ab(activation.size());
            Device<__half> ax(activation.size()); tqllm_cuda::f32_to_bf16(af.p, ab.p, af.count, nullptr);
            if (gsq_q8k) {
                if (exact_f32()) f32_to_half_host(af.p, ax.p, af.count);
                else tqllm_cuda::gsq_q8_round_h(af.p, ax.p, af.count, nullptr);
                if (checkpoint.validate_gsq_round && !exact_f32()) validate_round(ax, activation, false, prefix + "expert_down_input");
            } else if (mixed_formats) tqllm_cuda::bf16_to_h(ab.p, ax.p, int(ab.count), nullptr);
            else tqllm_cuda::tq3_rotate_h(ab.p, ax.p, c.expert_ff, n * c.topk, nullptr);
            if (checkpoint.f32_expert_output) routed = mixed_f32_projection(down_matrix(), ax.p, h, true);
            else {
                Device<__nv_bfloat16> ed(size_t(n) * c.topk * h); auto matrix = down_matrix();
                matrix.fmt->expert_matvec(matrix, ax.p, ed.p, h, routed_ids.p, n, c.topk, true, nullptr);
                routed = bf16_values(ed);
            }
        }
        Device<float> sg(size_t(n) * c.shared_ff), su(sg.count), sa(sg.count), sd(size_t(n) * h);
        checkpoint.project(prefix + "ffn_gate_shexp.weight", plain.p, sg.p, h, c.shared_ff, n);
        checkpoint.project(prefix + "ffn_up_shexp.weight", plain.p, su.p, h, c.shared_ff, n);
        auto shared_activation = sg.get(), shared_up = su.get();
        for (size_t j = 0; j < shared_activation.size(); ++j)
            shared_activation[j] = shared_activation[j] / (1.f + std::exp(-shared_activation[j])) * shared_up[j];
        sa.put(shared_activation); checkpoint.project(prefix + "ffn_down_shexp.weight", sa.p, sd.p, c.shared_ff, h, n);
        const auto shared = sd.get(), rw = weights.get(), sw = shared_weights.get();
        if (selected_trace) { trace_selected("expert_down", routed); trace_selected("shared", shared); trace_selected("routing_weights", rw); trace_selected("shared_weights", sw); }
        std::vector<float> result(size_t(n) * h);
        for (int t = 0; t < n; ++t) for (int j = 0; j < h; ++j) {
            float value = 0.f;
            for (int k = 0; k < c.topk; ++k) value += rw[t * c.topk + k] * routed[(size_t(t) * c.topk + k) * h + j];
            result[size_t(t) * h + j] = value + sw[t] * shared[size_t(t) * h + j];
        }
        if (selected_trace) trace_selected("ffn_output", result);
        Device<float> fo(result.size()), fi(mix.inject.size()), updated(residual.size()); fo.put(result); fi.put(mix.inject);
        tqllm_cuda::hc_combine(fr.p, fo.p, fi.p, updated.p, h, c.hc, n, nullptr); residual = updated.get();
        require(std::all_of(residual.begin(), residual.end(), [](float x) { return std::isfinite(x); }), "nonfinite chained residual");
        if (!g_chain_quiet) {
            std::ofstream state_file(std::filesystem::path(argv[1]) / ("native-chain-" + mode + "-layer-" + std::to_string(layer) + ".f32"), std::ios::binary);
            require(bool(state_file.write(reinterpret_cast<const char *>(residual.data()), residual.size() * sizeof(float))), "cannot save chained layer trace");
            std::ofstream route_file(std::filesystem::path(argv[1]) / ("native-chain-" + mode + "-routes-" + std::to_string(layer) + ".i32"), std::ios::binary);
            require(bool(route_file.write(reinterpret_cast<const char *>(actual_ids.data()), actual_ids.size() * sizeof(int))), "cannot save native routing trace");
            std::cout << "NATIVE_CHAIN_LAYER " << layer << " tokens=" << n << std::endl;
        }
    }
    if (g_chain_quiet) { g_chain_residual = residual; return; }
    Device<float> last(residual.size()), norm(residual.size()), gamma(width); last.put(residual);
    gamma.put(checkpoint.dense("output_hc_norm.weight"));
    tqllm_cuda::hc_norm(last.p, gamma.p, norm.p, h, c.hc, 1, float(checkpoint.epsilon), nullptr);
    const auto head = replay_mix(fixtures, checkpoint, "", 0, norm, true, false);
    trace("head", head.mixed);
    const auto logits = checkpoint.head_logits(head.mixed), expected = fixtures.values("result_output");
    require(logits.size() == expected.size(), "expected last-token logit dimensions differ");
    std::ofstream raw(std::filesystem::path(argv[1]) / ("native-chain-" + mode + "-logits.f32"), std::ios::binary);
    require(bool(raw.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float))), "cannot save chained logits");
    double zp = 0., zq = 0., mean = 0.;
    const float mp = *std::max_element(expected.begin(), expected.end()), mq = *std::max_element(logits.begin(), logits.end());
    for (size_t i = 0; i < logits.size(); ++i) { zp += std::exp(double(expected[i]) - mp); zq += std::exp(double(logits[i]) - mq); }
    for (size_t i = 0; i < logits.size(); ++i) {
        require(std::isfinite(logits[i]) && std::isfinite(expected[i]), "nonfinite chained logits");
        const double probability = std::exp(double(expected[i]) - mp) / zp;
        mean += probability * (double(expected[i]) - mp - double(logits[i]) + mq);
    }
    const double kld = mean + std::log(zq / zp);
    const auto actual_top = std::max_element(logits.begin(), logits.end()) - logits.begin();
    const auto reference_top = std::max_element(expected.begin(), expected.end()) - expected.begin();
    std::cout << "NATIVE_CHAIN_DONE layers=" << c.layers << " input_tokens=" << nt << " fixture_hidden_inputs=0"
              << " cpu_q8_diagnostic=" << cpu_q8 << " gsq_q8k_diagnostic=" << gsq_q8k
              << " f32_q6_diagnostic=" << checkpoint.f32_q6_projection
              << " f32_expert_output_diagnostic=" << checkpoint.f32_expert_output
              << " last_token_kld=" << kld << " top1=" << actual_top
              << " reference_top1=" << reference_top << " expert_intermediates="
              << (gsq_q8k ? "f32-q8k/q8-h" : f32_experts ? "f32" : "bf16")
              << " production_session=false" << std::endl;
    require(kld <= .01, "chained last-token KLD exceeds .01 screen");
}
// Many-token quality gate: teacher-forced native chain over the chunks of a llama.cpp --kl-divergence-base file (chunks of
// <= 64 tokens), positions n_ctx/2 .. n_ctx-2 scored with llama.cpp's KLD (same math as tqkld --llama-base).
struct KldSums {
    double sum_nll = 0, sum_nll2 = 0, sum_nll_base = 0, sum_nll_base2 = 0, sum_nll_nll_base = 0;
    double sum_kld = 0, sum_kld2 = 0, sum_p_diff = 0, sum_p_diff2 = 0, sum_p_diff4 = 0;
    float max_p_diff = 0;
    size_t n_same_top = 0, count = 0;
};
std::pair<double, float> kld_token(int n_vocab, const float * logits, const uint16_t * base_log_prob, int tok, KldSums & kld) {
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n_vocab; ++i)
        if (logits[i] > max_logit) { max_logit = logits[i]; imax = i; }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_logit);
    const float log_sum_exp = log(sum_exp);
    const float * d = reinterpret_cast<const float *>(base_log_prob);
    const float scale = d[0], min_log_prob = d[1];
    base_log_prob += 4;
    const float nll = max_logit + log_sum_exp - logits[tok];
    kld.sum_nll += nll;
    kld.sum_nll2 += nll * nll;
    const float nll_base = -(scale * base_log_prob[tok] + min_log_prob);
    kld.sum_nll_base += nll_base;
    kld.sum_nll_base2 += nll_base * nll_base;
    kld.sum_nll_nll_base += nll * nll_base;
    max_logit += log_sum_exp;
    double sum = 0;
    int imax_base = -1;
    float p_log_base_max = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float p_log_base = scale * base_log_prob[i] + min_log_prob;
        if (i == 0 || p_log_base > p_log_base_max) { p_log_base_max = p_log_base; imax_base = i; }
        if (p_log_base > -16.f) {
            const float p_base = expf(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    kld.sum_kld += sum;
    kld.sum_kld2 += sum * sum;
    ++kld.count;
    if (imax == imax_base) ++kld.n_same_top;
    const float p_base = expf(-nll_base), p = expf(-nll), p_diff = p - p_base;
    kld.sum_p_diff += p_diff;
    const double p_diff2 = p_diff * p_diff;
    kld.sum_p_diff2 += p_diff2;
    kld.sum_p_diff4 += p_diff2 * p_diff2;
    kld.max_p_diff = std::max(kld.max_p_diff, std::fabs(p_diff));
    return {sum, p_diff};
}


// Greedy decode prototype: the validated stage operations with carried state (GDN conv + recurrent state, BF16 KV,
// PLE conv history), resident weights and an LRU expert cache. Plain precision: FP16 activations into native formats.
static void run_pack_experts(int argc, char ** argv) {
    require(argc >= 3, "usage: tqfn-stage --pack-experts OUT.bin shard1.gguf [more] [--gsq-rco]");
    const std::string out = argv[1];
    std::vector<char *> args{argv[0], argv[0]}; for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
    Checkpoint checkpoint(int(args.size()), args.data()); const auto & c = checkpoint.config;
    g_tile_experts = true;
    tqllm::write_packed_experts(out, c.layers, c.experts, g_tile_experts,
                                [&](int l, int e) { return repack_expert(checkpoint, l, e); });
    std::printf("PACK_DONE %s\n", out.c_str());
}

static void run_decode(int argc, char ** argv) {
    require(argc >= 5, "usage: tqfn-stage --decode fixture-dir prompt-ids.txt N shard1.gguf [more] [--gsq-rco] [--slots S] [--budget-gb G] [--chunk C] [--check]");
    const std::string ids_file = argv[2]; const int max_new = std::stoi(argv[3]);
    int slots = 160, chunk = 8; double budget_gb = 14; bool check = false, spec_on = false; int drafts = 2; std::string packed_path;
    std::vector<char *> args{argv[0], argv[1]};
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--slots" && i + 1 < argc) { slots = std::stoi(argv[++i]); continue; }
        if (a == "--budget-gb" && i + 1 < argc) { budget_gb = std::stod(argv[++i]); continue; }
        if (a == "--chunk" && i + 1 < argc) { chunk = std::stoi(argv[++i]); continue; }
        if (a == "--check") { check = true; continue; }
        if (a == "--experts" && i + 1 < argc) { packed_path = argv[++i]; continue; }
        if (a == "--spec") { spec_on = true; continue; }
        if (a == "--drafts" && i + 1 < argc) { drafts = std::stoi(argv[++i]); continue; }
        args.push_back(argv[i]);
    }
    { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); uint64_t keep = UINT64_MAX;
      ck(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep)); }
    // TQFN_ARENA_MB: scratch arena size (default 512 MB, or what a layer-major chunk needs); the high-water mark is printed at exit
    static ScratchArena arena;
    arena.size = std::getenv("TQFN_ARENA_MB") ? size_t(std::atoi(std::getenv("TQFN_ARENA_MB"))) << 20
                                              : std::max(size_t(512) << 20, chunk >= 256 ? tqllm::prefill_arena_bytes(chunk) : 0);
    static struct ArenaReport { ~ArenaReport() { if (g_arena) std::printf("ARENA high-water %.1f MB of %.1f MB\n", g_arena->high / 1048576.0, g_arena->size / 1048576.0); } } arena_report; ck(cudaMalloc(&arena.base, arena.size)); g_arena = &arena;
    Fixtures fixtures(args[1]); Checkpoint checkpoint(int(args.size()), args.data()); checkpoint.resident = true;
    const auto & c = checkpoint.config;
    // TQFN_CPU_BENCH=threads: time the CPU miss pool alone on real expert bytes (no GPU work), K experts x n tokens per call
    // TQFN_PACK_CHECK=packed.tqfnexp: is each packed expert part byte-identical to its GGUF slice? (CPU pool could read it)
    if (const char * pc = std::getenv("TQFN_PACK_CHECK")) {
        const auto pk = open_packed(pc); std::map<std::string, int> tally;
        for (int l = 0; l < c.layers; ++l) {
            const auto pl = checkpoint.expert_payload(l, 7); const char * blk = pk.at(l, 7); size_t off = 0;
            for (int m = 0; m < 3; ++m) {
                const size_t d = pk.sizes[l][2 * m], a = pk.sizes[l][2 * m + 1];
                const bool same = a == 0 && d == pl[m].bytes.size && std::memcmp(blk + off, pl[m].bytes.data, d) == 0;
                ++tally["type " + std::to_string(pl[m].type) + " u4h " + std::to_string(expert_u4h_ids(checkpoint, l)[m]) + " tbpb " +
                        std::to_string(tqllm_cuda::tiled_block_bytes(expert_u4h_ids(checkpoint, l)[m])) + (m == 2 ? " down" : " gate/up") + (same ? " SAME" : " DIFFERS") +
                        " (gguf " + std::to_string(pl[m].bytes.size) + ", data " + std::to_string(d) + ", aux " + std::to_string(a) + ")"];
                off += d + a;
            }
        }
        for (auto & [k, v] : tally) std::printf("PACK_CHECK %s x%d\n", k.c_str(), v);
        std::fflush(stdout); std::_Exit(0);
    }
    if (const char * cb = std::getenv("TQFN_CPU_BENCH")) {
        CpuMissPool pool(std::atoi(cb), c.hidden, c.expert_ff);
        std::vector<float> x(size_t(4) * c.hidden); for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.37f * float(i)) * 0.5f;
        std::vector<__nv_bfloat16> out(size_t(4) * c.topk * c.hidden);
        for (int k : {1, 2, 3, 4}) for (int n : {1, 4}) {
            double total = 0; int calls = 0;
            for (int rep = 0; rep < 3; ++rep) for (int layer = 0; layer < c.layers; ++layer) {
                std::vector<CpuMissPool::Job> jobs;
                for (int e = 0; e < k; ++e) {
                    CpuMissPool::Job j; const auto pl = checkpoint.expert_payload(layer, (37 * e + 11 * layer) % c.experts);
                    const int rows[3] = {c.expert_ff, c.expert_ff, c.hidden};
                    for (int m = 0; m < 3; ++m) { j.w[m] = pl[m].bytes.data; j.type[m] = pl[m].type; j.row_bytes[m] = pl[m].bytes.size / size_t(rows[m]); }
                    j.ncols = n == 1 ? 1 : 2; for (int q = 0; q < j.ncols; ++q) j.cols_t[q] = q;  // a verify window: ~2 tokens per missed expert
                    jobs.push_back(j);
                }
                const auto t0 = std::chrono::steady_clock::now(); pool.start(x.data(), n, std::move(jobs), out.data()); pool.wait();
                if (rep) { total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); ++calls; }
            }
            std::printf("CPU_BENCH threads %s experts %d tokens %d: %.3f ms per call\n", cb, k, n, total / calls);
        }
        std::fflush(stdout); std::_Exit(0);
    }
    // A ".list" file names several token-id files: run them in order on one cache (like a server), report each + mean.
    std::vector<std::pair<std::string, std::vector<int>>> prompts;
    auto read_ids = [](const std::string & f) { std::vector<int> v; std::ifstream in(f); for (int id; in >> id;) v.push_back(id); return v; };
    if (ids_file.size() > 5 && ids_file.substr(ids_file.size() - 5) == ".list") {
        std::ifstream in(ids_file); for (std::string line; std::getline(in, line);) if (!line.empty()) prompts.push_back({line, read_ids(line)});
    } else prompts.push_back({ids_file, read_ids(ids_file)});
    // prompts past the indexer budget use QSA, as in the server (the KV holds every position)
    for (const auto & [name, ids] : prompts) {
        require(!ids.empty(), "prompt empty: " + name);
        g_kv_capacity = std::max(g_kv_capacity, int(ids.size()) + max_new + 16);
    }
    const auto prompt = prompts[0].second;
    MixedExpertCache cache(c.layers, c.experts, c.hidden, c.expert_ff, slots, size_t(budget_gb * 1024 * 1024 * 1024),
                           [&](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
    apply_layer_slots(cache, packed_path, slots);
    PackedExperts packed;
    if (!packed_path.empty()) {
        packed = open_packed(packed_path);
        require(packed.layers == c.layers && packed.experts == c.experts, "packed experts do not match the model");
        const bool pin = std::getenv("TQFN_NO_PIN") == nullptr;
        if (pin) {
            // File-backed mappings cannot be registered: copy the file into pinned anonymous memory once, then drop the
            // page-cache copy so RAM does not hold it twice.
            const auto t0 = std::chrono::steady_clock::now();
            char * host = nullptr; ck(cudaMallocHost(reinterpret_cast<void **>(&host), packed.bytes));
            const int fd = ::open(packed_path.c_str(), O_RDONLY); require(fd >= 0, "cannot reopen " + packed_path);
            for (size_t done = 0; done < packed.bytes;) {
                const ssize_t r = ::pread(fd, host + done, std::min<size_t>(size_t(1) << 28, packed.bytes - done), off_t(done));
                require(r > 0, "short read of " + packed_path); done += size_t(r);
            }
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd);
            ::munmap(const_cast<char *>(packed.map), packed.bytes); packed.map = host;
            std::printf("DECODE packed experts in pinned memory (%.1f GB) in %.1f s\n", packed.bytes / 1e9,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        // TQFN_TILED with a plain TQFNEXP1 file: tile the code parts in place in host memory (no second 50 GB file)
        if (!packed.tiled) {
            require(pin, "TQFN_TILED on a TQFNEXP1 file needs the pinned copy (unset TQFN_NO_PIN)");
            const auto t0 = std::chrono::steady_clock::now(); char * base = const_cast<char *>(packed.map);
            for (int l = 0; l < packed.layers; ++l) {
                size_t off = 0; const auto ids = expert_u4h_ids(checkpoint, l);
                for (int pp = 0; pp < 3; ++pp) {
                    const int cols = pp == 2 ? c.expert_ff : c.hidden, rows = pp == 2 ? c.hidden : c.expert_ff;
                    const int bpb = tqllm_cuda::tiled_block_bytes(ids[pp]);  // the device format
                    const size_t dsz = packed.sizes[l][2 * pp];
                    if (bpb) require(rows % 16 == 0 && (cols / 32) % 4 == 0 && dsz == size_t(rows) * (cols / 32) * bpb,
                                     "tileable expert part with an unexpected layout");
                    if (bpb) {
                        static int tiled_parts = 0; ++tiled_parts;
                        static struct Rep { ~Rep() { std::printf("DECODE tiled expert parts %d\n", tiled_parts); } } rep;
                        const int nt = 16; std::vector<std::thread> th;
                        for (int w = 0; w < nt; ++w) th.emplace_back([&, w] {
                            std::vector<char> tmp(dsz);
                            for (int e = w; e < packed.experts; e += nt) {
                                char * d = base + packed.base[l] + uint64_t(e) * packed.block[l] + off;
                                tqllm_cuda::tile_expert_rows(d, tmp.data(), rows, cols, bpb); std::memcpy(d, tmp.data(), dsz);
                            }
                        });
                        for (auto & t : th) t.join();
                    }
                    off += dsz + packed.sizes[l][2 * pp + 1];
                }
            }
            packed.tiled = true;
            std::printf("DECODE expert code parts tiled in host memory in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        if (packed.tiled) { tqllm_cuda::set_tiled_experts(true); g_tile_experts = true; std::printf("DECODE tiled expert layout\n"); }
        if (std::getenv("TQFN_CPU_MISS") && pin && std::getenv("TQFN_CPU_GGUF") == nullptr) {  // CPU pool reads the pinned copy
            static CpuPackedView pv; pv.map = packed.map; pv.base = packed.base; pv.block = packed.block; pv.sizes = packed.sizes;
            pv.tiled = packed.tiled; pv.layers = packed.layers;
            for (int l = 0; l < packed.layers; ++l) pv.fmt.push_back(expert_u4h_ids(checkpoint, l));
            g_cpu_packed = &pv; std::printf("DECODE CPU pool reads the pinned packed experts\n");
        }
        cache.set_prepacked([&](int layer, int id) -> const char * { return layer < packed.layers ? packed.at(layer, id) : nullptr; }, pin);
        std::printf("DECODE experts from %s (%.1f GB, no repack on a miss)\n", packed_path.c_str(), packed.bytes / 1e9);
    }
    std::unique_ptr<MixedExpertCache> mtp_cache; char * mtp_packed = nullptr;
    if (checkpoint.has_mtp) {
        const int mtp_slots = std::getenv("TQFN_MTP_SLOTS") ? std::atoi(std::getenv("TQFN_MTP_SLOTS")) : 256;
        mtp_cache = std::make_unique<MixedExpertCache>(c.layers + 1, c.experts, c.hidden, c.expert_ff, mtp_slots, size_t(mtp_slots) * (6u << 20) + (64u << 20),
            [&](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
        std::printf("DECODE MTP block loaded (blk.%d)%s\n", c.layers, spec_on ? ", speculative verify of 2 drafts" : "");
        if (!packed_path.empty() && std::getenv("TQFN_NO_PIN") == nullptr) {
            // MTP experts in the device layout in pinned memory: a miss is a kernel copy, not a CPU repack + upload.
            const auto t0 = std::chrono::steady_clock::now();
            const auto first = repack_expert(checkpoint, c.layers, 0); size_t block = 0; for (const auto & p : first) block += p.size();
            char * host = nullptr; ck(cudaMallocHost(reinterpret_cast<void **>(&host), block * size_t(c.experts)));
            std::vector<std::thread> pool; std::atomic<int> next{0};
            for (int t = 0; t < 16; ++t) pool.emplace_back([&] {
                for (int e; (e = next++) < c.experts;) {
                    const auto parts = repack_expert(checkpoint, c.layers, e); size_t off = 0;
                    for (const auto & p : parts) { std::memcpy(host + size_t(e) * block + off, p.data(), p.size()); off += p.size(); }
                    require(off == block, "MTP expert layout varies");
                }
            });
            for (auto & t : pool) t.join();
            mtp_packed = host;
            mtp_cache->set_prepacked([&, block](int layer, int id) -> const char * { return layer == c.layers ? mtp_packed + size_t(id) * block : nullptr; }, true);
            std::printf("DECODE MTP experts prepacked in pinned memory (%.2f GB) in %.1f s\n", block * double(c.experts) / 1e9,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    uint64_t spec_steps = 0, spec_tokens = 0;
    auto run = [&](const std::vector<int> & prompt_ids, int generate, std::vector<int> * all_argmax) {
        DecodeModel model(checkpoint, fixtures, cache, prompt_ids.size() + generate + 16);
        model.mtp_experts = mtp_cache.get();
        if (mtp_cache && model.miss_stream) mtp_cache->set_miss_stream(model.miss_stream);  // MTP experts on the split grouped path
        if (const char * pf = std::getenv("TQFN_PF_LOGITS")) {
            // Prefill numerics probe: the whole prompt through the layer-major prefill (chunks of `chunk`), then for every
            // position the top-32 next-token log-probs as (int32 id, float logp) x 32; tools/pf_logits_kl.py compares two dumps
            constexpr int K = 32; const int hdim = checkpoint.config.hidden;
            std::FILE * f = std::fopen(pf, "wb"); require(f != nullptr, std::string("cannot open ") + pf); size_t positions = 0;
            for (size_t b = 0; b < prompt_ids.size(); b += size_t(chunk)) {
                const size_t e = std::min(prompt_ids.size(), b + size_t(chunk));
                auto rows = model.step(std::vector<int>(prompt_ids.begin() + b, prompt_ids.begin() + e));
                for (size_t o = 0; o < e - b; o += 256) {
                    const int n = int(std::min<size_t>(256, e - b - o));
                    const auto lg = checkpoint.head_logits_device(rows->p + o * size_t(hdim), n);
                    const size_t V = lg.size() / size_t(n); std::vector<int> idx(V);
                    for (int r = 0; r < n; ++r) {
                        const float * row = lg.data() + size_t(r) * V;
                        float mx = -INFINITY; for (size_t v = 0; v < V; ++v) mx = std::max(mx, row[v]);
                        double lse = 0; for (size_t v = 0; v < V; ++v) lse += std::exp(double(row[v] - mx));
                        const float lz = mx + float(std::log(lse));
                        std::iota(idx.begin(), idx.end(), 0);
                        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [row](int a, int b2) { return row[a] > row[b2]; });
                        for (int k = 0; k < K; ++k) { const int32_t id = idx[size_t(k)]; const float lp = row[id] - lz;
                            require(std::fwrite(&id, 4, 1, f) == 1 && std::fwrite(&lp, 4, 1, f) == 1, "short write of prefill logits"); }
                    }
                }
                positions += e - b;
            }
            std::fclose(f); std::printf("PF_LOGITS %s positions %zu K %d\n", pf, positions, K);
            return std::vector<int>{};
        }
        if (const char * tf = std::getenv("TQFN_TF_LOGITS")) {
            // Teacher-forced decode probe: prefill 64 tokens, then feed the prompt's remaining tokens in verify-width blocks
            // of 3 through the decode kernels; writes float32 logits [blocks * 3][vocab] to compare numerics changes.
            const size_t P0 = 64; require(prompt_ids.size() >= P0 + 3, "TQFN_TF_LOGITS needs a prompt of 67+ tokens");
            for (size_t b = 0; b < P0; b += size_t(chunk)) { auto r = model.step(std::vector<int>(prompt_ids.begin() + b, prompt_ids.begin() + std::min(P0, b + size_t(chunk)))); (void) r; }
            std::FILE * f = std::fopen(tf, "wb"); require(f != nullptr, std::string("cannot open ") + tf); int blocks = 0;
            for (size_t b = P0; b + 3 <= prompt_ids.size(); b += 3, ++blocks) {
                auto rows = model.step(std::vector<int>(prompt_ids.begin() + b, prompt_ids.begin() + b + 3));
                const auto lg = checkpoint.head_logits_device(rows->p, 3);
                require(std::fwrite(lg.data(), sizeof(float), lg.size(), f) == lg.size(), "short write of teacher-forced logits");
            }
            std::fclose(f); std::printf("TF_LOGITS %s blocks %d vocab %zu\n", tf, blocks, size_t(checkpoint.head_resident().m.rows));
            return std::vector<int>{};
        }
        const bool use_spec = spec_on && mtp_cache && !all_argmax && generate > 1;
        std::unique_ptr<Device<float>> prompt_res;
        if (use_spec) { PersistentAlloc keep; prompt_res = std::make_unique<Device<float>>(prompt_ids.size() * size_t(model.width)); }
        std::vector<int> out;
        // TQFN_PROFILE_PROMPT=N: cudaProfilerStart/Stop around the N-th prompt's prefill (nsys --capture-range=cudaProfilerApi)
        static const int prof_prompt = std::getenv("TQFN_PROFILE_PROMPT") ? std::atoi(std::getenv("TQFN_PROFILE_PROMPT")) : 0; static int prompt_no = 0;
        const bool profiling = prof_prompt > 0 && !all_argmax && ++prompt_no == prof_prompt;
        if (profiling) ck(cudaProfilerStart());
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t b = 0; b < prompt_ids.size(); b += chunk) {
            std::vector<int> part(prompt_ids.begin() + b, prompt_ids.begin() + std::min(prompt_ids.size(), b + chunk));
            auto rows = model.step(part);
            if (prompt_res) ck(cudaMemcpyAsync(prompt_res->p + b * size_t(model.width), model.res_keep->p, part.size() * size_t(model.width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
            if (all_argmax) for (int i = 0; i < int(part.size()); ++i) all_argmax->push_back(checkpoint.head_argmax_device(rows->p + size_t(i) * c.hidden, 1));
            if (b + chunk >= prompt_ids.size()) {
                out.push_back(checkpoint.head_argmax_device(rows->p, int(part.size())));
                // TQFN_LAST_LOGITS=file: float32 logits of the last prompt position to file.<prompt length> (prefill numerics gate)
                if (const char * lf = std::getenv("TQFN_LAST_LOGITS"); lf && !all_argmax) {
                    const auto lg = checkpoint.head_logits_device(rows->p + size_t(part.size() - 1) * c.hidden, 1);
                    const auto fname = std::string(lf) + "." + std::to_string(prompt_ids.size()); std::FILE * f = std::fopen(fname.c_str(), "wb"); require(f && std::fwrite(lg.data(), 4, lg.size(), f) == lg.size(), "cannot write TQFN_LAST_LOGITS");
                    std::fclose(f);
                }
            }
        }
        ck(cudaDeviceSynchronize());
        if (profiling) ck(cudaProfilerStop());
        if (auto & ps = DecodeModel::stage(); ps.copy_bytes > 0) { std::printf("PREFILL staged experts %.2f GB\n", ps.copy_bytes / 1e9); ps.copy_bytes = 0; }
        if (std::getenv("TQFN_STEP_PROBE") && !all_argmax) {
            // Cost of one warm step carrying k tokens (Plan A: multi-token verify). Advances the state; timing only.
            for (int k = 1; k <= 4; ++k) {
                for (int w = 0; w < 4; ++w) { auto r = model.step(std::vector<int>(k, out.back())); (void) r; }
                ck(cudaDeviceSynchronize()); const auto a = std::chrono::steady_clock::now();
                for (int w = 0; w < 16; ++w) { auto r = model.step(std::vector<int>(k, out.back())); checkpoint.head_argmax_device(r->p, k); }
                ck(cudaDeviceSynchronize());
                std::printf("STEP_PROBE k=%d %.2f ms/step\n", k, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count() / 16);
                std::fflush(stdout);
            }
            return out;
        }
        const auto t1 = std::chrono::steady_clock::now(); g_host_ms[0] = g_host_ms[1] = g_host_ms[2] = 0;
        const auto stats0 = cache.stats();
        static const bool stop_eos = std::getenv("TQFN_NO_EOS_STOP") == nullptr && !all_argmax;
        auto is_eos = [](int t) { return t == 248046 || t == 248044; };
        if (use_spec) {
            // MTP catch-up over the prompt: pairs (token i+1, residual i), the last pair uses the first generated token.
            const int P = int(prompt_ids.size()); int d1 = 0;
            for (int b = 0; b < P; b += 4) {
                std::vector<int> toks;
                for (int i = b; i < std::min(P, b + 4); ++i) toks.push_back(i + 1 < P ? prompt_ids[i + 1] : out[0]);
                d1 = model.mtp_batch(toks, prompt_res->p + size_t(b) * model.width);
            }
            model.set_spec(true);
            static const bool st = std::getenv("TQFN_SPEC_TRACE") != nullptr; double tt[4] = {0, 0, 0, 0};
            auto mark = [&](int k, std::chrono::steady_clock::time_point & a) {
                if (!st) return; ck(cudaDeviceSynchronize()); const auto b = std::chrono::steady_clock::now();
                tt[k] += std::chrono::duration<double, std::milli>(b - a).count(); a = b; };
            while (int(out.size()) < generate && !(stop_eos && is_eos(out.back()))) {
                auto a = std::chrono::steady_clock::now();
                const int y = out.back();
                const std::vector<int> dr = model.mtp_drafts(d1, drafts);
                mark(0, a);
                std::vector<int> block{y}; block.insert(block.end(), dr.begin(), dr.end());
                auto rows = model.step(block);
                const auto v = checkpoint.head_argmax_rows(rows->p, int(block.size()));
                mark(1, a);
                int m = 1; while (m <= drafts && v[m - 1] == dr[m - 1]) ++m;  // m - 1 drafts accepted, plus the verified next token
                for (int i = 0; i < m && int(out.size()) < generate; ++i) { out.push_back(v[i]); if (stop_eos && is_eos(v[i])) { m = i + 1; break; } }
                model.accept(m); mark(2, a);
                ++spec_steps; spec_tokens += m;
                d1 = model.mtp_batch(std::vector<int>(v.begin(), v.begin() + m), model.res_keep->p); mark(3, a);
            }
            if (st && spec_steps) std::printf("SPEC_TRACE per cycle: chain %.2f ms, verify %.2f, accept %.2f, mtp catch-up %.2f\n",
                tt[0] / spec_steps, tt[1] / spec_steps, tt[2] / spec_steps, tt[3] / spec_steps);
            model.set_spec(false);
        } else
        while (int(out.size()) < generate && !(stop_eos && is_eos(out.back()))) {
            auto rows = model.step({out.back()}); out.push_back(checkpoint.head_argmax_device(rows->p, 1));
        }
        ck(cudaDeviceSynchronize());
        const auto t2 = std::chrono::steady_clock::now();
        const auto stats = cache.stats();
        const double pre = std::chrono::duration<double>(t1 - t0).count(), dec = std::chrono::duration<double>(t2 - t1).count();
        const double hits = double(stats.hits - stats0.hits), misses = double(stats.misses - stats0.misses);
        if (generate > 1 && out.size() > 1)
            std::printf("DECODE prompt %zu prefill %.2f s (%.1f tok/s) | generated %d decode %.2f s = %.2f tok/s | expert hit rate %.1f %% (%.0f / %.0f) transfer %.2f GB prefetched %llu (used %llu)\n",
                        prompt_ids.size(), pre, prompt_ids.size() / pre, int(out.size()) - 1, dec, (int(out.size()) - 1) / dec,
                        100 * hits / std::max(1.0, hits + misses), hits, hits + misses, double(stats.transfer_bytes - stats0.transfer_bytes) / 1e9,
                        (unsigned long long) cache.prefetched(), (unsigned long long) cache.prefetch_used());
        std::printf("DECODE prefetch overlap: %llu prepares with queued copies, %llu still running, %llu waited for unread experts\n",
                    (unsigned long long) (stats.pf_pending - stats0.pf_pending), (unsigned long long) (stats.pf_running - stats0.pf_running),
                    (unsigned long long) (stats.pf_waited_unread - stats0.pf_waited_unread));
        { size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot); cudaMemPool_t pool; cudaDeviceGetDefaultMemPool(&pool, 0); uint64_t res = 0, used = 0;
          cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &res); cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
          std::printf("VRAM used %.0f MiB of %.0f MiB; pool reserved %.0f MiB, in use %.0f MiB\n", (tot - fr) / 1048576.0, tot / 1048576.0, res / 1048576.0, used / 1048576.0); }
        std::printf("HOST per generated token: ids %.2f ms, prepare %.2f ms, prefetch %.2f ms\n", g_host_ms[0] / std::max<size_t>(1, out.size() - 1),
                    g_host_ms[1] / std::max<size_t>(1, out.size() - 1), g_host_ms[2] / std::max<size_t>(1, out.size() - 1));
        if (use_spec && spec_steps) std::printf("SPEC steps %llu tokens %llu = %.2f tokens per verify\n", (unsigned long long) spec_steps,
                                                (unsigned long long) spec_tokens, double(spec_tokens) / spec_steps);
        return out;
    };
    if (prompts.size() > 1) {
        double total = 0;
        for (const auto & [name, ids] : prompts) {
            const auto t0 = std::chrono::steady_clock::now(); const auto out = run(ids, max_new, nullptr); (void) t0;
            int stop = int(out.size()); for (int i = 0; i < int(out.size()); ++i) if (out[i] == 248046 || out[i] == 248044) { stop = i + 1; break; }
            std::printf("PROMPT %s generated %zu (first stop at %d)\n", name.c_str(), out.size(), stop);
            std::printf("TOKENS %s", name.c_str()); for (int t : out) std::printf(" %d", t); std::printf("\n"); std::fflush(stdout);
            (void) total;
        }
        return;
    }
    if (std::getenv("TQFN_TF_LOGITS")) { run(prompt, 0, nullptr); return; }
    const auto generated = run(prompt, max_new, nullptr);
    std::printf("TOKENS"); for (int t : generated) std::printf(" %d", t); std::printf("\n"); std::fflush(stdout);
    if (check) {
        std::vector<int> forced = prompt; forced.insert(forced.end(), generated.begin(), generated.end() - 1);
        std::vector<int> argmax; run(forced, 1, &argmax);
        int bad = 0;
        for (size_t i = 0; i + 1 < generated.size(); ++i) {
            const int expected = generated[i], got = argmax[prompt.size() - 1 + i];
            if (expected != got) { ++bad; std::printf("CHECK mismatch at generated %zu: step %d batch %d\n", i, expected, got); }
        }
        std::printf("CHECK %s: %d of %zu generated tokens differ from the teacher-forced batch argmax\n", bad ? "FAIL" : "PASS", bad, generated.size() - 1);
    }
}

static void run_chain_kld(int argc, char ** argv) {
    require(argc >= 5, "usage: tqfn-stage --chain-kld-gsq BASE.kld fixture-dir shard1.gguf [more shards] [--gsq-rco] [--chunks N] [--first C] [--pos-out FILE] [--logits-out FILE] [--act-f16]");
    const std::string base = argv[1];
    int max_chunks = 0, first_chunk = 0; std::string pos_out, logits_out;
    std::vector<char *> args{argv[0]};
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--chunks" && i + 1 < argc) { max_chunks = std::stoi(argv[++i]); continue; }
        if (std::string(argv[i]) == "--act-f16") { g_chain_f16_act = true; continue; }
        if (std::string(argv[i]) == "--first" && i + 1 < argc) { first_chunk = std::stoi(argv[++i]); continue; }
        if (std::string(argv[i]) == "--pos-out" && i + 1 < argc) { pos_out = argv[++i]; continue; }
        if (std::string(argv[i]) == "--logits-out" && i + 1 < argc) { logits_out = argv[++i]; continue; }
        args.push_back(argv[i]);
    }
    std::ifstream in(base, std::ios::binary);
    char check[9] = {};
    in.read(check, 8);
    require(!in.fail() && std::strncmp("_logits_", check, 8) == 0, base + " is not a llama.cpp log-prob file");
    uint32_t n_ctx = 0; int n_vocab = 0, n_chunk = 0;
    in.read(reinterpret_cast<char *>(&n_ctx), 4); in.read(reinterpret_cast<char *>(&n_vocab), 4); in.read(reinterpret_cast<char *>(&n_chunk), 4);
    require(n_ctx >= 4 && n_ctx <= 64, "chain KLD needs a base made with -c <= 64");
    std::vector<int32_t> tokens(size_t(n_ctx) * n_chunk);
    in.read(reinterpret_cast<char *>(tokens.data()), std::streamsize(tokens.size() * 4));
    require(!in.fail(), "cannot read the base header / tokens");
    if (max_chunks > 0) n_chunk = std::min(n_chunk, first_chunk + max_chunks);
    require(first_chunk >= 0 && first_chunk < n_chunk, "--first beyond the base");
    Fixtures fixtures(args[1]); Checkpoint checkpoint(int(args.size()), args.data()); checkpoint.cpu_q8_projection = true;
    const auto & c = checkpoint.config; const int h = c.hidden, width = h * c.hc;
    const int nv = 2 * ((n_vocab + 1) / 2) + 4, first = int(n_ctx) / 2, nscore = int(n_ctx) - 1 - first;
    std::vector<uint16_t> lp(size_t(nscore) * nv);
    KldSums kld;
    std::vector<int> chunk(n_ctx);
    g_chain_tokens = &chunk; g_chain_quiet = true;
    std::ofstream logits_file;
    if (!logits_out.empty()) { logits_file.open(logits_out, std::ios::binary); require(bool(logits_file), "cannot open " + logits_out); }
    std::ofstream pos_file;
    if (!pos_out.empty()) { pos_file.open(pos_out); require(bool(pos_file), "cannot open " + pos_out); }
    in.seekg(std::streamoff(lp.size()) * 2 * first_chunk, std::ios::cur);
    for (int ci = first_chunk; ci < n_chunk; ++ci) {
        std::copy(tokens.begin() + size_t(ci) * n_ctx, tokens.begin() + size_t(ci + 1) * n_ctx, chunk.begin());
        run_chain(int(args.size()), args.data(), false, false, true);
        require(g_chain_residual.size() == size_t(width) * n_ctx, "chain residual size differs");
        Device<float> last(g_chain_residual.size()), norm(g_chain_residual.size()), gamma(width); last.put(g_chain_residual);
        gamma.put(checkpoint.dense("output_hc_norm.weight"));
        tqllm_cuda::hc_norm(last.p, gamma.p, norm.p, h, c.hc, int(n_ctx), float(checkpoint.epsilon), nullptr);
        const auto head = replay_mix(fixtures, checkpoint, "", 0, norm, true, false);
        std::vector<float> rows(head.mixed.begin() + size_t(first) * h, head.mixed.begin() + size_t(first + nscore) * h);
        const auto logits = checkpoint.head_logits_rows(rows, nscore);
        const size_t V = logits.size() / size_t(nscore);
        require(V >= size_t(n_vocab), "head vocabulary smaller than the base");
        in.read(reinterpret_cast<char *>(lp.data()), std::streamsize(lp.size() * 2));
        require(!in.fail(), "failed reading log-probs of chunk " + std::to_string(ci));
        std::vector<float> row(n_vocab);
        if (logits_file) for (int i = 0; i < nscore; ++i)
            logits_file.write(reinterpret_cast<const char *>(logits.data() + size_t(i) * V), std::streamsize(size_t(n_vocab) * sizeof(float)));
        for (int i = 0; i < nscore; ++i) {
            std::copy(logits.begin() + size_t(i) * V, logits.begin() + size_t(i) * V + n_vocab, row.begin());
            const double before = kld.sum_kld; const size_t same_before = kld.n_same_top;
            kld_token(n_vocab, row.data(), lp.data() + size_t(i) * nv, chunk[first + i + 1], kld);
            if (pos_file) pos_file << ci << '\t' << first + i << '\t' << kld.sum_kld - before << '\t' << kld.n_same_top - same_before << '\n';
        }
        std::printf("chunk %d  mean KLD %.5f  same top %.4f  positions %zu\n", ci, kld.sum_kld / kld.count,
                    double(kld.n_same_top) / kld.count, kld.count);
        std::fflush(stdout);
    }
    g_chain_tokens = nullptr; g_chain_quiet = false;
    std::printf("NATIVE_CHAIN_KLD positions %zu  mean KLD %.5f  same top %.4f  PPL %.4f  PPL(base) %.4f\n", kld.count,
                kld.sum_kld / kld.count, double(kld.n_same_top) / kld.count, std::exp(kld.sum_nll / kld.count),
                std::exp(kld.sum_nll_base / kld.count));
}
static void run(int argc, char ** argv) {
    require(argc >= 3, "usage: tqfn-stage fixture-dir shard1.gguf [shard2.gguf ...]");
    Fixtures fixtures(argv[1]); Checkpoint checkpoint(argc, argv); const auto & c = checkpoint.config;
    require(fixtures.cpu_f32_router(), "F32 router projection gate requires router=cpu-f32 fixtures; the default reference uses custom TF32 MMF");
    const int h = c.hidden, streams = c.hc, width = h * streams;
    const auto tokens = fixtures.tokens(); const int nt = int(tokens.size());
    PleTokenHistory history(c.ple, tokens.size()); const auto row_ids = history.append_batch(tokens);
    std::vector<uint64_t> rows(row_ids.begin(), row_ids.end());
    auto gathered = checkpoint.file("per_layer_token_embd.weight").gather_rows("per_layer_token_embd.weight", rows);
    const auto ple_expected = fixtures.values("ple_embd");
    require(gathered.size() == ple_expected.size() && !std::memcmp(gathered.data(), ple_expected.data(), gathered.size() * 4),
            "PLE hash/gather differs from independent reference forward");
    std::cout << "PLE_HASH_GATHER_BIT_EXACT rows=" << rows.size() << '\n';
    std::vector<float> residual = fixtures.values("hc_init");
    Device<float> embedding(h * nt), initialized(width * nt); embedding.put(fixtures.values("model.input_embed"));
    tqllm_cuda::hc_init(embedding.p, initialized.p, h, streams, nt, nullptr);
    compare("hc_init", initialized.get(), residual, 0.);
    Tq3ExpertCache cache(c.layers, c.experts, h, c.expert_ff, 64, 256 * 1024 * 1024,
                         [&](int layer, int expert) { return checkpoint.expert(layer, expert); }, nullptr);
    for (int layer = 0; layer < c.layers; ++layer) {
        const std::string suffix = "-" + std::to_string(layer), prefix = "blk." + std::to_string(layer) + ".";
        const bool ple_layer = std::find(c.ple_layers.begin(), c.ple_layers.end(), layer) != c.ple_layers.end();
        if (ple_layer) {
            require(gathered.size() == size_t(h) * nt, "PLE concatenated embedding width differs");
            Device<float> dp(gathered.size()), key(size_t(width) * nt), value(size_t(h) * nt), query(residual.size());
            Device<float> key_norm(key.count), query_norm(query.count), gates(size_t(streams) * nt), native_gated(residual.size());
            Device<float> norm_weight{size_t(width)}; dp.put(gathered); query.put(residual);
            checkpoint.project(prefix + "ple_key.weight", dp.p, key.p, h, width, nt);
            checkpoint.project(prefix + "ple_value.weight", dp.p, value.p, h, h, nt);
            norm_weight.put(checkpoint.dense(prefix + "ple_norm_key.weight"));
            tqllm_cuda::hc_norm(key.p, norm_weight.p, key_norm.p, h, streams, nt, float(checkpoint.epsilon), nullptr);
            norm_weight.put(checkpoint.dense(prefix + "ple_norm_query.weight"));
            tqllm_cuda::hc_norm(query.p, norm_weight.p, query_norm.p, h, streams, nt, float(checkpoint.epsilon), nullptr);
            tqllm_cuda::ple_gate(key_norm.p, query_norm.p, gates.p, h, streams, nt, nullptr);
            tqllm_cuda::ple_broadcast(value.p, gates.p, native_gated.p, h, streams, nt, nullptr);
            compare("ple_projected_gate" + suffix, gates.get(), fixtures.values("ple_gate" + suffix), .02);
            auto gated = native_gated.get(), conv_expected = fixtures.values("ple_conv_out" + suffix);
            compare("ple_projected_gated_value" + suffix, gated, fixtures.values("ple_gated_value" + suffix), .02);
            auto gamma = checkpoint.dense(prefix + "ple_norm_conv.weight"), kernel = checkpoint.dense(prefix + "ple_conv1d.weight");
            Device<float> dg(gated.size()), dw(width), dk(kernel.size()), norm(gated.size()), conv(gated.size());
            const int hist = (c.ple_kernel - 1) * c.ple.ngram;
            Device<float> dh(width * hist), next(width * hist), dr(residual.size()), after(residual.size());
            dg.put(gated); dw.put(gamma); dk.put(kernel); dh.put(std::vector<float>(width * hist, 0.f)); dr.put(residual);
            tqllm_cuda::hc_norm(dg.p, dw.p, norm.p, h, streams, nt, float(checkpoint.epsilon), nullptr);
            tqllm_cuda::ple_conv(norm.p, dh.p, dk.p, conv.p, next.p, width, nt, c.ple_kernel, c.ple.ngram, nullptr);
            compare("ple_conv" + suffix, conv.get(), conv_expected, .02);
            tqllm_cuda::ple_apply(dr.p, dg.p, conv.p, after.p, int(residual.size()), nullptr); residual = after.get();
        }
        const auto attention_norm = fixtures.values("hc_norm" + suffix, 0);
        const int attn_tokens = int(attention_norm.size() / width);
        Device<float> dr(residual.size()), dg(width), dn(residual.size()); dr.put(residual);
        dg.put(checkpoint.dense(prefix + "hc_attn_norm.weight"));
        tqllm_cuda::hc_norm(dr.p, dg.p, dn.p, h, streams, attn_tokens, float(checkpoint.epsilon), nullptr);
        compare("hc_attn_norm" + suffix, dn.get(), attention_norm, ple_layer ? .02 : 2e-5);
        const auto attn_mix = replay_mix(fixtures, checkpoint, suffix, 0, dn);
        // Last-layer output selection happens between attention and the FFN mixer.
        auto attention_out = fixtures.values((fixtures.has("linear_attn_out" + suffix) ? "linear_attn_out" : "attn_output") + suffix);
        auto attention_inject = attn_mix.inject;
        const auto combined_expected = fixtures.values("hc_combine" + suffix);
        const int selected = int(combined_expected.size() / width);
        require(residual.size() >= combined_expected.size() && attention_out.size() >= size_t(selected) * h &&
                attention_inject.size() >= size_t(selected) * streams, "attention output selection dimensions differ");
        const auto last = [](std::vector<float> & values, size_t count) { values.erase(values.begin(), values.end() - count); };
        last(residual, size_t(selected) * width); last(attention_out, size_t(selected) * h); last(attention_inject, size_t(selected) * streams);
        Device<float> dares(residual.size()), dao(attention_out.size()), dai(attention_inject.size()), dac(residual.size());
        dares.put(residual); dao.put(attention_out); dai.put(attention_inject);
        tqllm_cuda::hc_combine(dares.p, dao.p, dai.p, dac.p, h, streams, selected, nullptr);
        compare("hc_attn_combine" + suffix, dac.get(), combined_expected, .02);
        residual = combined_expected;
        const int n = int(residual.size() / width);
        Device<float> dres(residual.size()), dn2(residual.size()); dres.put(residual);
        dg.put(checkpoint.dense(prefix + "hc_ffn_norm.weight"));
        tqllm_cuda::hc_norm(dres.p, dg.p, dn2.p, h, streams, n, float(checkpoint.epsilon), nullptr);
        compare("hc_ffn_norm" + suffix, dn2.get(), fixtures.values("hc_norm" + suffix, 1), 2e-5);
        const auto ffn_mix = replay_mix(fixtures, checkpoint, suffix, 1, dn2);
        const auto mixed = fixtures.values("hc_mixed" + suffix, 1);
        const auto reference_ids = fixtures.ids("ffn_moe_topk" + suffix);
        const auto reference_logits = fixtures.values("ffn_moe_logits" + suffix);
        require(reference_ids.size() == size_t(n * c.topk) && reference_logits.size() == size_t(n * c.experts), "routing fixture shape differs");
        std::vector<float> logits(size_t(n) * (c.experts + 1));
        for (int token = 0; token < n; ++token)
            std::copy_n(reference_logits.data() + token * c.experts, c.experts, logits.data() + token * (c.experts + 1));
        Device<float> dl(logits.size()), dwroute(reference_ids.size()), ds(n); Device<int> di(reference_ids.size()); dl.put(logits);
        tqllm_cuda::moe_topk(dl.p, di.p, dwroute.p, ds.p, c.experts, c.topk, n, 1.f, nullptr);
        require(di.get() == reference_ids, "native top-k differs from reference selected experts at " + std::to_string(layer));
        compare("router_weights" + suffix, dwroute.get(), fixtures.values("ffn_moe_weights_norm" + suffix), 3e-6);
        const auto router = checkpoint.dense(prefix + "ffn_gate_inp.weight");
        const auto shared_gate = checkpoint.dense(prefix + "ffn_gate_inp_shexp.weight");
        require(router.size() == size_t(h) * c.experts && shared_gate.size() == size_t(h), "router weight shape differs");
        Device<float> drouter(router.size()), dshared(shared_gate.size()), plain(mixed.size());
        drouter.put(router); dshared.put(shared_gate); plain.put(mixed);
        tqllm_cuda::moe_router_f32(drouter.p, dshared.p, plain.p, dl.p, h, c.experts, n, nullptr);
        const auto projected = dl.get(); std::vector<float> dense_logits(reference_logits.size()), dense_shared(n);
        for (int token = 0; token < n; ++token) {
            std::copy_n(projected.data() + token * (c.experts + 1), c.experts, dense_logits.data() + token * c.experts);
            dense_shared[token] = projected[token * (c.experts + 1) + c.experts];
        }
        compare("router_logits" + suffix, dense_logits, reference_logits, 2e-5);
        compare("shared_gate_logits" + suffix, dense_shared, fixtures.values("shared_expert_gate" + suffix), 2e-5);
        tqllm_cuda::moe_topk(dl.p, di.p, dwroute.p, ds.p, c.experts, c.topk, n, 1.f, nullptr);
        require(di.get() == reference_ids, "projected native router changed selected experts at " + std::to_string(layer));
        const auto physical = cache.prepare(layer, reference_ids); di.put(physical);
        const auto cold = cache.stats(); require(cache.prepare(layer, reference_ids) == physical, "warm expert placement differs");
        require(cache.stats().misses == cold.misses && cache.stats().transfer_bytes == cold.transfer_bytes, "warm expert replay transferred weights");
        Device<float> dx(mixed.size()); dx.put(mixed); Device<__nv_bfloat16> db(mixed.size()); Device<__half> rotated(mixed.size());
        tqllm_cuda::f32_to_bf16(dx.p, db.p, int64_t(mixed.size()), nullptr);
        tqllm_cuda::tq3_rotate_h(db.p, rotated.p, h, n, nullptr);
        for (int projection = 0; projection < 2; ++projection) {
            const auto matrix = projection == 0 ? cache.gate() : cache.up(); Device<__nv_bfloat16> out(size_t(n) * c.topk * c.expert_ff);
            matrix.fmt->expert_matvec(matrix, rotated.p, out.p, c.expert_ff, di.p, n, c.topk, false, nullptr);
            compare((projection == 0 ? "expert_gate" : "expert_up") + suffix, bf16_values(out),
                    fixtures.values((projection == 0 ? "ffn_moe_gate" : "ffn_moe_up") + suffix), .02);
        }
        // Isolate the down projection using the reference activation, rather than hiding earlier errors.
        auto activation = fixtures.values("ffn_moe_swiglu" + suffix);
        Device<float> da(activation.size()); da.put(activation); Device<__nv_bfloat16> dab(activation.size()); Device<__half> dar(activation.size());
        tqllm_cuda::f32_to_bf16(da.p, dab.p, int64_t(activation.size()), nullptr);
        tqllm_cuda::tq3_rotate_h(dab.p, dar.p, c.expert_ff, n * c.topk, nullptr);
        const auto matrix = cache.down(); Device<__nv_bfloat16> down(size_t(n) * c.topk * h);
        matrix.fmt->expert_matvec(matrix, dar.p, down.p, h, di.p, n, c.topk, true, nullptr);
        compare("expert_down" + suffix, bf16_values(down), fixtures.values("ffn_moe_down" + suffix), .02);
        auto block = fixtures.values("ffn_out" + suffix), inject = ffn_mix.inject;
        Device<float> dout(block.size()), dinject(inject.size()), combined(residual.size()); dout.put(block); dinject.put(inject);
        tqllm_cuda::hc_combine(dres.p, dout.p, dinject.p, combined.p, h, streams, n, nullptr);
        residual = fixtures.values("l_last" + suffix); compare("hc_ffn_combine" + suffix, combined.get(), residual, .02);
    }
    Device<float> final_residual(residual.size()), head_norm(residual.size()), head_gamma{size_t(width)};
    final_residual.put(residual); head_gamma.put(checkpoint.dense("output_hc_norm.weight"));
    tqllm_cuda::hc_norm(final_residual.p, head_gamma.p, head_norm.p, h, streams, int(residual.size() / width),
                       float(checkpoint.epsilon), nullptr);
    compare("hc_head_norm", head_norm.get(), fixtures.values("hc_norm"), 2e-5);
    replay_mix(fixtures, checkpoint, "", 0, head_norm, true);
    const auto stats = cache.stats();
    std::cout << "NATIVE_STAGE_REPLAY_DONE layers=" << c.layers << " cache_device_bytes=" << stats.device_bytes
              << " misses=" << stats.misses << " hits=" << stats.hits << " evictions=" << stats.evictions
              << " transfer_bytes=" << stats.transfer_bytes << " full_forward=false\n";
}
int main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--chain-f32-experts") run_chain(argc - 1, argv + 1, false, true);
        else if (argc > 1 && std::string(argv[1]) == "--chain-cpu-q8-f32-experts") run_chain(argc - 1, argv + 1, true, true);
        else if (argc > 1 && std::string(argv[1]) == "--chain") run_chain(argc - 1, argv + 1, false);
        else if (argc > 1 && std::string(argv[1]) == "--chain-cpu-q8") run_chain(argc - 1, argv + 1, true);
        else if (argc > 1 && std::string(argv[1]) == "--chain-gsq-cpu-q8k") run_chain(argc - 1, argv + 1, false, false, true);
        else if (argc > 1 && std::string(argv[1]) == "--chain-kld-gsq") run_chain_kld(argc - 1, argv + 1);
        else if (argc > 1 && std::string(argv[1]) == "--decode") run_decode(argc - 1, argv + 1);
        else if (argc > 1 && std::string(argv[1]) == "--pack-experts") run_pack_experts(argc - 1, argv + 1);
        else if (argc > 1 && std::string(argv[1]) == "--recurrent-cpu-q8k") run_recurrent(argc - 1, argv + 1, true);
        else if (argc > 1 && std::string(argv[1]) == "--recurrent") run_recurrent(argc - 1, argv + 1);
        else if (argc > 1 && std::string(argv[1]) == "--attention-cpu-q8") run_attention(argc - 1, argv + 1, true);
        else if (argc > 1 && std::string(argv[1]) == "--attention") run_attention(argc - 1, argv + 1);
        else run(argc, argv);
        return 0;
    } catch (const std::exception & e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
