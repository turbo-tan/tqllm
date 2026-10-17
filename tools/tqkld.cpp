// tqkld: KL divergence of a TQLLM model against a reference's top-K log-probs, teacher forced.
// Usage: tqkld --model M.gguf [--tokenizer T] --ref DIR --out PREFIX [--skip 16]
// DIR holds ids.bin (uint32 [chunks][L]), ref_ids.bin / ref_lp.bin (uint32 / float32 [chunks][L-1][K]) and meta.json
// (tqllm/kld/make_ref.py). Every chunk starts at position 0 and is fed in verify blocks of up to 8 tokens; the logits at
// position t (predicting t + 1) are compared with the reference: KL(ref || model) over the reference's top K plus one
// bucket for the rest, top-1 agreement and the model's NLL of the actual next token.
// Writes PREFIX.kld.f32, PREFIX.top1.u8, PREFIX.nll.f32 ([chunks][L-1]; positions < skip are NaN / 255) and a summary.
#include "tokenizer_path.hpp"
#include "tqllm/options.hpp"
#include <cstring>
#include "tqllm/model.hpp"
#include <third_party/nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static tqllm::EngineOptions & kld_engine() {  // -ctk / -ctv / -md ... for every model load
    static tqllm::EngineOptions o;
    return o;
}

namespace {
template <typename T> std::vector<T> read_all(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<T> v(size_t(f.tellg()) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
    return v;
}
template <typename T> void write_all(const std::string & path, const std::vector<T> & v) {
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
}

// ---- llama.cpp-standard KLD (tools/perplexity/perplexity.cpp, kl_divergence): the base file written by
// `llama-perplexity --kl-divergence-base`, positions n_ctx/2 .. n_ctx-2 of every chunk, full-vocabulary KL with base
// log-probs below -16 skipped, NLL / Delta-p / same-top statistics and the same summary lines.
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

int llama_base_mode(const std::string & model_path, const std::string & tok_path, const std::string & base, int block,
                    int max_chunks, bool prefill) {
    std::ifstream in(base, std::ios::binary);
    char check[9] = {};
    in.read(check, 8);
    if (in.fail() || std::strncmp("_logits_", check, 8) != 0) throw std::runtime_error(base + " is not a llama.cpp log-prob file");
    uint32_t n_ctx = 0;
    int n_vocab = 0, n_chunk = 0;
    in.read(reinterpret_cast<char *>(&n_ctx), 4);
    in.read(reinterpret_cast<char *>(&n_vocab), 4);
    in.read(reinterpret_cast<char *>(&n_chunk), 4);
    std::vector<int32_t> tokens(size_t(n_ctx) * n_chunk);
    in.read(reinterpret_cast<char *>(tokens.data()), std::streamsize(tokens.size() * 4));
    if (in.fail()) throw std::runtime_error("cannot read header / tokens of " + base);
    if (max_chunks > 0) n_chunk = std::min(n_chunk, max_chunks);
    tqllm::Model model(model_path, tok_path, n_ctx + 64, 1, kld_engine());
    model.set_full_logits(true);  // KLD needs every logit, not the two-stage top-16
    if (int(model.vocab_size()) != n_vocab)
        std::fprintf(stderr, "tqkld: vocabulary %zu vs base %d (the extra rows are ignored)\n", model.vocab_size(), n_vocab);
    const size_t V = model.vocab_size();
    const int nv = 2 * ((n_vocab + 1) / 2) + 4, first = int(n_ctx) / 2, nscore = int(n_ctx) - 1 - first;
    const int B = std::clamp(block, 1, 8);
    std::vector<uint16_t> lp(size_t(nscore) * nv);
    std::vector<float> lg(size_t(B) * V), chunk_logits(size_t(nscore) * n_vocab);
    std::vector<float> kld_values, p_diff_values;
    KldSums kld;
    auto mean_unc = [](double sum, double sum2, size_t count) {
        if (count < 1) return std::make_pair(0., 0.);
        double f = sum / count, df = sum2 / count - f * f;
        df = df > 0 && count > 10 ? std::sqrt(df / (count - 1)) : 0.;
        return std::make_pair(f, df);
    };
    auto covariance = [](double a, double b, double ab, size_t count) {
        if (count < 10) return 0.0;
        return (ab / count - (a / count) * (b / count)) / double(count - 1);
    };
    std::printf("chunk  KLD (running mean)  same top\n");
    for (int c = 0; c < n_chunk; ++c) {
        const int32_t * tk = tokens.data() + size_t(c) * n_ctx;  // Qwen vocabularies add no BOS (llama.cpp add_bos false)
        // --llama-prefill: the unscored first half goes through the chunked prefill path (its GEMM shapes the scored KV)
        if (prefill && !model.prefill_state(std::vector<uint32_t>(tk, tk + first), 0))
            throw std::runtime_error("--llama-prefill needs a model with chunked prefill");
        for (int pos = prefill ? first : 0; pos < int(n_ctx) - 1; pos += B) {
            const int n = std::min(B, int(n_ctx) - 1 - pos);
            std::vector<uint32_t> blk(tk + pos, tk + pos + n), next;
            model.verify(blk, uint32_t(pos), next);
            if (pos + n > first) {
                if (cudaMemcpy(lg.data(), model.verify_logits(), size_t(n) * V * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
                    throw std::runtime_error("cudaMemcpy of verify logits failed");
                for (int j = 0; j < n; ++j)
                    if (pos + j >= first)
                        std::memcpy(chunk_logits.data() + size_t(pos + j - first) * n_vocab, lg.data() + size_t(j) * V,
                                    size_t(n_vocab) * 4);
            }
            model.commit(n);
        }
        in.read(reinterpret_cast<char *>(lp.data()), std::streamsize(lp.size() * 2));
        if (in.fail()) throw std::runtime_error("failed reading log-probs of chunk " + std::to_string(c));
        for (int i = 0; i < nscore; ++i) {
            const auto v = kld_token(n_vocab, chunk_logits.data() + size_t(i) * n_vocab, lp.data() + size_t(i) * nv,
                                     tk[first + i + 1], kld);
            kld_values.push_back(float(v.first));
            p_diff_values.push_back(v.second);
        }
        std::printf("%4d  %10.6f  %6.3f %%\n", c + 1, kld.sum_kld / kld.count, 100.0 * kld.n_same_top / kld.count);
        std::fflush(stdout);
    }
    std::sort(kld_values.begin(), kld_values.end());
    std::sort(p_diff_values.begin(), p_diff_values.end());
    auto percentile = [](const std::vector<float> & v, float f) {
        if (f <= 0) return v.front();
        if (f >= 1) return v.back();
        float p = f * float(v.size() - 1);
        const size_t ip = size_t(p);
        p -= float(ip);
        return (1 - p) * v[ip] + p * v[std::min(ip + 1, v.size() - 1)];
    };
    auto median = [](const std::vector<float> & v) {
        return v.size() % 2 == 0 ? 0.5f * (v[v.size() / 2] + v[v.size() / 2 - 1]) : v[v.size() / 2];
    };
    const auto lq = mean_unc(kld.sum_nll, kld.sum_nll2, kld.count), lb = mean_unc(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
    const double cov = covariance(kld.sum_nll, kld.sum_nll_base, kld.sum_nll_nll_base, kld.count);
    const double ratio = lq.first - lb.first, ratio_unc = std::sqrt(lq.second * lq.second + lb.second * lb.second - 2.0 * cov);
    std::printf("====== Perplexity statistics ======\n");
    std::printf("Mean PPL(Q)                   : %10.6f ± %10.6f\n", std::exp(lq.first), std::exp(lq.first) * lq.second);
    std::printf("Mean PPL(base)                : %10.6f ± %10.6f\n", std::exp(lb.first), std::exp(lb.first) * lb.second);
    std::printf("Mean ln(PPL(Q)/PPL(base))     : %10.6f ± %10.6f\n", ratio, ratio_unc);
    std::printf("Mean PPL(Q)/PPL(base)         : %10.6f ± %10.6f\n", std::exp(ratio), std::exp(ratio) * ratio_unc);
    std::printf("====== KL divergence statistics ======\n");
    const auto kl = mean_unc(kld.sum_kld, kld.sum_kld2, kld.count);
    std::printf("Mean    KLD: %10.6f ± %10.6f\n", kl.first, kl.second);
    std::printf("Maximum KLD: %10.6f\n", kld_values.back());
    std::printf("99.9%%   KLD: %10.6f\n", percentile(kld_values, 0.999f));
    std::printf("99.0%%   KLD: %10.6f\n", percentile(kld_values, 0.990f));
    std::printf("95.0%%   KLD: %10.6f\n", percentile(kld_values, 0.950f));
    std::printf("90.0%%   KLD: %10.6f\n", percentile(kld_values, 0.900f));
    std::printf("Median  KLD: %10.6f\n", median(kld_values));
    std::printf("====== Token probability statistics ======\n");
    const auto pd = mean_unc(kld.sum_p_diff, kld.sum_p_diff2, kld.count);
    std::printf("Mean    Δp: %6.3f ± %5.3f %%\n", 100.0 * pd.first, 100.0 * pd.second);
    std::printf("Median  Δp: %6.3f%%\n", 100.0 * median(p_diff_values));
    const auto mse = mean_unc(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);
    const double rms = std::sqrt(mse.first);
    std::printf("RMS Δp    : %6.3f ± %5.3f %%\n", 100.0 * rms, 100.0 * 0.5 / rms * mse.second);
    const double st = double(kld.n_same_top) / kld.count;
    std::printf("Same top p: %6.3f ± %5.3f %%\n", 100.0 * st, 100.0 * std::sqrt(st * (1.0 - st) / (kld.count - 1)));
    return 0;
}
} // namespace

// --dump-ref: teacher-force the ids of --ref and write the model's own top-K log-probs as a reference in DIR (same
// layout as make_ref.py). With a 16-bit KV cache (the default bf16) this is the KV-only reference: other caches on the
// same weights are then measured for their KV error alone.
int dump_ref_mode(const std::string & model_path, const std::string & tok_path, const std::string & ref, const std::string & dir,
                  int block, int tail) {
    std::stringstream ms;
    ms << std::ifstream(ref + "/meta.json").rdbuf();
    const std::string meta_text = ms.str();
    const auto meta = tqllm::parse_json(meta_text);
    const int L = int(meta.get("chunk")->num), K = int(meta.get("topk")->num), C = int(meta.get("chunks")->num);
    const auto ids = read_all<uint32_t>(ref + "/ids.bin");
    if (ids.size() != size_t(C) * L) throw std::runtime_error("ids.bin does not match meta.json");
    tqllm::Model model(model_path, tok_path, uint32_t(L + 64), 1, kld_engine());
    model.set_full_logits(true);  // KLD needs every logit, not the two-stage top-16
    const size_t V = model.vocab_size();
    const int B = std::clamp(block, 1, 8);
    std::vector<float> lg(size_t(B) * V);
    std::vector<uint32_t> rid(size_t(C) * (L - 1) * K);
    std::vector<float> rlp(rid.size());
    std::vector<uint32_t> order(V);
    const int start = tail > 0 ? std::max(0, L - 1 - tail) : 0;
    for (int c = 0; c < C; ++c) {
        if (start > 0 && !model.prefill_state(std::vector<uint32_t>(ids.begin() + size_t(c) * L, ids.begin() + size_t(c) * L + start), 0))
            throw std::runtime_error("--tail needs a model with chunked prefill");
        int compacted = 0;  // --kv-compact: what an idle server would compact before the next request
        for (; model.kv_idle_pending(); ++compacted) model.kv_idle_step();
        if (compacted) std::printf("chunk %2d: %d blocks compacted before scoring\n", c, compacted);
        for (int pos = start; pos < L - 1; pos += B) {
            const int n = std::min(B, L - 1 - pos);
            std::vector<uint32_t> blk(ids.begin() + size_t(c) * L + pos, ids.begin() + size_t(c) * L + pos + n), next;
            model.verify(blk, uint32_t(pos), next);
            if (cudaMemcpy(lg.data(), model.verify_logits(), size_t(n) * V * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("cudaMemcpy of verify logits failed");
            model.commit(n);
            for (int j = 0; j < n; ++j) {
                const float * q = lg.data() + size_t(j) * V;
                float m = q[0];
                for (size_t v = 1; v < V; ++v) m = std::max(m, q[v]);
                double se = 0;
                for (size_t v = 0; v < V; ++v) se += std::exp(double(q[v] - m));
                const double lse = m + std::log(se);
                for (size_t v = 0; v < V; ++v) order[v] = uint32_t(v);
                std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](uint32_t a, uint32_t b) { return q[a] > q[b]; });
                const size_t at = (size_t(c) * (L - 1) + pos + j) * K;
                for (int k = 0; k < K; ++k) rid[at + k] = order[k], rlp[at + k] = float(double(q[order[k]]) - lse);
            }
        }
        std::printf("chunk %2d dumped\n", c);
        std::fflush(stdout);
    }
    write_all(dir + "/ids.bin", ids);
    write_all(dir + "/ref_ids.bin", rid);
    write_all(dir + "/ref_lp.bin", rlp);
    std::ofstream(dir + "/meta.json") << meta_text;
    return 0;
}

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqkld");
    std::string model_path, tok_path, ref, out;
    std::string llama_base, dump_ref;
    int skip = 16, block = 8, max_chunks = 0, tail = 0;
    bool llama_prefill = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") model_path = val();
        else if (a == "--tokenizer") tok_path = val();
        else if (a == "--ref") ref = val();
        else if (a == "--out") out = val();
        else if (a == "--skip") skip = std::stoi(val());
        else if (a == "--block") block = std::stoi(val());  // verify block, 1..8
        else if (a == "--chunks") max_chunks = std::stoi(val());
        else if (a == "--llama-base") llama_base = val();
        else if (a == "--llama-prefill") llama_prefill = true;
        else if (a == "--dump-ref") dump_ref = val();
        else if (a == "--tail") tail = std::stoi(val());  // long context: prefill each chunk without the head, score only its last N positions  // write this model's own top-K reference (same format) into DIR  // llama.cpp-standard KLD against a --kl-divergence-base file
        else if (tqllm::parse_engine_option(argc, argv, i, kld_engine())) continue;
        else throw std::invalid_argument("unknown argument " + a);
    }
    tok_path = tqllm::tokenizer_for(model_path, tok_path);
    if (!llama_base.empty() && !model_path.empty() && !tok_path.empty()) {
        try {
            return llama_base_mode(model_path, tok_path, llama_base, block, max_chunks, llama_prefill);
        } catch (const std::exception & e) {
            std::cerr << "tqkld: " << e.what() << "\n";
            return 1;
        }
    }
    if (!dump_ref.empty() && !model_path.empty() && !tok_path.empty() && !ref.empty()) {
        try {
            return dump_ref_mode(model_path, tok_path, ref, dump_ref, block, tail);
        } catch (const std::exception & e) {
            std::cerr << "tqkld: " << e.what() << "\n";
            return 1;
        }
    }
    if (model_path.empty() || tok_path.empty() || ref.empty() || out.empty()) {
        std::cerr << "usage: tqkld --model M.gguf [--tokenizer T (default: the GGUF)] --ref DIR --out PREFIX [--skip 16]\n";
        return 2;
    }
    try {
        std::stringstream ms;
        ms << std::ifstream(ref + "/meta.json").rdbuf();
        const auto meta = tqllm::parse_json(ms.str());
        const int L = int(meta.get("chunk")->num), K = int(meta.get("topk")->num), C = int(meta.get("chunks")->num);
        const auto ids = read_all<uint32_t>(ref + "/ids.bin");
        const auto rid = read_all<uint32_t>(ref + "/ref_ids.bin");
        const auto rlp = read_all<float>(ref + "/ref_lp.bin");
        if (ids.size() != size_t(C) * L || rid.size() != size_t(C) * (L - 1) * K || rlp.size() != rid.size())
            throw std::runtime_error("reference files do not match meta.json");
        tqllm::Model model(model_path, tok_path, uint32_t(L + 64), 1, kld_engine());
        model.set_full_logits(true);  // KLD needs every logit, not the two-stage top-16
        const size_t V = model.vocab_size();
        const int B = std::clamp(block, 1, 8);  // verify block (<= NVER)
        const int CR = max_chunks > 0 ? std::min(C, max_chunks) : C;
        std::vector<float> lg(size_t(B) * V);
        const size_t P = size_t(C) * (L - 1);
        std::vector<float> kld(P, NAN), nll(P, NAN);
        std::vector<uint8_t> top1(P, 255);
        const int start = tail > 0 ? std::max(0, L - 1 - tail) : 0;
        if (tail > 0) skip = std::max(skip, start);
        for (int c = 0; c < CR; ++c) {
            if (start > 0 && !model.prefill_state(std::vector<uint32_t>(ids.begin() + size_t(c) * L, ids.begin() + size_t(c) * L + start), 0))
                throw std::runtime_error("--tail needs a model with chunked prefill");
            for (int pos = start; pos < L - 1; pos += B) {
                const int n = std::min(B, L - 1 - pos);
                std::vector<uint32_t> blk(ids.begin() + size_t(c) * L + pos, ids.begin() + size_t(c) * L + pos + n), next;
                model.verify(blk, uint32_t(pos), next);
                if (cudaMemcpy(lg.data(), model.verify_logits(), size_t(n) * V * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
                    throw std::runtime_error("cudaMemcpy of verify logits failed");
                model.commit(n);
                for (int j = 0; j < n; ++j) {
                    const int t = pos + j;
                    if (t < skip) continue;
                    const float * q = lg.data() + size_t(j) * V;
                    const size_t at = size_t(c) * (L - 1) + t;
                    float m = q[0];
                    size_t am = 0;
                    for (size_t v = 1; v < V; ++v) if (q[v] > m) { m = q[v]; am = v; }
                    double se = 0;
                    for (size_t v = 0; v < V; ++v) se += std::exp(double(q[v] - m));
                    const double lse = m + std::log(se);
                    const uint32_t * ri = rid.data() + at * K;
                    const float * rl = rlp.data() + at * K;
                    double kl = 0, pm = 0, qm = 0;
                    for (int k = 0; k < K; ++k) {
                        const double lq = double(q[ri[k]]) - lse, p = std::exp(double(rl[k]));
                        kl += p * (double(rl[k]) - lq);
                        pm += p;
                        qm += std::exp(lq);
                    }
                    const double pr = std::max(1e-12, 1 - pm), qr = std::max(1e-12, 1 - qm);
                    kl += pr * std::log(pr / qr);
                    kld[at] = float(std::max(0.0, kl));
                    top1[at] = am == ri[0];
                    nll[at] = float(lse - q[ids[size_t(c) * L + t + 1]]);
                }
            }
            double s = 0, a = 0;
            int cnt = 0;
            for (int t = skip; t < L - 1; ++t) {
                s += kld[size_t(c) * (L - 1) + t];
                a += top1[size_t(c) * (L - 1) + t];
                ++cnt;
            }
            std::printf("chunk %2d  KLD %.5f  top1 %.4f\n", c, s / cnt, a / cnt);
            std::fflush(stdout);
        }
        write_all(out + ".kld.f32", kld);
        write_all(out + ".top1.u8", top1);
        write_all(out + ".nll.f32", nll);
        std::vector<float> all;
        double ta = 0, tn = 0;
        for (size_t i = 0; i < P; ++i)
            if (!std::isnan(kld[i])) { all.push_back(kld[i]); ta += top1[i]; tn += nll[i]; }
        std::sort(all.begin(), all.end());
        double mean = 0;
        for (float x : all) mean += x;
        mean /= double(all.size());
        auto pct = [&](double p) { return all[std::min(all.size() - 1, size_t(p * double(all.size())))]; };
        std::printf("positions %zu  mean KLD %.5f  median %.5f  p99 %.4f  p99.9 %.4f  top1 %.4f  PPL %.4f\n", all.size(), mean,
                    pct(0.5), pct(0.99), pct(0.999), ta / double(all.size()), std::exp(tn / double(all.size())));
    } catch (const std::exception & e) {
        std::cerr << "tqkld: " << e.what() << "\n";
        return 1;
    }
}
