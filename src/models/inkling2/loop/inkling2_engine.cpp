// Inkling-2-Mini behind the serving Engine interface (tqserve, tqllm::Model): one slot, chunked prefill, greedy decode
// with the MTP drafter (--mtp: mtp.safetensors in the folder) (spec == plain, src/models/inkling2/loop/inkling2_model.hpp),
// sampled decode (temperature / top-k / top-p on the host from the full logits) without the drafter.
#include "inkling2_model.hpp"
#include "inkling2_spec_policy.hpp"
#include "debug.hpp"

#include "../../../engine/engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <random>
#include <stdexcept>

namespace tqllm {
namespace {
// prompt chunk: 8192 tokens where memory allows (GB10 +12-16 % prefill: fuller grouped-expert tiles), else 2048
int prompt_chunk() { return inkling2_prompt_chunk(); }
constexpr uint32_t kEos = 200006;  // generation_config eos_token_id

class Inkling2Engine final : public Engine {
public:
    // drafter: -md FOLDER (a DFlash2 drafter folder), else --mtp (the DSpark head in the checkpoint's mtp.safetensors)
    Inkling2Engine(const std::string & dir, uint32_t context, bool mtp, bool exact, const std::string & draft_model)
        : m_(dir, int(context), prompt_chunk(), false, SIZE_MAX, exact,
             Inkling2Model::drafter_bytes(!draft_model.empty() ? draft_model : mtp ? dir : std::string())),
          context_(context) {
        if (!draft_model.empty()) {
            m_.enable_mtp(draft_model);
            mtp_ = true;
        } else if (mtp) {
            if (!std::filesystem::exists(std::filesystem::path(dir) / "mtp.safetensors")) throw std::runtime_error("inkling2: --mtp needs mtp.safetensors in " + dir);
            m_.enable_mtp(dir);
            mtp_ = true;
        }
    }
    uint32_t step(uint32_t token, uint32_t pos, bool sample, float * capture) override {
        if (capture) throw std::runtime_error("inkling2: hidden-state capture is not supported");
        if (pos == 0) { m_.reset(); ctx_.clear(); keep_.clear(); }
        if (int(pos) != m_.length()) throw std::runtime_error("inkling2: step position does not follow the sequence");
        const int t = int(token);
        ctx_.push_back(token);
        return uint32_t(m_.forward_greedy(t));
    }
    void verify(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) override {
        if (int(pos) != m_.length()) throw std::runtime_error("inkling2: verify position does not follow the sequence");
        std::vector<int> t(tokens, tokens + n);
        m_.verify(t.data(), n, lg_);
        pend_.assign(tokens, tokens + n);
        for (int i = 0; i < n; ++i) next[i] = argmax(i);
    }
    void commit(int k) override {
        m_.commit(k);
        ctx_.insert(ctx_.end(), pend_.begin(), pend_.begin() + std::min<size_t>(size_t(k), pend_.size()));
        pend_.clear();
    }
    bool has_drafter() const override { return mtp_; }
    bool native_mtp() const override { return mtp_; }
    void set_ignore_eos(bool on) override { ignore_eos_ = on; }
    void set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float) override {
        temp_ = temperature > 0.f ? temperature : 0.f;
        top_p_ = top_p > 0.f && top_p <= 1.f ? top_p : 1.f;
        top_k_ = top_k;
        rng_.seed(seed);
    }
    bool incremental() const override { return true; }
    void spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int d) override {
        if (prompt.empty()) throw std::runtime_error("inkling2: empty prompt");
        if (prompt.size() + 8 >= context_) throw std::runtime_error("inkling2: prompt fills the context");
        out_.clear(); steps_ = drafted_ = accepted_ = 0; count_ = count;
        fixed_d_ = d > 0 ? std::min(d, 7) : 0;
        const auto t0 = std::chrono::steady_clock::now();
        // conversation reuse, exact: every prompt is prefilled in chunks aligned at multiples of the chunk from position 0,
        // and the state at the start of its final chunk is kept (snapshot). A prompt that extends the kept prefix goes back
        // to it and prefills from there in the same aligned chunks: the same results bitwise as a fresh prefill from 0.
        const size_t chunk = size_t(m_.chunk());
        size_t done = 0;
        if (!keep_.empty() && prompt.size() > keep_.size() && std::equal(keep_.begin(), keep_.end(), prompt.begin()) && m_.restore()) done = keep_.size();
        if (!done) m_.reset();
        reused_ = done;
        if (tqllm::debug_on("reuse"))
            std::fprintf(stderr, "inkling2: prompt %zu tokens, %zu reused (%s)\n", prompt.size(), done, done ? "snapshot" : "prefill from 0");
        std::vector<int> p(prompt.begin(), prompt.end());
        keep_.clear();
        for (size_t i = done; i < p.size(); i += chunk) {
            const int n = int(std::min<size_t>(chunk, p.size() - i));
            if (i + chunk >= p.size() && i > 0) { m_.snapshot(); keep_.assign(prompt.begin(), prompt.begin() + i); }  // the final chunk's start
            m_.forward(p.data() + i, n, 1, lg_);
        }
        ctx_ = prompt;
        m_.replan_experts();  // partial residency (16 GB cards): the prompt's most-routed experts onto the device
        t1_ = std::chrono::steady_clock::now();
        prefill_us_ = uint64_t(std::chrono::duration<double, std::micro>(t1_ - t0).count());
        tok_ = pick(0);
        pol_ = Inkling2SpecPolicy(fixed_d_);
        done_ = count == 0;
        if (!done_) emit(tok_);
    }
    bool spec_advance() override {
        if (done_) return false;
        const int room = int(context_) - m_.length() - 1;
        if (room <= 1) { done_ = true; return false; }
        // greedy with the drafter: the policy picks a spec step (L drafts verified at once) or a plain step, by measured speed
        const int L = !mtp_ || temp_ > 0.f ? 0 : pol_.next(room);
        const auto ts = std::chrono::steady_clock::now();
        if (L == 0) {  // plain step (sampled requests: no drafter)
            const int t = int(tok_);
            if (temp_ > 0.f) { m_.forward(&t, 1, 1, lg_); tok_ = pick(0); }
            else tok_ = uint32_t(m_.forward_greedy(t));
            if (mtp_ && temp_ <= 0.f) pol_.plain_done(std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
            ctx_.push_back(uint32_t(t));
            ++steps_;
            emit(tok_);
            return !done_;
        }
        std::vector<int> d, blk(size_t(L) + 1);
        m_.draft(int(tok_), d, L);
        blk[0] = int(tok_);
        for (int i = 0; i < L; ++i) blk[size_t(i) + 1] = d[size_t(i)];
        int nx[8];
        m_.verify_greedy(blk.data(), L + 1, nx);  // greedy: the verify rows' tokens without their logits
        int a = 0;
        while (a < L && nx[a] == d[size_t(a)]) ++a;
        m_.commit(a + 1);
        pol_.spec_done(a, L, std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
        ctx_.insert(ctx_.end(), blk.begin(), blk.begin() + a + 1);
        ++steps_; drafted_ += uint64_t(L); accepted_ += uint64_t(a);
        for (int i = 0; i < a && !done_; ++i) emit(uint32_t(d[size_t(i)]));
        tok_ = uint32_t(nx[a]);
        if (!done_) emit(tok_);
        return !done_;
    }
    std::vector<uint32_t> spec_finish(uint64_t * stats) override {
        if (stats) {
            stats[0] = steps_; stats[1] = drafted_; stats[2] = accepted_;
            stats[3] = uint64_t(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1_).count());
            stats[4] = prefill_us_;
        }
        return std::move(out_);
    }
    const std::vector<uint32_t> & spec_tokens(int) const override { return out_; }
    std::vector<uint32_t> generate_spec(const std::vector<uint32_t> & prompt, uint32_t count, int d, uint64_t * stats) override {
        spec_begin(prompt, count, d);
        while (spec_advance()) {}
        return spec_finish(stats);
    }
    void synchronize() override { cudaDeviceSynchronize(); }
    uint32_t layers() const override { return 48; }  // visits
    uint32_t hidden() const override { return 2304; }
    uint32_t vocab() const override { return uint32_t(m_.vocab()); }
    uint32_t eos() const override { return kEos; }
    uint32_t context() const override { return context_; }

private:
    uint32_t argmax(int row) const {
        const size_t V = size_t(m_.vocab());
        return uint32_t(std::max_element(lg_.begin() + row * V, lg_.begin() + (row + 1) * V) - (lg_.begin() + row * V));
    }
    // greedy, or temperature / top-k / top-p over the row's softcapped logits
    uint32_t pick(int row) {
        if (temp_ <= 0.f) return argmax(row);
        const size_t V = size_t(m_.vocab());
        const float * x = lg_.data() + row * V;
        const int k = top_k_ > 0 ? std::min<int>(top_k_, int(V)) : int(std::min<size_t>(V, 1024));
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return x[a] > x[b]; });
        std::vector<double> p(static_cast<size_t>(k));
        double z = 0;
        for (int i = 0; i < k; ++i) { p[size_t(i)] = std::exp(double(x[idx[size_t(i)]] - x[idx[0]]) / temp_); z += p[size_t(i)]; }
        int keep = k;
        double c = 0;
        for (int i = 0; i < k; ++i) { c += p[size_t(i)] / z; if (c >= top_p_) { keep = i + 1; break; } }
        std::discrete_distribution<int> dist(p.begin(), p.begin() + keep);
        return uint32_t(idx[size_t(dist(rng_))]);
    }
    void emit(uint32_t t) {
        out_.push_back(t);
        if ((t == kEos && !ignore_eos_) || out_.size() >= count_) done_ = true;
    }
    Inkling2Model m_;
    uint32_t context_;
    bool mtp_ = false, ignore_eos_ = false, done_ = true;
    float temp_ = 0.f, top_p_ = 1.f;
    int top_k_ = 0, fixed_d_ = 0;
    std::mt19937_64 rng_{0};
    std::vector<float> lg_;
    std::vector<uint32_t> out_;
    // conversation reuse: the tokens the model holds (prompt + forwarded tokens), the last prompt (its snapshot) and the
    // prompt tokens the last request did not prefill
    std::vector<uint32_t> ctx_, keep_, pend_;
    size_t reused_ = 0;
    uint32_t tok_ = 0, count_ = 0;
    uint64_t steps_ = 0, drafted_ = 0, accepted_ = 0, prefill_us_ = 0;
    Inkling2SpecPolicy pol_;
    std::chrono::steady_clock::time_point t1_;
};
}  // namespace

std::unique_ptr<Engine> make_inkling2_engine(const std::string & dir, uint32_t context, bool mtp, bool exact, const std::string & draft_model) {
    return std::make_unique<Inkling2Engine>(dir, context, mtp, exact, draft_model);
}

}  // namespace tqllm
