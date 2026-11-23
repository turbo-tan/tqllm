// Inkling-2-Mini behind the serving Engine interface (tqserve, tqllm::Model): one slot, chunked prefill, greedy decode
// with the MTP drafter (--mtp: mtp.safetensors in the folder) (spec == plain, src/models/inkling2/loop/inkling2_model.hpp),
// sampled decode (temperature / top-k / top-p on the host from the full logits) without the drafter.
#include "inkling2_model.hpp"

#include "../../../engine/engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
    Inkling2Engine(const std::string & dir, uint32_t context, bool mtp, bool exact) : m_(dir, int(context), prompt_chunk(), false, SIZE_MAX, exact), context_(context) {
        if (mtp) {
            if (!std::filesystem::exists(std::filesystem::path(dir) / "mtp.safetensors")) throw std::runtime_error("inkling2: --mtp needs mtp.safetensors in " + dir);
            m_.enable_mtp(dir);
            mtp_ = true;
        }
    }
    uint32_t step(uint32_t token, uint32_t pos, bool sample, float * capture) override {
        if (capture) throw std::runtime_error("inkling2: hidden-state capture is not supported");
        if (pos == 0) m_.reset();
        if (int(pos) != m_.length()) throw std::runtime_error("inkling2: step position does not follow the sequence");
        const int t = int(token);
        m_.forward(&t, 1, 1, lg_);
        return sample ? argmax(0) : 0;
    }
    void verify(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) override {
        if (int(pos) != m_.length()) throw std::runtime_error("inkling2: verify position does not follow the sequence");
        std::vector<int> t(tokens, tokens + n);
        m_.verify(t.data(), n, lg_);
        for (int i = 0; i < n; ++i) next[i] = argmax(i);
    }
    void commit(int k) override { m_.commit(k); }
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
        m_.reset();
        std::vector<int> p(prompt.begin(), prompt.end());
        const size_t chunk = size_t(m_.chunk());
        for (size_t i = 0; i < p.size(); i += chunk) {
            const int n = int(std::min<size_t>(chunk, p.size() - i));
            m_.forward(p.data() + i, n, 1, lg_);
        }
        t1_ = std::chrono::steady_clock::now();
        prefill_us_ = uint64_t(std::chrono::duration<double, std::micro>(t1_ - t0).count());
        tok_ = pick(0);
        ema_ = 3.0;
        done_ = count == 0;
        if (!done_) emit(tok_);
    }
    bool spec_advance() override {
        if (done_) return false;
        const int room = int(context_) - m_.length() - 1;
        if (room <= 1) { done_ = true; return false; }
        if (!mtp_ || temp_ > 0.f) {  // plain step (sampled requests: no drafter)
            const int t = int(tok_);
            m_.forward(&t, 1, 1, lg_);
            tok_ = pick(0);
            ++steps_;
            emit(tok_);
            return !done_;
        }
        // greedy spec step: L drafts (round(EMA of tokens gained) + 1, 2 .. 7, or the request's fixed d) verified at once
        int L = fixed_d_ ? fixed_d_ : std::max(2, std::min(7, int(ema_ + 0.5) + 1));
        L = std::min(L, room - 1);
        std::vector<int> d, blk(size_t(L) + 1);
        m_.draft(int(tok_), d, L);
        blk[0] = int(tok_);
        for (int i = 0; i < L; ++i) blk[size_t(i) + 1] = d[size_t(i)];
        m_.verify(blk.data(), L + 1, lg_);
        int a = 0;
        while (a < L && argmax(a) == uint32_t(d[size_t(a)])) ++a;
        ema_ = 0.7 * ema_ + 0.3 * (a == L ? a + 1.0 : double(a));
        m_.commit(a + 1);
        ++steps_; drafted_ += uint64_t(L); accepted_ += uint64_t(a);
        for (int i = 0; i < a && !done_; ++i) emit(uint32_t(d[size_t(i)]));
        tok_ = argmax(a);
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
    uint32_t tok_ = 0, count_ = 0;
    uint64_t steps_ = 0, drafted_ = 0, accepted_ = 0, prefill_us_ = 0;
    double ema_ = 3.0;
    std::chrono::steady_clock::time_point t1_;
};
}  // namespace

std::unique_ptr<Engine> make_inkling2_engine(const std::string & dir, uint32_t context, bool mtp, bool exact) {
    return std::make_unique<Inkling2Engine>(dir, context, mtp, exact);
}

}  // namespace tqllm
