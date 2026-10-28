// tqllm::Model on the native Android/Arm CPU backend: the server's single-slot incremental path (spec_begin /
// spec_advance / spec_tokens / spec_finish) over android_backend::Model. Greedy requests use the adaptive Speculator
// (prompt-lookup proposals, plus a same-vocabulary draft model with -md); its output equals the target's own greedy
// decode. A request with temperature > 0 samples one token per step (top-k, top-p).
//
// The recurrent state only moves forward, so a new request continues the previous conversation when the tokens the
// model consumed are a prefix of the new prompt. Chat templates often render an earlier answer differently from how it
// was generated (Qwen drops the empty think block), so each prompt also leaves a state snapshot just before its last
// turn marker (<|im_start|>); a follow-up that shares the prompt up to there restores it and reads only the rest.
//
// Environment: TQLLM_ANDROID_THREADS (default 8, capped by the CPU affinity mask), TQLLM_ANDROID_DRAFT_DEPTH (6, 0
// disables speculation), TQLLM_ANDROID_DRAFT_HEAD (151936: the draft head's rows, plus EOS).
#include "tqllm/model.hpp"
#include "tqllm/tokenizer.hpp"
#include "android_model.hpp"
#include "android_speculative.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>

namespace tqllm {
namespace {
using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }
int env_int(const char * name, int fallback) {
    const char * v = std::getenv(name);
    return v && *v ? std::atoi(v) : fallback;
}
}  // namespace

struct Model::Impl {
    std::unique_ptr<android_backend::Model> target, draft;
    std::vector<uint32_t> history;  // every token the target consumed
    std::vector<uint32_t> snapshot;  // the tokens before the target's snapshot (empty: none)
    uint32_t turn = ~0u;  // <|im_start|>
    std::unique_ptr<android_backend::Speculator> speculator;
    android_backend::SpeculativeStats stats;
    float temperature = 0.f, top_p = 1.f; int top_k = 0; std::mt19937_64 rng{0};
    bool ignore_eos = false;
    // the active request
    std::vector<uint32_t> output;
    uint32_t count = 0, pending = 0;
    bool active = false, done = true, sampling = false;
    double prefill_us = 0;
    Clock::time_point decode_start;

    bool stops(uint32_t token) const { return token == target->eos() && !ignore_eos; }
    void restart() {
        target->reset();
        if (draft) draft->reset();
        history.clear(); snapshot.clear();
        speculator->restart();
    }
    bool starts(const std::vector<uint32_t> & prefix, const std::vector<uint32_t> & prompt) const {
        return !prefix.empty() && prefix.size() < prompt.size() && std::equal(prefix.begin(), prefix.end(), prompt.begin());
    }
    // Prompt tokens in chunks: one pass over the weights per chunk. Returns the last token's logits.
    const std::vector<float> & prefill(const std::vector<uint32_t> & tokens) {
        constexpr size_t chunk = 32;
        const std::vector<float> * last = nullptr;
        for (size_t i = 0; i < tokens.size(); i += chunk) {
            const std::vector<uint32_t> part(tokens.begin() + i, tokens.begin() + std::min(tokens.size(), i + chunk));
            last = &target->forward_batch(part, false).back();
        }
        history.insert(history.end(), tokens.begin(), tokens.end());
        return *last;
    }
    uint32_t pick(const std::vector<float> & logits) {
        if (!sampling) return android_backend::greedy(logits);
        // top-k (64 when unset), then top-p over the softmax of logits / temperature
        const int k = std::min<int>(top_k > 0 ? top_k : 64, int(logits.size()));
        std::vector<uint32_t> ids(logits.size());
        std::iota(ids.begin(), ids.end(), 0u);
        std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
        ids.resize(size_t(k));
        std::vector<double> p(static_cast<size_t>(k));
        double total = 0;
        for (int i = 0; i < k; ++i) total += p[size_t(i)] = std::exp(double(logits[ids[size_t(i)]] - logits[ids[0]]) / temperature);
        double kept = 0;
        int n = 0;
        while (n < k && (n == 0 || kept < top_p * total)) kept += p[size_t(n++)];
        std::uniform_real_distribution<double> u(0.0, kept);
        double r = u(rng);
        for (int i = 0; i < n; ++i) if ((r -= p[size_t(i)]) <= 0) return ids[size_t(i)];
        return ids[size_t(n - 1)];
    }
};

Model::Model(const std::string & weights, const std::string & tokenizer, uint32_t context, int slots, const EngineOptions & options)
    : impl_(std::make_unique<Impl>()) {
    if (slots != 1) throw std::invalid_argument("the Android backend serves one request at a time (--np 1)");
    auto & i = *impl_;
    android_backend::Options o;
    o.context = int(context);
    o.threads = std::clamp(env_int("TQLLM_ANDROID_THREADS", 8), 1, 64);
    i.target = std::make_unique<android_backend::Model>(weights, o);
    if (!options.draft_model.empty()) {
        auto d = o;
        d.head_rows = env_int("TQLLM_ANDROID_DRAFT_HEAD", 151936);
        i.draft = std::make_unique<android_backend::Model>(options.draft_model, d);
        if (i.draft->vocabulary() != i.target->vocabulary() || i.draft->eos() != i.target->eos())
            throw std::invalid_argument("draft model vocabulary differs from the target");
    }
    Tokenizer tok;
    tok.load(tokenizer.empty() ? weights : tokenizer);
    i.turn = tok.special_id("<|im_start|>");
    i.speculator = std::make_unique<android_backend::Speculator>(*i.target, i.draft.get(),
                                                                 std::max(0, env_int("TQLLM_ANDROID_DRAFT_DEPTH", 6)), i.history);
    std::fprintf(stderr, "tqserve: Android CPU backend, %d threads, %s\n", o.threads,
                 i.draft ? ("draft " + options.draft_model).c_str() : "prompt-lookup speculation");
}
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;

int Model::slots() const { return 1; }
uint32_t Model::context() const { return uint32_t(impl_->target->context()); }
bool Model::kv_idle_pending() const { return false; }
bool Model::kv_idle_step() { return false; }
bool Model::has_drafter() const { return impl_->draft != nullptr; }
bool Model::native_mtp() const { return false; }
std::string Model::fit_note() const { return {}; }
bool Model::incremental() const { return true; }
void Model::select_slot(int slot) { if (slot != 0) throw std::invalid_argument("the Android backend has one slot"); }
void Model::set_prompt_images(const std::vector<PromptImage> & images) {
    if (!images.empty()) throw std::invalid_argument("image input is not supported by the Android backend");
}
void Model::set_ignore_eos(bool on) { impl_->ignore_eos = on; }
void Model::set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float) {
    auto & i = *impl_;
    i.temperature = temperature; i.top_p = top_p; i.top_k = top_k; i.rng.seed(seed);
}
uint32_t Model::vocab_size() const { return uint32_t(impl_->target->vocabulary()); }
void Model::synchronize() {}

void Model::spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int) {
    auto & i = *impl_;
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    if (prompt.size() + count + 1 > uint32_t(i.target->context())) throw std::invalid_argument("prompt and answer exceed the context");
    const auto t0 = Clock::now();
    i.output.clear(); i.count = count; i.stats = {};
    i.sampling = i.temperature > 0.f;
    // Continue the previous conversation when the new prompt extends what the model consumed, else from the snapshot.
    const char * how = "continued";
    if (!i.starts(i.history, prompt)) {
        if (i.starts(i.snapshot, prompt) && i.target->restore() == int(i.snapshot.size())) {
            i.history = i.snapshot;
            i.speculator->resync_draft();
            how = "restored the snapshot";
        } else {
            i.restart();
            how = "started fresh";
        }
    }
    const size_t from = i.history.size();
    size_t mark = prompt.size();
    while (mark > from && prompt[mark - 1] != i.turn) --mark;
    if (mark > from) {  // snapshot before the last turn marker
        --mark;
        if (mark > from) i.prefill(std::vector<uint32_t>(prompt.begin() + from, prompt.begin() + mark));
        i.target->snapshot();
        i.snapshot = i.history;
    }
    const auto & logits = i.prefill(std::vector<uint32_t>(prompt.begin() + i.history.size(), prompt.end()));
    i.pending = i.pick(logits);
    i.output.push_back(i.pending);
    i.prefill_us = us_since(t0);
    i.decode_start = Clock::now();
    i.active = true;
    i.done = i.output.size() >= i.count || i.stops(i.pending);
    std::fprintf(stderr, "tqserve: prompt %zu tokens, %s, read %zu\n", prompt.size(), how, prompt.size() - from);
}

bool Model::spec_advance() {
    auto & i = *impl_;
    if (!i.active || i.done) return false;
    const int want = int(i.count) - int(i.output.size());
    if (want < 1) { i.done = true; return false; }
    if (i.sampling) {
        if (i.target->position() + 1 >= i.target->context()) { i.done = true; return false; }
        const auto & logits = i.target->forward(i.pending);
        i.history.push_back(i.pending);
        ++i.stats.verifies;
        i.pending = i.pick(logits);
        i.output.push_back(i.pending);
    } else {
        std::vector<uint32_t> accepted;
        if (!i.speculator->step(i.pending, want, accepted, i.stats)) { i.done = true; return false; }
        for (uint32_t token : accepted) {
            i.output.push_back(token);
            if (i.stops(token)) { i.done = true; return false; }
        }
        i.output.push_back(i.pending);
    }
    i.done = i.output.size() >= i.count || i.stops(i.output.back());
    return !i.done;
}

const std::vector<uint32_t> & Model::spec_tokens(int) const { return impl_->output; }

std::vector<uint32_t> Model::spec_finish(uint64_t * stats) {
    auto & i = *impl_;
    if (stats) {
        stats[0] = uint64_t(i.stats.verifies);
        stats[1] = uint64_t(i.stats.drafted + i.stats.text_drafted);
        stats[2] = uint64_t(i.stats.accepted + i.stats.text_accepted);
        stats[3] = uint64_t(us_since(i.decode_start));
        stats[4] = uint64_t(i.prefill_us);
    }
    i.active = false;
    return i.output;  // EOS stays: finish() turns it into finish_reason "stop"
}

void Model::spec_step(const std::vector<int> &, std::vector<char> &) {
    throw std::invalid_argument("the Android backend has no batched multi-slot step (--np 1)");
}
uint32_t Model::forward(uint32_t, uint32_t, bool) {
    throw std::invalid_argument("the Android backend decodes through spec_begin / spec_advance");
}
bool Model::prefill_state(const std::vector<uint32_t> &, uint32_t) { return false; }
}  // namespace tqllm
