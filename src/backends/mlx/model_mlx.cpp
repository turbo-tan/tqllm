// tqllm::Model on the MLX backend (Apple Silicon): the server's single-slot speculative path (spec_begin / spec_advance /
// spec_tokens / spec_finish) over Qwen35 + the DFlash2 drafter, with the settings of the measured exact configuration as
// defaults (an environment variable that is already set wins). Greedy is exact (output == the target's own greedy
// decode); a request with temperature > 0 uses speculative sampling with the same drafter (exact distribution).
#include "tqllm/model.hpp"
#include "tqllm/tokenizer.hpp"
#include "dflash.hpp"
#include "qwen35.hpp"
#include "wired_limit.hpp"
#include "gguf.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <stdexcept>

namespace mx = mlx::core;
using tqllm::mlx_backend::DFlash2;
using tqllm::mlx_backend::verify_width;
using tqllm::mlx_backend::Qwen35;

namespace tqllm {
namespace {
using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t) {return std::chrono::duration<double, std::micro>(Clock::now() - t).count();}
mx::array rows(const mx::array& x, int a, int b) {return mx::slice(x, {a, 0}, {b, x.shape(1)});}
void default_env(const char* name, const char* value) {setenv(name, value, 0);}
bool has_tensor(const std::string& path, const std::string& name) {
    return gguf_open(path).tensors.count(name) > 0;
}
}  // namespace

struct ScopedWiredLimitHolder { tqllm::mlx_backend::ScopedWiredLimit limit; };
struct Model::Impl {
    std::unique_ptr<ScopedWiredLimitHolder> residency;
    std::unique_ptr<Qwen35> engine;
    std::unique_ptr<DFlash2> drafter;
    uint32_t context_ = 0, eos = 0, vocab = 0, layers_ = 0, hidden_ = 0;
    int width = 15, chunk = 2048, shortlist = 32768;
    bool head4 = false;
    float temperature = 0.f, top_p = 1.f; int top_k = 0; std::mt19937_64 rng{0};
    // the active request
    std::vector<uint32_t> output;
    uint32_t count = 0, pending = 0;
    bool active = false, done = true, sampling = false;
    uint64_t steps = 0, drafted = 0, accepted = 0;
    double prefill_us = 0, decode_us = 0;
    Clock::time_point decode_start;
};
}  // namespace tqllm

namespace tqllm {

Model::Model(const std::string& weights, const std::string& tokenizer, uint32_t context, int slots, const EngineOptions& options)
    : impl_(std::make_unique<Impl>()) {
    if (slots != 1) throw std::invalid_argument("the MLX backend serves one request at a time (--np 1)");
    if (options.draft_model.empty()) throw std::invalid_argument("the MLX backend needs the DFlash2 drafter (-md DRAFTER.gguf)");
    // the measured exact configuration (j690-j694): scan GDN with the compact prefix tape, batch-invariant decode-rows
    // attention, serial dense rows, device-side draft selection, compiled elementwise epilogues, 16-row drafts
    default_env("MLX_ENABLE_TF32", "0");
    for (const char* name : {"TQLLM_MLX_GDN_SCAN", "TQLLM_MLX_SERIAL_VERIFY_ATTENTION", "TQLLM_MLX_SERIAL_VERIFY_DENSE",
                             "TQLLM_MLX_COMPACT_GDN_TAPE", "TQLLM_MLX_DEVICE_SELECTOR", "TQLLM_MLX_BATCH_DECODE_DENSE",
                             "TQLLM_MLX_ATTN_DECODE_ROWS", "TQLLM_MLX_DRAFT_HEAD_UNPAD", "TQLLM_MLX_FUSE_EW"})
        default_env(name, "1");
    default_env("TQLLM_MLX_DFLASH_BLOCK", "16");
    auto& i = *impl_;
    if (const char* c = std::getenv("TQLLM_MLX_PREFILL_CHUNK")) i.chunk = std::max(128, std::atoi(c));
    i.width = std::atoi(std::getenv("TQLLM_MLX_DFLASH_BLOCK")) >= 16 ? 15 : 7;
    // the 4-bit proposal head exists only in files that carry output_draft.weight (TQ4_L); elsewhere a shortlist head
    i.head4 = has_tensor(weights, "output_draft.weight");
    mx::set_cache_limit(size_t(512) * 1024 * 1024);
    i.residency = std::make_unique<ScopedWiredLimitHolder>();
    i.engine = std::make_unique<Qwen35>(weights, int(context));
    i.engine->gdn_mode(Qwen35::GdnMode::scan);
    i.engine->verify_tiles(true);
    i.drafter = std::make_unique<DFlash2>(options.draft_model, *i.engine);
    if (i.head4) i.engine->enable_draft_head4();
    tqllm::Tokenizer tok; tok.load(tokenizer.empty() ? weights : tokenizer);
    i.eos = tok.eos_id();
    i.context_ = context; i.vocab = uint32_t(i.engine->vocabulary());
    std::fprintf(stderr, "tqserve: MLX backend, drafter %s, block %s, %s head\n", options.draft_model.c_str(),
                 std::getenv("TQLLM_MLX_DFLASH_BLOCK"), i.head4 ? "4-bit proposal" : "shortlist");
}
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

int Model::slots() const {return 1;}
uint32_t Model::context() const {return impl_->context_;}
bool Model::kv_idle_pending() const {return false;}
bool Model::kv_idle_step() {return false;}
bool Model::has_drafter() const {return true;}
bool Model::native_mtp() const {return false;}
std::string Model::fit_note() const {return {};}
bool Model::incremental() const {return true;}
void Model::select_slot(int slot) {if (slot != 0) throw std::invalid_argument("the MLX backend has one slot");}
void Model::set_prompt_images(const std::vector<PromptImage>& images) {
    if (!images.empty()) throw std::invalid_argument("image input is not supported by the MLX backend yet");
}
void Model::set_sampling(float temperature, float top_p, int top_k, uint64_t seed) {
    auto& i = *impl_;
    i.temperature = temperature; i.top_p = top_p; i.top_k = top_k; i.rng.seed(seed);
}
uint32_t Model::vocab_size() const {return impl_->vocab;}
void Model::synchronize() {mx::synchronize();}

void Model::spec_begin(const std::vector<uint32_t>& prompt, uint32_t count, int) {
    auto& i = *impl_;
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    if (prompt.size() + count + 16 > i.context_) throw std::invalid_argument("prompt and answer exceed the context");
    i.output.clear(); i.count = count; i.steps = i.drafted = i.accepted = 0; i.decode_us = 0;
    i.sampling = i.temperature > 0.f;
    const auto t0 = Clock::now();
    i.engine->reset(); i.drafter->reset();
    mx::array logits = mx::array(0.0f);
    for (size_t off = 0; off < prompt.size(); off += size_t(i.chunk)) {
        const int n = std::min(i.chunk, int(prompt.size() - off));
        logits = i.engine->forward(std::span<const uint32_t>(prompt.data() + off, size_t(n)));
        i.drafter->inject(i.engine->feature_states(), int(off));
    }
    if (!i.head4) i.engine->draft_shortlist(logits, i.shortlist);
    i.pending = mx::argmax(logits).item<uint32_t>();
    if (i.sampling) {
        // sample the first token from the last prompt position's distribution
        auto scaled = mx::multiply(mx::astype(mx::reshape(logits, {-1}), mx::float32), mx::array(1.0f / i.temperature));
        mx::eval(scaled);
        const float* p = scaled.data<float>(); const size_t v = scaled.size();
        std::vector<std::pair<float, uint32_t>> top;
        const int k = i.top_k > 0 ? i.top_k : 64;
        for (size_t t = 0; t < v; ++t) {
            if (int(top.size()) < k) {top.push_back({p[t], uint32_t(t)}); std::push_heap(top.begin(), top.end(), std::greater<>());}
            else if (p[t] > top.front().first) {std::pop_heap(top.begin(), top.end(), std::greater<>()); top.back() = {p[t], uint32_t(t)}; std::push_heap(top.begin(), top.end(), std::greater<>());}
        }
        std::sort(top.rbegin(), top.rend());
        double z = 0; std::vector<double> w(top.size());
        for (size_t t = 0; t < top.size(); ++t) {w[t] = std::exp(double(top[t].first - top[0].first)); z += w[t];}
        double mass = 0; size_t keep = top.size();
        for (size_t t = 0; t < top.size(); ++t) {mass += w[t] / z; if (mass >= i.top_p) {keep = t + 1; break;}}
        std::discrete_distribution<size_t> pick(w.begin(), w.begin() + long(keep));
        i.pending = top[pick(i.rng)].second;
    }
    i.output.push_back(i.pending);
    i.prefill_us = us_since(t0);
    i.decode_start = Clock::now();
    i.active = true; i.done = i.output.size() >= i.count || i.pending == i.eos;
}

bool Model::spec_advance() {
    auto& i = *impl_;
    if (!i.active || i.done) return false;
    const uint32_t limit = i.count;
    static const bool sample_ar = std::getenv("TQLLM_MLX_SAMPLE_AR") != nullptr;  // debug: token-by-token sampling
    if (i.sampling && sample_ar) {
        auto logits = i.engine->forward({&i.pending, 1});
        auto scaled = mx::multiply(mx::astype(mx::reshape(logits, {-1}), mx::float32), mx::array(1.0f / i.temperature));
        mx::eval(scaled);
        const float* p = scaled.data<float>(); const size_t v = scaled.size();
        std::vector<std::pair<float, uint32_t>> top;
        const int k = i.top_k > 0 ? i.top_k : 64;
        for (size_t t = 0; t < v; ++t) {
            if (int(top.size()) < k) {top.push_back({p[t], uint32_t(t)}); std::push_heap(top.begin(), top.end(), std::greater<>());}
            else if (p[t] > top.front().first) {std::pop_heap(top.begin(), top.end(), std::greater<>()); top.back() = {p[t], uint32_t(t)}; std::push_heap(top.begin(), top.end(), std::greater<>());}
        }
        std::sort(top.rbegin(), top.rend());
        double z = 0; std::vector<double> w(top.size());
        for (size_t t = 0; t < top.size(); ++t) {w[t] = std::exp(double(top[t].first - top[0].first)); z += w[t];}
        double mass = 0; size_t keep = top.size();
        for (size_t t = 0; t < top.size(); ++t) {mass += w[t] / z; if (mass >= i.top_p) {keep = t + 1; break;}}
        std::discrete_distribution<size_t> pick(w.begin(), w.begin() + long(keep));
        i.pending = top[pick(i.rng)].second;
        i.output.push_back(i.pending); ++i.steps;
        i.done = i.output.size() >= limit || i.pending == i.eos;
        return !i.done;
    }
    const int pos = i.engine->position(), want = std::min(verify_width(i.width, pos), int(limit - i.output.size()));
    if (want < 1) {i.done = true; return false;}
    // draft -> verify as one graph (the draft ids stay on the GPU), one host sync per step
    auto drafted = i.drafter->draft_device(i.pending, pos, want);
    mx::async_eval(drafted);
    const uint32_t anchor[1] = {i.pending};
    auto block = mx::concatenate({mx::array(anchor, {1}, mx::uint32), drafted}, 0);
    Qwen35::Verification ticket;
    auto verified = i.engine->verify_device(block, ticket);
    std::vector<uint32_t> sampled_ids;
    const uint32_t* ids = nullptr;
    mx::array choices = mx::array(0u);
    int acc = 0;
    const uint32_t* proposed = nullptr;
    if (!i.sampling) {
        choices = mx::argmax(verified, -1);
        mx::eval(choices, drafted);
        ids = choices.data<uint32_t>();
        proposed = drafted.data<uint32_t>();
        while (acc < want && proposed[acc] == ids[acc]) ++acc;
    } else {
        // speculative sampling with a point-mass draft: accept proposed[j] with probability p_j(x); on rejection sample
        // from p_j without x (renormalized) and stop; after a full accept sample the bonus token from the last row.
        // The output distribution equals token-by-token sampling from the same filtered p (top-k, then top-p).
        const int rows = want + 1, k = i.top_k > 0 ? i.top_k : 64;
        auto scaled = mx::multiply(mx::astype(mx::reshape(verified, {rows, -1}), mx::float32), mx::array(1.0f / i.temperature));
        auto idx = mx::slice(mx::argpartition(mx::negative(scaled), k - 1, -1), {0, 0}, {rows, k});
        auto val = mx::take_along_axis(scaled, idx, -1);
        idx = mx::astype(idx, mx::uint32);
        mx::eval(idx, val, drafted);
        proposed = drafted.data<uint32_t>();
        const auto* pi = idx.data<uint32_t>(); const auto* pv = val.data<float>();
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        sampled_ids.assign(size_t(rows), 0u);
        auto row_dist = [&](int r, std::vector<std::pair<double, uint32_t>>& d) {
            d.clear();
            for (int t = 0; t < k; ++t) d.push_back({double(pv[r * k + t]), pi[r * k + t]});
            std::sort(d.begin(), d.end(), [](auto& x, auto& y) {return x.first > y.first;});
            const double peak = d[0].first;  // read once: the loop overwrites d[0]
            double z = 0; for (auto& e : d) {e.first = std::exp(e.first - peak); z += e.first;}
            double mass = 0; size_t keep = d.size();
            for (size_t t = 0; t < d.size(); ++t) {mass += d[t].first / z; if (mass >= i.top_p) {keep = t + 1; break;}}
            d.resize(keep); z = 0; for (auto& e : d) z += e.first; for (auto& e : d) e.first /= z;
        };
        auto draw = [&](const std::vector<std::pair<double, uint32_t>>& d, uint32_t skip) {
            double z = 0; for (auto& e : d) if (e.second != skip) z += e.first;
            double u = unit(i.rng) * z, c = 0; uint32_t last = d[0].second;
            for (auto& e : d) {if (e.second == skip) continue; last = e.second; c += e.first; if (u < c) return e.second;}
            return last;
        };
        std::vector<std::pair<double, uint32_t>> d;
        for (; acc < want; ++acc) {
            row_dist(acc, d);
            double px = 0; for (auto& e : d) if (e.second == proposed[acc]) px = e.first;
            if (unit(i.rng) < px) {sampled_ids[size_t(acc)] = proposed[acc]; continue;}
            sampled_ids[size_t(acc)] = d.size() == 1 && d[0].second == proposed[acc] ? proposed[acc] : draw(d, proposed[acc]);
            break;
        }
        if (acc == want) {row_dist(want, d); sampled_ids[size_t(want)] = draw(d, UINT32_MAX);}
        static const bool debug = std::getenv("TQLLM_MLX_SAMPLE_DEBUG") != nullptr;
        if (debug && i.steps < 3) {
            auto arg = mx::argmax(verified, -1); mx::eval(arg);
            const auto* am = arg.data<uint32_t>();
            for (int r = 0; r <= std::min(want, 3); ++r) {
                row_dist(r, d);
                std::fprintf(stderr, "SAMPLE_DEBUG step %llu row %d argmax %u proposed %u top:", (unsigned long long)i.steps, r, am[r], r < want ? proposed[r] : 0u);
                for (size_t t = 0; t < std::min<size_t>(d.size(), 3); ++t) std::fprintf(stderr, " %u:%.3f", d[t].second, d[t].first);
                std::fprintf(stderr, "\n");
            }
            std::fprintf(stderr, "SAMPLE_DEBUG verified shape %d x %d, idx %d x %d, acc %d\n", int(verified.shape(0)), verified.ndim() > 1 ? int(verified.shape(1)) : -1, int(idx.shape(0)), int(idx.shape(1)), acc);
        }
        ids = sampled_ids.data();
    }
    auto features = i.engine->feature_states();
    ++i.steps; i.drafted += uint64_t(want); i.accepted += uint64_t(acc);
    for (int j = 0; j < acc && i.output.size() < limit; ++j) {
        i.output.push_back(proposed[j]);
        if (proposed[j] == i.eos) {i.done = true; break;}
    }
    if (!i.done && i.output.size() < limit) i.output.push_back(ids[acc]);
    i.pending = ids[acc];
    i.engine->accept_prefix(ticket, size_t(acc + 1));
    i.drafter->inject(rows(features, 0, acc + 1), pos);
    if (!i.done) i.done = i.output.size() >= limit || i.output.back() == i.eos;
    return !i.done;
}

std::vector<uint32_t> Model::spec_tokens(int) const {return impl_->output;}

std::vector<uint32_t> Model::spec_finish(uint64_t* stats) {
    auto& i = *impl_;
    i.decode_us = us_since(i.decode_start);
    if (stats) {
        stats[0] = i.steps; stats[1] = i.drafted; stats[2] = i.accepted;
        stats[3] = uint64_t(i.decode_us); stats[4] = uint64_t(i.prefill_us);
    }
    i.active = false;
    return i.output;  // EOS stays: finish() turns it into finish_reason "stop"
}

void Model::spec_step(const std::vector<int>&, std::vector<char>&) {
    throw std::invalid_argument("the MLX backend has no batched multi-slot step (--np 1)");
}
uint32_t Model::forward(uint32_t, uint32_t, bool) {
    throw std::invalid_argument("the MLX backend decodes through spec_begin / spec_advance");
}
}  // namespace tqllm
