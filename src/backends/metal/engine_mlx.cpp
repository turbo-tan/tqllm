// The Metal engine (Apple Silicon, MLX), registered in builtin_mlx.cpp behind the shared tqllm::Model: the single-slot
// speculative path (spec_begin / spec_advance /
// spec_tokens / spec_finish) over Qwen35 + the DFlash2 drafter, with the settings of the measured exact configuration as
// defaults (an environment variable that is already set wins). Greedy is exact (output == the target's own greedy
// decode); a request with temperature > 0 uses speculative sampling with the same drafter (exact distribution).
#include "debug.hpp"
#include "release_defaults.hpp"
#include "engine_mlx.hpp"
#include "tqllm/tokenizer.hpp"
#include "dflash.hpp"
#include "sampling.hpp"
#include "sampling_topk.hpp"
#include "qwen35.hpp"
#include "kv_tcq.hpp"
#include "spec_pipeline.hpp"
#include "width_policy.hpp"
#include "wired_limit.hpp"
#include "gguf.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <stdexcept>

namespace mx = mlx::core;
using tqllm::mlx_backend::DFlash2;
using tqllm::mlx_backend::filtered_distribution;
using tqllm::mlx_backend::speculative_pick;
using tqllm::mlx_backend::draw_excluding;
using tqllm::mlx_backend::row_topk;
using tqllm::mlx_backend::Qwen35;

namespace tqllm {
namespace {
using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t) {return std::chrono::duration<double, std::micro>(Clock::now() - t).count();}
mx::array rows(const mx::array& x, int a, int b) {return mx::slice(x, {a, 0}, {b, x.shape(1)});}
}  // namespace

struct ScopedWiredLimitHolder { tqllm::mlx_backend::ScopedWiredLimit limit; };
class MlxQwen35Engine final : public Engine {
public:
    MlxQwen35Engine(const std::string& weights, const std::string& tokenizer, uint32_t context, int slots, const EngineOptions& options);
    ~MlxQwen35Engine() override;
    int slots() const override;
    uint32_t context() const override;
    bool kv_idle_pending() const override;
    bool kv_idle_step() override;
    bool has_drafter() const override;
    bool native_mtp() const override;
    bool incremental() const override;
    bool prefill_state(const std::vector<uint32_t>&, uint32_t) override;
    void select_slot(int slot) override;
    void set_prompt_images(const std::vector<PromptImage>& images) override;
    void set_ignore_eos(bool) override;
    void set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence) override;
    void synchronize() override;
    uint32_t layers() const override;
    uint32_t hidden() const override;
    uint32_t vocab() const override;
    uint32_t eos() const override;
    void spec_begin(const std::vector<uint32_t>& prompt, uint32_t count, int) override;
    bool spec_advance() override;
    const std::vector<uint32_t>& spec_tokens(int) const override;
    std::vector<uint32_t> spec_finish(uint64_t* stats) override;
    void spec_step(const int*, int, bool*) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
struct MlxQwen35Engine::Impl {
    std::unique_ptr<ScopedWiredLimitHolder> residency;
    std::unique_ptr<Qwen35> engine;
    std::unique_ptr<DFlash2> drafter;
    // greedy requests: verify(k) and draft(k+1) chained on the GPU (spec_pipeline.hpp, gated j892)
    std::optional<tqllm::mlx_backend::DraftAhead> ahead;
    tqllm::mlx_backend::WidthPolicy widths;  // verify 7 or 15 drafted tokens per step (time lines kept across requests)
    uint32_t context_ = 0, eos = 0, vocab = 0, layers_ = 0, hidden_ = 0;
    int width = 15, chunk = 8192;  // block-16 drafts; prefill batches of 8192 run layer-major in 2048-row slices
    bool exact = true;  // --exact on|off
    float temperature = 0.f, top_p = 1.f; int top_k = 0; std::mt19937_64 rng{0};
    // the active request
    std::vector<uint32_t> output;
    uint32_t count = 0, pending = 0;
    bool active = false, done = true, sampling = false;
    uint64_t steps = 0, drafted = 0, accepted = 0;
    double prefill_us = 0, decode_us = 0;
    // host phases summed over the request's draft-ahead steps (ms); gap = host time between one spec_advance's end
    // and the next one's start (the server loop: publish, repeat guard, scheduling)
    double h_build = 0, h_submit = 0, h_wait = 0, h_post = 0, h_gap = 0;
    std::optional<Clock::time_point> last_advance_end;
    std::vector<double> first_steps; double first_gap_ms = 0;  // first 4 step walls of a request + prefill end -> first step
    std::array<double, 4> first_phases{};  // the first step's host phases: build / submit / wait / post
    Clock::time_point decode_start;
};
}  // namespace tqllm

namespace tqllm {

MlxQwen35Engine::MlxQwen35Engine(const std::string& weights, const std::string& tokenizer, uint32_t context, int slots, const EngineOptions& options)
    : impl_(std::make_unique<Impl>()) {
    if (slots != 1) throw std::invalid_argument("the MLX backend serves one request at a time (--np 1)");
    if (options.draft_model.empty()) throw std::invalid_argument("the MLX backend needs the DFlash2 drafter (-md DRAFTER.gguf)");
    mlx_backend::apply_release_defaults();
    auto& i = *impl_;
    if (const char* c = std::getenv("TQLLM_MLX_PREFILL_CHUNK")) i.chunk = std::max(128, std::atoi(c));
    mx::set_cache_limit(size_t(512) * 1024 * 1024);
    i.residency = std::make_unique<ScopedWiredLimitHolder>();
    i.engine = std::make_unique<Qwen35>(weights, int(context));
    i.drafter = std::make_unique<DFlash2>(options.draft_model, *i.engine);
    i.engine->enable_draft_head4();  // 4-bit proposal head, matrix-unit layout (gated j892)
    i.exact = options.exact;
    i.engine->set_exact(i.exact);
    mlx_backend::warm_compressed_kv_kernels(i.engine->q_heads(), i.engine->kv_heads());
    tqllm::Tokenizer tok; tok.load(tokenizer.empty() ? weights : tokenizer);
    i.eos = tok.eos_id();
    i.context_ = context; i.vocab = uint32_t(i.engine->vocabulary());
    std::fprintf(stderr, "tqserve: MLX backend, drafter %s, block 16, 4-bit proposal head, exact %s\n", options.draft_model.c_str(),
                 options.exact ? "on" : "off");
}

MlxQwen35Engine::~MlxQwen35Engine() = default;  // Impl is complete here
int MlxQwen35Engine::slots() const {return 1;}
uint32_t MlxQwen35Engine::context() const {return impl_->context_;}
bool MlxQwen35Engine::kv_idle_pending() const {return false;}
bool MlxQwen35Engine::kv_idle_step() {return false;}
bool MlxQwen35Engine::has_drafter() const {return true;}
bool MlxQwen35Engine::native_mtp() const {return false;}
bool MlxQwen35Engine::incremental() const {return true;}
bool MlxQwen35Engine::prefill_state(const std::vector<uint32_t>&, uint32_t) {return false;}  // no chunked plain-decode prompt path: token by token
void MlxQwen35Engine::select_slot(int slot) {if (slot != 0) throw std::invalid_argument("the MLX backend has one slot");}
void MlxQwen35Engine::set_prompt_images(const std::vector<PromptImage>& images) {
    if (!images.empty()) throw std::invalid_argument("image input is not supported by the MLX backend yet");
}
void MlxQwen35Engine::set_ignore_eos(bool) {}  // the MLX decode loop stops at its own end tokens
void MlxQwen35Engine::set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence) {
    (void) presence;  // the MLX sampler has no presence penalty yet
    auto& i = *impl_;
    i.temperature = temperature; i.top_p = top_p; i.top_k = top_k; i.rng.seed(seed);
}
uint32_t MlxQwen35Engine::vocab() const {return impl_->vocab;}
uint32_t MlxQwen35Engine::eos() const {return impl_->eos;}
uint32_t MlxQwen35Engine::layers() const {return impl_->layers_;}
uint32_t MlxQwen35Engine::hidden() const {return impl_->hidden_;}
void MlxQwen35Engine::synchronize() {mx::synchronize();}

// TQLLM_DEBUG=memlog: MLX memory in use / peak since the request began / buffer cache at each stage of a request
static void memlog_point(const char* stage, long a = -1) {
    static const bool on = tqllm::debug_on("memlog");
    if (!on) return;
    if (a >= 0) std::fprintf(stderr, "MEMLOG %-14s %7ld active %6.2f GB peak %6.2f GB cache %5.2f GB\n", stage, a, mx::get_active_memory() / 1e9, mx::get_peak_memory() / 1e9, mx::get_cache_memory() / 1e9);
    else std::fprintf(stderr, "MEMLOG %-22s active %6.2f GB peak %6.2f GB cache %5.2f GB\n", stage, mx::get_active_memory() / 1e9, mx::get_peak_memory() / 1e9, mx::get_cache_memory() / 1e9);
}

void MlxQwen35Engine::spec_begin(const std::vector<uint32_t>& prompt, uint32_t count, int) {
    auto& i = *impl_;
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    if (prompt.size() + count + 16 > i.context_) throw std::invalid_argument("prompt and answer exceed the context");
    i.output.clear(); i.count = count; i.steps = i.drafted = i.accepted = 0; i.decode_us = 0; i.widths.begin(int(prompt.size())); i.drafter->native_block(i.widths.native());
    i.h_build = i.h_submit = i.h_wait = i.h_post = i.h_gap = 0; i.last_advance_end.reset();
    i.sampling = i.temperature > 0.f;
    const auto t0 = Clock::now();
    i.engine->reset(); i.drafter->reset();
    i.engine->reserve_kv(int(prompt.size() + count) + 16);
    bool lean_prefill = false;
    {
        const auto plan = i.engine->plan_kv_format(int(prompt.size()), int(count), i.chunk, i.drafter->far_row_bytes());
        const auto tier = plan.tier;
        i.engine->kv_format(tier); i.drafter->far_context(plan.far); i.drafter->reserve_far(int(prompt.size() + count) + 16); lean_prefill = plan.lean;
        if (i.exact && tier != mlx_backend::KvTier::Bf16)
            std::fprintf(stderr, "tqserve: WARN --exact on: KV forced to %s by memory for this request\n", mlx_backend::kv_tier_name(tier));
        if (tier != mlx_backend::KvTier::Bf16 || prompt.size() > 32768)
            std::fprintf(stderr, "tqserve: KV format %s%s for %zu prompt + %u new tokens (in use %.1f GB, usable working set %.1f GB)\n",
                         mlx_backend::kv_tier_name(tier), plan.far ? (plan.lean ? " (lean prefill)" : "") : (plan.lean ? " (drafter window only, lean prefill)" : " (drafter window only)"), prompt.size(), count, mx::get_active_memory() / 1e9,
                         i.engine->kv_budget(i.chunk).working_set / 1e9);
    }
    mx::reset_peak_memory();
    memlog_point("planned");
    mx::array logits = mx::array(0.0f);
    // MLX buffer cache 4 GiB during the prefill, 512 MB for decode: under the decode limit the 8192-row chunk
    // temporaries are reallocated per layer (j786 tqserve doc8k 462 vs 418 tok/s, fewer pageouts); 512 MB when the KV plan
    // needs that memory (KvPlan::lean)
    mx::set_cache_limit(lean_prefill ? size_t(512) << 20 : size_t(4096) << 20);
    for (size_t off = 0; off < prompt.size(); off += size_t(i.chunk)) {
        const int n = std::min(i.chunk, int(prompt.size() - off));
        logits = i.engine->forward(std::span<const uint32_t>(prompt.data() + off, size_t(n)));
        i.drafter->inject(i.engine->feature_states(), int(off));
        memlog_point("prefill chunk at", long(off));
    }
    i.pending = mx::argmax(logits).item<uint32_t>();
    mx::set_cache_limit(size_t(512) << 20);
    // the prefill's cached buffers go back to the system before the decode, lean or not: up to 4 GiB left cached pushed a
    // 256K decode into swap whenever the plan judged it fit (j1446: 21.8 vs 27.3 tok/s on a fresh server)
    mx::clear_cache();
    memlog_point("prefill done");

    if (i.sampling) {
        // sample the first token from the last prompt position's distribution (same filtering as spec_advance)
        const int k = i.top_k > 0 ? i.top_k : 64;
        auto scaled = mx::multiply(mx::astype(mx::reshape(logits, {1, -1}), mx::float32), mx::array(1.0f / i.temperature));
        auto [idx, val] = row_topk(scaled, k);
        mx::eval(idx, val);
        i.pending = draw_excluding(filtered_distribution(val.data<float>(), idx.data<uint32_t>(), k, i.top_p), UINT32_MAX, i.rng);
    }
    i.output.push_back(i.pending);
    i.ahead.reset();
    if (!i.sampling && i.pending != i.eos && i.output.size() < i.count) {
        i.ahead.emplace(*i.engine, *i.drafter, i.width);
        i.ahead->start(i.pending, i.widths.native() ? i.widths.choose(i.engine->position()) : i.width);
        memlog_point("draft-ahead started");
    }
    i.prefill_us = us_since(t0);
    i.decode_start = Clock::now();
    i.first_steps.clear(); i.first_gap_ms = 0; i.first_phases = {};
    i.active = true; i.done = i.output.size() >= i.count || i.pending == i.eos;
}

bool MlxQwen35Engine::spec_advance() {
    auto& i = *impl_;
    if (!i.active || i.done) return false;
    const uint32_t limit = i.count;
    // trellis KV runs its 8-row kernels: at most 7 drafted tokens per step (int8 v12 takes 16 rows)
    const int cap_width = i.engine->kv_format() != mlx_backend::KvTier::Trellis ? i.width : std::min(i.width, 7);
    // draft-ahead: the policy's width goes to the draft issued this step (verified next step); this step verifies at
    // most what was drafted for it
    const int pos = i.engine->position(), chosen = std::min(cap_width, i.widths.choose(pos)),
              verify = i.ahead ? std::min(chosen, i.ahead->drafted_width()) : chosen,
              want = std::min(verify, int(limit - i.output.size()));
    if (want < 1) {i.done = true; return false;}
    const auto step_start = Clock::now();
    if (i.first_steps.empty()) i.first_gap_ms = std::chrono::duration<double, std::milli>(step_start - i.decode_start).count();
    if (i.last_advance_end) i.h_gap += std::chrono::duration<double, std::milli>(step_start - *i.last_advance_end).count();
    struct EndMark {Impl& i; Clock::time_point t; ~EndMark() {
        i.last_advance_end = Clock::now();
        if (i.first_steps.size() < 4) i.first_steps.push_back(std::chrono::duration<double, std::milli>(*i.last_advance_end - t).count());
    }} end_mark{i, step_start};
    if (i.ahead) {
        const auto st = i.ahead->step(verify, i.widths.native() ? chosen : i.width);
        i.h_build += st.build_ms; i.h_submit += st.submit_ms; i.h_wait += st.wait_ms; i.h_post += st.post_ms;
        if (i.steps == 0) i.first_phases = {st.build_ms, st.submit_ms, st.wait_ms, st.post_ms};
        if (i.steps < 3) memlog_point("decode step", long(i.steps));
        if (want == verify) i.widths.record(verify, st.accepted, pos, us_since(step_start) / 1000.0);
        const int acc = std::min(st.accepted, want);
        ++i.steps; i.drafted += uint64_t(want); i.accepted += uint64_t(acc);
        for (int j = 0; j < acc && i.output.size() < limit; ++j) {
            i.output.push_back(st.proposed[size_t(j)]);
            if (st.proposed[size_t(j)] == i.eos) {i.done = true; break;}
        }
        if (!i.done && i.output.size() < limit) i.output.push_back(st.choices[size_t(acc)]);
        i.pending = st.choices[size_t(acc)];
        if (!i.done) i.done = i.output.size() >= limit || i.output.back() == i.eos;
        if (i.done) i.ahead.reset();
        return !i.done;
    }
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
        auto [idx, val] = row_topk(scaled, k);
        mx::eval(idx, val, drafted);
        proposed = drafted.data<uint32_t>();
        const auto* pi = idx.data<uint32_t>(); const auto* pv = val.data<float>();
        sampled_ids.assign(size_t(rows), 0u);
        auto row_dist = [&](int r, std::vector<std::pair<double, uint32_t>>& d) {d = filtered_distribution(pv + r * k, pi + r * k, k, i.top_p);};
        std::vector<std::pair<double, uint32_t>> d;
        for (; acc < want; ++acc) {
            row_dist(acc, d);
            const auto [taken, token] = speculative_pick(d, proposed[acc], i.rng);
            sampled_ids[size_t(acc)] = token;
            if (!taken) break;
        }
        if (acc == want) {row_dist(want, d); sampled_ids[size_t(want)] = draw_excluding(d, UINT32_MAX, i.rng);}
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
    if (want == verify && !i.sampling) i.widths.record(verify, acc, pos, us_since(step_start) / 1000.0);
    if (!i.done) i.done = i.output.size() >= limit || i.output.back() == i.eos;
    return !i.done;
}

const std::vector<uint32_t>& MlxQwen35Engine::spec_tokens(int) const {return impl_->output;}

std::vector<uint32_t> MlxQwen35Engine::spec_finish(uint64_t* stats) {
    auto& i = *impl_;
    i.decode_us = us_since(i.decode_start);
    if (stats) {
        stats[0] = i.steps; stats[1] = i.drafted; stats[2] = i.accepted;
        stats[3] = uint64_t(i.decode_us); stats[4] = uint64_t(i.prefill_us);
    }
    if (i.widths.narrow_steps() + i.widths.wide_steps() > 0)
        std::fprintf(stderr, "tqserve: verify widths 7 x %d, 15 x %d\n", i.widths.narrow_steps(), i.widths.wide_steps());
    if (!i.first_steps.empty()) {  // per-request one-off costs land in the first steps (doc8k: 14 steps)
        std::fprintf(stderr, "tqserve: first steps ms");
        for (double v : i.first_steps) std::fprintf(stderr, " %.1f", v);
        std::fprintf(stderr, " (first step build %.1f submit %.1f wait %.1f post %.1f; GPU memory peak %.0f MB active %.0f MB cache %.0f MB)\n",
                     i.first_phases[0], i.first_phases[1], i.first_phases[2], i.first_phases[3], mx::get_peak_memory() / 1048576.0,
                     mx::get_active_memory() / 1048576.0, mx::get_cache_memory() / 1048576.0);
    }
    if (i.steps > 1) {
        const double n = double(i.steps);
        std::fprintf(stderr, "tqserve: host ms per step: build %.2f submit %.2f wait %.2f post %.2f gap %.2f (wall %.2f, %llu steps)\n",
                     i.h_build / n, i.h_submit / n, i.h_wait / n, i.h_post / n, i.h_gap / n, i.decode_us / 1000.0 / n,
                     static_cast<unsigned long long>(i.steps));
    }
    i.active = false; i.ahead.reset();
    return i.output;  // EOS stays: finish() turns it into finish_reason "stop"
}

void MlxQwen35Engine::spec_step(const int*, int, bool*) {
    throw std::invalid_argument("the MLX backend has no batched multi-slot step (--np 1)");
}
std::unique_ptr<Engine> make_mlx_qwen35_engine(const std::string& weights, const std::string& tokenizer, uint32_t context, int slots,
                                               const EngineOptions& options) {
    return std::make_unique<MlxQwen35Engine>(weights, tokenizer, context, slots, options);
}
}  // namespace tqllm
