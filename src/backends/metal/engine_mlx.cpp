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
#include "attention_decode.hpp"
#include "attention_exact.hpp"
#include "kv_tcq.hpp"
#include "spec_pipeline.hpp"
#include "width_policy.hpp"
#include "wired_limit.hpp"
#include "gguf.hpp"
#include "tqllm/options.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
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

struct ScopedWiredLimitHolder { tqllm::mlx_backend::ScopedWiredLimit limit; tqllm::mlx_backend::RequestResidency request; };
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
    void set_continuation(int phase) override;
    void set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence) override;
    void synchronize() override;
    uint32_t layers() const override;
    uint32_t hidden() const override;
    uint32_t vocab() const override;
    uint32_t eos() const override;
    void spec_begin(const std::vector<uint32_t>& prompt, uint32_t count, int) override;
    bool spec_advance() override;
    bool tree_step(int pos, uint32_t limit);
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
    std::optional<tqllm::mlx_backend::TreeAhead> tree_ahead;  // tree requests: the device tree pipeline
    tqllm::mlx_backend::WidthPolicy widths;  // verify 7 or 15 drafted tokens per step (time lines kept across requests)
    uint32_t context_ = 0, eos = 0, vocab = 0, layers_ = 0, hidden_ = 0;
    int width = 15, chunk = 8192;  // block-16 drafts; prefill batches of 8192 run layer-major in 2048-row slices
    bool exact = true;  // --exact on|off
    bool far = false;   // the drafter retrieves far context for the active request
    bool tail_reuse = false;  // tail_prefill_reuse(): --exact off, or --prompt-cache on
    // thinking budget (set_continuation): phase 1 keeps its state, phase 2 prefills only its tail (the closing tag) on it;
    // both phases are one request, so this holds under --exact on too (the continuation is the state the request made)
    int continuation = 0;
    bool held_for_continuation = false;
    float temperature = 0.f, top_p = 1.f; int top_k = 0; std::mt19937_64 rng{0};
    // the tokens the target's KV / GDN state and the drafter hold after the last request (its prompt + emitted text minus
    // the pending token), valid only after a token-tree request that committed exactly its text; a prompt that extends
    // them prefills only its tail (a thinking budget's second phase: prompt + thinking + the closing tag)
    std::vector<uint32_t> prompt, held;
    // the active request
    std::vector<uint32_t> output;
    uint32_t count = 0, pending = 0;
    bool active = false, done = true, sampling = false;
    bool tree = false;  // this request drafts token trees (BF16 KV; sampled rows draw on the device): DFlash2 lattice -> best-first tree -> one verify
    uint64_t steps = 0, drafted = 0, accepted = 0;
    double prefill_us = 0, decode_us = 0;
    // host phases summed over the request's draft-ahead steps (ms); gap = host time between one spec_advance's end
    // and the next one's start (the server loop: publish, repeat guard, scheduling)
    double h_build = 0, h_submit = 0, h_wait = 0, h_post = 0, h_gap = 0;
    double h_inject = 0, h_encode = 0;  // inside h_submit: the drafter's inject, then the next tree's encode
    double h_vsubmit = 0;               // inside h_build: what the verify's submission blocked on
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
    i.layers_ = uint32_t(i.engine->layer_count()); i.hidden_ = uint32_t(i.engine->hidden_size());
    i.drafter = std::make_unique<DFlash2>(options.draft_model, *i.engine);
    i.engine->enable_draft_head4();  // 4-bit proposal head, matrix-unit layout (gated j892)
    i.exact = options.exact;
    i.engine->set_exact(i.exact);
    i.engine->qsa_from(options.qsa_from);
    i.tail_reuse = tqllm::tail_prefill_reuse(options);
    if (i.exact && options.prompt_cache)
        tqllm::warn_exact_conflicts({"--prompt-cache on: a prompt that extends the previous request's text prefills only its tail, "
                                     "so the text can differ from a fresh prefill"}, std::cerr);
    if (i.engine->head_dim() == 256) {
    mlx_backend::warm_compressed_kv_kernels(i.engine->q_heads(), i.engine->kv_heads());
    mlx_backend::warm_split_attention_kernels(i.engine->q_heads(), i.engine->kv_heads());
    mlx_backend::warm_exact_attention_kernels(i.engine->q_heads(), i.engine->kv_heads());
    }
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
void MlxQwen35Engine::set_continuation(int phase) {
    auto& i = *impl_;
    i.continuation = phase;
    if (phase == 0 && i.held_for_continuation && !i.active) {  // phase 1 ended without a phase 2: free what it kept
        i.held_for_continuation = false; i.held.clear();
        i.engine->reset(); i.engine->release_kv(); i.drafter->reset();
        mx::clear_cache();
    }
}
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
    i.h_build = i.h_submit = i.h_wait = i.h_post = i.h_gap = 0; i.h_inject = i.h_encode = i.h_vsubmit = 0; i.last_advance_end.reset();
    i.sampling = i.temperature > 0.f;
    const auto t0 = Clock::now();
    size_t reuse = 0;
    const bool may_reuse = mlx_backend::may_reuse_held(i.tail_reuse, i.continuation, i.held_for_continuation);
    if (may_reuse && !i.held.empty() && prompt.size() > i.held.size() && std::equal(i.held.begin(), i.held.end(), prompt.begin()) &&
        i.engine->position() == int(i.held.size()))
        reuse = i.held.size();
    i.held.clear(); i.held_for_continuation = false;
    bool lean_prefill = false;
    {
        const auto plan = i.engine->plan_kv_format(int(prompt.size()), int(count), i.chunk, i.drafter->far_row_bytes());
        const auto tier = plan.tier;
        // no_far: the drafter's ring window only, what it costs against the far-context retrieval
        const bool far = plan.far && !tqllm::debug_on("no_far");
        if (reuse && (tier != i.engine->kv_format() || far != i.far)) reuse = 0;
        if (!reuse) {i.engine->reset(); i.drafter->reset(); i.engine->kv_format(tier);}
        i.engine->reserve_kv(int(prompt.size() + count) + 16);
        i.far = far;
        i.drafter->far_context(far); i.drafter->reserve_far(int(prompt.size() + count) + 16); lean_prefill = plan.lean;
        if (i.exact && tier != mlx_backend::KvTier::Bf16)
            std::fprintf(stderr, "tqserve: WARN --exact on: KV forced to %s by memory for this request (int8 attention: int8 query)\n", mlx_backend::kv_tier_name(tier));
        if (tier != mlx_backend::KvTier::Bf16 || prompt.size() > 32768)
            std::fprintf(stderr, "tqserve: KV format %s%s for %zu prompt + %u new tokens (in use %.1f GB, usable working set %.1f GB)\n",
                         mlx_backend::kv_tier_name(tier), far ? (plan.lean ? " (lean prefill)" : "") : (plan.lean ? " (drafter window only, lean prefill)" : " (drafter window only)"), prompt.size(), count, mx::get_active_memory() / 1e9,
                         i.engine->kv_budget(i.chunk).working_set / 1e9);
    }
    mx::reset_peak_memory();
    memlog_point("planned");
    mx::array logits = mx::array(0.0f);
    // MLX buffer cache 4 GiB during the prefill, 512 MB for decode: under the decode limit the 8192-row chunk
    // temporaries are reallocated per layer (j786 tqserve doc8k 462 vs 418 tok/s, fewer pageouts); 512 MB when the KV plan
    // needs that memory (KvPlan::lean)
    mx::set_cache_limit(lean_prefill ? size_t(512) << 20 : size_t(4096) << 20);
    if (reuse) std::fprintf(stderr, "tqserve: prompt extends the previous request's text: %zu of %zu tokens already held, prefill %zu\n",
                            reuse, prompt.size(), prompt.size() - reuse);
    i.prompt = prompt;
    for (size_t off = reuse; off < prompt.size(); off += size_t(i.chunk)) {
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
    if (!i.residency->limit.overridden()) {
        const size_t wired = i.residency->request.hold(mx::get_active_memory(), size_t(1) << 30);
        if (prompt.size() > 32768) std::fprintf(stderr, "tqserve: request keeps %.1f GB resident\n", wired / 1e9);
    }

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
    i.tree = i.engine->kv_format() == mlx_backend::KvTier::Bf16;
    i.tree_ahead.reset();
    if (i.tree && i.pending != i.eos && i.output.size() < i.count) {
        // TQLLM_DEBUG=tree_rows:32: a 31-node tree in one 32-row verify (the release default stays 16 rows: j1583 priced
        // the 32-row verify at +70 % of the step for +7 % of the tokens)
        // tree_children / tree_tau / tree_edge: the shape of the 15-node tree, for a cheap acceptance sweep (it does not
        // move the verify's cost)
        mlx_backend::DeviceTreeOptions tree_options;
        if (const char* rows = tqllm::debug_cstr("tree_rows")) {
            const int n = std::atoi(rows);
            if (n != 16 && n != 32) throw std::invalid_argument("TQLLM_DEBUG tree_rows takes 16 or 32");
            if (n == 32 && i.engine->exact() && mlx_backend::matrix_unit_available())
                std::fprintf(stderr, "tqserve: WARN TQLLM_DEBUG=tree_rows:32 under --exact on: on the matrix unit a 17-32 row verify "
                                     "does not run the exact attention kernel its AR steps use, so output can differ from plain decoding\n");
            tree_options.nodes = n - 1;
        }
        if (const char* k = tqllm::debug_cstr("tree_children")) tree_options.children = std::atoi(k);
        if (const char* t = tqllm::debug_cstr("tree_tau")) tree_options.tau = float(std::atof(t));
        if (const char* e = tqllm::debug_cstr("tree_edge")) tree_options.edge = float(std::atof(e));
        i.tree_ahead.emplace(*i.engine, *i.drafter, tree_options);
        if (i.sampling) i.tree_ahead->sample(i.temperature, i.top_k, i.top_p, &i.rng);
        i.tree_ahead->start(i.pending);
        memlog_point("tree draft-ahead started");
    }
    if (!i.sampling && !i.tree && i.pending != i.eos && i.output.size() < i.count) {
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
    if (i.tree) return tree_step(pos, limit);
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

// One token-tree step (TensorFold's DFlash2 tree, src/spec/draft_tree.hpp): the drafter's selector lattice for the
// block's 15 positions becomes a best-first tree of up to 15 tokens under the anchor, built on the device
// (tree_device.hpp); the target verifies anchor + tree in one 16-row pass under the ancestor mask, the target's argmax
// walks down the tree from the anchor, and the deepest matching path is committed (KV rows, GDN state, drafter features).
// The next tree is drafted behind the verify on the GPU (TreeAhead), so the host waits once per step. Greedy output equals
// the chain's: every kept token is the target's argmax after its own path.
bool MlxQwen35Engine::tree_step(int, uint32_t limit) {
    auto& i = *impl_;
    if (!i.tree_ahead) {i.done = true; return false;}
    const auto st = i.tree_ahead->step(int(limit - i.output.size()));
    i.h_build += st.build_ms; i.h_submit += st.submit_ms; i.h_wait += st.wait_ms; i.h_post += st.post_ms;
    i.h_inject += st.inject_ms; i.h_encode += st.encode_ms; i.h_vsubmit += st.vsubmit_ms;
    if (i.steps == 0) i.first_phases = {st.build_ms, st.submit_ms, st.wait_ms, st.post_ms};
    ++i.steps; i.drafted += uint64_t(st.ids.size() - 1); i.accepted += uint64_t(st.accepted);
    for (int j = 1; j <= st.accepted && i.output.size() < limit; ++j) {
        i.output.push_back(st.ids[size_t(st.path[size_t(j)])]);
        if (st.ids[size_t(st.path[size_t(j)])] == i.eos) {i.done = true; break;}
    }
    const uint32_t bonus = st.choices[size_t(st.path.back())];
    if (!i.done && i.output.size() < limit) i.output.push_back(bonus);
    i.pending = bonus;
    if (!i.done) i.done = i.output.size() >= limit || i.output.back() == i.eos;
    if (i.done) i.tree_ahead.reset();
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
        std::fprintf(stderr, "tqserve: host ms per step: build %.2f (vsub %.2f) submit %.2f (inject %.2f encode %.2f) wait %.2f post %.2f gap %.2f (wall %.2f, %llu steps)\n",
                     i.h_build / n, i.h_vsubmit / n, i.h_submit / n, i.h_inject / n, i.h_encode / n, i.h_wait / n, i.h_post / n,
                     i.h_gap / n, i.decode_us / 1000.0 / n, static_cast<unsigned long long>(i.steps));
    }
    // the state holds prompt + emitted text minus the pending token when the tree committed exactly what it emitted
    const bool exact_text = i.tree && !i.output.empty() && i.engine->position() == int(i.prompt.size() + i.output.size() - 1);
    i.active = false; i.ahead.reset(); i.tree_ahead.reset();
    i.residency->request.release();
    if (mlx_backend::keep_state_for(i.tail_reuse, i.continuation, exact_text)) {
        i.held_for_continuation = i.continuation == 1;
        i.drafter->settle_to_host(i.engine->position());
        i.held = i.prompt;
        i.held.insert(i.held.end(), i.output.begin(), i.output.end() - 1);
    } else {
        i.engine->reset(); i.engine->release_kv(); i.drafter->reset();
        mx::clear_cache();
    }
    memlog_point("request done");
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
