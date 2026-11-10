// tqllm::Model: the backend-neutral front of every engine. The registry (registry.hpp) picks the engine for a model path;
// every method forwards to it. No backend headers here: CUDA and Metal builds share this file.
#include "tqllm/model.hpp"
#include "engine.hpp"
#include "model_impl.hpp"
#include "registry.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <vector>

namespace tqllm {
namespace {
// --exact on conflicts (approximations asked for explicitly): one WARN each, the asked options win; returns the /health note
std::string check_mode(const std::filesystem::path & p, const EngineOptions & o, const EngineEntry & engine, int slots) {
    const auto conflicts = exact_conflicts(o, p.extension() == ".tqfn", !engine.exact_path,
                                           [](const char * n) -> const char * { return std::getenv(n); }, slots);
    if (conflicts.empty()) return {};
    warn_exact_conflicts(conflicts, std::cerr);
    std::string note = "exact on, overridden by:";
    for (const auto & c : conflicts) note += " " + c.substr(0, c.find(':')) + ";";
    return note;
}
}  // namespace

// `weights`: a GGUF file, a Hugging Face checkpoint folder, a Flash-Next .tqfn manifest, or DIR/weights.tqbin (Qwen2)
Model::Model(const std::string & w, const std::string & t, uint32_t c, int slots, const EngineOptions & o)
    : impl_(std::make_unique<Impl>()) {
    const std::filesystem::path p(w);
    const EngineEntry * engine = find_engine(p);
    if (!engine) {
        if (p.extension() == ".tqfn") throw std::invalid_argument("this binary has no Flash-Next engine (use tqserve)");
        throw std::invalid_argument("no engine in this build serves " + w);
    }
    const std::string mode_note = check_mode(p, o, *engine, slots);
    impl_->e = engine->make(w, t, c, slots, o);
    auto & note = impl_->e->note;
    if (!mode_note.empty()) note = note.empty() ? mode_note : mode_note + " " + note;
}
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;

uint32_t Model::forward(uint32_t token, uint32_t pos, bool sample) { return impl_->e->step(token, pos, sample, nullptr); }
void Model::synchronize() { impl_->e->synchronize(); }
bool Model::prefill_state(const std::vector<uint32_t> & tokens, uint32_t pos) { return impl_->e->prefill_state(tokens, pos); }
int Model::slots() const { return impl_->e->slots(); }
bool Model::incremental() const { return impl_->e->incremental(); }
uint32_t Model::context() const { return impl_->e->context(); }
bool Model::kv_idle_pending() const { return impl_->e->kv_idle_pending(); }
bool Model::kv_idle_step() { return impl_->e->kv_idle_step(); }
bool Model::has_drafter() const { return impl_->e->has_drafter(); }
bool Model::native_mtp() const { return impl_->e->native_mtp(); }
std::string Model::fit_note() const { return impl_->e->note; }
void Model::select_slot(int slot) { impl_->e->select_slot(slot); }
void Model::set_prompt_images(const std::vector<PromptImage> & images) { impl_->e->set_prompt_images(images); }
void Model::spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int draft) {
    impl_->e->spec_begin(prompt, count, draft);
}
bool Model::spec_advance() { return impl_->e->spec_advance(); }
void Model::spec_step(const std::vector<int> & slots, std::vector<char> & done) {
    std::unique_ptr<bool[]> d(new bool[slots.size()]());
    impl_->e->spec_step(slots.data(), int(slots.size()), d.get());
    done.assign(d.get(), d.get() + slots.size());
}
std::vector<uint32_t> Model::spec_finish(uint64_t * stats) { return impl_->e->spec_finish(stats); }
const std::vector<uint32_t> & Model::spec_tokens(int slot) const { return impl_->e->spec_tokens(slot); }
uint32_t Model::prefix_reuse(int slot, const std::vector<uint32_t> & prompt) const { return impl_->e->prefix_reuse(slot, prompt); }
int Model::drafter_features(const std::vector<uint32_t> & tokens, std::vector<uint16_t> & feats) {
    return impl_->e->drafter_features(tokens, feats);
}
void Model::verify(const std::vector<uint32_t> & tokens, uint32_t pos, std::vector<uint32_t> & next) {
    next.resize(tokens.size());
    impl_->e->verify(tokens.data(), int(tokens.size()), pos, next.data());
}
void Model::commit(int accepted) { impl_->e->commit(accepted); }
void Model::spec_abort(int slot) { impl_->e->spec_abort(slot); }
const float * Model::verify_logits() const { return impl_->e->verify_logits(); }
void Model::set_full_logits(bool on) { impl_->e->set_full_logits(on); }
void Model::set_ignore_eos(bool on) { impl_->e->set_ignore_eos(on); }
void Model::set_first_token_hook(std::function<void()> hook) { impl_->e->set_first_token_hook(std::move(hook)); }
void Model::set_continuation(int phase) { impl_->e->set_continuation(phase); }
void Model::set_penalties(float repeat, float frequency) { impl_->e->set_penalties(repeat, frequency); }
void Model::set_sampling(float temperature, float top_p, int top_k, uint64_t seed, float presence) {
    impl_->e->set_sampling(temperature, top_p, top_k, seed, presence);
}

std::vector<uint32_t> Model::generate_spec(const std::vector<uint32_t> & prompt, uint32_t count, int draft,
                                           uint64_t * stats) {
    return impl_->e->generate_spec(prompt, count, draft, stats);
}
uint32_t Model::layers() const { return impl_->e->layers(); }
uint32_t Model::hidden_size() const { return impl_->e->hidden(); }
uint32_t Model::vocab_size() const { return impl_->e->vocab(); }


std::vector<uint32_t> Model::generate(const std::vector<uint32_t> & prompt, uint32_t count) {
    auto & e = *impl_->e;
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    if (prompt.size() > e.context() || uint64_t(prompt.size()) + count > e.context())
        throw std::runtime_error("request exceeds context capacity");
    const auto n = uint32_t(prompt.size());
    for (uint32_t i = 0; i + 1 < n; ++i) e.step(prompt[i], i, false, nullptr);
    uint32_t next = e.step(prompt[n - 1], n - 1, true, nullptr);
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < count; ++i) {
        out.push_back(next);
        if (next == e.eos()) break;
        if (i + 1 < count) next = e.step(next, n + i, true, nullptr);
    }
    return out;
}

} // namespace tqllm
