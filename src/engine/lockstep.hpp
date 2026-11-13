#pragma once
// Layer split across GPUs (docs plan multi-gpu-20261109): one process per GPU. Rank 0 serves HTTP; every engine call that
// changes state is sent to rank 1 over a host channel before it runs, and rank 1 (follow()) makes the same call on its
// own engine. Both engines run the same host code on the same token ids, so they make the same decisions; the device
// work between them (hidden rows at the layer boundary) goes over NCCL inside the engine. Pure C++: tests/lockstep.cpp.
#include "engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tqllm {

struct ByteChannel {
    virtual ~ByteChannel() = default;
    virtual void send(const void * p, size_t n) = 0;
    virtual void recv(void * p, size_t n) = 0;
};

namespace lockstep {
enum class Op : uint32_t {
    stop = 1, step, verify, commit, generate_spec, ignore_eos, continuation, penalties, sampling, select_slot, spec_begin,
    spec_advance, spec_abort, spec_step, spec_finish, drafter_features, prefill_state, full_logits, kv_idle_step,
};

// one call: [u64 bytes][u32 op][args]
class Writer {
public:
    explicit Writer(Op op) { put(uint32_t(op)); }
    template <class T> Writer & put(const T & v) {
        static_assert(std::is_trivially_copyable<T>::value, "plain values only");
        const auto * c = reinterpret_cast<const char *>(&v);
        b_.insert(b_.end(), c, c + sizeof v);
        return *this;
    }
    template <class T> Writer & vec(const T * p, size_t n) {
        put(uint64_t(n));
        const auto * c = reinterpret_cast<const char *>(p);
        b_.insert(b_.end(), c, c + n * sizeof(T));
        return *this;
    }
    void send(ByteChannel & ch) const {
        const uint64_t n = b_.size();
        ch.send(&n, sizeof n);
        ch.send(b_.data(), b_.size());
    }
private:
    std::vector<char> b_;
};

class Reader {
public:
    explicit Reader(ByteChannel & ch) {
        uint64_t n = 0;
        ch.recv(&n, sizeof n);
        if (n < 4 || n > (uint64_t(1) << 34)) throw std::runtime_error("lockstep: bad call size");
        b_.resize(size_t(n));
        ch.recv(b_.data(), b_.size());
    }
    template <class T> T get() {
        T v;
        take(&v, sizeof v);
        return v;
    }
    template <class T> std::vector<T> vec() {
        const auto n = get<uint64_t>();
        if (n > (b_.size() - at_) / sizeof(T)) throw std::runtime_error("lockstep: truncated call");
        std::vector<T> v(static_cast<size_t>(n));
        take(v.data(), size_t(n) * sizeof(T));
        return v;
    }
    bool done() const { return at_ == b_.size(); }
private:
    void take(void * p, size_t n) {
        if (n > b_.size() - at_) throw std::runtime_error("lockstep: truncated call");
        std::memcpy(p, b_.data() + at_, n);
        at_ += n;
    }
    std::vector<char> b_;
    size_t at_ = 0;
};

// rank 1: one call from rank 0 applied to `e`; false at stop
inline bool apply(Engine & e, Reader & r) {
    const auto op = Op(r.get<uint32_t>());
    switch (op) {
    case Op::stop: return false;
    case Op::step: {
        const auto id = r.get<uint32_t>(), pos = r.get<uint32_t>();
        e.step(id, pos, r.get<uint8_t>() != 0, nullptr);
        break;
    }
    case Op::verify: {
        const auto t = r.vec<uint32_t>();
        const auto pos = r.get<uint32_t>();
        std::vector<uint32_t> next(t.size());
        e.verify(t.data(), int(t.size()), pos, next.data());
        break;
    }
    case Op::commit: e.commit(r.get<int32_t>()); break;
    case Op::generate_spec: {
        const auto p = r.vec<uint32_t>();
        const auto count = r.get<uint32_t>();
        e.generate_spec(p, count, r.get<int32_t>(), nullptr);
        break;
    }
    case Op::ignore_eos: e.set_ignore_eos(r.get<uint8_t>() != 0); break;
    case Op::continuation: e.set_continuation(r.get<int32_t>()); break;
    case Op::penalties: {
        const auto rep = r.get<float>();
        e.set_penalties(rep, r.get<float>());
        break;
    }
    case Op::sampling: {
        const auto t = r.get<float>(), tp = r.get<float>();
        const auto tk = r.get<int32_t>();
        const auto seed = r.get<uint64_t>();
        e.set_sampling(t, tp, tk, seed, r.get<float>());
        break;
    }
    case Op::select_slot: e.select_slot(r.get<int32_t>()); break;
    case Op::spec_begin: {
        const auto p = r.vec<uint32_t>();
        const auto count = r.get<uint32_t>();
        e.spec_begin(p, count, r.get<int32_t>());
        break;
    }
    case Op::spec_advance: e.spec_advance(); break;
    case Op::spec_abort: e.spec_abort(r.get<int32_t>()); break;
    case Op::spec_step: {
        const auto s = r.vec<int32_t>();
        std::unique_ptr<bool[]> done(new bool[s.size() ? s.size() : 1]());
        e.spec_step(s.data(), int(s.size()), done.get());
        break;
    }
    case Op::spec_finish: e.spec_finish(nullptr); break;
    case Op::drafter_features: {
        const auto t = r.vec<uint32_t>();
        std::vector<uint16_t> f;
        e.drafter_features(t, f);
        break;
    }
    case Op::prefill_state: {
        const auto t = r.vec<uint32_t>();
        e.prefill_state(t, r.get<uint32_t>());
        break;
    }
    case Op::full_logits: e.set_full_logits(r.get<uint8_t>() != 0); break;
    case Op::kv_idle_step: e.kv_idle_step(); break;
    default: throw std::runtime_error("lockstep: unknown call " + std::to_string(uint32_t(op)));
    }
    if (!r.done()) throw std::runtime_error("lockstep: call has trailing bytes");
    return true;
}
}  // namespace lockstep

// Rank 0 wraps its engine: each state-changing call goes to rank 1 first. Rank 1 wraps its engine too and runs follow().
class LockstepEngine final : public Engine {
public:
    LockstepEngine(std::unique_ptr<Engine> inner, std::shared_ptr<ByteChannel> ch, int rank)
        : e_(std::move(inner)), ch_(std::move(ch)), rank_(rank) {
        note = e_->note;
    }
    ~LockstepEngine() override {
        try { stop_followers(); } catch (...) {}
    }
    uint32_t step(uint32_t id, uint32_t pos, bool sample, float * capture) override {
        if (capture) throw std::runtime_error("--split layer: hidden-state capture is single-GPU only");
        call(lockstep::Writer(lockstep::Op::step).put(id).put(pos).put(uint8_t(sample)));
        return e_->step(id, pos, sample, nullptr);
    }
    void verify(const uint32_t * t, int n, uint32_t pos, uint32_t * next) override {
        call(lockstep::Writer(lockstep::Op::verify).vec(t, size_t(std::max(n, 0))).put(pos));
        e_->verify(t, n, pos, next);
    }
    void commit(int k) override { call(lockstep::Writer(lockstep::Op::commit).put(int32_t(k))); e_->commit(k); }
    std::vector<uint32_t> generate_spec(const std::vector<uint32_t> & p, uint32_t count, int d, uint64_t * stats) override {
        call(lockstep::Writer(lockstep::Op::generate_spec).vec(p.data(), p.size()).put(count).put(int32_t(d)));
        return e_->generate_spec(p, count, d, stats);
    }
    void set_ignore_eos(bool b) override { call(lockstep::Writer(lockstep::Op::ignore_eos).put(uint8_t(b))); e_->set_ignore_eos(b); }
    void set_first_token_hook(std::function<void()> h) override { e_->set_first_token_hook(std::move(h)); }
    void set_continuation(int c) override { call(lockstep::Writer(lockstep::Op::continuation).put(int32_t(c))); e_->set_continuation(c); }
    void set_penalties(float rep, float freq) override {
        call(lockstep::Writer(lockstep::Op::penalties).put(rep).put(freq));
        e_->set_penalties(rep, freq);
    }
    void set_sampling(float t, float tp, int tk, uint64_t seed, float presence) override {
        call(lockstep::Writer(lockstep::Op::sampling).put(t).put(tp).put(int32_t(tk)).put(seed).put(presence));
        e_->set_sampling(t, tp, tk, seed, presence);
    }
    int slots() const override { return e_->slots(); }
    void follow() override {
        if (rank_ == 0) throw std::runtime_error("lockstep: follow() is for rank 1");
        for (;;) {
            lockstep::Reader r(*ch_);
            try {
                if (!lockstep::apply(*e_, r)) return;
            } catch (const std::exception & ex) {
                // rank 0 runs the same checks on the same input and fails the same call
                std::fprintf(stderr, "tqllm: rank %d: %s\n", rank_, ex.what());
            }
        }
    }
    void stop_followers() override {
        if (rank_ != 0 || stopped_) return;
        stopped_ = true;
        lockstep::Writer(lockstep::Op::stop).send(*ch_);
    }
    bool has_drafter() const override { return e_->has_drafter(); }
    bool native_mtp() const override { return e_->native_mtp(); }
    bool incremental() const override { return e_->incremental(); }
    void select_slot(int s) override { call(lockstep::Writer(lockstep::Op::select_slot).put(int32_t(s))); e_->select_slot(s); }
    void spec_begin(const std::vector<uint32_t> & p, uint32_t count, int d) override {
        call(lockstep::Writer(lockstep::Op::spec_begin).vec(p.data(), p.size()).put(count).put(int32_t(d)));
        e_->spec_begin(p, count, d);
    }
    bool spec_advance() override { call(lockstep::Writer(lockstep::Op::spec_advance)); return e_->spec_advance(); }
    void spec_abort(int s) override { call(lockstep::Writer(lockstep::Op::spec_abort).put(int32_t(s))); e_->spec_abort(s); }
    void set_prompt_images(const std::vector<PromptImage> &) override {
        throw std::runtime_error("--split layer: image input is single-GPU only so far");
    }
    void spec_step(const int * s, int n, bool * done) override {
        std::vector<int32_t> v(s, s + std::max(n, 0));
        call(lockstep::Writer(lockstep::Op::spec_step).vec(v.data(), v.size()));
        e_->spec_step(s, n, done);
    }
    std::vector<uint32_t> spec_finish(uint64_t * stats) override {
        call(lockstep::Writer(lockstep::Op::spec_finish));
        return e_->spec_finish(stats);
    }
    const std::vector<uint32_t> & spec_tokens(int s) const override { return e_->spec_tokens(s); }
    uint32_t prefix_reuse(int s, const std::vector<uint32_t> & p) const override { return e_->prefix_reuse(s, p); }
    int drafter_features(const std::vector<uint32_t> & t, std::vector<uint16_t> & f) override {
        call(lockstep::Writer(lockstep::Op::drafter_features).vec(t.data(), t.size()));
        return e_->drafter_features(t, f);
    }
    bool prefill_state(const std::vector<uint32_t> & t, uint32_t pos) override {
        call(lockstep::Writer(lockstep::Op::prefill_state).vec(t.data(), t.size()).put(pos));
        return e_->prefill_state(t, pos);
    }
    const float * logits() const override { return e_->logits(); }
    const float * verify_logits() const override { return e_->verify_logits(); }
    void set_full_logits(bool b) override { call(lockstep::Writer(lockstep::Op::full_logits).put(uint8_t(b))); e_->set_full_logits(b); }
    void synchronize() override { e_->synchronize(); }
    uint32_t layers() const override { return e_->layers(); }
    uint32_t hidden() const override { return e_->hidden(); }
    uint32_t vocab() const override { return e_->vocab(); }
    uint32_t eos() const override { return e_->eos(); }
    uint32_t context() const override { return e_->context(); }
    size_t kv_token_bytes() const override { return e_->kv_token_bytes(); }
    size_t kv_host_bytes() const override { return e_->kv_host_bytes(); }
    size_t kv_device_bytes() const override { return e_->kv_device_bytes(); }
    size_t kv_round_bytes() const override { return e_->kv_round_bytes(); }
    size_t host_placed_bytes(bool head) const override { return e_->host_placed_bytes(head); }
    bool kv_idle_pending() const override { return e_->kv_idle_pending(); }
    bool kv_idle_step() override { call(lockstep::Writer(lockstep::Op::kv_idle_step)); return e_->kv_idle_step(); }

private:
    void call(const lockstep::Writer & w) {
        if (rank_ != 0) throw std::runtime_error("lockstep: rank 1 takes calls from rank 0 only (follow())");
        if (stopped_) throw std::runtime_error("lockstep: rank 1 was stopped");
        w.send(*ch_);
    }
    std::unique_ptr<Engine> e_;
    std::shared_ptr<ByteChannel> ch_;
    int rank_ = 0;
    bool stopped_ = false;
};

}  // namespace tqllm
