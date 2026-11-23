#pragma once
// Expert parallelism, the parts without a GPU: the forward rules (rank 0) and the lease protocol over any ordered control
// channel. The component that uses them: expert_parallel.hpp. Tests: tests/expert_parallel.cpp.
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>

namespace tqllm {

// ---- forward rules (pure) ----

// Result rows per layer exchanged for a forward of `assignments` (rows x topk). About half the assignments route to the
// other rank (p50 ~0.5, max ~0.66); 5/8 of them fits nearly every layer and the rest are computed locally (both ranks
// keep every expert), so the exchange has one fixed size per row count, known on both hosts.
inline int ep_result_cap(int assignments) { return std::min(std::max(assignments, 0), (5 * std::max(assignments, 0) + 7) / 8); }

// Rank 0's view of the forward being exchanged. Each forward is announced at layer 0; rank 1 accepts (then runs exactly
// `layers` exchanges) or declines (every layer runs locally). A forward that cannot finish its accepted exchanges leaves
// rank 1 out of step, so expert parallelism then stays off for the process.
struct EpForwardState { bool open = false, accepted = false, off = false; int next = 0, n = 0; };
enum class EpCall { Local, Exchange, Fail };

// Before a MoE layer: can = this layer may exchange (model: resident bank, split kernels; rules: row cap); alive = rank 1
// is there. announce(n) sends layer 0's header and returns whether the lease is out. A plain re-run of a layer 0 whose
// graph capture failed is not announced again. Fail: an accepted forward lost its exchanges (the request fails).
template<class Announce>
inline EpCall ep_before_layer(EpForwardState & s, int layer, int n, bool can, bool alive, Announce && announce) {
    if (s.off) return EpCall::Local;
    if (!alive) {
        const bool fail = s.open && s.accepted && layer == s.next;
        s = {}; s.off = true; return fail ? EpCall::Fail : EpCall::Local;
    }
    if (layer == 0) {
        if (s.open && s.next == 0 && s.n == n) return s.accepted ? EpCall::Exchange : EpCall::Local;  // the re-run of layer 0
        if (s.open && s.accepted) { s = {}; s.off = true; return EpCall::Local; }  // an accepted forward was left open
        s = {};
        if (!can) return EpCall::Local;
        s.open = true; s.n = n; s.accepted = announce(n);
        return s.accepted ? EpCall::Exchange : EpCall::Local;
    }
    if (!s.open) return EpCall::Local;
    if (layer != s.next || n != s.n || (s.accepted && !can)) {
        const bool fail = s.accepted; s = {}; s.off = fail; return fail ? EpCall::Fail : EpCall::Local;
    }
    return s.accepted ? EpCall::Exchange : EpCall::Local;
}
// After a MoE layer's work (exchanged or local): the open forward moves on; past the last layer it closes
inline void ep_after_layer(EpForwardState & s, int layer, int layers) {
    if (!s.open || layer != s.next) return;
    if (++s.next == layers) { s.open = false; s.accepted = false; s.next = 0; }
}

// ---- control channel and lease (pure; the link is any ordered byte pipe) ----

enum : int32_t { kEpForward = 1, kEpStop = 2, kEpLeaseEnd = 3, kEpLease = 21, kEpRevoke = 22 };
struct EpMsg { int32_t kind = 0, n = 0; };
struct EpControl {  // ordered, reliable; recv blocks; both throw when the other side went away
    virtual ~EpControl() = default;
    virtual void send(const EpMsg & m) = 0;
    virtual EpMsg recv() = 0;
};

// Rank 0: is the lease out? Fed by a reader of rank 1's messages; sends go out under the same lock, so a forward is never
// announced after the lease end that follows a revoke.
class EpLeaseHolder {
public:
    explicit EpLeaseHolder(EpControl & ctl) : ctl_(ctl) {}
    bool announce(int rows);  // under the lease: send kEpForward rows and return true; else false
    void stop();              // tell rank 1 to stop serving forwards
    // one message from rank 1 (kEpLease / kEpRevoke); false = the link failed (lease dropped)
    bool on_message(const EpMsg & m);
    void lost();              // rank 1 went away
    bool leased() const { std::lock_guard<std::mutex> g(m_); return leased_; }
private:
    EpControl & ctl_;
    mutable std::mutex m_;
    bool leased_ = false;
};

// Rank 1: lends the lease while this box is idle and its banks are ready; takes it back before a request of its own.
class EpLeaseGiver {
public:
    EpLeaseGiver(EpControl & ctl, std::function<bool()> idle, std::function<bool()> ready) : ctl_(ctl), idle_(std::move(idle)), ready_(std::move(ready)) {}
    void request_begin();  // this box's request: revoke and wait for rank 0's lease end (call without holding the box lock)
    void note_idle();      // this box may be idle now: grant the lease if it is and the banks are ready
    void lease_ended();    // rank 0's kEpLeaseEnd (or rank 0 went away): wake a waiting request_begin
    void close();          // the engine is going away: never grant again
    uint64_t requests() const { return requests_; }
private:
    EpControl & ctl_;
    std::function<bool()> idle_, ready_;
    std::mutex m_, w_; std::condition_variable cv_;
    bool out_ = false, closed_ = false;
    uint64_t requests_ = 0, ready_at_ = ~uint64_t(0);
};

}  // namespace tqllm
