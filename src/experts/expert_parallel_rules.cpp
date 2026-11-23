#include "expert_parallel_rules.hpp"

#include <cstdio>
#include <stdexcept>

namespace tqllm {

// ---- lease, rank 0 ----

bool EpLeaseHolder::announce(int rows) {
    std::lock_guard<std::mutex> g(m_);
    if (!leased_) return false;  // rank 1 serves its own users
    ctl_.send({kEpForward, rows});
    return true;
}
void EpLeaseHolder::stop() { std::lock_guard<std::mutex> g(m_); ctl_.send({kEpStop, 0}); }
bool EpLeaseHolder::on_message(const EpMsg & m) {
    std::lock_guard<std::mutex> g(m_);
    if (m.kind == kEpLease) leased_ = true;
    else if (m.kind == kEpRevoke) {  // every forward announced so far was sent before this lease end
        leased_ = false;
        try { ctl_.send({kEpLeaseEnd, 0}); } catch (const std::exception &) { return false; }
    }
    return true;
}
void EpLeaseHolder::lost() { std::lock_guard<std::mutex> g(m_); leased_ = false; }

// ---- lease, rank 1 ----

void EpLeaseGiver::request_begin() {
    std::unique_lock<std::mutex> l(m_);
    ++requests_;
    if (!out_) return;
    try { std::lock_guard<std::mutex> w(w_); ctl_.send({kEpRevoke, 0}); }
    catch (const std::exception &) { out_ = false; return; }
    cv_.wait(l, [&] { return !out_; });  // rank 0's lease end follows its last forward
}
void EpLeaseGiver::note_idle() {
    {
        std::lock_guard<std::mutex> l(m_);
        if (closed_ || out_ || !idle_()) return;
        if (requests_ != ready_at_) {  // this box's requests may have reshaped its banks
            if (!ready_()) {
                static bool said = false;
                if (!said) { said = true; std::fprintf(stderr, "tqllm: --split experts rank 1: a bank is not full after this box's request; rank 0 runs alone until one is\n"); }
                return;
            }
            ready_at_ = requests_;
        }
        out_ = true;
    }
    try { std::lock_guard<std::mutex> w(w_); ctl_.send({kEpLease, 1}); }
    catch (const std::exception &) { std::lock_guard<std::mutex> l(m_); out_ = false; }
}
void EpLeaseGiver::lease_ended() { std::lock_guard<std::mutex> l(m_); out_ = false; cv_.notify_all(); }
void EpLeaseGiver::close() { std::lock_guard<std::mutex> l(m_); closed_ = true; }

}  // namespace tqllm
