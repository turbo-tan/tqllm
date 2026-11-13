#pragma once
// TQueue / TPark: the rules of the host-to-device expert copy queue and of the VRAM overflow park (plans: tqueue).
// No threads, no CUDA: the engine drives them (push on a miss or a guess, start when the copy engine has room,
// complete on the copy's event, layer_ran / layer_passed as decode moves on). tests/tqueue.cpp states the rules.
// (ExpertFetchQueue is the disk -> host stage for cold experts; this is the host -> device stage.)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tqllm {

struct TQKey {
    int layer = 0, expert = 0;
    bool operator==(const TQKey & o) const { return layer == o.layer && expert == o.expert; }
    bool operator<(const TQKey & o) const { return std::tie(layer, expert) < std::tie(o.layer, o.expert); }
};

// Priorities: 0 demand miss of the running layer, 1 next-layer top guesses, 2 weaker or further guesses, 3 next step.
class TQueue {
public:
    struct Started { TQKey key; int priority; };
    struct Stats { uint64_t pushed = 0, promoted = 0, expired = 0, started = 0, demands = 0, shared = 0; };
    explicit TQueue(int max_inflight) : max_inflight_(max_inflight) {
        if (max_inflight < 1) throw std::invalid_argument("TQueue: at least one copy in flight");
    }
    // guesses on the link at once (expert_prediction.hpp kGuessLinkDepth by default; tests pin smaller depths)
    void set_max_inflight(int max_inflight) {
        if (max_inflight < 1) throw std::invalid_argument("TQueue: at least one copy in flight");
        max_inflight_ = max_inflight;
    }
    // a key already queued keeps one entry (a stronger priority promotes it); a key on the link is not queued again
    void push(TQKey key, int priority) {
        if (in_flight(key)) return;
        for (auto & e : queued_) if (e.key == key) {
            if (priority < e.priority) { e.priority = priority; ++stats_.promoted; }
            return;
        }
        queued_.push_back({key, priority, seq_++}); ++stats_.pushed;
    }
    // the next copy for the link: strongest priority, then nearest layer, then oldest; none while the link is full
    std::optional<Started> start() {
        if (queued_.empty()) return std::nullopt;
        auto best = std::min_element(queued_.begin(), queued_.end(), [](const Entry & a, const Entry & b) {
            return std::tie(a.priority, a.key.layer, a.seq) < std::tie(b.priority, b.key.layer, b.seq); });
        // P0 (experts known to be needed: an early demand) is never held back; the depth limits guesses only
        if (best->priority > 0 && int(inflight_.size()) >= max_inflight_) return std::nullopt;
        const Started s{best->key, best->priority};
        queued_.erase(best); inflight_.push_back(s.key); ++stats_.started;
        return s;
    }
    void complete() { if (!inflight_.empty()) inflight_.pop_front(); }  // copies finish in the order they started
    // a demand miss goes on the link now, behind only the copies already there (never behind a queued guess). Its queued
    // guess is taken out (promoted: one copy). false: its copy is on the link already, the caller waits for that one
    bool demand(TQKey key) {
        if (in_flight(key)) { ++stats_.shared; return false; }
        const auto it = std::find_if(queued_.begin(), queued_.end(), [&](const Entry & e) { return e.key == key; });
        if (it != queued_.end()) { queued_.erase(it); ++stats_.promoted; }
        ++stats_.demands; return true;
    }
    // a started entry that never reached the link (no park space): it gives its place back
    void abandon(TQKey key) {
        const auto it = std::find(inflight_.begin(), inflight_.end(), key);
        if (it != inflight_.end()) { inflight_.erase(it); --stats_.started; }
    }
    int flying() const { return int(inflight_.size()); }
    // the layer computed: its entries not yet on the link are useless now
    void layer_ran(int layer) {
        const auto before = queued_.size();
        queued_.erase(std::remove_if(queued_.begin(), queued_.end(), [&](const Entry & e) { return e.key.layer == layer; }), queued_.end());
        stats_.expired += before - queued_.size();
    }
    int queued() const { return int(queued_.size()); }
    bool in_flight(TQKey key) const { return std::find(inflight_.begin(), inflight_.end(), key) != inflight_.end(); }
    const Stats & stats() const { return stats_; }

private:
    struct Entry { TQKey key; int priority; uint64_t seq; };
    int max_inflight_;
    uint64_t seq_ = 0;
    std::vector<Entry> queued_;
    std::deque<TQKey> inflight_;
    Stats stats_;
};

// Fixed spaces in VRAM outside the cache slots: prefetched experts wait here, so a wrong guess never evicts a cached one.
class TPark {
public:
    struct Stats { uint64_t parked = 0, picked = 0, late_hits = 0, dead_overwritten = 0, live_evicted = 0, refused = 0; };
    explicit TPark(int spaces) : spaces_(size_t(spaces)) {
        if (spaces < 1) throw std::invalid_argument("TPark: at least one space");
    }
    void set_running(int layer) { running_ = layer; }
    // a space for the key: free first, then the oldest dead letter, then the weakest / furthest live ticket that is not
    // the running layer's; -1 when only the running layer's tickets are left
    int park(TQKey key, int priority) {
        if (const int s = find(key); s >= 0) return s;
        int pick = -1;
        for (size_t i = 0; i < spaces_.size() && pick < 0; ++i) if (!spaces_[i].used) pick = int(i);
        if (pick < 0) {
            for (size_t i = 0; i < spaces_.size(); ++i)
                if (spaces_[i].dead && (pick < 0 || spaces_[i].dead_seq < spaces_[size_t(pick)].dead_seq)) pick = int(i);
            if (pick >= 0) ++stats_.dead_overwritten;
        }
        if (pick < 0) {
            for (size_t i = 0; i < spaces_.size(); ++i) {
                const auto & s = spaces_[i];
                if (s.key.layer == running_) continue;
                if (pick < 0 || std::tie(s.priority, s.key.layer) > std::tie(spaces_[size_t(pick)].priority, spaces_[size_t(pick)].key.layer)) pick = int(i);
            }
            if (pick >= 0) ++stats_.live_evicted;
        }
        if (pick < 0) { ++stats_.refused; return -1; }
        spaces_[size_t(pick)] = {true, false, key, priority, 0};
        ++stats_.parked;
        return pick;
    }
    // the layer ran: its tickets not picked up become dead letters (overwritten first, still collectable)
    void layer_passed(int layer) {
        for (auto & s : spaces_) if (s.used && !s.dead && s.key.layer == layer) { s.dead = true; s.dead_seq = ++clock_; }
    }
    // the space holding the key, freed for the next park (the caller copies it into a cache slot first); -1 if not parked
    int pickup(TQKey key) {
        const int s = find(key); if (s < 0) return -1;
        if (spaces_[size_t(s)].dead) ++stats_.late_hits;
        spaces_[size_t(s)] = {}; ++stats_.picked;
        return s;
    }
    bool holds(TQKey key) const { return find(key) >= 0; }
    int dead() const { return int(std::count_if(spaces_.begin(), spaces_.end(), [](const Space & s) { return s.used && s.dead; })); }
    const Stats & stats() const { return stats_; }

private:
    struct Space { bool used = false, dead = false; TQKey key{}; int priority = 0; uint64_t dead_seq = 0; };
    int find(TQKey key) const {
        for (size_t i = 0; i < spaces_.size(); ++i) if (spaces_[i].used && spaces_[i].key == key) return int(i);
        return -1;
    }
    std::vector<Space> spaces_;
    int running_ = -1;
    uint64_t clock_ = 0;
    Stats stats_;
};

// The link's last entries (queued, started, done, demand, pickup, expired), kept in a ring and printed on the first GPU
// error: which copies were on the link, to which address, when a fault arrived.
class TQTrace {
public:
    enum Op : uint8_t { Queued, Started, Done, Demand, Pickup, Refused, Reset };
    void note(Op op, TQKey key, int space, const void * dst, const void * src, size_t bytes) {
        auto & e = ring_[at_++ % ring_.size()];
        e = {std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(),
             dst, src, bytes, key, space, op};
    }
    uint64_t noted() const { return at_; }
    void dump(std::FILE * out, const char * why) const {
        static const char * const names[] = {"queued", "started", "done", "demand", "pickup", "refused", "reset"};
        const uint64_t n = std::min<uint64_t>(at_, ring_.size());
        const int64_t now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::fprintf(out, "TQTRACE %s: last %llu of %llu link entries (us before now, op, layer, expert, space, dst, src, bytes)\n",
                     why, (unsigned long long) n, (unsigned long long) at_);
        for (uint64_t i = at_ - n; i < at_; ++i) {
            const auto & e = ring_[i % ring_.size()];
            std::fprintf(out, "TQTRACE %9lld %-7s L%-3d E%-4d S%-3d %p %p %zu\n", (long long) (now - e.us), names[e.op], e.key.layer,
                         e.key.expert, e.space, e.dst, e.src, e.bytes);
        }
        std::fflush(out);
    }

private:
    struct Entry { int64_t us; const void * dst, * src; size_t bytes; TQKey key; int space; Op op; };
    std::vector<Entry> ring_ = std::vector<Entry>(4096);
    uint64_t at_ = 0;
};

}  // namespace tqllm
