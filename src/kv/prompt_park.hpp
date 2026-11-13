#pragma once
// PromptPark: kept prompt states parked outside the serving slots (the TPark rules of TQueue, for conversations instead of
// experts). A slot handed to another conversation parks its kept state (KV rows up to the kept position + the recurrent
// state at it); the conversation's next turn picks it up into whichever slot it gets. Spaces are taken: free first, then
// the oldest finished (dead) state, then the oldest live one that no queued request is waiting for; refused otherwise.
// A conversation keeps one space (its newest state replaces the old one). No CUDA: the engine copies; tests/prompt_park.cpp.
#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

namespace tqllm {
class PromptPark {
public:
    struct Stats { uint64_t parked = 0, picked = 0, replaced = 0, dead_overwritten = 0, live_evicted = 0, refused = 0; };
    explicit PromptPark(int spaces) : spaces_(size_t(spaces)) {
        if (spaces < 1) throw std::invalid_argument("PromptPark: at least one space");
    }
    // a space for the conversation's state, or -1 (every other space protected): its newest state replaces an older one
    int park(uint64_t conv) {
        if (const int s = find(conv); s >= 0) { spaces_[size_t(s)].seq = ++clock_; spaces_[size_t(s)].dead = false; ++stats_.replaced; return s; }
        int pick = -1;
        for (size_t i = 0; i < spaces_.size() && pick < 0; ++i) if (!spaces_[i].used) pick = int(i);
        if (pick < 0) {
            pick = oldest([](const Space & s) { return s.dead; });
            if (pick >= 0) ++stats_.dead_overwritten;
        }
        if (pick < 0) {
            pick = oldest([this](const Space & s) { return !protected_.count(s.conv); });
            if (pick >= 0) ++stats_.live_evicted;
        }
        if (pick < 0) { ++stats_.refused; return -1; }
        spaces_[size_t(pick)] = {true, false, conv, ++clock_};
        ++stats_.parked;
        return pick;
    }
    // the space holding the conversation, freed (the caller copies it into a slot first); -1 if not parked
    int pickup(uint64_t conv) {
        const int s = find(conv);
        if (s < 0) return -1;
        spaces_[size_t(s)] = {};
        protected_.erase(conv);
        ++stats_.picked;
        return s;
    }
    int where(uint64_t conv) const { return find(conv); }
    // a request for this conversation is queued: its state is not evicted until picked up or released
    void protect(uint64_t conv) { protected_.insert(conv); }
    void release(uint64_t conv) { protected_.erase(conv); }
    // the conversation ended (or was abandoned): its state goes first when room is needed, still collectable until then
    void retire(uint64_t conv) { if (const int s = find(conv); s >= 0) spaces_[size_t(s)].dead = true; }
    int size() const { return int(spaces_.size()); }
    const Stats & stats() const { return stats_; }

private:
    struct Space { bool used = false, dead = false; uint64_t conv = 0, seq = 0; };
    int find(uint64_t conv) const {
        for (size_t i = 0; i < spaces_.size(); ++i) if (spaces_[i].used && spaces_[i].conv == conv) return int(i);
        return -1;
    }
    template <class F> int oldest(F ok) const {
        int pick = -1;
        for (size_t i = 0; i < spaces_.size(); ++i)
            if (spaces_[i].used && ok(spaces_[i]) && (pick < 0 || spaces_[i].seq < spaces_[size_t(pick)].seq)) pick = int(i);
        return pick;
    }
    std::vector<Space> spaces_;
    std::set<uint64_t> protected_;
    uint64_t clock_ = 0;
    Stats stats_;
};
}  // namespace tqllm
