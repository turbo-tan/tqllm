// TQueue / TPark rules (plans/tqueue-20261013.md, "Rules spec"): the host-to-device expert copy queue and the VRAM
// overflow park, as a CPU model with no GPU. Expectations:
//   E1 a demand miss waits behind at most max_inflight copies already on the link, never behind queued guesses
//   E2 a guess whose layer has run is dropped if its copy has not started (expiry)
//   E3 a full park evicts dead letters first (oldest first), then the lowest-priority / furthest live ticket, never a
//      ticket of the running layer; with nothing evictable it refuses the copy
//   E4 a guess that becomes a demand is copied once: a queued guess is promoted, an in-flight or parked one is waited
//      for / picked up
//   E5 a ticket whose layer passed is a dead letter: still picked up by a later step (late hit)
#include "tqueue.hpp"
#include "expert_prediction.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using tqllm::TQueue; using tqllm::TPark; using tqllm::TQKey;

static void require(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }

static void demand_jumps_queued_guesses() {
    TQueue q(1);
    for (int e = 0; e < 20; ++e) q.push({5, e}, 1);  // twenty next-layer guesses
    require(q.start().has_value(), "a guess starts on the idle link");
    q.push({4, 99}, 0);  // the running layer misses expert 99
    const auto d = q.start();
    require(d && d->key == TQKey{4, 99} && d->priority == 0, "the demand goes on now, behind the one flying copy, ahead of 19 queued guesses");
    require(!q.start(), "the link is full: no guess starts");
}

// E1 on the engine's path: the demand is issued at once (demand()), behind only what is on the link
static void demand_goes_behind_flying_copies_only() {
    TQueue q(2);
    for (int e = 0; e < 20; ++e) q.push({5, e}, 1);
    require(q.start() && q.start() && !q.start(), "two guesses fly, the third waits for room");
    require(q.demand({5, 7}), "a demand for a queued guess goes on the link now");
    require(q.flying() == 2, "the demand waits behind the two flying copies, not the 17 queued ones");
    require(q.stats().promoted == 1 && q.queued() == 17, "its queued guess is gone: one copy");
    require(!q.demand({5, 0}) && q.stats().shared == 1, "a demand for a flying guess shares that copy");
    require(q.demand({4, 3}) && q.stats().demands == 2 && q.stats().promoted == 1, "a demand nobody guessed is a plain copy");
}

static void known_experts_are_not_held_back() {
    TQueue q(2);
    q.push({3, 1}, 1); q.push({3, 2}, 1);
    require(q.start() && q.start(), "two guesses fly");
    for (int e = 10; e < 18; ++e) q.push({4, e}, 0);  // the next token's own experts (GLM prefetch_released)
    int known = 0; while (q.start()) ++known;
    require(known == 8, "P0 entries start beside the flying guesses, all of them");
    q.push({4, 30}, 1);
    require(!q.start(), "a guess behind them still waits for room");
}

static void abandoned_start_frees_the_link() {
    TQueue q(1);
    q.push({5, 1}, 1); q.push({5, 2}, 1);
    const auto a = q.start(); require(a && !q.start(), "one flies");
    q.abandon(a->key);  // the park had no space for it
    const auto b = q.start(); require(b && b->key == TQKey{5, 2}, "the next guess takes the place");
    require(q.stats().started == 1, "an abandoned start is not a copy");
}

static void trace_keeps_the_last_entries() {
    tqllm::TQTrace t;
    for (int i = 0; i < 5000; ++i) t.note(tqllm::TQTrace::Started, {1, i}, 0, nullptr, nullptr, 8);
    char buf[1 << 19]; std::FILE * f = fmemopen(buf, sizeof buf, "w"); t.dump(f, "test"); std::fclose(f);
    const std::string s(buf);
    require(s.find("last 4096 of 5000") != std::string::npos, "the header counts kept and seen entries");
    require(s.find("E4999") != std::string::npos && s.find("E903 ") == std::string::npos, "newest kept, oldest overwritten");
}

static void priority_then_layer_order() {
    TQueue q(4);
    q.push({7, 1}, 2); q.push({6, 2}, 1); q.push({5, 3}, 1); q.push({4, 4}, 0);
    std::vector<TQKey> got; while (auto s = q.start()) got.push_back(s->key);
    require(got == std::vector<TQKey>{{4, 4}, {5, 3}, {6, 2}, {7, 1}}, "copies start by priority, then nearest layer");
}

static void guesses_expire_when_their_layer_ran() {
    TQueue q(1);
    q.push({3, 1}, 1); q.push({3, 2}, 1); q.push({5, 7}, 1);
    const auto first = q.start(); require(first && first->key == TQKey{3, 1}, "first guess on the link");
    q.layer_ran(3);  // layer 3 computed: its queued guess {3,2} is useless now; {3,1} is already copying
    require(q.queued() == 1 && q.stats().expired == 1, "the not-started guess of the layer that ran is dropped");
    q.complete();
    const auto s = q.start(); require(s && s->key == TQKey{5, 7}, "a later layer's guess stays");
}

static void promoted_guess_copied_once() {
    TQueue q(1);
    q.push({2, 9}, 1); q.push({2, 8}, 2);
    q.push({2, 8}, 0);  // the weaker guess becomes a demand
    require(q.queued() == 2 && q.stats().promoted == 1, "promoted in place, not queued twice");
    const auto s = q.start(); require(s && s->key == TQKey{2, 8} && s->priority == 0, "the promoted entry copies first");
    q.push({2, 8}, 0);  // demanded again while on the link
    require(q.queued() == 1 && q.in_flight(TQKey{2, 8}), "an in-flight copy is waited for, not queued again");
}

static void park_evicts_dead_then_weakest_never_running() {
    TPark park(3);
    require(park.park({4, 1}, 1) >= 0 && park.park({5, 2}, 2) >= 0 && park.park({6, 3}, 1) >= 0, "three free spaces");
    park.set_running(4); park.layer_passed(4);  // layer 4 ran without picking up {4,1}: a dead letter
    park.set_running(5);
    require(park.park({7, 4}, 1) >= 0 && !park.holds({4, 1}) && park.holds({7, 4}), "a dead letter is overwritten first");
    // full, no dead letter: {5,2} is the running layer's (kept even at the weaker priority 2); of the live P1 tickets
    // {6,3} and {7,4}, the furthest layer goes
    require(park.park({8, 5}, 1) >= 0, "a live ticket makes room");
    require(park.holds({5, 2}) && park.holds({6, 3}) && park.holds({8, 5}) && !park.holds({7, 4}),
            "the furthest of the weakest live tickets goes; the running layer's stays");
    TPark weak(2); weak.set_running(1);
    weak.park({2, 1}, 1); weak.park({3, 2}, 2);
    require(weak.park({4, 3}, 1) >= 0 && weak.holds({2, 1}) && !weak.holds({3, 2}), "a weaker priority goes before a nearer layer");
    TPark full(1); full.set_running(1); full.park({1, 1}, 1);
    require(full.park({2, 2}, 0) < 0 && full.holds({1, 1}), "only the running layer's ticket: the park refuses the copy");
}

static void dead_letter_late_hit() {
    TPark park(2);
    park.park({3, 5}, 1);
    park.layer_passed(3);
    require(park.dead() == 1, "unpicked ticket of a passed layer is a dead letter");
    const int s = park.pickup({3, 5});
    require(s >= 0 && park.stats().late_hits == 1 && !park.holds({3, 5}), "a later step still picks it up (late hit)");
    require(park.pickup({3, 5}) < 0, "picked up once");
}

static void dead_letters_oldest_first() {
    TPark park(2);
    park.park({1, 1}, 1); park.layer_passed(1);
    park.park({2, 2}, 1); park.layer_passed(2);
    park.park({5, 3}, 1);
    require(park.holds({2, 2}) && !park.holds({1, 1}), "the oldest dead letter goes first");
}

// E6 the engine's guess pipeline (expert_prediction.hpp) holds a whole layer's demand misses: the pipeline depth is what
// decides whether a guessed miss is on the link before the layer that needs it runs. Measured on the 3090 at 16 GB, 25.3K
// decode: depth 2 copied 3,426 guesses per decode and padded 2,298 pickups, depth 6 copied 6,958 and picked up 4,557.
static void guess_pipeline_covers_a_layer() {
    require(tqllm::kGuessLinkDepth >= tqllm::kLayerDemandMisses, "the link holds one layer's guessed misses at once");
    require(tqllm::kGuessParkSpaces >= tqllm::kLayerDemandMisses, "so does the park");
    TQueue q(tqllm::kGuessLinkDepth);
    TPark park(tqllm::kGuessParkSpaces);
    for (int i = 0; i < tqllm::kLayerDemandMisses; ++i) q.push({1, i}, 1);  // the next layer's guesses, in rank order
    int started = 0;
    while (const auto st = q.start()) { require(park.park(st->key, st->priority) >= 0, "every guess that starts has a park space"); ++started; }
    require(started == tqllm::kLayerDemandMisses, "one layer's guesses all go on the link together");
    require(park.stats().refused == 0, "the park takes them all");
    for (int i = 0; i < tqllm::kLayerDemandMisses; ++i) {
        require(!q.demand({1, i}), "a guess already on the link is waited for, not copied a second time");
        require(park.pickup({1, i}) >= 0, "and lands in its slot out of the park");
    }
}

int main() {
    const struct { const char * name; void (*run)(); } cases[] = {
        {"demand_jumps_queued_guesses", demand_jumps_queued_guesses},
        {"demand_goes_behind_flying_copies_only", demand_goes_behind_flying_copies_only},
        {"known_experts_are_not_held_back", known_experts_are_not_held_back},
        {"abandoned_start_frees_the_link", abandoned_start_frees_the_link},
        {"trace_keeps_the_last_entries", trace_keeps_the_last_entries},
        {"priority_then_layer_order", priority_then_layer_order},
        {"guesses_expire_when_their_layer_ran", guesses_expire_when_their_layer_ran},
        {"promoted_guess_copied_once", promoted_guess_copied_once},
        {"park_evicts_dead_then_weakest_never_running", park_evicts_dead_then_weakest_never_running},
        {"dead_letter_late_hit", dead_letter_late_hit},
        {"dead_letters_oldest_first", dead_letters_oldest_first},
        {"guess_pipeline_covers_a_layer", guess_pipeline_covers_a_layer},
    };
    int failed = 0;
    for (const auto & c : cases) {
        try { c.run(); std::printf("PASS %s\n", c.name); }
        catch (const std::exception & e) { ++failed; std::printf("FAIL %s: %s\n", c.name, e.what()); }
    }
    std::printf("%d of %zu cases failed\n", failed, sizeof(cases) / sizeof(cases[0]));
    return failed ? 1 : 0;
}
