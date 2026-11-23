// Host RAM expert tier (src/experts/host_expert_tier.hpp): block bookkeeping and parallel reads, no GPU.
//   T1 least recently used block goes first; a hit refreshes it
//   T2 a block a copy may still read (touched or released in an epoch that has not completed) is not reused without a fence
//   T3 blocks pinned in one cache call are never taken by it; with every block pinned a take is refused (-1)
//   T4 a released block is reused only after its epoch completes, and is then taken before more recently used blocks
//   T4b a released expert stays held until its block is needed (no copy back on a bank eviction, no second read)
//   T5 ensure() reads each missing expert once, in parallel, and a failed read leaves no block behind
//   T6 demote() takes a block only for an expert not held
//   T10 fit_tier_chunks: detaches from the last chunk until `need` is available, attaches while one more chunk leaves it
//   T9 elastic chunks: detached chunks hold nothing; an attached chunk's blocks live in its memory; detach drops exactly its
//      experts (the base blocks keep theirs) and refuses while one of its blocks is pinned
#include "host_expert_tier.hpp"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using tqllm::HostTierIndex; using tqllm::HostExpertTier;
static void require(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }
static const std::function<void()> no_fence;

static void lru_order() {
    HostTierIndex t(2, 8, 3);
    for (int e = 0; e < 3; ++e) require(t.take(0, e, no_fence) >= 0, "three free blocks");
    t.unpin_all(); t.complete(t.mark() + 1);  // the fills' copies are done
    require(t.find(0, 0) >= 0, "expert 0 held");  // 0 is now the most recent; 1 the least
    t.complete(t.mark() + 1);
    const int b = t.take(1, 5, no_fence);
    require(b >= 0 && t.peek(0, 1) < 0 && t.peek(0, 0) >= 0 && t.peek(0, 2) >= 0, "T1: the least recently used (expert 1) went");
}

static void unfenced_blocks_wait() {
    HostTierIndex t(1, 8, 2);
    require(t.take(0, 0, no_fence) >= 0 && t.take(0, 1, no_fence) >= 0, "two blocks");
    t.unpin_all(); t.mark();  // both were touched in the closed epoch: a copy may still read them
    int fences = 0;
    const int b = t.take(0, 2, [&] { ++fences; });
    require(b >= 0 && fences == 1 && t.stats().fences == 1, "T2: the reuse waited for one fence");
    t.unpin_all(); const uint64_t m = t.mark();
    t.complete(m + 1);  // everything up to the call before is done
    fences = 0; t.mark();
    t.find(0, 2);  // read in the open epoch again
    const int c = t.take(0, 3, [&] { ++fences; });
    require(c >= 0 && fences == 0 && t.peek(0, 2) >= 0, "T2: the block untouched since the completed epoch went without a fence");
    bool threw = false;
    try { t.unpin_all(); t.mark(); t.find(0, 2); t.find(0, 3); t.mark(); (void) t.take(0, 4, no_fence); } catch (const std::logic_error &) { threw = true; }
    require(threw, "T2: an unsafe reuse with no fence given is an error, not a silent overwrite");
}

// T2 a block released in the open epoch (the cache call still holds its copy on the host, unsubmitted) is never reused in
// that call, fence or not: a fence drains only what was submitted (one that also declared the open epoch done let the next
// call overwrite the block under a copy submitted after the fence: test-host-tier-cache, layer 1's bytes in a layer 0 slot)
static void fence_covers_submitted_only() {
    HostTierIndex t(1, 8, 2);
    require(t.take(0, 0, no_fence) >= 0 && t.take(0, 1, no_fence) >= 0, "two blocks");
    t.unpin_all(); t.mark();
    int fences = 0; const auto fence = [&] { ++fences; };
    require(t.find(0, 1) >= 0, "expert 1 read in this call");
    t.release(0, 0);  // its copy to the card is queued in this call
    require(t.take(0, 3, fence) == -1 && fences == 1, "T2: the block released in this call is not reused in it, even after a fence");
    t.unpin_all(); t.mark();  // next call: the released block's copy was submitted before it
    const int b = t.take(0, 4, fence);
    require(b >= 0 && fences == 2, "T2: the next call reuses it after a fence");
}

static void pins_hold() {
    HostTierIndex t(1, 8, 2);
    require(t.take(0, 0, no_fence) >= 0 && t.take(0, 1, no_fence) >= 0, "two blocks");
    require(t.take(0, 2, [] {}) == -1 && t.stats().refused == 1, "T3: every block pinned: refused");
    t.unpin_all(); t.complete(t.mark() + 1);
    t.pin(t.find(0, 0));  // this call's hit
    const int b = t.take(0, 2, no_fence);
    require(b >= 0 && t.peek(0, 0) >= 0 && t.peek(0, 1) < 0, "T3: the pinned hit stays, the other block is taken");
}

static void released_blocks() {
    HostTierIndex t(1, 8, 2);
    require(t.take(0, 0, no_fence) >= 0 && t.take(0, 1, no_fence) >= 0, "two blocks");
    t.unpin_all(); t.complete(t.mark() + 1);
    t.release(0, 0);  // its bytes are being copied to the card in the open epoch
    require(t.peek(0, 0) >= 0 && t.resident() == 2, "T4: released: still held while nothing needs the block");
    int fences = 0;
    const int b = t.take(0, 5, [&] { ++fences; });
    require(b >= 0 && fences == 0 && t.peek(0, 1) < 0 && t.peek(0, 0) >= 0,
            "T4: the release whose copy is unfinished is skipped, the safe resident block is evicted");
    t.unpin_all(); t.complete(t.mark() + 1);
    const int c = t.take(0, 6, no_fence);
    require(c >= 0 && t.peek(0, 0) < 0 && t.peek(0, 5) >= 0, "T4: a completed release is taken before a more recently used block");
}

// T4b inclusive while there is room: an expert released to the GPU stays in its block, so a later bank eviction of it
// demotes nothing (no device-to-host copy) and a later miss finds it without a drive read
static void released_stays() {
    std::vector<char> mem(4 * 64); int reads = 0;
    HostExpertTier tier(1, 8, std::vector<size_t>{64}, mem.data(), mem.size(), [&](int, int e, char * d) { ++reads; std::memset(d, e, 64); }, 1);
    tier.ensure(0, {3}, no_fence); tier.unpin_all();
    tier.release(0, 3);
    require(tier.demote(0, 3, no_fence) == nullptr && tier.stats().demotes == 0, "T4b: an evicted bank expert the tier still holds is not copied back");
    tier.ensure(0, {3}, no_fence); tier.unpin_all();
    require(reads == 1 && tier.find(0, 3)[63] == 3, "T4b: the released expert is found again without a read");
}

static void ensure_reads_once() {
    const int L = 2, E = 64, B = 40; const size_t bytes = 4096;
    std::vector<char> mem(size_t(B) * bytes);
    std::atomic<int> loads{0}; std::atomic<bool> fail{false};
    HostExpertTier tier(L, E, std::vector<size_t>(size_t(L), bytes), mem.data(), mem.size(), [&](int l, int e, char * dst) {
        if (fail && e == 13) throw std::runtime_error("drive error");
        ++loads; std::memset(dst, (l * E + e) & 127, bytes);
    }, 8);
    std::vector<int> ids; for (int e = 0; e < 32; ++e) ids.push_back(e);
    tier.ensure(1, ids, no_fence);
    require(loads == 32, "T5: 32 reads");
    for (int e : ids) { const char * p = tier.find(1, e); require(p && p[0] == char((E + e) & 127) && p[bytes - 1] == p[0], "T5: bytes in place"); }
    tier.ensure(1, ids, no_fence);
    require(loads == 32, "T5: held experts are not read again");
    tier.unpin_all(); tier.complete(tier.mark() + 1);
    fail = true; bool threw = false;
    try { tier.ensure(0, {12, 13, 14}, no_fence); } catch (const std::runtime_error &) { threw = true; }
    require(threw && tier.peek(0, 13) < 0 && tier.peek(0, 12) >= 0, "T5: the failed read holds no block, the others stay");
}

// T7 size classes: layers with different expert sizes get their own blocks, the bytes split by their share
static void size_classes() {
    const std::vector<size_t> sizes{100, 300, 100, 0};
    std::vector<char> mem(10000);
    require(HostExpertTier::blocks_for(sizes, mem.size()) == 40 + 20, "T7: the bytes split 2:3 (layer bytes per class): 40 blocks of 100, 20 of 300");
    HostExpertTier tier(4, 8, sizes, mem.data(), mem.size(), [](int l, int e, char * d) { std::memset(d, l * 8 + e, l == 1 ? 300 : 100); }, 2);
    require(tier.classes() == 2 && tier.block_bytes(1) == 300 && tier.block_bytes(2) == 100, "T7: two classes");
    tier.ensure(1, {0, 1}, no_fence); tier.ensure(2, {5}, no_fence);
    require(tier.find(1, 1)[299] == 9 && tier.find(2, 5)[99] == 21 && tier.find(0, 5) == nullptr, "T7: each layer's bytes in its class");
    bool threw = false; try { tier.find(3, 0); } catch (const std::out_of_range &) { threw = true; }
    require(threw, "T7: a layer the tier does not hold is an error");
}

static void demote_once() {
    std::vector<char> mem(4 * 64);
    HostExpertTier tier(1, 8, std::vector<size_t>{64}, mem.data(), mem.size(), [](int, int, char *) {}, 2);
    tier.ensure(0, {1}, no_fence);
    require(tier.demote(0, 1, no_fence) == nullptr, "T6: a held expert is not demoted twice");
    char * d = tier.demote(0, 2, no_fence);
    require(d && tier.peek(0, 2) >= 0 && tier.stats().demotes == 1, "T6: an evicted expert gets a block");
}

static void elastic_chunks() {
    const std::vector<size_t> sizes{100, 300};
    std::vector<char> base(2000), c0(2000), c1(2000);  // bytes split 1:3: 5 blocks of 100 + 5 of 300, in the base and per chunk
    HostExpertTier tier(2, 64, sizes, base.data(), base.size(), [](int l, int e, char * d) { std::memset(d, 1 + l * 64 + e, l ? 300 : 100); }, 2,
                        c0.size(), 2);
    require(tier.chunks() == 2 && tier.blocks() == 10 && tier.attached_chunks() == 0, "T9: the chunks start detached (10 base blocks)");
    tier.attach(0, c0.data());
    require(tier.blocks() == 20 && tier.attached(0) && !tier.attached(1), "T9: chunk 0 adds its 10 blocks");
    std::vector<int> ids; for (int e = 0; e < 10; ++e) ids.push_back(e);
    tier.ensure(0, ids, no_fence); tier.unpin_all();
    int held = 0, in_chunk = 0;
    for (int e = 0; e < 10; ++e) if (const char * p = tier.find(0, e)) {
        ++held; require(p[99] == char(1 + e), "T9: the bytes are the expert's");
        in_chunk += p >= c0.data() && p < c0.data() + c0.size();
    }
    require(held == 10 && in_chunk == 5, "T9: 10 layer-0 experts in 5 base + 5 chunk blocks");
    tier.ensure(0, {0}, no_fence);  // pinned
    bool threw = false; try { (void) tier.detach(0); } catch (const std::logic_error &) { threw = true; }
    require(threw, "T9: a chunk with a pinned block is not detached");
    tier.unpin_all();
    require(tier.detach(0) == c0.data() && tier.blocks() == 10 && tier.attached_chunks() == 0, "T9: detach returns the chunk's memory");
    held = 0; for (int e = 0; e < 10; ++e) held += tier.peek(0, e) >= 0;
    require(held == 5, "T9: the base blocks keep their 5 experts, the chunk's 5 are gone");
    for (int e = 10; e < 30; ++e) { tier.complete(tier.mark() + 1); tier.ensure(0, {e}, no_fence); tier.unpin_all(); }
    held = 0;
    for (int e = 0; e < 30; ++e) if (const char * p = tier.find(0, e)) {
        ++held; require(p >= base.data() && p < base.data() + base.size(), "T9: nothing lands in a detached chunk"); }
    require(held == 5, "T9: layer 0 holds its 5 base blocks");
    tier.attach(1, c1.data()); tier.ensure(1, {3, 4, 5}, no_fence); tier.unpin_all();
    require(tier.find(1, 5)[299] == char(1 + 64 + 5) && tier.blocks() == 20, "T9: chunk 1 in use (layer 1's class too)");
}

static void fit_chunks() {
    const std::vector<size_t> sizes{100};
    std::vector<char> base(1000); std::vector<std::vector<char>> mem;
    HostExpertTier tier(1, 64, sizes, base.data(), base.size(), [](int, int e, char * d) { std::memset(d, e, 100); }, 1, 1000, 3);
    long avail = 3500; int allocs = 0, frees = 0;
    auto available = [&] { return uint64_t(avail); };
    auto alloc = [&](size_t b) -> char * { if (long(b) > avail) return nullptr; avail -= long(b); ++allocs; mem.emplace_back(b); return mem.back().data(); };
    auto release = [&](char *) { avail += 1000; ++frees; };
    require(fit_tier_chunks(tier, 1500, available, alloc, release) == 2 && avail == 1500, "T10: two chunks attached, 1500 left");
    require(fit_tier_chunks(tier, 1500, available, alloc, release) == 2 && allocs == 2 && frees == 0, "T10: a second fit changes nothing");
    require(fit_tier_chunks(tier, 2200, available, alloc, release) == 1 && avail == 2500 && tier.attached(0) && !tier.attached(1),
            "T10: a long prompt's need detaches the last attached chunk");
    require(fit_tier_chunks(tier, 3000, available, alloc, release) == 0 && avail == 3500, "T10: a larger need detaches the rest");
    avail = 100;
    require(fit_tier_chunks(tier, 500, available, alloc, release) == 0, "T10: nothing to detach, nothing attached");
}

// T11 warm_tier fills every free block, each layer its class's share, reading each expert once and evicting nothing
static void warm_start() {
    const std::vector<size_t> sizes{100, 100, 300};
    std::vector<char> mem(2000); int reads = 0;
    HostExpertTier tier(3, 64, sizes, mem.data(), mem.size(), [&](int l, int e, char * d) { ++reads; std::memset(d, 1 + l * 64 + e, l == 2 ? 300 : 100); }, 2);
    const int room0 = tier.room(0) + tier.room(2);
    require(warm_tier(tier, {1, 1, 1}, 64, 4) == room0 && reads == room0, "T11: every free block read once");
    require(tier.room(0) == 0 && tier.room(1) == 0 && tier.room(2) == 0 && tier.index_stats().evictions == 0, "T11: full, nothing evicted");
    int l0 = 0, l1 = 0; for (int e = 0; e < 64; ++e) { l0 += tier.peek(0, e) >= 0; l1 += tier.peek(1, e) >= 0; }
    require(l0 > 0 && l1 > 0 && (l0 - l1 <= 4 && l1 - l0 <= 4), "T11: the two layers of one class share it");
    require(tier.find(2, 0) && tier.find(2, 0)[299] == char(1 + 128), "T11: the bytes are the expert's");
    require(warm_tier(tier, {1, 1, 1}, 64, 4) == 0, "T11: a full tier reads nothing");
}

int main() {
    try {
        lru_order(); unfenced_blocks_wait(); fence_covers_submitted_only(); pins_hold(); released_blocks(); released_stays(); ensure_reads_once(); size_classes(); demote_once();
        elastic_chunks(); fit_chunks(); warm_start();
    } catch (const std::exception & e) { std::fprintf(stderr, "host_expert_tier: FAIL %s\n", e.what()); return 1; }
    std::printf("host_expert_tier: ok\n");
    return 0;
}
