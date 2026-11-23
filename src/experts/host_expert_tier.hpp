#pragma once
// Host RAM tier of expert blocks: experts read from the GGUF on demand (a box whose RAM cannot pin every expert) are kept
// here in the expert banks' row layout (repacked and tiled once), so a bank miss is a DMA out of pinned memory instead of
// a drive read and a CPU repack. Exclusive with the GPU banks: an expert copied into a bank gives its block back, and an
// expert a bank evicts is copied back here (device to host), so RAM holds what the card does not (replay of the 11-20
// .77 decode routing, 17 GB tier beside 177 slots/layer: 13-16 drive reads per step on chats instead of 32-44 for a tier
// that keeps what it read).
//
// HostTierIndex is the bookkeeping (no CUDA, no threads; tests/host_expert_tier.cpp). A block is reused only once every
// copy that read or wrote it has completed: the caller opens an epoch (mark()) at the start of every batch of copies (a
// cache call) and records events on its copy stream (complete() once an event has passed covers the epochs before its
// mark); a block touched in an earlier epoch whose copies may still fly is reused only after fence() (the caller drains
// what it submitted). A block touched in the open epoch is never reused in it (its copy may not be submitted yet).
// Blocks pinned during one cache call are never taken by that call.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {

class HostTierIndex {
public:
    struct Stats { uint64_t hits = 0, misses = 0, taken_free = 0, evictions = 0, fences = 0, released = 0, refused = 0, retired = 0; };
    HostTierIndex(int layers, int experts, int blocks)
        : layers_(layers), experts_(experts), where_(size_t(layers) * size_t(experts), -1), blocks_(size_t(blocks)) {
        if (layers < 1 || experts < 1 || blocks < 0) throw std::invalid_argument("host tier: invalid shape");
        for (int b = 0; b < blocks; ++b) free_.push_back({b, 0});
    }
    int blocks() const { return int(blocks_.size()); }
    int active() const { return int(blocks_.size()) - off_; }
    int resident() const { return resident_; }
    // the block holding (layer, expert), -1 if none; a hit is touched (most recently used, read in this epoch)
    int find(int layer, int expert) {
        const int b = where_[key(layer, expert)];
        if (b < 0) { ++stats_.misses; return -1; }
        touch(b); ++stats_.hits; return b;
    }
    int peek(int layer, int expert) const { return where_[key(layer, expert)]; }
    void touch(int b) { auto & x = blocks_.at(size_t(b)); x.used = epoch_; unlink(b); push_mru(b); }
    void pin(int b) { if (!blocks_.at(size_t(b)).pins++) pinned_.push_back(b); }
    void unpin_all() { for (int b : pinned_) blocks_[size_t(b)].pins = 0; pinned_.clear(); }
    // a block for (layer, expert), which must not be held: a free block whose release has completed, else the least
    // recently used unpinned block whose last copy has completed, else (after `fence`) any free or unpinned block. The
    // block comes back pinned and touched; -1 when every block is pinned.
    int take(int layer, int expert, const std::function<void()> & fence) {
        if (where_[key(layer, expert)] >= 0) throw std::logic_error("host tier: take of a held expert");
        int b = take_safe();
        if (b < 0 && any_unpinned()) {
            if (!fence) throw std::logic_error("host tier: a block needs a fence and none was given");
            // the fence drains the copies submitted so far: the epochs before the open one (the open epoch's copies may
            // still be queued on the host, unsubmitted: a block released in it stays unsafe)
            fence(); complete(epoch_); ++stats_.fences; b = take_safe();
        }
        if (b < 0) { ++stats_.refused; return -1; }
        auto & x = blocks_[size_t(b)];
        x.key = int64_t(key(layer, expert)); where_[size_t(x.key)] = b; ++resident_;
        touch(b); pin(b); return b;
    }
    // the expert went to the GPU: its block is the first one taken once the copy that read it has completed (exclusive
    // under pressure), but it keeps the expert until then, so a bank eviction finds it here and copies nothing back
    // (inclusive while the RAM has room: .188 in 64 GB holds 24,569 of 24,576 experts, and every eviction was a D2H copy)
    void release(int layer, int expert) {
        const int b = where_[key(layer, expert)]; if (b < 0) return;
        auto & x = blocks_[size_t(b)];
        x.used = epoch_; unlink(b); push_lru(b); ++stats_.released;
    }
    // a block whose fill failed: free at once (nothing read it)
    void drop(int b) {
        auto & x = blocks_.at(size_t(b)); if (x.key < 0) return;
        unlink(b); where_[size_t(x.key)] = -1; x.key = -1; --resident_; free_.push_front({b, 0});
    }
    // blocks whose memory is handed back (the elastic part of the tier, given to a long prompt's host buffers): their
    // experts dropped, never taken until restored. The caller has drained every copy (device sync) and holds no pins.
    void retire(const std::vector<int> & bs) {
        std::vector<char> gone(blocks_.size(), 0);
        for (int b : bs) {
            auto & x = blocks_.at(size_t(b)); if (x.off) continue;
            if (x.pins) throw std::logic_error("host tier: retire of a pinned block");
            if (x.key >= 0) { unlink(b); where_[size_t(x.key)] = -1; x.key = -1; --resident_; }
            x.off = true; ++off_; gone[size_t(b)] = 1; ++stats_.retired;
        }
        free_.erase(std::remove_if(free_.begin(), free_.end(), [&](const Freed & f) { return gone[size_t(f.block)] != 0; }), free_.end());
    }
    // retired blocks back in use with fresh memory (no copy ever touched it): free at once
    void restore(const std::vector<int> & bs) {
        for (int b : bs) { auto & x = blocks_.at(size_t(b)); if (!x.off) continue; x.off = false; x.used = 0; --off_; free_.push_front({b, 0}); }
    }
    bool retired(int b) const { return blocks_.at(size_t(b)).off; }
    // epochs: mark() closes the current epoch (the caller records an event on its copy stream right after); complete(e):
    // every copy issued in epochs before e has finished
    uint64_t mark() { return epoch_++; }
    void complete(uint64_t through) { if (through > safe_) safe_ = through; }
    uint64_t epoch() const { return epoch_; }
    const Stats & stats() const { return stats_; }
    std::pair<int, int> key_of(int b) const {
        const int64_t k = blocks_.at(size_t(b)).key; return k < 0 ? std::make_pair(-1, -1) : std::make_pair(int(k / experts_), int(k % experts_));
    }

private:
    struct Block { int64_t key = -1; uint64_t used = 0; int pins = 0, prev = -1, next = -1; bool linked = false, off = false; };
    struct Freed { int block; uint64_t at; };
    size_t key(int layer, int expert) const {
        if (layer < 0 || layer >= layers_ || expert < 0 || expert >= experts_) throw std::out_of_range("host tier: expert out of range");
        return size_t(layer) * size_t(experts_) + size_t(expert);
    }
    bool safe(const Block & x) const { return x.used < safe_; }
    bool any_unpinned() const {
        for (const auto & f : free_) if (!blocks_[size_t(f.block)].pins) return true;
        for (int b = lru_; b >= 0; b = blocks_[size_t(b)].next) if (!blocks_[size_t(b)].pins) return true;
        return false;
    }
    int take_safe() {
        for (auto it = free_.begin(); it != free_.end() && it->at < safe_; ++it)  // freed in release order: the safe ones lead
            if (!blocks_[size_t(it->block)].pins) { const int b = it->block; free_.erase(it); ++stats_.taken_free; return b; }
        for (int b = lru_; b >= 0; b = blocks_[size_t(b)].next) {
            auto & x = blocks_[size_t(b)];
            if (x.pins || !safe(x)) continue;
            unlink(b); where_[size_t(x.key)] = -1; x.key = -1; --resident_; ++stats_.evictions;
            return b;
        }
        return -1;
    }
    void unlink(int b) {
        auto & x = blocks_[size_t(b)]; if (!x.linked) return;
        if (x.prev >= 0) blocks_[size_t(x.prev)].next = x.next; else lru_ = x.next;
        if (x.next >= 0) blocks_[size_t(x.next)].prev = x.prev; else mru_ = x.prev;
        x.prev = x.next = -1; x.linked = false;
    }
    void push_lru(int b) {
        auto & x = blocks_[size_t(b)]; if (x.key < 0) return;
        x.next = lru_; x.prev = -1; x.linked = true;
        if (lru_ >= 0) blocks_[size_t(lru_)].prev = b; else mru_ = b;
        lru_ = b;
    }
    void push_mru(int b) {
        auto & x = blocks_[size_t(b)]; if (x.key < 0) return;
        x.prev = mru_; x.next = -1; x.linked = true;
        if (mru_ >= 0) blocks_[size_t(mru_)].next = b; else lru_ = b;
        mru_ = b;
    }
    int layers_, experts_;
    std::vector<int> where_;
    std::vector<Block> blocks_;
    std::deque<Freed> free_;
    std::vector<int> pinned_;
    int lru_ = -1, mru_ = -1, resident_ = 0, off_ = 0;
    uint64_t epoch_ = 1, safe_ = 1;
    Stats stats_;
};

// The tier's memory and its drive reads. Layers whose experts have the same bank-layout size share one size class (an
// index and a slice of the memory, the slices in proportion to their layers' expert bytes): one block size for every
// layer wasted 30 % of the RAM on Flash-Next GSQ IQ3_S (1.5-2.66 MB experts). `load(layer, expert, dst)` writes one
// expert's bank-layout bytes (thread-safe: up to `readers` run at once).
// Elastic part: `chunks` chunks of `chunk_bytes` (each split over the classes as the base memory is) that start detached;
// attach(c, memory) puts chunk c's blocks in use, detach(c) drops their experts and hands the memory back (the caller frees
// it). A box whose RAM the tier shares with a long prompt's host buffers (31 GB .77: K/V tail and rows of a 128K prompt)
// holds more experts in decode and gives the room back for a long prompt.
class HostExpertTier {
public:
    using Load = std::function<void(int layer, int expert, char * dst)>;
    struct Stats { uint64_t loads = 0, demotes = 0, stream_reads = 0; double load_ms = 0; };
    // layer_bytes[l]: layer l's expert size (0: a layer the tier never holds); `memory` of `bytes` (caller-owned, pinned)
    HostExpertTier(int layers, int experts, const std::vector<size_t> & layer_bytes, char * memory, size_t bytes, Load load, int readers,
                   size_t chunk_bytes = 0, int chunks = 0)
        : layers_(layers), experts_(experts), load_(std::move(load)), readers_(std::max(1, readers)), class_of_(size_t(layers), -1),
          chunk_bytes_(chunk_bytes), chunk_mem_(size_t(std::max(0, chunks)), nullptr) {
        if (!load_ || int(layer_bytes.size()) != layers) throw std::invalid_argument("host tier: no loader or layer sizes");
        std::vector<size_t> sizes; std::vector<double> share;
        for (int l = 0; l < layers; ++l) {
            if (!layer_bytes[size_t(l)]) continue;
            auto it = std::find(sizes.begin(), sizes.end(), layer_bytes[size_t(l)]);
            if (it == sizes.end()) { sizes.push_back(layer_bytes[size_t(l)]); share.push_back(0); it = sizes.end() - 1; }
            const size_t k = size_t(it - sizes.begin()); class_of_[size_t(l)] = int(k); share[k] += double(layer_bytes[size_t(l)]);
        }
        double total = 0; for (double x : share) total += x;
        char * at = memory; size_t chunk_at = 0;
        for (size_t k = 0; k < sizes.size(); ++k) {
            const size_t room = total > 0 ? size_t(double(bytes) * share[k] / total) : 0;
            const int blocks = int(room / sizes[k]);
            const int per_chunk = total > 0 ? int(size_t(double(chunk_bytes) * share[k] / total) / sizes[k]) : 0;
            const int all = blocks + per_chunk * int(chunk_mem_.size());
            classes_.push_back(std::make_unique<Class>(Class{HostTierIndex(layers, experts, all), sizes[k], at, blocks, per_chunk, chunk_at}));
            at += size_t(blocks) * sizes[k]; chunk_at += size_t(per_chunk) * sizes[k];
            std::vector<int> elastic; for (int b = blocks; b < all; ++b) elastic.push_back(b);
            classes_.back()->index.retire(elastic);
        }
        if (bytes && !memory) throw std::invalid_argument("host tier: no memory");
    }
    int chunks() const { return int(chunk_mem_.size()); }
    size_t chunk_bytes() const { return chunk_bytes_; }
    bool attached(int chunk) const { return chunk_mem_.at(size_t(chunk)) != nullptr; }
    int attached_chunks() const { int n = 0; for (char * m : chunk_mem_) n += m != nullptr; return n; }
    // chunk `chunk`'s blocks in use, in `memory` (chunk_bytes, caller-owned, pinned)
    void attach(int chunk, char * memory) {
        if (!memory || attached(chunk)) throw std::logic_error("host tier: attach of an attached chunk or of no memory");
        chunk_mem_[size_t(chunk)] = memory;
        for (auto & c : classes_) c->index.restore(chunk_blocks(*c, chunk));
    }
    // chunk `chunk`'s experts dropped and its memory returned; no copy may still touch it (the caller drained them) and no
    // block may be pinned
    char * detach(int chunk) {
        char * m = chunk_mem_.at(size_t(chunk)); if (!m) return nullptr;
        for (auto & c : classes_) c->index.retire(chunk_blocks(*c, chunk));
        chunk_mem_[size_t(chunk)] = nullptr; return m;
    }
    // how many blocks `bytes` gives these layer sizes (what the constructor will make)
    static int blocks_for(const std::vector<size_t> & layer_bytes, size_t bytes) {
        std::vector<size_t> sizes; std::vector<double> share; double total = 0;
        for (size_t b : layer_bytes) if (b) {
            auto it = std::find(sizes.begin(), sizes.end(), b);
            if (it == sizes.end()) { sizes.push_back(b); share.push_back(0); it = sizes.end() - 1; }
            share[size_t(it - sizes.begin())] += double(b); total += double(b);
        }
        int n = 0; for (size_t k = 0; k < sizes.size(); ++k) n += int(size_t(double(bytes) * share[k] / total) / sizes[k]);
        return n;
    }
    size_t block_bytes(int layer) const { return cls(layer).bytes; }
    // the expert's bytes if held (touched), else null
    const char * find(int layer, int expert) { auto & c = cls(layer); const int b = c.index.find(layer, expert); return b < 0 ? nullptr : block(c, b); }
    int peek(int layer, int expert) const { return cls(layer).index.peek(layer, expert); }
    // every id held and pinned until unpin_all(): the missing ones read from the drive in parallel. An id that finds no
    // block (every block pinned) stays missing: find() returns null for it and the caller takes its own path.
    void ensure(int layer, const std::vector<int> & ids, const std::function<void()> & fence) {
        auto & c = cls(layer);
        std::vector<std::pair<int, int>> jobs;  // (expert, block)
        for (int id : ids) {
            int b = c.index.find(layer, id);
            if (b >= 0) { c.index.pin(b); continue; }
            b = c.index.take(layer, id, fence);
            if (b >= 0) jobs.push_back({id, b});
        }
        run(jobs.size(), [&](size_t i) { load_(layer, jobs[i].first, block(c, jobs[i].second)); }, [&](size_t i) { c.index.drop(jobs[i].second); });
        std::lock_guard<std::mutex> hold(stat_lock_); stats_.loads += jobs.size();
    }
    // reads experts straight into caller memory (a prefill stream ring), the tier unchanged
    void read_into(int layer, const std::vector<int> & ids, const std::vector<char *> & dst) {
        if (ids.size() != dst.size()) throw std::invalid_argument("host tier: read_into sizes differ");
        run(ids.size(), [&](size_t i) { load_(layer, ids[i], dst[i]); }, nullptr);
        std::lock_guard<std::mutex> hold(stat_lock_); stats_.stream_reads += ids.size();
    }
    // a block for an expert a bank is evicting (the caller copies its device bytes in); null when held already or no
    // block is free of pins
    char * demote(int layer, int expert, const std::function<void()> & fence) {
        auto & c = cls(layer);
        if (c.index.peek(layer, expert) >= 0) return nullptr;
        const int b = c.index.take(layer, expert, fence);
        if (b < 0) return nullptr;
        { std::lock_guard<std::mutex> hold(stat_lock_); ++stats_.demotes; } return block(c, b);
    }
    void release(int layer, int expert) { cls(layer).index.release(layer, expert); }
    void unpin_all() { for (auto & c : classes_) c->index.unpin_all(); }
    uint64_t mark() { uint64_t m = 0; for (auto & c : classes_) m = c->index.mark(); return m; }
    void complete(uint64_t through) { for (auto & c : classes_) c->index.complete(through); }
    // every held block: f(layer, expert, bytes)
    template <class F> void for_each(F && f) const {
        for (const auto & c : classes_) for (int b = 0; b < c->index.blocks(); ++b) {
            const auto [l, e] = c->index.key_of(b); if (l >= 0) f(l, e, static_cast<const char *>(block(*c, b)));
        }
    }
    int blocks() const { int n = 0; for (const auto & c : classes_) n += c->index.active(); return n; }
    // blocks of layer's size class still free (taking one evicts nothing)
    int room(int layer) const { const auto & c = cls(layer); return c.index.active() - c.index.resident(); }
    int resident() const { int n = 0; for (const auto & c : classes_) n += c->index.resident(); return n; }
    int classes() const { return int(classes_.size()); }
    HostTierIndex::Stats index_stats() const {
        HostTierIndex::Stats t;
        for (const auto & c : classes_) { const auto & x = c->index.stats(); t.hits += x.hits; t.misses += x.misses; t.taken_free += x.taken_free;
            t.evictions += x.evictions; t.fences += x.fences; t.released += x.released; t.refused += x.refused; }
        return t;
    }
    const Stats & stats() const { return stats_; }

private:
    // blocks [0, base_blocks) in the base memory, then per_chunk blocks per elastic chunk (at chunk_at inside each chunk)
    struct Class { HostTierIndex index; size_t bytes; char * base; int base_blocks, per_chunk; size_t chunk_at; };
    char * block(const Class & c, int b) const {
        if (b < c.base_blocks) return c.base + size_t(b) * c.bytes;
        const int k = (b - c.base_blocks) / c.per_chunk, i = (b - c.base_blocks) % c.per_chunk;
        return chunk_mem_[size_t(k)] + c.chunk_at + size_t(i) * c.bytes;
    }
    static std::vector<int> chunk_blocks(const Class & c, int chunk) {
        std::vector<int> v; for (int i = 0; i < c.per_chunk; ++i) v.push_back(c.base_blocks + chunk * c.per_chunk + i); return v;
    }
    Class & cls(int layer) {
        const int k = layer >= 0 && layer < layers_ ? class_of_[size_t(layer)] : -1;
        if (k < 0) throw std::out_of_range("host tier: a layer the tier does not hold");
        return *classes_[size_t(k)];
    }
    const Class & cls(int layer) const { return const_cast<HostExpertTier *>(this)->cls(layer); }
    // runs n jobs on up to readers_ threads (the calling thread included); a failed job runs `undo` and the first error
    // is rethrown after every job has finished
    void run(size_t n, const std::function<void(size_t)> & job, const std::function<void(size_t)> & undo) {
        if (!n) return;
        const auto t0 = std::chrono::steady_clock::now();
        std::atomic<size_t> next{0}; std::exception_ptr error; std::mutex lock; std::vector<size_t> failed;
        auto work = [&] {
            for (size_t i; (i = next++) < n;) {
                try { job(i); } catch (...) { std::lock_guard<std::mutex> g(lock); if (!error) error = std::current_exception(); failed.push_back(i); }
            }
        };
        std::vector<std::thread> pool;
        for (size_t t = 1; t < std::min<size_t>(n, size_t(readers_)); ++t) pool.emplace_back(work);
        work();
        for (auto & t : pool) t.join();
        { std::lock_guard<std::mutex> hold(stat_lock_); stats_.load_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
        if (undo) for (size_t i : failed) undo(i);
        if (error) std::rethrow_exception(error);
    }
    int layers_, experts_;
    Load load_;
    int readers_;
    std::vector<int> class_of_;
    std::vector<std::unique_ptr<Class>> classes_;
    size_t chunk_bytes_;
    std::vector<char *> chunk_mem_;
    Stats stats_; std::mutex stat_lock_;  // read_into runs on a prefetch thread beside the cache's reads (the index is the caller's thread only)
};

// The tier's elastic chunks follow the RAM left available (decode_rules.hpp host_tier_keep): chunks are detached (their
// memory freed) until `need` bytes are available, then detached chunks attached again while one more still leaves `need`.
// The caller has drained every copy (device sync) and holds no tier pins. Returns the chunks attached.
template <class Available, class Alloc, class Free>
inline int fit_tier_chunks(HostExpertTier & tier, uint64_t need, Available available, Alloc alloc, Free release) {
    for (int k = tier.chunks() - 1; k >= 0 && available() < need; --k)
        if (tier.attached(k)) release(tier.detach(k));
    for (int k = 0; k < tier.chunks(); ++k) {
        if (tier.attached(k)) continue;
        if (available() < need + tier.chunk_bytes()) break;
        char * m = alloc(tier.chunk_bytes()); if (!m) break;
        tier.attach(k, m);
    }
    return tier.attached_chunks();
}

// Fill the tier's free blocks at start (a cold tier made the first requests read every expert from the drive: .188 in a
// 64 GB scope ran its first short prompts at 24-60 tok/s and the same prompts warm at 67-80). Layers take turns, `step`
// experts each, so every layer gets its share of its size class; nothing is evicted. owned[l]: the tier holds layer l.
// Returns the experts read.
inline int warm_tier(HostExpertTier & tier, const std::vector<char> & owned, int experts, int step = 64) {
    int read = 0;
    for (int start = 0; start < experts; start += step) {
        bool any = false;
        for (int l = 0; l < int(owned.size()); ++l) {
            if (!owned[size_t(l)]) continue;
            const int n = std::min({step, experts - start, tier.room(l)});
            if (n <= 0) continue;
            std::vector<int> ids; for (int i = 0; i < n; ++i) ids.push_back(start + i);
            tier.ensure(l, ids, [] {}); tier.unpin_all(); read += n; any = true;
        }
        if (!any) break;
    }
    return read;
}

}  // namespace tqllm
