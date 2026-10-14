#pragma once
#include "formats/weight_format.hpp"
#include "gguf_mapped.hpp"
#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace tqllm {

struct ExpertCacheStats {
    uint64_t hits = 0, misses = 0, evictions = 0, transfer_bytes = 0;
    size_t device_bytes = 0;
    // prefetch overlap (split mode): prepares whose bank still had copies queued, of those the ones whose copies were
    // still running on the GPU (event not complete), and prepares that waited for copies of experts they do not read
    uint64_t pf_pending = 0, pf_running = 0, pf_waited_unread = 0;
};

// One stream owns preparation and projections. Views are valid until the next prepare/clear.
// The source returns immutable raw TQ3_4S gate/up/down slices, not a materialized expert set.
class Tq3ExpertCache {
    struct State;
    std::unique_ptr<State> state_;
public:
    using Source = std::function<std::array<GgufBytes, 3>(int layer, int expert)>;
    Tq3ExpertCache(int layers, int experts, int hidden, int ff, int slots, size_t budget_bytes,
                   Source, cudaStream_t);
    ~Tq3ExpertCache();
    Tq3ExpertCache(const Tq3ExpertCache &) = delete;
    Tq3ExpertCache & operator=(const Tq3ExpertCache &) = delete;
    // Duplicate assignments preserve order. All unique assignments must fit simultaneously.
    std::vector<int> prepare(int layer, const std::vector<int> & expert_ids);
    Mat gate() const;
    Mat up() const;
    Mat down() const;
    ExpertCacheStats stats() const;
    void clear();
};

struct ExpertPayload { uint32_t type; GgufBytes bytes; };
// Per-layer stacked banks retain their native quantized layout. The byte budget includes loader staging.
class MixedExpertCache {
    struct State;
    std::unique_ptr<State> state_;
    void prefetch_impl(int layer, const std::vector<int> & expert_ids, cudaStream_t copy_stream, bool after_release);
public:
    using Source = std::function<std::array<ExpertPayload, 3>(int layer, int expert)>;
    MixedExpertCache(int layers, int experts, int hidden, int ff, int slots, size_t budget_bytes, Source, cudaStream_t);
    ~MixedExpertCache();
    MixedExpertCache(const MixedExpertCache &) = delete;
    MixedExpertCache & operator=(const MixedExpertCache &) = delete;
    std::vector<int> prepare(int layer, const std::vector<int> & expert_ids);
    // Optional source of experts already in the bank row layout: one contiguous block per expert,
    // [gate data][gate aux][up data][up aux][down data][down aux]. Misses then skip the loader repack.
    using Prepacked = std::function<const char *(int layer, int expert)>;
    // registered: the blocks live in page-locked (cudaHostRegister) memory and are DMA-ed directly, no staging copy.
    void set_prepacked(Prepacked, bool registered = false);
    // Speculative upload of a later layer's likely experts on a copy stream (registered prepacked source only).
    // Slots are claimed immediately; the next prepare() of that layer waits for the copies on its own stream.
    void prefetch(int layer, const std::vector<int> & expert_ids, cudaStream_t copy_stream);
    // ids from this index on (in the next prefetch calls) are admitted only over stale or rarely used victims (TQFN_PF_RANKED)
    void set_prefetch_strict(int from);
    // Same, for the layer about to run: the copies wait only for the bank's last release (its previous use), not for
    // work enqueued since, so they overlap the layer's attention. Returns false when the bank cannot take it yet.
    bool prefetch_released(int layer, const std::vector<int> & expert_ids, cudaStream_t copy_stream);
    // Split mode: misses upload on their own stream; the compute stream waits only in wait_misses(), so routed
    // experts that hit (last_missed() == 0 per assignment of the last prepare) can run while misses land.
    void set_miss_stream(cudaStream_t);
    bool split() const;
    const std::vector<uint8_t> & last_missed() const;
    void wait_misses();
    uint64_t prefetched() const;
    uint64_t prefetch_used() const;  // prefetched slots a later prepare actually hit
    Mat gate() const;
    Mat up() const;
    Mat down() const;
    ExpertCacheStats stats() const;
    size_t peak_device_bytes() const;
    void clear();

    // Device-side bookkeeping hand-over (decode keeps the slot tables on the GPU). The tables are exported once the
    // banks exist and imported back before the host path is used again.
    struct BankLayout {
        char * dst[6] = {}; size_t piece[6] = {}; const char * src0 = nullptr; size_t block = 0;  // gate/up/down data+aux
    };
    struct BankState { std::vector<int> ids; std::vector<uint64_t> ages; std::vector<uint64_t> freq; };
    int slots() const;           // the largest layer's slots
    int slots(int layer) const;
    void set_layer_slots(const std::vector<int> & per_layer);  // before the first prepare; same total bytes is the caller's choice
    int experts() const; int layers() const;
    bool bank_ready(int layer) const;
    bool resident(int layer, int expert) const;  // in the layer's bank now (no side effects)
    BankLayout bank_layout(int layer) const;                // piece sizes; src0 set when a prepacked source exists
    BankState bank_state(int layer) const;                  // waits for pending host-side copies first
    void bank_import(int layer, const BankState &);
    uint64_t clock() const; void set_clock(uint64_t);
    Mat bank_mat(int layer, int projection) const;          // gate 0, up 1, down 2
    void add_device_counters(uint64_t hits, uint64_t misses, uint64_t prefetched, uint64_t bytes);
};
} // namespace tqllm
