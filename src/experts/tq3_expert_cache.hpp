#pragma once
#include "weight_format.hpp"
#include "gguf_mapped.hpp"
#include <array>
#include <functional>
#include <future>
#include <memory>
#include <vector>

namespace tqllm {

struct ExpertCacheStats {
    uint64_t hits = 0, misses = 0, evictions = 0, transfer_bytes = 0;
    size_t device_bytes = 0;
    // prefetch overlap (split mode): prepares whose bank still had copies queued, of those the ones whose copies were
    // still running on the GPU (event not complete), and prepares that waited for copies of experts they do not read
    uint64_t pf_pending = 0, pf_running = 0, pf_waited_unread = 0, pf_skipped = 0;  // pf_skipped: no device room for the staging
    uint64_t bank_builds = 0, bank_drops = 0;  // whole layer banks created / dropped (evicted or released)
    uint64_t drops_budget = 0, drops_oom = 0;   // of the drops: over the cache budget / a failed device allocation
    uint64_t miss_copies = 0, miss_copy_bytes = 0, pf_copy_bytes = 0;  // H2D DMAs of demand misses (pieces), prefetch bytes
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

struct ExpertPayload {
    uint32_t type; GgufBytes bytes;
    // Optional lifetime token for a source with evictable host buffers. prepare() retains it through repacking.
    std::shared_ptr<const void> owner = {};
    bool row_layout = false;  // data already matches the bank, without an auxiliary plane
};
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
    using BatchSource = std::function<std::vector<std::array<ExpertPayload,3>>(int layer,const std::vector<int> &)>;
    void set_batch_source(BatchSource);
    // Ordered read futures; known staged banks start copying each record before later reads finish.
    using AsyncBatchSource = std::function<std::vector<std::shared_future<std::array<ExpertPayload,3>>>(int,const std::vector<int> &)>;
    void set_async_batch_source(AsyncBatchSource);
    struct HostVictimStats { size_t bytes = 0; uint64_t hits = 0, source_loads = 0, spills = 0, d2h_bytes = 0; };
    void set_host_victim_budget(size_t bytes);
    void enable_host_victims(bool enabled);
    size_t prime_host_victims(int layer, const std::vector<int> & expert_ids);
    HostVictimStats host_victim_stats() const;
    std::vector<int> prepare(int layer, const std::vector<int> & expert_ids);
    // Optional source of experts already in the bank row layout: one contiguous block per expert,
    // [gate data][gate aux][up data][up aux][down data][down aux]. Misses then skip the loader repack.
    using Prepacked = std::function<const char *(int layer, int expert)>;
    // registered: the blocks live in page-locked (cudaHostRegister) memory and are DMA-ed directly, no staging copy.
    enum class PrepackedLayout { Contiguous, Indexed };
    void set_prepacked(Prepacked, bool registered = false, PrepackedLayout = PrepackedLayout::Contiguous);
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
    void set_staged_misses(bool enabled);
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
    // the slots set so far become each layer's target; a bank built while the free memory cannot hold the target gets the
    // largest count that fits, not below floor[layer]
    void set_layer_tiers(const std::vector<int> & floor, size_t keep_bytes = size_t(256) << 20);  // keep: left free beside the banks
    int plan_slots(int layer);  // the slots the layer's next bank gets (sized to the free memory when it has none)
    int regrow(size_t keep_free);  // rebuilds banks below their target at the target, same experts, while free memory holds it (+ keep_free)
    int experts() const; int layers() const;
    bool bank_ready(int layer) const;
    bool has_layout(int layer) const;  // a bank existed once: bank_layout / bank_mat answer without one (no device pointers)
    // Frees least-recently-used banks (never `protect`) until `want` bytes of device memory are free, after all queued work.
    // Returns the evicted layers; prepare() recreates a bank on its next use.
    std::vector<int> evict_until_free(size_t want, int protect = -1);
    std::vector<int> banks_lru() const;  // layers with a bank, least recently used first
    size_t bank_bytes(int layer) const;  // device bytes of that layer's bank (0: none)
    void release_bank(int layer);        // frees the bank now (after all queued work); prepare() recreates it on use
    void reset_usage();                  // forgets the expert frequencies (victim policy), keeps the banks
    void reserve_staging(int min_cap = 0);  // staging for a whole bank (at least min_cap experts), so no prepare grows it mid-request
    bool resident(int layer, int expert) const;  // in the layer's bank now (no side effects)
    BankLayout bank_layout(int layer) const;                // piece sizes; src0 only for a contiguous expert-major source
    BankState bank_state(int layer) const;                  // waits for pending host-side copies first
    void bank_import(int layer, const BankState &);
    uint64_t clock() const; void set_clock(uint64_t);
    Mat bank_mat(int layer, int projection) const;          // gate 0, up 1, down 2
    void add_device_counters(uint64_t hits, uint64_t misses, uint64_t prefetched, uint64_t bytes);
};

// Expert slots per layer: the fitted caps of this model (`env`, or a "<experts>.slots" file beside its packed experts,
// read by layer_slots_text and scaled to `slots` per layer by parse_layer_slots). Applied before the first bank; prints
// what it applied; a no-op when neither exists.
void apply_layer_slots(MixedExpertCache & cache, const std::string & experts_path, int slots,
                       const char * env = "TQFN_LAYER_SLOTS");
} // namespace tqllm
