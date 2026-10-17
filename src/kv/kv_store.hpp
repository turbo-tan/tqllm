#pragma once
// KV-cache storage: where each position's K and V rows live, separate from how a row is encoded (KvFormat). A KvPool
// owns device and host page arenas; each sequence has a KvStore that maps its positions to pool pages. Policies are
// plugins chosen by name (--kv-store, default "flat"):
//   flat    one page of the whole context per sequence: the contiguous layout every attention kernel reads today
//   paged   fixed-size pages taken on demand
//   tiered  paged; when the device pool is full the oldest unpinned page moves to host memory
// Positions are cache positions (KvStep::dpos), never RoPE positions. The DFlash drafter's caches are not stored here.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tqllm {

enum class KvTier : uint8_t { none, device, host };

// Device and host allocations and copies: CUDA in the engine (make_cuda_kv_memory), a byte budget in tests.
struct KvMemory {
    virtual ~KvMemory() = default;
    virtual void * device_alloc(size_t bytes) = 0;  // nullptr when out of memory
    virtual void device_free(void * p) = 0;
    virtual void * host_alloc(size_t bytes) = 0;    // pinned in the engine; nullptr when out of memory
    virtual void host_free(void * p) = 0;
    virtual void copy(void * dst, const void * src, size_t bytes) = 0;  // any direction, in stream order
    // Optional (CUDA virtual memory management): one reserved address range whose parts are mapped to device or host
    // memory, so a kernel reads both through one pointer. granule() 0: not supported.
    virtual size_t granule() const { return 0; }
    virtual void * reserve(size_t) { return nullptr; }
    virtual bool map(void *, size_t, KvTier) { return false; }  // [at, at + bytes) must lie in one reserved range
    virtual void unreserve(void *, size_t) {}                    // unmaps and releases everything mapped in the range
    // Physical device memory mapped whole (CUDA maps no part of an allocation) at one or more addresses: the staged host
    // tail aliases its old positions onto shared staging granules and moves device granules between addresses.
    virtual void * phys_alloc(size_t) { return nullptr; }       // nullptr: out of memory / not supported
    virtual void phys_free(void *) {}
    virtual bool map_phys(void *, void *) { return false; }      // at, phys: maps the whole allocation
    virtual void unmap(void *, size_t) {}                       // one earlier map / map_phys, exactly
    virtual void sync() {}                                       // queued copies done (before an unmap)
};

struct KvGeometry {
    int layers = 0;      // attention layers that hold KV
    size_t key_row = 0;  // bytes of one position (all KV heads) in the cache's format
    size_t value_row = 0;
    int64_t page = 0;    // positions per page
    // Positions of each device page backed by device memory; the rest of the page (its tail) is host memory reached
    // through the same pointer (needs KvMemory::granule). -1: the whole page on the device.
    int64_t device_rows = -1;
    // staged: device_rows is a window of the newest positions; when it fills, the oldest granule of every layer-side is
    // copied to host memory and its address range re-mapped onto one of two shared staging buffers (by attention-layer
    // parity) that the engine refills from host at link speed before that layer's attention. Old positions are never
    // written again, so the aliasing is safe. One page (one slot) per pool.
    bool staged = false;
};

// Page arenas shared by the sequences of one engine: one arena per (layer, side) and tier; page id p starts at row
// p * page. Every byte it holds is counted, so a budget does not depend on cudaMemGetInfo (which under-reports free
// memory under WDDM).
class KvPool {
public:
    KvPool(KvMemory & mem, const KvGeometry & g, int64_t device_pages, int64_t host_pages);  // throws "out of memory"
    ~KvPool();
    KvPool(const KvPool &) = delete;
    KvPool & operator=(const KvPool &) = delete;

    const KvGeometry & geometry() const { return g_; }
    int32_t take(KvTier t);  // -1 when the tier has no free page
    void give(KvTier t, int32_t page);
    int64_t pages(KvTier t) const;
    int64_t free_pages(KvTier t) const;
    size_t bytes(KvTier t) const;  // bytes held in that memory (device pages with a host tail count their tail as host)
    size_t page_bytes() const;     // one page: all layers, K and V
    void * page_ptr(KvTier t, int layer, bool key, int32_t page) const;
    void move(KvTier from, int32_t pf, KvTier to, int32_t pt);  // copies one page, all layers and sides

    // staged mode (geometry.staged)
    bool staged() const { return st_.on; }
    bool stage_ensure(int64_t start, int64_t end);  // writes to [start, end) hit device memory; false: window too small
    void stage_truncate(int64_t pos);               // positions >= pos will be rewritten: bring them back to the device
    size_t staged_bytes(bool key) const;            // bytes per layer to copy host -> staging before its attention
    void * host_copy(int layer, bool key) const;    // that layer's evicted bytes, in position order
    int64_t evictions() const { return st_.evictions; }

private:
    struct Arena {
        std::vector<void *> base;    // [layer * 2 + side]
        std::vector<size_t> stride;  // bytes from one page to the next, per layer and side
        std::vector<int32_t> free;
        int64_t pages = 0;
        bool split = false;          // device pages with a host tail (one reserved range per layer and side)
        size_t dev_bytes = 0, host_bytes = 0;
    };
    Arena & arena(KvTier t);
    const Arena & arena(KvTier t) const;
    void alloc(Arena & a, KvTier t, int64_t pages);
    void release(Arena & a, KvTier t);
    void stage_alloc();
    void stage_release();
    struct Staged {
        bool on = false;
        size_t gran = 0;
        int64_t n[2] = {0, 0}, d[2] = {0, 0}, e[2] = {0, 0};  // per side: granules per page, device ring size, evicted
        std::vector<std::vector<void *>> ring;                 // [layer * 2 + side][d] device granules
        std::vector<void *> staging[2][2];                     // [parity][side][n - d] staging granules
        std::vector<void *> host;                              // [layer * 2 + side] evicted bytes (n - d granules)
        int64_t evictions = 0;
    } st_;
    KvMemory & mem_;
    KvGeometry g_;
    Arena dev_, host_;
};

// One sequence: positions [0, reserved()) mapped to pool pages, logical page i = positions [i * page, (i + 1) * page).
class KvStore {
public:
    KvStore(KvPool & pool, int64_t context);
    virtual ~KvStore();
    KvStore(const KvStore &) = delete;
    KvStore & operator=(const KvStore &) = delete;

    virtual const char * name() const = 0;
    // Backs positions [0, end) with device pages. false (pages taken so far are kept) when the pool cannot hold them
    // or end is past the context.
    virtual bool reserve(int64_t end);
    void truncate(int64_t end);  // returns pages that start at or past end (speculative rollback keeps the rest)
    virtual bool offload(int64_t lpage);  // device -> host; false when pinned, not on the device or no host page
    virtual bool restore(int64_t lpage);  // host -> device; false when no device page can be freed
    void pin(int64_t positions);          // pages holding [0, positions) never leave the device

    int64_t context() const { return context_; }
    int64_t page() const { return pool_.geometry().page; }
    int64_t reserved() const;  // positions backed by pages (any tier)
    int64_t pages() const { return int64_t(tier_.size()); }
    KvTier tier(int64_t lpage) const { return tier_[size_t(lpage)]; }
    const std::vector<int32_t> & device_table() const { return dev_table_; }  // -1: not on the device (for kernels)
    void * row(int layer, bool key, int64_t pos) const;  // device address of one position; nullptr when off the device
    void * base(int layer, bool key) const;              // one device page spans the context: its first row, else nullptr

protected:
    int32_t take_device();  // a free device page, after evict() when the pool is full; -1 when none
    virtual int64_t evict_candidate(int64_t keep) const;  // logical page to offload for room (-1: none); never keep
    bool push_page();
    KvPool & pool_;
    int64_t context_;
    int64_t pinned_pages_ = 0;
    bool evicts_ = false;           // tiered: take_device may offload
    std::vector<KvTier> tier_;      // per logical page
    std::vector<int32_t> id_;       // page id in that tier
    std::vector<int32_t> dev_table_;
};

// Compaction hook: a small stand-in for an offloaded page. Interface only for now
// (tiered KV compaction is planned; no implementation yet).
struct KvCompactor {
    virtual ~KvCompactor() = default;
    virtual const char * name() const = 0;
    virtual bool compact(KvStore & s, int64_t lpage) = 0;
};

// A storage policy: page size for a context and the store it makes. Add a policy: a class in kv_store.cpp (or
// elsewhere) and register_kv_store_policy.
struct KvStorePolicy {
    virtual ~KvStorePolicy() = default;
    virtual const char * name() const = 0;
    virtual const char * summary() const = 0;
    virtual int64_t page(int64_t context) const = 0;
    virtual std::unique_ptr<KvStore> make(KvPool & pool, int64_t context) const = 0;
};

const KvStorePolicy * kv_store_policy(const std::string & name);  // throws on unknown names
std::vector<std::string> kv_store_names();
void register_kv_store_policy(std::unique_ptr<KvStorePolicy> p);
std::unique_ptr<KvMemory> make_cuda_kv_memory(void * stream);  // kv_store_cuda.cpp (CUDA builds only)

}  // namespace tqllm
