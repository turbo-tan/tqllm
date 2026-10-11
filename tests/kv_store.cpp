// Host-only tests for KV storage (src/kv_store.hpp): pool accounting, flat layout, paging, rollback, tiered offload and
// restore. A fake memory with byte budgets stands in for the GPU and pinned host memory.
#include "src/kv_store.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace tqllm;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } } while (0)

struct FakeMemory : KvMemory {
    size_t dev_budget, host_budget, dev_live = 0, host_live = 0;
    std::map<void *, size_t> dev, host;
    int copies = 0;
    FakeMemory(size_t d, size_t h) : dev_budget(d), host_budget(h) {}
    ~FakeMemory() override {
        if (fd >= 0) close(fd);
        for (auto & [p, n] : dev) std::free(p);
        for (auto & [p, n] : host) std::free(p);
    }
    void * get(std::map<void *, size_t> & m, size_t & live, size_t budget, size_t n) {
        if (live + n > budget) return nullptr;
        void * p = std::calloc(1, n);
        m[p] = n;
        live += n;
        return p;
    }
    void put(std::map<void *, size_t> & m, size_t & live, void * p) {
        live -= m.at(p);
        m.erase(p);
        std::free(p);
    }
    void * device_alloc(size_t n) override { return get(dev, dev_live, dev_budget, n); }
    void device_free(void * p) override { put(dev, dev_live, p); }
    void * host_alloc(size_t n) override { return get(host, host_live, host_budget, n); }
    void host_free(void * p) override { put(host, host_live, p); }
    void copy(void * d, const void * s, size_t n) override { std::memcpy(d, s, n); ++copies; }
    // virtual memory as on a GPU: a reserved range is PROT_NONE address space, map() backs a piece with fresh memory and
    // charges the tier's budget, phys_alloc() is a memfd that map_phys() maps at any address (real aliasing)
    size_t gran = 0;
    struct Map { char * at; size_t bytes; KvTier where; bool owned; };
    struct Range { size_t bytes; std::vector<Map> maps; };
    struct Phys { size_t off, bytes; };  // a piece of one shared memfd (one fd per granule would hit the fd limit)
    int fd = -1;
    size_t fd_size = 0;
    std::map<char *, Range> ranges;
    int phys_live = 0;
    size_t granule() const override { return gran; }
    void * reserve(size_t n) override {
        void * p = mmap(nullptr, n, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return nullptr;
        ranges[static_cast<char *>(p)] = Range{n, {}};
        return p;
    }
    Range * range_of(char * a, size_t n) {
        auto it = ranges.upper_bound(a);
        if (it == ranges.begin()) return nullptr;
        --it;
        if (a + n > it->first + it->second.bytes || (size_t(a - it->first) % gran) || n % gran) return nullptr;
        return &it->second;
    }
    bool map(void * at, size_t n, KvTier where) override {
        char * a = static_cast<char *>(at);
        Range * r = range_of(a, n);
        if (!r) return false;
        size_t & live = where == KvTier::host ? host_live : dev_live;
        if (live + n > (where == KvTier::host ? host_budget : dev_budget)) return false;
        if (mmap(a, n, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) == MAP_FAILED) return false;
        live += n;
        r->maps.push_back({a, n, where, true});
        return true;
    }
    void * phys_alloc(size_t n) override {
        if (dev_live + n > dev_budget) return nullptr;
        if (fd < 0) fd = memfd_create("kvphys", 0);
        if (fd < 0 || ftruncate(fd, off_t(fd_size + n))) return nullptr;
        dev_live += n;
        ++phys_live;
        fd_size += n;
        return new Phys{fd_size - n, n};
    }
    void phys_free(void * p) override {
        if (!p) return;
        auto * P = static_cast<Phys *>(p);
        dev_live -= P->bytes;
        --phys_live;
        delete P;
    }
    bool map_phys(void * at, void * p) override {  // whole allocation only, as cuMemMap
        char * a = static_cast<char *>(at);
        const size_t n = static_cast<Phys *>(p)->bytes, off = 0;
        Range * r = range_of(a, n);
        if (!r) return false;
        for (const auto & m : r->maps)
            if (a < m.at + m.bytes && m.at < a + n) return false;  // the address must be unmapped first, as with cuMemMap
        if (mmap(a, n, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, fd, off_t(static_cast<Phys *>(p)->off + off)) == MAP_FAILED) return false;
        r->maps.push_back({a, n, KvTier::device, false});
        return true;
    }
    void unmap(void * at, size_t n) override {
        char * a = static_cast<char *>(at);
        Range * r = range_of(a, n);
        if (!r) return;
        for (size_t i = 0; i < r->maps.size(); ++i)
            if (r->maps[i].at == a && r->maps[i].bytes == n) {
                if (r->maps[i].owned) (r->maps[i].where == KvTier::host ? host_live : dev_live) -= n;
                mmap(a, n, PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                r->maps.erase(r->maps.begin() + ptrdiff_t(i));
                return;
            }
    }
    void unreserve(void * base, size_t) override {
        const auto it = ranges.find(static_cast<char *>(base));
        for (const auto & m : it->second.maps)
            if (m.owned) (m.where == KvTier::host ? host_live : dev_live) -= m.bytes;
        munmap(it->first, it->second.bytes);
        ranges.erase(it);
    }
};

// tq3 on Qwen3.8-27B: 16 attention layers, 4 KV heads x 112 B per position for K and for V
static KvGeometry geom(int64_t page) { return KvGeometry{16, 4 * 112, 4 * 112, page}; }

static void fill(KvStore & s, int64_t pos, uint8_t v) {
    for (int l = 0; l < 16; ++l)
        for (bool key : {true, false}) std::memset(s.row(l, key, pos), v + l + (key ? 0 : 100), 448);
}

static bool same(KvStore & s, int64_t pos, uint8_t v) {
    for (int l = 0; l < 16; ++l)
        for (bool key : {true, false}) {
            const auto * p = static_cast<const uint8_t *>(s.row(l, key, pos));
            if (!p) return false;
            for (int i = 0; i < 448; ++i)
                if (p[i] != uint8_t(v + l + (key ? 0 : 100))) return false;
        }
    return true;
}

static void test_registry() {
    const auto names = kv_store_names();
    for (const char * must : {"flat", "paged", "tiered"}) {
        bool found = false;
        for (const auto & n : names) found = found || n == must;
        CHECK(found, "policy %s not registered", must);
    }
    for (const auto & n : names) CHECK(n == kv_store_policy(n)->name(), "%s: name mismatch", n.c_str());
    bool threw = false;
    try { kv_store_policy("no-such-store"); } catch (const std::runtime_error &) { threw = true; }
    CHECK(threw, "unknown policy must throw");
}

static void test_flat() {
    const int64_t ctx = 4096;
    FakeMemory mem(size_t(1) << 30, 0);
    const auto * pol = kv_store_policy("flat");
    KvPool pool(mem, geom(pol->page(ctx)), 2, 0);  // two slots
    CHECK(pool.bytes(KvTier::device) == mem.dev_live, "pool bytes %zu != live allocations %zu", pool.bytes(KvTier::device), mem.dev_live);
    CHECK(mem.dev_live == size_t(2) * 16 * ctx * (448 + 448), "flat holds exactly today's per-slot key_bytes(context) + value_bytes");
    auto a = pol->make(pool, ctx), b = pol->make(pool, ctx);
    CHECK(a->reserve(ctx) && b->reserve(ctx), "both slots reserve the whole context");
    CHECK(!a->reserve(ctx + 1), "past the context fails");
    for (int l = 0; l < 16; ++l)
        for (bool key : {true, false}) {
            auto * base = static_cast<char *>(a->base(l, key));
            CHECK(base, "flat has a contiguous base");
            for (int64_t pos : {0, 1, 255, 256, 4095})
                CHECK(a->row(l, key, pos) == base + pos * 448, "flat row = base + pos * row (layer %d pos %lld)", l, (long long)pos);
        }
    CHECK(a->base(0, true) != b->base(0, true), "slots do not share pages");
    a->truncate(100);  // rollback inside the single page keeps it
    CHECK(a->pages() == 1 && a->base(0, true), "flat keeps its page on rollback");
    auto c = pol->make(pool, ctx);
    CHECK(!c->reserve(1), "a third slot does not fit a two-slot pool");
}

static void test_paged() {
    const int64_t ctx = 4096;
    FakeMemory mem(size_t(1) << 30, 0);
    const auto * pol = kv_store_policy("paged");
    const int64_t pg = pol->page(ctx);
    KvPool pool(mem, geom(pg), 8, 0);
    auto s = pol->make(pool, ctx);
    CHECK(s->reserve(1) && s->pages() == 1, "first position takes one page");
    CHECK(s->reserve(pg) && s->pages() == 1, "a full page needs no second page");
    CHECK(s->reserve(pg + 1) && s->pages() == 2, "crossing a page boundary takes a page");
    fill(*s, pg - 1, 10);
    fill(*s, pg, 20);
    CHECK(same(*s, pg - 1, 10) && same(*s, pg, 20), "rows either side of the boundary hold their bytes");
    CHECK(s->row(3, false, pg + 5) == static_cast<char *>(pool.page_ptr(KvTier::device, 3, false, s->device_table()[1])) + 5 * 448,
          "row address = page of the device table + offset in the page");
    CHECK(s->base(0, true) == nullptr, "paged has no contiguous base");
    s->truncate(pg + 1);
    CHECK(s->pages() == 2, "truncate inside the last page keeps it");
    s->truncate(pg);
    CHECK(s->pages() == 1 && pool.free_pages(KvTier::device) == 7, "truncate at a boundary returns the page");
    CHECK(same(*s, pg - 1, 10), "rollback keeps earlier rows");
    CHECK(!s->reserve(9 * pg), "past the pool fails without throwing");
    CHECK(s->pages() == 8 && pool.free_pages(KvTier::device) == 0, "pages taken before failing are kept");
    s->truncate(0);
    CHECK(pool.free_pages(KvTier::device) == 8, "reset returns every page");
}

static void test_tiered() {
    const int64_t ctx = 8192;
    FakeMemory mem(size_t(1) << 30, size_t(1) << 30);
    const auto * pol = kv_store_policy("tiered");
    const int64_t pg = pol->page(ctx);
    KvPool pool(mem, geom(pg), 4, 8);  // 4 device pages, 8 host pages
    CHECK(pool.bytes(KvTier::device) + pool.bytes(KvTier::host) == mem.dev_live + mem.host_live, "pool counts both tiers");
    auto s = pol->make(pool, ctx);
    s->pin(pg);  // the system prompt page never leaves the device
    CHECK(s->reserve(4 * pg), "fills the device pool");
    for (int64_t p = 0; p < 4; ++p) fill(*s, p * pg + 7, uint8_t(p + 1));
    CHECK(s->reserve(5 * pg), "a fifth page offloads one");
    CHECK(s->tier(0) == KvTier::device, "pinned head stays on the device");
    CHECK(s->tier(1) == KvTier::host, "oldest unpinned page goes to host first");
    CHECK(s->row(0, true, pg + 7) == nullptr, "an offloaded row has no device address");
    CHECK(s->device_table()[1] == -1, "device table marks it off the device");
    CHECK(s->reserve(6 * pg) && s->tier(2) == KvTier::host, "next oldest goes next");
    CHECK(s->restore(1), "restore brings page 1 back");
    CHECK(s->tier(1) == KvTier::device && same(*s, pg + 7, 2), "restored bytes are identical");
    CHECK(s->tier(0) == KvTier::device, "restore never evicts the pinned head");
    CHECK(s->tier(5) == KvTier::device, "the newest page is never evicted");
    CHECK(s->restore(2) && same(*s, 2 * pg + 7, 3), "a second restore evicts another page, bytes intact");
    CHECK(!s->offload(0), "pinned page cannot be offloaded");
    CHECK(!s->restore(0), "restore of a device page is refused");
    int on_dev = 0;
    for (int64_t p = 0; p < s->pages(); ++p) on_dev += s->tier(p) == KvTier::device;
    CHECK(on_dev == 4 && pool.free_pages(KvTier::device) == 0, "device holds exactly its 4 pages");
    s->truncate(0);
    CHECK(pool.free_pages(KvTier::device) == 4 && pool.free_pages(KvTier::host) == 8, "reset returns pages in both tiers");
}

static void test_newest_page_stays() {
    FakeMemory mem(size_t(1) << 30, size_t(1) << 30);
    const auto * pol = kv_store_policy("tiered");
    const int64_t pg = pol->page(4096);
    KvPool pool(mem, geom(pg), 2, 4);
    auto s = pol->make(pool, 4096);
    s->pin(pg);
    CHECK(s->reserve(2 * pg), "two device pages");
    CHECK(!s->reserve(2 * pg + 1), "only the pinned head and the newest page are left: nothing may be offloaded");
    CHECK(s->tier(1) == KvTier::device && pool.free_pages(KvTier::host) == 4, "the page being written stays on the device");
}

static void test_host_tail() {
    // flat page of 4096 positions, last 1000 in host memory; granule 4 KiB (2 MiB on GPUs)
    const int64_t ctx = 4096, dev_rows = ctx - 1000;
    FakeMemory mem(size_t(1) << 30, size_t(1) << 30);
    mem.gran = 4096;
    KvGeometry g = geom(ctx);
    g.device_rows = dev_rows;
    {
        KvPool pool(mem, g, 2, 0);  // two slots
        const size_t row = 448, page = size_t(ctx) * row, dev = (dev_rows * row + 4095) / 4096 * 4096;
        const size_t stride = (page + 4095) / 4096 * 4096;
        CHECK(pool.bytes(KvTier::device) == size_t(2) * 32 * dev, "device bytes: each page's head, rounded to the granule");
        CHECK(pool.bytes(KvTier::host) == size_t(2) * 32 * (stride - dev), "host bytes: each page's tail");
        CHECK(mem.dev_live == pool.bytes(KvTier::device) && mem.host_live == pool.bytes(KvTier::host),
              "pool accounting equals what the memory mapped");
        auto a = kv_store_policy("flat")->make(pool, ctx), b = kv_store_policy("flat")->make(pool, ctx);
        CHECK(a->reserve(ctx) && b->reserve(ctx), "both slots reserve the whole context");
        for (int l = 0; l < 16; ++l)
            for (bool key : {true, false}) {
                auto * base = static_cast<char *>(a->base(l, key));
                CHECK(base, "one pointer spans device head and host tail");
                for (int64_t pos : {int64_t(0), dev_rows - 1, dev_rows, ctx - 1})
                    CHECK(a->row(l, key, pos) == base + pos * 448, "row = base + pos * row across the boundary (pos %lld)", (long long)pos);
            }
        CHECK(static_cast<char *>(b->base(0, true)) - static_cast<char *>(a->base(0, true)) == ptrdiff_t(stride) ||
                  static_cast<char *>(a->base(0, true)) - static_cast<char *>(b->base(0, true)) == ptrdiff_t(stride),
              "slots are one granule-aligned stride apart");
        fill(*a, dev_rows - 1, 30);
        fill(*a, dev_rows, 40);
        CHECK(same(*a, dev_rows - 1, 30) && same(*a, dev_rows, 40), "rows either side of the boundary hold their bytes");
    }
    CHECK(mem.dev_live == 0 && mem.host_live == 0 && mem.ranges.empty(), "pool release unmaps both tiers");
    {
        FakeMemory small(size_t(1) << 30, 4096);  // host budget too small for the tails
        small.gran = 4096;
        bool threw = false;
        try { KvPool pool(small, g, 2, 0); } catch (const std::runtime_error & e) { threw = std::strstr(e.what(), "out of memory"); }
        CHECK(threw, "host tails that do not fit throw out of memory");
        CHECK(small.dev_live == 0 && small.host_live == 0 && small.ranges.empty(), "a half-built split pool releases everything");
    }
    {
        FakeMemory novmm(size_t(1) << 30, size_t(1) << 30);  // granule 0: no virtual memory management
        bool threw = false;
        try { KvPool pool(novmm, g, 1, 0); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "a host tail without virtual memory management is refused, not silently placed");
    }
    {
        KvGeometry all = g;
        all.device_rows = 0;  // whole KV in host memory
        KvPool pool(mem, all, 1, 0);
        CHECK(pool.bytes(KvTier::device) == 0 && pool.bytes(KvTier::host) > 0, "device_rows 0 puts the whole page on host");
    }
}

// staged host tail: rows are written through the page's pointer in blocks; old granules move to host and alias staging
static void test_staged() {
    const int64_t ctx = 2048, window = 512, blk = 48;
    const int layers = 4;
    FakeMemory mem(size_t(1) << 30, size_t(1) << 30);
    mem.gran = 4096;  // 448 B rows straddle granules, as tq3 rows do on 2 MiB
    KvGeometry g{layers, 448, 448, ctx, window, true};
    {
    KvPool pool(mem, g, 1, 0);
    auto s = kv_store_policy("flat")->make(pool, ctx);
    CHECK(pool.staged() && s->reserve(ctx), "staged pool reserves the context");
    const size_t flat_b = size_t(layers) * 2 * size_t(ctx) * 448;
    CHECK(pool.bytes(KvTier::device) < flat_b * 2 / 3, "device holds the window + staging, not the whole context (%zu of %zu)",
          pool.bytes(KvTier::device), flat_b);
    auto byte_of = [](int l, bool key, int64_t pos) { return uint8_t(pos * 7 + l * 31 + (key ? 0 : 101)); };
    auto prefetch = [&](int l) {  // the engine's copy before layer l's attention
        for (bool key : {true, false})
            if (pool.staged_bytes(key)) std::memcpy(s->base(l, key), pool.host_copy(l, key), pool.staged_bytes(key));
    };
    auto check_upto = [&](int64_t end) {
        int bad = 0;
        prefetch(0), prefetch(1);  // the engine's order: two layers ahead of the reads, one staging buffer per parity
        for (int l = 0; l < layers; ++l) {
            for (bool key : {true, false})
                for (int64_t p = 0; p < end; ++p) {
                    const auto * r = static_cast<const uint8_t *>(s->base(l, key)) + p * 448;
                    bad += r[0] != byte_of(l, key, p) || r[447] != byte_of(l, key, p);
                }
            if (l + 2 < layers) prefetch(l + 2);
        }
        return bad;
    };
    auto write_block = [&](int64_t p0, int64_t p1) {
        for (int l = 0; l < layers; ++l)
            for (bool key : {true, false})
                for (int64_t p = p0; p < p1; ++p) std::memset(static_cast<char *>(s->base(l, key)) + p * 448, byte_of(l, key, p), 448);
    };
    bool ok = true;
    for (int64_t p = 0; p < ctx; p += blk) {
        const int64_t e = std::min(ctx, p + blk);
        ok = ok && pool.stage_ensure(p, e);
        write_block(p, e);
    }
    CHECK(ok, "every block fits the device window after evictions");
    CHECK(pool.evictions() > 0 && pool.staged_bytes(true) > 0, "old granules moved to host (%lld evictions)", (long long)pool.evictions());
    CHECK(check_upto(ctx) == 0, "every row reads back through the same pointer after staging (straddling rows included)");
    // truncate below the evicted boundary (request carryover): rows below stay, rows above are rewritten
    const int64_t t = 300;
    pool.stage_truncate(t);
    CHECK(pool.staged_bytes(true) <= size_t(t) * 448, "truncate brings the granules at and above position %lld back", (long long)t);
    for (int64_t p = t; p < ctx; p += blk) {
        const int64_t e = std::min(ctx, p + blk);
        ok = pool.stage_ensure(p, e);
        if (!ok) break;
        write_block(p, e);
    }
    CHECK(ok && check_upto(ctx) == 0, "rewritten after truncation, every row still correct");
    s.reset();
    }
    CHECK(mem.phys_live == 0 && mem.ranges.empty() && mem.dev_live == 0 && mem.host_live == 0,
          "a staged pool releases every mapping, physical chunk and host copy");
    {
        KvGeometry tiny{layers, 448, 448, ctx, 8, true};  // window of 8 rows: a 48-row block cannot fit
        KvPool p2(mem, tiny, 1, 0);
        CHECK(!p2.stage_ensure(0, 48) || !p2.stage_ensure(40, 88), "a block bigger than the window is refused, not corrupted");
    }
}

static void test_out_of_memory() {
    FakeMemory mem(1 << 20, 0);  // far too small
    bool threw = false;
    try { KvPool pool(mem, geom(4096), 1, 0); } catch (const std::runtime_error & e) { threw = std::strstr(e.what(), "out of memory"); }
    CHECK(threw, "a pool that does not fit throws out of memory (the fit retries on that message)");
    CHECK(mem.dev_live == 0, "a failed pool releases what it took");
}

int main() {
    test_registry();
    test_flat();
    test_paged();
    test_tiered();
    test_newest_page_stays();
    test_host_tail();
    test_staged();
    test_out_of_memory();
    if (fails) std::printf("kv_store: %d failure(s)\n", fails);
    else std::printf("kv_store: ok\n");
    return fails ? 1 : 0;
}
