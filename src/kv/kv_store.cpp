#include "kv_store.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>

namespace tqllm {

// ---- pool ----

KvPool::KvPool(KvMemory & mem, const KvGeometry & g, int64_t device_pages, int64_t host_pages) : mem_(mem), g_(g) {
    if (g.layers <= 0 || g.page <= 0 || !g.key_row || !g.value_row || device_pages < 0 || host_pages < 0 || g.device_rows > g.page)
        throw std::invalid_argument("KV pool: bad geometry");
    if (g.device_rows == g.page) g_.device_rows = -1, g_.staged = false;
    if (g_.staged && (g_.device_rows <= 0 || device_pages != 1 || host_pages))
        throw std::invalid_argument("KV pool: a staged host tail needs one page and a device window");
    try {
        if (g_.staged) stage_alloc();
        else alloc(dev_, KvTier::device, device_pages);
        alloc(host_, KvTier::host, host_pages);
    } catch (...) {
        if (st_.on) stage_release();
        else release(dev_, KvTier::device);
        release(host_, KvTier::host);
        throw;
    }
}

KvPool::~KvPool() {
    if (st_.on) stage_release();
    else release(dev_, KvTier::device);
    release(host_, KvTier::host);
}

void KvPool::stage_alloc() {
    st_.on = true;
    st_.gran = mem_.granule();
    const size_t gr = st_.gran;
    if (!gr) throw std::runtime_error("KV host tail needs virtual memory management (not supported here)");
    auto up = [gr](size_t n) { return (n + gr - 1) / gr; };  // granules
    for (int s = 0; s < 2; ++s) {
        const size_t row = s ? g_.value_row : g_.key_row;
        st_.n[s] = int64_t(up(size_t(g_.page) * row));
        // + 1: a block of writes may straddle into the next granule
        st_.d[s] = std::min<int64_t>(st_.n[s], int64_t(up(size_t(g_.device_rows) * row)) + 1);
    }
    dev_.pages = 1;
    dev_.free.assign(1, 0);
    auto granule = [&](const char * what) {
        void * p = mem_.phys_alloc(gr);
        if (!p) throw std::runtime_error(std::string("out of memory (") + what + ")");
        dev_.dev_bytes += gr;
        return p;
    };
    for (int l = 0; l < g_.layers; ++l)
        for (int s = 0; s < 2; ++s) {
            const int64_t n = st_.n[s], d = st_.d[s];
            void * base = mem_.reserve(size_t(n) * gr);
            if (!base) throw std::runtime_error("out of memory (KV address range)");
            dev_.base.push_back(base);
            dev_.stride.push_back(size_t(n) * gr);
            st_.ring.emplace_back();
            for (int64_t k = 0; k < d; ++k) {
                st_.ring.back().push_back(granule("KV device window"));
                if (!mem_.map_phys(static_cast<char *>(base) + size_t(k) * gr, st_.ring.back().back()))
                    throw std::runtime_error("out of memory (mapping the KV device window)");
            }
            void * h = n > d ? mem_.host_alloc(size_t(n - d) * gr) : nullptr;
            st_.host.push_back(h);
            if (n > d && !h) throw std::runtime_error("out of memory (KV host copy)");
            dev_.host_bytes += size_t(n - d) * gr;
        }
    for (int p = 0; p < 2; ++p)
        for (int s = 0; s < 2; ++s)
            for (int64_t k = 0; k < st_.n[s] - st_.d[s]; ++k) st_.staging[p][s].push_back(granule("KV staging"));
}

void KvPool::stage_release() {
    for (size_t i = 0; i < dev_.base.size(); ++i) mem_.unreserve(dev_.base[i], dev_.stride[i]);
    for (auto & r : st_.ring)
        for (void * g : r) mem_.phys_free(g);
    for (void * h : st_.host)
        if (h) mem_.host_free(h);
    for (auto & p : st_.staging)
        for (auto & s : p)
            for (void * g : s) mem_.phys_free(g);
    st_ = Staged{};
    dev_ = Arena{};
}

bool KvPool::stage_ensure(int64_t start, int64_t end) {
    if (!st_.on) return true;
    const size_t gr = st_.gran;
    if (size_t(start) * g_.key_row < size_t(st_.e[0]) * gr || size_t(start) * g_.value_row < size_t(st_.e[1]) * gr)
        stage_truncate(start);  // a new request (or carryover) rewrites positions below the evicted boundary
    for (int s = 0; s < 2; ++s) {
        const size_t row = s ? g_.value_row : g_.key_row;
        while (size_t(st_.e[s] + st_.d[s]) * gr < size_t(end) * row) {
            const int64_t c = st_.e[s], n = st_.n[s], d = st_.d[s];
            if (c + d >= n || size_t(c + 1) * gr > size_t(start) * row) return false;  // only written rows leave the device
            for (int l = 0; l < g_.layers; ++l) {
                char * base = static_cast<char *>(dev_.base[size_t(l) * 2 + s]);
                mem_.copy(static_cast<char *>(st_.host[size_t(l) * 2 + s]) + size_t(c) * gr, base + size_t(c) * gr, gr);
            }
            mem_.sync();
            for (int l = 0; l < g_.layers; ++l) {
                char * base = static_cast<char *>(dev_.base[size_t(l) * 2 + s]);
                mem_.unmap(base + size_t(c) * gr, gr);
                if (!mem_.map_phys(base + size_t(c) * gr, st_.staging[l % 2][s][size_t(c)]) ||
                    !mem_.map_phys(base + size_t(c + d) * gr, st_.ring[size_t(l) * 2 + s][size_t(c % d)]))
                    throw std::runtime_error("KV store: remapping an evicted granule failed");
            }
            ++st_.e[s];
            ++st_.evictions;
        }
    }
    return true;
}

void KvPool::stage_truncate(int64_t pos) {
    if (!st_.on) return;
    const size_t gr = st_.gran;
    for (int s = 0; s < 2; ++s) {
        const size_t row = s ? g_.value_row : g_.key_row;
        const int64_t c0 = int64_t(size_t(std::max<int64_t>(0, pos)) * row / gr), n = st_.n[s], d = st_.d[s];
        bool moved = false;
        while (st_.e[s] > c0) {
            const int64_t c = --st_.e[s];
            for (int l = 0; l < g_.layers; ++l) {
                char * base = static_cast<char *>(dev_.base[size_t(l) * 2 + s]);
                if (c + d < n) mem_.unmap(base + size_t(c + d) * gr, gr);
                mem_.unmap(base + size_t(c) * gr, gr);
                if (!mem_.map_phys(base + size_t(c) * gr, st_.ring[size_t(l) * 2 + s][size_t(c % d)]))
                    throw std::runtime_error("KV store: restoring an evicted granule failed");
                mem_.copy(base + size_t(c) * gr, static_cast<char *>(st_.host[size_t(l) * 2 + s]) + size_t(c) * gr, gr);
            }
            moved = true;
        }
        if (moved) mem_.sync();
    }
}

size_t KvPool::staged_bytes(bool key) const { return st_.on ? size_t(st_.e[key ? 0 : 1]) * st_.gran : 0; }
void * KvPool::host_copy(int layer, bool key) const { return st_.on ? st_.host[size_t(layer) * 2 + (key ? 0 : 1)] : nullptr; }

KvPool::Arena & KvPool::arena(KvTier t) { return t == KvTier::host ? host_ : dev_; }
const KvPool::Arena & KvPool::arena(KvTier t) const { return t == KvTier::host ? host_ : dev_; }

void KvPool::alloc(Arena & a, KvTier t, int64_t pages) {
    if (!pages) return;
    a.split = t == KvTier::device && g_.device_rows >= 0;
    a.pages = pages;  // set first: release() of a half-built split arena unreserves whole ranges
    const size_t gr = a.split ? mem_.granule() : 0;
    if (a.split && !gr) throw std::runtime_error("KV host tail needs virtual memory management (not supported here)");
    auto up = [gr](size_t n) { return (n + gr - 1) / gr * gr; };
    for (int l = 0; l < g_.layers; ++l)
        for (int side = 0; side < 2; ++side) {
            const size_t row = side ? g_.value_row : g_.key_row, pb = size_t(g_.page) * row;
            if (!a.split) {
                void * p = t == KvTier::host ? mem_.host_alloc(pb * size_t(pages)) : mem_.device_alloc(pb * size_t(pages));
                if (!p) throw std::runtime_error("out of memory");
                a.base.push_back(p);
                a.stride.push_back(pb);
                (t == KvTier::host ? a.host_bytes : a.dev_bytes) += pb * size_t(pages);
                continue;
            }
            // each page starts on a granule; its first device_rows rows (rounded up to a granule) are device memory
            const size_t stride = up(pb), dev = std::min(stride, up(size_t(g_.device_rows) * row));
            void * base = mem_.reserve(stride * size_t(pages));
            if (!base) throw std::runtime_error("out of memory");
            a.base.push_back(base);
            a.stride.push_back(stride);
            for (int64_t p = 0; p < pages; ++p) {
                char * at = static_cast<char *>(base) + size_t(p) * stride;
                if ((dev && !mem_.map(at, dev, KvTier::device)) || (stride > dev && !mem_.map(at + dev, stride - dev, KvTier::host)))
                    throw std::runtime_error("out of memory");
                a.dev_bytes += dev;
                a.host_bytes += stride - dev;
            }
        }
    for (int64_t i = pages - 1; i >= 0; --i) a.free.push_back(int32_t(i));  // lowest id first
}

void KvPool::release(Arena & a, KvTier t) {
    for (size_t i = 0; i < a.base.size(); ++i) {
        if (a.split) mem_.unreserve(a.base[i], a.stride[i] * size_t(a.pages));
        else t == KvTier::host ? mem_.host_free(a.base[i]) : mem_.device_free(a.base[i]);
    }
    a = Arena{};
}

int32_t KvPool::take(KvTier t) {
    auto & f = arena(t).free;
    if (f.empty()) return -1;
    const int32_t p = f.back();
    f.pop_back();
    return p;
}

void KvPool::give(KvTier t, int32_t page) { arena(t).free.push_back(page); }
int64_t KvPool::pages(KvTier t) const { return arena(t).pages; }
int64_t KvPool::free_pages(KvTier t) const { return int64_t(arena(t).free.size()); }
size_t KvPool::page_bytes() const { return size_t(g_.layers) * size_t(g_.page) * (g_.key_row + g_.value_row); }
size_t KvPool::bytes(KvTier t) const { return t == KvTier::host ? host_.host_bytes + dev_.host_bytes : dev_.dev_bytes; }

void * KvPool::page_ptr(KvTier t, int layer, bool key, int32_t page) const {
    const auto & a = arena(t);
    const size_t i = size_t(layer) * 2 + (key ? 0 : 1);
    return static_cast<char *>(a.base[i]) + size_t(page) * a.stride[i];
}

void KvPool::move(KvTier from, int32_t pf, KvTier to, int32_t pt) {
    for (int l = 0; l < g_.layers; ++l)
        for (bool key : {true, false})
            mem_.copy(page_ptr(to, l, key, pt), page_ptr(from, l, key, pf), size_t(g_.page) * (key ? g_.key_row : g_.value_row));
}

// ---- store ----

KvStore::KvStore(KvPool & pool, int64_t context) : pool_(pool), context_(context) {}
KvStore::~KvStore() { truncate(0); }

int64_t KvStore::reserved() const { return std::min(context_, pages() * page()); }

bool KvStore::reserve(int64_t end) {
    if (end > context_) return false;
    while (reserved() < end)
        if (!push_page()) return false;
    return true;
}

bool KvStore::push_page() {
    const int32_t id = take_device();
    if (id < 0) return false;
    tier_.push_back(KvTier::device);
    id_.push_back(id);
    dev_table_.push_back(id);
    return true;
}

int32_t KvStore::take_device() {
    int32_t id = pool_.take(KvTier::device);
    if (id >= 0 || !evicts_) return id;
    const int64_t lp = evict_candidate(-1);
    if (lp < 0 || !offload(lp)) return -1;
    return pool_.take(KvTier::device);
}

int64_t KvStore::evict_candidate(int64_t keep) const {
    // oldest unpinned device page; the newest page is never moved (the step being verified writes into it)
    for (int64_t lp = pinned_pages_; lp + 1 < pages(); ++lp)
        if (lp != keep && tier_[size_t(lp)] == KvTier::device) return lp;
    return -1;
}

void KvStore::truncate(int64_t end) {
    const int64_t keep = std::max<int64_t>(0, (end + page() - 1) / page());
    while (pages() > keep) {
        if (tier_.back() != KvTier::none) pool_.give(tier_.back(), id_.back());
        tier_.pop_back();
        id_.pop_back();
        dev_table_.pop_back();
    }
    pinned_pages_ = std::min(pinned_pages_, pages());
}

bool KvStore::offload(int64_t lp) {
    if (lp < pinned_pages_ || lp >= pages() || tier_[size_t(lp)] != KvTier::device) return false;
    const int32_t h = pool_.take(KvTier::host);
    if (h < 0) return false;
    pool_.move(KvTier::device, id_[size_t(lp)], KvTier::host, h);
    pool_.give(KvTier::device, id_[size_t(lp)]);
    tier_[size_t(lp)] = KvTier::host;
    id_[size_t(lp)] = h;
    dev_table_[size_t(lp)] = -1;
    return true;
}

bool KvStore::restore(int64_t lp) {
    if (lp >= pages() || tier_[size_t(lp)] != KvTier::host) return false;
    int32_t d = pool_.take(KvTier::device);
    if (d < 0 && evicts_) {
        const int64_t victim = evict_candidate(lp);
        if (victim >= 0 && offload(victim)) d = pool_.take(KvTier::device);
    }
    if (d < 0) return false;
    pool_.move(KvTier::host, id_[size_t(lp)], KvTier::device, d);
    pool_.give(KvTier::host, id_[size_t(lp)]);
    tier_[size_t(lp)] = KvTier::device;
    id_[size_t(lp)] = d;
    dev_table_[size_t(lp)] = d;
    return true;
}

void KvStore::pin(int64_t positions) {
    pinned_pages_ = (positions + page() - 1) / page();
    for (int64_t lp = 0; lp < std::min(pinned_pages_, pages()); ++lp)
        if (tier_[size_t(lp)] == KvTier::host) restore(lp);
}

void * KvStore::row(int layer, bool key, int64_t pos) const {
    const int64_t lp = pos / page();
    if (pos < 0 || lp >= pages() || tier_[size_t(lp)] != KvTier::device) return nullptr;
    const auto & g = pool_.geometry();
    return static_cast<char *>(pool_.page_ptr(KvTier::device, layer, key, id_[size_t(lp)])) +
           size_t(pos % page()) * (key ? g.key_row : g.value_row);
}

void * KvStore::base(int layer, bool key) const {
    if (page() < context_ || pages() != 1 || tier_[0] != KvTier::device) return nullptr;
    return pool_.page_ptr(KvTier::device, layer, key, id_[0]);
}

// ---- policies ----

namespace {
struct StoreOf : KvStore {
    const char * nm;
    StoreOf(KvPool & p, int64_t ctx, const char * n, bool evicts) : KvStore(p, ctx), nm(n) { evicts_ = evicts; }
    const char * name() const override { return nm; }
};

struct Flat : KvStorePolicy {
    const char * name() const override { return "flat"; }
    const char * summary() const override { return "one page of the whole context per sequence (default; every kernel)"; }
    int64_t page(int64_t context) const override { return context; }
    std::unique_ptr<KvStore> make(KvPool & p, int64_t ctx) const override { return std::make_unique<StoreOf>(p, ctx, "flat", false); }
};

constexpr int64_t kPage = 256;

struct Paged : KvStorePolicy {
    const char * name() const override { return "paged"; }
    const char * summary() const override { return "256-position pages taken on demand"; }
    int64_t page(int64_t) const override { return kPage; }
    std::unique_ptr<KvStore> make(KvPool & p, int64_t ctx) const override { return std::make_unique<StoreOf>(p, ctx, "paged", false); }
};

struct Tiered : KvStorePolicy {
    const char * name() const override { return "tiered"; }
    const char * summary() const override { return "paged; oldest unpinned page to host memory when the device pool is full"; }
    int64_t page(int64_t) const override { return kPage; }
    std::unique_ptr<KvStore> make(KvPool & p, int64_t ctx) const override { return std::make_unique<StoreOf>(p, ctx, "tiered", true); }
};

std::map<std::string, std::unique_ptr<KvStorePolicy>> & table() {
    static std::map<std::string, std::unique_ptr<KvStorePolicy>> t;
    return t;
}

void builtins() {
    static std::once_flag once;
    std::call_once(once, [] {
        for (auto * p : std::initializer_list<KvStorePolicy *>{new Flat, new Paged, new Tiered})
            table().emplace(p->name(), std::unique_ptr<KvStorePolicy>(p));
    });
}
}  // namespace

void register_kv_store_policy(std::unique_ptr<KvStorePolicy> p) {
    builtins();
    const std::string n = p->name();
    if (!table().emplace(n, std::move(p)).second) throw std::runtime_error("KV store policy " + n + " registered twice");
}

const KvStorePolicy * kv_store_policy(const std::string & name) {
    builtins();
    const auto it = table().find(name);
    if (it == table().end()) {
        std::string known;
        for (const auto & [n, p] : table()) known += (known.empty() ? "" : ", ") + n;
        throw std::runtime_error("unknown KV store '" + name + "' (known: " + known + ")");
    }
    return it->second.get();
}

std::vector<std::string> kv_store_names() {
    builtins();
    std::vector<std::string> v;
    for (const auto & [n, p] : table()) v.push_back(n);
    return v;
}

}  // namespace tqllm
