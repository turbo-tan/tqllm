#include "tq3_expert_cache.hpp"
#include "moe_kernels.cuh"
#include <algorithm>
#include <limits>
#include <map>
#include <cstring>
#include <set>
#include <tuple>
#include <cstdlib>
#include <unordered_map>
#include <stdexcept>
// TQFN_* switches that won their A/Bs are on by default; set NAME=0 to turn one off
static inline bool tqfn_default_on(const char * n) { const char * v = std::getenv(n); return !v || (v[0] && v[0] != '0'); }

namespace tqllm {
namespace {
void checked(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
}
struct MixedExpertCache::State {
    struct Allocation { void * p; size_t bytes; };
    struct Bank {
        State & owner;
        std::vector<Allocation> memory;
        std::array<Mat, 3> matrices{};
        std::array<uint32_t, 3> types{};
        std::array<size_t, 3> raw_stride{}, stride{}, aux_stride{};
        int cap;
        std::vector<int> ids;
        std::vector<uint64_t> ages;
        uint64_t touched = 0;
        cudaEvent_t ready = nullptr; bool pending = false;
        cudaEvent_t released = nullptr; bool used = false;  // compute stream passed the bank's last routed matvec
        std::vector<uint8_t> speculative;  // slot filled by prefetch and not yet used by a prepare
        std::vector<uint8_t> inflight;     // slot written by the bank's pending prefetch (ready not yet waited for)
        std::vector<uint64_t> pf_age;      // clock of the slot's last prefetch fill (TQFN_VALUE_VICTIM)
        std::vector<int> pf_seq;           // TQFN_PF_EVENTS: index of the event recorded after the slot's prefetch copy
        std::vector<cudaEvent_t> pf_ev;
        Bank(State & s, int c) : owner(s), cap(c), ids(c, -1), ages(c), speculative(c, 0), inflight(c, 0), pf_age(c, 0), pf_seq(c, -1) {}
        ~Bank() { if (ready) cudaEventDestroy(ready); if (released) cudaEventDestroy(released); for (auto e : pf_ev) cudaEventDestroy(e); owner.free(memory); }
    };
    int layers, experts, hidden, ff, capacity, active = -1;
    std::vector<int> layer_cap;  // per-layer slots (TQFN_LAYER_SLOTS); empty: capacity for every layer
    int cap_of(int layer) const { return layer_cap.empty() ? capacity : layer_cap[size_t(layer)]; }
    size_t budget, live = 0, peak = 0;
    Source source;
    cudaStream_t stream;
    ExpertCacheStats counters;
    std::map<int, std::unique_ptr<Bank>> banks;
    std::vector<uint64_t> frequency;
    uint64_t clock = 0;
    // Misses repack into this pinned buffer and copy straight into their slots (one sync per prepare).
    char * staging = nullptr; size_t staging_bytes = 0;
    char * device_staging = nullptr;  // prepacked misses: one upload per prepare, then device-side scatter
    cudaStream_t miss_stream = nullptr; cudaEvent_t miss_done = nullptr; bool miss_pending = false;
    std::vector<uint8_t> missed;
    std::vector<uint32_t> mark; std::vector<int> where; uint32_t gen = 0;  // prepare(): flat per-expert bookkeeping
    char * pf_staging = nullptr; size_t pf_bytes = 0;            // prefetch: one DMA per expert, then one scatter launch
    int pf_strict_from = 1 << 30;  // set_prefetch_strict: caller index from which prefetches are admitted strictly
    Prepacked prepacked; bool registered = false; uint64_t prefetch_count = 0, prefetch_used = 0;
    ~State() { if (miss_stream) cudaStreamSynchronize(miss_stream); if (miss_done) cudaEventDestroy(miss_done); banks.clear(); if (staging) cudaFreeHost(staging); if (device_staging) cudaFree(device_staging); if (pf_staging) cudaFree(pf_staging); }
    State(int l, int e, int h, int f, int c, size_t b, Source s, cudaStream_t st)
        : layers(l), experts(e), hidden(h), ff(f), capacity(c), budget(b), source(std::move(s)), stream(st) {
        if (l <= 0 || e <= 0 || h <= 0 || f <= 0 || c <= 0 || c > e || !b || !source ||
            int64_t(c) * std::max(h, f) > std::numeric_limits<int>::max() ||
            uint64_t(l) * e > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
            throw std::invalid_argument("mixed cache: invalid dimensions or source");
        frequency.resize(size_t(l) * e); mark.assign(size_t(e), 0u); where.assign(size_t(e), -1);
    }
    void wait_misses_impl() { if (miss_pending) { checked(cudaStreamWaitEvent(stream, miss_done, 0)); miss_pending = false; } }
    void free(std::vector<Allocation> & memory) {
        cudaStreamSynchronize(stream);
        for (const auto & a : memory) { cudaFree(a.p); live -= a.bytes; }
        memory.clear(); counters.device_bytes = live;
    }
    const void * upload(const void * host, size_t bytes, std::vector<Allocation> & owned, int protect) {
        if (!bytes || bytes > budget) throw std::invalid_argument("mixed cache: upload exceeds budget");
        while (live > budget - bytes) {
            auto victim = banks.end();
            for (auto it = banks.begin(); it != banks.end(); ++it)
                if (it->first != protect && (victim == banks.end() || it->second->touched < victim->second->touched)) victim = it;
            if (victim == banks.end()) throw std::invalid_argument("mixed cache: bank plus loader staging exceeds budget");
            for (int id : victim->second->ids) if (id >= 0) ++counters.evictions;
            banks.erase(victim);
        }
        void * p = nullptr; checked(cudaMalloc(&p, bytes));
        owned.push_back({p, bytes}); live += bytes; peak = std::max(peak, live); counters.device_bytes = live;
        checked(cudaMemcpyAsync(p, host, bytes, cudaMemcpyHostToDevice, stream));
        counters.transfer_bytes += bytes; return p;
    }
    void validate(const ExpertPayload & p, int projection, const Bank * bank = nullptr) const {
        uint64_t block = 0, bytes = 0;
        const int cols = projection == 2 ? ff : hidden, rows = projection == 2 ? hidden : ff;
        const auto * format = weight_format(p.type);
        if (!format || format->basis() != Basis::Plain || !gguf_block_layout(p.type, block, bytes) ||
            cols % block || !p.bytes.data || p.bytes.size != uint64_t(cols) * rows / block * bytes ||
            (bank && p.type != bank->types[projection]))
            throw std::invalid_argument("mixed cache: source shape, type or byte count differs");
    }
    std::unique_ptr<Bank> create(int layer, const std::array<ExpertPayload, 3> & sample) {
        auto bank = std::make_unique<Bank>(*this, cap_of(layer)); const int capacity = bank->cap;
        for (int p = 0; p < 3; ++p) {
            const int cols = p == 2 ? ff : hidden, rows = p == 2 ? hidden : ff;
            if (sample[p].bytes.size > budget / capacity) throw std::invalid_argument("mixed cache: raw bank exceeds budget");
            bank->types[p] = sample[p].type; bank->raw_stride[p] = sample[p].bytes.size;
            std::vector<char> zeros(sample[p].bytes.size * capacity, 0);
            Upload put = [&](const void * data, size_t bytes) { return upload(data, bytes, bank->memory, layer); };
            auto m = weight_format(sample[p].type)->load(zeros, capacity * rows, cols, put, stream);
            if (!m.fmt || m.fmt->basis() != Basis::Plain || !m.fmt->row_bytes(cols))  // no grouped kernel: callers slice rows
                throw std::invalid_argument("mixed cache: native grouped row layout unavailable");
            bank->stride[p] = m.fmt->row_bytes(cols) * rows;
            bank->aux_stride[p] = m.fmt->aux_row_bytes(cols) * rows;
            auto allocation_matches = [&](const void * ptr, size_t bytes) {
                return std::any_of(bank->memory.begin(), bank->memory.end(), [&](const Allocation & a) { return a.p == ptr && a.bytes == bytes; });
            };
            if (!allocation_matches(m.p, bank->stride[p] * capacity) ||
                (bank->aux_stride[p] && !allocation_matches(m.aux, bank->aux_stride[p] * capacity)))
                throw std::invalid_argument("mixed cache: loader upload differs from native row layout");
            bank->matrices[p] = m;
        }
        checked(cudaStreamSynchronize(stream)); return bank;
    }
};
MixedExpertCache::MixedExpertCache(int l, int e, int h, int f, int c, size_t b, Source s, cudaStream_t st)
    : state_(std::make_unique<State>(l, e, h, f, c, b, std::move(s), st)) {}
MixedExpertCache::~MixedExpertCache() = default;
// Victim policy (found by trace replay): a slot's value is its expert's use count decayed by time since last use, plus
// one while it was prefetched in the last 50 clock ticks; the lowest value is evicted, for demand misses and prefetches
// alike. Placement only: the model's output does not change.
constexpr double kValueDecay = 0.05;
template <class B> static double slot_value(const B & bank, int j, uint64_t freq, uint64_t clock) {
    return double(freq) / (1.0 + kValueDecay * double(clock - bank.ages[j])) + (clock - bank.pf_age[j] < 50 && bank.pf_age[j] ? 1.0 : 0.0);
}
std::vector<int> MixedExpertCache::prepare(int layer, const std::vector<int> & ids) {
    auto & s = *state_;
    if (layer < 0 || layer >= s.layers || ids.empty()) throw std::invalid_argument("mixed cache: invalid layer or empty assignments");
    // flat bookkeeping (no set / map per call): generation stamp = membership, where[] = slot of a needed expert
    if (++s.gen == 0) { std::fill(s.mark.begin(), s.mark.end(), 0u); s.gen = 1; }
    std::vector<int> needed; needed.reserve(ids.size());
    for (int id : ids) {
        if (id < 0 || id >= s.experts) throw std::invalid_argument("mixed cache: expert out of range");
        if (s.mark[id] != s.gen) { s.mark[id] = s.gen; s.where[id] = -1; needed.push_back(id); }
    }
    std::sort(needed.begin(), needed.end());  // same order as the former std::set: same slot assignment
    auto is_needed = [&](int e) { return e >= 0 && s.mark[e] == s.gen; };
    if (needed.size() > size_t(s.cap_of(layer))) throw std::invalid_argument("mixed cache: active set exceeds slots");
    if (s.miss_stream && s.active >= 0) {
        auto & prev = *s.banks.at(s.active);
        if (!prev.released) checked(cudaEventCreateWithFlags(&prev.released, cudaEventDisableTiming));
        checked(cudaEventRecord(prev.released, s.stream)); prev.used = true;
    }
    if (s.miss_pending) s.wait_misses_impl();
    auto it = s.banks.find(layer);
    State::Bank * bank = it == s.banks.end() ? nullptr : it->second.get();
    // TQFN_PF_EVENTS: the compute stream waits only for the prefetch copies of slots this call reads (one event per
    // prefetched expert, recorded in priority order); slots still in flight stay pending and are not demand victims.
    static const bool pf_events = std::getenv("TQFN_PF_EVENTS") != nullptr;
    if (bank && bank->pending) {
        ++s.counters.pf_pending;
        const bool running = bank->ready && cudaEventQuery(bank->ready) == cudaErrorNotReady;
        if (running) ++s.counters.pf_running;
        if (running && !pf_events) {  // the bank-wide wait below also covers copies of experts this call does not read
            for (int j = 0; j < bank->cap; ++j) if (bank->inflight[j] && !is_needed(bank->ids[j])) { ++s.counters.pf_waited_unread; break; }
        }
    }
    if (bank && bank->pending && pf_events && s.miss_stream && s.registered && s.prepacked) {
        // Event indices restart with every prefetch call, so the highest index is not the latest copy: wait for each
        // read slot's own event (an index re-recorded by a later call is a later point on the copy stream: still safe).
        std::vector<uint8_t> waited(bank->pf_ev.size(), 0);
        for (int j = 0; j < bank->cap; ++j) if (bank->inflight[j] && is_needed(bank->ids[j])) {
            const int e = bank->pf_seq[j];
            if (!waited[size_t(e)]) { checked(cudaStreamWaitEvent(s.stream, bank->pf_ev[size_t(e)], 0)); waited[size_t(e)] = 1; }
            bank->inflight[j] = 0;
        }
        bank->pending = std::any_of(bank->inflight.begin(), bank->inflight.end(), [](uint8_t f) { return f != 0; });
    } else
    if (bank && bank->pending) { checked(cudaStreamWaitEvent(s.stream, bank->ready, 0)); bank->pending = false; std::fill(bank->inflight.begin(), bank->inflight.end(), 0); }
    if (bank) for (int j = 0; j < bank->cap; ++j) if (is_needed(bank->ids[j])) {
        s.where[bank->ids[j]] = j;
        if (bank->speculative[j]) { ++s.prefetch_used; bank->speculative[j] = 0; }
    }
    std::vector<std::pair<int, std::array<ExpertPayload, 3>>> incoming;
    for (int id : needed) if (s.where[id] < 0) {
        auto payload = s.source(layer, id);
        for (int p = 0; p < 3; ++p) s.validate(payload[p], p, bank);
        incoming.emplace_back(id, payload);
    }
    if (!bank) {
        const auto & first = incoming.front().second;
        for (const auto & [id, payload] : incoming) for (int p = 0; p < 3; ++p)
            if (payload[p].type != first[p].type) throw std::invalid_argument("mixed cache: types vary within a layer projection");
        s.active = -1;
        auto created = s.create(layer, first); bank = created.get(); s.banks.emplace(layer, std::move(created));
    }
    s.active = -1;
    size_t need = 0;
    for (size_t k = 0; k < incoming.size(); ++k) for (int p = 0; p < 3; ++p) need += bank->stride[p] + bank->aux_stride[p];
    if (need > s.staging_bytes) {
        checked(cudaStreamSynchronize(s.stream)); if (s.miss_stream) checked(cudaStreamSynchronize(s.miss_stream));
        if (s.staging) cudaFreeHost(s.staging);
        if (s.device_staging) { cudaFree(s.device_staging); s.device_staging = nullptr; }
        checked(cudaMallocHost(reinterpret_cast<void **>(&s.staging), need)); s.staging_bytes = need;
        if (s.prepacked) checked(cudaMalloc(reinterpret_cast<void **>(&s.device_staging), need));
    }
    struct Scatter { char * dst; size_t src, bytes; };
    std::vector<Scatter> scatter; std::vector<tqllm_cuda::ScatterItem> direct;
    // Host staging is rewritten below; registered prepacked misses never touch it (stream order guards device_staging).
    bool host_idle = false;
    auto idle_host = [&] { if (!host_idle) { checked(cudaStreamSynchronize(s.stream)); host_idle = true; } };
    // Split mode: registered misses go to the miss stream once the compute stream has released this bank.
    const bool split = s.miss_stream && s.registered && s.prepacked;
    cudaStream_t up = split ? s.miss_stream : s.stream;
    if (split && !incoming.empty() && bank->used) checked(cudaStreamWaitEvent(up, bank->released, 0));
    const bool skip_inflight = pf_events && bank && bank->pending;
    if (split && !incoming.empty() && bank->ready && !skip_inflight) checked(cudaStreamWaitEvent(up, bank->ready, 0));  // a victim may be an in-flight prefetch
    bool waited_ready = !skip_inflight;
    size_t offset = 0;
    for (const auto & [id, payload] : incoming) {
        int target = -1;
        for (int pass = 0; pass < 2 && target < 0; ++pass) {
        if (pass == 1) {
            if (waited_ready) break;
            if (split && bank->ready) checked(cudaStreamWaitEvent(up, bank->ready, 0));  // every free slot is in flight
            waited_ready = true;
        }
        for (int j = 0; j < bank->cap; ++j) {
            const int old = bank->ids[j]; if (is_needed(old)) continue;
            if (pass == 0 && skip_inflight && bank->inflight[j]) continue;
            if (target < 0 || old < 0) target = j;
            else {
                const double vj = slot_value(*bank, j, s.frequency[size_t(layer) * s.experts + old], s.clock);
                const double vt = slot_value(*bank, target, s.frequency[size_t(layer) * s.experts + bank->ids[target]], s.clock);
                if (vj < vt || (vj == vt && s.clock - bank->ages[j] < s.clock - bank->ages[target])) target = j;
            }
            if (old < 0) break;
        }
        }
        if (target < 0) throw std::logic_error("mixed cache: no unpinned slot");
        bank->inflight[target] = 0;
        if (bank->ids[target] >= 0) ++s.counters.evictions;
        bank->ids[target] = -1; bank->speculative[target] = 0;
        const char * packed_src = s.prepacked ? s.prepacked(layer, id) : nullptr;  // null: not in the packed file
        if (packed_src && s.registered) {
            // Kernel reads straight from pinned host memory: demand misses skip the H2D DMA queue behind prefetches.
            size_t off = 0;
            for (int p = 0; p < 3; ++p) {
                const auto & d = bank->matrices[p];
                direct.push_back({packed_src + off, static_cast<char *>(const_cast<void *>(d.p)) + size_t(target) * bank->stride[p], bank->stride[p]});
                off += bank->stride[p];
                if (bank->aux_stride[p]) {
                    direct.push_back({packed_src + off, static_cast<char *>(const_cast<void *>(d.aux)) + size_t(target) * bank->aux_stride[p], bank->aux_stride[p]});
                    off += bank->aux_stride[p];
                }
            }
            s.counters.transfer_bytes += off;
        } else if (packed_src) {
            size_t total = 0; for (int p = 0; p < 3; ++p) total += bank->stride[p] + bank->aux_stride[p];
            if (s.registered) checked(cudaMemcpyAsync(s.device_staging + offset, packed_src, total, cudaMemcpyHostToDevice, up));
            else { idle_host(); std::memcpy(s.staging + offset, packed_src, total); }
            for (int p = 0; p < 3; ++p) {
                const auto & destination = bank->matrices[p];
                scatter.push_back({static_cast<char *>(const_cast<void *>(destination.p)) + size_t(target) * bank->stride[p], offset, bank->stride[p]});
                offset += bank->stride[p];
                if (bank->aux_stride[p]) {
                    scatter.push_back({static_cast<char *>(const_cast<void *>(destination.aux)) + size_t(target) * bank->aux_stride[p], offset, bank->aux_stride[p]});
                    offset += bank->aux_stride[p];
                }
            }
            s.counters.transfer_bytes += total;
        } else
        for (int p = 0; p < 3 && (idle_host(), true); ++p) {
            const int cols = p == 2 ? s.ff : s.hidden, rows = p == 2 ? s.hidden : s.ff;
            // Repack straight into pinned staging; the loader's row data and aux land there back to back.
            std::vector<std::pair<const void *, size_t>> pieces;
            Upload stage = [&](const void * data, size_t bytes) {
                if (offset + bytes > s.staging_bytes) throw std::invalid_argument("mixed cache: staging overflow");
                std::memcpy(s.staging + offset, data, bytes); const void * at = s.staging + offset;
                offset += bytes; pieces.push_back({at, bytes}); return at;
            };
            const auto m = weight_format(payload[p].type)->load(
                std::vector<char>(payload[p].bytes.data, payload[p].bytes.data + payload[p].bytes.size), rows, cols, stage, nullptr);
            const auto & destination = bank->matrices[p];
            if (m.fmt != destination.fmt || m.type != destination.type) throw std::invalid_argument("mixed cache: per-expert runtime format changed");
            checked(cudaMemcpyAsync(static_cast<char *>(const_cast<void *>(destination.p)) + size_t(target) * bank->stride[p],
                                    m.p, bank->stride[p], cudaMemcpyHostToDevice, s.stream));
            s.counters.transfer_bytes += bank->stride[p];
            if (bank->aux_stride[p]) {
                checked(cudaMemcpyAsync(static_cast<char *>(const_cast<void *>(destination.aux)) + size_t(target) * bank->aux_stride[p],
                                        m.aux, bank->aux_stride[p], cudaMemcpyHostToDevice, s.stream));
                s.counters.transfer_bytes += bank->aux_stride[p];
            }
        }
        bank->ids[target] = id; bank->ages[target] = ++s.clock; bank->pf_age[target] = 0; s.where[id] = target;
        ++s.counters.misses;
    }
    if (!scatter.empty()) {
        if (!s.registered) checked(cudaMemcpyAsync(s.device_staging, s.staging, offset, cudaMemcpyHostToDevice, s.stream));
        std::vector<tqllm_cuda::ScatterItem> items; items.reserve(scatter.size());
        for (const auto & c : scatter) items.push_back({s.device_staging + c.src, c.dst, c.bytes});
        tqllm_cuda::scatter_copy(items.data(), int(items.size()), s.registered ? up : s.stream);
    }
    constexpr int kZeroCopyParts = 2;  // blocks per item for kernel reads of pinned memory
    if (!direct.empty()) tqllm_cuda::scatter_copy(direct.data(), int(direct.size()), up, kZeroCopyParts);
    if (split && (!scatter.empty() || !direct.empty())) {
        if (!s.miss_done) checked(cudaEventCreateWithFlags(&s.miss_done, cudaEventDisableTiming));
        checked(cudaEventRecord(s.miss_done, up)); s.miss_pending = true;
    }
    s.missed.assign(ids.size(), 0);
    for (const auto & in : incoming) s.mark[in.first] = s.gen + 0x80000000u;  // tag misses for the per-assignment mask
    for (size_t a = 0; a < ids.size(); ++a) s.missed[a] = s.mark[ids[a]] == s.gen + 0x80000000u ? 1 : 0;
    for (const auto & in : incoming) s.mark[in.first] = s.gen;
    s.counters.hits += needed.size() - incoming.size();
    std::vector<int> physical; physical.reserve(ids.size());
    for (int id : ids) { physical.push_back(s.where[id]); ++s.frequency[size_t(layer) * s.experts + id]; bank->ages[s.where[id]] = ++s.clock; }
    bank->touched = ++s.clock; s.active = layer; return physical;
}
bool MixedExpertCache::prefetch_released(int layer, const std::vector<int> & ids, cudaStream_t copy) {
    auto & s = *state_;
    if (!s.prepacked || !s.registered || layer < 0 || layer >= s.layers || layer == s.active) return false;
    auto it = s.banks.find(layer); if (it == s.banks.end() || !it->second->used) return false;
    prefetch_impl(layer, ids, copy, true); return true;
}
void MixedExpertCache::prefetch(int layer, const std::vector<int> & ids, cudaStream_t copy) { prefetch_impl(layer, ids, copy, false); }
void MixedExpertCache::prefetch_impl(int layer, const std::vector<int> & ids, cudaStream_t copy, bool after_release) {
    auto & s = *state_;
    if (!s.prepacked || !s.registered || layer < 0 || layer >= s.layers) return;
    auto it = s.banks.find(layer); if (it == s.banks.end()) return;  // bank shape comes from a real prepare first
    auto * bank = it->second.get();
    // Keep the caller's rank order; entries from strict_from on (ranks 4-6) are only admitted over a stale or rarely
    // used victim (policy found by trace replay).
    std::vector<int> wanted; wanted.reserve(ids.size());
    {
        std::set<int> seen, resident(bank->ids.begin(), bank->ids.end());
        for (size_t q = 0; q < ids.size(); ++q) {
            const int id = ids[q];
            if (resident.count(id) || !seen.insert(id).second) continue;
            wanted.push_back(int(q) >= s.pf_strict_from ? -2 - id : id);  // negative: strict admission
        }
    }
    if (wanted.empty()) return;
    std::set<int> keep(ids.begin(), ids.end());
    if (after_release) checked(cudaStreamWaitEvent(copy, bank->released, 0));  // last reader: the bank's previous use
    else {
        // The compute stream may still read this bank (previous token): order the copies after it.
        cudaEvent_t done; checked(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        checked(cudaEventRecord(done, s.stream)); checked(cudaStreamWaitEvent(copy, done, 0)); cudaEventDestroy(done);
    }
    if (s.miss_pending) checked(cudaStreamWaitEvent(copy, s.miss_done, 0));  // demand misses keep the link first
    size_t total = 0; for (int p = 0; p < 3; ++p) total += bank->stride[p] + bank->aux_stride[p];
    constexpr int kCap = 16;  // experts per staged batch (the staging buffer is reused in copy-stream order)
    if (s.pf_bytes < kCap * total) {
        checked(cudaStreamSynchronize(copy));
        if (s.pf_staging) cudaFree(s.pf_staging);
        checked(cudaMalloc(reinterpret_cast<void **>(&s.pf_staging), kCap * total)); s.pf_bytes = kCap * total;
    }
    std::vector<tqllm_cuda::ScatterItem> items; int staged = 0, n_ev = 0;
    auto flush = [&] {
        if (!items.empty()) tqllm_cuda::scatter_copy(items.data(), int(items.size()), copy);
        items.clear(); staged = 0;
    };
    for (int wid : wanted) {
        const bool strict = wid < -1; const int id = strict ? -2 - wid : wid;
        if (id < 0 || id >= s.experts) continue;
        int target = -1;
        for (int j = 0; j < bank->cap; ++j) {
            const int old = bank->ids[j]; if (keep.count(old)) continue;
            if (target < 0 || old < 0) { target = j; if (old < 0) break; continue; }
            const double vj = slot_value(*bank, j, s.frequency[size_t(layer) * s.experts + old], s.clock);
            const double vt = slot_value(*bank, target, s.frequency[size_t(layer) * s.experts + bank->ids[target]], s.clock);
            if (vj < vt || (vj == vt && s.clock - bank->ages[j] < s.clock - bank->ages[target])) target = j;
        }
        if (target < 0) break;
        if (strict && bank->ids[target] >= 0) {
            const uint64_t age = s.clock - bank->ages[target], f = s.frequency[size_t(layer) * s.experts + bank->ids[target]];
            constexpr double decay = kValueDecay, mage = 8.0, mval = 2.0;  // a victim younger than mage with value >= mval stays
            constexpr uint64_t mf = 4;                                       // ... and so does one used >= mf times within 2 x mage
            const double val = double(f) / (1.0 + decay * double(age));
            if ((double(age) < mage && val >= mval) || (f >= mf && double(age) < 2 * mage)) continue;  // fresh or hot victim: skip
        }
        if (bank->ids[target] >= 0) ++s.counters.evictions;
        const char * src = s.prepacked(layer, id); size_t off = 0;
        {
            static const bool pf_events = std::getenv("TQFN_PF_EVENTS") != nullptr;
            if (staged == kCap || (pf_events && staged)) flush();
            char * stage = s.pf_staging + size_t(staged) * total;
            checked(cudaMemcpyAsync(stage, src, total, cudaMemcpyHostToDevice, copy));  // the expert's packed block in one DMA
            for (int p = 0; p < 3; ++p) {
                const auto & d = bank->matrices[p];
                items.push_back({stage + off, static_cast<char *>(const_cast<void *>(d.p)) + size_t(target) * bank->stride[p], bank->stride[p]});
                off += bank->stride[p];
                if (bank->aux_stride[p]) {
                    items.push_back({stage + off, static_cast<char *>(const_cast<void *>(d.aux)) + size_t(target) * bank->aux_stride[p], bank->aux_stride[p]});
                    off += bank->aux_stride[p];
                }
            }
            ++staged;
        }
        s.counters.transfer_bytes += off; ++s.prefetch_count;
        bank->ids[target] = id; bank->ages[target] = ++s.clock; bank->speculative[target] = 1; bank->inflight[target] = 1; bank->pf_age[target] = s.clock;
        if (static const bool pf_events = std::getenv("TQFN_PF_EVENTS") != nullptr; pf_events) {
            flush();
            if (size_t(n_ev) == bank->pf_ev.size()) { cudaEvent_t e; checked(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); bank->pf_ev.push_back(e); }
            checked(cudaEventRecord(bank->pf_ev[size_t(n_ev)], copy)); bank->pf_seq[target] = n_ev++;
        }
        ++s.frequency[size_t(layer) * s.experts + id];
        keep.insert(id);
    }
    flush();
    if (!bank->ready) checked(cudaEventCreateWithFlags(&bank->ready, cudaEventDisableTiming));
    checked(cudaEventRecord(bank->ready, copy)); bank->pending = true;
}
void MixedExpertCache::set_miss_stream(cudaStream_t m) { state_->miss_stream = m; }
void MixedExpertCache::set_prefetch_strict(int from) { state_->pf_strict_from = from; }
bool MixedExpertCache::resident(int layer, int expert) const {
    const auto & s = *state_; auto it = s.banks.find(layer); if (it == s.banks.end()) return false;
    for (int id : it->second->ids) if (id == expert) return true;
    return false;
}
bool MixedExpertCache::split() const { return state_->miss_stream && state_->registered && state_->prepacked; }
const std::vector<uint8_t> & MixedExpertCache::last_missed() const { return state_->missed; }
void MixedExpertCache::wait_misses() { state_->wait_misses_impl(); }
uint64_t MixedExpertCache::prefetched() const { return state_->prefetch_count; }
uint64_t MixedExpertCache::prefetch_used() const { return state_->prefetch_used; }
void MixedExpertCache::set_prepacked(Prepacked p, bool registered) { state_->prepacked = std::move(p); state_->registered = registered; }
Mat MixedExpertCache::gate() const { if (state_->active < 0) throw std::logic_error("mixed cache: no active bank"); return state_->banks.at(state_->active)->matrices[0]; }
Mat MixedExpertCache::up() const { if (state_->active < 0) throw std::logic_error("mixed cache: no active bank"); return state_->banks.at(state_->active)->matrices[1]; }
Mat MixedExpertCache::down() const { if (state_->active < 0) throw std::logic_error("mixed cache: no active bank"); return state_->banks.at(state_->active)->matrices[2]; }
ExpertCacheStats MixedExpertCache::stats() const { return state_->counters; }
size_t MixedExpertCache::peak_device_bytes() const { return state_->peak; }
int MixedExpertCache::slots() const { auto & s = *state_; return s.layer_cap.empty() ? s.capacity : *std::max_element(s.layer_cap.begin(), s.layer_cap.end()); }
int MixedExpertCache::slots(int layer) const { return state_->cap_of(layer); }
void MixedExpertCache::set_layer_slots(const std::vector<int> & caps) {
    auto & s = *state_;
    if (!s.banks.empty()) throw std::logic_error("mixed cache: per-layer slots must be set before the first bank");
    if (int(caps.size()) != s.layers) throw std::invalid_argument("mixed cache: per-layer slots need one count per layer");
    for (int c : caps) if (c <= 0 || c > s.experts) throw std::invalid_argument("mixed cache: per-layer slot count out of range");
    s.layer_cap = caps;
}
int MixedExpertCache::experts() const { return state_->experts; }
int MixedExpertCache::layers() const { return state_->layers; }
bool MixedExpertCache::bank_ready(int layer) const { return state_->banks.count(layer) != 0; }
MixedExpertCache::BankLayout MixedExpertCache::bank_layout(int layer) const {
    auto & s = *state_; const auto & b = *s.banks.at(layer);
    BankLayout l; int k = 0;
    for (int p = 0; p < 3; ++p) {
        l.dst[k] = static_cast<char *>(const_cast<void *>(b.matrices[p].p)); l.piece[k] = b.stride[p]; ++k;
        l.dst[k] = b.aux_stride[p] ? static_cast<char *>(const_cast<void *>(b.matrices[p].aux)) : nullptr; l.piece[k] = b.aux_stride[p]; ++k;
    }
    for (int i = 0; i < 6; ++i) l.block += l.piece[i];
    if (!s.prepacked) return l;  // sizes only
    l.src0 = s.prepacked(layer, 0);
    if (!l.src0 || s.prepacked(layer, 1) != l.src0 + l.block) throw std::logic_error("mixed cache: prepacked blocks are not contiguous in the bank layout");
    return l;
}
MixedExpertCache::BankState MixedExpertCache::bank_state(int layer) const {
    auto & s = *state_; auto & b = *s.banks.at(layer);
    s.wait_misses_impl(); checked(cudaDeviceSynchronize()); b.pending = false;
    BankState st; st.ids = b.ids; st.ages = b.ages;
    st.freq.assign(s.frequency.begin() + size_t(layer) * s.experts, s.frequency.begin() + size_t(layer + 1) * s.experts);
    return st;
}
void MixedExpertCache::bank_import(int layer, const BankState & st) {
    auto & s = *state_; auto & b = *s.banks.at(layer);
    if (int(st.ids.size()) != b.cap || int(st.ages.size()) != b.cap || int(st.freq.size()) != s.experts)
        throw std::invalid_argument("mixed cache: imported bank state has the wrong shape");
    b.ids = st.ids; b.ages = st.ages; std::fill(b.speculative.begin(), b.speculative.end(), 0); std::fill(b.inflight.begin(), b.inflight.end(), 0); std::fill(b.pf_age.begin(), b.pf_age.end(), 0);
    std::copy(st.freq.begin(), st.freq.end(), s.frequency.begin() + size_t(layer) * s.experts);
}
Mat MixedExpertCache::bank_mat(int layer, int projection) const { return state_->banks.at(layer)->matrices.at(projection); }
void MixedExpertCache::add_device_counters(uint64_t hits, uint64_t misses, uint64_t prefetched, uint64_t bytes) {
    auto & s = *state_; s.counters.hits += hits; s.counters.misses += misses; s.counters.transfer_bytes += bytes; s.prefetch_count += prefetched;
}
uint64_t MixedExpertCache::clock() const { return state_->clock; }
void MixedExpertCache::set_clock(uint64_t c) { state_->clock = c; }
void MixedExpertCache::clear() {
    auto & s = *state_; s.wait_misses_impl(); checked(cudaStreamSynchronize(s.stream)); s.active = -1; s.banks.clear();
    std::fill(s.frequency.begin(), s.frequency.end(), 0); s.counters = {}; s.peak = s.clock = 0;
}
}
