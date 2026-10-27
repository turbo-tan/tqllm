#include "tq3_expert_cache.hpp"
#include "expert_bank_plan.hpp"
#include "moe_kernels.cuh"
#include "debug.hpp"
#include "device_free_cuda.hpp"
#include <cstdio>
#include <algorithm>
#include <limits>
#include <map>
#include <cstring>
#include <set>
#include <tuple>
#include <cstdlib>
#include <unordered_map>
#include <stdexcept>
#include <atomic>
#include <exception>
#include <thread>
#include <mutex>
// TQFN_* switches that won their A/Bs are on by default; set NAME=0 to turn one off
static inline bool tqfn_default_on(const char * n) { const char * v = std::getenv(n); return !v || (v[0] && v[0] != '0'); }

namespace tqllm {
namespace {
void checked(cudaError_t s, int line = __builtin_LINE()) { if (s != cudaSuccess) throw std::runtime_error(std::string(cudaGetErrorString(s)) + " (mixed_expert_cache.cpp:" + std::to_string(line) + ")"); }
}
constexpr int kPfBatch = 16;  // experts per staged prefetch batch (the staging buffer is reused in copy-stream order)
static bool host_coherent() {  // GB10-class: the GPU reads pageable host memory through the host page tables (C2C)
    static const bool v = [] { int dev = 0, on = 0; cudaGetDevice(&dev);
        return cudaDeviceGetAttribute(&on, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) == cudaSuccess && on; }();
    return v;
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
        Bank(State & s, int c) : owner(s), cap(c), ids(c, -1), ages(c), speculative(c, 0), inflight(c, 0), pf_age(c, 0) {}
        ~Bank() { if (ready) cudaEventDestroy(ready); if (released) cudaEventDestroy(released); owner.free(memory); }
    };
    int layers, experts, hidden, ff, capacity, active = -1;
    std::vector<int> layer_cap;  // per-layer slots (TQFN_LAYER_SLOTS); empty: capacity for every layer
    int cap_of(int layer) const { return layer_cap.empty() ? capacity : layer_cap[size_t(layer)]; }
    // Two slot tiers (set_layer_tiers): a layer's bank is built at its target when the free memory holds it, else at the
    // largest count that fits but never below its floor, instead of evicting other layers' whole banks (a long prompt's
    // attention state left two layers without room: each step their bank evicted another layer's, 67 -> 4.9 tok/s).
    std::vector<int> layer_target, layer_floor; size_t tier_keep = size_t(256) << 20;
    std::vector<char> planned;  // plan_slots sized the layer's next bank: prepare builds it at that size (no second decision)
    size_t slot_bytes(int layer) const {
        const auto it = saved_piece.find(layer); if (it == saved_piece.end()) return 0;
        size_t b = 0; for (size_t x : it->second) b += x; return b;
    }
    void choose_cap(int layer) {
        if (layer_target.empty() || layer_cap.empty()) return;
        const int target = layer_target[size_t(layer)], floor = layer_floor[size_t(layer)];
        const size_t per = slot_bytes(layer);
        if (!per) { layer_cap[size_t(layer)] = target; return; }
        { cudaMemPool_t pool; if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) cudaMemPoolTrimTo(pool, 0); }
        size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
        // the free memory is shared evenly by every layer still without its bank (a refill after a long prompt): the
        // first banks refilled must not take their target and leave the last ones none
        size_t share = per, pending = 1;
        for (int l = 0; l < layers; ++l) if (l != layer && !banks.count(l) && saved_piece.count(l)) { share += slot_bytes(l); ++pending; }
        // each bank's up to six allocations carry upload()'s 64 KB tail beside the slots
        const size_t margin = tier_keep + pending * 6 * (size_t(64) << 10), fit = fr > margin ? (fr - margin) / share : 0;
        layer_cap[size_t(layer)] = int(std::clamp<size_t>(fit, size_t(floor), size_t(target)));
    }
    size_t budget, live = 0, peak = 0;
    Source source;
    BatchSource batch_source;
    AsyncBatchSource async_batch_source;
    cudaStream_t stream;
    ExpertCacheStats counters;
    std::map<int, std::unique_ptr<Bank>> banks;
    std::vector<uint64_t> frequency;
    uint64_t clock = 0;
    // Misses repack into this pinned buffer and copy straight into their slots (one sync per prepare).
    char * staging = nullptr; size_t staging_bytes = 0;
    char * staging_alt = nullptr; size_t staging_alt_bytes = 0;
    cudaEvent_t staging_ready = nullptr, staging_alt_ready = nullptr;
    bool staged_misses = false;
    char * device_staging = nullptr;  // prepacked misses: one upload per prepare, then device-side scatter
    cudaStream_t miss_stream = nullptr; cudaEvent_t miss_done = nullptr; bool miss_pending = false;
    // TQueue: the cache owns its demand-miss stream (high priority; a caller's stream only switches split mode on, so the
    // trunk and MTP caches no longer share one request's stream) and the event that orders a prefetch after the compute stream
    cudaStream_t own_miss = nullptr; cudaEvent_t order_ev = nullptr;
    std::vector<uint8_t> missed;
    std::vector<uint32_t> mark; std::vector<int> where; uint32_t gen = 0;  // prepare(): flat per-expert bookkeeping
    char * pf_staging = nullptr; size_t pf_bytes = 0;            // prefetch: one DMA per expert, then one scatter launch
    int pf_strict_from = 1 << 30;  // set_prefetch_strict: caller index from which prefetches are admitted strictly
    Prepacked prepacked; bool registered = false; uint64_t prefetch_count = 0, prefetch_used = 0;
    // each layer's matrix descriptors (pointers cleared) and row-layout pieces, kept from its first bank: a layer whose bank
    // was evicted still has a layout (bank_layout / bank_mat), so prefill can run its experts from staging
    std::map<int, std::array<Mat, 3>> saved_mat; std::map<int, std::array<size_t, 6>> saved_piece;
    std::map<int, std::array<uint32_t, 3>> saved_type; std::map<int, int> saved_cap;
    ~State() {
        cudaDeviceSynchronize();
        if (miss_done) cudaEventDestroy(miss_done);
        if (order_ev) cudaEventDestroy(order_ev);
        if (own_miss) cudaStreamDestroy(own_miss);
        banks.clear();
        if (staging_ready) cudaEventDestroy(staging_ready);
        if (staging_alt_ready) cudaEventDestroy(staging_alt_ready);
        if (staging_alt) cudaFreeHost(staging_alt);
        if (staging) cudaFreeHost(staging);
        if (device_staging) cudaFree(device_staging);
        if (pf_staging) cudaFree(pf_staging);
    }
    State(int l, int e, int h, int f, int c, size_t b, Source s, cudaStream_t st)
        : layers(l), experts(e), hidden(h), ff(f), capacity(c), budget(b), source(std::move(s)), stream(st) {
        if (l <= 0 || e <= 0 || h <= 0 || f <= 0 || c <= 0 || c > e || !b || !source ||
            int64_t(c) * std::max(h, f) > std::numeric_limits<int>::max() ||
            uint64_t(l) * e > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
            throw std::invalid_argument("mixed cache: invalid dimensions or source");
        frequency.resize(size_t(l) * e); mark.assign(size_t(e), 0u); where.assign(size_t(e), -1);
    }
    // Grows the prefetch staging; false (the old buffer still valid) when the device has no room: prefetches are only
    // speculative, so the caller skips them. Banks size it when created, before a prefill borrow and refill can take the room.
    bool grow_pf(size_t need) {
        if (pf_bytes >= need) return true;
        char * p = nullptr;
        if (cudaMalloc(reinterpret_cast<void **>(&p), need) != cudaSuccess) { cudaGetLastError(); return false; }
        checked(cudaDeviceSynchronize());  // a copy may still read the old buffer
        if (pf_staging) cudaFree(pf_staging);
        pf_staging = p; pf_bytes = need; return true;
    }
    static size_t expert_bytes(const Bank & b) { size_t t = 0; for (int p = 0; p < 3; ++p) t += b.stride[p] + b.aux_stride[p]; return t; }
    void wait_misses_impl() { if (miss_pending) { checked(cudaStreamWaitEvent(stream, miss_done, 0)); miss_pending = false; } }
    void free(std::vector<Allocation> & memory) {
        cudaDeviceSynchronize();  // a miss or prefetch copy may still write the bank, not only the compute stream
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
        void * p = nullptr;
        // + a tail: the MMQ kernels read the last row up to a multiple of 512 columns (ggml pads quantized matrices so).
        // The device can be fuller than at start-up (a long context's attention state): a failed allocation evicts the most
        // recently used other bank and tries again, so the bank count follows the free memory instead of failing the request.
        // Layers run in a fixed cycle, so the least recently used bank is the next one needed: evicting it made every layer
        // evict the next (all misses); the most recent one is needed last, and a step rebuilds at most one bank.
        cudaError_t e = cudaMalloc(&p, bytes + (size_t(64) << 10));
        while (e == cudaErrorMemoryAllocation) {
            cudaGetLastError();
            auto victim = banks.end();
            for (auto it = banks.begin(); it != banks.end(); ++it)
                if (it->first != protect && (victim == banks.end() || it->second->touched > victim->second->touched)) victim = it;
            if (victim == banks.end()) break;
            for (int id : victim->second->ids) if (id >= 0) ++counters.evictions;
            if (active == victim->first) active = -1;
            banks.erase(victim);
            e = cudaMalloc(&p, bytes + (size_t(64) << 10));
        }
        if (e != cudaSuccess) {
            cudaGetLastError(); size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
            throw std::runtime_error(std::string(cudaGetErrorString(e)) + ": expert bank upload of " + std::to_string(bytes >> 20) +
                                     " MiB for layer " + std::to_string(protect) + " (cache " + std::to_string(live >> 20) + " of " +
                                     std::to_string(budget >> 20) + " MiB), " + std::to_string(fr >> 20) + " MiB free");
        }
        owned.push_back({p, bytes}); live += bytes; peak = std::max(peak, live); counters.device_bytes = live;
        if (host) { checked(cudaMemcpyAsync(p, host, bytes, cudaMemcpyHostToDevice, stream)); counters.transfer_bytes += bytes; }
        return p;
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
        // a layer whose bank existed before: its per-slot layout is known, so the bank is just device memory at any slot
        // count (no zero payload repacked on the CPU and uploaded: a slot tier refill at a new size took 4.8 s instead of
        // 0.76 s); every slot starts empty and is written before it is read
        if (const auto c = saved_cap.find(layer); c != saved_cap.end() && c->second > 0) {
            const int saved = c->second;
            const auto & sm = saved_mat.at(layer); const auto & sp = saved_piece.at(layer); const auto & st = saved_type.at(layer);
            for (int p = 0; p < 3; ++p) {
                bank->types[p] = st[size_t(p)]; bank->raw_stride[p] = sample[p].bytes.size;
                bank->stride[p] = sp[size_t(2 * p)]; bank->aux_stride[p] = sp[size_t(2 * p + 1)];
                Mat m = sm[size_t(p)]; m.rows = m.rows / saved * capacity;
                m.p = upload(nullptr, bank->stride[p] * size_t(capacity), bank->memory, layer);
                m.aux = bank->aux_stride[p] ? upload(nullptr, bank->aux_stride[p] * size_t(capacity), bank->memory, layer) : nullptr;
                bank->matrices[p] = m;
            }
            return bank;
        }
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
        checked(cudaStreamSynchronize(stream));
        auto & sm = saved_mat[layer]; auto & sp = saved_piece[layer];
        for (int p = 0; p < 3; ++p) { sm[p] = bank->matrices[p]; sm[p].p = nullptr; sm[p].aux = nullptr; sp[2 * p] = bank->stride[p]; sp[2 * p + 1] = bank->aux_stride[p]; }
        saved_type[layer] = bank->types; saved_cap[layer] = capacity;
        return bank;
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
    if (!s.banks.count(layer) && !(layer < int(s.planned.size()) && s.planned[size_t(layer)])) s.choose_cap(layer);
    if (layer < int(s.planned.size())) s.planned[size_t(layer)] = 0;
    if (needed.size() > size_t(s.cap_of(layer))) throw std::invalid_argument("mixed cache: active set exceeds slots");
    if (s.miss_stream && s.active >= 0) {
        auto & prev = *s.banks.at(s.active);
        if (!prev.released) checked(cudaEventCreateWithFlags(&prev.released, cudaEventDisableTiming));
        checked(cudaEventRecord(prev.released, s.stream)); prev.used = true;
    }
    if (s.miss_pending) s.wait_misses_impl();
    auto it = s.banks.find(layer);
    State::Bank * bank = it == s.banks.end() ? nullptr : it->second.get();
    if (bank && bank->pending) {
        ++s.counters.pf_pending;
        const bool running = bank->ready && cudaEventQuery(bank->ready) == cudaErrorNotReady;
        if (running) ++s.counters.pf_running;
        if (running) {  // the bank-wide wait below also covers copies of experts this call does not read
            for (int j = 0; j < bank->cap; ++j) if (bank->inflight[j] && !is_needed(bank->ids[j])) { ++s.counters.pf_waited_unread; break; }
        }
    }
    if (bank && bank->pending) { checked(cudaStreamWaitEvent(s.stream, bank->ready, 0)); bank->pending = false; std::fill(bank->inflight.begin(), bank->inflight.end(), 0); }
    if (bank) for (int j = 0; j < bank->cap; ++j) if (is_needed(bank->ids[j])) {
        s.where[bank->ids[j]] = j;
        if (bank->speculative[j]) { ++s.prefetch_used; bank->speculative[j] = 0; }
    }
    std::vector<std::pair<int, std::array<ExpertPayload, 3>>> incoming;
    std::vector<int> missing;
    for(int id:needed)if(s.where[id]<0)missing.push_back(id);
    const bool stream_payloads = s.async_batch_source && bank && !missing.empty();
    std::vector<std::shared_future<std::array<ExpertPayload,3>>> pending;
    if (stream_payloads) {
        if (!s.staged_misses || s.prepacked) throw std::invalid_argument("mixed cache: async source requires row-layout staged misses");
        pending = s.async_batch_source(layer, missing);
        if (pending.size() != missing.size()) throw std::invalid_argument("mixed cache: async source count mismatch");
        for (auto & future : pending) if (!future.valid()) throw std::invalid_argument("mixed cache: invalid async source future");
        for (int id : missing) incoming.emplace_back(id, std::array<ExpertPayload,3>{});
    } else if(s.batch_source && !missing.empty()) {
        auto payloads=s.batch_source(layer,missing);
        if(payloads.size()!=missing.size())throw std::invalid_argument("mixed cache: batch source count mismatch");
        for(size_t i=0;i<missing.size();++i) {
            for(int p=0;p<3;++p)s.validate(payloads[i][p],p,bank);
            incoming.emplace_back(missing[i],std::move(payloads[i]));
        }
    } else for(int id:missing) {
        auto payload=s.source(layer,id);
        for(int p=0;p<3;++p)s.validate(payload[p],p,bank);
        incoming.emplace_back(id,std::move(payload));
    }
    if (!bank) {
        const auto & first = incoming.front().second;
        for (const auto & [id, payload] : incoming) for (int p = 0; p < 3; ++p)
            if (payload[p].type != first[p].type) throw std::invalid_argument("mixed cache: types vary within a layer projection");
        s.active = -1;
        auto created = s.create(layer, first); bank = created.get(); s.banks.emplace(layer, std::move(created));
        if (s.prepacked && s.registered) s.grow_pf(kPfBatch * State::expert_bytes(*bank));
    }
    s.active = -1;
    if (s.staged_misses && !incoming.empty()) {
        if (!stream_payloads) for (const auto & item : incoming) for (int p=0;p<3;++p)
            if (!item.second[p].row_layout || bank->aux_stride[p] || item.second[p].bytes.size != bank->stride[p])
                throw std::invalid_argument("mixed cache: staged misses require bank-layout data without aux");
        std::swap(s.staging, s.staging_alt); std::swap(s.staging_bytes, s.staging_alt_bytes);
        std::swap(s.staging_ready, s.staging_alt_ready);
        if (s.staging_ready) checked(cudaEventSynchronize(s.staging_ready));
    }
    size_t need = 0;
    for (size_t k = 0; k < incoming.size(); ++k) for (int p = 0; p < 3; ++p) need += bank->stride[p] + bank->aux_stride[p];
    if (need > s.staging_bytes) {
        checked(cudaStreamSynchronize(s.stream)); if (s.miss_stream) checked(cudaStreamSynchronize(s.miss_stream));
        if (s.staging) cudaFreeHost(s.staging);
        if (s.device_staging) { cudaFree(s.device_staging); s.device_staging = nullptr; }
        checked(cudaMallocHost(reinterpret_cast<void **>(&s.staging), need)); s.staging_bytes = need;
        if (tqllm::debug_on("mem")) std::fprintf(stderr, "STAGING grow layer %d: %zu experts, %.1f MB pinned%s\n", layer, incoming.size(), need / 1048576.0, s.prepacked ? " + device" : "");
        if (s.prepacked) checked(cudaMalloc(reinterpret_cast<void **>(&s.device_staging), need));
    }
    struct Scatter { char * dst; size_t src, bytes; };
    std::vector<Scatter> scatter; std::vector<tqllm_cuda::ScatterItem> direct;
    // Host staging is rewritten below; registered prepacked misses never touch it (stream order guards device_staging).
    bool host_idle = false;
    auto idle_host = [&] { if (!host_idle) { if (!s.staged_misses) checked(cudaStreamSynchronize(s.stream)); host_idle = true; } };
    // Split mode: registered misses go to the miss stream once the compute stream has released this bank.
    const bool split = s.miss_stream && ((s.registered && s.prepacked) || s.staged_misses);
    cudaStream_t up = split ? s.miss_stream : s.stream;
    if (split && !incoming.empty() && bank->used) checked(cudaStreamWaitEvent(up, bank->released, 0));
    if (split && !incoming.empty() && bank->ready) checked(cudaStreamWaitEvent(up, bank->ready, 0));  // a victim may be an in-flight prefetch
    bool dma_issued = false;
    size_t offset = 0;
    size_t read_at = 0;
    // Misses that need the loader (a format conversion) are converted on worker threads first, each into its own slice of
    // the staging buffer, in the order the loop below uploads them: one thread faulting in and repacking one expert after
    // another was the whole miss cost (GLM-5.3 on a GB10: ~0.3 GB/s).
    struct Repacked { std::array<const void *, 3> p{}, aux{}; };
    std::vector<Repacked> repacked;
    size_t per_expert = 0;
    if (!incoming.empty()) for (int p = 0; p < 3; ++p) per_expert += bank->stride[p] + bank->aux_stride[p];
    bool all_loader = !stream_payloads && !s.staged_misses && incoming.size() > 1;
    if (all_loader && s.prepacked) for (const auto & item : incoming) if (s.prepacked(layer, item.first)) { all_loader = false; break; }
    if (all_loader) {
        idle_host();
        repacked.resize(incoming.size());
        std::atomic<size_t> next{0}; std::exception_ptr error; std::mutex error_mutex;
        auto work = [&] {
            for (size_t k; (k = next++) < incoming.size();) {
                try {
                    size_t at = k * per_expert;
                    const size_t end = at + per_expert;
                    for (int p = 0; p < 3; ++p) {
                        const int cols = p == 2 ? s.ff : s.hidden, rows = p == 2 ? s.hidden : s.ff;
                        Upload stage = [&](const void * data, size_t bytes) {
                            if (at + bytes > end) throw std::invalid_argument("mixed cache: repacked expert larger than its slot");
                            std::memcpy(s.staging + at, data, bytes); const void * where = s.staging + at; at += bytes; return where;
                        };
                        const auto & payload = incoming[k].second[p];
                        const auto m = weight_format(payload.type)->load(
                            std::vector<char>(payload.bytes.data, payload.bytes.data + payload.bytes.size), rows, cols, stage, nullptr);
                        if (m.fmt != bank->matrices[p].fmt || m.type != bank->matrices[p].type)
                            throw std::invalid_argument("mixed cache: per-expert runtime format changed");
                        repacked[k].p[size_t(p)] = m.p; repacked[k].aux[size_t(p)] = m.aux;
                    }
                } catch (...) { std::lock_guard<std::mutex> lock(error_mutex); if (!error) error = std::current_exception(); }
            }
        };
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const size_t workers = std::min<size_t>(incoming.size(), std::min(16u, std::max(1u, hw / 2)));
        std::vector<std::thread> pool;
        for (size_t t = 1; t < workers; ++t) pool.emplace_back(work);
        work();
        for (auto & t : pool) t.join();
        if (error) std::rethrow_exception(error);
    }
    try {
    for (auto & [id, payload] : incoming) {
        if (stream_payloads) {
            payload = pending[read_at].get(); pending[read_at++] = {};
            for (int p=0; p<3; ++p) {
                s.validate(payload[p],p,bank);
                if (!payload[p].row_layout || bank->aux_stride[p] || payload[p].bytes.size != bank->stride[p])
                    throw std::invalid_argument("mixed cache: async source differs from bank row layout");
            }
        }
        int target = -1;
        for (int j = 0; j < bank->cap; ++j) {
            const int old = bank->ids[j]; if (is_needed(old)) continue;
            if (target < 0 || old < 0) target = j;
            else {
                const double vj = slot_value(*bank, j, s.frequency[size_t(layer) * s.experts + old], s.clock);
                const double vt = slot_value(*bank, target, s.frequency[size_t(layer) * s.experts + bank->ids[target]], s.clock);
                if (vj < vt || (vj == vt && s.clock - bank->ages[j] < s.clock - bank->ages[target])) target = j;
            }
            if (old < 0) break;
        }
        if (target < 0) throw std::logic_error("mixed cache: no unpinned slot");
        bank->inflight[target] = 0;
        if (bank->ids[target] >= 0) ++s.counters.evictions;
        bank->ids[target] = -1; bank->speculative[target] = 0;
        const char * packed_src = s.prepacked ? s.prepacked(layer, id) : nullptr;  // null: not in the packed file
        if (packed_src && s.registered && split && !host_coherent()) {
            // F25: on a PCIe GPU split-mode misses go by the copy engine straight into the slot on the miss stream: zero-copy kernels
            // beside the compute stream faulted / hung the 3090's front end in long decodes, and in compute-stream order they cost ~10 %
            // decode. GB10 (C2C, coherent host memory) never faulted and reads host memory faster by kernel: it keeps zero-copy.
            size_t off = 0;
            for (int p = 0; p < 3; ++p) {
                const auto & d = bank->matrices[p];
                checked(cudaMemcpyAsync(static_cast<char *>(const_cast<void *>(d.p)) + size_t(target) * bank->stride[p], packed_src + off, bank->stride[p], cudaMemcpyHostToDevice, up));
                off += bank->stride[p];
                if (bank->aux_stride[p]) {
                    checked(cudaMemcpyAsync(static_cast<char *>(const_cast<void *>(d.aux)) + size_t(target) * bank->aux_stride[p], packed_src + off, bank->aux_stride[p], cudaMemcpyHostToDevice, up));
                    off += bank->aux_stride[p];
                }
            }
            s.counters.transfer_bytes += off; dma_issued = true;
        } else if (packed_src && s.registered) {
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
            if (!repacked.empty()) {
                const auto & r = repacked[size_t(reinterpret_cast<const char *>(&id) - reinterpret_cast<const char *>(&incoming.front().first)) / sizeof(incoming.front())];
                const auto & destination = bank->matrices[p];
                checked(cudaMemcpyAsync(static_cast<char *>(const_cast<void *>(destination.p)) + size_t(target) * bank->stride[p],
                                        r.p[size_t(p)], bank->stride[p], cudaMemcpyHostToDevice, s.stream));
                s.counters.transfer_bytes += bank->stride[p];
                if (bank->aux_stride[p]) {
                    checked(cudaMemcpyAsync(static_cast<char *>(const_cast<void *>(destination.aux)) + size_t(target) * bank->aux_stride[p],
                                            r.aux[size_t(p)], bank->aux_stride[p], cudaMemcpyHostToDevice, s.stream));
                    s.counters.transfer_bytes += bank->aux_stride[p];
                }
                offset += bank->stride[p] + bank->aux_stride[p];
                continue;
            }
            if (s.staged_misses) {
                const auto & destination = bank->matrices[p];
                std::memcpy(s.staging + offset, payload[p].bytes.data, bank->stride[p]);
                checked(cudaMemcpyAsync(static_cast<char *>(const_cast<void *>(destination.p)) + size_t(target) * bank->stride[p],
                        s.staging + offset, bank->stride[p], cudaMemcpyHostToDevice, up));
                offset += bank->stride[p]; s.counters.transfer_bytes += bank->stride[p]; dma_issued = true;
                continue;
            }
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
        if (stream_payloads) payload = {}; // pinned staging owns the bytes until its completion event
    }
    } catch (...) {
        // A later read may fail after earlier records started DMA. Drain before staging can be reused.
        if (dma_issued) checked(cudaStreamSynchronize(up));
        throw;
    }
    if (!scatter.empty()) {
        if (!s.registered) checked(cudaMemcpyAsync(s.device_staging, s.staging, offset, cudaMemcpyHostToDevice, s.stream));
        std::vector<tqllm_cuda::ScatterItem> items; items.reserve(scatter.size());
        for (const auto & c : scatter) items.push_back({s.device_staging + c.src, c.dst, c.bytes});
        tqllm_cuda::scatter_copy(items.data(), int(items.size()), s.registered ? up : s.stream);
    }
    constexpr int kZeroCopyParts = 2;  // blocks per item for kernel reads of pinned memory
    // Zero-copy misses: on the miss stream on GB10 (never faulted, fastest there); otherwise only without split mode.
    if (!direct.empty()) tqllm_cuda::scatter_copy(direct.data(), int(direct.size()), split ? up : s.stream, kZeroCopyParts);
    if (s.staged_misses && dma_issued) {
        if (!s.staging_ready) checked(cudaEventCreateWithFlags(&s.staging_ready, cudaEventDisableTiming));
        checked(cudaEventRecord(s.staging_ready, up));
    }
    if (split && (!scatter.empty() || !direct.empty() || dma_issued)) {
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
    auto it = s.banks.find(layer);
    if (it == s.banks.end() || !it->second->used) return false;
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
        if (!s.order_ev) checked(cudaEventCreateWithFlags(&s.order_ev, cudaEventDisableTiming));
        checked(cudaEventRecord(s.order_ev, s.stream)); checked(cudaStreamWaitEvent(copy, s.order_ev, 0));
    }
    if (s.miss_pending) checked(cudaStreamWaitEvent(copy, s.miss_done, 0));  // demand misses keep the link first
    const size_t total = State::expert_bytes(*bank);
    constexpr int kCap = kPfBatch;
    if (!s.grow_pf(kCap * total)) { ++s.counters.pf_skipped; return; }
    std::vector<tqllm_cuda::ScatterItem> items; int staged = 0;
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
            if (staged == kCap) flush();
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
        ++s.frequency[size_t(layer) * s.experts + id];
        keep.insert(id);
    }
    flush();
    if (!bank->ready) checked(cudaEventCreateWithFlags(&bank->ready, cudaEventDisableTiming));
    checked(cudaEventRecord(bank->ready, copy)); bank->pending = true;
}
void MixedExpertCache::set_miss_stream(cudaStream_t m) {
    auto & s = *state_;
    if (!m) { if (s.own_miss) checked(cudaStreamSynchronize(s.own_miss)); s.miss_stream = nullptr; return; }
    if (!s.own_miss) {
        int least = 0, greatest = 0; checked(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        checked(cudaStreamCreateWithPriority(&s.own_miss, cudaStreamNonBlocking, greatest));
    }
    s.miss_stream = s.own_miss;
}
void MixedExpertCache::set_prefetch_strict(int from) { state_->pf_strict_from = from; }
bool MixedExpertCache::resident(int layer, int expert) const {
    const auto & s = *state_; auto it = s.banks.find(layer); if (it == s.banks.end()) return false;
    for (int id : it->second->ids) if (id == expert) return true;
    return false;
}
void MixedExpertCache::set_batch_source(BatchSource source) { state_->batch_source=std::move(source); }
void MixedExpertCache::set_async_batch_source(AsyncBatchSource source) { state_->async_batch_source=std::move(source); }
void MixedExpertCache::set_staged_misses(bool enabled) {
    auto & s = *state_; checked(cudaStreamSynchronize(s.stream));
    if (s.miss_stream) checked(cudaStreamSynchronize(s.miss_stream));
    s.staged_misses = enabled;
}
bool MixedExpertCache::split() const { return state_->miss_stream && ((state_->registered && state_->prepacked) || state_->staged_misses); }
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
void MixedExpertCache::set_layer_tiers(const std::vector<int> & floor, size_t keep_bytes) {
    auto & s = *state_; s.tier_keep = keep_bytes;
    if (int(floor.size()) != s.layers) throw std::invalid_argument("mixed cache: slot tiers need one floor per layer");
    if (s.layer_cap.empty()) s.layer_cap.assign(size_t(s.layers), s.capacity);
    s.layer_target = s.layer_cap; s.layer_floor = floor;
    for (int l = 0; l < s.layers; ++l) s.layer_floor[size_t(l)] = std::clamp(floor[size_t(l)], 1, s.layer_target[size_t(l)]);
}
int MixedExpertCache::plan_slots(int layer) {
    auto & s = *state_;
    if (!s.banks.count(layer)) { s.choose_cap(layer); if (s.planned.size() != size_t(s.layers)) s.planned.assign(size_t(s.layers), 0); s.planned[size_t(layer)] = 1; }
    return s.cap_of(layer);
}
int MixedExpertCache::regrow(size_t keep_free) {
    auto & s = *state_; if (s.layer_target.empty()) return 0;
    int n = 0;
    size_t reserved = 0;  // the full banks the released layers rebuild: their freed slots show as free memory but are spoken for
    for (int l = 0; l < s.layers; ++l) {
        const auto it = s.banks.find(l);
        if (it == s.banks.end() || it->second->cap >= s.layer_target[size_t(l)]) continue;
        size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
        const size_t grow = size_t(s.layer_target[size_t(l)] - it->second->cap) * s.slot_bytes(l);
        if (fr < keep_free + reserved + grow) break;
        reserved += size_t(s.layer_target[size_t(l)]) * s.slot_bytes(l);
        release_bank(l); ++n;  // rebuilt at its target by the next prepare (choose_cap)
    }
    return n;
}
int MixedExpertCache::experts() const { return state_->experts; }
int MixedExpertCache::layers() const { return state_->layers; }
bool MixedExpertCache::bank_ready(int layer) const { return state_->banks.count(layer) != 0; }
MixedExpertCache::BankLayout MixedExpertCache::bank_layout(int layer) const {
    auto & s = *state_;
    if (!s.banks.count(layer)) {  // evicted: the remembered layout, no device pointers
        BankLayout l; const auto & sp = s.saved_piece.at(layer);
        for (int i = 0; i < 6; ++i) { l.piece[i] = sp[size_t(i)]; l.block += sp[size_t(i)]; }
        if (!s.prepacked) return l;
        l.src0 = s.prepacked(layer, 0);
        if (!l.src0 || s.prepacked(layer, 1) != l.src0 + l.block) throw std::logic_error("mixed cache: prepacked blocks are not contiguous in the bank layout");
        return l;
    }
    const auto & b = *s.banks.at(layer);
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
Mat MixedExpertCache::bank_mat(int layer, int projection) const {
    const auto & s = *state_; const auto it = s.banks.find(layer);
    return it != s.banks.end() ? it->second->matrices.at(projection) : s.saved_mat.at(layer).at(size_t(projection));
}
bool MixedExpertCache::has_layout(int layer) const { return state_->saved_mat.count(layer) != 0; }
void MixedExpertCache::reserve_staging(int min_cap) {
    auto & s = *state_; size_t need = 0;
    for (const auto & [l, b] : s.banks) { size_t e = 0; for (int p = 0; p < 3; ++p) e += b->stride[p] + b->aux_stride[p]; need = std::max(need, e * size_t(std::max(b->cap, min_cap))); }
    if (need <= s.staging_bytes) return;
    checked(cudaStreamSynchronize(s.stream)); if (s.miss_stream) checked(cudaStreamSynchronize(s.miss_stream));
    if (s.staging) cudaFreeHost(s.staging);
    if (s.device_staging) { cudaFree(s.device_staging); s.device_staging = nullptr; }
    checked(cudaMallocHost(reinterpret_cast<void **>(&s.staging), need)); s.staging_bytes = need;
    if (s.prepacked) checked(cudaMalloc(reinterpret_cast<void **>(&s.device_staging), need));
}
std::vector<int> MixedExpertCache::evict_until_free(size_t want, int protect) {
    auto & s = *state_; std::vector<int> out;
    s.wait_misses_impl(); checked(cudaDeviceSynchronize());  // no kernel or copy may still use a victim
    for (;;) {
        if (tqllm::device_free_bytes() >= want) break;
        auto victim = s.banks.end();
        for (auto it = s.banks.begin(); it != s.banks.end(); ++it)
            if (it->first != protect && (victim == s.banks.end() || it->second->touched < victim->second->touched)) victim = it;
        if (victim == s.banks.end()) break;
        for (int id : victim->second->ids) if (id >= 0) ++s.counters.evictions;
        if (s.active == victim->first) s.active = -1;
        out.push_back(victim->first); s.banks.erase(victim);
    }
    return out;
}
std::vector<int> MixedExpertCache::banks_lru() const {
    auto & s = *state_; std::vector<std::pair<uint64_t, int>> v;
    for (const auto & [l, b] : s.banks) v.emplace_back(b->touched, l);
    std::sort(v.begin(), v.end());
    std::vector<int> out; for (const auto & p : v) out.push_back(p.second);
    return out;
}
size_t MixedExpertCache::bank_bytes(int layer) const {
    auto & s = *state_; const auto it = s.banks.find(layer); if (it == s.banks.end()) return 0;
    size_t b = 0; for (const auto & a : it->second->memory) b += a.bytes;
    return b;
}
void MixedExpertCache::release_bank(int layer) {
    auto & s = *state_; const auto it = s.banks.find(layer); if (it == s.banks.end()) return;
    s.wait_misses_impl(); checked(cudaDeviceSynchronize());  // no kernel or copy may still use the bank
    for (int id : it->second->ids) if (id >= 0) ++s.counters.evictions;
    if (s.active == layer) s.active = -1;
    s.banks.erase(it);
}
void MixedExpertCache::reset_usage() { auto & s = *state_; std::fill(s.frequency.begin(), s.frequency.end(), 0); }
void MixedExpertCache::add_device_counters(uint64_t hits, uint64_t misses, uint64_t prefetched, uint64_t bytes) {
    auto & s = *state_; s.counters.hits += hits; s.counters.misses += misses; s.counters.transfer_bytes += bytes; s.prefetch_count += prefetched;
}
// Per-layer slots fitted for this model (see expert_bank_plan.hpp): the environment wins over the file beside the packed
// experts; counts fitted for another slot count are scaled to keep the same total.
void apply_layer_slots(MixedExpertCache & cache, const std::string & experts_path, int slots, const char * env) {
    const char * v = env ? std::getenv(env) : nullptr;
    const bool from_env = v && *v;
    const std::string line = layer_slots_text(env, experts_path);
    if (line.empty()) return;
    const auto caps = parse_layer_slots(line, slots, cache.experts());
    if (caps.empty()) return;
    cache.set_layer_slots(caps);
    std::printf("DECODE per-layer expert slots %d..%d from %s\n", *std::min_element(caps.begin(), caps.end()), cache.slots(),
                (from_env ? env : experts_path + ".slots").c_str());
}
uint64_t MixedExpertCache::clock() const { return state_->clock; }
void MixedExpertCache::set_clock(uint64_t c) { state_->clock = c; }
void MixedExpertCache::clear() {
    auto & s = *state_; s.wait_misses_impl(); checked(cudaStreamSynchronize(s.stream)); s.active = -1; s.banks.clear();
    std::fill(s.frequency.begin(), s.frequency.end(), 0); s.counters = {}; s.peak = s.clock = 0;
}
}
