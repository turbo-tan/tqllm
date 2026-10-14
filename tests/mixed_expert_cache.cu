#include "../src/tq3_expert_cache.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <stdexcept>

using namespace tqllm;
static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
template<class T> struct Device {
    T * p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & x) { check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) { std::vector<T> x(n); check(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x; }
};
static void run() {
    const int layers = 4, experts = 5, hidden = 2560, ff = 640, slots = 3;
    const std::array<std::array<uint32_t, 3>, 4> types{{{18, 18, 20}, {21, 21, GGML_GSQ_Q2_0}, {22, 22, 20}, {23, 23, GGML_GSQ_Q2_0}}};
    std::vector<std::array<std::vector<char>, 3>> raw(layers * experts);
    std::mt19937 rng(73);
    for (int l = 0; l < layers; ++l) for (int id = 0; id < experts; ++id) for (int p = 0; p < 3; ++p) {
        uint64_t block, bytes; require(gguf_block_layout(types[l][p], block, bytes), "missing test layout");
        auto & data = raw[l * experts + id][p]; data.resize(uint64_t(hidden) * ff / block * bytes);
        for (auto & byte : data) byte = char(rng() & 255);
        for (size_t b = 0; b < data.size(); b += bytes) {
            const auto scale = __float2half_rn(.001f * (1 + id + l)); std::memcpy(data.data() + b, &scale, 2);
        }
    }
    int calls = 0; bool bad = false;
    auto source = [&](int l, int id) {
        ++calls; std::array<ExpertPayload, 3> payload;
        for (int p = 0; p < 3; ++p) {
            const auto & data = raw[l * experts + id][p]; payload[p] = {types[l][p], {data.data(), data.size() - (bad ? 1 : 0)}};
        }
        return payload;
    };
    const size_t budget = 32 * 1024 * 1024;
    MixedExpertCache cache(layers, experts, hidden, ff, slots, budget, source, nullptr);
    MixedExpertCache * selected = &cache;
    auto compare = [&](int l, const std::vector<int> & ids, bool per = false) {
        auto & current = *selected;
        const auto physical = current.prepare(l, ids); current.wait_misses(); Device<int> di(ids.size()); di.put(physical);
        const int topk = 2, n = ids.size() / topk;
        const std::array<Mat, 3> matrices{current.gate(), current.up(), current.down()};
        for (int p = 0; p < 3; ++p) {
            const int cols = p == 2 ? ff : hidden, rows = p == 2 ? hidden : ff;
            std::vector<__half> x(size_t(cols) * (per ? ids.size() : n));
            for (size_t i = 0; i < x.size(); ++i) x[i] = __float2half_rn(.125f * std::sin(float(i * 5 + p)));
            Device<__half> dx(x.size()); dx.put(x);
            Device<__nv_bfloat16> dy(size_t(rows) * ids.size()), reference(rows);
            const auto & m = matrices[p]; m.fmt->expert_matvec(m, dx.p, dy.p, rows, di.p, n, topk, per, nullptr);
            const auto actual = dy.get(size_t(rows) * ids.size());
            for (size_t a = 0; a < ids.size(); ++a) {
                std::vector<void *> memory;
                Upload upload = [&](const void * data, size_t bytes) {
                    void * device = nullptr; check(cudaMalloc(&device, bytes)); memory.push_back(device);
                    check(cudaMemcpy(device, data, bytes, cudaMemcpyHostToDevice)); return static_cast<const void *>(device);
                };
                const auto expected = weight_format(types[l][p])->load(raw[l * experts + ids[a]][p], rows, cols, upload, nullptr);
                expected.fmt->matvec(expected, dx.p + (per ? a : a / topk) * cols, reference.p, 1, nullptr);
                const auto got = reference.get(rows);
                const bool equal = std::memcmp(actual.data() + a * rows, got.data(), rows * sizeof(got[0])) == 0;
                for (void * ptr : memory) cudaFree(ptr);
                require(equal, "mixed cached grouped projection differs from independent native load");
            }
        }
        require(current.stats().device_bytes <= budget && current.peak_device_bytes() <= budget, "mixed cache resident or staging budget exceeded");
    };
    for (int l = 0; l < layers; ++l) {
        compare(l, {0, 1, 1, 2}); const auto before = cache.stats(); const int fetches = calls;
        require(before.transfer_bytes >= before.device_bytes, "mixed transfers omitted bank initialization or formatted bytes");
        compare(l, {2, 0, 1, 2}, true);
        require(cache.stats().misses == before.misses && cache.stats().transfer_bytes == before.transfer_bytes && calls == fetches, "warm mixed cache fetched or copied experts");
    }
    const auto stable = cache.stats(); bool rejected = false; bad = true;
    try { cache.prepare(3, {3, 4}); } catch (const std::invalid_argument &) { rejected = true; }
    bad = false; require(rejected && cache.stats().misses == stable.misses, "bad mixed source mutated cache");
    compare(3, {0, 4}); require(cache.stats().evictions > 0, "mixed replacement did not evict");
    rejected = false; try { cache.prepare(3, {0, 1, 2, 3}); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "oversized mixed active set accepted");
    cache.clear(); require(cache.stats().device_bytes == 0 && cache.stats().misses == 0, "mixed clear retained bytes or counters");
    rejected = false; try { cache.gate(); } catch (const std::logic_error &) { rejected = true; }
    require(rejected, "mixed clear retained an active view");
    MixedExpertCache tiny(layers, experts, hidden, ff, slots, 1, source, nullptr);
    rejected = false; try { tiny.prepare(0, {0, 1}); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && tiny.stats().device_bytes == 0, "too-small mixed budget leaked allocations");
    MixedExpertCache partial(layers, experts, hidden, ff, slots, 3 * 1024 * 1024, source, nullptr);
    rejected = false; try { partial.prepare(0, {0, 1}); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && partial.stats().device_bytes == 0 && partial.peak_device_bytes() > 0,
            "partial mixed bank allocation did not roll back");
    const size_t tight_budget = 8 * 1024 * 1024;
    MixedExpertCache tight(layers, experts, hidden, ff, slots, tight_budget, source, nullptr); selected = &tight;
    for (int l : {0, 1, 2, 3, 0}) {
        compare(l, {0, 1, 1, 2}, true);
        require(tight.peak_device_bytes() <= tight_budget, "layer-bank eviction ignored staging budget");
    }
    require(tight.stats().evictions > 0 && tight.stats().misses == 15, "tight cache did not evict and reload native banks");
    std::vector<std::vector<char>> packed(raw.size());
    for (int l = 0; l < layers; ++l) for (int id = 0; id < experts; ++id) {
        auto & bytes = packed[l * experts + id];
        std::vector<void *> memory;
        Upload upload = [&](const void * data, size_t size) {
            const auto * begin = static_cast<const char *>(data);
            bytes.insert(bytes.end(), begin, begin + size);
            void * ptr = nullptr; check(cudaMalloc(&ptr, size)); memory.push_back(ptr);
            check(cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice));
            return static_cast<const void *>(ptr);
        };
        for (int p = 0; p < 3; ++p)
            weight_format(types[l][p])->load(raw[l * experts + id][p], p == 2 ? hidden : ff,
                                           p == 2 ? ff : hidden, upload, nullptr);
        check(cudaDeviceSynchronize());
        for (void * ptr : memory) check(cudaFree(ptr));
    }
    size_t packed_bytes = 0;
    for (const auto & bytes : packed) packed_bytes += bytes.size();
    char * pinned = nullptr; check(cudaMallocHost(reinterpret_cast<void **>(&pinned), packed_bytes));
    std::vector<const char *> blocks(packed.size());
    size_t at = 0;
    for (size_t i = 0; i < packed.size(); ++i) {
        blocks[i] = pinned + at; std::memcpy(pinned + at, packed[i].data(), packed[i].size()); at += packed[i].size();
    }
    cudaStream_t misses, copies;
    check(cudaStreamCreateWithFlags(&misses, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&copies, cudaStreamNonBlocking));
    {
        {
            MixedExpertCache current(layers, experts, hidden, ff, slots, budget, source, nullptr);
            current.set_prepacked([&](int l, int id) { return blocks[l * experts + id]; }, true);
            current.set_miss_stream(misses); selected = &current;
            for (int l = 0; l < layers; ++l) {
                compare(l, {0, 1, 1, 2});
                const auto before = current.stats();
                compare(l, {2, 0, 1, 2}, true);
                require(current.stats().transfer_bytes == before.transfer_bytes, "registered warm hit transferred bytes");
                current.prefetch(l, {3, 4, 3}, copies);
                compare(l, {4, 3, 3, 4}, true);
                compare(l, {0, 4});
            }
            require(current.prefetch_used() > 0, "registered prefetch was never consumed");
            current.clear(); compare(0, {0, 1});
            current.set_miss_stream(nullptr);
        }
        check(cudaStreamSynchronize(misses)); check(cudaStreamSynchronize(copies));
        std::printf("registered prepacked misses %s: byte-exact projections / mixed formats / duplicates / prefetch / eviction / reset PASS\n",
                    "mapped scatter");
    }
    check(cudaStreamDestroy(copies)); check(cudaStreamDestroy(misses)); check(cudaFreeHost(pinned));
    std::printf("cache owned-device peak=%zu tight-budget=%zu bytes (includes loader staging)\n", tight.peak_device_bytes(), tight_budget);
    {  // per-layer slots: each bank holds its own count, projections stay byte-exact, the per-layer limit is enforced
        MixedExpertCache layered(layers, experts, hidden, ff, slots, budget, source, nullptr);
        layered.set_layer_slots({2, 5, 3, 4}); selected = &layered;
        require(layered.slots() == 5 && layered.slots(0) == 2 && layered.slots(3) == 4, "per-layer slot counts not reported");
        compare(1, {0, 1, 2, 3, 4, 4}, true); compare(0, {3, 4}); compare(2, {0, 2, 4, 0}, true); compare(3, {1, 2, 3, 4});
        compare(0, {0, 1}); compare(1, {4, 3, 2, 1, 0, 0}, true);
        bool refused = false;
        try { layered.prepare(0, {0, 1, 2}); } catch (const std::invalid_argument &) { refused = true; }
        require(refused, "a layer took more experts than its own slots");
        refused = false;
        try { layered.set_layer_slots({3, 3, 3, 3}); } catch (const std::logic_error &) { refused = true; }
        require(refused, "per-layer slots changed after the first bank");
        selected = &cache;
        std::puts("per-layer slots 2/5/3/4: byte-exact projections / own limit / set-once PASS");
    }
    std::puts("mixed IQ3_XXS/IQ3_S/IQ2_S/IQ4_XS + IQ4_NL/Q2_0 banks: cold/warm/grouped/duplicates/per-expert/eviction/budget/clear PASS");
}
int main() { try { run(); return 0; } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; } }
