// Split expert path under decode-like load (F25): every step routes random experts per layer through a registered prepacked
// cache with a miss stream, runs the hit projections before the miss wait and the miss projections after it, prefetches the
// next layer on a copy stream, and allocates its maps / outputs stream-ordered on the legacy stream (pool trimmed now and
// then) with no host sync between steps. Periodically: sync, error check, and one assignment byte-exact vs an independent load.
// TQ_STRESS_STEPS (default 3000) sets the step count.
#include "tq3_expert_cache.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <set>
#include <stdexcept>

using namespace tqllm;
static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }

static void run() {
    const int layers = 6, experts = 16, hidden = 2560, ff = 640, slots = 6, topk = 2, n = 4, A = n * topk;
    const uint32_t types[3] = {18, 18, 20};
    const long steps = std::getenv("TQ_STRESS_STEPS") ? std::atol(std::getenv("TQ_STRESS_STEPS")) : 3000;
    std::vector<std::array<std::vector<char>, 3>> raw(size_t(layers) * experts);
    std::mt19937 rng(91);
    for (int l = 0; l < layers; ++l) for (int id = 0; id < experts; ++id) for (int p = 0; p < 3; ++p) {
        uint64_t block, bytes; require(gguf_block_layout(types[p], block, bytes), "missing test layout");
        auto & data = raw[size_t(l) * experts + id][p]; data.resize(uint64_t(hidden) * ff / block * bytes);
        for (auto & byte : data) byte = char(rng() & 255);
        for (size_t b = 0; b < data.size(); b += bytes) {
            const auto scale = __float2half_rn(.001f * (1 + id + l)); std::memcpy(data.data() + b, &scale, 2);
        }
    }
    auto source = [&](int l, int id) {
        std::array<ExpertPayload, 3> payload;
        for (int p = 0; p < 3; ++p) { const auto & d = raw[size_t(l) * experts + id][p]; payload[p] = {types[p], {d.data(), d.size()}}; }
        return payload;
    };
    // prepacked blocks: the bytes each format's loader uploads, in one pinned buffer (as the engine's registered pack file)
    std::vector<std::vector<char>> packed(raw.size());
    for (int l = 0; l < layers; ++l) for (int id = 0; id < experts; ++id) {
        auto & bytes = packed[size_t(l) * experts + id]; std::vector<void *> memory;
        Upload upload = [&](const void * data, size_t size) {
            const auto * begin = static_cast<const char *>(data); bytes.insert(bytes.end(), begin, begin + size);
            void * ptr = nullptr; check(cudaMalloc(&ptr, size)); memory.push_back(ptr);
            check(cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice)); return static_cast<const void *>(ptr);
        };
        for (int p = 0; p < 3; ++p) weight_format(types[p])->load(raw[size_t(l) * experts + id][p], p == 2 ? hidden : ff, p == 2 ? ff : hidden, upload, nullptr);
        check(cudaDeviceSynchronize()); for (void * ptr : memory) check(cudaFree(ptr));
    }
    size_t packed_bytes = 0; for (const auto & b : packed) packed_bytes += b.size();
    char * pinned = nullptr; check(cudaMallocHost(reinterpret_cast<void **>(&pinned), packed_bytes));
    std::vector<const char *> blocks(packed.size());
    for (size_t i = 0, at = 0; i < packed.size(); at += packed[i].size(), ++i) { blocks[i] = pinned + at; std::memcpy(pinned + at, packed[i].data(), packed[i].size()); }

    cudaStream_t misses, copies;
    check(cudaStreamCreateWithFlags(&misses, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&copies, cudaStreamNonBlocking));
    cudaMemPool_t pool; check(cudaDeviceGetDefaultMemPool(&pool, 0));
    {
        MixedExpertCache cache(layers, experts, hidden, ff, slots, size_t(256) << 20, source, nullptr);
        cache.set_prepacked([&](int l, int id) { return blocks[size_t(l) * experts + id]; }, true);
        cache.set_miss_stream(misses);
        require(cache.split(), "stress cache did not take the split path");
        // one persistent input per projection kind (gate / up read hidden columns, down reads ff columns)
        std::vector<__half> xg(size_t(hidden) * n), xd(size_t(ff) * A);
        for (size_t i = 0; i < xg.size(); ++i) xg[i] = __float2half_rn(.125f * std::sin(float(i * 5)));
        for (size_t i = 0; i < xd.size(); ++i) xd[i] = __float2half_rn(.125f * std::sin(float(i * 7 + 2)));
        __half * dxg = nullptr, * dxd = nullptr;
        check(cudaMalloc(&dxg, xg.size() * 2)); check(cudaMemcpy(dxg, xg.data(), xg.size() * 2, cudaMemcpyHostToDevice));
        check(cudaMalloc(&dxd, xd.size() * 2)); check(cudaMemcpy(dxd, xd.data(), xd.size() * 2, cudaMemcpyHostToDevice));
        int * maps_host = nullptr; check(cudaMallocHost(reinterpret_cast<void **>(&maps_host), size_t(3) * A * sizeof(int)));
        std::vector<double> weights(experts); for (int e = 0; e < experts; ++e) weights[size_t(e)] = 1.0 / (1 + e);
        std::discrete_distribution<int> pick(weights.begin(), weights.end());
        long verified = 0;
        for (long step = 0; step < steps; ++step) {
            for (int l = 0; l < layers; ++l) {
                std::vector<int> ids(A); std::set<int> uniq;
                do { uniq.clear(); for (auto & id : ids) { id = pick(rng); uniq.insert(id); } } while (int(uniq.size()) > slots);
                const auto physical = cache.prepare(l, ids); const auto missed = cache.last_missed();
                // maps as the engine sends them: all / hit / miss slots, copied from pinned memory on the legacy stream
                check(cudaStreamSynchronize(nullptr));  // the pinned map buffer is reused per layer
                for (int a = 0; a < A; ++a) { maps_host[a] = physical[size_t(a)]; maps_host[A + a] = missed[size_t(a)] ? -1 : physical[size_t(a)]; maps_host[2 * A + a] = missed[size_t(a)] ? physical[size_t(a)] : -1; }
                int * maps = nullptr; check(cudaMallocAsync(reinterpret_cast<void **>(&maps), size_t(3) * A * sizeof(int), nullptr));
                check(cudaMemcpyAsync(maps, maps_host, size_t(3) * A * sizeof(int), cudaMemcpyHostToDevice, nullptr));
                __nv_bfloat16 * yg = nullptr, * yd = nullptr;
                check(cudaMallocAsync(reinterpret_cast<void **>(&yg), size_t(ff) * A * 2, nullptr));
                check(cudaMallocAsync(reinterpret_cast<void **>(&yd), size_t(hidden) * A * 2, nullptr));
                const Mat g = cache.gate(), d = cache.down();
                g.fmt->expert_matvec(g, dxg, yg, ff, maps + A, n, topk, false, nullptr);  // hits first
                std::vector<int> want; for (int j = 0; j < 3; ++j) want.push_back(pick(rng));
                cache.prefetch((l + 1) % layers, want, copies);  // the next layer, while the hits run
                cache.wait_misses();
                g.fmt->expert_matvec(g, dxg, yg, ff, maps + 2 * A, n, topk, false, nullptr);  // then the misses
                d.fmt->expert_matvec(d, dxd, yd, hidden, maps, A, 1, true, nullptr);
                if (step % 50 == 7 && l == layers - 1) {  // sync, then one assignment of each projection vs an independent load
                    check(cudaDeviceSynchronize()); check(cudaGetLastError());
                    const int a = int(rng() % A);
                    for (int p : {0, 2}) {
                        const int rows = p == 2 ? hidden : ff, cols = p == 2 ? ff : hidden;
                        std::vector<__nv_bfloat16> got(rows); check(cudaMemcpy(got.data(), (p == 2 ? yd : yg) + size_t(a) * rows, size_t(rows) * 2, cudaMemcpyDeviceToHost));
                        std::vector<void *> memory;
                        Upload upload = [&](const void * data, size_t bytes) { void * dev = nullptr; check(cudaMalloc(&dev, bytes)); memory.push_back(dev);
                                                                              check(cudaMemcpy(dev, data, bytes, cudaMemcpyHostToDevice)); return static_cast<const void *>(dev); };
                        const auto ref = weight_format(types[p])->load(raw[size_t(l) * experts + ids[size_t(a)]][p], rows, cols, upload, nullptr);
                        __nv_bfloat16 * yr = nullptr; check(cudaMalloc(&yr, size_t(rows) * 2));
                        ref.fmt->matvec(ref, (p == 2 ? dxd + size_t(a) * cols : dxg + size_t(a / topk) * cols), yr, 1, nullptr);
                        std::vector<__nv_bfloat16> want_y(rows); check(cudaMemcpy(want_y.data(), yr, size_t(rows) * 2, cudaMemcpyDeviceToHost));
                        cudaFree(yr); for (void * ptr : memory) cudaFree(ptr);
                        if (std::memcmp(got.data(), want_y.data(), size_t(rows) * 2))
                            throw std::runtime_error("step " + std::to_string(step) + " layer " + std::to_string(l) + " projection " + std::to_string(p) + " differs from an independent load");
                    }
                    ++verified;
                }
                check(cudaFreeAsync(maps, nullptr)); check(cudaFreeAsync(yg, nullptr)); check(cudaFreeAsync(yd, nullptr));
                if (l == 0 && step % 64 == 3) check(cudaMemPoolTrimTo(pool, 0));  // as refill_borrowed: no sync before the trim
            }
            if (step % 1000 == 999) { check(cudaDeviceSynchronize()); std::printf("  step %ld ok (%ld checks, hits %llu misses %llu)\n", step + 1, verified,
                                                                                   (unsigned long long) cache.stats().hits, (unsigned long long) cache.stats().misses); std::fflush(stdout); }
        }
        check(cudaDeviceSynchronize()); check(cudaGetLastError());
        require(cache.stats().misses > 0 && cache.prefetch_used() > 0, "stress run had no misses or no consumed prefetch");
        std::printf("split expert path stress: %ld steps x %d layers, %ld byte-exact checks, %llu misses PASS\n", steps, layers, verified,
                    (unsigned long long) cache.stats().misses);
        cache.set_miss_stream(nullptr);
        cudaFreeHost(maps_host); cudaFree(dxg); cudaFree(dxd);
    }
    cudaStreamDestroy(misses); cudaStreamDestroy(copies); cudaFreeHost(pinned);
}
int main() { try { run(); return 0; } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; } }
