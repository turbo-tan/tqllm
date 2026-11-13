// np > 1 on scarce expert slots (G2, the 12 GB np 4 fault): up to 4 requests share one registered prepacked cache as the
// Flash-Next engine runs them. Each request has its own copy and miss streams for its lifetime (DecodeModel: the
// constructor switches the shared cache to split mode, the destructor switches it off and destroys its streams); every
// scheduler step decodes the active requests as one batch of 1-4 verify rows each; a layer whose distinct experts exceed
// its slots runs in waves (prepare a wave, wait for its misses, project, sync), otherwise the split path (hits, a prefetch
// of the next layer on the batch leader's copy stream, the miss wait, misses, down); maps and outputs are stream-ordered
// allocations on the legacy stream with pool trims. Requests start and end at random steps. Periodically: sync, error
// check, and one assignment of each projection byte-exact vs an independent load. TQ_STRESS_STEPS sets the step count.
#include "tq3_expert_cache.hpp"
#include "decode_rules.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>

using namespace tqllm;
static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }

static void run() {
    const int layers = 6, experts = 24, hidden = 2560, ff = 640, slots = 6, topk = 2, max_rows = 16, maxA = max_rows * topk;
    const uint32_t types[3] = {18, 18, 20};
    const long steps = std::getenv("TQ_STRESS_STEPS") ? std::atol(std::getenv("TQ_STRESS_STEPS")) : 1500;
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

    cudaMemPool_t pool; check(cudaDeviceGetDefaultMemPool(&pool, 0));
    std::mt19937 req_rng(7);
    {
        MixedExpertCache cache(layers, experts, hidden, ff, slots, size_t(256) << 20, source, nullptr);
        cache.set_prepacked([&](int l, int id) { return blocks[size_t(l) * experts + id]; }, true);
        // a request as DecodeModel holds it
        struct Request {
            MixedExpertCache & cache; cudaStream_t copy = nullptr, miss = nullptr; int rows; long left;
            Request(MixedExpertCache & c, int r, long l) : cache(c), rows(r), left(l) {
                check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
                check(cudaStreamCreateWithFlags(&miss, cudaStreamNonBlocking)); cache.set_miss_stream(miss);
            }
            ~Request() {
                cudaStreamSynchronize(copy); cudaStreamDestroy(copy);
                cache.set_miss_stream(nullptr); cudaStreamSynchronize(miss); cudaStreamDestroy(miss);
            }
        };
        std::vector<std::unique_ptr<Request>> active;
        std::vector<__half> xg(size_t(hidden) * max_rows), xd(size_t(ff) * maxA);
        for (size_t i = 0; i < xg.size(); ++i) xg[i] = __float2half_rn(.125f * std::sin(float(i * 5)));
        for (size_t i = 0; i < xd.size(); ++i) xd[i] = __float2half_rn(.125f * std::sin(float(i * 7 + 2)));
        __half * dxg = nullptr, * dxd = nullptr;
        check(cudaMalloc(&dxg, xg.size() * 2)); check(cudaMemcpy(dxg, xg.data(), xg.size() * 2, cudaMemcpyHostToDevice));
        check(cudaMalloc(&dxd, xd.size() * 2)); check(cudaMemcpy(dxd, xd.data(), xd.size() * 2, cudaMemcpyHostToDevice));
        int * maps_host = nullptr; check(cudaMallocHost(reinterpret_cast<void **>(&maps_host), size_t(3) * maxA * sizeof(int)));
        std::vector<double> weights(experts); for (int e = 0; e < experts; ++e) weights[size_t(e)] = 1.0 / (1 + e);
        std::discrete_distribution<int> pick(weights.begin(), weights.end());
        long verified = 0, wave_layers = 0, split_layers = 0, started = 0, ended = 0;
        for (long step = 0; step < steps; ++step) {
            // admission and completion, as the scheduler: up to 4 requests, each 1-4 verify rows, 20-200 steps
            for (size_t i = 0; i < active.size();) if (--active[i]->left <= 0) { active.erase(active.begin() + long(i)); ++ended; } else ++i;
            while (active.size() < 4 && (active.empty() || req_rng() % 3 == 0)) {
                active.push_back(std::make_unique<Request>(cache, 1 + int(req_rng() % 4), 20 + long(req_rng() % 180))); ++started;
            }
            int rows = 0; for (auto & r : active) rows += r->rows;
            const int A = rows * topk;
            for (int l = 0; l < layers; ++l) {
                std::vector<int> ids(static_cast<size_t>(A)); for (auto & id : ids) id = pick(rng);
                std::vector<int> u(ids); std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end()), u.end());
                __nv_bfloat16 * yg = nullptr, * yd = nullptr;
                check(cudaMallocAsync(reinterpret_cast<void **>(&yg), size_t(ff) * A * 2, nullptr));
                check(cudaMallocAsync(reinterpret_cast<void **>(&yd), size_t(hidden) * A * 2, nullptr));
                int * maps = nullptr; check(cudaMallocAsync(reinterpret_cast<void **>(&maps), size_t(3) * A * sizeof(int), nullptr));
                const bool waves = tqllm::flashnext::layer_takes_waves(ids, rows, topk, slots, true);
                if (waves) {
                    ++wave_layers;
                    for (size_t w0 = 0; w0 < u.size(); w0 += size_t(slots)) {
                        const std::vector<int> wave(u.begin() + long(w0), u.begin() + long(std::min(u.size(), w0 + size_t(slots))));
                        const auto ws = cache.prepare(l, wave);
                        if (cache.split()) cache.wait_misses();
                        check(cudaStreamSynchronize(nullptr));  // the pinned map buffer is reused
                        for (int a = 0; a < A; ++a) {
                            const auto it = std::find(wave.begin(), wave.end(), ids[size_t(a)]);
                            maps_host[a] = it == wave.end() ? -1 : ws[size_t(it - wave.begin())];
                        }
                        check(cudaMemcpyAsync(maps, maps_host, size_t(A) * sizeof(int), cudaMemcpyHostToDevice, nullptr));
                        const Mat g = cache.gate(), d = cache.down();
                        g.fmt->expert_matvec(g, dxg, yg, ff, maps, rows, topk, false, nullptr);
                        d.fmt->expert_matvec(d, dxd, yd, hidden, maps, A, 1, true, nullptr);
                        check(cudaStreamSynchronize(nullptr));  // same-bank waves wait before reusing slots
                    }
                } else {
                    ++split_layers;
                    const auto physical = cache.prepare(l, ids); const auto missed = cache.last_missed();
                    check(cudaStreamSynchronize(nullptr));
                    for (int a = 0; a < A; ++a) { maps_host[a] = physical[size_t(a)]; maps_host[A + a] = missed[size_t(a)] ? -1 : physical[size_t(a)]; maps_host[2 * A + a] = missed[size_t(a)] ? physical[size_t(a)] : -1; }
                    check(cudaMemcpyAsync(maps, maps_host, size_t(3) * A * sizeof(int), cudaMemcpyHostToDevice, nullptr));
                    const Mat g = cache.gate(), d = cache.down();
                    g.fmt->expert_matvec(g, dxg, yg, ff, maps + A, rows, topk, false, nullptr);
                    std::vector<int> want; for (int j = 0; j < 3; ++j) want.push_back(pick(rng));
                    if (cache.split()) cache.prefetch((l + 1) % layers, want, active.front()->copy);
                    cache.wait_misses();
                    g.fmt->expert_matvec(g, dxg, yg, ff, maps + 2 * A, rows, topk, false, nullptr);
                    d.fmt->expert_matvec(d, dxd, yd, hidden, maps, A, 1, true, nullptr);
                }
                if (step % 25 == 7 && l == int(step / 25) % layers) {  // sync, then one assignment of each projection vs an independent load
                    check(cudaDeviceSynchronize()); check(cudaGetLastError());
                    const int a = int(rng() % A);
                    for (int p : {0, 2}) {
                        const int prow = p == 2 ? hidden : ff, cols = p == 2 ? ff : hidden;
                        std::vector<__nv_bfloat16> got(prow); check(cudaMemcpy(got.data(), (p == 2 ? yd : yg) + size_t(a) * prow, size_t(prow) * 2, cudaMemcpyDeviceToHost));
                        std::vector<void *> memory;
                        Upload upload = [&](const void * data, size_t bytes) { void * dev = nullptr; check(cudaMalloc(&dev, bytes)); memory.push_back(dev);
                                                                              check(cudaMemcpy(dev, data, bytes, cudaMemcpyHostToDevice)); return static_cast<const void *>(dev); };
                        const auto ref = weight_format(types[p])->load(raw[size_t(l) * experts + ids[size_t(a)]][p], prow, cols, upload, nullptr);
                        __nv_bfloat16 * yr = nullptr; check(cudaMalloc(&yr, size_t(prow) * 2));
                        ref.fmt->matvec(ref, (p == 2 ? dxd + size_t(a) * cols : dxg + size_t(a / topk) * cols), yr, 1, nullptr);
                        std::vector<__nv_bfloat16> want_y(prow); check(cudaMemcpy(want_y.data(), yr, size_t(prow) * 2, cudaMemcpyDeviceToHost));
                        cudaFree(yr); for (void * ptr : memory) cudaFree(ptr);
                        if (std::memcmp(got.data(), want_y.data(), size_t(prow) * 2))
                            throw std::runtime_error("step " + std::to_string(step) + " layer " + std::to_string(l) + " projection " + std::to_string(p) +
                                                     (waves ? " (waves)" : " (split)") + " differs from an independent load");
                    }
                    ++verified;
                }
                check(cudaFreeAsync(maps, nullptr)); check(cudaFreeAsync(yg, nullptr)); check(cudaFreeAsync(yd, nullptr));
                if (l == 0 && step % 64 == 3) check(cudaMemPoolTrimTo(pool, 0));
            }
            if (step % 500 == 499) { check(cudaDeviceSynchronize()); std::printf("  step %ld ok (%ld checks, %ld wave / %ld split layers, %ld requests)\n",
                                                                                 step + 1, verified, wave_layers, split_layers, started); std::fflush(stdout); }
        }
        check(cudaDeviceSynchronize()); check(cudaGetLastError());
        active.clear();
        require(wave_layers > 0 && split_layers > 0 && ended > 0 && cache.stats().misses > 0, "stress run missed a path");
        std::printf("np > 1 scarce-slot expert stress: %ld steps x %d layers (%ld wave / %ld split), %ld requests, %ld byte-exact checks PASS\n",
                    steps, layers, wave_layers, split_layers, started, verified);
        cudaFreeHost(maps_host); cudaFree(dxg); cudaFree(dxd);
    }
    cudaFreeHost(pinned);
}
int main() { try { run(); return 0; } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; } }
