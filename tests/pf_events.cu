// TQFN_PF_EVENTS: a demand miss and the compute stream must not wait for a prefetch of an expert the call does not read,
// and every slot must hold its expert's exact packed bytes once the copies land. The prefetch copy stream is held by a
// host function; with --legacy (the bank-wide wait) the demand call cannot finish while it is held, so that run fails.
#include "../src/tq3_expert_cache.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tqllm;
static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static std::atomic<bool> g_release{false};
static void CUDART_CB hold(void *) {
    const auto t0 = std::chrono::steady_clock::now();
    while (!g_release.load() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

int main(int argc, char ** argv) {
    const bool legacy = argc > 1 && std::string(argv[1]) == "--legacy";
    if (legacy) unsetenv("TQFN_PF_EVENTS"); else setenv("TQFN_PF_EVENTS", "1", 1);
    try {
        const int layers = 2, experts = 6, hidden = 2560, ff = 640, slots = 3;
        const std::array<uint32_t, 3> types{18, 18, 20};
        std::vector<std::array<std::vector<char>, 3>> raw(layers * experts);
        std::mt19937 rng(11);
        for (auto & e : raw) for (int p = 0; p < 3; ++p) {
            uint64_t block, bytes; require(gguf_block_layout(types[p], block, bytes), "missing test layout");
            e[p].resize(uint64_t(hidden) * ff / block * bytes);
            for (auto & b : e[p]) b = char(rng() & 255);
            for (size_t b = 0; b < e[p].size(); b += bytes) { const auto sc = __float2half_rn(.002f); std::memcpy(e[p].data() + b, &sc, 2); }
        }
        auto source = [&](int l, int id) {
            std::array<ExpertPayload, 3> pl;
            for (int p = 0; p < 3; ++p) pl[p] = {types[p], {raw[l * experts + id][p].data(), raw[l * experts + id][p].size()}};
            return pl;
        };
        // packed blocks in the bank row layout, as the engine's experts file holds them
        std::vector<std::vector<char>> packed(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            std::vector<void *> memory;
            Upload up = [&](const void * data, size_t size) {
                const auto * b = static_cast<const char *>(data); packed[i].insert(packed[i].end(), b, b + size);
                void * d = nullptr; check(cudaMalloc(&d, size)); memory.push_back(d); check(cudaMemcpy(d, data, size, cudaMemcpyHostToDevice));
                return static_cast<const void *>(d);
            };
            for (int p = 0; p < 3; ++p) weight_format(types[p])->load(raw[i][p], p == 2 ? hidden : ff, p == 2 ? ff : hidden, up, nullptr);
            check(cudaDeviceSynchronize()); for (void * d : memory) check(cudaFree(d));
        }
        size_t total = 0; for (const auto & b : packed) total += b.size();
        char * pinned = nullptr; check(cudaMallocHost(reinterpret_cast<void **>(&pinned), total));
        std::vector<const char *> blocks(packed.size()); size_t at = 0;
        for (size_t i = 0; i < packed.size(); ++i) { blocks[i] = pinned + at; std::memcpy(pinned + at, packed[i].data(), packed[i].size()); at += packed[i].size(); }
        cudaStream_t compute, misses, copies;
        check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        check(cudaStreamCreateWithFlags(&misses, cudaStreamNonBlocking));
        check(cudaStreamCreateWithFlags(&copies, cudaStreamNonBlocking));
        MixedExpertCache cache(layers, experts, hidden, ff, slots, 64u << 20, source, compute);
        cache.set_prepacked([&](int l, int id) { return blocks[l * experts + id]; }, true);
        cache.set_miss_stream(misses);
        // every slot the call returned holds that expert's packed bytes (data and aux of gate, up, down)
        auto verify = [&](int l, const std::vector<int> & ids, const std::vector<int> & phys) {
            check(cudaDeviceSynchronize());
            const auto lay = cache.bank_layout(l);
            for (size_t a = 0; a < ids.size(); ++a) {
                std::vector<char> got; got.reserve(lay.block);
                for (int k = 0; k < 6; ++k) {
                    if (!lay.piece[k]) continue;
                    std::vector<char> part(lay.piece[k]);
                    check(cudaMemcpy(part.data(), lay.dst[k] + size_t(phys[a]) * lay.piece[k], lay.piece[k], cudaMemcpyDeviceToHost));
                    got.insert(got.end(), part.begin(), part.end());
                }
                const auto & want = packed[l * experts + ids[a]];
                require(got.size() == want.size() && std::memcmp(got.data(), want.data(), want.size()) == 0, "slot bytes differ from the expert's packed block");
            }
        };
        for (int l = 0; l < layers; ++l) {
            g_release = false;
            auto phys = cache.prepare(l, {0, 1}); cache.wait_misses(); verify(l, {0, 1}, phys);  // bank: experts 0, 1 and one free slot
            check(cudaLaunchHostFunc(copies, hold, nullptr));                                   // the prefetch DMA cannot start
            cache.prefetch(l, {2}, copies);                                                     // claims the free slot, in flight
            phys = cache.prepare(l, {3, 0}); cache.wait_misses();                               // demand miss 3, hit 0: neither reads 2
            cudaEvent_t done; check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming)); check(cudaEventRecord(done, compute));
            bool finished = false;
            for (int i = 0; i < 3000 && !finished; ++i) {
                const auto rc = cudaEventQuery(done);
                if (rc == cudaSuccess) finished = true; else if (rc != cudaErrorNotReady) check(rc);
                else std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            g_release = true; check(cudaDeviceSynchronize()); cudaEventDestroy(done);
            require(finished, "the demand call waited for a prefetch of an expert it does not read");
            verify(l, {3, 0}, phys);
            phys = cache.prepare(l, {2, 3}); cache.wait_misses();                               // the prefetched expert, now landed
            verify(l, {2, 3}, phys);
            require(cache.prefetch_used() > 0, "the prefetched expert was copied again instead of used");
            // a prefetched expert that the next call does read is waited for: hold, prefetch 4, prepare {4}, release later
            g_release = false;
            check(cudaLaunchHostFunc(copies, hold, nullptr));
            cache.prefetch(l, {4}, copies);
            phys = cache.prepare(l, {4, 3}); cache.wait_misses();
            std::thread later([] { std::this_thread::sleep_for(std::chrono::milliseconds(200)); g_release = true; });
            verify(l, {4, 3}, phys); later.join();
        }
        // two prefetch calls before one prepare: an older call's expert (event 2, landed) and a newer call's held expert
        // (event 0) are both read, so the compute stream must wait for the newer one, not the highest index
        if (!legacy) {
            MixedExpertCache wide(layers, experts, hidden, ff, 6, 64u << 20, source, compute);
            wide.set_prepacked([&](int l, int id) { return blocks[l * experts + id]; }, true);
            wide.set_miss_stream(misses);
            auto phys = wide.prepare(0, {0, 1}); wide.wait_misses();
            wide.prefetch(0, {2, 3, 4}, copies); check(cudaStreamSynchronize(copies));  // events 0..2, all landed
            g_release = false;
            check(cudaLaunchHostFunc(copies, hold, nullptr));
            wide.prefetch(0, {5}, copies);                                                 // event 0 again, held
            phys = wide.prepare(0, {4, 5}); wide.wait_misses();
            cudaEvent_t done; check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming)); check(cudaEventRecord(done, compute));
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            const bool early = cudaEventQuery(done) == cudaSuccess;
            g_release = true; check(cudaDeviceSynchronize()); cudaEventDestroy(done);
            require(!early, "the compute stream ran before a read expert's prefetch landed (event of an older call)");
            const auto lay = wide.bank_layout(0);
            for (size_t a = 0; a < 2; ++a) {
                std::vector<char> got;
                for (int k = 0; k < 6; ++k) if (lay.piece[k]) {
                    std::vector<char> part(lay.piece[k]);
                    check(cudaMemcpy(part.data(), lay.dst[k] + size_t(phys[a]) * lay.piece[k], lay.piece[k], cudaMemcpyDeviceToHost));
                    got.insert(got.end(), part.begin(), part.end());
                }
                const auto & want = packed[size_t(a == 0 ? 4 : 5)];
                require(got.size() == want.size() && std::memcmp(got.data(), want.data(), want.size()) == 0, "slot bytes differ after two prefetch calls");
            }
        }
        check(cudaFreeHost(pinned));
        std::printf("%s: demand miss beside a held prefetch finished, read slots byte-exact, held prefetch waited for when read PASS\n",
                    legacy ? "legacy bank-wide wait" : "TQFN_PF_EVENTS");
        return 0;
    } catch (const std::exception & e) {
        g_release = true; cudaDeviceSynchronize();
        std::fprintf(stderr, "FAIL%s: %s\n", legacy ? " (legacy, expected)" : "", e.what());
        return 1;
    }
}
