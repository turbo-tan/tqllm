// np > 1: a sequence that ends must not unbind the shared expert cache's miss stream while another sequence holds it
// (G2: ~DecodeModel set it to null, so the remaining sequences' misses fell back to zero-copy kernel reads of pinned host
// memory, the F25 fault class on PCIe GPUs). Two bound streams, the first released: misses stay on the copy engine; with
// none bound, misses on a PCIe GPU still take the copy engine.
#include "tq3_expert_cache.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

using namespace tqllm;
static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }

static void run() {
    const int layers = 2, experts = 8, hidden = 2560, ff = 640, slots = 4;
    const uint32_t types[3] = {18, 18, 20};
    std::vector<std::array<std::vector<char>, 3>> raw(size_t(layers) * experts);
    std::mt19937 rng(7);
    for (auto & e : raw) for (int p = 0; p < 3; ++p) {
        uint64_t block, bytes; require(gguf_block_layout(types[p], block, bytes), "missing test layout");
        e[p].resize(uint64_t(hidden) * ff / block * bytes);
        for (auto & byte : e[p]) byte = char(rng() & 255);
        for (size_t b = 0; b < e[p].size(); b += bytes) { const auto scale = __float2half_rn(.001f); std::memcpy(e[p].data() + b, &scale, 2); }
    }
    auto source = [&](int l, int id) {
        std::array<ExpertPayload, 3> payload;
        for (int p = 0; p < 3; ++p) { const auto & d = raw[size_t(l) * experts + id][p]; payload[p] = {types[p], {d.data(), d.size()}}; }
        return payload;
    };
    std::vector<std::vector<char>> packed(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        std::vector<void *> memory;
        Upload upload = [&](const void * data, size_t size) {
            const auto * begin = static_cast<const char *>(data); packed[i].insert(packed[i].end(), begin, begin + size);
            void * ptr = nullptr; check(cudaMalloc(&ptr, size)); memory.push_back(ptr);
            check(cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice)); return static_cast<const void *>(ptr);
        };
        for (int p = 0; p < 3; ++p) weight_format(types[p])->load(raw[i][p], p == 2 ? hidden : ff, p == 2 ? ff : hidden, upload, nullptr);
        check(cudaDeviceSynchronize()); for (void * ptr : memory) check(cudaFree(ptr));
    }
    size_t total = 0; for (const auto & b : packed) total += b.size();
    char * pinned = nullptr; check(cudaMallocHost(reinterpret_cast<void **>(&pinned), total));
    std::vector<const char *> blocks(packed.size());
    for (size_t i = 0, at = 0; i < packed.size(); at += packed[i].size(), ++i) { blocks[i] = pinned + at; std::memcpy(pinned + at, packed[i].data(), packed[i].size()); }
    cudaStream_t a, b;
    check(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking)); check(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
    {
        MixedExpertCache cache(layers, experts, hidden, ff, slots, size_t(64) << 20, source, nullptr);
        cache.set_prepacked([&](int l, int id) { return blocks[size_t(l) * experts + id]; }, true);
        cache.set_miss_stream(a);  // sequence A binds
        cache.set_miss_stream(b);  // sequence B binds (the engine's bind() before B's step)
        cache.release_miss_stream(a);  // A's request ends while B is bound
        require(cache.split(), "the cache lost B's miss stream when A ended");
        for (int step = 0; step < 8; ++step) for (int l = 0; l < layers; ++l) {
            std::vector<int> ids; for (int k = 0; k < slots; ++k) ids.push_back((step * slots + k) % experts);
            cache.prepare(l, ids);
            if (cache.split()) cache.wait_misses();
        }
        check(cudaDeviceSynchronize()); check(cudaGetLastError());
        const auto st = cache.stats();
        std::printf("misses %llu, copy-engine pieces %llu, zero-copy pieces %llu\n", (unsigned long long) st.misses,
                    (unsigned long long) st.miss_copies, (unsigned long long) st.zero_copy_items);
        require(st.misses > 0 && st.miss_copies > 0, "no copy-engine misses");
        require(st.zero_copy_items == 0, "misses fell back to zero-copy kernel reads of host memory");
        cache.release_miss_stream(b);  // the owner unbinds
        require(!cache.split(), "the bound sequence's release left the miss stream bound");
        // no sequence bound (between requests, warm-up): PCIe GPUs still copy misses by the copy engine, never zero-copy
        int coherent = 0; check(cudaDeviceGetAttribute(&coherent, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
        const auto copies = cache.stats().miss_copies;
        for (int step = 0; step < 8; ++step) for (int l = 0; l < layers; ++l) {
            std::vector<int> ids; for (int k = 0; k < slots; ++k) ids.push_back((step * slots + k + 3) % experts);
            cache.prepare(l, ids);
        }
        check(cudaDeviceSynchronize()); check(cudaGetLastError());
        const auto un = cache.stats();
        std::printf("unbound: copy-engine pieces %llu, zero-copy pieces %llu (coherent host memory %d)\n",
                    (unsigned long long) (un.miss_copies - copies), (unsigned long long) un.zero_copy_items, coherent);
        if (!coherent) require(un.zero_copy_items == 0 && un.miss_copies > copies, "unbound misses read host memory by kernel on a PCIe GPU");
    }
    cudaStreamDestroy(a); cudaStreamDestroy(b); cudaFreeHost(pinned);
    std::printf("expert cache unbind (np > 1): PASS\n");
}
int main() { try { run(); return 0; } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; } }
