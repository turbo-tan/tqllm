#include "tq3_expert_cache.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

using namespace tqllm;
static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
template<class T> struct Device {
    T * p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & v, cudaStream_t stream) {
        check(cudaMemcpyAsync(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    std::vector<T> get(size_t n, cudaStream_t stream) {
        check(cudaStreamSynchronize(stream)); std::vector<T> v(n);
        check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
    }
};
static void run() {
    const int layers = 3, experts = 6, hidden = 640, ff = 256, slots = 3;
    const size_t slice = size_t(hidden) * ff / 2, budget = slice * 3 * slots;
    std::vector<std::array<std::vector<char>, 3>> raw(layers * experts);
    std::mt19937 rng(51);
    for (auto & entry : raw) for (auto & projection : entry) {
        projection.resize(slice);
        for (size_t block = 0; block < slice / 16; ++block)
            for (int byte = 0; byte < 16; ++byte) projection[block * 16 + byte] = char(byte < 4 ? 90 + rng() % 80 : rng() % 256);
    }
    bool bad_source = false;
    int source_calls = 0;
    auto source = [&](int layer, int expert) {
        ++source_calls; std::array<GgufBytes, 3> result;
        for (int projection = 0; projection < 3; ++projection) {
            const auto & bytes = raw.at(layer * experts + expert)[projection];
            result[projection] = {bytes.data(), bytes.size() - (bad_source ? 1 : 0)};
        }
        return result;
    };
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    struct Stream { cudaStream_t s; ~Stream() { cudaStreamDestroy(s); } } stream_owner{stream};
    Tq3ExpertCache cache(layers, experts, hidden, ff, slots, budget, source, stream);
    require(cache.stats().device_bytes == budget, "device slot accounting differs");
    auto compare = [&](int layer, const std::vector<int> & ids, bool per_expert = false) {
        const int k = 2, n = int(ids.size()) / k;
        const auto physical = cache.prepare(layer, ids);
        require(ids.size() % k == 0 && n > 0, "bad test assignment count");
        Device<int> dids(ids.size()); dids.put(physical, stream);
        const std::array<Mat, 3> matrices{cache.gate(), cache.up(), cache.down()};
        for (int projection = 0; projection < 3; ++projection) {
            const int cols = projection == 2 ? ff : hidden, rows = projection == 2 ? hidden : ff;
            std::vector<__half> x(size_t(cols) * (per_expert ? ids.size() : n));
            for (size_t i = 0; i < x.size(); ++i) x[i] = __float2half(.25f * std::sin(float(i * 13 + projection)));
            Device<__half> dx(x.size()); dx.put(x, stream);
            Device<__nv_bfloat16> dy(size_t(rows) * ids.size()), reference(rows);
            const auto & m = matrices[projection];
            m.fmt->expert_matvec(m, dx.p, dy.p, rows, dids.p, n, k, per_expert, stream);
            const auto got = dy.get(size_t(rows) * ids.size(), stream);
            for (size_t assignment = 0; assignment < ids.size(); ++assignment) {
                std::vector<void *> memory;
                Upload upload = [&](const void * data, size_t bytes) {
                    void * device = nullptr; check(cudaMalloc(&device, bytes)); memory.push_back(device);
                    check(cudaMemcpyAsync(device, data, bytes, cudaMemcpyHostToDevice, stream)); return static_cast<const void *>(device);
                };
                const auto expected = weight_format(GGML_TQ3_4S)->load(raw[layer * experts + ids[assignment]][projection], rows, cols, upload, stream);
                expected.fmt->matvec(expected, dx.p + (per_expert ? assignment : assignment / k) * cols, reference.p, 1, stream);
                const auto values = reference.get(rows, stream);
                const bool equal = std::memcmp(values.data(), got.data() + assignment * rows, rows * sizeof(__nv_bfloat16)) == 0;
                for (auto device : memory) cudaFree(device);
                require(equal, "cached projection differs from independently loaded expert");
            }
        }
    };
    compare(0, {0, 1, 1, 2, 0, 2});
    require(cache.stats().misses == 3 && cache.stats().transfer_bytes == slice * 9, "cold transfer accounting differs");
    const auto before = cache.stats(); const int calls = source_calls;
    compare(0, {2, 0, 1, 2, 1, 0}, true);
    require(cache.stats().misses == before.misses && cache.stats().transfer_bytes == before.transfer_bytes && source_calls == calls,
            "warm cache copied or fetched expert bytes");
    compare(0, {0, 4});
    require(cache.stats().evictions == 1, "expected exactly one eviction");
    compare(1, {0, 1});
    compare(2, {4, 5});
    const auto stable = cache.stats();
    bool rejected = false;
    try { cache.prepare(0, {0, 1, 2, 3}); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected && cache.stats().misses == stable.misses, "oversized active set mutated cache");
    for (const auto & invalid : {std::vector<int>{-1}, std::vector<int>{experts}}) {
        rejected = false; try { cache.prepare(0, invalid); } catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "invalid expert accepted");
    }
    rejected = false; try { cache.prepare(layers, {0}); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid layer accepted");
    bad_source = true; rejected = false;
    try { cache.prepare(0, {3}); } catch (const std::invalid_argument &) { rejected = true; }
    bad_source = false;
    require(rejected && cache.stats().misses == stable.misses, "invalid source mutated cache");
    compare(2, {5, 4});
    // Keep a projection in flight while preparing another layer on the owning stream.
    const auto physical = cache.prepare(0, {0, 1});
    Device<int> pending_ids(2); pending_ids.put(physical, stream);
    Device<__half> pending_x(hidden); pending_x.put(std::vector<__half>(hidden, __float2half(.125f)), stream);
    Device<__nv_bfloat16> pending_out(ff * 2), expected_out(ff * 2);
    const auto mat = cache.gate();
    mat.fmt->expert_matvec(mat, pending_x.p, expected_out.p, ff, pending_ids.p, 1, 2, false, stream);
    const auto expected_values = expected_out.get(ff * 2, stream);
    mat.fmt->expert_matvec(mat, pending_x.p, pending_out.p, ff, pending_ids.p, 1, 2, false, stream);
    cache.prepare(1, {4, 5});
    const auto pending_values = pending_out.get(ff * 2, stream);
    require(std::memcmp(expected_values.data(), pending_values.data(), ff * 2 * sizeof(__nv_bfloat16)) == 0,
            "slot overwritten before prior stream projection finished");
    cache.clear();
    require(cache.stats().misses == 0 && cache.stats().device_bytes == budget, "clear lost device budget or retained counters");
    compare(0, {0, 1});
    require(cache.stats().misses == 2, "clear retained old slot hits");
    rejected = false;
    try { Tq3ExpertCache too_large(layers, experts, hidden, ff, slots, budget - 1, source, stream); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "cache ignored device budget");
    std::puts("bounded TQ3 expert cache: cold/warm, eviction, layer ownership, duplicate/per-expert assignments, clear, bounds: PASS");
}
int main() { try { run(); return 0; } catch (const std::exception & error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; } }
