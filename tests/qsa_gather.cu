// QsaGather: the selected K/V rows of a decode step copied into one compact buffer (TQFN_KV_HOST decode).
// Contract checked here against a CPU model:
//   - a cell is copied when some token t selects it with valid != 0 and 0 <= pos < first_cells + t
//   - copied cells keep their order (compact index = number of copied cells below it), each copied once
//   - positions() holds the compact index per selection, -1 where the attention skips the row
//   - attention over the compact rows gives the same bits as attention over the rows in place
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void require(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }

template <class T> struct Device {
    T * p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & x) { ck(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) const { std::vector<T> x(n); ck(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x; }
};

struct Selection {
    int tokens, width, first;
    std::vector<int32_t> positions; std::vector<uint8_t> valid;
    Selection(int t, int w, int f) : tokens(t), width(w), first(f), positions(size_t(t) * w), valid(size_t(t) * w) {}
    int cells() const { return first + tokens - 1; }
};

struct Model { std::vector<int> map; int rows = 0; std::vector<int32_t> positions; };

static Model reference(const Selection & s) {
    Model m; m.map.assign(size_t(s.cells()), -1); std::vector<uint8_t> used(size_t(s.cells()));
    auto seen = [&](int t, int w) { const size_t i = size_t(t) * s.width + w; const int p = s.positions[i];
                                    return s.valid[i] && p >= 0 && p < s.first + t ? p : -1; };
    for (int t = 0; t < s.tokens; ++t) for (int w = 0; w < s.width; ++w) if (const int p = seen(t, w); p >= 0) used[size_t(p)] = 1;
    for (int c = 0; c < s.cells(); ++c) if (used[size_t(c)]) m.map[size_t(c)] = m.rows++;
    m.positions.resize(s.positions.size());
    for (int t = 0; t < s.tokens; ++t) for (int w = 0; w < s.width; ++w) {
        const int p = seen(t, w); m.positions[size_t(t) * s.width + w] = p >= 0 ? m.map[size_t(p)] : -1;
    }
    return m;
}

// one bf16 K/V row per cell, every row different
struct Rows {
    int cells; size_t row;  // row in elements
    std::vector<__nv_bfloat16> keys, values;
    Rows(int c, size_t r) : cells(c), row(r), keys(size_t(c) * r), values(size_t(c) * r) {
        for (size_t i = 0; i < keys.size(); ++i) {
            keys[i] = __float2bfloat16(float(int(i * 13 % 197) - 98) / 43.f);
            values[i] = __float2bfloat16(float(int(i * 31 % 181) - 90) / 47.f);
        }
    }
    size_t row_bytes() const { return row * sizeof(__nv_bfloat16); }
};

// gather on the device and compare positions and every copied row with the CPU model
static void expect_gather(tqllm_cuda::QsaGather & g, const Selection & s, const Rows & r, const void * keys, const void * values,
                          cudaStream_t stream = nullptr, int host_cells = 0) {
    Device<int32_t> dp(s.positions.size()); Device<uint8_t> dm(s.valid.size()); dp.put(s.positions); dm.put(s.valid);
    g.run(dp.p, dm.p, s.tokens, s.width, s.first, keys, values, stream, host_cells);
    ck(cudaStreamSynchronize(stream)); ck(cudaDeviceSynchronize());
    const Model m = reference(s);
    std::vector<int32_t> got(s.positions.size());
    ck(cudaMemcpy(got.data(), g.positions(), got.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i)
        require(got[i] == m.positions[i], "position " + std::to_string(i) + ": got " + std::to_string(got[i]) + " want " + std::to_string(m.positions[i]));
    std::vector<__nv_bfloat16> k(size_t(m.rows) * r.row), v(k.size());
    if (m.rows) {
        ck(cudaMemcpy(k.data(), g.keys(), k.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(v.data(), g.values(), v.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    }
    for (int c = 0; c < s.cells(); ++c) if (const int j = m.map[size_t(c)]; j >= 0) {
        require(!std::memcmp(&k[size_t(j) * r.row], &r.keys[size_t(c) * r.row], r.row_bytes()), "key row of cell " + std::to_string(c));
        require(!std::memcmp(&v[size_t(j) * r.row], &r.values[size_t(c) * r.row], r.row_bytes()), "value row of cell " + std::to_string(c));
    }
}

static Selection mixed(int tokens, int width, int first) {  // overlapping picks, some invalid, some past each token's cut
    Selection s(tokens, width, first);
    for (int t = 0; t < tokens; ++t) for (int w = 0; w < width; ++w) {
        const size_t i = size_t(t) * width + w;
        s.valid[i] = (w + t) % 9 != 0;
        s.positions[i] = w < width - 3 ? (w * 2 + t * 5) % (first + t) : first + t + w % 3;
    }
    return s;
}

static void matches_reference() {
    const Selection s = mixed(6, 2048, 4995); const Rows r(s.cells(), 512);
    Device<__nv_bfloat16> k(r.keys.size()), v(r.values.size()); k.put(r.keys); v.put(r.values);
    tqllm_cuda::QsaGather g(s.cells(), r.row_bytes());
    expect_gather(g, s, r, k.p, v.p);
}

static void duplicates_copied_once() {
    Selection s(4, 64, 100);  // every token picks the same 8 cells, 8 times each
    for (size_t i = 0; i < s.positions.size(); ++i) { s.positions[i] = int(i % 8) * 11; s.valid[i] = 1; }
    const Rows r(s.cells(), 256);
    Device<__nv_bfloat16> k(r.keys.size()), v(r.values.size()); k.put(r.keys); v.put(r.values);
    tqllm_cuda::QsaGather g(s.cells(), r.row_bytes());
    expect_gather(g, s, r, k.p, v.p);
    require(reference(s).rows == 8, "8 distinct cells");
}

static void skips_invalid_negative_and_future() {
    Selection s(3, 16, 50);
    for (int t = 0; t < 3; ++t) for (int w = 0; w < 16; ++w) {
        const size_t i = size_t(t) * 16 + w;
        s.valid[i] = w != 3;
        s.positions[i] = w == 5 ? -1 : w == 7 ? 50 + t : w == 9 ? 49 + t : w * 3;  // 50 + t is past token t's cut, 49 + t its last cell
    }
    const Rows r(s.cells(), 256);
    Device<__nv_bfloat16> k(r.keys.size()), v(r.values.size()); k.put(r.keys); v.put(r.values);
    tqllm_cuda::QsaGather g(s.cells(), r.row_bytes());
    expect_gather(g, s, r, k.p, v.p);
}

static void empty_selection() {
    Selection s(2, 32, 40);
    for (size_t i = 0; i < s.positions.size(); ++i) { s.positions[i] = int(i); s.valid[i] = 0; }
    const Rows r(s.cells(), 256);
    Device<__nv_bfloat16> k(r.keys.size()), v(r.values.size()); k.put(r.keys); v.put(r.values);
    tqllm_cuda::QsaGather g(s.cells(), r.row_bytes());
    expect_gather(g, s, r, k.p, v.p);
    require(reference(s).rows == 0, "no rows");
}

static void grows_past_first_shape() {
    const Rows r(70000, 256);  // one long request after a short one: more cells and more selections than at construction
    Device<__nv_bfloat16> k(r.keys.size()), v(r.values.size()); k.put(r.keys); v.put(r.values);
    tqllm_cuda::QsaGather g(1000, r.row_bytes());
    expect_gather(g, mixed(2, 256, 999), r, k.p, v.p);
    expect_gather(g, mixed(6, 2048, 69995), r, k.p, v.p);
    expect_gather(g, mixed(1, 128, 500), r, k.p, v.p);  // and back to a short one
}

static void reads_host_memory() {
    const Selection s = mixed(6, 2048, 8000); const Rows r(s.cells(), 512);
    void * hk = nullptr, * hv = nullptr;  // mapped pinned host pages, as K/V moved to host by TQFN_KV_HOST
    ck(cudaHostAlloc(&hk, r.keys.size() * sizeof(__nv_bfloat16), cudaHostAllocMapped));
    ck(cudaHostAlloc(&hv, r.values.size() * sizeof(__nv_bfloat16), cudaHostAllocMapped));
    std::memcpy(hk, r.keys.data(), r.keys.size() * sizeof(__nv_bfloat16)); std::memcpy(hv, r.values.data(), r.values.size() * sizeof(__nv_bfloat16));
    void * dk = nullptr, * dv = nullptr; ck(cudaHostGetDevicePointer(&dk, hk, 0)); ck(cudaHostGetDevicePointer(&dv, hv, 0));
    try { tqllm_cuda::QsaGather g(s.cells(), r.row_bytes()); expect_gather(g, s, r, dk, dv); }
    catch (...) { cudaFreeHost(hk); cudaFreeHost(hv); throw; }
    ck(cudaFreeHost(hk)); ck(cudaFreeHost(hv));
}

static void ordered_on_its_stream() {
    const Selection s = mixed(5, 1024, 3000); const Rows r(s.cells(), 512);
    Device<__nv_bfloat16> k(r.keys.size()), v(r.values.size()); k.put(r.keys); v.put(r.values);
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try { tqllm_cuda::QsaGather g(s.cells(), r.row_bytes()); expect_gather(g, s, r, k.p, v.p, stream); }
    catch (...) { cudaStreamDestroy(stream); throw; }
    ck(cudaStreamDestroy(stream));
}


// Row cache (cache_rows > 0): rows of cells below host_cells (K/V in host memory) are kept in a direct-mapped device cache
// between calls. Expectations: R1 every call's compact rows equal the uncached gather's (checked against the CPU model);
// R2 a repeated selection reads every host row from the cache; R3 two cells of one call on the same slot stay correct in
// later calls (one claims the slot, no torn row / tag); R4 rows of device cells (>= host_cells) are never cached.
struct HostRows {
    void * hk = nullptr, * hv = nullptr, * dk = nullptr, * dv = nullptr;
    explicit HostRows(const Rows & r) {
        ck(cudaHostAlloc(&hk, r.keys.size() * sizeof(__nv_bfloat16), cudaHostAllocMapped));
        ck(cudaHostAlloc(&hv, r.values.size() * sizeof(__nv_bfloat16), cudaHostAllocMapped));
        std::memcpy(hk, r.keys.data(), r.keys.size() * sizeof(__nv_bfloat16)); std::memcpy(hv, r.values.data(), r.values.size() * sizeof(__nv_bfloat16));
        ck(cudaHostGetDevicePointer(&dk, hk, 0)); ck(cudaHostGetDevicePointer(&dv, hv, 0));
    }
    ~HostRows() { cudaFreeHost(hk); cudaFreeHost(hv); }
};

static Selection shifted(int tokens, int width, int first, int shift) {  // the next step's selection: mostly the same cells
    Selection s(tokens, width, first);
    for (int t = 0; t < tokens; ++t) for (int w = 0; w < width; ++w) {
        const size_t i = size_t(t) * width + w;
        s.valid[i] = (w + t) % 11 != 0;
        s.positions[i] = (w * 7 + t * 3 + (w % 5 == 0 ? shift * 13 : 0)) % (first + t);
    }
    return s;
}

static void cache_matches_uncached_over_steps() {
    const Rows r(20100, 512); HostRows h(r);
    tqllm_cuda::QsaGather g(20100, r.row_bytes(), 4096);
    for (int step = 0; step < 6; ++step) expect_gather(g, shifted(4, 2048, 20000 + step * 3, step), r, h.dk, h.dv, nullptr, 15000);
}

static void repeat_selection_hits_cache() {
    const Rows r(20100, 512); HostRows h(r);
    tqllm_cuda::QsaGather g(20100, r.row_bytes(), 32768);  // more slots than cells: no conflicts
    const Selection s = shifted(4, 2048, 20000, 0);
    expect_gather(g, s, r, h.dk, h.dv, nullptr, 15000);
    const auto first = g.stats();
    expect_gather(g, s, r, h.dk, h.dv, nullptr, 15000);
    const auto second = g.stats();
    const uint64_t host_rows = second.host_rows - first.host_rows, hits = second.cache_hits - first.cache_hits;
    require(host_rows > 0 && hits == host_rows, "repeat: " + std::to_string(hits) + " of " + std::to_string(host_rows) + " host rows from the cache");
}

static void same_slot_conflict_is_safe() {
    const Rows r(5000, 512); HostRows h(r);
    tqllm_cuda::QsaGather g(5000, r.row_bytes(), 64);
    Selection both(1, 8, 4000);  // cells 10 and 74 (and 138, 202) share slot 10 of 64
    const int cells[8] = {10, 74, 138, 202, 11, 75, 300, 364};
    for (int w = 0; w < 8; ++w) { both.positions[size_t(w)] = cells[w]; both.valid[size_t(w)] = 1; }
    expect_gather(g, both, r, h.dk, h.dv, nullptr, 4000);
    for (int w = 0; w < 8; ++w) {
        Selection one(1, 1, 4000); one.positions[0] = cells[w]; one.valid[0] = 1;
        expect_gather(g, one, r, h.dk, h.dv, nullptr, 4000);
    }
}

static void device_cells_not_cached() {
    const Rows r(5000, 512); HostRows h(r);
    tqllm_cuda::QsaGather g(5000, r.row_bytes(), 8192);
    Selection s(2, 256, 4000);
    for (size_t i = 0; i < s.positions.size(); ++i) { s.positions[i] = 1000 + int(i % 2900); s.valid[i] = 1; }
    expect_gather(g, s, r, h.dk, h.dv, nullptr, 1000);
    const auto a = g.stats();
    expect_gather(g, s, r, h.dk, h.dv, nullptr, 1000);
    const auto b = g.stats();
    require(b.host_rows == a.host_rows && b.cache_hits == a.cache_hits, "device cells read or cached as host rows");
}


// R5: K/V rows rewritten at or below the cached cells (a new prompt that shares only part of the old one) are never
// served stale: run() with written_from at or below a cached cell, or with other K/V buffers, empties the cache
static void rewrite_invalidates_cache() {
    Rows r(5000, 512); HostRows h(r);
    tqllm_cuda::QsaGather g(5000, r.row_bytes(), 8192);
    Selection s(1, 64, 4000);
    for (int w = 0; w < 64; ++w) { s.positions[size_t(w)] = 100 + w; s.valid[size_t(w)] = 1; }
    expect_gather(g, s, r, h.dk, h.dv, nullptr, 4000);  // cells 100..163 cached
    for (size_t i = size_t(120) * r.row; i < size_t(121) * r.row; ++i) {  // the new prompt's row 120 differs
        r.keys[i] = __float2bfloat16(7.f); r.values[i] = __float2bfloat16(-7.f);
    }
    std::memcpy(h.hk, r.keys.data(), r.keys.size() * sizeof(__nv_bfloat16)); std::memcpy(h.hv, r.values.data(), r.values.size() * sizeof(__nv_bfloat16));
    Device<int32_t> dp(s.positions.size()); Device<uint8_t> dm(s.valid.size()); dp.put(s.positions); dm.put(s.valid);
    g.run(dp.p, dm.p, s.tokens, s.width, s.first, h.dk, h.dv, nullptr, 4000, 120);  // this call wrote from position 120 on
    ck(cudaDeviceSynchronize());
    const Model m = reference(s); const int j = m.map[120];
    std::vector<__nv_bfloat16> k(r.row);
    ck(cudaMemcpy(k.data(), static_cast<const char *>(g.keys()) + size_t(j) * r.row_bytes(), r.row_bytes(), cudaMemcpyDeviceToHost));
    require(!std::memcmp(k.data(), &r.keys[size_t(120) * r.row], r.row_bytes()), "a rewritten row came from the cache (stale)");
}


// R6: the gather runs inside CUDA graph capture (decode layers are captured): a call with more cells and tokens than an
// earlier one must not allocate or sync during capture when the gather was built for that size (max_rows)
static void runs_inside_graph_capture() {
    const Rows r(4200, 512);
    Device<__nv_bfloat16> k(r.keys.size()), v(r.values.size()); k.put(r.keys); v.put(r.values);
    tqllm_cuda::QsaGather g(4200, r.row_bytes(), 0, 6 * 2048);
    expect_gather(g, mixed(1, 256, 1000), r, k.p, v.p);  // small first call
    const Selection s = mixed(6, 2048, 4100);
    Device<int32_t> dp(s.positions.size()); Device<uint8_t> dm(s.valid.size()); dp.put(s.positions); dm.put(s.valid);
    cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    std::string failed;
    try { g.run(dp.p, dm.p, s.tokens, s.width, s.first, k.p, v.p, st); } catch (const std::exception & e) { failed = e.what(); }
    const cudaError_t end = cudaStreamEndCapture(st, &graph); cudaGetLastError();
    require(failed.empty() && end == cudaSuccess && graph, "capture of a larger call failed: " + (failed.empty() ? std::string(cudaGetErrorString(end)) : failed));
    ck(cudaGraphInstantiate(&exec, graph, 0)); ck(cudaGraphLaunch(exec, st)); ck(cudaStreamSynchronize(st));
    const Model m = reference(s);
    std::vector<int32_t> got(s.positions.size()); ck(cudaMemcpy(got.data(), g.positions(), got.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) require(got[i] == m.positions[i], "captured gather position " + std::to_string(i));
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(st);
}

static void rejects_bad_shapes() {
    auto throws = [](const std::function<void()> & f) { try { f(); } catch (const std::invalid_argument &) { return true; } return false; };
    require(throws([] { tqllm_cuda::QsaGather g(0, 512); }), "zero cells");
    require(throws([] { tqllm_cuda::QsaGather g(10, 0); }), "zero row bytes");
    require(throws([] { tqllm_cuda::QsaGather g(10, 520); }), "row bytes not a multiple of 16");
    tqllm_cuda::QsaGather g(10, 512);
    require(throws([&] { g.run(nullptr, nullptr, 0, 4, 5, nullptr, nullptr, nullptr); }), "zero tokens");
    require(throws([&] { g.run(nullptr, nullptr, 1, 0, 5, nullptr, nullptr, nullptr); }), "zero width");
    require(throws([&] { g.run(nullptr, nullptr, 1, 4, 0, nullptr, nullptr, nullptr); }), "zero cells");
}

static void attention_bits_unchanged() {
    constexpr int kvh = 2, heads = 24, dim = 256;
    const Selection s = mixed(6, 2048, 4995); const Rows r(s.cells(), size_t(kvh) * dim);
    std::vector<float> query(size_t(s.tokens) * heads * dim);
    for (size_t i = 0; i < query.size(); ++i) query[i] = float(int(i * 7 % 113) - 56) / 29.f;
    Device<__nv_bfloat16> dk(r.keys.size()), dv(r.values.size()); Device<float> dq(query.size()), o1(query.size()), o2(query.size());
    Device<int32_t> dp(s.positions.size()); Device<uint8_t> dm(s.valid.size());
    dk.put(r.keys); dv.put(r.values); dq.put(query); dp.put(s.positions); dm.put(s.valid);
    tqllm_cuda::qsa_attention_direct(dq.p, dk.p, dv.p, {dp.p, dm.p, s.width}, o1.p, s.tokens, s.first, heads, kvh, dim, nullptr);
    tqllm_cuda::QsaGather g(s.cells(), r.row_bytes());
    g.run(dp.p, dm.p, s.tokens, s.width, s.first, dk.p, dv.p, nullptr);
    tqllm_cuda::qsa_attention_direct(dq.p, static_cast<const __nv_bfloat16 *>(g.keys()), static_cast<const __nv_bfloat16 *>(g.values()),
                                     {g.positions(), dm.p, s.width}, o2.p, s.tokens, s.tokens * s.width + 1, heads, kvh, dim, nullptr);
    ck(cudaDeviceSynchronize());
    const auto a = o1.get(query.size()), b = o2.get(query.size());
    for (size_t i = 0; i < a.size(); ++i) require(!std::memcmp(&a[i], &b[i], sizeof(float)), "attention output differs at " + std::to_string(i));
}

int main() {
    const struct { const char * name; void (*run)(); } cases[] = {
        {"matches_reference", matches_reference},
        {"duplicates_copied_once", duplicates_copied_once},
        {"skips_invalid_negative_and_future", skips_invalid_negative_and_future},
        {"empty_selection", empty_selection},
        {"grows_past_first_shape", grows_past_first_shape},
        {"reads_host_memory", reads_host_memory},
        {"ordered_on_its_stream", ordered_on_its_stream},
        {"rejects_bad_shapes", rejects_bad_shapes},
        {"attention_bits_unchanged", attention_bits_unchanged},
        {"cache_matches_uncached_over_steps", cache_matches_uncached_over_steps},
        {"repeat_selection_hits_cache", repeat_selection_hits_cache},
        {"same_slot_conflict_is_safe", same_slot_conflict_is_safe},
        {"device_cells_not_cached", device_cells_not_cached},
        {"rewrite_invalidates_cache", rewrite_invalidates_cache},
        {"runs_inside_graph_capture", runs_inside_graph_capture},
    };
    int failed = 0;
    for (const auto & c : cases) {
        try { c.run(); std::printf("PASS %s\n", c.name); }
        catch (const std::exception & e) { ++failed; std::printf("FAIL %s: %s\n", c.name, e.what()); cudaGetLastError(); }
    }
    std::printf("%d of %zu cases failed\n", failed, sizeof(cases) / sizeof(cases[0]));
    return failed ? 1 : 0;
}
