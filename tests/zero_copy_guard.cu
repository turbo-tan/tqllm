// Load-time LM-head conversions never let a kernel touch host memory on a PCIe GPU (1.6.1 zero-copy list item 4): the
// launchers reject host pointers there, and the chunked path the engine uses for a head in host memory (convert a row
// chunk into device staging, copy it by the copy engine) is byte-identical to the one-shot device conversion.
#include "quant_kernels.cuh"
#include "dflash_kernels.cuh"
#include "qwen35_kernels.cuh"
#include "zero_copy_guard.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
template <class F> static bool throws(F f) { try { f(); } catch (const std::invalid_argument &) { cudaGetLastError(); return true; } return false; }

static void chunked_matches(bool q5) {
    const int rows = 300, k = 512, chunk = 128;  // chunks of 128, 128, 44 rows
    const size_t qrow = size_t(k) / 256 * (q5 ? 176 : 210);
    std::vector<uint8_t> q(size_t(rows) * qrow); std::mt19937 rng(q5 ? 3 : 2);
    for (auto & b : q) b = uint8_t(rng());
    for (size_t r = 0; r < size_t(rows) * (k / 256); ++r) {  // finite block scales
        const __half d = __float2half_rn(0.01f * float(1 + r % 7));
        std::memcpy(q.data() + r * (q5 ? 176 : 210) + (q5 ? 0 : 208), &d, 2);
        if (q5) std::memcpy(q.data() + r * 176 + 2, &d, 2);
    }
    uint8_t * dq = nullptr; check(cudaMalloc(&dq, q.size())); check(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
    int8_t * w = nullptr; __half * s = nullptr; check(cudaMalloc(&w, size_t(rows) * k)); check(cudaMalloc(&s, size_t(rows) * k / 16 * 2));
    if (q5) tqllm_cuda::q5k_to_i8(dq, w, s, rows, k, nullptr); else tqllm_cuda::q6k_to_i8(dq, w, s, rows, k, nullptr);
    std::vector<int8_t> w1(size_t(rows) * k); std::vector<__half> s1(size_t(rows) * k / 16);
    check(cudaMemcpy(w1.data(), w, w1.size(), cudaMemcpyDeviceToHost)); check(cudaMemcpy(s1.data(), s, s1.size() * 2, cudaMemcpyDeviceToHost));
    // the engine's chunked path into pinned host memory
    int8_t * hw = nullptr; check(cudaMallocHost(reinterpret_cast<void **>(&hw), size_t(rows) * k * 9 / 8));
    auto * hs = reinterpret_cast<char *>(hw + size_t(rows) * k);
    int8_t * cw = nullptr; __half * cs = nullptr; check(cudaMalloc(&cw, size_t(chunk) * k)); check(cudaMalloc(&cs, size_t(chunk) * k / 16 * 2));
    for (int r0 = 0; r0 < rows; r0 += chunk) {
        const int nr = std::min(chunk, rows - r0);
        if (q5) tqllm_cuda::q5k_to_i8(dq + size_t(r0) * qrow, cw, cs, nr, k, nullptr); else tqllm_cuda::q6k_to_i8(dq + size_t(r0) * qrow, cw, cs, nr, k, nullptr);
        check(cudaMemcpyAsync(hw + size_t(r0) * k, cw, size_t(nr) * k, cudaMemcpyDeviceToHost, nullptr));
        check(cudaMemcpyAsync(hs + size_t(r0) * k / 16 * 2, cs, size_t(nr) * k / 16 * 2, cudaMemcpyDeviceToHost, nullptr));
    }
    check(cudaDeviceSynchronize());
    require(!std::memcmp(hw, w1.data(), w1.size()) && !std::memcmp(hs, s1.data(), s1.size() * 2), "chunked int8 head differs from the one-shot conversion");
    // the 4-bit head from the host copy, a chunk at a time, equals the one-shot device conversion
    uint8_t * w4a = nullptr, * w4b = nullptr; __half2 * m4a = nullptr, * m4b = nullptr;
    check(cudaMalloc(&w4a, size_t(rows) * k / 2)); check(cudaMalloc(&w4b, size_t(rows) * k / 2));
    check(cudaMalloc(&m4a, size_t(rows) * k / 32 * 4)); check(cudaMalloc(&m4b, size_t(rows) * k / 32 * 4));
    tqllm_cuda::u4h_from_i8h(w, s, w4a, m4a, int64_t(rows) * k, nullptr);
    for (int r0 = 0; r0 < rows; r0 += chunk) {
        const int nr = std::min(chunk, rows - r0);
        check(cudaMemcpyAsync(cw, hw + size_t(r0) * k, size_t(nr) * k, cudaMemcpyHostToDevice, nullptr));
        check(cudaMemcpyAsync(cs, hs + size_t(r0) * k / 16 * 2, size_t(nr) * k / 16 * 2, cudaMemcpyHostToDevice, nullptr));
        tqllm_cuda::u4h_from_i8h(cw, cs, w4b + size_t(r0) * k / 2, m4b + size_t(r0) * k / 32, int64_t(nr) * k, nullptr);
    }
    check(cudaDeviceSynchronize());
    std::vector<uint8_t> a(size_t(rows) * k / 2 + size_t(rows) * k / 32 * 4), b(a.size());
    check(cudaMemcpy(a.data(), w4a, size_t(rows) * k / 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(a.data() + size_t(rows) * k / 2, m4a, size_t(rows) * k / 32 * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(), w4b, size_t(rows) * k / 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(b.data() + size_t(rows) * k / 2, m4b, size_t(rows) * k / 32 * 4, cudaMemcpyDeviceToHost));
    require(a == b, "chunked 4-bit head differs from the one-shot conversion");
    std::printf("%s head: chunked copy-engine path byte-identical (int8 + 4-bit)\n", q5 ? "Q5_K" : "Q6_K");
    cudaFree(dq); cudaFree(w); cudaFree(s); cudaFreeHost(hw); cudaFree(cw); cudaFree(cs); cudaFree(w4a); cudaFree(w4b); cudaFree(m4a); cudaFree(m4b);
}

// the host KV tail staged by the copy engine (Qwen35Engine::kv_prefetch): [off, end) of a pinned host copy into the same
// range of a device staging buffer, the rest of the buffer untouched
static void stage_by_copy_engine() {
    const size_t bytes = size_t(3) << 20, off = 1 << 20, end = bytes - 4096;
    char * host = nullptr; check(cudaMallocHost(reinterpret_cast<void **>(&host), bytes));
    for (size_t i = 0; i < bytes; ++i) host[i] = char(i * 131 + 7);
    char * dev = nullptr; check(cudaMalloc(&dev, bytes)); check(cudaMemset(dev, 0x5a, bytes));
    check(cudaDeviceSynchronize());  // the fill (default stream) before the copy on its own stream
    cudaStream_t side; check(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    check(cudaMemcpyAsync(dev + off, host + off, end - off, cudaMemcpyHostToDevice, side)); check(cudaStreamSynchronize(side));
    std::vector<char> got(bytes); check(cudaMemcpy(got.data(), dev, bytes, cudaMemcpyDeviceToHost));
    bool ok = !std::memcmp(got.data() + off, host + off, end - off);
    for (size_t i = 0; ok && i < bytes; i += (i < off || i >= end) ? 1 : end - off) if ((i < off || i >= end) && got[i] != 0x5a) ok = false;
    require(ok, "staged host tail range differs or the copy wrote outside [off, end)");
    std::printf("host KV tail staged by the copy engine: [off, end) exact, outside untouched\n");
    cudaStreamDestroy(side); cudaFree(dev); cudaFreeHost(host);
}

int main() {
    try {
        int dev = 0; check(cudaSetDevice(dev));
        chunked_matches(false); chunked_matches(true); stage_by_copy_engine();
        if (tqllm_cuda::host_memory_coherent()) { std::printf("coherent host memory (GB10 class): zero-copy allowed, guard checks skipped: PASS\n"); return 0; }
        const int rows = 4, k = 256;
        uint8_t * dq = nullptr; int8_t * dw = nullptr; __half * ds = nullptr; uint32_t * ids = nullptr;
        check(cudaMalloc(&dq, size_t(rows) * 210)); check(cudaMalloc(&dw, size_t(rows) * k)); check(cudaMalloc(&ds, size_t(rows) * k / 16 * 2)); check(cudaMalloc(&ids, 16));
        check(cudaMemset(dq, 0, size_t(rows) * 210)); check(cudaMemset(ids, 0, 16));
        char * h = nullptr; check(cudaMallocHost(reinterpret_cast<void **>(&h), size_t(rows) * k * 2));
        int rejected = 0;
        rejected += throws([&] { tqllm_cuda::q6k_to_i8(dq, reinterpret_cast<int8_t *>(h), ds, rows, k, nullptr); });          // int8 rows into host memory
        rejected += throws([&] { tqllm_cuda::q5k_to_i8(dq, dw, reinterpret_cast<__half *>(h), rows, k, nullptr); });           // scales into host memory
        rejected += throws([&] { tqllm_cuda::u4h_from_i8h(reinterpret_cast<int8_t *>(h), ds, reinterpret_cast<uint8_t *>(dw), reinterpret_cast<__half2 *>(dq), int64_t(rows) * k, nullptr); });
        rejected += throws([&] { tqllm_cuda::gather_rows(h, dw, ids, 1, k, nullptr); });                                     // rows from host memory
        int64_t * nb = nullptr; check(cudaMalloc(&nb, 16)); check(cudaMemset(nb, 0, 16));
        rejected += throws([&] { tqllm_cuda::kv_stage_copy(h, dw, nb, 1, nullptr); });                                        // host KV tail by kernel
        check(cudaDeviceSynchronize());
        std::printf("host pointers rejected: %d of 5\n", rejected);
        require(rejected == 5, "a kernel accepted a host pointer on a PCIe GPU (zero-copy)");
        require(!throws([&] { tqllm_cuda::q6k_to_i8(dq, dw, ds, rows, k, nullptr); }), "device pointers rejected");
        check(cudaDeviceSynchronize());
        std::printf("load-time head conversions without zero-copy: PASS\n");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
