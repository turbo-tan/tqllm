// Grouped-expert MMQ on IQ4_XS (loaded as IQ4N, the Flash-Next IQ4_XS file's gate/up experts [2560] -> 640) against a
// CPU reference from the GGUF blocks: listed assignments match within the q8_1 activation rounding, unlisted rows stay
// untouched. Buffers are allocated to their exact size, so compute-sanitizer memcheck reports any read past them
// (the server hit an out-of-bounds read in mul_mat_q<IQ4_XS> during the layer-major warm-up).
#include "formats/weight/weight_format.hpp"
#include "quant_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
template <typename T> static T * upload(const std::vector<T> & v) { T * d; check(cudaMalloc(&d, v.size() * sizeof(T))); check(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }

static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static bool run_case(int K, int ROWS, int SLOTS, int A, int div = 1, int pad = 0, bool fused = false) {
    std::mt19937 rng(11); std::uniform_int_distribution<int> byte(0, 255); std::normal_distribution<float> nd(0.f, 1.f);
    const int nb = K / 256;
    // GGUF IQ4_XS blocks for SLOTS experts of [ROWS][K] (one matrix of SLOTS * ROWS rows, as a bank holds them)
    std::vector<char> raw(size_t(SLOTS) * ROWS * nb * 136);
    std::vector<float> w(size_t(SLOTS) * ROWS * K);
    for (size_t b = 0; b < size_t(SLOTS) * ROWS * nb; ++b) {
        uint8_t * blk = reinterpret_cast<uint8_t *>(raw.data()) + b * 136;
        const __half d = __float2half(0.0005f * (1.f + std::fabs(nd(rng))));
        std::memcpy(blk, &d, 2);
        for (int i = 2; i < 136; ++i) blk[i] = uint8_t(byte(rng));
        uint16_t sh; std::memcpy(&sh, blk + 2, 2);
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            const float dl = __half2float(d) * float(ls - 32);
            for (int j = 0; j < 16; ++j) {
                const uint8_t q = blk[8 + ib * 16 + j];
                w[b * 256 + ib * 32 + j] = dl * kv[q & 0xF];
                w[b * 256 + ib * 32 + j + 16] = dl * kv[q >> 4];
            }
        }
    }
    const auto * fmt = tqllm::weight_format(23);
    std::vector<void *> keep;
    tqllm::Upload up = [&](const void * h, size_t bytes) { void * p; check(cudaMalloc(&p, bytes)); check(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice)); keep.push_back(p); return static_cast<const void *>(p); };
    const tqllm::Mat m = fmt->load(raw, SLOTS * ROWS, K, up, nullptr);
    const int id = m.fmt->u4h_id();

    const int NY = (A + div - 1) / div;  // div > 1: assignment a reads input row a / div (top-k experts per token)
    std::vector<__half> x(size_t(NY) * K); for (auto & v : x) v = __float2half(nd(rng));
    std::vector<std::vector<int>> lists(SLOTS);
    for (int a = 0; a < A; ++a) if (rng() % 4) { const int s = int(rng() % SLOTS); if (s != 1) lists[size_t(s)].push_back(a); }
    int cap = 1; for (auto & l : lists) cap = std::max(cap, int(l.size()));
    std::vector<int> counts(SLOTS), groups(size_t(SLOTS) * cap, pad);  // pad: what the unused list entries hold (the server's pinned list is not cleared)
    for (int s = 0; s < SLOTS; ++s) { counts[size_t(s)] = int(lists[size_t(s)].size()); for (size_t i = 0; i < lists[size_t(s)].size(); ++i) groups[size_t(s) * cap + i] = lists[size_t(s)][i]; }
    __half * dx = upload(x); int * dcounts = upload(counts); int * dgroups = upload(groups);
    void * yq; check(cudaMalloc(&yq, tqllm_cuda::mmq_act_bytes(NY, K)));
    tqllm_cuda::mmq_quantize_act(dx, NY, K, yq, nullptr);
    const int ntiles = tqllm_cuda::mmq_max_tiles(cap, SLOTS); int2 * tiles; check(cudaMalloc(&tiles, size_t(ntiles) * sizeof(int2)));
    __nv_bfloat16 * dy; check(cudaMalloc(&dy, size_t(A) * ROWS * 2)); check(cudaMemset(dy, 0x7b, size_t(A) * ROWS * 2));
    if (!tqllm_cuda::mmq_supports(id, K)) { std::printf("IQ4_XS (id %d) K %d: MMQ unsupported\n", id, K); return false; }
    // fused: the up projection with the SwiGLU in its epilogue (gate rows in, fp16 activations out), as the server runs it
    std::vector<__nv_bfloat16> gate(size_t(A) * ROWS); for (auto & g : gate) g = __float2bfloat16(nd(rng));
    __nv_bfloat16 * dgate = fused ? upload(gate) : nullptr; __half * dact = nullptr;
    if (fused) { check(cudaMalloc(&dact, size_t(A) * ROWS * 2)); check(cudaMemset(dact, 0, size_t(A) * ROWS * 2)); }
    const bool ok = tqllm_cuda::mmq_expert_gemm(id, static_cast<const uint8_t *>(m.p), m.aux, false, ROWS, K, dcounts, counts.data(), dgroups,
                                                cap, SLOTS, div, yq, NY, dy, tiles, ntiles, nullptr, dgate, dact);
    if (fused) { check(cudaDeviceSynchronize()); cudaFree(dgate); cudaFree(dact);
                 std::printf("IQ4_XS fused K %d rows %d slots %d A %d div %d cap %d: ran\n", K, ROWS, SLOTS, A, div, cap);
                 for (void * p : keep) cudaFree(p); return ok; }
    check(cudaDeviceSynchronize());
    if (!ok) { std::printf("IQ4_XS: gemm refused\n"); return false; }
    std::vector<uint16_t> y(size_t(A) * ROWS); check(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<int> slot_of(A, -1); for (int s = 0; s < SLOTS; ++s) for (int a : lists[size_t(s)]) slot_of[size_t(a)] = s;
    double worst = 0; size_t untouched_bad = 0, checked = 0;
    for (int a = 0; a < A; ++a) {
        if (slot_of[size_t(a)] < 0) { for (int r = 0; r < ROWS; ++r) untouched_bad += y[size_t(a) * ROWS + r] != 0x7b7b; continue; }
        if (a % 7) continue;  // a sample of the listed assignments against the CPU reference
        const int s = slot_of[size_t(a)]; double e2 = 0, r2 = 0;
        for (int r = 0; r < ROWS; ++r) {
            double ref = 0; const float * wr = &w[(size_t(s) * ROWS + r) * K];
            for (int c = 0; c < K; ++c) ref += double(wr[c]) * double(__half2float(x[size_t(a / div) * K + c]));
            __nv_bfloat16 b; std::memcpy(&b, &y[size_t(a) * ROWS + r], 2);
            const double d = double(__bfloat162float(b)) - ref; e2 += d * d; r2 += ref * ref;
        }
        worst = std::max(worst, std::sqrt(e2 / std::max(r2, 1e-30))); ++checked;
    }
    const bool pass = worst < 2e-2 && untouched_bad == 0 && checked > 0;
    std::printf("IQ4_XS K %d rows %d slots %d A %d div %d pad %d cap %d: worst rms rel err %.2e over %zu rows, %zu unlisted values written: %s\n",
                K, ROWS, SLOTS, A, div, pad, cap, worst, checked, untouched_bad, pass ? "PASS" : "FAIL");
    for (void * p : keep) cudaFree(p);
    cudaFree(dx), cudaFree(dcounts), cudaFree(dgroups), cudaFree(yq), cudaFree(tiles), cudaFree(dy);
    return pass;
}

int main() {
    bool pass = run_case(2560, 640, 3, 200);         // small
    pass = run_case(2560, 640, 48, 2000) && pass;    // Flash-Next IQ4_XS gate / up at a staging group
    pass = run_case(2560, 640, 48, 4096, 8) && pass;           // 512 tokens x top-8, input row per token (the warm-up pass)
    pass = run_case(2560, 640, 48, 4096, 8, 1 << 28) && pass;  // same, unused list entries hold garbage
    // the server's failing pass (Flash-Next IQ4_XS warm-up, layer 0): 512 tokens x top-10, 192 staged slots, cap 18
    pass = run_case(2560, 640, 192, 5120, 10, 1 << 28) && pass;
    pass = run_case(2560, 640, 192, 5120, 10, 1 << 28, true) && pass;
    std::printf(pass ? "MMQ_IQ4XS PASS\n" : "MMQ_IQ4XS FAIL\n");
    return pass ? 0 : 1;
}
