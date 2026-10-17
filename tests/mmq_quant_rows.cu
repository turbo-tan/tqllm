// mmq_quantize_act_rows (layer-major prefill: q8_1 activations of one grouped-expert pass) must write the listed rows
// bit for bit as mmq_quantize_act quantises them, including slots with far more rows than the grid's row stride, and
// leave every other row untouched.
#include "quant_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
int main() {
    constexpr size_t kBlock = 144;  // block_q8_1_mmq: 4 scales + 128 int8 values
    bool ok = true;
    for (int k : {640, 2560}) for (int rows : {37, 1000, 4096}) for (int slots : {1, 7, 48}) {
        std::mt19937 rng(k * 7 + rows * 131 + slots); std::normal_distribution<float> nd(0.f, 2.f);
        std::vector<__half> x(size_t(rows) * k); for (auto & v : x) v = __float2half(nd(rng));
        // uneven slot lists (one slot gets most rows when slots == 1), some slots empty
        std::vector<std::vector<int>> lists(static_cast<size_t>(slots));
        for (int r = 0; r < rows; ++r) if (rng() % 3) { const int s = int(rng() % unsigned(slots)); if (s % 5 != 4) lists[size_t(s)].push_back(r); }
        int cap = 1; for (auto & l : lists) cap = std::max(cap, int(l.size()));
        std::vector<int> counts(static_cast<size_t>(slots)), groups(static_cast<size_t>(slots) * cap, 0);
        for (int s = 0; s < slots; ++s) {
            counts[size_t(s)] = int(lists[size_t(s)].size());
            for (size_t i = 0; i < lists[size_t(s)].size(); ++i) groups[size_t(s) * cap + i] = lists[size_t(s)][i];
        }
        const size_t bytes = tqllm_cuda::mmq_act_bytes(rows, k);
        if (bytes % (size_t(rows) * kBlock)) throw std::runtime_error("unexpected q8_1 block size");
        const size_t nb = bytes / (size_t(rows) * kBlock);
        __half * dx; void * full, * part; int * dc, * dgr;
        check(cudaMalloc(&dx, x.size() * 2)); check(cudaMalloc(&full, bytes)); check(cudaMalloc(&part, bytes));
        check(cudaMalloc(&dc, counts.size() * 4)); check(cudaMalloc(&dgr, groups.size() * 4));
        check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dc, counts.data(), counts.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dgr, groups.data(), groups.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemset(part, 0x5a, bytes));
        tqllm_cuda::mmq_quantize_act(dx, rows, k, full, nullptr);
        tqllm_cuda::mmq_quantize_act_rows(dx, rows, k, dgr, dc, cap, slots, part, nullptr);
        check(cudaDeviceSynchronize());
        std::vector<unsigned char> f(bytes), p(bytes);
        check(cudaMemcpy(f.data(), full, bytes, cudaMemcpyDeviceToHost)); check(cudaMemcpy(p.data(), part, bytes, cudaMemcpyDeviceToHost));
        std::vector<char> listed(static_cast<size_t>(rows), 0); for (auto & l : lists) for (int r : l) listed[size_t(r)] = 1;
        std::vector<unsigned char> sentinel(kBlock, 0x5a);
        size_t bad = 0;
        for (size_t j = 0; j < nb; ++j) for (int r = 0; r < rows; ++r) {
            const size_t at = (j * size_t(rows) + size_t(r)) * kBlock;
            const unsigned char * want = listed[size_t(r)] ? f.data() + at : sentinel.data();
            bad += std::memcmp(p.data() + at, want, kBlock) != 0;
        }
        ok = ok && bad == 0;
        if (bad) std::printf("k %d rows %d slots %d cap %d: %zu of %zu blocks wrong\n", k, rows, slots, cap, bad, nb * size_t(rows));
        cudaFree(dx); cudaFree(full); cudaFree(part); cudaFree(dc); cudaFree(dgr);
    }
    std::printf(ok ? "MMQ_QUANT_ROWS PASS\n" : "MMQ_QUANT_ROWS FAIL\n");
    return ok ? 0 : 1;
}
