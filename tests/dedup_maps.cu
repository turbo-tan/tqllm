// moe_dedup_maps (device-built slot and dedup maps for a layer whose bank holds every expert) must equal the host build
// of the split path: same groups in the same order, same slot, columns, inputs and outputs; the groups after them slot -1.
#include "experts/moe_kernels.cuh"
#include "formats/weight/quant_kernels.cuh"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
int main() {
    constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
    const int experts = 512; bool ok = true; int cases = 0;
    for (int topk : {8, 10}) for (int n = 1; n <= 64; n = n < 8 ? n + 1 : n * 2) for (int skew = 0; skew < 3; ++skew) {
        std::mt19937 rng(topk * 1000 + n * 10 + skew); const int A = n * topk;
        std::vector<int> slot_of(experts); std::iota(slot_of.begin(), slot_of.end(), 0); std::shuffle(slot_of.begin(), slot_of.end(), rng);
        // skew 0: any expert; 1: 24 experts (many groups of more than kDedupCols); 2: 3 experts
        const int pool = skew == 0 ? experts : skew == 1 ? 24 : 3;
        std::vector<int> ids(A);
        for (int t = 0; t < n; ++t) {  // distinct experts within a token, as top-k gives
            std::vector<int> e(experts); std::iota(e.begin(), e.end(), 0); std::shuffle(e.begin(), e.begin() + pool, rng);
            for (int q = 0; q < topk; ++q) ids[t * topk + q] = e[size_t(q % pool)];
        }
        std::vector<int> want_gu(size_t(A) * DS, -1), want_dn(want_gu.size(), -1); int G = 0;
        for (int part = 0; part < 2; ++part) {
            int * out = (part ? want_dn : want_gu).data(); std::vector<int> open_at(size_t(experts) + 1, -1); int g = 0;
            for (int a = 0; a < A; ++a) {
                const int slot = slot_of[ids[a]]; int & at = open_at[size_t(slot)];
                if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = slot; out[at * DS + 1] = 0; }
                int * e = out + at * DS; const int col = e[1]++;
                e[2 + col] = part ? a : a / topk; e[2 + DC + col] = a;
            }
            G = g;
        }
        int * d_ids, * d_so, * d_slots, * d_gu, * d_dn;
        check(cudaMalloc(&d_ids, A * 4)); check(cudaMalloc(&d_so, experts * 4)); check(cudaMalloc(&d_slots, A * 4));
        check(cudaMalloc(&d_gu, size_t(A) * DS * 4)); check(cudaMalloc(&d_dn, size_t(A) * DS * 4));
        check(cudaMemcpy(d_ids, ids.data(), A * 4, cudaMemcpyHostToDevice)); check(cudaMemcpy(d_so, slot_of.data(), experts * 4, cudaMemcpyHostToDevice));
        tqllm_cuda::moe_dedup_maps(d_ids, d_so, A, topk, d_slots, d_gu, d_dn, A, nullptr);
        std::vector<int> slots(A), gu(size_t(A) * DS), dn(gu.size());
        check(cudaMemcpy(slots.data(), d_slots, A * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(gu.data(), d_gu, gu.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(dn.data(), d_dn, dn.size() * 4, cudaMemcpyDeviceToHost));
        bool same = true;
        for (int a = 0; a < A; ++a) same &= slots[a] == slot_of[ids[a]];
        for (int g = 0; g < A; ++g) for (int part = 0; part < 2; ++part) {
            const int * w = (part ? want_dn : want_gu).data() + g * DS, * got = (part ? dn : gu).data() + g * DS;
            if (g >= G) { same &= got[0] == -1; continue; }
            same &= got[0] == w[0] && got[1] == w[1];
            for (int c = 0; c < w[1]; ++c) same &= got[2 + c] == w[2 + c] && got[2 + DC + c] == w[2 + DC + c];
        }
        if (!same) { std::printf("FAIL topk %d n %d skew %d (groups %d)\n", topk, n, skew, G); ok = false; }
        ++cases;
        cudaFree(d_ids); cudaFree(d_so); cudaFree(d_slots); cudaFree(d_gu); cudaFree(d_dn);
    }
    std::printf("%s: %d cases\n", ok ? "dedup maps OK" : "dedup maps FAILED", cases);
    return ok ? 0 : 1;
}
