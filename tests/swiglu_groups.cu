// fn_swiglu_bf16_h_groups (layer-major prefill: SwiGLU on one grouped-expert pass) must equal fn_swiglu_bf16_h bit for
// bit on the listed assignments and leave every other row untouched.
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
int main() {
    bool ok = true; const int ff = 640;
    for (int A : {37, 1000, 4096}) for (int slots : {1, 7, 48}) {
        std::mt19937 rng(A * 131 + slots); std::normal_distribution<float> nd(0.f, 3.f);
        std::vector<__nv_bfloat16> g(size_t(A) * ff), u(g.size());
        for (size_t i = 0; i < g.size(); ++i) { g[i] = __float2bfloat16(nd(rng)); u[i] = __float2bfloat16(nd(rng)); }
        // a random subset of assignments spread over the slots, uneven counts, some slots empty
        std::vector<std::vector<int>> lists; lists.resize(size_t(slots));
        for (int a = 0; a < A; ++a) if (rng() % 3) { const int s = int(rng() % unsigned(slots)); if (s % 5 != 4) lists[size_t(s)].push_back(a); }
        int cap = 1; for (auto & l : lists) cap = std::max(cap, int(l.size()));
        std::vector<int> counts(static_cast<size_t>(slots)), groups(static_cast<size_t>(slots) * cap, 0);
        for (int s = 0; s < slots; ++s) { counts[size_t(s)] = int(lists[size_t(s)].size()); for (size_t i = 0; i < lists[size_t(s)].size(); ++i) groups[size_t(s) * cap + i] = lists[size_t(s)][i]; }
        __nv_bfloat16 * dg, * du; __half * full, * part; int * dc, * dgr;
        check(cudaMalloc(&dg, g.size() * 2)); check(cudaMalloc(&du, u.size() * 2)); check(cudaMalloc(&full, g.size() * 2)); check(cudaMalloc(&part, g.size() * 2));
        check(cudaMalloc(&dc, counts.size() * 4)); check(cudaMalloc(&dgr, groups.size() * 4));
        check(cudaMemcpy(dg, g.data(), g.size() * 2, cudaMemcpyHostToDevice)); check(cudaMemcpy(du, u.data(), u.size() * 2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dc, counts.data(), counts.size() * 4, cudaMemcpyHostToDevice)); check(cudaMemcpy(dgr, groups.data(), groups.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemset(part, 0x7b, g.size() * 2));
        tqllm_cuda::fn_swiglu_bf16_h(dg, du, full, int64_t(A) * ff, nullptr);
        tqllm_cuda::fn_swiglu_bf16_h_groups(dg, du, part, dc, dgr, slots, cap, ff, nullptr);
        check(cudaDeviceSynchronize());
        std::vector<uint16_t> f(g.size()), p(g.size()); check(cudaMemcpy(f.data(), full, f.size() * 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(p.data(), part, p.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<char> listed(static_cast<size_t>(A), 0); for (auto & l : lists) for (int a : l) listed[size_t(a)] = 1;
        size_t bad = 0;
        for (int a = 0; a < A; ++a) for (int j = 0; j < ff; ++j) {
            const size_t i = size_t(a) * ff + j;
            if (listed[size_t(a)] ? p[i] != f[i] : p[i] != 0x7b7b) ++bad;
        }
        std::printf("A %4d slots %2d cap %4d: %s (%zu differ)\n", A, slots, cap, bad ? "FAIL" : "ok", bad); ok = ok && !bad;
        cudaFree(dg); cudaFree(du); cudaFree(full); cudaFree(part); cudaFree(dc); cudaFree(dgr);
    }
    std::printf("%s\n", ok ? "SWIGLU_GROUPS_PASS" : "SWIGLU_GROUPS_FAIL"); return ok ? 0 : 1;
}
