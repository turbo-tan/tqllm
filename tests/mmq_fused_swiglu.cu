// MMQ up projection with the SwiGLU fused into its epilogue (mmq_expert_gemm gate / act) must equal the two-step path
// (up stored as bf16, fn_swiglu_bf16_h_groups over gate and up) bit for bit, on a synthetic IQ4_NL expert bank at the
// Flash-Next grouping (slot lists, per-assignment input), and leave the act rows of unlisted assignments untouched.
#include "quant_kernels.cuh"
#include "qwen4exp_kernels.cuh"
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

static bool run_case(int fmt, int K, int ROWS, int SLOTS, int A) {
    const int NB = K / 32;
    std::mt19937 rng(7); std::uniform_int_distribution<int> nib(0, 255); std::normal_distribution<float> nd(0.f, 1.f);
    // tqLLM IQ4_NL bank: per expert [ROWS][NB] blocks of 16 nibble bytes (natural order) + a half scale per block
    const bool iq3s = fmt == 4;
    std::vector<uint8_t> codes(size_t(SLOTS) * ROWS * NB * (iq3s ? 12 : 16)); for (auto & b : codes) b = uint8_t(nib(rng));
    std::vector<uint8_t> aux(iq3s ? size_t(SLOTS) * ROWS * (K / 256) * 16 : size_t(SLOTS) * ROWS * NB * 2);
    if (iq3s) for (size_t i = 0; i < aux.size(); i += 16) { const __half d = __float2half(0.002f + 0.01f * std::abs(nd(rng))); std::memcpy(&aux[i], &d, 2); for (int j = 2; j < 16; ++j) aux[i + j] = uint8_t(nib(rng)); }
    else for (size_t i = 0; i < aux.size(); i += 2) { const __half d = __float2half(0.01f + 0.05f * std::abs(nd(rng))); std::memcpy(&aux[i], &d, 2); }
    std::vector<__half> x(size_t(A) * K); for (auto & v : x) v = __float2half(nd(rng));
    std::vector<__nv_bfloat16> gate(size_t(A) * ROWS); for (auto & g : gate) g = __float2bfloat16(2.f * nd(rng));
    // slot lists: an uneven subset of the assignments, one slot empty
    std::vector<std::vector<int>> lists(SLOTS);
    for (int a = 0; a < A; ++a) if (rng() % 4) { const int s = int(rng() % SLOTS); if (s != 1) lists[size_t(s)].push_back(a); }
    int cap = 1; for (auto & l : lists) cap = std::max(cap, int(l.size()));
    std::vector<int> counts(SLOTS), groups(size_t(SLOTS) * cap, 0);
    for (int s = 0; s < SLOTS; ++s) { counts[size_t(s)] = int(lists[size_t(s)].size()); for (size_t i = 0; i < lists[size_t(s)].size(); ++i) groups[size_t(s) * cap + i] = lists[size_t(s)][i]; }

    uint8_t * dcodes = upload(codes); uint8_t * dscales = upload(aux); __half * dx = upload(x); __nv_bfloat16 * dgate = upload(gate);
    int * dcounts = upload(counts); int * dgroups = upload(groups);
    void * yq; check(cudaMalloc(&yq, tqllm_cuda::mmq_act_bytes(A, K)));
    tqllm_cuda::mmq_quantize_act(dx, A, K, yq, nullptr);
    const int ntiles = tqllm_cuda::mmq_max_tiles(cap, SLOTS); int2 * tiles; check(cudaMalloc(&tiles, size_t(ntiles) * sizeof(int2)));
    __nv_bfloat16 * dup; __half * act_ref, * act_fused;
    check(cudaMalloc(&dup, size_t(A) * ROWS * 2)); check(cudaMalloc(&act_ref, size_t(A) * ROWS * 2)); check(cudaMalloc(&act_fused, size_t(A) * ROWS * 2));
    check(cudaMemset(dup, 0, size_t(A) * ROWS * 2)); check(cudaMemset(act_ref, 0x7b, size_t(A) * ROWS * 2)); check(cudaMemset(act_fused, 0x7b, size_t(A) * ROWS * 2));
    if (!tqllm_cuda::mmq_supports(fmt, K)) { std::printf("fmt %d unsupported here\n", fmt); return true; }
    // two steps: up -> bf16, then the grouped swiglu
    bool ok = tqllm_cuda::mmq_expert_gemm(fmt, dcodes, dscales, false, ROWS, K, dcounts, counts.data(), dgroups, cap, SLOTS, 1, yq, A, dup, tiles, ntiles, nullptr);
    tqllm_cuda::fn_swiglu_bf16_h_groups(dgate, dup, act_ref, dcounts, dgroups, SLOTS, cap, ROWS, nullptr);
    // one launch: the epilogue reads the gate and writes the activation
    ok = ok && tqllm_cuda::mmq_expert_gemm(fmt, dcodes, dscales, false, ROWS, K, dcounts, counts.data(), dgroups, cap, SLOTS, 1, yq, A, dup, tiles, ntiles, nullptr, dgate, act_fused);
    check(cudaDeviceSynchronize());
    if (!ok) { std::printf("fmt %d: gemm refused\n", fmt); return false; }
    std::vector<uint16_t> r(size_t(A) * ROWS), f(r.size()), u(r.size());
    check(cudaMemcpy(r.data(), act_ref, r.size() * 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(f.data(), act_fused, f.size() * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(u.data(), dup, u.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<char> listed(A, 0); for (auto & l : lists) for (int a : l) listed[size_t(a)] = 1;
    size_t bad = 0, nonzero_up = 0;
    for (int a = 0; a < A; ++a) for (int i = 0; i < ROWS; ++i) {
        const size_t idx = size_t(a) * ROWS + i;
        if (listed[size_t(a)]) {
            if (r[idx] != f[idx] && bad < 3) {
                __half hr, hf; std::memcpy(&hr, &r[idx], 2); std::memcpy(&hf, &f[idx], 2); __nv_bfloat16 bu, bg; std::memcpy(&bu, &u[idx], 2); std::memcpy(&bg, &gate[idx], 2);
                const float gg = __bfloat162float(bg), uu = __bfloat162float(bu);
                std::printf("  mismatch a %d i %d: ref %.5f fused %.5f (gate %.4f up %.4f -> silu*up %.5f)\n", a, i, __half2float(hr), __half2float(hf), gg, uu, gg / (1.f + std::exp(-gg)) * uu);
            }
            bad += r[idx] != f[idx]; nonzero_up += u[idx] != 0;
        } else bad += f[idx] != 0x7b7b;
    }
    std::printf("fmt %d K %d rows %d slots %d A %d: fused vs two-step %zu of %zu values differ (%zu listed up values non-zero)\n", fmt, K, ROWS, SLOTS, A, bad, r.size(), nonzero_up);
    cudaFree(dcodes); cudaFree(dscales); cudaFree(dx); cudaFree(dgate); cudaFree(dcounts); cudaFree(dgroups); cudaFree(yq); cudaFree(tiles); cudaFree(dup); cudaFree(act_ref); cudaFree(act_fused);
    return bad == 0 && nonzero_up > 0;
}
int main() {
    bool pass = run_case(14, 256, 64, 3, 200);          // IQ4_NL, small
    pass = run_case(4, 2560, 640, 48, 2000) && pass;    // IQ3_S at the Flash-Next gate/up shape (k = hidden, ff rows, a staging group)
    pass = run_case(4, 640, 2560, 48, 2000) && pass;    // IQ3_S down shape
    pass = run_case(14, 640, 2560, 48, 2000) && pass;   // IQ4_NL down shape (Flash-Next IQ4_XS file: ffn_down_exps IQ4_NL [640])
    std::printf(pass ? "MMQ_FUSED_SWIGLU PASS\n" : "MMQ_FUSED_SWIGLU FAIL\n");
    return pass ? 0 : 1;
}
