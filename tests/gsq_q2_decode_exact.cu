// gsq_q2_expert_matvec (k = 640 unrolled, code - 1 by 2^23 bias) and gsq_q2_expert_dedup equal the previous kernel form
// (I2F per weight, rolled loop; copied below) bit for bit on full-range code bytes, for the grouped and dedup launches.
#include "formats/weight/gsq_q2.cuh"
#include "formats/weight/quant_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

__global__ void old_dot_warp(const unsigned char * weights, const __half * x, __nv_bfloat16 * y, int rows, int cols, int experts,
                             const int * ids, int topk, bool per_expert) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31, assignment = blockIdx.y;
    if (row >= rows) return;
    const int expert = ids ? ids[assignment] : 0;
    const int input = per_expert ? assignment : assignment / topk;
    float sum = 0.f;
    if (expert >= 0 && expert < experts) {
        const unsigned char * r = weights + (int64_t(expert) * rows + row) * (cols / 64) * 18;
        const __half2 * xv = reinterpret_cast<const __half2 *>(x + int64_t(input) * cols);
        for (int b = 0; b < cols / 64; ++b) {
            const unsigned char * block = r + b * 18;
            const float scale = __half2float(*reinterpret_cast<const __half *>(block));
            const unsigned q = block[2 + lane / 2] >> (4 * (lane & 1));
            const float2 xx = __half22float2(xv[b * 32 + lane]);
            sum += scale * (int(q & 3u) - 1) * xx.x;
            sum += scale * (int((q >> 2) & 3u) - 1) * xx.y;
        }
    }
    for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
    if (!lane) y[int64_t(assignment) * rows + row] = __float2bfloat16_rn(expert < 0 || expert >= experts ? nanf("") : sum);
}

int main() {
    const int rows = 2560, cols = 640, experts = 64, topk = 10, n = 4, A = n * topk, DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
    std::mt19937 rng(5);
    std::vector<unsigned char> w(size_t(experts) * rows * (cols / 64) * 18);
    for (auto & b : w) b = rng() & 255;
    std::uniform_real_distribution<float> e(-12.f, -6.f);
    for (size_t b = 0; b < w.size(); b += 18) { const __half v = __float2half(std::exp2(e(rng)) * ((rng() & 1) ? -1.f : 1.f)); std::memcpy(&w[b], &v, 2); }
    std::vector<__half> x(size_t(A) * cols);
    std::uniform_real_distribution<float> xd(-2.f, 2.f);
    for (auto & v : x) v = __float2half(xd(rng));
    std::vector<int> ids(A);
    for (int t = 0; t < n; ++t) for (int j = 0; j < topk; ++j) ids[t * topk + j] = (t * 3 + j * 5) % experts;  // repeats across tokens
    unsigned char * dw; __half * dx; int * di; __nv_bfloat16 * y0, * y1, * y2;
    cudaMalloc(&dw, w.size()); cudaMalloc(&dx, x.size() * 2); cudaMalloc(&di, A * 4);
    cudaMalloc(&y0, size_t(A) * rows * 2); cudaMalloc(&y1, size_t(A) * rows * 2); cudaMalloc(&y2, size_t(A) * rows * 2);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice); cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(di, ids.data(), A * 4, cudaMemcpyHostToDevice);
    bool ok = true;
    for (bool per : {false, true}) {  // gate / up (token input) and down (assignment input) indexing
        old_dot_warp<<<dim3((rows + 7) / 8, A), 256>>>(dw, dx, y0, rows, cols, experts, di, topk, per);
        tqllm_cuda::gsq_q2_expert_matvec(dw, dx, y1, rows, cols, experts, di, n, topk, per, nullptr);
        // dedup map: one group per distinct expert, its assignments as columns
        std::vector<int> map(size_t(A) * DS, -1); int g = 0; std::vector<int> at(experts, -1);
        for (int a = 0; a < A; ++a) {
            int & q = at[ids[a]];
            if (q < 0 || map[q * DS + 1] == DC) { q = g++; map[q * DS] = ids[a]; map[q * DS + 1] = 0; }
            int * m = map.data() + q * DS; const int c = m[1]++; m[2 + c] = per ? a : a / topk; m[2 + DC + c] = a;
        }
        int * dm; cudaMalloc(&dm, map.size() * 4); cudaMemcpy(dm, map.data(), map.size() * 4, cudaMemcpyHostToDevice);
        tqllm_cuda::gsq_q2_expert_dedup(dw, dx, y2, rows, cols, dm, g, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("FAIL cuda\n"); return 1; }
        std::vector<uint16_t> a(size_t(A) * rows), b(a.size()), c(a.size());
        cudaMemcpy(a.data(), y0, a.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), y1, b.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(c.data(), y2, c.size() * 2, cudaMemcpyDeviceToHost);
        const bool g1 = std::memcmp(a.data(), b.data(), a.size() * 2) == 0, g2 = std::memcmp(a.data(), c.data(), a.size() * 2) == 0;
        std::printf("%s input: grouped %s, dedup %s\n", per ? "assignment" : "token", g1 ? "equal" : "DIFFERS", g2 ? "equal" : "DIFFERS");
        ok = ok && g1 && g2; cudaFree(dm);
    }
    std::printf(ok ? "PASS gsq_q2_decode_exact\n" : "FAIL gsq_q2_decode_exact\n");
    return ok ? 0 : 1;
}
