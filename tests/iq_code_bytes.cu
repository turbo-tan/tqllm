// iq3_code / iq3x_code (byte-local 32-bit arithmetic) equal the per-byte __vadd4 / __vsub4 select they replaced, for
// every magnitude byte the grids hold (each grid's bytes are checked to lie in that range) and all 16 sign patterns.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "formats/weight/iq_codes.cuh"

__device__ __forceinline__ uint32_t old_code(uint32_t base, uint32_t g, uint32_t sbits) {
    const uint32_t neg = ((sbits * 0x00204081u) & 0x01010101u) * 0xFFu;
    return (__vadd4(base, g) & ~neg) | (__vsub4(base, g) & neg);
}
// one thread per (4 magnitude bytes drawn from [lo, hi], sign pattern): bad[0] counts mismatches
__global__ void check_k(int lo, int hi, bool x, unsigned long long * bad) {
    const int span = hi - lo + 1;
    const long long n = (long long) span * span * span * span * 16;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x) {
        long long r = i;
        const uint32_t sb = uint32_t(r % 16); r /= 16;
        uint32_t g = 0;
        for (int b = 0; b < 4; ++b) { g |= uint32_t(lo + r % span) << (8 * b); r /= span; }
        const uint32_t got = x ? iq3x_code(g, sb) : iq3_code(g, sb), want = old_code(x ? 0x40404040u : 0x10101010u, g, sb);
        if (got != want) atomicAdd(bad, 1ull);
    }
}
static const uint32_t g3s[] = {
#include "formats/weight/iq3s_grid.inc"
};
static const uint32_t g3x[] = {
#include "formats/weight/iq3xxs_grid.inc"
};
static const uint64_t g2s[] = {
#include "formats/weight/iq2s_grid.inc"
};
static const uint64_t g2xs[] = {
#include "formats/weight/iq2xs_grid.inc"
};
static const uint64_t g2xxs[] = {
#include "formats/weight/iq2xxs_grid.inc"
};
template <class T, size_t N> static bool in_range(const T (&g)[N], int lo, int hi, const char * what) {
    for (size_t i = 0; i < N; ++i)
        for (size_t b = 0; b < sizeof(T); ++b) {
            const int v = int((uint64_t(g[i]) >> (8 * b)) & 0xFF);
            if (v < lo || v > hi) { std::printf("FAIL %s grid byte %d outside [%d, %d]\n", what, v, lo, hi); return false; }
        }
    return true;
}
int main() {
    bool ok = in_range(g3s, 0, 15, "IQ3_S") && in_range(g3x, 0, 63, "IQ3_XXS") && in_range(g2s, 0, 63, "IQ2_S") &&
              in_range(g2xs, 0, 63, "IQ2_XS") && in_range(g2xxs, 0, 63, "IQ2_XXS");
    unsigned long long * bad; cudaMallocManaged(&bad, 16); bad[0] = bad[1] = 0;
    check_k<<<1024, 256>>>(0, 15, false, bad);
    check_k<<<1024, 256>>>(0, 63, true, bad + 1);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("FAIL cuda\n"); return 1; }
    std::printf("iq3_code mismatches %llu, iq3x_code mismatches %llu\n", bad[0], bad[1]);
    ok = ok && !bad[0] && !bad[1];
    std::printf(ok ? "PASS iq_code_bytes\n" : "FAIL iq_code_bytes\n");
    return ok ? 0 : 1;
}
