#pragma once
// Device helpers shared by the format kernels (src/formats/*/*.cu) (tensor-core MMA wrappers, packed half2 math, ldmatrix).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_set>

namespace tqllm_dev {

__device__ __forceinline__ void mma_f16(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ uint32_t hadd2u(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("add.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ void mma_f16acc(uint32_t (&c)[2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(uint32_t(__cvta_generic_to_shared(p))));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(uint32_t(__cvta_generic_to_shared(p))));
}

// One level of the transposing butterfly over a lane's N values: lanes with bit OFF set keep their upper half and add
// the partner's lower half, the others the reverse; N / 2 values remain. Compile-time OFF and N keep the array in
// registers (a runtime loop turned the 15 shuffles into ~500 instructions of register-array emulation). The pairs are
// those of the shfl_xor reduction tree (a + b == b + a), so each value's sum has the same bits.
template<int OFF, int N>
__device__ __forceinline__ void butterfly_fold(float (&a)[16], int lane) {
    const bool upper = lane & OFF;
    #pragma unroll
    for (int j = 0; j < N / 2; ++j) {
        const float mine = upper ? a[j + N / 2] : a[j], send = upper ? a[j] : a[j + N / 2];
        a[j] = mine + __shfl_xor_sync(0xffffffffu, send, OFF);
    }
    if constexpr (OFF > 2) butterfly_fold<OFF / 2, N / 2>(a, lane);
}

// Two-part (tree) verify attention: a query at position qpos takes keys [0, split) from the cache kernel and [split, qpos]
// from the in-block kernel. split <= qpos - TREE_MAX_DEPTH keeps it at or below the block start for any path of up to
// 16 nodes, and depends only on qpos.
constexpr int TREE_MAX_DEPTH = 15;
__host__ __device__ inline int tq_split_point(int qpos) {
    return qpos >= TREE_MAX_DEPTH ? (qpos - TREE_MAX_DEPTH) / 32 * 32 : 0;
}

// L1 for streaming matvec kernels (host, once per kernel): the smallest shared-memory carveout that keeps the kernel's
// full occupancy. The driver default for a kernel with a few KB of static shared memory is a 64 KB carveout (L1 ~60 KB on
// sm_86), too small for the 8-column activations (80-280 KB) that every 16-row block re-reads; with 32 KB the TQT verify
// matvec reads more of them from L1 (Qwen3.8-27B ffn gate+up at 8 columns in the verify 130 -> 120 us on a 3090 at its
// power cap). Launch configuration only: the arithmetic, and so every output bit, does not change. The carveout is per
// function: a kernel that also serves 1-column decode gets a second instance for this.
inline void prefer_l1(const void * kern, int threads, size_t dyn_smem = 0) {
    static std::mutex mu;
    static std::unordered_set<const void *> done;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!done.insert(kern).second) return;
    }
    int dev = 0, per_sm = 0, smem_sm = 0, reserved = 0;
    cudaFuncAttributes fa{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaFuncGetAttributes(&fa, kern) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, threads, dyn_smem) != cudaSuccess ||
        cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev) != cudaSuccess || smem_sm <= 0) {
        (void)cudaGetLastError();  // a hint only: keep the driver's choice
        return;
    }
    const size_t need = size_t(per_sm) * (fa.sharedSizeBytes + dyn_smem + size_t(reserved));
    const int pct = int(std::min<size_t>(100, (need * 100 + size_t(smem_sm) - 1) / size_t(smem_sm)));
    if (cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, pct) != cudaSuccess) (void)cudaGetLastError();
}

}  // namespace tqllm_dev
