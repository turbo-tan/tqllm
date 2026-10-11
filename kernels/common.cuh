#pragma once
// Device helpers shared by kernels/formats/*.cu (tensor-core MMA wrappers, packed half2 math, ldmatrix).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

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

// Two-part (tree) verify attention: a query at position qpos takes keys [0, split) from the cache kernel and [split, qpos]
// from the in-block kernel. split <= qpos - TREE_MAX_DEPTH keeps it at or below the block start for any path of up to
// 16 nodes, and depends only on qpos.
constexpr int TREE_MAX_DEPTH = 15;
__host__ __device__ inline int tq_split_point(int qpos) {
    return qpos >= TREE_MAX_DEPTH ? (qpos - TREE_MAX_DEPTH) / 32 * 32 : 0;
}

}  // namespace tqllm_dev
