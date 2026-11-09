#pragma once
// NVIDIA (CUDA) device primitives: the PTX each tqllm_dev helper stands for.
// Included by src/core/common.cuh only (the one place a backend is selected for device code). Defines the capability macros
// (TQLLM_DEV_*) and every primitive in tqllm_dev that differs by hardware; a new backend is a new src/core/platform/<name>_device.cuh
// with the same names, and no shared file changes beyond common.cuh's one include line.
#define TQLLM_DEV_PTX 1         // NVIDIA PTX inline assembly (mma.sync, cp.async, ldmatrix, redux)
#define TQLLM_DEV_CP_ASYNC 1    // asynchronous global -> shared copies: tiles can be double-buffered
#define TQLLM_DEV_LDS_64K 0     // at most 64 KB of dynamic shared memory per block: smaller row tiles, one buffer

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

// cp.async has no RDNA3 analogue with the same lifetime rules; the staging loops always pair it with
// a block barrier, so a direct vector copy keeps the ordering and the double buffers work unchanged.
__device__ __forceinline__ void cp_async16(void * dst, const void * src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))),
                 "l"(src));
}
// cp_async16 with a predicate: false writes 16 zero bytes (cp.async src-size 0), so masked tails never keep old bytes
__device__ __forceinline__ void cp_async16_zfill(void * dst, const void * src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src),
                 "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async4(void * dst, const void * src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
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

__device__ __forceinline__ void mma_s8_k16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_bf16(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t or_half16(uint32_t v) {
    const unsigned lane = threadIdx.x & 31;
    return __reduce_or_sync(lane < 16 ? 0x0000ffffu : 0xffff0000u, v);
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(uint32_t(__cvta_generic_to_shared(p))));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(uint32_t(__cvta_generic_to_shared(p))));
}
}  // namespace tqllm_dev
