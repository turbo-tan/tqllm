#pragma once
// Device helpers shared by the format kernels (src/formats/*/*.cu) (tensor-core MMA wrappers, packed half2 math, ldmatrix).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace tqllm_dev {

#if defined(TQLLM_WITH_HIP)
// ---- AMD GPU native device math --------------------------------------------------------------
// RDNA3 has no CUDA tensor-core PTX. The m16n8k16 fragment contract is kept bit-for-bit in the
// register layout and computed with native packed instructions: v_dot2_f32_f16 / _bf16 (two
// multiply-adds into an f32 accumulator per instruction) and v_dot4 (four int8 MACs). Every path
// below is a native AMD instruction, not a CUDA-PTX interpreter; the memory-bound decode kernels
// need no more than this, and prefill runs on hipBLAS matrix cores.

__device__ __forceinline__ uint32_t h2u(__half2 h) {
    uint32_t x;
    __builtin_memcpy(&x, &h, 4);
    return x;
}
__device__ __forceinline__ __half2 u2h(uint32_t x) {
    __half2 h;
    __builtin_memcpy(&h, &x, 4);
    return h;
}
// ext_vector_type is a typedef attribute: on a declaration clang rejects it ("only applies to typedefs"), and the
// ROCm clang in CI is stricter here than a hand-built toolchain. Name the type once.
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float fdot2(uint32_t a, uint32_t b, float c) {
    f16x2 va, vb;
    __builtin_memcpy(&va, &a, 4);
    __builtin_memcpy(&vb, &b, 4);
    return __builtin_amdgcn_fdot2(va, vb, c, false);
}
__device__ __forceinline__ float fdot2_bf16(uint32_t a, uint32_t b, float c) {
    __nv_bfloat162 va, vb;
    __builtin_memcpy(&va, &a, 4);
    __builtin_memcpy(&vb, &b, 4);
    return __builtin_amdgcn_fdot2_f32_bf16(va, vb, c, false);
}

__device__ __forceinline__ void mma_f16(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t bc0[8], bc1[8], ar[8], br[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        bc0[2 * j] = __shfl_sync(0xffffffffu, b0, 8 * t + j);
        bc1[2 * j] = __shfl_sync(0xffffffffu, b0, 8 * t + 4 + j);
        bc0[2 * j + 1] = __shfl_sync(0xffffffffu, b1, 8 * t + j);
        bc1[2 * j + 1] = __shfl_sync(0xffffffffu, b1, 8 * t + 4 + j);
        ar[2 * j] = __shfl_sync(0xffffffffu, a0, 4 * g + j);
        ar[2 * j + 1] = __shfl_sync(0xffffffffu, a2, 4 * g + j);
        br[2 * j] = __shfl_sync(0xffffffffu, a1, 4 * g + j);
        br[2 * j + 1] = __shfl_sync(0xffffffffu, a3, 4 * g + j);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        c[0] = fdot2(ar[j], bc0[j], c[0]);
        c[1] = fdot2(ar[j], bc1[j], c[1]);
        c[2] = fdot2(br[j], bc0[j], c[2]);
        c[3] = fdot2(br[j], bc1[j], c[3]);
    }
}

__device__ __forceinline__ void mma_bf16(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t bc0[8], bc1[8], ar[8], br[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        bc0[2 * j] = __shfl_sync(0xffffffffu, b0, 8 * t + j);
        bc1[2 * j] = __shfl_sync(0xffffffffu, b0, 8 * t + 4 + j);
        bc0[2 * j + 1] = __shfl_sync(0xffffffffu, b1, 8 * t + j);
        bc1[2 * j + 1] = __shfl_sync(0xffffffffu, b1, 8 * t + 4 + j);
        ar[2 * j] = __shfl_sync(0xffffffffu, a0, 4 * g + j);
        ar[2 * j + 1] = __shfl_sync(0xffffffffu, a2, 4 * g + j);
        br[2 * j] = __shfl_sync(0xffffffffu, a1, 4 * g + j);
        br[2 * j + 1] = __shfl_sync(0xffffffffu, a3, 4 * g + j);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        c[0] = fdot2_bf16(ar[j], bc0[j], c[0]);
        c[1] = fdot2_bf16(ar[j], bc1[j], c[1]);
        c[2] = fdot2_bf16(br[j], bc0[j], c[2]);
        c[3] = fdot2_bf16(br[j], bc1[j], c[3]);
    }
}

// f16 accumulator: the contract rounds to f16 once per mma. Accumulate in f32 (native dot2) and
// fold the rounded result into the caller's half2 lanes.
__device__ __forceinline__ void mma_f16acc(uint32_t (&c)[2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1) {
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_f16(d, a0, a1, a2, a3, b0, b1);
    c[0] = h2u(__hadd2(u2h(c[0]), __floats2half2_rn(d[0], d[1])));
    c[1] = h2u(__hadd2(u2h(c[1]), __floats2half2_rn(d[2], d[3])));
}

__device__ __forceinline__ void mma_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t bc0[8], bc1[8], ar[8], br[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        bc0[j] = __shfl_sync(0xffffffffu, b0, 8 * t + j);
        bc1[j] = __shfl_sync(0xffffffffu, b0, 8 * t + 4 + j);
        bc0[4 + j] = __shfl_sync(0xffffffffu, b1, 8 * t + j);
        bc1[4 + j] = __shfl_sync(0xffffffffu, b1, 8 * t + 4 + j);
        ar[j] = __shfl_sync(0xffffffffu, a0, 4 * g + j);
        br[j] = __shfl_sync(0xffffffffu, a1, 4 * g + j);
        ar[4 + j] = __shfl_sync(0xffffffffu, a2, 4 * g + j);
        br[4 + j] = __shfl_sync(0xffffffffu, a3, 4 * g + j);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        c[0] = __dp4a(int(ar[j]), int(bc0[j]), c[0]);
        c[1] = __dp4a(int(ar[j]), int(bc1[j]), c[1]);
        c[2] = __dp4a(int(br[j]), int(bc0[j]), c[2]);
        c[3] = __dp4a(int(br[j]), int(bc1[j]), c[3]);
    }
}

// ldmatrix semantics with native LDS reads: lane l, register i holds matrix i's elements
// (row = l/4, cols = 2*(l%4), +1). The row addresses come from the lanes the ISA contract names;
// they are broadcast with a per-lane shuffle and read from LDS.
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void * p) {
    const int lane = threadIdx.x & 31;
    const unsigned long long pa = reinterpret_cast<unsigned long long>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned long long rowp =
            __shfl_sync(0xffffffffu, pa, 8 * i + (lane >> 2));
        r[i] = *reinterpret_cast<const uint32_t *>(uintptr_t(rowp) + 4 * (lane & 3));
    }
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void * p) {
    const int lane = threadIdx.x & 31;
    const unsigned long long pa = reinterpret_cast<unsigned long long>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned long long r0 = __shfl_sync(0xffffffffu, pa, 8 * i + 2 * (lane & 3));
        const unsigned long long r1 = __shfl_sync(0xffffffffu, pa, 8 * i + 2 * (lane & 3) + 1);
        const uint32_t lo = *reinterpret_cast<const uint16_t *>(uintptr_t(r0) + 2 * (lane >> 2));
        const uint32_t hi = *reinterpret_cast<const uint16_t *>(uintptr_t(r1) + 2 * (lane >> 2));
        r[i] = lo | (hi << 16);
    }
}
#else
__device__ __forceinline__ void mma_f16(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
#if defined(TQLLM_WITH_HIP)
    return h2u(__hmul2(u2h(a), u2h(b)));
#else
    uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
#endif
}

__device__ __forceinline__ uint32_t hadd2u(uint32_t a, uint32_t b) {
#if defined(TQLLM_WITH_HIP)
    return h2u(__hadd2(u2h(a), u2h(b)));
#else
    uint32_t r;
    asm("add.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
#endif
}

__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t b, uint32_t c) {
#if defined(TQLLM_WITH_HIP)
    return h2u(__hfma2(u2h(a), u2h(b), u2h(c)));
#else
    uint32_t r;
    asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
#endif
}

__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

// cp.async has no RDNA3 analogue with the same lifetime rules; the staging loops always pair it with
// a block barrier, so a direct vector copy keeps the ordering and the double buffers work unchanged.
__device__ __forceinline__ void cp_async16(void * dst, const void * src) {
#if defined(TQLLM_WITH_HIP)
    *reinterpret_cast<uint4 *>(dst) = __ldg(reinterpret_cast<const uint4 *>(src));
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))),
                 "l"(src));
#endif
}
__device__ __forceinline__ void cp_async4(void * dst, const void * src) {
#if defined(TQLLM_WITH_HIP)
    *reinterpret_cast<uint32_t *>(dst) = __ldg(reinterpret_cast<const uint32_t *>(src));
#else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))),
                 "l"(src));
#endif
}
__device__ __forceinline__ void cp_commit() {
#if !defined(TQLLM_WITH_HIP)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_wait() {
#if !defined(TQLLM_WITH_HIP)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

#if !defined(TQLLM_WITH_HIP)
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
#endif

// Two-part (tree) verify attention: a query at position qpos takes keys [0, split) from the cache kernel and [split, qpos]
// from the in-block kernel. split <= qpos - TREE_MAX_DEPTH keeps it at or below the block start for any path of up to
// 16 nodes, and depends only on qpos.
constexpr int TREE_MAX_DEPTH = 15;
__host__ __device__ inline int tq_split_point(int qpos) {
    return qpos >= TREE_MAX_DEPTH ? (qpos - TREE_MAX_DEPTH) / 32 * 32 : 0;
}

}  // namespace tqllm_dev
