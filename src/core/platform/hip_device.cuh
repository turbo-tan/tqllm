#pragma once
// AMD (ROCm / HIP, wave32 RDNA3 / RDNA4) device primitives: native AMD instructions behind the NVIDIA fragment contract.
// Included by src/core/common.cuh only (the one place a backend is selected for device code). Defines the capability macros
// (TQLLM_DEV_*) and every primitive in tqllm_dev that differs by hardware; a new backend is a new src/core/platform/<name>_device.cuh
// with the same names, and no shared file changes beyond common.cuh's one include line.
#define TQLLM_DEV_PTX 0         // NVIDIA PTX inline assembly (mma.sync, cp.async, ldmatrix, redux)
#define TQLLM_DEV_CP_ASYNC 0    // asynchronous global -> shared copies: tiles can be double-buffered
#define TQLLM_DEV_LDS_64K 1     // at most 64 KB of dynamic shared memory per block: smaller row tiles, one buffer

namespace tqllm_dev {

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
    // one rounding, as the PTX f16 MMA: the f16 accumulator joins the f32 sum, then rounds once (two roundings drifted
    // long-context attention away from the CUDA result)
    const float2 c0 = __half22float2(u2h(c[0])), c1 = __half22float2(u2h(c[1]));
    float d[4] = {c0.x, c0.y, c1.x, c1.y};
    mma_f16(d, a0, a1, a2, a3, b0, b1);
    c[0] = h2u(__floats2half2_rn(d[0], d[1]));
    c[1] = h2u(__floats2half2_rn(d[2], d[3]));
}

// m16n8k16 s8: the first k16 half of mma_s8's fragment map (A rows g / g + 8 in a0 / a1, B column n in lanes 4n..4n+3)
__device__ __forceinline__ void mma_s8_k16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int ar = int(__shfl_sync(0xffffffffu, a0, 4 * g + j)), br = int(__shfl_sync(0xffffffffu, a1, 4 * g + j));
        const int b_0 = int(__shfl_sync(0xffffffffu, b0, 8 * t + j)), b_1 = int(__shfl_sync(0xffffffffu, b0, 8 * t + 4 + j));
        c[0] = __dp4a(ar, b_0, c[0]);
        c[1] = __dp4a(ar, b_1, c[1]);
        c[2] = __dp4a(br, b_0, c[2]);
        c[3] = __dp4a(br, b_1, c[3]);
    }
}
// OR of v over each 16-lane half of the warp (every lane of the half gets it)
__device__ __forceinline__ uint32_t or_half16(uint32_t v) {
#pragma unroll
    for (int o = 8; o; o >>= 1) v |= __shfl_xor_sync(0xffffffffu, v, o, 16);
    return v;
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

__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
    return h2u(__hmul2(u2h(a), u2h(b)));
}

__device__ __forceinline__ uint32_t hadd2u(uint32_t a, uint32_t b) {
    return h2u(__hadd2(u2h(a), u2h(b)));
}

__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t b, uint32_t c) {
    return h2u(__hfma2(u2h(a), u2h(b), u2h(c)));
}

// cp.async has no RDNA3 analogue with the same lifetime rules; the staging loops always pair it with
// a block barrier, so a direct vector copy keeps the ordering and the double buffers work unchanged.
__device__ __forceinline__ void cp_async16(void * dst, const void * src) {
    *reinterpret_cast<uint4 *>(dst) = __ldg(reinterpret_cast<const uint4 *>(src));
}
// cp_async16 with a predicate: false writes 16 zero bytes (cp.async src-size 0), so masked tails never keep old bytes
__device__ __forceinline__ void cp_async16_zfill(void * dst, const void * src, bool pred) {
    *reinterpret_cast<uint4 *>(dst) = pred ? __ldg(reinterpret_cast<const uint4 *>(src)) : make_uint4(0u, 0u, 0u, 0u);
}
__device__ __forceinline__ void cp_async4(void * dst, const void * src) {
    *reinterpret_cast<uint32_t *>(dst) = __ldg(reinterpret_cast<const uint32_t *>(src));
}
__device__ __forceinline__ void cp_commit() {
}
template <int N>
__device__ __forceinline__ void cp_wait() {
}

}  // namespace tqllm_dev
