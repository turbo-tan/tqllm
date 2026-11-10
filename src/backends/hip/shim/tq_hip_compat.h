#pragma once
// tqLLM HIP shim: common part every shim header re-includes, so device intrinsics and the API spellings
// are visible no matter which CUDA header the file included first (kernel files list only what they need).
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// wave size as a preprocessor constant: warpSize is a __device__ variable that the host pass of a
// HIP TU does not see, and this header's helpers are parsed in both passes. AMDGPU defines the
// macro per target (32 on gfx10/11, 64 on CDNA).
// --- device intrinsics HIP spells differently -----------------------------------------------
// HIP's own warp-sync set requires 64-bit masks; tqLLM's kernels are CUDA wave32 (32-bit masks). With
// HIP_DISABLE_WARP_SYNC_BUILTINS=1 (set in CMake) the builtins are off and this header provides the 32-bit
// set over the plain __shfl_* / __ballot intrinsics: one per wave, mask used only as a ballot filter.
#if defined(__AMDGCN_WAVEFRONT_SIZE)
#define TQ_WAVE __AMDGCN_WAVEFRONT_SIZE
#else
#define TQ_WAVE 32
#endif

__device__ inline unsigned long long tq_sync_or_mask(unsigned mask) {
    return mask ? static_cast<unsigned long long>(mask) : ~0ull;
}
template <typename T>
__device__ inline T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = TQ_WAVE) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __shfl_xor(var, laneMask, width);
#else
    return var;  // host pass: parsed only, device code above is what runs
#endif
}
template <typename T>
__device__ inline T __shfl_down_sync(unsigned mask, T var, int delta, int width = TQ_WAVE) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __shfl_down(var, delta, width);
#else
    return var;
#endif
}
template <typename T>
__device__ inline T __shfl_up_sync(unsigned mask, T var, int delta, int width = TQ_WAVE) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __shfl_up(var, delta, width);
#else
    return var;
#endif
}
template <typename T>
__device__ inline T __shfl_sync(unsigned mask, T var, int srcLane, int width = TQ_WAVE) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __shfl(var, srcLane, width);
#else
    return var;
#endif
}
__device__ inline unsigned long long __ballot_sync(unsigned mask, int predicate) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __ballot(predicate) & tq_sync_or_mask(mask);
#else
    return 0;  // host pass only parses this; device code runs the native ballot above
#endif
}
__device__ inline int __all_sync(unsigned mask, int predicate) {
    return __ballot_sync(mask, predicate) == tq_sync_or_mask(mask);
}
__device__ inline int __any_sync(unsigned mask, int predicate) { return __ballot_sync(mask, predicate) != 0; }
__device__ inline void __syncwarp() {}  // wave-level barrier is implicit on rdna warps of 32
__device__ inline void __syncwarp(unsigned mask) {}

// generic -> shared address space: LDS lives at the low 32 bits (ds_read/ds_write address width), and any
// two shared pointers share the base, so the low 32 bits compare (monotonic) exactly like the CUDA idiom.
__device__ inline unsigned __cvta_generic_to_shared(const void * p) {
    return static_cast<unsigned>(reinterpret_cast<uintptr_t>(p));
}

// __vsub4: four 8-bit lanes, a - b per lane, wrap (no saturation, no cross-lane borrow)
__device__ __forceinline__ unsigned __vsub4(unsigned a, unsigned b) {
    unsigned lo = ((a & 0x00FF00FFu) + 0x01000100u - (b & 0x00FF00FFu)) & 0x00FF00FFu;
    unsigned hi = (((a >> 8) & 0x00FF00FFu) + 0x01000100u - ((b >> 8) & 0x00FF00FFu)) & 0x00FF00FFu;
    return lo | (hi << 8);
}
// __vadd4: four 8-bit lanes, a + b per lane, wrap (carries die inside their lane)
__device__ __forceinline__ unsigned __vadd4(unsigned a, unsigned b) {
    unsigned lo = ((a & 0x00FF00FFu) + (b & 0x00FF00FFu)) & 0x00FF00FFu;
    unsigned hi = (((a >> 8) & 0x00FF00FFu) + ((b >> 8) & 0x00FF00FFu)) & 0x00FF00FFu;
    return lo | (hi << 8);
}

// __reduce_add_sync / __reduce_or_sync: CUDA sm_80 redux.sync folded with xor shuffles. A partial
// mask names the lanes that take part (the TQ2 KV writer uses a 16-lane half): skip a partner lane
// outside it. Full-wave masks take the branch-free path.
template <typename T>
__device__ inline T __reduce_add_sync(unsigned mask, T v) {
#if defined(__HIP_DEVICE_COMPILE__)
    const int lane = int(threadIdx.x) & (TQ_WAVE - 1);
    const bool full = mask == 0xffffffffu;
#pragma unroll
    for (int o = TQ_WAVE / 2; o; o >>= 1) {
        const T x = __shfl_xor(v, o);
        if (full || ((mask >> (lane ^ o)) & 1u)) v += x;
    }
#endif
    return v;
}
__device__ inline unsigned __reduce_or_sync(unsigned mask, unsigned v) {
#if defined(__HIP_DEVICE_COMPILE__)
    const int lane = int(threadIdx.x) & (TQ_WAVE - 1);
    const bool full = mask == 0xffffffffu;
#pragma unroll
    for (int o = TQ_WAVE / 2; o; o >>= 1) {
        const unsigned x = __shfl_xor(v, o);
        if (full || ((mask >> (lane ^ o)) & 1u)) v |= x;
    }
#endif
    return v;
}

// __dp4a: dot product of four 8-bit lanes with accumulate. On RDNA3 this is the hardware v_dot4_i32_i8 /
// v_dot4_u32_u8 instruction through amd_mixed_dot (--ockl_sdot4 / --ockl_udot4); verified lane-exact on gfx1100.
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
    char4 av; __builtin_memcpy(&av, &a, 4);
    char4 bv; __builtin_memcpy(&bv, &b, 4);
    return amd_mixed_dot(av, bv, c, false);
#else
    int r = c;  // host pass: same arithmetic, so the body parses and any host test stays correct
    for (int i = 0; i < 4; ++i) r += int(int8_t((a >> (8 * i)) & 0xFF)) * int(int8_t((b >> (8 * i)) & 0xFF));
    return r;
#endif
}
__device__ __forceinline__ unsigned __dp4a(unsigned a, unsigned b, unsigned c) {
#if defined(__HIP_DEVICE_COMPILE__)
    uchar4 av; __builtin_memcpy(&av, &a, 4);
    uchar4 bv; __builtin_memcpy(&bv, &b, 4);
    return amd_mixed_dot(av, bv, c, false);
#else
    unsigned r = c;
    for (int i = 0; i < 4; ++i) r += unsigned((a >> (8 * i)) & 0xFF) * unsigned((b >> (8 * i)) & 0xFF);
    return r;
#endif
}

// cudaFuncSetAttribute takes a kernel symbol: convert it for HIP (which takes const void*)
template <typename Fn>
__host__ inline hipError_t cudaFuncSetAttribute(Fn f, hipFuncAttribute attr, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void *>(f), attr, value);
}
