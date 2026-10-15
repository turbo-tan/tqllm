#pragma once
// ptx.cuh: the raw PTX / ldmatrix / funnel-shift primitives the X3 kernels build on. Ported
// verbatim from llama-paw-k25 ggml/src/ggml-cuda/paw-x3.cu (source line comments kept) so the
// integer windows decode bit-identically to the fork.
#include <cuda_runtime.h>
#include <cstdint>

namespace tqllm {
namespace x3 {

__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c) {
    int d;
    asm ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// exl3_dq.cuh fshift: plain 64-bit merge+shift (shift may exceed 32, up to bits*7+16;
// __funnelshift_r would clamp at 32 and corrupt K=3 windows)
__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift) {
    uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (merged >> shift);
}

#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr) {
    const int bytes = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}

__device__ inline void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

}  // namespace x3
}  // namespace tqllm
