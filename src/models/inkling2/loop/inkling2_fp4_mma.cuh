#pragma once
// Minimal Blackwell FP4 MMA primitives for the Inkling expert kernel. Same tile layout and PTX as the vendored
// ggml_mmq/mma.cuh, without its ggml host dependencies (inkling2-check links only CUDA / cuBLAS).
#include <cuda_runtime.h>
#include <cstdint>

// the block-scaled FP4 MMA exists only in the arch-specific targets (sm_120a / sm_121a, or the sm_12x family)
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200)
#define INKLING2_FP4_MMA 1
#endif

namespace inkling2_fp4_mma {

template <int I_, int J_, typename T> struct tile;

template <typename T>
struct tile<16, 8, T> {
    static constexpr int I = 16, J = 8, ne = 4;
    T x[ne] = {};
    static __device__ __forceinline__ int get_i(int l) { return ((l / 2) * 8) + (threadIdx.x / 4); }
    static __device__ __forceinline__ int get_j(int l) { return ((threadIdx.x % 4) * 2) + (l % 2); }
};

template <typename T>
struct tile<8, 8, T> {
    static constexpr int I = 8, J = 8, ne = 2;
    T x[ne] = {};
    static __device__ __forceinline__ int get_i(int l) { return threadIdx.x / 4; }
    static __device__ __forceinline__ int get_j(int l) { return l * 4 + (threadIdx.x % 4); }
};

template <typename T>
static __device__ __forceinline__ void load_ldmatrix(tile<16, 8, T> & t, const T * xs0, int stride) {
    int * xi = reinterpret_cast<int *>(t.x);
    const int * xs = reinterpret_cast<const int *>(xs0) + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
                 : "l"(xs));
}

template <typename T>
static __device__ __forceinline__ void load_generic(tile<8, 8, T> & t, const T * xs0, int stride) {
#pragma unroll
    for (int l = 0; l < t.ne; ++l) t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
}

static __device__ __forceinline__ void mma_nvfp4(tile<16, 8, float> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B,
                                                uint32_t a_scale, uint32_t b_scale) {
#if defined(INKLING2_FP4_MMA)
    const int * Axi = A.x;
    const int * Bxi = B.x;
    float * Dxi = D.x;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]),
          "r"(a_scale), "r"(b_scale));
#else
    (void) D; (void) A; (void) B; (void) a_scale; (void) b_scale;
#endif
}

}  // namespace inkling2_fp4_mma
