// dispatch.cu: the ONLY place that chooses an X3 kernel by n / shape. Everything else calls a
// Codec-typed entry point. Ported from llama-paw-k25 ggml/src/ggml-cuda/paw-x3.cu host side
// (x3v_try_launch / launch_sq): A and C are f32 (a_f32 = 1, c_fp32 = true) so results bit-match
// the fork; the engine converts around this layer.
#include "codec_mul1.cuh"
#include "x3_record.hpp"
#include "codec_testscalar.cuh"
#include "gemv_sq.cu"
#include "gemv_x3v.cu"
#include "gemv_window.cu"
#include "reconstruct.cu"
#include "x3.cuh"
#include "debug.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tqllm {
namespace x3 {

namespace {

// Default: every n (1..8) runs the x3v gemv with MMODE 1 so column j is bit-identical across n
// (the engine's verify uses the batched path; decode must match it token-for-token). TQLLM_DEBUG=x3sq
// (off by default) forces the faster but int8-activation sq path at n == 1 for A/B measurement.
bool x3_use_x3v() {
    static const bool v = tqllm::debug_on("x3v") || tqllm::debug_on("x3sq");
    return v;
}

bool x3_use_sq() {
    static const bool v = tqllm::debug_on("x3sq");
    return v;
}

int dispatch_num_sms() {
    static const int n = [] { int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v > 0 ? v : 82; }();
    return n;
}

template <class Codec, int MMODE, int CFG>
void x3v_launch(const float* A, const uint16_t* B, float* C, int size_m, int size_k, int size_n,
                const half* suh, half* a_had, const half* svh, cudaStream_t stream) {
    auto fn = x3v_gemv_kernel<Codec, true, MMODE, CFG>;
    const int block_dim = CFG == 0 ? 512 : 256;
    const int cols = CFG == 0 ? 32 : 64;
    int maxb = 1;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, block_dim, 0) != cudaSuccess) maxb = 1;
    int grid = std::min(size_n / cols, maxb * dispatch_num_sms());
    if (grid < 1) grid = 1;

    int* locks = nullptr;
    const int a_f32 = 1;
    void* args[] = {(void*) &A, (void*) &B, (void*) &C, (void*) &size_m, (void*) &size_k, (void*) &size_n,
                    (void*) &locks, (void*) &suh, (void*) &a_had, (void*) &svh, (void*) &a_f32};
    cudaError_t err = cudaLaunchCooperativeKernel((const void*) fn, dim3(grid), dim3(block_dim), args, 0, stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("x3 x3v cooperative launch: ") + cudaGetErrorString(err));
}

// CFG choice: narrow fits attention sizes, wide covers large-n FFN. Phase 5 tunes only this.
inline int x3v_cfg(int size_m, int size_k, int size_n, int bits) {
    (void) size_k;
    (void) bits;
    if (size_m > X3V_MAX_M) return -1;
    if (size_n % 128) return -1;
    return size_n <= 8192 ? 0 : 1;
}

template <class Codec>
void codec_gemv(const float* A, const uint16_t* B, float* C, int size_m, int size_k, int size_n,
                const half* suh, half* a_had, const half* svh, int* locks, cudaStream_t stream) {
    if (size_m == 1) {
        // Codecs with no x3v gemv (K==1) keep sq; others use x3v unless TQLLM_DEBUG=x3sq.
        if constexpr (Codec::supports_sq)
            if (x3_use_sq() || !Codec::supports_gemv_x3v) { launch_sq<Codec, true>(A, B, C, size_k, size_n, suh, svh, locks, stream); return; }
    }
    if constexpr (!Codec::supports_gemv_x3v) {
        throw std::runtime_error("x3: this codec has no gemv for a batch > 1");
    } else {
        const int cfg = x3v_cfg(size_m, size_k, size_n, Codec::bits);
        if (cfg < 0) throw std::runtime_error("x3: no gemv config for this shape");
        // MMODE 1 for every n (1 included): column j is then bit-identical across n, matching verify.
        if (cfg == 0) x3v_launch<Codec, 1, 0>(A, B, C, size_m, size_k, size_n, suh, a_had, svh, stream);
        else          x3v_launch<Codec, 1, 1>(A, B, C, size_m, size_k, size_n, suh, a_had, svh, stream);
    }
}

// reconstruct emits W in the fork's [in][out] layout (canonical trellis order: n=in=cols, m=out=rows).
template <class Codec>
void codec_dequant(half* W, const uint16_t* T, const half* suh, const half* svh, int rows, int cols,
                   cudaStream_t stream) {
    reconstruct_had<Codec>(W, T, suh, svh, cols, rows, stream);
}

template <class Codec>
size_t codec_ws_bytes(int size_k, int size_n) {
    if constexpr (Codec::supports_sq) return sq_workspace_bytes(size_k, size_n);
    else { (void) size_k; (void) size_n; return 0; }
}

template <class Codec>
constexpr auto codec_window() -> void (*)(const XwList&, const half*, __nv_bfloat16*, int, cudaStream_t) {
    if constexpr (Codec::supports_gemv_x3v) return &xw_matvec<Codec>;
    else return nullptr;
}

template <int KH>
constexpr X3Codec mul1_row(const char* name) {
    return {X3_CODEC_MUL1, 8 * KH, name, true, Mul1Codec<KH>::supports_gemv_x3v,
            &codec_gemv<Mul1Codec<KH>>, &codec_dequant<Mul1Codec<KH>>, &codec_ws_bytes<Mul1Codec<KH>>,
            codec_window<Mul1Codec<KH>>()};
}

const X3Codec kTable[] = {
    mul1_row<2>("mul1-K1.0"),  mul1_row<3>("mul1-K1.5"), mul1_row<4>("mul1-K2.0"),
    mul1_row<5>("mul1-K2.5"),  mul1_row<6>("mul1-K3.0"), mul1_row<7>("mul1-K3.5"),
    mul1_row<8>("mul1-K4.0"),
    {2, 128, "testscalar-K8.0", false, true,
     &codec_gemv<TestScalarCodec>, &codec_dequant<TestScalarCodec>, &codec_ws_bytes<TestScalarCodec>, nullptr},
};

}  // namespace

bool x3_window_enabled() { return !x3_use_x3v(); }

const X3Codec* x3_codec_table(int& count) {
    count = int(sizeof(kTable) / sizeof(kTable[0]));
    return kTable;
}

__global__ void x3_cast_h_f32_kernel(const half* __restrict__ src, float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}
__global__ void x3_cast_f32_bf16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

void x3_cast_h_f32(const half* src, float* dst, int n, cudaStream_t stream) {
    if (n > 0) x3_cast_h_f32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}
void x3_cast_f32_bf16(const float* src, __nv_bfloat16* dst, int n, cudaStream_t stream) {
    if (n > 0) x3_cast_f32_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

__global__ void x3_copy_strided_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                       int part_rows, int ld, int off, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // over n * part_rows
    if (i >= total) return;
    const int t = i / part_rows, c = i % part_rows;
    dst[size_t(t) * ld + off + c] = __float2bfloat16(src[i]);
}

void x3_copy_f32_bf16_strided(const float* src, __nv_bfloat16* dst, int part_rows, int ld, int off, int n,
                              cudaStream_t stream) {
    const int total = n * part_rows;
    if (total > 0) x3_copy_strided_kernel<<<(total + 255) / 256, 256, 0, stream>>>(src, dst, part_rows, ld, off, total);
}

// dst[r*ld + c] = src[c*rows + r]: turn reconstruct's [in][out] into the engine's [rows][cols].
__global__ void x3_transpose_h_kernel(const half* __restrict__ src, half* __restrict__ dst, int rows, int ld,
                                      int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // over rows * ld
    if (i >= total) return;
    const int r = i / ld, c = i % ld;
    dst[size_t(r) * ld + c] = src[size_t(c) * rows + r];
}

void x3_transpose_h(const half* src, half* dst, int rows, int cols, int ld, cudaStream_t stream) {
    const int total = rows * ld;
    if (total > 0) x3_transpose_h_kernel<<<(total + 255) / 256, 256, 0, stream>>>(src, dst, rows, ld, total);
}

}  // namespace x3
}  // namespace tqllm
