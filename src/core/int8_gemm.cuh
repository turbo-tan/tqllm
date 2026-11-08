#pragma once
// INT8 prefill GEMM for GPUs without FP8 tensor cores (Ampere, sm_80 / sm_86: RTX 30 series, A100): int8 tensor
// throughput is twice FP16 (RTX 3090 cuBLASLt IMMA 226-291 TOPS vs 135-138 TFLOPS FP16 at 27B shapes).
// Y[m][r_off + n] = sum_k W[n][k] * X[m][k]: W rows and X rows (tokens) quantised to int8 with their own scales
// (amax / 127), cuBLASLt IMMA int8 x int8 -> int32 (Ampere writes int32 only), scales applied when the int32 result
// is converted (int8_i32_to_bf16_scaled / int8_i32_to_h_scaled). Real TQ3 rows: int8 per-row error 1.06 % vs E4M3 2.65 %.
#if !defined(TQLLM_WITH_HIP)
#include <cublasLt.h>  // cuBLASLt is NVIDIA-only: the AMD build uses the BF16 hipBLAS prefill path
#endif
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>

namespace tqllm_cuda {

#if defined(TQLLM_WITH_HIP)
// AMD/HIP: the INT8 prefill GEMM is a cuBLASLt IMMA feature (sm_80+) and is not part of this build; BF16 prefill
// (hipBLAS) runs instead, chosen by supported() == false. These stubs exist to keep the shared forward code compiling,
// so they must FAIL LOUDLY if a call site ever reaches one: a silent no-op here would return unscaled or unwritten
// activations, i.e. wrong numbers with no error (AGENTS.md: "fail with a clear message, never fall back silently").
namespace int8_hip_detail {
[[noreturn]] inline void unavailable() {
    throw std::runtime_error("INT8 prefill GEMM is not built on the AMD (HIP) backend: it needs cuBLASLt IMMA (sm_80+). "
                             "The engine asks Int8Gemm::supported() and uses BF16 prefill (hipBLAS) instead.");
}
}  // namespace int8_hip_detail
inline void int8_quant_rows(const __half *, int, int, int, int8_t *, float *, cudaStream_t) { int8_hip_detail::unavailable(); }
inline void int8_i32_to_bf16_scaled(const int32_t *, __nv_bfloat16 *, int, int, const float *, const float *, cudaStream_t) { int8_hip_detail::unavailable(); }
inline void int8_i32_to_h_scaled(const int32_t *, __half *, int, int, const float *, const float *, cudaStream_t) { int8_hip_detail::unavailable(); }
class Int8Gemm {
public:
    static bool supported() { return false; }
    static bool default_on() { return false; }
    void init(size_t, size_t, int, int) {}
    ~Int8Gemm() = default;
    void quant_x(const __half *, int, int, int, cudaStream_t) { int8_hip_detail::unavailable(); }
    void gemm(const __half *, int, int, int, int32_t *, int, float *, cudaStream_t) { int8_hip_detail::unavailable(); }
    void gemm_prequant(int, int, int, int32_t *, int, cudaStream_t) { int8_hip_detail::unavailable(); }
    int8_t * w8() const { return nullptr; }
    const float * sx() const { return nullptr; }
};
#else

// out[r][c] = round(in[r][c] / scale[r]), scale[r] = max_c |in[r][c]| / 127 (1 for an all-zero row)
void int8_quant_rows(const __half * in, int rows, int cols, int ld_in, int8_t * out, float * scale, cudaStream_t s);
// out[m * ldo + n] = bf16(c[m * ldo + n] * sw[n] * sx[m]) / fp16 (same), n < ldo, m < ntok
void int8_i32_to_bf16_scaled(const int32_t * c, __nv_bfloat16 * out, int ntok, int ldo, const float * sw, const float * sx, cudaStream_t s);
void int8_i32_to_h_scaled(const int32_t * c, __half * out, int ntok, int ldo, const float * sw, const float * sx, cudaStream_t s);

class Int8Gemm {
public:
    static bool supported();   // int8 tensor cores: compute capability >= 8.0
    static bool default_on();  // Ampere and RTX 50 measured, RTX 40 by analogy; GB10 uses FP8
    void init(size_t max_w, size_t max_x, int max_rows, int max_tok);
    ~Int8Gemm();
    // X quantised once per GEMM group; W [rows][k] FP16; Y int32 [tok][ldo] (unscaled), row scales to sw_out [rows]
    void quant_x(const __half * x, int ldx, int tok, int k, cudaStream_t s);
    void gemm(const __half * w, int rows, int k, int tok, int32_t * y, int ldo, float * sw_out, cudaStream_t s);
    // Weights already quantised into w8_ (a format's straight-to-int8 reconstruct) with the row scales in place:
    // multiply only, no quantise pass.
    void gemm_prequant(int rows, int k, int tok, int32_t * y, int ldo, cudaStream_t s);
    int8_t * w8() const { return w8_; }
    const float * sx() const { return sx_; }

private:
    cublasLtHandle_t lt_ = nullptr;
    int8_t * w8_ = nullptr, * x8_ = nullptr;
    float * sw_ = nullptr, * sx_ = nullptr;
    void * ws_ = nullptr;
    void matmul(int rows, int k, int tok, int32_t * y, int ldo, cudaStream_t s);
    size_t ws_bytes_ = size_t(32) << 20;
    std::map<std::tuple<int, int, int, int>, cublasLtMatmulAlgo_t> algos_;
};

#endif

}  // namespace tqllm_cuda
