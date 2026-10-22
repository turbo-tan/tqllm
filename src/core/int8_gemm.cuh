#pragma once
// INT8 prefill GEMM for GPUs without FP8 tensor cores (Ampere, sm_80 / sm_86: RTX 30 series, A100): int8 tensor
// throughput is twice FP16 (RTX 3090 cuBLASLt IMMA 226-291 TOPS vs 135-138 TFLOPS FP16 at 27B shapes).
// Y[m][r_off + n] = sum_k W[n][k] * X[m][k]: W rows and X rows (tokens) quantised to int8 with their own scales
// (amax / 127), cuBLASLt IMMA int8 x int8 -> int32 (Ampere writes int32 only), scales applied when the int32 result
// is converted (int8_i32_to_bf16_scaled / int8_i32_to_h_scaled). Real TQ3 rows: int8 per-row error 1.06 % vs E4M3 2.65 %.
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>

namespace tqllm_cuda {

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
    const float * sx() const { return sx_; }

private:
    cublasLtHandle_t lt_ = nullptr;
    int8_t * w8_ = nullptr, * x8_ = nullptr;
    float * sw_ = nullptr, * sx_ = nullptr;
    void * ws_ = nullptr;
    size_t ws_bytes_ = size_t(32) << 20;
    std::map<std::tuple<int, int, int, int>, cublasLtMatmulAlgo_t> algos_;
};

}  // namespace tqllm_cuda
