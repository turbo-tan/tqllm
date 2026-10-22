#pragma once
// FP8 (E4M3) prefill GEMM on GPUs with FP8 tensor cores (sm_89+: RTX 40, RTX 50 / 5060 Ti, GB10).
// Y[m][r_off + n] = sum_k W[n][k] * X[m][k], W and X FP16 row-major (k contiguous), Y FP16 with row stride ldo.
// Each W row and each X row (token) is quantised to E4M3 with its own scale; cuBLASLt runs the FP8 GEMM with unit
// scales (GB10's cuBLASLt has no per-row / per-token FP8 scale mode), and the two scale vectors are applied after it
// (they factor out of every dot product): either by `run` (in place) or, with `gemm`, by the caller's next pass
// (fp8_h_to_bf16_scaled), which saves a full read and write of the output.
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>

namespace tqllm_cuda {

// out[r][c] = E4M3(in[r][c] / scale[r]), scale[r] = max_c |in[r][c]| / 448 (1 for an all-zero row)
void fp8_quant_rows(const __half * in, int rows, int cols, int ld_in, uint8_t * out, float * scale, cudaStream_t s);
// y[m * ldo + n] *= sw[n] * sx[m] * mul   for n < nrows, m < ncols
void fp8_scale_out(__half * y, int ldo, int nrows, int ncols, const float * sw, const float * sx, float mul, cudaStream_t s);
// out[m * ldo + n] = bf16(c[m * ldo + n] * sw[n] * sx[m] * mul), n < ldo, m < ntok
void fp8_h_to_bf16_scaled(const __half * c, __nv_bfloat16 * out, int ntok, int ldo, const float * sw, const float * sx,
                          float mul, cudaStream_t s);

class Fp8Gemm {
public:
    static constexpr float kMul = 448.f * 448.f;  // the GEMM runs with alpha 1 / 448^2 (FP16 output cannot overflow)
    static bool supported();  // compute capability >= 8.9
    static bool default_on();  // off everywhere: costs KLD ~0.01 vs FP16 prefill (GB10, 2026-10-09); --prefill-fp8 on opts in
    // buffers sized for W up to max_w elements, X up to max_x elements, max_rows W rows, max_tok tokens;
    // resident_bytes > 0: keep up to that many bytes of quantised weights (keyed by the weight pointer)
    void init(size_t max_w, size_t max_x, int max_rows, int max_tok, size_t resident_bytes = 0);
    ~Fp8Gemm();
    // one call: W [rows][k] FP16, X [tok][k] FP16 (row stride ldx), Y [tok][ldo] FP16, scaled in place
    void run(const __half * w, int rows, int k, const __half * x, int ldx, int tok, __half * y, int ldo, cudaStream_t s);

    // split form: quantise X once for several weights; `cached(key)`: W needs no FP16 copy (resident);
    // gemm writes Y unscaled and the row scales to sw_out [rows]; apply them with fp8_h_to_bf16_scaled / fp8_scale_out
    void quant_x(const __half * x, int ldx, int tok, int k, cudaStream_t s);
    bool cached(const void * key) const { return res_.count(key) != 0; }
    void gemm(const void * key, const __half * w, int rows, int k, int tok, __half * y, int ldo, float * sw_out, cudaStream_t s);
    const float * sx() const { return sx_; }
    size_t resident_used() const { return res_used_; }

private:
    void matmul(const uint8_t * w8, int rows, int k, int tok, __half * y, int ldo, cudaStream_t s);
    struct Resident { uint8_t * w8; float * sw; };
    cublasLtHandle_t lt_ = nullptr;
    uint8_t * w8_ = nullptr, * x8_ = nullptr;
    float * sw_ = nullptr, * sx_ = nullptr;
    void * ws_ = nullptr;
    size_t ws_bytes_ = size_t(64) << 20;
    size_t res_budget_ = 0, res_used_ = 0;
    std::map<const void *, Resident> res_;
    std::map<std::tuple<int, int, int, int>, cublasLtMatmulAlgo_t> algos_;
};

}  // namespace tqllm_cuda
