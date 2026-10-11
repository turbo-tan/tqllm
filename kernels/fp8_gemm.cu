#include "dev_alloc.cuh"
#include "fp8_gemm.cuh"

#include <cuda_fp8.h>

#include <stdexcept>
#include <string>

namespace tqllm_cuda {
namespace {
// |W8 . X8| <= 448 * 448 * k; with alpha = 1 / 448^2 the FP16 output holds sum |w / sw| |x / sx| <= k (k <= 32768)
constexpr float kE4M3Max = 448.f;

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("fp8 gemm: ") + what + ": " + cudaGetErrorString(e));
}
void cl(cublasStatus_t e, const char * what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("fp8 gemm: ") + what + ": cublas status " + std::to_string(int(e)));
}

__global__ void __launch_bounds__(256) quant_rows_k(const __half * in, int cols, int ld_in, uint8_t * out, float * scale) {
    const int r = blockIdx.x;
    const __half * row = in + size_t(r) * ld_in;
    float m = 0.f;
    for (int c = threadIdx.x; c < cols; c += 256) m = fmaxf(m, fabsf(__half2float(row[c])));
    __shared__ float red[8];
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
        for (int o = 4; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (threadIdx.x == 0) red[0] = m;
    }
    __syncthreads();
    const float sc = red[0] > 0.f ? red[0] / kE4M3Max : 1.f, inv = 1.f / sc;
    if (threadIdx.x == 0) scale[r] = sc;
    uint8_t * o = out + size_t(r) * cols;
    for (int c = threadIdx.x; c < cols; c += 256) {
        const __nv_fp8_e4m3 q(__half2float(row[c]) * inv);
        o[c] = q.__x;
    }
}

__global__ void scale_out_k(__half * y, int ldo, int nrows, int ncols, const float * sw, const float * sx, float mul) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (n >= nrows || m >= ncols) return;
    __half & v = y[size_t(m) * ldo + n];
    v = __float2half(__half2float(v) * sw[n] * sx[m] * mul);
}
__global__ void h_to_bf16_scaled_k(const __half * c, __nv_bfloat16 * out, int ldo, const float * sw, const float * sx, float mul) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (n >= ldo) return;
    const size_t i = size_t(m) * ldo + n;
    out[i] = __float2bfloat16(__half2float(c[i]) * sw[n] * sx[m] * mul);
}
}  // namespace

void fp8_h_to_bf16_scaled(const __half * c, __nv_bfloat16 * out, int ntok, int ldo, const float * sw, const float * sx,
                          float mul, cudaStream_t s) {
    if (ntok <= 0 || ldo <= 0) return;
    h_to_bf16_scaled_k<<<dim3((ldo + 255) / 256, ntok), 256, 0, s>>>(c, out, ldo, sw, sx, mul);
}

void fp8_quant_rows(const __half * in, int rows, int cols, int ld_in, uint8_t * out, float * scale, cudaStream_t s) {
    if (rows <= 0) return;
    quant_rows_k<<<rows, 256, 0, s>>>(in, cols, ld_in, out, scale);
}

void fp8_scale_out(__half * y, int ldo, int nrows, int ncols, const float * sw, const float * sx, float mul, cudaStream_t s) {
    if (nrows <= 0 || ncols <= 0) return;
    scale_out_k<<<dim3((nrows + 255) / 256, ncols), 256, 0, s>>>(y, ldo, nrows, ncols, sw, sx, mul);
}

bool Fp8Gemm::supported() {
    int dev = 0, ma = 0, mi = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
    return ma * 10 + mi >= 89;
}

bool Fp8Gemm::default_on() {
    int dev = 0, ma = 0, mi = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
    // GB10 measured 2026-10-09: +40 % prefill but KLD 0.010-0.014 vs FP16 prefill (TQ_S / TQ_XS, PPL +2.7 %) and a wrong
    // 16K answer - quality first: opt-in everywhere (--prefill-fp8 on) until a finer-scaled FP8 passes that gate
    return (void)ma, (void)mi, false;
}

void Fp8Gemm::init(size_t max_w, size_t max_x, int max_rows, int max_tok, size_t resident_bytes) {
    res_budget_ = resident_bytes;
    cl(cublasLtCreate(&lt_), "create");
    ck(tqllm_cuda::dev_malloc(&w8_, max_w), "w8");
    ck(tqllm_cuda::dev_malloc(&x8_, max_x), "x8");
    ck(tqllm_cuda::dev_malloc(&sw_, size_t(max_rows) * 4), "sw");
    ck(tqllm_cuda::dev_malloc(&sx_, size_t(max_tok) * 4), "sx");
    ck(tqllm_cuda::dev_malloc(&ws_, ws_bytes_), "workspace");
}

Fp8Gemm::~Fp8Gemm() {
    for (auto & [k, r] : res_) {
        tqllm_cuda::dev_free(r.w8);
        tqllm_cuda::dev_free(r.sw);
    }
    for (void * p : {static_cast<void *>(w8_), static_cast<void *>(x8_), static_cast<void *>(sw_), static_cast<void *>(sx_), ws_})
        if (p) tqllm_cuda::dev_free(p);
    if (lt_) cublasLtDestroy(lt_);
}

void Fp8Gemm::matmul(const uint8_t * w8, int rows, int k, int tok, __half * y, int ldo, cudaStream_t s) {
    cublasLtMatmulDesc_t op;
    cl(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cl(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
    cl(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
    cublasLtMatrixLayout_t la, lb, lc;
    cl(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, k, rows, k), "la");
    cl(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, k, tok, k), "lb");
    cl(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, rows, tok, ldo), "lc");
    const auto key = std::make_tuple(rows, k, tok, ldo);
    auto it = algos_.find(key);
    if (it == algos_.end()) {
        cublasLtMatmulPreference_t pref;
        cl(cublasLtMatmulPreferenceCreate(&pref), "pref");
        cl(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes_, sizeof(ws_bytes_)), "pref ws");
        cublasLtMatmulHeuristicResult_t h[8]{};
        int nh = 0;
        cl(cublasLtMatmulAlgoGetHeuristic(lt_, op, la, lb, lc, lc, pref, 8, h, &nh), "heuristic");
        cublasLtMatmulPreferenceDestroy(pref);
        if (!nh) throw std::runtime_error("fp8 gemm: no cuBLASLt algorithm for this shape");
        // the heuristic's first pick is not always the fastest (GB10 27B shapes: up to 7 % for FP8): time each once
        // per shape (the output is overwritten by the real call below)
        const float a0 = 1.f / (kE4M3Max * kE4M3Max), b0 = 0.f;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        int best = 0;
        float best_ms = 1e30f;
        // inside a CUDA graph capture (verify steps) nothing may synchronise: keep the heuristic's first pick for this shape
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s, &cap);
        for (int i = 0; i < nh && cap == cudaStreamCaptureStatusNone; ++i) {
            if (cublasLtMatmul(lt_, op, &a0, w8, la, x8_, lb, &b0, y, lc, y, lc, &h[i].algo, ws_, ws_bytes_, s) != CUBLAS_STATUS_SUCCESS) continue;
            cudaEventRecord(e0, s);
            const bool ok = cublasLtMatmul(lt_, op, &a0, w8, la, x8_, lb, &b0, y, lc, y, lc, &h[i].algo, ws_, ws_bytes_, s) == CUBLAS_STATUS_SUCCESS;
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (ok && ms < best_ms) best_ms = ms, best = i;
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        it = algos_.emplace(key, h[best].algo).first;
    }
    const float alpha = 1.f / (kE4M3Max * kE4M3Max), beta = 0.f;
    const std::string shape = "matmul rows " + std::to_string(rows) + " k " + std::to_string(k) + " tok " + std::to_string(tok) +
                              " ldo " + std::to_string(ldo) + " y%16 " + std::to_string(reinterpret_cast<uintptr_t>(y) % 16);
    cl(cublasLtMatmul(lt_, op, &alpha, w8, la, x8_, lb, &beta, y, lc, y, lc, &it->second, ws_, ws_bytes_, s), shape.c_str());
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
}

void Fp8Gemm::quant_x(const __half * x, int ldx, int tok, int k, cudaStream_t s) { fp8_quant_rows(x, tok, k, ldx, x8_, sx_, s); }

void Fp8Gemm::gemm(const void * key, const __half * w, int rows, int k, int tok, __half * y, int ldo, float * sw_out, cudaStream_t s) {
    if (k % 16) throw std::invalid_argument("fp8 gemm: k must be a multiple of 16");
    auto it = res_.find(key);
    if (it == res_.end()) {
        if (!w) throw std::invalid_argument("fp8 gemm: weights neither resident nor given");
        const size_t bytes = size_t(rows) * k + size_t(rows) * 4;
        if (key && res_used_ + bytes <= res_budget_) {  // quantise once, keep
            Resident r{};
            ck(tqllm_cuda::dev_malloc(&r.w8, size_t(rows) * k), "resident w8");
            ck(tqllm_cuda::dev_malloc(&r.sw, size_t(rows) * 4), "resident sw");
            fp8_quant_rows(w, rows, k, k, r.w8, r.sw, s);
            res_used_ += bytes;
            it = res_.emplace(key, r).first;
        }
    }
    const uint8_t * w8 = w8_;
    const float * sw = sw_;
    if (it != res_.end()) {
        w8 = it->second.w8;
        sw = it->second.sw;
    } else {
        fp8_quant_rows(w, rows, k, k, w8_, sw_, s);
    }
    matmul(w8, rows, k, tok, y, ldo, s);
    if (sw_out) ck(cudaMemcpyAsync(sw_out, sw, size_t(rows) * 4, cudaMemcpyDeviceToDevice, s), "row scales");
}

void Fp8Gemm::run(const __half * w, int rows, int k, const __half * x, int ldx, int tok, __half * y, int ldo, cudaStream_t s) {
    quant_x(x, ldx, tok, k, s);
    gemm(nullptr, w, rows, k, tok, y, ldo, nullptr, s);  // transient: the row scales stay in sw_
    fp8_scale_out(y, ldo, rows, tok, sw_, sx_, kMul, s);
}

}  // namespace tqllm_cuda
