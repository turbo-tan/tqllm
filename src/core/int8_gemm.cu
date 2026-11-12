#include <algorithm>
#include "dev_alloc.cuh"
#include "int8_gemm.cuh"

#include <stdexcept>
#include <type_traits>
#include <string>

namespace tqllm_cuda {
namespace {
void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("int8 gemm: ") + what + ": " + cudaGetErrorString(e));
}
void cl(cublasStatus_t e, const char * what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("int8 gemm: ") + what + ": cublas status " + std::to_string(int(e)));
}

__global__ void __launch_bounds__(256) quant_rows_k(const __half * in, int cols, int ld_in, int8_t * out, float * scale) {
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
    const float sc = red[0] > 0.f ? red[0] / 127.f : 1.f, inv = 1.f / sc;
    if (threadIdx.x == 0) scale[r] = sc;
    int8_t * o = out + size_t(r) * cols;
    for (int c = threadIdx.x; c < cols; c += 256) o[c] = int8_t(__float2int_rn(fminf(fmaxf(__half2float(row[c]) * inv, -127.f), 127.f)));
}

// quant_rows_k with column factors: row r of in * cm (MUL) or / cm (!MUL), quantised with its own amax/127
template <bool MUL>
__global__ void __launch_bounds__(256) quant_rows_cm_k(const __half * in, int cols, int ld_in, int8_t * out, float * scale, const float * cm) {
    const int r = blockIdx.x;
    const __half * row = in + size_t(r) * ld_in;
    auto val = [&](int c) { const float v = __half2float(row[c]); return MUL ? v * cm[c] : v / cm[c]; };
    float m = 0.f;
    for (int c = threadIdx.x; c < cols; c += 256) m = fmaxf(m, fabsf(val(c)));
    __shared__ float red[8];
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    m = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    const float sc = m > 0.f ? m / 127.f : 1.f, inv = 1.f / sc;
    if (threadIdx.x == 0) scale[r] = sc;
    int8_t * o = out + size_t(r) * cols;
    for (int c = threadIdx.x; c < cols; c += 256) o[c] = int8_t(__float2int_rn(fminf(fmaxf(val(c) * inv, -127.f), 127.f)));
}

// quant_rows_cm_k<false> (x / cm) with 16-byte loads and 8-byte stores: cols and ld multiples of 8
__global__ void __launch_bounds__(256) quant_rows_div8_k(const __half * in, int cols, int ld_in, int8_t * out, float * scale, const float * cm) {
    const int r = blockIdx.x, c8 = cols / 8;
    const uint4 * row = reinterpret_cast<const uint4 *>(in + size_t(r) * ld_in);
    const float4 * cm4 = reinterpret_cast<const float4 *>(cm);
    auto load = [&](int i, float (&f)[8]) {
        const uint4 u = row[i];
        const __half * h = reinterpret_cast<const __half *>(&u);
        const float4 a = cm4[2 * i], b = cm4[2 * i + 1];
        const float d[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) f[e] = __half2float(h[e]) / d[e];
    };
    float m = 0.f;
    for (int i = threadIdx.x; i < c8; i += 256) {
        float f[8];
        load(i, f);
#pragma unroll
        for (int e = 0; e < 8; ++e) m = fmaxf(m, fabsf(f[e]));
    }
    __shared__ float red[8];
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    m = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    const float sc = m > 0.f ? m / 127.f : 1.f, inv = 1.f / sc;
    if (threadIdx.x == 0) scale[r] = sc;
    uint2 * o = reinterpret_cast<uint2 *>(out + size_t(r) * cols);
    for (int i = threadIdx.x; i < c8; i += 256) {
        float f[8];
        load(i, f);
        __align__(8) int8_t q[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) q[e] = int8_t(__float2int_rn(fminf(fmaxf(f[e] * inv, -127.f), 127.f)));
        o[i] = *reinterpret_cast<const uint2 *>(q);
    }
}

// out[c] = max(out[c], max_r |a[r][c]|), 8 columns per thread (16-byte loads); grid.y splits the rows (non-negative
// floats order as their int bits, so atomicMax on the bits is a float max)
__global__ void colmax_k(const __half * a, int rows, int cols, int ld, float * out) {
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (c0 >= cols) return;
    float m[8] = {};
    for (int r = blockIdx.y; r < rows; r += gridDim.y) {
        const uint4 u = *reinterpret_cast<const uint4 *>(a + size_t(r) * ld + c0);
        const __half * h = reinterpret_cast<const __half *>(&u);
#pragma unroll
        for (int e = 0; e < 8; ++e) m[e] = fmaxf(m[e], fabsf(__half2float(h[e])));
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) atomicMax(reinterpret_cast<int *>(out + c0 + e), __float_as_int(m[e]));
}

__global__ void smooth_k(const float * ax, const float * aw, float * s, int cols, float alpha) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    const float x = fmaxf(ax[c], 1e-5f), w = fmaxf(aw[c], 1e-5f);
    s[c] = fminf(fmaxf(powf(x, alpha) / powf(w, 1.f - alpha), 1e-3f), 1e3f);
}

template <class T>
__global__ void i32_scaled_k(const int32_t * c, T * out, int ldo, const float * sw, const float * sx) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (n >= ldo) return;
    const size_t i = size_t(m) * ldo + n;
    const float v = float(c[i]) * sw[n] * sx[m];
    if constexpr (sizeof(T) == 2 && std::is_same_v<T, __half>) out[i] = __float2half(v);
    else out[i] = __float2bfloat16(v);
}
}  // namespace

void int8_quant_rows(const __half * in, int rows, int cols, int ld_in, int8_t * out, float * scale, cudaStream_t s) {
    if (rows > 0) quant_rows_k<<<rows, 256, 0, s>>>(in, cols, ld_in, out, scale);
}
void int8_colmax_h(const __half * a, int rows, int cols, int ld, float * out, cudaStream_t s) {
    if (cols % 8 || ld % 8) throw std::invalid_argument("int8 colmax: columns and ld must be multiples of 8");
    if (rows > 0) colmax_k<<<dim3((cols / 8 + 127) / 128, std::min(128, rows)), 128, 0, s>>>(a, rows, cols, ld, out);
}
void int8_smooth_scales(const float * ax, const float * aw, float * scale, int cols, float alpha, cudaStream_t s) {
    if (cols > 0) smooth_k<<<(cols + 255) / 256, 256, 0, s>>>(ax, aw, scale, cols, alpha);
}
void int8_i32_to_bf16_scaled(const int32_t * c, __nv_bfloat16 * out, int ntok, int ldo, const float * sw, const float * sx, cudaStream_t s) {
    if (ntok > 0 && ldo > 0) i32_scaled_k<__nv_bfloat16><<<dim3((ldo + 255) / 256, ntok), 256, 0, s>>>(c, out, ldo, sw, sx);
}
void int8_i32_to_h_scaled(const int32_t * c, __half * out, int ntok, int ldo, const float * sw, const float * sx, cudaStream_t s) {
    if (ntok > 0 && ldo > 0) i32_scaled_k<__half><<<dim3((ldo + 255) / 256, ntok), 256, 0, s>>>(c, out, ldo, sw, sx);
}

bool Int8Gemm::supported() {
    int dev = 0, ma = 0, mi = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
    return ma * 10 + mi >= 80;
}

bool Int8Gemm::default_on() {
    int dev = 0, ma = 0, mi = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
    const int cc = ma * 10 + mi;
    // Ampere: 3090 20.7K prompt 1640 -> 2123 tok/s; RTX 50 (sm_120): 5060 Ti 27.5K 1066 -> 1200 (+11-13 %, 2026-10-09);
    // RTX 40 (sm_89) by analogy (same full-rate int8 vs half-rate FP32-accumulate FP16/FP8 on GeForce), not measured
    return (ma == 8 && mi < 9) || cc == 89 || cc == 120;
}

void Int8Gemm::init(size_t max_w, size_t max_x, int max_rows, int max_tok, int8_t * w8, void * ws, size_t ws_bytes) {
    cl(cublasLtCreate(&lt_), "create");
    own_w8_ = !w8;
    own_ws_ = !ws;
    if (w8) w8_ = w8;
    else ck(tqllm_cuda::dev_malloc(&w8_, max_w), "w8");
    ck(tqllm_cuda::dev_malloc(&x8_, max_x), "x8");
    ck(tqllm_cuda::dev_malloc(&sw_, size_t(max_rows) * 4), "sw");
    ck(tqllm_cuda::dev_malloc(&sx_, size_t(max_tok) * 4), "sx");
    if (ws) ws_ = ws, ws_bytes_ = ws_bytes;
    else ck(tqllm_cuda::dev_malloc(&ws_, ws_bytes_), "workspace");
}

Int8Gemm::~Int8Gemm() {
    for (void * p : {own_w8_ ? static_cast<void *>(w8_) : nullptr, static_cast<void *>(x8_), static_cast<void *>(sw_),
                     static_cast<void *>(sx_), own_ws_ ? ws_ : nullptr})
        if (p) tqllm_cuda::dev_free(p);
    if (lt_) cublasLtDestroy(lt_);
}

void Int8Gemm::quant_x(const __half * x, int ldx, int tok, int k, cudaStream_t s, const float * col_div) {
    if (!col_div) int8_quant_rows(x, tok, k, ldx, x8_, sx_, s);
    else if (tok > 0 && k % 8 == 0 && ldx % 8 == 0) quant_rows_div8_k<<<tok, 256, 0, s>>>(x, k, ldx, x8_, sx_, col_div);
    else if (tok > 0) quant_rows_cm_k<false><<<tok, 256, 0, s>>>(x, k, ldx, x8_, sx_, col_div);
}

// cuBLASLt int8 x int8 -> int32 from the resident w8_/x8_ buffers; y [tok][ldo] unscaled (scales applied by the
// int8_i32_to_* pass). Algorithm picked once per shape and cached.
void Int8Gemm::matmul(int rows, int k, int tok, int32_t * y, int ldo, cudaStream_t s) {
    cublasLtMatmulDesc_t op;
    cl(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I), "desc");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cl(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
    cl(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
    cublasLtMatrixLayout_t la, lb, lc;
    cl(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, k, rows, k), "la");
    cl(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, k, tok, k), "lb");
    cl(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, rows, tok, ldo), "lc");
    const int32_t alpha = 1, beta = 0;
    const auto key = std::make_tuple(rows, k, tok, ldo);
    auto it = algos_.find(key);
    if (it == algos_.end()) {  // time each heuristic candidate once per shape (first picks are not always fastest)
        cublasLtMatmulPreference_t pref;
        cl(cublasLtMatmulPreferenceCreate(&pref), "pref");
        cl(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes_, sizeof(ws_bytes_)), "pref ws");
        cublasLtMatmulHeuristicResult_t h[8]{};
        int nh = 0;
        cl(cublasLtMatmulAlgoGetHeuristic(lt_, op, la, lb, lc, lc, pref, 8, h, &nh), "heuristic");
        cublasLtMatmulPreferenceDestroy(pref);
        if (!nh) throw std::runtime_error("int8 gemm: no cuBLASLt algorithm for this shape");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        int best = 0;
        float best_ms = 1e30f;
        // inside a CUDA graph capture (verify steps) nothing may synchronise: keep the heuristic's first pick for this shape
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s, &cap);
        for (int i = 0; i < nh && cap == cudaStreamCaptureStatusNone; ++i) {
            if (cublasLtMatmul(lt_, op, &alpha, w8_, la, x8_, lb, &beta, y, lc, y, lc, &h[i].algo, ws_, ws_bytes_, s) != CUBLAS_STATUS_SUCCESS) continue;
            cudaEventRecord(e0, s);
            const bool ok = cublasLtMatmul(lt_, op, &alpha, w8_, la, x8_, lb, &beta, y, lc, y, lc, &h[i].algo, ws_, ws_bytes_, s) == CUBLAS_STATUS_SUCCESS;
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
    cl(cublasLtMatmul(lt_, op, &alpha, w8_, la, x8_, lb, &beta, y, lc, y, lc, &it->second, ws_, ws_bytes_, s), "matmul");
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
}

void Int8Gemm::gemm(const __half * w, int rows, int k, int tok, int32_t * y, int ldo, float * sw_out, cudaStream_t s,
                    const float * col_mul) {
    if (k % 4 || ldo % 4) throw std::invalid_argument("int8 gemm: k and ldo must be multiples of 4");
    if (!col_mul) int8_quant_rows(w, rows, k, k, w8_, sw_, s);
    else if (rows > 0) quant_rows_cm_k<true><<<rows, 256, 0, s>>>(w, k, k, w8_, sw_, col_mul);
    matmul(rows, k, tok, y, ldo, s);
    if (sw_out) ck(cudaMemcpyAsync(sw_out, sw_, size_t(rows) * 4, cudaMemcpyDeviceToDevice, s), "row scales");
}

// Weights already quantised into w8_ (a format's straight-to-int8 reconstruct with load-time row scales): multiply
// only; the caller has placed the row scales where the output conversion reads them.
void Int8Gemm::gemm_prequant(int rows, int k, int tok, int32_t * y, int ldo, cudaStream_t s) {
    if (k % 4 || ldo % 4) throw std::invalid_argument("int8 gemm: k and ldo must be multiples of 4");
    matmul(rows, k, tok, y, ldo, s);
}

}  // namespace tqllm_cuda
