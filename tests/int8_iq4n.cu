// int8 prefill for IQ4N (IQ4_XS) GGUF weights:
// 1. iq4n_dequant_i8 is bit-identical to iq4n_dequant_h + int8_quant_rows (int8 operand and row scales);
// 2. with column factors it matches that FP16 row times cm, quantised by the same rule (host reference);
// 3. SmoothQuant scales (int8_colmax_h + int8_smooth_scales, x / s and w * s) keep the int8 GEMM's product and cut its
//    error on inputs with activation outlier channels, the plain-basis case (no rotation spreads them).
// Needs an sm_8x+ GPU; skipped elsewhere.
#include "int8_gemm.cuh"
#include "formats/weight/quant_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char * what, double v) {
    std::printf("%-64s %.6g %s\n", what, v, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}
// random IQ4N: nibbles, and per 256 weights an 8-byte record (d FP16, scales_h u16, scales_l u32)
void make_iq4n(int rows, int k, std::mt19937 & rng, std::vector<uint8_t> & w, std::vector<uint8_t> & rec) {
    w.resize(size_t(rows) * k / 2);
    rec.resize(size_t(rows) * k / 256 * 8);
    for (auto & b : w) b = uint8_t(rng());
    std::uniform_real_distribution<float> du(0.0005f, 0.004f);
    for (size_t i = 0; i < rec.size() / 8; ++i) {
        const float d = du(rng) * (i % 41 == 3 ? 20.f : 1.f);  // some super-blocks far larger: row scale set by them
        const uint16_t dh = __half_as_ushort(__float2half(d));
        const uint16_t sh = uint16_t(rng());
        const uint32_t sl = uint32_t(rng());
        std::memcpy(&rec[i * 8], &dh, 2);
        std::memcpy(&rec[i * 8 + 2], &sh, 2);
        std::memcpy(&rec[i * 8 + 4], &sl, 4);
    }
}
}  // namespace

int main() {
    if (!tqllm_cuda::Int8Gemm::supported()) { std::puts("int8_iq4n: skipped (no int8 tensor cores)"); return 0; }
    std::mt19937 rng(11);
    const int R = 1056, K = 5120;  // 1056: not a multiple of the 64-row / 256-thread tiles
    std::vector<uint8_t> w, rec;
    make_iq4n(R, K, rng, w, rec);
    uint8_t *dw, *drec;
    __half * dh;
    int8_t *q_ref, *q_new, *q_cm;
    float *s_ref, *s_new, *s_cm, *cm;
    cudaMalloc(&dw, w.size()); cudaMalloc(&drec, rec.size());
    cudaMalloc(&dh, size_t(R) * K * 2);
    cudaMalloc(&q_ref, size_t(R) * K); cudaMalloc(&q_new, size_t(R) * K); cudaMalloc(&q_cm, size_t(R) * K);
    cudaMalloc(&s_ref, R * 4); cudaMalloc(&s_new, R * 4); cudaMalloc(&s_cm, R * 4); cudaMalloc(&cm, K * 4);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(drec, rec.data(), rec.size(), cudaMemcpyHostToDevice);
    std::vector<float> hcm(K);
    std::lognormal_distribution<float> ln(0.f, 1.f);
    for (auto & c : hcm) c = ln(rng);
    cudaMemcpy(cm, hcm.data(), K * 4, cudaMemcpyHostToDevice);

    tqllm_cuda::iq4n_dequant_h(dw, drec, dh, R, K, nullptr);
    tqllm_cuda::int8_quant_rows(dh, R, K, K, q_ref, s_ref, nullptr);
    tqllm_cuda::iq4n_dequant_i8(dw, drec, q_new, s_new, R, K, nullptr);
    tqllm_cuda::iq4n_dequant_i8(dw, drec, q_cm, s_cm, R, K, nullptr, cm);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::puts("FAIL: CUDA error"); return 1; }
    std::vector<int8_t> a(size_t(R) * K), b(size_t(R) * K), c(size_t(R) * K);
    std::vector<float> sa(R), sb(R), sc(R);
    std::vector<__half> h(size_t(R) * K);
    cudaMemcpy(a.data(), q_ref, a.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), q_new, b.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(c.data(), q_cm, c.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(sa.data(), s_ref, R * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(sb.data(), s_new, R * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(sc.data(), s_cm, R * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h.data(), dh, h.size() * 2, cudaMemcpyDeviceToHost);
    check(a == b, "dequant_i8 int8 operand == dequant_h + int8_quant_rows", 0);
    check(std::memcmp(sa.data(), sb.data(), R * 4) == 0, "dequant_i8 row scales == int8_quant_rows (bitwise)", 0);
    // host reference of the column-scaled rule on the same FP16 values
    size_t bad_q = 0, bad_s = 0;
    for (int r = 0; r < R; ++r) {
        float m = 0.f;
        for (int k = 0; k < K; ++k) m = std::fmax(m, std::fabs(__half2float(h[size_t(r) * K + k]) * hcm[k]));
        const float s = m > 0.f ? m / 127.f : 1.f, inv = 1.f / s;
        bad_s += std::memcmp(&s, &sc[r], 4) != 0;
        for (int k = 0; k < K; ++k) {
            const float v = __half2float(h[size_t(r) * K + k]) * hcm[k] * inv;
            const int8_t q = int8_t(std::nearbyint(std::fmin(std::fmax(v, -127.f), 127.f)));
            bad_q += q != c[size_t(r) * K + k];
        }
    }
    check(bad_s == 0, "dequant_i8 with column factors: row scales (host reference)", double(bad_s));
    check(bad_q == 0, "dequant_i8 with column factors: int8 operand (host reference)", double(bad_q));

    // SmoothQuant on an int8 GEMM: activations with a few outlier channels (plain basis), FP32 reference
    const int M = 512;
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<__half> hx(size_t(M) * K);
    for (int t = 0; t < M; ++t)
        for (int k = 0; k < K; ++k) hx[size_t(t) * K + k] = __float2half(g(rng) * (k % 701 == 5 ? 60.f : 1.f));
    __half * dx;
    float *fw, *fx, *fy, *ax, *aw, *sm;
    cudaMalloc(&dx, hx.size() * 2);
    cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    std::vector<float> w32(size_t(R) * K), x32(hx.size());
    for (size_t i = 0; i < w32.size(); ++i) w32[i] = __half2float(h[i]);
    for (size_t i = 0; i < x32.size(); ++i) x32[i] = __half2float(hx[i]);
    cudaMalloc(&fw, w32.size() * 4); cudaMalloc(&fx, x32.size() * 4); cudaMalloc(&fy, size_t(M) * R * 4);
    cudaMemcpy(fw, w32.data(), w32.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(fx, x32.data(), x32.size() * 4, cudaMemcpyHostToDevice);
    cublasHandle_t bl;
    cublasCreate(&bl);
    cublasSetMathMode(bl, CUBLAS_PEDANTIC_MATH);
    const float one = 1.f, zero = 0.f;
    cublasSgemm(bl, CUBLAS_OP_T, CUBLAS_OP_N, R, M, K, &one, fw, K, fx, K, &zero, fy, R);
    cudaMalloc(&ax, K * 4); cudaMalloc(&aw, K * 4); cudaMalloc(&sm, K * 4);
    cudaMemset(ax, 0, K * 4); cudaMemset(aw, 0, K * 4);
    tqllm_cuda::int8_colmax_h(dx, M, K, K, ax, nullptr);
    tqllm_cuda::int8_colmax_h(dh, R, K, K, aw, nullptr);
    tqllm_cuda::int8_smooth_scales(ax, aw, sm, K, 0.75f, nullptr);
    tqllm_cuda::Int8Gemm g8;
    g8.init(size_t(R) * K, size_t(M) * K, R, M);
    int32_t * acc;
    float * sw;
    __nv_bfloat16 *y0, *y1;
    cudaMalloc(&acc, size_t(M) * R * 4); cudaMalloc(&sw, R * 4);
    cudaMalloc(&y0, size_t(M) * R * 2); cudaMalloc(&y1, size_t(M) * R * 2);
    g8.quant_x(dx, K, M, K, nullptr);  // plain per-token
    g8.gemm(dh, R, K, M, acc, R, sw, nullptr);
    tqllm_cuda::int8_i32_to_bf16_scaled(acc, y0, M, R, sw, g8.sx(), nullptr);
    g8.quant_x(dx, K, M, K, nullptr, sm);  // smoothed: x / s, w * s (through the IQ4N kernel)
    tqllm_cuda::iq4n_dequant_i8(dw, drec, g8.w8(), sw, R, K, nullptr, sm);
    g8.gemm_prequant(R, K, M, acc, R, nullptr);
    tqllm_cuda::int8_i32_to_bf16_scaled(acc, y1, M, R, sw, g8.sx(), nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::puts("FAIL: CUDA error"); return 1; }
    std::vector<float> ref(size_t(M) * R);
    std::vector<__nv_bfloat16> o0(ref.size()), o1(ref.size());
    cudaMemcpy(ref.data(), fy, ref.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o0.data(), y0, o0.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(o1.data(), y1, o1.size() * 2, cudaMemcpyDeviceToHost);
    double e0 = 0, e1 = 0, rr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double t = ref[i];
        e0 += (__bfloat162float(o0[i]) - t) * (__bfloat162float(o0[i]) - t);
        e1 += (__bfloat162float(o1[i]) - t) * (__bfloat162float(o1[i]) - t);
        rr += t * t;
    }
    const double r0 = std::sqrt(e0 / rr), r1 = std::sqrt(e1 / rr);
    check(r1 < 0.05, "smoothed int8 GEMM relative error (outlier channels)", r1);
    check(r1 < 0.5 * r0, "smoothing at least halves the plain per-token int8 error", r0);
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::puts("int8_iq4n: ok");
    return 0;
}
