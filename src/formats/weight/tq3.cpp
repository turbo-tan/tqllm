#include "quant_kernels.cuh"
#include "weight_format.hpp"
#include "debug.hpp"

#include <cstring>

namespace tqllm::formats {
namespace {
// TQ3 (GGUF 301, TQLLM-only): TQ4's rotation (Tq4Rot128) with the 8-level code, 3.125 bpw. File block per
// 128 weights: FP16 scale + 48 bytes = 4 x (uint32 lo0, lo1, hi) planes per 32 weights; split at load into per-row code planes
// [rows][A0 | A1 | C][cols / 32] (uint32) and scales [rows][cols / 128]. Same basis as TQ4, so the two mix in one model.
struct Tq3 final : WeightFormat {
    const char * name() const override { return "TQ3"; }
    Basis basis() const override { return Basis::Tq4Rot128; }
    int max_batch_cols() const override { return 32; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        // device layout per row: planes [A0 x nb][A1 x nb][C x nb], nb = cols / 32 (coalesced loads in the matvec)
        const int64_t ng = int64_t(rows) * cols / 128, nb = cols / 32;
        std::vector<uint32_t> codes(size_t(rows) * nb * 3);
        std::vector<__half> sc(static_cast<size_t>(ng));
        for (int64_t g = 0; g < ng; ++g) {
            std::memcpy(&sc[g], raw.data() + g * 50, 2);
            const int64_t r = g / (cols / 128), b0 = (g % (cols / 128)) * 4;
            for (int j = 0; j < 4; ++j)
                for (int pl = 0; pl < 3; ++pl)
                    std::memcpy(&codes[size_t(r * nb * 3 + pl * nb + b0 + j)], raw.data() + g * 50 + 2 + (j * 3 + pl) * 4, 4);
        }
        const void * p = up(codes.data(), codes.size() * sizeof(uint32_t));
        const void * scp = up(sc.data(), sc.size() * sizeof(__half));
        return {p, 301, rows, cols, scp, this};
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::tq4b3_matvec_bf16_ld(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows,
                                       m.cols, n, ldy, s);
        return true;
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::tq4b3_matvec_bf16_ld(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows,
                                       m.cols, n, m.rows, s);
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::tq4b3_matvec(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows, m.cols,
                               n, s);
    }
    // TQLLM_DEBUG=tq_gemm:off: no native GEMM (the pre-GEMM dequant + cuBLAS path, for A/B and the KLD reference)
    static bool gemm_off() { const std::string * v = tqllm::debug_value("tq_gemm"); return v && *v == "off"; }
    bool has_gemm() const override { return !gemm_off(); }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        if (m.cols % 128 || gemm_off()) return false;
        tqllm_cuda::tq4b3_gemm_h(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows, m.cols, n, m.rows, s);
        return true;
    }
    size_t row_bytes(int cols) const override { return size_t(cols) / 32 * 12; }  // per row: A0, A1, C planes (uint32)
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 128 * 2; }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::tq4b3_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), out, m.rows,
                                  m.cols, s);
    }
};
}  // namespace
void register_tq3() { register_weight_format(301, std::make_unique<Tq3>()); }
}  // namespace tqllm::formats
