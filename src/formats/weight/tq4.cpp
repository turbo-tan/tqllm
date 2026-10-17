#include "quant_kernels.cuh"
#include "weight_format.hpp"
#include "debug.hpp"

#include <cstring>

namespace tqllm::formats {
namespace {
// TQ4 (GGUF 300, TQLLM-only): rotated 4-bit, 16-level codebook. File block per 128: FP16 scale + 64 bytes of codes (natural
// nibble order); split at load into codes [rows][cols / 2] and scales [rows][cols / 128].
struct Tq4 final : WeightFormat {
    const char * name() const override { return "TQ4"; }
    Basis basis() const override { return Basis::Tq4Rot128; }
    // single-request verify up to 16 columns on the matvec kernel (tree drafts of 9-15 nodes); above it the engine
    // falls back to the prefill GEMM, which dequantises the whole matrix per step
    int max_matvec_cols() const override { return 16; }
    int max_batch_cols() const override { return 32; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        const int64_t ng = int64_t(rows) * cols / 128;
        std::vector<uint8_t> codes(size_t(ng) * 64);
        std::vector<__half> sc(static_cast<size_t>(ng));
        for (int64_t g = 0; g < ng; ++g) {
            std::memcpy(&sc[g], raw.data() + g * 66, 2);
            std::memcpy(codes.data() + g * 64, raw.data() + g * 66 + 2, 64);
        }
        const void * p = up(codes.data(), codes.size());
        const void * scp = up(sc.data(), sc.size() * sizeof(__half));
        return {p, 300, rows, cols, scp, this};
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::tq4_matvec_bf16_ld(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows,
                                       m.cols, n, ldy, s);
        return true;
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::tq4_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows,
                                    m.cols, n, s);
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::tq4_matvec(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows, m.cols,
                               n, s);
    }
    // TQLLM_DEBUG=tq_gemm:off: no native GEMM (the pre-GEMM dequant + cuBLAS path, for A/B and the KLD reference)
    static bool gemm_off() { const std::string * v = tqllm::debug_value("tq_gemm"); return v && *v == "off"; }
    bool has_gemm() const override { return !gemm_off(); }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        if (m.cols % 128 || gemm_off()) return false;
        tqllm_cuda::tq4_gemm_h(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows, m.cols, n, m.rows, s);
        return true;
    }
    size_t row_bytes(int cols) const override { return size_t(cols) / 2; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 128 * 2; }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::tq4_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), out, m.rows,
                                  m.cols, s);
    }
};
}  // namespace
void register_tq4() { register_weight_format(300, std::make_unique<Tq4>()); }
}  // namespace tqllm::formats
