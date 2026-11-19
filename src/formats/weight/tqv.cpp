#include "quant_kernels.cuh"
#include "weight_format.hpp"
#include "gguf.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace tqllm::formats {
namespace {
// TQV-12 (docs/reference/formats.md; files: GGUF type 305 + tqllm.tqv.codebook = "tqv12-v1", the engine id
// 305): TQ4's rotated basis, per 128 weights an FP16 scale + 32 groups of 4 weights, each a 12-bit index into the official
// int8 codebook (tqv12_codebook.inc) + 4 sign bits; 66 bytes as TQ4. Split at load into codes [rows][cols / 2] (the file's
// 16-byte units unchanged) and scales [rows][cols / 128], the TQ4 layout, so row ranges and the scale handling match TQ4.
struct Tqv12 final : WeightFormat {
    const char * name() const override { return "TQV12"; }
    Basis basis() const override { return Basis::Tq4Rot128; }
    int max_matvec_cols() const override { return 16; }  // single-request tree verify up to 16 nodes (as TQ4)
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        if (cols % 128) throw std::runtime_error("TQV12: the row length must be a multiple of 128");
        const int64_t ng = int64_t(rows) * cols / 128;
        if (raw.size() != size_t(ng) * 66) throw std::runtime_error("TQV12: tensor bytes do not match its shape");
        std::vector<uint8_t> codes(size_t(ng) * 64);
        std::vector<__half> sc(static_cast<size_t>(ng));
        for (int64_t g = 0; g < ng; ++g) {
            std::memcpy(&sc[g], raw.data() + g * 66, 2);
            std::memcpy(codes.data() + g * 64, raw.data() + g * 66 + 2, 64);
        }
        const void * p = up(codes.data(), codes.size());
        const void * scp = up(sc.data(), sc.size() * sizeof(__half));
        return {p, GGML_TQV12, rows, cols, scp, this};
    }
    static const uint8_t * w(const Mat & m) { return static_cast<const uint8_t *>(m.p); }
    static const __half * s(const Mat & m) { return static_cast<const __half *>(m.aux); }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t st) const override {
        tqllm_cuda::tqv12_matvec_bf16(w(m), s(m), x, y, m.rows, m.cols, n, ldy, st);
        return true;
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t st) const override {
        tqllm_cuda::tqv12_matvec_bf16(w(m), s(m), x, y, m.rows, m.cols, n, m.rows, st);
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t st) const override {
        tqllm_cuda::tqv12_matvec(w(m), s(m), x, y, m.rows, m.cols, n, st);
    }
    size_t row_bytes(int cols) const override { return size_t(cols) / 2; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 128 * 2; }
    void dequant_h(const Mat & m, __half * out, cudaStream_t st) const override {
        tqllm_cuda::tqv12_dequant_h(w(m), s(m), out, m.rows, m.cols, st);
    }
    bool dequant_rows_h(const Mat & m, __half * out, int row0, int nrows, cudaStream_t st) const override {
        tqllm_cuda::tqv12_dequant_h(w(m) + size_t(row0) * row_bytes(m.cols), s(m) + size_t(row0) * (m.cols / 128), out, nrows,
                                    m.cols, st);
        return true;
    }
    // native prefill GEMM (as TQ4): chunks below the cuBLAS crossover and the FP16 path without an FP16 weight copy
    bool has_gemm() const override { return true; }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t st) const override {
        tqllm_cuda::tqv12_gemm_h(w(m), s(m), x, y, m.rows, m.cols, n, m.rows, st);
        return true;
    }
    bool has_dequant_i8() const override { return true; }
    bool dequant_i8(const Mat & m, int8_t * w8, float * row_scale, cudaStream_t st) const override {
        tqllm_cuda::tqv12_dequant_i8(w(m), s(m), w8, row_scale, m.rows, m.cols, st);
        return true;
    }
    bool has_dequant_i8_cm() const override { return true; }
    bool dequant_i8_cm(const Mat & m, const float * cm, int8_t * w8, float * row_scale, cudaStream_t st) const override {
        tqllm_cuda::tqv12_dequant_i8(w(m), s(m), w8, row_scale, m.rows, m.cols, st, cm);
        return true;
    }
};
}  // namespace
void register_tqv12() { register_weight_format(GGML_TQV12, std::make_unique<Tqv12>()); }
}  // namespace tqllm::formats
