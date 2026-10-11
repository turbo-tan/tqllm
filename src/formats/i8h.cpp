#include "quant_kernels.cuh"
#include "weight_format.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace tqllm::formats {
namespace {
// I8H (engine id 1000): int8 + FP16 scale per 16 (drafter matrices, int8 LM-head copy).
struct I8h final : WeightFormat {
    const char * name() const override { return "I8H"; }
    size_t row_bytes(int cols) const override { return size_t(cols); }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 16 * sizeof(__half); }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("I8H is produced by other loaders");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::i8h_matvec_bf16(static_cast<const int8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows,
                                    m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::i8h_matvec_bf16(static_cast<const int8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows,
                                    m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::i8h_matvec(static_cast<const int8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows, m.cols,
                               n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::i8h_dequant_h(static_cast<const int8_t *>(m.p), static_cast<const __half *>(m.aux), out, m.rows, m.cols,
                                  s);
    }
};
// Q8_0 (GGUF 8) loader: int8 + FP16 scale per 32 -> I8H (scale repeated per 16), exact.
struct Q8_0 final : WeightFormat {
    const WeightFormat * i8h;
    explicit Q8_0(const WeightFormat * f) : i8h(f) {}
    const char * name() const override { return "Q8_0"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        const int64_t nb = int64_t(rows) * cols / 32;
        std::vector<int8_t> q(size_t(nb) * 32);
        std::vector<__half> sc(size_t(nb) * 2);
        for (int64_t b = 0; b < nb; ++b) {
            const char * blk = raw.data() + b * 34;
            __half d;
            std::memcpy(&d, blk, 2);
            sc[2 * b] = sc[2 * b + 1] = d;
            std::memcpy(q.data() + b * 32, blk + 2, 32);
        }
        const void * p = up(q.data(), q.size());
        return {p, 1000, rows, cols, up(sc.data(), sc.size() * sizeof(__half)), i8h};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("Q8_0 is a loader; matrices run as I8H");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("Q8_0 is a loader; matrices run as I8H");
    }
};
// Q6_K (GGUF 14) loader: per 16 weights a 6-bit code - 32 and an int8 scale times the block's FP16 d -> I8H.
struct Q6_K final : WeightFormat {
    const WeightFormat * i8h;
    explicit Q6_K(const WeightFormat * f) : i8h(f) {}
    const char * name() const override { return "Q6_K"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        if (cols % 256) throw std::runtime_error("Q6_K needs a multiple of 256 columns");
        const int64_t nb = int64_t(rows) * cols / 256;
        std::vector<int8_t> q(size_t(nb) * 256);
        std::vector<__half> sc(size_t(nb) * 16);
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 210;
            const uint8_t * ql = blk, * qh = blk + 128;
            const auto * s8 = reinterpret_cast<const int8_t *>(blk + 192);
            __half hd;
            std::memcpy(&hd, blk + 208, 2);
            const float d = __half2float(hd);
            for (int j = 0; j < 16; ++j) sc[b * 16 + j] = __float2half_rn(d * s8[j]);
            int8_t * y = q.data() + b * 256;
            for (int h = 0; h < 2; ++h, ql += 64, qh += 32, y += 128)
                for (int l = 0; l < 32; ++l) {
                    y[l] = int8_t(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                    y[l + 32] = int8_t(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    y[l + 64] = int8_t(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                    y[l + 96] = int8_t(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                }
        }
        const void * p = up(q.data(), q.size());
        return {p, 1000, rows, cols, up(sc.data(), sc.size() * sizeof(__half)), i8h};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("Q6_K is a loader; matrices run as I8H");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("Q6_K is a loader; matrices run as I8H");
    }
};
}  // namespace
void register_i8h() {  // no weight_format() lookups here (runs inside the registry's call_once)
    auto i8h = std::make_unique<I8h>();
    auto q80 = std::make_unique<Q8_0>(i8h.get());
    auto q6k = std::make_unique<Q6_K>(i8h.get());
    register_weight_format(1000, std::move(i8h));
    register_weight_format(8, std::move(q80));
    register_weight_format(14, std::move(q6k));
}
}  // namespace tqllm::formats
