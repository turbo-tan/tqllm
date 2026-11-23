#include "parallel_blocks.hpp"
#include "quant_kernels.cuh"
#include "weight_format.hpp"
#include "gguf.hpp"

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tqllm::formats {
namespace {
// FP8T (GGML_FP8T): per 128 weights the matrix's F32 scale, then 128 E4M3 bytes (modelopt FP8 projections as stored).
// Device: rows of cols E4M3 bytes, aux = one F32 scale per row (row-concatenated parts keep their own); prefill through
// dequant_h. The GGUF repeats the scale per block: a row whose blocks disagree is refused.
struct Fp8t final : WeightFormat {
    const char * name() const override { return "FP8T"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        if (cols % 128 || raw.size() != size_t(rows) * (cols / 128) * 132)
            throw std::runtime_error("FP8T: expected blocks of an F32 scale + 128 E4M3 bytes, cols % 128 == 0");
        const int nb = cols / 128;
        std::vector<uint8_t> w(size_t(rows) * cols);
        std::vector<float> sc(static_cast<size_t>(rows));
        std::atomic<bool> mixed{false};
        tqllm::parallel_blocks(rows, [&](int64_t r) {
            const auto * row = reinterpret_cast<const uint8_t *>(raw.data()) + r * int64_t(nb) * 132;
            for (int b = 0; b < nb; ++b) {
                if (b && std::memcmp(row + b * 132, row, 4) != 0) mixed = true;
                std::memcpy(w.data() + size_t(r) * cols + size_t(b) * 128, row + b * 132 + 4, 128);
            }
            std::memcpy(&sc[size_t(r)], row, 4);
        });
        if (mixed) throw std::runtime_error("FP8T: a row has more than one scale");
        const void * p = up(w.data(), w.size());
        const void * a = up(sc.data(), sc.size() * 4);
        return {p, GGML_FP8T, rows, cols, a, this};
    }
    static const float * scale(const Mat & m) { return static_cast<const float *>(m.aux); }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::fp8t_matvec(m.p, scale(m), x, y, false, m.rows, m.cols, n, s, 0);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::fp8t_matvec(m.p, scale(m), x, y, false, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::fp8t_matvec(m.p, scale(m), x, y, true, m.rows, m.cols, n, s, 0);
    }
    // 1..16 columns per launch, a column's bits independent of the launch width (tests/fp8t.cu); the dispatch cap engines
    // choose their path by stays at the default 8, as MXFP8 / BF16
    int max_batch_cols() const override { return 16; }
    int max_exact_cols() const override { return 16; }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::fp8t_dequant_h(m.p, scale(m), out, m.rows, m.cols, s);
    }
    size_t row_bytes(int cols) const override { return size_t(cols); }
    size_t aux_row_bytes(int) const override { return 4; }
};
}  // namespace
void register_fp8t() { register_weight_format(GGML_FP8T, std::make_unique<Fp8t>()); }
}  // namespace tqllm::formats
