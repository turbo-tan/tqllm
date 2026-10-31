#include "quant_kernels.cuh"
#include "weight_format.hpp"
#include "gguf.hpp"

namespace tqllm::formats {
namespace {
// MXFP8 (GGML_MXFP8): rows of cols E4M3 then the row's cols / 32 E8M0 scales; run as stored, prefill through dequant_h.
struct Mxfp8 final : WeightFormat {
    const char * name() const override { return "MXFP8"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        if (cols % 512 || raw.size() != size_t(rows) * cols + size_t(rows) * (cols / 32))
            throw std::runtime_error("MXFP8: expected rows of cols E4M3 + cols/32 scales, cols % 512 == 0");
        return {up(raw.data(), raw.size()), GGML_MXFP8, rows, cols, nullptr, this};
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::mxfp8_matvec(m.p, x, y, false, m.rows, m.cols, n, s, 0);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::mxfp8_matvec(m.p, x, y, false, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::mxfp8_matvec(m.p, x, y, true, m.rows, m.cols, n, s, 0);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override { tqllm_cuda::mxfp8_dequant_h(m.p, out, m.rows, m.cols, s); }
    size_t row_bytes(int cols) const override { return size_t(cols) + size_t(cols) / 32; }
};
}  // namespace
void register_mxfp8() { register_weight_format(GGML_MXFP8, std::make_unique<Mxfp8>()); }
}  // namespace tqllm::formats
