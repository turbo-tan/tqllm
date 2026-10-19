#include "quant_kernels.cuh"
#include "weight_format.hpp"

#include <cstring>
#include <vector>

namespace tqllm::formats {
namespace {
// Dense GGUF matrices: BF16 (30) and F16 (1) run as stored; F32 (0) is stored as BF16 (round to nearest even), as the
// 27B loader does. Prefill uses dequant_h + cuBLAS (no fused gemm).
struct Fp final : WeightFormat {
    bool bf16; uint32_t id;
    Fp(bool b, uint32_t i) : bf16(b), id(i) {}
    const char * name() const override { return id == 0 ? "F32" : bf16 ? "BF16" : "F16"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        const size_t n = size_t(rows) * cols;
        if (id != 0) return {up(raw.data(), n * 2), id, rows, cols, nullptr, this};
        std::vector<uint16_t> b(n);
        for (size_t i = 0; i < n; ++i) {
            uint32_t u; std::memcpy(&u, raw.data() + 4 * i, 4);
            b[i] = uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
        }
        return {up(b.data(), n * 2), 30, rows, cols, nullptr, this};
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::fp_matvec(bf16, m.p, x, y, false, m.rows, m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::fp_matvec(bf16, m.p, x, y, false, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::fp_matvec(bf16, m.p, x, y, true, m.rows, m.cols, n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::fp_dequant_h(bf16, m.p, out, size_t(m.rows) * m.cols, s);
    }
    size_t row_bytes(int cols) const override { return size_t(cols) * 2; }
};
}  // namespace
void register_fp() {
    register_weight_format(30, std::make_unique<Fp>(true, 30));
    register_weight_format(1, std::make_unique<Fp>(false, 1));
    register_weight_format(0, std::make_unique<Fp>(true, 0));
}
}  // namespace tqllm::formats
