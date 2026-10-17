#include "weight_format.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tqllm {
namespace {
constexpr uint32_t ID_COMPOSITE = 1100;

// Each part writes its own row block of y through matvec_ld (no scratch, no copies); per-row results are those of the
// part's format alone.
struct Composite final : WeightFormat {
    std::vector<Mat> ps;
    std::string nm;
    explicit Composite(std::vector<Mat> p) : ps(std::move(p)) {
        for (const auto & m : ps) nm += (nm.empty() ? "" : "+") + std::string(m.fmt->name());
    }
    const char * name() const override { return nm.c_str(); }
    Basis basis() const override { return ps[0].fmt->basis(); }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("composite matrices are built from loaded parts");
    }
    int max_matvec_cols() const override {
        int c = ps[0].fmt->max_matvec_cols();
        for (const auto & m : ps) c = std::min(c, m.fmt->max_matvec_cols());
        return c;
    }
    int max_batch_cols() const override {
        int c = ps[0].fmt->max_batch_cols();
        for (const auto & m : ps) c = std::min(c, m.fmt->max_batch_cols());
        return c;
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat &, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        int r0 = 0;
        for (const auto & m : ps) {
            m.fmt->matvec_ld(m, x, y + r0, n, ldy, s);
            r0 += m.rows;
        }
        return true;
    }
    void matvec(const Mat & c, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        matvec_ld(c, x, y, n, c.rows, s);
    }
    void dequant_h(const Mat & c, __half * out, cudaStream_t s) const override {
        int r0 = 0;
        for (const auto & m : ps) {
            m.fmt->dequant_h(m, out + size_t(r0) * c.cols, s);
            r0 += m.rows;
        }
    }
    int fused_gemm_cols() const override {
        int c = ps[0].fmt->fused_gemm_cols();
        for (const auto & m : ps) c = std::min(c, m.fmt->fused_gemm_cols());
        return c;
    }
    bool gemm(const Mat & c, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(c, x, y, n, c.rows, s);
    }
    bool gemm_ld(const Mat &, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        if (fused_gemm_cols() < n) return false;  // every part must take the fused path
        int r0 = 0;
        for (const auto & m : ps) {
            if (!m.fmt->gemm_ld(m, x, y + r0, n, ldy, s)) throw std::runtime_error(std::string(m.fmt->name()) + ": no strided gemm");
            r0 += m.rows;
        }
        return true;
    }
    const std::vector<Mat> * parts() const override { return &ps; }
};
}  // namespace

std::unique_ptr<WeightFormat> composite_format(std::vector<Mat> parts) {
    if (parts.size() < 2) return nullptr;
    for (const auto & m : parts)
        if (!m.fmt || !m.fmt->has_matvec_ld() || m.fmt->parts() || m.cols != parts[0].cols ||
            m.fmt->basis() != parts[0].fmt->basis())
            return nullptr;
    return std::make_unique<Composite>(std::move(parts));
}

Mat composite_mat(const WeightFormat & c) {
    const auto & ps = *c.parts();
    int rows = 0;
    for (const auto & m : ps) rows += m.rows;
    return {ps[0].p, ID_COMPOSITE, rows, ps[0].cols, nullptr, &c};
}

}  // namespace tqllm
