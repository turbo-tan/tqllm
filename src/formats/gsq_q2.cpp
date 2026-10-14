#include "weight_format.hpp"
#include "../gguf.hpp"
#include "formats/gsq_q2.cuh"
#include <stdexcept>

namespace tqllm::formats {
namespace {
struct GsqQ2 final : WeightFormat {
    const char * name() const override { return "GSQ_Q2_0"; }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols / 64) * 18; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & upload, cudaStream_t) const override {
        if (rows <= 0 || cols <= 0 || cols % 64 || raw.size() != size_t(rows) * row_bytes(cols))
            throw std::invalid_argument("GSQ Q2_0: invalid matrix shape or byte count");
        return {upload(raw.data(), raw.size()), GGML_GSQ_Q2_0, rows, cols, nullptr, this};
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::gsq_q2_matvec(m.p, x, y, m.rows, m.cols, n, s);
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::gsq_q2_matvec_f32(m.p, x, y, m.rows, m.cols, n, s);
    }
    bool has_expert_matvec() const override { return true; }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows, const int * counts, const int * groups,
                     int capacity, int experts, int topk, bool per_expert, cudaStream_t s) const override {
        tqllm_cuda::gsq_q2_expert_grouped(m.p, x, y, rows, m.cols, counts, groups, capacity, experts, topk, per_expert, s);
        return true;
    }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows, const int * ids,
                       int n, int topk, bool per_expert, cudaStream_t s) const override {
        if (rows <= 0 || m.rows % rows) throw std::invalid_argument("GSQ Q2_0: invalid stacked experts");
        tqllm_cuda::gsq_q2_expert_matvec(m.p, x, y, rows, m.cols, m.rows / rows, ids, n, topk, per_expert, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::gsq_q2_dequant(m.p, out, m.rows, m.cols, s);
    }
};
}
void register_gsq_q2() { register_weight_format(GGML_GSQ_Q2_0, std::make_unique<GsqQ2>()); }
}
