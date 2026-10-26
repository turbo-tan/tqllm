#include "quant_kernels.cuh"
#include "weight_format.hpp"
#include <stdexcept>

namespace tqllm::formats {
namespace {
// TQ3_4S (GGUF 46): 3-bit, 32-point rotation; repacked to the PRMT layout at load.
struct Tq3_4s final : WeightFormat {
    const char * name() const override { return "TQ3_4S"; }
    Basis basis() const override { return Basis::Tq3Rot32; }
    int max_matvec_cols() const override { return 12; }
    int max_batch_cols() const override { return 32; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        if (rows <= 0 || cols <= 0 || cols % 32 || raw.size() != size_t(rows) * (cols / 32) * 16)
            throw std::invalid_argument("TQ3_4S: invalid matrix shape or byte count");
        const void * p = up(raw.data(), raw.size());
        tqllm_cuda::tq3_repack(const_cast<void *>(p), int64_t(raw.size() / 16), s);
        tqllm_cuda::tq3_tune(p, rows, cols, s);
        cudaStreamSynchronize(s);
        return {p, 46, rows, cols, nullptr, this};
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::tq3_matvec_h(m.p, x, y, m.rows, m.cols, n, s);
    }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert, cudaStream_t s) const override {
        if (rows <= 0 || m.rows % rows || topk <= 0 || topk > m.rows / rows)
            throw std::invalid_argument("TQ3_4S: invalid stacked experts");
        tqllm_cuda::tq3_expert_matvec(m.p, x, y, rows, m.cols, ids, n, topk, per_expert, s);
    }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert, cudaStream_t s) const override {
        // the stack may hold more experts than the grouped map addresses: the qwen35 engine appends the shared
        // expert to the routed stack for decode, and a grouped launch simply never selects the extra slots.
        if (rows <= 0 || int64_t(rows) * experts > m.rows)
            throw std::invalid_argument("TQ3_4S: invalid stacked experts");
        tqllm_cuda::tq3_expert_gemm(m.p, x, y, rows, m.cols, counts, groups, capacity, experts,
                                  topk, per_expert, s);
        return true;
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::tq3_dequant_h(m.p, out, m.rows, m.cols, s);
    }
    size_t row_bytes(int cols) const override { return size_t(cols) / 32 * 16; }  // 16-byte blocks of 32, scale inline
    bool has_gemm() const override { return true; }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::tq3_gemm_h(m.p, x, y, m.rows, m.cols, n, s);
        return true;
    }
};
}  // namespace
void register_tq3_4s() { register_weight_format(46, std::make_unique<Tq3_4s>()); }
}  // namespace tqllm::formats
