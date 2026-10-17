#include "quant_kernels.cuh"
#include "weight_format.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace tqllm::formats {
namespace {
// Grouped-expert entry points of the u4h kernel family. `fmt` is the family index
// (0 u4h, 1 k4, 2 k3, 4 iq3, 5 iq3x, 6 k5, 7 k6, 8-12 UD types, 13 iq4n, 14 iq4l, 15 i8h) --
// the same index this format's dense gemm()/matvec() already pass, which is what keeps grouped
// output bit-identical to a row-sliced dense call (tests/moe_device.cu).
inline bool expert_gemm_u4h(int fmt, const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                            const int * counts, const int * groups, int capacity, int experts,
                            int topk, bool per_expert_input, cudaStream_t s) {
    tqllm_cuda::k_expert_gemm(fmt, static_cast<const uint8_t *>(m.p), m.aux, x, y, rows, m.cols,
                              counts, groups, capacity, experts, topk, per_expert_input, s);
    return true;
}
inline void expert_matvec_u4h(int fmt, const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                              const int * ids, int n, int topk, bool per_expert_input,
                              cudaStream_t s) {
    tqllm_cuda::k_expert_matvec(fmt, static_cast<const uint8_t *>(m.p), m.aux, x, y, rows, m.cols,
                                ids, n, topk, per_expert_input, s);
}
// I8H (engine id 1000): int8 + FP16 scale per 16 (drafter matrices, int8 LM-head copy).
struct I8h final : WeightFormat {
    const char * name() const override { return "I8H"; }
    int u4h_id() const override { return 15; }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_gemm(15, static_cast<const uint8_t *>(m.p), m.aux, x, y, rows, m.cols, counts, groups, capacity, experts,
                                  topk, per_expert_input, stream);
        return true;
    }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_matvec(15, static_cast<const uint8_t *>(m.p), m.aux, x, y, rows, m.cols, ids, n, topk, per_expert_input, stream);
    }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        if (m.cols % 64) return false;
        tqllm_cuda::u4h_family_gemm(15, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override { return 288; }  // (cols % 64 == 0; checked in gemm_ld)  // fused vs dequant + cuBLAS crossover (GB10 gemm-bench)
    size_t row_bytes(int cols) const override { return size_t(cols); }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 16 * sizeof(__half); }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("I8H is produced by other loaders");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        if (m.cols % 64) tqllm_cuda::i8h_matvec_bf16(static_cast<const int8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows, m.cols, n, s);
        else tqllm_cuda::u4h_family_matvec_bf16(15, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        if (m.cols % 64) tqllm_cuda::i8h_matvec_bf16(static_cast<const int8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows, m.cols, n, s, ldy);
        else tqllm_cuda::u4h_family_matvec_bf16(15, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        if (m.cols % 64) tqllm_cuda::i8h_matvec(static_cast<const int8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows, m.cols, n, s);
        else tqllm_cuda::u4h_family_matvec_f32(15, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s);
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
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
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
        const void * a = up(sc.data(), sc.size() * sizeof(__half));
        if (cols % 64 == 0) tqllm_cuda::u4h_family_tune(15, static_cast<const uint8_t *>(p), a, rows, cols, s);
        return {p, 1000, rows, cols, a, i8h};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("Q8_0 is a loader; matrices run as I8H");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("Q8_0 is a loader; matrices run as I8H");
    }
};
}  // namespace
void register_i8h() {  // no weight_format() lookups here (runs inside the registry's call_once)
    auto i8h = std::make_unique<I8h>();
    auto q80 = std::make_unique<Q8_0>(i8h.get());
    register_weight_format(1000, std::move(i8h));
    register_weight_format(8, std::move(q80));
}
}  // namespace tqllm::formats
