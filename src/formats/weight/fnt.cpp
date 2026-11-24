#include "weight_format.hpp"
#include "gguf.hpp"
#include "fnt.cuh"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm::formats {
namespace {
// FNT-k (docs/reference/formats.md "FNT"; GGUF types 330-333 + tqllm.fnt.code = "fnt-c2w16-v1"): per 640 weights an FP16
// scale + 5 tail-biting trellis streams of k * 8 bytes (128 weights each). Split at load into streams [rows][cols / 128]
// [k * 8 B] and scales [rows][cols / 640], so a row range is a matrix of its own (expert slots: row_bytes / aux_row_bytes).
// Plain basis for the engine: the kernels rotate their inputs (fnt.cu).
struct Fnt final : WeightFormat {
    explicit Fnt(uint32_t type) : type_(type), kb_(gguf_fnt_kb(type)), name_("FNT" + std::to_string(gguf_fnt_kb(type))) {}
    const char * name() const override { return name_.c_str(); }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols / 128) * 8 * kb_; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols / 640) * 2; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        const size_t gb = 2 + 40 * size_t(kb_);
        if (rows <= 0 || cols <= 0 || cols % 640 || raw.size() != size_t(rows) * (cols / 640) * gb)
            throw std::invalid_argument(name_ + ": invalid matrix shape or byte count");
        const size_t ng = size_t(rows) * (cols / 640), sb = 40 * size_t(kb_);
        std::vector<char> streams(ng * sb);
        std::vector<char> sc(ng * 2);
        for (size_t g = 0; g < ng; ++g) {
            std::memcpy(sc.data() + 2 * g, raw.data() + g * gb, 2);
            std::memcpy(streams.data() + g * sb, raw.data() + g * gb + 2, sb);
        }
        const void * p = up(streams.data(), streams.size());
        const void * a = up(sc.data(), sc.size());
        return {p, type_, rows, cols, a, this};
    }
    static const __half * s(const Mat & m) { return static_cast<const __half *>(m.aux); }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t st) const override {
        tqllm_cuda::fnt_expert_matvec(kb_, m.p, s(m), x, y, m.rows, m.cols, 1, nullptr, n, 1, true, st);
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t st) const override {
        tqllm_cuda::fnt_matvec_f32(kb_, m.p, s(m), x, y, m.rows, m.cols, n, st);
    }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows, const int * ids, int n, int topk,
                       bool per_expert, cudaStream_t st) const override {
        if (rows <= 0 || m.rows % rows) throw std::invalid_argument(name_ + ": invalid stacked experts");
        tqllm_cuda::fnt_expert_matvec(kb_, m.p, s(m), x, y, rows, m.cols, m.rows / rows, ids, n, topk, per_expert, st);
    }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows, const int * counts, const int * groups,
                     int capacity, int experts, int topk, bool per_expert, cudaStream_t st) const override {
        tqllm_cuda::fnt_expert_grouped(kb_, m.p, s(m), x, y, rows, m.cols, counts, groups, capacity, experts, topk, per_expert, st);
        return true;
    }
    bool has_expert_dedup() const override { return true; }
    void expert_matvec_dedup(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows, const int * dmap, int groups,
                             cudaStream_t st) const override {
        tqllm_cuda::fnt_expert_dedup(kb_, m.p, s(m), x, y, rows, m.cols, dmap, groups, st);
    }
    void expert_matvec_dedup2(const Mat & a, const Mat & b, const __half * x, __nv_bfloat16 * ya, __nv_bfloat16 * yb, int rows,
                              const int * dmap, int groups, cudaStream_t st) const override {
        tqllm_cuda::fnt_expert_dedup(kb_, a.p, s(a), x, ya, rows, a.cols, dmap, groups, st);
        tqllm_cuda::fnt_expert_dedup(kb_, b.p, s(b), x, yb, rows, b.cols, dmap, groups, st);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t st) const override {
        tqllm_cuda::fnt_dequant_h(kb_, m.p, s(m), out, m.rows, m.cols, st);
    }
  private:
    uint32_t type_;
    int kb_;
    std::string name_;
};
}  // namespace
void register_fnt() {
    for (uint32_t t = GGML_FNT4; t <= GGML_FNT7; ++t) register_weight_format(t, std::make_unique<Fnt>(t));
}
}  // namespace tqllm::formats
