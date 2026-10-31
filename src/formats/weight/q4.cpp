#include "matvec_rules.hpp"
#include "dequant.hpp"
#include "quant_kernels.cuh"
#include "weight_format.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

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
constexpr uint32_t ID_U4H = 1001, ID_W4S = 1002, ID_K4 = 1003, ID_K3 = 1004, /* 1005: retired IQ4 */ ID_IQ3 = 1006, ID_IQ3X = 1007,
                   ID_IQ4N = 1008, ID_IQ4L = 1009, ID_K5 = 1010, ID_K6 = 1011, ID_UDQ = 1012 /* 1012..1016: UD types, see Udq */;

// u4h (engine id 1001): 4-bit codes in 16-groups (element 4u + j -> byte (u / 2) * 4 + j, nibble u % 2),
// FP16 (scale, min) per 32. Produced from Q4_K (drafter) or non-uniform Q4_0.
struct U4h final : WeightFormat {
    const char * name() const override { return "U4H"; }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        return expert_gemm_u4h(0, m, x, y, rows, counts, groups, capacity, experts,
                               topk, per_expert_input, stream);
    }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input,
                       cudaStream_t stream) const override {
        expert_matvec_u4h(0, m, x, y, rows, ids, n, topk, per_expert_input, stream);
    }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_gemm(0, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override { return 640; }  // fused vs dequant + cuBLAS crossover (GB10 gemm-bench)
    size_t row_bytes(int cols) const override { return size_t(cols) / 2; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 32 * sizeof(__half2); }
    int max_batch_cols() const override { return 32; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("U4H is produced by other loaders");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::u4h_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const __half2 *>(m.aux), x, y, m.rows,
                                    m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const __half2 *>(m.aux), x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::u4h_matvec(static_cast<const uint8_t *>(m.p), static_cast<const __half2 *>(m.aux), x, y, m.rows, m.cols,
                               n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::u4h_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const __half2 *>(m.aux), out, m.rows,
                                  m.cols, s);
    }
};

// K4 (engine id 1003): native Q4_K, u4h codes + the raw 16-byte Q4_K superblock header (d, dmin, 6-bit scales / mins)
// per 256 weights: 4.5 bpw on the device instead of u4h's 5.0.
struct K4 final : WeightFormat {
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_gemm(1, static_cast<const uint8_t *>(m.p), m.aux, x, y, rows, m.cols,
                                 counts, groups, capacity, experts, topk, per_expert_input, stream);
        return true;
    }

    int u4h_id() const override { return 1; }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_matvec(1, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                   rows, m.cols, ids, n, topk, per_expert_input, stream);
    }

    const char * name() const override { return "K4"; }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_gemm(1, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override { return 640; }  // fused vs dequant + cuBLAS crossover (GB10 gemm-bench)
    int max_batch_cols() const override { return 32; }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols) / 2; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 256 * 16; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("K4 is produced by the Q4_K loader");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::k4_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                   m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::k4_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::k4_matvec(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows, m.cols,
                              n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::k4_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), out, m.rows,
                                 m.cols, s);
    }
};

// IQ3X (engine id 1007): native IQ3_XXS, per 32 weights 12 B (grid-index bytes + scale/sign word) + 4 B per 256 (d).
struct Iq3x final : WeightFormat {
    int u4h_id() const override { return 5; }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_matvec(5, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                  rows, m.cols, ids, n, topk, per_expert_input, stream);
    }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_gemm(5, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                rows, m.cols, counts, groups, capacity, experts, topk, per_expert_input, stream);
        return true;
    }
    const char * name() const override { return "IQ3X"; }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_gemm(5, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override { return 704; }  // fused vs dequant + cuBLAS crossover (GB10 gemm-bench)
    int max_batch_cols() const override { return 32; }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols) / 32 * 12; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 256 * 4; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("IQ3X is produced by the IQ3_XXS loader");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::iq3x_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                     m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::iq3x_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::iq3x_matvec(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                m.cols, n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::iq3x_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), out, m.rows,
                                   m.cols, s);
    }
};

// IQ3_XXS (GGUF 18) loader -> IQ3X: block d, qs[64], scales_and_signs[32] regrouped per 32 weights.
struct IQ3_XXS final : WeightFormat {
    const WeightFormat * iq3x;
    explicit IQ3_XXS(const WeightFormat * f) : iq3x(f) {}
    const char * name() const override { return "IQ3_XXS"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        if (cols % 256) throw std::runtime_error("IQ3_XXS needs a multiple of 256 columns");
        const int64_t nb = int64_t(rows) * cols / 256;
        std::vector<uint8_t> w3(size_t(nb) * 96), rec(size_t(nb) * 4, 0);
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 98;
            for (int ib = 0; ib < 8; ++ib) {
                std::memcpy(w3.data() + b * 96 + ib * 12, blk + 2 + ib * 8, 8);
                std::memcpy(w3.data() + b * 96 + ib * 12 + 8, blk + 66 + ib * 4, 4);
            }
            std::memcpy(rec.data() + b * 4, blk, 2);
        }
        const void * p = up(w3.data(), w3.size());
        const void * a = up(rec.data(), rec.size());
        tqllm_cuda::iq3x_tune(static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(a), rows, cols, s);
        return {p, ID_IQ3X, rows, cols, a, iq3x};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("IQ3_XXS is a loader; matrices run as IQ3X");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("IQ3_XXS is a loader; matrices run as IQ3X");
    }
};

// IQ3 (engine id 1006): native IQ3_S, per 32 weights 12 B (grid-index low bytes, sign bytes) + per 256 a 16-byte record
// (d, qh[8], scales[4], pad): 3.5 bpw on the device.
struct Iq3 final : WeightFormat {
    int u4h_id() const override { return 4; }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_matvec(4, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                  rows, m.cols, ids, n, topk, per_expert_input, stream);
    }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_gemm(4, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                rows, m.cols, counts, groups, capacity, experts, topk, per_expert_input, stream);
        return true;
    }
    const char * name() const override { return "IQ3"; }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_gemm(4, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override { return 576; }  // fused vs dequant + cuBLAS crossover (GB10 gemm-bench)
    int max_batch_cols() const override { return 32; }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols) / 32 * 12; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 256 * 16; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("IQ3 is produced by the IQ3_S loader");
    }
    // split-K or u4h kernel per shape: src/formats/weight/matvec_rules.hpp (bits 21)
    static tqllm_cuda::mvr::Pick pick(const Mat & m, int n) { return tqllm_cuda::mvr::choose(21, m.rows, m.cols, n); }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        if (const auto p = pick(m, n); p.kern == tqllm_cuda::mvr::Kern::Split16)
            tqllm_cuda::iq3_matvec_split_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y,
                                              m.rows, m.cols, n, p.pf, s);
        else
            tqllm_cuda::iq3_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                        m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        if (const auto p = pick(m, n); p.kern == tqllm_cuda::mvr::Kern::Split16)
            tqllm_cuda::iq3_matvec_split_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y,
                                              m.rows, m.cols, n, p.pf, s, ldy);
        else
            tqllm_cuda::iq3_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                        m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        if (const auto p = pick(m, n); p.kern == tqllm_cuda::mvr::Kern::Split16)
            tqllm_cuda::iq3_matvec_split(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y,
                                         m.rows, m.cols, n, p.pf, s);
        else
            tqllm_cuda::iq3_matvec(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                   m.cols, n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::iq3_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), out, m.rows,
                                  m.cols, s);
    }
};

// IQ3_S (GGUF 21) loader -> IQ3: block d, qs[64], qh[8], signs[32], scales[4] regrouped per 32 weights.
struct IQ3_S final : WeightFormat {
    const WeightFormat * iq3;
    explicit IQ3_S(const WeightFormat * f) : iq3(f) {}
    const char * name() const override { return "IQ3_S"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        if (cols % 256) throw std::runtime_error("IQ3_S needs a multiple of 256 columns");
        const int64_t nb = int64_t(rows) * cols / 256;
        std::vector<uint8_t> w3(size_t(nb) * 96), rec(size_t(nb) * 16, 0);
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 110;
            for (int ib = 0; ib < 8; ++ib) {
                std::memcpy(w3.data() + b * 96 + ib * 12, blk + 2 + ib * 8, 8);
                std::memcpy(w3.data() + b * 96 + ib * 12 + 8, blk + 74 + ib * 4, 4);
            }
            std::memcpy(rec.data() + b * 16, blk, 2);
            std::memcpy(rec.data() + b * 16 + 2, blk + 66, 8);
            std::memcpy(rec.data() + b * 16 + 10, blk + 106, 4);
        }
        const void * p = up(w3.data(), w3.size());
        const void * a = up(rec.data(), rec.size());
        tqllm_cuda::iq3_tune(static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(a), rows, cols, s);
        return {p, ID_IQ3, rows, cols, a, iq3};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("IQ3_S is a loader; matrices run as IQ3");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("IQ3_S is a loader; matrices run as IQ3");
    }
};

// IQ4N (engine id 1008): IQ4_XS in natural nibble order (byte j of a 32-weight block = weights 2j, 2j + 1) + the
// unchanged 8-byte record per 256 weights (d, scales_h, scales_l), on the 16-row split-K kernel (<= 16 columns; wider
// batches dequantise to FP16). 4.25 bpw on the device, same as the file.
struct Iq4n final : WeightFormat {
    int u4h_id() const override { return 13; }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_matvec(13, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                  rows, m.cols, ids, n, topk, per_expert_input, stream);
    }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_gemm(13, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                rows, m.cols, counts, groups, capacity, experts, topk, per_expert_input, stream);
        return true;
    }
    const char * name() const override { return "IQ4N"; }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_gemm(13, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override { return 416; }  // fused vs dequant + cuBLAS crossover (GB10 gemm-bench)
    int max_batch_cols() const override { return 16; }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols) / 2; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 256 * 8; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("IQ4N is produced by the IQ4_XS loader");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_matvec_bf16(13, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_matvec_bf16(13, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_matvec_f32(13, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::iq4n_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), out, m.rows,
                                   m.cols, s);
    }
};

// IQ4L (engine id 1009): IQ4_NL in natural nibble order (16 B per 32 weights) + one FP16 scale per 32 weights (aux).
struct Iq4l final : WeightFormat {
    int u4h_id() const override { return 14; }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_matvec(14, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                  rows, m.cols, ids, n, topk, per_expert_input, stream);
    }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_gemm(14, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                rows, m.cols, counts, groups, capacity, experts, topk, per_expert_input, stream);
        return true;
    }
    const char * name() const override { return "IQ4L"; }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_gemm(14, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override { return 896; }  // fused vs dequant + cuBLAS crossover (GB10 gemm-bench)
    int max_batch_cols() const override { return 16; }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols) / 2; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 32 * 2; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("IQ4L is produced by the IQ4_NL loader");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_matvec_bf16(14, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_matvec_bf16(14, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_matvec_f32(14, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::iq4l_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), out, m.rows,
                                   m.cols, s);
    }
};
// IQ4_NL (GGUF 20) loader -> IQ4L: block d, qs[16] (weight j low nibble of qs[j], weight 16 + j high nibble).
struct IQ4_NL final : WeightFormat {
    const WeightFormat * iq4l;
    explicit IQ4_NL(const WeightFormat * f) : iq4l(f) {}
    const char * name() const override { return "IQ4_NL"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        if (cols % 128) {  // the IQ4L kernels need 128-column rows: the same values as Q8_0 (block scale x int8 table value)
            if (cols % 32) throw std::runtime_error("IQ4_NL needs a multiple of 32 columns");
            static const int8_t kv[16] = {
#include "kvalues_iq4nl.inc"
            };
            const int64_t nb = int64_t(rows) * cols / 32;
            std::vector<char> q8(size_t(nb) * 34);
            for (int64_t b = 0; b < nb; ++b) {
                const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 18;
                char * o = q8.data() + b * 34; std::memcpy(o, blk, 2);
                for (int j = 0; j < 16; ++j) { o[2 + j] = char(kv[blk[2 + j] & 0xF]); o[2 + 16 + j] = char(kv[blk[2 + j] >> 4]); }
            }
            const WeightFormat * q80 = weight_format(8);
            if (!q80) throw std::runtime_error("IQ4_NL: Q8_0 format not registered");
            return q80->load(q8, rows, cols, up, s);
        }
        const int64_t nb = int64_t(rows) * cols / 32;
        std::vector<uint8_t> w(size_t(nb) * 16);
        std::vector<uint16_t> d(static_cast<size_t>(nb));
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 18;
            std::memcpy(&d[size_t(b)], blk, 2);
            const uint8_t * qs = blk + 2;
            auto q = [&](int e) { return e < 16 ? qs[e] & 0xF : qs[e - 16] >> 4; };
            for (int j = 0; j < 16; ++j) w[size_t(b) * 16 + j] = uint8_t(q(2 * j) | (q(2 * j + 1) << 4));
        }
        const void * p = up(w.data(), w.size());
        const void * a = up(d.data(), d.size() * 2);
        tqllm_cuda::u4h_family_tune(14, static_cast<const uint8_t *>(p), a, rows, cols, s);
        return {p, ID_IQ4L, rows, cols, a, iq4l};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("IQ4_NL is a loader; matrices run as IQ4L");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("IQ4_NL is a loader; matrices run as IQ4L");
    }
};
// IQ4_XS (GGUF 23) loader -> IQ4N.
struct IQ4_XS final : WeightFormat {
    const WeightFormat * iq4n;
    explicit IQ4_XS(const WeightFormat * f) : iq4n(f) {}
    const char * name() const override { return "IQ4_XS"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        if (cols % 256) throw std::runtime_error("IQ4_XS needs a multiple of 256 columns");
        const int64_t nb = int64_t(rows) * cols / 256;
        // natural nibble order for the split-K kernel; the record stays as stored
        std::vector<uint8_t> w(size_t(nb) * 128), rec(size_t(nb) * 8);
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 136;
            std::memcpy(rec.data() + b * 8, blk, 8);
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t * qs = blk + 8 + ib * 16;
                auto q = [&](int e) { return e < 16 ? qs[e] & 0xF : qs[e - 16] >> 4; };
                for (int j = 0; j < 16; ++j) w[size_t(b) * 128 + ib * 16 + j] = uint8_t(q(2 * j) | (q(2 * j + 1) << 4));
            }
        }
        const void * p = up(w.data(), w.size());
        const void * a = up(rec.data(), rec.size());
        tqllm_cuda::u4h_family_tune(13, static_cast<const uint8_t *>(p), a, rows, cols, s);
        return {p, ID_IQ4N, rows, cols, a, iq4n};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("IQ4_XS is a loader; matrices run as IQ4N");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("IQ4_XS is a loader; matrices run as IQ4N");
    }
};

// K3 (engine id 1004): native Q3_K, 3-bit planes (12 B per 32 weights) + per 256 weights a 20-byte record (16 int8
// scales, FP16 d): 3.6 bpw on the device. Matvec / dequant: the u4h kernels with FMT 2.
struct K3 final : WeightFormat {
    const char * name() const override { return "K3"; }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        return expert_gemm_u4h(2, m, x, y, rows, counts, groups, capacity, experts,
                               topk, per_expert_input, stream);
    }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input,
                       cudaStream_t stream) const override {
        expert_matvec_u4h(2, m, x, y, rows, ids, n, topk, per_expert_input, stream);
    }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_gemm(2, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override { return 576; }  // fused vs dequant + cuBLAS crossover (GB10 gemm-bench)
    int max_batch_cols() const override { return 32; }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols) / 32 * 12; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 256 * 20; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("K3 is produced by the Q3_K loader");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::k3_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                   m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::k3_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::k3_matvec(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows, m.cols,
                              n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::k3_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), out, m.rows,
                                 m.cols, s);
    }
};

// Q3_K (GGUF 11) loader -> K3: codes value + 4 regrouped into the K3 planes, scales and d copied.
struct Q3_K final : WeightFormat {
    const WeightFormat * k3;
    explicit Q3_K(const WeightFormat * f) : k3(f) {}
    const char * name() const override { return "Q3_K"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        if (cols % 256) throw std::runtime_error("Q3_K needs a multiple of 256 columns");
        const int64_t nb = int64_t(rows) * cols / 256;
        std::vector<uint8_t> w3(size_t(nb) * 96, 0), rec(size_t(nb) * 20, 0);
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 110;
            q3k_scales(blk + 96, reinterpret_cast<int8_t *>(rec.data() + b * 20));
            std::memcpy(rec.data() + b * 20 + 16, blk + 108, 2);
            for (int e = 0; e < 256; ++e) {
                const int n = e / 128, j = (e % 128) / 32, w = e % 32;
                const int code = ((blk[32 + 32 * n + w] >> (2 * j)) & 3) | (((blk[w] >> (4 * n + j)) & 1) << 2);
                // code = Q3_K value + 4 (high bit set means value 0..3, clear means -4..-1)
                uint8_t * o = w3.data() + b * 96 + (e / 32) * 12;  // the 32-weight block's L0, L1, H words
                const int g = (e % 32) / 16, ei = e % 16, u = ei / 4, jj = ei % 4;
                o[g * 4 + jj] |= uint8_t((code & 3) << (2 * u));
                o[8 + jj] |= uint8_t((code >> 2) << (u + 4 * g));
            }
        }
        const void * p = up(w3.data(), w3.size());
        const void * a = up(rec.data(), rec.size());
        tqllm_cuda::k3_tune(static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(a), rows, cols, s);
        return {p, ID_K3, rows, cols, a, k3};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("Q3_K is a loader; matrices run as K3");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("Q3_K is a loader; matrices run as K3");
    }
};

// W4S (engine id 1002): symmetric 4-bit in the u4h code layout, FP16 scale per 128 (the W4A16 group-128 layout).
struct W4s final : WeightFormat {
    const char * name() const override { return "W4S"; }
    size_t row_bytes(int cols) const override { return size_t(cols) / 2; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 128 * sizeof(__half); }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("W4S is produced by the Q4_0 loader");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::w4s_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows,
                                    m.cols, n, s);
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::w4s_matvec(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), x, y, m.rows, m.cols,
                               n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::w4s_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), out, m.rows,
                                  m.cols, s);
    }
};

// Q4_0 (GGUF 2) loader: exact repack to W4S when the 4 blocks of every 128-group share d (W4A16 group 128), else
// to u4h (scale d, min -8 d per 32).
struct Q4_0 final : WeightFormat {
    const WeightFormat * w4s, * u4h;
    Q4_0(const WeightFormat * a, const WeightFormat * b) : w4s(a), u4h(b) {}
    const char * name() const override { return "Q4_0"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        const int64_t nb = int64_t(rows) * cols / 32;
        std::vector<uint8_t> w4(size_t(nb) * 16, 0);
        std::vector<__half2> sm(static_cast<size_t>(nb));
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 18;
            __half hd;
            std::memcpy(&hd, blk, 2);
            const float d = __half2float(hd);
            sm[b] = __floats2half2_rn(d, -8.f * d);
            uint8_t * out = w4.data() + b * 16;
            for (int e = 0; e < 32; ++e) {
                const uint8_t q = e < 16 ? blk[2 + e] & 0xF : blk[2 + e - 16] >> 4;
                const int grp = e / 16, ei = e % 16, u = ei / 4, j = ei % 4;
                out[grp * 8 + (u / 2) * 4 + j] |= q << ((u % 2) * 4);
            }
        }
        bool g128 = cols % 128 == 0;
        for (int64_t b = 0; g128 && b < nb; b += 4)
            for (int i = 1; i < 4; ++i) g128 = g128 && std::memcmp(raw.data() + b * 18, raw.data() + (b + i) * 18, 2) == 0;
        const void * p = up(w4.data(), w4.size());
        if (g128) {
            std::vector<__half> sc(static_cast<size_t>(nb / 4));
            for (int64_t b = 0; b < nb; b += 4) sc[b / 4] = __low2half(sm[b]);
            return {p, ID_W4S, rows, cols, up(sc.data(), sc.size() * sizeof(__half)), w4s};
        }
        return {p, ID_U4H, rows, cols, up(sm.data(), sm.size() * sizeof(__half2)), u4h};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("Q4_0 is a loader; matrices run as W4S or U4H");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("Q4_0 is a loader; matrices run as W4S or U4H");
    }
};
// Q4_K (GGUF 12) loader -> K4: u4h-layout codes + the raw 16-byte superblock header.
struct Q4_K final : WeightFormat {
    const WeightFormat * k4;
    explicit Q4_K(const WeightFormat * c) : k4(c) {}
    const char * name() const override { return "Q4_K"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        if (cols % 256) throw std::runtime_error("Q4_K needs a multiple of 256 columns");
        const int64_t nb = int64_t(rows) * cols / 256;
        std::vector<uint8_t> w4(size_t(nb) * 128, 0), kh(size_t(nb) * 16);
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 144;
            std::memcpy(kh.data() + b * 16, blk, 16);
            uint8_t * out = w4.data() + b * 128;
            for (int e = 0; e < 256; ++e) {  // element e: byte 16 + (e / 64) * 32 + e % 32, high nibble for the second 32
                const uint8_t q = (blk[16 + (e / 64) * 32 + e % 32] >> ((e / 32) % 2 * 4)) & 0xF;
                const int grp = e / 16, ei = e % 16, u = ei / 4, j = ei % 4;
                out[grp * 8 + (u / 2) * 4 + j] |= q << ((u % 2) * 4);
            }
        }
        const void * p = up(w4.data(), w4.size());
        const void * a = up(kh.data(), kh.size());
        tqllm_cuda::k4_tune(static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(a), rows, cols, s);
        return {p, ID_K4, rows, cols, a, k4};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("Q4_K is a loader; matrices run as U4H");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("Q4_K is a loader; matrices run as U4H");
    }
};
}  // namespace

// K6 (engine id 1011): native Q6_K. Per 32 weights 16 nibble bytes in K4 order + 2 words of 2-bit highs; aux = per 256
// weights the 20-byte K3 record (16 int8 scales, FP16 d); value d * scale * (code - 32).
struct K6 final : WeightFormat {
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_gemm(7, static_cast<const uint8_t *>(m.p), m.aux, x, y, rows, m.cols,
                                 counts, groups, capacity, experts, topk, per_expert_input, stream);
        return true;
    }

    int u4h_id() const override { return 7; }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_matvec(7, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                   rows, m.cols, ids, n, topk, per_expert_input, stream);
    }

    const char * name() const override { return "K6"; }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_gemm(7, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override { return 256; }  // fused vs dequant + cuBLAS crossover (GB10 gemm-bench)
    int max_batch_cols() const override { return 32; }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols) / 32 * 24; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 256 * 20; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("K6 is produced by the Q6_K loader");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::k6_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                   m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::k6_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::k6_matvec(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows, m.cols,
                              n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::k6_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), out, m.rows,
                                 m.cols, s);
    }
};
// Q6_K (GGUF 14) loader -> K6: block ql[128], qh[64], scales[16], d; per 128 weights (h) element l + 32 q of ql / qh.
struct Q6_K final : WeightFormat {
    const WeightFormat * k6;
    explicit Q6_K(const WeightFormat * c) : k6(c) {}
    const char * name() const override { return "Q6_K"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        if (cols % 256) throw std::runtime_error("Q6_K needs a multiple of 256 columns");
        const int64_t nb = int64_t(rows) * cols / 256;
        std::vector<uint8_t> w6(size_t(nb) * 192, 0), kr(size_t(nb) * 20, 0);
        uint8_t v[256];
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 210;
            std::memcpy(kr.data() + b * 20, blk + 192, 18);
            const uint8_t * ql = blk, * qh = blk + 128;
            for (int h = 0; h < 2; ++h, ql += 64, qh += 32)
                for (int l = 0; l < 32; ++l) {
                    v[128 * h + l] = uint8_t((ql[l] & 0xF) | ((qh[l] & 3) << 4));
                    v[128 * h + l + 32] = uint8_t((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4));
                    v[128 * h + l + 64] = uint8_t((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4));
                    v[128 * h + l + 96] = uint8_t((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
                }
            for (int e = 0; e < 256; ++e) {
                uint8_t * out = w6.data() + size_t(b) * 192 + (e / 32) * 24;
                const int h = (e % 32) / 16, ei = e % 16, u = ei / 4, j = ei % 4;
                out[h * 8 + (u / 2) * 4 + j] |= uint8_t((v[e] & 0xF) << ((u % 2) * 4));
                out[16 + 4 * h + (2 * ei) / 8] |= uint8_t(((v[e] >> 4) & 3) << ((2 * ei) % 8));
            }
        }
        const void * p = up(w6.data(), w6.size());
        const void * a = up(kr.data(), kr.size());
        tqllm_cuda::k6_tune(static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(a), rows, cols, s);
        return {p, ID_K6, rows, cols, a, k6};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("Q6_K is a loader; matrices run as K6");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("Q6_K is a loader; matrices run as K6");
    }
};
// K5 (engine id 1010): native Q5_K. Per 32 weights 16 nibble bytes in K4 order + one word of high code bits; aux = the
// 16-byte Q5_K header (d, dmin, scales) per 256 weights per row.
struct K5 final : WeightFormat {
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_gemm(6, static_cast<const uint8_t *>(m.p), m.aux, x, y, rows, m.cols,
                                 counts, groups, capacity, experts, topk, per_expert_input, stream);
        return true;
    }

    int u4h_id() const override { return 6; }
    bool has_expert_matvec() const override { return true; }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_matvec(6, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                   rows, m.cols, ids, n, topk, per_expert_input, stream);
    }

    const char * name() const override { return "K5"; }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_gemm(6, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override { return 352; }  // fused vs dequant + cuBLAS crossover (GB10 gemm-bench)
    int max_batch_cols() const override { return 32; }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols) / 32 * 20; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 256 * 16; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error("K5 is produced by the Q5_K loader");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::k5_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                   m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::k5_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::k5_matvec(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows, m.cols,
                              n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::k5_dequant_h(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), out, m.rows,
                                 m.cols, s);
    }
};
// Q5_K (GGUF 13) loader -> K5: block d, dmin, scales[12], qh[32], qs[128]; element e: nibble of qs[(e / 64) * 32 + e % 32]
// (high nibble for the second 32), high bit qh[e % 32] >> (e / 32).
struct Q5_K final : WeightFormat {
    const WeightFormat * k5;
    explicit Q5_K(const WeightFormat * c) : k5(c) {}
    const char * name() const override { return "Q5_K"; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        if (cols % 256) throw std::runtime_error("Q5_K needs a multiple of 256 columns");
        const int64_t nb = int64_t(rows) * cols / 256;
        std::vector<uint8_t> w5(size_t(nb) * 160, 0), kh(size_t(nb) * 16);
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * 176;
            std::memcpy(kh.data() + b * 16, blk, 16);
            const uint8_t * qh = blk + 16, * qs = blk + 48;
            for (int e = 0; e < 256; ++e) {
                const uint8_t q = (qs[(e / 64) * 32 + e % 32] >> ((e / 32) % 2 * 4)) & 0xF;
                const uint32_t hbit = (qh[e % 32] >> (e / 32)) & 1;
                uint8_t * out = w5.data() + size_t(b) * 160 + (e / 32) * 20;
                const int h = (e % 32) / 16, ei = e % 16, u = ei / 4, j = ei % 4;
                out[h * 8 + (u / 2) * 4 + j] |= q << ((u % 2) * 4);
                out[16 + (16 * h + ei) / 8] |= uint8_t(hbit << ((16 * h + ei) % 8));
            }
        }
        const void * p = up(w5.data(), w5.size());
        const void * a = up(kh.data(), kh.size());
        tqllm_cuda::k5_tune(static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(a), rows, cols, s);
        return {p, ID_K5, rows, cols, a, k5};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error("Q5_K is a loader; matrices run as K5");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error("Q5_K is a loader; matrices run as K5");
    }
};
// UD low-bit types (engine ids 1012..1016, kernel fmt 8..12) on the u4h kernel: per 32 weights the GGUF block's code
// bytes, per row and 256 weights a record of its scalars (src/formats/weight/quant_kernels.cuh, udq_*).
struct Udq final : WeightFormat {
    int u4h_id() const override { return fmt == 10 ? 10 : -1; }
    int gemm_f32_id() const override { return fmt; }
    bool has_expert_matvec() const override { return true; }  // every UD format (grouped kernels cover u4h fmt 0..15)
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                       const int * ids, int n, int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_matvec(fmt, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                  rows, m.cols, ids, n, topk, per_expert_input, stream);
    }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert_input, cudaStream_t stream) const override {
        tqllm_cuda::k_expert_gemm(fmt, static_cast<const uint8_t *>(m.p), m.aux, x, y,
                                rows, m.cols, counts, groups, capacity, experts, topk, per_expert_input, stream);
        return true;
    }
    const char * nm;
    int fmt, bpb, rb;
    Udq(const char * n, int f, int b, int r) : nm(n), fmt(f), bpb(b), rb(r) {}
    const char * name() const override { return nm; }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        return gemm_ld(m, x, y, n, m.rows, s);
    }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::u4h_family_gemm(fmt, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, ldy);
        return true;
    }
    int fused_gemm_cols() const override {  // crossover vs dequant + cuBLAS (GB10 gemm-bench), fmt 8..12
        static const int c[5] = {832, 768, 576, 384, 768};
        return c[fmt - 8];
    }
    int max_batch_cols() const override { return 32; }
    Basis basis() const override { return Basis::Plain; }
    size_t row_bytes(int cols) const override { return size_t(cols) / 32 * bpb; }
    size_t aux_row_bytes(int cols) const override { return size_t(cols) / 256 * rb; }
    Mat load(const std::vector<char> &, int, int, const Upload &, cudaStream_t) const override {
        throw std::runtime_error(std::string(nm) + " is produced by its GGUF loader");
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        tqllm_cuda::udq_matvec_bf16(fmt, static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                    m.cols, n, s);
    }
    bool has_matvec_ld() const override { return true; }
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        tqllm_cuda::udq_matvec_bf16(fmt, static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                                    m.cols, n, s, ldy);
        return true;
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override {
        tqllm_cuda::udq_matvec(fmt, static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), x, y, m.rows,
                               m.cols, n, s);
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        tqllm_cuda::udq_dequant_h(fmt, static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), out, m.rows,
                                  m.cols, s);
    }
};
// GGUF loaders for the UD types: split each 256-weight block into per-32 code bytes and the record.
struct UdqLoader final : WeightFormat {
    const WeightFormat * fmt;
    const char * nm;
    uint32_t id;
    int gguf_type, block_bytes, bpb, rb;
    UdqLoader(const WeightFormat * f, const char * n, uint32_t i, int t, int bb, int b, int r)
        : fmt(f), nm(n), id(i), gguf_type(t), block_bytes(bb), bpb(b), rb(r) {}
    const char * name() const override { return nm; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        if (cols % 256) throw std::runtime_error(std::string(nm) + " needs a multiple of 256 columns");
        const int64_t nb = int64_t(rows) * cols / 256;
        std::vector<uint8_t> w(size_t(nb) * 8 * bpb, 0), rec(size_t(nb) * rb, 0);
        for (int64_t b = 0; b < nb; ++b) {
            const auto * blk = reinterpret_cast<const uint8_t *>(raw.data()) + b * block_bytes;
            uint8_t * o = w.data() + size_t(b) * 8 * bpb, * r = rec.data() + size_t(b) * rb;
            switch (gguf_type) {
                case 16:  // IQ2_XXS: d, qs[64] (8 bytes per 32)
                    std::memcpy(r, blk, 2);
                    std::memcpy(o, blk + 2, 64);
                    break;
                case 17:  // IQ2_XS: d, qs[64] (8 bytes per 32), scales[8]
                    std::memcpy(r, blk, 2);
                    std::memcpy(r + 4, blk + 66, 8);
                    std::memcpy(o, blk + 2, 64);
                    break;
                case 22:  // IQ2_S: d, qs[32] grid low bytes, signs[32], qh[8], scales[8]
                    std::memcpy(r, blk, 2);
                    std::memcpy(r + 2, blk + 66, 16);
                    for (int ib = 0; ib < 8; ++ib) {
                        std::memcpy(o + ib * 8, blk + 2 + ib * 4, 4);
                        std::memcpy(o + ib * 8 + 4, blk + 34 + ib * 4, 4);
                    }
                    break;
                case 19:  // IQ1_S: d, qs[32] (4 bytes per 32), qh[8] x uint16
                    std::memcpy(r, blk, 2);
                    std::memcpy(r + 2, blk + 34, 16);
                    std::memcpy(o, blk + 2, 32);
                    break;
                case 10:  // Q2_K: scales[16], qs[64], d, dmin; code of weight e: qs[(e / 128) * 32 + e % 32] >> 2 ((e % 128) / 32)
                    std::memcpy(r, blk, 16);
                    std::memcpy(r + 16, blk + 80, 4);
                    for (int e = 0; e < 256; ++e) {
                        const uint32_t q = (blk[16 + (e / 128) * 32 + e % 32] >> (2 * ((e % 128) / 32))) & 3;
                        const int word = (e / 32) * 2 + (e % 32) / 16, ei = e % 16;
                        o[word * 4 + (2 * ei) / 8] |= uint8_t(q << ((2 * ei) % 8));
                    }
                    break;
            }
        }
        const void * p = up(w.data(), w.size());
        const void * a = up(rec.data(), rec.size());
        tqllm_cuda::udq_tune(int(id - ID_UDQ) + 8, static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(a), rows, cols, s);
        return {p, id, rows, cols, a, fmt};
    }
    void matvec(const Mat &, const __half *, __nv_bfloat16 *, int, cudaStream_t) const override {
        throw std::runtime_error(std::string(nm) + " is a loader");
    }
    void dequant_h(const Mat &, __half *, cudaStream_t) const override {
        throw std::runtime_error(std::string(nm) + " is a loader");
    }
};
void register_q4() {  // no weight_format() lookups here: registration runs inside the registry's call_once
    auto u4h = std::make_unique<U4h>();
    auto w4s = std::make_unique<W4s>();
    auto q40 = std::make_unique<Q4_0>(w4s.get(), u4h.get());
    auto k4 = std::make_unique<K4>();
    auto k3 = std::make_unique<K3>();
    auto iq3 = std::make_unique<Iq3>();
    auto iq3x = std::make_unique<Iq3x>();
    auto iq3xxs = std::make_unique<IQ3_XXS>(iq3x.get());
    auto iq3s = std::make_unique<IQ3_S>(iq3.get());
    auto iq4n = std::make_unique<Iq4n>();
    auto iq4xs = std::make_unique<IQ4_XS>(iq4n.get());
    auto iq4l = std::make_unique<Iq4l>();
    auto iq4nl = std::make_unique<IQ4_NL>(iq4l.get());
    auto q3k = std::make_unique<Q3_K>(k3.get());
    auto q4k = std::make_unique<Q4_K>(k4.get());
    auto k5 = std::make_unique<K5>();
    auto k6 = std::make_unique<K6>();
    auto q6k = std::make_unique<Q6_K>(k6.get());
    auto q5k = std::make_unique<Q5_K>(k5.get());
    register_weight_format(ID_U4H, std::move(u4h));
    register_weight_format(ID_W4S, std::move(w4s));
    register_weight_format(ID_K4, std::move(k4));
    register_weight_format(ID_K3, std::move(k3));
    register_weight_format(ID_IQ4N, std::move(iq4n));
    register_weight_format(ID_IQ4L, std::move(iq4l));
    register_weight_format(ID_IQ3, std::move(iq3));
    register_weight_format(ID_IQ3X, std::move(iq3x));
    register_weight_format(18, std::move(iq3xxs));
    register_weight_format(21, std::move(iq3s));
    register_weight_format(23, std::move(iq4xs));
    register_weight_format(20, std::move(iq4nl));
    register_weight_format(11, std::move(q3k));
    register_weight_format(2, std::move(q40));
    register_weight_format(12, std::move(q4k));
    register_weight_format(ID_K5, std::move(k5));
    register_weight_format(13, std::move(q5k));
    register_weight_format(ID_K6, std::move(k6));
    register_weight_format(14, std::move(q6k));
    struct UdType { const char * fmt_name, * gguf_name; int gguf_type, block_bytes, bpb, rb; };
    static const UdType ud[] = {{"IQ2X", "IQ2_XXS", 16, 66, 8, 4}, {"IQ2W", "IQ2_XS", 17, 74, 8, 12},
                                {"IQ2S", "IQ2_S", 22, 82, 8, 20}, {"IQ1S", "IQ1_S", 19, 50, 4, 20},
                                {"K2", "Q2_K", 10, 84, 8, 20}};
    for (int i = 0; i < 5; ++i) {
        const auto & t = ud[i];
        auto f = std::make_unique<Udq>(t.fmt_name, 8 + i, t.bpb, t.rb);
        auto l = std::make_unique<UdqLoader>(f.get(), t.gguf_name, ID_UDQ + i, t.gguf_type, t.block_bytes, t.bpb, t.rb);
        register_weight_format(ID_UDQ + i, std::move(f));
        register_weight_format(uint32_t(t.gguf_type), std::move(l));
    }
}
}  // namespace tqllm::formats

namespace tqllm {
bool WeightFormat::expert_gemm_i8(const Mat & m, const int8_t * xq, const float * xs, const int16_t * xsum, __nv_bfloat16 * y, int rows,
                                   const int * counts, const int * groups, int capacity, int experts, int topk, bool per_expert_input,
                                   cudaStream_t stream, int mode) const {
    const int id = u4h_id();
    if (id != 1 && id != 4 && id != 5 && id != 6 && id != 7 && id != 10 && id != 13 && id != 14) return false;
    // TQFN_PREFILL_INT8_FMT: comma list of u4h ids allowed on the int8 path (default all); e.g. 13,14 = IQ4 formats only
    static const std::string allow = std::getenv("TQFN_PREFILL_INT8_FMT") ? std::getenv("TQFN_PREFILL_INT8_FMT") : "";
    if (!allow.empty() && ("," + allow + ",").find("," + std::to_string(id) + ",") == std::string::npos) return false;
    tqllm_cuda::k_expert_gemm_i8(id, static_cast<const uint8_t *>(m.p), m.aux, xq, xs, xsum, y, rows, m.cols, counts, groups,
                                 capacity, experts, topk, per_expert_input, stream, mode);
    return true;
}
}  // namespace tqllm
