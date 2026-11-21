#pragma once
// Exact sampling through a screened LM head (token_sampling.hpp: sample_reference / sample_bounded).
//
// The qwen35 head is two-stage: a 4-bit first pass a_r over every row (head4, u4h) and an exact int8 re-score of a few
// rows (i8h, possibly in mapped host memory). Each first-pass logit carries an error estimate
// sigma_r = sqrt(sum_s m_rs^2 |x_s|^2) (m_rs: the largest 4-bit scale / sqrt(12) of the 320-weight segment s of row r,
// |x_s|: the input's norm there; RTN rounding errors are uniform), and the screen trusts l_r in a_r +- (kSigmas sigma_r
// + slack). From that it knows, per position: the exact logits of a set of rows, an upper bound on every other row and
// bounds on their softmax mass -- enough to fix the top-k / top-p prefix (sample_bounded) without the other rows. When
// the bounds cannot decide, a round re-scores more rows (the largest upper bounds, 64, 1024, 8192), and in the last resort
// every row (the full int8 head, the TQLLM_DEBUG=full_head reference arithmetic). Whole-vocabulary draws (top_p 1, no
// top_k) screen the Gumbel scores a_r + T g_r the same way and re-score every row that could still win.
//
// Without a first pass (first-pass logits already exact: lm_head in a weight format, or no 4-bit copy) the same rounds
// run with sigma = 0 and read the exact logits instead of re-scoring.
#include "token_sampling.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace tqllm {

struct HeadSampleCol {
    int col = 0;                   // verify column (index into the head pass's logits / top-16)
    uint32_t pos = 0;              // position of the sampled token (Gumbel key)
    std::vector<uint32_t> recent;  // output before it (penalty window)
};

struct HeadSampleStats {
    uint64_t cols = 0, round1 = 0, rounds64 = 0, rounds1024 = 0, rounds8192 = 0, gumbel = 0, full = 0;
    // TQLLM_DEBUG=head_check: every sampled column also against the every-row reference, and the first-pass errors
    // |a - l| in units of the estimated sigma (rows past 3 / 4 / 5 / 6 sigma, the largest)
    uint64_t chk_cols = 0, chk_diff = 0, chk_rows = 0, chk_over[4] = {};
    double chk_max = 0;
};

class HeadSampler {
public:
    static constexpr int kSegs = 16;        // error-table segments per row
    static constexpr float kSigmas = 6.f;   // screen half-width in estimated standard deviations
    static constexpr int kMaxReq = 16;      // columns per round launch
    static constexpr int kCap = 8192;       // re-scored rows per column and round (at most)
    static constexpr int kMaxExtra = 80;    // extra exact rows per column (top-16 + penalty window)

    HeadSampler() = default;
    ~HeadSampler();
    HeadSampler(const HeadSampler &) = delete;
    HeadSampler & operator=(const HeadSampler &) = delete;

    // rows x k head, at most max_cols columns per head pass; the first pass's scales: u4h (scale, min) pairs per 32
    // weights, or k3 records (FP16 d + 16 int8 scales per 256 weights); neither: the first-pass logits are exact
    enum class FirstPass { Exact, U4h, K3 };
    void init(int rows, int k, int max_cols, FirstPass fp, const void * scales, cudaStream_t s);
    bool ready() const { return rows_ > 0; }
    bool screened() const { return sig_ != nullptr; }
    size_t device_bytes() const { return dev_bytes_; }

    // In the head pass (capturable): keep the FP16 head input of columns [0, n) for later re-scores.
    void save_x(const __half * xh, int n, cudaStream_t s);
    // After the head pass, before the stream sync: per-column screen statistics of columns [c0, c0 + n) (first-pass
    // logits a [col][rows], top-16 ids by a and their exact logits, temperature per column) and their copies to the host.
    void queue_stats(const float * a, const uint32_t * ids16, const float * ex16, int c0, int n, const float * temps,
                     cudaStream_t s);
    // After the sync: the exact draws of the given columns (one sequence's parameters). w8 / sc: the exact int8 head
    // (nullptr when unscreened). a: the same first-pass logits as queue_stats (still intact).
    std::vector<uint32_t> sample(const std::vector<HeadSampleCol> & cols, const SamplingParams & sp, const float * a,
                                 const int8_t * w8, const __half * sc, cudaStream_t s);
    // TQLLM_DEBUG=full_head: the reference draw from the exact logits of every row (a holds them).
    std::vector<uint32_t> sample_full(const std::vector<HeadSampleCol> & cols, const SamplingParams & sp, const float * a,
                                      cudaStream_t s);

    // Every row of n columns with the exact re-score's arithmetic (out [n][rows]): the full_head reference's logits.
    void rescore_all(const int8_t * w8, const __half * sc, const __half * xh, int n, float * out, cudaStream_t s) const;

    HeadSampleStats stats;

private:
    struct State;
    void round(std::vector<State *> & st, int cap, const SamplingParams & sp, const float * a, const int8_t * w8,
               const __half * sc, cudaStream_t s);
    void full_rows(State & st, const SamplingParams & sp, const int8_t * w8, const __half * sc, const float * a,
                   cudaStream_t s);
    void * dalloc(size_t bytes);
    void check(const State & x, const SamplingParams & sp, const float * a, const int8_t * w8, const __half * sc, cudaStream_t s);
    std::vector<__half> h_sig_;  // head_check: the error table on the host

    int rows_ = 0, k_ = 0, max_cols_ = 0, per_ = 1, nblk_ = 0;
    size_t dev_bytes_ = 0;
    std::vector<void *> allocs_;
    __half * sig_ = nullptr;     // [rows][kSegs]
    __half * xh_ = nullptr;      // [max_cols][k]
    float * xg_ = nullptr;       // [max_cols][kSegs]
    float * temps_ = nullptr;    // [max_cols] 1 / T
    float * part_ = nullptr;     // [nblk][max(max_cols, kMaxReq)][3]
    float * stat_ = nullptr;     // [max_cols][4]: m_out, ref, z_lo, z_hi
    void * req_ = nullptr;       // round requests
    int * hist_ = nullptr;       // [kMaxReq][kBins]
    int * cnt_ = nullptr;        // [kMaxReq] selected rows
    uint32_t * ids_ = nullptr;   // [kMaxReq][kCap + kMaxExtra]
    float * ex_ = nullptr;       // [kMaxReq][kCap + kMaxExtra] exact logits
    float * tail_ = nullptr;     // [kMaxReq][4]: m_out, -, z_lo, z_hi
    float * full_ = nullptr;     // [rows] one column's exact logits
    // pinned host mirrors
    void * hbuf_ = nullptr;
    uint32_t * h_ids16_ = nullptr;  // [max_cols][16]
    float * h_ex16_ = nullptr;      // [max_cols][16]
    float * h_stat_ = nullptr;      // [max_cols][4]
    float * h_temps_ = nullptr;     // [max_cols]
    void * h_req_ = nullptr;        // [kMaxReq] round requests
    uint32_t * h_ids_ = nullptr;    // [kMaxReq][kCap + kMaxExtra]
    float * h_ex_ = nullptr;        // [kMaxReq][kCap + kMaxExtra]
    int * h_cnt_ = nullptr;         // [kMaxReq]
    float * h_tail_ = nullptr;      // [kMaxReq][4]
    float * h_full_ = nullptr;      // [rows]
};

}  // namespace tqllm
