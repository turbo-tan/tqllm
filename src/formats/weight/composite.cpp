#include "weight_format.hpp"
#include "debug.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tqllm {
namespace {
constexpr uint32_t ID_COMPOSITE = 1100;

void ck(cudaError_t e, const char * what) {  // a failed fork / join (e.g. during graph capture) stops here, not later
    if (e != cudaSuccess) throw std::runtime_error(std::string("composite: ") + what + ": " + cudaGetErrorString(e));
}

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
    // Parts 1.. run on side streams forked from s and joined back (graph-capturable fork / join): every part reads the
    // same input and writes its own row block, so results are those of the sequential order, bit for bit. Small parts
    // (e.g. a Q5_K qkv block beside an IQ4_XS gate block) then overlap instead of each paying its own ramp-up / tail.
    // TQLLM_DEBUG=composite_serial runs the parts one after another on the main stream (A/B).
    // one small pool of side streams / events shared by every composite (they run one at a time on the main stream)
    struct Pool {
        cudaEvent_t fork = nullptr;
        std::vector<cudaStream_t> side;
        std::vector<cudaEvent_t> joins;
    };
    static Pool & pool(size_t n) {
        static Pool p;
        if (!p.fork) ck(cudaEventCreateWithFlags(&p.fork, cudaEventDisableTiming), "fork event");
        while (p.side.size() < n) {
            cudaStream_t st;
            cudaEvent_t ev;
            ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "side stream");
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "join event");
            p.side.push_back(st);
            p.joins.push_back(ev);
        }
        return p;
    }
    static bool serial() {
        static const bool v = tqllm::debug_on("composite_serial");
        return v;
    }
    bool matvec_ld(const Mat &, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        if (serial() || ps.size() < 2) {
            int r0 = 0;
            for (const auto & m : ps) {
                m.fmt->matvec_ld(m, x, y + r0, n, ldy, s);
                r0 += m.rows;
            }
            return true;
        }
        auto & P = pool(ps.size() - 1);
        const cudaEvent_t fork = P.fork;
        const auto & side = P.side;
        auto & joins = P.joins;
        ck(cudaEventRecord(fork, s), "record fork");
        int r0 = ps[0].rows;
        for (size_t i = 1; i < ps.size(); ++i) {
            ck(cudaStreamWaitEvent(side[i - 1], fork, 0), "side waits for fork");
            ps[i].fmt->matvec_ld(ps[i], x, y + r0, n, ldy, side[i - 1]);
            ck(cudaEventRecord(joins[i - 1], side[i - 1]), "record join");
            r0 += ps[i].rows;
        }
        ps[0].fmt->matvec_ld(ps[0], x, y, n, ldy, s);
        for (size_t i = 0; i + 1 < ps.size(); ++i) ck(cudaStreamWaitEvent(s, joins[i], 0), "main waits for join");
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
