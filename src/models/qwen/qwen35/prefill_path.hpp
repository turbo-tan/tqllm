#pragma once
// Which kernel a prefill projection over n columns takes (mm_h). Pure, so tests pin the dispatch.
// dequant + cuBLAS writes an FP16 copy of the whole matrix per chunk: a fixed cost that follows memory bandwidth
// (5060 Ti ~254 ms per chunk for the 27B, 3090 ~130 ms). The matvec kernel in batches and the native kernels have no
// fixed cost but are compute-bound per column, so the crossover differs per GPU: it is measured at load
// (prefill_crossover on timings of the real weights) instead of a fixed column count.
#include <algorithm>
#include <cmath>

namespace tqllm {
enum class PrefillPath { matvec, batched, native, dequant };
constexpr int kNgemmBlas = 128;  // the old fixed threshold (uncalibrated fallback)
struct PrefillFormat {
    int max_matvec_cols;  // matvec kernel width
    int max_batch_cols;   // columns per pass when the matvec kernel is run in batches
    bool native_gemm;     // the format has its own multi-column kernel
    int cross = -1;       // calibrated: chunks of <= cross columns avoid dequant + cuBLAS (-1: not calibrated)
    bool cross_native = false;  // calibrated alternative is the native kernel (else the batched matvec)
};
// Timings of one representative matrix of a format (ms): pass = one batched matvec pass of pass_cols columns;
// native(n) = native_fix + n * native_per_col; dequant(n) = dq_fix + n * dq_col.
struct PrefillCosts {
    double pass_ms; int pass_cols;
    double native_fix, native_per_col;  // native_per_col <= 0: no native kernel
    double dq_fix, dq_col;
};
struct PrefillCross { int cross; bool native; };
// Largest n in (lo, hi] for which the cheaper alternative still beats dequant + cuBLAS (0: never).
inline PrefillCross prefill_crossover(const PrefillCosts & c, int lo, int hi) {
    PrefillCross r{0, false};
    for (int n = lo + 1; n <= hi; ++n) {
        const double dq = c.dq_fix + n * c.dq_col;
        const double batched = std::ceil(double(n) / c.pass_cols) * c.pass_ms;
        const double native = c.native_per_col > 0 ? c.native_fix + n * c.native_per_col : 1e30;
        const double alt = std::min(batched, native);
        if (alt < dq) r = {n, native < batched};
    }
    return r;
}
// have_dequant: the FP16 prefill weight buffer exists; batched_verify: np > 1 / draft columns (np = 1 numerics)
inline PrefillPath prefill_path(int n, const PrefillFormat & f, bool have_dequant, bool batched_verify) {
    if (n <= f.max_matvec_cols) return PrefillPath::matvec;
    if (batched_verify) return PrefillPath::batched;
    if (f.cross < 0) {  // uncalibrated: the old rule
        if ((n < kNgemmBlas || !have_dequant) && f.native_gemm) return PrefillPath::native;
        return have_dequant ? PrefillPath::dequant : PrefillPath::batched;
    }
    if (!have_dequant || n <= f.cross) return f.cross_native && f.native_gemm ? PrefillPath::native : PrefillPath::batched;
    return PrefillPath::dequant;
}
}  // namespace tqllm
