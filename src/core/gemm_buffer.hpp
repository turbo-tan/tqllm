#pragma once
// FP16 prefill weight buffer: the largest quantised matrix dequantised whole, or (the fit's gemm_rows) blocks of rows.
// Formats without a row stride (TQ3_4S) are always dequantised whole, so the buffer never drops below them.
#include <algorithm>
#include <cstddef>
#include <vector>

namespace tqllm {
struct GemmMat {
    size_t rows, cols;
    bool row_sliceable;  // format has a per-row byte stride (row_bytes(cols) != 0)
};
struct GemmBufferPlan {
    int block_rows;  // 0: whole matrices
    size_t elems;    // FP16 elements in the buffer
};
// `capped_cols`: widest matrix that is dequantised in row blocks
inline GemmBufferPlan gemm_buffer_plan(int requested_rows, size_t capped_cols, const std::vector<GemmMat> & mats) {
    size_t mx = 0, whole = 0;  // whole: largest matrix that cannot be cut by rows
    for (const auto & m : mats) {
        mx = std::max(mx, m.rows * m.cols);
        if (!m.row_sliceable) whole = std::max(whole, m.rows * m.cols);
    }
    if (!requested_rows || whole == mx) return {0, mx};
    const int rows = std::max(256, requested_rows);
    return {rows, std::min(mx, std::max(whole, size_t(rows) * capped_cols))};
}
}  // namespace tqllm
