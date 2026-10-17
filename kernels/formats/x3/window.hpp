#pragma once
// window.hpp: the host-visible plan of the X3 window GEMV (kernels in gemv_window.cu). Plain C++ so the
// format layer (src/formats/x3.cpp) can build a list without seeing a kernel: one XwList = the row parts of
// one matvec that share a codec; the codec table's `window` entry runs it.
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace tqllm {
namespace x3 {

constexpr int XW_MAX_PARTS = 4;
constexpr int XW_MAX_N = 16;
constexpr int XW_MAX_ITEM_SLICES = 160;   // k slices per gemv item: the A stage stays <= 41 KB at n = 8 (two blocks per SM)

// gemv block geometry: WK warps split an item's k slices, each warp owns WNT adjacent 16-column tiles,
// PF k slices of weights in flight per warp, fp16 accumulators folded to fp32 every FOLD slices.
struct XwGeom {
    static constexpr int WK = 8, WNT = 4, PF = 4, FOLD = 2;
    static constexpr int THREADS = WK * 32;
    static constexpr int COLS = WNT * 16;
};

// One row part of the matrix as the window kernels see it.
struct XwPart {
    const uint32_t* B;   // trellis words, tiles k-major (x3_tile_offset)
    const half* suh;     // [cols]
    const half* svh;     // [rows]
    half* A;             // scratch [n][cols]: H128(x * suh)
    float* P;            // scratch [kparts][n][rows]: per-k-part partial sums
    int rows;            // output rows of the part
    int kparts;          // k split across items (divides cols / 16)
    int item0;           // first gemv item of this part
    int out_off;         // row offset of the part inside y
};

struct XwList {
    XwPart p[XW_MAX_PARTS];
    int count;
    int cols;            // input size (shared by all parts)
    int n;               // batch columns 1..XW_MAX_N
    int items;           // total gemv items
};

// k split for one part: doubled while the launch is short of items or an item exceeds XW_MAX_ITEM_SLICES,
// keeping >= 8 slices per warp. Depends on (rows, cols) only, never n: column t is then bit-identical for
// every n (the drafter's verify batch equals plain decode).
inline int xw_kparts(int rows, int cols, int num_sms) {
    const int groups = rows / XwGeom::COLS, kslices = cols / 16;
    const int target = 4 * num_sms;
    int kp = 1;
    while ((groups * kp < target || kslices / kp > XW_MAX_ITEM_SLICES) && kslices % (2 * kp) == 0 &&
           kslices / (2 * kp) >= 8 * XwGeom::WK)
        kp *= 2;
    return kp;
}

inline size_t xw_align(size_t b) { return (b + 255) & ~(size_t) 255; }

// Scratch of one list of parts (rows[i], shared cols), sized for XW_MAX_N: A then the partial sums.
inline size_t xw_scratch_bytes(const int* rows, int count, int cols, int num_sms) {
    size_t p = 0;
    for (int i = 0; i < count; ++i) p += (size_t) xw_kparts(rows[i], cols, num_sms) * XW_MAX_N * rows[i] * sizeof(float);
    return xw_align((size_t) count * XW_MAX_N * cols * sizeof(half)) + p;
}

inline void xw_begin(XwList& L, int cols, int n) {
    L.count = 0; L.cols = cols; L.n = n; L.items = 0;
}

// Appends a part; returns false when the list is full.
inline bool xw_add(XwList& L, const uint16_t* B, const half* suh, const half* svh, int rows, int out_off, int num_sms) {
    if (L.count == XW_MAX_PARTS) return false;
    XwPart& p = L.p[L.count++];
    p.B = reinterpret_cast<const uint32_t*>(B);
    p.suh = suh; p.svh = svh; p.rows = rows; p.out_off = out_off;
    p.kparts = xw_kparts(rows, L.cols, num_sms);
    p.item0 = L.items;
    p.A = nullptr; p.P = nullptr;
    L.items += (rows / XwGeom::COLS) * p.kparts;
    return true;
}

// Points the parts' A / P at `scratch` (at least xw_scratch_bytes of the same parts).
inline void xw_bind_scratch(XwList& L, void* scratch) {
    char* base = static_cast<char*>(scratch);
    float* P = reinterpret_cast<float*>(base + xw_align((size_t) L.count * XW_MAX_N * L.cols * sizeof(half)));
    for (int i = 0; i < L.count; ++i) {
        L.p[i].A = reinterpret_cast<half*>(base) + (size_t) i * XW_MAX_N * L.cols;
        L.p[i].P = P;
        P += (size_t) L.p[i].kparts * XW_MAX_N * L.p[i].rows;
    }
}

}  // namespace x3
}  // namespace tqllm
