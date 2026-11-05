#pragma once
// x3.cuh: the host-facing X3 kernel API. src/formats/x3.cpp includes only this; the kernels live in
// kernels/formats/x3/*. The codec table is keyed by (codec id, words) so a new codec/K is a table
// row, not a code edit.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace tqllm {
namespace x3 {

struct XwList;   // window.hpp
struct XwMoeList;

// One codec row: metadata plus the launch entry points for one (codec, words) pair.
struct X3Codec {
    uint16_t    id;      // X3CodecId
    int         words;   // uint16 per 16x16 tile (selects K)
    const char* name;
    bool        has_sq;   // a dedicated n == 1 int8 path
    bool        has_x3v;  // the register-path gemv works for a batch > 1
    // C[size_m][size_n] = W x, W trellis B, A is f32 (size_m x size_k), C is f32. No allocation.
    void (*gemv)(const float* A, const uint16_t* B, float* C, int size_m, int size_k, int size_n,
                 const half* suh, half* a_had, const half* svh, int* locks, cudaStream_t stream);
    void (*dequant)(half* W, const uint16_t* T, const half* suh, const half* svh, int rows, int cols,
                    cudaStream_t stream);
    // The same matrix in the engine's GEMM layout: W[r * ld + c], r < rows (out), c < cols (in); ld % 4 == 0.
    void (*dequant_rows)(half* W, int ld, const uint16_t* T, const half* suh, const half* svh, int rows, int cols,
                         cudaStream_t stream);
    // A row range of it: out rows [r0, r0 + nr) (both multiples of 128) into W[0 .. nr); rows / cols are the part's
    // full extents, which set the trellis' k-major stride. The head dump (4c) walks a 200K-row matrix in chunks
    // because the engine has no whole-matrix FP16 scratch.
    void (*dequant_rows_range)(half* W, int ld, const uint16_t* T, const half* suh, const half* svh, int rows, int cols,
                               int r0, int nr, cudaStream_t stream);
    // The same matrix quantised straight to int8 (no FP16 round trip): W[r * ld + c] = round(clamp(half2float(v) / sc[r])),
    // sc the load-time row scales (amax/127, 1 for a zero row); ld % 4 == 0.
    void (*dequant_rows_i8)(int8_t* W, int ld, const uint16_t* T, const half* suh, const half* svh, int rows, int cols,
                            const float* sc, cudaStream_t stream);
    // Caller workspace bytes for the n == 1 sq path (counters + qsums + partials).
    size_t (*ws_bytes)(int size_k, int size_n);
    // Window GEMV (gemv_window.cu) over a list of this codec's parts: y[t][out_off + r] = bf16(W x[t]),
    // x half [n][cols], y bf16 with row stride ldy. nullptr when the codec has no register path.
    void (*window)(const XwList& L, const half* x, __nv_bfloat16* y, int ldy, cudaStream_t stream);
    // The same GEMV with an f32 output: y[t][out_off + r] = f32(bf16(W x[t])), the values of `window` widened, so an
    // engine that converts matvec's output to f32 can skip its own bf16 buffer and the conversion pass.
    void (*window_f32)(const XwList& L, const half* x, float* y, int ldy, cudaStream_t stream);
    // Routed-expert window GEMV (moe_window.cuh) over a device-built list of at most `parts` parts; nullptr: none.
    // Part u's column t lands at y[t * ldy + u * rows].
    void (*moe)(const XwMoeList* L, int parts, int n, int rows, int cols, int kparts, __nv_bfloat16* y, int ldy,
                cudaStream_t stream);
    // Prompt-chunk routed experts (moe_window.cuh xw_pmoe_matvec): a host-built list of (expert, pass) parts at
    // n = XW_MAX_N reading x through each pass's assignment ids, writing y [assignment][rows]; nullptr: none.
    void (*pmoe)(const XwMoeList* L, int parts, int rows, int cols, int kparts, const half* x, __nv_bfloat16* y,
                 cudaStream_t stream);
};

// Verify blocks: one part per distinct expert with all n columns; aindex[u * n + t] = the assignment of token t routed
// to part u (-1 for none), which the out launch uses to write y [assignment][rows] directly and, for a per-slot input
// (the down projection), which had_in uses to read the assignment's row of x: no staged per-part copy.
void x3_moe_build_dedup(const int* ids, int n, int topk, const void* bank, int rows, int cols, int kparts, int ncodec,
                        const half* x, bool per_slot_input, half* A, float* P, int* aindex, XwMoeList* lists,
                        cudaStream_t stream);

// Prompt path: rows of x picked by an expert's assignments, and that expert's [count][rows] output scattered to y.
void x3_moe_gather(const half* x, const int* group, int count, int topk, bool per_slot_input, int cols, half* out,
                   cudaStream_t stream);
void x3_moe_scatter(const float* in, const int* group, int count, int rows, __nv_bfloat16* y, cudaStream_t stream);
void x3_moe_scatter_bf16(const __nv_bfloat16* in, const int* group, int count, int rows, __nv_bfloat16* y, cudaStream_t stream);

// The same pass over the routing for gate and up together (two banks, one codec index space, the caller's two outputs in
// one allocation `up_base` = yb - ya elements apart): two parts per routed slot / distinct expert, A / P slot 2u + m.
void x3_moe_build2(const int* ids, int count, int topk, const void* bankA, const void* bankB, int rows, int cols,
                   int kparts, int ncodec, const half* x, int up_base, half* A, float* P, XwMoeList* lists,
                   cudaStream_t stream);
void x3_moe_build_dedup2(const int* ids, int n, int topk, const void* bankA, const void* bankB, int rows, int cols,
                         int kparts, int ncodec, const half* x, int up_base, half* A, float* P,
                         int* aindex, XwMoeList* lists, cudaStream_t stream);

// Device bank of routed experts -> one list per codec slot, from the routing ids (count = tokens x top-k).
void x3_moe_build(const int* ids, int count, int topk, bool per_slot_input, const void* bank, int rows, int cols,
                  int kparts, int ncodec, const half* x, half* A, float* P, XwMoeList* lists, cudaStream_t stream);

// The full table (null-terminated count). Defined in dispatch.cu.
const X3Codec* x3_codec_table(int& count);

// True when matvec should use the codecs' window GEMV (default); TQLLM_DEBUG=x3v selects the x3v/sq path.
bool x3_window_enabled();

// Element-wise casts the format layer uses to present f32 buffers to the codec entry points.
void x3_cast_h_f32(const half* src, float* dst, int n, cudaStream_t stream);
void x3_cast_f32_bf16(const float* src, __nv_bfloat16* dst, int n, cudaStream_t stream);
// dst[t*ld + off + c] = bf16(src[t*part_rows + c]) - a row-part of a row-concatenated matvec output.
void x3_copy_f32_bf16_strided(const float* src, __nv_bfloat16* dst, int part_rows, int ld, int off, int n,
                              cudaStream_t stream);

}  // namespace x3
}  // namespace tqllm
