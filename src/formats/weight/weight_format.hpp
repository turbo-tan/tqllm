#pragma once
// Weight formats as strategies. A format turns a GGUF tensor into a device matrix and supplies the kernels the engine
// needs for it; the engine never branches on a format. Add a format: one .cpp in src/formats/ (kernels in
// src/formats/weight/) plus one line in registry.cpp. GGUF type ids >= 300 are reserved for TQLLM-only formats.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace tqllm {

// How a matmul input must be prepared (applied once per input vector, shared by every matrix that reads it).
enum class Basis { Plain, Tq3Rot32, Tq4Rot128 };

struct WeightFormat;

struct Mat {
    const void * p = nullptr;
    uint32_t type = 30;  // GGUF type id of the source (BF16 = 30), or an engine-internal id >= 1000
    int rows = 0, cols = 0;
    const void * aux = nullptr;           // per-format side data (scales)
    const WeightFormat * fmt = nullptr;   // null: dense BF16, handled by the engine (cuBLAS)
};

// Uploads host bytes to device memory owned by the engine; returns the device pointer.
using Upload = std::function<const void *(const void * host, size_t bytes)>;

struct WeightFormat {
    virtual ~WeightFormat() = default;
    virtual const char * name() const = 0;
    virtual Basis basis() const = 0;
    // `raw`: GGUF bytes of `rows` rows (one tensor or several row-concatenated tensors of this type).
    virtual Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const = 0;
    virtual int max_matvec_cols() const { return 8; }
    // Column limit of matvec() in a batched verify (per-column results identical to narrower calls).
    virtual int max_batch_cols() const { return max_matvec_cols(); }
    // y[n][rows] = W x[n] for n <= max_matvec_cols(); x in basis().
    virtual void matvec(const Mat &, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t) const = 0;
    virtual void matvec_f32(const Mat &, const __half * x, float * y, int n, cudaStream_t) const;
    // Output rows ldy apart (y [n][ldy], first m.rows written); false: not supported (callers copy instead).
    virtual bool matvec_ld(const Mat &, const __half *, __nv_bfloat16 *, int, int, cudaStream_t) const { return false; }
    virtual bool has_matvec_ld() const { return false; }
    virtual bool expert_gemm(const Mat &, const __half *, __nv_bfloat16 *, int,
                              const int *, const int *, int, int,
                              int, bool, cudaStream_t) const { return false; }
    // Same as expert_gemm with int8 activations (per-32-block scales + per-16 sums); false: no int8 kernel for this format.
    virtual bool expert_gemm_i8(const Mat &, const int8_t *, const float *, const int16_t *, __nv_bfloat16 *, int,
                                 const int *, const int *, int, int, int, bool, cudaStream_t, int mode) const;
    // gate and up (same format) in one launch with the swiglu epilogue; y [assignments][rows] half
    virtual bool expert_gemm_gateup(const Mat & g, const Mat & u, const __half * x, __half * y, int rows,
                                    const int * counts, const int * groups, int capacity, int experts, int topk, cudaStream_t) const;
    virtual bool has_expert_matvec() const { return false; }
    // Stacked weights [experts * rows][cols], device IDs [n][topk], output [n * topk][rows].
    virtual void expert_matvec(const Mat &, const __half *, __nv_bfloat16 *, int rows,
                               const int *, int n, int topk, bool per_expert_input, cudaStream_t) const;
    // u4h-family kernel id (-1: none). Dedup: one block per unique expert, map layout in quant_kernels.cuh.
    virtual int u4h_id() const { return -1; }
    bool has_expert_dedup() const { return u4h_id() >= 0; }
    void expert_matvec_dedup(const Mat &, const __half *, __nv_bfloat16 *, int rows, const int * dmap, int groups, cudaStream_t) const;
    // two matrices of this format (gate, up) over the same dedup map in one launch
    void expert_matvec_dedup2(const Mat &, const Mat &, const __half *, __nv_bfloat16 *, __nv_bfloat16 *, int rows, const int * dmap, int groups, cudaStream_t) const;
    // two stacked matrices of this format (gate, up) over the same device IDs in one launch; per row expert_matvec
    void expert_matvec2(const Mat &, const Mat &, const __half *, __nv_bfloat16 *, __nv_bfloat16 *, int rows, const int * ids, int n,
                        int topk, cudaStream_t) const;
    // FP16 [rows][cols] in basis(), for the cuBLAS prefill GEMM.
    virtual void dequant_h(const Mat &, __half * out, cudaStream_t) const = 0;
    // int8 prefill without an FP16 round trip: weights quantised straight to int8 [rows][cols] (row-major) plus the
    // per-row scales row_scale[r] = amax/127 (1 for a zero row), the same rule as int8_quant_rows. false: use the
    // dequant_h + per-chunk quantise path (the default for every format).
    virtual bool dequant_i8(const Mat &, int8_t * w8, float * row_scale, cudaStream_t) const { return false; }
    // Device bytes per row of the data / aux arrays when rows are stored contiguously (a row range is then a Mat of its
    // own: p + r0 * row_bytes, aux + r0 * aux_row_bytes); 0 = not row-sliceable.
    virtual size_t row_bytes(int /*cols*/) const { return 0; }
    virtual size_t aux_row_bytes(int /*cols*/) const { return 0; }
    // Optional kernel for max_matvec_cols() < n < the cuBLAS threshold; false = use dequant + GEMM.
    virtual bool gemm(const Mat &, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t) const { return false; }
    virtual bool has_gemm() const { return fused_gemm_cols() > 0; }  // gemm() above is implemented (prefill dispatch, prefill_path.hpp)
    // y [n][rows] F32 through the u4h-family GEMM (weights read once per 64 columns); false: no GEMM for this format
    bool gemm_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const;
    // u4h kernel family id with an F32-output GEMM (u4h_family_gemm_f32), including the UD types that are not u4h_id()
    // formats (gemm_f32 above keeps u4h_id(), so engines that call it are unchanged)
    virtual int gemm_f32_id() const { return u4h_id(); }
    bool gemm_i8_f32(const Mat & m, const int8_t * xq, const float * xs, const int16_t * xsum, float * y, int n, cudaStream_t s, int mode) const;
    // gemm() also replaces the dequant + cuBLAS prefill up to this many columns (0: only below the cuBLAS threshold).
    virtual int fused_gemm_cols() const { return 0; }
    // gemm() with output rows ldy apart (y [n][ldy]); false: not supported.
    virtual bool gemm_ld(const Mat &, const __half *, __nv_bfloat16 *, int, int, cudaStream_t) const { return false; }
    // Row blocks of a composite matrix (non-null only for composite_format); each is a matrix of its own format.
    virtual const std::vector<Mat> * parts() const { return nullptr; }
};

// Format that loads GGUF type `gguf_type`, or null. The loaded Mat may point at a different (runtime) format,
// e.g. Q4_0 loads as W4S when every 128-group shares one scale.
const WeightFormat * weight_format(uint32_t gguf_type);
void register_weight_format(uint32_t gguf_type, std::unique_ptr<WeightFormat>);

// Row-concatenation of matrices of different formats (a fused group such as gate + up quantized to different types):
// null unless every part has a format with matvec_ld, all share cols and basis. The returned format owns the parts'
// descriptors (not their device memory) and must outlive the Mat from composite_mat().
std::unique_ptr<WeightFormat> composite_format(std::vector<Mat> parts);
Mat composite_mat(const WeightFormat & composite);

}  // namespace tqllm
