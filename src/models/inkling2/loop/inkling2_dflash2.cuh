#pragma once
// Inkling-2-Mini DFlash2 drafter kernels (incoai/Inkling-2-Mini-DFlash2, Apache-2.0; reference sglang 0.5.21
// srt/models/dflash.py with the drafter's Inkling patch, as tools/inkling2/dflash2_reference.py). The drafter only
// proposes tokens; the target's verify decides, so spec == plain whatever these round to.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace tqllm_cuda::inkling2 {

// two-tap grouped dynamic convolution over the block rows (DFlashGroupedConv): taps w0 / w1 of column c in row r =
// dyn[r][(2 side + j) * G + c / group] + base[(2 side + j) * H + c]; y[r][c] = bf16(w0 x[r][c] + w1 x[r - 1][c]) (x[-1] = 0),
// with a residual bf16(res + y). x / res / y F32 [m][H], dyn BF16 [m][4 G] (G = H / group), base F32 [4 H]
void dflash_conv(const float * x, const __nv_bfloat16 * dyn, const float * base, int m, int H, int group, int side, const float * res, float * y,
                 cudaStream_t s);
// act = bf16(bf16(g sigmoid(g)) u) from gu BF16 [m][2 I] (gate, then up)
void dflash_silu_mul(const __nv_bfloat16 * gu, __nv_bfloat16 * act, int m, int I, cudaStream_t s);
// the 16 largest of each row of x F32 [rows][ld] over columns [0, n) (larger value first, lower id on ties):
// ids [rows][16], vals [rows][16]; scratch: dflash_top16_floats(rows, n) floats
int dflash_top16_floats(int rows, int n);
void dflash_top16(const float * x, int rows, int n, int ld, int * ids, float * vals, float * scratch, cudaStream_t s);
// the selector's greedy walk (CandidateSelector): from the anchor, step i (1 .. count) picks the candidate c of row i with
// the largest succ[id][:] . (pred[previous][:] * gate[i][:]) + unary[i][c]; tokens[i - 1] = its id (vocab and above stop:
// the rest are set to vocab). gate / pred / succ BF16: [rows][256], [>= vocab][256]
void dflash_select(const int * ids, const float * unary, const __nv_bfloat16 * gate, const __nv_bfloat16 * succ, const __nv_bfloat16 * pred, const int * anchor,
                   int count, int vocab, int * tokens, cudaStream_t s);

}  // namespace tqllm_cuda::inkling2
