#pragma once
// The grouped-expert guard used by the qwen35 engine's load_moe. Pure (no CUDA), so tests pin it:
// a loaded expert matrix must have a format with a grouped-expert projection. The forward path
// calls the format's own expert_matvec / expert_gemm, which handle any basis, so the basis is not
// checked here. (The old guard also required Basis::Plain, which rejected our own TQ3_4S MoE files.)
#include "weight_format.hpp"
#include <stdexcept>
#include <string>

namespace tqllm {
void check_grouped_expert_format(const Mat & m, const std::string & prefix);

// The BF16 Gated DeltaNet [b|a] rows next to a quantized, rotated [qkv|z]: true when they are stored un-rotated
// and so must read an un-rotated copy of the normalised activation. The hub's TQ3_4S files store them
// un-rotated; our TQ4 files store them in the rotated basis and read the rotated copy the [qkv|z] rows read
// (feeding those the un-rotated copy made Qwen3.8-27B TQ_L greedy decode loop). Plain: one copy, nothing to pick.
inline bool bf16_ba_unrotated(Basis basis) { return basis == Basis::Tq3Rot32; }

// The verify path's Gated DeltaNet mix staging (qwen35 allocate() / mix_mm()). A file can keep the GDN [b|a]
// projections BF16 next to a quantized [qkv|z] (W4A16-style GGUFs, our TQ4 and the hub's TQ3_4S ones); the verify
// path then projects the two groups separately and interleaves them, which needs the staging buffers, plus the
// un-rotated copy when bf16_ba_unrotated. (The old code skipped the staging for Basis::Tq3Rot32, which left the
// buffers null and made the verify path write through a null output at 13..NGEMM_BLAS columns.)
struct MixStaging {
    bool quantized = false;  // v_xb / v_mixa / v_mixb: separate [qkv|z] and BF16 [b|a] projections
    bool unrotated = false;  // v_xu: the un-rotated FP16 normalised activation the BF16 rows read
};
MixStaging mix_staging(Basis basis, bool bf16_ba, bool quantized_model);
}  // namespace tqllm
