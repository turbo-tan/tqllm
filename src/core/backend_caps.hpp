#pragma once
// What the GPU backend of this build can do, for host code (the counterpart of common.cuh's TQLLM_DEV_* device macros).
// Components and models ask these, never TQLLM_WITH_<backend> (AGENTS.md rule 13); this file and src/backends/ are where
// the backend name is tested.
namespace tqllm::caps {
#if defined(TQLLM_WITH_HIP)
inline constexpr bool kFragmentPrefillGemm = false;  // TQ3 / TQ4 custom prefill GEMMs are NVIDIA mma.sync fragment code
inline constexpr bool kDenseSolver = false;          // cuSOLVER Spotrf / Spotrs (TqCache's stand-in solve); not linked
inline constexpr bool kBatchCopyAttributes = false;  // cudaMemcpyBatchAsync access-order / overlap attributes
#else
inline constexpr bool kFragmentPrefillGemm = true;
inline constexpr bool kDenseSolver = true;
inline constexpr bool kBatchCopyAttributes = true;
#endif
}  // namespace tqllm::caps
