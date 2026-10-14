# Native Qwen35 MoE development

The `feat/moe-rc3-20261007` lane includes rc3 `4d9bc14` and native UD formats
`be99d2e`. MoE models load by default (TQLLM_MOE_DEV is retired).

Implemented:

- F32 router weights, device softmax, top-k selection, selected-weight
  normalization and shared-expert sigmoid gate.
- Grouped K4, K5 and K6 expert projections with device IDs and no CPU routing
  synchronization. Inputs can be per token or per selected expert.
- Routed SwiGLU, shared FFN and residual integration in the batched forward.
- Device grouping of prefill assignments and native quantized grouped GEMM;
  inactive expert tiles exit without touching weights.
- Linear BF16/Q8 attention automatically splits verify blocks to fit the
  query-head grouping, including group-8 Qwen35 MoE.
- Early rejection of unsupported expert formats; rotated expert inputs and MoE
  MTP remain unsupported.

The device test compares grouped projections bit-for-bit with individual
row-sliced projections for one and multiple tokens, both input layouts, tail
rows and CUDA graph replay. Grouped GEMM is bit-identical to gathered per-expert
GEMM for both input layouts, inactive experts, hot experts at full assignment
capacity and 64-column tile boundaries. Group-8 attention verify(8) is
bit-identical to individual causal queries. Router tests use an independent CPU reference,
including exact ties and finite extreme logits. Near-tied probabilities can
round differently between CPU and CUDA exponentials; deterministic ties in this
engine use ascending expert IDs. This is not an exact-tie parity promise versus
llama.cpp's unstable sort.

Pending gates: same-file standard llama.cpp KLD on Wikitext-2 (100 x 512), sane
text, and matched no-spec speed. Decode uses grouped matvec; prefill uses grouped GEMM.
Do not promote this lane or quote performance before those gates pass.
