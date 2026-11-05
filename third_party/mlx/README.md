# MLX patch for tqLLM's Metal backend

`mlx-0.32.3-sdpa-lse.patch` applies to MLX v0.32.3 (https://github.com/ml-explore/mlx, MIT licence). It adds
`mlx::core::fast::scaled_dot_product_attention_lse`: the fused attention kernel on the M5 matrix unit that also returns
each row's log-sum-exp, so tqLLM can split attention over a long KV cache into segments and merge them exactly. The
header defines `MLX_HAS_SDPA_LSE`; without the patch tqLLM builds and uses a slower path at long context.
Build steps: [docs/guides/mac.md](../../docs/guides/mac.md).
