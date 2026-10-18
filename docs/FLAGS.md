# TQLLM flags

Generated from `config/flags.yaml` (`tools/gen_flags.py`). `tqserve --list-flags` prints the same table;
every tool stops at start-up when a retired name is set and warns about unknown `TQLLM_*` names. Settings are
command-line options (`docs/CONFIG.md`).

## Environment

| Flag | Default | What it does | Use instead |
|---|---|---|---|
| `TQLLM_MLX_ANE_FFN` | - | DIR: Neural Engine share of the prefill FFN from per-layer Core ML models (opt-in fast prefill; quality changes, see docs/MLX_BACKEND.md) |  |
| `TQLLM_MLX_ANE_WAVE` | - | 1 / N: two-half wavefront prefill that overlaps the ANE share with the GPU (opt-in with TQLLM_MLX_ANE_FFN) |  |
| `TQLLM_MLX_CACHE_LIMIT_MIB` | - | MiB: MLX buffer-cache limit for the Mac tools (default 512) |  |
| `TQLLM_MLX_DFLASH_BLOCK` | - | N: drafter block rows (8, or 16 for block-16 drafts and verify) |  |
| `TQLLM_MLX_ATTN_DECODE_V5` | - | =1: own decode / verify attention v5 (matrix-unit S and P.V, queries staged once per 32, in-register softmax); takes precedence over SDPA / v1 |  |
| `TQLLM_MLX_ATTN_DECODE_SDPA` | 1 | default on; =0 returns to v1. Decode / verify attention (1-16 rows, BF16 KV) through MLX's fused SDPA matrix kernel, queries padded to 16 rows (v4) |  |
| `TQLLM_MLX_VERIFY_NARROW_AT` | 0 | position from which spec verify narrows to TQLLM_MLX_VERIFY_NARROW_WIDTH rows (0 = always full width) |  |
| `TQLLM_MLX_VERIFY_NARROW_WIDTH` | 7 | drafted tokens verified per step past TQLLM_MLX_VERIFY_NARROW_AT |  |
| `TQLLM_MLX_PREFILL_CHUNK` | - | N: prefill chunk rows for tqmlx-dflash (>= 128; 2048 is the measured best) |  |
| `TQLLM_MLX_PREFILL_INT8` | - | 1: int8 matrix-unit prefill for FFN gate/up (opt-in, quality changes; 2 = exact slow mode) |  |
| `TQLLM_MATVEC_RULE` | - | override the matvec kernel rule table (src/formats/weight/matvec_rules.hpp) per format bits: "BITS:KERNEL[:PF],... (KERNEL legacy or split16)", e.g. "3:split16:2,21:legacy" (3 TQ3, 4 TQ4, 14 IQ4N, 21 IQ3) |  |

## Debug

| Flag | Default | What it does | Use instead |
|---|---|---|---|
| `TQLLM_MLX_ABLATE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ABLATE_BLOCKS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ABLATE_SMALL` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ANE_CHECK` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ANE_FULL_DEQUANT` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ANE_NOISE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ANE_SERIAL` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ANE_SKIP_PREDICT` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ANE_TIMING` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ATTN_DECODE_ROWS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ATTN_DECODE_V2` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ATTN_DECODE_V3` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ATTN_PREFILL_BF16` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ATTN_Q_CHUNK` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ATTN_V3_PB` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_COUNT_NODES` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_SAMPLE_AR` | - | =1: temperature > 0 decodes token by token (no drafter), the pre-speculative-sampling path |  |
| `TQLLM_MLX_SAMPLE_DEBUG` | - | =1: print per-row top tokens / argmax / proposal for the first 3 speculative sampling steps |  |
| `TQLLM_MLX_ATTN_V5_PB` | 256 | v5 key partition length (multiple of 8) |  |
| `TQLLM_MLX_DFLASH_CHAIN` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_PIPELINE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_PIPELINE_SYNC_BUILD` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_RANKSTATS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_SHARDED_TOP16` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_SPEC_ONLY` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_WARM_TOKENS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DRAFT_ABLATE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DRAFT_BUILD_PROFILE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DROP_TILED_ORIGINALS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DUMP_FFN` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DUMP_FFN_SLICE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_EAGER_ACCEPT` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_EAGER_COMMIT` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_FFN_GEMM` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_FUSE_EW` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_FUSE_SWIGLU_ROTATION` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_GDN_CHUNKED` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_GDN_FUSE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_GDN_FUSE_PREFILL` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_GDN_STEP` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_HEAD_FRAG` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_KV_F32` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_LAZY_INJECT` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_LAZY_VERIFY_COMMIT` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_MMA8` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_MMA8_DRAFT` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_MMA8_UNTILED_ONLY` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_DRAFT_HEAD4` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_DRAFT_SHORTLIST` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_FUSE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_MAX_SPLIT` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_MR64` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_NO_INPUT_CACHE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_PREFILL_GEMM` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_Q6_HEAD` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_ROW_CHECK` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_SPLIT_TARGET` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_TQ4` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_PAIRED_SWIGLU_ROTATION` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_PREFILL_GEMM` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_PREFILL_QMM` | 1 | Apple MLX backend: prefill 4-bit affine matrices through mx::quantized_matmul on weights repacked to MLX layout (1, default) or dequant + matmul (0) |  |
| `TQLLM_MLX_PREFILL_INT8_MASK` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_SPLIT_COMMIT_TIMING` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_STAGED_BKM` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_STAGED_KS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_STAGED_MT` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_STAGED_PERSG` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_STAGED_SIMDS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_TARGET_HEAD_HOST` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_TARGET_HEAD_IDS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_TARGET_HEAD_SHORTLIST` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_TQ4_ROTATE_UNFUSED` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MOE_FUSE_SHARED` | on | 0: decode keeps the shared expert as separate dense matvecs instead of one more assignment in the routed expert launches |  |
| `TQLLM_MOE_STAGE` | on | 0: --moe-host-mb decode reads host experts in place instead of copying the routed ones to a VRAM stage first (diagnostics) |  |
| `TQLLM_MOE_PROFILE` | - | FILE: write MoE routing counts per layer and expert (read by --moe-profile for --moe-host-mb); rewritten after every prefill |  |
| `TQLLM_DEBUG` | off | comma list of diagnostics: memlog (device memory per load stage), profile (spec round phases), steps (decode step timings), stageprof (eager verify, per-stage time), matvec (kernel picks), tune (load-time tuner picks), prefill (progress), draft (first draft / accept decisions), tree_chain (tree verify as a chain), tf_dump:FILE (teacher-forced argmax per prompt position), fakeq_k:N / fakeq_v:N (quantise-dequantise K / V to N bits) |  |
| `TQLLM_MLX_AFFINE_FFN` | unset (backend default) | Experimental Apple MLX affine ffn override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_AFFINE_GU` | unset (backend default) | Experimental Apple MLX affine gu override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_AFFINE_HALF` | unset (backend default) | Experimental Apple MLX affine half override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_AFFINE_MIXER` | unset (backend default) | Experimental Apple MLX affine mixer override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_AFFINE_PAD` | unset (backend default) | Experimental Apple MLX affine pad override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_AFFINE_TQ4` | unset (backend default) | Experimental Apple MLX affine tq4 override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_AFFINE_TQ4_FP32` | unset (backend default) | Experimental Apple MLX affine tq4 fp32 override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_ATTENTION_KV_VIEWS` | unset (backend default) | Experimental Apple MLX attention kv views override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_BATCH_DECODE_DENSE` | unset (backend default) | Experimental Apple MLX batch decode dense override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_CAPTURE_VERIFY` | unset (backend default) | Experimental Apple MLX capture verify override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_COMPACT_GDN_TAPE` | unset (backend default) | Experimental Apple MLX compact gdn tape override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_COMPILED_GDN` | unset (backend default) | Experimental Apple MLX compiled gdn override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_DECODED_FFN` | unset (backend default) | Experimental Apple MLX decoded ffn override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_DEVICE_SELECTOR` | unset (backend default) | Experimental Apple MLX device selector override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_DRAFT_HEAD4` | unset (backend default) | Experimental Apple MLX draft head4 override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_DRAFT_HEAD_UNPAD` | unset (backend default) | Experimental Apple MLX draft head unpad override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_FUSED_ATTENTION_ROWS` | unset (backend default) | Experimental Apple MLX fused attention rows override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_FUSE_EPILOGUE` | unset (backend default) | Experimental Apple MLX fuse epilogue override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_GDN_SCAN` | unset (backend default) | Experimental Apple MLX gdn scan override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_INT8_N` | unset (backend default) | Experimental Apple MLX int8 n override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_INT8_SIMDS` | unset (backend default) | Experimental Apple MLX int8 simds override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_INT8_TILE` | unset (backend default) | Experimental Apple MLX int8 tile override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_MATMUL_COLS` | unset (backend default) | Experimental Apple MLX matmul cols override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_MATMUL_SIMDS` | unset (backend default) | Experimental Apple MLX matmul simds override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_NATIVE_GDN` | unset (backend default) | Experimental Apple MLX native gdn override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_NATIVE_GDN_PREP` | unset (backend default) | Experimental Apple MLX native gdn prep override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_Q6_AFFINE` | unset (backend default) | Experimental Apple MLX q6 affine override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_REUSE_ROTATION` | unset (backend default) | Experimental Apple MLX reuse rotation override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_SERIAL_VERIFY_ATTENTION` | unset (backend default) | Experimental Apple MLX serial verify attention override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_SERIAL_VERIFY_DENSE` | unset (backend default) | Experimental Apple MLX serial verify dense override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TENSOR_K` | unset (backend default) | Experimental Apple MLX tensor k override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TENSOR_M` | unset (backend default) | Experimental Apple MLX tensor m override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TENSOR_MEMORY` | unset (backend default) | Experimental Apple MLX tensor memory override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TENSOR_N` | unset (backend default) | Experimental Apple MLX tensor n override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TENSOR_SIMDS` | unset (backend default) | Experimental Apple MLX tensor simds override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TENSOR_SPLIT` | unset (backend default) | Experimental Apple MLX tensor split override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TENSOR_TRANSPOSE` | unset (backend default) | Experimental Apple MLX tensor transpose override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TENSOR_WIDE` | unset (backend default) | Experimental Apple MLX tensor wide override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TILED_DFLASH_FP32` | unset (backend default) | Experimental Apple MLX tiled dflash fp32 override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TILED_MIXER_FP32` | unset (backend default) | Experimental Apple MLX tiled mixer fp32 override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TILED_TQ3_FP16` | unset (backend default) | Experimental Apple MLX tiled tq3 fp16 override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TILED_TQ3_FP32` | unset (backend default) | Experimental Apple MLX tiled tq3 fp32 override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TILED_TQ4_FP32` | unset (backend default) | Experimental Apple MLX tiled tq4 fp32 override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TQ3_STREAM` | unset (backend default) | Experimental Apple MLX tq3 stream override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TQ3_SUBGROUP` | unset (backend default) | Experimental Apple MLX tq3 subgroup override; only used by MLX tools. Leave unset unless running the corresponding validation experiment. |  |
| `TQLLM_MLX_TILED_TQ4_FP16` | unset (backend default) | Experimental Apple MLX tiled TQ4 FFN with FP16 operands, FP32 accumulation; set to 1, mutually exclusive with TQLLM_MLX_TILED_TQ4_FP32. Quality gate required; does not requantize stored codes. |  |
| `TQLLM_MLX_PAIRED_GATE_UP` | unset (separate projections) | Experimental Apple MLX shared-tile gate/up projections with fused activation; set to 1. Uses explicitly enabled tiled FFN precision; incompatible pairs retain the original path. Arithmetic and whole-model gates required. |  |
| `TQLLM_MLX_TILED_MIXER_FP16` | unset (backend default) | Experimental Apple MLX tiled TQ3 GDN projections with FP16 operands and FP32 accumulation; set to 1, mutually exclusive with TQLLM_MLX_TILED_MIXER_FP32. Recurrent-state precision is unchanged. Requires quality and width-parity gates. |  |
| `TQLLM_MLX_WIRED_MIB` | unset (retain MLX default) | Experimental process-local persistent Metal residency budget in MiB for MLX benchmark tools; 0 disables wiring. Capped by the device recommendation, restored on scope exit. Does not alter system settings. |  |

## Retired (the tool stops; use the replacement)

| Flag | Default | What it does | Use instead |
|---|---|---|---|
| `TQLLM_ATTN_NBUF` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_BATCH_COLS` | - | removed | automatic (per GPU) |
| `TQLLM_COMPOSITE` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_CONV_SERIAL` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DEBUG_DRAFT` | - | removed | TQLLM_DEBUG=draft |
| `TQLLM_DEFER` | - | removed | automatic (per GPU) |
| `TQLLM_DFLASH` | - | removed | -md / --model-draft FILE |
| `TQLLM_DFLASH_BF16` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DFLASH_I8H` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DFLASH_NOGRAPH` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DFLASH_TQ3` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DRAFT_D` | - | removed | tqserve --draft-max N |
| `TQLLM_DRAFT_HEAD8` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DRAFT_VOCAB` | - | removed | --draft-vocab FILE |
| `TQLLM_DSUB_COPY` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_EMBED_HOST` | - | removed | automatic (tqserve memory fit) |
| `TQLLM_FAKEQ_K` | - | removed | TQLLM_DEBUG=fakeq_k:N |
| `TQLLM_FAKEQ_V` | - | removed | TQLLM_DEBUG=fakeq_v:N |
| `TQLLM_FORCE_BATCH` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_GEMM_F32` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_GEMM_F32ACC` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_GEMM_PAD` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_GEMM_ROWS` | - | removed | automatic (tqserve memory fit) |
| `TQLLM_GEN_ATTN_OLD` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_HEAD8` | - | removed | tqkld requests full-vocabulary logits itself |
| `TQLLM_HEAD_HOST` | - | removed | automatic (tqserve memory fit) |
| `TQLLM_HEAD_PER_SEG` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_HOST_CHAIN` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_HW_MT` | - | removed | removed (fixed launch policy) |
| `TQLLM_IQ3_OLD` | - | removed | TQLLM_MATVEC_RULE=21:legacy |
| `TQLLM_IQ3_PF` | - | removed | TQLLM_MATVEC_RULE=21:split16:PF |
| `TQLLM_IQ4N_PF` | - | removed | TQLLM_MATVEC_RULE=14:split16:PF |
| `TQLLM_IQ4_OLD` | - | removed | removed (IQ4_XS loads as IQ4N) |
| `TQLLM_KV` | - | removed | --kv FORMAT or -ctk / -ctv |
| `TQLLM_KV8` | - | removed | -ctk q8 (or --kv q8) |
| `TQLLM_KV_K` | - | removed | -ctk / --cache-type-k |
| `TQLLM_KV_T4K3V` | - | removed | --kv t4k-turbo3v |
| `TQLLM_KV_T4KV` | - | removed | --kv t4kv |
| `TQLLM_KV_TQ3` | - | removed | --kv tq3 |
| `TQLLM_KV_TURBO3_V` | - | removed | --kv q8k-turbo3v |
| `TQLLM_KV_V` | - | removed | -ctv / --cache-type-v |
| `TQLLM_LOOKUP` | - | removed | --lookup L |
| `TQLLM_LOOKUP_FIRST` | - | removed | --lookup-first K |
| `TQLLM_LOOKUP_N` | - | removed | removed (--lookup matches 6 tokens) |
| `TQLLM_MATVEC_LOG` | - | removed | TQLLM_DEBUG=matvec |
| `TQLLM_MEMLOG` | - | removed | TQLLM_DEBUG=memlog |
| `TQLLM_MIX_COPY` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_MQ_SPLIT` | - | removed | automatic |
| `TQLLM_NO_BLKMERGE` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_NO_BTREE` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_NO_RMSROPE` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_NO_SIDE` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_PF_GU_BF16` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_PF_MIX_COPY` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_PREFILL_ATTN_GROUPED` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_PREFILL_ATTN_OLD` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_PREFILL_BUF` | - | removed | -ub / --ubatch-size N |
| `TQLLM_PREFILL_CHUNK` | - | removed | automatic (-ub sets the prefill buffer) |
| `TQLLM_PREFILL_PROGRESS` | - | removed | TQLLM_DEBUG=prefill |
| `TQLLM_PROFILE` | - | removed | TQLLM_DEBUG=profile |
| `TQLLM_Q4K_U4H` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_REPEAT_EXIT` | - | removed | removed |
| `TQLLM_REPEAT_LIMIT` | - | removed | tqserve --repeat-limit N |
| `TQLLM_SERIAL_DRAFT` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_SERIAL_SEGS` | - | removed | fixed behaviour (settled A/B switch removed) |
| `TQLLM_STAGEPROF` | - | removed | TQLLM_DEBUG=stageprof |
| `TQLLM_STEP_PROF` | - | removed | TQLLM_DEBUG=steps |
| `TQLLM_TF_DUMP` | - | removed | TQLLM_DEBUG=tf_dump:FILE |
| `TQLLM_TQ3_HW` | - | removed | removed (fixed launch policy) |
| `TQLLM_TQ3_MT` | - | removed | removed (fixed launch policy) |
| `TQLLM_TQ3_MT_KMIN` | - | removed | removed (fixed launch policy: k >= 12288) |
| `TQLLM_TQ3_OLD` | - | removed | TQLLM_MATVEC_RULE=3:legacy |
| `TQLLM_TQ3_P` | - | removed | removed (fixed launch policy) |
| `TQLLM_TQ3_TUNE` | - | removed | removed (the load-time tuner always runs) |
| `TQLLM_TQ3_TUNE_LOG` | - | removed | TQLLM_DEBUG=tune |
| `TQLLM_TQ4_OLD` | - | removed | TQLLM_MATVEC_RULE=4:legacy |
| `TQLLM_TQ4_TUNE` | - | removed | removed (no in-model gain) |
| `TQLLM_TQ4_TUNE_LOG` | - | removed | TQLLM_DEBUG=tune |
| `TQLLM_TQ4_VARIANT` | - | removed | removed (variant 0) |
| `TQLLM_TREE` | - | removed | tree verify is on with -md; --no-tree turns it off |
| `TQLLM_TREE_CHAIN` | - | removed | TQLLM_DEBUG=tree_chain |
| `TQLLM_TREE_TEMP` | - | removed | removed (1.0) |
| `TQLLM_U4H` | - | removed | removed |
| `TQLLM_VRAM_MARGIN_MB` | - | removed | -fitt MiB (-fit on|off) |
| `TQLLM_MOE_DEV` | - | removed | nothing: qwen35moe (MoE) models load by default |
