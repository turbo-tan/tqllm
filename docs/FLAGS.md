# TQLLM flags

Generated from `config/flags.yaml` (`tools/gen_flags.py`). `tqserve --list-flags` prints the same table;
every tool stops at start-up when a retired name is set and warns about unknown `TQLLM_*` names. Settings are
command-line options (`docs/CONFIG.md`).

## Environment

| Flag | Default | What it does | Use instead |
|---|---|---|---|
| `TQLLM_MATVEC_RULE` | - | override the matvec kernel rule table (kernels/formats/matvec_rules.hpp) per format bits: "BITS:KERNEL[:PF],... (KERNEL legacy or split16)", e.g. "3:split16:2,21:legacy" (3 TQ3, 4 TQ4, 14 IQ4N, 21 IQ3) |  |

## Debug

| Flag | Default | What it does | Use instead |
|---|---|---|---|
| `TQLLM_MOE_FUSE_SHARED` | on | 0: decode keeps the shared expert as separate dense matvecs instead of one more assignment in the routed expert launches |  |
| `TQLLM_MOE_STAGE` | on | 0: --moe-host-mb decode reads host experts in place instead of copying the routed ones to a VRAM stage first (diagnostics) |  |
| `TQLLM_MOE_PROFILE` | - | FILE: write MoE routing counts per layer and expert (read by --moe-profile for --moe-host-mb); rewritten after every prefill |  |
| `TQLLM_LOG_TIMINGS` | on | 0: no llama.cpp-style print_timing lines (per-request prefill / decode / total with draft acceptance, plus the 3 s n_gen pulse) |  |
| `TQLLM_DEBUG` | off | comma list of diagnostics: memlog (device memory per load stage), profile (spec round phases), steps (decode step timings), stageprof (eager verify, per-stage time), matvec (kernel picks), tune (load-time tuner picks), prefill (progress), draft (first draft / accept decisions), tree_chain (tree verify as a chain), tf_dump:FILE (teacher-forced argmax per prompt position), fakeq_k:N / fakeq_v:N (quantise-dequantise K / V to N bits) |  |

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
