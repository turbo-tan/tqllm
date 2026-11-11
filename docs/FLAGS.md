# TQLLM flags

Generated from `config/flags.yaml` (`tools/gen_flags.py`). `tqserve --list-flags` prints the same table;
every tool stops at start-up when a retired name is set and warns about unknown `TQLLM_*` names. Settings are
command-line options (`docs/CONFIG.md`).

## Environment

| Flag | Default | What it does | Use instead |
|---|---|---|---|
| `TQLLM_MLX_CACHE_LIMIT_MIB` | - | MiB: MLX buffer-cache limit for the Mac tools (default 512) |  |
| `TQLLM_MLX_PREFILL_CHUNK` | - | N: prefill chunk rows for tqmlx-dflash (>= 128; 2048 is the measured best) |  |
| `TQLLM_MATVEC_RULE` | - | override the matvec kernel rule table (src/formats/weight/matvec_rules.hpp) per format bits: "BITS:KERNEL[:PF],... (KERNEL legacy or split16)", e.g. "3:split16:2,21:legacy" (3 TQ3, 4 TQ4, 14 IQ4N, 21 IQ3) |  |
| `TQLLM_GLM_EXPERT_SLOTS` | 0 | GLM-5.3-Flash: keep only this many routed experts of every MoE layer resident, in a MixedExpertCache bank (0: the whole expert stack on the device). The experts a token needs are streamed from the GGUF / the packed file, so the tokens do not change; it also turns off the decode CUDA graph |  |
| `TQLLM_GLM_EXPERT_BUDGET_MB` | banks + 25 % | GLM-5.3-Flash: device bytes the expert banks plus loader staging may hold. With TQLLM_GLM_EXPERT_SLOTS it caps them; on its own it turns on the shared byte-budget plan (src/experts/expert_bank_plan.hpp), which gives every MoE layer top-k slots and the rest of the bytes to the layers whose routing spreads over more experts (measured over the prompt, re-planned when it ends) |  |
| `TQLLM_GLM_EXPERTS` | - | GLM-5.3-Flash: packed expert file (glm-pack-experts MODEL.gguf OUT.tqfnexp) for the expert banks: a bank miss copies the expert instead of repacking it; the engine page-locks it when it is at most half the host RAM, which also lets it prefetch (TQLLM_GLM_EXPERT_PF); a bigger file copies misses through pinned staging |  |
| `TQFN_ARENA_MB` | - | MB: Flash-Next scratch arena size instead of the measured rule (the high-water mark is printed at exit) |  |
| `TQFN_CPU_MISS` | 0 | N: compute up to N missed experts per layer call on the CPU instead of copying them to the GPU (default on small cards: min(15, cores / 2 - 1); else off) |  |
| `TQFN_DENSE_INT8` | 0 | 1 / 2: int8 activations for the dense prefill projections (per-32 / per-token scales); 0 = fp16 (exact path) |  |
| `TQFN_DRAFT_VOCAB` | - | FILE: the MTP draft vocabulary (overrides the manifest's draft_vocab) |  |
| `TQFN_KV` | - | KV format for Flash-Next attention (name from the KV format registry; default bf16) |  |
| `TQFN_KV_HOST` | auto | auto (default): on for discrete GPUs, off on unified memory; 1 / 0 force it. On: K/V older than TQFN_KV_HOST_KEEP positions lives in host memory after a long prompt (frees VRAM for expert banks; decode copies only the selected rows) |  |
| `TQFN_KV_HOST_KEEP` | 32768 | positions of K/V kept on the device with TQFN_KV_HOST=1 |  |
| `TQFN_LAYER_SLOTS` | - | comma list of expert slots per layer (overrides the packed file's .slots profile) |  |
| `TQFN_MTP_SLOTS` | 256 | MTP expert slots (every expert when the trunk keeps every expert; 64 on small cards) |  |
| `TQFN_PREFILL_INT8` | 0 | 1 / 2: int8 activations for the layer-major expert GEMMs (per-32 / per-token scales) |  |
| `TQFN_TAIL_SKIP` | - | R: in decode, skip non-resident experts ranked R-th or lower for every token of a call (approximate) |  |

## Debug

| Flag | Default | What it does | Use instead |
|---|---|---|---|
| `TQLLM_DEBUG` | off | comma list of diagnostics: memlog (device memory per load stage), profile (spec round phases), steps (decode step timings), stageprof (eager verify, per-stage time), matvec (kernel picks), tune (load-time tuner picks), prefill (progress), draft (first draft / accept decisions), tree_chain (tree verify as a chain), tf_dump:FILE (teacher-forced argmax per prompt position), fakeq_k:N / fakeq_v:N (quantise-dequantise K / V to N bits); Flash-Next / MoE diagnostics: arena, fn_prefill_log, cpu_check, qsa_check, pf_mmq_check:N, profile_req:N, profile_decode, route_dump:FILE, pred_dump:FILE, dump_gdn:FILE, expert_profile:FILE, keep_map:FILE, moe_profile:FILE, minimax_dump:FILE, minimax_routes:FILE, stagemap (MiniMax verify-step stage map), accept (MiniMax acceptance per draft depth), dump_feats:DIR / replay_feats:DIR (DSpark drafter training rows), glm_dump, glm_no_graph, graphs (each CUDA graph cache miss with the cache size, evictions and free device memory), composite_serial (composite matrix parts one after another on the main stream instead of concurrently: A/B, same results), mtp_eager (one-token MTP draft passes eagerly instead of as captured graphs: A/B, same results), commit_overlap (opt-in: --mtp commit replay on a side stream overlapping the draft passes; same results, single slot only so far); Metal: no_matrix_unit (the M1-M4 paths on an M5: MLX affine weights, no tensor-op attention), ane_serial, ane_timing, count_nodes, dflash_rankstats, stream_group:N (a draft-ahead verify submits every N layers, default 4), nostream (one submission per forward), tree_rows:N (token-tree verify rows, 16 or 32), no_far (drafter ring window only), retr_split, retr_share, lattice_split, build_split, dump_ffn:DIR, dump_ffn_slice:N, wired_mib:N; Metal bench tools: dflash_spec_only, dflash_warm_tokens:N, draft_stages, draft_bench:N, kl_rows:K, kl_slice:N |  |

## Retired (the tool stops; use the replacement)

| Flag | Default | What it does | Use instead |
|---|---|---|---|
| `TQLLM_DEBUG_DRAFT` | - | removed | TQLLM_DEBUG=draft |
| `TQLLM_DFLASH` | - | removed | -md / --model-draft FILE |
| `TQLLM_DRAFT_D` | - | removed | tqserve --draft-max N |
| `TQLLM_DRAFT_VOCAB` | - | removed | --draft-vocab FILE |
| `TQLLM_FAKEQ_K` | - | removed | TQLLM_DEBUG=fakeq_k:N |
| `TQLLM_FAKEQ_V` | - | removed | TQLLM_DEBUG=fakeq_v:N |
| `TQLLM_IQ3_OLD` | - | removed | TQLLM_MATVEC_RULE=21:legacy |
| `TQLLM_IQ3_PF` | - | removed | TQLLM_MATVEC_RULE=21:split16:PF |
| `TQLLM_IQ4N_PF` | - | removed | TQLLM_MATVEC_RULE=14:split16:PF |
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
| `TQLLM_MATVEC_LOG` | - | removed | TQLLM_DEBUG=matvec |
| `TQLLM_MEMLOG` | - | removed | TQLLM_DEBUG=memlog |
| `TQLLM_PREFILL_BUF` | - | removed | -ub / --ubatch-size N |
| `TQLLM_PREFILL_PROGRESS` | - | removed | TQLLM_DEBUG=prefill |
| `TQLLM_PROFILE` | - | removed | TQLLM_DEBUG=profile |
| `TQLLM_REPEAT_LIMIT` | - | removed | tqserve --repeat-limit N |
| `TQLLM_STAGEPROF` | - | removed | TQLLM_DEBUG=stageprof |
| `TQLLM_STEP_PROF` | - | removed | TQLLM_DEBUG=steps |
| `TQLLM_TF_DUMP` | - | removed | TQLLM_DEBUG=tf_dump:FILE |
| `TQLLM_TQ3_OLD` | - | removed | TQLLM_MATVEC_RULE=3:legacy |
| `TQLLM_TQ3_TUNE_LOG` | - | removed | TQLLM_DEBUG=tune |
| `TQLLM_TQ4_OLD` | - | removed | TQLLM_MATVEC_RULE=4:legacy |
| `TQLLM_TQ4_TUNE_LOG` | - | removed | TQLLM_DEBUG=tune |
| `TQLLM_TREE` | - | removed | tree verify is on with -md; --no-tree turns it off |
| `TQLLM_TREE_CHAIN` | - | removed | TQLLM_DEBUG=tree_chain |
| `TQLLM_VRAM_MARGIN_MB` | - | removed | -fitt MiB (-fit on|off) |

## Removed in 1.5.0 (no effect; use the replacement)

| Flag | Use instead |
|---|---|
| `TQLLM_MLX_ANE_FFN` | none: the Neural Engine prefill split is the default (no Core ML models needed) |
| `TQLLM_MLX_ANE_WAVE` | none: the Neural Engine prefill split is the default |

## Removed in 1.4.0 (no effect; use the replacement)

| Flag | Use instead |
|---|---|
| `TQFN_ALLOW_NP_ABOVE_8` | none: fixed at its measured default |
| `TQFN_ARENA_LOG` | TQLLM_DEBUG=arena |
| `TQFN_AUTO_KEEP_MB` | none: fixed at its measured default |
| `TQFN_BF16_ACT` | none: fixed at its measured default |
| `TQFN_CPU_CHECK` | TQLLM_DEBUG=cpu_check |
| `TQFN_CPU_GGUF_ROWS` | none: fixed at its measured default |
| `TQFN_CPU_MISS_MAX` | none: fixed at its measured default |
| `TQFN_CPU_OWNED` | removed: the CPU-owned eighth is a fixed rule (decode_rules.hpp cpu_owned), kept out of the GPU banks |
| `TQFN_CPU_OWNED_COLD` | removed (profile-owned cold experts lost on the 3090) |
| `TQFN_CPU_OWNED_PROFILE` | removed with TQFN_CPU_OWNED_COLD |
| `TQFN_CPU_REFILL` | none: fixed at its measured default |
| `TQFN_CPU_SPIN_MS` | none: fixed at its measured default |
| `TQFN_DRAFT_TAU` | none: the engine sets the draft stop at load |
| `TQFN_DUMP_GDN` | TQLLM_DEBUG=dump_gdn:FILE |
| `TQFN_EXACT_F32` | none: fixed at its measured default |
| `TQFN_EXPERT_GEMM_MIN` | none: fixed at its measured default |
| `TQFN_EXPERT_PROFILE` | TQLLM_DEBUG=expert_profile:FILE |
| `TQFN_F32_ROUTER` | none: fixed at its measured default |
| `TQFN_GDN_CHUNK_MIN` | none: fixed at its measured default |
| `TQFN_GDN_SEQ_FAST` | none: fixed at its measured default |
| `TQFN_GDN_VPW` | fixed behaviour (8 value columns per warp, bit-identical) |
| `TQFN_GDN_WY` | none: fixed at its measured default |
| `TQFN_GEMV_TILE` | none: fixed at its measured default |
| `TQFN_KEEP_MAP` | TQLLM_DEBUG=keep_map:FILE |
| `TQFN_LAYER_MAJOR` | none: fixed at its default |
| `TQFN_LAYER_MAJOR_MIN` | none: fixed at its measured default |
| `TQFN_MTP_CATCHUP` | none: fixed at its measured default |
| `TQFN_MTP_LAYER_MAJOR` | none: fixed at its measured default |
| `TQFN_NO_LAYER_RESET` | none: fixed at its measured default |
| `TQFN_NO_PREFETCH` | none: fixed at its measured default |
| `TQFN_OLD_Q2_DOT` | fixed behaviour (warp-per-row GSQ Q2_0 dot) |
| `TQFN_PF_CHUNK_KB` | fixed behaviour (one DMA per prefetched expert; chunking lost on the 3090) |
| `TQFN_PF_EVENTS` | fixed behaviour (bank-wide prefetch wait; per-slot events were neutral and could find no slot, F33) |
| `TQFN_PF_K` | none: fixed at its measured default |
| `TQFN_PF_KEEP` | none: fixed at its measured default |
| `TQFN_PF_MMQ_CHECK` | TQLLM_DEBUG=pf_mmq_check:N |
| `TQFN_PREDICT_N` | none: the engine sets the prediction depth at load |
| `TQFN_PRED_DUMP` | TQLLM_DEBUG=pred_dump:FILE |
| `TQFN_PREFILL_INT8_FMT` | none: fixed at its default |
| `TQFN_PREFILL_LOG` | TQLLM_DEBUG=fn_prefill_log |
| `TQFN_PROFILE_DECODE` | TQLLM_DEBUG=profile_decode |
| `TQFN_PROFILE_REQ` | TQLLM_DEBUG=profile_req:N |
| `TQFN_Q2_DOT_GROUPED` | none: fixed at its default |
| `TQFN_QSA` | none: fixed at its default |
| `TQFN_QSA_CHECK` | TQLLM_DEBUG=qsa_check |
| `TQFN_QSA_SPLIT` | none: fixed at its default |
| `TQFN_ROUND32` | none: fixed at its measured default |
| `TQFN_ROUTER_GEMM` | fixed behaviour (tiled router GEMV; the GEMM flipped near-tie routing) |
| `TQFN_ROUTE_DUMP` | TQLLM_DEBUG=route_dump:FILE |
| `TQFN_TQ_AHEAD` | TQueue prefetch queue lives on branch tqllm/tqueue (not in the release) |
| `TQFN_VERIFY_WAVES` | none: fixed at its measured default |
| `TQFN_WAVE_GEMM_AVG` | none: fixed at its measured default |
| `TQLLM_ATTN_NBUF` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_BATCH_COLS` | automatic (per GPU) |
| `TQLLM_COMPOSITE` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_CONV_SERIAL` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DEFER` | automatic (per GPU) |
| `TQLLM_DFLASH_BF16` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DFLASH_I8H` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DFLASH_NOGRAPH` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DFLASH_TQ3` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DRAFT_HEAD8` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_DSUB_COPY` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_EMBED_HOST` | automatic (tqserve memory fit) |
| `TQLLM_FORCE_BATCH` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_GEMM_F32` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_GEMM_F32ACC` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_GEMM_PAD` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_GEMM_ROWS` | automatic (tqserve memory fit) |
| `TQLLM_GEN_ATTN_OLD` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_GLM_DUMP` | TQLLM_DEBUG=glm_dump |
| `TQLLM_GLM_EXPERT_PF` | none: fixed at its measured default |
| `TQLLM_GLM_NO_GRAPH` | TQLLM_DEBUG=glm_no_graph |
| `TQLLM_HEAD8` | tqkld requests full-vocabulary logits itself |
| `TQLLM_HEAD_HOST` | automatic (tqserve memory fit) |
| `TQLLM_HEAD_PER_SEG` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_HOST_CHAIN` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_HW_MT` | removed (fixed launch policy) |
| `TQLLM_IQ4_OLD` | removed (IQ4_XS loads as IQ4N) |
| `TQLLM_LOOKUP_N` | removed (--lookup matches 6 tokens) |
| `TQLLM_MINIMAX_DUMP` | TQLLM_DEBUG=minimax_dump:FILE |
| `TQLLM_MINIMAX_ROUTES` | TQLLM_DEBUG=minimax_routes:FILE |
| `TQLLM_MIX_COPY` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_MLX_ABLATE` | none: Metal release default built in |
| `TQLLM_MLX_ABLATE_BLOCKS` | none: Metal release default built in |
| `TQLLM_MLX_ABLATE_SMALL` | none: Metal release default built in |
| `TQLLM_MLX_AFFINE_FFN` | none: Metal release default built in |
| `TQLLM_MLX_AFFINE_GU` | none: Metal release default built in |
| `TQLLM_MLX_AFFINE_HALF` | none: Metal release default built in |
| `TQLLM_MLX_AFFINE_MIXER` | none: Metal release default built in |
| `TQLLM_MLX_AFFINE_PAD` | none: Metal release default built in |
| `TQLLM_MLX_AFFINE_TQ4` | none: Metal release default built in |
| `TQLLM_MLX_AFFINE_TQ4_FP32` | none: Metal release default built in |
| `TQLLM_MLX_ANE_CHECK` | none: Metal release default built in |
| `TQLLM_MLX_ANE_FULL_DEQUANT` | none: Metal release default built in |
| `TQLLM_MLX_ANE_NOISE` | none: Metal release default built in |
| `TQLLM_MLX_ANE_SERIAL` | none: the Core ML Neural Engine path was removed in 1.5.0 (the Neural Engine prefill split is the default) |
| `TQLLM_MLX_ANE_SKIP_PREDICT` | none: Metal release default built in |
| `TQLLM_MLX_ANE_TIMING` | none: the Core ML Neural Engine path was removed in 1.5.0 (the Neural Engine prefill split is the default) |
| `TQLLM_MLX_ATTENTION_KV_VIEWS` | none: Metal release default built in |
| `TQLLM_MLX_ATTN_DECODE_ROWS` | none: Metal release default built in |
| `TQLLM_MLX_ATTN_DECODE_SDPA` | none: always on (fused SDPA decode attention) |
| `TQLLM_MLX_ATTN_DECODE_V2` | none: Metal release default built in |
| `TQLLM_MLX_ATTN_DECODE_V3` | none: Metal release default built in |
| `TQLLM_MLX_ATTN_PREFILL_BF16` | none: Metal release default built in |
| `TQLLM_MLX_ATTN_Q_CHUNK` | none: Metal release default built in |
| `TQLLM_MLX_ATTN_V3_PB` | none: Metal release default built in |
| `TQLLM_MLX_BATCH_DECODE_DENSE` | none: Metal release default built in |
| `TQLLM_MLX_CAPTURE_VERIFY` | none: Metal release default built in |
| `TQLLM_MLX_COMPACT_GDN_TAPE` | none: Metal release default built in |
| `TQLLM_MLX_COMPILED_GDN` | none: Metal release default built in |
| `TQLLM_MLX_COUNT_NODES` | TQLLM_DEBUG=count_nodes |
| `TQLLM_MLX_DECODED_FFN` | none: Metal release default built in |
| `TQLLM_MLX_DEVICE_SELECTOR` | none: Metal release default built in |
| `TQLLM_MLX_DFLASH_BLOCK` | none: the drafter GGUF's block size (DFlash2 block_rows) |
| `TQLLM_MLX_DFLASH_CHAIN` | none: Metal release default built in |
| `TQLLM_MLX_DFLASH_PIPELINE` | none: Metal release default built in |
| `TQLLM_MLX_DFLASH_PIPELINE_SYNC_BUILD` | none: Metal release default built in |
| `TQLLM_MLX_DFLASH_RANKSTATS` | TQLLM_DEBUG=dflash_rankstats |
| `TQLLM_MLX_DFLASH_SHARDED_TOP16` | none: Metal release default built in |
| `TQLLM_MLX_DFLASH_SPEC_ONLY` | TQLLM_DEBUG=dflash_spec_only |
| `TQLLM_MLX_DFLASH_WARM_TOKENS` | TQLLM_DEBUG=dflash_warm_tokens:N |
| `TQLLM_MLX_DRAFT_ABLATE` | none: Metal release default built in |
| `TQLLM_MLX_DRAFT_BUILD_PROFILE` | none: Metal release default built in |
| `TQLLM_MLX_DRAFT_HEAD4` | none: Metal release default built in |
| `TQLLM_MLX_DRAFT_HEAD_UNPAD` | none: Metal release default built in |
| `TQLLM_MLX_DROP_TILED_ORIGINALS` | none: Metal release default built in |
| `TQLLM_MLX_DUMP_FFN` | none: the Core ML Neural Engine path was removed in 1.5.0 (the Neural Engine prefill split is the default) |
| `TQLLM_MLX_DUMP_FFN_SLICE` | none: the Core ML Neural Engine path was removed in 1.5.0 (the Neural Engine prefill split is the default) |
| `TQLLM_MLX_DUMP_KV` | none: Metal release default built in |
| `TQLLM_MLX_EAGER_ACCEPT` | none: Metal release default built in |
| `TQLLM_MLX_EAGER_COMMIT` | none: Metal release default built in |
| `TQLLM_MLX_FFN_GEMM` | none: Metal release default built in |
| `TQLLM_MLX_FUSED_ATTENTION_ROWS` | none: Metal release default built in |
| `TQLLM_MLX_FUSE_EPILOGUE` | none: Metal release default built in |
| `TQLLM_MLX_FUSE_EW` | none: Metal release default built in |
| `TQLLM_MLX_FUSE_SWIGLU_ROTATION` | none: Metal release default built in |
| `TQLLM_MLX_GDN_CHUNKED` | none: Metal release default built in |
| `TQLLM_MLX_GDN_FUSE` | none: Metal release default built in |
| `TQLLM_MLX_GDN_FUSE_PREFILL` | none: Metal release default built in |
| `TQLLM_MLX_GDN_SCAN` | none: Metal release default built in |
| `TQLLM_MLX_GDN_SCAN8` | none: Metal release default built in |
| `TQLLM_MLX_GDN_STEP` | none: Metal release default built in |
| `TQLLM_MLX_HEAD_FRAG` | none: Metal release default built in |
| `TQLLM_MLX_INT8_N` | none: Metal release default built in |
| `TQLLM_MLX_INT8_SIMDS` | none: Metal release default built in |
| `TQLLM_MLX_INT8_TILE` | none: Metal release default built in |
| `TQLLM_MLX_KV_F32` | none: Metal release default built in |
| `TQLLM_MLX_LAZY_INJECT` | none: Metal release default built in |
| `TQLLM_MLX_LAZY_VERIFY_COMMIT` | none: Metal release default built in |
| `TQLLM_MLX_MATMUL_COLS` | none: Metal release default built in |
| `TQLLM_MLX_MATMUL_SIMDS` | none: Metal release default built in |
| `TQLLM_MLX_MMA8` | none: Metal release default built in |
| `TQLLM_MLX_MMA8_DRAFT` | none: Metal release default built in |
| `TQLLM_MLX_MMA8_UNTILED_ONLY` | none: Metal release default built in |
| `TQLLM_MLX_NATIVE_GDN` | none: Metal release default built in |
| `TQLLM_MLX_NATIVE_GDN_PREP` | none: Metal release default built in |
| `TQLLM_MLX_NAX_DRAFT_HEAD4` | none: Metal release default built in |
| `TQLLM_MLX_NAX_DRAFT_SHORTLIST` | none: removed (experiment, off) |
| `TQLLM_MLX_NAX_FUSE` | none: Metal release default built in |
| `TQLLM_MLX_NAX_MAX_SPLIT` | none: Metal release default built in |
| `TQLLM_MLX_NAX_MR64` | none: Metal release default built in |
| `TQLLM_MLX_NAX_NO_INPUT_CACHE` | none: Metal release default built in |
| `TQLLM_MLX_NAX_PREFILL_GEMM` | none: Metal release default built in |
| `TQLLM_MLX_NAX_Q6_HEAD` | none: Metal release default built in |
| `TQLLM_MLX_NAX_ROW_CHECK` | none: Metal release default built in |
| `TQLLM_MLX_NAX_SPLIT_TARGET` | none: Metal release default built in |
| `TQLLM_MLX_NAX_TQ4` | none: Metal release default built in |
| `TQLLM_MLX_PAIRED_GATE_UP` | none: Metal release default built in |
| `TQLLM_MLX_PAIRED_SWIGLU_ROTATION` | none: Metal release default built in |
| `TQLLM_MLX_PREFILL_GEMM` | none: Metal release default built in |
| `TQLLM_MLX_PREFILL_INT8` | none: removed (opt-in experiment; its quality loss closed it) |
| `TQLLM_MLX_PREFILL_INT8_MASK` | none: Metal release default built in |
| `TQLLM_MLX_Q6_AFFINE` | none: Metal release default built in |
| `TQLLM_MLX_REUSE_ROTATION` | none: Metal release default built in |
| `TQLLM_MLX_SERIAL_VERIFY_ATTENTION` | none: Metal release default built in |
| `TQLLM_MLX_SERIAL_VERIFY_DENSE` | none: Metal release default built in |
| `TQLLM_MLX_SPLIT_COMMIT_TIMING` | none: Metal release default built in |
| `TQLLM_MLX_STAGED_BKM` | none: Metal release default built in |
| `TQLLM_MLX_STAGED_KS` | none: Metal release default built in |
| `TQLLM_MLX_STAGED_MT` | none: Metal release default built in |
| `TQLLM_MLX_STAGED_PERSG` | none: Metal release default built in |
| `TQLLM_MLX_STAGED_SIMDS` | none: Metal release default built in |
| `TQLLM_MLX_TARGET_HEAD_HOST` | none: Metal release default built in |
| `TQLLM_MLX_TARGET_HEAD_IDS` | none: Metal release default built in |
| `TQLLM_MLX_TARGET_HEAD_SHORTLIST` | none: Metal release default built in |
| `TQLLM_MLX_TENSOR_K` | none: Metal release default built in |
| `TQLLM_MLX_TENSOR_M` | none: Metal release default built in |
| `TQLLM_MLX_TENSOR_MEMORY` | none: Metal release default built in |
| `TQLLM_MLX_TENSOR_N` | none: Metal release default built in |
| `TQLLM_MLX_TENSOR_SIMDS` | none: Metal release default built in |
| `TQLLM_MLX_TENSOR_SPLIT` | none: Metal release default built in |
| `TQLLM_MLX_TENSOR_TRANSPOSE` | none: Metal release default built in |
| `TQLLM_MLX_TENSOR_WIDE` | none: Metal release default built in |
| `TQLLM_MLX_TILED_DFLASH_FP32` | none: Metal release default built in |
| `TQLLM_MLX_TILED_MIXER_FP16` | none: Metal release default built in |
| `TQLLM_MLX_TILED_MIXER_FP32` | none: Metal release default built in |
| `TQLLM_MLX_TILED_TQ3_FP16` | none: Metal release default built in |
| `TQLLM_MLX_TILED_TQ3_FP32` | none: Metal release default built in |
| `TQLLM_MLX_TILED_TQ4_FP16` | none: Metal release default built in |
| `TQLLM_MLX_TILED_TQ4_FP32` | none: Metal release default built in |
| `TQLLM_MLX_TQ3_STREAM` | none: Metal release default built in |
| `TQLLM_MLX_TQ3_SUBGROUP` | none: Metal release default built in |
| `TQLLM_MLX_TQ4_ROTATE_UNFUSED` | none: Metal release default built in |
| `TQLLM_MLX_WIRED_MIB` | TQLLM_DEBUG=wired_mib:N |
| `TQLLM_MOE_DEV` | nothing: qwen35moe (MoE) models load by default |
| `TQLLM_MOE_FUSE_SHARED` | none: fixed at its measured default |
| `TQLLM_MOE_PROFILE` | TQLLM_DEBUG=moe_profile:FILE |
| `TQLLM_MOE_STAGE` | none: fixed at its measured default |
| `TQLLM_MQ_SPLIT` | automatic |
| `TQLLM_NO_BLKMERGE` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_NO_BTREE` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_NO_RMSROPE` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_NO_SIDE` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_PF_GU_BF16` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_PF_MIX_COPY` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_PREFILL_ATTN_GROUPED` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_PREFILL_ATTN_OLD` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_PREFILL_CHUNK` | automatic (-ub sets the prefill buffer) |
| `TQLLM_Q4K_U4H` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_REPEAT_EXIT` | removed |
| `TQLLM_SERIAL_DRAFT` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_SERIAL_SEGS` | fixed behaviour (settled A/B switch removed) |
| `TQLLM_TQ3_HW` | removed (fixed launch policy) |
| `TQLLM_TQ3_MT` | removed (fixed launch policy) |
| `TQLLM_TQ3_MT_KMIN` | removed (fixed launch policy: k >= 12288) |
| `TQLLM_TQ3_P` | removed (fixed launch policy) |
| `TQLLM_TQ3_TUNE` | removed (the load-time tuner always runs) |
| `TQLLM_TQ4_TUNE` | removed (no in-model gain) |
| `TQLLM_TQ4_VARIANT` | removed (variant 0) |
| `TQLLM_TREE_TEMP` | removed (1.0) |
| `TQLLM_U4H` | removed |
