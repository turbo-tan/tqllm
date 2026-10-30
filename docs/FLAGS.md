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
| `TQLLM_MLX_DRAFT_AHEAD` | 1 | Apple MLX greedy decode: verify(k) and draft(k+1) chained on the GPU (gated j892); 0 turns it off |  |
| `TQLLM_MLX_GDN_STEP` | 1 | Apple MLX decode: one GDN launch per verify step (gated j892); 0 turns it off |  |
| `TQLLM_MLX_PREFILL_CHUNK` | - | N: prefill chunk rows for tqmlx-dflash (>= 128; 2048 is the measured best) |  |
| `TQLLM_MATVEC_RULE` | - | override the matvec kernel rule table (src/formats/weight/matvec_rules.hpp) per format bits: "BITS:KERNEL[:PF],... (KERNEL legacy or split16)", e.g. "3:split16:2,21:legacy" (3 TQ3, 4 TQ4, 14 IQ4N, 21 IQ3) |  |
| `TQLLM_GLM_EXPERT_SLOTS` | 0 | GLM-5.3-Flash: keep only this many routed experts of every MoE layer resident, in a MixedExpertCache bank (0: the whole expert stack on the device). The experts a token needs are streamed from the GGUF / the packed file, so the tokens do not change; it also turns off the decode CUDA graph |  |
| `TQLLM_GLM_EXPERT_BUDGET_MB` | banks + 25 % | GLM-5.3-Flash: device bytes the expert banks plus loader staging may hold. With TQLLM_GLM_EXPERT_SLOTS it caps them; on its own it turns on the shared byte-budget plan (src/experts/expert_bank_plan.hpp), which gives every MoE layer top-k slots and the rest of the bytes to the layers whose routing spreads over more experts (measured over the prompt, re-planned when it ends) |  |
| `TQLLM_GLM_EXPERTS` | - | GLM-5.3-Flash: packed expert file (glm-pack-experts MODEL.gguf OUT.tqfnexp) for the expert banks: a bank miss copies the expert instead of repacking it; the engine page-locks it when it is at most half the host RAM, which also lets it prefetch (TQLLM_GLM_EXPERT_PF); a bigger file copies misses through pinned staging |  |
| `TQLLM_MLX_DRAFT_HEAD4` | 2 | Apple MLX draft proposal head: 2 = 4-bit head in the matrix-unit layout (gated j892, default), 1 = row layout, 0 = off |  |
| `TQFN_ARENA_MB` | - | MB: Flash-Next scratch arena size instead of the measured rule (the high-water mark is printed at exit) |  |
| `TQFN_CPU_MISS` | 0 | N: compute up to N missed experts per layer call on the CPU instead of copying them to the GPU (default on small cards: min(15, cores / 2 - 1); else off) |  |
| `TQFN_DRAFT_VOCAB` | - | FILE: the MTP draft vocabulary (overrides the manifest's draft_vocab) |  |
| `TQFN_KV` | - | KV format for Flash-Next attention (name from the KV format registry; default bf16) |  |
| `TQFN_KV_HOST` | auto | auto (default): on for discrete GPUs, off on unified memory; 1 / 0 force it. On: K/V older than TQFN_KV_HOST_KEEP positions lives in host memory after a long prompt (frees VRAM for expert banks; decode copies only the selected rows) |  |
| `TQFN_KV_HOST_KEEP` | 32768 | positions of K/V kept on the device with TQFN_KV_HOST=1 |  |
| `TQFN_LAYER_SLOTS` | - | comma list of expert slots per layer (overrides the packed file's .slots profile) |  |
| `TQFN_MTP_SLOTS` | 256 | MTP expert slots (every expert when the trunk keeps every expert; 64 on small cards) |  |

## Debug

| Flag | Default | What it does | Use instead |
|---|---|---|---|
| `TQFN_KEEP_MAP` | - | FILE: private compact-expert diagnostic map; incompatible with the full-routing hot/cold path |  |
| `TQLLM_MLX_ANE_SERIAL` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ANE_STUB` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ANE_TIMING` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_COUNT_NODES` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_RANKSTATS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_COMPILE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_RING` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_SPEC_ONLY` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_WARM_TOKENS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_PREFILL_GROUP` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_GDN_SCAN_STAGED` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_PREFILL_FUSED` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_KL_ROWS` | - | K: with tqmlx-probe --kl-reference, print the K worst prompt positions of each chunk by KL (diagnoses which rows the ANE split gets wrong) |  |
| `TQLLM_KL_SLICE` | - | N: with tqmlx-probe --kl-reference, print the mean KL and the count of rows with KL > 1 for each N-position slice of the prompt (finds whether the ANE split's error sits in one slice or follows tokens) |  |
| `TQLLM_MLX_ATTN_QK_FUSED` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_PREFILL_PACK` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DRAFT_STAGES` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DRAFT_BENCH` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DUMP_FFN` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DUMP_FFN_SLICE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_DRAFT_SHORTLIST` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MOE_FUSE_SHARED` | on | 0: decode keeps the shared expert as separate dense matvecs instead of one more assignment in the routed expert launches |  |
| `TQLLM_MOE_STAGE` | on | 0: --moe-host-mb decode reads host experts in place instead of copying the routed ones to a VRAM stage first (diagnostics) |  |
| `TQLLM_MOE_PROFILE` | - | FILE: write MoE routing counts per layer and expert (read by --moe-profile for --moe-host-mb); rewritten after every prefill |  |
| `TQLLM_DEBUG` | off | comma list of diagnostics: memlog (device memory per load stage), profile (spec round phases), steps (decode step timings), stageprof (eager verify, per-stage time), matvec (kernel picks), tune (load-time tuner picks), prefill (progress), draft (first draft / accept decisions), tree_chain (tree verify as a chain), tf_dump:FILE (teacher-forced argmax per prompt position), fakeq_k:N / fakeq_v:N (quantise-dequantise K / V to N bits) |  |
| `TQLLM_GLM_NO_GRAPH` | off | set (any value): the GLM-5.3-Flash engine decodes eagerly instead of replaying one captured single-token CUDA graph (same tokens, slower). The bring-up scripts' GLM_NO_GRAPH is read as well |  |
| `TQLLM_GLM_DUMP` | off | set (any value): GLM-5.3-Flash prints the per-stage sums of the forward (same names as glm-ref and llama-eval-callback); implies TQLLM_GLM_NO_GRAPH. The bring-up scripts' GLM_DUMP is read as well |  |
| `TQLLM_MINIMAX_DUMP` | off | FILE: MiniMax-M3.1-Flash writes the residual stream after every layer to FILE (records int32 layer, pos, n + float [n][hidden]) for the per-layer comparison with tools/minimax_ref.py |  |
| `TQLLM_MINIMAX_ROUTES` | off | FILE: MiniMax-M3.1-Flash writes its per-layer routed-expert counts (int64 [layers][experts]) at exit, the input of a hot / cold expert map |  |
| `TQLLM_GLM_EXPERT_PF` | on | 0: GLM-5.3-Flash does not prefetch the experts a MoE layer wanted at the previous token while that layer runs its attention (needs a pinned TQLLM_GLM_EXPERTS file; placement only, the tokens do not change) |  |
| `TQLLM_MLX_WIRED_MIB` | unset (retain MLX default) | Experimental process-local persistent Metal residency budget in MiB for MLX benchmark tools; 0 disables wiring. Capped by the device recommendation, restored on scope exit. Does not alter system settings. |  |
| `TQFN_ALLOW_NP_ABOVE_8` | off | allow Flash-Next servers with more than 8 slots (known to end some requests after one token above 8) |  |
| `TQFN_ARENA_LOG` | off | print the scratch arena high-water mark after each prefill |  |
| `TQFN_BF16_ACT` | off | BF16 instead of FP16 activations for the Flash-Next native projections (A/B) |  |
| `TQFN_CPU_CHECK` | off | check the CPU miss pool's int8 gate/up against fp32 and its packed-row rebuild against the GGUF |  |
| `TQFN_CPU_GGUF_ROWS` | off | debug A/B: the CPU miss pool reads GGUF rows instead of rebuilding them from the pinned packed copy |  |
| `TQFN_AUTO_KEEP_MB` | - | MB slots=auto keeps free beside the expert banks instead of the full-context attention room (small cards: 512); long prompts borrow banks |  |
| `TQFN_EXPERT_PROFILE` | - | seed the expert banks from a routing profile (tools/fn_expert_profile.py); measured no effect on the 12 GB first request (prune) |  |
| `TQFN_CPU_MISS_MAX` | - | K: at most K distinct experts per layer call on the CPU miss pool; the rest are GPU misses (small cards: 5) |  |
| `TQFN_CPU_REFILL` | off | refill the expert banks from the CPU pool's copies (A/B) |  |
| `TQFN_CPU_SPIN_MS` | 2 | ms the CPU miss workers spin before sleeping |  |
| `TQFN_DENSE_INT8` | 0 | 1 / 2: int8 activations for the dense prefill projections (per-32 / per-token scales); 0 = fp16 (exact path) |  |
| `TQFN_DUMP_GDN` | - | FILE: dump the GDN layer outputs (fixture comparison) |  |
| `TQFN_EXACT_F32` | off | every Flash-Next projection on the fp32 reference path (slow; numerics gate) |  |
| `TQFN_EXPERT_GEMM_MIN` | 65 | rows from which prefill experts use the grouped GEMM instead of the dedup kernel |  |
| `TQFN_F32_ROUTER` | off | the fp32 per-token router kernel instead of the BF16 GEMV / segmented launch |  |
| `TQFN_GDN_CHUNK_MIN` | 64 | tokens from which GDN prefill uses the chunked (WY) kernel; 0 = off |  |
| `TQFN_GDN_SEQ_FAST` | on | 0: the plain sequential GDN kernel instead of the fast pass at dim 128 |  |
| `TQFN_GDN_WY` | on | 0: GDN chunked prefill without the WY form (A/B) |  |
| `TQFN_GEMV_TILE` | 16 | router GEMV tile: 8, 16, 164 (16 x 4) or 32 (same output bits) |  |
| `TQFN_LAYER_MAJOR` | on | 0: prefill chunks token-major instead of layer by layer (routed experts copied once per layer) |  |
| `TQFN_LAYER_MAJOR_MIN` | 256 | tokens from which a prefill chunk runs layer-major |  |
| `TQFN_MTP_CATCHUP` | - | positions per MTP catch-up batch when the MTP cache holds every expert |  |
| `TQFN_MTP_LAYER_MAJOR` | on | 0: the MTP catch-up runs token-major instead of layer-major |  |
| `TQFN_NO_LAYER_RESET` | off | keep the scratch arena across layers (A/B) |  |
| `TQFN_NO_PREFETCH` | off | no predicted-expert prefetch in decode |  |
| `TQFN_PF_K` | 6 | prefetch each token's K best predicted experts |  |
| `TQFN_PF_KEEP` | off | 1: keep the layer-major buffers and staging halves across requests |  |
| `TQFN_DRAFT_TAU` | 0.9 (np 1) | t: stop MTP drafting once the last draft's top probability is below t (text unchanged); one sequence defaults to 0.9 and up to 5 drafts; 0 = off |  |
| `TQFN_VERIFY_WAVES` | off | 1: verify blocks pick the waves path by assignment count (pre-F39 rule; np > 1 text then differs from np 1) |  |
| `TQFN_PREDICT_N` | 4 (6 with adaptive drafts) | next-layer prediction (prefetch) for verify blocks of up to N tokens |  |
| `TQFN_PF_MMQ_CHECK` | 0 | N: check up to N layer-major MMQ expert GEMMs against the fp16 grouped kernel |  |
| `TQFN_PRED_DUMP` | - | FILE: append routed and predicted next-layer ids per trunk layer call (cache simulation) |  |
| `TQFN_PREFILL_INT8` | 0 | 1 / 2: int8 activations for the layer-major expert GEMMs (per-32 / per-token scales) |  |
| `TQFN_PREFILL_INT8_FMT` | - | comma list of u4h format ids allowed on the int8 prefill path (default all) |  |
| `TQFN_PREFILL_LOG` | off | per-layer and per-stage prefill timings (synchronises; slower) |  |
| `TQFN_PROFILE_DECODE` | off | 1: the TQFN_PROFILE_REQ capture covers decode instead of prefill |  |
| `TQFN_PROFILE_REQ` | - | N: profile the N-th request; also logs where a slow first token goes |  |
| `TQFN_Q2_DOT_GROUPED` | off | GSQ Q2_0 grouped experts: 1 = the scalar kernel instead of tensor cores (A/B) |  |
| `TQFN_QSA` | on | 0: dense attention at every length instead of QSA sparse attention past the indexer budget (changes output) |  |
| `TQFN_QSA_CHECK` | off | QSA selection: compare the top-k select with a full sort (raises on any difference) |  |
| `TQFN_QSA_SPLIT` | 16 | slices per decode QSA attention row; 0 or 1 = direct attention |  |
| `TQFN_ROUND32` | - | diagnostic: Q8 activation rounding of the dense projections on the GPU side (all = every tensor; default: none) |  |
| `TQFN_ROUTE_DUMP` | - | FILE: append routed expert ids per trunk layer call |  |
| `TQFN_TAIL_SKIP` | - | R: in decode, skip non-resident experts ranked R-th or lower for every token of a call (approximate) |  |
| `TQFN_WAVE_GEMM_AVG` | - | average tokens per expert from which prefill waves use the grouped GEMM |  |

## Retired (the tool stops; use the replacement)

| Flag | Default | What it does | Use instead |
|---|---|---|---|
| `TQLLM_MLX_ATTN_DECODE_V7` | - | removed | fixed behaviour (v7 is the only engine path) |
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
| `TQFN_CPU_OWNED` | - | removed | removed: the CPU-owned eighth is a fixed rule (decode_rules.hpp cpu_owned), kept out of the GPU banks |
| `TQFN_GDN_VPW` | - | removed | fixed behaviour (8 value columns per warp, bit-identical) |
| `TQFN_OLD_Q2_DOT` | - | removed | fixed behaviour (warp-per-row GSQ Q2_0 dot) |
| `TQFN_PF_EVENTS` | - | removed | fixed behaviour (bank-wide prefetch wait; per-slot events were neutral and could find no slot, F33) |
| `TQFN_CPU_OWNED_COLD` | - | removed | removed (profile-owned cold experts lost on the 3090) |
| `TQFN_CPU_OWNED_PROFILE` | - | removed | removed with TQFN_CPU_OWNED_COLD |
| `TQFN_PF_CHUNK_KB` | - | removed | fixed behaviour (one DMA per prefetched expert; chunking lost on the 3090) |
| `TQFN_TQ_AHEAD` | - | removed | TQueue prefetch queue lives on branch tqllm/tqueue (not in the release) |
| `TQFN_ROUTER_GEMM` | - | removed | fixed behaviour (tiled router GEMV; the GEMM flipped near-tie routing) |
