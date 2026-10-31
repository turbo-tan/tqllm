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
| `TQLLM_MLX_PREFILL_CHUNK` | - | N: prefill chunk rows for tqmlx-dflash (>= 128; 2048 is the measured best) |  |
| `TQLLM_MATVEC_RULE` | - | override the matvec kernel rule table (src/formats/weight/matvec_rules.hpp) per format bits: "BITS:KERNEL[:PF],... (KERNEL legacy or split16)", e.g. "3:split16:2,21:legacy" (3 TQ3, 4 TQ4, 14 IQ4N, 21 IQ3) |  |
| `TQLLM_GLM_EXPERT_SLOTS` | 0 | GLM-5.3-Flash: keep only this many routed experts of every MoE layer resident, in a MixedExpertCache bank (0: the whole expert stack on the device). The experts a token needs are streamed from the GGUF / the packed file, so the tokens do not change; it also turns off the decode CUDA graph |  |
| `TQLLM_GLM_EXPERT_BUDGET_MB` | banks + 25 % | GLM-5.3-Flash: device bytes the expert banks plus loader staging may hold. With TQLLM_GLM_EXPERT_SLOTS it caps them; on its own it turns on the shared byte-budget plan (src/experts/expert_bank_plan.hpp), which gives every MoE layer top-k slots and the rest of the bytes to the layers whose routing spreads over more experts (measured over the prompt, re-planned when it ends) |  |
| `TQLLM_GLM_EXPERTS` | - | GLM-5.3-Flash: packed expert file (glm-pack-experts MODEL.gguf OUT.tqfnexp) for the expert banks: a bank miss copies the expert instead of repacking it; the engine page-locks it when it is at most half the host RAM, which also lets it prefetch (TQLLM_GLM_EXPERT_PF); a bigger file copies misses through pinned staging |  |
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
| `TQLLM_MLX_ANE_SERIAL` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_ANE_TIMING` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_COUNT_NODES` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_RANKSTATS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_SPEC_ONLY` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DFLASH_WARM_TOKENS` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_PREFILL_FUSED` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_KL_ROWS` | - | K: with tqmlx-probe --kl-reference, print the K worst prompt positions of each chunk by KL (diagnoses which rows the ANE split gets wrong) |  |
| `TQLLM_KL_SLICE` | - | N: with tqmlx-probe --kl-reference, print the mean KL and the count of rows with KL > 1 for each N-position slice of the prompt (finds whether the ANE split's error sits in one slice or follows tokens) |  |
| `TQLLM_MLX_ATTN_QK_FUSED` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_PREFILL_PACK` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DRAFT_STAGES` | - | tqmlx-dflash draft bench only: time each stage of the draft-ahead draft (layers, head, top-16, selector) |  |
| `TQLLM_MLX_DRAFT_BENCH` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DUMP_FFN` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_DUMP_FFN_SLICE` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_MLX_NAX_DRAFT_SHORTLIST` | - | Apple MLX backend experiment or diagnostic switch; verdict and removal plan in docs/steering FLAGS-TO-PRUNE (MLX rows) |  |
| `TQLLM_DEBUG` | off | comma list of diagnostics: memlog (device memory per load stage), profile (spec round phases), steps (decode step timings), stageprof (eager verify, per-stage time), matvec (kernel picks), tune (load-time tuner picks), prefill (progress), draft (first draft / accept decisions), tree_chain (tree verify as a chain), tf_dump:FILE (teacher-forced argmax per prompt position), fakeq_k:N / fakeq_v:N (quantise-dequantise K / V to N bits); Flash-Next / MoE diagnostics: arena, fn_prefill_log, cpu_check, qsa_check, pf_mmq_check:N, profile_req:N, profile_decode, route_dump:FILE, pred_dump:FILE, dump_gdn:FILE, expert_profile:FILE, keep_map:FILE, moe_profile:FILE, minimax_dump:FILE, minimax_routes:FILE, glm_dump, glm_no_graph |  |
| `TQLLM_MLX_WIRED_MIB` | unset (retain MLX default) | Experimental process-local persistent Metal residency budget in MiB for MLX benchmark tools; 0 disables wiring. Capped by the device recommendation, restored on scope exit. Does not alter system settings. |  |
| `TQFN_DENSE_INT8` | 0 | 1 / 2: int8 activations for the dense prefill projections (per-32 / per-token scales); 0 = fp16 (exact path) |  |
| `TQFN_DRAFT_TAU` | 0.9 (np 1) | t: stop MTP drafting once the last draft's top probability is below t (text unchanged); one sequence defaults to 0.9 and up to 5 drafts; 0 = off |  |
| `TQFN_PREDICT_N` | 4 (6 with adaptive drafts) | next-layer prediction (prefetch) for verify blocks of up to N tokens |  |
| `TQFN_PREFILL_INT8` | 0 | 1 / 2: int8 activations for the layer-major expert GEMMs (per-32 / per-token scales) |  |
| `TQFN_TAIL_SKIP` | - | R: in decode, skip non-resident experts ranked R-th or lower for every token of a call (approximate) |  |

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
