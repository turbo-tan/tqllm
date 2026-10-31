# Server options and numerical limits

[Documentation](../README.md) / [Project home](../../README.md)

## Options

`tqserve --help` prints the same list. In Docker, the variables below set the common ones and any other option goes
after the image name (`docker run ... ghcr.io/turbo-tan/tqllm:<tag> --prefill-int8 on`).

Docker variables: `MODEL_FILE`, `TOKENIZER_FILE`, `DRAFT_MODEL_FILE`, `DRAFT_VOCAB_FILE`, `KV_FORMAT`, `CONTEXT`
(16384), `PARALLEL` (1), `MAX_TOKENS` (1024), `REASONING` (off).

### Model and serving

| Option | Default | What it does |
|---|---|---|
| `--model FILE` / `--model-dir DIR` | | the GGUF (or a safetensors directory) to serve |
| `--tokenizer FILE` | the GGUF itself | only for safetensors or a deliberate override |
| `--host IPv4`, `--port N` | 127.0.0.1, 8086 | where the API listens |
| `--context N` | 2048 | context per server, split evenly across `--np` slots; the memory fit may lower it |
| `--max-tokens N` | 16384 | default output cap per request (also the cap for requests that ask for more; thinking counts against it) |
| `--np N` | 1 | requests served at once (batched verify); `GET`/`POST /admin/slots` reads or changes the live limit |
| `--name NAME`, `--system TEXT` | | model name in the API; a default system prompt |
| `--api-key KEY` / `--api-key-file FILE` | | require a bearer key |
| `--mmproj FILE [--image-min-tokens N] [--image-max-tokens N]` | 8, 4096 | vision projector for image input |
| `--repeat-limit N` | 256 | end a request after N identical tokens in a row (0 off) |

### More engine options

Long forms of short options: `-ctk` / `--cache-type-k`, `-ctv` / `--cache-type-v`, `-md` / `--model-draft`,
`-ub` / `--ubatch-size`, `-ncmoe` / `--n-cpu-moe`, `-fit` / `--fit`, `-fitt` / `--fit-target`, `-fitc` / `--fit-ctx`,
`-fitl` / `--fit-lossy`, `--temperature` / `--temp`.

| Option | Default | What it does |
|---|---|---|
| `--temperature T` `--top-p P` `--top-k K` `--presence-penalty X` | greedy, no penalty | sampling defaults for requests that set none |
| `--monitor-port N` | the API port | serve the read-only dashboard on its own port |
| `-fit on\|off` | on | fit the model, KV and buffers into GPU memory at load |
| `-fitt MiB` | 768 (256 under WSL) | GPU memory to leave free after loading |
| `-fitc N` | 4096 | smallest context per slot the fit may choose |
| `-fitl on\|off` | on | on: the fit may pick KV codec / context / prefill buffers and announces them; off: refuse to start instead |
| `-ub N` | engine default | prefill micro-batch rows |
| `-ncmoe N` | 0 | keep the MoE experts of the first N layers in host memory |
| `--moe-host-mb M\|auto` `--moe-profile FILE` | off | keep the least-routed MoE experts (by FILE's routing counts) in host memory, M MiB of them |
| `--moe-cache-mb C\|auto` | auto | VRAM cache of recently used host experts |
| `--hot-map FILE` | none | MiniMax: routing counts; the busiest experts are preloaded, the rest stay on disk |
| `--plugin PATH` | none | load a weight-format plugin (repeatable, CUDA builds): new GGUF types or kernel patches; each must pass the format contract at load ([plugins](../guides/plugins.md)) |
| `--tp-rank R --tp-peer IP [--tp-port P]` | single box, port 29600 | MiniMax over two linked boxes (rank 0 serves; rank 1 runs `glm-gpu MODEL --tokens 1 --tp-rank 1 ...`) |
| `--yarn-orig-ctx N` `--yarn-ext-factor F` `--yarn-attn-factor F` `--yarn-beta-fast F` `--yarn-beta-slow F` | model metadata | YaRN context extension, llama.cpp meanings (with `--rope-scaling yarn`) |
| Flash-Next prefill: `--prefill-chunk N` `--prefill-sub N` `--prefill-stage N` | 16384, 2048, 192 | rows per prefill chunk, per sub-batch, and per expert stage |
| Flash-Next: `--prefill-reserve-mb N` `--prefill-arena-mb N` `--dense-cache-mb N` | 1000, 3800, 600 | GPU memory kept free during prefill, the prefill arena, the dense-weight cache |
| Flash-Next: `--prefill-borrow auto\|all\|N` | auto | expert banks the prefill may borrow (all, or N of them) for its arena |

Environment variables (`TQLLM_*`, `TQFN_*`) are listed with their defaults and status in [FLAGS.md](../FLAGS.md).

### Exact and speed (`--exact on|off`)

`--exact on` (default) runs every step at the reference precision: FP16 / BF16 activations and KV, no int8 / FP8 GEMMs,
no CPU-served experts. Greedy output is then the same from build to build. `--exact off` lets each engine use its
measured approximations (the middle column below): faster, with a small, measured change in the output. Published speed
numbers say which one they ran.

If memory forces a smaller KV cache with `--exact on`, the server says so on stderr and in `/health`
("exact on: ..."). An approximation asked for explicitly with `--exact on` (the last column) is a conflict: the server
prints one `WARN --exact on with ...` line per option, runs the asked option, and `/health` shows
"exact on, overridden by: ...".

| Engine | What `--exact off` changes | `--exact on`, and its conflicts |
|---|---|---|
| Qwen3.8 27B, CUDA | int8 prefill GEMMs on RTX 30 / 40 / 50, FP8 on GB10 | FP16 prefill; conflicts: `--prefill-int8 on`, `--prefill-fp8 on`, a lossy `-ctk` / `-ctv` / `--kv` |
| Flash-Next (.tqfn), CUDA | the CPU miss pool on small cards (`TQFN_CPU_MISS`; route-tail skip `TQFN_TAIL_SKIP` stays opt-in) | no CPU miss pool; conflicts: `TQFN_CPU_MISS`, `TQFN_TAIL_SKIP`, `TQFN_PREFILL_INT8`, `TQFN_DENSE_INT8` |
| Qwen3.8 27B, Metal | int8 KV from 32K keys, int8-query verify attention (KL 0.003-0.010 vs `--exact on` at 32K-128K) | BF16 KV at every length; FP32 query if memory forces int8 |
| MiniMax, GLM | (their only path) | no exact path yet: they run their approximate path with a WARN |

### Reasoning and chat template

| Option | Default | What it does |
|---|---|---|
| `--reasoning on\|off` | off | thinking mode |
| `--reasoning-effort low\|medium\|xhigh` | | effort hint for the template |
| `--reasoning-budget N` (alias `--thinking-budget`) | -1 | thinking token budget (-1 unlimited, 0 end now; llama.cpp semantics) |
| `--reasoning-budget-message TEXT` | | text inserted when the budget ends the thinking |
| `--reasoning-answer-reserve N` | 8192 | tokens kept for the answer after a budgeted think (0 = budget counts inside max_tokens) |
| `--jinja` / `--no-jinja` | jinja | the model's Jinja chat template (llama.cpp's engine) or the native Qwen layout |
| `--chat-template-file FILE`, `--chat-template-kwargs JSON` | | override the template / pass template variables |

### Speculative decoding (drafter)

| Option | Default | What it does |
|---|---|---|
| `-md FILE` | | DFlash2 drafter GGUF (exact: the output equals decoding without it) |
| `--draft-vocab FILE` | | the drafter's token list (65,536 ids) |
| `--draft-max N` (`--draft N`) | 7 | drafted tokens per step (1-15); hybrid Qwen3.5 / 3.8 models verify at most 8 columns, so 7 is the effective maximum there. MiniMax with its DSpark drafter: unset, the length follows the measured acceptance (1 to the drafter's block of 7); set, it is fixed |
| `--no-tree` | tree on | verify a single draft chain instead of a token tree |
| `--draft-window N` | GGUF value | drafter attention window over the context (0 = whole context) |
| `--lookup L [--lookup-first K]` | off | prompt-lookup drafts (copies of earlier text) next to the tree; shares the verify budget |

### KV cache and long context (TqCache)

| Option | Default | What it does |
|---|---|---|
| `--kv FORMAT` / `-ctk K -ctv V` | bf16 | KV cache format; `--list-kv` lists them (tq3 = 3-bit, trellis tcq3 / tcq2, q8, q4, splits) |
| `--kv-host N` | memory fit | KV positions kept in system RAM (exact); by default the fit moves only what does not fit on the GPU |
| `--kv-stage on\|off` | on | stream the RAM part back to the GPU layer by layer while decoding |
| `--kv-compact R [--kv-compact-at idle\|turn]` | off, idle | compress old KV R-fold (not exact; one slot, tq3 or trellis KV) |
| `--prompt-cache on\|off` | off | keep the previous turn's KV so a follow-up does not re-read the document (one slot; on with `--kv-compact`) |
| `--kv-store flat` | flat | KV storage policy (only flat is served today) |
| `--rope-scaling none\|linear\|yarn --rope-scale F [--yarn-orig-ctx N]` | none | context past the trained length (llama.cpp names) |

### Memory fit and prefill

| Option | Default | What it does |
|---|---|---|
| `-fit on\|off` | on | load, measure, and step down until the margin is free, in this order: token embeddings and the LM head to host memory, smaller prefill buffers, a smaller KV format (when `--kv` is not set), prefill weight blocks, the KV tail to system RAM, then a shorter context |
| `-fitt MiB` | 768 (256 under WSL) | free memory to leave on the GPU (64 MiB floor) |
| `-fitc N` | 4096 | smallest context the fit may choose (with `--np`: the smallest context per slot) |
| `-fitl on\|off` | on | on: when the requested context / users do not fit at full settings, pick a compact KV cache (q4/tcq2), smaller prefill buffers or weight blocks, or a shorter context, and say so at start-up (and in `/health`); off: refuse instead |
| `--gemm-rows N` | 0 (fit) | prefill FP16 weight blocks of N rows (less VRAM); the memory fit picks 2048 when needed and says so |
| `--mtp` | off | draft with the model's own MTP (nextn) head instead of `-md` (text-only, greedy, `--np 1`; default depth 4, `--draft-max N`) |
| `--draft-retrieve N` | 0 (off) | with `-md`: N far chunks of 16 positions in the drafter window; long prompts get faster (32: +8-12 % at 31K-51K on a 3090) but the drafter keeps K/V for the whole context (~20 KB / token: 1.25 GiB at 64K, 3.9 GiB at 200K), so turn it on when memory allows |
| `--slot-reserve MiB` | 128 | free memory per parallel slot after loading; an `--np` that does not fit is refused at startup with the largest that does |
| `--graph-cache N` | 256 | captured CUDA graphs kept (~12 MB each); least recently used are evicted, also when free memory runs low |
| `-ub N` | automatic | prefill chunk (ubatch) size |
| `--prefill-fp8 on\|off` | off | FP8 prefill GEMMs (sm_89+): about +40 % prefill on GB10 but KLD ~0.01 vs FP16 prefill, so opt-in; slower on 16 GB cards |
| `--prefill-int8 on\|off` | on for RTX 30 / 40 / 50 | int8 prefill GEMMs (RTX 50: +11-13 % prefill on a 5060 Ti; RTX 40 by analogy) |

### Memory fit: what it does and how to override it

You normally pass only `--context` (and `--np`). If that does not fit at full settings, the server picks what fits and
prints one line, for example:

```
tqserve: fit: to fit 1 slot(s) x 131072 in GPU memory: KV cache q4/tcq2 (asked bf16), prefill buffers -ub 512,
prefill FP16 weight blocks of 2048 rows (choose yourself with --kv, -ub, --gemm-rows or a smaller --context)
```

`GET /health` reports the same in its `fit` field. Your own settings always win; examples:

| you want | add |
|---|---|
| full-quality BF16 KV, accept a shorter context if needed | `--kv bf16` |
| a specific compact cache | `--kv q4/tcq2` (trellis V, lowest KLD of the compact caches), `--kv q8/q8`, `--kv tq3` (`--list-kv`) |
| bigger prompt buffers (faster prefill, more memory) | `-ub 2048` |
| full weight matrices for prefill (faster, more memory) | `--gemm-rows 0` |
| exactly what you asked for, or an error | `-fitl off` |
| several users at a long context (e.g. 4 x 32K) | `--np 4 --context 131072` (the context is split across `--np`) |
| keep more free memory for other programs | `-fitt 2048` (MiB left free) |

### Diagnostics

| Option / variable | What it does |
|---|---|
| `--list-kv`, `--list-flags` | KV formats; every environment flag with its status |
| `TQLLM_DEBUG=key,...` | memlog, profile, steps, stageprof, matvec, tune, prefill, draft, tree_chain, tf_dump:FILE, fakeq_k:N, fakeq_v:N |
| `TQLLM_MATVEC_RULE=BITS:KERNEL[:PF],...` | override the decode matvec kernel per format (e.g. `3:legacy:0`) |

## Request sampling numbers

`top_k` accepts integers from 0 to INT_MAX; sampling currently uses at most 16 candidates.
`seed` accepts integers from 0 to 9007199254740991 (the JSON parser's exact integer range).
Negative seeds are rejected with HTTP 400; omitted, null or zero retains the default seed behavior.
Fractional, out-of-range and nonnumeric values are rejected before integer conversion.

## Target-head shortlist

The default greedy verifier ranks the full vocabulary with a 4-bit head, then re-scores its top 16 candidates with
an int8 head. The re-score is exact for those candidates; an omitted vocabulary row can still win the full int8
projection. This is an empirical speed optimisation, not a certified full-vocabulary argmax. A large shortlist
margin alone does not bound the scores of omitted rows.

For full-vocabulary verification and logit diagnostics, the C++ API provides `Model::set_full_logits(true)`.
Set it before the first forward or verify call on a fresh model: existing CUDA graphs retain the head path captured
when they were created. The reference is the runtime's converted int8 head, not the original GGUF head tensor.
Some wider single-slot verify calls already use that full int8 path; a width sweep must record the path exercised.
