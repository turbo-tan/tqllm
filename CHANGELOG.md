# Changelog

All notable changes to tqLLM. Images: `ytan2000/tqllm:<version>` and `ghcr.io/turbo-tan/tqllm:<version>` (multi-arch:
x86-64 RTX 30 / 40 / 50 and arm64 GB10 / DGX Spark). `latest` is the newest release.

## 1.2.0 (2026-10-22)

### Added
- Live completed-token prefill progress and throughput in the runtime monitor, with cached-prefix counts excluded from speed. Prefill and output rates remain visible together in the request card, including after completion.
- Native MTP drafting on mixture-of-experts (qwen35moe) files that carry an MTP block (`--mtp`): the block's experts
  load like the trunk's and stay in VRAM. Qwen3.6-35B-A3B UD-Q4_K_XL on an RTX 3090 (default settings: the memory
  fit moves the int8 LM head and the token embeddings to host memory, every expert stays on the GPU): 246.2-329.6 tok/s
  over five chats (177.5-182.8 without MTP), 251.3 after a 4K prompt (179.7); output identical with and without MTP.
- `--moe-host-mb auto` (default for files larger than the free GPU memory): the experts beyond it go to system RAM,
  spread over the layers when there is no routing profile; `--moe-cache-mb auto` sizes the VRAM cache of host experts.
- Built-in `tqserve` monitor on the API port (optional separate `--monitor-port`) with request telemetry, host/device readings and a right-hand
  terminal chat. Includes in-memory theme/key controls, thinking toggle, Stop/Escape and explicit unknown counters.
  The request workbench adds selectable request journeys, pinned-reference latency/rate comparisons and visible hardware details sampled every three seconds. Theme switching uses a sun/moon icon.
- `tqllm launch pi` (the pi coding agent) and `tqllm launch dsh` (DeepSeek Harness, `web` or `headless` profile): each
  gets the server as a provider through a temporary config the launcher deletes afterwards (`PI_CODING_AGENT_DIR` /
  `dsh --patch`). Both wrote and ran a file against Qwen3.8-27B on an RTX 3090. Implemented by `tqllm launch codex exec`
  running on that same model, then reviewed.
- `tqserve --temperature T --top-p P --top-k K`: sampling for requests that set none (default stays greedy). `tqllm launch`
  gives agents 0.6 / 0.95 / 20 (Qwen3.8's thinking-mode settings): Codex sends no temperature, and greedy decoding looped
  on an ASCII-art task until it hit the token limit.
- MoE: `--moe-cache-mb C` keeps recently used host-memory experts in a VRAM cache, managed on the GPU inside the decode
  graph. On a 16 GB budget, MiniCPM-V 4.7 35B-A3B Q5_K_M goes from 71.9 / 112.5 / 54.9 to 96.2 / 146.3 / 85.9 tok/s
  (RTX 3090 capped to 16 GB; llama.cpp `-ncmoe 19`: 62.6 / 62.8 / 62.8). Output identical.
- Native OpenAI Responses API (`POST /v1/responses`, streaming and not, with tool calls): Codex CLI connects straight to
  the server with standard settings (`wire_api = "responses"`, no key). Tested with Codex 0.162.1 on an RTX 3090.
- Native Anthropic Messages API (`POST /v1/messages`, plus `/v1/messages/count_tokens`): Claude Code connects straight
  to the server (`ANTHROPIC_BASE_URL`). `tqllm launch codex` / `claude` now point the agent at the server directly.

- Flash-Next adaptive draft length (default for one sequence): up to 5 MTP drafts per step, drafting stops once the
  draft's top probability is below 0.9 (`TQFN_DRAFT_TAU`, 0 = off). Greedy output identical to a fixed 3-token draft;
  RTX 3090 chats +1.5 %, decode after 32K +3 %; GB10 decode after 32K +4.4 %.
- Flash-Next CUDA graphs (default on PCIe GPUs; GB10 keeps eager launches): each decode layer's attention part is
  replayed from a graph shared by every layer of its kind. RTX 3090 warm chats 92.1 -> 94.1 tok/s, output identical.
- Flash-Next expert slot tiers (one sequence, `slots=auto`): the banks keep only the decode reserve free (144 -> 157
  slots per layer on a 3090); a long prompt borrows banks for its state, the banks refilled after it share the free
  memory evenly (never below a 64-slot floor), and they grow back at the next request. RTX 3090 warm chats 94.2 ->
  100.4 tok/s, decode after 32K 67.3 -> 71.2, after 64K 63.3 -> 69.3; text identical, `--np 2/4/8` unchanged.
- `tools/rc_gate.py`: release-candidate gate (an old and a new build on one GPU: identical greedy text per setting, a
  decode-loss limit, and `--np N` answers equal to `--np 1`).

### Changed
- The runtime monitor is available at `/` and `/monitor` on the API port by default; `--monitor-port` remains optional.
- Shortened the project README and moved detailed setup, options, formats, agent instructions, benchmarks and credits into linked guides; added a recorded monitor walkthrough.
- Monitor: aligned accessible icon controls, keyboard/IME handling, stable request focus, non-disruptive streaming scroll and retryable errors. Trusted panel extensions have isolated snapshots, a contributor guide and Chromium/Firefox/WebKit CI coverage.
- `tqllm launch`: `--max-tokens` default 8192 -> 32768 (an agent's reasoning used up 8192 before it wrote the file); the
  drafter and vocabulary are also found one folder down in a model folder (e.g. `~/models/dflash2/`); a second Ctrl+C
  while the server shuts down no longer prints a traceback. A CUDA build without `-DCMAKE_CUDA_ARCHITECTURES` targets the
  GPUs in the machine (`native`), and CMake older than 3.25 stops with a clear message.
- `tqllm launch`: a model given as a file that matches a catalog model also gets that model's DFlash2 drafter, downloaded automatically if missing (`--no-drafter` to opt out).
- CI: a `vX.Y.Z` tag publishes `X.Y.Z`, `X.Y.Z-cu13` and `latest` to ghcr.io and Docker Hub (no `beta-` tags).
- CI: `checks` workflow (CPU build, ctest, Python tests) runs on every pull request; required before merge.
- MoE decode runs the shared expert inside the routed expert launches. MiniCPM-V 4.7 35B-A3B on one RTX 3090:
  IQ3_S 207.8 -> 220.1 tok/s, Q5_K_M (2000 MiB experts in host memory) 176.5 -> 187.1 tok/s. Output identical.
- Flash-Next long-prompt prefill: QSA top-k over 32-bit score bits, a measured scratch-arena rule (16K chunks fit at
  245K), first-chunk host prep overlapped with the memory plan, fewer pool trims, and an MTP catch-up that computes only
  the MTP layer's keys / values (plus its indexer keys) for every prompt position but the last. Qwen3.8-Flash-Next
  GSQ-RCO IQ3_S on one RTX 3090, first token vs Strata 0.1.45 on the same file: 32K 14.09 vs 15.12 s, 64K 27.57 vs
  29.50 s, 128K 55.58 vs 58.32 s, 245K 121.6 vs 124.4 s; decode after the prompt 82.0 / 80.1 / 66.0 / 54.1 vs
  69.5 / 68.5 / 53.8 / 48.1 tok/s. Output identical to the previous build at every length.
- Flash-Next engine split from its bring-up tool: component units under `src/models/qwen/qwen4exp/` (`fn_runtime`,
  `fn_checkpoint`, `fn_state`, `fn_cpu_experts`, `fn_layers`, `fn_model` with `fn_layer_forward` / `fn_prefill` /
  `fn_step` / `fn_mtp`, and the server engine `fn_engine_load` / `fn_engine_prefill` / `fn_engine_decode`); the tool is
  `tools/tqfn_stage.cpp`. Output and drafts identical to before at 32K and 128K. Every `TQFN_*` variable is in
  `config/flags.yaml`, checked by `flags_registry`.
- Flash-Next manifests accept `slots=auto`: the expert banks take the memory a full context does not keep (measured
  after the banks are built). Opt-in: the slot count can change greedy output (known issue). The warm-up no longer holds
  the start-up arena and full staging while the banks are built, so more slots fit (IQ4_XS on a 3090: 120 instead of
  112, chats 56.2 -> 61.1 tok/s, 32K first token 17.2 -> 15.3 s; Strata 0.1.45 on the same file 61.2 tok/s / 15.3 s).
- Source tree: one folder per component and per model architecture (`AGENTS.md` has the map and the layering rules,
  enforced by test-layering); `tools/dev/relayout.py` moves a branch onto the new layout.

- CUDA builds use the per-thread default stream (no legacy stream): stream 0 no longer synchronizes with every other
  blocking stream, and decode work can be captured into CUDA graphs. Output identical; 27B and Flash-Next speed unchanged.
- Retired A/B switches (setting one stops the tool and names the replacement): `TQFN_PF_EVENTS`, `TQFN_ROUTER_GEMM`,
  `TQFN_OLD_Q2_DOT`, `TQFN_GDN_VPW`, `TQFN_PF_CHUNK_KB`, `TQFN_CPU_OWNED_COLD` / `_PROFILE`, `TQFN_TQ_AHEAD`.

### Fixed
- Flash-Next on small cards (12 GB VRAM + 64 GB RAM, RTX 3090 held to 12 GB): chats 44.6 / 46.2 -> 45.9 / 46.4 tok/s and
  decode after a 16K prompt 10 -> 45-50 tok/s. A verify block now fits the CPU expert pool (3 drafts there; adaptive
  length stays on larger cards), every layer gets an expert bank back after a long prompt (slot tiers on small cards),
  and the CPU-owned experts are a fixed rule kept out of the GPU banks: the same prompt gives the same text whatever the
  server ran before. 24 GB unchanged (text identical, chat 100.6 tok/s).
- Flash-Next on GB10: a long prompt could fail with "prefill plan: ... MB needed, ... MB free after releasing N
  expert banks" a few MB short (releasing a bank frees a little less than its size); the prefill now releases the next
  bank until the plan fits.
- Docker image build: the Dockerfile copied `kernels/`, removed in the source layout move, so the image no longer
  built; test `dockerfile_paths` checks every copied path exists.
- Flash-Next on an RTX 3090 (PCIe): long generations could stop with "unspecified launch failure" (GPU front-end
  Xid 11 / 32 / 69): expert misses read pinned host memory by kernel beside the compute stream. Misses now go by the copy
  engine straight into their slots; GB10 keeps kernel reads (it never faulted). Three 55-72K-token pagoda runs clean.
- Flash-Next `--np N`: answers differed from `--np 1` (F39). A multi-sequence verify block passed the slot count by
  assignments and took the expert path that sums the down projection in another order; it now takes the same path as one
  sequence whenever its distinct experts fit. `--np` 2 / 4 / 8 identical to `--np 1` (RTX 3090 and GB10).
- Flash-Next `slots=auto` with `--np 4` at 32K ran out of memory in the second request's state: the fit kept room for one
  sequence's attention / recurrent / MTP state; it keeps it for every sequence (F38).
- The answer streams when the request carries tool schemas: `StreamWriter` sent no content deltas until the end whenever
  `tools` was non-empty, so agent clients saw the thinking live and then the whole answer at once. Prose before the first
  tool-call marker now streams, and a tail that could still become a marker is held back. Found and fixed by Andrew Zhu
  (@xhinker) in his fork azllm; test-serve covers the hold-back.
- `tqserve --max-tokens` defaults to 16384 (was 512): the option caps every request, and thinking counts against it, so
  turns ended with `finish_reason: length` mid-answer. Reported by Andrew Zhu (@xhinker).
- `tqllm launch codex` failed with Codex CLI 0.16x, which no longer accepts `wire_api = "chat"`; it now uses the native
  Responses API.
- A request without `max_tokens` was cut at 64 tokens; it now gets the server's `--max-tokens` (OpenAI clients such as
  Codex leave it out).
- GB10 / DGX Spark: Flash-Next read free memory from `cudaMemGetInfo`, which on unified memory counts the page cache
  as used (6.7 GB "free" with 55.8 GB available), and refused a 262K context that fits. The context check, the prefill
  memory plan and expert-bank eviction now use `MemAvailable` on unified-memory GPUs; discrete cards are unchanged.
- Flash-Next 245K prompts on a 24 GB card could run out of memory at the expert staging allocation; the plan keeps a
  448 MB margin above 128K and the allocation trims the memory pool and retries once.
- Flash-Next on files with a quantized hyper-connection mixer (the IQ4_XS file; the Q8_0 MTP block): prefill pieces
  over 64 tokens left the mixer's fp32 rows unwritten. IQ4_XS routed tokens to a non-existent expert and faulted; the
  MTP layer drafted from stale rows (decode after 128K 66.0 -> 75.3 tok/s with the fix, output unchanged).
- Flash-Next IQ4_NL tensors whose rows are not a multiple of 128 columns load as Q8_0 (same values) instead of failing.

## 1.0.0 - 2026-10-14

First public release.

### Models
- Qwen3.8 / Qwen3.5 dense models (hybrid attention + Gated DeltaNet), e.g. Qwen3.8-27B in the TQ release files
  (`YTan2000/Qwen3.8-27B-TQ`: TQ_XS 11.85 GB, TQ_S 12.55 GB, TQ_M 13.14 GB, TQ_L 14.72 GB).
- Mixture-of-experts `qwen35moe` models, e.g. MiniCPM-V 4.7 35B-A3B (text). `--moe-host-mb N` keeps the least-routed
  experts in pinned system RAM when the file is larger than the GPU, placed from a routing profile
  (`TQLLM_MOE_PROFILE`); host experts run on a side stream, staged through VRAM.
- GGUF tensor types: TQ3, TQ4, TQ3_4S, Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q4_0, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, Q2_K, IQ2_S,
  IQ2_XS, IQ2_XXS, IQ1_S, F16, BF16 (Q4_K_M, Q5_K_M, Q6_K, Q8_0 and 2-bit Unsloth UD files load).

### Speculative decoding (exact)
- DFlash2 drafter with tree verify, chain verify (`--no-tree`) and the model's own MTP head (`--mtp`): greedy output is
  identical to plain decoding in every mode (one attention arithmetic for plain and verify blocks).
- DSpark drafters (openbmb) for MiniCPM-V 4.7, chain verify by default.
- `--draft-retrieve N`: the drafter also reads far passages of long prompts.

### Memory and context
- Automatic memory fit: picks the KV format, context and prefill buffers for the GPU and prints what it chose.
- Compressed KV caches: tq3, turbo2/3/4, trellis-coded tcq2/tcq3, q4 and more (`--list-kv`).
- TqCache: the oldest KV positions live in system RAM when the context does not fit on the GPU, with identical output.
- Up to 256K context on one RTX 3090 (measured: 140 tok/s at 256K with TQ_L, tq3 KV, DFlash2); YaRN RoPE scaling
  (`--rope-scaling yarn`) beyond the trained length.

### Speed
- Prefill GEMMs in int8 on RTX 30 (Ampere) and FP8 on RTX 40 / 50 and GB10; prefill padding rows zeroed so the first
  token is stable across server starts.
- Qwen3.8-27B on one RTX 3090: 273.1 tok/s (LocalMaxxing reasoning-v1, approved) / 199.5 (code-v1).
- MiniCPM-V 4.7 35B-A3B on one RTX 3090: IQ3_S 202-208 tok/s vs llama.cpp 136-147; Q5_K_M 168-182 vs 123-125
  (`-ncmoe 2`). DGX Spark with DSpark: 145-165 vs 106-115 on code prompts, 88 vs 72 on prose.

### Server and tools
- OpenAI-compatible HTTP server: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/admin/slots`;
  tool calls, reasoning on / off with effort and budget, Jinja chat templates from the GGUF.
- `--np N` parallel slots with batched verify.
- `tools/launch/tqllm.py`: `launch opencode | claude` starts the server (Docker or native) and the agent; model
  catalog with Hugging Face download and sha256 check.
- Docker image for x86-64 and arm64 in one tag; Linux and Windows (WSL 2).
