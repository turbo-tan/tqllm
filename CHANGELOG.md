# Changelog

All notable changes to tqLLM. Images: `ytan2000/tqllm:<version>` and `ghcr.io/turbo-tan/tqllm:<version>` (multi-arch:
x86-64 RTX 30 / 40 / 50 and arm64 GB10 / DGX Spark). `latest` is the newest release.

## 1.5.0 (2026-11-10)

### Added
- Several boxes behind one address: `tqserve --np N --peer http://HOST:PORT` (repeatable) makes this server the front door
  for other tqserve boxes running the same model. Each chat request goes to the box expected to answer first: free slots,
  the users already decoding there, and the part of the prompt that box's K/V does not already hold, so a conversation's
  next turn goes back to the box that has it. Requests are forwarded byte for byte (streaming included); a box that does
  not answer is skipped. `POST /admin/route` reports a box's state. Two DGX Sparks (GB10), Flash-Next IQ3_S, 384-token
  answers, total tok/s (two rounds): 4 users 156.7 / 129.4 (one box 111.5 / 93.0), 8 users 208.0 / 182.9 (134.8 / 130.9),
  16 users 220.6 / 214.2 (one box 125.3 / 134.3). Texts are the same as one box.

- `--mtp` drafts for sampled requests (temperature > 0) as well, with exact speculative sampling (the MTP drafts are
  argmax tokens, accepted with the model's probability, else resampled). RTX 3090, Qwen3.8-27B IQ4_XS GGUF, 32K context,
  temperature 0.7: 104-117 tok/s against 50 when sampled requests decoded without drafting.
- llama.cpp IQ4_XS GGUFs of Qwen3.8 load: their IQ4_XS token embeddings are read in place by new `iq4xs_row` /
  `embed_multi_iq4xs` lookups (0.67 GB, bitwise equal to the host IQ4_XS dequantiser in BF16). Any other embedding type
  without a lookup kernel is dequantized to BF16 once at load instead of failing with "unsupported matrix type".
- Native Windows build (MSVC + CUDA, no WSL or Docker): `tqserve` serves Qwen3.8 GGUFs. One file per OS for file mapping,
  positioned reads and shared libraries (`src/core/platform/os_*.cpp`) and one socket layer (`tools/serve/net.hpp`). RTX
  5060 Ti 16 GB, Windows 11, CUDA 12.8, Qwen3.8-27B TQ3_XS: prompt 1,186-1,228 tok/s at 2K / 6K, decode 33.8 tok/s, with
  DFlash2-TQ3 160 / 116 / 102 tok/s (chat / 2K / 6K). The MoE engines (Flash-Next, GLM-5.3-Flash, MiniMax) are left out on
  Windows for now and say so (`TQLLM_MOE_ENGINES`, OFF on Windows); run them under WSL 2. CUDA 13.3 builds too (#35).
- DSpine drafter (opt-in, `-md` with a DSpine GGUF; `tools/convert/dflash/dspine_hf_to_gguf.py` converts Tencent's
  checkpoint) for Qwen3.8-27B and Qwen3-8B targets. Exact (greedy text equal to plain decoding, 20/20). DFlash2 stays the
  default: on the same targets it is faster on an RTX 3090 (162 vs 133 tok/s, code + math 227 vs 209), a GB10 (61 vs 50)
  and an RTX 5060 Ti (130 vs 85).

### Changed
- Flash-Next with more than 32 rows in a verify block (np 16 x 4 rows): dense and head projections read their weights
  once per 32 rows instead of once per 8. One GB10, 16 users: 116.3 / 123.8 -> 125.3 / 134.3 tok/s; texts unchanged.
- Qwen3.5 4B and smaller (tied embeddings, no `output.weight`) load again (they failed with "missing GGUF tensor:
  output.weight"). RTX 3090, Qwen3.5-4B Q4_K_M: 106-185 tok/s.
- macOS (Apple Silicon): long prompts run part of every FFN on the Neural Engine beside the GPU (an int8 program for the
  last 12,800 of the 17,408 FFN channels, planned from costs measured once per Mac and cached). M5 Pro, Qwen3.8-27B
  TQ_L-QSA: prompts 685-704 / 659-671 / 598-605 tok/s at 8K / 16K / 32K (1.4.4: 477-484 / 456-460 / 429-430). Outputs stay
  exact (server == plain decoding); the split changes numbers slightly (KL vs the GPU-only prompt 0.006 at 16K), so texts
  can differ from 1.4.4. Without a Neural Engine prompts run on the GPU alone.
- macOS: shorter first token for prompts of 17-1,000 tokens (a 64-row prompt tile up to 1,024 rows, one 32-row matrix op
  for 17-32-row prompts): mlx-community 4-bit 114-123 ms at 19-29 tokens (1.4.4: 120-132), 165-172 ms at 35-60
  (193-219), ~400 ms at 129-161 (522-532).
- macOS: builds on the stock MLX 0.32.4 wheel (pip / Homebrew) with the Neural Engine path; the opt-in Core ML path and its
  variables (TQLLM_MLX_ANE_FFN, TQLLM_MLX_ANE_WAVE) are removed.

### Fixed
- GB10 (memory shared with the CPU): a model loaded while another model process held most of the memory exhausted it and
  hung the machine until the watchdog panicked. tqserve now refuses that load and says which memory it found.
- Several users on a small card (np > 1, 16 / 12 GB): a request whose K/V could not fit beside the running ones failed
  the running requests' next step with HTTP 500. It now waits in the queue, the admission counts a new sequence's own
  state (284 MB at 8K), and a request that does fail frees its slot instead of turning the next request into a 500.
  RTX 3090 limited to 16 GB, Flash-Next, 4 users: 12 rounds with no failure. tqserve logs the cause of every HTTP 500.
- Router: a box that answered its status late (busy prefilling a burst of requests) was skipped, and its share queued on
  the full front door; it now keeps its last answer and is skipped only after three misses in a row.
- Streaming: a piece could end inside a multi-byte character (the stop-string hold is a byte count); pieces now end on a
  character boundary, and invalid UTF-8 in a reply becomes U+FFFD instead of invalid JSON (#35).
- macOS 27: the Metal kernels and the MLX CMake package build again (#24); the wrong-output report on macOS 27 is still
  under investigation.
- With a Jinja chat template, `--reasoning-effort` never reached the template, so every request that set no effort ran
  at the template's default (xhigh). The resolved effort (the request's, else the server's) is now passed to the template.

### Known issues
- Flash-Next on a 12 GB card with several users (`--np` > 1) under constant load can stop after several minutes with a
  GPU fault (Xid 31 / 32 in the kernel log). One user on a 12 GB card and four users on a 16 GB card ran clean. Use
  `--np 1` on 12 GB cards for now; a fix is planned for 1.6.0.

## 1.4.4 (2026-11-07)

1.4.3 was not published; 1.4.4 carries everything below.

### Added
- Flash-Next: Unsloth Dynamic v3.1 GGUFs (UD-IQ1_M, UD-IQ3_XXS) load and serve, F32 projections included. RTX 3090, one
  request, UD-IQ3_XXS: 144.5 / 93.5 tok/s (two rounds), prompts 2,409 / 2,547 / 2,586 tok/s at 8K / 16K / 32K.
- Flash-Next: NVIDIA's NVFP4 experts (`nvidia/Qwen3.8-Flash-Next-NVFP4`) load as NVFP4 (no requantising) beside tqLLM's
  dense layers in one manifest (`tools/fn_nvfp4_gguf.py` writes the GGUF). One DGX Spark: 121.8 / 59.3 tok/s, the same as
  tqLLM's own file.
- Qwen3.8-27B: EXL3 checkpoint folders (`Qwen3_5ForConditionalGeneration`) load directly.
- `tqkld` scores Flash-Next manifests (`.tqfn`) against a llama.cpp `--kl-divergence-base` file.
- Server: a request that runs out of device memory during its prompt while others decode waits in the queue and runs
  when one finishes, instead of failing.
- macOS (Apple Silicon): `brew install turbo-tan/tap/tqllm` or `curl -fsSL https://tqllm.com/install.sh | bash` build tqserve against the stock
  MLX 0.32.4 wheel; the `tqllm` command: `tqllm serve` downloads the model and drafter on first run, the first of
  TQ_L-QSA, the mlx-community 4-bit file (40 GB+ Macs; `--qsa` its QSA file), TQ_L that is already downloaded or published
  on Hugging Face; `--model owner/repo` or `owner/repo/file.gguf` downloads that model.
- `tqllm claude` / `tqllm codex` / `tqllm opencode` start the coding agent pointed at the local server, and start the
  server first when none answers on the port (it keeps running; `tqllm stop` ends it).
- Metal: QSA indexers load on TQ4 files (TQ_L) too.

### Changed
- Flash-Next, `--exact off` with several requests: the CPU expert pool reads its pinned expert copy. RTX 3090, 8 requests:
  78.5 / 79.0 tok/s total (was 68.2 / 69.9).
- Flash-Next: a built-in per-layer expert slot profile when the manifest has none (UD files, RTX 3090 one request:
  130.6 -> 144.5 tok/s).
- Flash-Next: F32 router and projection weights decode through one GEMV kernel (UD files +3-10 %).
- Flash-Next prompts on RTX 30-series cards: BF16-mixer and UD files take the FP16 row-scale path and the F32 projections
  run as FP16 tensor-core GEMMs. RTX 3090, one request, 8K / 16K / 32K: tqLLM's IQ3_S file 2,548 / 2,764 / 2,804 tok/s
  (1.4.2: 2,351 / 2,563 / 2,551), UD-IQ3_XXS 2,592 / 2,815 / 2,829 (2,409 / 2,547 / 2,586); GSM8K-200 unchanged.
- DGX Spark / GB10: unified memory counts reclaimable page cache as free when sizing the context (27B, 8 x 131,072:
  128K prompts 251 -> 786 tok/s); the 27B batched verify holds at most 24 columns (8 requests 104.5 -> 135.1 tok/s).
- Metal, TQ4 files (TQ_L): decode / verify 2-3 % faster, same tokens: matrices split K 4 ways, the Q6 head reads FP16 d +
  int8 sub-scales instead of an FP32 scale per 16 weights (head 2.93 -> 2.45 ms per step), and the decode RMS norm writes
  the rotated FP16 copy in the same launch.

### Fixed
- Flash-Next, RTX 3090 and other 24 GB cards: with automatic expert slots and one request up to a 64K context, long
  answers (thousands of tokens) could run out of device memory (HTTP 500). Automatic slots now keep a 1 GB decode reserve
  for one request (GSM8K-200 at `--context 65536 --max-tokens 4096`: 31 failures -> 0).
- Flash-Next, UD-IQ1_M: prompts no longer crash the server (the MTP layer's experts overran the prefill staging).
- Flash-Next, UD files: F32 projection weights no longer break CUDA graph capture.
- tqserve: a second Ctrl-C exits at once; an exception in a request thread returns a logged HTTP 500 instead of an empty
  reply.
- Metal: the exact attention kernel was switched off on stock MLX builds (it checked for the patched MLX instead of the
  matrix unit).

## 1.4.2 (2026-11-05)

### Fixed
- Flash-Next, `slots=auto` without per-layer slots: a 128K prompt on a 24 GB card no longer runs out of memory at the
  first decode step. The device buffers that read the K/V rows moved to host memory are now made before the expert
  banks refill after the prompt, so the refill sizes the banks around them (RTX 3090, tqLLM's own Flash-Next file,
  145K-token prompt: three requests out of memory with 1.4.1, all answered with 1.4.2).
- Flash-Next on DGX Spark / GB10 with `slots=auto`: when every expert fits on the device, one request drafts up to 7
  tokens as a manifest with fixed slots does (the draft length was chosen before the slots were known). tqLLM's own
  Flash-Next file, one request: 120.7 vs 104.1 tok/s.

### Added
- tqLLM's own Flash-Next file on Hugging Face ([YTan2000/Qwen3.8-Flash-Next-tqLLM](https://huggingface.co/YTan2000/Qwen3.8-Flash-Next-tqLLM),
  `mixed/`, 82.4 GB in two parts, manifest `flashnext-tqllm.tqfn`): the ISTA-DASLab IQ3_S experts, IQ4_XS dense layers
  and 8-bit hyper-connection mixers. One manifest for RTX 30 / 40 / 50 (24 GB) and GB10.

## 1.4.1 (2026-11-05)

The first public release since 1.2.3: it carries everything below (the 1.3.0 and 1.4.0 candidates were not published).

### Added
- Flash-Next serves several requests at once in one batched verify step (`--np` 2-8), with every request's text equal
  to its `--np 1` text under `--exact on`. RTX 3090, same IQ3_S GGUF and 8 prompts, total tok/s per round against
  Strata 0.2.10: `--np 2` 113 / 93 vs 72-79, `--np 4` 89 / 86 vs 73-79, `--np 1` 154 / 92 vs 93-115, `--np 8` 68 / 72
  vs 70-74. Expert slots are sized per request count from the measured state (`slots=auto`).
- Prompt cache across request slots (Flash-Next, `--prompt-cache on` with `--np` > 1): each slot keeps its conversation,
  a request goes to the slot that already holds most of its prompt, and only the new part is prefilled. Eight concurrent
  agent sessions on a GB10: 11.9 vs 4.5 tok/s total, first token 7 s vs 30 s. A reused prompt is not byte-identical to an
  uncached one, so with `--exact on` the server prints a warning.
- `repeat_penalty` (LM Studio / llama.cpp; `repetition_penalty` accepted) and `frequency_penalty` per request, and
  `--repeat-penalty` / `--frequency-penalty` server defaults, next to `presence_penalty` (27B engine; sampling only).
- Flash-Next: faster decode at long context (multi-row BF16 matrix-vector kernel, indexer keys in a ring with a host
  mirror, expert banks grown into free memory after a long prompt): 128K exact 79.2 vs 73.5 tok/s on an RTX 3090, chats
  +3.5-8 %, same output.
- Quantised mmproj tensors (Q5_0 / Q5_1) load for the vision encoder.
- MiniMax: `--draft-vocab FILE` runs the DSpark drafter's head over a token subset (exact: the target verifies every token).
- `--exact on|off`, on by default: every step at the reference precision (FP16 / BF16 activations and KV, no int8 / FP8
  GEMMs, no CPU-served experts), so greedy output is the same from build to build. `--exact off` turns on each engine's
  measured approximations; an approximation asked for explicitly with `--exact on` runs with one `WARN` line and a
  `/health` note. Per-engine table in [options](docs/reference/options.md#exact-and-speed---exact-onoff).
- One server front for every backend: CUDA and Apple Silicon (Metal) builds of `tqserve` share the same model front and
  pick the engine from a registry (`src/backends/<platform>/`); a new architecture registers there
  ([Adding a model](docs/guides/adding-a-model.md)).
- Weight-format plugins (experimental, CUDA): `--plugin PATH` loads a shared library that adds a GGUF tensor type or a
  faster kernel for a built-in format. Each is checked against the format contract at load, and a patch that claims to
  be exact is compared bit for bit with the built-in kernel; `/health` lists what was loaded ([Plugins](docs/guides/plugins.md)).
- Format contract test: every GGUF weight format's GPU dequantisation against a CPU reference and its matrix-vector
  product against a double-precision dot (14 of 19 formats pass; IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ1_S are known gaps).
- Metal (Apple Silicon): Qwen3.8-27B decode with the DFlash2 drafter on the same mlx-community 4-bit weights as
  Splash 1.4.0 is faster on 13 of 18 measured prompts on an M5 Pro 48 GB (2 ties, 3 losses). Short prompts run 2-20 % faster. Contexts
  from 32K to 128K run 2-15 % faster. Changes: the drafter's native 8-row blocks on prompts of 8,000 tokens or more,
  draft-ahead and a per-request width policy, int8 KV from 32K keys, and far-context retrieval for the drafter.
  Verify attention reads the int8 KV in place. Greedy speculative output equals plain decoding.
- Metal: 256K-token prompts on a 48 GB M5 Pro decode at 27.6 tok/s (Splash 1.4.0 24.5; TensorFold 1.2.1 cannot hold
  the context). The KV cache is freed when a request ends (idle server 17.9 GB, was 26.3 GB after a 256K request), and
  under `--exact on` an int8 KV cache forced by memory uses the int8-query attention (bit-identical rows, so greedy
  speculative output still equals plain decoding; 23.2 -> 27.6 tok/s at 256K). Build guide: [Mac](docs/guides/mac.md),
  with the MLX 0.32.3 patch in `third_party/mlx`.
- MiniMax-M3.1-Flash (experimental: its greedy output has not yet been checked against a reference implementation and
  changed in this release; treat its text and speed as preview) from the EXL3 2.15 bpw pack (Hugging Face folder, no
  conversion) on one GB10 / DGX Spark, with the DSpark drafter: prompt chunks on tensor cores (attention, indexer, routed experts in batched trellis lists), FP8 KV
  when the context needs it, and a draft length that follows the measured acceptance. Prefill 537 tok/s on an 8K prompt
  at 16K context, 492 tok/s at 32K.
- GLM-5.3-Flash (GSQ GGUF): sparse attention with the DSA indexer, contexts up to 128K on a GB10, prompt chunks of
  512 tokens (prefill ~100 tok/s), end-of-turn stop and reasoning split.
- Hugging Face checkpoint folders served in place (`--model FOLDER`): safetensors shards, tokenizer and chat template read
  as published, no conversion and no second copy on disk. Encodings: EXL3 trellis, MXFP8, BF16 / F16 / F32.
- Guide: [MiniMax-M3.1-Flash and GLM-5.3-Flash on one GB10](docs/guides/gb10-big-moe.md).
- Guide: [RTX 5090](docs/guides/rtx-5090.md): Flash-Next and Qwen3.8-27B numbers and what the engine sizes by itself.

### Changed
- **Flash-Next prompt processing on GB10 / DGX Spark**, against TensorFold 1.2.2 on the same machine, same client and
  prompts (tqLLM on its mixed Flash-Next file with 8-bit mixers, TensorFold on its MLX 4-bit folder): 8K / 16K / 32K / 64K prompts take 4.32 / 8.14 /
  16.83 / 35.64 s vs 4.90 / 9.21 / 18.58 / 38.11 s (1.07-1.13x faster). The layer-major prefill runs dense attention on
  tensor cores, the router and the hyper-connection mixers as FP16 / BF16 GEMMs (down and inject in one GEMM), writes
  the 16-bit rows each projection reads instead of converting, combines the MoE and hyper-connection updates in one pass
  and keeps the prompt's residual rows in BF16 between layers (files with 8-bit hyper-connection mixers). Decode is unchanged (1 / 2 / 4 / 8
  requests: 122 / 97 / 106 / 120 tok/s on a repeated prompt); 8 requests' text still equals each request alone; GSM8K-200
  96.0 % (paired with the previous build: 3 lost, 3 gained).
- **Mac (Metal): `--exact on` output now equals plain one-token-at-a-time decoding, token for token**, also with the token
  tree: one attention kernel serves plain decoding, chains and trees, summing each row's last 16-79 keys in plain-decoding
  order. Decode on an M5 Pro 48 GB, same mlx-community 4-bit file as TensorFold 1.2.1 / Splash 1.4.0: 8K 51.9, 16K 62.1,
  32K 56.1, 64K 48.8, 128K 38.9, 256K 27.5 tok/s (TensorFold 61.0 / 54.4 / 49.9 / 41.4 / 34.7 / does not fit; Splash
  60.8 / 62.4 / 48.4 / 43.9 / 35.9 / 24.5).
- **Mac: QSA sparse attention** for model files that carry QSA indexers (the 16 of higa2515/Qwen3.8-27B-QSA-BF16, added
  with `tools/qsa_indexer_gguf.py` + `tools/gguf_add.py`): past `--qsa-from N` keys (default 49152) a step attends the
  2,048 keys the indexer picks plus the recent ones. 128K decode 38.9 -> 49.6 tok/s. QSA changes the text; spec output
  still equals the same file's plain decoding.
- Mac: a thinking budget's second phase continues on the first phase's state under `--exact on` (only the closing tag is
  prefilled, not the prompt and thinking again).
- Flash-Next with several requests reads each dense weight once per step instead of once per 8 verify rows (the
  hyper-connection mixers, the router and every quantised dense projection), the expert down projection takes two
  16-row tiles per block, and the per-layer token embedding is read into memory again after the expert banks are built
  (the bank build had pushed it out, so each step read random rows from the disk). GB10 / DGX Spark, total tok/s per
  round: `--np 2` 75 / 59 (was 61 / 48), `--np 4` 90 / 80 (63 / 59), `--np 8` 99 / 100 (64 / 89), `--np 1` 55 / 48
  (55 / 42); same output.
- The decode reserve on a 3090-class card is 256 MB up to a 64K context (512 MB sat idle in decode): more expert slots.
- Every environment switch the engine reads is a registered `TQLLM_*` / `TQFN_*` flag (`tools/gen_flags.py --check` fails
  on any other); three unregistered switches were removed.
- **Default precision:** with `--exact on` (the new default) the Qwen3.8-27B prefill runs FP16 GEMMs on CUDA where
  1.3.0 used int8 (RTX 30 / 40 / 50) or FP8 (GB10). `--exact off` restores the 1.3.0 prefill.
- **Environment variables:** 55 remain (20 settings, `TQLLM_DEBUG`, 35 retired names that stop the tool and name their
  replacement). Diagnostics are `TQLLM_DEBUG` keys; settled A/B switches are gone, fixed at their measured defaults.
  [docs/FLAGS.md](docs/FLAGS.md) lists every removed name with what replaces it.
- Flash-Next decode launches its per-layer kernels directly on Blackwell PCIe cards (RTX 5090, `sm_120`) and keeps CUDA
  graphs on unified-memory GPUs (GB10 / DGX Spark, +5-6 % decode) and older PCIe cards (RTX 3090: -3 % chat without them).
  RTX 5090 with a server CPU: chat 128 -> 154, 64K context 118 -> 134 tok/s. Output identical.
- Flash-Next: an expert bank lent to a long prompt's prefill comes back with the experts it held before the prompt
  (64K context on an RTX 5090: +3 % decode).
- Flash-Next keeps the K/V of a long context in host memory by default on discrete GPUs with virtual memory management
  (RTX 30 / 40 / 50; the last 32K positions stay on the GPU): 256K context on an RTX 3090 decodes at 65-69 instead of
  52-56 tok/s, up to 128K unchanged. Off on unified-memory GPUs (GB10 / DGX Spark). `TQFN_KV_HOST=0` / `1` forces it.

### Fixed
- Flash-Next on GB10 / DGX Spark no longer pins host memory for long-prompt residuals at start-up (on unified memory it
  took 6.6 GB of the memory the expert banks need and left 759 MB free).
- The approximate tail skip (`TQFN_TAIL_SKIP=R`, opt-in) applies to verify blocks of any size (it stopped at 4 tokens).
- System memory is the machine's RAM capped by the container limit (cgroup `memory.max`): in a Docker / cloud container
  with less memory than the host, Flash-Next no longer pins an expert pack sized for the host (a 5090 pod reported 944 GB
  of host RAM with a 117 GB limit and kept hitting the limit while loading).
- Expert cache on a full device: a bank that does not fit evicts the most recently used other bank, not the least
  recently used one. Layers run in a fixed cycle, so the old victim was always the next layer needed and every layer
  evicted the next (3090, 128K, larger expert banks: 0.5 % hits and 5 tok/s -> 77 % and 27.7 tok/s).
- Expert cache over its own budget picks the victim the same way (most recent other bank), and `regrow` after a long
  prompt rebuilds each bank at its target in place with the experts it held instead of releasing it (the refill took the
  freed memory first and decode then built those banks by evicting others). RTX 3090, 128K, 109 GB cap: 69-71 -> 77-78
  tok/s on a repeat request, zero bank rebuilds during decode.
- Flash-Next with `TQFN_KV_HOST=1`: each attention layer keeps the host K/V rows it read recently in a 16K-row device cache
  (consecutive steps select mostly the same positions); a new request starts with an empty cache. RTX 3090, 128K, 109 GB
  cap: 76-78 -> 82 tok/s on a repeat request, output identical.
- Flash-Next verify blocks of 5-8 tokens (the default up to 5 drafts) read each BF16 weight matrix once: the BF16 GEMVs and
  the fused router launch take up to 8 tokens (were 4, so a 6-token block read every BF16 matrix twice). Bit-identical
  (tests: bf16_gemv token grouping, gemv_seg and topk2 to 8 tokens). RTX 3090 128K repeat request 82 -> 83-84 tok/s.
- Flash-Next pins the multi-chunk prefill's residual buffer once at start-up when the host easily holds it (a tenth of RAM,
  a quarter of what is available) instead of inside the first request of each new prompt length: RTX 3090 prefill
  32K / 64K / 128K 2,104 / 2,130 / 2,026 -> 2,197 / 2,251 / 2,133 tok/s (as 1.2.3), 128K time to first token 77 -> 73 s.
- Flash-Next keeps K/V older than 32K positions in host memory by default on discrete GPUs (`TQFN_KV_HOST` auto; GB10
  keeps it on the device; `TQFN_KV_HOST=0` turns it off). RTX 3090 decode at 256K 52-56 -> 65-71 tok/s, 128K first request
  69 -> 75-77; contexts up to 64K move nothing (unchanged).
- Decode memory ledger (`TQLLM_DEBUG=mem`): bank builds and drops per request, by cause.
- Tests carry a ctest label: `gpu` tests run locally (`ctest -L gpu`), `cpu` tests run in CI (CONTRIBUTING.md).
- Flash-Next with `TQFN_KV_HOST=1`: decode copies only the attention-selected K/V rows from host memory into a compact
  device buffer instead of reading every old position over PCIe, and the expert banks grow back into the VRAM the
  host KV frees. RTX 3090 at 128K with a 109 GB RAM limit: 70.7 tok/s against 68 with all K/V on the device, output identical
  (the old path read every position over PCIe: 14-19 tok/s on a 5090).
- Qwen3.5 / 3.8 TQ4 files (e.g. Qwen3.8-27B TQ_L): the Gated DeltaNet [b|a] rows and the MoE router read the rotated
  activation again. Builds after 1.2.3 fed them an un-rotated copy meant only for hub TQ3_4S files, and greedy decode
  could loop (40-question MMLU-Pro subset 23/40 -> 36/40). Released 1.2.x builds were not affected.
- MiniMax raw completions start with the model's beginning-of-document token (without it the first tokens degenerate).
- Builds with GCC 12 (Ubuntu 22.04), test fixes for x86 builds and GPU stream ordering in the test suite.

## 1.2.3 (2026-10-22)

### Fixed
- Qwen3.8 Flash-Next from the Hugging Face files now runs at full speed with no pack file (RTX 3090, cold page cache:
  chats 96-103 tok/s, 32K prompt first token 14.6 s; output identical). The gap 1.2.1 and 1.2.2 showed (79-88 tok/s,
  17.3 s) was not the expert layout. The per-layer token embedding (28.8 GB, read row by row at every token) was not
  cached, so each new token waited on the disk, and 1.2.1's page release made that worse. The engine now reads it once
  at start-up (about 6 s) and only drops the expert pages after building the cache.
- Correction to the 1.2.1 notes: the optional 50.9 GB pack file does not change speed, and is no longer needed.

## 1.2.2 (2026-10-22)

### Fixed
- Qwen3.8 Flash-Next on GB10 / DGX Spark with the Hugging Face manifest: every request failed ("mixed cache: active set
  exceeds slots"). The manifest's 3090 cache plan and its 18 GB bank cap no longer limit a device that holds every
  expert: with `slots=auto` (now the default) the engine sizes the expert cache and its cap from the device at start-up.
- A Flash-Next manifest without `chunk=` prefilled in chunks of 8 tokens (a 32K prompt took 263 s instead of 15 s). The
  prefill chunk is now always the engine's setting (`--prefill-chunk`, 16384). For about 30 minutes on 22 Oct the
  Hugging Face manifest lacked that line, which made long prompts slow on 1.2.1; it is restored.
- Releases are now gated by running the Hugging Face card's commands verbatim on an RTX 3090 and a GB10 (health, known
  answers, a 27K-token prompt).

## 1.2.1 (2026-10-22)

### Fixed
- Qwen3.8 Flash-Next no longer needs the 51 GB packed-expert file: at start-up the engine arranges the experts of the
  IQ3_S GGUF in pinned memory itself (about 50 s; server ready in about 2 min, as with the file), byte-identical to the
  file, and then drops the GGUF's page cache. The manifest may carry the per-layer cache plan as `layer_slots=`.
  Known gap: on an RTX 3090 short chats decode about 15 % slower this way (79-87 against 93-102 tok/s) and a 32K prompt
  takes 17.3 s to first token instead of 14.6 s; long-context decode is the same. For full speed, download the pack
  file (`hf download YTan2000/Qwen3.8-Flash-Next-tqLLM experts-packed.tqfnexp`, 50.9 GB, byte-identical to the
  in-memory build) or write it once with `tqfn-pack` (now in the image, about 90 s), and add `experts=` to the manifest.
- Coming next: a single, more compact Flash-Next file with the experts already arranged (no side files, full speed
  from the start).
  Setup: https://huggingface.co/YTan2000/Qwen3.8-Flash-Next-tqLLM

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
