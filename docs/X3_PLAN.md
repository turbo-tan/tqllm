# PAW X3 (EXL3 trellis) in tqLLM: assessment and execution plan

Branch: `exl3`. Goal: serve PAW-27B-X3.1 (`general.architecture = paw-dense`, 12,346,158,560 bytes) in `tqserve` at
tqLLM speed, with the minimum change to existing files. The PAW x3 kernels are good and already measured; we port
them, we do not redesign them.

Paths used below (set them in your shell; never write private paths into shipped files):

```sh
X3_GGUF=...      # PAW-27B-X3.1.gguf
PAW_SRC=...      # llama-paw checkout at main >= 873c8b4f5 (the k25 tree); kernels in ggml/src/ggml-cuda/paw-x3.cu
PAW_BIN=...      # its build/bin (llama-server, llama-perplexity, test-paw-x3-frac)
DRAFTER=models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf
VOCAB=models/mtp-vocab/atx_65536.txt
BASE=models/Qwen3.8-27B-TQ3_4S-v2.gguf   # the speed reference
```

## 1. What the file is (assessed 2026-10-15)

- 1652 tensors. Non-matmul tensors are ordinary qwen35 (llama.cpp layout): F32 norms (already 1 + w), `ssm_a`,
  `ssm_dt.bias`, `ssm_conv1d`, `ssm_norm`, F32 `ssm_alpha` / `ssm_beta` [5120, 48], `token_embd` Q4_K,
  `output` Q5_K. `mach1.ne_tlut` (F16 [2, 512]) belongs to the legacy rt codec and is unused by x3.
- Metadata keys are `paw-dense.*` with the same names and meanings as `qwen35.*`; extra keys `paw.format_version = 3`,
  `paw.x3_rewrite = mul1-v1`, `paw-dense.rht_block = 1024` (rt codec only, ignore for x3). No MTP layers.
- Every matmul (400) is a triple instead of `<name>.weight`:
  - `<name>.m3_trellis` I16 `[16*K, (rows/16)*(cols/16)]`: one 16x16 tile per row of 16*K uint16. 56 words =
    K3.5 (395 matrices), 48 = K3 (5). Kernels also exist for 2, 2.5, 3, 4 and 1.5.
  - `<name>.m3_suh` F16 `[cols]` (input scale), `<name>.m3_svh` F16 `[rows]` (output scale).
  - Math: `y = svh * H128( W_hat * H128( suh * x ) )`, block-diagonal 128-point Hadamard (1/sqrt(128) per side),
    W_hat decoded from the mul1 codebook (exllamav3 trellis; fractional K uses mask 0xAAAA).
  - Matrix names match qwen35: `attn_qkv`, `attn_gate`, `ssm_out`, `attn_q` (q + gate, 12288 rows), `attn_k`,
    `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`.
- GDN value-head order: the x3 payload was encoded from the parent llama.cpp GGUF, so its rows (and `ssm_out` input
  columns) are already in the tiled order the tqLLM loader expects ("tiled value heads", `src/qwen35.cpp` GGUF
  section). llama-paw's V reorder is for the legacy rt codec and is OFF for x3. Confirm with the parity gate (G3), do
  not add a reorder.

### How llama-paw runs it (`$PAW_SRC/ggml/src/ggml-cuda/paw-x3.cu`, `ggml_cuda_op_paw_x3_mm`)

| columns nt | fractional K (K3.5, 99% of this file) | integer K (K3) |
|---|---|---|
| 1..8 | `x3v_gemv_kernel` (GGML_PAW_X3_GEMV=2): suh + H128 prologue, mma decode, svh + H128 epilogue | same |
| 9..128 | `x3r_reconstruct_ws` (fused reconstruct + both Hadamards, ORIGINAL basis fp16 W) + cuBLAS | `x3g_gemm_kernel` (tensor-core trellis GEMM) |
| > 128 | reconstruct + cuBLAS (fp16 accumulate on sm_86) | plain reconstruct + activation Hadamards up to 1023, then fused |

The fractional x3g GEMM does not exist; K3.5 above 8 columns always reconstructs.

### Speed reference (RTX 3090, GPU 2)

| engine / model | AR tok/s | DFlash2 tok/s | source |
|---|---:|---:|---|
| tqLLM, Qwen3.8-27B-TQ3_4S-v2 (14.50 GB) | 51.7 (2k prompt), 50.6 (8k) | 121.9 (8k), 107.1 (32k) | `results_gpu2.jsonl`, 2026-10-15 |
| llama-paw, X3.1 (12.35 GB) | ~30 | 73.7 pinned / 129 HumanEval40 | paw27b SERVING.md |

X3.1 is 15% smaller than TQ3_4S-v2, so a bandwidth-bound decode could beat the TQ3_4S line. The llama-paw AR gap
(30 vs 52) is mostly engine overhead (ggml graph, f32 activations, separate cast / norm launches), which tqLLM does
not have. The open risk is the x3v kernel's own time at nt=1 and at the verify width; Phase 0 measures it before
anything is built.

**Targets** (same GPU, same session, `bench/relbench.py`, same prompts as the TQ3_4S line):
- AR decode >= TQ3_4S-v2 AR (>= ~51 tok/s at 2k).
- DFlash2 decode >= 0.95x TQ3_4S-v2 (>= ~115 tok/s at 8k). Acceptance differs by model; report it next to tok/s.
- Prefill >= 0.8x TQ3_4S-v2 (fused reconstruct + cuBLAS is the expected path).

## 2. Design

The engine never learns about x3. Two seams carry the whole change:

1. **A GGUF adapter** turns a `paw-dense` file into a `qwen35` view at open time: metadata keys are aliased, and each
   `m3_*` triple becomes one virtual tensor `<name>.weight` of a new tqLLM type `GGML_X3 = 303`.
2. **A weight format** `X3` (registered for 303) loads those virtual tensors and supplies `matvec`, `dequant_h`, and
   later `gemm`. Basis is `Plain`: suh/H128 live inside the kernel (each matrix has its own suh, so a shared engine
   basis is impossible and is not needed).

### Virtual tensor = self-describing records

The loader concatenates row groups of the same type (`g_cat`: `attn_qkv + attn_gate`, possibly `ffn_gate + ffn_up`)
into one byte buffer before `load()`. x3 matrices cannot be merged (separate suh / svh / K), so the bytes must be
splittable. Each virtual tensor reads as one record:

```
struct X3Record {            // src/formats/x3_record.hpp, shared by the adapter and the format
    char     magic[4];       // "X3R1"
    uint32_t rows, cols;     // output rows, input cols (multiples of 128)
    uint32_t words;          // uint16 per 16x16 tile: 16*K (56 = K3.5, 48 = K3, ...)
};                           // followed by trellis (rows/16*cols/16*words*2 B), suh (cols*2 B), svh (rows*2 B)
```

`X3::load(raw, rows, cols)` walks the records, checks that rows sum to `rows` and that cols match, uploads each and
returns a `Mat` whose `aux` points at a host-owned part table (same ownership pattern as `composite.cpp`). With one
part, `matvec` is one launch. With several, one launch per part writes its row block (`y + row_offset`, row stride
`m.rows`), so the x3v epilogue needs an `ldc` parameter. A later optimisation can batch parts in one launch (Phase 5).

### Files

New (each under ~1000 lines; ported code keeps an `// exllamav3 <file>, via llama-paw paw-x3.cu:<lines>` source line):

| file | content | est. lines |
|---|---|---:|
| `src/paw_gguf.hpp/.cpp` | `void paw_adapt(GgufFile &)`: no-op unless arch is `paw-dense`; checks `paw.format_version == 3` and `paw.x3_rewrite == mul1-v1`, aliases `paw-dense.X` to `qwen35.X`, sets arch `qwen35`, builds the 303 virtual tensors, removes the `m3_*` entries and `mach1.ne_tlut`; refuses clearly a file with rt-only (`rt_trellis`) matrices | 150 |
| `src/formats/x3_record.hpp` | the record header and a parse helper | 50 |
| `src/formats/x3.cpp` | `WeightFormat` `X3`: load, matvec (n <= 8), `max_batch_cols` 8, `dequant_h`, `row_bytes` 0 (not row-sliceable) | 150 |
| `kernels/formats/x3.cuh` | host launch declarations (`tqllm_cuda::x3_matvec`, `x3_dequant_h`) | 40 |
| `kernels/formats/x3_common.cuh` | ptx helpers, mul1 codebook decode, window extraction incl. fractional, H128 inner functions | 500 |
| `kernels/formats/x3_gemv.cu` | `x3v_gemv_kernel` + `x3v_cfg` heuristic + plan cache, half input, bf16 output, `ldc` | 700 |
| `kernels/formats/x3_reconstruct.cu` | `reconstruct_had_kernel` to fp16 `[rows][cols]` in the original basis | 450 |
| `tests/x3_format.cu` | see G2 | 250 |
| `tests/paw_gguf.cpp` | see G1 (synthetic GGUF, no model file needed) | 120 |

Touch points in existing files (this is the whole diff outside new files):

| file | change |
|---|---|
| `src/gguf.hpp` | `GGML_I16 = 25`, `GGML_X3 = 303` in the enum; `GgufTensor` gains `std::vector<char> header; std::vector<GgufTensor> parts;` (virtual tensor: header then parts, `bytes` = total) |
| `src/gguf.cpp` | `gguf_block_layout`: `GGML_I16` (1 elem, 2 B); `read()`: if `parts` non-empty, return header + each part's bytes; end of `gguf_open`: `paw_adapt(f);` |
| `src/formats/registry.cpp` | `void register_x3();` and its call |
| `CMakeLists.txt` | `src/paw_gguf.cpp` in the `tqllm` sources; the two tests with `add_test` (`src/formats/*.cpp` and `kernels/formats/*.cu` are globbed already) |
| `README.md`, `NOTICE`, `licenses/`, `CHANGELOG.md` | credit exllamav3 (MIT, turboderp) for the trellis codec and ported kernels; add `licenses/exllamav3-MIT.txt`; PAW X3 in the supported-format list; "Unreleased" line |

Nothing in `src/qwen35.cpp`, `src/model.cpp` or `tools/` changes. If a phase seems to need one, stop and write down
why in `docs/X3_STATUS.md` before editing it (likely candidates: a consumer that reads tensor bytes through the mmap
reader `gguf_mapped.cpp` instead of `GgufFile::read`; the memory fit if it sizes by `gguf_block_layout` instead of
`bytes`). Check both in Phase 1 and handle them in the adapter layer if at all possible.

## 3. Execution plan with gates

Work strictly in order; do not start a phase until the previous gate passes. Record every number (GPU, files,
commit, settings) in `docs/X3_STATUS.md`. GPU 2 is shared: check `nvidia-smi` first, and stop any of your own
`tqserve` before benchmarking. Long jobs run detached (`setsid ... > log 2>&1 < /dev/null &`) and are polled.

### Phase 0: assess the kernel before porting (half a day)

1. Run `X3FRAC_BENCH_REPS=300 $PAW_BIN/test-paw-x3-frac` and record K3.5 times at nt 1, 2, 4, 8 for the real
   shapes: 5120->10240 (qkv), 5120->6144 (gate, attn_output in), 6144->5120 (ssm_out, attn_output), 5120->12288 (q),
   5120->1024 (k, v), 5120->17408 (gate, up), 17408->5120 (down). If nt 1 / 2 / 4 are not covered, add them via its
   env or a 20-line driver in the scratch area; do not change llama-paw.
2. Matmul time per token = sum over the 64 layers (48 GDN, 16 attention). Compare with the same sum for TQ3_4S from
   `build/tqkbench` (or `tests/matvec_bench.cu`) on the same shapes.
3. **Gate G0**: projected x3 matmul time per token at nt=1 <= TQ3_4S time x 1.05. If it fails, write the numbers in
   `docs/X3_STATUS.md` and stop for review (option B: a lossless repack of the trellis to a tqLLM-friendly layout at
   load time; option C: tune `x3v_cfg` for these shapes). Also record nt=16 (two nt=8 launches vs reconstruct +
   cuBLAS): this decides the verify path in Phase 5.

### Phase R: make the K3.5 matmul cost TQ3_4S-class, lossless (added after G0 FAIL, 2026-10-15)

G0 measured X3 at 1.70x TQ3_4S per token (24.77 vs 14.58 ms). Why, per weight, from the code:

- TQ3_4S: 8-level scalar codebook. One PRMT gives 4 int8 weights, one dp4a does 4 MACs: ~0.5 instr / weight.
- x3v (fp16 path): window extract (K3.5: 64-bit variable shifts + ring wrap select) + IMUL (mul1 hash) + dp4a
  (byte sum) + half pack + HFMA2 + MMA fragment work: ~6-8 instr / weight.
- Issue budget on a 3090 to stay at DRAM speed: 82 SM x 4 issue x 1.7 GHz x 32 = ~17.8 T thread-instr/s; a K3.5
  5120x17408 matrix (39 MB, 44 us at 890 GB/s) allows ~8.8 instr / weight in total, ~4-5 to leave headroom. Today's
  87 us is ~17 / weight. Prior work (paw27b 20260907 gemv_ceiling, phase0_results) measured: not DRAM-bound, no
  single dominant item, 3-bit decode 2.3x 4-bit decode because of the per-lane variable shift, and the int8 sq
  kernel beats x3v at m=1 for 3-bit by 13-37%.

The two levers that cut instructions per weight without touching values:

1. **int8 sq path** (exists, ported from exllamav3, supports fractional K): per weight = extract + IMUL + ONE dp4a
   against the splatted int8 activation; the bias goes through exact per-slice sums. This is the minimum arithmetic
   for mul1. Cost: activations are int8 per 128-slice after the Hadamard (exllamav3 default for m <= 2) -> needs a
   quality gate, not a bit-identity check.
2. **Aligned K3.5 re-layout** (new, lossless): permute the trellis bits at load so each lane's windows start at a
   constant bit offset (e.g. 4 lanes per tile x 32 pairs = 7 words each, history bits from the neighbour lane by
   shfl at a compile-time shift). Same bits, same values, same bytes; extraction becomes K4-style constant-shift
   BFE.

Measure before building (scratch harness under /tmp/opencode, llama-paw untouched, warm runs, spread reported):

- R1: sq vs x3v for K3.5 and K3, nt 1 / 2 / 4 / 8, the 7 real shapes. Per-token projection for "sq at nt <= 2, x3v
  above" and for "sq everywhere it wins".
- R2: upper bound of lever 2: copy of the sq (and x3v) kernel with the fractional extraction replaced by K4-style
  constant-shift extraction on the same bytes (values wrong, timing only). Also K4 kernels on K3.5-sized byte
  counts.
- R3: floors for the best variant: no-decode (extract + IMUL replaced by a register move) and no-load, to see what
  is left after both levers.
- R4: quality of lever 1 alone: llama-paw X3.1 with the sq path forced at nt = 1 vs the x3v path: KLD on wikitext
  32 x 2048 against the x3v logits, plus the paired GSM8K / HumanEval+ the paw27b reports use for numerics changes.

**Gate GR**: projected per-token matmul with the best measured combination <= 1.15x TQ3_4S (<= 16.8 ms), and R4
quality null. If met, Phase 2 ports that combination (sq + aligned layout) instead of x3v-only. If not, report the
numbers and stop.

### Phase 1: GGUF adapter (no CUDA)

1. Add the enum ids, `GgufTensor::header/parts`, `read()` concatenation, I16 sizing.
2. Write `paw_adapt` + `x3_record.hpp`. Virtual tensor `ne = {cols, rows}`, `type = GGML_X3`, `offset` unused,
   `bytes = sizeof(X3Record) + trellis + suh + svh`.
3. Grep every reader of `GgufTensor::offset` and every `gguf_block_layout(t.type...)` caller; make sure each one
   either never sees type 303 or handles `parts` (`tests/gguf_validation.cpp`, `fit_context`, `gguf_mapped`,
   `tools/tqmoe_info.cpp`).
4. **Gate G1**: `tests/paw_gguf.cpp` writes a tiny synthetic paw-dense GGUF (2 matrices, one K3.5 and one K3) and
   checks the arch, key aliases, virtual shapes, and that `read()` returns header + trellis + suh + svh bytes exactly.
   Plus: `config_from_gguf` on the real `$X3_GGUF` returns 64 layers, hidden 5120, ffn 17408, 24/4 heads, lin
   16/48 heads (a one-off check, not a ctest). CPU build and `ctest` pass.

### Phase 2: X3 weight format and kernels

1. Port verbatim from `paw-x3.cu`: the H128 inner functions, codebook / extraction for every K in {1, 1.5, 2, 2.5, 3, 3.5, 4} (X3 files are K3 + K4, X3.1 files are K3.5 + K3,
   K2.5 + K2 or K1.5; same container, K from the trellis shape),
   `x3v_gemv_kernel` with its MMODE 0 and 1 and CFG 0 and 1 and `x3v_cfg` (mode 2 = the llama-paw serving setting,
   hard-coded; no env flag), and `reconstruct_had_kernel`. Keep the 2-block `__launch_bounds__` fix from
   `873c8b4f5`. Adaptations only:
   - input: `__half` plain activations (use `had_hf_r_128_inner`, the half-input prologue, not the f32 one);
   - output: `__nv_bfloat16` (and `float` for `matvec_f32` if cheap) with an `ldc` row stride;
   - workspace (`a_had`, ~cols*8 halves) is preallocated in `load()` per matrix or once per format sized to the
     largest cols; no allocation inside `matvec` (CUDA graphs capture decode);
   - reconstruct writes `[rows][cols]` row-major (exllamav3 emits `[cols][rows]`; transpose in the store, not after).
2. `src/formats/x3.cpp`: `name "X3"`, `Basis::Plain`, `max_matvec_cols 8`, `max_batch_cols 8`, `dequant_h` through
   reconstruct, no `gemm` yet.
3. **Gate G2** (`tests/x3_format.cu`, random trellis words + random suh/svh, every K, shapes 128x256, 5120x1024,
   6144x5120, 5120x17408, multi-part record):
   - `dequant_h` equals a host reference decode (port the mul1 decode + H128 on host, about 60 lines) to fp16
     rounding;
   - `matvec` n = 1..8 equals `dequant_h` then a host / cuBLAS GEMV within 2e-3 relative RMS;
   - width independence: column j of an n-column matvec is bit-identical to a 1-column matvec of that column;
   - multi-part record: equal to the parts run separately;
   - one-off (not ctest): a real `$X3_GGUF` matrix output bit-matches llama-paw x3v for the same fp16 input (dump with
     `GGML_PAW_X3_DUMP`); differences allowed only from the f32 vs bf16 output store.

### Phase 3: end-to-end bring-up and parity

1. `build/tqrun --model $X3_GGUF` short greedy prompt; then `tqserve` with `--kv` default, context 8192.
2. **Gate G3** (quality / exactness, records in `docs/X3_STATUS.md`):
   - greedy 256 tokens on 5 fixed prompts (2 code, 1 chat, 1 math, 1 long-context ~8k): tqLLM vs llama-paw (server line
     from paw27b SERVING.md, `-ctk f16`). Expect identical text for most prompts and late divergence elsewhere; an
     early divergence (< 20 tokens) means a layout bug (check GDN V order first, then q/gate split);
   - KLD of tqLLM X3.1 logits vs llama-paw X3.1 logits (`tools/tqkld` against `llama-perplexity
     --kl-divergence-base`) on wikitext 32 x 2048: expect mean KLD < 1e-3;
   - `test-dflash-exact`-style check: greedy output with the drafter equals greedy without it (tqLLM invariant).

### Phase 4: speed measurement (no tuning yet)

`bench/relbench.py` on GPU 2, same session, BASE and X3.1, prompts 2k / 8k / 32k, nospec and dflash2 (DRAFTER,
VOCAB), and `TQLLM_DEBUG=decode` for per-step timing. Use nsys on one decode step and one verify step to split the
time into x3 matvec, attention / GDN, head, and launch gaps. Also the real workload, xhigh thinking at depth: one fixed
reasoning prompt, `--reasoning on`, >= 8k generated tokens, 32K and 64K context, DFlash2; BASE and X3.1 in tqLLM plus
llama-paw X3.1 (measured 40-50 tok/s there); report tok/s and acceptance. **Gate G4**: the targets in section 1, and
X3.1 thinking tok/s >= 0.95x BASE thinking tok/s. If met, go to
Phase 6.

### Phase 5: tuning, only where Phase 4 says the time goes (measure each change alone)

In expected order of value:
1. **Verify width.** DFlash2 verifies `dfl.block` columns (16): with `max_batch_cols 8` the engine reads every x3
   matrix twice. Options, decided by the Phase 0 nt=16 numbers: (a) extend x3v to M <= 16 (a third MMODE; watch
   registers, check occupancy with `--resource-usage`); (b) `gemm()` = reconstruct + cuBLAS above 8 columns.
2. **Multi-part fusion.** qkv + gate and ffn gate + up as one launch (a part table in constant memory, blocks
   indexed over parts); they share the input, so the suh prologue is the only per-part difference.
3. **Prefill.** Pick the reconstruct cutoff vs the engine's `NGEMM_BLAS` threshold; cache the input Hadamard per
   (input, suh) only if nsys shows it matters.
4. `x3v_cfg` rows for the 27B shapes, if Phase 0 showed a config miss; a rule row goes in
   `kernels/formats/matvec_rules.hpp` with its measurement.

Each change: G2 tests still pass, greedy outputs of G3 unchanged (or the difference explained), relbench before /
after in the status file.

### Phase 6: finish

- `ctest` (GPU and CPU builds), `tests/test_no_private_paths.py`, `python3 tools/gen_flags.py` if any flag was added
  (prefer none).
- README: PAW X3 in the supported list, a speed row with conditions; Credits: exllamav3 (trellis codec, kernels ported
  via llama-paw); NOTICE + licence file; CHANGELOG "Unreleased".
- Commits on `exl3`, one per phase, messages in the repo's style (`area: what, measured why`).

## 4. Rules for the executor

- Port, do not rewrite: the kernels are measured. Every ported function keeps a source comment with the upstream file
  and line range. Only the adaptations listed in Phase 2 are allowed without a measurement.
- The engine (`src/qwen35.cpp`), `src/model.cpp` and `tools/` stay untouched unless the status file records why.
- No new environment flags for A/B; settle the choice by measurement and keep the winner.
- One change per measurement; warm repetitions only; state GPU, files, commit and settings for every number.
- If a gate fails, write what failed and the numbers in `docs/X3_STATUS.md` and stop for review instead of working
  around it.

## Phase D: X3.1-distilled DFlash2 drafter, full run (added 2026-10-15 after D-POC)

D-POC: fc-only fine-tune on 300 seqs raised live tok/round 3.11 -> 3.88 (+25%), tg 66.6 -> 82.5 on the
40 hard held-out prompts (`docs/X3_STATUS.md` Phase D-POC). Phase D scales it with staged data, not a fixed
time budget. Hard cap 48 h of GPU 2. Rules: GPU 2 only (`flock /tmp/bonsai_gpu2.lock`); detached chained
jobs (`setsid bash -c "<cmd>; tmux send-keys ... [job] <name> done rc=$rc"`), no sleep/poll anywhere;
search reports/X3_STATUS before any measurement; report one line per gate to tmux claude:0.0.

**Order:** finish D-POC (f16 stock control + identity vs non-spec greedy) -> P-EXL (lossless logit check,
then AR/DFlash2 speed on GPU 2) -> D1 -> D2 -> D3 -> D4.

**Data (per stage, appended, never regenerated):**
- Prompt-disjoint split: the 40 P1 prompts stay the live eval set; a new fixed 100-seq offline held-out set
  (prompts never used for training) replaces the 20-seq one. Train prompts never overlap either.
- Mix: 40% xhigh reasoning, 40% agentic code (8k-32k context, real tool-call transcripts), 20% chat/prose.
  70% greedy, 30% temp 0.6 (serving is not always greedy). Generate with the best drafter so far (faster).
- Capture with `llama-x3-tapdump`, convert to `.pt` bf16, delete raw caps per batch. Before every batch,
  require >= 100 GB free on /mnt/hdd, else stop and report BLOCKED.

**Stages (each = collect -> train from best ckpt -> live eval):**

| stage | cumulative seqs | trainable | typical wall |
|---|---|---|---|
| D1 | 1,000 | fc | ~6 h |
| D2 | 3,000 | fc, then fc+last2 if fc plateaued | ~12 h |
| D3 | 10,000 | fc+last2 (all layers only if last2 plateaued) | ~24 h, only if D2 gate passes |

**Plateau rules (decide automatically, no waiting for the lead):**
- Within a training run: offline eval (accept/block on the 100 held-out) every 200 steps. Early-stop when
  the best value has not improved by >= 1% relative over the last 3 evals; keep the best checkpoint only.
- Overfit guard: also eval on 100 training seqs; if train-heldout gap grows > 15% while held-out is flat,
  stop and lower lr or freeze back to fc.
- Layer escalation: only after fc early-stops on the current data.
- Data-scaling gate (live, 40 prompts, f16 drafter, n_max 5): go to the next stage only if live
  tok/round improved >= 5% relative over the previous stage. Otherwise data has plateaued: go to D4.
- Live eval cost is ~5 min; run it at every stage end, never mid-run.

**D4 finalize:** live eval of the best drafter at n_max 5 and 7 (llama-paw), report tok/round, accept,
tg per kind (reason/code/prose), at drafter precisions f16, Q8_0, Q4_K_M (shipping pick = smallest within 2% of f16 tok/round), plus one 262144-ctx load+decode check (q4_0 KV, n_max 5 and 7) reporting resident MiB; identity vs non-spec greedy at the same rate as the stock drafter; export
GGUF (and EXL3 if P-EXL shows exllamav3 is the faster runtime); write results + training curve
(steps vs offline accept/block, stage vs live tok/round) to `docs/X3_STATUS.md` Phase D.

**Gates reported:** GD-POC, GP-EXL, GD1, GD2, GD3, GD4. Stop and wait only on FAIL/BLOCKED or the 48 h cap.

**2026-10-15 decision (after GD0, bounds the work above):** Phase D **closed**. Ship the **stock z-lab DFlash2 drafter at Q4_K_M** (1.08 GB; live tok/round 3.815 vs f16 3.821, i.e. within 0.2%; Q2_K 3.113). **No training** — D1/D2/D3 are **cancelled**: the fc fine-tune moved live tok/round only 3.821->3.881 (+1.6%) and the trainer's offline accept/block is miscalibrated (stock 1.522 offline vs 3.821 live), so it is not a trustworthy training signal; that gap is **parked, not chased**. Remaining work: D4-lite with the stock Q4_K_M drafter (n_max 5 vs 7, plus one 262144-ctx q4_0-KV resident-MiB check), the 4 wide-divergence numeric-vs-bug probe, the bounded P-EXL diagnosis, then Phases 1-6 (tqLLM port).

## Phase 2+ engineering rules (added 2026-10-16, applies to Phases 2-6; overrides the file table above where they differ)

Goal: X3.1 runs, and the next codec / next PAW model is a new file plus a table row, not an edit.

### Three seams

1. **Codec seam (device).** Only one header knows how trellis words become weights.
   - `kernels/formats/x3/codec_mul1.cuh`: `template<int K> struct Mul1Codec { static constexpr int words = 16*K/...; __device__ static void decode(const uint16_t* tile, int lane, half2 (&w)[N]); }` covering K in {1,1.5,2,2.5,3,3.5,4}.
   - Kernel skeletons are templated on `Codec`: `gemv_sq<Codec>` (n=1, the Phase R int8 sq kernel), `gemv_x3v<Codec>` (n=2..8), `reconstruct<Codec>`. They never name mul1 or K directly.
   - A future codec (better trellis, TurboQuant-style scalar, ...) = one new `codec_<name>.cuh` + one row in the codec table. No skeleton edits.
2. **Codec table (host).** `src/formats/x3_codecs.cpp`: `struct X3Codec { uint16_t id; const char* name; int words_per_tile; MatvecFn matvec; DequantFn dequant; }` with one row per (codec, K) instantiation. `X3Record` gains `uint16_t codec, flags` (magic -> `X3R2`; adapter sets `codec = MUL1`). `X3::load` looks the codec up by id and fails clearly on unknown ids.
3. **Model seam (adapter).** `src/paw_gguf.cpp` becomes table-driven: `struct PawArch { const char* paw_arch; const char* tq_arch; std::span<const KeyAlias> aliases; TensorMapFn map; }`. paw-dense -> qwen35 is the first row; paw-moe (paw-125b, `qwen4exp`) later is a second row with its own map function, no other edits.

### Layout

```
kernels/formats/x3/
  x3.cuh            host launch decls (only header the .cpp side includes)
  ptx.cuh           ptx/ldmatrix helpers
  had128.cuh        H128 in/out (half input prologue)
  codec_mul1.cuh    mul1 codebook + window extraction, every K
  gemv_sq.cu        n=1 skeleton
  gemv_x3v.cu       n=2..8 skeleton + x3v_cfg
  reconstruct.cu    dequant to fp16 [rows][cols]
  dispatch.cu       the ONLY place choosing kernel by n/shape (Phase 5 tuning edits only this)
src/formats/x3.cpp, x3_codecs.cpp, x3_record.hpp
tests/x3_format.cu  parameterised over the codec table: a new codec row gets G2 tests for free
```

### Code rules

- New files <= 600 lines, functions <= 80 lines; no copy-paste between skeletons (shared pieces go in the .cuh helpers).
- Existing files: only the touch-point table above (+ CMake globs). Existing kernel files are not edited. If one must change, stop and write why in X3_STATUS first.
- No env flags for kernel choice; no allocation in matvec (graph capture); every ported block keeps its source line comment.
- Each gate report includes `wc -l` of new files and `git diff --stat` of existing files.

### Gate G2 additions

- Seam proof: a test-only `codec_testscalar.cuh` (8-bit scalar codebook, ~40 lines) runs through the same skeletons and passes the same matvec-vs-dequant tests. This proves a new codec needs no skeleton edits.
- sq (n=1) and x3v (n=2..8) both bit-match llama-paw on one real X3.1 matrix; dispatch picks sq at n=1.

## Phase T: TRUSS-design trellis GEMV for X3.1 decode (added 2026-10-16, lead took over the critical path)

Why: GR measured the llama-paw/exllamav3 kernel family (sq, x3v) at ~87% non-decode structure (R3: on 5120x17408 K3.5,
zero traffic + zero decode still leaves 55.5 us vs a 44 us read). TRUSS (`~/trellis-kernel`, same mul1 codec, incl.
`Mul1Frac` K=KA+0.5) built a different kernel family whose decode items run at the codec ceiling: decode+mma from
registers 2.1-2.7 Tw/s (TRACKER #17), K3 at 84-91% of DRAM, `moe_window` v7a persistent dataflow kernel (#26). That
lever was never tried for the dense 27B. Projection: K3.5 dense matrices are read-bound at ~890 GB/s -> per-token
matmul ~13-15 ms if a kernel reaches ~85% of max(read, decode), vs 24.8 ms today and 14.58 ms TQ3_4S.
Same weights, same values: lossless.

- **T0 (measure, no new code):** in `~/trellis-kernel` (build/ is current) run `tk-bench-ceiling` / `tk-bench-decode`
  on GPU 2 for Mul1<3> and Mul1Frac<3> (K3.5): stream GB/s and decode Tw/s. Survey `tk-bench-gemv-split`,
  `gemv_multi_test`, `moe_window` for a kernel that can run one dense matrix (E=1, rows 5120..17408). Project the
  per-token matmul over the 400 X3.1 matrices as sum(max(bytes/stream, weights/decode)) / eff for eff 0.75/0.85.
  **Gate GT0:** projection at eff 0.85 <= 16.8 ms (1.15x TQ3_4S). Report the table; stop on FAIL.
- **T1 (build):** `kernels/formats/x3/gemv_window.cu` (<= 600 lines): dense GEMV for n = 1..8 in ONE kernel (same
  per-column math for every n -> drafter exactness by construction), modelled on `moe_window` v7a (persistent grid,
  items = 16-row x K-slab, decode in registers via the codec, mma, Hadamard in/out as TRUSS does). Half input, bf16
  output with ldc straight from the kernel (removes the x3.cpp cast/copy launches). Codec comes from the seam
  (port TRUSS `Mul1Frac` register path into `codec_mul1.cuh` if ours differs). Credit TRUSS source lines in comments.
  **Gate GT1:** tests/x3_format.cu rows pass (vs reconstruct < 2e-3, column bit-identity n=1..8); per-matrix time on
  the 7 real shapes <= 1/0.85 of max(read, decode) ceiling; per-token sum <= 16.8 ms.
- **T2:** select it in `dispatch.cu` only; rerun test-dflash-exact, KLD (< 1e-3 vs llama-paw), G4 table.
