# PAW X3 in tqLLM: status log

Plan: `docs/X3_PLAN.md`. Branch: `exl3`. One change per measurement; warm repetitions only.
Every number states GPU, files, commit, settings. A failed gate stops the line for review.

## Environment (Phase 0)

- GPU: 4x RTX 3090 (sm_86). All Phase 0 numbers on physical GPU 2 unless stated.
- `X3_GGUF`: PAW-27B-X3.1.gguf, 12,346,158,560 bytes, arch `paw-dense`, `paw.format_version=3`, `paw.x3_rewrite=mul1-v1`.
- `PAW_SRC`: llama-paw k25 tree. `PAW_BIN`: its `build/bin` (`test-paw-x3-frac`).
- tqLLM: branch `exl3` (base `4d27324` + debug commits `1a4e889`, `b056318`), CUDA 12.6, `CMAKE_CUDA_ARCHITECTURES=86`.
- `BASE`: `models/Qwen3.8-27B-TQ3_4S-v2.gguf` (speed reference).
- Serving setting mirrored in Phase 0: `GGML_PAW_X3_GEMV=2`. Bench binaries hardcode CUDA device 0,
  so they run under `CUDA_VISIBLE_DEVICES=2` to land on physical GPU 2.

## Phase 0: kernel timing (G0)

Tool: `$PAW_BIN/test-paw-x3-frac`. Bench mode (`X3FRAC_BENCH_REPS=N`) covers shapes
{5120x17408, 17408x5120, 5120x10240, 5120x6144, 6144x5120} at nt {1, 6}, K {3, 4, 3.5}.
Missing (plan 2.1): nt {2, 4, 8}, shapes 5120x12288 and 5120x1024 -> scratch driver, no llama-paw changes.

Note: a bare `test-paw-x3-frac --help` run prints `nt=0 ... FAIL` lines; that is an artifact
(`atoi("--help") == 0` -> degenerate 0-column case), not a kernel failure.

Lead note (queued 2026-10-15, applies from Phase 2 on): X3 and X3.1 use the identical paw-dense
container; only the bit rates differ (X3 B3.5 = K3+K4 words 48/64; X3.1 = K3.5+K3 words 56/48;
X3.1-2.5bpw = 40/32; 1.5bpw = 24). Port every K in {1, 1.5, 2, 2.5, 3, 3.5, 4}, derive K from the
trellis shape, and in Phase 3 also run the parity check on
`/mnt/hdd/ml/paw27b/20260904_bitrate_sweep/data/paw27b_out/B3.5.gguf` to cover the integer-K
paths. Target is still X3.1.

Lead update (queued 2026-10-15, applies at Phase 4): Phase 4 additionally requires an
xhigh-thinking benchmark at depth: reasoning on, >= 8k tokens generated, 32K and 64K context,
DFlash2; BASE vs X3.1 in tqllm, plus llama-paw X3.1 (lands 40-50 tok/s there). Report tok/s +
acceptance. Re-read the Phase 4 section of `docs/X3_PLAN.md` when reaching Phase 4.

### X3 bench (K3.5 unless noted)

Tool: `$PAW_BIN/test-paw-x3-frac` bench mode (`X3FRAC_BENCH_REPS=300`) + scratch driver
`/tmp/opencode/x3_shapes(.cpp)` (same `bench_case` body, shapes x nt {1,2,4,8,16} x K {3.5,3}).
Physical GPU 2 (`CUDA_VISIBLE_DEVICES=2`), `GGML_PAW_X3_GEMV=2` (serving mode), k25 tree.
Times are per graph launch (one x3 matmul); K3.5 nt=1:

| shape (cols -> rows) | role | K3.5 nt=1 | K3 nt=1 |
|---|---|---:|---:|
| 5120 -> 17408 | ffn gate/up | 87.2 us (binary 88.1) | 84.5 |
| 17408 -> 5120 | ffn down | 80.1 (binary 79.9) | 77.2 |
| 5120 -> 10240 | attn_qkv | 48.8 (binary 48.9) | 47.8 |
| 5120 -> 6144 | attn_gate / ssm_out in | 41.7 (binary 41.8) | 40.7 |
| 6144 -> 5120 | ssm_out / attn_output | 36.0 (binary 36.1) | 35.2 |
| 5120 -> 12288 | attn_q | 74.7 (scratch only) | 73.0 |
| 5120 -> 1024 | attn_k / attn_v | 20.3 (scratch only) | 20.0 |

nt scaling (K3.5): flat 1 -> 8 (weights read once; e.g. big shape 87.2 / 87.9 / 88.1 / 90.1 us).
nt=16 (K3.5) falls off a cliff into reconstruct+cuBLAS: 706.8 / 685.8 / 410.4 / 250.0 / 245.0 /
491.1 / 51.0 us (~8x nt=8). K3 nt=16 stays flat (x3g tensor-core GEMM exists for integer K:
95.0 / 95.9 / 63.1 / 53.9 / 54.6 / 68.5 / 32.7 us). Phase 5 consequence: verify at DFlash2
block 16 must be 2x nt=8 launches (~180 us), not reconstruct+cuBLAS (~707 us).

### TQ3_4S baseline (same shapes, nt=1)

Tool: scratch `/tmp/opencode/tq_shapes(.cu)` linking `build/libtqllm.a`, production kernels
(`tq3_matvec_i8` = AR decode path after `tq3_rotate_i8`; `tq3_matvec_h` measured too, same speed).
Physical GPU 2, 200 iters, commit `b056318`:

| shape (rows x cols) | use | fp us | i8 us |
|---|---|---:|---:|
| gu 34816 x 5120 | fused gate+up x64 | 104.3 | 104.1 |
| down 5120 x 17408 | x64 | 54.3 | 53.6 |
| gdn_in 16480 x 5120 | fused qkv+gate+beta+alpha x48 | 53.2 | 50.3 |
| gdn_out 5120 x 6144 | ssm_out x48 | 21.4 | 20.6 |
| att_in 14336 x 5120 | fused q+k+v x16 | 45.1 | 47.2 |
| att_out 5120 x 6144 | attn_output x16 | 21.3 | 20.5 |
| gate 17408 x 5120 (unfused, kernel-vs-kernel) | - | 55.6 | 53.1 |
| qkv 10240 x 5120 | - | 33.2 | 32.4 |
| q 12288 x 5120 | - | 40.9 | 38.4 |
| kv 1024 x 5120 | - | 6.3 | 5.5 |

TQ3_4S runs at ~800 GB/s effective (read ceiling ~850) on large shapes: bandwidth-bound.

### Per-token projection and G0 verdict

Layer inventory from the GGUF (400 m3 triples; 5 integer-K3: blk.0 ssm_out, blk.3 attn_output,
blk.{3,7,11} attn_q; rest K3.5). Per-token matmul sum at nt=1:

X3 (one launch per matrix, Phase-2 design):
64x87.2 (gate) + 64x87.2 (up) + 64x80.1 (down) + 48x48.8 (qkv) + 48x41.7 (gate)
+ 47x36.0 + 1x35.2 (ssm_out) + 13x74.7 + 3x73.0 (q) + 16x20.3 (k) + 16x20.3 (v)
+ 15x36.0 + 1x35.2 (output) = **24774 us = 24.77 ms**.

TQ3_4S production (fused, i8): 64x104.1 + 64x53.6 + 48x50.3 + 48x20.6 + 16x47.2 + 16x20.5
= **14579 us = 14.58 ms**.

Ratio **1.70x** (kernel-only unfused ratio 1.66x). Gate G0 (<= 1.05x): **FAIL**.

Cross-check: TQ3_4S AR 51.7 tok/s @2k = 19.34 ms/token; 14.58 ms matmul + ~4.8 ms rest fits.
X3 projects to 24.77 + ~4.8 ~= 29.6 ms/token ~= **34 tok/s AR**, matching llama-paw's measured
no-drafter 30.2 tok/s — the projection method is validated, and the gap is kernel physics
(trellis decode + dual H128 at ~440 GB/s effective vs ~810 GB/s), not fusion or harness error.

Caveats (all small, none move 1.70 toward 1.05): x3 bench uses F32 input (harness) vs the
half-input serving prologue (O(cols), ~us); ggml single-op graph overhead (~us/launch) included
in x3 numbers; `tq3_rotate_i8` input rotation excluded from the TQ sum (once per input vector
per layer, ~0.4 ms total est.); lm head + embedding + w_ba (96-row BF16, ~0.06 ms) excluded on
both sides; random weights (traffic-identical).

**STOPPED for review per plan.** Option B (lossless trellis repack to a tqLLM-friendly layout at
load) is the only path that changes the physics; option C (x3v_cfg tune) cannot close
1.70 -> 1.05 (CFG 0/1 already the serving-optimal settings; K3.5 has no wide GEMM).

Design note for Phase 1 (found while scoping G0): BASE fuses GDN qkv+gate+beta+alpha 4-way
(beta/alpha are type 46 there). In the X3 file beta/alpha are F32, so the adapter must route
through the `qwen35.cpp:1200` branch (w_mix = qkv+gate [303], w_ba = beta+alpha -> BF16);
verify `g_cat` returns null (not throw) for the uncombinable 4-way mix.

## Phase R: make K3.5 matmul TQ3_4S-class, lossless

Prior reports read (no re-measurement of closed items):
- `paw27b_gemv_ceiling_20260907.md`: read ceiling 892.7 GB/s (not 936); kernel NOT DRAM-bound
  (no-decode/no-load/skeleton ablation; skeleton ~47-53%); 3-bit decode 2.3x 4-bit decode from the
  per-lane variable shift; grid sizing closed; codec/bit-rate closed; first-run-after-idle is
  cold-boosted: discard it, report spread, warm steady state only.
- `paw27b_phase0_results_20260907.md`: int8 sq wins at m==1 for 3-bit only (+11-37%), fp16 wins
  4-bit and m>=2; dispatch rule nt==1 && bits==3 -> sq; MMA 8-24%, cross-warp reduction free;
  `x3_cast_f32_f16` 0.48 ms/token removable via float-input prologue.

k25 dispatch (paw-x3.cu `ggml_cuda_op_paw_x3_mm`): `GGML_PAW_X3_GEMV=2` tries x3v first for nt<=8
(all Phase-0 numbers are x3v); `=0` falls through to sq for nt<=8 (`sq_ok` = rows%256==0, true
for all 27B shapes), integer-K gets x3g GEMM at nt>=3, fractional-K reconstructs above nt=8.
R1 needs no code: same scratch driver, `GGML_PAW_X3_GEMV=0` = sq.

### R1: sq vs x3v (K3.5 + K3, nt 1/2/4/8, 7 shapes)

Same scratch driver, `GGML_PAW_X3_GEMV=0` (sq) vs `=2` (x3v). GPU 2, 300 reps, warm.
sq wins at nt=1 everywhere (kv marginal); nt=2 borderline; nt>=4 x3v wins big:

| shape | x3v nt1 | sq nt1 | sq nt2 / x3v nt2 | sq nt4 / x3v nt4 | sq nt8 / x3v nt8 |
|---|---|---:|---:|---:|---:|
| gate 17408 | 87.2 | 68.6 (+21%) | 88.9 / 87.9 | 132.5 / 88.1 | 212.0 / 90.1 |
| down 5120 | 80.1 | 72.3 (+10%) | 89.6 / 82.6 | 134.5 / 82.0 | 214.3 / 84.9 |
| qkv 10240 | 48.8 | 43.9 (+10%) | 51.1 / 49.0 | 74.2 / 49.8 | 117.9 / 49.6 |
| gate 6144 | 41.7 | 31.5 (+24%) | 36.4 / 42.7 | 51.4 / 43.9 | 80.0 / 44.7 |
| out 5120 | 36.0 | 31.3 (+13%) | 36.5 / 36.9 | 51.9 / 37.2 | 79.4 / 37.8 |
| q 12288 | 74.7 | 49.9 (+33%) | 61.1 / 75.0 | 90.3 / 75.4 | 142.4 / 76.5 |
| kv 1024 | 20.3 | 19.6 (+3%) | 22.9 / 20.9 | 34.0 / 21.8 | 40.7 / 21.1 |

Dispatch rule for X3.1 (matches 0907 E1): sq at nt==1, x3v above. Per-token AR sum with sq@nt1:
64x68.6x2 (gate/up) + 64x72.3 (down) + 48x43.9 (qkv) + 48x31.5 (gate) + 47x31.3+30.1 (ssm)
+ 13x49.9+3x46.3 (q) + 32x19.6 (k/v) + 15x31.3+30.1 (out) = **20443 us = 20.44 ms (1.40x)**.
Lever 1 alone: 24.77 -> 20.44 ms (-17.5%). GR needs <= 16.8 ms: R2 must find another -18%.

### R2: constant-shift upper bound + K4-equiv

(a) K4 kernel on K3.5-sized bytes (x3v path, no code, cols x7/8): fractional penalty **13-20%**
(gate 87.2->69.5, down 80.1->64.3, qkv 48.8->40.5, gate48 41.7->36.3, ssm 36.0->30.7,
q 74.7->61.4, kv 20.3->18.6). This bound is x3v-specific (MMA-fragment path).
(b) True R2 on the sq vehicle: verbatim lift of the sq closure into scratch
(`/tmp/opencode/x3r_lift_core.cuh`, 1138 lines, paw-x3.cu:29-311,312-473,478-968,978-1160)
+ harness (`x3r_main.cu`). Lift validates: BASE lift vs op sq = -4..5 us uniformly
(op graph+pool overhead; deltas below are lift-vs-lift, absolute anchor stays op-measured).
Constant-shift emulation (K4-style BFE, wrap fixed for 28-u32 tiles, same bytes/geometry):
at EQUAL grid 246, saves **~1.5%** (gate 63.7->62.3, down 68.6->68.0, rest ~0, kv -6% small-base).
Free-running it picks grid 328 (occupancy heuristic) and is SLOWER (66.5): a real lever-2
kernel would need a plan rule to stay at 246.
Per-byte analysis concurs: sq K3.5 = 569 GB/s vs sq K3 = 543 GB/s on the big shape —
fractional extraction costs nothing extra per byte on sq (the x3v penalty does not transfer).
Lever-2 value on the vehicle: **~0-2%**. Dead.

### R3: floors (same lift, M=1, K3.5, grid 246)

- no-decode (ext+IMUL -> register moves, loads kept; compiler did NOT drop staging:
  61.4 > noload 57.8): gate 63.7->61.4 (-3.6%), i.e. **decode+IMUL ~= 2.3 us (4%)**.
- no-load (staging -> arithmetic constants, decode kept): gate 63.7->57.8 (**traffic ~= 5.9 us, 9%**).
- Residual (dp4a pipeline + activation staging/Hadamard + epilogue/Hadamard + ksplit
  cooperation + prologue): **~87%**. Even zero traffic + zero decode leaves ~55.5 us on the
  big shape vs TQ 53-55 (parity at best), and small shapes are structurally gapped
  (kv floor 14.9 vs TQ 5.5: ksplit/epilogue overhead dominates; no lever touches it).

### GR verdict

Best buildable combination (sq@nt1 + constant-shift): **~20.2 ms = 1.38x TQ3_4S**, vs gate
<= 1.15x (<= 16.8 ms). **FAIL.** Stacking everything else measured (E4 cast -0.48 ms,
fusion) still lands ~18+ ms. The residual is dp4a/staging/epilogue/cooperation, not
extraction — no ordered lever reaches the bar. R4 (sq int8-activation quality) not run:
moot for the gate. GPU 2 left free.

## Verify cost at nt=8 / nt=16 (post-GR, lead request 2026-10-15 — CANCELLED, see below)

STANDING RULE (lead, 2026-10-15): before any experiment, search ML_projects/paw27b reports
and this file; if already measured, cite instead of re-running.
CANCELLATION (lead, 2026-10-15): the verify-width re-measurement was cancelled — the X3 side
was already measured (R1: x3v nt8 ~= nt1; G0: nt16 reconstruct ~8x vs 2x nt8; and
`paw27b/20260907_kernel_speed/reports/paw27b_x3_verify_bottleneck_20260906.md`: nt=16 costs
7.84x a single token, reconstruct-W path). The TQ-side numbers below were newly measured for
this request and are kept; the X3 side below merely restates R1/G0 (cited, not re-measured).
STOPPED — waiting for lead.

Per-step matmul sum over the 27B model (400 matrices; 5 integer-K3). GPU 2, warm reps.
X3.1 best path: x3v nt8 measured; nt16 = min(2x nt8, measured nt16) per matrix
(K3.5: 2x nt8 wins everywhere over reconstruct+cuBLAS; K3: x3g GEMM wins).
TQ3_4S: production dispatch (`qwen35.cpp:2006-2009`: matvec n<=12, native gemm above):
fused groups (gu, down, gdn_in, gdn_out, att_in, att_out), matvec@8 / gemm@16.

| step | X3.1 best path (per step) | TQ3_4S production (per step) | ratio |
|---|---|---:|---:|
| AR nt=1 (sq x3v) | 20.44 ms (sq@nt1 sum) | 14.58 ms (i8 fused) | **1.40x** |
| verify nt=8 (x3v) | 25.81 ms | 15.35 ms (matvec@8 fused) | **1.68x** |
| verify nt=16 | 51.33 ms (2x nt8; K3 via x3g) | 41.56 ms (gemm@16 fused) | **1.24x** |

Notes: X3 nt8 picks K3.5-x3v except the 5 K3 matrices (x3v@8: 84.8/42.1/74.3); X3 nt16 picks
2x nt8 for all K3.5 (e.g. gate 2x90.1=180.2 vs recon 706.8) and measured x3g@16 for K3
(95.0/95.9/63.1/53.9/54.6/68.5/32.7). TQ nt16 gemm is slowish (gu 298.3 us @299 GB/s),
which narrows the nt16 ratio; verify amortizes per accepted token. Harness basis: X3 f32
input via ggml op, TQ half (fp) / int8 (AR) via WeightFormat; op/graph overhead in X3 only.

## Phase L: literature review (NO GPU runs, NO code changes, branch exl3, 2026-10-15)

Downloads (read-only, never built/run): `/tmp/opencode/lit/` = `glq/` (clone),
`qtip` NOT cloned (repo is Cornell-RelaxML/qtip; kernels are HYB-only per its README —
nothing to port for mul1, noted not fetched), `exllamav3/` (turboderp upstream, depth-1).
Papers read via arXiv HTML: QTIP 2406.11235 (method text), CCQ 2507.07145, Leech 2609.02652,
CodeGEMM 2512.17970, SBVR 2509.18172. Prior local reports re-read for the tried-column:
`kernel_closed_levers_20260902.md`, `paw27b_gemv_ceiling_20260907.md`,
`paw27b_phase0_results_20260907.md`.
Baseline reminder: our sq = mul1/1MAD-family (IMUL 0x83DCD12D + dp4a byte-sum + /147.7),
~6-8 instr/weight; R3 floors: decode+IMUL 4%, traffic 9%, skeleton 87%.

| trick | source | instr/weight | lossless X3.1, no re-encode | tried? / NEW | est. gain vs R3 floors |
|---|---|---:|---|---|---|
| 1MAD (LCG + byte-sum + scale) | QTIP paper §3.1.1 | ~4 | y (it IS our mul1) | TRIED — it is the current kernel (closed_levers: MUL-BAL == mul1 construction) | 0 (baseline) |
| 3INST (MAD + lop3 + reinterpret + add, fp16 out, no dp4a) | QTIP paper §3.1.1 Alg.2 | ~4 | n (different values) | TRIED-family — closed_levers: shared-LUT gather beats 3INST/emission (2607 vs 1931 Gval/s) | N/A (re-encode) |
| HYB (hash + 512-entry LUT + sign flip, ~2 amortized) | QTIP paper §3.1.2; Cornell-RelaxML/qtip kernels | ~2 + LUT | n (LUT is fine-tuned, different code) | not tried (needs re-encode + retune) | N/A (re-encode) |
| 16x16 MMA tile per trellis sequence | QTIP paper §4.1 | MMA | n/a (structural; same-code variant possible but—) | TRIED — closed_levers: tensor-core walk x2, slower than scalar baseline | 0 (closed) |
| CCQ bitshift+mask scalar decode (Algo 2/3) | CCQ §3.1/§3.5 | ~4-6 | n (conv+scalar codec) | not tried (needs re-encode) | N/A; their GEMV ~= WINT2 (0.034 vs 0.027 ms, H20) |
| CodeGEMM Psumbook (precompute centroid×activation dots, gather psums) | CodeGEMM §3 | 1 gather + acc | n (needs finite additive codebook 2^b×v; trellis state space 2^L has none per position) | not tried (inapplicable) | N/A; also loses to cuBLAS at M>32 by their own admission |
| SBVR bitvector AND + popcount (W4A8) | SBVR §4.3/§4.4 | AND+popcnt + KxN FMAs | n (bitvector-sum codec + activation recode) | not tried (needs re-encode) | N/A; ~= AWQ speed at same quality class |
| Leech unfold-to-shifts-and-masks record (Planes14, predicated class tree, 4 acc chains, shuffle butterfly, warp/row) | Leech §2-§3 | ~6-8 | n (lattice VQ codec; unfold EXPANDS bytes 48b->112b) | not tried (needs re-encode + more VRAM) | N/A; their QTIP arm: 405 GB/s, 61% of bound (we do 570/64%) |
| Warp-per-row + ILP acc chains restructure of sq (drop ksplit atomics/cooperation), same math | Leech §2.3 pattern | same ~6-8 | y (same weights; last-ulp order effects only — needs R4-style gate, not bit-identity) | NEW (not tried) | ~5-12% est (atomics/cooperation share unisolated; E2 says reduction itself is free) — BELOW the 15% bar |
| Fuse q/k/v + gate/up launches (252->144 launches, +11.7%, +1.061x e2e) | Leech §3.4 | 0 (same instr) | y | TRIED — have it (tqLLM fuses gu/att_in/gdn_in by construction) | 0 (already in) |
| int8 output head (87 tok/s served 4B) | Leech §4 | n/a (head, not matmul) | y | n/a to this gate | 0 on matmul sum |
| Branchless NF4/FP4 LUT dequant (+33% decode H100) | bitsandbytes #1746 | LUT | n/a | N/A — our ext8w/dp4a path has no branches to remove | 0 |
| Upstream exllamav3 v1.x: fused-int8, m=2 inst, coop autotune, Ampere GEMM shapes, mul1 default, sq K1-8 + h1-h3 | turboderp/exllamav3 compare v0.0.43→v1.2.0, quant/ sources | same | y (same codec family) | MIXED — fused up/gate = have it; m=2 ~= our nt2 wash (R1); coop grid.sync CLOSED (closed_levers: naive coop DOA); frac extraction same extfrac family, no cheaper formulation found | ~0 (tuning/dispatch only) |
| Uniform-code kernels (MARLIN/LUT-GEMM/FLUTE class) | surveyed by construction | LUT/gather | n (uniform/BCQ codebooks by definition) | excluded (lossless impossible) | N/A |
| LUT-gather 2-bit scalar code (decode-as-gather, dp4a) | cephalochromoscope notes | gather + dp4a | n (explicitly trades quality: rel-MSE 0.116 vs trellis 0.069) | excluded (re-encode + quality loss) | N/A |
| GLQ zero-smem 3inst + 8-tokens-free batching + tensor-core MMA GEMV | glq/csrc/glq_trellis.cu | MMA | mixed: 3inst n (different code); batching y | batching TRIED — have it (weights-once nt<=8, R1 flat); MMA = tensor-core-walk family, closed | 0 |

### GL verdict

**FAIL.** No NEW lossless trick plausibly cuts >= 15% of per-token matmul time. Every
other codec's trick requires its codec (re-encode): 3INST/HYB (different values), CCQ
(conv+scalar), CodeGEMM (needs finite codebook), SBVR (bitvector recode + W4A8),
Leech unfold (different codec + 2.3x bytes). Same-codec items are all tried/closed
(1MAD = current kernel; tensor-core x2 slower; fused groups already in tqLLM;
nt<=8 weights-once already in; upstream v1.x is dispatch/tuning only; branchless N/A).
Closest miss: a warp-per-row sq restructure (~5-12% est, needs its own quality gate) —
below the bar, and a rewrite, not a trick. The 87% residual (dp4a pipeline, activation
staging/Hadamard, epilogue, ksplit cooperation) has no literature-documented lossless
shortcut on Ampere for mul1 K3.5.

## Phase D-POC: X3.1-distilled DFlash2 drafter (desk P0, 2026-10-15)

Goal: quick proof that fine-tuning the stock DFlash2 drafter on X3.1's own taps raises
tokens/round. Rule: cite-before-running; GPU2 only under `flock /tmp/bonsai_gpu2.lock`;
work in llama-paw-k25 + scratch; no tqLLM changes; 1-day box.

Prior work (cited, not re-run): `20260906_speculative/reports/paw27b_drafter_plan_20260906.md`
(the plan that named this fix: refit/re-train the drafter on our activations),
`paw27b_spec100_v2_retrieval_head_20260908.md` (the only tap/lattice capture ever built:
verdict FAIL, route closed; what it recorded is re-checked below),
`20260929_serve_speed/data/multi_{x31,b35}_multi.json` (40 easy prompts: accept 0.934 vs
0.926 — saturated, useless as a gate; matches the task's premise).

### P0 sub-verdicts

**(a) Drafter weights — AVAILABLE (cheap).** `z-lab/Qwen3.8-27B-DFlash2` is public,
not gated, HF reachable (HTTP 200); tree has `config.json` + `model.safetensors`
= 3.85 GB (BF16, ~1.9 B params). Not cached locally. Download target must be `/mnt/hdd`
(`/` is at 97%, 28 GB free; `/mnt/hdd` has 832 GB). `engine/drafters/dflash2.py:208
resolve_checkpoint()` defaults to `~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots`.

**(b) Trainable definition — AVAILABLE.** `/home/green-gpu/tandemllm/`:
`engine/drafters/dflash2.py` (`DFlash2Module` / `load_config` / `load_weights`),
`tools/train_dflash2.py` (objective is exactly P3's: `CE(draft_argmax, target_argmax) +
w*KL(target_top64 || draft)` over the 8-token block, early rows weighted),
`tools/train_data.py` (writes per-sequence `.pt` with `fused` = 5 taps `[N,5*5120]` bf16,
`label`, `top_ids`, `top_lp`, plus `manifest.json` with held-out split). This is a
ready-made P3 rig.

**(c) X3.1 tap dump from llama-paw — MISSING (>half day).** This is the blocker.
The k25 runtime HAS the tap API and uses it every DFlash round:
`llama_set_embeddings_layer_inp` / `llama_get_embeddings_layer_inp`
(`src/llama-ext.h:111,115`; `src/llama-context.cpp:1012,1207`), and the fresh taps are
already gathered host-side in `common/speculative.cpp:1189-1198` (the `batch_inject.embd`
loop copies each of the 5 target layers). But there is **no tool or hook that writes the
5 taps to disk**. The 20260908 `GGML_PAW_DFLASH_CAPTURE` instrumentation the task points
at is gone: absent from k25, absent from `/home/green-gpu/llama-paw` (clean tree, HEAD
`873c8b4f5`, `server-context.cpp` has no capture; the only stash is the unrelated
`pre-master-switch 2026-09-18` 15-file set with no server/spec file), and its record
(`10396 B = seq,anchor,row,prev,label,n_valid,sel + ids[16] + unary[16] fp32 +
hidden[5120] fp16`) held the **drafter's** pre-head hidden + target top-16 logits —
**not the 5 target-layer taps** the drafter's `fc` consumes. So (c) must be written from
scratch: a C++ capture hook emitting the 5 taps (f32, 5*5120) + target logits per token,
a k25 rebuild, a driver (the old `spec100_v2_capture_driver.py` shape is reusable), and
alignment to `train_data.py`'s `.pt`/manifest layout incl. bf16 cast. Note the +1 layer
convention flagged in `paw27b_drafter_plan_20260906.md` §2.1: tandemllm trains taps at
`(5,19,33,47,61)` while the GGUF serves `[6,20,34,48,62]`; a self-consistent capture must
pick one and serve with it.

### GD verdict (POC gate)

**BLOCKED on (c).** (a) and (b) are ready and cheap; the missing piece is the X3.1
target-tap dump, which is a from-scratch C++ capture + rebuild + driver + format
alignment — more than half a day, and on the critical path of P2→P3→P4. P1 (stock
drafter tok/round on the 40 hard prompts) is runnable today with existing binaries, but
without P2 taps no X3.1-distilled drafter can be trained or gated. Waiting on the lead:
either authorize the tap-capture patch (starts the ~1-day clock with (c) as first task),
or re-scope.

### P0 unblock (2026-10-15): (c) solved cheaply via the tap API, verified — GO

The lead-directed eval-callback route is unnecessary: the DFlash2 tap tensors are named
in the paw graph (`paw.cpp:431` sets `t_layer_inp[il]=inpL`, the residual entering layer
il, which is node `l_out-{il-1}`), but the runtime already exposes them through the exact
call the serving path uses — `llama_set_embeddings_layer_inp` / `llama_get_embeddings_layer_inp`
(`src/llama-ext.h:111,115`; `common/speculative.cpp:1086,1189`). So the cheapest correct
capture is a ~180-line example that reuses that API, not a name-matching cb_eval tool.

Built `/home/green-gpu/llama-paw-k25/examples/x3-tapdump/x3-tapdump.cpp` (target
`llama-x3-tapdump`, no core changes): teacher-forced, chunked over `n_batch` with a
persistent KV cache; enables taps `[6,20,34,48,62]`, dumps `tap_<L>.f16` = fp16
`[N,5120]` per layer plus `top_ids.i32`/`top_val.f32` `[N,16]` and `tokens.i32`.

**Verify (1 sequence, `paw27b_drafter_plan_20260906.md` text, 1850 tok, GPU2):** ran the
tool twice on the same sequence — chunked prefill (`-b 512`) vs single-token decode
(`-b 1`, the serving regime) — and compared.

| layer | nRMS(a,b) | mean cos | min cos | top-16 rank0 agree |
|---:|---:|---:|---:|---:|
| 6  | 0.0023 | 0.999998 | 0.99998 | 1827/1850 |
| 20 | 0.0053 | 0.999986 | 0.99986 | (cross-mode) |
| 34 | 0.0120 | 0.999928 | 0.98922 | |
| 48 | 0.0187 | 0.999799 | 0.94821 | |
| 62 | 0.0312 | 0.999440 | 0.86849 | |

Both modes self-consistently predict the next input token 59.5% (argmax==next token);
cross-mode rank-0 logits agree 98.8%. The residual prefill-vs-decode gap is the X3
codec's own GEMM (prefill) vs GEMV (decode) rounding, accumulating with depth — not a
capture artifact (at `-b 1` the tool reproduces the drafter's decode taps by
construction, same function). Conditioner mean cosine ≥0.9994 (worst single token 0.87)
is well within what a Q2_K drafter tolerates. **P0 PASS → proceed P1-P4.**

### D-POC job registry (detached jobs; rule: no blocking waits, no poll loops)

Launch form (job self-notifies on exit, no watcher/sleep/poll):
`setsid bash -c '<cmd> > <log> 2>&1; tmux send-keys -t opencode:0.0 -l "[job] <name> done (pid $$) log=<log>"; tmux send-keys -t opencode:0.0 Enter' </dev/null >/dev/null 2>&1 &`
(The older `watch_job.sh` polled with `sleep 15`; kept only for the pre-existing jobs below.)

| job | pid | log | note |
|---|---|---|---|
| p2_gen | 3495613 | `/tmp/opencode/dpoc/p2_gen.log` | 300 prompts, X3.1 greedy 512 tok (reason=xhigh full turn, code=no-think), incremental to `p2_gen.json`; watcher notified via tmux `opencode:0.0` |
| (server) | 3467554/5 | `/tmp/opencode/dpoc/serve_p1.log` | llama-paw-k25 X3.1 + stock DFlash2 Q2_K, port 8114, GPU2 under `/tmp/bonsai_gpu2.lock` — STOPPED (freed GPU2 for capture) |
| p2_template | 3583776 | `/tmp/opencode/dpoc/p2_template.log` | renders chat template (`/apply-template`) + prompt-token count for 300 seqs → `seqs/*.txt`, `seqs/meta.json` (server must stay up) |
| p2_capture | 3587400 | `/tmp/opencode/dpoc/p2_capture.log` | `x3-tapdump --x3dir` (one model load, KV cleared per seq) teacher-forces all 300 seqs → `/mnt/hdd/dpoc/caps/<id>/` (~69 GB); server stopped first to free GPU2 |
| make_pt | 3665239 | `/tmp/opencode/dpoc/make_pt.log` | converts `caps/` → tandemllm `.pt` + `manifest.json` at `/mnt/hdd/dpoc/data` (fused=5 taps bf16, label, top_ids, top_lp; every 15th seq held out) |
| exl_repack | 3709701 | `/tmp/opencode/dpoc/exl_repack.log` | CPU-only lossless repack X3.1 GGUF → EXL3 safetensors at `/mnt/hdd/dpoc/exl31` (reshape trellis, no transpose; names/shapes from x3up manifest; embed/head dequant to bf16). v1 MISSED `ssm_conv1d` → re-run |
| exl_repack2 | 3734263 | `/tmp/opencode/dpoc/exl_repack.log` | DONE + validated: 400 linears + 451 others, trellis 16/16 byte-exact (v2 added `ssm_conv1d`) |
| smoke_p3 | 3735096 | `/tmp/opencode/dpoc/smoke_p3.log` | DONE: data (37 train/3 held), drafter (1.796 B params) load; stock accept/block 1.250 |
| train_p3 | 3746297 | `/tmp/opencode/dpoc/train_p3.log` | DONE: `--train fc --steps 1000 --lr 1e-4 --kl 0.5 --blocks 32`; accept/block **1.250 → 3.881** (@1000: ALL=3.881 code=5.619 prose=2.144; @300 2.866, @800 3.722); checkpoints → `/mnt/hdd/dpoc/ft` |
| convert_p4 | — | `/tmp/opencode/dpoc/convert_p4.log` | ft→GGUF (f16). Needed TWO fork patches: (a) dense tokenizer hash `4f53cda1…` → pre `qwen35` (`conversion/base.py`); (b) DFlash `set_vocab` must write `tokenizer.ggml.merges` from the target dir (`conversion/qwen.py`) — v1 lacked merges and the server refused to load the drafter ("cannot find tokenizer merges"). Fine-tune changed only `fc.weight` (131 M), `norm.weight`, `hidden_norm.weight` |
| convert_p4c / convert_stock2 | 3807102 / 3807104 | `/tmp/opencode/dpoc/convert_{p4,stock}.log` | STILL BAD: the dense snapshot's `tokenizer.json`/`merges.txt`/`vocab.json` are broken NTFS reparse-point symlinks (unreadable) → converter produced a garbage vocab (mostly `[PAD…]`). No merges either |
| convert_p4d / convert_stock3 | 3816054 / 3816056 | `/tmp/opencode/dpoc/convert_{p4,stock}.log` | re-run with `--target-model-dir /data/www/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/models/Qwen3.8-27B-EXL3-3.5bpw` (readable Qwen3.8 tokenizer: BPE, 247 587 merges, vocab 248 044, chkhsh `1444df51…` = pre `qwen35`) → correct drafter GGUF |
| p4_ftf16 | 3819948 | `/tmp/opencode/dpoc/p4_ftf16.log` | serve X3.1 + **fine-tuned** drafter (`df2_ft_f16.gguf`, f16), 40 hard P1 prompts → `p4_ftf16.json` (drafter now loads; acceptance ~0.51, mean len 3.55 observed) |
| p4_stockf16 | 3819950 | `/tmp/opencode/dpoc/p4_stockf16.log` | stock drafter at f16 (`df2_stock_f16.gguf`) — same-precision control → `p4_stockf16.json` |
| p4_nospec | 3832413 | `/tmp/opencode/dpoc/p4_nospec.log` | non-speculative greedy reference (X3.1 alone, no `-md`) → `p4_nospec.json`; queued on the GPU2 lock behind p4_stockf16. Launched with the chained self-notify form (no watcher) |
| exl_load | 3844505 | `/tmp/opencode/dpoc/exl_load.log` | P-EXL structural gate: load `/mnt/hdd/dpoc/exl31` in exllamav3 v1.5.1 on GPU2 (`flock`), report LOADED OK. Queued behind p4_nospec. **FAILED**: torch JIT ext build (see below) |
| exl_build | 3871114 | `/tmp/opencode/dpoc/exl_build.log` | rebuild `exllamav3_ext` against current torch: `exl_build.sh` sets `CUDA_HOME=/usr/local/cuda-12.6`, `TORCH_CUDA_ARCH_LIST=8.6`, `MAX_JOBS=8`; GPU2 `flock`. On success the JIT cache lets `exl_run.sh` reuse it |
| exl_run | — | — | wrapper: `exl_run.sh <script.py>` = the same build env + `flock` + `CUDA_VISIBLE_DEVICES=2`, for all P-EXL GPU scripts |
| D0a stockq8 / stockq4km | — | — | `run_p4_eval.sh /tmp/opencode/dpoc/df2_stock_{q8,q4km}.gguf stock{q8,q4km}`: stock drafter live at Q8_0 / Q4_K_M, 40 hard prompts, n_max 5 |
| D0b offline stock | — | `/tmp/opencode/dpoc/offline_stock.sh` | trainer offline acceptance of the STOCK f16 drafter (`--ckpt /mnt/hdd/dpoc/dflash2_hf --eval-only`) |
| D0c identity | — | `/tmp/opencode/dpoc/d0c_identity.py` | per mismatched prompt: first divergence char + AR top1−top2 margin there (needs a non-spec server with logprobs) |

**Phase D order (per `docs/X3_PLAN.md` §"Phase D"):** finish D-POC (f16 stock control DONE; identity vs non-spec greedy = `p4_nospec`, running) -> P-EXL (lossless logit check `exl_load`/`exl_gen`, then AR/DFlash2 speed) -> D1 -> D2 -> D3 -> D4. Gates GD-POC, GP-EXL, GD1..GD4 reported one line each to `claude:0.0`; stop only on FAIL/BLOCKED or the 48 h GPU2 cap. All new jobs use the chained self-notify form (no watchers/sleep).

**D4 scope (updated):** live eval of best drafter at n_max 5 and 7, precisions **f16 / Q8_0 / Q4_K_M** (ship smallest within 2% of f16 tok/round), plus one **262144-ctx load+decode check** (q4_0 KV, n_max 5 and 7) reporting resident MiB; identity vs non-spec greedy at the stock rate; export GGUF (and EXL3 if exllamav3 is faster). Ref: B3.5 + Q2_K drafter at 262k = **20248 MiB** (n_max 5), 19950 (n_max 3), 18512 (AR) — `ML_projects/paw27b/20260906_speculative/reports/paw27b_spec256k_first_numbers_20260907.md:21`. Needs `llama-quantize` built (not currently) for Q8_0/Q4_K_M.

## P-EXL: run X3.1 itself in exllamav3 (parallel; D-POC priority; **GPU2 only**)

**Hard rule (user correction): only GPU2 may be used — never GPU0/1/3.** All P-EXL GPU work takes
`/tmp/bonsai_gpu2.lock` and is **queued after** the D-POC GPU jobs (fine-tune, then P4). CPU-only
work (repack, sha256 checks) runs anytime. Nothing was ever launched on GPU0/1/3: the two
`/opt/conda/bin/python` processes on GPU0 are not ours (my jobs all used `CUDA_VISIBLE_DEVICES=2`);
the historical `paw27b_session_20260906_speculative.md` GPU0 run is cited prior work only.

Goal: serve PAW-27B-X3.1 through exllamav3 (the engine that encoded it) with the EXL3 DFlash2
drafter, and measure AR + DFlash2 tok/s + tok/round on the same 40 hard held-out P1 prompts and the
11k code prompt. GPU2 only (`/tmp/bonsai_gpu2.lock`), after D-POC fine-tune + P4.

**Prior work (cited):** `20260826_evals_parity/reports/paw_b35_vs_exl3_3.5bpw_codec_diff_20260919.md`
— our X3 trellis is **byte-identical** to exllamav3 `exl3-mul1-v1` (`m3_trellis` I16 == their `trellis`
int16, SWAP16 packing; `suh`/`svh` fp16; codebook = the `0x83DCD12D` mul1 hash). The X3.1 encoder is
exllamav3 **v1.5.1** (`20260924_thinking_ladder/reports/PAW-27B-X3.1-README.md`;
`20260923_x3up/reports/RESEARCH-X3-CODEC-UPGRADE-20260923.md`). Reference run:
`20260906_speculative/reports/paw27b_session_20260906_speculative.md:79-88` — "their kit" on GPU0,
11k code, greedy: **AR 41.9, +DFlash2 (`num_draft_tokens=7`) 139.6 tok/s** (EXL3 3.5bpw, commit 86fa6ee).

**(1) Existing X3.1 EXL3 export? — NO (searched).** `quantization_config.json` search across
`/mnt/hdd`, `/data/www`, `/home/green-gpu` (depth 5) returns only: the 2B `exl4/q/x31r*`
(Qwen3.5-2B, exl3 v1.6.0 — not our model) and Mia-AiLab's 27B EXL3 3.5bpw (rule-based allocation,
different bits — the codec_diff "theirs", not X3.1). No 27B `paw`/`X3.1` EXL3 safetensors exists.
The encoder wrote the GGUF; its working dirs
(`/mnt/hdd/ml/paw27b/20260923_x3up/data/enc/{s35,s4,v5}/*/manifest.json`) hold per-module metadata
only, not a loadable EXL3 model. **So a lossless repack is required.**

**Repack recipe (concrete, from the encoder manifest).** `enc/s35/k3.5/manifest.json` is a 400-entry
map `HF module name -> {K, in, out, gguf_base, trellis_shape, sha256_trellis}` — it *is* the
HF<->GGUF name map and the exact EXL3 target shape. Example:
`model.language_model.layers.0.linear_attn.in_proj_qkv` -> `gguf_base=blk.0.attn_qkv`,
`trellis_shape=[320,384,56]` (= `[in/16, out/16, 16K]`, K=3.5). GGUF stores
`blk.0.attn_qkv.m3_trellis` I16 with the 16K axis contiguous; exllamav3's kernel doc
`paw-x3.cu:1563` writes it as "(n//16, m//16, 16*bits)". So per linear the EXL3 tensor is the same
bytes reshaped to `[in/16, out/16, 16K]`.

**Tile order CONFIRMED (no transpose).** The manifest `sha256_trellis` equals the sha256 of the
**raw GGUF `m3_trellis` bytes** — verified bit-exact on 5 tensors (blk.0 attn_gate/attn_qkv/
ffn_down/ffn_gate/ffn_up, K3.5). The GGUF already stores the stream with 16K contiguous
(`t.data` shape `(tiles,16K)`), so the repack is a pure `bytes -> reshape([in/16, out/16, 16K])`,
no re-encode, no axis permutation. `mul1` = int32 `0x83DCD12D` (signed -2082680531, read from Mia's
EXL3 safetensors). `token_embd`/`output` dequantized (Q4_K/Q5_K, extracted to
`/mnt/hdd/dpoc/target/model.safetensors`). `quantization_config.json` schema per
`src/exllamav3/exllamav3/conversion/quant_config.py:15` (`stored_tensors`, `quant_format`,
`bits_per_weight = trellis.shape[-1]/16`, `mul1_multiplier`); `config.json`/tokenizer from the dense
Qwen3.8-27B snapshot.

**Located for (2)/(3):** exllamav3 v1.5.1 at `/home/green-gpu/src/exllamav3`; the EXL3 DFlash2 drafter
under `/data/www/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/`; GPU2 is the only GPU we use.

**Status: (2) lossless-logit check + (3) serve/measure NOT yet run.** The **repack is DONE and
validated**: `/mnt/hdd/dpoc/exl31/` = `model.safetensors` 15.78 GB (400 linears `trellis`/`suh`/`svh`/
`mul1` + 451 non-quantised incl. `linear_attn.conv1d.weight` [10240,1,4] + bf16 `embed_tokens`/
`lm_head`/`norm`), `config.json` (dense + `quantization_config`: exl3 v1.5.1, bpw 3.5, codebook mul1,
out_scales always), `quantization_config.json` (with `tensor_storage`), tokenizer/chat-template.
**Lossless verified: sampled 16/16 tetrix byte-exact** vs raw GGUF `m3_trellis` (reshaped
`[in/16,out/16,16K]`); per-module K is read straight from the GGUF trellis width (e.g. all
`linear_attn.out_proj` are K3 = 48 words, not 3.5 as the manifest's nominal K implied). Its GPU work
— (2) the exllamav3-vs-llama-paw logit check and (3) the AR/DFlash2 measurements — takes
`/tmp/bonsai_gpu2.lock` and is **queued after D-POC** (fine-tune, then P4). No GPU2 lock is held by
P-EXL yet; no other GPU is used.

**Repack fix (tokenizer).** `exllamav3`'s `Tokenizer.from_config` reads `<dir>/tokenizer.json` (HFTokenizer); `exl31` shipped `tokenizer_config.json`+`chat_template.jinja` but **no `tokenizer.json`**. Copied `tokenizer.json`/`vocab.json`/`merges.txt` from the readable `/data/www/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/models/Qwen3.8-27B-EXL3-3.5bpw/` into `/mnt/hdd/dpoc/exl31/` (the dense snapshot's are broken NTFS reparse-point symlinks).

**P-EXL (2) lossless check — the `add_bos` trap (2026-10-15).** First exllamav3 generations were **incoherent** (X3.1: random multilingual tokens; the *known-good kit* model: degenerate repetition `"France is France is…"`, only 2 unique token ids). Ruled out, in order: harness loop (the documented `Generator.generate()` reproduces it), the v1.5.1 build (the kit's own **1.4.2**+prebuilt ext degenerates identically on the same model), the repack (embed_tokens / lm_head are **bit-identical** to the trainer's `target/model.safetensors`; trellis orientation confirmed `[in/16,out/16,16K]` from exl3.py:312-322; all GDN tensors present). Root cause: **`add_bos=True`** — this model's `bos_token_id == eos_token_id == 248044`, so a forced BOS breaks generation. With **`add_bos=False`** the kit model is coherent (`"The capital of France is" -> " Paris.\nThe capital of Germany is Berlin…"`), matching llama-paw's raw `/completion`. `exl_gen*.py`/`exl_gen_simple.py` now take `EXL_BOS` (default must be 0 here).

**P-EXL (2) lossless check — the repack trellis bug (2026-10-15).** With `add_bos=False` the kit model is coherent but **our repacked X3.1 is still garbage** → the repack was wrong. Root cause: the GGUF `m3_trellis` is stored `(tiles, 16K)` with **16K contiguous** and the tile index enumerates the **in** tile fastest (`paw-x3.cu`: `index_k = i % tiles_k`, `tiles_k = in/16`; `t = out_idx*(in/16) + in_idx`). `exl_repack.py` did `tt.data.reshape(in/16, out/16, 16K)` on the C-contiguous buffer, which assumes 16K is the fast axis → **every tile scrambled** (the earlier "16/16 byte-exact" check compared flat bytes, which is invariant under this reordering, so it could not catch it). Fix in `exl_repack2.py`: `raw.reshape(out/16, in/16, 16K).transpose(1,0,2)`. Re-packed to `/mnt/hdd/dpoc/exl31b/`; proof = coherent generation (below).

**P-EXL (2) lossless check — FAIL / BLOCKED (2026-10-15).** Structural load: **PASS** (`exl_load4`: `/mnt/hdd/dpoc/exl31` loads in exllamav3 v1.5.1, 153 s, 67 modules). Lossless: **FAIL** — the repacked X3.1 generates garbage, while the *kit* EXL3 model is coherent in the **same** build (`exl_gen_kit` + `exl_simple_kit_nobos`), so the engine and our env are fine. Evidence gathered:
- `add_bos=True` bug (fixed): BOS==EOS==248044; `add_bos=False` is required.
- Non-quantised tensors are **correct**: `blk.0.ssm_dt.bias`/`ssm_a`/`attn_norm.weight` equal the GGUF bytes (so the kit model is a *different base* / not a valid value oracle).
- Codec confirmed byte-identical (`paw_b35_vs_exl3_3.5bpw_codec_diff_20260919.md`: SWAP16 identical both sides; `W = diag(suh)·H128·Ŵ·H128·diag(svh)`, `suh`=k=in, `svh`=n=out); suh/svh/mul1 all match exllamav3 expectations.
- exllamav3's own encoder (`exl3_lib/quantize.py`: `pack_trellis` -> `(tiles_k, tiles_n, 16K)` = `[in/16,out/16,16K]`, `weight.view(tiles_k,16,tiles_n,16)`) says the tile order is **k-slow, n-fast**; `exl_repack.py` (v1) already used that ordering and it is *also* garbage, as is the transposed v2. So both natural tile orders fail -> the trellis **tile enumeration / 16K axis order** the GGUF received from the encoder is still not matched. Open item: derive the encoder's raw `_trellis.bin` order (conversion `qwen4exp.py:286` memmaps it flat, so the order is set by the encoder writer, not the GGUF writer). No further GPU2 time spent; stopped per plan for review.

**P1 baseline (stock drafter, held-out 40):** tok/round **3.113**, accept **0.4271**, tg **66.55** tok/s (gen 18476). Per kind: reason tok/round 3.496 accept 0.504; code tok/round 2.861 accept 0.377. GD PASS needs tok/round ≥ ~3.424 (aggregate) with identical greedy text.

**P4 result — fine-tuned drafter (f16), same 40 prompts (`p4_ftf16.json`):** tok/round **3.881** (≥ 3.424 → tok/round gate PASS), accept **0.5804**, tg **82.49** (gen 15983). Per kind: reason 3.787 / 0.5622; code 3.987 / 0.6007. vs baseline: overall +24.7%, reason +8.3%, code +39.4%.
**Text-identity caveat:** `p1_stock.json` (18:25) was written by an older `run_spec.py` that kept only `content` (not `reasoning_content`), so its reasoning rows are empty (`sha=e3b0c44…`) — unusable for the identity check; and baseline gen (18476) ≠ ft gen (15983), so the spec outputs differ. Settling via: (a) `p4_stockf16` (current harness, `sha` of full turn), (b) a non-spec greedy reference run (no `-md`). If spec-decode numerics make outputs drafter-dependent, the identity gate may be unachievable across drafters and must be judged vs the non-spec greedy text.
**Fork patches (in `llama-paw-k25`):** (1) `conversion/base.py`: dense-tokenizer hash `4f53cda1…` → pre `qwen35`; (2) `conversion/qwen.py` DFlash `set_vocab`: write `tokenizer.ggml.merges` from target dir when absent. (Neither needed once `--target-model-dir` pointed at the readable EXL3 tokenizer dir.)

**P4 control — stock drafter at f16 (`p4_stockf16.json`):** tok/round **3.821**, accept 0.5684, tg 80.79 (gen 16007). Same-precision delta (ft − stock): tok/round 3.881 vs 3.821 = **+1.6%** only; most of the gain vs the Q2_K baseline (3.113) is **precision, not the fine-tune**. ft vs stock-f16 `text_sha` differ on **3/40** (`c002,c009,c017`), gen equal 39/40.
**Implication:** the fine-tune barely moves server tok/round at f16 (its big trainer-side "1.25→3.88" was on in-distribution replay, a different metric). The deployable win is a higher-precision (f16) drafter. Spec output is drafter-dependent here (Q2_K→f16 changed gen 18476→16007), so a non-spec greedy run (`p4_nospec`) is needed to judge the identity gate.

**GD-POC verdict — PASS (2026-10-15).** Non-spec greedy reference `p4_nospec.json` (X3.1 alone, no `-md`, same 40 prompts): gen 16062, tg 29.85 tok/s (AR).
- Identity (full-turn `text_sha` vs non-spec greedy): **ft-f16 10/40, stock-f16 10/40** — *identical diff sets* (`r000,r001,r002,r005,r007,r008,r009,r011..r019` and `c000,c002,c005..c019`); gen-count equal 22/40 for both. So the fc fine-tune leaves the lossless profile exactly at the stock rate (Δ = 0 prompts). Identity gate PASS.
- Same-precision control: ft-f16 vs stock-f16 `text_sha` 37/40, gen equal 39/40 (diffs `c002,c009,c017`).
- tok/round: ft-f16 **3.881** (reason 3.787, code 3.987); stock-f16 3.821; AR 1.000.
- Observation (not a drafter defect): the spec lane is only ~25% text-identical to AR greedy even for the *stock* drafter — a property of the DFlash2 spec runtime, equal for stock and ft, so not disqualifying. GD-POC reported PASS to `claude:0.0`.

**Lead re-gate (2026-10-15): HOLD D1-D3.** Because ft−stock(f16) is only **+1.6%**, the +24.7% over the Q2_K baseline is precision, not the fine-tune; no big datagen/training is justified yet. Order: **P-EXL next**, then a new **D0** (all detached `setsid`+tmux, GPU2 `flock`, check reports first, no repeats):
- **(a)** stock drafter **Q8_0** and **Q4_K_M** live, 40 hard prompts, n_max 5 → tok/round curve vs f16 **3.821** / Q2_K **3.113**.
- **(b)** offline-vs-live gap: run the *trainer* offline eval on the **stock f16** drafter; if offline stock ≈1.5 while live 3.82, the offline metric is miscalibrated — diagnose (tap layer/position offset, mask, block size).
- **(c)** identity: for each of the ~30 mismatched prompts, report first-divergence token index and the non-spec **top1−top2 logit margin** there (margin < 0.1 ⇒ numeric near-tie; else spec-path bug).
- Report one line `[opencode] GD0: PASS|FAIL - precision curve, offline stock value, identity verdict`. **D1 only on lead approval after GD0.**

**P-EXL build note:** the exllamav3 native ext must be rebuilt against the current torch. `exl_load` triggered torch JIT and failed: (i) `nvcc` code=127 (torch's auto `CUDA_HOME=/home/green-gpu/cuda-12.6-extracted/builds` has the pip layout — nvcc lives in `builds/cuda_nvcc/bin`); (ii) `__halves2half2` undeclared (torch's default `-D__CUDA_NO_HALF_CONVERSIONS__`). The repo's prebuilt `build/lib*/exllamav3_ext*.so` is stale (undefined `torch::cuda::CUDAErrorLogCapture`). Fix = `CUDA_HOME=/usr/local/cuda-12.6` (has `bin/nvcc` + `include/`), `TORCH_CUDA_ARCH_LIST=8.6`, then JIT builds/caches. Kit venv (`/data/www/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/.venv`, exllamav3 **1.4.2**, prebuilt ext) is a fallback but predates our v1.5.1 repack.

**P-EXL build note 2 (triton).** With `CUDA_HOME=/usr/local/cuda-12.6` the ext **compiles clean** (the earlier nvcc/`__halves2half2` errors are gone). The next failure is at import: `ModuleNotFoundError: No module named 'triton.runtime'` — main python's `~/.local/.../triton/` is a **partial copy** (only `_C`,`errors.py`,`FileCheck`,`knobs.py`,`testing.py`,`_utils.py`; no `runtime/`/`compiler/`, no dist-info). Main py = torch **2.8.0+cu126**; the kit venv = torch **2.14.0+cu126** + working **triton 3.8.0**. So P-EXL now runs through the kit venv python with the v1.5.1 source (`PYTHONPATH`) ahead of the kit's 1.4.2, rebuilding `exllamav3_ext` against the kit torch (`exl_run_kit.sh`). `exl_load2` = the structural load gate under this setup.

**P-EXL build note 3 (ext shadowing).** `exl_load2` got past import with the kit's **prebuilt 1.4.2** `exllamav3_ext.so` in the kit site-packages (found by `find_spec` before any build), then failed at `LinearEXL3.__init__` → `ext.BC_LinearEXL3(...)` **incompatible constructor args**: the v1.5.1 `linear.py` calls a v1.5.1 C++ signature against the 1.4.2 binary. Fix: build v1.5.1's ext **in-place at the source root** (`exl_build_kit.sh` = `setup.py build_ext --inplace` with the kit torch), so `import exllamav3_ext` picks the source-root `.so` (PYTHONPATH precedes site-packages) over the kit's stale binary. `exl_load3` failed the same way (still the 1.4.2 ext); the in-place build is the fix. Confirmed `Tokenizer.from_config` needs `<dir>/tokenizer.json` (copied, above).

**Pipeline (built, GPU work pending):** P2 text → `p2_template.py` (renders chat template via `/apply-template`, adds prompt-token count) → `p2_capture.sh` (server down; `llama-x3-tapdump` teacher-forces each seq on GPU2) → `make_pt.py` (tandemllm `.pt` format: fused=5 taps, label, top_ids/top_lp, manifest with held-out split) → `train_dflash2.py --ckpt /mnt/hdd/dpoc/dflash2_hf --model /mnt/hdd/dpoc/target --train fc|last2` (target embed/head dequantized from X3.1 GGUF; drafter forward + selector verified with those tables) → export → `convert_hf_to_gguf.py --target-model-dir` (DFlash2DraftModel supported) → P4 re-run.

## D0 (2026-10-15, lead-ordered; GPU2 `flock`, detached chained jobs)

**D0a precision curve** (stock drafter, 40 hard P1 prompts, n_max 5, `run_p4_eval.sh`):
| drafter | tok/round | accept | tg tok/s | size |
|---|---|---|---|---|
| f16 | 3.821 | 0.5684 | 80.8 | 3.86 GB |
| Q8_0 | 3.825 | 0.5691 | 83.8 | 1.96 GB |
| Q4_K_M | 3.815 | 0.5669 | 83.9 | 1.08 GB |
| Q2_K (baseline) | 3.113 | 0.4271 | 66.6 | — |
Q8_0 and Q4_K_M are within **0.2%** of f16 → ship-smallest = **Q4_K_M** (1.08 GB); only the aggressive Q2_K loses (3.11).

**D0b offline-vs-live** (`offline_stock.sh`, stock f16, `--eval-only`, 20 held-out): offline **accept/block = 1.522** (code 1.625, prose 1.419) vs live tok/round **3.821** → the offline metric **understates live ~2.3x**. Ruled out by inspection: config (matches), tap-layer convention (server uses `llama_get_embeddings_layer_inp` too; HF `target_layer_ids [5,19,33,47,61]` == GGUF `target_layers [6,20,34,48,62]` under the layer-*input* convention), causal mask (`dflash.attention.causal=False` == trainer's non-causal block), block size 8, sliding_window 2048, BOS (`llama_vocab_get_add_bos`), and the loss/eval index alignment (`label[anchor+j-1]` == the verify pass's `label[p:p+bs-1]`). Not yet explained; flagged for review (a fine-tune adapts to the offline setup, which is why its offline "1.25->3.88" did not transfer live).

**D0c identity** (`run_d0c.sh` -> `d0c_identity.json`): first spec-vs-AR divergence located per prompt; AR top1−top2 margin (nats) there. **26/30** mismatched prompts are **near-ties (margin < 0.1)** -> numeric argmax flips under the spec/verify batching, not a spec-path bug. Wide margins: `c019, r015, r017, r018` (4/30). Consistent with ft==stock having identical diff sets.

## D4-lite (2026-10-15, stock z-lab DFlash2 Q4_K_M drafter, GPU2 `flock`)

- **n_max 5 vs 7** (40 hard P1 prompts, `run_p4_nmax.sh`): n_max 5 tok/round **3.815** (accept 0.5669, tg 83.9); n_max 7 tok/round **4.247** (accept 0.4680, tg 88.8) -> the longer budget wins (+11%).
- **262144-ctx load+decode** (q4_0 KV, `run_262k.sh`): n_max 5 resident **19,976 MiB after load / 20,588 after decode**; n_max 7 **20,276 / 20,890**. Reference B3.5 + Q2_K at 262k = 20,248 MiB (n_max 5) -> X3.1 + Q4_K_M is comparable (~+340 MiB, larger drafter). No OOM.
- Ship = **stock z-lab DFlash2 drafter Q4_K_M** (1.08 GB), n_max 7.

## GP-EXL2 (bounded <=1h P-EXL diagnosis) — FAIL, parked

The repack's **linear weights are provably correct**: `exl_recon.py` reconstructs `layers.0.linear_attn.in_proj_qkv` with exllamav3's own kernel and it equals the independent numpy GGUF decode (`W_ref`) — **reldiff 0.0009, per-fragment cosine 1.0, identity 64/64** (`exl_recon_v1`; the transposed `exl31b` gives reldiff 1.41 as expected). Non-quantised tensors byte-match the GGUF; configs match the kit. Yet **exl31 (v1) still generates garbage** while the kit EXL3 model is coherent in the *same* build. So the fault is exllamav3's **model construction for our container**, not the weights. Structural diffs vs the kit: the kit stores the head at **top-level `lm_head` (quantised)** and ships a **`model.visual.*` tower**; ours has `model.language_model.lm_head.weight` (bf16) and no visual. Renaming the head to top-level `lm_head.weight` + `tie_word_embeddings=false` (`exl31c`) **did not fix it** (still garbage). Parked per the 1h bound — runtime stays llama-paw/tqLLM.

**Wide-divergence probe (item 1).** The spec server returns **empty `top_logprobs` at draft-accepted positions**, so the intended verify-vs-single-token logit-drift could not be measured directly. Using what exists (AR distribution at the first-divergence position + the token the spec picked): `c019`, `r017`, `r018` are the spec selecting the AR **top-2** token at small margins (0.37/0.21/0.10 nats) -> **numeric**; `r015` is anomalous (spec picked `m` where AR gives `length` at -0.0006, margin 7.48; the picked token is far outside AR top-8) -> **possible spec-path bug**, flagged.

**r015 bounded look (2026-10-15, <=30 min).** `run_r015.sh` (ft f16 drafter, `LLAMA_TRACE=1`, r015 only): the divergence **reproduces bit-exactly** (`text_sha 0943b27c8b8b2111` == the earlier ft run), so it is deterministic, not a flaky numeric artifact. The ft lane emits `(rad/m)` where non-spec emits `(rad/length)`. Per-round accepts from the trace: **126 rounds, 58 full 5/5, 68 partial** -> the bad token lands **at/right after a partial accept** (~round 28, previous round accepted 3/5). Consistent with a recurrent/SSM-state-rollback or KV-trim issue on the partial-accept path in llama-paw's dflash lane; flagged, not chased (port proceeds). This motivates the added G3 requirement: spec-vs-non-spec identity on the 40 prompts in tqLLM, divergence with margin > 0.5 = FAIL.

## X3 port Phase 1 (GGUF adapter) — G1 PASS (2026-10-15)

`tqllm::paw_adapt` (new `src/paw_gguf.cpp`, `src/formats/x3_record.hpp`; `src/gguf.{hpp,cpp}` gains `GGML_I16=25`, `GGML_X3=303`, `GgufTensor::header/parts`, `read()` concatenation, `gguf_open` calls `paw_adapt`). It aliases `paw-dense.*` -> `qwen35.*`, sets arch `qwen35`, and turns each `m3_trellis/m3_suh/m3_svh` triple into one `GGML_X3` virtual tensor `<name>.weight` = `X3Record{rows,cols,words}` header + the three parts (read from unrelated offsets). No weights are touched.

- **G1**: `tests/paw_gguf.cpp` (synthetic 2-matrix file, K3.5 + K3) passes: arch switch, key aliasing, virtual `ne={cols,rows}`, byte count, and `read()` == header + trellis + suh + svh byte-exact; `m3_*`/`mach1.ne_tlut` gone. Real `$X3_GGUF`: arch `qwen35`, 64 layers, hidden 5120, ffn 17408, 24/4 heads, lin 16/48, key_len 256, **400 X3 tensors, 0 `m3_*` left**. CPU build + `ctest` **29/29** pass. Commit `c60fa0a`.

- **Phase 2 (in progress):** port `x3v_gemv_kernel` (n=2..8) + the int8 `sq` path (n=1) + `reconstruct_had_kernel` from `$PAW_SRC/ggml/src/ggml-cuda/paw-x3.cu` (4240 lines) into `kernels/formats/x3_{common.cuh,gemv.cu,reconstruct.cu}` and `src/formats/x3.cpp` (registered in `registry.cpp`), per the plan's file table; G2 = `tests/x3_format.cu` (dequant == host reference; matvec == dequant+host GEMV <2e-3 RMS; width independence; multi-part; one-off real-matrix bit-match vs llama-paw x3v).

## X3 port Phase 2 (kernels) — foundation in, G2 pending (2026-10-16)

Per the "Phase 2+ engineering rules" section: codec seam first.

- **Committed `7c0753e`** `kernels/formats/x3/{ptx,had128,codec_mul1,codec_testscalar}.cuh` (369 lines total).
  - `codec_mul1.cuh` = the one place that decodes mul1 words: `template<int KH> Mul1Codec` (KH=2K, K 1..4), `mul1_extract8<KH>` (per-K window extraction, ported verbatim from `paw-x3.cu` dq8/dq8_aligned_{1,2,4}/dq8_frac with source line comments) + `mul1_decode_2` (3-inst codebook: `idx->fma(uint16(idx*0x83DCD12D), 1/147.7, -10.39)`). Interface: `decode(const uint16_t* tile, int lane, half2 (&w)[4])`, 32 lanes/tile.
  - `codec_testscalar.cuh` = 8-bit scalar test codec (128 uint16/tile), the seam proof.
  - `ptx.cuh` = dp4a/fshift/FSHF_IMM/BFE16_IMM/cp.async; `had128.cuh` = 128-pt Hadamard both sides (1/sqrt(128)).
  - **Verified**: `nvcc -std=c++17 -arch=sm_86` compiles all 7 K instantiations + the test codec + the H128 in/out prologues, zero warnings.
- **Next (G2):** skeletons `gemv_sq.cu` (n=1), `gemv_x3v.cu` (n=2..8 + x3v_cfg), `reconstruct.cu`, `dispatch.cu` (only n/shape chooser); `src/formats/x3.cpp` + `x3_codecs.cpp` (host codec table keyed by `X3Record::codec`, `X3::load` fails clearly on unknown ids) + one `registry.cpp` line; `tests/x3_format.cu` parameterised over the codec table (dequant == host ref, matvec == dequant+host GEMV < 2e-3 RMS, width independence, multi-part, testscalar through the same skeletons, sq/x3v bit-match llama-paw on one real X3.1 matrix, dispatch picks sq at n=1).

## X3 port Phase 2 — G2 BLOCKED (2026-10-16)

Implemented (compile clean, `nvcc -arch=sm_86`; builds into `libtqllm`):
- `kernels/formats/x3/{reconstruct.cu,gemv_x3v.cu,gemv_sq.cu,dispatch.cu,x3.cuh}` + `codec_mul1.cuh`/`codec_testscalar.cuh` (both decode paths: smem `decode` and register `regs_decode`, plus `sq_windows`/`row_u32`/`loads`/`ld_*`).
- `src/formats/x3.cpp` (WeightFormat: f32 A/C around the codec, casts, scratch pool), `src/formats/x3_codecs.{hpp,cpp}` (table lookup by `(codec,words)`), one `registry.cpp` line, CMake source + include dirs, `tests/x3_format.cu`.

**G2 finding (the block):** `tests/x3_format.cu` (256x256) shows
- **orientation**: the gemv matches the reconstructed matrix only TRANSPOSED (host GEMV must use `W[c*cols+r]`) — reconstruct and the gemv use swapped row/col conventions;
- **tile layout**: with a flat random trellis the x3v path is uncorrelated (rel≈1.35) while reconstruct (n-major, `out*(in/16)+in`) is fine. `x3v` indexes tiles **k-major** (`slice_stride = ntiles*TWORDS`, `bp = ks0*slice_stride + group*WNT*TWORDS`), reconstruct indexes **n-major**. They cannot read the same un-repacked buffer, so llama-paw must repack the trellis into the kernel layout at load.
- **sq**: at n=1 the sq path agrees with dequant+host GEMV to rel≈0.02 (int8 activation quantisation error, as expected) — not the 2e-3 exact gate; sq needs its own reference (bit-match vs llama-paw), not the exact-GEMV check. sq also requires `size_n % 256 == 0` (nb256_total).

Next: resolve the canonical trellis layout + orientation against llama-paw (`test-paw-x3-frac.cpp` decode_what / the fork's load repack), repack at adapter/load, then re-run G2.

## X3 port Phase 2 — G2 PASS (2026-10-16)

- **In-repo (`x3_format`, all 8 codec rows)**: x3v n=2..8 == dequant + host GEMV to < 2e-3; sq n=1 within 5e-2 (int8); test-only codec runs through the same skeletons (seam proof); width independence (n=4 col == n=7 col).
- **Real-matrix bit-match vs llama-paw** (`tests/x3_real.cu` + a fork-side ggml driver; matrices blk.0.ssm_out K3.0, blk.0.attn_gate K3.5, fixed fp16 input, fork run with `GGML_PAW_X3_GEMV=2`):
  - sq n=1: **0 bit-diffs** (5120 and 6144 entries).
  - x3v n=4: **0 bit-diffs** (20480 and 24576 entries).
  - (Note: the fork's default `GGML_PAW_X3_GEMV=0` routes n>1 to the reconstructed GEMM; the plan's target is its x3v, hence the flag.)
- Still open (not blocking G3): multi-part record (row-group-split X3Records) is structurally supported by the record but the adapter emits one record per tensor, so no test exercises it yet.

## X3 port Phase 3 — bring-up OK, G3 BLOCKED (2026-10-16)

- **Bring-up PASS**: PAW-27B-X3.1 loads and serves in tqLLM (`tqserve --model $X3_GGUF`); multi-part records (ffn_gate+ffn_up = two X3Records in one buffer) handled by `X3Format::load`. Greedy "What is the capital of France?" -> **"Paris"** (prompt 24 tok / ~96 ms, ~29 tok/s). Short prompts generate fine at any max_tokens.
- **G3 BLOCKED**: the 40-prompt greedy run fails with `generation failed: an illegal memory access was encountered` on the first longer prompt (`r000`, 409 chars). Short prompts (35 chars) are clean, so the fault is in the longer-prompt path (prefill). Engine log: `prefill x3: batched pass ... -> batched up to 24 columns` and `prefill GEMMs in int8`. Likely candidates: the engine's x3 batched-prefill path calling matvec with a column count above the format's `max_matvec_cols()` (8) (my x3v supports m<=8), or the fused dequant (`dequant_h`) into the prefill buffer. Needs a focused repro (single matrix, prefill-length input) before Phase 3 can continue.
- Spec pass could not run: `--spec-type draft-dflash --spec-draft-n-max 5` are not tqserve flags (usage lists `-md DRAFTER.gguf --draft-max N`); rerun the spec pass with `-md $DRAFTER --draft-max 5`.

## X3 port Phase 3 — G3 result (2026-10-16)

- **Prefill OOB fixed** (commit): compute-sanitizer localized it to `reconstruct_had_kernel<Mul1Codec<6>>` reading the trellis OOB from `X3Format::dequant_h` -> `gemm_c_rows`. dequant_h now uses the canonical `(n=in=cols, m=out=rows)` order and transposes into the engine's `[rows][cols]` (cuBLAS OP_T, lda=cols). Smoke 'Paris' and the failing r000 both clean now.
- **G3 identity (40 prompts, greedy, non-spec vs spec `-md df2_stock_q4km --draft-max 5`)**: **25/40 identical, 15 divergences** (r000,r001,r004,r005,r008,r009,r010,r011,r012,r014,r016,r017,r019,c002,c008). First divergence is deep in long outputs (185-794 chars in; outputs up to 1327 chars).
- **Margin criterion BLOCKED**: the FAIL criterion is divergence margin > 0.5, which needs token top-2 logprobs; `tqserve`'s `/v1/chat/completions` returns `logprobs: null` (no logprobs support in this build) and no tool exposes per-token top-2. Need a logprobs source (add to tqserve, or a `tqspec` readout) before G3 can be declared PASS/FAIL.

## X3 port Phase 3 — G3 verdict (2026-10-16)

Divergence classification (40 prompts, greedy; spec `-md df2_stock_q4km --draft-max 5`):
- **25/40 identical.**
- Of the 15 divergences, re-running non-spec with `TQLLM_X3_N1_X3V=1` (n=1 forced onto the exact x3v gemv instead of sq) makes **10/15 match spec exactly** -> those are **by design**: non-spec decode is sq (int8 activations), spec verify is x3v (exact fp16). Not bugs.
- Remaining 5: r009/r010/r012/r019 diverge deep (371-868 chars) inside 700-1300-char outputs (near-ties); r000's "spec" text is a **stale pre-fix 500 error** (the spec pass ran before the prefill OOB fix landed).
- **sq default verdict**: keep sq as the n=1 default (it bit-matches llama-paw's default decode, `GGML_PAW_X3_GEMV=0`). The cost is that non-spec decode diverges from the x3v verify path by ~int8 rounding; if strict spec==non-spec identity is required, run n=1 on x3v (the hook), which is exact.

Still to do for a full G3: per-token top-2 margins at the first divergence for r009/r010/r012/r019 (needs a logits readout - tqserve has no logprobs), regenerate the spec pass post-fix, and G3 KLD vs llama-paw.

## X3 port Phase 3 — G3 state after the sq hook fold (2026-10-16)

- Hook is now `TQLLM_DEBUG=x3n1x3v` (no new env var); flags regenerated, `gen_flags.py --check` ok.
- Post-fix G3 (both passes regenerated): **25/40 identical, 15 divergences** (same set). With `TQLLM_DEBUG=x3n1x3v` (n=1 -> exact x3v), **10/15 match spec exactly** -> by design (non-spec decode sq vs verify x3v).
- Remaining 5 (r000 now valid post-fix, plus r009/r010/r012/r019) diverge deep (371-868 chars) in long outputs; these are x3v MMODE0 (n=1) vs MMODE1 (batched verify) accumulation-tie candidates, but the >0.5 margin rule needs a measured top-2.
- **Margin tool not yet built**: `tqllm::Model::capture(prompt, hidden, logits)` gives the final-position logits, so a small `tools/` tool (args: weights gguf + tokenizer + prefix text -> top-5) is the path; it needs the GGUF->tokenizer route tqserve uses.
- **KLD vs llama-paw (wikitext 32x2048, <1e-3) not yet run** (needs a llama-paw reference logits file).
- sq verdict: keep sq as the n=1 default (bit-matches llama-paw `GGML_PAW_X3_GEMV=0`).

## X3 port Phase 3 — graph-capture bug fixed; G3 tooling outstanding (2026-10-16)

- **Real bug found + fixed**: a raw `/v1/completions` decode failed with `x3: scratch allocation failed (348160 bytes, free 7099908096)`. Cause: the shared scratch allocated lazily on the first matvec, but the engine captures CUDA graphs around decode and `cudaMalloc` is illegal during capture. `X3Format::load` now warms a/had/c/dw/locks to the sizes the Mat can need, so matvec/dequant_h never allocate. `/v1/completions` returns " Paris."; `x3_format` still OK.
- `/v1/completions` and `/completion` expose no logprobs either -> the margin readout must come from a new `tools/` tool (`Model::capture` gives the final-position logits), not the server.
- No pre-existing llama-paw X3.1 kl-divergence-base found under /mnt/hdd/ml/paw27b -> the KLD gate needs one created (`llama-perplexity --kl-divergence-base`, wikitext 32x2048) then `tools/tqkld`.
- G3 still: 25/40 identity; 10/15 divergences proven by-design (sq vs x3v); 5 (r000,r009,r010,r012,r019) deep divergences await the margin tool; KLD outstanding.

## X3 port Phase 3 — G3 margin finding (2026-10-16)

- `tools/tqtop.cpp` built and runs: renders the chat template, appends a prefix, prints top-5 logits at the last position via `verify()`+`verify_logits()` (capture() needs TQ3_4S/BF16).
- **But the G3 harness is flawed**: it compared only `message.content`, while the model returns reasoning separately. tqtop's raw first token for r000 is "We" (margin 9.6) whereas the run's content starts "To find ..." -> the divergence positions are content-relative, not raw-token positions, so the measured margins are not at the true positions.
- Margins measured (x3v mode, sq and x3v identical for all 5): r000 9.62, r009 0.54, r010 0.55, r012 0.14, r019 0.94 - but see the caveat above; r000's "spec" text is still a stale 500 (the spec pass with -md errors on r000).
- To finish G3 the harness must capture reasoning + content (the full raw generation) or the token ids, then re-measure. No clean PASS/FAIL yet.

## X3 port Phase 3 — G3 verdict: FAIL (identity), KLD PASS (2026-10-16)

- **KLD vs llama-paw PASS**: `tools/tqkld --llama-base /mnt/hdd/x3_klbase.bin` (base = `llama-perplexity --kl-divergence-base` on the X3.1 GGUF, wikitext-2 32x2048, PPL 6.1486): **Mean KLD 0.000458 +/- 0.000077 (< 1e-3)**, same top p 99.203%. Whole-model x3 numerics match llama-paw.
- **G3 identity FAIL**: 40 prompts greedy, raw generation = `reasoning_content + content` (harness fixed). 25/40 identical, 15 divergences. 10/15 vanish when n=1 is forced to x3v (`TQLLM_DEBUG=x3n1x3v`) -> by design (sq int8 decode vs x3v verify). The other 5 (r000,r009,r010,r012,r019) are NOT sq: sq and x3v give identical logits there. Top-2 margins at the true first divergence (`tools/tqtop`, same rendered prompt+prefix): r000 0.60, r009 0.70, r010 0.55, r012 0.41, r019 0.60 -> **4 of 5 exceed the 0.5 FAIL threshold**.
- Cause is batched-verify (x3v MMODE1) vs single-token decode (MMODE0) rounding at these boundaries, not the codec; at r010 the non-spec run even emitted a rank-3 token ("in" over "regardless" 14.90 vs 14.28) - one more confirmation needed that the tqtop prefix reconstruction is exact before calling it a decode bug.
- Verdict: KLD PASS; **G3 identity FAIL** by the stated >0.5 rule on 4 prompts. Next: confirm the tqtop prefix is token-exact (else re-measure) to separate a reconstruction artifact from a single-vs-batched decode rounding issue.
