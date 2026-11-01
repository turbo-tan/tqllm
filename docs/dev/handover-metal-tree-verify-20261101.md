# Handover: Metal token-tree verify, tests to write and run (2026-11-01)

For the agent on the Mac lane. Branch **`ccr-dfd66d38-ysr5wl`**, three commits on top of
`tqllm/metal-tree-step` (0f45acbd):

| Commit | What |
|---|---|
| `c8a6cc9a` | Tree step on the device: draft-ahead for token trees (`TreeAhead`), GDN anchor state, tree mask built once per verify |
| `b00ce274` | `SuffixCopy` (copy drafts from the context, host only; low priority, same lever as the closed j1072 / j1073) |
| `117d6719` | 17-32 row verify for token trees, behind `TQLLM_DEBUG=tree_rows:32` (default stays 16) |

None of this has run on Metal. It compiles against the MLX 0.32.4 headers with no new errors, and the MLX-op parts pass
on the CPU backend. Your job: build it on the Mac, write the missing tests below, run every gate, and report.

Reporting rule: results go out as tok/s (decode or prefill) and PASS / FAIL only.

## 0. Before you build: merge the local exact-attention fix

The Mac lane's local `tqllm/metal-tree-step` has an unpushed commit (11-01 ~10:09, "exact tree attention": side-branch
rows' tail segments on their own path keys, job j1565). This branch edits the same tree attention code
(`attention_decode.hpp`: `tree_key_mask` takes an `[n, nb]` block, `cached_tree_mask`; `attention_rows.hpp`:
`causal_attention_rows(..., exact_rows)`). Rebase this branch onto your local branch and keep both: the fix's per-segment
path keys plus the mask cache and the 17-32 row split. Run j1565's gate (text equal to the chain on the 11 gate + long
prompts) on the merged build before anything below.

## 1. Build and the existing gates

```sh
cmake -S . -B build -DTQLLM_CUDA=OFF -DTQLLM_MLX=ON -DCMAKE_PREFIX_PATH=<mlx install> -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build && ctest -R 'mlx_tree|draft-tree|suffix-copy|mlx_gdn|mlx_attention_rows|mlx_nax_affine' --output-on-failure
```

Must pass:
- `mlx_tree_kernels`: now three cases, a 10-row tree, a 32-row tree (13-deep spine, branches) and a 32-row chain.
  GDN rows equal their own path run as a chain; attention rows (through `causal_attention_rows(..., exact_rows=true)`)
  equal their path stored contiguously. Bit for bit.
- `mlx_tree_device` (new): the device tree equals the host selector on the 8K fixture at 15 and 31 nodes; ancestors,
  depths and the device walk equal the host's. Also passes on the CPU backend.
- `draft-tree`, `draft-tree-8k` (its recorded baselines), `suffix-copy`.
- All existing `mlx_gdn*`, `mlx_attention_rows`, `mlx_nax_affine`: the 1-16 row paths must be unchanged.

## 2. Tests to write

Put each next to the closest existing test and register it in `cmake/mlx.cmake` (rule 10: existing ctest names
do not change). Every comparison is bit for bit (`mx::array_equal`), with `MLX_ENABLE_TF32=0` (the ctest property
sets it).

1. **32-row projections == one-row projections** (`tests/mlx_nax_affine.cpp`): `nax_affine_matmul` with 17, 24 and
   32 rows (the `kernel32` tile) against the same rows run one at a time; 4-bit affine and TQ4, `round_bf16` on.
2. **FFN fallback == fused SwiGLU** (`tests/mlx_nax_affine.cpp`): for 1-16 rows,
   `bf(silu(bf(gate))) * up` from two `nax_affine_matmul` calls, cast to FP16, equals `nax_affine_swiglu`. This is the
   assumption behind 17-32 verify rows (they take gate / up + SwiGLU; AR takes the fused kernel). If it fails, the
   17-32 row verify is not exact: add a 32-row `nax_affine_swiglu` instead.
3. **GDN step 17-32 rows == one-row steps** (`tests/mlx_gdn_fuse.cpp` or `mlx_tree_kernels.cpp`): a 32-row
   `gdn_step_fused` chain against 32 single-row calls, each fed the previous call's state and history: outputs, deltas,
   state, keys, decay and history. Both gate forms (alpha / beta given, and the in-kernel ab projection `ABF`).
4. **Accept at 17-32 rows**: `gdn_replay_prefix` with count 17..32 equals the state after that many single-row steps.
5. **Mask cache never goes stale** (`tests/mlx_tree_kernels.cpp`): two different trees of the same size, verified one
   after the other through `causal_attention_rows` at the same lengths; each must equal its own path reference. Then
   16 and 32 rows alternating.
6. **Device tree verify on the model** (`tests/mlx_tree_verify.cpp`, needs `TQLLM_TEST_MODEL`): build a tree with
   `device_tree` from a real `draft_lattice_device`, verify it with the device overload of `verify_tree_device`, and
   check: (a) every row on the greedy path gives the logits of feeding that path one token at a time; (b) the same for a
   31-node tree (32 rows); (c) `accept_path(ticket, path, &parents)` then one AR token equals the plain AR continuation.
7. **TreeAhead == synchronous decoding** (served, `gate.py`): greedy text of the tree pipeline equals plain AR decoding
   on the gate prompts and the long prompts, `--exact on`, at `tree_rows:16` and `tree_rows:32`. Any difference is a
   bug; report the first differing token and prompt.

## 3. Decision runs (only after section 2 is green)

All served through `tqserve` with the same client as the TensorFold rows (owner rule: decisions from tqserve, not tools).

| Run | Arms | Decides |
|---|---|---|
| A | tree step today (`0f45acbd`) vs this branch at `tree_rows:16` | the device pipeline (draft-ahead, anchor state, mask once) |
| B | this branch `tree_rows:16` vs `tree_rows:32` | the 32-row verify |
| C | the winner of B vs TensorFold 1.2.0 | the release rows: 8K (TensorFold 61.8) and 16K (55.0) decode tok/s |

Before B, one diagnostic with the tool (not a decision): a bare 32-row verify against a bare 16-row verify at 8K. The
32-row tree only pays if the 32-row verify costs at most about 1.15x the 16-row one. If it is well above that, the
first suspect is the FFN (gate / up + SwiGLU as separate launches at 17-32 rows): report the stage split, do not tune
blind.

## 4. Where things are

| Piece | File |
|---|---|
| Device tree build, ancestors, walk | `src/backends/metal/tree_device.hpp` |
| Tree draft-ahead pipeline | `src/backends/metal/spec_pipeline.hpp` (`TreeAhead`) |
| Engine use, `tree_rows` key | `src/backends/metal/engine_mlx.cpp` (`spec_begin`, `tree_step`) |
| Device lattice | `src/backends/metal/dflash.cpp` (`draft_lattice_device`, `selector_predecessor`) |
| Device-tree verify, accept with parents | `src/backends/metal/qwen35.cpp` (`verify_tree_device`, `accept_path`) |
| GDN anchor state, 17-32 rows (qx / ox scratch) | `src/backends/metal/gdn.cpp` (`gdn_step_fused`) |
| Attention split, mask cache | `src/backends/metal/attention_rows.hpp`, `attention_decode.hpp` |
| Host selector, 8K fixture | `src/spec/draft_tree.cpp`, `tests/lattice_fixture.hpp`, `tests/data/lattice8k.bin` |

## 5. Known limits

- The selector equals TensorFold's on identical lattices (its `propose()` ported: same trees on 12 of 12 steps, same
  acceptance). Tuned scoring did not hold up on held-out steps; do not ship selector tuning from the 12-step fixture.
  A larger dump (the full 86-step 8K plus 16K) is needed before any selector change.
- Tree verify needs BF16 KV; 17-32 row verify is BF16 KV only.
- If the host walk ever disagrees with the device walk, `TreeAhead::step` throws ("the device path is not the
  target's walk"). Report it with the prompt; do not catch it.
