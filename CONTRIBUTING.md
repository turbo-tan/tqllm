# Contributing to tqLLM

Thank you for helping. tqLLM is a C++/CUDA inference engine for open-weight models, tuned for one person on one GPU.
This guide covers how to build and test, where new pieces plug in, and what a pull request needs.

Start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a map of the code.

## Ground rules

1. **Exact first.** Anything that claims to keep the output (a drafter, a kernel, a cache, a memory placement) must
   produce the same greedy tokens as the path it replaces, and a test must say so. If a change is not exact, say that in
   the PR title and show the quality cost (KLD against BF16, and a task score such as GSM8K).
2. **Numbers name their conditions.** Every speed or quality figure states the GPU, the model and drafter files, the
   context, the prompt type and the engine commit. "Faster" without a measurement does not merge.
3. **Same machine, same session.** Compare against the base commit on the same GPU in the same session; clocks and
   drivers drift between sessions.
4. **Credit borrowed ideas.** If your change follows a paper or another project, add it to the Credits table in
   README.md. Copied code is listed in NOTICE with its licence, and only licences compatible with Apache-2.0 are
   accepted.
5. **No private paths or hosts** in shipped files. `tests/test_no_private_paths.py` checks this.

## Build and test

Requirements: CMake 3.22+, a C++20 compiler, and for the GPU build a recent CUDA toolkit (12.9 is tested).

```sh
# GPU build (all targets and tests)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
ctest --test-dir build --output-on-failure

# CPU-only build: tokenizer, GGUF, options, serving logic and the Python checks, no GPU needed
cmake -S . -B build-cpu -DTQLLM_CUDA=OFF
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

Run the server from the build: `./build/tqserve --model models/model.gguf --port 8086`. `tqserve --help` lists every
option; `--list-kv` and `--list-flags` list the KV formats and environment flags.

Bug fixes come with a test that fails on the old code and passes on the new one. Put it next to the closest existing
test in `tests/` and register it in `CMakeLists.txt` with `add_test`.

### GPU tests run locally, not in CI

Every test carries a ctest label. `cpu` tests run in CI on every pull request (`.github/workflows/checks.yml`, GitHub
runners have no GPU). `gpu` tests (kernels, expert caches, attention, engines) only run on a machine with a CUDA card,
so run them yourself before opening a pull request that touches GPU code, and paste the result line in the pull request:

```sh
ctest --test-dir build -L gpu --output-on-failure   # this card's GPU tests
ctest --test-dir build -L cpu --output-on-failure   # what CI runs
```

Some GPU tests fill the device on purpose (`RUN_SERIAL`); others share it, so keep `-j` low (2) on a 24 GB card.

### How a test is laid out

One program per area: `tests/<area>.cu` (GPU) or `.cpp` (CPU). State the expected behaviour in the comment at the top
of the case, then check it against a reference (a CPU model, a dequantise-then-multiply product, or the path the change
replaces). New files list their cases in a table and report each one (`tests/qsa_gather.cu`):

```cpp
const struct { const char * name; void (*run)(); } cases[] = {{"matches_reference", matches_reference}, ...};
// prints PASS <name> / FAIL <name>: <why> per case, runs all cases, exits 1 if any failed
```

## Where things plug in

The source layout and its layering rules are in [AGENTS.md](AGENTS.md); read it first.

Most extensions are a strategy object plus one registration line; the engine does not branch on the type.

### A weight format

Weight formats turn a GGUF tensor into a device matrix and supply its kernels. The interface is
[src/formats/weight/weight_format.hpp](src/formats/weight/weight_format.hpp).

1. Add `src/formats/weight/<name>.cpp` implementing `WeightFormat`: `load()` (GGUF bytes to device layout), `matvec()` (decode,
   up to `max_matvec_cols()` columns), `dequant_h()` (FP16 for the prefill GEMM), and the optional `gemm()`,
   `matvec_ld()` and `row_bytes()` when the format can do them.
2. Put its CUDA kernels in `src/formats/weight/<name>.cu`, next to its host code.
3. Register it: add `register_<name>()` to the list in [src/formats/weight/registry.cpp](src/formats/weight/registry.cpp) with the
   GGUF type id it loads. Ids of 300 and above are reserved for tqLLM-only formats; pick the next free one and say so in
   the PR.
4. If the format has more than one decode kernel, add rows to the rule table in
   [src/formats/weight/matvec_rules.hpp](src/formats/weight/matvec_rules.hpp). Each rule cites the measurement that justified
   it.
5. Tests: matvec against a dequantise-then-multiply reference, width independence (one column at a time equals a batch),
   and, for formats used by MoE experts, the grouped-expert tests in `tests/`.

### A KV-cache format

KV formats own the record size, the cache writer and the attention for verify and prefill. The interface is
[src/formats/kv/kv_format.hpp](src/formats/kv/kv_format.hpp).

1. Add `src/formats/kv/kv_<name>.cpp` implementing `KvFormat`: `key_bytes()` and `value_bytes()` per context,
   `write_and_attend()`, and the capability flags (`tree_ok()` for tree verify, `compressed()`, `compact_codecs()` for
   TqCache stand-ins).
2. Fill in `provenance()`: who made it, the source, the licence, and whether the idea or code was borrowed. It is
   printed by `tqserve --list-kv`.
3. Register it by name in [src/formats/kv/kv_registry.cpp](src/formats/kv/kv_registry.cpp). Users select it with `--kv <name>`.
4. Tests: attention against the BF16 cache within the stated tolerance (or bit-identical if the format claims so),
   tree verify if `tree_ok()`, and the memory count in `tests/kv_mem_count.cu`.

### A model architecture (engine)

Each architecture is an engine behind `tqllm::Model`. The interface is [src/engine/engine.hpp](src/engine/engine.hpp): `step()` for
one token, `verify()` and `commit()` for speculative decoding, and optional multi-slot and drafter methods. Today's
engine is [src/models/qwen/qwen35/qwen35.cpp](src/models/qwen/qwen35/qwen35.cpp): Qwen3.5 / Qwen3.8 hybrid attention + Gated DeltaNet, dense and mixture-of-experts
(`qwen35moe`, grouped expert kernels in [src/experts/moe.cu](src/experts/moe.cu)).

1. Add `src/models/<family>/<arch>/` with an `Engine` subclass and a `make_<arch>_engine()` factory (see AGENTS.md). Reuse the weight and KV formats
   through their registries rather than calling kernels directly.
2. Dispatch to it from [src/engine/model.cpp](src/engine/model.cpp) on the GGUF `general.architecture` (or the safetensors
   `model_type`).
3. Validate the config at load and refuse clearly what is not implemented yet.
4. Tests: greedy tokens against a reference implementation on a few prompts (a golden file under `tests/`, produced by a
   script that states its source), and `verify()` equal to repeated `step()`.

Please open an issue before starting a new architecture, so the work is not duplicated.

### Environment flags

Settings are command-line options. Environment variables are only for diagnostics or overrides without a CLI form, and
every one is declared in [config/flags.yaml](config/flags.yaml) (`kind: user`, `debug` or `retired`). Run
`python3 tools/gen_flags.py` to regenerate `docs/FLAGS.md` and `src/core/flags_gen.hpp`; the `flags_registry` test fails if
they are out of date. A/B switches used while developing are removed before merge, keeping the winner as the default.

### Plugins (experimental)

[include/tqllm/plugin.h](include/tqllm/plugin.h) is an experimental C ABI for out-of-tree kernels, loaded from a
shared library. ABI 1 is a proof of the loading path only; it is not yet a format or quantisation ABI. If you want to
build on it, open an issue describing what your plugin needs to reach.

### Monitor UI and telemetry panels

The monitor has a source-level registry for trusted panels, with isolated snapshot
copies and per-panel error handling. It is separate from the kernel-plugin ABI.
See [the monitor contributor guide](docs/MONITOR_CONTRIBUTING.md) for the source map,
panel example, hardware collector contract, UI checklist and three-browser tests.

## Performance claims

Use the shipped benchmark so others can reproduce your number:

```sh
sh bench/get_wikitext.sh
python3 bench/relbench.py --bin build/tqserve --gpu "<your GPU>" --wiki bench/data/wiki.test.raw --out results.jsonl \
  --model name=models/<model>.gguf [--drafter models/<drafter>.gguf --vocab models/<vocab>.txt]
```

Run the base commit and your branch on the same GPU in the same session, and paste both lines of `results.jsonl` in
the PR. For changes that should not move speed, show that they do not.

## Code style

- Match the code around you: naming, comment density and idiom. C++20, CUDA, no new dependencies without discussion.
- Comments explain what the code cannot: an invariant, a measured reason, a source. Do not restate the code.
- ASCII in code and commit messages.
- Keep a pull request to one change. Large or cross-cutting changes start as an issue.

## Pull requests

Before you open one:

- [ ] Builds, and `ctest` passes (CPU build at least; `ctest -L gpu` on a CUDA card for kernel or engine changes).
- [ ] New behaviour has a test; a bug fix has a test that failed before.
- [ ] Exactness stated: identical tokens, or the measured quality cost.
- [ ] Speed claims include base and branch numbers from the same session, with GPU, files and settings.
- [ ] New environment flags are in `config/flags.yaml`; borrowed ideas are credited.
- [ ] User-visible changes have a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).

## AI-assisted contributions

AI tools are welcome as assistants. You remain the author: you must understand every line you submit, be able to
explain and debug it without the tool, and answer review in your own words. Say in the PR when an AI tool wrote a
meaningful part of the change. Pull requests that are mostly generated, with no evidence that the author ran and
understood them, will be closed.

## Licence

tqLLM is Apache-2.0. By contributing you agree that your contribution is licensed under the same terms.
