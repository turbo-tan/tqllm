# Architecture

A map of the source tree for contributors. See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to add pieces.

## Request path

```
HTTP request (tools/tqserve.cpp, tools/serve/)
  -> chat template (tools/serve/jinja_template.cpp, third_party/jinja) and tokenizer (src/core/tokenizer.cpp)
  -> scheduler and sessions (tools/serve/scheduler.hpp, session.hpp): one slot per concurrent request
  -> tqllm::Model (include/tqllm/model.hpp, src/engine/model.cpp): memory fit, engine choice
  -> Engine (src/engine/engine.hpp), e.g. src/models/qwen/qwen35/qwen35.cpp: prefill, decode step, speculative verify / commit
       weights  -> WeightFormat registry (src/formats/weight/weight_format.hpp, registry.cpp) -> src/formats/weight/ (host + kernels, quant.cu)
       KV cache -> KvFormat registry (src/formats/kv/kv_format.hpp, kv_registry.cpp) -> src/models/qwen/qwen35/attention.cu, src/formats/kv/kv_*.cu
       drafter  -> DFlash2 (src/spec/dflash.cu) or the model's MTP head; token tree / chain verify
  -> streamed tokens and per-request timings back to the client
```

## Directories

The layering rules and the diagram are in [AGENTS.md](../AGENTS.md); `tests/test_layering.py` enforces them.

| Path | What lives there |
|---|---|
| `include/tqllm/` | public C++ API: `Model`, `EngineOptions`, tokenizer, runtime, the experimental plugin ABI (`plugin.h`) |
| `src/engine/` | `tqllm::Model` (`model.cpp`: opens GGUF or safetensors, runs the memory fit `fit_context.hpp`, creates the engine) and the engine interface `engine.hpp` |
| `src/models/qwen/qwen35/` | Qwen3.5 / Qwen3.8 engine, dense and qwen35moe: hybrid attention + Gated DeltaNet, prefill, decode, DFlash2 and MTP verify, multi-slot |
| `src/models/qwen/qwen4exp/` | Flash-Next: model description (`qwen4exp.cpp`), kernels (GDN, QSA sparse attention) and the engine in component headers: `fn_runtime` (arena, tuning), `fn_checkpoint` (weights, projection dispatch), `fn_state` (recurrent state, K/V), `fn_cpu_experts` (CPU miss pool), `fn_layers` (GDN, attention, mixer), `fn_model` (decode, layer-major prefill, MTP); `prefill_plan.hpp` holds the tested memory and slot rules; served through `tqserve_flashnext.cpp`, packed experts writer `tools/tqfn_pack.cpp` |
| `src/models/qwen/qwen2/` | the draft-model forward kernels |
| `src/formats/weight/` | weight formats (`tq3`, `tq4`, `tq3_4s`, `q4`, `i8h`, `gsq_q2`, `nvfp4`, `w4s`, `composite`), the registry, k-quant / IQ kernels (`quant.cu`, grids) and the decode rule table (`matvec_rules.hpp`) |
| `src/formats/kv/` | KV-cache formats (`kv_*`), their registry and the TqCache compaction kernels |
| `src/kv/` | KV store: positions on the device or in system RAM (TqCache), CUDA virtual memory tiers |
| `src/experts/` | expert routing, grouped expert GEMM, expert caches (VRAM, host, mixed), the CPU expert path |
| `src/spec/` | DFlash2 drafting and token-tree verify |
| `src/vision/` | image encoder for `--mmproj` |
| `src/core/` | CUDA helpers, int8 / fp8 GEMM, graph capture, GGUF / safetensors readers, tokenizer, options, runtime, generated flags |
| `tools/` | entry points: `tqserve` (the server, `tools/serve/`) and developer tools: `tqrun`, `tqbench`, `tqkld`, `tqverify`, `tqspec`, `tqgolden`, `tqkbench`, `tqvision`, `tqtokenize`, `tqconvert.py` |
| `tests/` | CTest suite (C++ / CUDA unit and exactness tests, Python checks for the server, Docker, shipped files and layering) |
| `bench/` | `relbench.py`, the reproducible speed benchmark behind the README tables |
| `config/flags.yaml` | the registry of every environment variable; `docs/FLAGS.md` is generated from it |
| `docker/`, `Dockerfile`, `compose.yaml` | the image and its entrypoint (Docker variables to `tqserve` options) |
| `third_party/`, `licenses/`, `NOTICE` | bundled code (jinja, nlohmann json, stb_image) and its licences |

## Invariants worth knowing

- **One attention arithmetic.** Plain decoding and every verify path (tree, chain, MTP) compute attention the same way,
  so greedy output with a drafter equals greedy output without one. `test-tree`, `test-dflash-exact` and
  `test-native-mtp` guard this; any new verify path must keep it.
- **Formats are strategies.** The engine never branches on a weight or KV type; it asks the format. New formats need no
  engine changes.
- **The memory fit decides placement at load** and prints what it chose. New memory consumers must be counted by the
  fit, or a configuration that loads can run out of memory later.
- **CUDA graphs** capture decode and verify steps; anything inside a step must keep its device pointers fixed and avoid
  host synchronisation.
