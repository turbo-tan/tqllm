# Architecture

A map of the source tree for contributors. See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to add pieces.

## Request path

```
HTTP request (tools/tqserve.cpp, tools/serve/)
  -> chat template (tools/serve/jinja_template.cpp, third_party/jinja) and tokenizer (src/tokenizer.cpp)
  -> scheduler and sessions (tools/serve/scheduler.hpp, session.hpp): one slot per concurrent request
  -> tqllm::Model (include/tqllm/model.hpp, src/model.cpp): memory fit, engine choice
  -> Engine (src/engine.hpp), e.g. src/qwen35.cpp: prefill, decode step, speculative verify / commit
       weights  -> WeightFormat registry (src/formats/weight_format.hpp, registry.cpp) -> kernels/formats/, kernels/quant.cu
       KV cache -> KvFormat registry (src/formats/kv_format.hpp, kv_registry.cpp) -> kernels/attention.cu, kernels/formats/kv_*.cu
       drafter  -> DFlash2 (kernels/dflash.cu) or the model's MTP head; token tree / chain verify
  -> streamed tokens and per-request timings back to the client
```

## Directories

| Path | What lives there |
|---|---|
| `include/tqllm/` | public C++ API: `Model`, `EngineOptions`, tokenizer, runtime, the experimental plugin ABI (`plugin.h`) |
| `src/model.cpp` | `tqllm::Model`: opens GGUF or safetensors, runs the memory fit (`fit_context.hpp`), creates the engine |
| `src/engine.hpp` | the per-architecture engine interface |
| `src/qwen35.cpp` | Qwen3.5 / Qwen3.8 engine: hybrid attention + Gated DeltaNet, prefill, decode, DFlash2 and MTP verify, multi-slot |
| `src/formats/` | weight formats (`tq3.cpp`, `tq4.cpp`, `tq3_4s.cpp`, `q4.cpp`, `i8h.cpp`, `composite.cpp`) and KV formats (`kv_*.cpp`), each registered by id or name |
| `src/kv_store*.cpp` | TqCache: KV positions in system RAM, staged back while decoding |
| `src/gguf.cpp`, `src/safetensors.cpp` | model file readers |
| `src/options.cpp` | command-line options shared by the tools |
| `src/vision.cpp` | image encoder for `--mmproj` |
| `kernels/` | CUDA kernels: attention, quantised matvec / GEMM (`quant.cu`, `int8_gemm.cu`, `fp8_gemm.cu`), DFlash2, KV compaction, vision |
| `kernels/formats/` | per-format kernels and the decode kernel rule table (`matvec_rules.hpp`) |
| `tools/` | `tqserve` (the server) and developer tools: `tqrun`, `tqbench`, `tqkld`, `tqverify`, `tqspec`, `tqgolden`, `tqkbench`, `tqvision`, `tqtokenize`, `tqconvert.py` |
| `tests/` | CTest suite (C++ / CUDA unit and exactness tests, Python checks for the server, Docker and shipped files) |
| `bench/` | `relbench.py`, the reproducible speed benchmark behind the README tables |
| `config/flags.yaml` | the registry of every environment variable; `docs/FLAGS.md` is generated from it |
| `docker/`, `Dockerfile`, `compose.yaml` | the image and its entrypoint (Docker variables to `tqserve` options) |
| `third_party/`, `licenses/`, `NOTICE` | bundled code and its licences |

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
