#pragma once
// paw-dense GGUF adapter: turn a PAW X3 (`paw-dense`) file into a `qwen35` view at open time.
// Metadata keys are aliased, and each `m3_trellis` / `m3_suh` / `m3_svh` triple becomes one
// virtual tensor `<name>.weight` of type GGML_X3. No-op for any other architecture.
namespace tqllm {

struct GgufFile;

void paw_adapt(GgufFile & f);

} // namespace tqllm
