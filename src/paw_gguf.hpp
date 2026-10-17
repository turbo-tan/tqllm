#pragma once
// PAW GGUF adapter: turn a PAW checkpoint (currently `paw-dense`) into the arch the engine knows
// (`qwen35`) at open time. The arch table drives the key renames and the tensor rewrite: each
// `m3_trellis` / `m3_suh` / `m3_svh` triple becomes one virtual tensor `<name>.weight` of type
// GGML_X3 carrying an X3Record (rows/cols/words + codec id). No-op for any other architecture.
namespace tqllm {

struct GgufFile;

void paw_adapt(GgufFile & f);

} // namespace tqllm
