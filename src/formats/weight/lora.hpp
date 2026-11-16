#pragma once
// Runtime LoRA adapters (--lora FILE, llama.cpp's GGUF LoRA files from convert_lora_to_gguf.py): an adapted matrix W
// computes y = W x + scale * factor * B (A x), A = lora_a [rank][cols], B = lora_b [rows][rank], factor = alpha / rank.
// The base weights are never changed, so a request with scale 0 runs the base model unchanged (the device kernels in
// lora.cuh return at once for scale 0).
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace tqllm {

struct LoraTensor {
    int rows = 0, cols = 0, rank = 0;
    std::vector<float> a;  // [rank][cols]
    std::vector<float> b;  // [rows][rank]
};

struct LoraFile {
    std::string path;
    float alpha = 0.f;                          // adapter.lora.alpha (0: none given)
    std::map<std::string, LoraTensor> tensors;  // by the base tensor's GGUF name, e.g. "blk.3.ffn_down.weight"
};

// Reads a GGUF LoRA adapter (general.type "adapter", adapter.type "lora"; lora_a / lora_b pairs in F32, F16 or BF16).
// Throws on a missing half of a pair or inconsistent shapes.
LoraFile lora_read(const std::string & path);

// PEFT / llama.cpp scaling of B: alpha / rank, or 1 when the file has no alpha
inline float lora_factor(float alpha, int rank) { return alpha != 0.f && rank > 0 ? alpha / float(rank) : 1.f; }

// CPU reference: y[n][rows] += scale * factor * B (A x[n]) for x [n][cols] (tests)
void lora_reference(const LoraTensor & t, float factor, float scale, const float * x, float * y, int n);

}  // namespace tqllm
