#pragma once
// Vision encoder for mmproj GGUFs (llama.cpp clip format). Supported: projector qwen3vl_merger (Qwen3-VL ViT + 2x2
// patch merger, as Qwen3.5 / 3.8). Preprocessing and math follow llama.cpp tools/mtmd.
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tqllm {

// Resized, padded, normalised input: CHW FP32, w and h multiples of patch * merge.
struct VisionImage {
    int w = 0, h = 0;
    std::vector<float> chw;
};

class VisionEncoder {
public:
    static std::unique_ptr<VisionEncoder> load(const std::string & mmproj, cudaStream_t stream);
    virtual ~VisionEncoder() = default;
    virtual int embd() const = 0;  // output width (the text model's hidden size)
    virtual int align() const = 0;  // patch * merge: input pixels per output token along each axis
    // token budget per image (llama.cpp image_min_tokens / image_max_tokens)
    virtual void set_token_limits(int min_tokens, int max_tokens) = 0;
    virtual VisionImage preprocess(const uint8_t * rgb, int w, int h) const = 0;
    // embeddings [nx * ny][embd()] FP32 on the device, row-major over the nx x ny output grid; valid until the next call
    virtual const float * encode(const VisionImage &, int & nx, int & ny) = 0;
};

// JPEG / PNG / BMP / GIF (first frame) / ... -> RGB8; false if the bytes are not an image
bool decode_image(const void * data, size_t bytes, std::vector<uint8_t> & rgb, int & w, int & h);

}  // namespace tqllm
