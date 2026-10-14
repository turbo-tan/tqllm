#pragma once
// Minimal GGUF v3 reader: metadata (scalars and arrays) and tensor directory.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace tqllm {

enum GgmlType : uint32_t { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q8_0 = 8, GGML_TQ4 = 300,  /* TQLLM-only type ids start at 300, clear of ggml */ GGML_TQ3 = 301, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_IQ3_XXS = 18, GGML_IQ3_S = 21, GGML_IQ4_XS = 23, GGML_Q6_K = 14, GGML_BF16 = 30, GGML_TQ3_4S = 46,
    /* tq_mlx: kernel-ready MLX layouts, rows padded as the kernels need */
    GGML_TQMLX_TILED3 = 310, GGML_TQMLX_TILED4 = 311, GGML_TQMLX_MMA8_3 = 312, GGML_TQMLX_MMA8_4 = 313 };

struct GgufTensor {
    uint32_t type = 0;
    std::vector<uint64_t> ne;  // ne[0] is the contiguous (column) dimension
    uint64_t offset = 0;       // absolute file offset
    uint64_t bytes = 0;
};

struct GgufValue {
    std::string str;
    double num = 0;
    std::vector<double> arr;  // numeric arrays
    std::vector<std::string> strs;  // string arrays
};

struct GgufFile {
    std::string path;
    std::map<std::string, GgufValue> kv;
    std::map<std::string, GgufTensor> tensors;
    uint64_t n_kv = 0, kv_end = 0, alignment = 32;  // kv_end: byte offset just after the metadata block

    double num(const std::string & key) const;
    double num(const std::string & key, double fallback) const;
    std::vector<char> read(const GgufTensor & t) const;
};

GgufFile gguf_open(const std::string & path);
uint64_t gguf_tensor_bytes(uint32_t type, const std::vector<uint64_t> & ne, const std::string & name);

} // namespace tqllm
