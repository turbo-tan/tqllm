#pragma once
// Minimal reader for (sharded) Hugging Face safetensors checkpoints.
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace tqllm {

struct SafeTensor {
    std::string file;       // shard path
    std::string dtype;      // "BF16", "F32", ...
    std::vector<uint64_t> shape;
    uint64_t begin = 0;     // absolute byte offset in the shard
    uint64_t bytes = 0;
};

// Indexes every tensor in `dir` (model.safetensors or model.safetensors.index.json shards). Names are kept as-is.
std::map<std::string, SafeTensor> safetensors_index(const std::filesystem::path & dir);

// Reads one tensor's raw bytes.
std::vector<char> safetensors_read(const SafeTensor & t);

} // namespace tqllm
