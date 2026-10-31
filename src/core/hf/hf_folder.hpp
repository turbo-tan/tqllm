#pragma once
// Hugging Face checkpoint folders (config.json + safetensors shards) opened in place as a GgufFile in the engine's own
// tensor names, so tqserve --model FOLDER serves a checkpoint as published: no conversion, no second copy on disk.
//
// Three layers, so a new architecture or a new quantisation is one small addition:
//   HfFolder      the folder: config (text_config when the model is multimodal), the safetensors index, and the
//                 tensor builders below; it knows the storage encodings, not the models
//   builders      matrix() picks the encoding from the keys a module has (exl3 trellis / suh / svh -> GGML_X3 parts read
//                 from the shards; MXFP8 weight + weight_scale_inv -> BF16; BF16 / F16 / F32 weights as stored), stack()
//                 makes an expert bank, norm() a plain RMSNorm weight (plus_one for Gemma-style 1 + w)
//   HfArch        one row per architecture (hf_arch.cpp): the HF `architectures` name, the engine arch, and a map
//                 function in its own file (arch_<name>.cpp) that only names tensors and metadata keys
#include "gguf.hpp"
#include "safetensors.hpp"
#include "nlohmann/json.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace tqllm {

class HfFolder {
public:
    explicit HfFolder(const std::string & path);
    HfFolder(const HfFolder &) = delete;  // cfg_ points into root_
    HfFolder & operator=(const HfFolder &) = delete;
    const std::filesystem::path & dir() const { return dir_; }
    const Json & root() const { return root_; }
    const Json & config() const { return *cfg_; }  // text_config of a multimodal checkpoint, else the root
    std::string architecture() const;              // architectures[0]
    double num(const char * key) const;            // a number of config(); throws when missing
    const Json * section(const char * key) const;  // an object of config(), or nullptr
    bool has(const std::string & tensor) const { return index_.count(tensor) != 0; }
    const SafeTensor & at(const std::string & tensor) const;

    // A matrix of `module` (the tensor names without .weight / .trellis), in the encoding it is stored in.
    GgufTensor matrix(const std::string & module) const;
    // Routed experts: one tensor [cols, rows, experts] with one part per module (every module the same shape).
    GgufTensor stack(const std::vector<std::string> & modules) const;
    // A 1-D weight as F32; plus_one stores 1 + w (the engine's norms are plain x * w).
    GgufTensor norm(const std::string & tensor, bool plus_one) const;
    // A tensor stored as is (BF16 / F16 / F32), the engine type from its dtype.
    GgufTensor plain(const std::string & tensor) const;
    // tokenizer.json + tokenizer_config.json (+ chat_template.jinja): byte-level BPE keys of the GGUF tokenizer.
    void tokenizer(GgufFile & f, const std::string & pre, uint64_t vocab) const;

private:
    std::filesystem::path dir_;
    std::string config_text_;
    Json root_;
    const Json * cfg_ = nullptr;
    std::map<std::string, SafeTensor> index_;
};

// The engine architecture of a checkpoint folder hf_arch.cpp maps ("" for any other path).
std::string hf_folder_arch(const std::string & path);
// True when `path` is a checkpoint folder of an architecture hf_arch.cpp maps.
inline bool hf_folder(const std::string & path) { return !hf_folder_arch(path).empty(); }

// The folder as a GgufFile; throws for an architecture with no map.
GgufFile hf_open(const std::string & path);

}  // namespace tqllm
