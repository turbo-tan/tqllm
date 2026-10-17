#pragma once
// Minimal GGUF v3 reader: metadata (scalars and arrays) and tensor directory.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace tqllm {

// Numeric GGUF type ids follow ggml exactly. Keep entries in id order so a new format is easy to
// add and to diff. TQLLM-only formats start at 300, clear of ggml's ids.
enum GgmlType : uint32_t {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q8_0 = 8,
    GGML_Q2_K = 10,      // MoE expert tensors (GLM-5.3-Flash ffn_down_exps)
    GGML_Q3_K = 11,
    GGML_Q4_K = 12,
    GGML_Q5_K = 13,
    GGML_Q6_K = 14,
    GGML_IQ2_XXS = 16,
    GGML_IQ2_XS = 17,
    GGML_IQ3_XXS = 18,
    GGML_IQ1_S = 19,
    GGML_IQ4_NL = 20,
    GGML_IQ3_S = 21,
    GGML_IQ2_S = 22,     // MoE expert tensors (GLM-5.3-Flash ffn_gate_exps / ffn_up_exps)
    GGML_IQ4_XS = 23,
    GGML_IQ1_M = 29,
    GGML_BF16 = 30,
    GGML_TQ3_4S = 46,
    GGML_TQ4 = 300, GGML_TQ3 = 301,  // TQLLM-only ids
    GGML_GSQ_Q2_0 = 302,  // explicit GSQ-RCO dialect: external id42, not fork Q1_0
};


struct GgufTensor {
    uint32_t type = 0;
    std::vector<uint64_t> ne;  // ne[0] is the contiguous (column) dimension
    uint64_t offset = 0;       // absolute file offset
    uint64_t bytes = 0;
    uint64_t type_header_offset = 0; // original type field, for explicit dialect reader adapters
};

// Block layout of a quantised GGUF type: `elems` weights per block, `bytes` per block.
// Returns false for types this reader cannot size. Single source of truth for every byte-count
// computation in the loader (tensor_bytes and the MoE expert slicer both use it) so a block
// size can never disagree between the two.
bool gguf_block_layout(uint32_t type, uint64_t & elems, uint64_t & bytes);

// One expert slice of a 3-D MoE expert tensor [cols, rows, n_experts].
//
// MoE checkpoints store every expert of a projection in ONE tensor with the expert index as
// ne[2] (GLM-5.3-Flash: ffn_gate_exps [4096,2048,288], 774 MB). Routing picks ~8 of 288
// experts per token, so a loader must be able to address one expert's bytes without reading or
// uploading all of them.
//
// Returns a tensor descriptor whose offset/bytes cover exactly expert `idx`, and whose ne is
// reduced to the 2-D [cols, rows] matrix that expert holds. Throws on any inconsistency rather
// than returning something plausible: a wrong slice silently reads a neighbouring expert's
// weights and still produces fluent text.
GgufTensor gguf_expert_slice(const GgufTensor & t, uint64_t idx, const std::string & name);

// Number of experts in a tensor, or 0 if it is not a 3-D expert tensor.
uint64_t gguf_expert_count(const GgufTensor & t);

// Human-readable shape "[d0,d1,...]" for error messages. Kept next to the reader so loaders and
// validators report a tensor's declared dims identically instead of each hand-rolling the format.
std::string gguf_shape_string(const GgufTensor & t);

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

    double num(const std::string & key) const;
    double num(const std::string & key, double fallback) const;
    // String KV (e.g. general.architecture). Throws when the key is absent or is not a string;
    // `has` lets a caller branch on presence instead of catching.
    std::string str(const std::string & key) const;
    bool has(const std::string & key) const;
    std::vector<char> read(const GgufTensor & t) const;
};

enum class GgufDialect { Native, GsqRco };
GgufFile gguf_open(const std::string & path, GgufDialect dialect = GgufDialect::Native);

} // namespace tqllm
