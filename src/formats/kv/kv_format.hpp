#pragma once
// KV-cache formats as strategies: record size, cache writer and attention (verify and prefill shapes). The engine picks
// one by name (TQLLM_KV=<name>; legacy TQLLM_KV_* flags still map). Add a format: one .cpp here + one line in
// kv_registry.cpp.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tqllm {

struct KvStep {  // one attention layer, n new tokens (verify: n <= nver; prefill chunk: n > nver)
    const __nv_bfloat16 * k0;  // token i: K at k0 + i * ts, V right after (kvheads * head_dim later)
    int ts, n;
    void * kc, * vc;
    const int * dpos;          // device: [0] = first position, [1] = first position + 1
    int heads, kvheads, head_dim, nver, split;
    int64_t context;
    const __nv_bfloat16 * q;   // [n][heads][head_dim]
    __nv_bfloat16 * o;         // [n][heads][head_dim]
    float * part;              // split partials scratch
    float scale;
    cudaStream_t stream;
    const int * par = nullptr;  // tree verify (device, n parents; chain allowed): two-part attention (tree_ok formats)
    const int * dep = nullptr;  // node depths (null: chain)
    const void * compact = nullptr;  // TqCache stand-in segment (tqllm_cuda::KvCompactSeg); only formats with compact_ok()
};

// One layer of sparse attention: each query attends to its `width` selected positions (valid[i] == 0: padding).
struct KvSelected {
    const void * keys;            // the format's K rows of the layer
    const void * values;          // and its V rows
    const float * query;          // [tokens][heads][head_dim]
    float * output;               // [tokens][heads][head_dim]
    const int32_t * positions;    // [tokens][width]
    const uint8_t * valid;        // [tokens][width]
    int width = 0, tokens = 0, first_cells = 0, heads = 0, kvheads = 0, head_dim = 0;  // query t sees cells below first_cells + t
    float * scores = nullptr;     // prefill blocks (tokens >= 32): [tokens][heads][width] scratch; null: the per-query path
    float * inv = nullptr;        // [tokens][heads]
    cudaStream_t stream = nullptr;
};

// Who made the format and what was borrowed; printed by `tqserve --list-kv` and logged when a format loads (TQLLM_DEBUG=memlog).
// borrowed: "own", "idea" (follows published work), "code" (adapted, licence checked) or "see NOTICE".
struct KvProvenance {
    const char * summary = "";
    const char * authors = "TQLLM";
    const char * source = "tqllm";
    const char * licence = "Apache-2.0";
    const char * borrowed = "own";
};

struct KvFormat {
    virtual ~KvFormat() = default;
    virtual const char * name() const = 0;
    virtual KvProvenance provenance() const { return {}; }
    // false: write_and_attend() needs only the multi-query partials of the engine's verify scratch (nq * heads * nsplit *
    // (head_dim + 2) floats, independent of context). true (default, conservative): also provision the per-query
    // context-sized partials of the BF16 path (NVER x attention_partial_floats(heads, context, head_dim)).
    virtual bool needs_context_scratch() const { return true; }
    virtual bool compressed() const { return true; }  // false: the single-token step path can read it
    virtual size_t key_bytes(int64_t context, int kvheads, int head_dim) const = 0;
    virtual size_t value_bytes(int64_t context, int kvheads, int head_dim) const = 0;
    virtual int split_factor() const { return 1; }    // verify key splits per SM and KV head
    // reads a TqCache stand-in segment (KvStep::compact); k / v: the tqllm_cuda::KvCodec ids the compactor decodes
    virtual bool compact_codecs(int & k, int & v) const { (void)k, (void)v; return false; }
    bool compact_ok() const { int k = 0, v = 0; return compact_codecs(k, v); }
    // true: write_and_attend() takes KvStep::par / dep (two-part tree verify; partials need split + 1 splits) and a cache row
    // (one position, all KV heads: key_bytes(1, ...) / value_bytes(1, ...)) is self-contained, so a commit may move rows
    virtual bool tree_ok() const { return false; }
    virtual void write_and_attend(const KvStep &) const = 0;
    // Sparse attention over selected positions (Flash-Next's QSA): one layer's K and V rows in this format's layout
    // (bf16: [position][kvheads][head_dim], K and V in separate buffers). Formats without it are refused by engines
    // that attend sparsely.
    virtual bool selected_ok() const { return false; }
    virtual void attend_selected(const KvSelected &) const;
};

const KvFormat * kv_format(const std::string & name);  // throws on unknown names
void register_kv_format(const std::string & name, std::unique_ptr<KvFormat>);
std::vector<std::string> kv_format_names();  // registered formats, sorted
// One block per format: name, bytes per token over `layers` attention layers (K + V), scratch need, provenance.
std::string kv_format_listing(int layers, int kvheads, int head_dim);

}  // namespace tqllm
