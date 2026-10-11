#include "kv_format.hpp"
#include "qwen35_kernels.cuh"

#include <stdexcept>

namespace tqllm::formats {
namespace {
// llama.cpp-tq3 TQ3_0 K + V (8 x 32-value blocks per 256-dim row, fp16 RMS each; 112 B per head per token).
struct KvTq3 final : KvFormat {
    const char * name() const override { return "tq3"; }
    bool needs_context_scratch() const override { return false; }  // attention_tq3kv uses only the multi-query partials
    KvProvenance provenance() const override {
        return {"TQ3_0 K and V: 8 x 32-value blocks per 256-dim row, fp16 RMS each, 3-bit rotated codebook (112 B per head per token)",
                "turbo-tan (llama.cpp-tq3 TQ3_0 layout)", "llama.cpp-tq3 / tqllm", "Apache-2.0", "see NOTICE"};
    }
    int split_factor() const override { return 2; }
    bool tree_ok() const override { return true; }
    bool compact_codecs(int & k, int & v) const override { k = v = int(tqllm_cuda::KvCodec::Tq3); return true; }
    size_t key_bytes(int64_t c, int kvh, int hd) const override {
        if (hd != 256) throw std::runtime_error("KV format tq3 requires head_dim 256");
        return size_t(c) * kvh * sizeof(tqllm_cuda::Tq3KV256);
    }
    size_t value_bytes(int64_t c, int kvh, int hd) const override { return key_bytes(c, kvh, hd); }
    void write_and_attend(const KvStep & a) const override {
        auto * k3 = static_cast<tqllm_cuda::Tq3KV256 *>(a.kc), * v3 = static_cast<tqllm_cuda::Tq3KV256 *>(a.vc);
        tqllm_cuda::cache_write_multi_tq3kv(a.k0, a.k0 + a.kvheads * a.head_dim, a.ts, a.n, k3, v3, a.dpos, a.kvheads,
                                            a.head_dim, a.stream);
        tqllm_cuda::attention_kv_split(tqllm_cuda::KvCodec::Tq3, tqllm_cuda::KvCodec::Tq3, a.q, k3, v3,
                                       a.o, a.part, a.dpos + 1, a.n, a.heads, a.kvheads, a.scale, a.n > a.nver ? 1 : a.split, a.stream, a.par, a.dep,
                                       static_cast<const tqllm_cuda::KvCompactSeg *>(a.compact));
    }
};
}  // namespace
void register_kv_tq3() { register_kv_format("tq3", std::make_unique<KvTq3>()); }
}  // namespace tqllm::formats
