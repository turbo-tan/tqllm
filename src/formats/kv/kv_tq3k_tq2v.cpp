#include "kv_format.hpp"
#include "qwen35_kernels.cuh"

#include <stdexcept>

namespace tqllm::formats {
namespace {
struct KvTq3kTq2v final : KvFormat {
    const char * name() const override { return "tq3k-tq2v"; }
    bool needs_context_scratch() const override { return false; }
    KvProvenance provenance() const override {
        return {"TQ3_0 K and four-level 2-bit V: 112 B K + 80 B V per 256-dim head",
                "turbo-tan (TQ3_0 layout); TQLLM (2-bit V extension)", "llama.cpp-tq3 / tqllm", "Apache-2.0", "see NOTICE"};
    }
    int split_factor() const override { return 2; }
    bool tree_ok() const override { return true; }
    size_t key_bytes(int64_t c, int kvh, int hd) const override {
        if (hd != 256) throw std::runtime_error("KV format tq3k-tq2v requires head_dim 256");
        return size_t(c) * kvh * sizeof(tqllm_cuda::Tq3KV256);
    }
    size_t value_bytes(int64_t c, int kvh, int hd) const override {
        if (hd != 256) throw std::runtime_error("KV format tq3k-tq2v requires head_dim 256");
        return size_t(c) * kvh * sizeof(tqllm_cuda::Tq2V256);
    }
    void write_and_attend(const KvStep & a) const override {
        auto * k3 = static_cast<tqllm_cuda::Tq3KV256 *>(a.kc);
        auto * v2 = static_cast<tqllm_cuda::Tq2V256 *>(a.vc);
        tqllm_cuda::cache_write_multi_tq3k_tq2v(a.k0, a.k0 + a.kvheads * a.head_dim, a.ts, a.n, k3, v2, a.dpos,
                                                a.kvheads, a.head_dim, a.stream);
        tqllm_cuda::attention_kv_split(tqllm_cuda::KvCodec::Tq3, tqllm_cuda::KvCodec::Tq2s, a.q, k3, v2,
                                       a.o, a.part, a.dpos + 1, a.n, a.heads, a.kvheads, a.scale, a.n > a.nver ? 1 : a.split, a.stream, a.par, a.dep);
    }
};
}  // namespace

void register_kv_tq3k_tq2v() { register_kv_format("tq3k-tq2v", std::make_unique<KvTq3kTq2v>()); }
}  // namespace tqllm::formats
