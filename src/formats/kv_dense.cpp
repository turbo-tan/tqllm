#include "kv_format.hpp"
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"

#include <algorithm>
#include <cstdlib>

namespace tqllm::formats {
namespace {
// BF16 K/V (default; also readable by the single-token step path).
struct KvBf16 final : KvFormat {
    const char * name() const override { return "bf16"; }
    KvProvenance provenance() const override { return {"BF16 K and V (reference format, also read by the single-token path)"}; }
    bool compressed() const override { return false; }
    bool tree_ok() const override { return true; }
    size_t key_bytes(int64_t c, int kvh, int hd) const override { return size_t(c) * kvh * hd * 2; }
    size_t value_bytes(int64_t c, int kvh, int hd) const override { return size_t(c) * kvh * hd * 2; }
    void write_and_attend(const KvStep & a) const override {
        auto * kc = static_cast<__nv_bfloat16 *>(a.kc), * vc = static_cast<__nv_bfloat16 *>(a.vc);
        tqllm_cuda::cache_write_multi(a.k0, a.k0 + a.kvheads * a.head_dim, a.ts, a.n, kc, vc, a.dpos, a.kvheads,
                                      a.head_dim, a.stream);
        if (a.par) {  // tree / chain verify block: cache part + in-block ancestors (n <= 8)
            tqllm_cuda::attention_mq256(a.q, kc, vc, a.o, a.part, a.dpos + 1, a.n, a.heads, a.kvheads, a.scale, a.split,
                                        a.stream, 0, a.par, a.dep);
            return;
        }
        if (a.n > a.nver) {  // prefill chunk: the partials buffer holds >= n queries x 1 split
            tqllm_cuda::attention_mq256_prefill(a.q, kc, vc, a.o, a.part,
                                                size_t(tqllm_cuda::attention_mq_partial_floats(a.heads, a.n, 1)),
                                                a.dpos + 1, a.n, a.heads, a.kvheads, a.scale, a.split, a.stream);
            return;
        }
        for (int off = 0; off < a.n; off += 8) {
            const size_t qoff = size_t(off) * a.heads * a.head_dim;
            tqllm_cuda::attention_mq256(a.q + qoff, kc, vc, a.o + qoff, a.part, a.dpos + 1, std::min(8, a.n - off),
                                        a.heads, a.kvheads, a.scale, a.split, a.stream, off);
        }
    }
};

// int8 K and V with an FP16 scale per 32.
struct KvQ8 final : KvFormat {
    const char * name() const override { return "q8"; }
    KvProvenance provenance() const override { return {"int8 K and V with an fp16 scale per 32"}; }
    size_t key_bytes(int64_t c, int kvh, int hd) const override { const size_t n = size_t(c) * kvh * hd; return n + n / 32 * 2; }
    size_t value_bytes(int64_t c, int kvh, int hd) const override { return key_bytes(c, kvh, hd); }
    void write_and_attend(const KvStep & a) const override {
        const size_t kvn = size_t(a.context) * a.kvheads * a.head_dim;
        auto * k8 = static_cast<int8_t *>(a.kc), * v8 = static_cast<int8_t *>(a.vc);
        auto * ks = reinterpret_cast<__half *>(k8 + kvn), * vs = reinterpret_cast<__half *>(v8 + kvn);
        tqllm_cuda::cache_write_multi_q8(a.k0, a.k0 + a.kvheads * a.head_dim, a.ts, a.n, k8, v8, ks, vs, a.dpos, a.kvheads,
                                         a.head_dim, a.stream);
        for (int off = 0; off < a.n; off += 8) {
            const size_t qoff = size_t(off) * a.heads * a.head_dim;
            tqllm_cuda::attention_mq256_q8(a.q + qoff, k8, v8, ks, vs, a.o + qoff, a.part, a.dpos + 1,
                                           std::min(8, a.n - off), a.heads, a.kvheads, a.scale, a.split, a.stream, off);
        }
    }
};
}  // namespace
void register_kv_dense() {
    register_kv_format("bf16", std::make_unique<KvBf16>());
    register_kv_format("q8", std::make_unique<KvQ8>());
}
}  // namespace tqllm::formats
