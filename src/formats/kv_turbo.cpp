#include "kv_format.hpp"
#include "qwen35_kernels.cuh"

#include <algorithm>
#include <stdexcept>

namespace tqllm::formats {
namespace {
using tqllm_cuda::Turbo3V256;
using tqllm_cuda::Turbo4K256;
void need256(int hd, const char * name) {
    if (hd != 256) throw std::runtime_error(std::string("KV format ") + name + " requires head_dim 256");
}

// int8 K (FP16 scale per 32) + Turbo3 V (3-bit, 128-value blocks). Fails the free-running quality gate.
struct KvQ8Turbo3V final : KvFormat {
    const char * name() const override { return "q8k-turbo3v"; }
    KvProvenance provenance() const override {
        return {"int8 K (fp16 scale per 32) + Turbo3 V (3-bit, 128-value blocks); fails the free-running quality gate", "TQLLM",
                "TurboQuant+ record layouts (TheTom/turboquant_plus, Apache-2.0)", "Apache-2.0", "see NOTICE"};
    }
    int split_factor() const override { return 2; }
    size_t key_bytes(int64_t c, int kvh, int hd) const override { const size_t n = size_t(c) * kvh * hd; return n + n / 32 * 2; }
    size_t value_bytes(int64_t c, int kvh, int hd) const override { need256(hd, name()); return size_t(c) * kvh * sizeof(Turbo3V256); }
    void write_and_attend(const KvStep & a) const override {
        const size_t kvn = size_t(a.context) * a.kvheads * a.head_dim;
        auto * k8 = static_cast<int8_t *>(a.kc);
        auto * ks = reinterpret_cast<__half *>(k8 + kvn);
        auto * v3 = static_cast<Turbo3V256 *>(a.vc);
        tqllm_cuda::cache_write_multi_q8_turbo3v(a.k0, a.k0 + a.kvheads * a.head_dim, a.ts, a.n, k8, ks, v3, a.dpos,
                                                 a.kvheads, a.head_dim, a.stream);
        for (int off = 0; off < a.n; off += 8) {
            const size_t qoff = size_t(off) * a.heads * a.head_dim;
            tqllm_cuda::attention_mq256_q8_turbo3v(a.q + qoff, k8, v3, ks, a.o + qoff, a.part, a.dpos + 1,
                                                   std::min(8, a.n - off), a.heads, a.kvheads, a.scale, a.split, a.stream, off);
        }
    }
};

// Turbo4 K + Turbo3 V.
struct KvT4kTurbo3V final : KvFormat {
    const char * name() const override { return "t4k-turbo3v"; }
    KvProvenance provenance() const override {
        return {"Turbo4 K + Turbo3 V (the 16 GB turbo3-V setting)", "TQLLM",
                "TurboQuant+ record layouts (TheTom/turboquant_plus, Apache-2.0)", "Apache-2.0", "see NOTICE"};
    }
    int split_factor() const override { return 2; }
    size_t key_bytes(int64_t c, int kvh, int hd) const override { need256(hd, name()); return size_t(c) * kvh * sizeof(Turbo4K256); }
    size_t value_bytes(int64_t c, int kvh, int hd) const override { need256(hd, name()); return size_t(c) * kvh * sizeof(Turbo3V256); }
    void write_and_attend(const KvStep & a) const override {
        auto * k4 = static_cast<Turbo4K256 *>(a.kc);
        auto * v3 = static_cast<Turbo3V256 *>(a.vc);
        tqllm_cuda::cache_write_multi_t4k_turbo3v(a.k0, a.k0 + a.kvheads * a.head_dim, a.ts, a.n, k4, v3, a.dpos, a.kvheads,
                                                  a.head_dim, a.stream);
        for (int off = 0; off < a.n; off += 8) {
            const size_t qoff = size_t(off) * a.heads * a.head_dim;
            tqllm_cuda::attention_mq256_t4k_turbo3v(a.q + qoff, k4, v3, a.o + qoff, a.part, a.dpos + 1, std::min(8, a.n - off),
                                                    a.heads, a.kvheads, a.scale, a.split, a.stream, off);
        }
    }
};

// Turbo4 K + Turbo4 V.
struct KvT4kv final : KvFormat {
    const char * name() const override { return "t4kv"; }
    KvProvenance provenance() const override {
        return {"Turbo4 K + Turbo4 V (4-bit K and V)", "TQLLM",
                "TurboQuant+ record layouts (TheTom/turboquant_plus, Apache-2.0)", "Apache-2.0", "see NOTICE"};
    }
    int split_factor() const override { return 2; }
    size_t key_bytes(int64_t c, int kvh, int hd) const override { need256(hd, name()); return size_t(c) * kvh * sizeof(Turbo4K256); }
    size_t value_bytes(int64_t c, int kvh, int hd) const override { return key_bytes(c, kvh, hd); }
    void write_and_attend(const KvStep & a) const override {
        auto * k4 = static_cast<Turbo4K256 *>(a.kc), * v4 = static_cast<Turbo4K256 *>(a.vc);
        tqllm_cuda::cache_write_multi_t4kv(a.k0, a.k0 + a.kvheads * a.head_dim, a.ts, a.n, k4, v4, a.dpos, a.kvheads,
                                           a.head_dim, a.stream);
        if (a.n > a.nver) {  // prefill chunk: FA-style kernel, all queries in one launch
            const int ns = std::max(1, a.split * a.kvheads / (a.kvheads * ((a.n + 7) / 8)));
            tqllm_cuda::attention_mq256_t4kv_prefill(a.q, k4, v4, a.o, a.part, a.dpos + 1, a.n, a.heads, a.kvheads, a.scale,
                                                     ns, a.stream);
            return;
        }
        for (int off = 0; off < a.n; off += 8) {
            const size_t qoff = size_t(off) * a.heads * a.head_dim;
            tqllm_cuda::attention_mq256_t4kv(a.q + qoff, k4, v4, a.o + qoff, a.part, a.dpos + 1, std::min(8, a.n - off),
                                             a.heads, a.kvheads, a.scale, a.split, a.stream, off);
        }
    }
};
}  // namespace
void register_kv_turbo() {
    register_kv_format("q8k-turbo3v", std::make_unique<KvQ8Turbo3V>());
    register_kv_format("t4k-turbo3v", std::make_unique<KvT4kTurbo3V>());
    register_kv_format("t4kv", std::make_unique<KvT4kv>());
}
}  // namespace tqllm::formats
