#include "kv_format.hpp"
#include "qwen35_kernels.cuh"

#include <stdexcept>
#include <string>

namespace tqllm::formats {
namespace {
using tqllm_cuda::KvCodec;

struct CodecInfo {
    const char * name;
    KvCodec id;
    size_t row_bytes;  // per token, KV head, 256-dim row
    bool k_ok, v_ok;
    const char * summary;
};
constexpr CodecInfo CODECS[] = {
    {"tq3", KvCodec::Tq3, sizeof(tqllm_cuda::Tq3KV256), true, true, "TQ3_0 (3-bit rotated codebook, fp16 RMS per 32, 112 B)"},
    {"tq2s", KvCodec::Tq2s, sizeof(tqllm_cuda::Tq2V256), false, true, "scalar 2-bit (fp16 RMS per 32, 80 B)"},
    {"q8", KvCodec::Q8, sizeof(tqllm_cuda::Q8KV256), true, true, "rotated int8 (fp16 scale per 32, 272 B)"},
    {"tcq2", KvCodec::Tcq2, sizeof(tqllm_cuda::Tcq2V256), false, true,
     "trellis-coded 2-bit (2 x 128-value groups, fp16 norm each, 72 B; own codebook; idea: spiritbuun TCQ)"},
    {"tcq3", KvCodec::Tcq3, sizeof(tqllm_cuda::Tcq3K256), true, false,
     "trellis-coded 3-bit (2 x 128-value groups, 512-state trellis, fp16 norm each, 104 B; own codebook; idea: spiritbuun TCQ)"},
    {"q4", KvCodec::Q4, sizeof(tqllm_cuda::Q4KV256), true, true, "rotated 4-bit, q4_0 style (fp16 scale per 32, 144 B)"},
};

// K and V codecs picked independently: name "<k>/<v>", e.g. tq3/tq3 (= tq3), tq3/tq2s (= tq3k-tq2v).
struct KvSplit final : KvFormat {
    CodecInfo k, v;
    std::string nm, sum;
    KvSplit(const CodecInfo & k_, const CodecInfo & v_) : k(k_), v(v_), nm(std::string(k_.name) + "/" + v_.name),
        sum(std::string("K ") + k_.summary + "; V " + v_.summary) {}
    const char * name() const override { return nm.c_str(); }
    bool needs_context_scratch() const override { return false; }
    KvProvenance provenance() const override {
        return {sum.c_str(), "turbo-tan (llama.cpp-tq3 TQ3_0 layout); TQLLM (split K/V codecs, TCQ implementation and codebook)",
                "llama.cpp-tq3 / tqllm", "Apache-2.0", "idea (trellis KV: spiritbuun/buun-llama-cpp)"};
    }
    int split_factor() const override { return 2; }
    bool tree_ok() const override { return true; }
    bool compact_codecs(int & kc, int & vc) const override {
        kc = int(k.id), vc = int(v.id);
        return (k.id == KvCodec::Tq3 || k.id == KvCodec::Tcq3) && (v.id == KvCodec::Tq3 || v.id == KvCodec::Tcq2);
    }
    size_t key_bytes(int64_t c, int kvh, int hd) const override {
        if (hd != 256) throw std::runtime_error("KV format " + nm + " requires head_dim 256");
        return size_t(c) * kvh * k.row_bytes;
    }
    size_t value_bytes(int64_t c, int kvh, int hd) const override {
        if (hd != 256) throw std::runtime_error("KV format " + nm + " requires head_dim 256");
        return size_t(c) * kvh * v.row_bytes;
    }
    void write_and_attend(const KvStep & a) const override {
        tqllm_cuda::cache_write_kv_side(k.id, a.k0, a.ts, a.n, a.kc, a.dpos, a.kvheads, a.head_dim, a.stream);
        tqllm_cuda::cache_write_kv_side(v.id, a.k0 + a.kvheads * a.head_dim, a.ts, a.n, a.vc, a.dpos, a.kvheads, a.head_dim,
                                        a.stream);
        tqllm_cuda::attention_kv_split(k.id, v.id, a.q, a.kc, a.vc, a.o, a.part, a.dpos + 1, a.n, a.heads, a.kvheads, a.scale,
                                       a.n > a.nver ? 1 : a.split, a.stream, a.par, a.dep,
                                       static_cast<const tqllm_cuda::KvCompactSeg *>(a.compact));
    }
};
}  // namespace

void register_kv_split() {
    for (const auto & k : CODECS)
        if (k.k_ok)
            for (const auto & v : CODECS)
                if (v.v_ok) register_kv_format(std::string(k.name) + "/" + v.name, std::make_unique<KvSplit>(k, v));
}
}  // namespace tqllm::formats
