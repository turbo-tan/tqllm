// GGUF LoRA adapter reader and CPU reference (lora.hpp).
#include "lora.hpp"

#include "dequant.hpp"
#include "gguf.hpp"

#include <stdexcept>

namespace tqllm {

namespace {
std::vector<float> to_f32(const GgufFile & f, const GgufTensor & t, const std::string & name) {
    if (t.type != GGML_F32 && t.type != GGML_F16 && t.type != GGML_BF16)
        throw std::runtime_error("LoRA " + f.path + ": " + name + " must be F32, F16 or BF16 (convert with --outtype f32)");
    uint64_t n = 1;
    for (uint64_t d : t.ne) n *= d;
    const auto raw = f.read(t);
    std::vector<float> out(n);
    dequantize_host(t.type, raw.data(), out.data(), int64_t(n));
    return out;
}
}  // namespace

LoraFile lora_read(const std::string & path) {
    const GgufFile f = gguf_open(path);
    if (f.has("general.type") && f.str("general.type") != "adapter")
        throw std::runtime_error("LoRA " + path + ": general.type is " + f.str("general.type") + ", not adapter");
    if (f.has("adapter.type") && f.str("adapter.type") != "lora")
        throw std::runtime_error("LoRA " + path + ": adapter.type is " + f.str("adapter.type") + ", not lora");
    LoraFile out;
    out.path = path;
    out.alpha = float(f.num("adapter.lora.alpha", 0.0));
    static const std::string sa = ".lora_a", sb = ".lora_b";
    for (const auto & [name, ta] : f.tensors) {
        if (name.size() <= sa.size() || name.compare(name.size() - sa.size(), sa.size(), sa) != 0) continue;
        const std::string base = name.substr(0, name.size() - sa.size());
        const auto ib = f.tensors.find(base + sb);
        if (ib == f.tensors.end()) throw std::runtime_error("LoRA " + path + ": " + name + " has no " + base + sb);
        const GgufTensor & tb = ib->second;
        // ggml order: lora_a ne = {cols, rank} ([rank][cols]), lora_b ne = {rank, rows} ([rows][rank])
        if (ta.ne.size() != 2 || tb.ne.size() != 2 || ta.ne[1] != tb.ne[0] || ta.ne[1] == 0)
            throw std::runtime_error("LoRA " + path + ": " + base + " has inconsistent shapes " + gguf_shape_string(ta) + " / " +
                                     gguf_shape_string(tb));
        LoraTensor t;
        t.cols = int(ta.ne[0]);
        t.rank = int(ta.ne[1]);
        t.rows = int(tb.ne[1]);
        t.a = to_f32(f, ta, name);
        t.b = to_f32(f, tb, base + sb);
        out.tensors.emplace(base, std::move(t));
    }
    for (const auto & [name, t] : f.tensors)
        if (name.size() > sb.size() && name.compare(name.size() - sb.size(), sb.size(), sb) == 0 &&
            !out.tensors.count(name.substr(0, name.size() - sb.size())))
            throw std::runtime_error("LoRA " + path + ": " + name + " has no lora_a");
    if (out.tensors.empty()) throw std::runtime_error("LoRA " + path + ": no lora_a / lora_b tensors");
    return out;
}

void lora_reference(const LoraTensor & t, float factor, float scale, const float * x, float * y, int n) {
    std::vector<double> ax(size_t(t.rank));
    for (int j = 0; j < n; ++j) {
        const float * xj = x + size_t(j) * t.cols;
        for (int k = 0; k < t.rank; ++k) {
            double s = 0;
            for (int c = 0; c < t.cols; ++c) s += double(t.a[size_t(k) * t.cols + c]) * xj[c];
            ax[size_t(k)] = s;
        }
        for (int i = 0; i < t.rows; ++i) {
            double s = 0;
            for (int k = 0; k < t.rank; ++k) s += double(t.b[size_t(i) * t.rank + k]) * ax[size_t(k)];
            y[size_t(j) * t.rows + i] += float(double(scale) * factor * s);
        }
    }
}

}  // namespace tqllm
