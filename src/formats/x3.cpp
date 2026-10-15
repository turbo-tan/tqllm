// X3 (PAW-27B mul1 trellis) weight format. A Mat points at one uploaded trellis+suh+svh blob and a
// heap X3Desc; every kernel choice is made by the codec table (kernels/formats/x3/dispatch.cu). The
// codec entry points run in f32 (matching llama-paw), so matvec casts x -> f32 and C -> bf16 around
// them. Scratch is a process-wide pool grown on demand (never inside a capture-sized steady state).
#include "weight_format.hpp"

#include "gguf.hpp"
#include "x3_codecs.hpp"
#include "x3_record.hpp"
#include "x3.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace tqllm {
namespace {

struct X3Desc {
    const x3::X3Codec* codec;
    const uint16_t* trellis;
    const half* suh;
    const half* svh;
    int rows, cols, words;
};

struct Scratch {
    std::mutex mtx;
    float* a = nullptr;   size_t a_cap = 0;
    float* c = nullptr;   size_t c_cap = 0;
    half*  had = nullptr; size_t had_cap = 0;
    int*   locks = nullptr; size_t locks_cap = 0;
};

Scratch& scratch() { static Scratch s; return s; }

void* grow(void** p, size_t& cap, size_t need) {
    if (need <= cap) return *p;
    if (*p) { cudaFree(*p); *p = nullptr; cap = 0; }
    void* np = nullptr;
    if (cudaMalloc(&np, need) != cudaSuccess) throw std::runtime_error("x3: scratch allocation failed");
    *p = np; cap = need;
    return np;
}

struct X3Format final : WeightFormat {
    const char* name() const override { return "x3"; }
    Basis basis() const override { return Basis::Plain; }
    int max_matvec_cols() const override { return 8; }

    Mat load(const std::vector<char>& raw, int rows, int cols, const Upload& up, cudaStream_t) const override {
        if (raw.size() < sizeof(X3Record)) throw std::runtime_error("x3: tensor smaller than an X3Record");
        X3Record rec{};
        std::memcpy(&rec, raw.data(), sizeof(rec));
        if (!x3_record_valid(rec)) throw std::runtime_error("x3: bad X3Record magic/size");
        if (int(rec.rows) != rows || int(rec.cols) != cols)
            throw std::runtime_error("x3: record dims disagree with the tensor");
        if (cols % 128 || rows % 128) throw std::runtime_error("x3: dims must be multiples of 128");
        const x3::X3Codec* codec = x3_codec_lookup(rec.codec, int(rec.words));
        if (!codec) throw std::runtime_error(x3_codec_unknown(rec.codec, int(rec.words)));

        const size_t tiles = size_t(rows / 16) * size_t(cols / 16);
        const size_t trellis_bytes = tiles * rec.words * 2;
        const size_t expect = sizeof(X3Record) + trellis_bytes + size_t(cols) * 2 + size_t(rows) * 2;
        if (raw.size() != expect) throw std::runtime_error("x3: payload size mismatch");

        const char* body = raw.data() + sizeof(X3Record);
        const void* dev = up(body, raw.size() - sizeof(X3Record));
        if (!dev) throw std::runtime_error("x3: upload failed");

        auto* d = new X3Desc{};
        d->codec = codec;
        d->trellis = reinterpret_cast<const uint16_t*>(dev);
        d->suh = reinterpret_cast<const half*>(static_cast<const char*>(dev) + trellis_bytes);
        d->svh = reinterpret_cast<const half*>(static_cast<const char*>(dev) + trellis_bytes + size_t(cols) * 2);
        d->rows = rows; d->cols = cols; d->words = int(rec.words);

        Mat m;
        m.p = dev;
        m.type = GGML_X3;
        m.rows = rows;
        m.cols = cols;
        m.aux = d;  // the format owns the descriptor (freed with the process)
        m.fmt = this;
        return m;
    }

    void matvec(const Mat& m, const __half* x, __nv_bfloat16* y, int n, cudaStream_t s) const override {
        if (n < 1 || n > max_matvec_cols())
            throw std::runtime_error("x3: matvec batch out of range");
        const auto* d = static_cast<const X3Desc*>(m.aux);
        Scratch& sc = scratch();
        std::lock_guard<std::mutex> lock(sc.mtx);

        const size_t k = size_t(m.cols);
        float* a = static_cast<float*>(grow(reinterpret_cast<void**>(&sc.a), sc.a_cap, size_t(n) * k * sizeof(float)));
        half* had = static_cast<half*>(grow(reinterpret_cast<void**>(&sc.had), sc.had_cap, size_t(n) * k * sizeof(half)));
        float* c = static_cast<float*>(grow(reinterpret_cast<void**>(&sc.c), sc.c_cap, size_t(m.rows) * n * sizeof(float)));
        int* locks = nullptr;
        const size_t lb = d->codec->ws_bytes(int(k), m.rows);
        if (lb) locks = static_cast<int*>(grow(reinterpret_cast<void**>(&sc.locks), sc.locks_cap, lb));

        x3::x3_cast_h_f32(x, a, int(size_t(n) * k), s);
        d->codec->gemv(a, d->trellis, c, n, int(k), m.rows, d->suh, had, d->svh, locks, s);
        x3::x3_cast_f32_bf16(c, y, int(size_t(m.rows) * n), s);
    }

    void dequant_h(const Mat& m, __half* out, cudaStream_t s) const override {
        const auto* d = static_cast<const X3Desc*>(m.aux);
        d->codec->dequant(out, d->trellis, d->suh, d->svh, m.rows, m.cols, s);
    }
};

}  // namespace

namespace formats {
void register_x3() { register_weight_format(GGML_X3, std::make_unique<X3Format>()); }
}  // namespace formats

}  // namespace tqllm
