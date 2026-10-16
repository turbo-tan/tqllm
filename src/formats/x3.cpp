// X3 (PAW-27B mul1 trellis) weight format. A Mat points at one or more uploaded row-group records
// (the engine row-concatenates fused matrices, e.g. ffn_gate + ffn_up, into one buffer of several
// X3Records) plus a heap X3Desc listing the parts; every kernel choice is made by the codec table
// (kernels/formats/x3/dispatch.cu). The codec entry points run in f32 (matching llama-paw), so
// matvec casts x -> f32 and C -> bf16 around them.
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

// One row group: its own trellis + suh + svh inside the shared uploaded blob.
struct X3Part {
    const x3::X3Codec* codec;
    const uint16_t* trellis;
    const half* suh;
    const half* svh;
    int rows, cols;
};

struct X3Desc {
    std::vector<X3Part> parts;
    int rows, cols;
};

struct Scratch {
    std::mutex mtx;
    float* a = nullptr;   size_t a_cap = 0;
    float* c = nullptr;   size_t c_cap = 0;
    half*  had = nullptr; size_t had_cap = 0;
    half*  dw = nullptr;  size_t dw_cap = 0;
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
        if (cols % 128 || rows % 128) throw std::runtime_error("x3: dims must be multiples of 128");
        auto d = std::make_unique<X3Desc>();
        d->rows = rows; d->cols = cols;
        size_t pos = 0;
        int total = 0;
        while (pos < raw.size()) {
            if (raw.size() - pos < sizeof(X3Record)) throw std::runtime_error("x3: truncated X3Record");
            X3Record rec{};
            std::memcpy(&rec, raw.data() + pos, sizeof(rec));
            if (!x3_record_valid(rec)) throw std::runtime_error("x3: bad X3Record magic/size");
            if (int(rec.cols) != cols) throw std::runtime_error("x3: record cols disagree with the tensor");
            if (rec.rows % 128) throw std::runtime_error("x3: record rows not a multiple of 128");
            const x3::X3Codec* codec = x3_codec_lookup(rec.codec, int(rec.words));
            if (!codec) throw std::runtime_error(x3_codec_unknown(rec.codec, int(rec.words)));

            const size_t tb = size_t(rec.rows / 16) * size_t(cols / 16) * rec.words * 2;
            const size_t sb = size_t(cols) * 2, vb = size_t(rec.rows) * 2;
            if (pos + sizeof(X3Record) + tb + sb + vb > raw.size())
                throw std::runtime_error("x3: record payload overruns the buffer");
            const char* body = raw.data() + pos + sizeof(X3Record);
            const void* dev = up(body, tb + sb + vb);
            if (!dev) throw std::runtime_error("x3: upload failed");

            X3Part p;
            p.codec = codec;
            p.trellis = reinterpret_cast<const uint16_t*>(dev);
            p.suh = reinterpret_cast<const half*>(static_cast<const char*>(dev) + tb);
            p.svh = reinterpret_cast<const half*>(static_cast<const char*>(dev) + tb + sb);
            p.rows = int(rec.rows); p.cols = cols;
            d->parts.push_back(p);
            total += int(rec.rows);
            pos += sizeof(X3Record) + tb + sb + vb;
        }
        if (total != rows) throw std::runtime_error("x3: records total " + std::to_string(total) + " rows, tensor has " +
                                                    std::to_string(rows));

        Mat m;
        m.p = reinterpret_cast<const char*>(d->parts.front().trellis);  // first part's base (informational)
        m.type = GGML_X3;
        m.rows = rows;
        m.cols = cols;
        m.aux = d.release();  // the format owns the descriptor (freed with the process)
        m.fmt = this;
        return m;
    }

    void matvec(const Mat& m, const __half* x, __nv_bfloat16* y, int n, cudaStream_t s) const override {
        if (n < 1 || n > max_matvec_cols()) throw std::runtime_error("x3: matvec batch out of range");
        const auto* d = static_cast<const X3Desc*>(m.aux);
        Scratch& sc = scratch();
        std::lock_guard<std::mutex> lock(sc.mtx);

        const size_t k = size_t(m.cols);
        float* a = static_cast<float*>(grow(reinterpret_cast<void**>(&sc.a), sc.a_cap, size_t(n) * k * sizeof(float)));
        half* had = static_cast<half*>(grow(reinterpret_cast<void**>(&sc.had), sc.had_cap, size_t(n) * k * sizeof(half)));
        x3::x3_cast_h_f32(x, a, int(size_t(n) * k), s);

        int off = 0;
        for (const X3Part& p : d->parts) {
            float* c = static_cast<float*>(grow(reinterpret_cast<void**>(&sc.c), sc.c_cap, size_t(p.rows) * n * sizeof(float)));
            int* locks = nullptr;
            const size_t lb = p.codec->ws_bytes(p.cols, p.rows);
            if (lb) locks = static_cast<int*>(grow(reinterpret_cast<void**>(&sc.locks), sc.locks_cap, lb));
            p.codec->gemv(a, p.trellis, c, n, p.cols, p.rows, p.suh, had, p.svh, locks, s);
            x3::x3_copy_f32_bf16_strided(c, y, p.rows, m.rows, off, n, s);
            off += p.rows;
        }
    }

    // The engine wants [rows][cols] (cuBLAS OP_T, lda = cols). reconstruct emits [in][out] (the fork's
    // canonical order, n=in=cols, m=out=rows), so each part is reconstructed into a scratch then
    // transposed into place.
    void dequant_h(const Mat& m, __half* out, cudaStream_t s) const override {
        const auto* d = static_cast<const X3Desc*>(m.aux);
        Scratch& sc = scratch();
        std::lock_guard<std::mutex> lock(sc.mtx);
        int off = 0;
        for (const X3Part& p : d->parts) {
            half* tmp = static_cast<half*>(grow(reinterpret_cast<void**>(&sc.dw), sc.dw_cap,
                                                size_t(p.cols) * p.rows * sizeof(half)));
            p.codec->dequant(tmp, p.trellis, p.suh, p.svh, p.rows, p.cols, s);
            x3::x3_transpose_h(tmp, out + size_t(off) * m.cols, p.rows, p.cols, m.cols, s);
            off += p.rows;
        }
    }
};

}  // namespace

namespace formats {
void register_x3() { register_weight_format(GGML_X3, std::make_unique<X3Format>()); }
}  // namespace formats

}  // namespace tqllm
