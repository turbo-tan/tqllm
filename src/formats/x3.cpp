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
#include "window.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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
    float* sc = nullptr;  // load-time int8 row scales [rows], amax/127 (1 for a zero row); lives with the descriptor
};

struct Scratch {
    std::mutex mtx;
    float* a = nullptr;   size_t a_cap = 0;
    float* c = nullptr;   size_t c_cap = 0;
    half*  had = nullptr; size_t had_cap = 0;
    int*   locks = nullptr; size_t locks_cap = 0;
    void*  win = nullptr;   size_t win_cap = 0;   // window GEMV scratch (A + partial sums), sized at load
};

Scratch& scratch() { static Scratch s; return s; }

void* grow(void** p, size_t& cap, size_t need) {
    if (need <= cap) return *p;
    if (*p) { cudaFree(*p); *p = nullptr; cap = 0; }
    void* np = nullptr;
    if (cudaMalloc(&np, need) != cudaSuccess) {
        size_t free_b = 0, tot = 0;
        cudaMemGetInfo(&free_b, &tot);
        throw std::runtime_error("x3: scratch allocation failed (" + std::to_string(need) + " bytes, free " +
                                 std::to_string(free_b) + ")");
    }
    *p = np; cap = need;
    return np;
}

int num_sms() {
    static const int n = [] { int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v > 0 ? v : 82; }();
    return n;
}

// Runs of consecutive parts with one codec, at most XW_MAX_PARTS each: one window call per run.
// f(i, j, off): parts [i, j), output row offset of part i.
template <class F>
void for_each_window_run(const X3Desc& d, F&& f) {
    size_t i = 0;
    int off = 0;
    while (i < d.parts.size()) {
        size_t j = i + 1;
        while (j < d.parts.size() && j - i < size_t(x3::XW_MAX_PARTS) && d.parts[j].codec == d.parts[i].codec) ++j;
        f(i, j, off);
        for (size_t k = i; k < j; ++k) off += d.parts[k].rows;
        i = j;
    }
}

size_t window_scratch_bytes(const X3Desc& d) {
    size_t need = 0;
    for_each_window_run(d, [&](size_t i, size_t j, int) {
        int rows[x3::XW_MAX_PARTS];
        for (size_t k = i; k < j; ++k) rows[k - i] = d.parts[k].rows;
        need = std::max(need, x3::xw_scratch_bytes(rows, int(j - i), d.cols, num_sms()));
    });
    return need;
}

void* window_scratch(Scratch& sc, const X3Desc& d) { return grow(&sc.win, sc.win_cap, window_scratch_bytes(d)); }

bool window_ok(const X3Desc& d) {
    if (!x3::x3_window_enabled()) return false;
    for (const X3Part& p : d.parts)
        if (!p.codec->window) return false;
    return true;
}

struct X3Format final : WeightFormat {
    const char* name() const override { return "x3"; }
    Basis basis() const override { return Basis::Plain; }
    // the window GEMV takes a whole tree verify (12 columns) in one pass; the x3v fallback stops at 8
    int max_matvec_cols() const override { return x3::x3_window_enabled() ? x3::XW_MAX_N : 8; }

    Mat load(const std::vector<char>& raw, int rows, int cols, const Upload& up, cudaStream_t stream) const override {
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

        // Warm the shared scratch at load time (outside any CUDA graph capture). matvec/dequant_h
        // must never cudaMalloc: the engine captures CUDA graphs around decode, and an allocation
        // during capture fails (a raw-completions decode hit exactly that).
        {
            Scratch& sc = scratch();
            std::lock_guard<std::mutex> lock(sc.mtx);
            size_t lb = 0;
            for (const X3Part& p : d->parts) {
                lb = std::max(lb, p.codec->ws_bytes(p.cols, p.rows));
            }
            const int nb = max_matvec_cols();
            grow(reinterpret_cast<void**>(&sc.a), sc.a_cap, size_t(nb) * cols * sizeof(float));
            grow(reinterpret_cast<void**>(&sc.had), sc.had_cap, size_t(nb) * cols * sizeof(half));
            grow(reinterpret_cast<void**>(&sc.c), sc.c_cap, size_t(nb) * rows * sizeof(float));
            if (lb) grow(reinterpret_cast<void**>(&sc.locks), sc.locks_cap, lb);
            window_scratch(sc, *d);
        }

        // Row scales for the int8 prefill GEMM, once at load (the weights never change): reconstruct each part to a
        // temporary FP16 buffer and apply the same amax/127 rule as the per-chunk quantise pass (x3_row_scales), so
        // dequant_i8 can skip both the FP16 round trip and the quantise kernel.
        {
            if (cudaMalloc(&d->sc, size_t(rows) * 4) != cudaSuccess) throw std::runtime_error("x3: row scale allocation failed");
            half* tmp = nullptr;
            size_t tmp_cap = 0;
            int off = 0;
            for (const X3Part& p : d->parts) {
                const size_t need = size_t(p.rows) * cols * sizeof(half);
                if (need > tmp_cap) {
                    if (tmp) cudaFree(tmp);
                    if (cudaMalloc(&tmp, need) != cudaSuccess) throw std::runtime_error("x3: scale scratch allocation failed");
                    tmp_cap = need;
                }
                p.codec->dequant_rows(tmp, cols, p.trellis, p.suh, p.svh, p.rows, p.cols, stream);
                x3::x3_row_scales(tmp, p.rows, cols, d->sc + off, stream);
                off += p.rows;
            }
            if (tmp) cudaFree(tmp);
        }
        Mat m;
        m.p = reinterpret_cast<const char*>(d->parts.front().trellis);  // first part's base (informational)
        m.type = GGML_X3;
        m.rows = rows;
        m.cols = cols;
        m.aux = d.release();  // the format owns the descriptor (freed with the process)
        m.fmt = this;
        return m;
    }

    // y rows of n columns ldy apart (ldy >= m.rows): lets the engine write fused outputs in place and run
    // independent small matvecs on its side stream meanwhile (qwen35 mix_mm). Window path only.
    bool has_matvec_ld() const override { return x3::x3_window_enabled(); }
    bool matvec_ld(const Mat& m, const __half* x, __nv_bfloat16* y, int n, int ldy, cudaStream_t s) const override {
        const auto* d = static_cast<const X3Desc*>(m.aux);
        if (n < 1 || n > max_matvec_cols() || ldy < m.rows || !window_ok(*d)) return false;
        run_window(*d, x, y, n, ldy, s);
        return true;
    }

private:
    // Every part into y at its own rows, one window call per same-codec run.
    void run_window(const X3Desc& d, const __half* x, __nv_bfloat16* y, int n, int ldy, cudaStream_t s) const {
        Scratch& sc = scratch();
        std::lock_guard<std::mutex> lock(sc.mtx);
        void* ws = window_scratch(sc, d);
        for_each_window_run(d, [&](size_t i, size_t j, int off) {
            x3::XwList L;
            x3::xw_begin(L, d.cols, n);
            for (size_t k = i; k < j; ++k) {
                const X3Part& p = d.parts[k];
                x3::xw_add(L, p.trellis, p.suh, p.svh, p.rows, off, num_sms());
                off += p.rows;
            }
            x3::xw_bind_scratch(L, ws);
            d.parts[i].codec->window(L, x, y, ldy, s);
        });
    }

public:

    void matvec(const Mat& m, const __half* x, __nv_bfloat16* y, int n, cudaStream_t s) const override {
        if (n < 1 || n > max_matvec_cols()) throw std::runtime_error("x3: matvec batch out of range");
        if (matvec_ld(m, x, y, n, m.rows, s)) return;
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

    // The engine wants [rows][cols] (cuBLAS OP_T, lda = cols): each part is reconstructed straight into its
    // row block (dequant_rows, the out-major form of reconstruct; no scratch, no transpose pass).
    void dequant_h(const Mat& m, __half* out, cudaStream_t s) const override {
        const auto* d = static_cast<const X3Desc*>(m.aux);
        int off = 0;
        for (const X3Part& p : d->parts) {
            p.codec->dequant_rows(out + size_t(off) * m.cols, m.cols, p.trellis, p.suh, p.svh, p.rows, p.cols, s);
            off += p.rows;
        }
    }

    // int8 prefill: each part reconstructed straight to int8 with the load-time row scales (no FP16 round trip, no
    // per-chunk quantise pass); row_scale copied from the descriptor's load-time array.
    bool dequant_i8(const Mat& m, int8_t* w8, float* row_scale, cudaStream_t s) const override {
        const auto* d = static_cast<const X3Desc*>(m.aux);
        int off = 0;
        for (const X3Part& p : d->parts) {
            p.codec->dequant_rows_i8(w8 + size_t(off) * m.cols, m.cols, p.trellis, p.suh, p.svh, p.rows, p.cols,
                                     d->sc + off, s);
            off += p.rows;
        }
        if (cudaMemcpyAsync(row_scale, d->sc, size_t(m.rows) * 4, cudaMemcpyDeviceToDevice, s) != cudaSuccess)
            throw std::runtime_error("x3: row scale copy failed");
        return true;
    }
};

}  // namespace

namespace formats {
void register_x3() { register_weight_format(GGML_X3, std::make_unique<X3Format>()); }
}  // namespace formats

}  // namespace tqllm
