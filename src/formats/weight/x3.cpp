// X3 (PAW-27B mul1 trellis) weight format. A Mat points at one or more uploaded row-group records
// (the engine row-concatenates fused matrices, e.g. ffn_gate + ffn_up, into one buffer of several
// X3Records) plus a heap X3Desc listing the parts; every kernel choice is made by the codec table
// (kernels/formats/x3/dispatch.cu). The codec entry points run in f32 (matching llama-paw), so
// matvec casts x -> f32 and C -> bf16 around them.
#include "weight_format.hpp"

#include "gguf.hpp"
#include "x3_codecs.hpp"
#include "x3_record.hpp"
#include "int8_gemm.cuh"
#include "x3.cuh"
#include "window.hpp"

#include <cublas_v2.h>
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
    // routed-expert bank (every part the same rows): one XwExpert per part on the device, the bank's codecs by slot
    x3::XwExpert* bank = nullptr;
    std::vector<const x3::X3Codec*> codecs;
};

struct Scratch {
    std::mutex mtx;
    float* a = nullptr;   size_t a_cap = 0;
    float* c = nullptr;   size_t c_cap = 0;
    half*  had = nullptr; size_t had_cap = 0;
    int*   locks = nullptr; size_t locks_cap = 0;
    void*  win = nullptr;   size_t win_cap = 0;   // window GEMV scratch (A + partial sums), sized at load
    half*  moe_a = nullptr; size_t moe_a_cap = 0;  // routed-expert lists: A per slot, partial sums, one list per codec
    float* moe_p = nullptr; size_t moe_p_cap = 0;
    x3::XwMoeList* moe_l = nullptr; size_t moe_l_cap = 0;
    x3::XwMoeList* moe_pl = nullptr; size_t moe_pl_cap = 0;  // prompt path: the batched (expert, pass) lists and
    half*  pm_a = nullptr; size_t pm_a_cap = 0;                // their own A / partial sums (decode graphs hold moe_a / moe_p)
    float* pm_p = nullptr; size_t pm_p_cap = 0;
    half*  moe_s = nullptr; size_t moe_s_cap = 0;      // verify blocks: staged down inputs, part outputs, slot map
    __nv_bfloat16* moe_y = nullptr; size_t moe_y_cap = 0;
    int*   moe_i = nullptr; size_t moe_i_cap = 0;
    half*  gw = nullptr; size_t gw_cap = 0;   // prompt path: one expert in FP16, its gathered inputs, FP32 outputs
    half*  gx = nullptr; size_t gx_cap = 0;
    float* gy = nullptr; size_t gy_cap = 0;
    cublasHandle_t blas = nullptr;
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
        // one upload per tensor: every record's body at a 256-byte offset inside one host blob (a device allocation per
        // part rounded each of the ~22K expert parts of a MiniMax pack up, GiBs in all)
        struct Body { const x3::X3Codec* codec; size_t src, bytes, dst; uint32_t rows; };
        std::vector<Body> bodies;
        size_t pos = 0, blob = 0;
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
            bodies.push_back({codec, pos + sizeof(X3Record), tb + sb + vb, blob, rec.rows});
            blob += (tb + sb + vb + 255) / 256 * 256;
            total += int(rec.rows);
            pos += sizeof(X3Record) + tb + sb + vb;
        }
        std::vector<char> host(blob);
        for (const Body& b : bodies) std::memcpy(host.data() + b.dst, raw.data() + b.src, b.bytes);
        const char* dev = static_cast<const char*>(up(host.data(), host.size()));
        if (!dev) throw std::runtime_error("x3: upload failed");
        for (const Body& b : bodies) {
            const size_t tb = size_t(b.rows / 16) * size_t(cols / 16) * b.codec->words * 2, sb = size_t(cols) * 2;
            X3Part p;
            p.codec = b.codec;
            p.trellis = reinterpret_cast<const uint16_t*>(dev + b.dst);
            p.suh = reinterpret_cast<const half*>(dev + b.dst + tb);
            p.svh = reinterpret_cast<const half*>(dev + b.dst + tb + sb);
            p.rows = int(b.rows); p.cols = cols;
            d->parts.push_back(p);
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
            build_bank(*d, sc);
        }

        // dense matrices take the int8 prefill path: their row scales at load, as before (built on the first prefill they
        // cost the first request 27 %); expert banks never take it and skip them
        if (!d->bank) ensure_scales(*d, stream);
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

    // Row scales for the int8 prefill GEMM: the engine's own quantise kernel (int8_quant_rows) on each reconstructed part,
    // keeping only the scales - one source of the amax/127 rule. Dense matrices build them at load; an expert bank only
    // if dequant_i8 is ever called on it (prefill, never inside a graph capture).
    static void ensure_scales(X3Desc& d, cudaStream_t stream) {
        if (d.sc) return;
        const int rows = d.rows, cols = d.cols;
        {
            if (cudaMalloc(&d.sc, size_t(rows) * 4) != cudaSuccess) throw std::runtime_error("x3: row scale allocation failed");
            void* tmp_h = nullptr;
            size_t tmp_h_cap = 0;
            void* tmp_q = nullptr;
            size_t tmp_q_cap = 0;
            int off = 0;
            for (const X3Part& p : d.parts) {
                grow(&tmp_h, tmp_h_cap, size_t(p.rows) * cols * sizeof(half));
                grow(&tmp_q, tmp_q_cap, size_t(p.rows) * cols);
                p.codec->dequant_rows(static_cast<half*>(tmp_h), cols, p.trellis, p.suh, p.svh, p.rows, p.cols, stream);
                tqllm_cuda::int8_quant_rows(static_cast<const __half*>(tmp_h), p.rows, cols, cols,
                                            static_cast<int8_t*>(tmp_q), d.sc + off, stream);
                off += p.rows;
            }
            if (tmp_h) cudaFree(tmp_h);
            if (tmp_q) cudaFree(tmp_q);
        }
    }

    bool has_expert_matvec() const override { return true; }

    // y [count][rows], count = n * topk routed slots (ids [n][topk]); x [n][cols], or [count][cols] when
    // per_slot_input (the down projection reads each slot's own activation).
    void expert_matvec(const Mat& m, const __half* x, __nv_bfloat16* y, int rows, const int* ids, int n, int topk,
                       bool per_slot_input, cudaStream_t s) const override {
        const auto* d = static_cast<const X3Desc*>(m.aux);
        if (!d->bank || rows != d->parts.front().rows) throw std::runtime_error("x3: not a routed-expert bank");
        const int count = n * topk, kp = x3::xw_moe_kparts(d->cols);
        Scratch& sc = scratch();
        std::lock_guard<std::mutex> lock(sc.mtx);
        if (n == 1) {  // decode: one part per routed slot
            x3::x3_moe_build(ids, count, topk, per_slot_input, d->bank, rows, d->cols, kp, int(d->codecs.size()), x,
                             sc.moe_a, sc.moe_p, sc.moe_l, s);
            for (size_t c = 0; c < d->codecs.size(); ++c)
                d->codecs[c]->moe(sc.moe_l + c, count, 1, rows, d->cols, kp, y, 0, s);
            return;
        }
        // verify block: each distinct expert read once for all n columns (bit-identical per column to n = 1)
        int* slot = sc.moe_i;
        int* part_expert = sc.moe_i + x3::XW_MOE_MAX_PARTS;
        int* nparts = sc.moe_i + 2 * x3::XW_MOE_MAX_PARTS;
        const int ldy = count * rows;
        x3::x3_moe_build_dedup(ids, n, topk, d->bank, rows, d->cols, kp, int(d->codecs.size()), x,
                               per_slot_input ? sc.moe_s : nullptr, sc.moe_a, sc.moe_p, slot, part_expert, nparts, sc.moe_l, s);
        if (per_slot_input) x3::x3_moe_stage(x, slot, n, topk, d->cols, nparts, sc.moe_s, s);
        for (size_t c = 0; c < d->codecs.size(); ++c)
            d->codecs[c]->moe(sc.moe_l + c, count, n, rows, d->cols, kp, sc.moe_y, ldy, s);
        x3::x3_moe_unslot(sc.moe_y, slot, count, topk, rows, ldy, y, s);
    }

    // A prompt chunk, grouped by expert (counts [experts], groups [experts][capacity] of assignments): each expert
    // with tokens is reconstructed to FP16 once (dequant_rows) and multiplied by cuBLAS for all its tokens; the
    // decode step is the trellis matvec, the prompt is GEMM-bound.
    bool expert_gemm(const Mat& m, const __half* x, __nv_bfloat16* y, int rows, const int* counts, const int* groups,
                     int capacity, int experts, int topk, bool per_slot_input, cudaStream_t s) const override {
        const auto* d = static_cast<const X3Desc*>(m.aux);
        if (!d->bank || rows != d->parts.front().rows || experts != int(d->parts.size())) return false;
        std::vector<int> cnt(static_cast<size_t>(experts));
        if (cudaMemcpyAsync(cnt.data(), counts, cnt.size() * sizeof(int), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaStreamSynchronize(s) != cudaSuccess)
            throw std::runtime_error("x3: expert counts copy failed");
        const int most = *std::max_element(cnt.begin(), cnt.end());
        Scratch& sc = scratch();
        std::lock_guard<std::mutex> lock(sc.mtx);
        // window GEMV in batched lists: one part per (expert, pass of XW_MAX_N of its assignments), up to
        // XW_MOE_MAX_PARTS parts per launch, inputs read and outputs written through the assignment ids (one launch
        // set for many experts instead of gather + three launches + scatter per expert pass); the trellis is decoded in
        // registers, nothing is written back. reconstruct + cuBLAS remains for codecs without a window kernel.
        if (window_ok(*d) && rows % x3::XW_PMOE_COLS == 0 && std::all_of(d->codecs.begin(), d->codecs.end(), [](const x3::X3Codec* c) { return c->pmoe; })) {
            constexpr int N = x3::XW_MAX_N;
            // 16 parts per list: at 64 the lists' partial sums (12-25 MB) evicted the weights from L2 between an
            // expert's passes (GB10 8.1K prefill 310 vs 256 tok/s)
            constexpr int MP = 16;
            const int kp = x3::xw_moe_kparts(d->cols);
            half* A = static_cast<half*>(grow(reinterpret_cast<void**>(&sc.pm_a), sc.pm_a_cap, size_t(MP) * N * d->cols * sizeof(half)));
            float* P = static_cast<float*>(grow(reinterpret_cast<void**>(&sc.pm_p), sc.pm_p_cap, size_t(MP) * kp * N * rows * sizeof(float)));
            std::vector<x3::XwMoeList> lists;
            std::vector<int> list_codec;
            std::vector<int> open(d->codecs.size(), -1);  // the codec's list still taking parts
            for (int e = 0; e < experts; ++e) {
                const X3Part& p = d->parts[size_t(e)];
                const int ci = int(std::find(d->codecs.begin(), d->codecs.end(), p.codec) - d->codecs.begin());
                for (int t0 = 0; t0 < cnt[size_t(e)]; t0 += N) {
                    if (open[size_t(ci)] < 0 || lists[size_t(open[size_t(ci)])].count == MP) {
                        open[size_t(ci)] = int(lists.size());
                        lists.emplace_back();
                        x3::XwMoeList& L = lists.back();
                        L.count = 0; L.cols = d->cols; L.n = N; L.items = 0;
                        list_codec.push_back(ci);
                    }
                    x3::XwMoeList& L = lists[size_t(open[size_t(ci)])];
                    const int u = L.count++;
                    x3::XwPart& q = L.p[u];
                    q.B = reinterpret_cast<const uint32_t*>(p.trellis); q.suh = p.suh; q.svh = p.svh;
                    q.rows = rows; q.kparts = kp; q.item0 = L.items; q.out_off = 0; q.x = nullptr;
                    q.A = A + size_t(u) * N * d->cols;
                    q.P = P + size_t(u) * kp * N * rows;
                    q.gsrc = groups + size_t(e) * capacity + t0;
                    q.nv = std::min(N, cnt[size_t(e)] - t0);
                    q.sdiv = per_slot_input ? 1 : topk;
                    L.items += (rows / x3::XW_PMOE_COLS) * kp;
                }
            }
            if (lists.empty()) return true;
            auto* dl = static_cast<x3::XwMoeList*>(grow(reinterpret_cast<void**>(&sc.moe_pl), sc.moe_pl_cap, lists.size() * sizeof(x3::XwMoeList)));
            if (cudaMemcpyAsync(dl, lists.data(), lists.size() * sizeof(x3::XwMoeList), cudaMemcpyHostToDevice, s) != cudaSuccess)
                throw std::runtime_error("x3: prompt expert list upload failed");
            // the lists share the A / P scratch: launches in stream order, each list's outputs written before the next
            for (size_t i = 0; i < lists.size(); ++i)
                d->codecs[size_t(list_codec[i])]->pmoe(dl + i, lists[i].count, rows, d->cols, kp, x, y, s);
            return true;
        }
        half* w = nullptr;  // reconstruct path only (below)
        half* gx = static_cast<half*>(grow(reinterpret_cast<void**>(&sc.gx), sc.gx_cap, size_t(std::max(1, most)) * d->cols * sizeof(half)));
        float* gy = static_cast<float*>(grow(reinterpret_cast<void**>(&sc.gy), sc.gy_cap, size_t(std::max(1, most)) * rows * sizeof(float)));
        if (!sc.blas && cublasCreate(&sc.blas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("x3: cuBLAS init failed");
        cublasSetStream(sc.blas, s);
        const float one = 1.f, zero = 0.f;
        for (int e = 0; e < experts; ++e) {
            if (!cnt[size_t(e)]) continue;
            const X3Part& p = d->parts[size_t(e)];
            const int* group = groups + size_t(e) * capacity;
            if (!w) w = static_cast<half*>(grow(reinterpret_cast<void**>(&sc.gw), sc.gw_cap, size_t(rows) * d->cols * sizeof(half)));
            p.codec->dequant_rows(w, d->cols, p.trellis, p.suh, p.svh, rows, d->cols, s);
            x3::x3_moe_gather(x, group, cnt[size_t(e)], topk, per_slot_input, d->cols, gx, s);
            // row-major y_e [count][rows] = x_e [count][cols] . w [rows][cols]^T
            if (cublasGemmEx(sc.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, cnt[size_t(e)], d->cols, &one, w, CUDA_R_16F, d->cols,
                             gx, CUDA_R_16F, d->cols, &zero, gy, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("x3: expert GEMM failed");
            x3::x3_moe_scatter(gy, group, cnt[size_t(e)], rows, y, s);
        }
        return true;
    }

private:
    // A bank when there are several equal parts that all have a routed-expert kernel; sizes the shared MoE scratch.
    static void build_bank(X3Desc& d, Scratch& sc) {
        if (d.parts.size() < 2) return;
        std::vector<x3::XwExpert> host;
        for (const X3Part& p : d.parts) {
            if (p.rows != d.parts.front().rows || !p.codec->moe) return;
            auto it = std::find(d.codecs.begin(), d.codecs.end(), p.codec);
            if (it == d.codecs.end()) it = d.codecs.insert(d.codecs.end(), p.codec);
            host.push_back({reinterpret_cast<const uint32_t*>(p.trellis), p.suh, p.svh, int(it - d.codecs.begin())});
        }
        void* dev = nullptr;
        if (cudaMalloc(&dev, host.size() * sizeof(x3::XwExpert)) != cudaSuccess ||
            cudaMemcpy(dev, host.data(), host.size() * sizeof(x3::XwExpert), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("x3: expert bank upload failed");
        d.bank = static_cast<x3::XwExpert*>(dev);
        // parts x columns: a decode part has one column, a verify part (distinct expert) up to n; parts x n <= MAX x MAX_N / 4
        const size_t slots = x3::XW_MOE_MAX_PARTS, cols_n = slots * x3::XW_MAX_N, rows = size_t(d.parts.front().rows);
        grow(reinterpret_cast<void**>(&sc.moe_a), sc.moe_a_cap, cols_n * d.cols * sizeof(half));
        grow(reinterpret_cast<void**>(&sc.moe_p), sc.moe_p_cap, cols_n * x3::xw_moe_kparts(d.cols) * rows * sizeof(float));
        grow(reinterpret_cast<void**>(&sc.moe_l), sc.moe_l_cap, d.codecs.size() * sizeof(x3::XwMoeList));
        grow(reinterpret_cast<void**>(&sc.moe_s), sc.moe_s_cap, cols_n * d.cols * sizeof(half));
        grow(reinterpret_cast<void**>(&sc.moe_y), sc.moe_y_cap, size_t(x3::XW_MAX_N) * slots * rows * sizeof(__nv_bfloat16));
        grow(reinterpret_cast<void**>(&sc.moe_i), sc.moe_i_cap, 3 * slots * sizeof(int));
    }

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
        auto* d = static_cast<X3Desc*>(const_cast<void*>(m.aux));
        ensure_scales(*d, s);
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
