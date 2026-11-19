// TQT weight format (src/formats/weight/tqt.cpp / tqt.cu, docs/reference/formats.md) through the WeightFormat registry:
// 1. GGUF: type 306 + tqllm.tqt.code = "tqt-c2-v1" reads as GGML_TQT; another tag is refused; 306 without the tag is
//    refused.
// 2. dequant_h == the CPU reference (dequantize_host_rows) within 1 FP16 ulp, ragged row counts included.
// 3. matvec (1..16 columns) against a double dot over the CPU reference; every kernel variant (row tiles, prefetch,
//    persistent grid) and every column count give identical bits per column; matvec_f32 == matvec (BF16-rounded);
//    matvec_ld writes the same values into a wider row layout and leaves the gap alone.
// 4. dequant vs matvec: the FP16 dequant times x (double) agrees with the matvec (the prefill GEMM and the decode kernel
//    compute the same product).
// 5. dequant_i8 == dequant_h + int8_quant_rows (operand and row scales, bitwise); dequant_rows_h == a slice of dequant_h.
#include "dequant.hpp"
#include "gguf.hpp"
#include "int8_gemm.cuh"
#include "quant_kernels.cuh"
#include "weight_format.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string & what, double v = 0) {
    std::printf("%-78s %.4g %s\n", what.c_str(), v, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}
std::vector<void *> allocs;
template <class T> T * dev(const std::vector<T> & h) {
    void * p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T));
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    allocs.push_back(p);
    return static_cast<T *>(p);
}
template <class T> T * dalloc(size_t n) {
    void * p = nullptr;
    cudaMalloc(&p, n * sizeof(T));
    allocs.push_back(p);
    return static_cast<T *>(p);
}
template <class T> std::vector<T> host(const T * d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
// random TQT file blocks: FP16 scale (some blocks 20x larger), random stream bytes
std::vector<char> make_blocks(int rows, int k, std::mt19937 & rng) {
    std::vector<char> raw(size_t(rows) * k / 128 * 66);
    std::uniform_real_distribution<float> du(0.0002f, 0.0012f);
    for (size_t b = 0; b < raw.size() / 66; ++b) {
        const uint16_t h = __half_as_ushort(__float2half(du(rng) * (b % 37 == 5 ? 20.f : 1.f)));
        std::memcpy(&raw[b * 66], &h, 2);
        for (int i = 2; i < 66; ++i) raw[b * 66 + i] = char(rng());
    }
    return raw;
}
int ulps(__half a, __half b) {
    const uint16_t x = __half_as_ushort(a), y = __half_as_ushort(b);
    auto ord = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7fff) : int(v); };
    return std::abs(ord(x) - ord(y));
}

// minimal GGUF v3 with one string key and one 128 x 2 tensor of type `type`
std::string write_gguf(const std::string & path, const char * key, const char * val, uint32_t type, const std::vector<char> & data) {
    std::ofstream o(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { o.write(reinterpret_cast<const char *>(&v), 4); };
    auto u64 = [&](uint64_t v) { o.write(reinterpret_cast<const char *>(&v), 8); };
    auto str = [&](const std::string & s) { u64(s.size()); o.write(s.data(), std::streamsize(s.size())); };
    u32(0x46554747u); u32(3); u64(1); u64(key ? 1 : 0);
    if (key) { str(key); u32(8); str(val); }
    str("w"); u32(2); u64(128); u64(2); u32(type); u64(0);
    while (o.tellp() % 32) o.put(0);
    o.write(data.data(), std::streamsize(data.size()));
    return path;
}

// --bench: matvec timing at the Qwen3.8-27B decode shapes, every TQT variant vs the TQ4 matvec (engine rule), same
// bytes. Median over 40 samples of 10 back-to-back launches (event tick ~1 us). Needs the GPU to itself (GPU lock).
int bench() {
    struct S { int rows, k; const char * what; };
    const S shapes[] = {{34816, 5120, "ffn gate+up"}, {5120, 17408, "ffn down"}, {16384, 5120, "gdn qkv+z"},
                        {14336, 5120, "attn q+k+v"}, {5120, 6144, "gdn out / attn o"}};
    size_t maxe = 0;
    for (const S & s : shapes) maxe = std::max(maxe, size_t(s.rows) * s.k);
    std::mt19937 rng(3);
    std::vector<uint8_t> w(maxe / 2);
    for (auto & b : w) b = uint8_t(rng());
    std::vector<__half> sc(maxe / 128), x(size_t(16) * 17408);
    std::uniform_real_distribution<float> du(0.0002f, 0.0012f);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : sc) v = __float2half(du(rng));
    for (auto & v : x) v = __float2half(nd(rng));
    auto * dw = dev(w); auto * ds = dev(sc); auto * dx = dev(x);
    auto * dy = dalloc<float>(size_t(16) * 34816);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time_us = [&](auto && f) {
        for (int i = 0; i < 20; ++i) f();
        std::vector<float> v;
        for (int it = 0; it < 40; ++it) {
            cudaEventRecord(e0);
            for (int r = 0; r < 10; ++r) f();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1); v.push_back(ms * 100.f);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    for (const S & s : shapes)
        for (const int nc : {1, 2, 4, 8, 16}) {
            const float t4 = time_us([&] { tqllm_cuda::tq4_matvec(dw, ds, dx, dy, s.rows, s.k, nc, nullptr); });
            std::printf("%-17s %5d x %5d nc %2d | TQ4 %6.1f us (%4.0f GB/s) | TQT", s.what, s.rows, s.k, nc, t4,
                        double(s.rows) * s.k * 66 / 128 / t4 / 1e3);
            for (int v = 0; v <= 8; ++v) {
                const float t = time_us([&] { tqllm_cuda::tqt_matvec_variant(dw, ds, dx, dy, s.rows, s.k, nc, v, nullptr); });
                std::printf(" v%d %6.1f", v, t);
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    for (void * p : allocs) cudaFree(p);
    return 0;
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") return bench();
    std::mt19937 rng(7);
    // 1. GGUF type mapping
    {
        const std::string dir = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
        std::vector<char> data(2 * 66, 0);
        const auto a = tqllm::gguf_open(write_gguf(dir + "/tqt_a.gguf", "tqllm.tqt.code", "tqt-c2-v1", 306, data));
        check(a.tensors.at("w").type == tqllm::GGML_TQT && a.tensors.at("w").bytes == 132, "gguf: 306 + tqt-c2-v1 -> TQT, 132 bytes");
        bool refused = false;
        try { tqllm::gguf_open(write_gguf(dir + "/tqt_b.gguf", "tqllm.tqt.code", "tqt-c3-v1", 306, data)); } catch (const std::exception &) { refused = true; }
        check(refused, "gguf: another code tag is refused");
        refused = false;
        try { tqllm::gguf_open(write_gguf(dir + "/tqt_c.gguf", nullptr, nullptr, 306, data)); } catch (const std::exception &) { refused = true; }
        check(refused, "gguf: 306 without the code tag is refused");
        std::remove((dir + "/tqt_a.gguf").c_str()); std::remove((dir + "/tqt_b.gguf").c_str()); std::remove((dir + "/tqt_c.gguf").c_str());
    }
    const tqllm::WeightFormat * fmt = tqllm::weight_format(tqllm::GGML_TQT);
    check(fmt && fmt->basis() == tqllm::Basis::Tq4Rot128 && fmt->max_matvec_cols() == 16, "registry: TQT registered, TQ4 basis, 16 columns");
    if (!fmt) return 1;
    tqllm::Upload up = [](const void * h, size_t n) {
        void * p = nullptr;
        cudaMalloc(&p, n);
        cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);
        allocs.push_back(p);
        return static_cast<const void *>(p);
    };
    struct Shape { int rows, k; };
    for (const Shape sh : {Shape{1000, 5120}, Shape{77, 640}, Shape{520, 17408}}) {
        const int R = sh.rows, K = sh.k;
        const std::string tag = "[" + std::to_string(R) + " x " + std::to_string(K) + "] ";
        const auto raw = make_blocks(R, K, rng);
        std::vector<float> ref(size_t(R) * K);
        tqllm::dequantize_host_rows(tqllm::GGML_TQT, raw.data(), ref.data(), R, K);
        const tqllm::Mat m = fmt->load(raw, R, K, up, nullptr);
        // 2. dequant_h vs CPU reference
        auto * dh = dalloc<__half>(size_t(R) * K);
        fmt->dequant_h(m, dh, nullptr);
        const auto h = host(dh, size_t(R) * K);
        int worst = 0;
        for (size_t i = 0; i < h.size(); ++i) worst = std::max(worst, ulps(h[i], __float2half(ref[i])));
        check(worst <= 1, tag + "dequant_h == CPU reference (worst FP16 ulp)", worst);
        // 3. matvec vs double dot; variants and widths identical
        const int NC = 16;
        std::vector<__half> x(size_t(NC) * K);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : x) v = __float2half(nd(rng));
        auto * dx = dev(x);
        auto * y0 = dalloc<float>(size_t(NC) * R);
        auto * y1 = dalloc<float>(size_t(NC) * R);
        fmt->matvec_f32(m, dx, y0, NC, nullptr);
        const auto base = host(y0, size_t(NC) * R);
        double se = 0, sr = 0;
        for (int n = 0; n < NC; ++n)
            for (int r = 0; r < R; ++r) {
                double d = 0;
                for (int c = 0; c < K; ++c) d += double(ref[size_t(r) * K + c]) * __half2float(x[size_t(n) * K + c]);
                const double e = base[size_t(n) * R + r] - d;
                se += e * e, sr += d * d;
            }
        check(std::sqrt(se / sr) < 1e-5, tag + "matvec_f32 (16 columns) vs double dot over the CPU reference (rel-rms)", std::sqrt(se / sr));
        size_t diff = 0;
        for (int v = 0; v <= 8; ++v)
            for (const int nc : {1, 3, 8, 9, 16}) {
                tqllm_cuda::tqt_matvec_variant(static_cast<const uint8_t *>(m.p), static_cast<const __half *>(m.aux), dx, y1, R, K, nc, v, nullptr);
                const auto got = host(y1, size_t(nc) * R);
                diff += std::memcmp(got.data(), base.data(), got.size() * 4) != 0;
            }
        check(diff == 0, tag + "every variant / column count: identical bits per column (differing runs)", double(diff));
        // matvec (BF16) == BF16(matvec_f32); matvec_ld into a wider layout, gap untouched
        auto * yb = dalloc<__nv_bfloat16>(size_t(NC) * (R + 8));
        cudaMemset(yb, 0x7F, size_t(NC) * (R + 8) * 2);
        fmt->matvec_ld(m, dx, yb, NC, R + 8, nullptr);
        const auto gb = host(yb, size_t(NC) * (R + 8));
        size_t bad = 0;
        for (int n = 0; n < NC; ++n) {
            for (int r = 0; r < R; ++r) bad += __bfloat16_as_ushort(gb[size_t(n) * (R + 8) + r]) != __bfloat16_as_ushort(__float2bfloat16_rn(base[size_t(n) * R + r]));
            for (int r = R; r < R + 8; ++r) bad += __bfloat16_as_ushort(gb[size_t(n) * (R + 8) + r]) != 0x7F7F;
        }
        fmt->matvec(m, dx, yb, 5, nullptr);
        const auto g5 = host(yb, size_t(5) * R);
        for (int n = 0; n < 5; ++n)
            for (int r = 0; r < R; ++r) bad += __bfloat16_as_ushort(g5[size_t(n) * R + r]) != __bfloat16_as_ushort(__float2bfloat16_rn(base[size_t(n) * R + r]));
        check(bad == 0, tag + "matvec / matvec_ld (BF16) == BF16(matvec_f32), ld gap untouched", double(bad));
        // 4. dequant (FP16 weights) x activations vs matvec
        se = sr = 0;
        for (int n = 0; n < NC; n += 5)
            for (int r = 0; r < R; ++r) {
                double d = 0;
                for (int c = 0; c < K; ++c) d += double(__half2float(h[size_t(r) * K + c])) * __half2float(x[size_t(n) * K + c]);
                const double e = base[size_t(n) * R + r] - d;
                se += e * e, sr += d * d;
            }
        check(std::sqrt(se / sr) < 1e-3, tag + "dequant_h product vs matvec (rel-rms)", std::sqrt(se / sr));
        // native prefill GEMM (FP16 accumulation per 128-block) vs the double product of the dequant_h weights
        {
            const int NG = 70;
            std::vector<__half> xg(size_t(NG) * K);
            for (auto & v : xg) v = __float2half(nd(rng));
            auto * dxg = dev(xg);
            auto * yg = dalloc<__nv_bfloat16>(size_t(NG) * R);
            const bool ran = fmt->has_gemm() && fmt->gemm(m, dxg, yg, NG, nullptr);
            const auto gy = host(yg, size_t(NG) * R);
            double ge = 0, gr = 0;
            for (int n = 0; n < NG; n += 3)
                for (int r = 0; r < R; ++r) {
                    double d = 0;
                    for (int c = 0; c < K; ++c) d += double(__half2float(h[size_t(r) * K + c])) * __half2float(xg[size_t(n) * K + c]);
                    const double e = __bfloat162float(gy[size_t(n) * R + r]) - d;
                    ge += e * e, gr += d * d;
                }
            check(ran && std::sqrt(ge / gr) < 5e-3, tag + "gemm (70 columns) vs double product of dequant_h (rel-rms)", std::sqrt(ge / gr));
        }
        // 5. dequant_i8 == dequant_h + int8_quant_rows; dequant_rows_h == slice
        if (tqllm_cuda::Int8Gemm::supported()) {
            auto * qa = dalloc<int8_t>(size_t(R) * K);
            auto * qb = dalloc<int8_t>(size_t(R) * K);
            auto * sa = dalloc<float>(R);
            auto * sb = dalloc<float>(R);
            tqllm_cuda::int8_quant_rows(dh, R, K, K, qa, sa, nullptr);
            check(fmt->has_dequant_i8() && fmt->dequant_i8(m, qb, sb, nullptr), tag + "dequant_i8 available");
            const auto A = host(qa, size_t(R) * K), B = host(qb, size_t(R) * K);
            const auto SA = host(sa, R), SB = host(sb, R);
            check(A == B && std::memcmp(SA.data(), SB.data(), R * 4) == 0, tag + "dequant_i8 == dequant_h + int8_quant_rows (operand, row scales)");
        }
        const int r0 = R / 3, nr = R / 2;
        auto * ds = dalloc<__half>(size_t(nr) * K);
        fmt->dequant_rows_h(m, ds, r0, nr, nullptr);
        const auto hs = host(ds, size_t(nr) * K);
        check(std::memcmp(hs.data(), h.data() + size_t(r0) * K, hs.size() * 2) == 0, tag + "dequant_rows_h == rows of dequant_h");
    }
    const cudaError_t e = cudaDeviceSynchronize();
    check(e == cudaSuccess, std::string("CUDA: ") + cudaGetErrorString(e));
    for (void * p : allocs) cudaFree(p);
    std::printf("TQT_FORMAT %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
