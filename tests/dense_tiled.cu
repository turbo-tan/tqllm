// tile_dense: a Q6_K / Q5_K matrix rewritten as 16-row x 4-block tiles must give the untiled matrix's bits in every path the
// engines call (matvec_f32 1..64 columns, matvec / matvec_ld BF16, gemm_f32 / gemm above 64 columns, dequant_h), also on a
// row-gathered shape (rows not a multiple of 64) and at the LM-head height; formats without tiles stay unchanged.
#include "formats/weight/weight_format.hpp"
#include "core/gguf.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using namespace tqllm;
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
template <class T> static std::vector<T> get(const T * d, size_t n) { std::vector<T> h(n); check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

int main() {
    int fails = 0;
    struct Case { uint32_t type; int rows, cols; };
    for (const Case cs : {Case{14, 10240, 2560}, Case{14, 2560, 6144}, Case{14, 528, 2560}, Case{13, 6144, 2560}, Case{13, 2560, 6144}, Case{14, 81920, 2560}}) {
        const auto * f = weight_format(cs.type);
        std::mt19937 rng(cs.type * 7 + cs.rows); std::uniform_int_distribution<int> bd(0, 0x3B); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<char> raw(gguf_tensor_bytes(cs.type, {uint64_t(cs.cols), uint64_t(cs.rows)}, "w"));
        for (auto & c : raw) c = char(bd(rng));
        std::vector<void *> keep;
        Upload up = [&](const void * h, size_t n) { void * p; check(cudaMalloc(&p, n)); check(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice)); keep.push_back(p); return (const void *) p; };
        const Mat plain = f->load(raw, cs.rows, cs.cols, up, nullptr);
        Mat tiled = f->load(raw, cs.rows, cs.cols, up, nullptr);
        if (!tile_dense(tiled)) { std::printf("type %u %dx%d: not tiled  FAIL\n", cs.type, cs.rows, cs.cols); ++fails; continue; }
        constexpr int maxn = 256;
        std::vector<__half> xh(size_t(cs.cols) * maxn); for (auto & v : xh) v = __float2half(nd(rng));
        __half * dx; check(cudaMalloc(&dx, xh.size() * 2)); check(cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
        const size_t ny = size_t(cs.rows) * maxn;
        float * y1, * y2; check(cudaMalloc(&y1, ny * 4)); check(cudaMalloc(&y2, ny * 4));
        __nv_bfloat16 * b1, * b2; check(cudaMalloc(&b1, (ny + cs.rows) * 2)); check(cudaMalloc(&b2, (ny + cs.rows) * 2));
        bool ok = true; std::string bad;
        auto same_f = [&](size_t n, const char * what) {
            const auto a = get(y1, n), b = get(y2, n);
            if (std::memcmp(a.data(), b.data(), n * 4)) { ok = false; bad += std::string(" ") + what; }
        };
        auto same_b = [&](size_t n, const char * what) {
            const auto a = get(b1, n), b = get(b2, n);
            if (std::memcmp(a.data(), b.data(), n * 2)) { ok = false; bad += std::string(" ") + what; }
        };
        for (int n : {1, 2, 3, 5, 6, 8, 12, 16, 24, 32, 48, 64}) {
            if (n > plain.fmt->max_exact_cols() && n > 8) continue;
            check(cudaMemset(y1, 0, ny * 4)); check(cudaMemset(y2, 0xff, ny * 4));
            plain.fmt->matvec_f32(plain, dx, y1, n, nullptr); tiled.fmt->matvec_f32(tiled, dx, y2, n, nullptr); check(cudaDeviceSynchronize());
            same_f(size_t(n) * cs.rows, ("matvec_f32 n=" + std::to_string(n)).c_str());
        }
        for (int n : {1, 4, 8}) {
            check(cudaMemset(b1, 0, ny * 2)); check(cudaMemset(b2, 0xff, ny * 2));
            plain.fmt->matvec(plain, dx, b1, n, nullptr); tiled.fmt->matvec(tiled, dx, b2, n, nullptr); check(cudaDeviceSynchronize());
            same_b(size_t(n) * cs.rows, ("matvec n=" + std::to_string(n)).c_str());
            const int ld = cs.rows + 16;
            check(cudaMemset(b1, 0, ny * 2)); check(cudaMemset(b2, 0, ny * 2));
            plain.fmt->matvec_ld(plain, dx, b1, n, ld, nullptr); tiled.fmt->matvec_ld(tiled, dx, b2, n, ld, nullptr); check(cudaDeviceSynchronize());
            same_b(size_t(n) * ld, ("matvec_ld n=" + std::to_string(n)).c_str());
        }
        if (cs.rows <= 16384) {
            for (int n : {65, 128, 200}) {
                check(cudaMemset(y1, 0, ny * 4)); check(cudaMemset(y2, 0xff, ny * 4));
                const bool g1 = plain.fmt->gemm_f32(plain, dx, y1, n, nullptr), g2 = tiled.fmt->gemm_f32(tiled, dx, y2, n, nullptr); check(cudaDeviceSynchronize());
                if (g1 != g2) { ok = false; bad += " gemm_f32 support"; }
                if (g1 && g2) same_f(size_t(n) * cs.rows, ("gemm_f32 n=" + std::to_string(n)).c_str());
                check(cudaMemset(b1, 0, ny * 2)); check(cudaMemset(b2, 0xff, ny * 2));
                const bool h1 = plain.fmt->gemm(plain, dx, b1, n, nullptr), h2 = tiled.fmt->gemm(tiled, dx, b2, n, nullptr); check(cudaDeviceSynchronize());
                if (h1 != h2) { ok = false; bad += " gemm support"; }
                if (h1 && h2) same_b(size_t(n) * cs.rows, ("gemm n=" + std::to_string(n)).c_str());
            }
            __half * d1, * d2; const size_t nd2 = size_t(cs.rows) * cs.cols;
            check(cudaMalloc(&d1, nd2 * 2)); check(cudaMalloc(&d2, nd2 * 2));
            plain.fmt->dequant_h(plain, d1, nullptr); tiled.fmt->dequant_h(tiled, d2, nullptr); check(cudaDeviceSynchronize());
            if (get(d1, nd2) != get(d2, nd2)) { ok = false; bad += " dequant_h"; }
            cudaFree(d1); cudaFree(d2);
        }
        std::printf("%-6s -> %-4s %6d x %-5d: %s%s\n", f->name(), tiled.fmt->name(), cs.rows, cs.cols, ok ? "same" : "DIFF", bad.c_str());
        fails += !ok;
        for (void * p : keep) cudaFree(p);
        cudaFree(dx); cudaFree(y1); cudaFree(y2); cudaFree(b1); cudaFree(b2);
    }
    {   // a format without tiles, and a shape without whole tiles: the Mat stays as it was
        std::vector<void *> keep;
        Upload up = [&](const void * h, size_t n) { void * p; check(cudaMalloc(&p, n)); check(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice)); keep.push_back(p); return (const void *) p; };
        std::vector<char> raw4(gguf_tensor_bytes(12, {2560, 640}, "w"), char(1)), raw6(gguf_tensor_bytes(14, {2560, 520}, "w"), char(1));
        Mat a = weight_format(12)->load(raw4, 640, 2560, up, nullptr), b = weight_format(14)->load(raw6, 520, 2560, up, nullptr);
        const auto * fa = a.fmt, * fb = b.fmt;
        const bool ta = tile_dense(a), tb = tile_dense(b);
        const bool ok = !ta && !tb && a.fmt == fa && b.fmt == fb;
        std::printf("untileable (Q4_K, 520-row Q6_K) kept: %s\n", ok ? "yes" : "NO");
        fails += !ok;
        for (void * p : keep) cudaFree(p);
    }
    std::printf("dense_tiled: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
