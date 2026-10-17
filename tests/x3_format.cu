// G2: X3 weight format. For each codec table row: dequant() then a host GEMV of the dequantised
// matrix must match gemv() (dequant + host GEMV < 2e-3 RMS); the codec seam is proven by running
// the test-only scalar codec through the same reconstruct/gemv skeletons. sq (n=1) and x3v (n=2..8)
// are both exercised. No llama-paw needed here (that bit-match lives in the porting notes).
#include "x3.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tqllm;

namespace {

uint32_t g_state = 12345u;
uint32_t lcg() { g_state = g_state * 1664525u + 1013904223u; return g_state; }
float rnd() { return (float(int32_t(lcg() >> 8)) / float(1 << 23)) * 0.25f; }  // ~[-0.25, 0.25]

bool g_ok = true;
void check(bool c, const std::string& what) {
    if (!c) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); g_ok = false; }
}

template <typename T>
T* dev_up(const std::vector<T>& h) {
    T* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) { std::fprintf(stderr, "cudaMalloc failed\n"); std::exit(2); }
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::string fname(const char* n, int nv) { return std::string(n) + "=" + std::to_string(nv); }

void col_identity(const x3::X3Codec* c, int rows, int cols) {
    const int tiles = (rows / 16) * (cols / 16);
    std::vector<uint16_t> trellis(size_t(tiles) * c->words);
    for (auto& v : trellis) v = uint16_t(lcg() >> 13);
    std::vector<half> suh(cols, __float2half(1.f)), svh(rows, __float2half(1.f));
    uint16_t* d_t = dev_up(trellis); half* d_suh = dev_up(suh); half* d_svh = dev_up(svh);
    const int nmax = 8;
    std::vector<half> x(size_t(cols) * nmax);
    for (auto& v : x) v = __float2half(rnd());
    auto run = [&](int n) {
        std::vector<float> A(size_t(n) * cols);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < cols; ++i) A[size_t(j) * cols + i] = __half2float(x[size_t(i) * nmax + j]);
        float* dA = dev_up(A); float* dC = nullptr; cudaMalloc(&dC, size_t(n) * rows * 4);
        half* dH = nullptr; cudaMalloc(&dH, size_t(n) * cols * 2);
        std::vector<float> C(size_t(n) * rows);
        c->gemv(dA, d_t, dC, n, cols, rows, d_suh, dH, d_svh, nullptr, 0);
        cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(dA); cudaFree(dC); cudaFree(dH);
        return C;
    };
    const std::vector<float> C8 = run(nmax);
    bool eq = true;
    for (int n = 1; n < nmax && eq; ++n) {
        const std::vector<float> Cn = run(n);
        for (int j = 0; j < n && eq; ++j)
            for (int r = 0; r < rows && eq; ++r)
                if (Cn[size_t(j) * rows + r] != C8[size_t(j) * rows + r]) eq = false;
    }
    check(eq, std::string(c->name) + ": column bit-identical n=1..8 (rows=" + std::to_string(rows) + " cols=" + std::to_string(cols) + ")");
    cudaFree(d_t); cudaFree(d_suh); cudaFree(d_svh);
}

// dequant_rows (out-major, row stride ld) must hold exactly dequant's values transposed: W_rows[r*ld + c] ==
// W[c*rows + r], bit for bit, and must not touch the ld padding.
void rows_identity(const x3::X3Codec* c, int rows, int cols, int ld) {
    std::vector<uint16_t> trellis(size_t(rows / 16) * (cols / 16) * c->words);
    for (auto& v : trellis) v = uint16_t(lcg() >> 13);
    std::vector<half> suh(cols), svh(rows);
    for (auto& v : suh) v = __float2half(0.5f + rnd());
    for (auto& v : svh) v = __float2half(0.5f + rnd());
    uint16_t* d_t = dev_up(trellis);
    half* d_suh = dev_up(suh);
    half* d_svh = dev_up(svh);
    half *d_W = nullptr, *d_R = nullptr;
    cudaMalloc(&d_W, size_t(rows) * cols * sizeof(half));
    cudaMalloc(&d_R, size_t(rows) * ld * sizeof(half));
    cudaMemset(d_R, 0x7b, size_t(rows) * ld * sizeof(half));   // padding sentinel
    c->dequant(d_W, d_t, d_suh, d_svh, rows, cols, 0);
    c->dequant_rows(d_R, ld, d_t, d_suh, d_svh, rows, cols, 0);
    std::vector<uint16_t> W(size_t(rows) * cols), R(size_t(rows) * ld);
    cudaMemcpy(W.data(), d_W, W.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(R.data(), d_R, R.size() * 2, cudaMemcpyDeviceToHost);
    bool eq = true;
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < ld; ++k)
            eq &= R[size_t(r) * ld + k] == (k < cols ? W[size_t(k) * rows + r] : uint16_t(0x7b7b));
    check(eq, std::string(c->name) + ": dequant_rows == dequant transposed (rows=" + std::to_string(rows) +
              " cols=" + std::to_string(cols) + " ld=" + std::to_string(ld) + ")");
    cudaFree(d_t); cudaFree(d_suh); cudaFree(d_svh); cudaFree(d_W); cudaFree(d_R);
}

void test_row(const x3::X3Codec* c, int rows, int cols) {
    const int tiles = (rows / 16) * (cols / 16);
    const size_t trellis_words = size_t(tiles) * c->words;

    std::vector<uint16_t> trellis(trellis_words);
    for (auto& v : trellis) v = uint16_t(lcg() >> 13);
    std::vector<half> suh(cols), svh(rows);
    for (auto& v : suh) v = __float2half(0.5f + rnd());
    for (auto& v : svh) v = __float2half(0.5f + rnd());

    uint16_t* d_t = dev_up(trellis);
    half* d_suh = dev_up(suh);
    half* d_svh = dev_up(svh);

    // dequant -> W [rows][cols]
    std::vector<half> W(size_t(rows) * cols);
    half* d_W = nullptr;
    cudaMalloc(&d_W, W.size() * sizeof(half));
    c->dequant(d_W, d_t, d_suh, d_svh, rows, cols, 0);
    cudaMemcpy(W.data(), d_W, W.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // dequant must produce finite values.
    bool finite = true;
    for (half h : W) { const float f = __half2float(h); if (!std::isfinite(f)) finite = false; }
    check(finite, std::string(c->name) + std::string(": dequant finite"));

    for (int n : {1, 4, 7}) {
        if (n > 1 && !c->has_x3v) continue;
        std::vector<half> x(size_t(cols) * n);
        for (auto& v : x) v = __float2half(rnd());
        std::vector<float> A(size_t(n) * cols);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < cols; ++i) A[size_t(j) * cols + i] = __half2float(x[size_t(i) * n + j]);
        float* d_A = dev_up(A);
        float* d_C = nullptr;
        cudaMalloc(&d_C, size_t(n) * rows * sizeof(float));
        half* d_had = nullptr;
        cudaMalloc(&d_had, size_t(n) * cols * sizeof(half));
        int* d_locks = nullptr;
        const size_t lb = c->ws_bytes(cols, rows);
        if (lb) { cudaMalloc(&d_locks, lb); cudaMemset(d_locks, 0, lb); }

        c->gemv(d_A, d_t, d_C, n, cols, rows, d_suh, d_had, d_svh, d_locks, 0);
        cudaError_t err = cudaDeviceSynchronize();
        check(err == cudaSuccess, std::string(c->name) + ": gemv launch (" + cudaGetErrorString(err) + ")");

        std::vector<float> C(size_t(n) * rows);
        cudaMemcpy(C.data(), d_C, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Host reference in the fork's convention (tests/test-paw-x3-frac.cpp decode_what): the
        // reconstructed W is [in][out] (row = in, col = out) and y = x . W, reduction over `in`.
        // So for a stored W[rows][cols] with rows=out, cols=in the element (in,out) is W[in*cols+out].
        double se = 0.0, den = 1e-12;
        for (int j = 0; j < n; ++j)
            for (int r = 0; r < rows; ++r) {
                double acc = 0.0;
                for (int c2 = 0; c2 < cols; ++c2)
                    acc += double(__half2float(x[size_t(c2) * n + j])) * double(__half2float(W[size_t(c2) * cols + r]));
                const double diff = C[size_t(j) * rows + r] - acc;
                se += diff * diff; den += acc * acc;
            }
        const double rel = std::sqrt(se / double(size_t(n) * rows)) / (std::sqrt(den / double(size_t(n) * rows)) + 1e-12);
        // n == 1 runs the int8 sq path (activations quantised to int8), so it is ~2% off the exact
        // GEMV by construction; its true reference is the Phase R sq implementation. n >= 2 is the
        // exact fp16 x3v path and must match to 2e-3.
        const double thr = (n == 1) ? 5e-2 : 2e-3;
        check(rel < thr, std::string(c->name) + ": matvec vs dequant+host GEMV n=" + std::to_string(n) +
                         " (rel=" + std::to_string(rel) + ")");
        if (rel >= thr) std::fprintf(stderr, "  %s rows=%d cols=%d n=%d rel=%.3e\n", c->name, rows, cols, n, rel);

        cudaFree(d_A); cudaFree(d_C); cudaFree(d_had);
        if (d_locks) cudaFree(d_locks);
    }

    // Width independence: the x3v path processes batch columns independently, so column j of an
    // n=7 call must equal column j of an n=4 call on the same input.
    if (c->has_x3v) {
        const int n7 = 7, n4 = 4;
        std::vector<half> x(size_t(cols) * n7);
        for (auto& v : x) v = __float2half(rnd());
        auto run = [&](int n) {
            std::vector<float> A(size_t(n) * cols);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < cols; ++i) A[size_t(j) * cols + i] = __half2float(x[size_t(i) * n7 + j]);
            float* dA = dev_up(A);
            float* dC = nullptr; cudaMalloc(&dC, size_t(n) * rows * 4);
            half* dH = nullptr; cudaMalloc(&dH, size_t(n) * cols * 2);
            std::vector<float> C(size_t(n) * rows);
            c->gemv(dA, d_t, dC, n, cols, rows, d_suh, dH, d_svh, nullptr, 0);
            cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost);
            cudaFree(dA); cudaFree(dC); cudaFree(dH);
            return C;
        };
        const std::vector<float> C7 = run(n7), C4 = run(n4);
        bool eq = true; double mx = 0.0;
        for (int j = 0; j < n4; ++j)
            for (int r = 0; r < rows; ++r) {
                const double d = std::fabs(double(C7[size_t(j) * rows + r]) - double(C4[size_t(j) * rows + r]));
                if (C7[size_t(j) * rows + r] != C4[size_t(j) * rows + r]) eq = false;
                if (d > mx) mx = d;
            }
        check(eq, std::string(c->name) + ": width independence n=4 col == n=7 col (maxdiff=" + std::to_string(mx) + ")");
    }

    // Kernel contract: column j of an X3 matvec is bit-identical for every n in 1..8, so decode
    // (n=1) matches the batched verify token-for-token.
    if (c->has_x3v) {
        const int nmax = 8;
        std::vector<half> x(size_t(cols) * nmax);
        for (auto& v : x) v = __float2half(rnd());
        auto run = [&](int n) {
            std::vector<float> A(size_t(n) * cols);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < cols; ++i) A[size_t(j) * cols + i] = __half2float(x[size_t(i) * nmax + j]);
            float* dA = dev_up(A);
            float* dC = nullptr; cudaMalloc(&dC, size_t(n) * rows * 4);
            half* dH = nullptr; cudaMalloc(&dH, size_t(n) * cols * 2);
            std::vector<float> C(size_t(n) * rows);
            c->gemv(dA, d_t, dC, n, cols, rows, d_suh, dH, d_svh, nullptr, 0);
            cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost);
            cudaFree(dA); cudaFree(dC); cudaFree(dH);
            return C;
        };
        const std::vector<float> C8 = run(nmax);
        bool eq = true;
        for (int n = 1; n < nmax && eq; ++n) {
            const std::vector<float> Cn = run(n);
            for (int j = 0; j < n && eq; ++j)
                for (int r = 0; r < rows && eq; ++r)
                    if (Cn[size_t(j) * rows + r] != C8[size_t(j) * rows + r]) eq = false;
        }
        check(eq, std::string(c->name) + ": column bit-identical across n=1..8");
    }

    cudaFree(d_t); cudaFree(d_suh); cudaFree(d_svh); cudaFree(d_W);
}

}  // namespace

int main() {
    int count = 0;
    const x3::X3Codec* table = x3::x3_codec_table(count);
    check(count >= 8, "codec table has the mul1 K rows + testscalar");
    for (int i = 0; i < count; ++i) {
        test_row(&table[i], 256, 256);
        rows_identity(&table[i], 256, 384, 392);
        rows_identity(&table[i], 384, 256, 256);
        if (table[i].has_x3v) col_identity(&table[i], 8448, 5120);  // size_n>8192 -> CFG1
    }
    if (g_ok) std::printf("x3_format: OK\n");
    return g_ok ? 0 : 1;
}
