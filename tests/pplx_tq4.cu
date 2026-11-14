// The fast path's TQ4 (GGUF 300) decoder against a CPU reference: w = (L[code] * scale) R per 128, R = H diag(s1) / sqrt(128)
// (Sylvester Hadamard, Turbo4 signs), the layout tools/tq4_to_gguf.py writes. Max |err| <= 1 % of max |w| (BF16 output).
#include "pplx_embed.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const int groups = 64;
    const float L[16] = {6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};
    const uint32_t S[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};
    std::vector<uint8_t> blob(size_t(groups) * 66);
    std::vector<double> ref(size_t(groups) * 128);
    uint32_t st = 7;
    auto rnd = [&] { st = st * 1664525u + 1013904223u; return st >> 8; };
    for (int g = 0; g < groups; ++g) {
        const float sc = 0.001f + float(rnd() % 1000) * 1e-5f;
        const __half hs = __float2half_rn(sc);
        std::memcpy(&blob[size_t(g) * 66], &hs, 2);
        const double scd = double(__half2float(hs));
        std::vector<double> wr(128);
        for (int b = 0; b < 64; ++b) {
            const uint8_t byte = uint8_t(rnd());
            blob[size_t(g) * 66 + 2 + b] = byte;
            wr[2 * b] = L[byte & 15] * scd;
            wr[2 * b + 1] = L[byte >> 4] * scd;
        }
        for (int j = 0; j < 128; ++j) {
            double acc = 0;
            for (int i = 0; i < 128; ++i) acc += wr[i] * ((__builtin_popcount(unsigned(i & j)) & 1) ? -1.0 : 1.0);
            const double s = (S[j / 32] >> (j % 32) & 1) ? -1.0 : 1.0;
            ref[size_t(g) * 128 + j] = acc * s / std::sqrt(128.0);
        }
    }
    void * dw = nullptr, * dout = nullptr;
    cudaMalloc(&dw, blob.size());
    cudaMalloc(&dout, ref.size() * 2);
    cudaMemcpy(dw, blob.data(), blob.size(), cudaMemcpyHostToDevice);
    tqllm::pplx_tq4_to_bf16(dw, groups, dout);
    std::vector<uint16_t> got(ref.size());
    const bool ran = cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess;
    double err = 0, mx = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const uint32_t u = uint32_t(got[i]) << 16;
        float f; std::memcpy(&f, &u, 4);
        err = std::max(err, std::fabs(f - ref[i]));
        mx = std::max(mx, std::fabs(ref[i]));
    }
    const bool ok = ran && err <= 0.01 * mx;
    std::printf("tq4 decode: max |err| %.3g of max |w| %.3g: %s\n", err, mx, ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
