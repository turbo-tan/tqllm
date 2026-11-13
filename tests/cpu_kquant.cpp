// CPU K-quant expert matvec (src/experts/cpu_kquant.hpp): the AVX2 path equals the portable one, both stay within the
// int8-activation error of an fp32 dot on dequantize_host's values, for Q2_K / Q3_K / Q4_K; speed on an expert-sized matrix.
#include "../src/experts/cpu_kquant.hpp"
#include "dequant.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
static int failures = 0;
int main() {
    std::mt19937 rng(7);
    const int rows = 64, cols = 4096;
    for (uint32_t type : {10u, 11u, 12u}) {
        const int64_t rb = tqllm::cpu::kq_row_bytes(type, cols);
        std::vector<uint8_t> W(size_t(rb * rows));
        for (auto & b : W) b = uint8_t(rng());
        const int bb = int(rb / (cols / 256));
        for (int64_t blk = 0; blk < int64_t(W.size()) / bb; ++blk) {  // sane fp16 scales (random bytes can be inf / nan)
            uint8_t * p = W.data() + blk * bb;
            const uint16_t d = uint16_t(0x1400 + rng() % 0x0800), m = uint16_t(0x1000 + rng() % 0x0800);
            const int off = type == 10 ? 80 : type == 11 ? 108 : 0;
            std::memcpy(p + off, &d, 2);
            if (type != 11) std::memcpy(p + off + 2, &m, 2);
        }
        std::vector<float> x(cols), wrow(cols);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : x) v = nd(rng);
        tqllm::cpu::Xq xq;
        tqllm::cpu::quantize_x(x.data(), cols, xq);
        std::vector<float> fast(rows), ref(rows);
        tqllm::cpu::kq_matvec(type, W.data(), cols, xq, 0, rows, fast.data());
        tqllm::cpu::kq_matvec_scalar(type, W.data(), cols, xq, 0, rows, ref.data());
        double worst = 0, scale = 0;
        for (int r = 0; r < rows; ++r) {
            tqllm::dequantize_host(type, W.data() + r * rb, wrow.data(), cols);
            double exact = 0;
            for (int c = 0; c < cols; ++c) exact += double(wrow[c]) * x[c];
            worst = std::max(worst, std::fabs(double(fast[r]) - exact));
            scale = std::max(scale, std::fabs(exact));
            if (std::fabs(fast[r] - ref[r]) > 1e-5f * std::max(1.f, std::fabs(ref[r]))) { std::fprintf(stderr, "FAIL type %u row %d: avx2 %g vs scalar %g\n", type, r, fast[r], ref[r]); ++failures; break; }
        }
        const double rel = worst / std::max(scale, 1e-9);
        std::printf("type %u: max |kq - fp32 dot| %.3g of %.3g (%.2f %%)\n", type, worst, scale, 100 * rel);
        if (rel > 0.02) { std::fprintf(stderr, "FAIL type %u: error %.3f %% over 2 %%\n", type, 100 * rel); ++failures; }
    }
    {  // speed: one GLM expert gate matrix, 2048 x 4096 Q2_K, single thread
        const int R = 2048, C = 4096;
        std::vector<uint8_t> W(size_t(tqllm::cpu::kq_row_bytes(10, C) * R), 0x55);
        std::vector<float> x(C, 0.5f), y(R);
        tqllm::cpu::Xq xq;
        tqllm::cpu::quantize_x(x.data(), C, xq);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) tqllm::cpu::kq_matvec(10, W.data(), C, xq, 0, R, y.data());
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
        std::printf("Q2_K 2048x4096, one thread: %.2f ms (%.1f GB/s)\n", s * 1e3, double(W.size()) / s / 1e9);
    }
    if (failures) return 1;
    std::puts("cpu_kquant: ok");
    return 0;
}
