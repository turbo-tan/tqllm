// Prefill GEMM on a real matrix of a GGUF (NAX affine or TQ4): today's path (FP16 weight copy + MLX GEMM) and the MLX GEMM
// alone on a pre-dequantized copy (its ceiling). Median ms and TFLOPS per variant, relative error vs today's path. Also: MLX quantized_matmul on the same values in MLX's
// 4-bit layout and the GEMM on a K-major copy. usage: bench-mlx-prefill-gemm model.gguf tensor [rows=2048]
#include "nax_affine.hpp"
#include "tq4.hpp"
#include "gguf.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

namespace {
double median_ms(const std::function<mx::array()>& run) {
    for (int w = 0; w < 3; ++w) mx::eval(run());
    std::vector<double> t;
    for (int r = 0; r < 10; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        mx::eval(run());
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) throw std::invalid_argument("usage: bench-mlx-prefill-gemm model.gguf tensor [rows=2048]");
        const int m = argc == 4 ? std::stoi(argv[3]) : 2048;
        auto file = tqllm::gguf_open(argv[1]);
        const auto& t = file.tensors.at(argv[2]);
        auto raw = file.read(t);
        const int rows = int(t.ne[1]), cols = int(t.ne[0]);
        NaxAffine w = [&] {
            std::span<const uint8_t> payload(reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
            if (t.type == tqllm::GGML_NAX_A4_G64 || t.type == tqllm::GGML_NAX_A4_G128)
                return nax_affine_load(payload, rows, cols, 4, t.type == tqllm::GGML_NAX_A4_G128 ? 128 : 64);
            if (t.type == tqllm::GGML_TQ4) return nax_from_tq4(load_tq4_blocks(payload, rows, cols));
            throw std::invalid_argument("tensor is neither 4-bit NAX affine nor TQ4");
        }();
        mx::eval(w.words, w.scales, w.biases);
        auto x = mx::astype(mx::random::normal({m, cols}), mx::float16);
        auto dense = nax_affine_dequant(w);
        mx::eval(x, dense);
        const double flops = 2.0 * double(m) * rows * cols;
        auto report = [&](const char* name, double ms, double rel) {
            std::printf("PREFILL_GEMM %-34s %s m=%d n=%d k=%d %8.3f ms %6.2f TFLOPS rel_err %.2e\n", name, argv[2], m, rows, cols, ms,
                        flops / ms / 1e9, rel);
        };
        auto reference = nax_affine_gemm(w, x, true);
        mx::eval(reference);
        auto rel = [&](const mx::array& y) {
            auto d = mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(y, reference))), mx::sum(mx::square(reference))));
            mx::eval(d);
            return double(d.item<float>());
        };
        report("today (dequant + MLX GEMM)", median_ms([&] {return nax_affine_gemm(w, x, true);}), 0.0);
        report("MLX GEMM on a resident FP16 copy", median_ms([&] {return mx::matmul(x, mx::transpose(dense));}),
               rel(mx::astype(mx::astype(mx::astype(mx::matmul(x, mx::transpose(dense)), mx::float32), mx::bfloat16), mx::float32)));
        // the same values in MLX's own 4-bit layout (what Splash runs) and a K-major FP16 copy of the weights
        auto q = mx::quantize(dense, 64, 4);
        auto kmajor = mx::contiguous(mx::transpose(dense));
        mx::eval(q[0], q[1], q[2], kmajor);
        report("MLX quantized_matmul (MLX layout)", median_ms([&] {return mx::quantized_matmul(x, q[0], q[1], q[2], true, 64, 4);}),
               rel(mx::astype(mx::quantized_matmul(x, q[0], q[1], q[2], true, 64, 4), mx::float32)));
        report("MLX GEMM on a K-major FP16 copy", median_ms([&] {return mx::matmul(x, kmajor);}), rel(mx::astype(mx::matmul(x, kmajor), mx::float32)));
        if (w.bits == 4 && !w.q6) {
            // the staged prefill kernel on the stored codes (no FP16 weight copy), per tile: RS rows x TN columns, SGS simdgroups
            // best configs found on M5 Pro: gate / up / packs (wide N) like 32x64 sg 4 tb 1; down (wide K) like
            // 32x64 sg 8 tb 0. Both pf 0 and pf 1 kept: pf 1 helps the wide-N shapes, pf 0 the wide-K ones.
            for (auto [rs, tn, sgs, tb] : {std::array{32, 64, 4, 1}, std::array{32, 64, 4, 0}, std::array{32, 64, 8, 0},
                                           std::array{32, 64, 8, 1}, std::array{16, 64, 8, 0}, std::array{16, 64, 8, 1}})
                    for (int pf = 0; pf < 2; ++pf) for (int rf = 0; rf < 2; ++rf) {
                        char name[64];
                        std::snprintf(name, sizeof name, "nax prefill %dx%d sg %d tb %d pf %d rf %d", rs, tn, sgs, tb, pf, rf);
                        auto run = [&] {return nax_affine_prefill(w, x, 0, std::nullopt, rs, tn, sgs, tb, pf, rf);};
                        report(name, median_ms(run), rel(mx::astype(mx::astype(mx::astype(run(), mx::float32), mx::bfloat16), mx::float32)));
                    }
            // the fused epilogues on the shape's best config: EP 1 SwiGLU (up pass: silu(gate) * acc), EP 2 residual
            // (aux + acc). The aux operand is a resident FP16 [m, N] here; its cost is the same as the plain GEMM's.
            auto aux = mx::astype(mx::random::normal({m, rows}), mx::float16);
            mx::eval(aux);
            const bool wideN = rows >= cols;
            const int brs = 32, btn = 64, bsgs = 8, btb = wideN ? 1 : 0, bpf = wideN ? 1 : 0;
            char en[80];
            std::snprintf(en, sizeof en, "nax prefill %dx%d sg %d tb %d pf %d EP1 swiglu", brs, btn, bsgs, btb, bpf);
            report(en, median_ms([&] {return nax_affine_prefill(w, x, 1, aux, brs, btn, bsgs, btb, bpf);}), -1.0);
            std::snprintf(en, sizeof en, "nax prefill %dx%d sg %d tb %d pf %d EP2 residual", brs, btn, bsgs, btb, bpf);
            report(en, median_ms([&] {return nax_affine_prefill(w, x, 2, aux, brs, btn, bsgs, btb, bpf);}), -1.0);
        }
        if (w.bits == 4 && !w.tq4 && !w.q6) {
            // the stored codes repacked per call (no requantization), then quantized_matmul
            auto stored = nax_affine_to_mlx(w);
            mx::eval(stored.codes, stored.scales, stored.biases);
            report("repack stored codes + quantized_matmul", median_ms([&] {return nax_affine_qmm(nax_affine_to_mlx(w), x);}),
                   rel(mx::astype(mx::astype(mx::astype(nax_affine_qmm(stored, x), mx::float32), mx::bfloat16), mx::float32)));
            report("quantized_matmul on repacked codes", median_ms([&] {return nax_affine_qmm(stored, x);}), -1.0);
            report("repack alone", median_ms([&] {return nax_affine_to_mlx(w).codes;}), -1.0);
        }
    } catch (const std::exception& e) {std::fprintf(stderr, "BENCH_ERROR %s\n", e.what()); return 1;}
}
