// Accuracy + speed test for tqllm::mlx_backend::AneProgram (src/backends/mlx/ane_program.mm): our own private-API W8A8
// Neural Engine FFN program, driven from the CPU. Accuracy: random weights at the real shape (C 6656, hidden 5120,
// rows 2048) against an FP32 reference of the same FP16 slice (rel L2 + max abs; int8-level ~1e-2 expected, fail on
// NaN/inf or > 5e-2). Speed: median of 20 evaluations at C in {4096, 6656, 8192}, rows 2048, as ms and TOPS
// (2 * rows * hidden * C * 3 / s). Prints MLX_ANE_PROGRAM_PASS when it ran.
#import <Foundation/Foundation.h>

#include "ane_program.hpp"
#include "nax_affine.hpp"
#include "tq4.hpp"
#include <mlx/mlx.h>
#include <mlx/random.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mx = mlx::core;
using tqllm::mlx_backend::AneProgram;
using tqllm::mlx_backend::NaxAffine;
using tqllm::mlx_backend::nax_affine_pack;
using tqllm::mlx_backend::nax_affine_load;
using tqllm::mlx_backend::nax_affine_dequant;
using Clock = std::chrono::steady_clock;

namespace {

// FP32 reference FFN slice from the exact FP16 weights/input the ANE sees: y = down(silu(x Wg^T) * (x Wu^T)).
// wg,wu: [C, H], wd: [H, C], x: [rows, H], all float16 MLX arrays. Returns [rows, H] float32.
mx::array referenceFfn(const mx::array& x, const mx::array& wg, const mx::array& wu, const mx::array& wd) {
    auto xf = mx::astype(x, mx::float32);
    auto g = mx::matmul(xf, mx::transpose(mx::astype(wg, mx::float32)));   // [rows, C]
    auto u = mx::matmul(xf, mx::transpose(mx::astype(wu, mx::float32)));   // [rows, C]
    auto silu = mx::multiply(g, mx::sigmoid(g));
    auto h = mx::multiply(silu, u);                                        // [rows, C]
    return mx::matmul(h, mx::transpose(mx::astype(wd, mx::float32)));      // [rows, H]
}

double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// ---- step 2 helpers ----
uint16_t bf16_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
// A random 4-bit affine matrix [rows, columns] (group 64) built like tests/mlx_nax_affine.cpp: random codes, small
// positive scales and small biases, loaded through nax_affine_pack / nax_affine_load. The matrix the ANE share reads.
NaxAffine randomNax(int rows, int columns, std::mt19937& rng) {
    const int bits = 4, group = 64, gpr = columns / group;
    std::vector<uint8_t> codes(size_t(rows) * columns);
    for (auto& c : codes) c = uint8_t(rng() & 15u);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f), bd(-0.05f, 0.05f);
    std::vector<uint16_t> s(size_t(rows) * gpr), b(s.size());
    for (size_t i = 0; i < s.size(); ++i) { s[i] = bf16_bits(sd(rng)); b[i] = bf16_bits(bd(rng)); }
    return nax_affine_load(nax_affine_pack(codes, s, b, rows, columns, bits, group), rows, columns, bits, group);
}

// Copy an MLX array to a host fp16 vector (contiguous).
std::vector<uint16_t> toHalfVec(const mx::array& a) {
    auto c = mx::contiguous(mx::astype(a, mx::float16));
    mx::eval(c);
    return std::vector<uint16_t>(reinterpret_cast<const uint16_t*>(c.data<mx::float16_t>()),
                                 reinterpret_cast<const uint16_t*>(c.data<mx::float16_t>()) + c.size());
}

}  // namespace

int main() {
    @autoreleasepool {
        try {
            const int H = 5120, rows = 2048;
            // ===================== requantize cost, pipelined (no per-call eval round trip) =====================
            if (std::getenv("TQLLM_ANE_TEST_REQUANT_TIMING")) {
                for (int C : {6656, 13824}) {
                    std::mt19937 rng(7);
                    auto wgN = randomNax(17408, H, rng), wuN = randomNax(17408, H, rng), wdN = randomNax(H, 17408, rng);
                    auto prog = AneProgram::open(C, rows, H, 1);
                    prog->reserve_slots(2);
                    { auto w = prog->requantize_weights_gpu(0, wgN, wuN, wdN, C); mx::eval(w); }
                    const int reps = 20;
                    auto t = Clock::now();
                    std::vector<mx::array> keep;
                    for (int i = 0; i < reps; ++i) { auto w = prog->requantize_weights_gpu(i & 1, wgN, wuN, wdN, C); mx::async_eval(w); keep.push_back(w); }
                    mx::eval(keep.back());
                    std::printf("MLX_ANE_PROGRAM_REQUANT_PIPELINED C=%d ms_per_layer=%.3f\n", C, msSince(t) / reps);
                }
                // pack and join, pipelined, at the slice sizes the engine uses
                {
                    auto prog = AneProgram::open(6656, rows, H, 1);
                    prog->reserve_slots(2);
                    for (int n : {1024, 2048}) {
                        auto xn = mx::astype(mx::random::normal({n, H}), mx::float16);
                        mx::eval(xn);
                        { auto tk = prog->pack_input_gpu(xn); mx::eval(tk); auto yj = prog->join_output_gpu(tk); mx::eval(yj); }
                        const int reps = 30;
                        auto t = Clock::now();
                        std::vector<mx::array> keep;
                        for (int i = 0; i < reps; ++i) { auto tk = prog->pack_input_gpu(xn, i & 1); mx::async_eval(tk); keep.push_back(tk); }
                        mx::eval(keep.back());
                        const double packMs = msSince(t) / reps;
                        t = Clock::now();
                        std::vector<mx::array> outs;
                        for (int i = 0; i < reps; ++i) { auto yj = prog->join_output_gpu(keep[i], i & 1); mx::async_eval(yj); outs.push_back(yj); }
                        mx::eval(outs.back());
                        std::printf("MLX_ANE_PROGRAM_PACKJOIN_PIPELINED n=%d pack_ms=%.3f join_ms=%.3f\n", n, packMs, msSince(t) / reps);
                    }
                }
                return 0;   // timing mode only
            }

            mx::random::seed(20261020);

            // -------- accuracy at the real shape --------
            {
                const int C = 6656;
                auto wg = mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float16);
                auto wu = mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float16);
                auto wd = mx::astype(mx::multiply(mx::random::normal({H, C}), mx::array(0.02f)), mx::float16);
                auto x = mx::astype(mx::random::normal({rows, H}), mx::float16);
                mx::eval(wg, wu, wd, x);

                auto ref = mx::contiguous(referenceFfn(x, wg, wu, wd));
                mx::eval(ref);

                auto prog = AneProgram::open(C, rows, H, 1);
                std::printf("MLX_ANE_PROGRAM built C=%d rows=%d hidden=%d\n", C, rows, H);
                prog->bind_weights(0, wg.data<mx::float16_t>(), wu.data<mx::float16_t>(), wd.data<mx::float16_t>());

                std::vector<uint16_t> y(size_t(rows) * H);
                prog->evaluate(0, x.data<mx::float16_t>(), y.data());

                // rel L2 and max abs against the FP32 reference.
                const float* r = ref.data<float>();
                double num = 0, den = 0, maxAbs = 0;
                bool finite = true;
                for (size_t i = 0; i < size_t(rows) * H; ++i) {
                    const float got = float(std::bit_cast<_Float16>(y[i]));
                    if (!std::isfinite(got)) finite = false;
                    const double d = double(got) - double(r[i]);
                    num += d * d;
                    den += double(r[i]) * double(r[i]);
                    maxAbs = std::max(maxAbs, std::fabs(d));
                }
                const double relL2 = std::sqrt(num / std::max(den, 1e-30));
                std::printf("MLX_ANE_PROGRAM_ACCURACY C=%d rows=%d rel_l2=%.4e max_abs=%.4e finite=%d\n", C, rows,
                            relL2, maxAbs, finite ? 1 : 0);
                if (!finite) throw std::runtime_error("ANE output not finite");
                if (relL2 > 5e-2) throw std::runtime_error("ANE rel L2 error above 5e-2");
                if (relL2 > 1e-2)
                    std::printf("MLX_ANE_PROGRAM_NOTE rel L2 %.4e above the ~1e-2 int8 target (not failing)\n", relL2);
            }

            // -------- outlier (massive-activation) channels: before/after the input rotation --------
            // The fix for the end-to-end KL spike (j842): Qwen3.8 has massive-activation channels in the FFN input on
            // some tokens. Our per-token int8 pack with one scale and no rotation lets such a channel take the whole
            // scale, so the rest of that row loses its precision. Build an input with 3 channels at 500-2000 on 10% of
            // the tokens (the rest N(0,1)), and measure the ANE share's rel L2 and per-row worst error against an FP32
            // reference, with the rotation off (ane_program_set_input_rotate) and on (default). The rotation must lower
            // both; the plain random-input check above must not regress (it ran with the rotation on).
            {
                const int C = 6656;
                mx::random::seed(20261021);
                auto wg = mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float16);
                auto wu = mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float16);
                auto wd = mx::astype(mx::multiply(mx::random::normal({H, C}), mx::array(0.02f)), mx::float16);
                mx::eval(wg, wu, wd);

                // base N(0,1) input, then inject massive-activation channels on 10% of the tokens.
                std::vector<float> xf(size_t(rows) * H);
                {
                    auto base = mx::random::normal({rows, H});
                    auto bc = mx::contiguous(mx::astype(base, mx::float32));
                    mx::eval(bc);
                    std::memcpy(xf.data(), bc.data<float>(), xf.size() * sizeof(float));
                }
                std::mt19937 orng(20261022);
                const int outChans[3] = {37, 1900, 4211};   // three fixed channels get the massive activations
                std::uniform_real_distribution<float> mag(500.0f, 2000.0f);
                int hotTokens = 0;
                for (int t = 0; t < rows; ++t) {
                    if ((orng() % 10u) != 0u) continue;       // ~10% of tokens
                    ++hotTokens;
                    for (int ci = 0; ci < 3; ++ci) {
                        const float sgn = (orng() & 1u) ? 1.0f : -1.0f;
                        xf[size_t(t) * H + outChans[ci]] = sgn * mag(orng);
                    }
                }
                auto x = mx::astype(mx::array(xf.data(), {rows, H}, mx::float32), mx::float16);
                mx::eval(x);
                auto ref = mx::contiguous(referenceFfn(x, wg, wu, wd));
                mx::eval(ref);
                const float* r = ref.data<float>();

                const auto measure = [&](bool rotate, double& relL2, double& worstRow, int& finite) {
                    tqllm::mlx_backend::ane_program_set_input_rotate(rotate);
                    
                    auto prog = AneProgram::open(C, rows, H, 1);
                    prog->bind_weights(0, wg.data<mx::float16_t>(), wu.data<mx::float16_t>(), wd.data<mx::float16_t>());
                    std::vector<uint16_t> y(size_t(rows) * H);
                    prog->evaluate(0, x.data<mx::float16_t>(), y.data());
                    double num = 0, den = 0, worst = 0;
                    finite = 1;
                    for (int t = 0; t < rows; ++t) {
                        double rn = 0, rd = 0;
                        for (int ch = 0; ch < H; ++ch) {
                            const size_t i = size_t(t) * H + ch;
                            const float got = float(std::bit_cast<_Float16>(y[i]));
                            if (!std::isfinite(got)) finite = 0;
                            const double d = double(got) - double(r[i]);
                            rn += d * d; rd += double(r[i]) * double(r[i]);
                        }
                        num += rn; den += rd;
                        worst = std::max(worst, std::sqrt(rn / std::max(rd, 1e-30)));
                    }
                    relL2 = std::sqrt(num / std::max(den, 1e-30));
                    worstRow = worst;
                };

                double l2Off = 0, wrOff = 0, l2On = 0, wrOn = 0; int finOff = 0, finOn = 0;
                measure(false, l2Off, wrOff, finOff);
                measure(true, l2On, wrOn, finOn);
                tqllm::mlx_backend::ane_program_set_input_rotate(true);
                std::printf("MLX_ANE_PROGRAM_OUTLIER C=%d rows=%d hot_tokens=%d before_rel_l2=%.4e before_worst_row=%.4e "
                            "after_rel_l2=%.4e after_worst_row=%.4e finite_before=%d finite_after=%d\n",
                            C, rows, hotTokens, l2Off, wrOff, l2On, wrOn, finOff, finOn);
                if (!finOn) throw std::runtime_error("rotated outlier output not finite");
                if (l2On >= l2Off)
                    throw std::runtime_error("input rotation did not lower the outlier rel L2");
                if (wrOn >= wrOff)
                    throw std::runtime_error("input rotation did not lower the worst-row outlier error");
            }

            // -------- fp16-overflow finite guard (Splash: "an infinity in the ANE output stops it") --------
            // A hidden channel of ~1e5 (the pattern ane_ffn_test.mm lists) drives the fp16 intermediate h = silu(g)*us
            // past 65504 for some tokens, so the ANE's partial is +-Inf there. The join must keep the chunk's output
            // finite: the guard drops a non-finite token's contribution (0), so the GPU share stands for that row,
            // rather than writing Inf/NaN into the residual (which would poison every later layer, KL -> inf). The
            // finite tokens are unchanged. One big weight row makes the overflow reachable from a normal input.
            {
                const int C = 6656;
                mx::random::seed(20261101);
                std::vector<float> wgf(size_t(C) * H, 0.0f), wuf(size_t(C) * H, 0.0f), wdf(size_t(H) * C, 0.0f);
                {
                    auto g = mx::contiguous(mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float32));
                    auto u = mx::contiguous(mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float32));
                    auto d = mx::contiguous(mx::astype(mx::multiply(mx::random::normal({H, C}), mx::array(0.02f)), mx::float32));
                    mx::eval(g, u, d);
                    std::memcpy(wgf.data(), g.data<float>(), wgf.size() * sizeof(float));
                    std::memcpy(wuf.data(), u.data<float>(), wuf.size() * sizeof(float));
                    std::memcpy(wdf.data(), d.data<float>(), wdf.size() * sizeof(float));
                }
                // one intermediate channel (row 1234 of gate and up) is given large weights on a few hidden inputs, so
                // a token with a ~1e5 activation on those inputs makes gate*up on that channel overflow fp16.
                const int hotChan = 1234, hotIn[2] = {11, 2222};
                for (int ci = 0; ci < 2; ++ci) { wgf[size_t(hotChan) * H + hotIn[ci]] = 4.0f; wuf[size_t(hotChan) * H + hotIn[ci]] = 4.0f; }
                for (int row = 0; row < H; row += 512) wdf[size_t(row) * C + hotChan] = 1.0f;  // channel feeds the output
                auto wg = mx::astype(mx::array(wgf.data(), {C, H}, mx::float32), mx::float16);
                auto wu = mx::astype(mx::array(wuf.data(), {C, H}, mx::float32), mx::float16);
                auto wd = mx::astype(mx::array(wdf.data(), {H, C}, mx::float32), mx::float16);
                mx::eval(wg, wu, wd);
                std::vector<float> xf(size_t(rows) * H, 0.0f);
                { auto base = mx::contiguous(mx::astype(mx::random::normal({rows, H}), mx::float32)); mx::eval(base);
                  std::memcpy(xf.data(), base.data<float>(), xf.size() * sizeof(float)); }
                // tokens 100 and 777 get a ~1e5 hidden channel -> fp16 intermediate overflow on hotChan.
                const int hotTok[2] = {100, 777};
                for (int ti = 0; ti < 2; ++ti) for (int ci = 0; ci < 2; ++ci) xf[size_t(hotTok[ti]) * H + hotIn[ci]] = 1.0e5f;
                auto x = mx::astype(mx::array(xf.data(), {rows, H}, mx::float32), mx::float16);
                mx::eval(x);

                auto prog = AneProgram::open(C, rows, H, 1);
                prog->bind_weights(0, wg.data<mx::float16_t>(), wu.data<mx::float16_t>(), wd.data<mx::float16_t>());
                std::vector<uint16_t> y(size_t(rows) * H);
                prog->evaluate(0, x.data<mx::float16_t>(), y.data());
                int nonFinite = 0, hotRowsZeroed = 0;
                for (size_t i = 0; i < y.size(); ++i) if (!std::isfinite(float(std::bit_cast<_Float16>(y[i])))) ++nonFinite;
                for (int ti = 0; ti < 2; ++ti) {
                    bool allZero = true;
                    for (int ch = 0; ch < H; ++ch) if (float(std::bit_cast<_Float16>(y[size_t(hotTok[ti]) * H + ch])) != 0.0f) { allZero = false; break; }
                    hotRowsZeroed += allZero ? 1 : 0;
                }
                // a quiet token's row must still be the normal (finite, non-zero) partial.
                bool quietOk = false;
                for (int ch = 0; ch < H; ++ch) if (float(std::bit_cast<_Float16>(y[size_t(300) * H + ch])) != 0.0f) { quietOk = true; break; }
                std::printf("MLX_ANE_PROGRAM_OVERFLOW_GUARD C=%d rows=%d nonfinite_out=%d hot_rows_dropped=%d quiet_row_ok=%d\n",
                            C, rows, nonFinite, hotRowsZeroed, quietOk ? 1 : 0);
                if (nonFinite != 0) throw std::runtime_error("the join let a non-finite value into the output");
                if (!quietOk) throw std::runtime_error("the guard zeroed a finite (quiet) row");
            }

            // -------- speed --------
            std::printf("MLX_ANE_PROGRAM_SPEED rows=%d hidden=%d (quiet M5 Pro, j844: ~21.6 ms / ~19.4 TOPS at C=6656, as Splash's)\n",
                        rows, H);
            for (int C : {4096, 6656, 8192}) {
                auto wg = mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float16);
                auto wu = mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float16);
                auto wd = mx::astype(mx::multiply(mx::random::normal({H, C}), mx::array(0.02f)), mx::float16);
                auto x = mx::astype(mx::random::normal({rows, H}), mx::float16);
                mx::eval(wg, wu, wd, x);

                auto prog = AneProgram::open(C, rows, H, 1);
                prog->bind_weights(0, wg.data<mx::float16_t>(), wu.data<mx::float16_t>(), wd.data<mx::float16_t>());
                std::vector<uint16_t> y(size_t(rows) * H);

                prog->evaluate(0, x.data<mx::float16_t>(), y.data());  // warm up
                std::vector<double> whole, eval;
                for (int i = 0; i < 20; ++i) {
                    auto t = Clock::now();
                    prog->evaluate(0, x.data<mx::float16_t>(), y.data());
                    whole.push_back(msSince(t));
                    eval.push_back(prog->last_eval_ms());  // ANE evaluation alone (no CPU pack/join)
                }
                std::sort(whole.begin(), whole.end());
                std::sort(eval.begin(), eval.end());
                const double wholeMs = whole[whole.size() / 2], evalMs = eval[eval.size() / 2];
                const double flops = 2.0 * rows * double(H) * C * 3.0;
                const double wholeTops = flops / (wholeMs * 1e-3) * 1e-12;
                const double evalTops = flops / (evalMs * 1e-3) * 1e-12;
                std::printf("MLX_ANE_PROGRAM_SPEED C=%d ane_eval_ms=%.2f ane_eval_TOPS=%.2f whole_ms=%.2f whole_TOPS=%.2f\n",
                            C, evalMs, evalTops, wholeMs, wholeTops);
            }

            // ===================== step 2: GPU pack / join / requantize + async evaluate =====================
            // Run the whole step-2 path (requantize bit-equality, pack, join, accuracy, timings, overlap) at both ANE
            // shares the engine picks for Qwen3 (C = 6656 and C = 8704; the full intermediate F = 17408).
            for (int C : {6656, 8704, 12800, 13824}) {
                const int F = 17408, G = F - C;  // ANE share C, GPU share G = 17408 - C
                std::mt19937 rng(20261020);

                // random 4-bit affine weights at the real FFN shape, and a random FP16 input on the GPU.
                auto wgN = randomNax(F, H, rng), wuN = randomNax(F, H, rng), wdN = randomNax(H, F, rng);
                auto x16 = mx::astype(mx::random::normal({rows, H}), mx::float16);
                mx::eval(x16);

                // the exact FP16 weights the ANE share reads: dequantize, then the last C rows of gate/up and the last
                // C columns of down. bind_weights (CPU) and requantize_weights_gpu (GPU) quantize these same values.
                auto wgF = nax_affine_dequant(wgN), wuF = nax_affine_dequant(wuN), wdF = nax_affine_dequant(wdN);
                auto wgC = mx::contiguous(mx::slice(wgF, {F - C, 0}, {F, H}));   // [C, H]
                auto wuC = mx::contiguous(mx::slice(wuF, {F - C, 0}, {F, H}));   // [C, H]
                auto wdC = mx::contiguous(mx::slice(wdF, {0, F - C}, {H, F}));   // [H, C]
                mx::eval(wgC, wuC, wdC);
                auto wgH = toHalfVec(wgC), wuH = toHalfVec(wuC), wdH = toHalfVec(wdC);

                auto prog = AneProgram::open(C, rows, H, 1);

                // ---- CPU-path reference on the identical FP16 weights/input (pack+bind+join on the CPU). Capture the
                //      CPU-packed input and CPU-staged weight surfaces for the per-stage bit-equality checks. ----
                prog->bind_weights(0, wgH.data(), wuH.data(), wdH.data());
                auto xH = toHalfVec(x16);
                std::vector<uint16_t> yCpu(size_t(rows) * H);
                prog->evaluate(0, xH.data(), yCpu.data());
                std::vector<uint16_t> txCpu; std::vector<int8_t> x0Cpu; int x0r = 0, x0s = 0;
                prog->debug_copy_pack(txCpu, x0Cpu, x0r, x0s);
                std::vector<uint16_t> sgC, sdC; std::vector<int8_t> wg0C, wd0C; int wgr = 0, wgs = 0, wdr = 0, wds = 0;
                prog->debug_copy_weights(-1, sgC, sdC, wg0C, wd0C, wgr, wgs, wdr, wds);

                // ---- GPU requantize vs CPU bind: bit equality of the staged int8 weights + per-row scales ----
                prog->reserve_slots(2);
                auto wreq = prog->requantize_weights_gpu(0, wgN, wuN, wdN, C);
                mx::eval(wreq);  // stage slot 0's weights before enqueue
                {
                    std::vector<uint16_t> sgG, sdG; std::vector<int8_t> wg0G, wd0G; int a, b, c2, d;
                    prog->debug_copy_weights(0, sgG, sdG, wg0G, wd0G, a, b, c2, d);
                    size_t sgMm = 0, sdMm = 0, wgMm = 0, wdMm = 0;
                    for (size_t i = 0; i < sgG.size(); ++i) sgMm += sgG[i] != sgC[i];
                    for (size_t i = 0; i < sdG.size(); ++i) sdMm += sdG[i] != sdC[i];
                    for (size_t i = 0; i < wg0G.size() && i < wg0C.size(); ++i) wgMm += wg0G[i] != wg0C[i];
                    for (size_t i = 0; i < wd0G.size() && i < wd0C.size(); ++i) wdMm += wd0G[i] != wd0C[i];
                    std::printf("MLX_ANE_PROGRAM_GPU_REQUANT C=%d sg_mm=%zu sd_mm=%zu wg_mm=%zu wd_mm=%zu\n", C, sgMm,
                                sdMm, wgMm, wdMm);
                    if (sgMm || sdMm || wgMm || wdMm)
                        throw std::runtime_error("GPU requantize not bit-equal to CPU bind");
                }

                // ---- GPU pack vs CPU pack: bit equality of the per-token int8 input + per-token scale ----
                auto tok = prog->pack_input_gpu(x16);
                mx::eval(tok);  // pack the input into the surfaces
                {
                    std::vector<uint16_t> txG; std::vector<int8_t> x0G; int gr = 0, gs = 0;
                    prog->debug_copy_pack(txG, x0G, gr, gs);
                    size_t txMm = 0, xMm = 0;
                    for (int t = 0; t < rows; ++t) txMm += txG[t] != txCpu[t];
                    for (size_t i = 0; i < x0G.size() && i < x0Cpu.size(); ++i) xMm += x0G[i] != x0Cpu[i];
                    std::printf("MLX_ANE_PROGRAM_GPU_PACK C=%d tx_mm=%zu x_mm=%zu\n", C, txMm, xMm);
                    if (txMm || xMm) throw std::runtime_error("GPU pack not bit-equal to CPU pack");
                }

                prog->enqueue(0, /*after_pack=*/false);
                prog->wait();
                auto yGpuArr = prog->join_output_gpu(tok);  // [rows, H] fp32
                mx::eval(yGpuArr);
                const float* yGpu = yGpuArr.data<float>();

                // ---- GPU join vs CPU join: the whole path (pack + requantize + join on the GPU, same ANE eval) is
                //      bit-equal to the CPU path, so the join matches too. ----
                size_t mismatches = 0;
                double worstAbs = 0;
                for (size_t i = 0; i < size_t(rows) * H; ++i) {
                    const float cpu = float(std::bit_cast<_Float16>(yCpu[i]));
                    if (float(_Float16(yGpu[i])) != cpu) {
                        ++mismatches;
                        worstAbs = std::max(worstAbs, std::fabs(double(yGpu[i]) - double(cpu)));
                    }
                }
                std::printf("MLX_ANE_PROGRAM_GPU_JOIN C=%d rows=%d mismatches=%zu worst_abs=%.3e\n", C, rows,
                            mismatches, worstAbs);
                if (mismatches != 0)
                    throw std::runtime_error("GPU join (whole path) not bit-equal to CPU (" +
                                             std::to_string(mismatches) + " mismatches)");

                // ---- end-to-end accuracy vs an FP32 reference of the same FP16 share ----
                auto ref = mx::contiguous(referenceFfn(x16, wgC, wuC, wdC));
                mx::eval(ref);
                const float* r = ref.data<float>();
                double num = 0, den = 0, maxAbs = 0;
                bool finite = true;
                for (size_t i = 0; i < size_t(rows) * H; ++i) {
                    if (!std::isfinite(yGpu[i])) finite = false;
                    const double d = double(yGpu[i]) - double(r[i]);
                    num += d * d;
                    den += double(r[i]) * double(r[i]);
                    maxAbs = std::max(maxAbs, std::fabs(d));
                }
                const double relL2 = std::sqrt(num / std::max(den, 1e-30));
                std::printf("MLX_ANE_PROGRAM_GPU_ACCURACY C=%d rows=%d rel_l2=%.4e max_abs=%.4e finite=%d\n", C, rows,
                            relL2, maxAbs, finite ? 1 : 0);
                if (!finite) throw std::runtime_error("GPU-path ANE output not finite");
                if (relL2 > 5e-2) throw std::runtime_error("GPU-path rel L2 error above 5e-2");

                // ---- step-3 pipelined handoff, as the engine runs it with TQLLM_MLX_ANE_HANDOFF: four slices alternating
                // two inputs and two surface parities; each slice's pack is async_eval'd and the ANE waits on the event MLX
                // signals after it (no CPU wait), and slice k-1 is joined only after slice k is queued, so two ANE requests
                // are in flight and each pack overwrites the parity whose join was the slice before. Every joined output
                // must equal the CPU path bit for bit; a missing order shows as mismatches or a hang.
                {
                    auto x16b = mx::astype(mx::random::normal({rows, H}), mx::float16);
                    mx::eval(x16b);
                    auto xBH = toHalfVec(x16b);
                    std::vector<uint16_t> yCpuB(size_t(rows) * H);
                    prog->evaluate(0, xBH.data(), yCpuB.data());
                    std::vector<mx::array> toks, outs;
                    for (int k = 0; k < 4; ++k) {
                        const int p = k & 1;
                        auto tk = prog->pack_input_gpu(k % 2 ? x16b : x16, p);
                        mx::async_eval(tk);
                        prog->ready_after(tk);
                        prog->enqueue(0, /*after_pack=*/true, p);
                        toks.push_back(tk);
                        if (k > 0) {
                            prog->wait((k - 1) & 1);
                            auto yj = prog->join_output_gpu(toks[k - 1], (k - 1) & 1);
                            mx::async_eval(yj);
                            outs.push_back(yj);
                        }
                    }
                    prog->wait(1);
                    outs.push_back(prog->join_output_gpu(toks[3], 1));
                    mx::eval(outs);
                    size_t mm = 0;
                    for (int k = 0; k < 4; ++k) {
                        const float* yh = outs[k].data<float>();
                        const std::vector<uint16_t>& ref = k % 2 ? yCpuB : yCpu;
                        for (size_t i = 0; i < size_t(rows) * H; ++i)
                            if (float(_Float16(yh[i])) != float(std::bit_cast<_Float16>(ref[i]))) ++mm;
                    }
                    std::printf("MLX_ANE_PROGRAM_GPU_HANDOFF slices=4 mismatches=%zu\n", mm);
                    if (mm != 0) throw std::runtime_error("pipelined handoff diverged from CPU");

                    // same pipeline with the join ordered on the GPU timeline (wait_gpu) and no CPU wait before it; the
                    // CPU collects each request only before its parity is enqueued again. A missing GPU wait lets the
                    // join read y before the ANE wrote it (mismatches).
                    std::vector<mx::array> toksG, outsG;
                    for (int k = 0; k < 4; ++k) {
                        const int p = k & 1;
                        if (k >= 2) prog->wait(p);
                        auto tk = prog->pack_input_gpu(k % 2 ? x16b : x16, p);
                        mx::async_eval(tk);
                        prog->ready_after(tk);
                        prog->enqueue(0, /*after_pack=*/true, p);
                        toksG.push_back(tk);
                        if (k > 0) {
                            prog->wait_gpu((k - 1) & 1);
                            auto yj = prog->join_output_gpu(toksG[k - 1], (k - 1) & 1);
                            mx::async_eval(yj);
                            outsG.push_back(yj);
                        }
                    }
                    prog->wait_gpu(1);
                    outsG.push_back(prog->join_output_gpu(toksG[3], 1));
                    mx::eval(outsG);
                    prog->wait(0);
                    prog->wait(1);
                    size_t mmG = 0;
                    for (int k = 0; k < 4; ++k) {
                        const float* yh = outsG[k].data<float>();
                        const std::vector<uint16_t>& ref = k % 2 ? yCpuB : yCpu;
                        for (size_t i = 0; i < size_t(rows) * H; ++i)
                            if (float(_Float16(yh[i])) != float(std::bit_cast<_Float16>(ref[i]))) ++mmG;
                    }
                    std::printf("MLX_ANE_PROGRAM_GPU_WAIT slices=4 mismatches=%zu\n", mmG);
                    if (mmG != 0) throw std::runtime_error("GPU-timeline wait diverged from CPU");
                }

                // ---- timings: per-stage GPU, ANE eval, and the overlap test ----
                const auto median = [](std::vector<double>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
                std::vector<double> packMs, joinMs, reqMs, aneMs;
                for (int i = 0; i < 20; ++i) {
                    auto t = Clock::now();
                    auto tk = prog->pack_input_gpu(x16); mx::eval(tk);
                    packMs.push_back(msSince(t));
                    prog->enqueue(0, false); prog->wait();
                    aneMs.push_back(prog->last_eval_ms());
                    t = Clock::now();
                    auto yj = prog->join_output_gpu(tk); mx::eval(yj);
                    joinMs.push_back(msSince(t));
                }
                for (int i = 0; i < 10; ++i) {
                    auto t = Clock::now();
                    auto w = prog->requantize_weights_gpu(0, wgN, wuN, wdN, C); mx::eval(w);
                    reqMs.push_back(msSince(t));
                }
                const double pMs = median(packMs), jMs = median(joinMs), rMs = median(reqMs), aMs = median(aneMs);
                std::printf("MLX_ANE_PROGRAM_GPU_TIMING C=%d pack_ms=%.3f join_ms=%.3f requantize_ms=%.3f ane_eval_ms=%.2f\n",
                            C, pMs, jMs, rMs, aMs);

                // overlap test: the ANE share enqueued while the GPU runs the complementary share (two FP16 GEMMs for
                // the G = 17408 - C channels: gate/up [rows x H] x [H x G], down [rows x G] x [G x H]).
                auto wgg = mx::astype(mx::multiply(mx::random::normal({G, H}), mx::array(0.02f)), mx::float16);
                auto wug = mx::astype(mx::multiply(mx::random::normal({G, H}), mx::array(0.02f)), mx::float16);
                auto wdg = mx::astype(mx::multiply(mx::random::normal({H, G}), mx::array(0.02f)), mx::float16);
                mx::eval(wgg, wug, wdg);
                const auto gpuShare = [&] {
                    auto g = mx::matmul(x16, mx::transpose(wgg));
                    auto h = mx::multiply(mx::multiply(g, mx::sigmoid(g)), mx::matmul(x16, mx::transpose(wug)));
                    return mx::matmul(h, mx::transpose(wdg));  // [rows, H]
                };
                const auto best = [](auto f) { f(); double b = 1e30; for (int i = 0; i < 7; ++i) { auto t = Clock::now(); f(); b = std::min(b, msSince(t)); } return b; };

                // pack once so the ANE input is ready for the enqueue-only timings.
                { auto tk = prog->pack_input_gpu(x16); mx::eval(tk); }
                const double gpuOnly = best([&] { mx::eval(gpuShare()); });
                const double aneOnly = best([&] { prog->enqueue(0, false); prog->wait(); });
                const double both = best([&] {
                    auto g = gpuShare();
                    mx::async_eval(g);                 // GPU complementary share starts now
                    prog->enqueue(0, false);           // ANE share runs on the Neural Engine meanwhile
                    prog->wait();
                    mx::eval(g);
                });
                std::printf("MLX_ANE_PROGRAM_OVERLAP C=%d gpu_share_ms=%.2f ane_share_ms=%.2f both_ms=%.2f speedup=%.2fx\n",
                            C, gpuOnly, aneOnly, both, (gpuOnly + aneOnly) / both);
            }

            // ===================== variable rows: the program has a function per slice size =====================
            // A slice of n tokens runs the smallest compiled function with >= n rows. For every size the program offers,
            // pack the first n rows of a 2048-row input, evaluate, join, and compare with the FP32 reference of those n
            // rows (rel L2 must stay at the 2048-row level) and, informationally, with the 2048-row function's output.
            {
                const int C = 6656;
                mx::random::seed(20261024);
                auto wg = mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float16);
                auto wu = mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float16);
                auto wd = mx::astype(mx::multiply(mx::random::normal({H, C}), mx::array(0.02f)), mx::float16);
                auto xall = mx::astype(mx::random::normal({rows, H}), mx::float16);
                mx::eval(wg, wu, wd, xall);
                auto refAll = mx::contiguous(referenceFfn(xall, wg, wu, wd)); mx::eval(refAll);
                auto prog = AneProgram::open(C, rows, H, 1);
                prog->bind_weights(0, wg.data<mx::float16_t>(), wu.data<mx::float16_t>(), wd.data<mx::float16_t>());
                std::vector<uint16_t> yFull(size_t(rows) * H);
                prog->evaluate(0, xall.data<mx::float16_t>(), yFull.data());
                // GPU path slots use requantize_weights_gpu from NaxAffine; stage the same share through random NAX weights
                std::mt19937 rng(20261024);
                auto wgN = randomNax(17408, H, rng), wuN = randomNax(17408, H, rng), wdN = randomNax(H, 17408, rng);
                prog->reserve_slots(2);
                auto staged = prog->requantize_weights_gpu(0, wgN, wuN, wdN, C);
                mx::eval(staged);
                auto dgN = nax_affine_dequant(wgN), duN = nax_affine_dequant(wuN), ddN = nax_affine_dequant(wdN);
                auto wgS = mx::contiguous(mx::slice(dgN, {17408 - C, 0}, {17408, H}));
                auto wuS = mx::contiguous(mx::slice(duN, {17408 - C, 0}, {17408, H}));
                auto wdS = mx::contiguous(mx::slice(ddN, {0, 17408 - C}, {H, 17408}));
                mx::eval(wgS, wuS, wdS);
                std::printf("MLX_ANE_PROGRAM_ROW_COUNTS");
                for (int n : prog->row_counts()) std::printf(" %d", n);
                std::printf("\n");
                if (prog->fit_rows(1) != prog->row_counts().front() || prog->fit_rows(rows) != rows || prog->fit_rows(rows + 1) != -1 ||
                    prog->fit_rows(513) != 768)
                    throw std::runtime_error("fit_rows does not pick the smallest function >= n");
                for (int n : prog->row_counts()) {
                    auto xn = mx::contiguous(mx::slice(xall, {0, 0}, {n, H}));
                    auto refN = mx::contiguous(referenceFfn(xn, wgS, wuS, wdS)); mx::eval(refN);
                    auto tk = prog->pack_input_gpu(xn);
                    mx::eval(tk);
                    prog->enqueue(0, false);
                    prog->wait();
                    auto yn = prog->join_output_gpu(tk);
                    mx::eval(yn);
                    if (yn.shape(0) != n) throw std::runtime_error("join returned the wrong row count");
                    const float* yv = yn.data<float>(); const float* rv = refN.data<float>();
                    double num = 0, den = 0; bool finite = true;
                    for (size_t i = 0; i < size_t(n) * H; ++i) {
                        if (!std::isfinite(yv[i])) finite = false;
                        const double d = double(yv[i]) - double(rv[i]);
                        num += d * d; den += double(rv[i]) * double(rv[i]);
                    }
                    const double rel = std::sqrt(num / std::max(den, 1e-30));
                    std::printf("MLX_ANE_PROGRAM_ROWS n=%d rel_l2=%.4e finite=%d\n", n, rel, finite ? 1 : 0);
                    if (!finite || rel > 5e-2) throw std::runtime_error("variable-rows ANE output wrong at n=" + std::to_string(n));
                }
            }

            // ===================== accuracy against the share size C (the engine runs C up to ~14k) =====================
            // FP32 reference of the same FP16 slice at several C, plain affine-style random weights; the join must stay
            // finite and rel L2 must not grow with C. j-quality 2026-10-21: the model's KL jumps between C 12288 and 12800.
            if (std::getenv("TQLLM_ANE_TEST_C_SWEEP")) {
                for (int C : {11776, 12288, 12800, 13312, 13824, 14336}) {
                    mx::random::seed(20261023);
                    auto wg = mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float16);
                    auto wu = mx::astype(mx::multiply(mx::random::normal({C, H}), mx::array(0.02f)), mx::float16);
                    auto wd = mx::astype(mx::multiply(mx::random::normal({H, C}), mx::array(0.02f)), mx::float16);
                    auto x = mx::astype(mx::random::normal({rows, H}), mx::float16);
                    // TQLLM_ANE_TEST_OUTLIER=k: share row 192 gets k x larger gate/up weights, so tokens see an
                    // intermediate value far above the rest (the real model has such channels: layer 0 inter max 2.0 vs 0.3)
                    if (const char* ok = std::getenv("TQLLM_ANE_TEST_OUTLIER")) {
                        const float k = float(std::atof(ok));
                        auto rowScale = mx::ones({C, 1}, mx::float32);
                        rowScale = mx::scatter(rowScale, mx::array(std::vector<int>{192}.data(), {1}, mx::int32),
                                               mx::full({1, 1, 1}, k, mx::float32), 0);
                        wg = mx::astype(mx::multiply(mx::astype(wg, mx::float32), rowScale), mx::float16);
                        wu = mx::astype(mx::multiply(mx::astype(wu, mx::float32), rowScale), mx::float16);
                    }
                    mx::eval(wg, wu, wd, x);
                    auto ref = mx::contiguous(referenceFfn(x, wg, wu, wd)); mx::eval(ref);
                    auto prog = AneProgram::open(C, rows, H, 1);
                    prog->bind_weights(0, wg.data<mx::float16_t>(), wu.data<mx::float16_t>(), wd.data<mx::float16_t>());
                    std::vector<uint16_t> y(size_t(rows) * H);
                    prog->evaluate(0, x.data<mx::float16_t>(), y.data());
                    const float* r = ref.data<float>();
                    double num = 0, den = 0; size_t bad = 0;
                    for (size_t i = 0; i < size_t(rows) * H; ++i) {
                        const float got = float(std::bit_cast<_Float16>(y[i]));
                        if (!std::isfinite(got)) { ++bad; continue; }
                        const double d = double(got) - double(r[i]);
                        num += d * d; den += double(r[i]) * double(r[i]);
                    }
                    std::printf("MLX_ANE_PROGRAM_C_SWEEP C=%d rel_l2=%.4e nonfinite=%zu\n", C, std::sqrt(num / std::max(den, 1e-30)), bad);
                }
            }

            // ===================== TQ4: the ANE share on TQ4 weights (stored rotated, 128-blocks) =====================
            // Random TQ4 blocks (66 bytes: FP16 scale, 64 packed code bytes) at the real FFN shape. The model computes
            // y = W x with W' = W R^T stored (R = H D / sqrt(128) per 128-block), so with xr = R x it is y = W' xr; the
            // down weights' columns are rotated the same way over the intermediate. Reference (FP32, GPU): xr =
            // tq4_rotate(x), g = xr Wg'^T, u = xr Wu'^T, h = silu(g) u, y = tq4_rotate(h) Wd'^T, on the share's channels.
            // The ANE path takes the unrotated x and the stored weights: pack rotates in TQ4's basis, gate/up are
            // requantized as stored, down is un-rotated per 128-block then rotated by the ANE's 512 Hadamard.
            {
                const int C = 6656, F = 17408;
                std::mt19937 rng(20261021);
                const auto blocks = [&](int r, int c) {
                    const int nb = c / 128;
                    std::vector<uint8_t> b(size_t(r) * nb * 66);
                    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
                    for (size_t i = 0; i < size_t(r) * nb; ++i) {
                        const uint16_t h = std::bit_cast<uint16_t>(_Float16(sd(rng)));
                        b[i * 66] = uint8_t(h & 255u); b[i * 66 + 1] = uint8_t(h >> 8);
                        for (int j = 0; j < 64; ++j) b[i * 66 + 2 + j] = uint8_t(rng());
                    }
                    return b;
                };
                const auto tq = [&](int r, int c) {
                    auto b = blocks(r, c);
                    return tqllm::mlx_backend::nax_from_tq4(tqllm::mlx_backend::load_tq4_blocks(b, r, c));
                };
                auto wgT = tq(F, H), wuT = tq(F, H), wdT = tq(H, F);
                auto xq = mx::astype(mx::random::normal({rows, H}), mx::float16);
                mx::eval(xq);
                // reference on the last C channels
                auto dg = nax_affine_dequant(wgT), du = nax_affine_dequant(wuT), dd = nax_affine_dequant(wdT);
                auto wgC = mx::astype(mx::slice(dg, {F - C, 0}, {F, H}), mx::float32);
                auto wuC = mx::astype(mx::slice(du, {F - C, 0}, {F, H}), mx::float32);
                auto wdC = mx::astype(mx::slice(dd, {0, F - C}, {H, F}), mx::float32);
                auto xr = tqllm::mlx_backend::tq4_rotate(xq);                               // [rows, H] fp32
                auto gg = mx::matmul(xr, mx::transpose(wgC)), uu = mx::matmul(xr, mx::transpose(wuC));
                auto hh = mx::multiply(mx::multiply(gg, mx::sigmoid(gg)), uu);               // [rows, C] true basis
                auto refT = mx::contiguous(mx::matmul(tqllm::mlx_backend::tq4_rotate(hh), mx::transpose(wdC)));
                mx::eval(refT);

                auto prog = AneProgram::open(C, rows, H, 1);
                prog->reserve_slots(2);
                auto staged = prog->requantize_weights_gpu(0, wgT, wuT, wdT, C);
                mx::eval(staged);
                auto tk = prog->pack_input_gpu(xq);
                mx::eval(tk);
                prog->enqueue(0, false);
                prog->wait();
                auto yT = prog->join_output_gpu(tk);
                mx::eval(yT);
                const float* yv = yT.data<float>();
                const float* rv = refT.data<float>();
                double num = 0, den = 0;
                bool finite = true;
                for (size_t i = 0; i < size_t(rows) * H; ++i) {
                    if (!std::isfinite(yv[i])) finite = false;
                    const double d = double(yv[i]) - double(rv[i]);
                    num += d * d; den += double(rv[i]) * double(rv[i]);
                }
                const double relT = std::sqrt(num / std::max(den, 1e-30));
                std::printf("MLX_ANE_PROGRAM_TQ4_ACCURACY C=%d rows=%d rel_l2=%.4e finite=%d\n", C, rows, relT, finite ? 1 : 0);
                if (!finite) throw std::runtime_error("TQ4 ANE output not finite");
                if (relT > 5e-2) throw std::runtime_error("TQ4 ANE rel L2 error above 5e-2 (basis mismatch?)");

                // The GPU share of a TQ4 split layer: the staged 4-bit kernel on the leading G channels (gate EP0, up EP1
                // = silu(gate) x up, rotated intermediate, down over the leading columns) against dequant + FP16 matmul.
                const int G = F - C;
                const auto cutRows = [&](const NaxAffine& w, int n) {
                    NaxAffine v = w;
                    mx::Shape b(w.words.ndim(), 0), e = w.words.shape(); e[0] = n / 32; v.words = mx::slice(w.words, b, e);
                    mx::Shape bs(w.scales.ndim(), 0), es = w.scales.shape(); es[0] = n / 32; v.scales = mx::slice(w.scales, bs, es);
                    v.rows = n; return v;
                };
                const auto cutCols = [&](const NaxAffine& w, int n) {
                    NaxAffine v = w;
                    mx::Shape b(w.words.ndim(), 0), e = w.words.shape(); e[1] = n / 16; v.words = mx::contiguous(mx::slice(w.words, b, e));
                    mx::Shape bs(w.scales.ndim(), 0), es = w.scales.shape(); es[1] = n / w.group; v.scales = mx::contiguous(mx::slice(w.scales, bs, es));
                    v.columns = n; return v;
                };
                auto g4 = cutRows(wgT, G), u4 = cutRows(wuT, G), d4 = cutCols(wdT, G);
                auto xr16 = tqllm::mlx_backend::tq4_rotate_half(mx::astype(mx::multiply(xq, mx::array(0.05f)), mx::float16));  // small: keep FP16 finite
                auto g16 = tqllm::mlx_backend::nax_affine_prefill(g4, xr16, 0, std::nullopt, 32, 64, 4, 1, 1, 1);
                auto h16 = tqllm::mlx_backend::nax_affine_prefill(u4, xr16, 1, g16, 32, 64, 4, 1, 1, 1);
                auto hr16 = tqllm::mlx_backend::tq4_rotate_half(h16);
                auto yS = mx::astype(tqllm::mlx_backend::nax_affine_prefill(d4, hr16, 0, std::nullopt, 32, 64, 4, 0, 0, 1), mx::float32);
                auto rg = mx::matmul(xr16, mx::transpose(mx::slice(dg, {0, 0}, {G, H})));
                auto ru = mx::matmul(xr16, mx::transpose(mx::slice(du, {0, 0}, {G, H})));
                auto rh = mx::multiply(mx::multiply(rg, mx::sigmoid(rg)), ru);
                auto rhr = tqllm::mlx_backend::tq4_rotate_half(mx::astype(rh, mx::float16));
                auto yR = mx::astype(mx::matmul(rhr, mx::transpose(mx::slice(dd, {0, 0}, {H, G}))), mx::float32);
                auto dgap = mx::sqrt(mx::sum(mx::square(mx::subtract(yS, yR))));
                auto dref = mx::sqrt(mx::sum(mx::square(yR)));
                auto ggap = mx::sqrt(mx::sum(mx::square(mx::subtract(mx::astype(g16, mx::float32), mx::astype(rg, mx::float32)))));
                auto gref = mx::sqrt(mx::sum(mx::square(mx::astype(rg, mx::float32))));
                mx::eval(dgap, dref, ggap, gref);
                std::printf("MLX_ANE_PROGRAM_TQ4_GPU_SHARE gate_rel=%.3e share_rel=%.3e\n",
                            ggap.item<float>() / gref.item<float>(), dgap.item<float>() / dref.item<float>());
            }

            std::printf("MLX_ANE_PROGRAM_PASS\n");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "MLX_ANE_PROGRAM_FAIL %s\n", e.what());
            return 1;
        }
    }
    return 0;
}
