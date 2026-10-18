// CPU + GPU co-execution for the prefill FFN (one layer, 5120 -> 17408 -> 5120, M rows): the GPU (MLX FP16 GEMM, as
// the engine's prefill FFN) computes the first F - C intermediate columns while the CPU stream (Accelerate, FP32)
// computes the last C columns from an FP32 slice of the weights. Inputs are materialized before the split, otherwise
// MLX serializes the two streams inside one lazy graph. usage: test-mlx-cpu-split [M] [C,C,...]
#include <mlx/mlx.h>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace mx = mlx::core;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::stoi(argv[1]) : 2048, H = 5120, F = 17408;
    std::vector<int> cs;
    for (std::string s = argc > 2 ? argv[2] : "0,512,1024,1536,2048"; !s.empty();) {
        auto p = s.find(','); cs.push_back(std::stoi(s.substr(0, p))); s = p == std::string::npos ? "" : s.substr(p + 1);
    }
    auto gpu = mx::Device::gpu, cpu = mx::Device::cpu;
    auto x = mx::astype(mx::random::normal({M, H}), mx::float16);
    auto wgf = mx::astype(mx::multiply(mx::random::normal({F, H}), mx::array(0.02f)), mx::float16);
    auto wuf = mx::astype(mx::multiply(mx::random::normal({F, H}), mx::array(0.02f)), mx::float16);
    auto wdf = mx::astype(mx::multiply(mx::random::normal({H, F}), mx::array(0.02f)), mx::float16);
    mx::eval(x, wgf, wuf, wdf);
    auto ffn = [&](const mx::array& in, const mx::array& a, const mx::array& b, const mx::array& d, mx::StreamOrDevice s) {
        auto g = mx::matmul(in, mx::transpose(a, s), s);
        auto h = mx::multiply(mx::multiply(g, mx::sigmoid(g, s), s), mx::matmul(in, mx::transpose(b, s), s), s);
        return mx::matmul(h, mx::transpose(d, s), s);
    };
    auto best = [](auto f) { f(); double b = 1e30; for (int i = 0; i < 5; ++i) { auto t = Clock::now(); f(); b = std::min(b, ms_since(t)); } return b; };
    auto ref = ffn(x, wgf, wuf, wdf, gpu);
    mx::eval(ref);
    const double t_full = best([&] { mx::eval(ffn(x, wgf, wuf, wdf, gpu)); });
    const double flop = 2.0 * M * 3.0 * H * F;
    std::printf("CPU_SPLIT M=%d gpu_full %.2f ms (%.1f TFLOPS)\n", M, t_full, flop / t_full * 1e-9);
    for (int C : cs) {
        const int G = F - C;
        auto wg = mx::slice(wgf, {0, 0}, {G, H}), wu = mx::slice(wuf, {0, 0}, {G, H}), wd = mx::slice(wdf, {0, 0}, {H, G});
        // CPU share: FP32 operands (Accelerate sgemm), materialized once as a stand-in for a per-chunk FP32 slice dequant
        auto x32 = mx::astype(x, mx::float32, cpu);
        auto cg = mx::astype(mx::slice(wgf, {G, 0}, {F, H}), mx::float32, cpu), cu = mx::astype(mx::slice(wuf, {G, 0}, {F, H}), mx::float32, cpu);
        auto cd = mx::astype(mx::slice(wdf, {0, G}, {H, F}), mx::float32, cpu);
        mx::eval(wg, wu, wd, x32, cg, cu, cd);
        const double t_gpu = C < F ? best([&] { mx::eval(ffn(x, wg, wu, wd, gpu)); }) : 0.0;
        const double t_cpu = C > 0 ? best([&] { mx::eval(ffn(x32, cg, cu, cd, cpu)); }) : 0.0;
        const double t_both = best([&] {
            auto g = C < F ? ffn(x, wg, wu, wd, gpu) : mx::zeros({M, H}, mx::float16);
            auto c = C > 0 ? ffn(x32, cg, cu, cd, cpu) : mx::zeros({M, H}, mx::float32);
            mx::eval(g, c);
        });
        auto y = mx::add(mx::astype(C < F ? ffn(x, wg, wu, wd, gpu) : mx::zeros({M, H}, mx::float16), mx::float32),
                         C > 0 ? ffn(x32, cg, cu, cd, cpu) : mx::zeros({M, H}, mx::float32));
        auto err = mx::divide(mx::max(mx::abs(mx::subtract(y, mx::astype(ref, mx::float32)))), mx::max(mx::abs(mx::astype(ref, mx::float32))));
        mx::eval(err);
        std::printf("CPU_SPLIT M=%d C=%d gpu_share %.2f ms cpu_share %.2f ms (%.2f TFLOPS) concurrent %.2f ms -> %.2fx rel_err %.1e\n", M, C, t_gpu, t_cpu,
                    C > 0 ? 2.0 * M * 3.0 * H * C / t_cpu * 1e-9 : 0.0, t_both, t_full / t_both, err.item<float>());
    }
    std::puts("CPU_SPLIT_DONE");
    return 0;
}
