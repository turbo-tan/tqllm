// Inkling-2 verify expert stage, synthetic weights (no model): the time of one routed-expert product for a block of
// `tokens` rows x 8 experts with `distinct` experts among them, on the matrix-unit rows kernel (NVFP4) and on gather_qmm
// (NVFP4 / MXFP8, sorted), and the bytes of the distinct experts read per second.
// Usage: bench-mlx-expert-rows [tokens] [distinct] [reps] [dense]
// dense: also the row-invariant dense product (2304 -> 4096 BF16, 1 and `tokens` rows) against MLX's matmul
#include "mlx_floatx.hpp"
#include "inkling2_rows.hpp"
#include "inkling2_mlx_ops.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>
namespace mx = mlx::core;
using tqllm::mlx_backend::FloatxExperts;
using tqllm::mlx_backend::inkling2_rows_experts;
namespace {
FloatxExperts make(int E, int O, int K, int bits, uint32_t seed) {
    mx::random::seed(seed);
    auto codes = mx::astype(mx::random::randint(0, 256, {E, O, K * bits / 8}), mx::uint8);
    const int group = bits == 4 ? 16 : 32;
    // E4M3 scales 0x30..0x3f (0.25 .. ~0.94) for NVFP4, E8M0 120..127 for MXFP8
    auto scales = mx::astype(mx::random::randint(bits == 4 ? 0x30 : 120, bits == 4 ? 0x40 : 128, {E, O, K / group}), mx::uint8);
    std::optional<mx::array> global;
    if (bits == 4) global = mx::full({E}, 1.0f, mx::float32);
    mx::eval({codes, scales});
    return FloatxExperts(codes, scales, global, bits);
}
template <class F> double time_ms(F f, int reps) {
    mx::eval(f());
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<mx::array> keep;
    for (int r = 0; r < reps; ++r) keep.push_back(f());
    mx::eval(keep);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}
}  // namespace
int main(int argc, char** argv) {
    const int tokens = argc > 1 ? std::stoi(argv[1]) : 8, distinct = argc > 2 ? std::stoi(argv[2]) : 40, reps = argc > 3 ? std::stoi(argv[3]) : 50;
    constexpr int E = 128, topk = 8, D = 2304, I = 1152;
    mx::set_default_device(mx::Device::gpu);
    const int P = tokens * topk;
    // routes: each token 8 different experts from a pool of `distinct`, every pool expert used at least once
    std::mt19937 rng(7);
    std::vector<int> pool(E);
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), rng);
    pool.resize(size_t(std::max(distinct, topk)));
    std::vector<uint32_t> ids;
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> pick;
        for (int k = 0; k < topk; ++k) pick.push_back(pool[size_t((t * topk + k) % int(pool.size()))]);
        std::sort(pick.begin(), pick.end());
        pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
        while (int(pick.size()) < topk) { int e = pool[rng() % pool.size()]; if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e); }
        for (int e : pick) ids.push_back(uint32_t(e));
    }
    std::vector<uint32_t> order(static_cast<size_t>(P));
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return ids[a] < ids[b]; });
    std::vector<uint32_t> sorted(static_cast<size_t>(P)), token(static_cast<size_t>(P));
    for (int i = 0; i < P; ++i) { sorted[size_t(i)] = ids[order[size_t(i)]]; token[size_t(i)] = order[size_t(i)] / topk; }
    std::vector<uint32_t> uniq(sorted);
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    const int used = int(uniq.size());
    auto s = mx::array(sorted.data(), {P}, mx::uint32), tk = mx::array(token.data(), {P}, mx::uint32);
    auto x = mx::astype(mx::random::normal({tokens, D}), mx::bfloat16), a = mx::astype(mx::random::normal({P, I}), mx::bfloat16);
    auto xa = mx::astype(mx::random::normal({P, D}), mx::bfloat16);
    mx::eval({x, a, xa});
    std::printf("EXPERT_ROWS tokens %d pairs %d distinct %d\n", tokens, P, used);
    for (int bits : {4, 8}) {
        auto gu = make(E, 2 * I, D, bits, 1), dn = make(E, D, I, bits, 2);
        const double bytes_gu = double(used) * (gu.codes.nbytes() + gu.scales.nbytes()) / E;
        const double bytes_dn = double(used) * (dn.codes.nbytes() + dn.scales.nbytes()) / E;
        auto report = [&](const char* what, double ms, double bytes) {
            std::printf("EXPERT_ROWS %s bits %d ms %.4f GB/s %.1f\n", what, bits, ms, bytes / ms / 1e6);
        };
        auto xg = mx::expand_dims(mx::take(x, tk, 0), 1);
        report("gather_qmm gate_up", time_ms([&] { return gu(xg, s, true); }, reps), bytes_gu);
        report("gather_qmm down", time_ms([&] { return dn(mx::expand_dims(a, 1), s, true); }, reps), bytes_dn);
        // the row-invariant kernels the engine runs (decode and verify): gate_up with the fused SwiGLU, then down
        report("inkling2_rows gate_up+swiglu", time_ms([&] { return inkling2_rows_experts(x, tk, s, gu.codes, gu.scales, gu.global, bits, true, 4.f, 25.f); }, reps), bytes_gu);
        report("inkling2_rows down", time_ms([&] { return inkling2_rows_experts(a, mx::arange(P, mx::uint32), s, dn.codes, dn.scales, dn.global, bits); }, reps), bytes_dn);
        for (auto [sg, to] : std::vector<std::pair<int, int>>{{2, 2}, {8, 2}, {8, 4}}) {
            const std::string tag = " sg" + std::to_string(sg) + " to" + std::to_string(to);
            report(("simd gate_up+swiglu" + tag).c_str(), time_ms([&] { return inkling2_rows_experts(x, tk, s, gu.codes, gu.scales, gu.global, bits, true, 4.f, 25.f, sg, to); }, reps), bytes_gu);
        }
        if (bits == 4) {
            gu.pack_nax(); dn.pack_nax();
            report("rows gate_up", time_ms([&] { return gu.rows(x, tk, s); }, reps), bytes_gu);
            report("rows down", time_ms([&] { return dn.rows(a, mx::arange(P, mx::uint32), s); }, reps), bytes_dn);
        }
    }
    if (argc > 4 && std::string(argv[4]) == "attn") {  // one decode row / an 8-row block: the row-invariant attention vs MLX's
        const int H = 24, G = 8, HD = 128, DR = 16, E = 1024;
        for (auto [keys, window] : std::vector<std::pair<int, int>>{{1536, 1024}, {8192, 0}, {32768, 0}}) {
            for (int rows : {1, 8}) {
                const int S = keys + rows;
                auto kc = mx::astype(mx::random::normal({1, G, S + 1, HD}), mx::bfloat16), vc = mx::astype(mx::random::normal({1, G, S + 1, HD}), mx::bfloat16);
                auto q = mx::astype(mx::random::normal({1, H, rows, HD}), mx::bfloat16);
                auto rel = mx::random::normal({rows, H, DR});
                auto proj = mx::astype(mx::random::normal({DR, E}), mx::bfloat16);
                auto sinks = mx::astype(mx::random::normal({rows, H}), mx::bfloat16);
                mx::eval({kc, vc, q, rel, proj, sinks});
                const int start = window ? std::max(0, keys + rows - window - 8) : 0;
                auto kv = mx::slice(kc, {0, 0, start, 0}, {1, G, S + 1, HD}), vv = mx::slice(vc, {0, 0, start, 0}, {1, G, S + 1, HD});
                const int s = S - start;
                const double ours = time_ms([&] { return tqllm::mlx_backend::inkling2_rows_attention(q, kv, vv, rel, proj, sinks, keys, start, s, window, 0.088f); }, reps);
                const double mlx = time_ms([&] {
                    auto bias = mx::matmul(mx::astype(rel, mx::bfloat16), proj);
                    auto mask = tqllm::mlx_backend::inkling2_mask(bias, sinks, keys, start, s, window);
                    if (rows > 1) return mx::reshape(tqllm::mlx_backend::inkling2_sink_attention(mx::reshape(q, {1, H * rows, 1, HD}), kv, vv,
                                                     mx::reshape(mask, {1, H * rows, 1, s + 1}), 0.088f), {1, H, rows, HD});
                    return tqllm::mlx_backend::inkling2_sink_attention(q, kv, vv, mask, 0.088f); }, reps);
                std::printf("ATTN keys %d window %d rows %d us: ours %.1f mlx %.1f\n", keys, window, rows, ours * 1e3, mlx * 1e3);
            }
        }
        return 0;
    }
    if (argc > 4 && std::string(argv[4]) == "shapes") {  // the decode step's dense shapes, BF16 (exact mode)
        const std::vector<std::pair<int, int>> shapes{{3072, 2304}, {1024, 2304}, {384, 2304}, {24, 2304}, {32, 2304},
                                                      {2304, 3072}, {128, 2304}, {64, 4608}, {2304, 64}};
        // 24 copies of each matrix, cycled: the weights come from DRAM as in a decode step, not from the system cache
        for (auto [O, K] : shapes) {
            std::vector<mx::array> ws;
            for (int c = 0; c < 24; ++c) ws.push_back(mx::astype(mx::multiply(mx::random::normal({O, K}), mx::array(0.02f)), mx::bfloat16));
            auto x1 = mx::astype(mx::random::normal({1, K}), mx::bfloat16), x8 = mx::astype(mx::random::normal({8, K}), mx::bfloat16);
            mx::eval(ws); mx::eval({x1, x8});
            int at = 0;
            auto next = [&]() -> const mx::array& { at = (at + 1) % int(ws.size()); return ws[size_t(at)]; };
            const double a = time_ms([&] { return mx::matmul(x1, mx::transpose(next())); }, reps);
            const double c = time_ms([&] { return tqllm::mlx_backend::inkling2_rows_linear(x1, next()); }, reps);
            const double d = time_ms([&] { return tqllm::mlx_backend::inkling2_rows_linear(x8, next()); }, reps);
            const double e = time_ms([&] { return mx::matmul(x8, mx::transpose(next())); }, reps);
            std::printf("SHAPE %dx%d us (DRAM): mlx1 %.1f rows1 %.1f rows8 %.1f mlx8 %.1f\n", O, K, a * 1e3, c * 1e3, d * 1e3, e * 1e3);
        }
        return 0;
    }
    if (argc > 4) {
        auto w = mx::astype(mx::multiply(mx::random::normal({4096, D}), mx::array(0.02f)), mx::bfloat16);
        mx::eval(w);
        const double bytes = double(w.nbytes());
        for (int rows : {1, tokens}) {
            auto xr = mx::slice(x, {0, 0}, {rows, D});
            mx::eval(xr);
            const double a = time_ms([&] { return mx::matmul(xr, mx::transpose(w)); }, reps);
            const double b = time_ms([&] { return tqllm::mlx_backend::inkling2_rows_linear(xr, w); }, reps);
            std::printf("DENSE rows %d mlx ms %.4f GB/s %.1f inkling2_rows ms %.4f GB/s %.1f\n", rows, a, bytes / a / 1e6, b, bytes / b / 1e6);
            for (int bits : {8, 4}) {
                auto q = mx::quantize(w, 64, bits);
                mx::eval(q);
                const double qb = double(q[0].nbytes() + q[1].nbytes() + q[2].nbytes());
                const double m = time_ms([&] { return mx::quantized_matmul(xr, q[0], q[1], q[2], true, 64, bits); }, reps);
                const double a1 = time_ms([&] { return tqllm::mlx_backend::inkling2_rows_affine(xr, q[0], q[1], q[2], 64, bits); }, reps);
                std::printf("AFFINE%d rows %d mlx GB/s %.1f rows GB/s %.1f\n", bits, rows, qb / m / 1e6, qb / a1 / 1e6);
            }
            for (auto [sg, to] : std::vector<std::pair<int, int>>{{8, 1}, {8, 2}, {4, 2}, {2, 4}}) {
                const double c = time_ms([&] { return tqllm::mlx_backend::inkling2_rows_linear(xr, w, sg, to); }, reps);
                std::printf("DENSE rows %d simd sg%d to%d ms %.4f GB/s %.1f\n", rows, sg, to, c, bytes / c / 1e6);
            }
        }
    }
    return 0;
}
