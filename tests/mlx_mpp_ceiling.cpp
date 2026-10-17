// Ceiling of the M5 matrix unit for prefill-shaped GEMMs: Y[M,N] = X[M,K] * W[N,K]^T, FP16 operands staged in
// threadgroup memory, MPP matmul2d(MT, NT, BK) on execution_simdgroups<SIMDS>, FP32 accumulate. Weights are already
// FP16 here (no decode), so this is the upper bound a decode-on-the-fly prefill kernel can approach.
#include <mlx/mlx.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace mx = mlx::core;
namespace {
mx::array tiled(const mx::array& x, const mx::array& w, int mt, int nt, int bk, int simds) {
    const int m = x.shape(0), k = x.shape(1), n = w.shape(0);
    static auto kernel = mx::fast::metal_kernel("tqllm_mpp_ceiling", {"x", "w"}, {"y"}, R"METAL(
uint tid = thread_position_in_threadgroup.x;
uint n0 = threadgroup_position_in_grid.x * uint(NT), m0 = threadgroup_position_in_grid.y * uint(MT);
threadgroup half A[MT * BK], B[NT * BK];
tensor<threadgroup half, extents<int, BK, MT>, tensor_inline> ta(A, extents<int, BK, MT>());
tensor<threadgroup half, extents<int, BK, NT>, tensor_inline> tb(B, extents<int, BK, NT>());
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(MT, NT, BK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SIMDS>> op;
auto acc = op.template get_destination_cooperative_tensor<decltype(ta), decltype(tb), float>();
for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
for (uint base = 0; base < uint(K); base += uint(BK)) {
    for (uint i = tid; i < uint(MT * BK / 4); i += 32u * uint(SIMDS)) {
        uint row = (i * 4u) / uint(BK), col = (i * 4u) % uint(BK);
        *(threadgroup half4*)(A + i * 4u) = m0 + row < uint(M) ? *(const device half4*)(x + (m0 + row) * uint(K) + base + col) : half4(0.0h);
    }
    for (uint i = tid; i < uint(NT * BK / 4); i += 32u * uint(SIMDS)) {
        uint row = (i * 4u) / uint(BK), col = (i * 4u) % uint(BK);
        *(threadgroup half4*)(B + i * 4u) = n0 + row < uint(N) ? *(const device half4*)(w + (n0 + row) * uint(K) + base + col) : half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(ta, tb, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto c = acc.get_multidimensional_index(i);
    uint col = n0 + uint(c[0]), row = m0 + uint(c[1]);
    if (row < uint(M) && col < uint(N)) y[row * uint(N) + col] = acc[i];
}
)METAL", R"METAL(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL");
    return kernel({x, w}, {{m, n}}, {mx::float32}, {((n + nt - 1) / nt) * 32 * simds, (m + mt - 1) / mt, 1}, {32 * simds, 1, 1},
                  {{"MT", mt}, {"NT", nt}, {"BK", bk}, {"SIMDS", simds}, {"M", m}, {"N", n}, {"K", k}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
template <class F>
double best_ms(F f) {
    mx::eval(f());
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        auto t = std::chrono::steady_clock::now();
        mx::eval(f());
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
    }
    return best;
}
}  // namespace

int main() {
    const int M = 2048, K = 5120;
    for (int N : {17408, 5120}) {
        auto x = mx::astype(mx::random::normal({M, K}), mx::float16);
        auto w = mx::astype(mx::multiply(mx::random::normal({N, K}), mx::array(0.02f)), mx::float16);
        mx::eval(x, w);
        const double flop = 2.0 * M * N * K;
        auto ref = mx::astype(mx::matmul(x, mx::transpose(w)), mx::float32);
        const double t_ref = best_ms([&] { return mx::matmul(x, mx::transpose(w)); });
        std::printf("CEILING N=%d mlx_fp16_matmul %.2f ms %.1f TFLOPS\n", N, t_ref, flop / t_ref * 1e-9);
        struct V { int mt, nt, bk, simds; };
        for (V v : {V{64, 64, 32, 4}, V{64, 128, 32, 4}, V{128, 64, 32, 4}, V{128, 128, 32, 4}, V{64, 64, 64, 4},
                    V{128, 128, 16, 4}, V{64, 64, 32, 2}, V{128, 128, 32, 8}}) {
            if ((v.mt * v.bk + v.nt * v.bk) * 2 > 32768) continue;
            try {
                auto y = tiled(x, w, v.mt, v.nt, v.bk, v.simds);
                auto err = mx::max(mx::abs(mx::subtract(y, ref)));
                auto peak = mx::max(mx::abs(ref));
                mx::eval(err, peak);
                const double t = best_ms([&] { return tiled(x, w, v.mt, v.nt, v.bk, v.simds); });
                std::printf("CEILING N=%d mt=%d nt=%d bk=%d simds=%d %.2f ms %.1f TFLOPS rel_err %.2e\n", N, v.mt, v.nt, v.bk,
                            v.simds, t, flop / t * 1e-9, err.item<float>() / peak.item<float>());
            } catch (const std::exception& e) {
                std::printf("CEILING N=%d mt=%d nt=%d bk=%d simds=%d ERROR %s\n", N, v.mt, v.nt, v.bk, v.simds, std::string(e.what()).substr(0, 300).c_str());
            }
        }
    }
    std::puts("CEILING_DONE");
}
