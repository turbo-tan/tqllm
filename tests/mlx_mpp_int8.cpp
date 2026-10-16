// Matrix-unit throughput per operand type for prefill-shaped GEMMs: Y[M,N] = X[M,K] * W[N,K]^T, operands staged in
// threadgroup memory, MPP matmul2d(MT, NT, BK) on execution_simdgroups<SIMDS>. Same staging loop for every variant, so
// the numbers compare like for like: half x half -> float (today's prefill GEMM), int8 x int8 -> int32 (W8A8), and the
// mixed half x int8 / int8 x half -> float pairs the SDK header lists (fast only if the hardware runs them natively).
#include <mlx/mlx.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace mx = mlx::core;
namespace {
struct Variant { const char* name; const char* ta; const char* tb; const char* tc; };
mx::array tiled(const Variant& v, const mx::array& x, const mx::array& w, int mt, int nt, int bk, int simds) {
    const int m = x.shape(0), k = x.shape(1), n = w.shape(0);
    using Kernel = decltype(mx::fast::metal_kernel(std::string(), std::vector<std::string>(), std::vector<std::string>(), std::string(), std::string()));
    static std::map<std::string, Kernel> kernels;
    auto it = kernels.find(v.name);
    if (it == kernels.end()) {
        const std::string header = std::string("#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                                               "typedef ") + v.ta + " TA; typedef " + v.tb + " TB; typedef " + v.tc + " TC;\n";
        it = kernels.emplace(v.name, mx::fast::metal_kernel(std::string("tqllm_mpp_types_") + v.name, {"x", "w"}, {"y"}, R"METAL(
uint tid = thread_position_in_threadgroup.x;
uint n0 = threadgroup_position_in_grid.x * uint(NT), m0 = threadgroup_position_in_grid.y * uint(MT);
threadgroup TA A[MT * BK];
threadgroup TB B[NT * BK];
tensor<threadgroup TA, extents<int, BK, MT>, tensor_inline> ta(A, extents<int, BK, MT>());
tensor<threadgroup TB, extents<int, BK, NT>, tensor_inline> tb(B, extents<int, BK, NT>());
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(MT, NT, BK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SIMDS>> op;
auto acc = op.template get_destination_cooperative_tensor<decltype(ta), decltype(tb), TC>();
for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = TC(0);
for (uint base = 0; base < uint(K); base += uint(BK)) {
    for (uint i = tid; i < uint(MT * BK); i += 32u * uint(SIMDS)) {
        uint row = i / uint(BK), col = i % uint(BK);
        A[i] = m0 + row < uint(M) ? TA(x[(m0 + row) * uint(K) + base + col]) : TA(0);
    }
    for (uint i = tid; i < uint(NT * BK); i += 32u * uint(SIMDS)) {
        uint row = i / uint(BK), col = i % uint(BK);
        B[i] = n0 + row < uint(N) ? TB(w[(n0 + row) * uint(K) + base + col]) : TB(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(ta, tb, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto c = acc.get_multidimensional_index(i);
    uint col = n0 + uint(c[0]), row = m0 + uint(c[1]);
    if (row < uint(M) && col < uint(N)) y[row * uint(N) + col] = float(acc[i]);
}
)METAL", header)).first;
    }
    return it->second({x, w}, {{m, n}}, {mx::float32}, {((n + nt - 1) / nt) * 32 * simds, (m + mt - 1) / mt, 1}, {32 * simds, 1, 1},
                      {{"MT", mt}, {"NT", nt}, {"BK", bk}, {"SIMDS", simds}, {"M", m}, {"N", n}, {"K", k}},
                      std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

mx::array compute_only(const Variant& v, const mx::array& x, const mx::array& w, int mt, int nt, int bk, int simds, int reps) {
    const int m = x.shape(0), n = w.shape(0);
    using Kernel = decltype(mx::fast::metal_kernel(std::string(), std::vector<std::string>(), std::vector<std::string>(), std::string(), std::string()));
    static std::map<std::string, Kernel> kernels;
    auto it = kernels.find(v.name);
    if (it == kernels.end()) {
        const std::string header = std::string("#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                                               "typedef ") + v.ta + " TA; typedef " + v.tb + " TB; typedef " + v.tc + " TC;\n";
        it = kernels.emplace(v.name, mx::fast::metal_kernel(std::string("tqllm_mpp_compute_") + v.name, {"x", "w"}, {"y"}, R"METAL(
uint tid = thread_position_in_threadgroup.x;
uint n0 = threadgroup_position_in_grid.x * uint(NT), m0 = threadgroup_position_in_grid.y * uint(MT);
threadgroup TA A[MT * BK];
threadgroup TB B[NT * BK];
tensor<threadgroup TA, extents<int, BK, MT>, tensor_inline> ta(A, extents<int, BK, MT>());
tensor<threadgroup TB, extents<int, BK, NT>, tensor_inline> tb(B, extents<int, BK, NT>());
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(MT, NT, BK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SIMDS>> op;
auto acc = op.template get_destination_cooperative_tensor<decltype(ta), decltype(tb), TC>();
for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = TC(0);
for (uint i = tid; i < uint(MT * BK); i += 32u * uint(SIMDS)) A[i] = TA(x[(m0 + i / uint(BK)) * uint(BK) + i % uint(BK)]);
for (uint i = tid; i < uint(NT * BK); i += 32u * uint(SIMDS)) B[i] = TB(w[(n0 + i / uint(BK)) * uint(BK) + i % uint(BK)]);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint r = 0; r < uint(REPS); ++r) op.run(ta, tb, acc);
for (uint i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto c = acc.get_multidimensional_index(i);
    y[(m0 + uint(c[1])) * uint(N) + n0 + uint(c[0])] = float(acc[i]);
}
)METAL", header)).first;
    }
    return it->second({x, w}, {{m, n}}, {mx::float32}, {(n / nt) * 32 * simds, m / mt, 1}, {32 * simds, 1, 1},
                      {{"MT", mt}, {"NT", nt}, {"BK", bk}, {"SIMDS", simds}, {"N", n}, {"REPS", reps}},
                      std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

// staged GEMM with 16-byte loads (no bounds checks: M, N multiples of the tile, K of BK); int8 variants may apply a
// per-step, per-column float rescale of the int32 tile (exact per-group weight scales, group = BK)
mx::array staged16(const Variant& v, const mx::array& x, const mx::array& w, const mx::array& scales, int mt, int nt, int bk, int simds, bool rescale) {
    const int m = x.shape(0), k = x.shape(1), n = w.shape(0);
    using Kernel = decltype(mx::fast::metal_kernel(std::string(), std::vector<std::string>(), std::vector<std::string>(), std::string(), std::string()));
    static std::map<std::string, Kernel> kernels;
    const std::string key = std::string(v.name) + (rescale ? "_rs" : "");
    auto it = kernels.find(key);
    if (it == kernels.end()) {
        const std::string header = std::string("#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                                               "typedef ") + v.ta + " TA; typedef " + v.tb + " TB; typedef " + v.tc + " TC;\n#define RESCALE " + (rescale ? "1" : "0") + "\n";
        it = kernels.emplace(key, mx::fast::metal_kernel(std::string("tqllm_mpp_staged16_") + key, {"x", "w", "s"}, {"y"}, R"METAL(
uint tid = thread_position_in_threadgroup.x;
uint n0 = threadgroup_position_in_grid.x * uint(NT), m0 = threadgroup_position_in_grid.y * uint(MT);
threadgroup TA A[MT * BK];
threadgroup TB B[NT * BK];
threadgroup half S[NT];
tensor<threadgroup TA, extents<int, BK, MT>, tensor_inline> ta(A, extents<int, BK, MT>());
tensor<threadgroup TB, extents<int, BK, NT>, tensor_inline> tb(B, extents<int, BK, NT>());
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(MT, NT, BK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SIMDS>> op;
auto acc = op.template get_destination_cooperative_tensor<decltype(ta), decltype(tb), TC>();
constexpr uint CAP = uint(MT) * uint(NT) / (32u * uint(SIMDS));
for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = TC(0);
#if RESCALE
float facc[CAP];
ushort colidx[CAP];
for (uint i = 0; i < acc.get_capacity(); ++i) {facc[i] = 0.0f; auto c = acc.get_multidimensional_index(i); colidx[i] = ushort(c[0]);}
#endif
constexpr uint ROWB_A = uint(BK) * sizeof(TA) / 16u, ROWB_B = uint(BK) * sizeof(TB) / 16u;  // 16-byte chunks per tile row
for (uint base = 0; base < uint(K); base += uint(BK)) {
    for (uint i = tid; i < uint(MT) * ROWB_A; i += 32u * uint(SIMDS)) {
        uint row = i / ROWB_A, chunk = i % ROWB_A;
        ((threadgroup uint4*)A)[i] = *(const device uint4*)((const device char*)x + ((m0 + row) * uint(K) + base) * sizeof(TA) + chunk * 16u);
    }
    for (uint i = tid; i < uint(NT) * ROWB_B; i += 32u * uint(SIMDS)) {
        uint row = i / ROWB_B, chunk = i % ROWB_B;
        ((threadgroup uint4*)B)[i] = *(const device uint4*)((const device char*)w + ((n0 + row) * uint(K) + base) * sizeof(TB) + chunk * 16u);
    }
#if RESCALE
    for (uint i = tid; i < uint(NT); i += 32u * uint(SIMDS)) S[i] = s[(n0 + i) * (uint(K) / uint(BK)) + base / uint(BK)];
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(ta, tb, acc);
#if RESCALE
    for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {facc[i] += float(acc[i]) * float(S[colidx[i]]); acc[i] = TC(0);}
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto c = acc.get_multidimensional_index(i);
#if RESCALE
    y[(m0 + uint(c[1])) * uint(N) + n0 + uint(c[0])] = facc[i];
#else
    y[(m0 + uint(c[1])) * uint(N) + n0 + uint(c[0])] = float(acc[i]);
#endif
}
)METAL", header)).first;
    }
    return it->second({x, w, scales}, {{m, n}}, {mx::float32}, {(n / nt) * 32 * simds, m / mt, 1}, {32 * simds, 1, 1},
                      {{"MT", mt}, {"NT", nt}, {"BK", bk}, {"SIMDS", simds}, {"N", n}, {"K", k}},
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
mx::array operand(const std::string& type, mx::Shape shape) {
    // small integers so every reference product is exact in float32
    auto ints = mx::random::randint(-8, 8, shape);
    return type == "int8_t" ? mx::astype(ints, mx::int8) : mx::astype(ints, mx::float16);
}
}  // namespace

int main() {
    const int M = 2048, K = 5120;
    const std::vector<Variant> variants{{"hh_f", "half", "half", "float"}, {"ii_i", "int8_t", "int8_t", "int32_t"},
                                        {"hi_f", "half", "int8_t", "float"}, {"ih_f", "int8_t", "half", "float"}};
    for (int N : {17408, 5120}) {
        const double flop = 2.0 * M * N * K;
        for (const auto& v : variants) {
            auto x = operand(v.ta, {M, K}), w = operand(v.tb, {N, K});
            mx::eval(x, w);
            auto ref = mx::matmul(mx::astype(x, mx::float32), mx::transpose(mx::astype(w, mx::float32)));
            struct T { int mt, nt, bk, simds; };
            for (T t : {T{64, 64, 32, 4}, T{128, 128, 32, 4}, T{64, 128, 64, 4}}) {
                try {
                    auto y = tiled(v, x, w, t.mt, t.nt, t.bk, t.simds);
                    auto err = mx::max(mx::abs(mx::subtract(y, ref)));
                    mx::eval(err);
                    const double ms = best_ms([&] { return tiled(v, x, w, t.mt, t.nt, t.bk, t.simds); });
                    std::printf("MPP_TYPES N=%d %s mt=%d nt=%d bk=%d simds=%d %.2f ms %.1f TOPS max_abs_err %.1f\n", N, v.name, t.mt, t.nt,
                                t.bk, t.simds, ms, flop / ms * 1e-9, err.item<float>());
                } catch (const std::exception& e) {
                    std::printf("MPP_TYPES N=%d %s mt=%d nt=%d bk=%d simds=%d ERROR %s\n", N, v.name, t.mt, t.nt, t.bk, t.simds,
                                std::string(e.what()).substr(0, 400).c_str());
                }
            }
        }
    }
    // compute-only: the matrix unit's rate per operand type (one staged 32-deep tile, REPS accumulating runs)
    for (const auto& v : variants) {
        const int Mc = 2048, Nc = 17408, bk = 32, reps = 160;
        auto x = operand(v.ta, {Mc, bk}), w = operand(v.tb, {Nc, bk});
        mx::eval(x, w);
        auto ref = mx::multiply(mx::matmul(mx::astype(x, mx::float32), mx::transpose(mx::astype(w, mx::float32))), mx::array(float(reps)));
        struct T { int mt, nt, simds; };
        for (T t : {T{64, 64, 4}, T{32, 32, 1}, T{64, 32, 2}, T{128, 64, 4}, T{32, 64, 1}}) {
            try {
                auto y = compute_only(v, x, w, t.mt, t.nt, bk, t.simds, reps);
                auto err = mx::max(mx::abs(mx::subtract(y, ref)));
                mx::eval(err);
                const double ms = best_ms([&] { return compute_only(v, x, w, t.mt, t.nt, bk, t.simds, reps); });
                std::printf("MPP_COMPUTE %s mt=%d nt=%d bk=%d simds=%d %.2f ms %.1f TOPS max_abs_err %.1f\n", v.name, t.mt, t.nt, bk, t.simds, ms,
                            2.0 * Mc * Nc * bk * reps / ms * 1e-9, err.item<float>());
            } catch (const std::exception& e) {
                std::printf("MPP_COMPUTE %s mt=%d nt=%d simds=%d ERROR %s\n", v.name, t.mt, t.nt, t.simds, std::string(e.what()).substr(0, 300).c_str());
            }
        }
    }
    // staged (16-byte loads) GEMMs, the practical rate; int8 with the exact per-group rescale (group = BK) as well
    {
        const int Ms = 2048, Ks = 5120, Ns = 17408, bk = 32;
        const double flop = 2.0 * Ms * Ns * Ks;
        struct R { Variant v; bool rescale; };
        for (R r : {R{variants[0], false}, R{variants[1], false}, R{variants[1], true}}) {
            auto x = operand(r.v.ta, {Ms, Ks}), w = operand(r.v.tb, {Ns, Ks});
            auto sc = mx::astype(mx::add(mx::random::uniform({Ns, Ks / bk}), mx::array(0.5f)), mx::float16);
            mx::eval(x, w, sc);
            auto ref = r.rescale
                ? mx::sum(mx::multiply(mx::matmul(mx::reshape(mx::astype(x, mx::float32), {Ms, Ks / bk, 1, bk}), mx::transpose(mx::reshape(mx::astype(w, mx::float32), {1, Ns, Ks / bk, bk}), {0, 2, 3, 1})) , mx::astype(mx::transpose(sc), mx::float32)), 1)
                : mx::matmul(mx::astype(x, mx::float32), mx::transpose(mx::astype(w, mx::float32)));
            struct T { int mt, nt, simds; };
            for (T t : {T{64, 64, 4}, T{32, 32, 1}, T{64, 32, 2}, T{128, 64, 4}, T{64, 128, 4}}) {
                try {
                    auto y = staged16(r.v, x, w, sc, t.mt, t.nt, bk, t.simds, r.rescale);
                    auto err = mx::divide(mx::max(mx::abs(mx::subtract(y, ref))), mx::max(mx::abs(ref)));
                    mx::eval(err);
                    const double ms = best_ms([&] { return staged16(r.v, x, w, sc, t.mt, t.nt, bk, t.simds, r.rescale); });
                    std::printf("MPP_STAGED %s%s mt=%d nt=%d bk=%d simds=%d %.2f ms %.1f TOPS rel_err %.1e\n", r.v.name, r.rescale ? "_rescale" : "", t.mt, t.nt, bk,
                                t.simds, ms, flop / ms * 1e-9, err.item<float>());
                } catch (const std::exception& e) {
                    std::printf("MPP_STAGED %s%s mt=%d nt=%d simds=%d ERROR %s\n", r.v.name, r.rescale ? "_rescale" : "", t.mt, t.nt, t.simds, std::string(e.what()).substr(0, 300).c_str());
                }
            }
        }
    }
    std::puts("MPP_TYPES_DONE");
}
