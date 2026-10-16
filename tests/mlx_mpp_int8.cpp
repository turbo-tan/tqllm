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
mx::array staged16(const Variant& v, const mx::array& x, const mx::array& w, const mx::array& scales, int mt, int nt, int bk, int simds, bool rescale, bool dbuf = false) {
    const int m = x.shape(0), k = x.shape(1), n = w.shape(0);
    using Kernel = decltype(mx::fast::metal_kernel(std::string(), std::vector<std::string>(), std::vector<std::string>(), std::string(), std::string()));
    static std::map<std::string, Kernel> kernels;
    const std::string key = std::string(v.name) + (rescale ? "_rs" : "") + (dbuf ? "_db" : "");
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
#if DBUF
threadgroup TA A1[MT * BK];
threadgroup TB B1[NT * BK];
tensor<threadgroup TA, extents<int, BK, MT>, tensor_inline> ta1(A1, extents<int, BK, MT>());
tensor<threadgroup TB, extents<int, BK, NT>, tensor_inline> tb1(B1, extents<int, BK, NT>());
constexpr uint NA = (uint(MT) * ROWB_A + 32u * uint(SIMDS) - 1u) / (32u * uint(SIMDS)), NB = (uint(NT) * ROWB_B + 32u * uint(SIMDS) - 1u) / (32u * uint(SIMDS));
uint4 ra[NA], rb[NB];
auto fetch = [&](uint base) {
    for (uint j = 0; j < NA; ++j) {uint i = tid + j * 32u * uint(SIMDS); if (i < uint(MT) * ROWB_A) {uint row = i / ROWB_A, chunk = i % ROWB_A; ra[j] = *(const device uint4*)((const device char*)x + ((m0 + row) * uint(K) + base) * sizeof(TA) + chunk * 16u);}}
    for (uint j = 0; j < NB; ++j) {uint i = tid + j * 32u * uint(SIMDS); if (i < uint(NT) * ROWB_B) {uint row = i / ROWB_B, chunk = i % ROWB_B; rb[j] = *(const device uint4*)((const device char*)w + ((n0 + row) * uint(K) + base) * sizeof(TB) + chunk * 16u);}}
};
auto store = [&](threadgroup TA* Ad, threadgroup TB* Bd) {
    for (uint j = 0; j < NA; ++j) {uint i = tid + j * 32u * uint(SIMDS); if (i < uint(MT) * ROWB_A) ((threadgroup uint4*)Ad)[i] = ra[j];}
    for (uint j = 0; j < NB; ++j) {uint i = tid + j * 32u * uint(SIMDS); if (i < uint(NT) * ROWB_B) ((threadgroup uint4*)Bd)[i] = rb[j];}
};
fetch(0u); store(A, B);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint base = 0; base < uint(K); base += uint(BK)) {
    const bool cur = (base / uint(BK)) & 1u, more = base + uint(BK) < uint(K);
    if (more) fetch(base + uint(BK));
    if (cur) op.run(ta1, tb1, acc); else op.run(ta, tb, acc);
    if (more) {if (cur) store(A, B); else store(A1, B1);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
#else
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
#endif
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

// operands as device tensors (row-major, K inner), one matmul2d run over the whole K per output tile
mx::array device_tensor(const Variant& v, const mx::array& x, const mx::array& w, int mt, int nt, int simds) {
    const int m = x.shape(0), k = x.shape(1), n = w.shape(0);
    using Kernel = decltype(mx::fast::metal_kernel(std::string(), std::vector<std::string>(), std::vector<std::string>(), std::string(), std::string()));
    static std::map<std::string, Kernel> kernels;
    auto it = kernels.find(v.name);
    if (it == kernels.end()) {
        const std::string header = std::string("#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
                                               "typedef ") + v.ta + " TA; typedef " + v.tb + " TB; typedef " + v.tc + " TC;\n";
        it = kernels.emplace(v.name, mx::fast::metal_kernel(std::string("tqllm_mpp_device_") + v.name, {"x", "w"}, {"y"}, R"METAL(
uint n0 = threadgroup_position_in_grid.x * uint(NT), m0 = threadgroup_position_in_grid.y * uint(MT);
auto tx = tensor<const device TA, dextents<int, 2>, tensor_inline>(x + m0 * uint(K), dextents<int, 2>(K, MT));
auto tw = tensor<const device TB, dextents<int, 2>, tensor_inline>(w + n0 * uint(K), dextents<int, 2>(K, NT));
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(MT, NT, metal::dynamic_extent, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SIMDS>> op;
auto acc = op.template get_destination_cooperative_tensor<decltype(tx), decltype(tw), TC>();
for (uint i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = TC(0);
op.run(tx, tw, acc);
for (uint i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto c = acc.get_multidimensional_index(i);
    y[(m0 + uint(c[1])) * uint(N) + n0 + uint(c[0])] = float(acc[i]);
}
)METAL", header)).first;
    }
    return it->second({x, w}, {{m, n}}, {mx::float32}, {(n / nt) * 32 * simds, m / mt, 1}, {32 * simds, 1, 1},
                      {{"MT", mt}, {"NT", nt}, {"SIMDS", simds}, {"N", n}, {"K", k}},
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

int main(int argc, char** argv) {
    const bool all = argc > 1;
    const int M = 2048, K = 5120;
    const std::vector<Variant> variants{{"hh_f", "half", "half", "float"}, {"ii_i", "int8_t", "int8_t", "int32_t"},
                                        {"hi_f", "half", "int8_t", "float"}, {"ih_f", "int8_t", "half", "float"}};
    for (int N : {17408, 5120}) {
        if (!all) break;
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
        if (!all) break;
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
    // staged (16-byte loads) GEMMs at the practical rate: tile depth, double buffering, device-tensor operands;
    // int8 with the exact per-group rescale (group = BK) as well
    {
        const int Ms = 2048, Ks = 5120, Ns = 17408;
        const double flop = 2.0 * Ms * Ns * Ks;
        struct R { Variant v; bool rescale; bool dbuf; int bk; };
        std::vector<R> runs;
        for (int bk : {32, 64, 128}) {runs.push_back({variants[0], false, false, bk}); runs.push_back({variants[1], false, false, bk});}
        for (int bk : {32, 64}) {runs.push_back({variants[1], false, true, bk}); runs.push_back({variants[0], false, true, bk});}
        runs.push_back({variants[1], true, false, 32});
        for (const R& r : runs) {
            auto x = operand(r.v.ta, {Ms, Ks}), w = operand(r.v.tb, {Ns, Ks});
            auto sc = mx::astype(mx::add(mx::random::uniform({Ns, Ks / r.bk}), mx::array(0.5f)), mx::float16);
            mx::eval(x, w, sc);
            mx::array ref = mx::zeros({Ms, Ns});
            if (r.rescale) {
                for (int g = 0; g < Ks / r.bk; ++g) {
                    auto part = mx::matmul(mx::astype(mx::slice(x, {0, g * r.bk}, {Ms, (g + 1) * r.bk}), mx::float32),
                                           mx::transpose(mx::astype(mx::slice(w, {0, g * r.bk}, {Ns, (g + 1) * r.bk}), mx::float32)));
                    ref = mx::add(ref, mx::multiply(part, mx::astype(mx::transpose(mx::slice(sc, {0, g}, {Ns, g + 1})), mx::float32)));
                }
            } else ref = mx::matmul(mx::astype(x, mx::float32), mx::transpose(mx::astype(w, mx::float32)));
            mx::eval(ref);
            struct T { int mt, nt, simds; };
            for (T t : {T{64, 64, 4}, T{128, 64, 4}, T{64, 128, 4}, T{128, 128, 8}}) {
                if ((t.mt + t.nt) * r.bk * (r.v.ta[0] == 'h' ? 2 : 1) * (r.dbuf ? 2 : 1) > 32768) continue;
                try {
                    auto y = staged16(r.v, x, w, sc, t.mt, t.nt, r.bk, t.simds, r.rescale, r.dbuf);
                    auto err = mx::divide(mx::max(mx::abs(mx::subtract(y, ref))), mx::max(mx::abs(ref)));
                    mx::eval(err);
                    const double ms = best_ms([&] { return staged16(r.v, x, w, sc, t.mt, t.nt, r.bk, t.simds, r.rescale, r.dbuf); });
                    std::printf("MPP_STAGED %s%s%s mt=%d nt=%d bk=%d simds=%d %.2f ms %.1f TOPS rel_err %.1e\n", r.v.name, r.rescale ? "_rescale" : "",
                                r.dbuf ? "_dbuf" : "", t.mt, t.nt, r.bk, t.simds, ms, flop / ms * 1e-9, err.item<float>());
                } catch (const std::exception& e) {
                    std::printf("MPP_STAGED %s%s%s mt=%d nt=%d bk=%d simds=%d ERROR %s\n", r.v.name, r.rescale ? "_rescale" : "", r.dbuf ? "_dbuf" : "",
                                t.mt, t.nt, r.bk, t.simds, std::string(e.what()).substr(0, 300).c_str());
                }
            }
        }
        for (const auto& v : {variants[0], variants[1]}) {
            auto x = operand(v.ta, {Ms, Ks}), w = operand(v.tb, {Ns, Ks});
            mx::eval(x, w);
            auto ref = mx::matmul(mx::astype(x, mx::float32), mx::transpose(mx::astype(w, mx::float32)));
            struct T { int mt, nt, simds; };
            for (T t : {T{64, 64, 4}, T{32, 32, 1}, T{64, 64, 1}, T{128, 128, 4}}) {
                try {
                    auto y = device_tensor(v, x, w, t.mt, t.nt, t.simds);
                    auto err = mx::divide(mx::max(mx::abs(mx::subtract(y, ref))), mx::max(mx::abs(ref)));
                    mx::eval(err);
                    const double ms = best_ms([&] { return device_tensor(v, x, w, t.mt, t.nt, t.simds); });
                    std::printf("MPP_DEVICE %s mt=%d nt=%d simds=%d %.2f ms %.1f TOPS rel_err %.1e\n", v.name, t.mt, t.nt, t.simds, ms, flop / ms * 1e-9, err.item<float>());
                } catch (const std::exception& e) {
                    std::printf("MPP_DEVICE %s mt=%d nt=%d simds=%d ERROR %s\n", v.name, t.mt, t.nt, t.simds, std::string(e.what()).substr(0, 400).c_str());
                }
            }
        }
    }
    std::puts("MPP_TYPES_DONE");
}
