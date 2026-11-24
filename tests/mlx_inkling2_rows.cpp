// Row invariance of the Inkling-2 decode / verify products (inkling2_rows.hpp): row j of an 8-row call must equal the
// 1-row call on that row bit for bit (the spec == plain contract), and each product must match MLX's within rounding.
// MLX's own products fail the first check (gemv vs gemm, qmv vs sorted gather_qmm): printed as the reason for the kernels.
#include "inkling2_rows.hpp"
#include "inkling2_mlx_ops.hpp"
#include "mlx_floatx.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
int failures = 0;
bool same_bits(const mx::array& a, const mx::array& b) {
    auto x = mx::contiguous(a), y = mx::contiguous(b);
    mx::eval({x, y});
    return x.nbytes() == y.nbytes() && std::memcmp(x.data<void>(), y.data<void>(), x.nbytes()) == 0;
}
double rel(const mx::array& a, const mx::array& b) {
    auto d = mx::astype(mx::subtract(mx::astype(a, mx::float32), mx::astype(b, mx::float32)), mx::float32);
    auto e = mx::sqrt(mx::divide(mx::sum(mx::square(d)), mx::add(mx::sum(mx::square(mx::astype(b, mx::float32))), mx::array(1e-30f))));
    return e.item<float>();
}
mx::array row(const mx::array& x, int r) { return mx::slice(x, {r, 0}, {r + 1, x.shape(1)}); }
void check(const char* what, bool ok, double err, double tol) {
    const bool pass = ok && err <= tol;
    std::printf("%s %s row_invariant %d rel_vs_mlx %.3g (tol %.3g)\n", pass ? "PASS" : "FAIL", what, int(ok), err, tol);
    failures += !pass;
}
}  // namespace
int main() {
    mx::set_default_device(mx::Device::gpu);
    mx::random::seed(3);
    const int n = 8, K = 2304, O = 1152;
    auto x = mx::astype(mx::random::normal({n, K}), mx::bfloat16);
    {   // dense BF16
        auto w = mx::astype(mx::multiply(mx::random::normal({O, K}), mx::array(0.02f)), mx::bfloat16);
        auto all = inkling2_rows_linear(x, w);
        bool inv = true;
        for (int r = 0; r < n; ++r) inv = inv && same_bits(row(all, r), inkling2_rows_linear(row(x, r), w));
        check("linear_bf16", inv, rel(all, mx::matmul(x, mx::transpose(w))), 1e-2);
        bool mlx_inv = true;
        auto ref = mx::matmul(x, mx::transpose(w));
        for (int r = 0; r < n; ++r) mlx_inv = mlx_inv && same_bits(row(ref, r), mx::matmul(row(x, r), mx::transpose(w)));
        std::printf("INFO mlx matmul row_invariant %d\n", int(mlx_inv));
    }
    {   // a narrow product (a controller's down product, 64 x 4608): K split over simdgroups
        auto xn = mx::astype(mx::random::normal({n, 4608}), mx::bfloat16);
        auto w = mx::astype(mx::multiply(mx::random::normal({64, 4608}), mx::array(0.02f)), mx::bfloat16);
        auto all = inkling2_rows_linear(xn, w);
        bool inv = true;
        for (int r = 0; r < n; ++r) inv = inv && same_bits(row(all, r), inkling2_rows_linear(row(xn, r), w));
        check("linear_narrow_split", inv, rel(all, mx::matmul(xn, mx::transpose(w))), 1e-2);
    }
    {   // dense F32 (the router)
        auto w = mx::multiply(mx::random::normal({128, K}), mx::array(0.02f));
        auto xf = mx::astype(x, mx::float32);
        auto all = inkling2_rows_linear(xf, w);
        bool inv = true;
        for (int r = 0; r < n; ++r) inv = inv && same_bits(row(all, r), inkling2_rows_linear(row(xf, r), w));
        check("linear_f32", inv, rel(all, mx::matmul(xf, mx::transpose(w))), 1e-5);
    }
    for (int bits : {8, 4}) {  // dense affine (--exact off)
        auto w = mx::astype(mx::multiply(mx::random::normal({O, K}), mx::array(0.02f)), mx::bfloat16);
        auto q = mx::quantize(w, 64, bits);
        auto all = inkling2_rows_affine(x, q[0], q[1], q[2], 64, bits);
        bool inv = true;
        for (int r = 0; r < n; ++r) inv = inv && same_bits(row(all, r), inkling2_rows_affine(row(x, r), q[0], q[1], q[2], 64, bits));
        check(bits == 8 ? "affine8" : "affine4", inv, rel(all, mx::quantized_matmul(x, q[0], q[1], q[2], true, 64, bits)), 1e-2);
    }
    for (int bits : {4, 8}) {  // routed experts, 8 tokens x 8 experts out of 32, sorted by expert
        const int E = 32, topk = 8, I = 576, D = 1152;
        auto codes = mx::astype(mx::random::randint(0, 256, {E, 2 * I, D * bits / 8}), mx::uint8);
        if (bits == 8) codes = mx::bitwise_and(codes, mx::array(uint8_t(0xf7)));  // no E4M3 NaN codes (0x7f, 0xff)
        auto scales = mx::astype(mx::random::randint(bits == 4 ? 0x30 : 120, bits == 4 ? 0x40 : 128, {E, 2 * I, D / (bits == 4 ? 16 : 32)}), mx::uint8);
        std::optional<mx::array> outer;
        if (bits == 4) outer = mx::full({E}, 0.5f, mx::float32);
        FloatxExperts gu(codes, scales, outer, bits);
        auto xe = mx::astype(mx::random::normal({n, D}), mx::bfloat16);
        std::vector<uint32_t> ids;
        for (int t = 0; t < n; ++t) for (int k = 0; k < topk; ++k) ids.push_back(uint32_t((t * 3 + k * 5) % E));
        auto idarr = mx::array(ids.data(), {n, topk}, mx::uint32);
        {
        const int impl = 0;
        auto block = [&](const mx::array& xs, const mx::array& route) {
            const int P = route.size();
            auto sorting = inkling2_pair_sort(route);
            auto sorted = mx::reshape(mx::slice(sorting, {0, 0}, {1, P}), {P});
            auto token = mx::reshape(mx::slice(sorting, {1, 0}, {2, P}), {P});
            auto restore = mx::reshape(mx::slice(sorting, {2, 0}, {3, P}), {P});
            auto y = inkling2_rows_experts(xs, token, sorted, gu.codes, gu.scales, gu.global, bits);
            return mx::take(y, restore, 0);  // route order [P, 2I]
        };
        auto all = block(xe, idarr);
        bool inv = true;
        for (int t = 0; t < n; ++t) {
            auto one = block(row(xe, t), mx::slice(idarr, {t, 0}, {t + 1, topk}));
            inv = inv && same_bits(mx::slice(all, {t * topk, 0}, {(t + 1) * topk, 2 * I}), one);
        }
        auto ref = mx::reshape(mx::squeeze(gu(mx::reshape(xe, {n, 1, 1, D}), idarr), 2), {n * topk, 2 * I});
        if (impl == 0) {  // gather_qmm unsorted (each token-expert pair a matrix-vector product): the decode path at n rows
            bool ginv = true;
            for (int t = 0; t < n; ++t) {
                auto one = mx::reshape(mx::squeeze(gu(mx::reshape(row(xe, t), {1, 1, 1, D}), mx::slice(idarr, {t, 0}, {t + 1, topk})), 2), {topk, 2 * I});
                ginv = ginv && same_bits(mx::slice(ref, {t * topk, 0}, {(t + 1) * topk, 2 * I}), one);
            }
            check(bits == 4 ? "gather_qmm_unsorted_nvfp4" : "gather_qmm_unsorted_mxfp8", ginv, 0.0, 1.0);
        }
        check(bits == 4 ? "experts_nvfp4" : "experts_mxfp8", inv, rel(all, ref), 1e-2);
        }
        // the fused capped SwiGLU against the same math on the unfused rows
        auto sorting = inkling2_pair_sort(idarr);
        const int P = n * topk;
        auto sorted = mx::reshape(mx::slice(sorting, {0, 0}, {1, P}), {P});
        auto token = mx::reshape(mx::slice(sorting, {1, 0}, {2, P}), {P});
        auto raw = mx::astype(inkling2_rows_experts(xe, token, sorted, gu.codes, gu.scales, gu.global, bits), mx::float32);
        auto g = mx::slice(raw, {0, 0}, {P, 2 * I}, {1, 2}), u = mx::slice(raw, {0, 1}, {P, 2 * I}, {1, 2});
        const float gc = 4.f, uc = 25.f;
        auto act = mx::multiply(mx::multiply(mx::multiply(mx::tanh(mx::divide(g, mx::array(gc))), mx::array(gc)), mx::sigmoid(g)),
                                mx::multiply(mx::tanh(mx::divide(u, mx::array(uc))), mx::array(uc)));
        auto fused = inkling2_rows_experts(xe, token, sorted, gu.codes, gu.scales, gu.global, bits, true, gc, uc);
        check(bits == 4 ? "swiglu_nvfp4" : "swiglu_mxfp8", true, rel(fused, act), 1e-2);
    }
    {   // attention: a verify block of 8 rows vs each row as a decode step over its own cache view (window trimmed)
        const int H = 24, G = 8, HD = 128, DR = 16, E = 1024, window = 1024;
        // full = 2: 509 keys before the block, so a decode step's span is 2 chunks (one fused kernel) and the block's 3
        // (chunk kernel + merge kernel): both must give the same bits
        for (int full : {0, 1, 2}) {
            const int before = full == 2 ? 509 : 1500, start0 = 0;  // keys already cached at positions 0 .. before - 1
            const int S = before + n;
            auto zero = mx::zeros({1, G, 1, HD}, mx::bfloat16);  // the cache's zero sink row after the last key
            auto keys = mx::concatenate({mx::astype(mx::random::normal({1, G, S, HD}), mx::bfloat16), zero}, 2);
            auto vals = mx::concatenate({mx::astype(mx::random::normal({1, G, S, HD}), mx::bfloat16), zero}, 2);
            auto q = mx::astype(mx::multiply(mx::random::normal({1, H, n, HD}), mx::array(0.3f)), mx::bfloat16);
            auto relv = mx::astype(mx::random::normal({n, H, DR}), mx::float32);
            auto proj = mx::astype(mx::multiply(mx::random::normal({DR, E}), mx::array(0.2f)), mx::bfloat16);
            auto sinks = mx::astype(mx::random::normal({n, H}), mx::bfloat16);
            const int win = full ? 0 : window;
            auto all = inkling2_rows_attention(q, keys, vals, relv, proj, sinks, before, start0, S, win, 1.f / std::sqrt(float(HD)));
            bool inv = true;
            for (int r = 0; r < n; ++r) {
                // decode step r: the cache holds positions up to before + r, trimmed to the window (MlxKvCache)
                const int end = before + r + 1, start = win ? std::max(0, end - win) : 0;
                auto kv = mx::slice(keys, {0, 0, start, 0}, {1, G, end + 1, HD}), vv = mx::slice(vals, {0, 0, start, 0}, {1, G, end + 1, HD});
                auto one = inkling2_rows_attention(mx::slice(q, {0, 0, r, 0}, {1, H, r + 1, HD}), kv, vv,
                                                   mx::slice(relv, {r, 0, 0}, {r + 1, H, DR}), proj, mx::slice(sinks, {r, 0}, {r + 1, H}),
                                                   before + r, start, end - start, win, 1.f / std::sqrt(float(HD)));
                inv = inv && same_bits(mx::slice(all, {0, 0, r, 0}, {1, H, r + 1, HD}), one);
            }
            // reference: the bias by distance, the mask kernel and MLX's attention
            auto bias = mx::matmul(mx::astype(relv, mx::bfloat16), proj);  // [n, H, E]
            auto mask = inkling2_mask(bias, sinks, before, start0, S, win);
            auto ref = inkling2_sink_attention(q, keys, vals, mask, 1.f / std::sqrt(float(HD)));
            check(full == 2 ? "attention_full_two_kernels" : full ? "attention_full" : "attention_window", inv, rel(all, ref), 2e-2);
        }
    }
    std::printf("%s\n", failures ? "INKLING2_ROWS FAIL" : "INKLING2_ROWS PASS");
    return failures ? 1 : 0;
}
