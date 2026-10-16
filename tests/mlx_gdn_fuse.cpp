#include <mlx/mlx.h>
#include "gdn.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
mx::array rnd(std::mt19937& g, mx::Shape s, float scale) {
    size_t n = 1;
    for (int d : s) n *= size_t(d);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(g);
    return mx::array(v.data(), s, mx::float32);
}
mx::array bf(const mx::array& x) {return mx::astype(mx::astype(x, mx::bfloat16), mx::float32);}
// largest difference in bf16 ulps of the reference (values are bf16-rounded or derived from bf16 values)
double ulps(const mx::array& a, const mx::array& b) {
    auto x = mx::contiguous(a), y = mx::contiguous(b);
    mx::eval(x, y);
    double worst = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double r = y.data<float>()[i], d = std::fabs(double(x.data<float>()[i]) - r);
        worst = std::max(worst, d / (std::fabs(r) * 0.0078125 + 1e-6));
    }
    return worst;
}
}  // namespace

int main() {
    std::mt19937 g(5);
    bool ok = true;
    const int kh = 16, vh = 48, width = 2 * kh * 128 + vh * 128;
    for (int m : {1, 8}) {
        auto joined = rnd(g, {m + 3, width}, 1.0f), weights = rnd(g, {width, 4}, 0.5f);
        auto ref = gdn_convolve_native(joined, weights);
        auto q = mx::reshape(mx::slice(ref, {0, 0}, {m, kh * 128}), {1, m, kh, 128});
        auto k = mx::reshape(mx::slice(ref, {0, kh * 128}, {m, 2 * kh * 128}), {1, m, kh, 128});
        auto v = mx::slice(ref, {0, 2 * kh * 128}, {m, width});
        q = gdn_normalize(q, 1.0f / std::sqrt(128.0f));
        k = gdn_normalize(k, 1.0f);
        auto want = mx::concatenate({mx::reshape(q, {m, kh * 128}), mx::reshape(k, {m, kh * 128}), v}, 1);
        auto got = gdn_pre_fused(joined, weights, kh, 1.0f / std::sqrt(128.0f));
        const double e1 = ulps(got, want);
        auto y = rnd(g, {1, m, vh, 128}, 1.0f), z = rnd(g, {1, m, vh, 128}, 1.0f), w = rnd(g, {128}, 0.3f);
        auto want2 = bf(mx::multiply(mx::fast::rms_norm(bf(y), w, 1e-6f), mx::multiply(z, mx::sigmoid(z))));
        auto got2 = gdn_post_fused(y, z, w, 1e-6f);
        const double e2 = ulps(got2, want2);
        // rows are independent: the m=8 call's row 0 equals an m=1 call on that row
        std::printf("GDN_FUSE m=%d pre_max_bf16_ulps=%.3f post_max_bf16_ulps=%.3f\n", m, e1, e2);
        ok = ok && e1 <= 1.01 && e2 <= 1.01;
    }
    std::puts(ok ? "MLX_GDN_FUSE_PASS" : "MLX_GDN_FUSE_FAIL");
    return ok ? 0 : 1;
}
