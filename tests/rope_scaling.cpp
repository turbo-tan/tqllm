// RoPE scaling table (include/tqllm/rope_scaling.hpp) against ggml's rope_yarn for Qwen3.8-27B (rot 64, base 1e7,
// trained 262144): YaRN x4 to 1M.
#include "tqllm/rope_scaling.hpp"

#include <cmath>
#include <cstdio>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } } while (0)

int main() {
    using namespace tqllm;
    const RopeTable off = rope_scaling_table(RopeScaling{}, 64, 1e7f, 262144);
    bool ones = off.mscale == 1.f;
    for (float m : off.mul) ones = ones && m == 1.f;
    CHECK(ones && off.mul.size() == 32, "off: every factor exactly 1 (kernels stay bit-identical)");
    RopeScaling y;
    y.type = "yarn", y.factor = 4.f;
    const RopeTable t = rope_scaling_table(y, 64, 1e7f, 262144);
    // ggml corr dims: n_dims * ln(n_ctx_orig / (n_rot * 2 pi)) / (2 ln base) for beta 32 / 1 -> floor / ceil
    const float lo = std::floor(64.f * std::log(262144.f / (32.f * 2.f * float(M_PI))) / (2.f * std::log(1e7f)));
    const float hi = std::ceil(64.f * std::log(262144.f / (1.f * 2.f * float(M_PI))) / (2.f * std::log(1e7f)));
    CHECK(lo == 14.f && hi == 22.f, "corr dims %g %g (expected 14 22)", lo, hi);
    CHECK(t.mul[0] == 1.f && t.mul[14] == 1.f, "high-frequency pairs extrapolate (factor 1)");
    CHECK(std::fabs(t.mul[31] - 0.25f) < 1e-7f && std::fabs(t.mul[22] - 0.25f) < 1e-7f, "low-frequency pairs interpolate (1/4)");
    CHECK(std::fabs(t.mul[18] - (0.25f * 0.5f + 0.5f)) < 1e-6f, "ramp midpoint blends half and half");
    for (int d = 1; d < 32; ++d) CHECK(t.mul[size_t(d)] <= t.mul[size_t(d - 1)], "factors fall monotonically (pair %d)", d);
    CHECK(std::fabs(t.mscale - (1.f + 0.1f * std::log(4.f))) < 1e-6f, "YaRN cos/sin scale 1 + 0.1 ln 4");
    RopeScaling lin;
    lin.type = "linear", lin.factor = 2.f;
    const RopeTable l = rope_scaling_table(lin, 64, 1e7f, 262144);
    CHECK(l.mul[0] == 0.5f && l.mul[31] == 0.5f && l.mscale == 1.f, "linear: every pair 1/factor, no scale");
    RopeScaling noext = y;
    noext.ext_factor = 0.f;
    const RopeTable n = rope_scaling_table(noext, 64, 1e7f, 262144);
    CHECK(n.mul[0] == 0.25f && n.mscale == 1.f, "ext_factor 0: pure interpolation, no scale (ggml)");
    if (fails) std::printf("rope_scaling: %d failure(s)\n", fails);
    else std::printf("rope_scaling: ok (yarn x4: pairs 0-14 x1, 22-31 x0.25, scale %.4f)\n", t.mscale);
    return fails ? 1 : 0;
}
