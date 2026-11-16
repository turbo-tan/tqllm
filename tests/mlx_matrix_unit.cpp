// #38: the NAX layouts and the MPP attention kernels run only on the M5 matrix unit (GPU generation 17); M1-M4 load
// NAX-stored weights as MLX affine matrices. Checks the generation rule and that the affine path computes the same
// product as the stored matrix (dequant + GEMM), 4-bit (repacked codes) and 6-bit (requantized).
#include "matrix_unit.hpp"
#include "weights.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <map>
#include <string>
#include "nax_affine.hpp"

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
uint16_t bf16_bits(float v) {uint32_t u; std::memcpy(&u, &v, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);}
std::vector<float> rows_of(const mx::array& y) {
    auto c = mx::contiguous(mx::astype(y, mx::float32)); mx::eval(c);
    return std::vector<float>(c.data<float>(), c.data<float>() + c.size());
}
bool run(int rows, int k, int bits, int group, double tol, std::mt19937& rng) {
    const int gpr = k / group;
    std::vector<uint8_t> codes(size_t(rows) * k);
    for (auto& c : codes) c = uint8_t(rng() & ((1u << bits) - 1u));
    std::uniform_real_distribution<float> sd(0.002f, 0.02f), bd(-0.2f, 0.2f), xd(-2.0f, 2.0f);
    std::vector<uint16_t> s(size_t(rows) * gpr), b(s.size());
    for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(sd(rng)); b[i] = bf16_bits(bd(rng));}
    auto w = nax_affine_load(nax_affine_pack(codes, s, b, rows, k, bits, group), rows, k, bits, group);
    std::vector<float> x(size_t(3) * k);
    for (auto& v : x) v = float(mx::float16_t(xd(rng)));
    auto X = mx::array(x.data(), {3, k}, mx::float32);
    auto ref = rows_of(nax_affine_gemm(w, X, false));
    auto y = rows_of(linear(Matrix{affine_from_nax(w)}, X));
    double worst = 0, peak = 0;
    for (size_t i = 0; i < y.size(); ++i) {peak = std::max(peak, double(std::abs(ref[i]))); worst = std::max(worst, double(std::abs(ref[i] - y[i])));}
    std::printf("affine from NAX rows=%d k=%d bits=%d group=%d rel_err_vs_stored=%.3e\n", rows, k, bits, group, worst / peak);
    return worst / peak < tol;
}
}  // namespace

int main() {
    bool ok = true;
    // M1 g13 .. M4 g16 take the MLX paths; M5 (applegpu_g17s on our M5 Pro) the matrix unit
    for (auto [arch, gen] : std::vector<std::pair<const char*, int>>{{"applegpu_g13g", 13}, {"applegpu_g15s", 15}, {"applegpu_g16s", 16},
                                                                    {"applegpu_g17s", 17}, {"applegpu_g17g", 17}, {"", 0}, {"applegpu", 0}}) {
        const int got = apple_gpu_generation(arch);
        const bool m5 = got >= kMatrixUnitGeneration;
        if (got != gen || m5 != (gen >= 17)) {std::printf("FAIL: %s -> generation %d (want %d)\n", arch, got, gen); ok = false;}
    }
    const auto info = mx::device_info(mx::Device::gpu);
    const auto* arch = std::get_if<std::string>(&info.at("architecture"));
    std::printf("this GPU: %s, M5 matrix unit: %d\n", arch ? arch->c_str() : "?", int(m5_matrix_unit()));
    if (!tqllm::debug_on("no_matrix_unit") && arch && apple_gpu_generation(*arch) >= kMatrixUnitGeneration) {
        // #24: on an M5 the matrix-unit kernels must match the CPU references (else serving falls back to the M1-M4 paths)
        const auto check = matrix_unit_check();
        std::printf("matrix unit load check: %s (worst relative error %.3e%s%s)\n", check.ok ? "ok" : "FAIL", check.error,
                    check.ok ? "" : ", first failure: ", check.failed);
        if (!check.ok || !m5_matrix_unit()) {std::printf("FAIL: the M5 matrix unit is not usable on this system\n"); ok = false;}
    }
    if (m5_matrix_unit()) {
        // #24: the model check on 27B-like shapes (5120 / 17408 columns, split-K, 4- and 6-bit, the first layer's SwiGLU)
        std::mt19937 r2(24);
        auto make = [&](int rows, int k, int bits, int group) {
            std::vector<uint8_t> codes(size_t(rows) * k);
            for (auto& c : codes) c = uint8_t(r2() & ((1u << bits) - 1u));
            std::uniform_real_distribution<float> sd(0.002f, 0.02f), bd(-0.2f, 0.2f);
            std::vector<uint16_t> s(size_t(rows) * (k / group)), b(s.size());
            for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(sd(r2)); b[i] = bf16_bits(bd(r2));}
            return nax_affine_load(nax_affine_pack(codes, s, b, rows, k, bits, group), rows, k, bits, group);
        };
        std::map<std::string, NaxAffine> nax, none;
        nax.emplace("blk.0.ffn_gate.weight", make(512, 5120, 4, 64));
        nax.emplace("blk.0.ffn_up.weight", make(512, 5120, 4, 64));
        nax.emplace("blk.0.ffn_down.weight", make(256, 17408, 4, 64));
        nax.emplace("blk.0.attn_qkv.weight", make(256, 5120, 4, 128));
        nax.emplace("output.weight", make(256, 5120, 6, 64));
        const auto bad = check_model_matrix_unit(nax, none, none);
        std::printf("model matrix-unit check: %s\n", bad.empty() ? "ok" : bad.c_str());
        if (!bad.empty()) ok = false;
    }
    std::mt19937 rng(38);
    ok &= run(256, 512, 4, 64, 2e-3, rng);
    ok &= run(256, 512, 4, 128, 2e-3, rng);
    ok &= run(128, 512, 6, 64, 1e-2, rng);  // requantized: MLX picks its own scale / bias
    std::printf(ok ? "matrix unit: PASS\n" : "matrix unit: FAIL\n");
    return ok ? 0 : 1;
}
