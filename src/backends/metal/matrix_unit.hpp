#pragma once
// The M5 GPU matrix unit: the NAX weight layouts (nax_affine.hpp) and the MPP tensor-op attention kernels assume its lane
// order. On macOS 26 earlier chips (M1-M4) also compile and run tensor ops, with other values (issue #38: an M4 Pro
// answered random tokens), so a kernel that merely runs is not proof: the GPU generation decides, and on an M5 the
// kernels must also match CPU references once at load (issue #24: macOS 27 on an M5 / M5 Max gave wrong values).
#include "debug.hpp"
#include <mlx/mlx.h>
#include <atomic>
#include <cstdio>
#include <string>
#include <string_view>
#include <variant>

namespace tqllm::mlx_backend {
// Apple GPU generation from MLX's architecture string ("applegpu_g17s" -> 17, M5); 0 when it has none
inline int apple_gpu_generation(std::string_view arch) {
    const auto at = arch.find("_g");
    if (at == std::string_view::npos) return 0;
    int g = 0, digits = 0;
    for (size_t i = at + 2; i < arch.size() && arch[i] >= '0' && arch[i] <= '9' && digits < 4; ++i, ++digits) g = g * 10 + (arch[i] - '0');
    return digits ? g : 0;
}
inline constexpr int kMatrixUnitGeneration = 17;  // M5
struct MatrixUnitCheck {bool ok = true; const char* failed = ""; double error = 0;};
// the NAX matmul (1 / 16 rows), the NAX prefill GEMM and the tensor-op exact attention vs CPU references (matrix_unit_check.cpp)
MatrixUnitCheck matrix_unit_check();
// set when a model's own matrix-unit products disagree with their stored values (#24): every later m5_matrix_unit() is false
inline std::atomic<bool>& matrix_unit_vetoed() {static std::atomic<bool> v{false}; return v;}
inline void veto_matrix_unit(const char* why) {
    if (!matrix_unit_vetoed().exchange(true))
        std::fprintf(stderr, "tqllm: the GPU matrix unit gave wrong values on this model's weights (%s); using the M1-M4 paths "
                             "instead (correct, slower). Please report your macOS version at "
                             "https://github.com/turbo-tan/tqllm/issues/24\n", why);
}
// TQLLM_DEBUG=no_matrix_unit: the M1-M4 paths on an M5 (to check them where no older chip is at hand)
inline bool m5_matrix_unit() {
    static const bool ok = [] {
        if (tqllm::debug_on("no_matrix_unit")) return false;
        try {
            const auto info = mlx::core::device_info(mlx::core::Device::gpu);
            const auto it = info.find("architecture");
            if (it == info.end()) return false;
            const auto* arch = std::get_if<std::string>(&it->second);
            if (!arch || apple_gpu_generation(*arch) < kMatrixUnitGeneration) return false;
        } catch (const std::exception&) {return false;}
        const auto check = matrix_unit_check();
        if (!check.ok)
            std::fprintf(stderr, "tqllm: the GPU matrix unit gave wrong values in the load check (%s, relative error %.3g); "
                                 "using the M1-M4 paths instead (correct, slower). Please report your macOS version at "
                                 "https://github.com/turbo-tan/tqllm/issues/24\n", check.failed, check.error);
        return check.ok;
    }();
    return ok && !matrix_unit_vetoed().load();
}
}  // namespace tqllm::mlx_backend
