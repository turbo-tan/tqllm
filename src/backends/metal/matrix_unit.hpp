#pragma once
// The M5 GPU matrix unit: the NAX weight layouts (nax_affine.hpp) and the MPP tensor-op attention kernels assume its lane
// order. On macOS 26 earlier chips (M1-M4) also compile and run tensor ops, with other values (issue #38: an M4 Pro
// answered random tokens), so a kernel that merely runs is not proof; the GPU generation decides.
#include "debug.hpp"
#include <mlx/mlx.h>
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
// TQLLM_DEBUG=no_matrix_unit: the M1-M4 paths on an M5 (to check them where no older chip is at hand)
inline bool m5_matrix_unit() {
    static const bool ok = [] {
        if (tqllm::debug_on("no_matrix_unit")) return false;
        try {
            const auto info = mlx::core::device_info(mlx::core::Device::gpu);
            const auto it = info.find("architecture");
            if (it == info.end()) return false;
            const auto* arch = std::get_if<std::string>(&it->second);
            return arch && apple_gpu_generation(*arch) >= kMatrixUnitGeneration;
        } catch (const std::exception&) {return false;}
    }();
    return ok;
}
}  // namespace tqllm::mlx_backend
