#pragma once
#include <mlx/mlx.h>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

namespace tqllm::mlx_backend {
struct FloatxExperts {
    mlx::core::array codes, scales;
    std::optional<mlx::core::array> global;
    int bits, group;
    // Matrix-unit layout for few-row products (verify blocks), NVFP4 only: see src/backends/metal/floatx_nax.cpp.
    std::optional<mlx::core::array> nax_codes, nax_scales;
    void pack_nax();
    static const mlx::core::array& zero_rows(int k) {
        static std::map<int, mlx::core::array> cache;
        auto it = cache.find(k);
        if (it == cache.end()) {
            auto z = mlx::core::zeros({1, k}, mlx::core::uint32);
            mlx::core::eval(z);
            it = cache.emplace(k, z).first;
        }
        return it->second;
    }
    // y [R, O] = rows of x [T, K] (xrow[r]) times expert ids[r]; rows sharing an expert are consecutive (<= 16).
    mlx::core::array rows(const mlx::core::array& x, const mlx::core::array& xrow, const mlx::core::array& ids) const;

    FloatxExperts(const mlx::core::array& bytes, const mlx::core::array& scale,
                  std::optional<mlx::core::array> outer, int width)
        : codes(mlx::core::view(bytes, mlx::core::uint32)), scales(scale), global(std::move(outer)),
          bits(width), group(width == 4 ? 16 : 32) {
        if ((bits != 4 && bits != 8) || bytes.dtype() != mlx::core::uint8 || bytes.ndim() != 3 ||
            scale.dtype() != mlx::core::uint8 || scale.ndim() != 3 ||
            bytes.shape(0) != scale.shape(0) || bytes.shape(1) != scale.shape(1) ||
            bytes.shape(2) * 8 / bits != scale.shape(2) * group ||
            (bits == 4 && (!global || global->shape() != mlx::core::Shape{bytes.shape(0)})) ||
            (bits == 8 && global))
            throw std::invalid_argument("floatx expert shape/scale contract");
    }

    // sorted: ids are non-decreasing along the flattened batch, which selects the grouped-by-expert kernel.
    mlx::core::array operator()(const mlx::core::array& x, const mlx::core::array& ids, bool sorted = false) const {
        // One token ([1, 1, 1, K] x [1, k] ids): its row index is 0 for every expert; passed explicitly (a cached
        // constant) so gather_qmm does not build an arange each call.
        std::optional<mlx::core::array> lhs;
        if (!sorted && x.ndim() == 4 && x.shape(0) == 1 && x.shape(1) == 1 && ids.ndim() == 2 && ids.shape(0) == 1)
            lhs = zero_rows(ids.shape(1));
        return mlx::core::gather_qmm(x, codes, scales, std::nullopt, lhs, ids,
                                     true, group, bits, bits == 4 ? "nvfp4" : "mxfp8", global, sorted);
    }
};

// Routed rows ordered by expert, so gather_qmm reads each expert's weights once per chunk (prefill).
// order[r] is the flattened (token * topk + slot) index of sorted row r; restore inverts it.
struct ExpertGroups {
    mlx::core::array order, restore, ids;
    explicit ExpertGroups(const mlx::core::array& routes)  // [tokens, topk] expert ids
        : order(mlx::core::argsort(mlx::core::reshape(routes, {-1}), 0)),
          restore(mlx::core::argsort(order, 0)),
          ids(mlx::core::take(mlx::core::reshape(routes, {-1}), order, 0)) {
        if (routes.ndim() != 2) throw std::invalid_argument("expert groups need [tokens, topk] ids");
    }
    // [tokens, D] -> [tokens * topk, 1, D], one row per routed (token, expert) pair in expert order.
    mlx::core::array gather(const mlx::core::array& x, int topk) const {
        namespace mx = mlx::core;
        auto token = mx::floor_divide(order, mx::array(uint32_t(topk), mx::uint32));
        return mx::expand_dims(mx::take(x, token, 0), 1);
    }
    // [tokens * topk, 1, O] in expert order -> [tokens, topk, O] in route order.
    mlx::core::array scatter(const mlx::core::array& y, int tokens, int topk) const {
        namespace mx = mlx::core;
        return mx::reshape(mx::take(y, restore, 0), {tokens, topk, y.shape(-1)});
    }
};
}  // namespace tqllm::mlx_backend
