// dflash_top16: per row the 16 largest logits, largest first, ties by lower column, equal to a CPU sort; at the draft
// shape (16 x 248320) and at sizes off the 2048-column chunk (40, 2049, 4111), with exact ties across chunks.
#include "dflash.hpp"
#include <mlx/mlx.h>
#include <mlx/random.h>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <set>
#include <vector>

namespace mx = mlx::core;

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what, int n) {if (!ok) {std::printf("FAIL n=%d: %s\n", n, what); ++failures;}};
    mx::random::seed(3);
    for (int n : {40, 2049, 4111, 248320}) {
        const int rows = n == 248320 ? 16 : 5;
        auto logits = mx::multiply(mx::random::normal({rows, n}), mx::array(4.0f));
        if (n != 248320) logits = mx::round(logits);  // many exact ties, incl. at the 16th place and across chunks
        mx::eval(logits);
        auto [ids, values] = tqllm::mlx_backend::dflash_top16(logits);
        mx::eval(ids, values);
        check(ids.shape() == mx::Shape{rows, 16} && ids.dtype() == mx::uint32 && values.dtype() == mx::float32, "shape / dtype", n);
        const float* l = logits.data<float>();
        bool ordered = true, untied_set = true;
        for (int r = 0; r < rows; ++r) {
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            const float* row = l + size_t(r) * n;
            std::partial_sort(order.begin(), order.begin() + 16, order.end(),
                              [&](int a, int b) {return row[a] > row[b] || (row[a] == row[b] && a < b);});
            for (int k = 0; k < 16; ++k)
                ordered = ordered && ids.data<uint32_t>()[r * 16 + k] == uint32_t(order[k]) && values.data<float>()[r * 16 + k] == row[order[k]];
        }
        check(ordered, "ids + values == CPU sort (value desc, column asc)", n);
        if (n == 248320) {  // continuous values: no ties, argpartition gives the same set
            auto part = mx::slice(mx::argpartition(logits, n - 16, -1), {0, n - 16}, {rows, n});
            part = mx::astype(part, mx::uint32);
            mx::eval(part);
            for (int r = 0; r < rows; ++r) {
                std::set<uint32_t> a(part.data<uint32_t>() + r * 16, part.data<uint32_t>() + r * 16 + 16);
                std::set<uint32_t> b(ids.data<uint32_t>() + r * 16, ids.data<uint32_t>() + r * 16 + 16);
                untied_set = untied_set && a == b;
            }
            // reported, not asserted: j780 MLX argpartition's set differed from the exact CPU sort, which the kernel matches
            std::printf("DFLASH_TOP16 argpartition set equal: %s\n", untied_set ? "yes" : "no");
        }
    }
    std::puts(failures ? "DFLASH_TOP16_FAIL" : "DFLASH_TOP16_PASS");
    return failures ? 1 : 0;
}
