#pragma once
// Target KV cache capacity (rows) for an append that needs `need` rows. A request that announced its length (`reserve`:
// prompt + max new tokens + one verify block) gets one allocation of that size, so the cache never regrows: a regrowth
// holds the old and the new cache at once (x2.5 the cache at the x1.5 step: 246K BF16 = 26 GB transient, the 48 GB Mac
// swapped, j1137). Without a reservation: x1.5 geometric growth. Capacity in whole attention segments, at most `context`.
#include <algorithm>

namespace tqllm::mlx_backend {
inline int kv_capacity(int current, int need, int reserve, int context, int segment) {
    auto whole = [segment](int x) {return (x + segment - 1) / segment * segment;};
    if (current >= need) return current;
    const int grown = current + current / 2;
    const int want = std::max({need, reserve, grown, (need + 255) / 256 * 256 + 256});
    return std::min(whole(context), whole(want));
}
}  // namespace tqllm::mlx_backend
