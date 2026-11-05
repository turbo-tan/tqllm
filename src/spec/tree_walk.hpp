#pragma once
// The host's check of a device token-tree walk, shared by the backends (no GPU code). Rows are a tree's verify rows: row 0
// the anchor (parent -1), every parent before its row; sibling tokens are distinct, so the target's walk is one path.
#include <cstdint>
#include <vector>

namespace tqllm {
// The path must follow parents and the target's choices, and stop only where no child matches the choice after its last
// row (unless the request's row limit cut it). Returns nullptr, or what is wrong.
inline const char* tree_walk_error(const std::vector<int32_t>& parents, const std::vector<uint32_t>& ids,
                                   const std::vector<uint32_t>& choices, const std::vector<int32_t>& path, bool cut) {
    for (size_t j = 1; j < path.size(); ++j)
        if (parents[size_t(path[j])] != path[j - 1] || ids[size_t(path[j])] != choices[size_t(path[j - 1])])
            return "tree draft-ahead: the device path is not the target's walk";
    if (!cut)
        for (size_t r = 1; r < parents.size(); ++r)
            if (parents[r] == path.back() && ids[r] == choices[size_t(path.back())])
                return "tree draft-ahead: the device path stopped before the target's walk";
    return nullptr;
}
}  // namespace tqllm
