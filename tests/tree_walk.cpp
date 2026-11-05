// Unit tests for the host's check of a device token-tree walk (src/spec/tree_walk.hpp), no GPU: a full walk and a walk
// cut by the row limit pass; a path off the parents or the target's choices, and a stop before a matching child, fail.
#include <cstdio>
#include <vector>

#include "tree_walk.hpp"

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {std::printf("FAIL %s\n", what); ++failures;}
}
}  // namespace

int main() {
    using tqllm::tree_walk_error;
    // 0 anchor; 1 (0); 2 (0); 3 (2); 4 (2); 5 (4); 6 (1); 7 (5). The target walks 0 -> 2 -> 4 -> 5 and stops (99 matches no child).
    const std::vector<int32_t> parents = {-1, 0, 0, 2, 2, 4, 1, 5};
    const std::vector<uint32_t> ids = {10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<uint32_t> choices = {12, 99, 14, 99, 15, 99, 99, 99};
    expect(tree_walk_error(parents, ids, choices, {0, 2, 4, 5}, false) == nullptr, "the full walk");
    expect(tree_walk_error(parents, ids, choices, {0, 2}, true) == nullptr, "a walk cut by the row limit");
    expect(tree_walk_error(parents, ids, choices, {0, 2}, false) != nullptr, "a stop before a matching child");
    expect(tree_walk_error(parents, ids, choices, {0, 1}, false) != nullptr, "a path off the target's choices");
    expect(tree_walk_error(parents, ids, choices, {0, 2, 5}, false) != nullptr, "a path off the parents");
    expect(tree_walk_error(parents, ids, choices, {0}, false) != nullptr, "an empty walk where row 2 matches");
    std::printf(failures ? "TREE_WALK_FAIL\n" : "TREE_WALK_PASS\n");
    return failures ? 1 : 0;
}
