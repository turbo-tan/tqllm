// Token trees on the device (src/backends/metal/tree_device.hpp), against the host selector on the recorded 8K lattice:
// (1) device_tree builds the host's best-first tree in the host's depth-first row order (ids and parents equal), and the
//     same acceptance; (2) the ancestor table and depths equal the host's; (3) tree_path walks the target's choices like
//     the host walk (accepted count, path rows, next anchor), for choices that follow the recorded text and for
//     choices that leave it at every depth. MLX ops only: runs on the CPU backend (no Metal needed).
#include <mlx/mlx.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#include "draft_tree.hpp"
#include "lattice_fixture.hpp"
#include <chrono>
#include <cstring>
#include "tree_device.hpp"

namespace mx = mlx::core;
using namespace lattice_fixture;
using tqllm::mlx_backend::DeviceTree;

namespace {
int failures = 0;
void expect(bool ok, const char* what, int step) {
    if (!ok) {
        std::printf("FAIL %s (step %d)\n", what, step);
        ++failures;
    }
}
template <class T>
std::vector<T> host(const mx::array& a) {
    auto c = mx::contiguous(a);
    mx::eval(c);
    return std::vector<T>(c.data<T>(), c.data<T>() + c.size());
}

// the engine's host tree in depth-first rows (engine_mlx.cpp tree_step before the device path)
struct HostRows {std::vector<uint32_t> ids; std::vector<int32_t> parents;};
HostRows host_rows(const tqllm::DraftTree& tree, uint32_t anchor) {
    const int nodes = int(tree.tokens.size());
    std::vector<std::vector<int>> children(size_t(nodes) + 1);
    for (int r = 0; r < nodes; ++r) children[size_t(tree.parents[size_t(r)] + 1)].push_back(r);
    HostRows out{{anchor}, {-1}};
    std::vector<int32_t> row_of(size_t(nodes) + 1, 0);
    std::vector<int> stack(children[0].rbegin(), children[0].rend());
    while (!stack.empty()) {
        const int r = stack.back(); stack.pop_back();
        row_of[size_t(r) + 1] = int32_t(out.ids.size());
        out.ids.push_back(tree.tokens[size_t(r)]);
        out.parents.push_back(row_of[size_t(tree.parents[size_t(r)] + 1)]);
        stack.insert(stack.end(), children[size_t(r) + 1].rbegin(), children[size_t(r) + 1].rend());
    }
    return out;
}
// the engine's host walk: from row 0, the first child equal to the target's choice
std::vector<int32_t> host_walk(const std::vector<uint32_t>& ids, const std::vector<int32_t>& parents,
                               const std::vector<uint32_t>& choices) {
    std::vector<int32_t> path{0};
    for (int cur = 0;;) {
        int next = -1;
        for (int r = cur + 1; r < int(ids.size()) && next < 0; ++r) if (parents[size_t(r)] == cur && ids[size_t(r)] == choices[size_t(cur)]) next = r;
        if (next < 0) break;
        path.push_back(next); cur = next;
    }
    return path;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "bench") == 0;
    const char* path = std::getenv("TQLLM_TEST_LATTICE_8K");
    const auto steps = load(path ? path : "tests/data/lattice8k.bin");
    for (const int tree_nodes : {15, 31}) {  // the 16-row and the 32-row verify
    int same_tree = 0, same_accept = 0;
    std::printf("-- %d nodes\n", tree_nodes);
    for (const auto& step : steps) {
        const int s = step.produced;
        // the device reads the selector table by token id: give it the fixture's rows under compact ids
        std::map<uint32_t, uint32_t> compact;
        for (size_t i = 0; i < step.codebook_tokens.size(); ++i) compact[step.codebook_tokens[i]] = uint32_t(i);
        std::vector<uint32_t> mapped(step.tokens.size());
        for (size_t i = 0; i < mapped.size(); ++i) {
            auto it = compact.find(step.tokens[i]);
            mapped[i] = it == compact.end() ? 0u : it->second;  // row 0 (the anchor's) may hold unused tokens
        }
        const int rows = step.rows, P = step.count;
        auto lattice_ids = mx::array(mapped.data(), {rows, kTopK}, mx::uint32);
        auto unary = mx::array(step.unary.data(), {rows, kTopK}, mx::float32);
        auto gate = mx::array(step.gate.data(), {rows, kRank}, mx::float32);
        auto successor = mx::array(step.successor.data(), {rows, kTopK, kRank}, mx::float32);
        auto predecessor = mx::array(step.predecessor.data(), {int(step.codebook_tokens.size()), kRank}, mx::float32);
        auto anchor = mx::array(compact.at(step.anchor));
        tqllm::mlx_backend::DeviceTreeOptions options;
        options.nodes = std::min(tree_nodes, [&] {  // the implicit tree may be smaller than 15 nodes on a short draft
            int total = 0, width = 1;
            for (int d = 0; d < P; ++d) {width = std::min(tree_nodes, width * options.children); total += width;}
            return total;
        }());
        DeviceTree tree = tqllm::mlx_backend::device_tree(lattice_ids, unary, gate, successor, predecessor, anchor, P, options);
        if (bench && tree_nodes == 15) {  // device cost of one step's tree build + walk (eval per call, as a served step pays it)
            for (int w = 0; w < 5; ++w) mx::eval(device_tree(lattice_ids, unary, gate, successor, predecessor, anchor, P, options).ids);
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 50; ++r) {
                auto bt = device_tree(lattice_ids, unary, gate, successor, predecessor, anchor, P, options);
                auto walk = tqllm::mlx_backend::tree_path(bt, bt.ids);
                mx::eval({bt.ids, bt.parents, bt.depth, bt.visible, walk.accepted, walk.rows, walk.next_anchor});
            }
            std::printf("TREE_BENCH device_tree + tree_path %.3f ms per step (15 nodes, %d positions)\n",
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 50, P);
            return 0;
        }
        auto dev_ids = host<uint32_t>(tree.ids);
        for (auto& id : dev_ids) id = step.codebook_tokens[id];  // back to token ids
        const auto dev_parents = host<int32_t>(tree.parents);

        StepView view(step);
        tqllm::DraftTreeOptions host_options;
        host_options.nodes = options.nodes;
        const auto best_first = tqllm::best_first_tree(view.lattice(), step.anchor, host_options);
        const auto expected = host_rows(best_first, step.anchor);
        const bool equal = dev_ids == expected.ids && dev_parents == expected.parents;
        same_tree += equal;
        // acceptance against the recorded text, from the device rows
        tqllm::DraftTree as_tree;
        for (size_t r = 1; r < dev_ids.size(); ++r) {as_tree.tokens.push_back(dev_ids[r]); as_tree.parents.push_back(dev_parents[r] - 1);}
        same_accept += accepted(as_tree, step.truth) == accepted(best_first, step.truth);
        std::printf("step %3d: tree %s, accepted device %d host %d\n", s, equal ? "== host" : "DIFFERS", accepted(as_tree, step.truth),
                    accepted(best_first, step.truth));

        // (2) ancestors and depths
        const int n = tree.rows;
        const auto visible = host<bool>(tree.visible);
        const auto depth = host<int32_t>(tree.depth);
        bool ancestors_ok = true;
        for (int t = 0; t < n; ++t) {
            std::vector<bool> want(size_t(n), false);
            int d = 0;
            for (int a = t; a >= 0; a = dev_parents[size_t(a)]) {want[size_t(a)] = true; d += a != t;}
            ancestors_ok = ancestors_ok && depth[size_t(t)] == d;
            for (int a = 0; a < n; ++a) ancestors_ok = ancestors_ok && visible[size_t(t) * n + a] == want[size_t(a)];
        }
        expect(ancestors_ok, "ancestor table and depths", s);

        // (3) the walk: the target follows the recorded text, then leaves it at depth `leave` (every depth tried)
        for (int leave = 0; leave <= P; ++leave) {
            std::vector<uint32_t> choices(static_cast<size_t>(n), 0u);
            for (int r = 0; r < n; ++r) {
                // row r's path tokens (below the anchor) must be a prefix of the text for the target to continue it
                std::vector<uint32_t> p;
                for (int a = r; a > 0; a = dev_parents[size_t(a)]) p.push_back(dev_ids[size_t(a)]);
                std::reverse(p.begin(), p.end());
                const bool on_text = p.size() <= step.truth.size() && std::equal(p.begin(), p.end(), step.truth.begin());
                const size_t dr = p.size();
                choices[size_t(r)] = on_text && int(dr) < leave && dr < step.truth.size() ? step.truth[dr] : 7u + uint32_t(r);
            }
            auto choice_array = mx::array(choices.data(), {n}, mx::uint32);
            DeviceTree back = tree;
            back.ids = mx::array(dev_ids.data(), {n}, mx::uint32);
            auto walk = tqllm::mlx_backend::tree_path(back, choice_array);
            const auto want = host_walk(dev_ids, dev_parents, choices);
            const int acc = host<int32_t>(mx::reshape(walk.accepted, {1}))[0];
            const auto rows_dev = host<int32_t>(walk.rows);
            const uint32_t next = host<uint32_t>(mx::reshape(walk.next_anchor, {1}))[0];
            bool ok = acc == int(want.size()) - 1 && next == choices[size_t(want.back())];
            for (size_t j = 0; j < want.size(); ++j) ok = ok && rows_dev[j] == want[j];
            expect(ok, "device walk == host walk", s);
        }
    }
    std::printf("device tree == host tree on %d of %zu steps; same acceptance on %d\n", same_tree, steps.size(), same_accept);
    // the host scores in double, the device in float: a near tie may order two siblings differently, never more than one step
    expect(same_tree + 1 >= int(steps.size()), "device trees equal the host's", -1);
    expect(same_accept == int(steps.size()), "acceptance equal on every step", -1);
    }
    if (failures) return 1;
    std::printf("mlx_tree_device: ok\n");
    return 0;
}
