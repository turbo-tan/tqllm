// Token-tree verify timing on the Metal target (no drafter): one verify after a random prompt, per mode: "tree"
// (a fixed best-first-like tree with 4 side leaves, device arrays as TreeAhead submits them) and "chain" (a 16-row
// chain), "wide" (32 rows: a 16-row spine with side leaves, SpecPilot's wide tree). Each verify keeps row 0, so the
// context grows by one per call. Prints ms per verify (evaluated), the mean of `reps` after 3 warm-up calls.
// usage: tqmlx-tree-bench model.gguf prompt_len [reps]
#include "qwen35.hpp"
#include "tree_device.hpp"
#include <mlx/mlx.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

int main(int argc, char** argv) {
    if (argc < 3) {std::fprintf(stderr, "usage: tqmlx-tree-bench model.gguf prompt_len [reps]\n"); return 2;}
    const int len = std::atoi(argv[2]), reps = argc > 3 ? std::atoi(argv[3]) : 20;
    // a best-first-like tree: a 9-row main chain and side branches off rows 0, 1, 2, 4 (4 side leaves)
    const std::vector<int32_t> tree = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 1, 11, 2, 4, 14};
    std::vector<int32_t> chain(16);
    for (int t = 0; t < 16; ++t) chain[size_t(t)] = t - 1;
    // "wide": SpecPilot's 32-node tree within the 16 drafted positions: a 16-row spine and one side leaf per spine row
    std::vector<int32_t> wide(32);
    for (int t = 0; t < 16; ++t) wide[size_t(t)] = t - 1;
    for (int t = 16; t < 32; ++t) wide[size_t(t)] = t - 17 < 0 ? 0 : t - 17;
    Qwen35 engine(argv[1], len + 4 * (reps + 3) + 64);
    engine.set_exact(true);
    std::vector<uint32_t> prompt(static_cast<size_t>(len)), block(32);
    uint32_t x = 4242;
    for (auto& t : prompt) {x = x * 1103515245u + 12345u; t = 1000 + (x >> 8) % 40000;}
    for (auto& t : block) {x = x * 1103515245u + 12345u; t = 1000 + (x >> 8) % 40000;}
    
    for (const std::string mode : {"tree", "chain", "wide"}) {
        const auto& parents = mode == "chain" ? chain : mode == "wide" ? wide : tree;
        const int n = int(parents.size());
        const auto ids = mx::array(block.data(), {n}, mx::uint32);
        const auto dp = mx::array(parents.data(), {n}, mx::int32);
        auto [visible, depth] = tree_ancestors(dp);
        mx::eval(visible, depth);
        engine.reset(); engine.reserve_kv(len + reps + 32); engine.kv_format(KvTier::Bf16);
        mx::eval(engine.forward(prompt));
        double total = 0;
        for (int r = 0; r < reps + 3; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            Qwen35::Verification ticket;
            auto logits = engine.verify_tree_device(ids, dp, depth, visible, ticket);
            mx::eval(logits);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (r >= 3) total += ms;
            engine.accept_path(ticket, {0}, &parents);
        }
        std::printf("TREE_BENCH len %d mode %-5s %.2f ms per verify\n", len, mode.c_str(), total / reps);
    }
    return 0;
}
