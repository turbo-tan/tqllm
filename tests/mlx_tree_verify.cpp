// Token-tree verify on the real engine (needs TQLLM_TEST_MODEL=<gguf>; skips without it), BF16 KV, 3000-token prompt:
// (1) a chain-shaped tree verifies bit for bit like the chain verify (every row's logits);
// (2) a tree with decoy siblings: the rows on the greedy path give the logits of feeding that path one token at a time,
//     bit for bit (side-branch rows attend over their keys in AR order, --exact on);
// (3) accept_path on that branch, then one more token: the next logits equal the one-token-at-a-time run bit for bit.
#include "qwen35.hpp"
#include "tree_device.hpp"
#include <mlx/mlx.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

static mx::array row(const mx::array& logits, int r) {return mx::reshape(mx::slice(logits, {r, 0}, {r + 1, logits.shape(1)}), {-1});}
static float rel_diff(const mx::array& a, const mx::array& b) {
    return (mx::max(mx::abs(mx::subtract(a, b))) / mx::maximum(mx::max(mx::abs(b)), mx::array(1e-6f))).item<float>();
}

int main() {
    const char* path = std::getenv("TQLLM_TEST_MODEL");
    if (!path) {std::printf("TREE_VERIFY_SKIP (TQLLM_TEST_MODEL not set)\n"); return 0;}
    Qwen35 engine(path, 8192);
    // TQLLM_TEST_PROMPT_LEN: prompt length (default 3000; short chats are 30-60 tokens)
    std::vector<uint32_t> prompt(size_t(std::getenv("TQLLM_TEST_PROMPT_LEN") ? std::atoi(std::getenv("TQLLM_TEST_PROMPT_LEN")) : 3000)), block(16);
    uint32_t x = 4242;
    for (auto& t : prompt) {x = x * 1103515245u + 12345u; t = 1000 + (x >> 8) % 40000;}
    for (auto& t : block) {x = x * 1103515245u + 12345u; t = 1000 + (x >> 8) % 40000;}
    auto start = [&] {engine.reset(); engine.reserve_kv(int(prompt.size()) + 64); engine.kv_format(KvTier::Bf16); mx::eval(engine.forward(prompt));};
    int failures = 0;

    {   // (1) chain-shaped tree == chain verify
        start();
        Qwen35::Verification chain_ticket;
        auto chain = engine.verify_device(mx::array(block.data(), {16}, mx::uint32), chain_ticket); mx::eval(chain);
        start();
        std::vector<int32_t> parents(16);
        for (int t = 0; t < 16; ++t) parents[size_t(t)] = t - 1;
        Qwen35::Verification tree_ticket;
        auto tree = engine.verify_tree_device(mx::array(block.data(), {16}, mx::uint32), parents, tree_ticket); mx::eval(tree);
        int equal = 0;
        for (int r = 0; r < 16; ++r) equal += mx::array_equal(row(chain, r), row(tree, r)).item<bool>();
        failures += equal != 16;
        std::printf("%s TREE chain-shaped tree == chain verify bit for bit: %d / 16 rows\n", equal == 16 ? "PASS" : "FAIL", equal);
    }

    // greedy reference: anchor a, then the target's own continuation g1 .. g4, fed one token at a time
    const uint32_t anchor = block[0];
    start();
    std::vector<mx::array> ar;
    std::vector<uint32_t> greedy;
    uint32_t next = anchor;
    for (int s = 0; s < 5; ++s) {
        auto l = mx::reshape(engine.forward(std::vector<uint32_t>{next}), {-1}); mx::eval(l);
        ar.push_back(l);
        next = uint32_t(mx::argmax(l).item<int>());
        greedy.push_back(next);
    }
    {   // (2) tree: 0 anchor; 1 decoy (0); 2 g1 (0); 3 decoy (2); 4 g2 (2); 5 g3 (4); 6 decoy (1); 7 decoy (5)
        start();
        const std::vector<uint32_t> ids = {anchor, block[1], greedy[0], block[3], greedy[1], greedy[2], block[6], block[7]};
        const std::vector<int32_t> parents = {-1, 0, 0, 2, 2, 4, 1, 5};
        Qwen35::Verification ticket;
        auto logits = engine.verify_tree_device(mx::array(ids.data(), {int(ids.size())}, mx::uint32), parents, ticket); mx::eval(logits);
        const int on_path[4] = {0, 2, 4, 5};
        bool same_argmax = true; float worst = 0.0f;
        for (int k = 0; k < 4; ++k) {
            auto r = row(logits, on_path[k]);
            same_argmax = same_argmax && mx::argmax(r).item<int>() == mx::argmax(ar[size_t(k)]).item<int>();
            worst = std::max(worst, rel_diff(r, ar[size_t(k)]));
        }
        const bool ok2 = same_argmax && worst == 0.0f;
        failures += !ok2;
        std::printf("%s TREE branch rows on the greedy path: argmax %s, max rel diff %.2e vs one token at a time\n",
                    ok2 ? "PASS" : "FAIL", same_argmax ? "equal" : "DIFFERENT", worst);
        // (3) keep the path 0 -> 2 -> 4 -> 5 (anchor, g1, g2, g3), then feed g4
        engine.accept_path(ticket, {0, 2, 4, 5});
        auto after = mx::reshape(engine.forward(std::vector<uint32_t>{greedy[3]}), {-1}); mx::eval(after);
        const bool same = mx::argmax(after).item<int>() == mx::argmax(ar[4]).item<int>();
        const float d = rel_diff(after, ar[4]);
        const bool ok3 = same && d == 0.0f && engine.position() == int(prompt.size()) + 5;
        failures += !ok3;
        std::printf("%s TREE accept_path then one token: argmax %s, rel diff %.2e, position %d\n", ok3 ? "PASS" : "FAIL",
                    same ? "equal" : "DIFFERENT", d, engine.position());
    }
    {   // (2b) the served path: the same tree from device arrays only (TreeAhead: nothing read back before the verify)
        start();
        const std::vector<uint32_t> ids = {anchor, block[1], greedy[0], block[3], greedy[1], greedy[2], block[6], block[7]};
        const std::vector<int32_t> parents = {-1, 0, 0, 2, 2, 4, 1, 5};
        const int on_path[4] = {0, 2, 4, 5};
        const auto dparents = mx::array(parents.data(), {int(parents.size())}, mx::int32);
        auto [visible, depth] = tree_ancestors(dparents);
        Qwen35::Verification ticket;
        auto logits = engine.verify_tree_device(mx::array(ids.data(), {int(ids.size())}, mx::uint32), dparents, depth, visible, ticket);
        mx::eval(logits);
        bool bitwise = true;
        for (int k = 0; k < 4; ++k) bitwise = bitwise && mx::array_equal(row(logits, on_path[k]), ar[size_t(k)]).item<bool>();
        failures += !bitwise;
        std::printf("%s TREE device arrays (served path): greedy-path rows bit for bit %s\n", bitwise ? "PASS" : "FAIL", bitwise ? "yes" : "NO");
    }
    std::printf(failures ? "TREE_VERIFY_FAIL\n" : "TREE_VERIFY_PASS\n");
    return failures ? 1 : 0;
}
