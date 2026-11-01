// The 8K release-lattice selector gate: runs the selectors on the recorded 8K lattice fixture (our served text, j1540) and
// prices accepted tokens per verify pass. A node lands iff its root-to-node path is a prefix of the recorded greedy
// continuation - exactly what the target's greedy verify checks - so the number is the row's acceptance at <= 15 nodes.
//
// ctest runs the baseline assertions only (the chain and this build's default policy against recorded values). The release
// gate is `test-draft-tree-8k --gate`: it fails until the default policy reaches the bar that beats TensorFold at 8K
// (>= 4.90 accepted per pass, TensorFold's measured 4.88 at a 79.6 ms step; the candidate sets support 5.83).
//
// Fixture: tests/data/lattice8k.bin, 12 steps of a recorded 8K selector lattice (candidates, scores, gate / successor /
// predecessor rows, greedy continuation). Override the path and the gate bar with TQLLM_TEST_LATTICE_8K / TQLLM_GATE_8K.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "draft_tree.hpp"

#include "lattice_fixture.hpp"

namespace {
using namespace lattice_fixture;
// measured on the 12-step fixture (tests/data/lattice8k.bin, generated from the j1540 8K dump): chain 3.167, default
// policy 4.250. The full 86-step dump reads chain 2.97 / policy 4.05, so the sample is a fair, if slightly easier, stand-in.
constexpr double kChainBaseline = 3.167;
constexpr double kDefaultPolicyBaseline = 4.250;
constexpr double kGateAccepted = 4.90;   // beats TensorFold's 4.88 at 8K (79.6 ms step); the candidate sets support 5.83

}  // namespace

int main(int argc, char** argv) {
    try {
        const char* path = std::getenv("TQLLM_TEST_LATTICE_8K");
        const std::string fixture = path ? path : "tests/data/lattice8k.bin";
        const bool gate = argc > 1 && std::string(argv[1]) == "--gate";
        const auto steps = load(fixture);
        const tqllm::DraftTreeOptions defaults;  // this build's policy (the plugin's default)
        tqllm::DraftTreeOptions chain = defaults;
        chain.children = 1;
        chain.nodes = 15;
        chain.edge = 1.0f;
        chain.tau = 1.0f;
        double chain_sum = 0, policy_sum = 0;
        for (const auto& step : steps) {
            StepView view(step);
            const int chain_accepted = accepted(tqllm::best_first_tree(view.lattice(), step.anchor, chain), step.truth);
            const int policy_accepted = accepted(tqllm::best_first_tree(view.lattice(), step.anchor, defaults), step.truth);
            chain_sum += chain_accepted;
            policy_sum += policy_accepted;
            std::printf("step %2d: chain %2d  policy %2d\n", step.produced, chain_accepted, policy_accepted);
        }
        const double n = double(steps.size());
        std::printf("8K fixture: %d steps, chain %.3f, policy %.3f accepted per verify pass (gate %.2f)\n", int(steps.size()),
                    chain_sum / n, policy_sum / n, kGateAccepted);
        if (gate) {
            if (policy_sum / n < kGateAccepted) {
                std::printf("FAIL gate: policy %.3f < %.2f accepted per pass\n", policy_sum / n, kGateAccepted);
                return 1;
            }
            std::printf("PASS gate: policy %.3f >= %.2f accepted per pass\n", policy_sum / n, kGateAccepted);
            return 0;
        }
        // baseline sanity (ctest): the fixture and loader are right when the chain and the default policy land in band
        if (std::fabs(chain_sum / n - kChainBaseline) > 0.15) {
            std::printf("FAIL baseline: chain %.3f, recorded %.3f\n", chain_sum / n, kChainBaseline);
            return 1;
        }
        if (std::fabs(policy_sum / n - kDefaultPolicyBaseline) > 0.15) {
            std::printf("FAIL baseline: policy %.3f, recorded %.3f\n", policy_sum / n, kDefaultPolicyBaseline);
            return 1;
        }
        std::printf("draft_tree_8k: ok\n");
        return 0;
    } catch (const std::exception& error) {
        std::printf("draft_tree_8k: %s\n", error.what());
        return 1;
    }
}