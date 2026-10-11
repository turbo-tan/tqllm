#include "../src/batch_graph_key.hpp"
#include <cstdio>
#include <set>

int main() {
    std::set<uint64_t> keys;
    for (unsigned stash = 0; stash < 2; ++stash)
        for (int sampled = 0; sampled < 2; ++sampled)
            for (int slot = 0; slot < 32; ++slot)
                for (int width = 1; width <= 16; ++width)
                    for (int state = 0; state < 4; ++state) {
                        const auto layout = uint64_t((slot * 64 + width) * 4 + state);
                        const auto key = (tqllm::batch_graph_seed(sampled, stash) ^ layout) * 1099511628211ull;
                        if (!keys.insert(key).second) {
                            std::fprintf(stderr, "graph key collision: sampled=%d state=%d\n", sampled, state);
                            return 1;
                        }
                    }
    return 0;
}
