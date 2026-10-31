#pragma once
// GLM-5.3-Flash routed experts on the shared expert cache (src/experts): a layer's 288-expert stack is replaced by a bank
// of `slots` experts in the cache's row layout, and the grouped expert kernels run on the bank's slot ids instead of the
// expert ids (same rows, same bytes, same arithmetic: the output is identical to the whole-stack path). A chunk that
// routes to more experts than the layer holds runs in waves; each expert is loaded once per wave.
// tests/glm_expert_bank.cu checks the waves and the bank against the whole-stack path and a CPU reference.
#include "gguf_mapped.hpp"
#include "tq3_expert_cache.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {

// the three routed-expert projections of a GLM layer, in cache order (gate, up, down)
inline constexpr const char * kGlmExpertProjection[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};

// One expert's raw GGUF slices (the MixedExpertCache source of a GLM model). `map_of` resolves a tensor name to the
// mapping that holds its bytes (a split GGUF keeps each tensor in its own shard).
inline std::array<ExpertPayload, 3> glm_expert_payload(
    const std::string & prefix, const std::function<const MappedGguf &(const std::string &)> & map_of, int expert) {
    std::array<ExpertPayload, 3> out;
    for (int p = 0; p < 3; ++p) {
        const std::string n = prefix + kGlmExpertProjection[p];
        const auto & map = map_of(n);
        const auto it = map.file().tensors.find(n);
        if (it == map.file().tensors.end()) throw std::runtime_error("glm5next: missing expert tensor " + n);
        out[p].type = it->second.type;
        out[p].bytes = map.expert(n, uint64_t(expert));
    }
    return out;
}

// Assignments [first, first + tokens) of a [tokens][topk] routed id list, with the distinct experts they need.
struct GlmExpertWave {
    int first = 0, tokens = 0;
    std::vector<int> ids;  // distinct expert ids, ascending: prepare(layer, ids) puts them in the bank's slots
};

// Greedy: extend a wave while its distinct experts fit `slots`; one token per wave even when its own top-k does not fit
// (prepare() then reports the layer's slots are below the model's top-k).
inline std::vector<GlmExpertWave> glm_expert_waves(const std::vector<int> & routed, int topk, int slots) {
    std::vector<GlmExpertWave> waves;
    if (topk <= 0) return waves;
    const int tokens = int(routed.size()) / topk;
    for (int t = 0; t < tokens;) {
        GlmExpertWave w;
        w.first = t;
        for (int u = t; u < tokens; ++u) {
            std::vector<int> cand = w.ids;
            cand.insert(cand.end(), routed.begin() + size_t(u) * topk, routed.begin() + size_t(u + 1) * topk);
            std::sort(cand.begin(), cand.end());
            cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
            if (int(cand.size()) > slots && u > t) break;
            w.ids = std::move(cand);
            w.tokens = u - t + 1;
        }
        t += w.tokens;
        waves.push_back(std::move(w));
    }
    return waves;
}

}  // namespace tqllm
