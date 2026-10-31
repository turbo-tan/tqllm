#include "hf_arch.hpp"

namespace tqllm {

namespace {
constexpr HfArch kArches[] = {
    {"MiniMaxM3SparseForConditionalGeneration", "minimax-m3", hf_map_minimax_m3},
    {"DSparkMiniMaxDraftModel", "minimax-m3", hf_map_minimax_m3_dspark},
};
}  // namespace

const HfArch * hf_arch(const std::string & hf_name) {
    for (const HfArch & a : kArches)
        if (hf_name == a.hf) return &a;
    return nullptr;
}

}  // namespace tqllm
