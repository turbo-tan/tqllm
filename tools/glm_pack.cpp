// glm-pack-experts: write the packed expert file (TQFNEXP1) of a GLM-5.3-Flash GGUF, the file TQLLM_GLM_EXPERTS points at:
// every expert of every layer in the expert bank's row layout, so a bank miss is a copy instead of a loader repack.
// Layers without routed experts (the leading dense blocks) get an empty block, which the engine reads as "not packed".
//   glm-pack-experts MODEL.gguf OUT.tqfnexp
#include "glm_experts.hpp"
#include "gguf.hpp"
#include "gguf_mapped.hpp"
#include "packed_experts.hpp"

#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tqllm;

int main(int argc, char ** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: glm-pack-experts MODEL.gguf OUT.tqfnexp");
        const GgufFile meta = gguf_open(argv[1]);
        const std::string arch = meta.str("general.architecture");
        if (arch != "glm5-next" && arch != "glm5next") throw std::runtime_error("glm-pack-experts: not a glm5-next / glm5next file");
        const std::string a = arch + ".";
        const int layers = int(meta.num(a + "block_count")) - int(meta.num(a + "nextn_predict_layers", 0));
        const int experts = int(meta.num(a + "expert_count")), hidden = int(meta.num(a + "embedding_length"));
        const int ff = int(meta.num(a + "expert_feed_forward_length"));
        if (layers <= 0 || experts <= 0 || hidden <= 0 || ff <= 0) throw std::runtime_error("glm-pack-experts: bad model metadata");
        std::map<std::string, std::unique_ptr<MappedGguf>> maps;
        for (const auto & [n, t] : meta.tensors) {
            const std::string f = t.file.empty() ? meta.path : t.file;
            if (!maps.count(f)) maps[f] = std::make_unique<MappedGguf>(f);
        }
        auto map_of = [&](const std::string & n) -> const MappedGguf & {
            const auto it = meta.tensors.find(n);
            if (it == meta.tensors.end()) throw std::runtime_error("glm-pack-experts: missing tensor " + n);
            return *maps.at(it->second.file.empty() ? meta.path : it->second.file);
        };
        const int dense = int(meta.num(a + "leading_dense_block_count"));
        write_packed_experts(argv[2], layers, experts, false, [&](int l, int e) {
            if (l < dense) return std::array<std::vector<char>, 6>{};  // no routed experts in this layer
            return repack_expert(glm_expert_payload("blk." + std::to_string(l) + ".", map_of, e), hidden, ff, false);
        });
        std::printf("PACK_DONE %s layers %d experts %d hidden %d ff %d\n", argv[2], layers, experts, hidden, ff);
        return 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "glm-pack-experts: %s\n", ex.what());
        return 1;
    }
}
