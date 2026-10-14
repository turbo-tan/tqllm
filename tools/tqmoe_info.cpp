#include "gguf.hpp"
#include "qwen4exp.hpp"
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: tqmoe-info [--gsq-rco] shard1.gguf [shard2.gguf ...] (metadata only)");
        tqllm::GgufDialect dialect = tqllm::GgufDialect::Native;
        std::vector<std::string> paths;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--gsq-rco") dialect = tqllm::GgufDialect::GsqRco;
            else if (argv[i][0] == '-') throw std::runtime_error("unknown option: " + std::string(argv[i]));
            else paths.emplace_back(argv[i]);
        }
        if (paths.empty()) throw std::runtime_error("no GGUF shards given");
        tqllm::Qwen4ExpConfig config;
        uint64_t payload = 0, experts = 0, ple = 0, expert_triplet = 0;
        std::map<std::string, tqllm::GgufTensor> tensors;
        int expected_shards = 0;
        std::set<int> shard_ids;
        for (size_t i = 0; i < paths.size(); ++i) {
            const auto f = tqllm::gguf_open(paths[i], dialect);
            const auto c = i == 0 || f.has("general.architecture") ? tqllm::qwen4exp_config(f) : config;
            if (i == 0) {
                config = c;
                expected_shards = int(f.num("split.count", 1));
            } else if (c != config || f.num("split.count", 1) != expected_shards)
                throw std::runtime_error("shard architecture metadata differs");
            const int shard_id = int(f.num("split.no", 0));
            if (shard_id < 0 || shard_id >= expected_shards || !shard_ids.insert(shard_id).second)
                throw std::runtime_error("invalid or duplicate shard index");
            const auto size = std::filesystem::file_size(paths[i]);
            for (const auto & [name, tensor] : f.tensors) {
                if (!tensors.emplace(name, tensor).second) throw std::runtime_error("duplicate tensor: " + name);
                if (tensor.offset > size || tensor.bytes > size - tensor.offset)
                    throw std::runtime_error("tensor exceeds shard: " + name);
                payload += tensor.bytes;
                if (name.find("_exps.weight") != std::string::npos) {
                    if (tensor.ne.size() != 3 || tensor.ne[2] != uint64_t(c.experts))
                        throw std::runtime_error("unexpected expert dimensions: " + name);
                    experts += tensor.bytes;
                    expert_triplet = std::max(expert_triplet, tensor.bytes / c.experts);
                }
                if (name == "per_layer_token_embd.weight") ple += tensor.bytes;
            }
        }
        if (expected_shards != int(paths.size())) throw std::runtime_error("incomplete shard set");
        for (int layer = 0; layer < config.layers; ++layer) {
            uint64_t triplet = 0;
            const std::string p = "blk." + std::to_string(layer) + ".";
            for (const char * kind : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                const auto it = tensors.find(p + kind);
                if (it == tensors.end()) throw std::runtime_error("missing tensor: " + p + kind);
                const auto & t = it->second;
                const bool down = std::string(kind) == "ffn_down_exps.weight";
                if (t.ne[0] != uint64_t(down ? config.expert_ff : config.hidden) ||
                    t.ne[1] != uint64_t(down ? config.hidden : config.expert_ff))
                    throw std::runtime_error("expert shape mismatch: " + p + kind);
                triplet += t.bytes / config.experts;
            }
            expert_triplet = std::max(expert_triplet, triplet);
        }
        if (!config.ple_layers.empty()) {
            const auto it = tensors.find("per_layer_token_embd.weight");
            if (it == tensors.end() || it->second.ne.size() != 2 || it->second.ne[0] != uint64_t(config.ple_dim))
                throw std::runtime_error("missing or wrong PLE embedding table");
            for (size_t h = 0; h < config.ple.offsets.size(); ++h)
                if (it->second.ne[1] < config.ple.offsets[h] + config.ple.vocab_sizes[h])
                    throw std::runtime_error("PLE head range exceeds table");
        }
        std::cout << "{\n  \"architecture\": \"qwen4exp\",\n  \"inference_ready\": false,\n"
                  << "  \"dialect\": \"" << (dialect == tqllm::GgufDialect::GsqRco ? "gsq-rco" : "native") << "\",\n"
                  << "  \"shards\": " << expected_shards << ",\n  \"tensors\": " << tensors.size()
                  << ",\n  \"layers\": " << config.layers << ",\n  \"hidden\": " << config.hidden
                  << ",\n  \"experts\": " << config.experts << ",\n  \"topk\": " << config.topk
                  << ",\n  \"index_heads\": " << config.index_heads
                  << ",\n  \"index_dim\": " << config.index_dim
                  << ",\n  \"index_topk\": " << config.index_topk
                  << ",\n  \"hc_streams\": " << config.hc << ",\n  \"payload_bytes\": " << payload
                  << ",\n  \"expert_bytes\": " << experts << ",\n  \"non_expert_bytes\": " << payload - experts
                  << ",\n  \"ple_embedding_bytes\": " << ple << ",\n  \"max_expert_triplet_bytes\": " << expert_triplet
                  << ",\n  \"max_active_expert_bytes_per_layer\": " << expert_triplet * config.topk << "\n}\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << "tqmoe-info: " << e.what() << '\n'; return 1; }
}
