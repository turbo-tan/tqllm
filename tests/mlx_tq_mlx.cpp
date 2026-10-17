#include "prefill.hpp"
#include "tq_mlx.hpp"
#include <cstdio>
#include <mlx/mlx.h>
#include <random>

namespace mx = mlx::core;
using namespace tqllm;
using namespace tqllm::mlx_backend;

namespace {
bool check(uint32_t source_type, const std::string& name, int rows, int k, std::mt19937& rng) {
    const int block_bytes = source_type == GGML_TQ3 ? 50 : 66;
    std::vector<uint8_t> payload(size_t(rows) * (k / 128) * block_bytes);
    for (auto& b : payload) b = uint8_t(rng());
    for (size_t at = 0; at < payload.size(); at += block_bytes) {  // finite FP16 scale in [0.5, 1.5)
        payload[at] = uint8_t(rng());
        payload[at + 1] = 0x38 | (rng() & 3);
    }
    GgufTensor t;
    t.type = source_type;
    t.ne = {uint64_t(k), uint64_t(rows)};
    t.bytes = payload.size();
    uint32_t type = source_type;
    auto blob = tq_mlx_pack_tensor(name, t, payload, type);
    GgufTensor packed;
    packed.type = type;
    packed.ne = t.ne;
    packed.bytes = gguf_tensor_bytes(type, t.ne, name);
    if (blob.size() != packed.bytes) {
        std::printf("size mismatch %s: %zu vs %llu\n", name.c_str(), blob.size(), (unsigned long long)packed.bytes);
        return false;
    }
    auto stored = tq_mlx_load(packed, blob);
    auto original = stage_bf16(load_matrix(t, payload));
    auto dense = tq_mlx_dense(stored);
    auto same = mx::all(mx::equal(mx::view(dense, mx::uint16), mx::view(original, mx::uint16)));
    mx::eval(same);
    const bool ok = same.item<bool>();
    std::printf("%s type %u -> %u rows %d k %d: %s\n", name.c_str(), source_type, type, rows, k, ok ? "ok" : "MISMATCH");
    return ok;
}
}

int main() {
    try {
        std::mt19937 rng(7);
        bool ok = true;
        for (auto source : {uint32_t(GGML_TQ3), uint32_t(GGML_TQ4)})
            for (auto name : {"blk.0.ffn_down.weight", "blk.3.attn_q.weight"})
                for (auto [rows, k] : {std::pair{200, 512}, std::pair{48, 256}, std::pair{1024, 5120}})
                    ok &= check(source, name, rows, k, rng);
        std::puts(ok ? "TQ_MLX_PASS" : "TQ_MLX_FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("TQ_MLX_ERROR %s\n", e.what());
        return 2;
    }
}
