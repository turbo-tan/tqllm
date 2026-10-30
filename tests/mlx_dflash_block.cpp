// A 7-token draft from the block-16 drafter (TQLLM_MLX_DFLASH_BLOCK=16, the served default) runs the file's native
// 8-row block and must equal the block-8 drafter's draft token for token, from the same ring (host and device draft
// paths, window partly and fully filled) when native_block is on; off, a 7-token draft is the head of the 15-draft. Needs TQLLM_TEST_MODEL + TQLLM_TEST_DRAFTER
// (skips without them).
#include "dflash.hpp"
#include "qwen35.hpp"
#include <mlx/mlx.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
int main() {
    const char* model = std::getenv("TQLLM_TEST_MODEL");
    const char* draft = std::getenv("TQLLM_TEST_DRAFTER");
    if (!model || !draft) {std::printf("DFLASH_BLOCK_SKIP (TQLLM_TEST_MODEL / TQLLM_TEST_DRAFTER not set)\n"); return 0;}
    Qwen35 engine(model, 8192);
    setenv("TQLLM_MLX_DFLASH_BLOCK", "16", 1);
    auto wide = std::make_unique<DFlash2>(draft, engine);
    setenv("TQLLM_MLX_DFLASH_BLOCK", "8", 1);
    auto narrow = std::make_unique<DFlash2>(draft, engine);
    int failures = 0;
    for (int length : {600, 3000}) {   // below and above the drafter's window
        std::vector<uint32_t> prompt(static_cast<size_t>(length));
        uint32_t x = 777u + uint32_t(length);
        for (auto& t : prompt) {x = x * 1103515245u + 12345u; t = 1000 + (x >> 8) % 40000;}
        engine.reset(); engine.reserve_kv(length + 64);
        auto logits = engine.forward(prompt);
        const uint32_t pending = mx::argmax(logits).item<uint32_t>();
        auto features = engine.feature_states();
        wide->reset(); narrow->reset();
        wide->inject(features, 0); narrow->inject(features, 0);
        // native off (the served default below WidthPolicy::kNativeFrom): a 7-token draft is the head of the 15-draft
        wide->native_block(false);
        auto h7 = wide->draft(pending, length, 7), a15 = wide->draft(pending, length, 15);
        const bool head_ok = a15.size() == 15 && std::vector<uint32_t>(a15.begin(), a15.begin() + 7) == h7;
        wide->native_block(true);
        auto a7 = wide->draft(pending, length, 7), b7 = narrow->draft(pending, length, 7);
        const bool host_equal = a7 == b7, wide_ok = head_ok;
        // the device draft (draft-ahead path) from the same ring
        auto d7 = wide->draft_device_at(mx::array(pending), mx::array(length, mx::int32), 7);
        auto e7 = narrow->draft_device_at(mx::array(pending), mx::array(length, mx::int32), 7);
        mx::eval(d7, e7);
        const bool device_equal = mx::array_equal(d7, e7).item<bool>() &&
            std::vector<uint32_t>(d7.data<uint32_t>(), d7.data<uint32_t>() + 7) == b7;
        std::printf("DFLASH_BLOCK length=%d host_7_equal=%d device_7_equal=%d head_of_15_off=%d %s\n", length, int(host_equal),
                    int(device_equal), int(wide_ok), host_equal && device_equal && wide_ok ? "PASS" : "FAIL");
        failures += !(host_equal && device_equal && wide_ok);
    }
    std::printf(failures ? "DFLASH_BLOCK_FAIL\n" : "DFLASH_BLOCK_PASS\n");
    return failures ? 1 : 0;
}
