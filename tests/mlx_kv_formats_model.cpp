// KV formats on the real engine (needs TQLLM_TEST_MODEL=<gguf>; skips without it): for BF16, int8 and trellis KV, a
// 3000-token prompt (prefill path: chunk > 16 rows) then 16 tokens fed (a) one at a time and (b) as one 16-row verify
// block; every row's logits must be equal bit for bit (spec == AR depends on it), and each compressed format's argmax
// is compared with BF16's (agreement, not a gate).
#include "dflash.hpp"
#include "qwen35.hpp"
#include <mlx/mlx.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
int main() {
    const char* path = std::getenv("TQLLM_TEST_MODEL");
    if (!path) {std::printf("KV_FORMATS_MODEL_SKIP (TQLLM_TEST_MODEL not set)\n"); return 0;}
    Qwen35 engine(path, 8192);
    std::vector<uint32_t> prompt(3000), tail(16);
    uint32_t x = 12345;
    for (auto& t : prompt) {x = x * 1103515245u + 12345u; t = 1000 + (x >> 8) % 40000;}
    for (auto& t : tail) {x = x * 1103515245u + 12345u; t = 1000 + (x >> 8) % 40000;}
    int failures = 0;
    std::vector<int> ref_top;
    for (KvTier tier : {KvTier::Bf16, KvTier::Int8, KvTier::Trellis}) {
        auto start = [&] {engine.reset(); engine.reserve_kv(int(prompt.size() + tail.size()) + 16); engine.kv_format(tier); mx::eval(engine.forward(prompt));};
        start();
        std::vector<mx::array> ar;
        for (uint32_t t : tail) {auto l = engine.forward(std::vector<uint32_t>{t}); mx::eval(l); ar.push_back(mx::reshape(l, {-1}));}
        start();
        auto block = engine.forward(tail, true); mx::eval(block);
        int equal = 0; std::vector<int> top;
        for (int i = 0; i < 16; ++i) {
            auto row = mx::reshape(mx::slice(block, {i, 0}, {i + 1, block.shape(1)}), {-1});
            equal += mx::array_equal(row, ar[i]).item<bool>();
            top.push_back(mx::argmax(ar[i]).item<int>());
        }
        int agree = 0;
        if (ref_top.empty()) ref_top = top; else for (int i = 0; i < 16; ++i) agree += top[i] == ref_top[i];
        const bool ok = equal == 16;
        failures += !ok;
        std::printf("%s KV %s: 16-row verify = 16 single-row steps bit for bit: %d / 16 rows%s\n", ok ? "PASS" : "FAIL", kv_tier_name(tier), equal,
                    tier == KvTier::Bf16 ? "" : (std::string("; argmax agreement with BF16 ") + std::to_string(agree) + " / 16").c_str());
    }
    {   // a finished request's KV buffers are not "in use" for the next request's plan
        engine.reset(); mx::synchronize();
        const double raw = mx::get_active_memory() / 1e6, held = engine.kv_budget(512).in_use / 1e6;
        engine.kv_format(KvTier::Bf16); mx::synchronize();
        const double freed = engine.kv_budget(512).in_use / 1e6;
        const bool ok = std::abs(held - freed) < 1.0 && raw - freed > 10.0;
        failures += !ok;
        std::printf("%s plan budget after a trellis request: %.1f MB in use with its KV held, %.1f MB freed (active %.1f MB)\n",
                    ok ? "PASS" : "FAIL", held, freed, raw);
    }
    if (const char* dpath = std::getenv("TQLLM_TEST_DRAFTER")) {
        // the drafter's far-context path on the real engine (j1167: the first retrieving draft scanned the int8 codes with
        // a matmul and every request past the window failed): a 3000-row prefill inject fills the far caches past
        // window + retrieved rows, then a draft with the inline selection and one with the look-ahead rows
        DFlash2 drafter(dpath, engine);
        engine.reset(); drafter.reset(); engine.reserve_kv(int(prompt.size()) + 32); engine.kv_format(KvTier::Bf16);
        drafter.far_context(true); drafter.reserve_far(int(prompt.size()) + 32);
        mx::eval(engine.forward(prompt)); drafter.inject(engine.feature_states(), 0);
        bool ok = drafter.far_row_bytes() > 0;
        std::string err;
        try {
            for (int round = 0; round < 2 && ok; ++round) {
                auto ids = drafter.draft_device_at(mx::array(uint32_t(1000)), mx::array(engine.position(), mx::int32), 15);
                mx::eval(ids);
                ok = ids.shape() == mx::Shape{15} && ids.dtype() == mx::uint32 && mx::max(ids).item<uint32_t>() < uint32_t(engine.vocabulary());
            }
        } catch (const std::exception& e) {ok = false; err = e.what();}
        failures += !ok;
        std::printf("%s drafter far-context drafts after a 3000-row inject (inline selection, then look-ahead)%s%s\n", ok ? "PASS" : "FAIL",
                    err.empty() ? "" : ": ", err.c_str());
    }
    std::printf(failures ? "KV_FORMATS_MODEL_FAIL\n" : "KV_FORMATS_MODEL_PASS\n");
    return failures ? 1 : 0;
}
