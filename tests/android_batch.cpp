// Batched forward and speculative rollback against one-token-at-a-time forward, on a real qwen35 GGUF:
//   test-android-batch MODEL.gguf
#include "android_model.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <stdexcept>

using tqllm::android_backend::Model;
namespace {
void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
}
int main(int argc, char ** argv) {
    try {
        if (argc != 2) { std::cerr << "usage: test-android-batch MODEL.gguf\n"; return 2; }
        const std::vector<uint32_t> prompt{9419, 0, 2500, 628, 353, 1438, 488}, more{3242, 30, 198, 12305, 727};
        Model a(argv[1], {256, 4}), b(argv[1], {256, 4});
        std::vector<std::vector<float>> single;
        for (auto t : prompt) a.forward(t, false);
        for (auto t : more) single.push_back(a.forward(t));
        // Prefill as one batch, then the rest as a batch with every logit.
        b.forward_batch(prompt, false);
        auto batched = b.forward_batch(more);
        for (size_t i = 0; i < more.size(); ++i) check(same(single[i], batched[i]), "batched logits differ from single-token logits");
        check(a.position() == b.position(), "batched position");
        // Rollback: forward 5 speculative tokens, keep 2, continue; must equal never having forwarded the other 3.
        Model c(argv[1], {256, 4});
        c.forward_batch(prompt, false);
        c.checkpoint();
        c.forward_batch({3242, 30, 1, 2, 3});
        c.rollback(2);
        check(c.position() == int(prompt.size()) + 2, "rollback position");
        const auto & after = c.forward_batch({198, 12305, 727});
        for (int i = 0; i < 3; ++i) check(same(after[i], single[2 + i]), "rollback state differs from clean history");
        // Full accept and zero accept.
        c.checkpoint(); c.forward(5); c.rollback(1);
        c.checkpoint(); c.forward(6); c.forward(7); c.rollback(0);
        check(c.position() == int(prompt.size() + more.size()) + 1, "rollback bookkeeping");
        // Snapshot / restore and save / load: continuing from either equals the clean history.
        Model d(argv[1], {256, 4});
        d.forward_batch(prompt, false);
        d.snapshot();
        d.forward_batch({1, 2, 3, 4});
        check(d.restore() == int(prompt.size()), "restore position");
        const auto restored = d.forward_batch(more);
        for (size_t i = 0; i < more.size(); ++i) check(same(restored[i], single[i]), "restored state differs from clean history");
        const std::string state = std::string(argv[1]) + ".test-state";
        d.reset(); d.forward_batch(prompt, false); d.save(state);
        Model e(argv[1], {256, 4});
        check(e.load(state) && e.position() == int(prompt.size()), "load state");
        std::remove(state.c_str());
        const auto loaded = e.forward_batch(more);
        for (size_t i = 0; i < more.size(); ++i) check(same(loaded[i], single[i]), "loaded state differs from clean history");
        Model f(argv[1], {512, 4});
        check(f.position() == 0, "fresh model");
        std::cout << "android_batch: PASS (batched == single bitwise, prefill batch, rollback keep 2/all/none, "
                     "snapshot/restore, save/load)\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
