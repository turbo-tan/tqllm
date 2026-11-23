// tools/serve/admission.hpp: out of memory at a request's prefill queues it while others run (owner 11-06), fails alone.
#include "../tools/serve/admission.hpp"
#include <cstdio>
int main() {
    using namespace tqserve;
    int fails = 0;
    auto expect = [&](bool ok, const char * what) { if (!ok) { std::printf("FAIL %s\n", what); ++fails; } };
    expect(on_prefill_error("out of memory (fn_engine_prefill.cpp:120)", 3) == OnOom::Requeue, "OOM beside running requests: queued");
    expect(on_prefill_error("out of memory (fn_engine_prefill.cpp:120)", 0) == OnOom::Fail, "OOM with nothing else running: an error");
    expect(on_prefill_error("glm5next: prefill beyond the context", 3) == OnOom::Fail, "other errors still fail");
    expect(on_prefill_error(nullptr, 3) == OnOom::Fail, "no message: fail");
    expect(!admission_open(4, 5) && admission_open(5, 5) && admission_open(9, 0), "admission waits for one finish after a requeue");
    // shared prefixes: the slot holding 32K of this prompt lends it; a few hundred tokens more is not worth a copy
    {
        const unsigned held[4] = {0, 32000, 31000, 0};
        expect(share_source(4, 3, 0, [&](int o) { return held[o]; }) == 1, "the slot covering most of the prompt lends its state");
        expect(share_source(4, 1, 32000, [&](int o) { return held[o]; }) == -1, "the slot that holds it shares nothing with itself");
        expect(share_source(4, 3, 30000, [&](int o) { return held[o]; }) == -1, "under 4096 tokens more than own: no copy");
        expect(share_source(1, 0, 0, [&](int) { return 99999u; }) == -1, "one slot: nothing to share");
    }
    std::printf("serve admission: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
