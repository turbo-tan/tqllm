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
    std::printf("serve admission: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
