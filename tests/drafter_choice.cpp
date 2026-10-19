// One drafter per run: -md loads DFlash2 only, --mtp loads the MTP head only, neither loads nothing; -md with --mtp is
// refused, so the unused drafter is never loaded or run.
#include "tqllm/options.hpp"

#include <cstdio>
#include <stdexcept>

static int fails = 0;
static void expect(bool ok, const char * what) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", what); fails += !ok; }

int main() {
    tqllm::EngineOptions o;
    auto d = tqllm::drafters(o);
    expect(!d.dflash && !d.mtp, "no drafter option: neither loaded");
    o.draft_model = "dflash2.gguf";
    d = tqllm::drafters(o);
    expect(d.dflash && !d.mtp, "-md: DFlash2 only, no MTP head");
    o = {}; o.native_mtp = true;
    d = tqllm::drafters(o);
    expect(!d.dflash && d.mtp, "--mtp: MTP head only, no DFlash2");
    o.draft_model = "dflash2.gguf";
    bool threw = false;
    try { tqllm::drafters(o); } catch (const std::exception &) { threw = true; }
    expect(threw, "-md with --mtp: refused");
    return fails ? 1 : 0;
}
