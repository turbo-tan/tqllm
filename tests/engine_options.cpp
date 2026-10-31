// tqserve engine options: --mode and the Flash-Next prefill parameters parse into EngineOptions, keep -1 (engine default) when absent,
// and reject out-of-range values.
#include "tqllm/options.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static tqllm::EngineOptions parse(std::vector<std::string> args) {
    args.insert(args.begin(), "tqserve");
    std::vector<char *> argv; for (auto & a : args) argv.push_back(a.data());
    tqllm::EngineOptions o;
    for (int i = 1; i < int(argv.size()); ++i)
        if (!tqllm::parse_engine_option(int(argv.size()), argv.data(), i, o)) throw std::invalid_argument("not an engine option: " + args[size_t(i)]);
    return o;
}
static bool throws(std::vector<std::string> args) {
    try { parse(std::move(args)); } catch (const std::exception &) { return true; }
    return false;
}

int main() {
    int fail = 0;
    auto expect = [&](bool ok, const char * what) { if (!ok) { std::printf("FAIL %s\n", what); ++fail; } };
    const auto d = parse({});
    expect(d.prefill_borrow.empty() && d.prefill_chunk == -1 && d.prefill_sub == -1 && d.prefill_stage == -1 && d.prefill_reserve_mb == -1 &&
           d.prefill_arena_mb == -1 && d.dense_cache_mb == -1, "defaults stay -1 (engine default)");
    const auto o = parse({"--prefill-chunk", "16384", "--prefill-sub", "1024", "--prefill-stage", "96", "--prefill-reserve-mb", "5500",
                          "--prefill-arena-mb", "4000", "--dense-cache-mb", "0"});
    expect(o.prefill_chunk == 16384, "--prefill-chunk");
    expect(o.prefill_sub == 1024, "--prefill-sub");
    expect(o.prefill_stage == 96, "--prefill-stage");
    expect(o.prefill_reserve_mb == 5500, "--prefill-reserve-mb");
    expect(o.prefill_arena_mb == 4000, "--prefill-arena-mb");
    expect(o.dense_cache_mb == 0, "--dense-cache-mb 0 turns the cache off");
    expect(parse({"--prefill-borrow", "auto"}).prefill_borrow == "auto" && parse({"--prefill-borrow", "12"}).prefill_borrow == "12", "--prefill-borrow all|auto|N");
    expect(throws({"--prefill-borrow", "some"}), "bad borrow value rejected");
    expect(throws({"--prefill-chunk", "100"}), "--prefill-chunk below 256 rejected");
    expect(throws({"--prefill-sub", "8"}), "--prefill-sub below 16 rejected");
    expect(throws({"--prefill-stage", "0"}), "--prefill-stage below 1 rejected");
    expect(d.mode == tqllm::Mode::Speed, "--mode defaults to speed");
    expect(parse({"--mode", "exact"}).mode == tqllm::Mode::Exact && parse({"--mode", "speed"}).mode == tqllm::Mode::Speed, "--mode exact|speed");
    expect(throws({"--mode", "fast"}), "unknown --mode rejected");
    // --mode exact conflicts: listed, warned, and only accepted by --yes or a "y" on the terminal
    static const char * (*no_env)(const char *) = [](const char *) -> const char * { return nullptr; };
    static const char * (*cpu_miss)(const char *) = [](const char * n) -> const char * { return std::strcmp(n, "TQFN_CPU_MISS") ? nullptr : "4"; };
    expect(tqllm::exact_mode_conflicts(parse({"--mode", "exact"}), false, false, no_env).empty(), "exact alone: no conflict");
    expect(tqllm::exact_mode_conflicts(parse({"--prefill-int8", "on"}), false, false, no_env).empty(), "speed mode: never a conflict");
    expect(tqllm::exact_mode_conflicts(parse({"--mode", "exact", "--prefill-int8", "on"}), false, false, no_env).size() == 1, "exact + --prefill-int8 on");
    expect(tqllm::exact_mode_conflicts(parse({"--mode", "exact", "-ctk", "q8"}), false, false, no_env).size() == 1, "exact + lossy KV");
    expect(tqllm::exact_mode_conflicts(parse({"--mode", "exact"}), true, false, cpu_miss).size() == 1 &&
           tqllm::exact_mode_conflicts(parse({"--mode", "exact"}), false, false, cpu_miss).empty(), "TQFN_* only for Flash-Next");
    expect(tqllm::exact_mode_conflicts(parse({"--mode", "exact"}), false, true, no_env).size() == 1, "engine without an exact path");
    expect(parse({"-y"}).assume_yes && parse({"--yes"}).assume_yes && !parse({}).assume_yes, "-y / --yes");
    const std::vector<std::string> one = {"--prefill-int8 on: int8 prefill"};
    auto confirm = [&](bool yes, const char * typed, bool terminal) {
        std::istringstream in(typed); std::ostringstream err;
        const bool go = tqllm::confirm_mode_conflicts(one, yes, terminal ? &in : nullptr, err);
        return std::make_pair(go, err.str());
    };
    expect(confirm(true, "", false).first, "--yes goes ahead without a terminal");
    expect(confirm(false, "y\n", true).first && confirm(false, "Yes\n", true).first, "y / Yes on the terminal goes ahead");
    expect(!confirm(false, "n\n", true).first && !confirm(false, "\n", true).first && !confirm(false, "", true).first, "n / empty / EOF stops");
    expect(!confirm(false, "y\n", false).first, "no terminal and no --yes stops");
    expect(confirm(false, "n\n", true).second.find("WARN --mode exact with --prefill-int8 on") != std::string::npos, "the warning names the option");
    expect(tqllm::confirm_mode_conflicts({}, false, nullptr, std::cerr), "no conflict: no question");
    expect(throws({"--prefill-reserve-mb", "-5"}), "negative --prefill-reserve-mb rejected");
    expect(throws({"--prefill-chunk"}), "missing value rejected");
    std::printf(fail ? "ENGINE_OPTIONS FAIL\n" : "ENGINE_OPTIONS PASS\n");
    return fail ? 1 : 0;
}
