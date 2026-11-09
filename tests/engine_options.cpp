// tqserve engine options: --exact and the Flash-Next prefill parameters parse into EngineOptions, keep -1 (engine default) when absent,
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
    expect(d.exact, "--exact defaults to on");
    expect(parse({"--exact", "on"}).exact && !parse({"--exact", "off"}).exact, "--exact on|off");
    expect(throws({"--exact", "fast"}) && throws({"--mode", "exact"}) && throws({"--allow-inexact"}), "unknown values / old options rejected");
    // --exact on conflicts: listed and warned; the asked options win (no question, no stop)
    static const char * (*no_env)(const char *) = [](const char *) -> const char * { return nullptr; };
    static const char * (*cpu_miss)(const char *) = [](const char * n) -> const char * { return std::strcmp(n, "TQFN_CPU_MISS") ? nullptr : "4"; };
    expect(tqllm::exact_conflicts(parse({}), false, false, no_env).empty(), "exact alone: no conflict");
    expect(tqllm::exact_conflicts(parse({"--exact", "off", "--prefill-int8", "on"}), false, false, no_env).empty(), "--exact off: never a conflict");
    expect(tqllm::exact_conflicts(parse({"--prefill-int8", "on"}), false, false, no_env).size() == 1, "exact + --prefill-int8 on");
    expect(tqllm::exact_conflicts(parse({"-ctk", "q8"}), false, false, no_env).size() == 1, "exact + lossy KV");
    expect(tqllm::exact_conflicts(parse({}), true, false, cpu_miss).size() == 1 &&
           tqllm::exact_conflicts(parse({}), false, false, cpu_miss).empty(), "TQFN_* only for Flash-Next");
    expect(tqllm::exact_conflicts(parse({"--prompt-cache", "on"}), true, false, no_env, 4).size() == 1 &&
           tqllm::exact_conflicts(parse({"--prompt-cache", "on"}), true, false, no_env, 1).empty() &&
           tqllm::exact_conflicts(parse({"--prompt-cache", "on", "--exact", "off"}), true, false, no_env, 4).empty(), "Flash-Next prompt cache: a conflict with several slots only");
    expect(tqllm::exact_conflicts(parse({}), false, true, no_env).size() == 1, "engine without an exact path");
    expect(!tqllm::tail_prefill_reuse(parse({})), "Metal tail prefill: off under --exact on");
    expect(tqllm::tail_prefill_reuse(parse({"--prompt-cache", "on"})), "Metal tail prefill: --prompt-cache on asks for it");
    expect(tqllm::tail_prefill_reuse(parse({"--exact", "off"})), "Metal tail prefill: on with --exact off");
    std::ostringstream err;
    tqllm::warn_exact_conflicts(tqllm::exact_conflicts(parse({"--prefill-int8", "on"}), false, false, no_env), err);
    expect(err.str().find("WARN --exact on with --prefill-int8 on") != std::string::npos &&
           err.str().find("the asked option wins") != std::string::npos, "the warning names the option and who wins");
    expect(d.graph_keep_mb == 384 && parse({"--graph-keep-mb", "128"}).graph_keep_mb == 128 &&
           parse({"--graph-keep-mb", "0"}).graph_keep_mb == 0, "--graph-keep-mb (384 default, 0 = by count only)");
    expect(throws({"--graph-keep-mb", "-1"}), "negative --graph-keep-mb rejected");
    expect(throws({"--prefill-reserve-mb", "-5"}), "negative --prefill-reserve-mb rejected");
    expect(throws({"--prefill-chunk"}), "missing value rejected");
    std::printf(fail ? "ENGINE_OPTIONS FAIL\n" : "ENGINE_OPTIONS PASS\n");
    return fail ? 1 : 0;
}
