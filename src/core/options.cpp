#include "tqllm/options.hpp"

#include "flags_gen.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#if defined(__APPLE__)
#include <crt_externs.h>
#endif

namespace tqllm {
namespace {
std::string codec(const std::string & t) {  // llama.cpp cache type names accepted as well
    if (t == "q8_0") return "q8";
    if (t == "q4_0") return "q4";
    if (t == "tq3_0") return "tq3";
    if (t == "f16") return "bf16";
    return t;
}
}  // namespace

std::string EngineOptions::kv_format() const {
    if (!kv.empty()) return kv;
    if (cache_k.empty() && cache_v.empty()) return "bf16";
    const std::string k = cache_k.empty() ? "tq3" : codec(cache_k);
    const std::string v = cache_v.empty() ? k : codec(cache_v);
    return k == "bf16" && v == "bf16" ? "bf16" : k + "/" + v;
}

std::vector<std::string> exact_conflicts(const EngineOptions & o, bool flashnext, bool no_exact_path,
                                         const char * (*env)(const char *), int slots) {
    std::vector<std::string> out;
    if (!o.exact) return out;
    if (o.prefill_int8 == 1) out.push_back("--prefill-int8 on: int8 prefill GEMMs (KLD 0.003-0.007 vs FP16 prefill)");
    if (o.prefill_fp8 == 1) out.push_back("--prefill-fp8 on: FP8 prefill GEMMs (KLD 0.003-0.005 vs FP16 prefill)");
    if (const std::string kv = o.kv_format(); kv != "bf16") out.push_back("KV " + kv + " (-ctk / -ctv / --kv): a lossy KV cache");
    if (flashnext)
        for (const char * v : {"TQFN_PREFILL_INT8", "TQFN_DENSE_INT8", "TQFN_TAIL_SKIP", "TQFN_CPU_MISS"})
            if (const char * s = env(v); s && std::atoi(s) != 0) out.push_back(std::string(v) + " set: an approximation");
    if (no_exact_path) out.push_back("MiniMax / GLM have no exact path yet: they run their approximate path");
    if (flashnext && o.prompt_cache && slots > 1)
        out.push_back("--prompt-cache on: a reused prompt prefix splits the prefill, so the text can differ from an uncached prompt");
    for (const auto & p : o.inexact_patches) out.push_back("--plugin patch " + p + ": not byte-identical to the built-in kernel");
    return out;
}
bool tail_prefill_reuse(const EngineOptions & o) { return !o.exact || o.prompt_cache; }
void warn_exact_conflicts(const std::vector<std::string> & conflicts, std::ostream & err) {
    for (const auto & c : conflicts) err << "tqserve: WARN --exact on with " << c << " (the asked option wins)\n";
}

bool parse_engine_option(int argc, char ** argv, int & i, EngineOptions & o) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
        return argv[++i];
    };
    auto count = [&](int lo) {
        const int n = std::stoi(value());
        if (n < lo) throw std::invalid_argument(a + " must be >= " + std::to_string(lo));
        return n;
    };
    if (a == "-ctk" || a == "--cache-type-k") o.cache_k = value();
    else if (a == "-ctv" || a == "--cache-type-v") o.cache_v = value();
    else if (a == "--kv") o.kv = value();
    else if (a == "--exact") {
        const std::string v = value();
        if (v != "on" && v != "off") throw std::invalid_argument("--exact must be on or off");
        o.exact = v == "on";
    }
    else if (a == "--qsa-from") {
        const std::string v = value();
        o.qsa_from = v == "off" ? 0 : std::stoi(v);
        if (o.qsa_from != 0 && o.qsa_from < 4096) throw std::invalid_argument("--qsa-from must be off or >= 4096");
    }
    else if (a == "--kv-store") o.kv_store = value();
    else if (a == "--kv-host") o.kv_host = count(0);
    else if (a == "--kv-compact") o.kv_compact = count(0);
    else if (a == "--kv-stage") {
        const std::string v = value();
        if (v != "on" && v != "off") throw std::invalid_argument("--kv-stage must be on or off");
        o.kv_stage = v == "on";
    }
    else if (a == "--kv-compact-at") {
        const std::string v = value();
        if (v != "idle" && v != "turn") throw std::invalid_argument("--kv-compact-at must be idle or turn");
        o.kv_compact_turn = v == "turn";
    }
    else if (a == "--prefill-chunk") o.prefill_chunk = count(256);
    else if (a == "--prefill-sub") o.prefill_sub = count(16);
    else if (a == "--prefill-stage") o.prefill_stage = count(1);
    else if (a == "--prefill-reserve-mb") o.prefill_reserve_mb = count(0);
    else if (a == "--prefill-arena-mb") o.prefill_arena_mb = count(0);
    else if (a == "--dense-cache-mb") o.dense_cache_mb = count(0);
    else if (a == "--prefill-borrow") {
        const std::string v = value();
        if (v != "all" && v != "auto" && (v.empty() || v.find_first_not_of("0123456789") != std::string::npos))
            throw std::invalid_argument("--prefill-borrow must be all, auto or a bank count");
        o.prefill_borrow = v;
    }
    else if (a == "--prefill-int8") {
        const std::string v = value();
        if (v != "on" && v != "off") throw std::invalid_argument("--prefill-int8 must be on or off");
        o.prefill_int8 = v == "on" ? 1 : 0;
    }
    else if (a == "--prefill-fp8") {
        const std::string v = value();
        if (v != "on" && v != "off") throw std::invalid_argument("--prefill-fp8 must be on or off");
        o.prefill_fp8 = v == "on" ? 1 : 0;
    }
    else if (a == "--prompt-cache") {
        const std::string v = value();
        if (v != "on" && v != "off") throw std::invalid_argument("--prompt-cache must be on or off");
        o.prompt_cache = v == "on";
    }
    else if (a == "--rope-scaling") o.rope.type = value();
    else if (a == "--rope-scale") o.rope.factor = std::stof(value());
    else if (a == "--yarn-orig-ctx") o.rope.orig_ctx = std::stol(value());
    else if (a == "--yarn-ext-factor") o.rope.ext_factor = std::stof(value());
    else if (a == "--yarn-attn-factor") o.rope.attn_factor = std::stof(value());
    else if (a == "--yarn-beta-fast") o.rope.beta_fast = std::stof(value());
    else if (a == "--yarn-beta-slow") o.rope.beta_slow = std::stof(value());
    else if (a == "-md" || a == "--model-draft") o.draft_model = value();
    else if (a == "--tp-rank") o.tp_rank = std::stoi(value()), o.tp_world = 2;
    else if (a == "--tp-peer") o.tp_peer = value();
    else if (a == "--tp-port") o.tp_port = std::stoi(value());
    else if (a == "--plugin") o.plugins.push_back(value());
    else if (a == "--hot-map") o.hot_map = value();
    else if (a == "--draft-vocab") o.draft_vocab = value();
    else if (a == "--mtp") o.native_mtp = true;
    else if (a == "--draft-window") o.draft_window = count(0);
    else if (a == "--draft-retrieve") o.draft_retrieve = count(0);
    else if (a == "--no-tree") o.tree = false;
    else if (a == "--lookup") o.lookup = count(0);
    else if (a == "--lookup-first") o.lookup_first = count(0);
    else if (a == "-ub" || a == "--ubatch-size") o.ubatch = count(1);
    else if (a == "--gemm-rows") o.gemm_rows = count(0);
    else if (a == "-ncmoe" || a == "--n-cpu-moe") o.n_cpu_moe = count(0);
    else if (a == "--moe-host-mb") { const std::string v = value(); o.moe_host_mb = v == "auto" ? -1 : std::stol(v); if (o.moe_host_mb < -1) throw std::invalid_argument("--moe-host-mb must be auto or >= 0"); }
    else if (a == "--moe-cache-mb") { const std::string v = value(); o.moe_cache_mb = v == "auto" ? -1 : std::stol(v); if (o.moe_cache_mb < -1) throw std::invalid_argument("--moe-cache-mb must be auto or >= 0"); }
    else if (a == "--moe-profile") o.moe_profile = value();
    else if (a == "-fit" || a == "--fit") {
        const std::string v = value();
        if (v != "on" && v != "off") throw std::invalid_argument("--fit must be on or off");
        o.fit = v == "on";
    } else if (a == "-fitt" || a == "--fit-target") o.fit_target_mb = count(0);
    else if (a == "-fitc" || a == "--fit-ctx") o.fit_min_ctx = uint32_t(count(256));
    else if (a == "-fitl" || a == "--fit-lossy") {
        const std::string v = value();
        if (v != "on" && v != "off") throw std::invalid_argument("--fit-lossy must be on or off");
        o.fit_lossy = v == "on";
    }
    else if (a == "--slot-reserve") o.slot_reserve_mb = count(0);
    else if (a == "--graph-cache") o.graph_cache = count(1);
    else return false;
    return true;
}

const char * engine_options_usage() {
    return "  engine: [--exact on|off (on)] [-ctk K] [-ctv V] (KV codecs; llama.cpp names q8_0 / q4_0 / tq3_0 too) | [--kv FORMAT] (--list-kv) [--kv-store flat] [--kv-host N [--kv-stage on|off] [--kv-compact R [--kv-compact-at idle|turn]]] [--prompt-cache on|off] [--prefill-fp8 on|off] [--prefill-int8 on|off]\n"
           "          [--qsa-from N|off (QSA models: sparse attention past N keys; default the file's, else 49152)]\n"
           "          [--plugin PATH (repeatable: weight-format plugin, CUDA builds)]\n"
           "          [--rope-scaling none|linear|yarn --rope-scale F [--yarn-orig-ctx N]] (context past training, llama.cpp names)\n"
           "          [-md DRAFTER.gguf] [--draft-vocab FILE] [--draft-window N] [--draft-retrieve C] [--no-tree] [--lookup L [--lookup-first K]] [-ub N] [-ncmoe N (first N layers\' MoE experts in host memory)] [--moe-host-mb M|auto [--moe-profile FILE] (least-routed experts in host memory) [--moe-cache-mb C|auto (VRAM cache of recently used host experts, default auto)]]\n"
           "          Flash-Next prefill: [--prefill-chunk N (16384)] [--prefill-sub N (2048)] [--prefill-stage N (192)] [--prefill-reserve-mb N (1000)]\n"
           "          [--prefill-arena-mb N (3800)] [--dense-cache-mb N (600)] [--prefill-borrow auto|all|N (auto)]\n"
           "          [-fit on|off (on)] [-fitt MiB free after loading (768; 256 under WSL)] [-fitc smallest context the fit may set (4096)] [--mtp (draft with the model's own MTP head; text-only, greedy, --np 1) (not with -md)] [-fitl on|off (on: pick KV / context / prefill buffers to fit and announce them; off: refuse instead)] [--gemm-rows N (prefill FP16 weight blocks of N rows; 0 whole matrices)]\n"
           "          [--slot-reserve MiB free per parallel slot (128)] [--graph-cache captured CUDA graphs kept (256)]\n";
}

void check_env_flags(const char * tool) {
    int retired = 0;
#if defined(__APPLE__)
    char ** environment = *_NSGetEnviron();
#else
    char ** environment = environ;
#endif
    for (char ** e = environment; e && *e; ++e) {
        if (std::strncmp(*e, "TQLLM_", 6) != 0) continue;
        const std::string name(*e, std::strcspn(*e, "="));
        const auto * f = flags::find(name.c_str());
        if (!f) {
            std::fprintf(stderr, "%s: warning: unknown variable %s (docs/FLAGS.md)\n", tool, name.c_str());
        } else if (f->kind == flags::Kind::Retired) {
            std::fprintf(stderr, "%s: %s was removed; %s\n", tool, name.c_str(), f->replaced_by);
            ++retired;
        }
    }
    if (retired) std::exit(2);
}
Drafters drafters(const EngineOptions & o) {
    if (!o.draft_model.empty() && o.native_mtp)
        throw std::runtime_error("-md (DFlash2) and --mtp (MTP head) are exclusive: pick one drafter");
    return {!o.draft_model.empty(), o.native_mtp};
}
}  // namespace tqllm
