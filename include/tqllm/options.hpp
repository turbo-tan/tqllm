#pragma once
// Engine options, fixed when a Model is built. Every tool parses them with parse_engine_option (llama.cpp names
// where llama.cpp has the option); tqserve's memory fit fills the placement fields itself.
#include <cstdint>
#include <string>

#include "tqllm/rope_scaling.hpp"

namespace tqllm {
struct EngineOptions {
    std::string cache_k, cache_v;  // -ctk / -ctv KV codecs (tq3, q8, q4, tcq3 / tq3, tq2s, tcq2, q8, q4); only one set:
                                   // V = K, or K = tq3. --kv FORMAT names a whole-cache format (tqserve --list-kv)
    std::string kv;
    std::string kv_store = "flat";  // --kv-store POLICY: where KV rows live (src/kv/kv_store.hpp; flat, paged, tiered)
    RopeScaling rope;   // --rope-scaling none|linear|yarn --rope-scale F --yarn-orig-ctx N ... (llama.cpp names)
    int kv_compact = 0;    // --kv-compact R: TqCache compaction (opt-in): while the server is idle, old evicted 8K blocks become
                           // stand-ins R times smaller that the GPU reads directly (approximate; staged + tq3 KV only)
    bool kv_compact_turn = false;  // --kv-compact-at idle|turn: turn compacts after each request's prefill, fitted on its
                                   // queries (the question), at the cost of that time before the first token
    int prefill_int8 = -1;  // --prefill-int8 on|off: prefill GEMMs in int8 (per-row x per-token scales) default on RTX 30 / 40 / 50, opt-in elsewhere (sm_80+)
    int prefill_fp8 = -1;  // --prefill-fp8 on|off: prefill GEMMs in FP8 (E4M3, per-row x per-token scales); default on GB10 (validated), opt-in on other sm_89+ GPUs
    bool prompt_cache = false;  // --prompt-cache on|off: one slot keeps the previous request's cache and prefills only the
                                // part of a new prompt past the shared prefix (on with --kv-compact)
    bool kv_stage = true;  // --kv-stage on|off: with --kv-host, old positions move to host and are staged back per layer at
                           // link speed (one slot, compressed KV); off: the newest positions stay in host memory
    long kv_host = -1;  // --kv-host N: last N positions of each sequence's KV in host memory (read over PCIe); -1: the fit
                        // decides (0 when the context fits, else the fewest that do)
    std::string draft_model;   // -md / --model-draft: DFlash2 drafter GGUF (the MTP head is then not loaded)
    bool native_mtp = false;   // --mtp: draft with the model's own MTP (nextn) head, no -md (text-only, greedy, --np 1; depth 4 by default)
    std::string draft_vocab;   // --draft-vocab FILE: token ids of the drafter / MTP head vocabulary subset
    int draft_window = -1;     // --draft-window N: drafter attention window over the context (-1: the GGUF value, 0: whole context)
    int draft_retrieve = 0;    // --draft-retrieve C: C x 16 positions of the drafter window are far chunks it retrieves (one slot)
    bool tree = true;          // --no-tree: chain verify instead of tree verify with a drafter
    int lookup = 0;            // --lookup L: prompt-lookup chain of up to L tokens beside the draft tree (copy tasks)
    int lookup_first = 0;      // --lookup-first K: a lookup match of >= K tokens takes the whole verify block
    int ubatch = 0;            // -ub / --ubatch-size: prefill buffer columns (0: engine default)
    // Flash-Next layer-major prefill (.tqfn); -1 = the engine default
    int prefill_chunk = -1;       // --prefill-chunk N: prompt tokens per prefill chunk (default: the manifest's chunk=)
    int prefill_sub = -1;         // --prefill-sub N: tokens per dense sub-chunk inside a chunk (2048)
    int prefill_stage = -1;       // --prefill-stage N: experts per staging bank half for streamed experts (192)
    long prefill_reserve_mb = -1; // --prefill-reserve-mb N: extra memory freed (expert banks borrowed) per long prompt (3400)
    long prefill_arena_mb = -1;   // --prefill-arena-mb N: prefill scratch arena (3800)
    long dense_cache_mb = -1;     // --dense-cache-mb N: FP16 dense weight images reused by a chunk's sub-chunks (600; 0 off)
    std::string prefill_borrow;   // --prefill-borrow auto|all|N: expert banks released for a long prompt (auto: the fewest the plan needs)
    // memory placement; the fit (Model) sets them when the device is short
    bool embed_host = false;   // token embeddings in host memory
    bool head_host = false;    // int8 LM head in host memory
    int gemm_rows = 0;         // prefill FP16 weight blocks of this many rows (0: whole matrices)
    int n_cpu_moe = 0;         // -ncmoe N: MoE expert weights of the first N layers in pinned host memory
                               //   (llama.cpp --n-cpu-moe parity; read over PCIe by the decode kernels)
    long moe_host_mb = 0;      // --moe-host-mb M: the least-routed experts (by --moe-profile) in pinned host memory, about M MiB
                               //   (auto, -1: what the file needs beyond the free device memory);
                               //   every layer keeps its hot experts in VRAM (output identical to all-VRAM)
    long moe_cache_mb = -1;    // --moe-cache-mb C: VRAM cache of recently used host experts (decode), about C MiB, exact;
                               //   auto (-1): 12 % of the host experts (256..1536 MiB), the host share raised by the same
    std::string moe_profile;   // --moe-profile FILE: routing counts (default MODEL.experts); TQLLM_MOE_PROFILE=FILE writes one
    // memory fit (llama.cpp names): -fit on|off, -fitt MiB free after loading, -fitc smallest context it may set
    bool fit = true;
    long fit_target_mb = -1;  // -1: 768, or 256 under WSL (WDDM; measured 2026-10-08, results/20261008_kv_host_tail)
    uint32_t fit_min_ctx = 4096;
    bool fit_lossy = true;  // -fitl off: refuse a load that needs another KV codec / context / prefill split (default: apply and announce)
    // parallel slots (tqserve --np): --slot-reserve MiB free per slot after loading, else the load refuses the --np and
    // names the largest that fits; --graph-cache N captured CUDA graphs kept (~12 MB each, least recently used evicted)
    long slot_reserve_mb = 128;
    int graph_cache = 256;
    long graph_keep_mb = 384;  // a graph miss also evicts while less than this is free; the fit sets 0 under WSL

    std::string kv_format() const;  // the KV format name these options select ("bf16" when none)
};

// The one drafter a run loads: DFlash2 with -md, the model's MTP head with --mtp, neither without them. Throws for -md
// with --mtp: the unused drafter is never loaded or run.
struct Drafters { bool dflash = false, mtp = false; };
Drafters drafters(const EngineOptions & o);

// Parses the engine option at argv[i] (moving i past its value); false when argv[i] is not one.
bool parse_engine_option(int argc, char ** argv, int & i, EngineOptions & o);
const char * engine_options_usage();

// Stops the tool when a removed TQLLM_* variable is set (config/flags.yaml kind retired names the replacement) and
// warns about unknown TQLLM_* names.
void check_env_flags(const char * tool);
}  // namespace tqllm
