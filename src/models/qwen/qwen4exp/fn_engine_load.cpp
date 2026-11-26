#include <algorithm>
#include <cmath>
#include <cstdint>
// Flash-Next engine load: manifest, checkpoint and expert caches, warm-up, expert slots and the context fit.
#include <fstream>
#include <iterator>
#include "debug.hpp"
#include "device_free.hpp"
#include "fn_engine.hpp"
#include "fn_ablate.hpp"
#include "core/tp_comm.hpp"
#include "core/split_bounds.hpp"
#include <filesystem>
#include "decode_rules.hpp"
#include "memory_plan.hpp"
#include <chrono>
#include "hot_cold_policy.hpp"
#include "gsq_q2_host.hpp"
#include "expert_prediction.hpp"
#include "expert_block_batch.hpp"

namespace tqllm::flashnext {
namespace {
std::array<ExpertPayload,3> mapped_cold_payload(FnServe & f,int layer,int expert) {
    auto raw=f.cold_store->load_mapped(layer,expert);std::array<ExpertPayload,3> out;
    for(int p=0;p<3;++p)out[p]={GGML_GSQ_Q2_0,{raw.parts[p].data(),raw.parts[p].size()},raw.owner,true};
    return out;
}
}


// decode scratch per verify row (measured on a GB10: 11-13 1,604 MB high at np 16 x 3 rows, ~30.7 MB a row; 11-14 2,182 MB
// at np 32 x 2 rows, ~32.1 MB a row; a 31 MB row filled the arena to its last MB) and the rows of a batched decode step
// (kBatchRowsMax, decode_rules.hpp): no step's temporaries spill out of the arena
constexpr int kArenaRowMb = 34, kArenaRowsMax = tqllm::flashnext::kBatchRowsMax;
void FnServe::fit_tier(uint64_t need) {
    if (!tier || !tier->chunks()) return;
    const int before = tier->attached_chunks();
    const int now = tqllm::fit_tier_chunks(*tier, need,
        [] { std::ifstream mi("/proc/meminfo"); const std::string t((std::istreambuf_iterator<char>(mi)), std::istreambuf_iterator<char>());
             return uint64_t(tqllm::host_available_bytes(t)); },
        [](size_t bytes) -> char * { char * m = nullptr; if (cudaMallocHost(reinterpret_cast<void **>(&m), bytes) != cudaSuccess) { (void) cudaGetLastError(); return nullptr; } return m; },
        [](char * m) { if (m) ck(cudaFreeHost(m)); });
    if (now != before)
        std::fprintf(stderr, "tqllm: host expert tier: elastic chunks %d -> %d (%.1f GB kept available)\n", before, now, need / 1e9);
}

FlashNextEngine::FlashNextEngine(const std::string & manifest, uint32_t context, int nslots, const FnConfig & options) : context_(context), slots_(size_t(std::max(1, nslots))) {
    // up to 16 sequences: each layer keeps np x topk slots and a request makes room for its K/V at its start (decode_rules.hpp
    // lift_layer_floor, memory_plan.hpp request_shrink_bytes); 3090 np 10 / 12 exact, all texts == np 1 (11-05). With every
    // expert resident (slots = 512, a GB10) no layer runs short of slots: up to 32, the batched kernels' sequence limit
    // (GdnSeqs / AttnSeqs / QsaBlockJobs kMax); checked once the manifest is read
    require(nslots <= 32, "Flash-Next serves up to --np 32");
    auto & f = f_; std::ifstream in(manifest); require(bool(in), "cannot open Flash-Next manifest " + manifest);
    const auto dir = std::filesystem::path(manifest).parent_path();
    auto resolve = [&](const std::string & v) { return std::filesystem::path(v).is_absolute() ? v : (dir / v).string(); };
    bool slots_given = false;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        const auto eq = line.find('='); require(eq != std::string::npos, "manifest line without '=': " + line);
        const auto k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "gguf") f.ggufs.push_back(resolve(v)); else if (k == "mtp") f.mtp = resolve(v);
        else if (k == "dflash") f.dflash = resolve(v);
        else if (k == "hot_map") f.hot_map_path = resolve(v);
        else if (k == "cold_verify") { require(v=="fnv" || v=="xxh64", "invalid cold_verify mode"); f.cold_fast=v=="xxh64"; }
        else if (k == "cold_store") f.cold_store_path = resolve(v);
        else if (k == "cold_readers") {
            size_t pos=0; f.cold_readers=std::stoi(v,&pos);
            require(pos==v.size() && f.cold_readers>=0 && f.cold_readers<=8,"cold_readers requires0..8");
        }
        else if(k=="cold_prefetch_k") {
            size_t pos=0;f.cold_prefetch_k=std::stoi(v,&pos);
            require(pos==v.size() && f.cold_prefetch_k>=0 && f.cold_prefetch_k<=8,"cold_prefetch_k requires0..8");
        }
        else if(k=="cold_streaming") {
            require(v=="0" || v=="1","cold_streaming requires0 or1");f.cold_streaming=v=="1";
        }
        else if(k=="cold_overlap") {
            size_t pos=0;f.cold_overlap=std::stoi(v,&pos);
            require(pos==v.size() && f.cold_overlap>=0 && f.cold_overlap<=2,"cold_overlap requires0..2");
        }
        else if(k=="cold_mmap") {
            require(v=="0" || v=="1","cold_mmap requires0 or1");f.cold_mmap=v=="1";
        }
        else if(k=="cold_reclaim_hot") {
            require(v=="0" || v=="1","cold_reclaim_hot requires0 or1");f.cold_reclaim_hot=v=="1";
        }
        else if (k == "cold_host_mb") {
            size_t pos=0;const int mb=std::stoi(v,&pos);
            require(pos==v.size() && mb>=0 && mb<=65536,"cold_host_mb requires0..65536");
            f.cold_host.set_capacity(size_t(mb)<<20);
        }
        else if (k == "hot_slots") {
            size_t pos = 0; f.hot_slots = std::stoi(v, &pos);
            require(pos == v.size() && f.hot_slots >= 1 && f.hot_slots <= 512, "hot_slots requires1..512");
        }
        else if (k == "cold_slots") f.cold_slots = std::stoi(v);
        else if (k == "experts") f.experts = resolve(v); else if (k == "draft_vocab") f.draft_vocab = resolve(v);
        else if (k == "layer_slots") f.layer_slots = v;
        else if (k == "slots") { slots_given = true; f.slots_auto = v == "auto"; f.slots = f.slots_auto ? kAutoStartSlots : std::stoi(v); } else if (k == "budget_gb") f.budget_gb = std::stod(v);
        else if (k == "drafts") f.drafts = std::stoi(v); else if (k == "chunk") f.chunk = std::stoi(v);
        else if (k == "gsq_rco") f.gsq_rco = v != "0";
        else throw std::runtime_error("unknown Flash-Next manifest key " + k);
    }
    require(f.cold_store_path.empty() || !f.hot_map_path.empty(), "cold_store requires hot_map");
    require(!f.hot_slots || !f.hot_map_path.empty(), "hot_slots requires hot_map");
    require(!f.cold_streaming || (f.cold_mmap && f.cold_readers>=2),"cold_streaming requires cold_mmap and at least2 cold_readers");
    require(!f.cold_prefetch_k || (f.cold_mmap && f.cold_readers>=2),"cold_prefetch_k requires cold_mmap and at least2 cold_readers");
    require(!f.cold_mmap || !f.cold_store_path.empty(),"cold_mmap requires a persistent cold store");
    require(!f.cold_reclaim_hot || !f.cold_store_path.empty(),"cold_reclaim_hot requires a persistent cold store");
    require(!f.cold_readers || !f.cold_store_path.empty(),"cold_readers requires a persistent cold store");
    require(!f.ggufs.empty() && !f.mtp.empty(), "manifest needs gguf and mtp lines");
    if (!slots_given) { f.slots_auto = true; f.slots = kAutoStartSlots; }  // sized from the device at start-up
    f.chunk = prefill_chunk(options.tune.chunk, f.chunk);  // --prefill-chunk (16384) over the manifest, also when it has none
    if (!f.draft_vocab.empty()) setenv("TQFN_DRAFT_VOCAB", f.draft_vocab.c_str(), 1);
    { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); uint64_t keep = UINT64_MAX;
      ck(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep)); }
    // per-step temporaries grow with the prefill chunk (one layer at a time in prefill: measured 117 MB at chunk 16-64, 245 MB at 256; decode verify steps keep one span); TQFN_ARENA_MB overrides
    if (!f.hot_map_path.empty()) {
        require(f.experts.empty() && !tqllm::debug_cstr("keep_map") && !tqllm::debug_cstr("expert_profile"),
                "hot/cold: use original GGUFs, not pruning or a single-format packed bank/profile");
        require(nslots == 1 && f.cold_slots >= 1 && f.cold_slots <= 512, "hot/cold pilot: one session and cold_slots 1..512 required");
        f.chunk = std::min(f.chunk, 8); // pilot uses the same mixed-precision path for prefill and decode
    }
    f.arena.size = size_t(std::getenv("TQFN_ARENA_MB") ? std::atoi(std::getenv("TQFN_ARENA_MB")) : std::max({256,
                   // layer-major chunks (256+) run the dense part in 256-token sub-chunks: measured 690 MB at 4096, 1375 at 8192
                   // (start-up arena capped at the 8K-chunk size: a bigger chunk's arena is borrowed per request, not held while
                   // the banks are built)
                   f.chunk >= 256 ? std::min(f.chunk, 8192) / 5 + 64 : 2 * f.chunk + 128,
                   slots_.size() > 1 ? kArenaRowMb * std::min(kArenaRowsMax, int(slots_.size()) * (f.drafts + 1)) + 128 : 0})) << 20;  // + one span per batched verify row
    ck(cudaMalloc(&f.arena.base, f.arena.size)); g_arena = &f.arena; f.arena_prefill = f.arena.size;
    // --prefill-arena-mb: the prefill arena alone (bigger dense sub-chunks); the startup and decode arena keep their size,
    // the request's borrow evicts banks for the difference and they refill after the prompt
    f.arena_prefill = std::max({f.arena_prefill, size_t(std::max(0L, options.tune.arena_mb)) << 20, tqllm::prefill_arena_bytes(f.chunk)});
    // decode needs far less than a layer-major prefill (one step's temporaries); the arena shrinks after each long prompt
    f.arena_decode = std::getenv("TQFN_ARENA_MB") ? f.arena.size : std::min(f.arena.size, size_t(std::max(256,
                     slots_.size() > 1 ? kArenaRowMb * std::min(kArenaRowsMax, int(slots_.size()) * (f.drafts + 1)) + 128 : 0)) << 20);
    f.arg_store = {"tqserve", "-"}; for (const auto & g : f.ggufs) f.arg_store.push_back(g);
    f.arg_store.push_back("--mtp"); f.arg_store.push_back(f.mtp); if (f.gsq_rco) f.arg_store.push_back("--gsq-rco");
    for (auto & a : f.arg_store) f.argv.push_back(a.data());
    f.fixtures = std::make_unique<Fixtures>();
    f.cp = std::make_unique<Checkpoint>(int(f.argv.size()), f.argv.data()); f.cp->resident = true; f.cp->reclaim_expert_pages=f.cold_reclaim_hot; f.cp->run = options;
    if (!options.exact)  // approximate mode: the BF16 mixers as Q8_0 (0.63 of 1.26 GB per decode step)
        if (const int q = f.cp->requant_hc_mixers_q8()) std::fprintf(stderr, "tqllm: --exact off: %d hyper-connection mixer matrices BF16 -> Q8_0\n", q);
    { int dev = 0, um = 0; cudaGetDevice(&dev);
      f.cp->run.unified_memory = cudaDeviceGetAttribute(&um, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) == cudaSuccess && um; }
    const auto & c = f.cp->config; auto & checkpoint = *f.cp; FnConfig & run = checkpoint.run; FnRuntime & rt = checkpoint.rt;
    if (run.split) {  // --split layer: this rank's trunk layers; what one rank cannot cover yet is refused
        require(!run.ep, "--split layer and --split experts cannot be combined");
        require(f.hot_map_path.empty(), "--split layer: hot/cold expert manifests run on one GPU so far");
        // layers by free device memory: each layer's trunk experts (the banks fill what is left) against what every rank
        // holds besides them (dense tensors, the MTP layer, the prefill arena, buffers)
        std::vector<uint64_t> cost(size_t(c.layers), 0);
        for (int l = 0; l < c.layers; ++l)
            for (const auto & p : checkpoint.expert_payload(l, 0)) cost[size_t(l)] += uint64_t(p.bytes.size) * uint64_t(c.experts);
        std::error_code ec;
        const uint64_t fixed = uint64_t(std::filesystem::file_size(f.mtp, ec)) + (uint64_t(std::max(0L, run.tune.arena_mb) + 3072) << 20);
        // a discrete card's rank pins its layers' experts in RAM when they fit (build_pack_in_memory: 60 % of the RAM its box
        // gives each rank), else reads them from the GGUF per miss (2x 3090, .77 with 31 GB pinning nothing for 24 layers:
        // 28-53 tok/s vs 85-114 on one card): its layers are capped at what it can pin, less one layer for the rounding
        // ... and at most what keeps the box's 20 GB reserve beside the host K/V tail of those layers at this context and, on
        // the head rank, the prompt buffers of a full-context request (split_ram_cap)
        tqllm::HostRoom room;
        { int dev = 0, integ = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&integ, cudaDevAttrIntegrated, dev);
          if (!integ && f.experts.empty() && f.hot_map_path.empty()) {
              const int local = std::max(1, run.split->local_ranks());
              std::ifstream mi("/proc/meminfo"); const std::string text((std::istreambuf_iterator<char>(mi)), std::istreambuf_iterator<char>());
              const uint64_t avail = tqllm::meminfo_available(text) / uint64_t(local), ram = tqllm::host_ram_bytes() / uint64_t(local);
              uint64_t attn = 0; for (int l = 0; l < c.layers; ++l) attn += checkpoint.has_tensor("blk." + std::to_string(l) + ".indexer.k_proj.weight");
              const uint64_t tail = kv_host_enabled() && context > 32768 ? uint64_t(context) - 32768 : 0;
              const uint64_t kv_tail = attn * tail * 2 * uint64_t(checkpoint.attention_kvheads) * uint64_t(checkpoint.attention_dim) * 2;
              uint64_t pack = 0; for (uint64_t x : cost) pack += x;
              const bool head = nslots > 1 || run.split->rank() == run.split->world() - 1;
              const uint64_t head_bytes = head && int(context) > 4096 ? uint64_t(context) * uint64_t(c.hidden) * (uint64_t(c.hc) + 1) * 4 : 0;
              std::error_code mec;
              const uint64_t fixed_host = tqllm::flashnext::kSplitHostFixed + uint64_t(std::filesystem::file_size(f.mtp, mec));
              const double kvp = pack ? double(kv_tail) / double(pack) : 0.0;
              const uint64_t most = tqllm::flashnext::split_ram_cap(avail, ram, tqllm::flashnext::kSplitHostReserve, head_bytes, kvp, fixed_host);
              const uint64_t cheapest = *std::min_element(cost.begin(), cost.end());
              room.limited = true; room.cap = most; room.available = avail;
              room.need_one = head_bytes + fixed_host + uint64_t(double(cheapest) * (1.02 + kvp)) + tqllm::flashnext::kSplitHostReserve;
              std::fprintf(stderr, "tqllm: --split layer: %.1f GB available here, 20 GB kept, %.1f GB for a full prompt's buffers, "
                           "%.2f GB host K/V per GB of experts, 4 GB + the MTP pack fixed: experts pinned here up to %.1f GB\n", double(avail) / 1e9,
                           double(head_bytes) / 1e9, pack ? double(kv_tail) / double(pack) : 0.0, double(most) / 1e9);
          } }
        run.ranges = tqllm::agree_split(*run.split, cost, fixed, tqllm::device_free_bytes(), run.split_at, room);
        run.lr = run.ranges[size_t(run.split->rank())];
        std::fprintf(stderr, "tqllm: --split layer rank %d of %d: trunk layers %u..%u of %d (head, MTP and sampling on every rank)\n",
                     run.split->rank(), run.split->world(), run.lr.lo, run.lr.hi - 1, c.layers);
    }
    require(nslots <= tqllm::flashnext::np_max(f.slots_auto, f.slots, c.experts),
            "Flash-Next serves up to --np 16 unless every expert is resident (slots=" + std::to_string(c.experts) + " in the manifest)");
    rt.kv_capacity = int(context);  // per slot
    int integrated = 0; { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev); }
    double pack_bytes = 0;  // the GGUF bytes of the trunk experts, about the pinned pack's size
    // --split layer: a rank packs only its own trunk layers, and the ranks of one machine share its RAM
    for (int l = 0; l < c.layers; ++l) if (!run.split || run.lr.owns(uint32_t(l)))
        for (const auto & p : checkpoint.expert_payload(l, 0)) pack_bytes += double(p.bytes.size) * c.experts;
    const double host_ram = double(tqllm::host_ram_bytes()) / (run.split ? std::max(1, run.split->local_ranks()) : 1);  // physical RAM capped by the container limit
    double pack_avail = 0, pack_beside = 0;  // one rank: the pack beside what the host tier keeps free (decode_rules.hpp)
    if (!run.split && !integrated) {
        std::ifstream mi("/proc/meminfo"); const std::string text((std::istreambuf_iterator<char>(mi)), std::istreambuf_iterator<char>());
        pack_avail = double(std::min<uint64_t>(tqllm::host_available_bytes(text), tqllm::host_ram_bytes()));
        uint64_t attn = 0; for (int l = 0; l < c.layers; ++l) attn += checkpoint.has_tensor("blk." + std::to_string(l) + ".indexer.k_proj.weight");
        const uint64_t tail = kv_host_enabled() && context > 32768 ? uint64_t(context) - 32768 : 0;
        const uint64_t rows_host = uint64_t(context) * uint64_t(c.hidden) * (nslots > 1 ? uint64_t(c.hc) + 1 : 1) * 4;
        std::error_code mec;
        pack_beside = double(rows_host + attn * tail * 2 * uint64_t(checkpoint.attention_kvheads) * uint64_t(checkpoint.attention_dim) * 2 +
                             uint64_t(std::filesystem::file_size(f.mtp, mec)) + tqllm::flashnext::kHostTierReserve);
    }
    const bool pack_in_memory = build_pack_in_memory(!f.experts.empty(), integrated != 0, !f.hot_map_path.empty(), pack_bytes * 1.02, host_ram,
                                                     pack_avail, pack_beside);
    if (f.experts.empty() && !integrated && !pack_in_memory)
        std::fprintf(stderr, "tqllm: experts read from the GGUF on demand (%s)\n", !f.hot_map_path.empty() ? "hot/cold tiers" :
                     "a pinned pack would not fit this machine's RAM");
    const bool on_demand = f.experts.empty() && !integrated && !pack_in_memory && f.hot_map_path.empty();
    // experts read from the GGUF on demand: a pinned host tier under the banks in the pinned pack's tiled layout (decided
    // before any bank), sized to what RAM is left beside a full-context prompt's host buffers (decode_rules.hpp)
    if (on_demand) {
        const auto t0 = std::chrono::steady_clock::now();
        f.tier_loader = std::make_unique<tqllm::flashnext::TierLoader>(checkpoint, true);
        std::vector<size_t> layer_bytes(size_t(c.layers), 0); uint64_t block = 0;
        for (int l = 0; l < c.layers; ++l) if (!run.split || run.lr.owns(uint32_t(l))) {
            uint64_t b = 0; for (const auto & p : f.tier_loader->parts(l, 0)) b += p.size();
            layer_bytes[size_t(l)] = size_t(b); block = std::max(block, b);
        }
        std::ifstream mi("/proc/meminfo"); const std::string text((std::istreambuf_iterator<char>(mi)), std::istreambuf_iterator<char>());
        const uint64_t avail = std::min<uint64_t>(tqllm::host_available_bytes(text), tqllm::host_ram_bytes());
        uint64_t attn = 0; for (int l = 0; l < c.layers; ++l) attn += checkpoint.has_tensor("blk." + std::to_string(l) + ".indexer.k_proj.weight");
        const uint64_t tail = kv_host_enabled() && context > 32768 ? uint64_t(context) - 32768 : 0;
        // a full-context prompt's host buffers: one sequence keeps the prompt's embeddings (the MTP layer's K/V written
        // chunk by chunk, no residual rows); several sequences or --split layer the residual rows too
        const uint64_t rows_host = uint64_t(context) * uint64_t(c.hidden) * (nslots > 1 || run.split ? uint64_t(c.hc) + 1 : 1) * 4;
        const uint64_t prompt_host = rows_host + attn * tail * 2 * uint64_t(checkpoint.attention_kvheads) * uint64_t(checkpoint.attention_dim) * 2;
        const uint64_t ring = 2 * uint64_t(c.experts) * block;  // the prefill stream ring: a whole layer per half (layer-outer)
        std::error_code mec; const uint64_t mtp_pack = uint64_t(std::filesystem::file_size(f.mtp, mec));  // pinned below
        uint64_t mean = 0; int owned = 0; for (size_t b : layer_bytes) if (b) { mean += b; ++owned; }
        mean = owned ? mean / uint64_t(owned) : 0;
        // the blocks the RAM holds at the mean expert size (each size class gets its share of the bytes)
        const int want = tqllm::flashnext::host_tier_blocks(avail, prompt_host + mtp_pack, ring, tqllm::flashnext::kHostTierReserve, mean,
                                                            uint64_t(c.layers) * uint64_t(c.experts), 2 * c.experts);
        size_t bytes = size_t(want) * mean;
        while (bytes && cudaMallocHost(reinterpret_cast<void **>(&f.tier_mem), bytes) != cudaSuccess) {
            (void) cudaGetLastError(); bytes = bytes / 8 * 7; if (bytes < size_t(2 * c.experts) * block) bytes = 0;
        }
        int blocks = bytes ? tqllm::HostExpertTier::blocks_for(layer_bytes, bytes) : 0;
        if (blocks > 0) {
            auto * loader = f.tier_loader.get();
            // the elastic chunks hold experts with the RAM a long prompt's host buffers take while no such prompt runs
            const int chunks = tqllm::flashnext::host_tier_chunks(prompt_host);
            f.tier = std::make_unique<tqllm::HostExpertTier>(c.layers, c.experts, layer_bytes, f.tier_mem, bytes,
                [loader](int layer, int expert, char * dst) { loader->load(layer, expert, dst); }, 8,
                size_t(tqllm::flashnext::kHostTierChunk), chunks);
            f.tier_rows_per_token = rows_host / std::max<uint64_t>(1, context);
            f.tier_kv_per_token = attn * 2 * uint64_t(checkpoint.attention_kvheads) * uint64_t(checkpoint.attention_dim) * 2;
            f.tier_ring_half = uint64_t(c.experts) * block;  // fitted at each request start (after the MTP pack is pinned)
            tqllm_cuda::set_tiled_experts(true); run.tile_experts = true;  // the tier holds the pinned pack's tiled layout
        }
        if (f.tier) std::fprintf(stderr, "tqllm: host expert tier: %d elastic chunks of %.0f MB (attached at each request start while RAM allows)\n",
                                 f.tier->chunks(), tqllm::flashnext::kHostTierChunk / 1048576.0);
        std::fprintf(stderr, "tqllm: host expert tier: %d blocks of %.2f MB mean (%d size classes) = %.1f GB pinned (%.1f GB available, %.1f GB kept for a full prompt's host buffers, %.1f GB ring, %.0f GB reserve) in %.1f s\n",
                     blocks, mean / 1e6, f.tier ? f.tier->classes() : 0, bytes / 1e9, avail / 1e9, prompt_host / 1e9, ring / 1e9,
                     tqllm::flashnext::kHostTierReserve / 1e9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    run.small_card = f.slots_auto && (!f.experts.empty() || pack_in_memory) && tqllm::device_free_bytes() < (size_t(16) << 30);
    run.parallel = slots_.size() > 1;
    // One sequence: adaptive draft length (stop below a 0.9 draft probability, next-layer prediction for 6-token blocks):
    // 3090 chat +1.5 %, 32K +3 %; GB10 32K +4.4 %; text identical to drafts=3 (verify is batch-invariant). Not while a
    // CPU expert pool serves the misses: it takes blocks of at most its row budget (decode_rules.hpp sequence_drafts).
    // The pool only runs with --exact off, so the 16 GB cards that must run exact (the small-card path, 107 of 512 experts
    // resident) keep the adaptive block instead of the pool's 3 drafts.
    // --np > 1 verifies every slot's full block, so it keeps the manifest's drafts
    const int manifest_drafts = f.drafts;
    // the decode pool's row budget (0: no pool): the banks' verify floor and the memory plan's draft count follow it, so a
    // plan can no longer assume a 4-row block for a card whose pool does not run (11-12)
    auto cpu_pool_rows_now = [&] {
        return tqllm::flashnext::cpu_pool_rows(tqllm::flashnext::cpu_miss_threads(tqllm::flashnext::cpu_miss_env(),
                                                                                  run.exact, int(std::thread::hardware_concurrency()),
                                                                                  run.small_card && run.parallel, tqllm::flashnext::kCpuMissPoolBuilt,
                                                                                  f.tier != nullptr, slots_.size() == 1 && !run.small_card), run.small_card);
    };
    if (slots_.size() == 1) {
        const DraftPlan plan = sequence_drafts(f.drafts, run.exact,
                                               tqllm::flashnext::cpu_miss_env(),
                                               int(std::thread::hardware_concurrency()), !f.slots_auto && f.slots >= c.experts,
                                               run.small_card, f.tier != nullptr);
        f.drafts = plan.drafts;
        if (plan.adaptive) {
            run.draft_tau = 0.9f;
            run.predict_n = 6;
        }
    }
    if (run.small_card) {
        // the dense fp16 cache holds one layer (measured high-water 119 MB on Flash-Next IQ3_S), not 600 MB
        if (!run.tune.dense_cache_set) run.tune.dense_cache_mb = std::min(run.tune.dense_cache_mb, 160L);
        // 256 experts stage as two even slices of 128 (192 + 64 at the default): 325 MB less staging, the same slice count;
        // a prompt the engine chunk covers takes 64-expert slices per request (decode_rules.hpp small_card_stage)
        if (!run.tune.stage_set) run.tune.stage = std::min(run.tune.stage, 128);
    }
    checkpoint.cold_overlap=f.cold_overlap;
    if (!f.hot_map_path.empty()) {
        f.hot_map.resize(size_t(c.layers) * c.experts);
        std::ifstream map(f.hot_map_path, std::ios::binary);
        require(bool(map.read(reinterpret_cast<char *>(f.hot_map.data()), std::streamsize(f.hot_map.size() * 4))) && map.peek() == EOF,
                "hot/cold: map size differs");
        const int hot_count = validate_hot_map(f.hot_map, c.layers, c.experts);
        f.slots = hot_cache_slots(hot_count, f.hot_slots); f.slots_auto = false;
        if (!f.cold_store_path.empty()) {
            f.cold_store = std::make_unique<ColdExpertStore>(f.cold_store_path, size_t(c.hidden)*c.expert_ff/64*18,
                ColdExpertStore::identity(f.ggufs, f.hot_map, size_t(c.hidden)*c.expert_ff/64*18));
            f.cold_store->profile(tqllm::debug_on("mem"));
            f.cold_store->fast_verification(f.cold_fast);
            for (int l=0;l<c.layers;++l) for (int e=0;e<c.experts;++e)
                if (!expert_is_hot(f.hot_map,l,c.experts,e))
                    require(f.cold_store->contains(l,e) && (!f.cold_fast || f.cold_store->contains_fast(l,e)), "cold store: missing expert or fast checksum");
            std::fprintf(stderr,"tqllm: cold store enabled; no runtime Q2 conversion\n");
        }
        f.cold_cache = std::make_unique<MixedExpertCache>(c.layers, c.experts, c.hidden, c.expert_ff, f.cold_slots,
                size_t(2) << 30, [state = &f, cp = &checkpoint](int layer, int expert) {
            auto & f = *state; auto & checkpoint = *cp; const auto & c = cp->config;
            require(!expert_is_hot(f.hot_map, layer, c.experts, expert), "cold cache requested a hot expert");
            const auto key = std::make_pair(layer, expert);
            if(f.cold_fetch_queue)return f.cold_fetch_queue->demand({key})[0].get();
            if(f.cold_mmap)return mapped_cold_payload(f,layer,expert);
            auto payload = f.cold_host.get(key, [&] {
                std::array<std::vector<char>,3> raw;
                if (f.cold_store) raw = f.cold_store->load(layer, expert);
                else {
                const auto src = checkpoint.expert_payload(layer, expert);
                for (int m = 0; m < 3; ++m) {
                    const size_t count = size_t(c.hidden) * c.expert_ff; uint64_t block=0, bytes=0;
                    require(tqllm::gguf_block_layout(src[m].type,block,bytes) && count % block == 0 && count / block * bytes == src[m].bytes.size,
                            "cold Q2: source layout differs");
                    raw[m] = tqllm::requantize_gsq_q2_parallel(src[m].type,src[m].bytes.data,count,size_t(block),size_t(bytes),
                            std::max(1u,std::min(8u,std::thread::hardware_concurrency())));
                }
                }
                return raw;
            });
            std::array<ExpertPayload,3> result;
            for(int m = 0; m < 3; ++m) result[m] = {tqllm::GGML_GSQ_Q2_0, {(*payload)[m].data(), (*payload)[m].size()}, payload, true};
            return result;
        }, nullptr);
        if(f.cold_readers) {
            f.cold_read_pool=std::make_unique<ExpertReadPool>(unsigned(f.cold_readers));
            if(f.cold_prefetch_k) {
                f.cold_fetch_queue=std::make_unique<ColdFetchQueue>(*f.cold_read_pool,size_t(2*f.cold_prefetch_k),
                    [state=&f](ColdFetchQueue::Key key){return mapped_cold_payload(*state,key.first,key.second);});
                checkpoint.cold_prefetch=[state=&f](int layer,const std::vector<int> & ids) {
                    auto & f=*state;const auto & c=f.cp->config;
                    auto ranked=rank_expert_predictions(ids,c.topk,f.cold_prefetch_k,[&](int id){
                        return id>=0 && id<c.experts && !expert_is_hot(f.hot_map,layer,c.experts,id) && !f.cold_cache->resident(layer,id);
                    });
                    f.cold_fetch_queue->prefetch(layer,ranked);
                };
            }
            f.cold_cache->set_batch_source([state=&f](int layer,const std::vector<int> & ids) {
                auto & f=*state;std::vector<ColdExpertMemo::Key> keys;
                for(int id:ids) {
                    require(!expert_is_hot(f.hot_map,layer,f.cp->config.experts,id),"cold reader requested a hot expert");
                    keys.emplace_back(layer,id);
                }
                if(f.cold_fetch_queue) {
                    auto jobs=f.cold_fetch_queue->demand(keys);std::vector<std::array<ExpertPayload,3>> out;
                    for(auto & job:jobs)out.push_back(job.get());return out;
                }
                if(f.cold_mmap) {
                    std::vector<std::future<std::array<ExpertPayload,3>>> jobs;
                    for(int id:ids)jobs.push_back(f.cold_read_pool->submit([state,layer,id]{return mapped_cold_payload(*state,layer,id);}));
                    std::vector<std::array<ExpertPayload,3>> out;out.reserve(jobs.size());
                    for(auto & job:jobs)out.push_back(job.get());
                    return out;
                }
                auto raw=f.cold_host.get_batch(keys,*f.cold_read_pool,[&f](ColdExpertMemo::Key key) {return f.cold_store->load(key.first,key.second);});
                std::vector<std::array<ExpertPayload,3>> result(ids.size());
                for(size_t i=0;i<ids.size();++i)for(int p=0;p<3;++p)
                    result[i][p]={tqllm::GGML_GSQ_Q2_0,{(*raw[i])[p].data(),(*raw[i])[p].size()},raw[i],true};
                return result;
            });
        }
        if (f.cold_streaming) f.cold_cache->set_async_batch_source([state=&f](int layer,const std::vector<int> & ids) {
            auto & f=*state;std::vector<ColdFetchQueue::Key> keys;
            for(int id:ids) {
                require(!expert_is_hot(f.hot_map,layer,f.cp->config.experts,id),"cold streaming requested a hot expert");
                keys.emplace_back(layer,id);
            }
            if(f.cold_fetch_queue)return f.cold_fetch_queue->demand(keys);
            std::vector<std::shared_future<std::array<ExpertPayload,3>>> jobs;
            for(int id:ids)jobs.push_back(f.cold_read_pool->submit([state,layer,id]{return mapped_cold_payload(*state,layer,id);}).share());
            return jobs;
        });
        f.cold_cache->set_staged_misses(true);
        rt.hot_map = &f.hot_map; rt.cold_cache = f.cold_cache.get();
        std::fprintf(stderr, "tqllm: hot/cold pilot: %d current-format hot, %d Q2 cold experts per layer; %d cold GPU slots, no router mask\n",
                hot_count, c.experts - hot_count, f.cold_slots);
        std::fprintf(stderr, "tqllm: hot cache: %d GPU slots/layer for %d original-format hot experts\n", f.slots, hot_count);
    }
    size_t device_total = 0; (void) tqllm::device_free_bytes(&device_total);
    f.cache = std::make_unique<MixedExpertCache>(c.layers, c.experts, c.hidden, c.expert_ff, f.slots, bank_budget_bytes(f.slots_auto, f.budget_gb, device_total),
                [&checkpoint](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
    ::apply_layer_slots(*f.cache, f.experts, f.slots, f.layer_slots, slots_.size() > 1 ? int(slots_.size()) * c.topk : 0);
    if (f.tier) {
        f.cache->set_host_tier(f.tier.get());
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<char> owned(size_t(c.layers), 0);
        for (int l = 0; l < c.layers; ++l) owned[size_t(l)] = !run.split || run.lr.owns(uint32_t(l));
        uint64_t total = 0; for (char o : owned) total += o ? uint64_t(c.experts) : 0;
        const int read = tqllm::flashnext::warm_host_tier(uint64_t(f.tier->blocks()), total) ? tqllm::warm_tier(*f.tier, owned, c.experts) : 0;
        std::fprintf(stderr, "tqllm: host expert tier: %d experts read at start in %.1f s\n", read,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    // packed experts in pinned memory, tiled in place (same steps as --decode). Without an experts file (a GB10 with
    // every expert resident) the cache loads experts from the GGUF once and keeps them.
    if (!f.experts.empty() || pack_in_memory) {
    static std::vector<int32_t> keep_map; int keep_k = 0;  // TQLLM_DEBUG=keep_map (pack-file path only)
    if (pack_in_memory) {  // no experts file: the same bytes, built from the GGUF (decode_rules.hpp build_pack_in_memory)
        const auto t0 = std::chrono::steady_clock::now();
        f.packed = build_packed(checkpoint, std::max(4, std::min(32, int(std::thread::hardware_concurrency()))),
                                run.split ? int(run.lr.lo) : 0, run.split ? int(run.lr.hi) : c.layers);
        checkpoint.release_expert_pages();  // the expert pages would otherwise sit beside the pinned pack
        std::fprintf(stderr, "tqllm: packed experts built from the GGUF in memory: %.1f GB in %.0f s\n", f.packed.bytes / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    } else {
    f.packed = open_packed(f.experts);
    // TQLLM_DEBUG=keep_map:FILE (expert pruning): int32 [layers][experts], each expert's index in a compact pack or -1 (pruned); the router
    // masks pruned experts out (rt.keep_mask) and the pack holds only the kept ones (K per layer)
    if (const char * km = tqllm::debug_cstr("keep_map")) {
        keep_map.assign(size_t(c.layers) * c.experts, -1);
        FILE * mf = std::fopen(km, "rb"); require(mf && std::fread(keep_map.data(), 4, keep_map.size(), mf) == keep_map.size(), std::string("TQLLM_DEBUG=keep_map: cannot read ") + km);
        std::fclose(mf);
        std::vector<float> mask(keep_map.size());
        for (size_t i = 0; i < keep_map.size(); ++i) { mask[i] = keep_map[i] < 0 ? -INFINITY : 0.f; keep_k = std::max(keep_k, keep_map[i] + 1); }
        float * dm = nullptr; ck(cudaMalloc(reinterpret_cast<void **>(&dm), mask.size() * 4));
        ck(cudaMemcpy(dm, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice)); rt.keep_mask = dm;
        size_t kept = 0; for (int32_t v : keep_map) kept += v >= 0;
        std::fprintf(stderr, "tqllm: expert pruning: %zu of %d experts kept (%d per layer in the pack, %.1f %%)\n", kept, c.layers * c.experts, keep_k, 100.0 * kept / (c.layers * c.experts));
    }
    require(f.packed.layers == c.layers && f.packed.experts == (keep_map.empty() ? c.experts : keep_k), "packed experts do not match the model");
    // --split layer: only this rank's layers are copied into pinned memory (map + base[l] stays valid for them)
    if (run.split) { f.packed.lo = int(run.lr.lo); f.packed.hi = int(run.lr.hi); }
    const auto span = packed_span(f.packed.base, f.packed.block, f.packed.experts, std::max(0, f.packed.lo), std::min(f.packed.layers, f.packed.hi));
    const size_t file_bytes = f.packed.bytes, held = size_t(span.second - span.first);
    char * host = nullptr; ck(cudaMallocHost(reinterpret_cast<void **>(&host), held));
    { const int fd = ::open(f.experts.c_str(), O_RDONLY); require(fd >= 0, "cannot reopen " + f.experts);
      for (size_t done = 0; done < held;) {
          const ssize_t r = ::pread(fd, host + done, std::min<size_t>(size_t(1) << 28, held - done), off_t(span.first + done));
          require(r > 0, "short read of " + f.experts); done += size_t(r);
      }
      ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
    ::munmap(const_cast<char *>(f.packed.map), file_bytes); f.packed.map = host - span.first; f.packed.bytes = held;
    }
    if (!f.packed.tiled) {
        for (int l = 0; l < f.packed.layers; ++l) {
            if (!f.packed.has(l)) continue;
            size_t off = 0; const auto bpbs = expert_tile_bpbs(checkpoint, l);  // the grouped kernels' id, not u4h_id (UD types)
            for (int pp = 0; pp < 3; ++pp) {
                const int cols = pp == 2 ? c.expert_ff : c.hidden, rows = pp == 2 ? c.hidden : c.expert_ff;
                const int bpb = bpbs[size_t(pp)]; const size_t dsz = f.packed.sizes[l][2 * pp];
                if (bpb) {
                    require(rows % 16 == 0 && (cols / 32) % 4 == 0 && dsz == size_t(rows) * (cols / 32) * bpb, "tileable expert part with an unexpected layout");
                    std::vector<std::thread> th;
                    for (int w = 0; w < 16; ++w) th.emplace_back([&, w] {
                        std::vector<char> tmp(dsz);
                        for (int e = w; e < f.packed.experts; e += 16) {
                            char * d = const_cast<char *>(f.packed.map) + f.packed.base[l] + uint64_t(e) * f.packed.block[l] + off;
                            tqllm_cuda::tile_expert_rows(d, tmp.data(), rows, cols, bpb); std::memcpy(d, tmp.data(), dsz);
                        }
                    });
                    for (auto & t : th) t.join();
                }
                off += dsz + f.packed.sizes[l][2 * pp + 1];
            }
        }
        f.packed.tiled = true;
    }
    if (f.packed.tiled) { tqllm_cuda::set_tiled_experts(true); run.tile_experts = true; }
    // the CPU pool reads the pinned copy whenever it runs (decode_rules.hpp); the GGUF expert pages would hold the experts twice
    if (tqllm::flashnext::cpu_pool_reads_packed(tqllm::flashnext::cpu_miss_env(), run.exact,
                                                int(std::thread::hardware_concurrency()), run.small_card && run.parallel, f.tier != nullptr, slots_.size() == 1 && !run.small_card)) {
        static CpuPackedView pv; pv = CpuPackedView{}; pv.map = f.packed.map; pv.base = f.packed.base; pv.block = f.packed.block;
        pv.sizes = f.packed.sizes; pv.tiled = f.packed.tiled; pv.layers = f.packed.layers;
        for (int l = 0; l < f.packed.layers; ++l) pv.fmt.push_back(expert_u4h_ids(checkpoint, l));
        g_cpu_packed = &pv; std::printf("tqllm: CPU miss pool reads the pinned packed experts\n");
    }
    PackedExperts * pk = &f.packed;
    if (!keep_map.empty()) {
        const int32_t * km = keep_map.data(); const int E = c.experts;
        f.cache->set_prepacked([pk, km, E](int layer, int id) -> const char * {
            if (!pk->has(layer)) return nullptr; const int32_t q = km[size_t(layer) * E + id]; return q < 0 ? nullptr : pk->at(layer, q); }, true);
        g_cpu_packed = nullptr;  // the CPU pool's packed view indexes by global id: off with a compact pack
    } else
    f.cache->set_prepacked([pk](int layer, int id) -> const char * { return pk->has(layer) ? pk->at(layer, id) : nullptr; }, true);
    }
    // RAM still available now (after the pinned pack, cgroup-aware): whether the per-layer embedding can stay cached
    const auto ple_available = [] { std::ifstream mi("/proc/meminfo"); const std::string t((std::istreambuf_iterator<char>(mi)), std::istreambuf_iterator<char>());
                                    return double(tqllm::host_available_bytes(t)); };
    {   // the per-layer token embedding is read at every token: cold pages cost each new token a disk wait (decode_rules.hpp)
        const auto t0 = std::chrono::steady_clock::now();
        if (const size_t n = checkpoint.warm_tensors(8, run.owns_ple(c) ? host_ram : 0., ple_available()))
            std::fprintf(stderr, "tqllm: per-layer token embedding cached: %.1f GB in %.0f s\n", n / 1e9,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    // MTP experts prepacked in pinned memory
    // all experts resident (GB10): the MTP block's experts too, so its prefill catch-up can run whole chunks
    const int mtp_slots = std::getenv("TQFN_MTP_SLOTS") ? std::atoi(std::getenv("TQFN_MTP_SLOTS")) : run.small_card ? 64 : (f.slots >= c.experts ? c.experts : 256);
    f.mtp_cache = std::make_unique<MixedExpertCache>(c.layers + 1, c.experts, c.hidden, c.expert_ff, mtp_slots, size_t(mtp_slots) * (6u << 20) + (64u << 20),
                    [&checkpoint](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
    const auto first = repack_expert(checkpoint, c.layers, 0, run.tile_experts); for (const auto & p : first) f.mtp_block += p.size();
    ck(cudaMallocHost(reinterpret_cast<void **>(&f.mtp_packed), f.mtp_block * size_t(c.experts)));
    { std::vector<std::thread> pool; std::atomic<int> next{0};
      for (int t = 0; t < 16; ++t) pool.emplace_back([&] {
          for (int e; (e = next++) < c.experts;) {
              const auto parts = repack_expert(checkpoint, c.layers, e, run.tile_experts); size_t off = 0;
              for (const auto & p : parts) { std::memcpy(f.mtp_packed + size_t(e) * f.mtp_block + off, p.data(), p.size()); off += p.size(); }
          }
      });
      for (auto & t : pool) t.join(); }
    char * mp = f.mtp_packed; const size_t mb = f.mtp_block; const int L = c.layers;
    f.mtp_cache->set_prepacked([mp, mb, L](int layer, int id) -> const char * { return layer == L ? mp + size_t(id) * mb : nullptr; }, true);
    tqllm::flashnext::ablation().load(c.hidden, c.layers);  // TQFN_ABLATE (fn_ablate.hpp): before any capture
    std::fprintf(stderr, "tqllm: Flash-Next ready: %d layers, %d experts, %d slots/layer, context %u, %d drafts\n",
                 c.layers, c.experts, f.slots, context, f.drafts);
    // warm-up: one short prompt uploads every dense weight and creates every expert bank now, so the first request is
    // not slowed by it and free-memory readings (the prefill borrow) are true from the start
    tqllm_cuda::qsa_reserve_split_scratch();  // decode attention scratch before the banks take the free memory
    // MMQ kernels load before the banks take the free memory
    // DFlash2 drafter (manifest dflash=): one sequence at a time drafts with it, several with MTP (Strata's Auto rule);
    // loaded before the banks are sized so its weights, tables and K/V ring are counted
    if (!f.dflash.empty() && slots_.size() == 1) {
        PersistentAlloc keep;
        f.drafter = std::make_unique<tqllm::Dflash2Drafter>(f.dflash, c.hidden, checkpoint.vocab_rows(), float(checkpoint.epsilon), nullptr);
        // full 7-draft chain: tau 0.9 stopped it after 1.3-1.7 drafts (2x4090 split 11-14: python 183 -> 202, btree 162 ->
        // 180 tok/s without); prose, where the chain loses, drafts with MTP (pick_drafter)
        f.dfl_feat_rows = size_t(std::clamp(f.chunk, 64, 8192));  // prefill chunks are capped at this (210 MB at 8192)
        f.dfl_feat = std::make_unique<Device<__nv_bfloat16>>(f.dfl_feat_rows * size_t(f.drafter->features()));
        std::fprintf(stderr, "tqllm: DFlash2 drafter %s: block %d, target layers %d..%d, %.0f MB\n", f.dflash.c_str(), f.drafter->block(),
                     f.drafter->target_layers().front(), f.drafter->target_layers().back(), f.drafter->device_bytes() / 1048576.0);
    }
    std::fprintf(stderr, "tqllm: MMQ kernels loaded: %.1f MB\n", tqllm_cuda::mmq_warm() / 1048576.0);
    // the warm model's attention state is sized by the context capacity (8 GB at 262144): it must not hold that while
    // the banks are built, or the slots are capped by it; the warm-up only needs its 64 tokens
    const int kv_capacity_full = rt.kv_capacity; rt.kv_capacity = std::min(rt.kv_capacity, 4096);
    // the device memory a full context keeps beside the banks (memory_plan.hpp context_keep: attention state less the
    // K/V that lives in host memory in decode, and the decode reserve)
    auto mem_shape = [&] {
        tqllm::flashnext::MemShape m;
        for (int l = 0; l < c.layers; ++l) if (checkpoint.has_tensor("blk." + std::to_string(l) + ".indexer.k_proj.weight")) ++m.attention_layers;
        m.kv_heads = checkpoint.attention_kvheads; m.head_dim = checkpoint.attention_dim; m.index_dim = c.index_dim;
        m.experts = c.experts; m.topk = c.topk; return m;
    };
    auto mem_run = [&] {
        tqllm::flashnext::MemRun r; r.context = int(context); r.kv_host = kv_host_enabled();
        r.kv_host_keep = std::getenv("TQFN_KV_HOST_KEEP") ? std::atoi(std::getenv("TQFN_KV_HOST_KEEP")) : 32768;
        r.sequences = slots_.size(); r.small_card = run.small_card; r.caching = caching(); r.manifest_drafts = f.drafts;
        r.pool_rows = cpu_pool_rows_now(); return r;
    };
    auto context_keep = [&] {
        const auto k = tqllm::flashnext::context_keep(mem_shape(), mem_run());
        return std::pair<size_t, size_t>{k.state, k.reserve};
    };
    // the warm-up's steps (64 and 512 tokens) need a fraction of the start-up arena and of the staging halves: hold only
    // that while the banks are built, so the slot count is not capped by buffers the warm-up does not use
    if (const size_t warm_arena = std::max(f.arena_decode, tqllm::prefill_arena_bytes(512)); f.arena.size > warm_arena) {
        ck(cudaFree(f.arena.base)); ck(cudaMalloc(&f.arena.base, warm_arena)); f.arena.size = warm_arena; f.arena.used = 0; }
    const int stage_full = run.tune.stage; run.tune.stage = std::min(run.tune.stage, 32);
    { const auto w0 = std::chrono::steady_clock::now();
      DecodeModel warm(checkpoint, *f.fixtures, *f.cache, 640 + 3 * 64);  // 64 + 512 tokens, then up to three slots=auto rebuilds of 64
      if (warm.miss_stream) f.cache->set_miss_stream(warm.miss_stream);
      std::vector<int> ids(rt.hot_map ? 1 : 64); for (int i = 0; i < int(ids.size()); ++i) ids[size_t(i)] = 1000 + (i * 37) % 50000;
      auto rows = warm.step(ids); ck(cudaDeviceSynchronize());
      // then one 512-token layer-major chunk: the prefill kernels' modules load and the pinned buffers and resident
      // router copies are created now (nsys: 0.3 s of module loads and pinned allocations inside the first prompt)
      // (needs the prepacked expert file: a GB10 build keeps every expert resident and has none)
      // (--split layer: this rank's first layer; layer 0 has no bank on rank 1, and both ranks must run the same steps)
      const int l0 = run.split ? int(run.lr.lo) : 0;
      int packed = f.cache->has_layout(l0) && f.cache->bank_layout(l0).streamable() ? 1 : 0;
      if (run.split) DecodeModel::split_min(*run.split, &packed);
      if (tqllm::flashnext::warm_prefill_chunk({packed})) {
        std::vector<int> lm(512); for (int i = 0; i < 512; ++i) lm[size_t(i)] = 2000 + (i * 53) % 50000;
        (void) warm.step(lm); ck(cudaDeviceSynchronize()); warm.release_prefill(); }
      run.tune.stage = stage_full;  // requests stage their full groups (sized again under the request's borrow)
      (void) checkpoint.head_argmax_rows(rows->p, 1); rows.reset();  // the full LM head loads here, not at the first prompt's first token (1.9 s)
      warm.mtp_experts = f.mtp_cache.get();  // the MTP layer too: its bank and weights, so the first request does not build them
      (void) warm.mtp_batch(std::vector<int>(ids.begin(), ids.begin() + std::min<size_t>(8, ids.size())), warm.res_keep->p); ck(cudaDeviceSynchronize());
      // the indexer's query projection runs only past the QSA budget (2048 positions), which no warm-up step reaches: upload
      // it now, not inside the first long prompt (GB10: 13 x 2.6 MB from the mapped file, 0.3 s of a 3.4K prompt)
      for (int l = 0; l <= c.layers; ++l)
      {
          if (const std::string name = "blk." + std::to_string(l) + ".indexer.q_proj.weight"; checkpoint.has_tensor(name) && checkpoint.is_bf16(name))
              (void) checkpoint.bf16_device(name);
          // its norm too: first used inside a decode step's graph capture, its upload failed the capture
          if (const std::string name = "blk." + std::to_string(l) + ".indexer.q_norm.weight"; checkpoint.has_tensor(name))
              (void) checkpoint.dense_device(name);
      }
      // the first long prompt would otherwise grow the staging buffers mid-request (a whole MTP bank in the catch-up, a
      // whole trunk bank in the refill): size them now, the steady state every later request already has
      f.mtp_cache->reserve_staging(mtp_slots); f.cache->reserve_staging();
      // several sequences: what one sequence really holds at the full slot context, measured with a one-token probe (trunk
      // and MTP). The estimate below counts the attention state only; at 16K per slot a request start took ~990 MB against
      // 610 kept, so np 4 ran out of memory in the third request (11-01, plans/multi-user-c8-20261031.md)
      size_t measured_state = 0;
      if (slots_.size() > 1) {
          rt.kv_capacity = kv_capacity_full;
          ck(cudaDeviceSynchronize()); const size_t before = tqllm::device_free_bytes();
          if (tqllm::debug_on("seqmem")) std::fprintf(stderr, "SEQMEM probe begin (capacity %d)\n", kv_capacity_full);
          {
              DecodeModel probe(checkpoint, *f.fixtures, *f.cache, 64);
              if (probe.miss_stream) f.cache->set_miss_stream(probe.miss_stream);
              probe.mtp_experts = f.mtp_cache.get();
              auto pr = probe.step(std::vector<int>{1000}); ck(cudaDeviceSynchronize());
              (void) probe.mtp_batch(std::vector<int>{1000}, probe.res_keep->p); ck(cudaDeviceSynchronize());
              const size_t after = tqllm::device_free_bytes();
              measured_state = before > after ? before - after : 0;
              if (tqllm::debug_on("seqmem")) std::fprintf(stderr, "SEQMEM probe end: %zu MiB\n", measured_state >> 20);
          }
          if (warm.miss_stream) f.cache->set_miss_stream(warm.miss_stream);
          rt.kv_capacity = std::min(kv_capacity_full, 4096);
          // the probe's K/V is lazily sized (26 of 284 MB at 8K, 11-09), not the full-context formula: count all of it
          f.seq_fixed = measured_state;
          std::fprintf(stderr, "tqllm: one sequence holds %.0f MB of device memory beside its request-sized K/V (measured; slot context %d; counted at each new request)\n",
                       measured_state / 1048576.0, kv_capacity_full);
      }
      if (f.slots_auto) {  // slots=auto: every other buffer is in place; the banks get what the context does not keep
          // --split layer: only this rank's layers hold banks, so a slot costs half as much and each layer gets about twice the slots
          size_t per = 0; for (int l = 0; l < c.layers; ++l) if (f.cache->has_layout(l) && (!run.split || run.lr.owns(uint32_t(l)))) per += f.cache->bank_layout(l).block;
          const auto [state_est, reserve] = context_keep();
          // several sequences: each one's own buffers, measured on a short sequence (the probe); a request's K/V is sized by
          // the request and a long prompt borrows banks like a single sequence, so the full-context estimate is not kept per slot
          // (the prompt cache keeps every slot at its full context: then the full estimate)
          const size_t state = tqllm::flashnext::keep_state(mem_shape(), mem_run(), state_est, measured_state);
          // every slot (--np) keeps its own full-context attention / recurrent / MTP state beside the banks: sized for one,
          // np 4 at 32K ran out of memory in the second request's state (F38)
          const size_t nseq = slots_.size();
          const size_t need = run.small_card ? size_t(512) << 20 : auto_keep_bytes(nseq, state, reserve);
          if (run.small_card) std::fprintf(stderr, "tqllm: small card: %s, 64 MTP slots, banks up to a 512 MB keep (long prompts borrow banks)\n",
                                           run.parallel && tqllm::flashnext::cpu_miss_env() < 0 ? "CPU expert pool off (--np > 1; one request keeps it)" : "CPU expert pool");
          size_t extra = 0;  // what rebuilding the banks takes beyond the slots themselves (measured, not estimated)
          for (int round = 0; round < 3; ++round) {
              for (int l = 0; l < c.layers; ++l) f.cache->release_bank(l);
              { cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
              const size_t fr = tqllm::device_free_bytes();
              f.slots = tqllm::auto_expert_slots(fr, need + extra, per, c.experts);
              f.cache->set_layer_slots(std::vector<int>(size_t(c.layers), f.slots)); ::apply_layer_slots(*f.cache, f.experts, f.slots, f.layer_slots, slots_.size() > 1 ? int(slots_.size()) * c.topk : 0);
              { auto again = warm.step(ids); ck(cudaDeviceSynchronize()); }  // the banks again, at the new size
              const size_t after = tqllm::device_free_bytes();
              std::fprintf(stderr, "tqllm: expert slots auto: %.0f MB free without banks, %.0f MB kept, %.1f MB per slot -> %d slots/layer, %.0f MB free after\n",
                           fr / 1048576.0, (need + extra) / 1048576.0, per / 1048576.0, f.slots, after / 1048576.0);
              int done = after >= need || f.slots <= 1;
              if (run.split) DecodeModel::split_min(*run.split, &done);  // every rank runs the same rounds (each round is a forward)
              if (done) break;
              if (need > after) extra += need - after;
          }
          if (slots_.size() == 1) f.drafts = auto_slot_drafts(manifest_drafts, f.drafts, f.slots, c.experts, cpu_pool_rows_now());
          if (f.drafter) f.drafter->set_tau(dflash_tau(f.slots >= c.experts));
          // --split layer: each rank keeps the slots its card holds (placement only); the draft count sets every forward's
          // rows, so all ranks take the smallest
          if (run.split) {
              const int mine = f.drafts; DecodeModel::split_min(*run.split, &f.drafts);
              std::fprintf(stderr, "tqllm: --split layer: %d expert slots per layer here, %d drafts on every rank (this rank: %d)\n", f.slots, f.drafts, mine);
          }
          // tier A (the floor a bank never drops below): the slots a full-context state leaves, as with one sequence's whole
          // state kept free; tier B (the target) is what the smaller keep allows (decode_rules.hpp auto_keep_bytes)
          if (slot_tiers(nseq, false)) {
              const size_t fr = tqllm::device_free_bytes() + f.cache->stats().device_bytes;
              const int tier_a = std::min(f.slots, tqllm::auto_expert_slots(fr, state + reserve + (size_t(256) << 20) + extra, per, c.experts));
              std::vector<int> floor(size_t(c.layers));
              for (int l = 0; l < c.layers; ++l) floor[size_t(l)] = slot_floor(f.cache->slots(l), 0, f.slots, verify_slot_min(f.drafts, c.topk));  // the F39-safe minimum
              f.cache->set_layer_tiers(floor, run.small_card ? size_t(512) << 20 : size_t(256) << 20);  // small cards: the refill's keep
              std::fprintf(stderr, "tqllm: expert slots: %d per layer (%d with a full context's state kept; a long prompt's layers share what is free)\n", f.slots, tier_a);
          }
      }
      // TQLLM_DEBUG=expert_profile:FILE (int32 counts [layers][experts], tools/fn_expert_profile.py): the banks start with each
      // layer's most-routed experts instead of what the synthetic warm-up prompt happened to route (the first request
      // otherwise runs on a cold cache)
      if (const char * prof = tqllm::debug_cstr("expert_profile")) {
          std::ifstream in(prof, std::ios::binary); std::vector<int32_t> cnt(size_t(c.layers) * c.experts);
          in.read(reinterpret_cast<char *>(cnt.data()), std::streamsize(cnt.size() * 4));
          if (!in) throw std::runtime_error(std::string("TQLLM_DEBUG=expert_profile: cannot read ") + prof + " as int32 [layers][experts]");
          int seeded = 0;
          for (int l = 0; l < c.layers; ++l) {
              if (!f.cache->has_layout(l)) continue;
              std::vector<int> ids(size_t(c.experts)); for (int e = 0; e < c.experts; ++e) ids[size_t(e)] = e;
              const int32_t * row = cnt.data() + size_t(l) * c.experts;
              std::stable_sort(ids.begin(), ids.end(), [&](int x, int y) { return row[x] > row[y]; });
              ids.resize(size_t(std::min(f.cache->slots(l), c.experts)));
              (void) f.cache->prepare(l, ids); f.cache->wait_misses(); seeded += int(ids.size());
          }
          ck(cudaDeviceSynchronize());
          std::fprintf(stderr, "tqllm: expert banks seeded from %s: %d experts\n", prof, seeded);
      }
      {   // banks that hold every expert of their layer are filled now, not one prompt's misses at a time (decode_rules.hpp)
          using pclk = std::chrono::steady_clock; const auto t0 = pclk::now(); int layers_filled = 0, loaded = 0;
          for (int l = 0; l < c.layers; ++l) {
              if (!f.cache->has_layout(l) || !preload_all_experts(f.cache->slots(l), c.experts)) continue;
              std::vector<int> ids(size_t(c.experts)); for (int e = 0; e < c.experts; ++e) ids[size_t(e)] = e;
              const auto before = f.cache->stats().misses;
              (void) f.cache->prepare(l, ids); f.cache->wait_misses();
              loaded += int(f.cache->stats().misses - before); ++layers_filled;
          }
          if (f.mtp_cache && preload_all_experts(f.mtp_cache->slots(), c.experts)) {  // the MTP block too: its draft calls then build their maps on the device
              std::vector<int> ids(size_t(c.experts)); for (int e = 0; e < c.experts; ++e) ids[size_t(e)] = e;
              (void) f.mtp_cache->prepare(c.layers, ids); f.mtp_cache->wait_misses(); ++layers_filled;
          }
          ck(cudaDeviceSynchronize());
          if (layers_filled)
              std::fprintf(stderr, "tqllm: expert banks filled at start-up: %d layers hold all %d experts (%d loaded) in %.1f s\n",
                           layers_filled, c.experts, loaded, std::chrono::duration<double>(pclk::now() - t0).count());
          // every bank full (a GB10 keeps every expert resident, no pack): tile them in place as a pack is tiled, so the expert
          // kernels read lane-contiguous tiles (same bits; IQ gate/up -30 % per call at 40-110 groups, expert-dedup-bench 11-11).
          const bool mtp_full = f.mtp_cache && preload_all_experts(f.mtp_cache->slots(), c.experts);
          if (!f.packed.tiled && f.experts.empty() && layers_filled == c.layers + (mtp_full ? 1 : 0) && (!f.mtp_cache || mtp_full)) {
              const auto t1 = pclk::now();
              std::vector<int> all(size_t(c.experts)); for (int e = 0; e < c.experts; ++e) all[size_t(e)] = e;
              auto banks = [&](auto && fn) {
                  for (int l = 0; l <= c.layers; ++l) {
                      auto * cache = l < c.layers ? f.cache.get() : f.mtp_cache.get();
                      if (!cache) continue;
                      (void) cache->prepare(l, all);
                      fn(cache->gate(), c.expert_ff, c.hidden);
                      fn(cache->up(), c.expert_ff, c.hidden);
                      fn(cache->down(), c.hidden, c.expert_ff);
                  }
              };
              // per bank: formats with a tiled layout are tiled; others (e.g. GSQ_Q2_0) keep theirs: the tiled flag is read only by
              // the grouped u4h kernels of tileable formats (quant_u4h.inc: GROUPED && tiled_bpb(FMT) && c_tiled_experts)
              int tiled = 0, kept = 0;
              banks([&](const tqllm::Mat & m, int rows, int k) {  // m: the bank, its slots stacked (m.rows = slots x rows); data only
                  const int bpb = tqllm::expert_tile_bpb(m.fmt);
                  if (bpb > 0) {  // a tileable format must be tiled: the kernels read every bank of it as tiles once the flag is on
                      require(rows % 16 == 0 && (k / 32) % 4 == 0 && m.cols == k && m.rows % rows == 0, "tileable expert bank with an unexpected layout");
                      tqllm_cuda::tile_expert_rows_device(static_cast<char *>(const_cast<void *>(m.p)), m.rows / rows, rows, k, bpb, nullptr);
                      ++tiled;
                  } else ++kept;
              });
              if (tiled) {
                  ck(cudaDeviceSynchronize());
                  f.packed.tiled = true; tqllm_cuda::set_tiled_experts(true); run.tile_experts = true;
                  std::fprintf(stderr, "tqllm: resident expert banks tiled in place: %d tiled, %d in a format without tiles, %.1f s\n",
                               tiled, kept, std::chrono::duration<double>(pclk::now() - t1).count());
              }

          }
      }
      {   // the group count from which the paired gate / up launch prefetches, per expert format, timed on the loaded banks
          // (full banks on a GB10, the packed tiled banks of a discrete card; both variants give the same bits; was a fixed 160)
          std::vector<int> seen;
          for (int l = 0; l < c.layers; ++l) {
              if (!f.cache->has_layout(l) || f.cache->slots(l) < 16) continue;
              (void) f.cache->prepare(l, {0});
              const tqllm::Mat g = f.cache->gate(), u = f.cache->up();
              if (!g.fmt || !u.fmt || g.fmt->u4h_id() < 0 || g.fmt->u4h_id() != u.fmt->u4h_id()) continue;
              const int id = g.fmt->u4h_id();
              if (std::find(seen.begin(), seen.end(), id) != seen.end()) continue;
              seen.push_back(id);
              const int pf = tqllm_cuda::calibrate_dedup_pf(id, static_cast<const uint8_t *>(g.p), g.aux, static_cast<const uint8_t *>(u.p), u.aux,
                                                            c.expert_ff, c.hidden, g.rows / c.expert_ff, nullptr);
              if (pf == INT32_MAX) std::fprintf(stderr, "tqllm: paired %s experts: no prefetch (never faster here)\n", g.fmt->name());
              else std::fprintf(stderr, "tqllm: paired %s experts: prefetch from %d groups\n", g.fmt->name(), pf);
          }
      }
      // spec_begin grows pinned residuals to the actual prompt, not the configured slot capacity.
      f.cache->set_miss_stream(nullptr);
      if (tqllm::debug_on("mem")) checkpoint.resident_report();
      {   // a multi-chunk prefill's pinned residual rows, pinned here once instead of inside a request (decode_rules.hpp), if
          // its chunks could ever need them: they are kLayerMajorChunkMin tokens or more, which write the MTP K/V inline
          const bool host_rows = tqllm::flashnext::prefill_residuals_on_host(run.head_here(), tqllm::flashnext::kLayerMajorChunkMin);
          const size_t rows = int(context) > 4096 && host_rows ? size_t(context) * size_t(warm.width) : 0;
          std::ifstream mi("/proc/meminfo"); const std::string text((std::istreambuf_iterator<char>(mi)), std::istreambuf_iterator<char>());
          int dev = 0, unified = 0; cudaGetDevice(&dev);
          if (cudaDeviceGetAttribute(&unified, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) != cudaSuccess) unified = 0;
          if (pin_residual_at_start(rows * sizeof(float), tqllm::host_ram_bytes(), tqllm::host_available_bytes(text), unified != 0)) {
              ck(cudaMallocHost(reinterpret_cast<void **>(&f.res_host), rows * sizeof(float))); f.res_host_cap = rows;
              std::fprintf(stderr, "tqllm: prompt residual buffer pinned at start: %.1f GB for %u tokens\n", double(rows * sizeof(float)) / 1e9, context);
          }
          int attn = 0; for (int l = 0; l < c.layers; ++l) if (checkpoint.has_tensor("blk." + std::to_string(l) + ".indexer.k_proj.weight")) ++attn;
          const size_t kept = tqllm::prefill_context_bytes(int(context), 16, attn, checkpoint.attention_kvheads, checkpoint.attention_dim, c.index_dim);
          const size_t avail = tqllm::host_available_bytes(text);
          if (!run.prompt_cache_set) {
              run.prompt_cache = tqllm::flashnext::prompt_cache_by_fit(false, false, unified != 0, slots_.size(), avail, kept);
              if (unified) std::fprintf(stderr, "tqllm: prompt cache %s by fit (%zu x %.1f GB kept conversations, %.0f GB available; --prompt-cache on|off decides)\n",
                                        run.prompt_cache ? "on" : "off", slots_.size(), kept / 1e9, avail / 1e9);
              else std::fprintf(stderr, "tqllm: prompt cache on (default; --prompt-cache off turns it off)\n");
          }
      }
      if (f.arena.size != f.arena_decode) {  // start the first request as every later one: decode-sized arena, more room to borrow
          ck(cudaFree(f.arena.base)); ck(cudaMalloc(&f.arena.base, f.arena_decode)); f.arena.size = f.arena_decode; f.arena.used = 0; }
      size_t fr = 0, tot = 0; ck(cudaMemGetInfo(&fr, &tot));
      std::fprintf(stderr, "tqllm: Flash-Next warm-up %.1f s: expert banks %.0f MB, %.0f MB free\n",
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(), f.cache->stats().device_bytes / 1048576.0, fr / 1048576.0); }
    {   // building the banks read every expert through the page cache and pushed the per-layer token embedding out (a
        // GB10 then read 16 random rows per token from the disk: ~22 ms per np 4 step with the GPU idle); once every expert
        // is in a bank its file pages are dropped, and the embedding is read in again
        if (f.slots >= c.experts) checkpoint.release_expert_pages();
        const auto t0 = std::chrono::steady_clock::now();
        if (const size_t n = checkpoint.warm_tensors(8, run.owns_ple(c) ? host_ram : 0., ple_available()))
            std::fprintf(stderr, "tqllm: per-layer token embedding cached again after the banks: %.1f GB in %.0f s\n", n / 1e9,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    if(f.cold_reclaim_hot) {
        ck(cudaDeviceSynchronize());
        std::fprintf(stderr,"tqllm: uploaded checkpoint source page reclaim: %s\n",checkpoint.discard_source_pages() ? "ok" : "partial");
    }
    rt.kv_capacity = kv_capacity_full; f.kv_capacity_full = kv_capacity_full; rt.kv_capacity_full = kv_capacity_full; rt.qsa_batch.reset();  // the selection scratch was sized for the warm-up's capacity
    {   // the plan budget: what a prompt may plan with (device free + the expert banks), measured once with the pool trimmed and
        // rounded down to 256 MB, so the chunk / block a prompt length gets does not follow what earlier requests left in the
        // pool or in a partly refilled bank (a 100 MB difference flipped a 16K prompt between 3 x 6144 and 4 x 5120, F26)
        cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0));
        const size_t q = size_t(256) << 20;
        f.plan_budget = (tqllm::device_free_bytes() + f.cache->stats().device_bytes) / q * q;
        // the MTP bank a multi-chunk prompt drops before its trunk: it is a plan term, so take it here instead of from the
        // bank's state at the request (a partly filled bank would move the plan, and the plan moves the tokens)
        f.mtp_bank_bytes = f.mtp_cache ? f.mtp_cache->peak_device_bytes() : 0;
        std::fprintf(stderr, "tqllm: prefill plan room: %.0f MB fixed (pool trimmed, 256 MB steps), MTP bank %.0f MB\n",
                     f.plan_budget / 1048576.0, f.mtp_bank_bytes / 1048576.0);
    }
    {   // a long prompt allocates the attention state for the whole context next to the banks: refuse a slot count that
        // cannot hold it (it would prefill for minutes and then fail in decode)
        const auto [state, reserve] = context_keep();
        f.state_full = state; f.state_tokens = int(context); f.mem_shape = mem_shape();
        const size_t fr = tqllm::device_free_bytes();
        // slots=auto keeps only the reserve (and each extra sequence's own buffers) on purpose (decode_rules.hpp
        // auto_keep_bytes): a request's K/V is sized by the request and a long prompt borrows banks, at any --np
        if (fr < state + reserve && (run.small_card || f.slots_auto)) {
            std::fprintf(stderr, "tqllm: %zu MB free < %zu MB for a full context; long prompts borrow expert banks\n", fr >> 20, (state + reserve) >> 20);
        } else if (const int most = tqllm::flashnext::fit_max_slots({state, reserve}, fr, f.slots_auto, run.small_card, f.slots,
                                                                         f.cache->stats().device_bytes); most >= 0) {
            throw std::runtime_error("Flash-Next: " + std::to_string(f.slots) + " expert slots per layer leave " + std::to_string(fr >> 20) +
                                     " MB free, a full " + std::to_string(context) + "-token context needs " + std::to_string((state + reserve) >> 20) +
                                     " MB (attention state + decode reserve): use at most " + std::to_string(most) +
                                     " slots (manifest slots=) or a smaller --context");
        }
    }
    // the default stream is per thread: the engine thread's stream is not ordered after the loading thread's work
    ck(cudaDeviceSynchronize());
    // --split layer, one sequence: from here the head, MTP and drafter run on the last rank only (the warm-up above ran them
    // everywhere on the broadcast rows)
    f.cp->run.head_only = f.cp->run.split && slots_.size() == 1;
    f.cp->run.last_rank = !f.cp->run.split || f.cp->run.split->rank() == f.cp->run.split->world() - 1;
    if (f.cp->run.ep) {  // --split experts: rank 0 reads the lease; rank 1 lends this box while it has no request of its own
        f.cp->run.experts = std::make_shared<tqllm::ExpertParallel>(f.cp->run.ep, flashnext_expert_shard(*f.cp, *f.cache));
        f.cp->run.experts->start([this] { for (const auto & sl : slots_) if (sl.serving_) return false; return true; });
    }
}


}  // namespace tqllm::flashnext
