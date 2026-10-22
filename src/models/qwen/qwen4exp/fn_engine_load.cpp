// Flash-Next engine load: manifest, checkpoint and expert caches, warm-up, expert slots and the context fit.
#include <fstream>
#include "fn_engine.hpp"
#include "decode_rules.hpp"
#include <chrono>

namespace tqllm::flashnext {

FlashNextEngine::FlashNextEngine(const std::string & manifest, uint32_t context, int nslots) : context_(context), slots_(size_t(std::max(1, nslots))) {
    // above 8 concurrent slots some requests end after one token and np 16 fails (10-13, P8 build too): refuse until fixed
    require(nslots <= 8 || std::getenv("TQFN_ALLOW_NP_ABOVE_8") != nullptr,
            "Flash-Next serves up to --np 8 for now (a bug above 8 slots is being fixed); TQFN_ALLOW_NP_ABOVE_8=1 for diagnostics");
    auto & f = f_; std::ifstream in(manifest); require(bool(in), "cannot open Flash-Next manifest " + manifest);
    const auto dir = std::filesystem::path(manifest).parent_path();
    auto resolve = [&](const std::string & v) { return std::filesystem::path(v).is_absolute() ? v : (dir / v).string(); };
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        const auto eq = line.find('='); require(eq != std::string::npos, "manifest line without '=': " + line);
        const auto k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "gguf") f.ggufs.push_back(resolve(v)); else if (k == "mtp") f.mtp = resolve(v);
        else if (k == "experts") f.experts = resolve(v); else if (k == "draft_vocab") f.draft_vocab = resolve(v);
        else if (k == "layer_slots") f.layer_slots = v;
        else if (k == "slots") { f.slots_auto = v == "auto"; f.slots = f.slots_auto ? kAutoStartSlots : std::stoi(v); } else if (k == "budget_gb") f.budget_gb = std::stod(v);
        else if (k == "drafts") f.drafts = std::stoi(v); else if (k == "chunk") f.chunk = g_tune.chunk > 0 ? g_tune.chunk : std::stoi(v);  // --prefill-chunk (16384) over the manifest
        else if (k == "gsq_rco") f.gsq_rco = v != "0";
        else throw std::runtime_error("unknown Flash-Next manifest key " + k);
    }
    require(!f.ggufs.empty() && !f.mtp.empty(), "manifest needs gguf and mtp lines");
    if (!f.draft_vocab.empty()) setenv("TQFN_DRAFT_VOCAB", f.draft_vocab.c_str(), 1);
    g_kv_capacity = int(context);  // per slot
    { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); uint64_t keep = UINT64_MAX;
      ck(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep)); }
    // per-step temporaries grow with the prefill chunk (one layer at a time in prefill: measured 117 MB at chunk 16-64, 245 MB at 256; decode verify steps keep one span); TQFN_ARENA_MB overrides
    f.arena.size = size_t(std::getenv("TQFN_ARENA_MB") ? std::atoi(std::getenv("TQFN_ARENA_MB")) : std::max({256,
                   // layer-major chunks (256+) run the dense part in 256-token sub-chunks: measured 690 MB at 4096, 1375 at 8192
                   // (start-up arena capped at the 8K-chunk size: a bigger chunk's arena is borrowed per request, not held while
                   // the banks are built)
                   f.chunk >= 256 ? std::min(f.chunk, 8192) / 5 + 64 : 2 * f.chunk + 128,
                   slots_.size() > 1 ? 26 * std::min(64, int(slots_.size()) * (f.drafts + 1)) + 128 : 0})) << 20;  // + one span per batched verify (~26 MB a row)
    ck(cudaMalloc(&f.arena.base, f.arena.size)); g_arena = &f.arena; f.arena_prefill = f.arena.size;
    // --prefill-arena-mb: the prefill arena alone (bigger dense sub-chunks); the startup and decode arena keep their size,
    // the request's borrow evicts banks for the difference and they refill after the prompt
    f.arena_prefill = std::max({f.arena_prefill, size_t(std::max(0L, g_tune.arena_mb)) << 20, tqllm::prefill_arena_bytes(f.chunk)});
    // decode needs far less than a layer-major prefill (one step's temporaries); the arena shrinks after each long prompt
    f.arena_decode = std::getenv("TQFN_ARENA_MB") ? f.arena.size : std::min(f.arena.size, size_t(std::max(256,
                     slots_.size() > 1 ? 26 * std::min(64, int(slots_.size()) * (f.drafts + 1)) + 128 : 0)) << 20);
    f.arg_store = {"tqserve", "-"}; for (const auto & g : f.ggufs) f.arg_store.push_back(g);
    f.arg_store.push_back("--mtp"); f.arg_store.push_back(f.mtp); if (f.gsq_rco) f.arg_store.push_back("--gsq-rco");
    for (auto & a : f.arg_store) f.argv.push_back(a.data());
    f.fixtures = std::make_unique<Fixtures>();
    f.cp = std::make_unique<Checkpoint>(int(f.argv.size()), f.argv.data()); f.cp->resident = true;
    const auto & c = f.cp->config; auto & checkpoint = *f.cp;
    int integrated = 0; { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev); }
    const bool pack_in_memory = build_pack_in_memory(!f.experts.empty(), integrated != 0);
    g_small_card = f.slots_auto && (!f.experts.empty() || pack_in_memory) && tqllm::device_free_bytes() < (size_t(16) << 30);
    // One sequence: adaptive draft length (stop below a 0.9 draft probability, next-layer prediction for 6-token blocks):
    // 3090 chat +1.5 %, 32K +3 %; GB10 32K +4.4 %; text identical to drafts=3 (verify is batch-invariant). Not on small
    // cards: their CPU expert pool takes blocks of at most kCpuPoolRows rows (decode_rules.hpp one_sequence_drafts).
    // TQFN_DRAFT_TAU=0 turns it off; --np > 1 verifies every slot's full block, so it keeps the manifest's drafts
    if (const char * t = std::getenv("TQFN_DRAFT_TAU"); slots_.size() == 1 && (!t || std::strtof(t, nullptr) > 0.f)) {
        const DraftPlan plan = one_sequence_drafts(f.drafts, g_small_card);
        f.drafts = plan.drafts;
        if (plan.adaptive) {
            if (!t) setenv("TQFN_DRAFT_TAU", "0.9", 1);
            if (!std::getenv("TQFN_PREDICT_N")) setenv("TQFN_PREDICT_N", "6", 1);
        }
    }
    if (g_small_card) {
        // the dense fp16 cache holds one layer (measured high-water 119 MB on Flash-Next IQ3_S), not 600 MB
        if (!g_tune.dense_cache_set) g_tune.dense_cache_mb = std::min(g_tune.dense_cache_mb, 160L);
        // 256 experts stage as two even slices of 128 (192 + 64 at the default): 325 MB less staging, the same slice count
        if (!g_tune.stage_set) g_tune.stage = std::min(g_tune.stage, 128);
    }
    f.cache = std::make_unique<MixedExpertCache>(c.layers, c.experts, c.hidden, c.expert_ff, f.slots, size_t(f.budget_gb * 1024 * 1024 * 1024),
                [&checkpoint](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
    apply_layer_slots(*f.cache, f.experts, f.slots, f.layer_slots);
    // packed experts in pinned memory, tiled in place (same steps as --decode). Without an experts file (a GB10 with
    // every expert resident) the cache loads experts from the GGUF once and keeps them.
    if (!f.experts.empty() || pack_in_memory) {
    if (pack_in_memory) {  // no experts file: the same bytes, built from the GGUF (decode_rules.hpp build_pack_in_memory)
        const auto t0 = std::chrono::steady_clock::now();
        f.packed = build_packed(checkpoint, std::max(4, std::min(32, int(std::thread::hardware_concurrency()))));
        checkpoint.release_pages();  // the GGUF page cache would otherwise sit beside the pinned pack
        std::fprintf(stderr, "tqllm: packed experts built from the GGUF in memory: %.1f GB in %.0f s\n", f.packed.bytes / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    } else {
    f.packed = open_packed(f.experts);
    require(f.packed.layers == c.layers && f.packed.experts == c.experts, "packed experts do not match the model");
    char * host = nullptr; ck(cudaMallocHost(reinterpret_cast<void **>(&host), f.packed.bytes));
    { const int fd = ::open(f.experts.c_str(), O_RDONLY); require(fd >= 0, "cannot reopen " + f.experts);
      for (size_t done = 0; done < f.packed.bytes;) {
          const ssize_t r = ::pread(fd, host + done, std::min<size_t>(size_t(1) << 28, f.packed.bytes - done), off_t(done));
          require(r > 0, "short read of " + f.experts); done += size_t(r);
      }
      ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
    ::munmap(const_cast<char *>(f.packed.map), f.packed.bytes); f.packed.map = host;
    }
    if (!f.packed.tiled) {
        for (int l = 0; l < f.packed.layers; ++l) {
            size_t off = 0; const auto ids = expert_u4h_ids(checkpoint, l);
            for (int pp = 0; pp < 3; ++pp) {
                const int cols = pp == 2 ? c.expert_ff : c.hidden, rows = pp == 2 ? c.hidden : c.expert_ff;
                const int bpb = tqllm_cuda::tiled_block_bytes(ids[pp]); const size_t dsz = f.packed.sizes[l][2 * pp];
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
    if (f.packed.tiled) { tqllm_cuda::set_tiled_experts(true); g_tile_experts = true; }
    if (std::getenv("TQFN_CPU_MISS") || g_small_card) {  // CPU pool reads the pinned copy; the GGUF expert pages would hold the experts twice
        static CpuPackedView pv; pv = CpuPackedView{}; pv.map = f.packed.map; pv.base = f.packed.base; pv.block = f.packed.block;
        pv.sizes = f.packed.sizes; pv.tiled = f.packed.tiled; pv.layers = f.packed.layers;
        for (int l = 0; l < f.packed.layers; ++l) pv.fmt.push_back(expert_u4h_ids(checkpoint, l));
        g_cpu_packed = &pv; std::printf("tqllm: CPU miss pool reads the pinned packed experts\n");
    }
    PackedExperts * pk = &f.packed;
    f.cache->set_prepacked([pk](int layer, int id) -> const char * { return layer < pk->layers ? pk->at(layer, id) : nullptr; }, true);
    }
    // MTP experts prepacked in pinned memory
    // all experts resident (GB10): the MTP block's experts too, so its prefill catch-up can run whole chunks
    const int mtp_slots = std::getenv("TQFN_MTP_SLOTS") ? std::atoi(std::getenv("TQFN_MTP_SLOTS")) : g_small_card ? 64 : (f.slots >= c.experts ? c.experts : 256);
    f.mtp_cache = std::make_unique<MixedExpertCache>(c.layers + 1, c.experts, c.hidden, c.expert_ff, mtp_slots, size_t(mtp_slots) * (6u << 20) + (64u << 20),
                    [&checkpoint](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
    const auto first = repack_expert(checkpoint, c.layers, 0); for (const auto & p : first) f.mtp_block += p.size();
    ck(cudaMallocHost(reinterpret_cast<void **>(&f.mtp_packed), f.mtp_block * size_t(c.experts)));
    { std::vector<std::thread> pool; std::atomic<int> next{0};
      for (int t = 0; t < 16; ++t) pool.emplace_back([&] {
          for (int e; (e = next++) < c.experts;) {
              const auto parts = repack_expert(checkpoint, c.layers, e); size_t off = 0;
              for (const auto & p : parts) { std::memcpy(f.mtp_packed + size_t(e) * f.mtp_block + off, p.data(), p.size()); off += p.size(); }
          }
      });
      for (auto & t : pool) t.join(); }
    char * mp = f.mtp_packed; const size_t mb = f.mtp_block; const int L = c.layers;
    f.mtp_cache->set_prepacked([mp, mb, L](int layer, int id) -> const char * { return layer == L ? mp + size_t(id) * mb : nullptr; }, true);
    std::fprintf(stderr, "tqllm: Flash-Next ready: %d layers, %d experts, %d slots/layer, context %u, %d drafts\n",
                 c.layers, c.experts, f.slots, context, f.drafts);
    // warm-up: one short prompt uploads every dense weight and creates every expert bank now, so the first request is
    // not slowed by it and free-memory readings (the prefill borrow) are true from the start
    tqllm_cuda::qsa_reserve_split_scratch();  // decode attention scratch before the banks take the free memory
    // MMQ kernels load before the banks take the free memory
    std::fprintf(stderr, "tqllm: MMQ kernels loaded: %.1f MB\n", tqllm_cuda::mmq_warm() / 1048576.0);
    // the warm model's attention state is sized by the context capacity (8 GB at 262144): it must not hold that while
    // the banks are built, or the slots are capped by it; the warm-up only needs its 64 tokens
    const int kv_capacity_full = g_kv_capacity; g_kv_capacity = std::min(g_kv_capacity, 4096);
    // the device memory a full context keeps beside the banks: its attention state and the decode reserve
    auto context_keep = [&] {
        int attn = 0; for (int l = 0; l < c.layers; ++l) if (checkpoint.has_tensor("blk." + std::to_string(l) + ".indexer.k_proj.weight")) ++attn;
        // in decode the K/V of positions older than TQFN_KV_HOST_KEEP live in host memory (kv_to_host); the indexer keys stay
        const bool kv_host = std::getenv("TQFN_KV_HOST") && std::atoi(std::getenv("TQFN_KV_HOST")) != 0 && KvBuf::mem().granule();
        const int keep = std::getenv("TQFN_KV_HOST_KEEP") ? std::atoi(std::getenv("TQFN_KV_HOST_KEEP")) : 32768;
        const size_t kv_row = size_t(2) * checkpoint.attention_kvheads * checkpoint.attention_dim * 2;  // K + V bf16 per position
        const size_t moved = kv_host && int(context) > 2 * keep ? size_t(attn) * (size_t(context) - size_t(2 * keep)) * kv_row : 0;
        const size_t state = tqllm::prefill_context_bytes(int(context), 16, attn, checkpoint.attention_kvheads, checkpoint.attention_dim, c.index_dim) - moved;
        // decode scratch, the MTP bank rebuilt after a multi-chunk prompt, the prefill staging halves, refill staging
        // (2 GB for a long-context server: the 245K cells at 64 / 72 slots failed with 1 GB; a chat server at 33K runs
        // 140 slots with 1.8 GB free, so short contexts keep 512 MB)
        const size_t reserve = size_t(context > 65536 ? 2048 : 512) << 20;
        return std::pair<size_t, size_t>{state, reserve};
    };
    // the warm-up's steps (64 and 512 tokens) need a fraction of the start-up arena and of the staging halves: hold only
    // that while the banks are built, so the slot count is not capped by buffers the warm-up does not use
    if (const size_t warm_arena = std::max(f.arena_decode, tqllm::prefill_arena_bytes(512)); f.arena.size > warm_arena) {
        ck(cudaFree(f.arena.base)); ck(cudaMalloc(&f.arena.base, warm_arena)); f.arena.size = warm_arena; f.arena.used = 0; }
    const int stage_full = g_tune.stage; g_tune.stage = std::min(g_tune.stage, 32);
    { const auto w0 = std::chrono::steady_clock::now();
      DecodeModel warm(checkpoint, *f.fixtures, *f.cache, 640 + 3 * 64);  // 64 + 512 tokens, then up to three slots=auto rebuilds of 64
      if (warm.miss_stream) f.cache->set_miss_stream(warm.miss_stream);
      std::vector<int> ids(64); for (int i = 0; i < 64; ++i) ids[size_t(i)] = 1000 + (i * 37) % 50000;
      auto rows = warm.step(ids); ck(cudaDeviceSynchronize());
      // then one 512-token layer-major chunk: the prefill kernels' modules load and the pinned buffers and resident
      // router copies are created now (nsys: 0.3 s of module loads and pinned allocations inside the first prompt)
      // (needs the prepacked expert file: a GB10 build keeps every expert resident and has none)
      if (f.cache->has_layout(0) && f.cache->bank_layout(0).src0) {
        std::vector<int> lm(512); for (int i = 0; i < 512; ++i) lm[size_t(i)] = 2000 + (i * 53) % 50000;
        (void) warm.step(lm); ck(cudaDeviceSynchronize()); warm.release_prefill(); }
      g_tune.stage = stage_full;  // requests stage their full groups (sized again under the request's borrow)
      (void) checkpoint.head_argmax_rows(rows->p, 1); rows.reset();  // the full LM head loads here, not at the first prompt's first token (1.9 s)
      warm.mtp_experts = f.mtp_cache.get();  // the MTP layer too: its bank and weights, so the first request does not build them
      (void) warm.mtp_batch(std::vector<int>(ids.begin(), ids.begin() + 8), warm.res_keep->p); ck(cudaDeviceSynchronize());
      // the first long prompt would otherwise grow the staging buffers mid-request (a whole MTP bank in the catch-up, a
      // whole trunk bank in the refill): size them now, the steady state every later request already has
      f.mtp_cache->reserve_staging(mtp_slots); f.cache->reserve_staging();
      if (f.slots_auto) {  // slots=auto: every other buffer is in place; the banks get what the context does not keep
          size_t per = 0; for (int l = 0; l < c.layers; ++l) if (f.cache->has_layout(l)) per += f.cache->bank_layout(l).block;
          const auto [state, reserve] = context_keep();
          // TQFN_AUTO_KEEP_MB (debug A/B): what slots=auto keeps free beside the banks, instead of the full-context attention
          // state + decode reserve (a long prompt then borrows banks for its state)
          static const char * keep_env = std::getenv("TQFN_AUTO_KEEP_MB");
          // every slot (--np) keeps its own full-context attention / recurrent / MTP state beside the banks: sized for one,
          // np 4 at 32K ran out of memory in the second request's state (F38)
          const size_t nseq = slots_.size();
          const size_t need = keep_env ? size_t(std::atoll(keep_env)) << 20 : g_small_card ? size_t(512) << 20 : auto_keep_bytes(nseq, state, reserve);
          if (g_small_card) std::fprintf(stderr, "tqllm: small card: CPU expert pool, 64 MTP slots, banks up to a 512 MB keep (long prompts borrow banks)\n");
          size_t extra = 0;  // what rebuilding the banks takes beyond the slots themselves (measured, not estimated)
          for (int round = 0; round < 3; ++round) {
              for (int l = 0; l < c.layers; ++l) f.cache->release_bank(l);
              { cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
              const size_t fr = tqllm::device_free_bytes();
              f.slots = tqllm::auto_expert_slots(fr, need + extra, per, c.experts);
              f.cache->set_layer_slots(std::vector<int>(size_t(c.layers), f.slots)); apply_layer_slots(*f.cache, f.experts, f.slots, f.layer_slots);
              { auto again = warm.step(ids); ck(cudaDeviceSynchronize()); }  // the banks again, at the new size
              const size_t after = tqllm::device_free_bytes();
              std::fprintf(stderr, "tqllm: expert slots auto: %.0f MB free without banks, %.0f MB kept, %.1f MB per slot -> %d slots/layer, %.0f MB free after\n",
                           fr / 1048576.0, (need + extra) / 1048576.0, per / 1048576.0, f.slots, after / 1048576.0);
              if (after >= need || f.slots <= 1) break;
              extra += need - after;
          }
          // tier A (the floor a bank never drops below): the slots a full-context state leaves, as with one sequence's whole
          // state kept free; tier B (the target) is what the smaller keep allows (decode_rules.hpp auto_keep_bytes)
          if (slot_tiers(nseq, keep_env != nullptr)) {
              const size_t fr = tqllm::device_free_bytes() + f.cache->stats().device_bytes;
              const int tier_a = std::min(f.slots, tqllm::auto_expert_slots(fr, state + reserve + (size_t(256) << 20) + extra, per, c.experts));
              std::vector<int> floor(size_t(c.layers));
              for (int l = 0; l < c.layers; ++l) floor[size_t(l)] = slot_floor(f.cache->slots(l), 0, f.slots, verify_slot_min(f.drafts, c.topk));  // the F39-safe minimum
              f.cache->set_layer_tiers(floor, g_small_card ? size_t(512) << 20 : size_t(256) << 20);  // small cards: the refill's keep
              std::fprintf(stderr, "tqllm: expert slots: %d per layer (%d with a full context's state kept; a long prompt's layers share what is free)\n", f.slots, tier_a);
          }
      }
      // TQFN_EXPERT_PROFILE=FILE (int32 counts [layers][experts], tools/fn_expert_profile.py): the banks start with each
      // layer's most-routed experts instead of what the synthetic warm-up prompt happened to route (the first request
      // otherwise runs on a cold cache)
      if (const char * prof = std::getenv("TQFN_EXPERT_PROFILE")) {
          std::ifstream in(prof, std::ios::binary); std::vector<int32_t> cnt(size_t(c.layers) * c.experts);
          in.read(reinterpret_cast<char *>(cnt.data()), std::streamsize(cnt.size() * 4));
          if (!in) throw std::runtime_error(std::string("TQFN_EXPERT_PROFILE: cannot read ") + prof + " as int32 [layers][experts]");
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
      f.res_host_cap = size_t(context) * size_t(warm.width); ck(cudaMallocHost(reinterpret_cast<void **>(&f.res_host), f.res_host_cap * sizeof(float)));
      f.cache->set_miss_stream(nullptr);
      if (tqllm::debug_on("mem")) checkpoint.resident_report();
      if (f.arena.size != f.arena_decode) {  // start the first request as every later one: decode-sized arena, more room to borrow
          ck(cudaFree(f.arena.base)); ck(cudaMalloc(&f.arena.base, f.arena_decode)); f.arena.size = f.arena_decode; f.arena.used = 0; }
      size_t fr = 0, tot = 0; ck(cudaMemGetInfo(&fr, &tot));
      std::fprintf(stderr, "tqllm: Flash-Next warm-up %.1f s: expert banks %.0f MB, %.0f MB free\n",
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(), f.cache->stats().device_bytes / 1048576.0, fr / 1048576.0); }
    g_kv_capacity = kv_capacity_full; f.kv_capacity_full = kv_capacity_full; g_qsa_batch.reset();  // the selection scratch was sized for the warm-up's capacity
    {   // the plan budget: what a prompt may plan with (device free + the expert banks), measured once with the pool trimmed and
        // rounded down to 256 MB, so the chunk / block a prompt length gets does not follow what earlier requests left in the
        // pool or in a partly refilled bank (a 100 MB difference flipped a 16K prompt between 3 x 6144 and 4 x 5120, F26)
        cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0));
        const size_t q = size_t(256) << 20;
        f.plan_budget = (tqllm::device_free_bytes() + f.cache->stats().device_bytes) / q * q;
    }
    {   // a long prompt allocates the attention state for the whole context next to the banks: refuse a slot count that
        // cannot hold it (it would prefill for minutes and then fail in decode)
        const auto [state, reserve] = context_keep();
        f.state_full = state; f.state_tokens = int(context);
        const size_t fr = tqllm::device_free_bytes();
        // slots=auto with one sequence keeps only the reserve on purpose (decode_rules.hpp auto_keep_bytes): long prompts borrow
        if (fr < state + reserve && (std::getenv("TQFN_AUTO_KEEP_MB") || g_small_card || (f.slots_auto && slots_.size() == 1))) {
            std::fprintf(stderr, "tqllm: %zu MB free < %zu MB for a full context; long prompts borrow expert banks\n", fr >> 20, (state + reserve) >> 20);
        } else if (fr < state + reserve) {
            const size_t per_slot = f.cache->stats().device_bytes / size_t(std::max(1, f.slots));
            const int drop = per_slot ? int((state + reserve - fr + per_slot - 1) / per_slot) : f.slots;
            throw std::runtime_error("Flash-Next: " + std::to_string(f.slots) + " expert slots per layer leave " + std::to_string(fr >> 20) +
                                     " MB free, a full " + std::to_string(context) + "-token context needs " + std::to_string((state + reserve) >> 20) +
                                     " MB (attention state + decode reserve): use at most " + std::to_string(std::max(0, f.slots - drop)) +
                                     " slots (manifest slots=) or a smaller --context");
        }
    }
    // the default stream is per thread: the engine thread's stream is not ordered after the loading thread's work
    ck(cudaDeviceSynchronize());
}

}  // namespace tqllm::flashnext
