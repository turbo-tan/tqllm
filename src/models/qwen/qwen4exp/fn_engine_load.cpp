#include <cmath>
// Flash-Next engine load: manifest, checkpoint and expert caches, warm-up, expert slots and the context fit.
#include <fstream>
#include <iterator>
#include "device_free.hpp"
#include "fn_engine.hpp"
#include "decode_rules.hpp"
#include <chrono>
#include "hot_cold_policy.hpp"
#include "gsq_q2_host.hpp"
#include "expert_prediction.hpp"

namespace tqllm::flashnext {
namespace {
std::array<ExpertPayload,3> mapped_cold_payload(FnServe & f,int layer,int expert) {
    auto raw=f.cold_store->load_mapped(layer,expert);std::array<ExpertPayload,3> out;
    for(int p=0;p<3;++p)out[p]={GGML_GSQ_Q2_0,{raw.parts[p].data(),raw.parts[p].size()},raw.owner,true};
    return out;
}
}


FlashNextEngine::FlashNextEngine(const std::string & manifest, uint32_t context, int nslots) : context_(context), slots_(size_t(std::max(1, nslots))) {
    // above 8 concurrent slots some requests end after one token and np 16 fails (10-13, P8 build too): refuse until fixed
    require(nslots <= 8 || std::getenv("TQFN_ALLOW_NP_ABOVE_8") != nullptr,
            "Flash-Next serves up to --np 8 for now (a bug above 8 slots is being fixed); TQFN_ALLOW_NP_ABOVE_8=1 for diagnostics");
    auto & f = f_; std::ifstream in(manifest); require(bool(in), "cannot open Flash-Next manifest " + manifest);
    const auto dir = std::filesystem::path(manifest).parent_path();
    auto resolve = [&](const std::string & v) { return std::filesystem::path(v).is_absolute() ? v : (dir / v).string(); };
    bool slots_given = false;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        const auto eq = line.find('='); require(eq != std::string::npos, "manifest line without '=': " + line);
        const auto k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "gguf") f.ggufs.push_back(resolve(v)); else if (k == "mtp") f.mtp = resolve(v);
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
    f.chunk = prefill_chunk(g_tune.chunk, f.chunk);  // --prefill-chunk (16384) over the manifest, also when it has none
    if (!f.draft_vocab.empty()) setenv("TQFN_DRAFT_VOCAB", f.draft_vocab.c_str(), 1);
    g_kv_capacity = int(context);  // per slot
    { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); uint64_t keep = UINT64_MAX;
      ck(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep)); }
    // per-step temporaries grow with the prefill chunk (one layer at a time in prefill: measured 117 MB at chunk 16-64, 245 MB at 256; decode verify steps keep one span); TQFN_ARENA_MB overrides
    if (!f.hot_map_path.empty()) {
        require(f.experts.empty() && !std::getenv("TQFN_KEEP_MAP") && !std::getenv("TQFN_EXPERT_PROFILE"),
                "hot/cold: use original GGUFs, not pruning or a single-format packed bank/profile");
        require(nslots == 1 && f.cold_slots >= 1 && f.cold_slots <= 512, "hot/cold pilot: one session and cold_slots 1..512 required");
        f.chunk = std::min(f.chunk, 8); // pilot uses the same mixed-precision path for prefill and decode
    }
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
    f.cp = std::make_unique<Checkpoint>(int(f.argv.size()), f.argv.data()); f.cp->resident = true; f.cp->reclaim_expert_pages=f.cold_reclaim_hot;
    const auto & c = f.cp->config; auto & checkpoint = *f.cp;
    int integrated = 0; { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev); }
    double pack_bytes = 0;  // the GGUF bytes of the trunk experts, about the pinned pack's size
    for (int l = 0; l < c.layers; ++l) for (const auto & p : checkpoint.expert_payload(l, 0)) pack_bytes += double(p.bytes.size) * c.experts;
    const double host_ram = double(tqllm::host_ram_bytes());  // physical RAM capped by the container limit
    const bool pack_in_memory = build_pack_in_memory(!f.experts.empty(), integrated != 0, !f.hot_map_path.empty(), pack_bytes * 1.02, host_ram);
    if (f.experts.empty() && !integrated && !pack_in_memory)
        std::fprintf(stderr, "tqllm: experts read from the GGUF on demand (%s)\n", !f.hot_map_path.empty() ? "hot/cold tiers" :
                     "a pinned pack would not fit this machine's RAM");
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
        g_hot_map = &f.hot_map; g_cold_cache = f.cold_cache.get();
        std::fprintf(stderr, "tqllm: hot/cold pilot: %d current-format hot, %d Q2 cold experts per layer; %d cold GPU slots, no router mask\n",
                hot_count, c.experts - hot_count, f.cold_slots);
        std::fprintf(stderr, "tqllm: hot cache: %d GPU slots/layer for %d original-format hot experts\n", f.slots, hot_count);
    }
    size_t device_total = 0; (void) tqllm::device_free_bytes(&device_total);
    f.cache = std::make_unique<MixedExpertCache>(c.layers, c.experts, c.hidden, c.expert_ff, f.slots, bank_budget_bytes(f.slots_auto, f.budget_gb, device_total),
                [&checkpoint](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
    ::apply_layer_slots(*f.cache, f.experts, f.slots, f.layer_slots);
    // packed experts in pinned memory, tiled in place (same steps as --decode). Without an experts file (a GB10 with
    // every expert resident) the cache loads experts from the GGUF once and keeps them.
    if (!f.experts.empty() || pack_in_memory) {
    static std::vector<int32_t> keep_map; int keep_k = 0;  // TQFN_KEEP_MAP (pack-file path only)
    if (pack_in_memory) {  // no experts file: the same bytes, built from the GGUF (decode_rules.hpp build_pack_in_memory)
        const auto t0 = std::chrono::steady_clock::now();
        f.packed = build_packed(checkpoint, std::max(4, std::min(32, int(std::thread::hardware_concurrency()))));
        checkpoint.release_expert_pages();  // the expert pages would otherwise sit beside the pinned pack
        std::fprintf(stderr, "tqllm: packed experts built from the GGUF in memory: %.1f GB in %.0f s\n", f.packed.bytes / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    } else {
    f.packed = open_packed(f.experts);
    // TQFN_KEEP_MAP=FILE (expert pruning): int32 [layers][experts], each expert's index in a compact pack or -1 (pruned); the router
    // masks pruned experts out (g_keep_mask) and the pack holds only the kept ones (K per layer)
    if (const char * km = std::getenv("TQFN_KEEP_MAP")) {
        keep_map.assign(size_t(c.layers) * c.experts, -1);
        FILE * mf = std::fopen(km, "rb"); require(mf && std::fread(keep_map.data(), 4, keep_map.size(), mf) == keep_map.size(), std::string("TQFN_KEEP_MAP: cannot read ") + km);
        std::fclose(mf);
        std::vector<float> mask(keep_map.size());
        for (size_t i = 0; i < keep_map.size(); ++i) { mask[i] = keep_map[i] < 0 ? -INFINITY : 0.f; keep_k = std::max(keep_k, keep_map[i] + 1); }
        float * dm = nullptr; ck(cudaMalloc(reinterpret_cast<void **>(&dm), mask.size() * 4));
        ck(cudaMemcpy(dm, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice)); g_keep_mask = dm;
        size_t kept = 0; for (int32_t v : keep_map) kept += v >= 0;
        std::fprintf(stderr, "tqllm: expert pruning: %zu of %d experts kept (%d per layer in the pack, %.1f %%)\n", kept, c.layers * c.experts, keep_k, 100.0 * kept / (c.layers * c.experts));
    }
    require(f.packed.layers == c.layers && f.packed.experts == (keep_map.empty() ? c.experts : keep_k), "packed experts do not match the model");
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
    if (!keep_map.empty()) {
        const int32_t * km = keep_map.data(); const int E = c.experts;
        f.cache->set_prepacked([pk, km, E](int layer, int id) -> const char * {
            if (layer >= pk->layers) return nullptr; const int32_t q = km[size_t(layer) * E + id]; return q < 0 ? nullptr : pk->at(layer, q); }, true);
        g_cpu_packed = nullptr;  // the CPU pool's packed view indexes by global id: off with a compact pack
    } else
    f.cache->set_prepacked([pk](int layer, int id) -> const char * { return layer < pk->layers ? pk->at(layer, id) : nullptr; }, true);
    }
    {   // the per-layer token embedding is read at every token: cold pages cost each new token a disk wait (decode_rules.hpp)
        const auto t0 = std::chrono::steady_clock::now();
        if (const size_t n = checkpoint.warm_tensors(8, host_ram))
            std::fprintf(stderr, "tqllm: per-layer token embedding cached: %.1f GB in %.0f s\n", n / 1e9,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
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
      std::vector<int> ids(g_hot_map ? 1 : 64); for (int i = 0; i < int(ids.size()); ++i) ids[size_t(i)] = 1000 + (i * 37) % 50000;
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
      (void) warm.mtp_batch(std::vector<int>(ids.begin(), ids.begin() + std::min<size_t>(8, ids.size())), warm.res_keep->p); ck(cudaDeviceSynchronize());
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
              f.cache->set_layer_slots(std::vector<int>(size_t(c.layers), f.slots)); ::apply_layer_slots(*f.cache, f.experts, f.slots, f.layer_slots);
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
      // spec_begin grows pinned residuals to the actual prompt, not the configured slot capacity.
      f.cache->set_miss_stream(nullptr);
      if (tqllm::debug_on("mem")) checkpoint.resident_report();
      {   // a multi-chunk prefill's pinned residual rows, pinned here once instead of inside a request (decode_rules.hpp)
          const size_t rows = int(context) > 4096 ? size_t(context) * size_t(warm.width) : 0;
          std::ifstream mi("/proc/meminfo"); const std::string text((std::istreambuf_iterator<char>(mi)), std::istreambuf_iterator<char>());
          if (pin_residual_at_start(rows * sizeof(float), tqllm::host_ram_bytes(), tqllm::meminfo_available(text))) {
              ck(cudaMallocHost(reinterpret_cast<void **>(&f.res_host), rows * sizeof(float))); f.res_host_cap = rows;
              std::fprintf(stderr, "tqllm: prompt residual buffer pinned at start: %.1f GB for %u tokens\n", double(rows * sizeof(float)) / 1e9, context);
          }
      }
      if (f.arena.size != f.arena_decode) {  // start the first request as every later one: decode-sized arena, more room to borrow
          ck(cudaFree(f.arena.base)); ck(cudaMalloc(&f.arena.base, f.arena_decode)); f.arena.size = f.arena_decode; f.arena.used = 0; }
      size_t fr = 0, tot = 0; ck(cudaMemGetInfo(&fr, &tot));
      std::fprintf(stderr, "tqllm: Flash-Next warm-up %.1f s: expert banks %.0f MB, %.0f MB free\n",
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(), f.cache->stats().device_bytes / 1048576.0, fr / 1048576.0); }
    if(f.cold_reclaim_hot) {
        ck(cudaDeviceSynchronize());
        std::fprintf(stderr,"tqllm: uploaded checkpoint source page reclaim: %s\n",checkpoint.discard_source_pages() ? "ok" : "partial");
    }
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
