// Flash-Next behind the tqllm Engine interface (tqserve), registered as the .tqfn model type.
#include "fn_model.hpp"
// Flash-Next behind the tqllm Engine interface (tqserve). The model is a manifest file NAME.tqfn of key=value lines:
//   gguf=<shard> (one line per shard, in order), mtp=<MTP gguf>, experts=<packed experts>, draft_vocab=<file>,
//   slots=176, budget_gb=18, drafts=3, chunk=8, gsq_rco=1
// One slot, greedy MTP speculative decoding (exact: tokens identical to plain greedy). A fresh DecodeModel per request
// resets the recurrent, PLE, attention and MTP state; the expert caches persist across requests.
#include "engine.hpp"
namespace {
struct FnServe {
    std::vector<std::string> ggufs; std::string mtp, experts, draft_vocab;
    int slots = 176, drafts = 3, chunk = 8; double budget_gb = 18; bool gsq_rco = true, slots_auto = false;
    std::vector<std::string> arg_store; std::vector<char *> argv;
    std::unique_ptr<Fixtures> fixtures; std::unique_ptr<Checkpoint> cp;
    std::unique_ptr<MixedExpertCache> cache, mtp_cache; PackedExperts packed; char * mtp_packed = nullptr; size_t mtp_block = 0;
    ScratchArena arena; size_t arena_prefill = 0, arena_decode = 0; int kv_capacity_full = 0;
    float * res_host = nullptr; size_t res_host_cap = 0;  // pinned prompt residuals of a multi-chunk prefill (one prefill at a time)
};
class FlashNextEngine final : public tqllm::Engine {
    FnServe f_; uint32_t context_;
    // one sequence per slot (KV, recurrent and MTP state, sampling); weights, expert caches and scratch are shared
    struct SlotState {
        std::unique_ptr<DecodeModel> model_; std::unique_ptr<Device<float>> prompt_res_;
        std::unique_ptr<tqllm::BankLedger> ledger_;  // this request's borrowed expert banks
        tqllm::ExpertCacheStats stats_at_decode_{}, mtp_stats_at_decode_{};  // TQLLM_DEBUG=mem: cache counters when the decode of this request began
        size_t plan_catchup_bytes_ = 0;             // the plan's catch-up peak: what an overlapped refill must leave free
        std::vector<int> out_; int generate_ = 0, d1_ = 0; bool done_ = true; uint64_t steps_ = 0, toks_ = 0; uint64_t prefill_us_ = 0, decode_us_ = 0;
        std::chrono::steady_clock::time_point t_begin_, t_decode_;
        float temp_ = 0.f, top_p_ = 1.f; int top_k_ = 0; std::mt19937_64 rng_{1};
    };
    std::vector<SlotState> slots_; int cur_ = 0;
    SlotState & S() { return slots_[size_t(cur_)]; }
    uint32_t slot_context() const { return context_; }  // Model passes the per-slot context (tqserve: --context / --np)
    // the shared caches copy misses on the active slot's stream
    void bind() { if (S().model_ && S().model_->miss_stream) { f_.cache->set_miss_stream(S().model_->miss_stream); f_.mtp_cache->set_miss_stream(S().model_->miss_stream); } }
    static bool is_eos(int t) { return t == 248046 || t == 248044; }
    // T > 0: sample each verify position from the target's own distribution; a draft is accepted only if it equals that
    // sample, so the output follows the target exactly (greedy drafts just lower the acceptance)
    std::vector<int> pick(const float * rows, int n) {
        auto & checkpoint = *f_.cp;
        if (S().temp_ <= 0.f) return checkpoint.head_argmax_rows(rows, n);
        const auto lg = checkpoint.head_logits_device(rows, n); const size_t V = lg.size() / size_t(n);
        std::vector<int> r(n); std::vector<std::pair<float, int>> cand;
        for (int i = 0; i < n; ++i) {
            const float * l = lg.data() + size_t(i) * V; float mx = l[0]; for (size_t v = 1; v < V; ++v) mx = std::max(mx, l[v]);
            cand.clear();
            for (size_t v = 0; v < V; ++v) { const float p = std::exp((l[v] - mx) / S().temp_); if (p > 1e-9f) cand.push_back({p, int(v)}); }
            std::sort(cand.begin(), cand.end(), [](auto a, auto b) { return a.first > b.first; });
            if (S().top_k_ > 0 && int(cand.size()) > S().top_k_) cand.resize(size_t(S().top_k_));
            double total = 0; for (auto & c : cand) total += c.first;
            if (S().top_p_ < 1.f) { double acc = 0; size_t keep = 0; for (; keep < cand.size(); ++keep) { acc += cand[keep].first / total; if (acc >= S().top_p_) { ++keep; break; } }
                                cand.resize(std::max<size_t>(1, std::min(keep, cand.size()))); total = 0; for (auto & c : cand) total += c.first; }
            double u = std::uniform_real_distribution<double>(0, total)(S().rng_), acc = 0; r[i] = cand.back().second;
            for (auto & c : cand) { acc += c.first; if (acc >= u) { r[i] = c.second; break; } }
        }
        return r;
    }
public:
    FlashNextEngine(const std::string & manifest, uint32_t context, int nslots) : context_(context), slots_(size_t(std::max(1, nslots))) {
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
            else if (k == "slots") { f.slots_auto = v == "auto"; f.slots = f.slots_auto ? 64 : std::stoi(v); } else if (k == "budget_gb") f.budget_gb = std::stod(v);
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
        f.cache = std::make_unique<MixedExpertCache>(c.layers, c.experts, c.hidden, c.expert_ff, f.slots, size_t(f.budget_gb * 1024 * 1024 * 1024),
                    [&checkpoint](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
        apply_layer_slots(*f.cache, f.experts, f.slots);
        // packed experts in pinned memory, tiled in place (same steps as --decode). Without an experts file (a GB10 with
        // every expert resident) the cache loads experts from the GGUF once and keeps them.
        if (!f.experts.empty()) {
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
                                char * d = host + f.packed.base[l] + uint64_t(e) * f.packed.block[l] + off;
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
        PackedExperts * pk = &f.packed;
        f.cache->set_prepacked([pk](int layer, int id) -> const char * { return layer < pk->layers ? pk->at(layer, id) : nullptr; }, true);
        }
        // MTP experts prepacked in pinned memory
        // all experts resident (GB10): the MTP block's experts too, so its prefill catch-up can run whole chunks
        const int mtp_slots = std::getenv("TQFN_MTP_SLOTS") ? std::atoi(std::getenv("TQFN_MTP_SLOTS")) : (f.slots >= c.experts ? c.experts : 256);
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
              const auto [state, reserve] = context_keep(); const size_t need = state + reserve + (size_t(256) << 20);
              size_t extra = 0;  // what rebuilding the banks takes beyond the slots themselves (measured, not estimated)
              for (int round = 0; round < 3; ++round) {
                  for (int l = 0; l < c.layers; ++l) f.cache->release_bank(l);
                  { cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
                  const size_t fr = tqllm::device_free_bytes();
                  f.slots = tqllm::auto_expert_slots(fr, need + extra, per, c.experts);
                  f.cache->set_layer_slots(std::vector<int>(size_t(c.layers), f.slots)); apply_layer_slots(*f.cache, f.experts, f.slots);
                  { auto again = warm.step(ids); ck(cudaDeviceSynchronize()); }  // the banks again, at the new size
                  const size_t after = tqllm::device_free_bytes();
                  std::fprintf(stderr, "tqllm: expert slots auto: %.0f MB free without banks, %.0f MB kept, %.1f MB per slot -> %d slots/layer, %.0f MB free after\n",
                               fr / 1048576.0, (need + extra) / 1048576.0, per / 1048576.0, f.slots, after / 1048576.0);
                  if (after >= need || f.slots <= 1) break;
                  extra += need - after;
              }
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
        {   // a long prompt allocates the attention state for the whole context next to the banks: refuse a slot count that
            // cannot hold it (it would prefill for minutes and then fail in decode)
            const auto [state, reserve] = context_keep();
            const size_t fr = tqllm::device_free_bytes();
            if (fr < state + reserve) {
                const size_t per_slot = f.cache->stats().device_bytes / size_t(std::max(1, f.slots));
                const int drop = per_slot ? int((state + reserve - fr + per_slot - 1) / per_slot) : f.slots;
                throw std::runtime_error("Flash-Next: " + std::to_string(f.slots) + " expert slots per layer leave " + std::to_string(fr >> 20) +
                                         " MB free, a full " + std::to_string(context) + "-token context needs " + std::to_string((state + reserve) >> 20) +
                                         " MB (attention state + decode reserve): use at most " + std::to_string(std::max(0, f.slots - drop)) +
                                         " slots (manifest slots=) or a smaller --context");
            }
        }
    }
    // TQLLM_DEBUG=mem: where the device memory is (MiB); "other" = KV, QSA keys, recurrent state, CUDA context
    void ledger(const char * at) const {
        size_t fr = 0, tot = 0; ck(cudaMemGetInfo(&fr, &tot));
        cudaMemPool_t pool; uint64_t res = 0, used = 0; ck(cudaDeviceGetDefaultMemPool(&pool, 0));
        ck(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &res)); ck(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
        const double M = 1048576.0, trunk = f_.cache->stats().device_bytes / M, mtp = f_.mtp_cache->stats().device_bytes / M,
                     stage = 2.0 * DecodeModel::stage().half_bytes / M, arena = f_.arena.size / M, gdn = tqllm_cuda::fn_gdn_workspace_bytes() / M,
                     inuse = (tot - fr) / M, other = inuse - res / M - trunk - mtp - stage - arena - gdn;
        std::fprintf(stderr, "LEDGER %s: in use %.0f = pool %.0f (used %.0f) + trunk banks %.0f + MTP banks %.0f + staging %.0f + arena %.0f + "
                     "GDN ws %.0f + other %.0f | free %.0f\n", at, inuse, res / M, used / M, trunk, mtp, stage, arena, gdn, other, fr / M);
    }
    // borrowed banks back, filled with the prompt's most-routed experts (Strata does the same)
    // after release_prefill (buffers freed, pool trimmed): the ledger rebuilds the borrowed banks; a bank that no longer
    // fits refills on its first use in decode
    void refill_borrowed() {
        auto & led = S().ledger_; if (!led) return;
        using pclk = std::chrono::steady_clock; const auto r0 = pclk::now();
        const size_t all = led->borrowed_total();
        if (led->state() == tqllm::BankLedger::State::Borrowed) led->buffers_released();
        const int done = led->refill(); cudaGetLastError();
        ck(cudaDeviceSynchronize());
        if (all) std::fprintf(stderr, "tqllm: refilled %d of %zu borrowed expert banks in %.0f ms\n", done, all, std::chrono::duration<double, std::milli>(pclk::now() - r0).count());
        S().model_->pf_refill.clear(); led.reset();
    }
    void arena_resize(size_t bytes) {
        if (f_.arena.size == bytes || !f_.arena.base) return;
        ck(cudaDeviceSynchronize()); ck(cudaFree(f_.arena.base)); f_.arena.base = nullptr;
        ck(cudaMalloc(&f_.arena.base, bytes)); f_.arena.size = bytes; f_.arena.used = 0;
    }
    void spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int) override {
        auto & checkpoint = *f_.cp; const auto & c = checkpoint.config;
        require(!prompt.empty() && prompt.size() + count + 16 <= slot_context(), "prompt plus max_tokens exceeds the slot context");
        S().model_.reset(); S().prompt_res_.reset(); g_trim_below = SIZE_MAX;
        if (slots_.size() == 1 && f_.kv_capacity_full) {  // one sequence: its attention state is sized by the request, not the slot context
            const int need = int(std::min<size_t>(size_t(f_.kv_capacity_full), (prompt.size() + count + 16 + 1023) / 1024 * 1024));
            if (need != g_kv_capacity) { if (g_qsa_batch) { ck(cudaDeviceSynchronize()); g_qsa_batch.reset(); } g_kv_capacity = need; }
        }
        if (g_arena) g_arena->used = 0;
        // the pool keeps what the last request freed (release threshold max); hand it back so plain cudaMalloc users (expert
        // banks, staging, attention workspaces) can have it
        // a failed earlier request may have left a non-sticky CUDA error pending (QSA / CUB would report it on every later
        // request); a sticky one means the context is lost
        if (const cudaError_t e = tqllm::clear_stale_cuda_error(); e != cudaSuccess)
            throw std::runtime_error(std::string("CUDA context lost by an earlier request (") + cudaGetErrorString(e) + "); restart tqserve");
        { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
        if (tqllm::debug_on("mem")) { std::fprintf(stderr, "LEDGER request of %zu prompt tokens\n", prompt.size()); ledger("request start"); }
        S().model_ = std::make_unique<DecodeModel>(checkpoint, *f_.fixtures, *f_.cache, prompt.size() + count + 16);
        S().model_->mtp_experts = f_.mtp_cache.get();
        bind();
        const std::vector<int> ids(prompt.begin(), prompt.end());
        // TQFN_PROFILE_REQ=N: cudaProfilerStart at the N-th request's prefill, stop after it (nsys --capture-range=cudaProfilerApi)
        static const int prof_req = std::getenv("TQFN_PROFILE_REQ") ? std::atoi(std::getenv("TQFN_PROFILE_REQ")) : 0; static int req_no = 0;
        // TQFN_PROFILE_DECODE=1: the capture covers that request's decode instead of its prefill
        static const bool prof_decode = std::getenv("TQFN_PROFILE_DECODE") && std::atoi(std::getenv("TQFN_PROFILE_DECODE")) != 0;
        const bool profiling = prof_req > 0 && ++req_no == prof_req && !prof_decode;
        g_profile_decode_req = prof_req > 0 && req_no == prof_req && prof_decode;
        if (profiling) ck(cudaProfilerStart());
        S().t_begin_ = std::chrono::steady_clock::now(); S().prefill_us_ = S().decode_us_ = 0;
        S().out_.clear(); S().generate_ = int(count); S().steps_ = S().toks_ = 0;
        // Trunk prefill first, then the MTP catch-up, so their working sets do not add up: a prompt longer than one chunk
        // keeps its residuals in pinned host memory, drops the MTP bank before the trunk and the layer-major buffers
        // after it. The catch-up runs the same batches in the same order as when it followed each chunk.
        const int P = int(ids.size()); const int W = S().model_->width;
        // the prefill chunk for this request: the engine's chunk, halved while the plan cannot fit even with every bank
        // released (a 128K prompt at 104 slots does not fit 16K chunks on a 24 GB card)
        int chunk = f_.chunk; size_t arena_req = f_.arena_prefill;  // the arena a chunk needs: the start-up size, or less once the chunk has halved
        if (P >= 256 && chunk >= 256) {
            const auto & c = checkpoint.config;
            tqllm::PrefillShape sh; sh.hidden = c.hidden; sh.hc = c.hc; sh.expert_ff = c.expert_ff; sh.topk = c.topk; sh.experts = c.experts; sh.layers = c.layers;
            // the staging halves are sized for the largest expert block of any layer (prefill_layer): mixed-format files have
            // bigger blocks in some layers (the first layer's block undercounted staging by 170 MB at 245K)
            for (int l = 0; l < c.layers; ++l) if (f_.cache->has_layout(l)) {
                const auto lay = f_.cache->bank_layout(l); size_t blk = 0;
                for (int i = 0; i < 6; ++i) blk += lay.piece[i];
                sh.expert_block_bytes = std::max(sh.expert_block_bytes, std::max(blk, size_t(lay.block)) + 6 * 256 / std::max(1, g_tune.stage));
            }
            tqllm::PrefillTuning tu; tu.sub = g_tune.sub; tu.stage = g_tune.stage;
            tu.dense_cache_bytes = dense_half_pool_bytes(); tu.mtp_bank_bytes = f_.mtp_cache->peak_device_bytes();
            tu.workspace_bytes = size_t(1024 + std::max(0L, g_tune.reserve_mb)) << 20;  // --prefill-reserve-mb: headroom for the kernels' own scratch
            // above 128K keep a small margin beyond the plan's own terms (allocator granularity, kernel scratch)
            if (P > 131072) tu.workspace_bytes += size_t(448) << 20;
            int attn = 0; for (int l = 0; l < c.layers; ++l) if (checkpoint.has_tensor("blk." + std::to_string(l) + ".indexer.k_proj.weight")) ++attn;
            auto context_term = [&](int block) { return tqllm::prefill_context_bytes(g_kv_capacity, block, attn, checkpoint.attention_kvheads, checkpoint.attention_dim, c.index_dim); };
            // the plan counts the QSA selection scratch in context_bytes: release the old one so the free memory it reads is right
            if (g_qsa_batch) { ck(cudaDeviceSynchronize()); g_qsa_batch.reset(); cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0)); }
            auto free_now = [] { return tqllm::device_free_bytes(); };
            const size_t mtp_room = P > chunk ? f_.mtp_cache->stats().device_bytes : 0;  // a multi-chunk prompt clears the MTP bank first
            std::vector<size_t> bank_bytes; for (int l : f_.cache->banks_lru()) bank_bytes.push_back(f_.cache->bank_bytes(l));
            const int block0 = std::clamp(g_tune.sub, 512, 2048);
            int block = block0;
            tqllm::PrefillPlan plan;
            for (;;) {
                tu.chunk = chunk; tu.sub = std::min(g_tune.sub, block); tu.context_bytes = context_term(block);
                arena_req = std::min<size_t>(f_.arena_prefill, tqllm::prefill_arena_bytes(chunk));  // the measured rule at every chunk size
                arena_req += size_t(block) * size_t(checkpoint.attention_heads) * 2052 * 8 + (size_t(64) << 20);  // one block's QSA score scratch + headroom
                tu.arena_bytes = arena_req;
                plan = tqllm::prefill_plan(sh, tu, P, free_now() + mtp_room);
                try { (void) tqllm::banks_to_borrow(plan, bank_bytes, g_tune.borrow, g_tune.borrow_count); break; }
                catch (const std::runtime_error & e) {
                    const auto & cb = plan.chunk;
                    std::fprintf(stderr, "tqllm: prefill of %d tokens: %s (chunk %d, block %d: activations %zu expert_io %zu staging %zu residual %zu arena %zu dense %zu workspace %zu, context %zu MB)\n",
                                 P, e.what(), chunk, block, cb.activations >> 20, cb.expert_io >> 20, cb.staging >> 20, cb.residual >> 20, cb.arena >> 20, cb.dense_cache >> 20, cb.workspace >> 20, tu.context_bytes >> 20);
                    // a smaller selection block first (~0.5 s at 32K); halving the chunk doubles the expert bytes streamed per prompt
                    if (block > 512) { block = block > 1024 ? 1024 : 512; continue; }
                    if (chunk <= 2048) throw;
                    chunk = std::max(2048, chunk * 7 / 8 / 1024 * 1024); block = block0;  // 16K -> 14K -> 12K -> 10.5K ...: every chunk streams all experts, so take the largest that fits
                }
            }
            g_qsa_block = block; g_sub_req = block;
            // the first chunk's host prep (embedding + PLE gathers, ~0.4 s at 16K) runs while the banks, KV and arena are set up
            { DecodeModel * m = S().model_.get(); std::vector<int> first(ids.begin(), ids.begin() + std::min(P, chunk));
              m->prep_next_ = std::async(std::launch::async, [m, first] { return m->prepare_host(first); }); }
            if (P > chunk) f_.mtp_cache->clear();  // a multi-chunk prompt needs the room; one chunk keeps the MTP bank
            tqllm::BankLedger::Ops ops;
            ops.lru_layers = [this] { return f_.cache->banks_lru(); };
            ops.bank_bytes = [this](int l) { return f_.cache->bank_bytes(l); };
            ops.release = [this](int l) { f_.cache->release_bank(l); };
            ops.refill = [this](int l) { auto & m = *S().model_; const auto it = m.pf_refill.find(l); if (it != m.pf_refill.end()) f_.cache->prepare(l, it->second); };
            ops.free_bytes = free_now;
            S().ledger_ = std::make_unique<tqllm::BankLedger>(std::move(ops));
            const auto ev = S().ledger_->borrow(plan, g_tune.borrow, g_tune.borrow_count); S().plan_catchup_bytes_ = plan.catchup_peak;
            g_trim_below = S().ledger_->borrowed_bytes() + f_.mtp_cache->peak_device_bytes() + (size_t(512) << 20);
            S().model_->pf_borrowed.assign(size_t(c.layers), 0);
            for (int l : ev) S().model_->pf_borrowed[size_t(l)] = 1;
            if (!ev.empty() || tqllm::debug_on("mem"))
                std::fprintf(stderr, "tqllm: prefill of %d tokens in %d chunks of %d: needs %.0f MB (chunk %.0f, catch-up %.0f), released %zu expert banks (%.0f MB), %.0f MB free, %.1f GB of experts to stream\n",
                             P, plan.chunks, chunk, plan.need / 1048576.0, plan.trunk_peak / 1048576.0, plan.catchup_peak / 1048576.0, ev.size(),
                             S().ledger_->borrowed_bytes() / 1048576.0, free_now() / 1048576.0, plan.host_copies / 1e9);
        }
        const bool host_res = P > chunk;
        using pclk = std::chrono::steady_clock; double trunk_ms = 0, mtp_ms = 0; g_qsa_ms = 0; const auto p0 = pclk::now();
        auto lap = [&](double & acc, pclk::time_point & t) { if (!g_prefill_log) return; ck(cudaDeviceSynchronize()); const auto u = pclk::now();
                                                             acc += std::chrono::duration<double, std::milli>(u - t).count(); t = u; };
        auto mem_note = [&](const char * at) {
            if (tqllm::debug_on("mem")) ledger(at);
            if (g_prefill_log) { ck(cudaDeviceSynchronize());
                std::fprintf(stderr, "tqllm: prefill +%.2f s %s\n", std::chrono::duration<double>(pclk::now() - p0).count(), at); }
        };
        mem_note("banks borrowed");
        if (const size_t back = S().model_->kv_to_device())
            std::fprintf(stderr, "tqllm: attention KV: %.0f MB back on the device for this prompt\n", back / 1048576.0);
        if (host_res) {
            f_.mtp_cache->clear();
            if (f_.res_host_cap < size_t(P) * W) {
                if (f_.res_host) ck(cudaFreeHost(f_.res_host));
                ck(cudaMallocHost(reinterpret_cast<void **>(&f_.res_host), size_t(P) * W * sizeof(float))); f_.res_host_cap = size_t(P) * W;
            }
        }
        mem_note("residual host buffer ready");
        if (P >= 256 && chunk >= 256) arena_resize(arena_req);  // layer-major prefill; decode shrinks it again
        mem_note("prefill arena ready");
        // TQFN_MTP_CATCHUP: tokens per MTP catch-up batch (each needs topk unique MTP slots)
        // a cache holding every expert takes any width (the whole chunk: grouped GEMM path); else unique experts <= slots
        static const int cw = f_.mtp_cache->slots() >= checkpoint.config.experts
            ? (std::getenv("TQFN_MTP_CATCHUP") ? std::atoi(std::getenv("TQFN_MTP_CATCHUP")) : chunk)
            : std::max(1, std::min(std::getenv("TQFN_MTP_CATCHUP") ? std::atoi(std::getenv("TQFN_MTP_CATCHUP")) : 16,
                                   f_.mtp_cache->slots() / checkpoint.config.topk));
        // TQFN_MTP_LAYER_MAJOR (default on): the catch-up runs the MTP layer layer-major in 1024-position pieces
        static const bool mtp_lm = !(std::getenv("TQFN_MTP_LAYER_MAJOR") && std::atoi(std::getenv("TQFN_MTP_LAYER_MAJOR")) == 0);
        auto catch_up = [&](int b, int e, const float * res) {  // res: residual rows b..e-1 on the device
            if (mtp_lm && chunk >= 256 && e - b >= 256) {  // layer-major MTP straight from the chunk's rows
                for (int q = b; q < e; q += 1024) {
                    const int qe = std::min(e, q + 1024); std::vector<int> toks;
                    for (int i = q; i < qe; ++i) toks.push_back(i + 1 < P ? ids[i + 1] : S().out_[0]);
                    if (g_arena) g_arena->used = 0;
                    S().d1_ = S().model_->mtp_prefill(toks, res + size_t(q - b) * W);
                }
                return;
            }
            for (int q = b; q < e; q += cw) {
                std::vector<int> toks;
                for (int i = q; i < std::min(e, q + cw); ++i) toks.push_back(i + 1 < P ? ids[i + 1] : S().out_[0]);
                if (g_arena) g_arena->used = 0;  // the catch-up only reads persistent residual rows
                S().d1_ = S().model_->mtp_batch(toks, res + size_t(q - b) * W);
            }
        };
        S().model_->keep_embeddings = host_res && mtp_lm && chunk >= 256; S().model_->prompt_emb.clear();
        if (S().model_->keep_embeddings) S().model_->prompt_emb.alloc(size_t(P) * size_t(c.hidden));
        for (int b = 0; b < P; b += chunk) {
            const int e = std::min(P, b + chunk);
            auto tl = pclk::now();
            const std::vector<int> next(ids.begin() + e, ids.begin() + std::min(P, e + chunk));
            auto rows = S().model_->step(std::vector<int>(ids.begin() + b, ids.begin() + e), &next);
            lap(trunk_ms, tl);
            static const bool req_log = std::getenv("TQFN_PROFILE_REQ") != nullptr;
            const auto stamp = [&](const char * what) { if (req_log) std::fprintf(stderr, "tqllm: req +%.3f s %s\n", std::chrono::duration<double>(pclk::now() - p0).count(), what); };
            stamp("step returned");
            if (e == P) S().out_.push_back(pick(rows->p + size_t(e - b - 1) * checkpoint.config.hidden, 1)[0]);
            stamp("first token picked");
            if (host_res) ck(cudaMemcpyAsync(f_.res_host + size_t(b) * W, S().model_->res_keep->p, size_t(e - b) * W * sizeof(float), cudaMemcpyDeviceToHost, nullptr));
            if (host_res) mem_note("chunk done");
            else { catch_up(b, e, S().model_->res_keep->p); lap(mtp_ms, tl); }
        }
        if (host_res) {
            mem_note("after trunk prefill");
            // the trunk buffers go first: the MTP bank (650 MB) needs their room (keeping them OOMed the bank upload at 140 slots)
            S().model_->release_prefill(); S().model_->pf_borrowed.clear();
            mem_note("prefill buffers released");
            if (std::getenv("TQFN_PROFILE_REQ")) std::fprintf(stderr, "tqllm: req +%.3f s buffers released\n", std::chrono::duration<double>(pclk::now() - p0).count());
            auto tl = pclk::now();
            // residual rows staged through a small device buffer, a whole number of catch-up batches, starting at each
            // chunk boundary so the batches are the ones the interleaved order ran
            const int rb = cw * std::max(1, 256 / cw);
            std::unique_ptr<Device<float>> res;
            { PersistentAlloc keep; res = std::make_unique<Device<float>>(size_t(rb) * W); }
            if (mtp_lm && chunk >= 256) {
                const int mb = std::min(P, chunk);  // one piece per chunk: the MTP experts are staged once per piece
                std::unique_ptr<Device<float>> rows;
                { PersistentAlloc keep; rows = std::make_unique<Device<float>>(size_t(mb) * W); }
                const auto & pemb = S().model_->prompt_emb; const size_t hd = size_t(c.hidden);
                std::vector<float> pe;
                // keys / values for every position but the last, then the full MTP layer for the last one (mtp_kv_prefill)
                for (int q = 0; q < P; q += mb) {
                    const int qe = std::min(P, q + mb);
                    ck(cudaMemcpyAsync(rows->p, f_.res_host + size_t(q) * W, size_t(qe - q) * W * sizeof(float), cudaMemcpyHostToDevice, nullptr));
                    std::vector<int> toks;
                    for (int i = q; i < qe; ++i) toks.push_back(i + 1 < P ? ids[i + 1] : S().out_[0]);
                    const float * emb = nullptr;
                    if (pemb.size() == size_t(P) * hd) {  // the next tokens' embeddings from the trunk's prep; the last one (the first output) is gathered
                        pe.resize(size_t(qe - q) * hd);
                        const int kept = std::min(qe, P - 1) - q;
                        std::memcpy(pe.data(), pemb.data() + size_t(q + 1) * hd, size_t(kept) * hd * sizeof(float));
                        if (qe == P) { const auto last = checkpoint.file("token_embd.weight").gather_rows("token_embd.weight", {uint64_t(S().out_[0])});
                                       std::memcpy(pe.data() + size_t(kept) * hd, last.data(), hd * sizeof(float)); }
                        emb = pe.data();
                    }
                    if (g_arena) g_arena->used = 0;
                    const int keep = qe == P ? qe - q - 1 : qe - q;  // the last position runs the whole layer below
                    if (keep > 0) S().model_->mtp_kv_prefill(std::vector<int>(toks.begin(), toks.begin() + keep), rows->p, emb);
                    if (qe == P) { if (g_arena) g_arena->used = 0; S().d1_ = S().model_->mtp_batch({toks.back()}, rows->p + size_t(qe - q - 1) * W); }
                }
                rows.reset(); S().model_->release_prefill();
                S().model_->prompt_emb.clear(); S().model_->keep_embeddings = false;
            } else
            for (int b = 0; b < P; b += chunk)
                for (int q = b, e = std::min(P, b + chunk); q < e; q += rb) {
                    const int qe = std::min(e, q + rb);
                    ck(cudaMemcpyAsync(res->p, f_.res_host + size_t(q) * W, size_t(qe - q) * W * sizeof(float), cudaMemcpyHostToDevice, nullptr));
                    catch_up(q, qe, res->p);
                }
            lap(mtp_ms, tl);
            mem_note("after MTP catch-up");
            if (std::getenv("TQFN_PROFILE_REQ")) std::fprintf(stderr, "tqllm: req +%.3f s after MTP catch-up\n", std::chrono::duration<double>(pclk::now() - p0).count());
        }
        if (g_prefill_log) std::fprintf(stderr, "tqllm: prefill %d tokens %.1f s: trunk %.1f s (QSA %.1f s), MTP catch-up %.1f s\n", P,
            std::chrono::duration<double>(pclk::now() - p0).count(), trunk_ms / 1000, g_qsa_ms / 1000, mtp_ms / 1000);
        if (!host_res && P >= 256 && chunk >= 256) S().model_->release_prefill();  // single-chunk layer-major prompt
        arena_resize(f_.arena_decode);
        if (P >= 256) {  // long prompt: the old K/V rows to host memory, the refill below takes the freed device memory
            using kclk = std::chrono::steady_clock; const auto k0 = kclk::now();
            if (const size_t freed = S().model_->kv_to_host())
                std::fprintf(stderr, "tqllm: attention KV: %.0f MB of old positions moved to host memory in %.0f ms\n", freed / 1048576.0,
                             std::chrono::duration<double, std::milli>(kclk::now() - k0).count());
        }
        refill_borrowed(); g_trim_below = SIZE_MAX;
        S().model_->set_spec(true);
        ck(cudaDeviceSynchronize()); S().t_decode_ = std::chrono::steady_clock::now();
        if (tqllm::debug_on("mem")) { S().stats_at_decode_ = f_.cache->stats(); S().mtp_stats_at_decode_ = f_.mtp_cache->stats(); }
        S().prefill_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(S().t_decode_ - S().t_begin_).count());
        if (profiling) ck(cudaProfilerStop());
        if (g_profile_decode_req) ck(cudaProfilerStart());
        if (g_arena && std::getenv("TQFN_ARENA_LOG")) std::fprintf(stderr, "tqllm: prefill %d tokens, arena high %.0f MB of %.0f\n", P, g_arena->high / 1048576.0, g_arena->size / 1048576.0);
        S().done_ = int(S().out_.size()) >= S().generate_ || is_eos(S().out_.back());
        (void) c;
    }
    bool spec_advance() override {
        if (S().done_) return false;
        if (S().steps_ == 8 && tqllm::debug_on("mem")) ledger("decode step 8");
        bind();
        refill_borrowed();  // no-op unless an earlier refill ran out of memory
        auto & checkpoint = *f_.cp; const int drafts = f_.drafts;
        const int y = S().out_.back();
        const std::vector<int> dr = S().model_->mtp_drafts(S().d1_, drafts);
        std::vector<int> block{y}; block.insert(block.end(), dr.begin(), dr.end());
        auto rows = S().model_->step(block);
        const auto v = pick(rows->p, int(block.size()));
        int m = 1; while (m <= drafts && v[m - 1] == dr[m - 1]) ++m;
        for (int i = 0; i < m && int(S().out_.size()) < S().generate_; ++i) { S().out_.push_back(v[i]); if (is_eos(v[i])) { m = i + 1; break; } }
        S().model_->accept(m); ++S().steps_; S().toks_ += m;
        S().d1_ = S().model_->mtp_batch(std::vector<int>(v.begin(), v.begin() + m), S().model_->res_keep->p);
        S().done_ = int(S().out_.size()) >= S().generate_ || is_eos(S().out_.back());
        if (S().done_ && g_profile_decode_req) { ck(cudaDeviceSynchronize()); ck(cudaProfilerStop()); g_profile_decode_req = false; }
        if (S().done_) { S().model_->set_spec(false); ck(cudaDeviceSynchronize());
                     S().decode_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - S().t_decode_).count()); }
        return !S().done_;
    }
    std::vector<uint32_t> spec_finish(uint64_t * stats) override {
        bind();
        while (spec_advance()) {}
        // steps, drafted, accepted drafts, decode us, prefill us (tqserve timings)
        if (stats) { stats[0] = S().steps_; stats[1] = S().steps_ * uint64_t(f_.drafts); stats[2] = S().toks_ - S().steps_; stats[3] = S().decode_us_; stats[4] = S().prefill_us_; }
        std::vector<uint32_t> r(S().out_.begin(), S().out_.end());
        if (tqllm::debug_on("mem")) {  // the decode phase's expert cache traffic (the warm-prompt decode question of 10-17)
            const auto s = f_.cache->stats(), & b = S().stats_at_decode_, m = f_.mtp_cache->stats(), & mb = S().mtp_stats_at_decode_;
            std::fprintf(stderr, "LEDGER decode of %zu tokens: expert hits %llu misses %llu (%.1f %% hit) transfer %.2f GB, banks %.0f MB | MTP hits %llu misses %llu transfer %.2f GB, bank %.0f MB\n", r.size(),
                         (unsigned long long) (s.hits - b.hits), (unsigned long long) (s.misses - b.misses),
                         100.0 * double(s.hits - b.hits) / std::max<double>(1, double(s.hits - b.hits + s.misses - b.misses)),
                         double(s.transfer_bytes - b.transfer_bytes) / 1e9, s.device_bytes / 1048576.0,
                         (unsigned long long) (m.hits - mb.hits), (unsigned long long) (m.misses - mb.misses), double(m.transfer_bytes - mb.transfer_bytes) / 1e9, m.device_bytes / 1048576.0);
        }
        S().model_.reset(); S().prompt_res_.reset();
        return r;
    }
    std::vector<uint32_t> spec_tokens(int s) const override {
        const auto & o = slots_[size_t(s)].out_; return std::vector<uint32_t>(o.begin(), o.end());
    }
    int slots() const override { return int(slots_.size()); }
    void select_slot(int s) override { require(s >= 0 && s < int(slots_.size()), "Flash-Next slot out of range"); cur_ = s; }
    // one step for every listed slot: each drafts with its own MTP state, one verify pass covers all of them
    // (TQFN_SLOT_BATCH=0, a single active slot, more than 64 rows or more experts than cache slots: one slot at a time)
    void spec_step(const int * ids, int n, bool * done) override {
        const int drafts = f_.drafts, topk = f_.cp->config.topk;
        std::vector<int> act;
        for (int i = 0; i < n; ++i) { done[i] = slots_[size_t(ids[i])].done_; if (!done[i]) act.push_back(i); }
        const int rows = int(act.size()) * (drafts + 1);
        if (act.size() < 2 || rows > 64 || (f_.cache->slots() < f_.cp->config.experts && rows * topk > f_.cache->slots())) {
            for (int i : act) { select_slot(ids[i]); done[i] = !spec_advance(); }
            return;
        }
        // every slot drafts and catches up in shared MTP passes
        std::vector<DecodeModel *> ms; std::vector<std::vector<int>> blocks, drs; std::vector<int> d1s;
        for (int i : act) { select_slot(ids[i]); refill_borrowed(); ms.push_back(S().model_.get()); d1s.push_back(S().d1_); }
        select_slot(ids[act[0]]); bind();
        drs = S().model_->mtp_drafts_multi(ms, d1s, drafts);
        for (size_t a = 0; a < act.size(); ++a) {
            std::vector<int> block{slots_[size_t(ids[act[a]])].out_.back()}; block.insert(block.end(), drs[a].begin(), drs[a].end());
            blocks.push_back(block);
        }
        select_slot(ids[act[0]]); bind();
        const auto rows_d = S().model_->step_multi(ms, blocks);
        std::vector<std::vector<int>> vs(act.size()); std::vector<int> ms_acc(act.size());
        size_t off = 0;
        for (size_t a = 0; a < act.size(); ++a) {  // pick and roll back each slot before any MTP catch-up reuses the arena
            select_slot(ids[act[a]]); auto & sl = S();
            const auto v = pick(rows_d->p + off * size_t(f_.cp->config.hidden), int(blocks[a].size())); off += blocks[a].size();
            int m = 1; while (m <= drafts && v[m - 1] == drs[a][m - 1]) ++m;
            for (int i = 0; i < m && int(sl.out_.size()) < sl.generate_; ++i) { sl.out_.push_back(v[i]); if (is_eos(v[i])) { m = i + 1; break; } }
            sl.model_->accept(m); ++sl.steps_; sl.toks_ += m; vs[a] = v; ms_acc[a] = m;
        }
        std::vector<int> d1n(act.size());
        {
            std::vector<std::vector<int>> t(act.size()); std::vector<const float *> hp(act.size());
            for (size_t a = 0; a < act.size(); ++a) { t[a].assign(vs[a].begin(), vs[a].begin() + ms_acc[a]); hp[a] = ms[a]->res_keep->p; }
            select_slot(ids[act[0]]); bind();
            d1n = S().model_->mtp_batch_multi(ms, t, hp);
        }
        for (size_t a = 0; a < act.size(); ++a) {
            select_slot(ids[act[a]]); bind(); auto & sl = S();
            sl.d1_ = d1n[a];
            sl.done_ = int(sl.out_.size()) >= sl.generate_ || is_eos(sl.out_.back());
            if (sl.done_) { sl.model_->set_spec(false); ck(cudaDeviceSynchronize());
                            sl.decode_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sl.t_decode_).count()); }
            done[act[a]] = sl.done_;
        }
    }
    bool incremental() const override { return true; }
    bool has_drafter() const override { return true; }
    void set_sampling(float temperature, float top_p, int top_k, uint64_t seed) override {
        S().temp_ = temperature; S().top_p_ = top_p > 0.f && top_p <= 1.f ? top_p : 1.f; S().top_k_ = top_k; S().rng_.seed(seed ? seed : 0x9e3779b97f4a7c15ull);
    }
    uint32_t step(uint32_t, uint32_t, bool, float *) override { throw std::runtime_error("Flash-Next serves through spec_begin / spec_advance"); }
    void verify(const uint32_t *, int, uint32_t, uint32_t *) override { throw std::runtime_error("Flash-Next: verify is internal"); }
    void commit(int) override { throw std::runtime_error("Flash-Next: commit is internal"); }
    const float * logits() const override { return nullptr; }
    void synchronize() override { ck(cudaDeviceSynchronize()); }
    uint32_t layers() const override { return uint32_t(f_.cp->config.layers); }
    uint32_t hidden() const override { return uint32_t(f_.cp->config.hidden); }
    uint32_t vocab() const override { return uint32_t(f_.cp->head_resident().m.rows); }
    uint32_t eos() const override { return 248046; }
    uint32_t context() const override { return context_; }
};
std::unique_ptr<tqllm::Engine> make_flashnext(const std::string & manifest, const std::string &, uint32_t context, int slots,
                                              const tqllm::EngineOptions & o) {
    if (o.prefill_chunk > 0) g_tune.chunk = o.prefill_chunk;
    if (o.prefill_sub > 0) g_tune.sub = o.prefill_sub;
    if (o.prefill_stage > 0) g_tune.stage = o.prefill_stage;
    if (o.prefill_reserve_mb >= 0) g_tune.reserve_mb = o.prefill_reserve_mb;
    if (o.prefill_arena_mb >= 0) g_tune.arena_mb = o.prefill_arena_mb;
    if (o.dense_cache_mb >= 0) g_tune.dense_cache_mb = o.dense_cache_mb;
    if (o.prefill_borrow == "all") g_tune.borrow = tqllm::BorrowMode::All;
    else if (!o.prefill_borrow.empty() && o.prefill_borrow != "auto") { g_tune.borrow = tqllm::BorrowMode::Count; g_tune.borrow_count = std::atoi(o.prefill_borrow.c_str()); }
    return std::make_unique<FlashNextEngine>(manifest, context, slots);
}
const bool registered_flashnext = (tqllm::register_flashnext_engine(&make_flashnext), true);
}  // namespace
