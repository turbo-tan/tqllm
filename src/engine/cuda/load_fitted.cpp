// The memory fit for the qwen35 GGUF engine (see load_fitted in cuda_engines.hpp).
#include "dev_alloc.cuh"
#include "tqllm/model.hpp"
#include "debug.hpp"
#include "../fit_context.hpp"
#include "dev_alloc.cuh"
#include "qwen2_kernels.cuh"
#include "cuda_engines.hpp"
#include "nlohmann/json.hpp"
#include "gguf.hpp"
#include "hf_folder.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tqllm {

namespace {
// Builds the engine leaving o.fit_target_mb (-fitt) of device memory free for run-time allocations (library workspaces,
// graphs; ~0.4 GB measured on a 3090 at 32K). A shortfall or an out-of-memory load applies the next step and reloads:
// embeddings, then the int8 LM head to host memory, smaller prefill buffers, the turbo3 V cache, prefill weight blocks,
// then half the context (not below -fitc). Options the user set are left alone; -fit off loads once.
std::string g_fit_note;  // settings this load's memory fit chose instead of the defaults (empty: none), -> Engine::note
std::unique_ptr<Engine> load_fitted_impl(const std::string & path, const std::string & tok, uint32_t ctx, int slots,
                                    EngineOptions opt) {
    // WSL / Windows (WDDM): cudaMemGetInfo reads ~0 near the budget, and 0.30 GB free at 262K ran at full speed with
    // no spill on a 5060 Ti (results/20261008_kv_host_tail), so the default margin and the floor are 256 MB there
    static const bool wsl = [] {
        std::ifstream v("/proc/version");
        std::string l;
        std::getline(v, l);
        return l.find("icrosoft") != std::string::npos;
    }();
    // parallel slots: each needs --slot-reserve MiB free after loading (its batched-verify / draft graphs); the fit makes
    // that room first (host placement, KV to host, context), then refuses an --np the memory cannot carry
    const size_t base_margin = fit_margin(opt.fit, opt.fit_target_mb, wsl);
    const size_t slot_b = slots > 1 ? size_t(std::max(0L, opt.slot_reserve_mb)) << 20 : 0;
    const size_t margin = base_margin ? base_margin + size_t(slots) * slot_b : 0;
    if (wsl) opt.graph_keep_mb = 0;  // the driver reads ~0 near the budget: the graph cache bounds itself by count only
    // in use before loading (CUDA context, desktop, other processes): read while far from the budget, so reliable on WSL too
    size_t free0 = 0, total0 = 0;
    if (cudaMemGetInfo(&free0, &total0) != cudaSuccess) free0 = total0 = 0;
    const size_t used_before = total0 - free0;
    if (wsl && opt.fit)
        std::fprintf(stderr, "tqllm: fit: WSL - judging free memory by the engine's own allocations (%.2f GB in use before load)\n",
                     double(used_before) / 1e9);
    auto measure = [&](size_t & fr, size_t & tot) {
        size_t df = 0;
        if (cudaMemGetInfo(&df, &tot) != cudaSuccess) return false;
        fr = fit_free(wsl, df, tot, used_before, tqllm_cuda::dev_bytes());
        if (tqllm::debug_on("memlog"))
            std::fprintf(stderr, "tqllm: fit measure: engine %.2f GB, driver free %.2f GB, total %.2f GB, before %.2f GB -> free %.2f GB\n",
                         double(tqllm_cuda::dev_bytes()) / 1e9, double(df) / 1e9, double(tot) / 1e9, double(used_before) / 1e9,
                         double(fr) / 1e9);
        return true;
    };
    // MoE experts in host memory: --moe-host-mb auto takes what the file needs beyond the free device memory (2 GiB left for
    // KV, buffers and the runtime; the steps below settle the rest); a VRAM cache of recently used host experts (--moe-cache-mb
    // auto) moves the same MiB of experts to host, so the device holds what it would without it
    if (opt.moe_host_mb != 0 || opt.moe_cache_mb > 0) {
        bool moe = false;
        try {
            const auto g = gguf_open(path);
            const std::string arch = g.has("general.architecture") ? g.str("general.architecture") : std::string();
            moe = g.num(arch + ".expert_count", 0) > 0;
        } catch (const std::exception &) {}
        if (!moe) opt.moe_host_mb = 0, opt.moe_cache_mb = 0;
        std::error_code ec;
        const uint64_t file_b = std::filesystem::file_size(path, ec);
        const bool host_auto_mb = opt.moe_host_mb < 0, cache_auto = opt.moe_cache_mb < 0;
        const MoePlace mp = moe_auto_place(ec ? 0 : file_b, free0, opt.moe_host_mb, opt.moe_cache_mb);
        opt.moe_host_mb = mp.host_mb, opt.moe_cache_mb = mp.cache_mb;
        if (moe && (host_auto_mb || (cache_auto && mp.cache_mb)))
            std::fprintf(stderr, "tqllm: fit: MoE experts in host memory %ld MiB%s, host-expert cache %ld MiB%s (%.2f GB free on the device)\n",
                         mp.host_mb, host_auto_mb ? " (--moe-host-mb auto)" : "", mp.cache_mb, cache_auto ? " (--moe-cache-mb auto)" : "",
                         double(free0) / 1e9);
    }
    const FitSetting asked{opt.kv_format(), ctx, long(opt.ubatch), long(opt.gemm_rows)};
    auto finish = [&](std::unique_ptr<Engine> en) {
        if (!en) return en;
        // the fit may move weights and KV between device and host (same results). Settings that change the arithmetic (KV
        // codec, context, prefill buffers, weight blocks) are applied automatically and announced (also in /health); an
        // explicit --kv / -ub / --gemm-rows / --context is never overridden. -fitl off refuses them instead.
        if (const std::string lossy = fit_result_changes(asked, {opt.kv_format(), en->context(), long(opt.ubatch), long(opt.gemm_rows)});
            !lossy.empty()) {
            if (!opt.fit_lossy)
                throw std::runtime_error("does not fit as asked (" + std::to_string(slots) + " slot(s), context " + std::to_string(asked.ctx) +
                                         " per slot, KV " + asked.kv + "): it would need " + lossy +
                                         ". Ask for that explicitly (--kv, -ub, --gemm-rows), use fewer slots (--np) or a smaller --context, or drop -fitl off");
            g_fit_note = "to fit " + std::to_string(slots) + " slot(s) x " + std::to_string(asked.ctx) + " in GPU memory: " + lossy;
            if (opt.exact) {
                g_fit_note = "exact on: " + g_fit_note;
                std::fprintf(stderr, "tqserve: WARN --exact on: %s forced by memory\n", lossy.c_str());
            }
            std::fprintf(stderr, "tqserve: fit: %s (choose yourself with --kv, -ub, --gemm-rows or a smaller --context)\n",
                         g_fit_note.c_str());
        }
        if (slots <= 1) return en;
        size_t fr = 0, tot = 0;
        if (!measure(fr, tot)) return en;
        const uint32_t ok = fit_slots(uint32_t(slots), en->context(), opt.fit_min_ctx, fr, base_margin, slot_b);
        if (ok >= uint32_t(slots)) return en;
        char msg[512];
        std::snprintf(msg, sizeof msg,
                      "--np %d does not fit: %.2f GB free after loading, context %u per slot; each slot needs %u positions "
                      "(-fitc) and %ld MiB (--slot-reserve) above the %.2f GB margin. Largest that fits: --np %u "
                      "(or a smaller --context)", slots, double(fr) / 1e9, en->context(), opt.fit_min_ctx, opt.slot_reserve_mb,
                      double(base_margin) / 1e9, ok);
        throw std::runtime_error(msg);
    };
    const bool user_pf = opt.ubatch != 0, user_kv = opt.kv_format() != "bf16";
    const bool user_head = opt.head_host, user_embed = opt.embed_host;  // asked for: never given back
    const size_t floor_b = size_t(256) << 20;
    const bool host_auto = opt.kv_host < 0;
    int host_tries = 0;         // KV-to-host steps applied so far
    size_t host_prev_free = 0;  // free memory measured before the last of them
    bool measuring = false;  // the last load put all KV in host memory to measure what the device can hold
    bool measured = false;   // that measuring load was made (once per fit)
    bool returned = false;   // spare memory after a host-tail fit was tried once
    uint32_t ctx_short = 0;  // the last context that loaded but left less than the margin free
    size_t free_short = 0;
    for (int step = 0;;) {
        std::unique_ptr<Engine> e;
        try {
            e = make_qwen35_engine(path, tok, ctx, slots, opt);
        } catch (const std::exception & ex) {
            if (!margin || !std::strstr(ex.what(), "out of memory")) throw;
        }
        size_t fr = 0, tot = 0;
        if (e && (!margin || (measure(fr, tot) && fr >= margin))) {
            if (measuring) {  // bring back to the device as many positions as the free memory holds
                measuring = false;
                const size_t tb = e->kv_token_bytes(), round_b = e->kv_round_bytes();
                const uint64_t dev_tok = tb && fr > margin + round_b ? (fr - margin - round_b) / tb : 0;
                const uint32_t h = dev_tok >= ctx ? 0 : uint32_t(std::min<uint64_t>(ctx, (ctx - dev_tok + 1023) / 1024 * 1024));
                if (h < uint32_t(opt.kv_host)) {
                    e.reset();
                    opt.kv_host = h;
                    std::fprintf(stderr, "tqllm: fit: %.2f GB free with all KV in host memory: last %u positions stay there\n",
                                 double(fr) / 1e9, h);
                    continue;
                }
            }
            // KV tail in host memory with room left on the GPU: bring the spare back once, keep it if the margin holds
            if (host_auto && opt.kv_host > 0 && !returned) {
                returned = true;
                // measured device cost per position (pages round to granules), not the nominal row size
                const uint32_t dev_pos = ctx > uint32_t(opt.kv_host) ? ctx - uint32_t(opt.kv_host) : 0;
                const size_t per = std::max(e->kv_token_bytes(), dev_pos ? e->kv_device_bytes() / dev_pos : size_t(0));
                const long prev = opt.kv_host;
                uint32_t h = fit_host_return(uint32_t(prev), fr, margin, per);
                for (int k = 0; k < 2 && h < uint32_t(prev); ++k) {  // a short return, then half of it
                    e.reset();
                    opt.kv_host = h;
                    std::unique_ptr<Engine> e2;
                    try {
                        e2 = make_qwen35_engine(path, tok, ctx, slots, opt);
                    } catch (const std::exception & ex) {
                        if (!std::strstr(ex.what(), "out of memory")) throw;
                    }
                    size_t fr2 = 0;
                    if (e2 && measure(fr2, tot) && fr2 >= margin) {
                        std::fprintf(stderr, "tqllm: fit: %ld positions back on the GPU (%u in host memory), %.2f GB free\n",
                                     prev - long(h), h, double(fr2) / 1e9);
                        return finish(std::move(e2));
                    }
                    h = uint32_t(prev) - (uint32_t(prev) - h) / 2 / 1024 * 1024;
                }
                if (opt.kv_host != prev) {
                    opt.kv_host = prev;
                    e = make_qwen35_engine(path, tok, ctx, slots, opt);
                    if (!e || !measure(fr, tot)) throw std::runtime_error("fit: reload after a failed KV return");
                }
            }
            // a halved context that fits with room to spare: take the largest context the two loads predict
            // a halved context that fits with room to spare: take the largest context the loads predict; an attempt that
            // falls short becomes the new upper point and the prediction is refined (up to 3 loads)
            uint32_t ok_ctx = ctx;
            size_t ok_free = fr;
            bool reloaded = false;
            for (int k = 0; ctx_short && k < 3; ++k) {
                const uint32_t best = fit_context(ctx_short, free_short, ok_ctx, ok_free, margin);
                if (best <= ok_ctx) break;
                e.reset();
                reloaded = true;
                std::unique_ptr<Engine> e2;
                try {
                    e2 = make_qwen35_engine(path, tok, best, slots, opt);
                } catch (const std::exception & ex) {
                    if (!std::strstr(ex.what(), "out of memory")) throw;
                }
                size_t fr2 = 0;
                if (e2) measure(fr2, tot);
                if (e2 && fr2 >= margin) {
                    std::fprintf(stderr, "tqllm: fit: context %u (interpolated), %.2f GB free after %d step(s)\n", best,
                                 double(fr2) / 1e9, step + 1 + k);
                    return finish(std::move(e2));
                }
                std::fprintf(stderr, "tqllm: fit: interpolated context %u left %.2f GB; refining\n", best, double(fr2) / 1e9);
                if (!e2) break;  // out of memory: no measurement to refine with
                ctx_short = best, free_short = fr2;
            }
            if (reloaded) {
                e = make_qwen35_engine(path, tok, ok_ctx, slots, opt);
                if (!e || !measure(fr, tot)) throw std::runtime_error("fit: reload at the interpolated context");
            }
            // give back placement steps that cost speed while the load leaves room for them (LM head first: every token)
            struct Back { bool * on; bool user; const char * what; };
            for (const Back & b : {Back{&opt.head_host, user_head, "int8 LM head back on the GPU"},
                                   Back{&opt.embed_host, user_embed, "token embeddings back on the GPU"}}) {
                if (!*b.on || b.user || !fit_try_return(fr, margin)) continue;
                e.reset();
                *b.on = false;
                std::unique_ptr<Engine> e2;
                try {
                    e2 = make_qwen35_engine(path, tok, ok_ctx, slots, opt);
                } catch (const std::exception & ex) {
                    if (!std::strstr(ex.what(), "out of memory")) throw;
                }
                size_t fr2 = 0;
                if (e2 && measure(fr2, tot) && fr2 >= margin) {
                    std::fprintf(stderr, "tqllm: fit: %s, %.2f GB free\n", b.what, double(fr2) / 1e9);
                    e = std::move(e2);
                    fr = fr2;
                    continue;
                }
                e2.reset();
                *b.on = true;
                e = make_qwen35_engine(path, tok, ok_ctx, slots, opt);
                if (!e || !measure(fr, tot)) throw std::runtime_error("fit: reload after a failed give-back");
                break;
            }
            if (step) std::fprintf(stderr, "tqllm: fit: %.2f GB free after %d step(s)\n", double(fr) / 1e9, step);
            return finish(std::move(e));
        }
        // every memory step applied (or all KV already in host memory): keep the context if the floor is free
        if (e && fr >= floor_b && (step >= 8 || (host_auto && opt.kv_host >= long(ctx)))) {
            std::fprintf(stderr, "tqllm: fit: WARNING %.2f GB free (< %.2f GB margin) after %d step(s); keeping context %u\n",
                         double(fr) / 1e9, double(margin) / 1e9, step, ctx);
            return finish(std::move(e));
        }
        const bool loaded = bool(e);
        const size_t token_b = e ? e->kv_token_bytes() : 0;
        e.reset();
        std::string what;
        for (bool applied = false; !applied; ++step) {
            switch (step) {
            case 0: if (!opt.embed_host) { opt.embed_host = true; what = "token embeddings to host memory"; applied = true; } break;
            case 1: if (!opt.head_host) { opt.head_host = true; what = "int8 LM head to host memory"; applied = true; } break;
            case 2: if (!user_pf) { opt.ubatch = 1024; what = "prefill buffers 1024"; applied = true; } break;
            case 3: if (!user_pf) { opt.ubatch = 512; what = "prefill buffers 512"; applied = true; } break;
            // q4 K + trellis 2-bit V: lowest KLD of the compact caches at 64K-200K (results/20261006_kv_longctx_kld)
            case 4: if (!user_kv) { opt.kv = "q4/tcq2"; what = "KV cache q4/tcq2"; applied = true; } break;
            case 5: if (!opt.gemm_rows) { opt.gemm_rows = 2048; what = "prefill FP16 weight blocks of 2048 rows"; applied = true; } break;
            case 6:  // KV tail to host memory (exact; decode reads it over PCIe) before giving up context
            case 7:  // refine while each step frees memory (the per-token estimate undershoots: buffers grow with the tail)
                if (!host_auto) break;
                if (host_tries > 0 && !fit_host_again(host_tries, fr, host_prev_free, uint32_t(std::max(0L, opt.kv_host)), ctx))
                    break;
                if (!loaded || !token_b) {  // out of memory: measure once with all KV on host
                    if (!measured) { opt.kv_host = ctx; measuring = measured = true; what = "all KV to host memory (measuring)"; applied = true; }
                    break;
                }
                if (const uint32_t h = fit_host_tokens(ctx, uint32_t(std::max(0L, opt.kv_host)), fr, margin, token_b, 1024, measured);
                    h > uint32_t(std::max(0L, opt.kv_host))) {
                    opt.kv_host = h;
                    measuring = h == ctx;
                    measured |= measuring;
                    what = measuring ? "all KV to host memory (measuring)" : "KV of the last " + std::to_string(h) + " positions to host memory";
                    applied = true;
                    if (host_tries++ > 0 && step == 7) step = 6;  // stay on the refine step (the loop increments to 7)
                    host_prev_free = fr;
                }
                break;
            default:
                if (ctx / 2 < opt.fit_min_ctx) {  // smallest setting: start with a warning if 256 MB are free, else give up
                    if (fr < (size_t(256) << 20))
                        throw std::runtime_error("model does not fit in device memory (even at context " + std::to_string(ctx) + ")");
                    std::fprintf(stderr, "tqllm: fit: WARNING only %.2f GB free at the smallest setting; long prompts may "
                                         "fail (-fitt lowers the margin)\n", double(fr) / 1e9);
                    return finish(make_qwen35_engine(path, tok, ctx, slots, opt));
                }
                if (loaded) ctx_short = ctx, free_short = fr;
                if (host_auto) opt.kv_host = -1, measuring = false, host_tries = 0;  // host memory did not make room: weights and buffers are too big
                ctx /= 2;
                what = "context " + std::to_string(ctx);
                applied = true;
            }
        }
        std::fprintf(stderr, "tqllm: fit: %.2f GB free (< %.2f GB margin): %s\n", fr ? double(fr) / 1e9 : 0.0,
                     double(margin) / 1e9, what.c_str());
    }
}
}  // namespace

std::unique_ptr<Engine> load_fitted(const std::string & path, const std::string & tok, uint32_t ctx, int slots, EngineOptions opt) {
    g_fit_note.clear();
    auto e = load_fitted_impl(path, tok, ctx, slots, std::move(opt));
    if (e) e->note = g_fit_note;
    return e;
}

}  // namespace tqllm
