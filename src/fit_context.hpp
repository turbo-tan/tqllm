#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tqllm {
// Largest context (multiple of `align`) that keeps `margin` bytes free, from two loads: ctx_short left free_short
// (< margin) and ctx_ok left free_ok (>= margin). Free memory falls linearly with the context (the KV cache), so the
// answer lies between them; returns ctx_ok when the measurements leave no room for more.
inline uint32_t fit_context(uint32_t ctx_short, size_t free_short, uint32_t ctx_ok, size_t free_ok, size_t margin,
                            uint32_t align = 4096) {
    if (!align || ctx_short <= ctx_ok || free_ok <= free_short || free_ok < margin) return ctx_ok;
    const double per_tok = double(free_ok - free_short) / double(ctx_short - ctx_ok);
    const double extra = double(free_ok - margin) / per_tok;
    uint64_t c = uint64_t(ctx_ok + extra);
    c = c / align * align;
    if (c >= ctx_short) c = (uint64_t(ctx_short) - 1) / align * align;
    return c > ctx_ok ? uint32_t(c) : ctx_ok;
}

// KV positions per sequence to keep in host memory so a load that left free_b bytes (with `host` positions already on
// host) leaves at least `margin`: every position moved frees token_b device bytes (all attention layers and slots).
// Rounded up to `align`; at most ctx. Returns host unchanged when nothing needs to move.
// A reading under kFreeSaturated says only "over budget" (WDDM reports ~0 however far over), so it is no measure of the
// shortfall: the answer is then ctx (all KV to host, a load that reads true free memory to size the split from).
constexpr size_t kFreeSaturated = size_t(32) << 20;
// measured: the split already came from a true reading, so a saturated one now means a small miss: count it as 0 free
inline uint32_t fit_host_tokens(uint32_t ctx, uint32_t host, size_t free_b, size_t margin, size_t token_b,
                                uint32_t align = 1024, bool measured = false) {
    if (free_b >= margin || !token_b || !align) return host;
    if (free_b < kFreeSaturated) {
        if (!measured) return ctx;
        free_b = 0;
    }
    const uint64_t need = (uint64_t(margin - free_b) + token_b - 1) / token_b;
    const uint64_t h = (uint64_t(host) + need + align - 1) / align * align;
    return h >= ctx ? ctx : uint32_t(h);
}
// Another KV-to-host refinement before the fit gives up context: `tries` host steps so far, free memory after the last
// (`free_b`) and before it (`prev_free`), `host` positions already in host memory of `ctx`.
// KV positions to keep in host memory after a load that left free_b with `host` positions there: the spare above
// `margin` brings whole `align` blocks back to the device (token_b device bytes each). Returns host when none fit.
inline uint32_t fit_host_return(uint32_t host, size_t free_b, size_t margin, size_t token_b, uint32_t align = 1024) {
    if (!host || !token_b || !align || free_b <= margin) return host;
    const uint64_t back = (uint64_t(free_b - margin) / token_b) / align * align;
    return back >= host ? 0 : host - uint32_t(back);
}
inline bool fit_host_again(int tries, size_t free_b, size_t prev_free, uint32_t host, uint32_t ctx) {
    if (host >= ctx || tries >= 6) return false;  // all KV already on host, or bounded reloads
    return tries < 2 || free_b > prev_free;       // first refinement always; then only while each step frees memory
}
// Free bytes the memory fit must leave after loading: -fitt MiB (default 768, 256 under WSL), never below 64 MiB - a WSL /
// Windows card past its VRAM spills into shared memory and reports ~0 free, so "0 free" must not pass; 0 when the fit is off.
inline size_t fit_margin(bool fit, long target_mb, bool wsl) {
    const long mb = target_mb < 0 ? (wsl ? 256 : 768) : target_mb;
    return fit ? size_t(mb < 64 ? 64 : mb) << 20 : 0;
}
// Free device memory the fit judges a load by. Linux: the driver's figure. WSL / Windows (WDDM): the driver reads ~0 near
// the budget whether or not the card has spilled into shared memory, so the free figure is our own accounting instead:
// total minus what was in use before loading (context, desktop, other processes) minus the engine's allocations.
inline size_t fit_free(bool wsl, size_t driver_free, size_t total, size_t used_before, size_t engine_bytes) {
    if (!wsl) return driver_free;
    const size_t slack = size_t(192) << 20;  // cuBLAS / cuBLASLt workspaces, CUDA graphs, allocator rounding
    const size_t used = used_before + engine_bytes + slack;
    return used >= total ? 0 : total - used;
}
// Parallel slots a load can serve: each needs at least min_ctx positions of the context and slot_b bytes of the memory
// left free after loading (its batched-verify / draft graphs, ~12 MB each), on top of `margin`. Returns np when they fit,
// else the largest count that does (at least 1: one slot is the plain fit). tqserve refuses to start rather than fail
// requests mid-run.
inline uint32_t fit_slots(uint32_t np, uint32_t context, uint32_t min_ctx, size_t free_b, size_t margin, size_t slot_b) {
    if (np <= 1) return np;
    const uint32_t by_ctx = min_ctx ? context / min_ctx : np;
    uint32_t by_mem = np;
    if (slot_b) by_mem = free_b <= margin ? 0 : uint32_t(std::min<uint64_t>(np, (free_b - margin) / slot_b));
    return std::max(1u, std::min({np, by_ctx, by_mem}));
}
}  // namespace tqllm
