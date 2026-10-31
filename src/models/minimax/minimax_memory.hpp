#pragma once
#include <cstddef>
#include <string>

// Host memory a GB10 keeps free beside a MiniMax load; earlyoom on the box must trigger below this (10-24: 20 -> 4 -> 2 GiB,
// earlyoom -M 1 GiB: the EXL3 pack with its MXFP8 DSpark draft fits one GX10 only so)
inline constexpr size_t kMiniMaxHostReserve = size_t(2) << 30;

// One layer's KV cache: K and V for the local KV heads plus the indexer keys, as the engine allocates them (elem 2: FP16,
// 1: FP8 E4M3, chosen only when the requested context does not fit in FP16)
inline size_t minimax_kv_layer_bytes(size_t context, size_t kv_heads, size_t head_dim, size_t index_dim, size_t elem = 2) {
    return context * (2 * kv_heads * head_dim + index_dim) * elem;
}

// Tensors the engine copies into memory it owns; the token embedding stays in the mapped file (rows are read on the
// host per token), so its pages are page cache the kernel can drop and do not count against the load.
inline bool minimax_resident(const std::string & name) { return name != "token_embd.weight"; }

// KV element bytes that let every expert stack load beside `fixed` (reserves) in `free`: 2 (FP16) when it fits, 1 (FP8)
// when only that fits, 0 when neither (the hot / cold banks). FP8 only when asked for already or when FP16 does not fit.
inline size_t minimax_expert_kv_elem(size_t stack, size_t free, size_t fixed, size_t kv16, size_t kv8, bool kv8_now) {
    auto fits = [&](size_t kv) { return free > fixed + kv && stack <= free - fixed - kv; };
    if (!kv8_now && fits(kv16)) return 2;
    return fits(kv8) ? 1 : 0;
}
