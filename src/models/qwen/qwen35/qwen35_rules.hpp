#pragma once
// Pure rules of the qwen35 engine (no CUDA), tested in tests/qwen35_rules.cpp.
namespace tqllm::qwen35 {
// Columns one batched verify step may hold across slots (drafts per slot shrink as slots grow; the greedy output does not
// depend on the draft length). GA10x GeForce: 16 (3090 c=4: 16 -> 283 / 198, 24 -> 281 / 173, 32 -> 249 / 138 tok/s).
// GB10 (integrated Blackwell): 24 (11-06, TQ4_SX + DFlash2-TQ3, 8 requests total: full width 104.5, 24 -> 135.1, 16 -> 128.0,
// 12 -> 127.0 tok/s; 4 requests 116.7 / 127.9 / 128.9 / 115.6; 1-2 requests unchanged): the verify step grew ~4 ms per column.
// Other Blackwell cards (5060 Ti): the full width, as measured there.
inline int verify_column_budget(int major, bool integrated, int full) {
    if (major < 10) return full < 16 ? full : 16;
    if (integrated) return full < 24 ? full : 24;
    return full;
}
}  // namespace tqllm::qwen35
