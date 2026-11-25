#pragma once
// Prefill chunk length (tokens per GEMM pass). The context rule keeps 24 GB cards within memory at long contexts;
// 24 GB cards with >= 2 GB left after the device-resident KV take at least 1024, devices with >= 16 GB left 4096 at any
// context, >= 32 GB (GB10, 128 GB unified) 8192.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "device_free.hpp"

namespace tqllm {
inline int prefill_chunk(int64_t context, size_t total_bytes, size_t free_bytes, size_t kv_device_bytes) {
    int n = context <= 131072 ? 2048 : context <= 196608 ? 1024 : 512;  // 256K + 2048 would pass 24 GB
    if (context <= 65536 && total_bytes >= (size_t(20) << 30)) n = 4096;  // 27B on the 3090: +4-5%, peak +1 GB at 32K
    // GB10 TQ_XS prefill, 4096 -> 8192: 8K 1170 -> 1208, 32K 1068 -> 1136, 128K 818 -> 863, 262K 629 -> 659 tok/s
    const size_t room = free_bytes > kv_device_bytes ? free_bytes - kv_device_bytes : 0;
    // 24 GB cards with >= 2 GB left: 1024 at any context (not tried on 16 GB cards). RTX 3090, 27B TQT, 256K context,
    // q8/q4 KV, embeddings and head in host memory (2.51 GB left): 250K-token prefill 590-643 -> 966 tok/s (every chunk
    // re-reads the whole prefix and dequantises the weights once); 2048 runs out of memory there.
    if (total_bytes >= (size_t(20) << 30) && room >= (size_t(2) << 30)) n = std::max(n, 1024);
    if (room >= (size_t(16) << 30)) n = std::max(n, 4096);
    if (room >= (size_t(32) << 30)) n = std::max(n, 8192);
    return n;
}

// Where a prompt keeps its state for the next turn (0: nowhere): on the chunk grid a cold prefill uses (multiples of
// `chunk` from 0), at least `tail` tokens before the end (the next turn's template re-renders the end), after `from` (the
// position this prefill resumed at). A later turn that resumes there prefills exactly the chunks a cold prefill would,
// so its reply equals the reply with the cache off (the kernels pick their path by chunk width).
// Chunks restart at every message start (starts: sorted positions of the chat template's message-start token, > 0):
// a prefill's chunks are a property of its prompt, the same for a cold prefill and for one resumed at a message start.
// The end of the chunk that begins at t.
inline uint32_t chunk_end(const std::vector<uint32_t> & starts, uint32_t t, uint32_t prompt, uint32_t chunk) {
    const auto next = std::upper_bound(starts.begin(), starts.end(), t);
    const uint32_t seg = next == starts.begin() ? 0 : *(next - 1);
    uint32_t end = seg + ((t - seg) / chunk + 1) * chunk;
    if (next != starts.end()) end = std::min(end, *next);
    return std::min(end, prompt);
}
// Where a chat prompt keeps its state for the next turn: the start of its last message (the generation prompt), which
// the next turn's prompt shares and where every prefill of it starts a chunk; 0 when that is not past `from`. Prompts
// without message starts keep the grid rule (kept_state_at).
inline uint32_t kept_state_msg(const std::vector<uint32_t> & starts, uint32_t prompt, uint32_t from) {
    const auto it = std::lower_bound(starts.begin(), starts.end(), prompt);
    if (it == starts.begin()) return 0;
    const uint32_t last = *(it - 1);
    return last > from ? last : 0;
}
// A second kept state at the end of the first message when that message (the system prompt) is long: requests that share
// it but not the user message resume there instead of reading it again. 0: no message start, the first message is
// shorter than `min_tokens`, or the prefill resumes at or past it, or it is the last-message state already.
inline uint32_t kept_state_sys(const std::vector<uint32_t> & starts, uint32_t prompt, uint32_t from, uint32_t snap, uint32_t min_tokens) {
    if (starts.empty() || starts[0] >= prompt) return 0;
    const uint32_t end = starts[0];
    return end >= min_tokens && end > from && end != snap ? end : 0;
}
// A third kept state at the start of the first message after the prompt's last user message (the turn anchor). Chat
// templates that drop the reasoning of earlier assistant messages once a new user message arrives (Qwen3) re-render
// every message from there on, so the next user turn shares the prompt only up to that point and the last-message state
// is past it (issue 82: an agent's every user turn read 120K tokens again). 0: no user message, or nothing follows it.
// tokens[s + 1] is the role name of the message that starts at s; a user message whose fourth token is `result_tok`
// ("<|im_start|>user\n<tool_response>": Qwen templates send tool results as user messages) is a tool result, not a turn.
inline uint32_t kept_state_turn(const std::vector<uint32_t> & starts, const uint32_t * tokens, uint32_t prompt, uint32_t user_tok,
                                uint32_t result_tok) {
    for (size_t i = starts.size(); i-- > 0;) {
        if (starts[i] + 1 >= prompt || tokens[starts[i] + 1] != user_tok) continue;
        if (starts[i] + 3 < prompt && tokens[starts[i] + 3] == result_tok) continue;
        return i + 1 < starts.size() && starts[i + 1] < prompt ? starts[i + 1] : 0;
    }
    return 0;
}
// The positions of a slot's kept states: the last message start, the system-message end and the turn anchor.
struct KeptStates { uint32_t snap = 0, sys = 0, anchor = 0; };
// The kept state a prompt resumes at: the longest one it shares (shares(n): the prompt starts with the n tokens the
// state was built from and is longer than n); 0 when none.
template <class Shares> uint32_t kept_resume(const KeptStates & k, Shares shares) {
    uint32_t best = 0;
    for (uint32_t n : {k.snap, k.anchor, k.sys}) if (n > best && shares(n)) best = n;
    return best;
}
// The turn anchor after a prefill that resumed at `from` (kept states past `from` already dropped), for a prompt whose
// anchor is at `a` and whose last-message state goes to `snap_new`: a state held at `a` stays (the anchor itself, or an
// earlier turn's last-message state, moved over: *move_snap); past `from` this prefill saves it, unless it is the
// last-message state itself (the next turn moves that one over); otherwise the old anchor while it is still shared.
inline uint32_t kept_anchor_next(const KeptStates & k, uint32_t from, uint32_t a, uint32_t snap_new, bool * move_snap) {
    *move_snap = false;
    if (a && a <= from && a == k.anchor) return a;
    if (a && a <= from && a == k.snap) { *move_snap = true; return a; }
    if (a > from && a != snap_new) return a;
    return k.anchor <= from ? k.anchor : 0;
}
inline uint32_t kept_state_at(uint32_t prompt, uint32_t from, uint32_t chunk, uint32_t tail) {
    if (chunk == 0 || prompt < tail) return 0;
    const uint32_t grid = (prompt - tail) / chunk * chunk;
    return grid > from ? grid : 0;
}
}  // namespace tqllm
