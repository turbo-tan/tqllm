#pragma once
// Turn reuse on Metal: the next turn of a conversation prefills only its new messages, with the reply a cold prefill gives.
// A chat client re-renders the last answer without its reasoning, so the next prompt shares the previous one only up to
// its generation prompt. The state is kept there, and prefill chunks restart at every message start: a prefill's chunks
// are then a property of the prompt's prefix, the same for a cold prefill and for one resumed at a kept state (the
// kernels pick their path by chunk rows, so equal chunks give equal numbers). An answer's message is a chunk of its own,
// so the server can prefill it while idle, before the next request arrives (Engine::prefill_ahead). Same rule as the CUDA
// engine's (src/models/qwen/qwen35/prefill_chunk.hpp), with no starts below kTurnMin.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tqllm::mlx_backend {
// message starts before this position do not split the prefill: short prompts keep their single pass (a split adds a
// forward of the generation prompt's few rows to the first token)
constexpr uint32_t kTurnMin = 1024;

// positions p >= kTurnMin of the chat template's message-start token, ascending
template <class P> inline std::vector<uint32_t> message_starts(const P& prompt, uint32_t im_start) {
    std::vector<uint32_t> out;
    if (im_start == ~0u) return out;
    for (size_t p = kTurnMin; p < prompt.size(); ++p)
        if (uint32_t(prompt[p]) == im_start) out.push_back(uint32_t(p));
    return out;
}
// the end of the prefill chunk that begins at t: the chunk grid restarts at each start
inline uint32_t turn_chunk_end(const std::vector<uint32_t>& starts, uint32_t t, uint32_t prompt, uint32_t chunk) {
    const auto next = std::upper_bound(starts.begin(), starts.end(), t);
    const uint32_t seg = next == starts.begin() ? 0 : *(next - 1);
    uint32_t end = seg + ((t - seg) / chunk + 1) * chunk;
    if (next != starts.end()) end = std::min(end, *next);
    return std::min(end, prompt);
}
// where this prefill keeps its state for the next turn: its last message start (a chat prompt's generation prompt), when
// that is past `from`, the position the prefill began at; 0: nothing new to keep
inline uint32_t turn_kept_at(const std::vector<uint32_t>& starts, uint32_t from) {
    return !starts.empty() && starts.back() > from ? starts.back() : 0;
}
// a state kept at a message start of a prompt serves a new prompt that begins with `kept`: that prompt's tokens through
// the start's own token and the role after it (so the position is a message start of the new prompt too)
template <class A, class B> inline bool turn_shared(const A& kept, const B& prompt) {
    if (kept.empty() || prompt.size() < kept.size()) return false;
    for (size_t i = 0; i < kept.size(); ++i) if (uint32_t(kept[i]) != uint32_t(prompt[i])) return false;
    return true;
}
}  // namespace tqllm::mlx_backend
