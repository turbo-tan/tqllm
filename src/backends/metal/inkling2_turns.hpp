#pragma once
// Inkling-2 turn reuse on Metal (engine_inkling2_mlx.cpp, inkling2-mlx-check --turns): the prompt chunks of a prefill.
#include "inkling2_mlx.hpp"
#include "turn_reuse.hpp"
#include <span>
#include <utility>
#include <vector>

namespace tqllm::mlx_backend {
// Prompt chunks of a prefill from `from`: `chunk` rows at most, the grid restarting at each message start (a role token)
// at or past kTurnMin (turn_reuse.hpp), so a prefill's chunks are a property of the prompt's prefix: the same for a cold
// prefill and for one resumed at a kept message start, and the row-dependent prefill kernels give the same numbers.
inline std::vector<std::pair<size_t, size_t>> inkling2_prefill_chunks(std::span<const uint32_t> prompt, size_t from, size_t chunk) {
    std::vector<uint32_t> starts;
    for (size_t p = kTurnMin; p < prompt.size(); ++p) if (inkling2_message_start(prompt[p])) starts.push_back(uint32_t(p));
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t off = from; off < prompt.size();) {
        const size_t end = turn_chunk_end(starts, uint32_t(off), uint32_t(prompt.size()), uint32_t(chunk));
        out.emplace_back(off, end);
        off = end;
    }
    return out;
}
}  // namespace tqllm::mlx_backend
