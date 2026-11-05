#pragma once
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
namespace tqllm::mlx_backend {
inline size_t wired_bytes(std::string_view text,size_t recommended) {
    uint64_t mib=0;auto r=std::from_chars(text.data(),text.data()+text.size(),mib);
    constexpr uint64_t unit=1024*1024;
    if(text.empty() || r.ec!=std::errc{} || r.ptr!=text.data()+text.size() ||
       mib>std::numeric_limits<size_t>::max()/unit || mib*unit>recommended)
        throw std::invalid_argument("invalid MLX wired MiB budget or exceeds device recommendation");
    return size_t(mib*unit);
}
// a request keeps resident what it holds after its prefill (weights, KV, drafter) plus the decode's working room
inline size_t request_wired_bytes(size_t active,size_t headroom,size_t recommended) {
    return std::min(active+headroom,recommended);
}
// after a request the KV and drafter state stay allocated only for a prompt that may extend this text (tail prefill
// reuse); otherwise an idle server holds the weights alone
inline bool keep_state_after_request(bool tail_reuse,bool exact_text) {return tail_reuse && exact_text;}
// a thinking budget's two phases are one request: phase 1 (continuation 1) keeps its exact text's state, and only phase 2
// (continuation 2) may prefill its tail on it; any other request starts from a fresh prefill unless tail reuse is on
inline bool keep_state_for(bool tail_reuse,int continuation,bool exact_text) {return keep_state_after_request(tail_reuse || continuation==1,exact_text);}
inline bool may_reuse_held(bool tail_reuse,int continuation,bool held_for_continuation) {
    return tail_reuse || (continuation==2 && held_for_continuation);
}
}
