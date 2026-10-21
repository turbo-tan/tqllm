#pragma once
// Flash-Next decode decisions that change exactness or memory, as pure functions (no CUDA) so tests/decode_rules.cpp can
// check them on any machine.
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace tqllm::flashnext {

// Expert path of one trunk layer call: true = waves (several prepares, grouped per wave), false = one split prepare.
// A prompt chunk (spec off, more than one token) always takes waves. A verify block takes the split path whenever its
// distinct experts fit the layer's slots: the waves path sums the down projection in another order, so choosing it by
// the assignment count made a multi-sequence block (np 4: 4 x 4 rows = 160 assignments) differ from np 1 (F39).
// old_rule = the pre-F39 choice (assignments vs slots), kept behind TQFN_VERIFY_WAVES for A/B.
inline bool layer_takes_waves(const std::vector<int> & routed, int n, int topk, int slots, bool spec, bool old_rule = false) {
    if (!(n * topk > slots || (!spec && n > 1))) return false;
    if (!spec || n <= 1 || old_rule) return true;
    std::vector<int> u(routed); std::sort(u.begin(), u.end());
    return std::unique(u.begin(), u.end()) - u.begin() > slots;
}

// Device bytes slots=auto keeps free beside the expert banks. One sequence keeps only the decode reserve and 256 MB of
// working room: a long prompt borrows banks for its attention state and the refill after it stops short of the last
// 512 MB (keeping its full-context state free cost 3090 chats ~16 slots per layer, 85.9 vs 95.7 tok/s at 140 vs 157).
// Several sequences (--np) each keep their own full-context attention / recurrent / MTP state and working room: sized
// for one, np 4 at 32K ran out of memory (F38).
inline size_t auto_keep_bytes(size_t sequences, size_t state, size_t reserve) {
    if (sequences <= 1) return reserve + (size_t(256) << 20);
    return sequences * state + reserve + sequences * (size_t(256) << 20);
}
// A layer's slot floor (tier A) from its target: scaled like the slot total (tier_a of tier_b per layer on average), and
// never below 64: a 6-row verify block routes to at most 60 distinct experts, so layer_takes_waves keeps the split path at
// every bank size (F39) and the text does not depend on how much a long prompt took.
inline int slot_floor(int target, int tier_a, int tier_b) {
    if (tier_b <= 0 || target <= 0) return target;
    return std::min(target, std::max(64, int(int64_t(target) * tier_a / tier_b)));
}

// Adaptive draft length: draft again only while the last draft's top probability reaches tau (tau <= 0: fixed count).
inline bool keep_drafting(float tau, float last_draft_probability) { return !(tau > 0.f && last_draft_probability < tau); }

}  // namespace tqllm::flashnext
