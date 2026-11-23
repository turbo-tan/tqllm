#pragma once
// Admission when memory runs short (owner 11-06: "when memory is not enough it should queue instead of stop").
// A request whose prefill fails with "out of memory" while other requests are running goes back to the front of the queue
// and waits until one of them finishes; only a request that cannot run with nothing else running is an error.
#include <cstring>
#include <string>

namespace tqserve {
enum class OnOom { Requeue, Fail };
inline bool is_out_of_memory(const char * what) { return what && std::strstr(what, "out of memory") != nullptr; }
inline OnOom on_prefill_error(const char * what, int others_running) {
    return is_out_of_memory(what) && others_running > 0 ? OnOom::Requeue : OnOom::Fail;
}
// after a requeue, admission resumes once `finished` passes the count recorded at the requeue
inline bool admission_open(unsigned long long finished, unsigned long long wait_until) { return finished >= wait_until; }
// Shared prefixes: before a request's prefill, the slot whose kept state covers at least kShareGain more of its prompt
// than its own slot (or the park) lends a copy to the park, so requests that arrive together with one long system
// prompt read it once (GB10 27B, 8 x 32K shared prefix: TTFT 154 s vs TensorFold 13.5 s; each read the 32K again).
// reuse[o]: prompt tokens slot o would reuse; own: the request's slot (park included). -1: nothing worth sharing.
constexpr unsigned kShareGain = 4096;
template <class Reuse>
inline int share_source(int slots, int own_slot, unsigned own, Reuse reuse) {
    int from = -1; unsigned best = own;
    for (int o = 0; o < slots; ++o) {
        if (o == own_slot) continue;
        const unsigned r = reuse(o);
        if (r >= best + kShareGain) { best = r; from = o; }
    }
    return from;
}
}  // namespace tqserve
