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
}  // namespace tqserve
