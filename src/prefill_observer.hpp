#pragma once
#include <cstdint>
#include <functional>
#include <utility>

namespace tqllm {
using PrefillObserver = std::function<void(uint32_t, uint32_t, uint32_t)>;
inline thread_local PrefillObserver prefill_observer;
// Scoped to the thread performing prefill, including the scheduler's engine thread.
class PrefillObservation {
    PrefillObserver previous_;
public:
    explicit PrefillObservation(PrefillObserver value) : previous_(std::move(prefill_observer)) { prefill_observer=std::move(value); }
    ~PrefillObservation() { prefill_observer=std::move(previous_); }
    PrefillObservation(const PrefillObservation &)=delete;
    PrefillObservation & operator=(const PrefillObservation &)=delete;
};
}
