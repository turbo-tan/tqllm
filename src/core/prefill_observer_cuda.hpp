#pragma once
#include "prefill_observer.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

namespace tqllm {
// Report completed work in stream order without synchronizing the inference thread.
inline void prefill_completed(cudaStream_t stream, uint32_t done, uint32_t total, uint32_t cached=0) {
    if (!prefill_observer) return;
    struct Event { PrefillObserver observer; uint32_t done, total, cached; };
    auto * event=new Event{prefill_observer,done,total,cached};
    const auto status=cudaLaunchHostFunc(stream,[](void * data) {
        std::unique_ptr<Event> e(static_cast<Event *>(data));
        try { e->observer(e->done,e->total,e->cached); } catch (...) {}
    },event);
    if (status!=cudaSuccess) { delete event; throw std::runtime_error("prefill progress callback enqueue failed"); }
}
}
