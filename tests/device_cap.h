// A fixed-size device for the tests that fill device memory on purpose (definitions in device_cap.cpp, host C++: nvcc
// defines a device-side cudaMalloc / cudaFree in every .cu file, so they cannot be defined there).
//
// Those tests took all the free memory and then asserted what the engine does in what is left. On a card that also drives a
// desktop that is not stable: with the device full, the compositor and browsers free and retake 16-55 MB every ~15 ms (an
// RTX 3090 under GNOME, cudaMemGetInfo polled every 2 ms). The engine took that memory as it appeared, so a bank that
// should have evicted another fit, a 64 KB tuning scratch got allocated, and an even split of the free memory between two
// banks came out a few slots off: mixed_expert_cache, expert_slot_tiers and tune_oom failed in 5-30 % of the runs.
//
// device_cap::limit(room) gives the process a device of (memory it holds now + room) bytes: cudaMalloc fails with
// cudaErrorMemoryAllocation past it and cudaMemGetInfo reports what is left of it, for the test and the engine alike (the
// engine is a static library: its calls resolve to these definitions). The memory under the limit is real device memory,
// so a full device is reached without taking the desktop's memory. Each allocation counts what the driver takes for it
// (a power of two up to 1 MiB, whole 2 MiB pages above), as the real device's fill-and-evict arithmetic did. Linux only
// (the real functions come from libcudart through dlsym); elsewhere limit() does nothing and the tests fill the real
// device as before.
#pragma once
#include <cstddef>

namespace device_cap {
// the device ends `room` bytes past what the process holds now (SIZE_MAX: no limit); does nothing off Linux
void limit(size_t room);
} // namespace device_cap
