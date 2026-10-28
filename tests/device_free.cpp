// Usable device memory: on a unified-memory GPU (GB10) cudaMemGetInfo counts the page cache as used, so free memory
// comes from /proc/meminfo MemAvailable (capped at the device total); discrete cards keep cudaMemGetInfo's figure.
// Case from .44 on 2026-10-18: after a 179 GB copy, cudaMemGetInfo read 6681 MB free, MemAvailable 55.8 GB; the
// Flash-Next server refused its 262K context (needs 9735 MB) although d1fb0d3 served it.
#include "device_free.hpp"

#include <cstdio>
#include <string>

namespace {
int failures = 0;
constexpr size_t MiB = size_t(1) << 20, GiB = size_t(1) << 30;
void expect(size_t got, size_t want, const char * what) {
    if (got != want) { std::fprintf(stderr, "FAIL %s: %zu MiB, want %zu MiB\n", what, got / MiB, want / MiB); ++failures; }
}
}

int main() {
    using tqllm::usable_free_bytes;
    const std::string gb10 = "MemTotal:       127535340 kB\nMemFree:         7144316 kB\nMemAvailable:   55831292 kB\nCached:         51727224 kB\n";
    const size_t avail = size_t(55831292) * 1024, total = size_t(124546) * MiB, cuda_free = size_t(6681) * MiB;
    expect(usable_free_bytes(cuda_free, total, true, gb10), avail, "GB10 page cache counts as free");
    if (usable_free_bytes(cuda_free, total, true, gb10) < size_t(9735) * MiB) { std::fprintf(stderr, "FAIL GB10 262K context still refused\n"); ++failures; }
    // discrete card: the host's page cache is not device memory
    expect(usable_free_bytes(size_t(11) * GiB, 24 * GiB, false, gb10), size_t(11) * GiB, "3090 keeps cudaMemGetInfo");
    // unified, MemAvailable above the device total (driver reserve): capped at the total
    expect(usable_free_bytes(cuda_free, 40 * GiB, true, gb10), 40 * GiB, "capped at device total");
    // unified, no /proc/meminfo text (or no MemAvailable line): cudaMemGetInfo
    expect(usable_free_bytes(cuda_free, total, true, ""), cuda_free, "no meminfo");
    expect(usable_free_bytes(cuda_free, total, true, "MemTotal: 1 kB\n"), cuda_free, "no MemAvailable");
    // cudaMemGetInfo above MemAvailable (never lower the figure)
    expect(usable_free_bytes(60 * GiB, total, true, gb10), 60 * GiB, "cudaMemGetInfo larger");
    // host RAM: the cgroup limit caps physical memory (RunPod 5090 pod: 944 GB physical, memory.max 116999999488)
    using tqllm::host_ram_limit;
    expect(host_ram_limit(944 * GiB, "116999999488\n"), size_t(116999999488), "cgroup v2 limit below physical");
    expect(host_ram_limit(125 * GiB, "max\n"), 125 * GiB, "no cgroup limit");
    expect(host_ram_limit(125 * GiB, ""), 125 * GiB, "no cgroup file");
    expect(host_ram_limit(31 * GiB, "9223372036854771712\n"), 31 * GiB, "cgroup v1 unlimited");
    expect(host_ram_limit(125 * GiB, "x"), 125 * GiB, "unreadable limit");
    if (tqllm::cgroup_v2_path("0::/user.slice/run-x.scope\n") != "/user.slice/run-x.scope") { std::fprintf(stderr, "FAIL cgroup path\n"); ++failures; }
    if (tqllm::cgroup_v2_path("0::/\n") != "/") { std::fprintf(stderr, "FAIL cgroup root path\n"); ++failures; }
    if (!tqllm::cgroup_v2_path("12:memory:/x\n").empty()) { std::fprintf(stderr, "FAIL v1-only text\n"); ++failures; }
    if (failures) return 1;
    std::puts("device_free: ok");
    return 0;
}
