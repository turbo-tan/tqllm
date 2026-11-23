// Usable device memory: on a unified-memory GPU (GB10) cudaMemGetInfo counts the page cache as used, so free memory
// comes from /proc/meminfo MemAvailable (capped at the device total); discrete cards keep cudaMemGetInfo's figure.
// Case from .44 on 2026-10-18: after a 179 GB copy, cudaMemGetInfo read 6681 MB free, MemAvailable 55.8 GB; the
// Flash-Next server refused its 262K context (needs 9735 MB) although d1fb0d3 served it.
#include "host_expert_tier.hpp"
#include <cstring>
#include <vector>
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
    // a 64 GB scope: the limit less anon + kernel + sock; file cache counts as free; no limit = SIZE_MAX (MemAvailable decides)
    const std::string stat = "anon 10737418240\nfile 32212254720\nkernel 1073741824\nkernel_stack 1000\nsock 0\nshmem 0\n";
    // pinned host memory (cudaMallocHost) is shmem, inside file: it is held, not reclaimable (B2 on .188: 64.4 GB shmem -> OOM kill)
    const std::string pinned = "anon 2354126848\nfile 66095411200\nkernel 231079936\nsock 0\nshmem 65935290368\n";
    expect(tqllm::cgroup_available("68719476736\n", pinned), size_t(198979584), "pinned shmem counts as held (190 MB left)");
    expect(tqllm::cgroup_available("68719476736\n", stat), 68719476736ull - 10737418240ull - 1073741824ull, "64 GB scope: limit - anon - kernel");
    expect(tqllm::cgroup_available("max\n", stat), SIZE_MAX, "no limit");
    expect(tqllm::cgroup_available("8589934592\n", stat), 0, "over the limit: nothing left");
    expect(tqllm::cgroup_available("68719476736\n", ""), 68719476736ull, "no stat: the limit");
    // B2 on .188 (11-20): the host tier's elastic chunks in a 64 GB scope. Every chunk attached is pinned host memory,
    // i.e. cgroup shmem; with shmem counted as free the fit kept attaching until the scope OOM-killed tqserve. Scaled: a
    // 10,000-byte limit, the base tier 6,000 bytes of shmem, 500 anon, four 1,000-byte chunks, 2,500 kept for decode.
    {
        const std::vector<size_t> sizes{100};
        std::vector<char> base(1000); std::vector<std::vector<char>> mem;
        tqllm::HostExpertTier tier(1, 64, sizes, base.data(), base.size(), [](int, int e, char * d) { std::memset(d, e, 100); }, 1, 1000, 4);
        uint64_t shmem = 6000;
        const auto available = [&] {
            const std::string st = "anon 500\nfile " + std::to_string(shmem + 3000) + "\nkernel 0\nsock 0\nshmem " + std::to_string(shmem) + "\n";
            return uint64_t(std::min<size_t>(size_t(100000), tqllm::cgroup_available("10000\n", st)));  // MemAvailable: the whole box
        };
        const auto alloc = [&](size_t b) -> char * { shmem += b; mem.emplace_back(b); return mem.back().data(); };
        const auto release = [&](char *) { shmem -= 1000; };
        const int decode = tqllm::fit_tier_chunks(tier, 2500, available, alloc, release);
        if (decode != 1 || available() < 2500) { std::fprintf(stderr, "FAIL 64 GB scope: %d chunks attached, %llu left (want 1, >= 2500)\n", decode, (unsigned long long) available()); ++failures; }
        const int long_prompt = tqllm::fit_tier_chunks(tier, 3500, available, alloc, release);
        if (long_prompt != 0 || shmem != 6000) { std::fprintf(stderr, "FAIL 64 GB scope, long prompt: %d chunks, shmem %llu (want 0, 6000)\n", long_prompt, (unsigned long long) shmem); ++failures; }
    }
    if (failures) return 1;
    std::puts("device_free: ok");
    return 0;
}
