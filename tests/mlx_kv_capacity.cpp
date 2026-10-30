// KV capacity: with a reservation the cache is allocated once for the whole request (no regrowth), without one it grows
// x1.5; never past the context, always in whole segments.
#include "kv_capacity.hpp"
#include <cstdio>

using tqllm::mlx_backend::kv_capacity;
int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {std::printf("%s %s\n", ok ? "PASS" : "FAIL", what); failures += !ok;};
    // 246K-token prompt in 8192-row prefill chunks, then 256 tokens in 16-row verify steps, 1024-key segments
    const int context = 270000, seg = 1024, prompt = 245595, out = 256, reserve = prompt + out + 16;
    int cap = 0, grows = 0, peak_pair = 0;
    for (int pos = 0; pos < prompt + out;) {
        const int m = pos < prompt ? std::min(8192, prompt - pos) : 16, next = kv_capacity(cap, pos + m, reserve, context, seg);
        if (next != cap) {++grows; peak_pair = std::max(peak_pair, cap + next); cap = next;}
        pos += m;
    }
    check(grows == 1, "reserved request: one allocation for the whole request");
    check(cap >= reserve && cap % seg == 0 && cap <= context, "reserved capacity covers the request, whole segments, within the context");
    check(peak_pair <= cap, "reserved request: never holds two caches at once");
    cap = 0; grows = 0; peak_pair = 0;
    for (int pos = 0; pos < prompt + out;) {
        const int m = pos < prompt ? std::min(8192, prompt - pos) : 16, next = kv_capacity(cap, pos + m, 0, context, seg);
        if (next != cap) {++grows; peak_pair = std::max(peak_pair, cap + next); cap = next;}
        pos += m;
    }
    check(grows > 1 && peak_pair > cap, "no reservation: geometric growth (the old path, two caches during a regrowth)");
    check(kv_capacity(0, 100, 0, 270000, 1024) == 1024, "short request without reservation: one segment");
    check(kv_capacity(0, 300000, 0, 270000, 1024) == 270336, "capacity capped at the context (whole segments)");
    std::printf(failures ? "KV_CAPACITY_FAIL\n" : "KV_CAPACITY_PASS\n");
    return failures ? 1 : 0;
}
