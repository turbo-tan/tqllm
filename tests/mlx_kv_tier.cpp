// KV tier choice: short requests stay BF16; the j1140 249K request on a 48 GB Mac (recommended working set ~38 GB,
// weights + drafter ~17.4 GB in use) does not fit in BF16 and gets int8; a larger one or a smaller machine gets trellis.
#include "kv_tier.hpp"
#include <cstdio>

using namespace tqllm::mlx_backend;
int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {std::printf("%s %s\n", ok ? "PASS" : "FAIL", what); failures += !ok;};
    const size_t GB = size_t(1) << 30;
    const KvBudget mac48{size_t(17.4 * GB), size_t(38.0 * GB), size_t(5) * GB};
    check(choose_kv_tier(8192 + 272, 16, 4, 256, mac48) == KvTier::Bf16, "8K request on the 48 GB Mac: BF16");
    check(choose_kv_tier(121000 + 272, 16, 4, 256, mac48) == KvTier::Bf16, "121K request on the 48 GB Mac: BF16 (fits, j1140 ran without paging)");
    check(choose_kv_tier(249391 + 272, 16, 4, 256, mac48) == KvTier::Bf16, "249K request on an idle 48 GB Mac: BF16 (39.4 of the recommended 40.2 GB)");
    check(choose_kv_tier(1000000, 16, 4, 256, mac48) == KvTier::Trellis, "1M request on the 48 GB Mac: trellis");
    const KvBudget mac24{size_t(17.4 * GB), size_t(20.0 * GB), size_t(2) * GB};
    check(choose_kv_tier(32768, 16, 4, 256, mac24) == KvTier::Trellis || choose_kv_tier(32768, 16, 4, 256, mac24) == KvTier::Int8,
          "32K on a 24 GB machine: compressed");
    check(kv_row_bytes(KvTier::Trellis, 256) * 5 < kv_row_bytes(KvTier::Bf16, 256), "trellis row < 1/5 of BF16");
    // far-context drafter caches: kept while the KV format holds, dropped when they alone would force trellis
    const KvBudget mac48run{size_t(17.4 * GB), size_t(38.0 * GB), (size_t(4096) << 20) + size_t(8192) * 5120 * 40};
    const size_t far = size_t(5.1 * GB) / 248000;  // ~5 GB of BF16 drafter K / V at 248K (j1158)
    const auto p248 = plan_kv(248000 + 272, 16, 4, 256, mac48run, far);
    check(p248.tier == KvTier::Int8 && p248.far, "248K + far caches, 38 GB working set: int8 with the far caches (37.7 GB)");
    const KvBudget mac48tight{size_t(17.4 * GB), size_t(36.0 * GB), mac48run.transient};
    const auto p248t = plan_kv(248000 + 272, 16, 4, 256, mac48tight, far);
    check(p248t.tier == KvTier::Int8 && !p248t.far, "248K + far caches, 36 GB working set: int8 without them (not trellis with them)");
    const auto p64 = plan_kv(65536 + 272, 16, 4, 256, mac48run, far);
    check(p64.tier == KvTier::Bf16 && p64.far, "64K, memory alone: BF16 with far caches");
    check(speed_kv_tier(8192 + 272) == KvTier::Bf16 && speed_kv_tier(32767) == KvTier::Bf16, "under 32K: BF16 first");
    check(speed_kv_tier(32768) == KvTier::Int8 && speed_kv_tier(249391 + 272) == KvTier::Int8, "32K and up: int8 first");
    check(request_kv_first(300, 16384) == KvTier::Bf16, "short chat, max_tokens omitted (server default 16384): BF16");
    check(request_kv_first(20000, 16384) == KvTier::Bf16 && request_kv_first(32766, 256) == KvTier::Int8, "the prompt decides: 20K + default answer BF16, 32K int8");
    const auto p64s = plan_kv(65536 + 272, 16, 4, 256, mac48run, far, 0, speed_kv_tier(65536 + 272));
    check(p64s.tier == KvTier::Int8 && p64s.far, "64K: int8 with far caches (j1224 44.4 vs BF16 42.1 tok/s)");
    const auto p8s = plan_kv(8192 + 272, 16, 4, 256, mac48run, far, 0, speed_kv_tier(8192 + 272));
    check(p8s.tier == KvTier::Bf16 && p8s.far, "8K: BF16 with far caches");
    const auto p0 = plan_kv(65536 + 272, 16, 4, 256, mac48run, 0);
    check(p0.tier == KvTier::Bf16 && !p0.far, "no far caches: plain tier choice");
    // what the system can hand us: j1168 vm_stat on the loaded 48 GB Mac (16 KiB pages): free 72877, inactive 1058466,
    // speculative 2277, purgeable 2218, file backed 405410 -> 7.9 GB; free + inactive + ... counted 18.6 GB and let
    // BF16 121K in three times (j1164 21.0, j1166 25.9, j1168 22.5 tok/s vs int8 32.2)
    const size_t page = 16384, avail = available_memory(72877, 1058466, 2277, 2218, 405410) * page;
    check(avail > size_t(7.8e9) && avail < size_t(8.0e9), "loaded Mac: 7.9 GB available (free + speculative + purgeable + file-backed inactive)");
    check(available_memory(100, 50, 10, 5, 400) == 165 && available_memory(100, 500, 10, 5, 40) == 155, "inactive counts only up to the file-backed pages");
    check(usable_working_set(size_t(40.2e9), size_t(18e9), size_t(30e9)) == size_t(40.2e9), "idle machine: the recommended working set");
    check(usable_working_set(size_t(40.2e9), size_t(18e9), 0) == size_t(40.2e9), "unknown free memory: the recommended working set");
    const size_t lean = (size_t(512) << 20) + size_t(8192) * 5120 * 40;
    // 121K on the loaded Mac, 22.3 GB ours: BF16 does not fit even lean (32.4 > 30.2), int8 fits with the lean prefill
    const KvBudget load121{size_t(22.3e9), usable_working_set(size_t(40.2e9), size_t(22.3e9), avail), mac48run.transient};
    const size_t far8 = 5 * 8 * (128 * 2 + 4 + 128 * 4 / 16);
    const auto p121 = plan_kv(120873 + 272, 16, 4, 256, load121, far8, lean);
    check(p121.tier == KvTier::Int8 && p121.lean, "121K on the loaded Mac: int8 with the lean prefill, never BF16");
    // 248K on the loaded Mac, 17.9 GB ours (usable 25.8 GB): int8 does not fit (29.3 lean); trellis without far caches.
    // 20 tok/s at 256K needs a machine that is not holding 28 GB of other apps' pages
    const KvBudget load248{size_t(17.9e9), usable_working_set(size_t(40.2e9), size_t(17.9e9), avail), mac48run.transient};
    const auto p248l = plan_kv(248000 + 272, 16, 4, 256, load248, far8, lean);
    check(p248l.tier == KvTier::Trellis && !p248l.far && p248l.lean, "248K on the loaded Mac: trellis, lean prefill, no far caches");
    // 248K on an idle 48 GB Mac (25 GB available -> the recommended 40.2 GB): int8 with the int8 far caches, no lean
    const KvBudget idle248{size_t(17.9e9), usable_working_set(size_t(40.2e9), size_t(17.9e9), size_t(25e9)), mac48run.transient};
    const auto p248i = plan_kv(248000 + 272, 16, 4, 256, idle248, far8, lean);
    check(p248i.tier == KvTier::Int8 && p248i.far && !p248i.lean, "248K on an idle 48 GB Mac: int8 with the far caches, full prefill cache");
    const auto short_plan = plan_kv(8192 + 272, 16, 4, 256, load121, far8, lean);
    check(short_plan.tier == KvTier::Bf16 && !short_plan.lean, "8K on the loaded Mac: BF16, full prefill cache");
    std::printf(failures ? "KV_TIER_FAIL\n" : "KV_TIER_PASS\n");
    return failures ? 1 : 0;
}
