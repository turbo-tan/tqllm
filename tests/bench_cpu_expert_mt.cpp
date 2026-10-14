// Multithread scaling bench: can the 0.15 ms/layer budget be met at all?
//
// WHY THIS EXISTS: the single-thread kernel is COMPUTE-bound (2.40 GB/s against a measured
// 25.71 GB/s single-thread read ceiling on this box). So the budget question is not about
// DRAM -- it is about how many cores the pool can use. .77 has 6 physical cores (12 threads);
// Claude asked for 15 threads, which this box does not have, so I measure the real scaling
// curve and extrapolate rather than assume linear.
//
// Per-layer miss work is gate + up + down. Sizes come from the real GGUF headers:
//   gate/up  640 x 2560  -> 640 * 80 * 18 = 921,600 B
//   down    2560 x 640   -> 2560 * 20 * 18 = 921,600 B
// so a gate+up+down triple is 3 x 0.92 MB = 2.76 MB of weight bytes per miss expert.
#include "../src/cpu_expert_matvec.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>   // std::function for the persistent worker pool
#include <chrono>
#include <algorithm>

using namespace tqllm::cpu;

static void pack_iq4nl(std::vector<uint8_t> &W, int rows, int cols, unsigned seed) {
    const int nb = cols / 32, row_bytes = nb * 18;
    W.assign((size_t)rows * row_bytes, 0);
    unsigned s = seed ? seed : 1u;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s; };
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            uint8_t *p = W.data() + (size_t)r * row_bytes + b * 18;
            p[0] = (uint8_t)(rnd() & 0xFF); p[1] = (uint8_t)(rnd() & 0xFF);
            for (int i = 0; i < 16; ++i) p[2 + i] = (uint8_t)rnd();
        }
}

struct Mat { std::vector<uint8_t> W; int rows, cols, row_bytes; };

int main() {
    // Real shapes (cols=2560 for gate/up, NOT the 2048 in the task description).
    const int GU_R = 640,  GU_C = 2560;   // gate / up
    const int DN_R = 2560, DN_C = 640;    // down
    const int gu_rb = (GU_C / 32) * 18, dn_rb = (DN_C / 32) * 18;

    const double MB_GU = (double)GU_R * gu_rb / 1e6;
    const double MB_DN = (double)DN_R * dn_rb / 1e6;
    printf("per-matrix bytes: gate/up %.4f MB, down %.4f MB\n", MB_GU, MB_DN);
    printf("one miss expert (gate+up+down) = %.4f MB\n\n", 2*MB_GU + MB_DN);

    // A work unit = one full miss expert (3 matvecs). The pool would run several experts
    // concurrently across cores, so model that: NEXPERTS experts, one per thread.
    const int NEXP = 24;
    std::vector<Mat> gu(NEXP), dn(NEXP);
    for (int i = 0; i < NEXP; ++i) {
        pack_iq4nl(gu[i].W, GU_R, GU_C, 11 + i * 7);
        gu[i].rows = GU_R; gu[i].cols = GU_C; gu[i].row_bytes = gu_rb;
        pack_iq4nl(dn[i].W, DN_R, DN_C, 91 + i * 13);
        dn[i].rows = DN_R; dn[i].cols = DN_C; dn[i].row_bytes = dn_rb;
    }
    const double MB_TOTAL = NEXP * (2*MB_GU + MB_DN);

    const int hw = (int)std::thread::hardware_concurrency();
    printf("hardware_concurrency = %d (6 physical cores, SMT2)\n\n", hw);
    printf("%-8s %-8s %10s %10s %10s %12s\n",
           "threads", "path", "ms/expert", "GB/s agg", "24exp ms", "vs 0.15ms");

    for (int nt : {1, 2, 3, 4, 6, 8, 12}) {
        if (nt > hw) continue;
        for (int use_int8 = 0; use_int8 <= 1; ++use_int8) {
            // Warm pass.
            for (int i = 0; i < NEXP; ++i) {
                std::vector<float> x(gu[i].cols, 0.5f), y(gu[i].rows);
                if (use_int8) iq4nl_expert_matvec_i16(gu[i].W.data(), gu[i].rows, gu[i].cols,
                                                      gu[i].row_bytes, x.data(), 1, y.data());
                else          iq4nl_expert_matvec_fp32(gu[i].W.data(), gu[i].rows, gu[i].cols,
                                                       gu[i].row_bytes, x.data(), 1, y.data());
            }
            const int reps = 5;
            auto t0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < reps; ++rep) {
                std::vector<std::thread> th;
                std::atomic<int> next{0};
                for (int t = 0; t < nt; ++t) th.emplace_back([&]() {
                    for (;;) {
                        int i = next.fetch_add(1);
                        if (i >= NEXP) break;
                        // gate -> up -> down, the real miss-expert sequence.
                        std::vector<float> xg(gu[i].cols, 0.5f), yg(gu[i].rows);
                        std::vector<float> xd(dn[i].cols, 0.5f), yd(dn[i].rows);
                        for (int k = 0; k < 2; ++k) {   // gate and up
                            if (use_int8) iq4nl_expert_matvec_i16(gu[i].W.data(), gu[i].rows, gu[i].cols,
                                                                  gu[i].row_bytes, xg.data(), 1, yg.data());
                            else          iq4nl_expert_matvec_fp32(gu[i].W.data(), gu[i].rows, gu[i].cols,
                                                                   gu[i].row_bytes, xg.data(), 1, yg.data());
                        }
                        if (use_int8) iq4nl_expert_matvec_i16(dn[i].W.data(), dn[i].rows, dn[i].cols,
                                                              dn[i].row_bytes, xd.data(), 1, yd.data());
                        else          iq4nl_expert_matvec_fp32(dn[i].W.data(), dn[i].rows, dn[i].cols,
                                                               dn[i].row_bytes, xd.data(), 1, yd.data());
                    }
                });
                for (auto &t : th) t.join();
            }
            auto t1 = std::chrono::steady_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
            double per_expert = total_ms / NEXP;
            double gbs = MB_TOTAL / 1000.0 / (total_ms / 1000.0);
            // Budget comparison: 0.15 ms is Claude's per-layer target for ONE expert's worth
            // of miss correction (2-4 MB). Report how one expert compares.
            printf("%-8d %-8s %10.4f %10.2f %10.2f %12s\n",
                   nt, use_int8 ? "int8" : "fp32", per_expert, gbs, total_ms,
                   per_expert <= 0.15 ? "MEETS" : (per_expert <= 0.30 ? "2x over" : "over"));
        }
    }

    printf("\n=== SINGLE-EXPERT LATENCY (the number that matters for the 0.15ms gate) ===\n");
    printf("One miss expert = gate(0.92MB) + up(0.92MB) + down(0.92MB) = 2.76 MB\n");
    // MEASUREMENT FIX: the first version of this block spawned threads INSIDE the timed
    // loop, so it timed pthread_create (~0.1ms each) rather than the kernel. It reported
    // 2 threads = 0.496 ms and 3 threads = 0.565 ms -- SLOWER than 1 thread (0.385 ms),
    // which is impossible for parallel work and is the tell-tale sign of measuring
    // overhead. Replaced with PERSISTENT worker threads woken by an atomic generation
    // counter, so thread creation happens once, outside every measurement.
    //
    // Parallelism here splits ROWS of each matrix across workers, which is the only way to
    // attack a SINGLE expert's latency (table 1 above instead amortises across 24 experts,
    // which is the throughput view).
    struct Pool {
        std::vector<std::thread> th;
        std::atomic<long> gen{0};
        std::atomic<int> done{0};
        std::atomic<bool> stop{false};
        std::function<void(int)> work;
        explicit Pool(int n) {
            for (int i = 0; i < n; ++i) th.emplace_back([&, i]() {
                long seen = 0;
                for (;;) {
                    long g = gen.load(std::memory_order_acquire);
                    if (g == seen) {
                        // BUG FIX: this loop never checked `stop`, so the destructor could
                        // not end the workers. Destroying a still-joinable std::thread calls
                        // std::terminate -> "terminate called without an active exception"
                        // (exit 134 / SIGABRT), which is exactly what the first run did.
                        if (stop.load(std::memory_order_acquire)) return;
                        std::this_thread::yield();
                        continue;
                    }
                    seen = g;
                    if (stop.load(std::memory_order_acquire)) return;
                    work(i);
                    done.fetch_add(1, std::memory_order_release);
                }
            });
        }
        void run(std::function<void(int)> w) {
            work = std::move(w);
            done.store(0, std::memory_order_release);
            gen.fetch_add(1, std::memory_order_release);
            while (done.load(std::memory_order_acquire) < (int)th.size()) std::this_thread::yield();
        }
        ~Pool() {
            stop.store(true, std::memory_order_release);
            gen.fetch_add(1, std::memory_order_release);   // wake any parked worker
            for (auto &t : th) if (t.joinable()) t.join();
        }
    };

    for (int nt : {1, 2, 4, 6, 12}) {
        if (nt > hw) continue;
        // Split the three matvecs' ROWS across nt workers. Rows divide evenly:
        // gate/up 640 rows, down 2560 rows.
        std::vector<float> xg(GU_C, 0.5f), yg(GU_R), xd(DN_C, 0.5f), yd(DN_R);
        Pool pool(nt);
        const int reps = 200;
        // Warm.
        pool.run([&](int t) { (void)t; });
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            pool.run([&](int t) {
                // Each worker handles a contiguous row slice of all three matrices.
                const int gu_lo = (int)((long)GU_R * t / nt), gu_hi = (int)((long)GU_R * (t+1) / nt);
                const int dn_lo = (int)((long)DN_R * t / nt), dn_hi = (int)((long)DN_R * (t+1) / nt);
                // gate and up share the same row range; call twice on the same slice.
                for (int k = 0; k < 2; ++k)
                    iq4nl_expert_matvec_i16(gu[k].W.data() + (size_t)gu_lo * gu_rb,
                                            gu_hi - gu_lo, GU_C, gu_rb,
                                            xg.data(), 1, yg.data() + gu_lo);
                iq4nl_expert_matvec_i16(dn[0].W.data() + (size_t)dn_lo * dn_rb,
                                        dn_hi - dn_lo, DN_C, dn_rb,
                                        xd.data(), 1, yd.data() + dn_lo);
            });
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        printf("  %2d persistent thread(s), int8, row-split: %.4f ms per expert  -> %s\n",
               nt, ms, ms <= 0.15 ? "MEETS 0.15ms" : "OVER budget");
    }
    printf("\nMT_BENCH_DONE\n");
    return 0;
}
