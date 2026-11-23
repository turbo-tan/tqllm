// The multi-GPU control star (src/core/ctl_star.hpp): three ranks in threads over loopback, any connect order.
#include "../src/core/ctl_star.hpp"

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;
static void check(bool ok, const char * what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; } }
static int port_base() { return 20000 + int(getpid() % 20000); }

int main() {
    {   // three ranks; rank 2 connects before rank 1; ranks 0 and 2 share a host
        const int port = port_base();
        const std::vector<char> id = {'n', 'c', 'c', 'l'};
        std::vector<std::vector<char>> blob(3);
        std::vector<int> local(3, 0);
        std::vector<std::vector<uint64_t>> gathered(3, std::vector<uint64_t>(3, 0));
        std::vector<std::string> got(3);
        std::vector<std::string> err(3);
        const char * hosts[3] = {"boxA", "boxB", "boxA"};
        auto run = [&](int r) {
            try {
                if (r == 1) usleep(200000);
                if (r == 0) blob[0] = id;
                tqllm::CtlStar s(r, 3, "127.0.0.1", port, blob[size_t(r)], "abc1234", hosts[r]);
                local[size_t(r)] = s.local_ranks();
                const uint64_t mine = 100 + uint64_t(r);
                s.all_gather(&mine, gathered[size_t(r)].data(), sizeof mine);
                char msg[6] = {};
                if (r == 0) s.fan_out("hello", 6); else s.recv(0, msg, 6), got[size_t(r)] = msg;
                if (r != 0) { const int back = r * 7; s.send(0, &back, sizeof back); }
                else for (int p = 1; p < 3; ++p) { int v = 0; s.recv(p, &v, sizeof v); if (v != p * 7) err[0] = "bad reply"; }
            } catch (const std::exception & e) { err[size_t(r)] = e.what(); }
        };
        std::thread t0(run, 0), t2(run, 2), t1(run, 1);
        t0.join(); t1.join(); t2.join();
        for (int r = 0; r < 3; ++r) if (!err[size_t(r)].empty()) std::fprintf(stderr, "rank %d: %s\n", r, err[size_t(r)].c_str());
        check(err[0].empty() && err[1].empty() && err[2].empty(), "three ranks connect in any order");
        check(blob[1] == id && blob[2] == id, "rank 0's bootstrap bytes reach every rank");
        check(local[0] == 2 && local[1] == 1 && local[2] == 2, "ranks per host");
        bool same = true;
        for (int r = 0; r < 3; ++r) same = same && gathered[size_t(r)] == std::vector<uint64_t>({100, 101, 102});
        check(same, "all-gather in rank order on every rank");
        check(got[1] == "hello" && got[2] == "hello", "rank 0 reaches every rank");
    }
    {   // a rank started with another world size, or another build, is refused on every rank (no hang)
        for (int mode = 0; mode < 2; ++mode) {
            const int port = port_base() + 1 + mode;
            std::vector<std::string> err(2);
            auto run = [&](int r) {
                std::vector<char> b = {'x'};
                try {
                    const int world = mode == 0 && r == 1 ? 3 : 2;
                    tqllm::CtlStar s(r, world, "127.0.0.1", port, b, mode == 1 && r == 1 ? "other" : "abc1234");
                } catch (const std::exception & e) { err[size_t(r)] = e.what(); }
            };
            std::thread a(run, 0), b(run, 1);
            a.join(); b.join();
            check(!err[0].empty() && !err[1].empty(), mode == 0 ? "world mismatch refused on both ranks" : "build mismatch refused on both ranks");
            check(err[1].find("refused") != std::string::npos, "the follower is told why");
        }
    }
    {   // control links exist between rank 0 and the others only
        bool threw = false;
        const int port = port_base() + 3;
        std::thread a([&] { std::vector<char> b = {'x'}; try { tqllm::CtlStar s(0, 2, "", port, b, "t"); } catch (...) {} });
        std::vector<char> b;
        try {
            tqllm::CtlStar s(1, 2, "127.0.0.1", port, b, "t");
            try { s.fan_out("x", 1); } catch (const std::logic_error &) { threw = true; }
        } catch (...) {}
        a.join();
        check(threw, "only rank 0 fans out");
    }
    if (failures) return 1;
    std::puts("ctl_star: ok");
    return 0;
}
