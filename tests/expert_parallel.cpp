// Expert parallelism (src/experts/expert_parallel.hpp), no GPU: the forward rules and the lease protocol over an
// in-memory control channel (grant, announce, revoke in the middle of forwards, lease end order, the peer going away).
#include "experts/expert_parallel_rules.hpp"
#include <chrono>
#include <vector>

#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

using namespace tqllm;
static int failures = 0;
static void check(bool ok, const std::string & what) { if (!ok) { std::printf("FAIL %s\n", what.c_str()); ++failures; } }

// one direction of an ordered pipe; closed() makes recv throw (the other box went away)
struct Pipe {
    std::mutex m; std::condition_variable cv; std::deque<EpMsg> q; bool closed = false;
    void put(const EpMsg & x) { std::lock_guard<std::mutex> g(m); if (closed) throw std::runtime_error("closed"); q.push_back(x); cv.notify_all(); }
    EpMsg take() { std::unique_lock<std::mutex> l(m); cv.wait(l, [&] { return closed || !q.empty(); }); if (q.empty()) throw std::runtime_error("closed"); auto x = q.front(); q.pop_front(); return x; }
    void close() { std::lock_guard<std::mutex> g(m); closed = true; cv.notify_all(); }
};
struct End final : EpControl {
    Pipe & out; Pipe & in;
    End(Pipe & o, Pipe & i) : out(o), in(i) {}
    void send(const EpMsg & m) override { out.put(m); }
    EpMsg recv() override { return in.take(); }
};

int main() {
    check(ep_result_cap(10) == 7 && ep_result_cap(80) == 50 && ep_result_cap(240) == 150 && ep_result_cap(1) == 1 && ep_result_cap(0) == 0,
          "--split experts result cap: 5/8 of the assignments, rounded up, never more than all of them");
    {   // --split experts forward state (3 trunk layers, rows 2)
        int announced = 0; bool answer = true;
        auto ann = [&](int) { ++announced; return answer; };
        auto layer = [&](EpForwardState & st, int l, bool can = true, bool alive = true) { const auto c = ep_before_layer(st, l, 2, can, alive, ann); if (c != EpCall::Fail) ep_after_layer(st, l, 3); return c; };
        auto whole = [&](EpForwardState & st, EpCall want) { bool ok = true; for (int l = 0; l < 3; ++l) ok = ok && layer(st, l) == want; return ok && !st.open; };
        EpForwardState s;
        check(whole(s, EpCall::Exchange) && announced == 1 && whole(s, EpCall::Exchange) && announced == 2, "--split experts: an accepted forward exchanges every layer, announced once");
        answer = false;
        check(whole(s, EpCall::Local) && announced == 3 && !s.off, "a declined forward runs locally; the next is asked again");
        answer = true; s = {}; announced = 0;
        check(ep_before_layer(s, 0, 2, true, true, ann) == EpCall::Exchange && ep_before_layer(s, 0, 2, true, true, ann) == EpCall::Exchange && announced == 1,
              "a layer 0 re-run after a failed capture is not announced twice");
        s = {}; layer(s, 0); layer(s, 1);
        check(layer(s, 2, true, false) == EpCall::Fail && s.off && layer(s, 0) == EpCall::Local, "rank 1 gone inside an accepted forward: that request fails, then local");
        s = {}; layer(s, 0); layer(s, 1);  // the forward failed at layer 2 (its send threw: rank 1 went away)
        check(ep_before_layer(s, 0, 2, true, false, ann) == EpCall::Local && s.off, "rank 1 gone, next forward: local, not a second failed request (c5e139d9)");
        s = {}; layer(s, 0); layer(s, 1);  // an accepted forward left open while rank 1 is alive (it still waits for layer 2)
        announced = 0;
        check(layer(s, 0) == EpCall::Local && s.off && announced == 0, "an accepted forward left open: the next runs locally, nothing announced (no out-of-step exchange)");
        answer = false; s = {}; layer(s, 0); layer(s, 1); answer = true; announced = 0;
        check(layer(s, 0) == EpCall::Exchange && announced == 1 && !s.off, "a declined forward left open: rank 1 waits for nothing, the next forward is asked");
        s = {}; layer(s, 0);
        check(layer(s, 1, false) == EpCall::Fail && s.off, "an accepted forward whose layer cannot exchange fails");
        s = {};
        check(layer(s, 0, false) == EpCall::Local && !s.open && !s.off && layer(s, 1) == EpCall::Local, "layer 0 cannot exchange: the forward is not announced, all local");
        s = {}; layer(s, 0);
        check(layer(s, 2) == EpCall::Fail && s.off, "an accepted forward's layer out of order: fails, parallelism off");
    }

    {   // lease: rank 1 grants while idle, rank 0 announces under it, a revoke waits for the lease end after the last forward
        Pipe to1, to0; End r0(to1, to0), r1(to0, to1);
        EpLeaseHolder holder(r0);
        bool idle = true, ready = true;
        EpLeaseGiver giver(r1, [&] { return idle; }, [&] { return ready; });
        std::thread reader([&] { for (;;) { EpMsg m; try { m = r0.recv(); } catch (...) { holder.lost(); return; } if (!holder.on_message(m)) return; } });
        check(!holder.announce(1), "no lease yet: rank 0 runs alone");
        giver.note_idle();
        for (int i = 0; i < 200 && !holder.leased(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(holder.leased(), "an idle rank 1 grants the lease");
        check(holder.announce(2) && holder.announce(3), "under the lease rank 0 announces forwards without waiting");
        // rank 1's helper: reads the two forwards, then a request of its own revokes; the lease end must follow both forwards
        std::vector<EpMsg> seen;
        std::thread helper([&] { for (;;) { EpMsg m; try { m = r1.recv(); } catch (...) { return; } seen.push_back(m); if (m.kind == kEpLeaseEnd) { giver.lease_ended(); return; } } });
        idle = false; giver.request_begin();  // returns only after the lease end
        helper.join();
        check(seen.size() == 3 && seen[0].kind == kEpForward && seen[0].n == 2 && seen[1].n == 3 && seen[2].kind == kEpLeaseEnd,
              "revoke: both forwards arrive before the lease end, then this box serves");
        check(!holder.leased() && !holder.announce(1), "after the revoke rank 0 runs alone");
        giver.note_idle();
        check(!holder.leased(), "a busy box does not lend");
        idle = true; ready = false; giver.note_idle();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        check(!holder.leased(), "a box whose banks are not ready after its request does not lend");
        ready = true; giver.note_idle();
        for (int i = 0; i < 200 && !holder.leased(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(holder.leased(), "banks ready again: the lease goes out");
        giver.close(); idle = true; giver.note_idle();
        to0.close(); reader.join();
        check(!holder.leased(), "rank 1 went away: the lease is dropped");
    }
    check(ep_result_cap(10) == 7 && ep_result_cap(0) == 0, "result cap");
    std::printf("expert parallel: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
