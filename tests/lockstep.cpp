// The layer split's call forwarding (src/engine/lockstep.hpp): rank 1 makes the calls rank 0 makes, with the same arguments.
#include "../src/engine/lockstep.hpp"

#include <cstdio>
#include <deque>
#include <sstream>

static int failures = 0;
static void check(bool ok, const char * what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; } }

struct Pipe : tqllm::ByteChannel {
    std::deque<char> q;
    void send(const void * p, size_t n) override { const auto * c = static_cast<const char *>(p); q.insert(q.end(), c, c + n); }
    void recv(void * p, size_t n) override {
        if (q.size() < n) throw std::runtime_error("pipe empty");
        auto * c = static_cast<char *>(p);
        for (size_t i = 0; i < n; ++i) c[i] = q.front(), q.pop_front();
    }
};

struct Log : tqllm::Engine {  // writes each call and its arguments as one line
    std::ostringstream o;
    uint32_t step(uint32_t id, uint32_t pos, bool s, float *) override { o << "step " << id << ' ' << pos << ' ' << s << '\n'; return id + 1; }
    void verify(const uint32_t * t, int n, uint32_t pos, uint32_t * next) override {
        o << "verify " << pos;
        for (int i = 0; i < n; ++i) o << ' ' << t[i], next[i] = t[i] * 2;
        o << '\n';
    }
    void commit(int k) override { o << "commit " << k << '\n'; }
    void set_sampling(float t, float tp, int tk, uint64_t seed, float pr) override { o << "sampling " << t << ' ' << tp << ' ' << tk << ' ' << seed << ' ' << pr << '\n'; }
    void set_penalties(float r, float f) override { o << "penalties " << r << ' ' << f << '\n'; }
    void set_ignore_eos(bool b) override { o << "ignore_eos " << b << '\n'; }
    void set_continuation(int c) override { o << "continuation " << c << '\n'; }
    int slots() const override { return 4; }
    void select_slot(int s) override { o << "slot " << s << '\n'; }
    void spec_begin(const std::vector<uint32_t> & p, uint32_t c, int d) override {
        o << "begin " << c << ' ' << d;
        for (auto t : p) o << ' ' << t;
        o << '\n';
        if (p.empty()) throw std::runtime_error("empty prompt");
    }
    bool spec_advance() override { o << "advance\n"; return true; }
    void spec_abort(int s) override { o << "abort " << s << '\n'; }
    void spec_step(const int * s, int n, bool * done) override {
        o << "spec_step";
        for (int i = 0; i < n; ++i) o << ' ' << s[i], done[i] = false;
        o << '\n';
    }
    std::vector<uint32_t> spec_finish(uint64_t *) override { o << "finish\n"; return {7}; }
    bool prefill_state(const std::vector<uint32_t> & t, uint32_t pos) override { o << "prefill " << t.size() << ' ' << pos << '\n'; return true; }
    void set_full_logits(bool b) override { o << "full_logits " << b << '\n'; }
    bool kv_idle_step() override { o << "idle\n"; return false; }
    void synchronize() override {}
    uint32_t layers() const override { return 64; }
    uint32_t hidden() const override { return 8; }
    uint32_t vocab() const override { return 100; }
    uint32_t eos() const override { return 0; }
    uint32_t context() const override { return 1024; }
};

int main() {
    auto pipe = std::make_shared<Pipe>();
    auto * l0 = new Log, * l1 = new Log;
    tqllm::LockstepEngine r0(std::unique_ptr<tqllm::Engine>(l0), pipe, 0), r1(std::unique_ptr<tqllm::Engine>(l1), pipe, 1);
    r0.set_sampling(0.7f, 0.9f, 16, 42, 0.5f);
    r0.set_penalties(1.1f, 0.2f);
    r0.set_ignore_eos(true);
    r0.set_continuation(2);
    r0.select_slot(3);
    r0.spec_begin({1, 2, 3, 70000}, 50, 7);
    bool threw = false;
    try { r0.spec_begin({}, 5, 1); } catch (const std::exception &) { threw = true; }
    check(threw, "rank 0 still raises the engine's error");
    check(r0.spec_advance(), "spec_advance result from rank 0's engine");
    const int ss[3] = {0, 2, 3};
    bool done[3];
    r0.spec_step(ss, 3, done);
    uint32_t vt[3] = {5, 6, 7}, nx[3] = {};
    r0.verify(vt, 3, 99, nx);
    check(nx[0] == 10 && nx[2] == 14, "verify results reach rank 0's caller");
    r0.commit(2);
    check(r0.step(9, 4, true, nullptr) == 10, "step result");
    r0.prefill_state(std::vector<uint32_t>(5000, 1), 12);
    r0.set_full_logits(true);
    r0.kv_idle_step();
    r0.spec_abort(1);
    check(r0.spec_finish(nullptr) == std::vector<uint32_t>{7}, "spec_finish result");
    check(r0.slots() == 4 && r0.layers() == 64, "queries are local");
    r0.stop_followers();
    r1.follow();  // replays everything, then returns at stop
    check(pipe->q.empty(), "rank 1 read every call");
    check(l0->o.str() == l1->o.str(), "rank 1 made the same calls with the same arguments");
    check(l1->o.str().find("begin 50 7 1 2 3 70000\n") != std::string::npos, "prompt tokens carried");
    check(l1->o.str().find("begin 5 1\n") != std::string::npos, "rank 1 also made the failing call and kept following");
    threw = false;
    try { r1.commit(1); } catch (const std::exception &) { threw = true; }
    check(threw, "rank 1 refuses calls of its own");
    threw = false;
    try { r0.commit(1); } catch (const std::exception &) { threw = true; }
    check(threw, "no calls after stop");
    if (failures) { std::fprintf(stderr, "rank 0:\n%s\nrank 1:\n%s", l0->o.str().c_str(), l1->o.str().c_str()); return 1; }
    std::puts("lockstep: ok");
    return 0;
}
