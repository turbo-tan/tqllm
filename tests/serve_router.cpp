// tools/serve/router.hpp: which box serves a request (tqserve --peer).
#include "tools/serve/router.hpp"

#include <cstdio>
#include <cstdlib>

using tqserve::BoxStatus;

static int failures = 0;
static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}

static BoxStatus box(int free, int busy, int queued = 0, double reuse = 0, int inflight = 0) {
    BoxStatus b;
    b.ok = true; b.free = free; b.busy = busy; b.queued = queued; b.reuse = reuse; b.inflight = inflight;
    b.prefill_tps = 1900; b.decode_tps = 100; b.hold_s = 20;
    return b;
}

int main() {
    // both quiet: the front door keeps it (no hop)
    check(tqserve::pick_box({box(8, 0), box(8, 0)}, 200, 384) == 0, "quiet boxes -> front door");
    {   // --split experts: a lone request goes to the helped front door although the peer measured faster (11-12: a lone
        // request routed to the peer decoded 66 instead of ~72 tok/s), unless the peer is much cheaper (it holds the prompt)
        auto f = box(8, 0), p = box(8, 0); f.helped = true; f.decode_tps = 64; p.decode_tps = 70;
        check(tqserve::pick_box({f, p}, 200, 384) == 0, "lone request -> the box with expert help");
        auto pu = p; pu.decode_tps = 0;  // not measured yet: route_cost reads 100 tok/s (11-12 smoke: q16 went to the peer)
        check(tqserve::pick_box({f, pu}, 200, 384) == 0, "lone request, peer not measured yet -> the box with expert help");
        auto pk = p; pk.reuse = 30000;
        check(tqserve::pick_box({f, pk}, 30500, 384) == 1, "lone follow-up -> the box that holds its K/V");
        auto fb = f; fb.busy = 1; fb.free = 7;
        check(tqserve::pick_box({fb, p}, 200, 384) == 1, "front door busy: the helper declines anyway -> the usual rule");
        {   // after lone requests all went to the helped front door, the peer is not measured: 4 at once must split 2 + 2
            std::vector<BoxStatus> b = {box(8, 0), box(8, 0)}; b[0].helped = true; b[0].decode_tps = 68; b[0].prefill_tps = 2200;
            b[1].decode_tps = 0; b[1].prefill_tps = 0; int n[2] = {0, 0};
            for (int r = 0; r < 4; ++r) { const int k = tqserve::pick_box(b, 200, 384); ++n[k]; ++b[size_t(k)].inflight; }
            check(n[0] == 2 && n[1] == 2, "4 at once, peer not measured yet -> 2 + 2");
        }
        auto pb = p; pb.busy = 1; pb.free = 7;
        check(tqserve::pick_box({f, pb}, 200, 384) == 0, "peer busy: usual rule");
    }
    // the front door decodes 4 sequences, the peer none: a new chat goes to the peer (decode is not slowed there)
    check(tqserve::pick_box({box(4, 4), box(8, 0)}, 200, 384) == 1, "busy front door -> idle peer");
    // equal load: stays on the front door
    check(tqserve::pick_box({box(6, 2), box(6, 2)}, 200, 384) == 0, "equal load -> front door");
    // a follow-up turn of a 30K conversation goes back to the box holding its K/V
    check(tqserve::pick_box({box(6, 2), box(6, 2, 0, 30000)}, 30500, 384) == 1, "follow-up -> box with the K/V");
    // ... unless that box is full and queued while the other is idle and the prompt is short
    check(tqserve::pick_box({box(8, 0), box(0, 8, 2, 2900)}, 3000, 384) == 0, "follow-up on a full box -> idle box");
    // a long new prompt avoids the box with many decoding users
    check(tqserve::pick_box({box(2, 6), box(6, 2)}, 32000, 384) == 1, "long prompt -> fewer decoders");
    // requests arriving together: the front door's in-flight counts spread them (16 users -> 8 + 8)
    {
        std::vector<BoxStatus> b = {box(8, 0), box(8, 0)};
        int n[2] = {0, 0};
        for (int i = 0; i < 16; ++i) {
            const int k = tqserve::pick_box(b, 200, 384);
            ++n[k];
            ++b[size_t(k)].inflight;
        }
        check(n[0] == 8 && n[1] == 8, "16 simultaneous requests -> 8 + 8");
    }
    // one box measured faster (11-11, .55 + .44 Flash-Next: costs 7.44 vs 5.65 at the same load): it took 10 of 16, two queued
    // behind its 8 slots while the other box had one free. A box with a free slot wins over a full one, whatever the rates.
    {
        std::vector<BoxStatus> b = {box(8, 0), box(8, 0)};
        b[1].decode_tps = 135; b[1].prefill_tps = 2600;
        int n[2] = {0, 0};
        for (int i = 0; i < 16; ++i) {
            const int k = tqserve::pick_box(b, 200, 384);
            ++n[k];
            ++b[size_t(k)].inflight;
        }
        check(n[0] == 8 && n[1] == 8, "16 simultaneous requests, one box faster -> 8 + 8 (no queue while a slot is free)");
    }
    // every slot taken on both: the one with the shorter queue
    check(tqserve::pick_box({box(0, 8, 3), box(0, 8, 1)}, 200, 384) == 1, "all full -> shorter queue");
    // a peer that did not answer is skipped; nobody answered -> front door
    {
        BoxStatus dead = box(8, 0); dead.ok = false;
        check(tqserve::pick_box({box(0, 8, 4), dead}, 200, 384) == 0, "dead peer skipped");
        BoxStatus d0 = dead;
        check(tqserve::pick_box({d0, dead}, 200, 384) == 0, "nobody answered -> front door");
    }
    // reuse: shared template header alone does not count; a held conversation does
    {
        const std::string sys = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
        check(tqserve::reuse_tokens(sys + "hello", sys + "bye", 4) == 0, "template header is not reuse");
        const std::string conv = sys + std::string(4000, 'x');
        check(tqserve::reuse_tokens(conv + "answer", conv + "answer<|im_end|>next", 4) > 1000, "held conversation counts");
    }
    // the box already admitted what the front door sent: no double count
    check(tqserve::route_load(box(4, 4, 0, 0, 4)) == 4, "admitted in-flight requests counted once");
    // 17th request with both boxes full and even: waits on either, never a crash
    {
        std::vector<BoxStatus> b = {box(0, 8, 0, 0, 8), box(0, 8, 0, 0, 8)};
        const int k = tqserve::pick_box(b, 200, 384);
        check(k == 0 || k == 1, "both full -> a box");
    }
    // a busy peer answers its status late (16 requests at once): it stays a candidate with its last answer + the
    // front door's in-flight count, instead of being skipped while the front door queues past its slots (11-09)
    {
        tqserve::PeerMemory mem;
        BoxStatus seen = box(8, 0); seen.ok = true;
        (void) tqserve::remembered_status(seen, mem);           // an earlier answer: 8 free
        BoxStatus late; late.ok = false; late.inflight = 3;      // this query timed out; 3 sent there, not finished
        const BoxStatus used = tqserve::remembered_status(late, mem);
        check(used.ok && used.inflight == 3 && used.free == 8, "late answer: last status, current in-flight");
        BoxStatus front = box(0, 8, 0, 0, 8); front.ok = true;   // the front door is full
        check(tqserve::pick_box({front, used}, 200, 384) == 1, "full front door, late peer with room -> peer");
        (void) tqserve::remembered_status(late, mem);
        const BoxStatus down = tqserve::remembered_status(late, mem);  // third miss in a row
        check(!down.ok, "three misses in a row -> skipped");
        tqserve::PeerMemory fresh;
        check(!tqserve::remembered_status(late, fresh).ok, "never answered -> skipped");
    }
    {   // conversation states across boxes (--state-listen): fetch-aware picks
        auto st = [](BoxStatus b, const char * a) { b.state = a; return b; };
        // the holder is fuller than the other box: the other box takes the follow-up and fetches the state from the holder
        BoxStatus holder = st(box(1, 7, 0, 900), "10.0.0.2:9000"), other = st(box(6, 2), "10.0.0.1:9000");
        auto r = tqserve::pick_box_fetch({other, holder}, 1000, 384, 3e9);
        check(r.box == 0 && r.owner == 1 && r.fetch_s > 0, "holder busier -> the other box, fetching from the holder");
        // same load: the holder keeps it (a fetch costs more than holding)
        holder = st(box(4, 4, 0, 900), "10.0.0.2:9000"); other = st(box(4, 4), "10.0.0.1:9000");
        r = tqserve::pick_box_fetch({other, holder}, 1000, 384, 3e9);
        check(r.box == 1 && r.owner == -1, "same load -> the holder, no fetch");
        // a box without a state listener cannot fetch: the old prefix rule decides (the holder)
        BoxStatus plain = box(6, 2); holder = st(box(1, 7, 0, 900), "10.0.0.2:9000");
        r = tqserve::pick_box_fetch({plain, holder}, 1000, 384, 3e9);
        check(r.owner == -1, "no listener on the other box -> no fetch");
        // nobody holds the prompt: the plain pick
        r = tqserve::pick_box_fetch({st(box(6, 2), "a:1"), st(box(1, 7), "b:1")}, 1000, 384, 3e9);
        check(r.box == 0 && r.owner == -1, "nothing held -> the usual pick, no fetch");
        // a slow link prices the fetch above the prefill it saves: no fetch
        holder = st(box(1, 7, 0, 900), "10.0.0.2:9000"); other = st(box(6, 2), "10.0.0.1:9000");
        r = tqserve::pick_box_fetch({other, holder}, 1000, 384, 2e7);
        check(r.owner == -1, "20 MB/s link -> prefill instead of a fetch");
    }
    {   // the holder survives a late status answer (remembered_status knows no reuse)
        tqserve::HolderMemory hm;
        const std::string prompt(3000, 'x');
        const size_t k = tqserve::HolderMemory::key(prompt);
        auto st = [](BoxStatus b, const char * a) { b.state = a; return b; };
        std::vector<BoxStatus> now = {st(box(4, 4), "a:1"), st(box(1, 7, 0, 700), "b:1")};
        hm.note(now, k, 0.0);
        std::vector<BoxStatus> late = {st(box(6, 2), "a:1"), st(box(1, 7, 0, 0), "b:1")};  // box 1 late: its reuse is unknown
        hm.apply(late, k, 5.0);
        auto r = tqserve::pick_box_fetch(late, 900, 384, 3e9);
        check(late[1].reuse == 700 && r.box == 0 && r.owner == 1, "late status -> still fetches from the remembered holder");
        std::vector<BoxStatus> later = {st(box(6, 2), "a:1"), st(box(1, 7, 0, 0), "b:1")};
        hm.apply(later, k, 40.0);
        check(later[1].reuse == 0, "past the TTL -> forgotten");
        hm.served(k, 0, 900, 41.0);
        std::vector<BoxStatus> next = {st(box(6, 2), "a:1"), st(box(1, 7), "b:1")};
        hm.apply(next, k, 42.0);
        check(next[0].reuse == 900, "the box a request went to holds it afterwards");
        check(tqserve::HolderMemory::key("short") == 0, "a short prompt (no held conversation) has no key");
        // a stale report from the box that lent the state away must not take the holder back (GB10 11-15: three third
        // turns went to the lender, which held nothing any more, and prefilled in full)
        tqserve::HolderMemory h2;
        h2.served(k, 1, 928, 100.0);  // turn 2 went to box 1 (it fetched the state)
        std::vector<BoxStatus> t3 = {st(box(4, 4, 0, 862), "a:1"), st(box(0, 8, 0, 0), "b:1")};  // box 0 still reports its old text
        h2.note(t3, k, 110.0);
        h2.apply(t3, k, 110.0);
        auto r3 = tqserve::pick_box_fetch(t3, 1000, 64, 3e9);
        check(t3[1].reuse >= 928 && (r3.box == 1 || r3.owner == 1), "a stale lower report from the lender -> the turn follows the newer holder");
        // shared-prefix tokens (the lender forgets by tokens, not by decoded text)
        check(tqserve::shared_prefix({1, 2, 3, 4}, {1, 2, 3, 9}) == 3 && tqserve::shared_prefix({}, {1}) == 0, "shared token prefix");
    }
    if (failures) return 1;
    std::puts("serve_router: ok");
    return 0;
}
