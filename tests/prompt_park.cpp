// PromptPark (src/kv/prompt_park.hpp): the TPark rules for kept prompt states outside the serving slots.
#include "../src/kv/prompt_park.hpp"
#include <cstdio>
static int failures = 0;
static void check(bool ok, const char * what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; } }
int main() {
    tqllm::PromptPark p(3);
    const int a = p.park(1), b = p.park(2), c = p.park(3);
    check(a >= 0 && b >= 0 && c >= 0 && a != b && b != c && a != c, "free spaces first");
    check(p.park(2) == b && p.stats().replaced == 1, "a conversation keeps one space: its newest state replaces the old");
    p.protect(1);  // a follow-up for conversation 1 is queued
    const int d = p.park(4);  // full: oldest live not protected is 3 (2 was refreshed), 1 is protected
    check(d == c && p.where(3) < 0 && p.where(1) == a, "the queued conversation's state is never evicted; the oldest unprotected goes");
    p.retire(2);
    check(p.park(5) == b && p.where(2) < 0, "a finished conversation's state goes before any live one");
    check(p.pickup(1) == a && p.where(1) < 0, "pickup frees the space");
    check(p.park(6) == a, "a picked-up space is free again");
    p.protect(4); p.protect(5); p.protect(6);
    check(p.park(7) == -1 && p.stats().refused == 1, "every space protected: refused (the new state is not kept)");
    p.release(5);
    check(p.park(7) == b && p.where(5) < 0, "released protection makes it evictable");
    check(p.pickup(99) == -1, "not parked");
    if (failures) return 1;
    std::puts("prompt_park: ok");
    return 0;
}
