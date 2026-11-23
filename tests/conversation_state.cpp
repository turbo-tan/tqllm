// Conversation state store / directory (src/kv/conversation_state.hpp), no GPU: the local store keeps the prompt park's
// rules (differential against PromptPark on random operations), entity tags, the byte budget, buffers handed back to the
// engine's pool, and the directory's ownership with stale owners.
#include "../src/kv/conversation_state.hpp"
#include "../src/kv/prompt_park.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

using namespace tqllm;
static int failures = 0;
static void check(bool ok, const std::string & what) { if (!ok) { std::printf("FAIL %s\n", what.c_str()); ++failures; } }

static int pool_returns = 0;
static StateBlob blob(uint64_t conv, size_t bytes, const std::string & entity = "a", uint32_t positions = 10) {
    StateBlob b;
    b.meta.conv = conv; b.meta.positions = positions; b.meta.entity = entity; b.meta.layout = "test";
    b.meta.tokens.assign(positions, uint32_t(conv));
    StatePart p;
    p.kind = StatePart::Kind::KvRows; p.layer = 0; p.bytes = bytes;
    p.data = std::shared_ptr<void>(new char[bytes ? bytes : 1], [](void * q) { delete[] static_cast<char *>(q); ++pool_returns; });
    b.parts.push_back(std::move(p));
    return b;
}

int main() {
    using Put = ConversationStateStore::Put;
    using W = StateLocation::Where;
    {   // the prompt park's walk-through, through the store
        auto s = make_state_store("local", {3});
        check(s->put(blob(1, 10)) == Put::Stored && s->put(blob(2, 10)) == Put::Stored && s->put(blob(3, 10)) == Put::Stored, "free spaces first");
        check(s->put(blob(2, 12)) == Put::Replaced && s->stats().bytes == 32, "one state per conversation: the newest replaces the old");
        s->protect(1);
        check(s->put(blob(4, 10)) == Put::Stored && s->where(3).where == W::None && s->where(1).where == W::Here, "full: the oldest unprotected goes, the queued one stays");
        s->retire(2);
        check(s->put(blob(5, 10)) == Put::Stored && s->where(2).where == W::None && s->stats().dead_overwritten == 1, "a finished conversation goes first");
        auto t = s->take(1, "a");
        check(t && t->meta.conv == 1 && s->where(1).where == W::None && s->stats().taken == 1, "take moves the state out");
        check(s->take(4, "b") == nullptr && s->stats().denied == 1 && s->where(4).where == W::Here, "another entity is denied; the state stays");
        s->protect(4); s->protect(5);
        check(s->put(blob(6, 10)) == Put::Stored, "the taken space is free again");
        s->protect(6);
        check(s->put(blob(7, 10)) == Put::NoRoom && s->stats().no_room == 1, "every state protected: no room");
        s->release(5);
        check(s->put(blob(7, 10)) == Put::Stored && s->where(5).where == W::None, "released protection makes it evictable");
        std::vector<uint64_t> order;
        s->for_each([&](const StateMeta & m) { order.push_back(m.conv); });
        check(order == std::vector<uint64_t>{7, 6, 4}, "for_each: newest first");
        check(s->evict(6) && !s->evict(6) && s->where(6).where == W::None, "evict drops the state once");
    }
    {   // differential: the store and PromptPark make the same choices on random operations
        std::mt19937 rng(11);
        for (int trial = 0; trial < 40; ++trial) {
            const int spaces = 1 + int(rng() % 6);
            auto s = make_state_store("local", {spaces});
            PromptPark p(spaces);
            bool same = true;
            for (int op = 0; op < 400 && same; ++op) {
                const uint64_t c = 1 + rng() % 12;
                switch (rng() % 6) {
                    case 0: case 1: { const bool kept = s->put(blob(c, 8)) != Put::NoRoom; same = kept == (p.park(c) >= 0); break; }
                    case 2: { const bool got = s->take(c, "a") != nullptr; same = got == (p.pickup(c) >= 0); break; }
                    case 3: s->protect(c); p.protect(c); break;
                    case 4: s->release(c); p.release(c); break;
                    case 5: s->retire(c); p.retire(c); break;
                }
                for (uint64_t q = 1; q <= 12 && same; ++q) same = (s->where(q).where == W::Here) == (p.where(q) >= 0);
            }
            check(same, "local store == PromptPark, trial " + std::to_string(trial));
        }
    }
    {   // byte budget: evictions in the park's order until the new state fits; larger than the budget: refused
        auto s = make_state_store("local", {8, 100});
        check(s->put(blob(1, 40)) == Put::Stored && s->put(blob(2, 40)) == Put::Stored, "two fit");
        s->retire(2);
        check(s->put(blob(3, 40)) == Put::Stored && s->where(2).where == W::None && s->where(1).where == W::Here && s->stats().bytes == 80,
              "over budget: the dead one goes first");
        s->protect(1); s->protect(3);
        check(s->put(blob(4, 40)) == Put::NoRoom && s->stats().bytes == 80, "only protected states left: no room");
        check(s->put(blob(5, 101)) == Put::Refused, "larger than the budget: refused");
        check(s->put(blob(1, 60)) == Put::Replaced && s->stats().bytes == 100, "a replacement counts its own old bytes as free");
        StateBlob d = blob(9, 4); d.parts[0].memory = StatePart::Memory::Device;
        check(s->put(std::move(d)) == Put::Refused, "the local store keeps host parts only");
    }
    {   // buffers go back to the engine's pool when a state is replaced, evicted or released by its taker
        pool_returns = 0;
        auto s = make_state_store("local", {2});
        s->put(blob(1, 8)); s->put(blob(1, 8));
        check(pool_returns == 1, "replaced: the old buffer returns");
        s->put(blob(2, 8)); s->put(blob(3, 8));
        check(pool_returns == 2, "evicted: its buffer returns");
        { auto t = s->take(2, "a"); check(pool_returns == 2, "taken: the buffer stays with the taker"); }
        check(pool_returns == 3, "the taker done: it returns");
    }
    {   // directory: one live owner per conversation; a stale owner loses it
        auto now = std::chrono::steady_clock::time_point{};
        auto d = make_state_directory("local", std::chrono::seconds(30), [&] { return now; });
        check(d->claim(7, 0) && !d->claim(7, 1) && d->owner(7) == 0, "one owner; another live box cannot claim");
        d->announce(7, 0, 512, 4096, "L");
        const auto e = d->entry(7);
        check(e.box == 0 && e.positions == 512 && e.bytes == 4096 && e.layout == "L", "announce records positions, bytes, layout");
        d->heartbeat(1);
        now += std::chrono::seconds(31); d->heartbeat(1);
        check(d->owner(7) == -1 && d->claim(7, 1) && d->owner(7) == 1, "box 0 not seen for 30 s: box 1 claims it");
        d->release(7, 0);
        check(d->owner(7) == 1, "a release by a box that no longer owns it changes nothing");
        d->release(7, 1);
        check(d->owner(7) == -1, "released");
    }
    {   // a store with a directory: puts claim and announce, takes and evictions release, another box's state is Peer
        auto d = make_state_directory("local");
        StateStoreOptions o; o.spaces = 2; o.box = 0; o.directory = d.get();
        auto s = make_state_store("local", o);
        check(s->put(blob(1, 8, "a", 64)) == Put::Stored && d->owner(1) == 0 && d->entry(1).positions == 64, "put claims and announces");
        check(d->claim(2, 1) && s->where(2).where == W::Peer && s->where(2).box == 1, "another box's conversation: Peer");
        check(s->put(blob(2, 8)) == Put::Refused, "another live box owns it: refused here");
        s->take(1, "a");
        check(d->owner(1) == -1, "take releases");
        s->put(blob(3, 8)); s->put(blob(4, 8)); s->put(blob(5, 8));
        check(d->owner(3) == -1 && d->owner(5) == 0, "an evicted state is released");
    }
    {   // one blob as a file, and the layout check names the first differing field
        StateBlob b = blob(42, 64, "ent", 9); b.meta.turn = 3;
        std::memset(b.parts[0].data.get(), 0x5a, 64);
        StatePart e; e.kind = StatePart::Kind::Engine; e.layer = -1; e.bytes = 0; e.data = std::shared_ptr<void>(new char[1], [](void * q) { delete[] static_cast<char *>(q); });
        b.parts.push_back(e);
        const std::string path = "conversation_state_test.bin";
        write_state_blob(path, b);
        const StateBlob r = read_state_blob(path);
        check(r.meta.conv == 42 && r.meta.turn == 3 && r.meta.positions == 9 && r.meta.tokens == b.meta.tokens && r.meta.layout == "test" &&
              r.meta.entity == "ent" && r.parts.size() == 2 && r.parts[0].bytes == 64 && r.parts[0].layer == 0 && r.parts[1].bytes == 0 &&
              std::memcmp(r.parts[0].data.get(), b.parts[0].data.get(), 64) == 0, "a state file reads back the same blob");
        { std::ofstream t(path, std::ios::binary | std::ios::trunc); t << "TQSTATE1"; }
        try { read_state_blob(path); check(false, "a truncated file throws"); } catch (const std::runtime_error &) {}
        std::remove(path.c_str());
        std::string wire;  // the socket form: the same bytes, its size known up front, a ceiling on the reader
        check(encode_state_blob(b, [&](const void * q, size_t n) { wire.append(static_cast<const char *>(q), n); return true; }) && wire.size() == encoded_size(b),
              "encoded_size is the stream's length");
        size_t at = 0;
        auto src = [&](void * q, size_t n) { if (at + n > wire.size()) return false; std::memcpy(q, wire.data() + at, n); at += n; return true; };
        const StateBlob w = decode_state_blob(src, wire.size(), "wire");
        check(w.meta.conv == 42 && w.parts.size() == 2 && std::memcmp(w.parts[0].data.get(), b.parts[0].data.get(), 64) == 0, "a streamed blob decodes back");
        at = 0;
        try { decode_state_blob(src, 100, "wire"); check(false, "over the ceiling throws"); } catch (const std::runtime_error & e) { check(std::string(e.what()).find("ceiling") != std::string::npos, "the ceiling is named"); }
        at = 0; wire.resize(wire.size() - 10);
        try { decode_state_blob(src, SIZE_MAX, "wire"); check(false, "a cut stream throws"); } catch (const std::runtime_error & e) { check(std::string(e.what()).find("truncated") != std::string::npos, "truncation is named"); }
        check(layout_mismatch("v=1|model=a|rank=0", "v=1|model=a|rank=0").empty(), "equal layouts");
        check(layout_mismatch("v=1|model=a|rank=1", "v=1|model=a|rank=0") == "rank: got 1, want 0", "the differing field is named");
        check(layout_mismatch("v=1|model=a", "v=1|model=a|part0=bf16[9][256]") == "part0: got (none), want bf16[9][256]", "a missing field is named");
    }
    try { make_state_store("nope", {1}); check(false, "unknown backend throws"); } catch (const std::invalid_argument &) {}
    try { make_state_store("local", {0}); check(false, "no spaces throws"); } catch (const std::invalid_argument &) {}
    if (failures) return 1;
    std::puts("conversation_state: ok");
    return 0;
}
