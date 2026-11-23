#pragma once
// Conversation state kept outside the serving slots, on this box or another, behind a replaceable interface (owner 11-14:
// another cache engine can take its place). A store keeps one state per conversation (its newest kept turn); a directory
// records which box holds it. Engines write and read the bytes through their own serialiser: nothing here knows a layout.
// Backends by name (--kv-store): "local" (host memory of this box, the prompt park's rules), later "peer" (fetched from
// the owning box over the TpComm link) and external engines. No CUDA: buffers come from the engine (pinned host memory);
// a part's deleter may hand its buffer back to the engine's pool. Tests: tests/conversation_state.cpp.
//
// Threading: every store and directory method may be called from any thread; each object serialises its calls with its
// own lock. for_each calls fn under that lock: fn must not call back into the same object. A blob handed out by take()
// belongs to the caller alone.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../core/state_blob.hpp"

namespace tqllm {

struct StateLocation {
    enum class Where : uint8_t { None, Here, Peer };
    Where where = Where::None;
    int box = -1;                    // Peer: the owning box
};

class ConversationStateStore {
public:
    enum class Put : uint8_t {
        Stored,                      // kept in a free space or one taken from another conversation
        Replaced,                    // the conversation's older state was replaced
        NoRoom,                      // every other state protected, or the byte budget cannot be met
        Refused,                     // not keepable here: larger than the byte budget, a Device part, or another box owns it
    };
    struct Stats { uint64_t stored = 0, replaced = 0, taken = 0, dead_overwritten = 0, live_evicted = 0, no_room = 0, refused = 0, fetched = 0, denied = 0; size_t bytes = 0; };
    virtual ~ConversationStateStore() = default;
    virtual const char * name() const = 0;
    // keeps the conversation's state (its newest replaces an older one)
    virtual Put put(StateBlob blob) = 0;
    // here, on another box (through the directory), or none
    virtual StateLocation where(uint64_t conv) const = 0;
    // the state moved out to the caller; nullptr: none, another entity (denied), or a failed fetch. A peer backend copies
    // first and the owner drops its state only after the transfer is confirmed, so a failed fetch leaves it intact.
    virtual std::unique_ptr<StateBlob> take(uint64_t conv, const std::string & entity) = 0;
    // metadata of the states kept here, newest first (the engine picks the best prefix match)
    virtual void for_each(const std::function<void(const StateMeta &)> & fn) const = 0;
    virtual void protect(uint64_t conv) = 0;   // a request for it is queued: not evicted until taken or released
    virtual void release(uint64_t conv) = 0;
    virtual void retire(uint64_t conv) = 0;    // the conversation ended: its state goes first when room is needed
    virtual bool evict(uint64_t conv) = 0;
    virtual Stats stats() const = 0;
    // the byte limit from now on (0: none); states above it go at the next put, in the park's order
    virtual void set_byte_budget(size_t bytes) = 0;
};

// Which box holds each conversation. Boxes call heartbeat() while alive; an owner not seen for stale_after (options of
// the backend) no longer owns anything: owner() reports -1 and a claim by another box succeeds. The directory records;
// the router decides where a turn runs (fetch vs re-prefill priced from positions / bytes per box pair).
class StateDirectory {
public:
    struct Entry { int box = -1; uint32_t positions = 0; size_t bytes = 0; std::string layout; };
    virtual ~StateDirectory() = default;
    virtual const char * name() const = 0;
    virtual int owner(uint64_t conv) const = 0;                          // the live box holding it, -1: none
    virtual Entry entry(uint64_t conv) const = 0;                        // box -1: none (or a stale owner)
    virtual bool claim(uint64_t conv, int box) = 0;                      // box keeps it now; false: another live box owns it
    virtual void release(uint64_t conv, int box) = 0;                    // box no longer holds it (taken, evicted)
    virtual void announce(uint64_t conv, int box, uint32_t positions, size_t bytes, const std::string & layout) = 0;
    virtual void heartbeat(int box) = 0;
};

struct StateStoreOptions {
    int spaces = 0;                  // conversations kept (> 0)
    size_t byte_budget = 0;          // 0: no byte limit (spaces only, as the prompt park)
    int box = 0;                     // this box in the directory
    StateDirectory * directory = nullptr;  // optional; not owned
};
// "local"; throws std::invalid_argument for an unknown backend or spaces < 1
std::unique_ptr<ConversationStateStore> make_state_store(const std::string & backend, const StateStoreOptions & options);

// A blob as one file (host parts only): magic, format version, meta, then each part's kind, memory, layer, bytes and data.
// read_state_blob throws std::runtime_error naming what is wrong (not a state file, another version, truncated).
void write_state_blob(const std::string & path, const StateBlob & blob);
StateBlob read_state_blob(const std::string & path);
// The same bytes through a sink / source (a socket): sink(p, n) / source(p, n) return false on failure; encoded_size is the
// sink's total. decode throws std::runtime_error (what: "<where>: ...") and stops past max_bytes.
size_t encoded_size(const StateBlob & blob);
bool encode_state_blob(const StateBlob & blob, const std::function<bool(const void *, size_t)> & sink);
StateBlob decode_state_blob(const std::function<bool(void *, size_t)> & source, size_t max_bytes, const std::string & where);
// The layout check of an import: the first '|'-separated field of `want` that differs in `got`, as "field: got X, want Y";
// empty when equal (fields are "name=value")
std::string layout_mismatch(const std::string & got, const std::string & want);

using StateClock = std::function<std::chrono::steady_clock::time_point()>;
// "local" (one process); clock: steady_clock::now unless given (tests)
std::unique_ptr<StateDirectory> make_state_directory(const std::string & backend, std::chrono::milliseconds stale_after = std::chrono::seconds(30),
                                                     StateClock clock = {});

}  // namespace tqllm
