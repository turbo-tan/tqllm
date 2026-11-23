#pragma once
// A kept conversation state as opaque parts (src/kv/conversation_state.hpp stores and moves them; each engine's serialiser
// alone reads the bytes). Here in core so the Engine interface can name it.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tqllm {

// one piece of a state, written and read only by the engine's serialiser
struct StatePart {
    enum class Kind : uint8_t { KvRows, Recurrent, Drafter, Engine };
    enum class Memory : uint8_t { Host, Device };  // Device: for a GPU-direct backend later; the local store keeps Host parts
    Kind kind = Kind::Engine;
    Memory memory = Memory::Host;
    int32_t layer = -1;              // KvRows / Recurrent: the layer; -1 otherwise
    std::shared_ptr<void> data;      // owns the bytes (pinned host buffer, received copy, ...)
    size_t bytes = 0;
};

struct StateMeta {
    uint64_t conv = 0;               // conversation id
    uint32_t turn = 0;               // the conversation turn this state ends
    uint32_t positions = 0;          // tokens covered: K/V rows [0, positions), recurrent state at positions
    std::vector<uint32_t> tokens;    // the tokens a next prompt must start with (the engine says how many: Flash-Next adds the one at positions)
    std::string layout;              // serialiser layout id (model file hash, engine, rank / split, formats); import refuses another
    std::string entity;              // entitlement / legal entity tag: handed out only to a caller with the same tag
};

struct StateBlob {
    StateMeta meta;
    std::vector<StatePart> parts;
    size_t bytes() const { size_t n = 0; for (const auto & p : parts) n += p.bytes; return n; }
};

}  // namespace tqllm
