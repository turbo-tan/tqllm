// Conversation state store and directory backends (src/kv/conversation_state.hpp). "local": the prompt park's rules
// (PromptPark chooses every space, so the two cannot drift) holding blobs in host memory, plus an optional byte budget.
#include "conversation_state.hpp"
#include "prompt_park.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

namespace tqllm {
namespace {

class LocalStore final : public ConversationStateStore {
public:
    explicit LocalStore(const StateStoreOptions & o) : park_(o.spaces), slots_(size_t(o.spaces)), seq_(size_t(o.spaces), 0), opt_(o) {}
    const char * name() const override { return "local"; }

    Put put(StateBlob blob) override {
        std::lock_guard<std::mutex> g(m_);
        const uint64_t conv = blob.meta.conv;
        const size_t need = blob.bytes();
        const bool device = std::any_of(blob.parts.begin(), blob.parts.end(), [](const StatePart & p) { return p.memory != StatePart::Memory::Host; });
        if (device || (opt_.byte_budget && need > opt_.byte_budget) ||
            (opt_.directory && !opt_.directory->claim(conv, opt_.box))) { ++st_.refused; return Put::Refused; }
        const bool had = park_.where(conv) >= 0;
        if (opt_.byte_budget) {  // make room by the park's own order (dead first, then live unprotected), never the conversation itself
            size_t freed_own = had ? slots_[size_t(park_.where(conv))]->bytes() : 0;
            while (st_.bytes - freed_own + need > opt_.byte_budget) {
                const int v = park_.victim(conv);
                if (v < 0) { ++st_.no_room; release_if_new(conv, had); return Put::NoRoom; }
                drop_space(v);
            }
        }
        const int s = park_.park(conv);
        if (s < 0) { ++st_.no_room; release_if_new(conv, had); return Put::NoRoom; }
        if (slots_[size_t(s)]) {  // replaced its own older state, or took another conversation's space
            const uint64_t was = slots_[size_t(s)]->meta.conv;
            if (was != conv) { (retired_.count(was) ? st_.dead_overwritten : st_.live_evicted)++; forget(was); }
            st_.bytes -= slots_[size_t(s)]->bytes();
        }
        retired_.erase(conv);
        st_.bytes += need;
        if (opt_.directory) opt_.directory->announce(conv, opt_.box, blob.meta.positions, need, blob.meta.layout);
        slots_[size_t(s)] = std::make_unique<StateBlob>(std::move(blob));
        seq_[size_t(s)] = ++clock_;
        if (had) { ++st_.replaced; return Put::Replaced; }
        ++st_.stored;
        return Put::Stored;
    }

    StateLocation where(uint64_t conv) const override {
        std::lock_guard<std::mutex> g(m_);
        if (park_.where(conv) >= 0) return {StateLocation::Where::Here, opt_.box};
        if (opt_.directory) {
            const int b = opt_.directory->owner(conv);
            if (b >= 0 && b != opt_.box) return {StateLocation::Where::Peer, b};
        }
        return {};
    }

    std::unique_ptr<StateBlob> take(uint64_t conv, const std::string & entity) override {
        std::lock_guard<std::mutex> g(m_);
        const int s = park_.where(conv);
        if (s < 0) return nullptr;  // a peer backend fetches here; the local one keeps only this box's states
        if (slots_[size_t(s)]->meta.entity != entity) { ++st_.denied; return nullptr; }
        park_.pickup(conv);
        auto out = std::move(slots_[size_t(s)]);
        st_.bytes -= out->bytes();
        forget(conv);
        ++st_.taken;
        return out;
    }

    void for_each(const std::function<void(const StateMeta &)> & fn) const override {
        std::lock_guard<std::mutex> g(m_);
        std::vector<size_t> order;
        for (size_t i = 0; i < slots_.size(); ++i) if (slots_[i]) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return seq_[a] > seq_[b]; });
        for (size_t i : order) fn(slots_[i]->meta);
    }
    void protect(uint64_t conv) override { std::lock_guard<std::mutex> g(m_); park_.protect(conv); }
    void release(uint64_t conv) override { std::lock_guard<std::mutex> g(m_); park_.release(conv); }
    void retire(uint64_t conv) override {
        std::lock_guard<std::mutex> g(m_);
        if (park_.where(conv) >= 0) { park_.retire(conv); retired_.insert(conv); }
    }
    bool evict(uint64_t conv) override {
        std::lock_guard<std::mutex> g(m_);
        const int s = park_.where(conv);
        if (s < 0) return false;
        drop_space(s);
        return true;
    }
    Stats stats() const override { std::lock_guard<std::mutex> g(m_); return st_; }
    void set_byte_budget(size_t bytes) override { std::lock_guard<std::mutex> g(m_); opt_.byte_budget = bytes; }

private:
    void drop_space(int s) {
        const uint64_t conv = slots_[size_t(s)]->meta.conv;
        (retired_.count(conv) ? st_.dead_overwritten : st_.live_evicted)++;
        st_.bytes -= slots_[size_t(s)]->bytes();
        park_.drop(conv);
        slots_[size_t(s)].reset();
        forget(conv);
    }
    void forget(uint64_t conv) { retired_.erase(conv); if (opt_.directory) opt_.directory->release(conv, opt_.box); }
    void release_if_new(uint64_t conv, bool had) { if (!had && opt_.directory) opt_.directory->release(conv, opt_.box); }

    mutable std::mutex m_;
    PromptPark park_;
    std::vector<std::unique_ptr<StateBlob>> slots_;
    std::vector<uint64_t> seq_;
    std::set<uint64_t> retired_;
    uint64_t clock_ = 0;
    StateStoreOptions opt_;
    Stats st_;
};

class LocalDirectory final : public StateDirectory {
public:
    LocalDirectory(std::chrono::milliseconds stale, StateClock clock)
        : stale_(stale), clock_(clock ? std::move(clock) : StateClock([] { return std::chrono::steady_clock::now(); })) {}
    const char * name() const override { return "local"; }
    int owner(uint64_t conv) const override { return entry(conv).box; }
    Entry entry(uint64_t conv) const override {
        std::lock_guard<std::mutex> g(m_);
        const auto it = map_.find(conv);
        return it != map_.end() && live(it->second.box) ? it->second : Entry{};
    }
    bool claim(uint64_t conv, int box) override {
        std::lock_guard<std::mutex> g(m_);
        seen_[box] = clock_();
        auto & e = map_[conv];
        if (e.box >= 0 && e.box != box && live(e.box)) return false;
        if (e.box != box) e = Entry{box, 0, 0, {}};
        return true;
    }
    void release(uint64_t conv, int box) override {
        std::lock_guard<std::mutex> g(m_);
        const auto it = map_.find(conv);
        if (it != map_.end() && it->second.box == box) map_.erase(it);
    }
    void announce(uint64_t conv, int box, uint32_t positions, size_t bytes, const std::string & layout) override {
        std::lock_guard<std::mutex> g(m_);
        seen_[box] = clock_();
        auto & e = map_[conv];
        if (e.box >= 0 && e.box != box && live(e.box)) return;  // another live box owns it: its claim stands
        e = Entry{box, positions, bytes, layout};
    }
    void heartbeat(int box) override { std::lock_guard<std::mutex> g(m_); seen_[box] = clock_(); }

private:
    bool live(int box) const {
        const auto it = seen_.find(box);
        return it != seen_.end() && clock_() - it->second <= stale_;
    }
    mutable std::mutex m_;
    std::map<uint64_t, Entry> map_;
    std::map<int, std::chrono::steady_clock::time_point> seen_;
    std::chrono::milliseconds stale_;
    StateClock clock_;
};

}  // namespace

namespace {
constexpr char kMagic[8] = {'T', 'Q', 'S', 'T', 'A', 'T', 'E', '1'};
constexpr uint32_t kFileVersion = 1;
}  // namespace

size_t encoded_size(const StateBlob & b) {
    size_t n = sizeof kMagic + 4 + 8 + 4 + 4 + 8 + b.meta.tokens.size() * 4 + 8 + b.meta.layout.size() + 8 + b.meta.entity.size() + 8;
    for (const auto & p : b.parts) n += 1 + 4 + 8 + p.bytes;
    return n;
}

bool encode_state_blob(const StateBlob & b, const std::function<bool(const void *, size_t)> & sink) {
    std::string h(kMagic, sizeof kMagic);
    auto pod = [&](const auto & v) { h.append(reinterpret_cast<const char *>(&v), sizeof v); };
    auto str = [&](const std::string & s) { pod(uint64_t(s.size())); h += s; };
    pod(kFileVersion); pod(b.meta.conv); pod(b.meta.turn); pod(b.meta.positions);
    pod(uint64_t(b.meta.tokens.size())); h.append(reinterpret_cast<const char *>(b.meta.tokens.data()), b.meta.tokens.size() * 4);
    str(b.meta.layout); str(b.meta.entity); pod(uint64_t(b.parts.size()));
    if (!sink(h.data(), h.size())) return false;
    for (const auto & p : b.parts) {
        if (p.memory != StatePart::Memory::Host) return false;
        std::string ph; auto pp = [&](const auto & v) { ph.append(reinterpret_cast<const char *>(&v), sizeof v); };
        pp(uint8_t(p.kind)); pp(int32_t(p.layer)); pp(uint64_t(p.bytes));
        if (!sink(ph.data(), ph.size()) || (p.bytes && !sink(p.data.get(), p.bytes))) return false;
    }
    return true;
}

StateBlob decode_state_blob(const std::function<bool(void *, size_t)> & source, size_t max_bytes, const std::string & where) {
    size_t used = 0;
    auto read = [&](void * p, size_t n) {
        if ((used += n) > max_bytes) throw std::runtime_error(where + ": larger than the " + std::to_string(max_bytes) + "-byte ceiling");
        if (n && !source(p, n)) throw std::runtime_error(where + ": truncated");
    };
    auto pod = [&](auto & v) { read(&v, sizeof v); };
    auto str = [&](std::string & s) { uint64_t n = 0; pod(n); if (n > (uint64_t(1) << 32)) throw std::runtime_error(where + ": corrupt string length"); s.resize(size_t(n)); read(s.data(), size_t(n)); };
    char magic[8]{}; read(magic, sizeof magic);
    if (std::memcmp(magic, kMagic, sizeof magic)) throw std::runtime_error(where + ": not a state file");
    uint32_t v = 0; pod(v);
    if (v != kFileVersion) throw std::runtime_error(where + ": format version " + std::to_string(v) + ", this build reads " + std::to_string(kFileVersion));
    StateBlob b;
    pod(b.meta.conv); pod(b.meta.turn); pod(b.meta.positions);
    uint64_t nt = 0; pod(nt);
    if (nt > (uint64_t(1) << 31)) throw std::runtime_error(where + ": corrupt token count");
    b.meta.tokens.resize(size_t(nt)); read(b.meta.tokens.data(), size_t(nt) * 4);
    str(b.meta.layout); str(b.meta.entity);
    uint64_t np = 0; pod(np);
    if (np > (uint64_t(1) << 24)) throw std::runtime_error(where + ": corrupt part count");
    for (uint64_t i = 0; i < np; ++i) {
        StatePart p; uint8_t kind = 0; int32_t layer = 0; uint64_t bytes = 0;
        pod(kind); pod(layer); pod(bytes);
        if (bytes > max_bytes) throw std::runtime_error(where + ": part " + std::to_string(i) + " larger than the ceiling");
        p.kind = StatePart::Kind(kind); p.layer = layer; p.bytes = size_t(bytes);
        std::shared_ptr<char> d(new char[p.bytes ? p.bytes : 1], std::default_delete<char[]>());
        read(d.get(), p.bytes);
        p.data = d;
        b.parts.push_back(std::move(p));
    }
    return b;
}

void write_state_blob(const std::string & path, const StateBlob & b) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("conversation state file " + path + ": cannot create");
    if (!encode_state_blob(b, [&](const void * p, size_t n) { f.write(static_cast<const char *>(p), std::streamsize(n)); return bool(f); }))
        throw std::runtime_error("conversation state file " + path + ": write failed (or a device part)");
}

StateBlob read_state_blob(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("conversation state file " + path + ": cannot open");
    return decode_state_blob([&](void * p, size_t n) { f.read(static_cast<char *>(p), std::streamsize(n)); return bool(f); }, SIZE_MAX,
                             "conversation state file " + path);
}

std::string layout_mismatch(const std::string & got, const std::string & want) {
    auto split = [](const std::string & s) { std::vector<std::string> v; size_t a = 0; for (;;) { const size_t b = s.find('|', a); v.push_back(s.substr(a, b - a)); if (b == std::string::npos) return v; a = b + 1; } };
    const auto g = split(got), w = split(want);
    for (size_t i = 0; i < std::max(g.size(), w.size()); ++i) {
        const std::string gi = i < g.size() ? g[i] : "", wi = i < w.size() ? w[i] : "";
        if (gi == wi) continue;
        const std::string name = wi.substr(0, wi.find('=')).empty() ? gi.substr(0, gi.find('=')) : wi.substr(0, wi.find('='));
        auto val = [](const std::string & f) { const size_t e = f.find('='); return e == std::string::npos ? f : f.substr(e + 1); };
        return name + ": got " + (gi.empty() ? "(none)" : val(gi)) + ", want " + (wi.empty() ? "(none)" : val(wi));
    }
    return {};
}

std::unique_ptr<ConversationStateStore> make_state_store(const std::string & backend, const StateStoreOptions & options) {
    if (options.spaces < 1) throw std::invalid_argument("conversation state store: at least one space");
    if (backend == "local") return std::make_unique<LocalStore>(options);
    throw std::invalid_argument("conversation state store: unknown backend '" + backend + "' (local)");
}

std::unique_ptr<StateDirectory> make_state_directory(const std::string & backend, std::chrono::milliseconds stale_after, StateClock clock) {
    if (backend == "local") return std::make_unique<LocalDirectory>(stale_after, std::move(clock));
    throw std::invalid_argument("conversation state directory: unknown backend '" + backend + "' (local)");
}

}  // namespace tqllm
