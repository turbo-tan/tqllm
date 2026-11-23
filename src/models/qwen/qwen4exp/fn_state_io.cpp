// Flash-Next conversation state <-> StateBlob (src/kv/conversation_state.hpp): a slot's prompt-cache snapshot at snap_ = p,
// i.e. what a next turn reuses. Parts, in this order (an absent piece is a part of 0 bytes):
//   per trunk layer l: GDN state, GDN conv history, PLE rows (the snapshot copies), K rows [0, p), V rows [0, p), QSA cells
//   the MTP block (layer = layers): K rows, V rows, QSA cells
//   the PLE token history [0, p) (int32, layer -1)
// QSA cells: int32 {cells, blocks, normalized blocks} then raw [cells][dim], pooled [blocks][dim], normalized [nb][dim] (f32).
// meta.tokens = the slot's tokens [0, p] (the reuse check includes the token at p: the MTP pair p - 1 holds it).
// The layout names the format version, the model files, the rank / layer range and every shape that sizes a part; an
// import refuses any difference, naming the field. Copies go through one pinned host block per state on a copy stream.
#include "fn_engine.hpp"
#include "conversation_state.hpp"
#include "device_free_cuda.hpp"

#include <filesystem>
#include <mutex>

namespace tqllm::flashnext {
namespace {

// pinned host blocks of exported states, reused (an export takes the smallest that fits); freed with the engine
struct PinnedPool {
    std::mutex m; std::vector<std::pair<size_t, char *>> free;
    char * take(size_t n, size_t & cap) {
        std::lock_guard<std::mutex> g(m);
        size_t pick = free.size();
        for (size_t i = 0; i < free.size(); ++i) if (free[i].first >= n && (pick == free.size() || free[i].first < free[pick].first)) pick = i;
        if (pick < free.size()) { auto r = free[pick]; free.erase(free.begin() + long(pick)); cap = r.first; return r.second; }
        while (!free.empty()) { cudaFreeHost(free.front().second); free.erase(free.begin()); }  // none fits: the spare goes (unified memory)
        char * p = nullptr; ck(cudaMallocHost(&p, std::max<size_t>(n, 1))); cap = std::max<size_t>(n, 1);
        return p;
    }
    void give(char * p, size_t cap) {  // one spare block kept for the next export; more go back to the system
        std::lock_guard<std::mutex> g(m);
        if (!free.empty()) { cudaFreeHost(p); return; }
        free.push_back({cap, p});
    }
    size_t spare() { std::lock_guard<std::mutex> g(m); size_t n = 0; for (auto & f : free) n += f.first; return n; }
    ~PinnedPool() { for (auto & f : free) cudaFreeHost(f.second); }
};
std::shared_ptr<PinnedPool> pinned_pool() { static auto p = std::make_shared<PinnedPool>(); return p; }
cudaStream_t copy_stream() { static cudaStream_t s = [] { cudaStream_t q = nullptr; ck(cudaStreamCreateWithFlags(&q, cudaStreamNonBlocking)); return q; }(); return s; }

constexpr int kQsaDim = 128, kQsaRatio = 4, kQsaTop = 2048, kQsaHeads = 4;
size_t qsa_bytes(int cells, int nb) { const int blocks = cells / kQsaRatio; return 12 + size_t(cells + blocks + nb) * kQsaDim * sizeof(float); }

}  // namespace

std::string FlashNextEngine::state_layout(int p) const {
    const auto & cp = *f_.cp; const auto & c = cp.config;
    std::string model;
    auto file = [&](const std::string & path) {
        std::error_code ec; const auto n = std::filesystem::file_size(path, ec);
        model += (model.empty() ? "" : ",") + std::filesystem::path(path).filename().string() + ":" + (ec ? std::string("?") : std::to_string(n));
    };
    for (const auto & g : f_.ggufs) file(g);
    if (!f_.mtp.empty()) file(f_.mtp);
    const int rank = cp.run.split ? cp.run.split->rank() : 0, world = cp.run.split ? cp.run.split->world() : 1;
    const uint32_t lo = cp.run.split ? cp.run.lr.lo : 0, hi = cp.run.split ? cp.run.lr.hi : uint32_t(c.layers);
    const int channels = (2 * cp.key_heads + cp.value_heads) * cp.state_dim, hist = (c.ple_kernel - 1) * c.ple.ngram;
    return "fmt=fnstate1|engine=flashnext|model=" + model + "|rank=" + std::to_string(rank) + "/" + std::to_string(world) +
           "|layers=" + std::to_string(lo) + ".." + std::to_string(hi) + "/" + std::to_string(c.layers) +
           "|gdn_state=f32[" + std::to_string(cp.value_heads) + "][" + std::to_string(cp.state_dim) + "][" + std::to_string(cp.state_dim) + "]" +
           "|gdn_conv=f32[" + std::to_string(cp.conv_kernel - 1) + "][" + std::to_string(channels) + "]" +
           "|ple=f32[" + std::to_string(hist) + "][" + std::to_string(c.hidden * c.hc) + "]" +
           "|kv=bf16[" + std::to_string(p) + "][" + std::to_string(cp.attention_kvheads) + "][" + std::to_string(cp.attention_dim) + "]" +
           "|qsa=f32[cells " + std::to_string(p) + "][" + std::to_string(kQsaDim) + "]/" + std::to_string(kQsaRatio) +
           "|history=i32[" + std::to_string(p) + "]";
}

std::unique_ptr<StateBlob> FlashNextEngine::export_state(int slot) {
    const auto lk = ep_lock();
    if (!sharing() || slot < 0 || slot >= int(slots_.size()) || slots_[size_t(slot)].serving_) return nullptr;
    return export_from(slots_[size_t(slot)]);
}

std::unique_ptr<StateBlob> FlashNextEngine::export_from(SlotState & s) {
    if (!s.model_ || s.snap_ <= 0 || s.seq_.size() <= size_t(s.snap_)) return nullptr;
    const auto & cp = *f_.cp; DecodeModel & m = *s.model_; const int p = s.snap_, L = cp.config.layers;
    const size_t row = size_t(cp.attention_kvheads) * cp.attention_dim * sizeof(__nv_bfloat16);
    ck(cudaDeviceSynchronize());  // once per export: the QSA host mirrors' queued copies have landed
    // the parts' sizes, then one pinned block for all of them
    struct Src { StatePart::Kind kind; int layer; size_t bytes; std::function<void(char *, cudaStream_t)> fill; };
    std::vector<Src> src;
    auto dev = [&](StatePart::Kind k, int l, const Device<float> * d) {
        const float * ptr = d ? d->p : nullptr; const size_t n = d ? d->count * sizeof(float) : 0;
        src.push_back({k, l, n, [ptr, n](char * h, cudaStream_t st) { if (n) ck(cudaMemcpyAsync(h, ptr, n, cudaMemcpyDeviceToHost, st)); }});
    };
    auto kv = [&](int l, const AttentionState & a) {
        for (const KvBuf * b : {a.k.get(), a.v.get()}) {
            const size_t n = b ? size_t(p) * row : 0; const void * ptr = b ? b->p : nullptr;
            require(!b || a.len >= p, "conversation state export: attention rows end before the kept position");
            src.push_back({StatePart::Kind::KvRows, l, n, [ptr, n](char * h, cudaStream_t st) { if (n) ck(cudaMemcpyAsync(h, ptr, n, cudaMemcpyDeviceToHost, st)); }});
        }
        const tqllm_cuda::QsaIndexer * q = a.qsa.get();
        const int nb = q ? std::min(q->keys().normalized_blocks(), p / kQsaRatio) : 0;
        require(!q || (q->size() >= p && q->keys().dim() == kQsaDim && q->keys().ratio() == kQsaRatio), "conversation state export: QSA cells end before the kept position");
        src.push_back({StatePart::Kind::Engine, l, q ? qsa_bytes(p, nb) : 0, [q, p, nb](char * h, cudaStream_t st) {
            if (!q) return;
            const int32_t hd[3] = {p, p / kQsaRatio, nb}; std::memcpy(h, hd, sizeof hd);
            float * raw = reinterpret_cast<float *>(h + 12), * pooled = raw + size_t(p) * kQsaDim, * norm = pooled + size_t(p / kQsaRatio) * kQsaDim;
            q->keys().export_cells(p, raw, pooled, norm, st);
        }});
    };
    for (int l = 0; l < L; ++l) {
        auto at = [l](const std::vector<std::unique_ptr<Device<float>>> & v) { return l < int(v.size()) ? v[size_t(l)].get() : nullptr; };
        dev(StatePart::Kind::Recurrent, l, at(s.snap_state_));
        dev(StatePart::Kind::Recurrent, l, at(s.snap_conv_));
        dev(StatePart::Kind::Engine, l, at(s.snap_ple_));
        kv(l, m.att[size_t(l)]);
    }
    kv(L, m.mtp_att);
    const auto & ht = m.history.tokens();
    require(ht.size() >= size_t(p), "conversation state export: the PLE history ends before the kept position");
    src.push_back({StatePart::Kind::Engine, -1, size_t(p) * sizeof(int32_t), [&ht, p](char * h, cudaStream_t) { std::memcpy(h, ht.data(), size_t(p) * sizeof(int32_t)); }});
    size_t total = 0; for (const auto & x : src) total += (x.bytes + 255) / 256 * 256;
    auto pool = pinned_pool(); size_t cap = 0; char * block = pool->take(total, cap);
    std::shared_ptr<char> owner(block, [pool, cap](char * q) { pool->give(q, cap); });
    auto blob = std::make_unique<StateBlob>();
    const cudaStream_t st = copy_stream(); size_t off = 0;
    for (auto & x : src) {
        x.fill(block + off, st);
        StatePart part; part.kind = x.kind; part.layer = x.layer; part.bytes = x.bytes;
        part.data = std::shared_ptr<void>(owner, block + off);
        blob->parts.push_back(std::move(part));
        off += (x.bytes + 255) / 256 * 256;
    }
    ck(cudaStreamSynchronize(st));
    blob->meta.conv = s.conv_;
    blob->meta.positions = uint32_t(p);
    blob->meta.tokens.assign(s.seq_.begin(), s.seq_.begin() + p + 1);
    blob->meta.layout = state_layout(p);
    return blob;
}

std::string FlashNextEngine::import_state(int slot, const StateBlob & b) {
    const auto lk = ep_lock();
    if (!sharing()) return "conversation state import: needs --prompt-cache on and --np > 1";
    if (slot < 0 || slot >= int(slots_.size())) return "conversation state import: no slot " + std::to_string(slot);
    SlotState & s = slots_[size_t(slot)];
    if (s.serving_) return "conversation state import: slot " + std::to_string(slot) + " is serving a request";
    return import_into(s, b);
}

std::string FlashNextEngine::import_into(SlotState & s, const StateBlob & b) {
    const int p = int(b.meta.positions);
    if (p <= 0 || b.meta.tokens.size() != size_t(p) + 1) return "conversation state import: tokens: got " + std::to_string(b.meta.tokens.size()) + ", want " + std::to_string(p + 1);
    if (p + 1 > int(slot_context())) return "conversation state import: positions: got " + std::to_string(p) + ", the slot holds " + std::to_string(slot_context());
    if (const auto why = layout_mismatch(b.meta.layout, state_layout(p)); !why.empty()) return "conversation state import: " + why;
    auto & cp = *f_.cp; const auto & c = cp.config; const int L = c.layers;
    if (b.parts.size() != size_t(6 * L + 4)) return "conversation state import: parts: got " + std::to_string(b.parts.size()) + ", want " + std::to_string(6 * L + 4);
    for (const auto & part : b.parts) if (part.memory != StatePart::Memory::Host) return "conversation state import: a device part (host parts only)";
    const size_t row = size_t(cp.attention_kvheads) * cp.attention_dim * sizeof(__nv_bfloat16);
    const size_t gdn_state = size_t(cp.value_heads) * cp.state_dim * cp.state_dim * sizeof(float);
    const size_t gdn_conv = size_t(cp.conv_kernel - 1) * (2 * cp.key_heads + cp.value_heads) * cp.state_dim * sizeof(float);
    const size_t ple = size_t((c.ple_kernel - 1) * c.ple.ngram) * size_t(c.hidden * c.hc) * sizeof(float);
    auto bad = [](const std::string & what, int l, size_t got, size_t want) {
        return "conversation state import: layer " + std::to_string(l) + " " + what + ": got " + std::to_string(got) + " bytes, want 0 or " + std::to_string(want);
    };
    for (int l = 0; l <= L; ++l) {  // every size before anything is replaced
        const size_t base = size_t(6) * size_t(l);
        if (l < L) {
            if (b.parts[base].bytes && b.parts[base].bytes != gdn_state) return bad("GDN state", l, b.parts[base].bytes, gdn_state);
            if (b.parts[base + 1].bytes && b.parts[base + 1].bytes != gdn_conv) return bad("GDN conv", l, b.parts[base + 1].bytes, gdn_conv);
            if (b.parts[base + 2].bytes && b.parts[base + 2].bytes != ple) return bad("PLE rows", l, b.parts[base + 2].bytes, ple);
        }
        const size_t k = l < L ? base + 3 : base;
        for (size_t j = k; j < k + 2; ++j) if (b.parts[j].bytes && b.parts[j].bytes != size_t(p) * row) return bad("K/V rows", l, b.parts[j].bytes, size_t(p) * row);
        if (const auto & q = b.parts[k + 2]; q.bytes) {
            const auto * hd = static_cast<const int32_t *>(q.data.get());
            if (q.bytes < 12 || hd[0] != p || hd[1] != p / kQsaRatio || hd[2] < 0 || hd[2] > hd[1] || q.bytes != qsa_bytes(p, hd[2]))
                return "conversation state import: layer " + std::to_string(l) + " QSA cells do not match the kept position";
        }
    }
    if (b.parts.back().bytes != size_t(p) * sizeof(int32_t)) return "conversation state import: PLE history: got " + std::to_string(b.parts.back().bytes) + " bytes, want " + std::to_string(size_t(p) * 4);
    // a fresh sequence holding the kept state, and the slot's snapshot of it: the next request resumes through reuse() and
    // restore_snapshot as for a state this slot kept itself
    ck(cudaDeviceSynchronize());
    s.model_.reset();
    s.model_ = std::make_unique<DecodeModel>(cp, *f_.fixtures, *f_.cache, size_t(slot_context()));
    s.model_->mtp_experts = f_.mtp_cache.get();
    DecodeModel & m = *s.model_;
    const cudaStream_t st = copy_stream();
    PersistentAlloc keep;
    auto put = [&](std::unique_ptr<Device<float>> & dst, std::unique_ptr<Device<float>> * live, const StatePart & part) {
        if (!part.bytes) { dst.reset(); if (live) live->reset(); return; }
        dst = std::make_unique<Device<float>>(part.bytes / sizeof(float));
        ck(cudaMemcpyAsync(dst->p, part.data.get(), part.bytes, cudaMemcpyHostToDevice, st));
        if (live) *live = std::make_unique<Device<float>>(part.bytes / sizeof(float));  // the sequence's own buffer: restore_snapshot copies into it
    };
    auto att = [&](AttentionState & a, size_t k) {
        const int capn = kv_cap(a, cp.rt.kv_capacity);
        if (capn < p) throw std::runtime_error("conversation state import: the attention capacity is below the kept position");
        if (b.parts[k].bytes) {
            const size_t most = size_t(std::max(capn, cp.rt.kv_capacity_full)) * row;
            a.k = std::make_unique<KvBuf>(size_t(capn) * row, most); a.v = std::make_unique<KvBuf>(size_t(capn) * row, most);
            ck(cudaMemcpyAsync(a.k->p, b.parts[k].data.get(), b.parts[k].bytes, cudaMemcpyHostToDevice, st));
            ck(cudaMemcpyAsync(a.v->p, b.parts[k + 1].data.get(), b.parts[k + 1].bytes, cudaMemcpyHostToDevice, st));
            a.len = p;
        }
        if (const auto & q = b.parts[k + 2]; q.bytes) {
            const auto * hd = static_cast<const int32_t *>(q.data.get());
            const float * raw = reinterpret_cast<const float *>(static_cast<const char *>(q.data.get()) + 12);
            const float * pooled = raw + size_t(p) * kQsaDim, * norm = pooled + size_t(hd[1]) * kQsaDim;
            a.qsa = std::make_unique<tqllm_cuda::QsaIndexer>(capn, kQsaRatio, kQsaTop, kQsaHeads, kQsaDim, std::min(cp.rotary, kQsaDim), cp.theta, float(cp.epsilon));
            a.qsa->keys().import_cells(p, hd[2], raw, pooled, norm, st);
        }
    };
    s.snap_state_.clear(); s.snap_conv_.clear(); s.snap_ple_.clear();
    s.snap_state_.resize(size_t(L)); s.snap_conv_.resize(size_t(L)); s.snap_ple_.resize(size_t(L));
    for (int l = 0; l < L; ++l) {
        const size_t base = size_t(6) * size_t(l);
        put(s.snap_state_[size_t(l)], &m.rec[size_t(l)].state, b.parts[base]);
        put(s.snap_conv_[size_t(l)], &m.rec[size_t(l)].history, b.parts[base + 1]);
        put(s.snap_ple_[size_t(l)], &m.ple_past[size_t(l)], b.parts[base + 2]);
        att(m.att[size_t(l)], base + 3);
    }
    att(m.mtp_att, size_t(6) * size_t(L));
    const auto * ht = static_cast<const int32_t *>(b.parts.back().data.get());
    m.history.assign(std::vector<int32_t>(ht, ht + p));
    m.position = p;
    ck(cudaStreamSynchronize(st));
    s.seq_.assign(b.meta.tokens.begin(), b.meta.tokens.end());
    s.snap_ = p; s.live_ = 0; s.snap_rows_ = 0;
    s.conv_ = b.meta.conv ? b.meta.conv : ++conv_clock_;
    return {};
}

}  // namespace tqllm::flashnext

namespace tqllm::flashnext {

void FlashNextEngine::park_init() {
    const int want = f_.cp->run.prompt_park;
    if (!sharing() || want == 0) return;
    // GB10: pinned host memory is the unified memory the expert banks use, so the park has a byte cap besides its count
    StateStoreOptions o; o.spaces = want > 0 ? want : 4 * int(slots_.size()); o.byte_budget = 1;  // set at each park (park_budget)
    park_ = make_state_store("local", o);
    std::fprintf(stderr, "tqllm: prompt park: up to %d kept conversations outside the slots, pinned host memory min(4 GB, (free - 20 GB) / 2) at each park\n", o.spaces);
}

int FlashNextEngine::park_best(const std::vector<uint32_t> & prompt, uint64_t * conv) const {
    int best = 0;
    if (conv) *conv = 0;
    if (!park_) return 0;
    park_->for_each([&](const StateMeta & m) {
        const int n = reusable_prefix(m.tokens, int(m.positions), prompt);
        if (n > best) { best = n; if (conv) *conv = m.conv; }
    });
    return best;
}

void FlashNextEngine::park_note(const char * what, ConversationStateStore::Put put) const {
    static const bool each = tqllm::debug_on("park");
    const auto st = park_->stats();
    const bool notable = put == ConversationStateStore::Put::NoRoom || put == ConversationStateStore::Put::Refused;
    if (!each && !notable && (st.stored + st.replaced) % 32 != 1) return;  // every 32nd park, and every one it could not keep
    int n = 0; park_->for_each([&](const StateMeta &) { ++n; });
    static const char * names[] = {"stored", "replaced", "no room", "refused"};
    std::fprintf(stderr, "tqllm: prompt park: %s (%s): %d conversations, %.2f GB; stored %llu, taken %llu, evicted %llu, no room %llu\n",
                 what, names[int(put)], n, double(st.bytes) / (1 << 30), (unsigned long long) (st.stored + st.replaced),
                 (unsigned long long) st.taken, (unsigned long long) (st.dead_overwritten + st.live_evicted), (unsigned long long) st.no_room);
}

// The park's byte budget now: what it holds plus half of the free memory above 20 GB (a GB10's unified memory is the
// device's: at the edge one hung and needed a reboot), at most 4 GB, the export pool's spare block counted in it
size_t FlashNextEngine::park_budget() const {
    constexpr size_t kCap = size_t(4) << 30, kKeep = size_t(20) << 30;
    const size_t freeb = tqllm::device_free_bytes(), spare = pinned_pool()->spare(), held = park_->stats().bytes + spare;
    const size_t extra = freeb > kKeep ? (freeb - kKeep) / 2 : 0;
    const size_t budget = std::min(kCap, held + extra);
    return budget > spare ? budget - spare : 0;
}

// before a request on the active slot: if a parked conversation shares more of the prompt than the slot's own kept state,
// the slot's state is parked and that one is imported into it; a new conversation parks the slot's old one
void FlashNextEngine::park_swap(const std::vector<uint32_t> & prompt) {
    if (!park_) { static bool tried = false; if (tried) return; tried = true; park_init(); if (!park_) return; }
    const int own = reuse(S(), prompt);
    uint64_t conv = 0;
    const int best = park_best(prompt, &conv);
    auto park_out = [&] {
        if (!S().model_ || S().snap_ <= 0) return;
        const size_t budget = park_budget();
        static size_t said = SIZE_MAX;
        if (budget != said && (budget == 0 || said == SIZE_MAX || budget < said / 2 || budget > said * 2)) {  // on large changes only
            said = budget;
            std::fprintf(stderr, "tqllm: prompt park: budget %.2f GB (%.1f GB free, %.2f GB held)\n", double(budget) / (1 << 30),
                         double(tqllm::device_free_bytes()) / (1 << 30), double(park_->stats().bytes) / (1 << 30));
        }
        park_->set_byte_budget(budget ? budget : 1);
        if (budget < size_t(64) << 20) return;  // no room above the 20 GB reserve: the state is dropped, as without a park
        auto b = export_from(S());
        if (!b) return;
        park_note("parked", park_->put(std::move(*b)));
    };
    static const bool trace = tqllm::debug_on("park");
    if (trace) std::fprintf(stderr, "tqllm: park: slot %d (conv %llu, snap %d) own %d parked best %d (conv %llu)\n", cur_, (unsigned long long) S().conv_, S().snap_, own, best, (unsigned long long) conv);
    if (!conv || best <= own) {
        if (!own) park_out();
        return;
    }
    park_->protect(conv);  // the slot's own state, parked next, must not take this one's space
    park_out();
    auto b = park_->take(conv, "");
    if (!b) { park_->release(conv); return; }
    if (const auto why = import_into(S(), *b); !why.empty()) std::fprintf(stderr, "tqllm: prompt park: %s (the prompt prefills in full)\n", why.c_str());
}

}  // namespace tqllm::flashnext

namespace tqllm::flashnext {

std::unique_ptr<StateBlob> FlashNextEngine::export_state_for(const std::vector<uint32_t> & prompt, const std::string & entity) {
    const auto lk = ep_lock();
    if (!sharing()) return nullptr;
    int best = 0, slot = -1;
    for (size_t s = 0; s < slots_.size(); ++s)
        if (!slots_[s].serving_) { const int n = reuse(slots_[s], prompt); if (n > best) { best = n; slot = int(s); } }
    uint64_t conv = 0;
    const int parked = park_best(prompt, &conv);
    if (parked > best && conv) {  // a copy of the parked state (the parts are shared, read only); the park keeps it
        auto b = park_->take(conv, entity);
        if (!b) return nullptr;  // another entity's state
        auto copy = std::make_unique<StateBlob>(*b);
        park_note("lent", park_->put(std::move(*b)));
        return copy;
    }
    if (slot < 0 || !entity.empty()) return nullptr;  // a slot's state has no entity tag: only an untagged request gets it
    auto b = export_from(slots_[size_t(slot)]);
    if (b && !b->meta.conv) b->meta.conv = slots_[size_t(slot)].conv_;
    return b;
}

std::string FlashNextEngine::adopt_state(StateBlob blob) {
    const auto lk = ep_lock();
    if (!sharing()) return "conversation state adopt: needs --prompt-cache on and --np > 1";
    if (!park_) { park_init(); if (!park_) return "conversation state adopt: the prompt park is off (--prompt-park 0)"; }
    const int p = int(blob.meta.positions);
    if (p <= 0 || blob.meta.tokens.size() != size_t(p) + 1) return "conversation state adopt: tokens do not match the positions";
    if (const auto why = layout_mismatch(blob.meta.layout, state_layout(p)); !why.empty()) return "conversation state adopt: " + why;
    blob.meta.conv = ++conv_clock_;  // conversation ids are per box: the owner's id stays with the owner (its ack names it)
    park_->set_byte_budget(std::max<size_t>(park_budget(), 1));  // a fetched state counts against the same budget
    const auto put = park_->put(std::move(blob));
    park_note("fetched", put);
    return put == ConversationStateStore::Put::Stored || put == ConversationStateStore::Put::Replaced ? std::string() : "conversation state adopt: the park has no room";
}

void FlashNextEngine::release_state(uint64_t conv) {
    const auto lk = ep_lock();
    if (!conv) return;
    if (park_) park_->evict(conv);
    // a slot that lent its kept state drops it too: one box holds a conversation's state, so a later turn finds the
    // newest one (fetched on) instead of resuming from this older, shorter prefix
    static const bool trace = tqllm::debug_on("park");
    for (size_t i = 0; i < slots_.size(); ++i) {
        auto & s = slots_[i];
        if (!s.serving_ && s.conv_ == conv) { if (trace) std::fprintf(stderr, "tqllm: park: released conv %llu from slot %zu\n", (unsigned long long) conv, i); s.snap_ = s.live_ = 0; s.seq_.clear(); s.model_.reset(); s.conv_ = 0; }
    }
}

}  // namespace tqllm::flashnext
