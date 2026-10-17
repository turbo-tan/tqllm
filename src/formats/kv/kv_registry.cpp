#include "kv_format.hpp"

#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {
namespace formats {  // built-in KV formats; a new format adds one declaration and one call below
void register_kv_dense();
void register_kv_turbo();
void register_kv_tq3();
void register_kv_tq3k_tq2v();
void register_kv_split();
}

namespace {
std::map<std::string, std::unique_ptr<KvFormat>> & table() {
    static std::map<std::string, std::unique_ptr<KvFormat>> t;
    return t;
}
void builtins() {
    static std::once_flag once;
    std::call_once(once, [] {
        formats::register_kv_dense();
        formats::register_kv_turbo();
        formats::register_kv_tq3();
        formats::register_kv_tq3k_tq2v();
        formats::register_kv_split();
    });
}
}  // namespace

void KvFormat::attend_selected(const KvSelected &) const {
    throw std::runtime_error(std::string("KV format ") + name() + " has no selected-position attention");
}

void register_kv_format(const std::string & name, std::unique_ptr<KvFormat> f) {
    if (!table().emplace(name, std::move(f)).second) throw std::runtime_error("KV format " + name + " registered twice");
}

const KvFormat * kv_format(const std::string & name) {
    builtins();
    const auto it = table().find(name);
    if (it == table().end()) throw std::runtime_error("unknown KV format '" + name + "'");
    return it->second.get();
}
}  // namespace tqllm

namespace tqllm {
std::vector<std::string> kv_format_names() {
    builtins();
    std::vector<std::string> v;
    for (const auto & kv : table()) v.push_back(kv.first);
    return v;  // std::map keeps them sorted
}

std::string kv_format_listing(int layers, int kvheads, int head_dim) {
    builtins();
    std::string out;
    for (const auto & kv : table()) {
        const KvFormat & f = *kv.second;
        const KvProvenance p = f.provenance();
        std::string bytes = "n/a (needs head_dim 256)";
        try {
            const double per_tok = double(layers) * double(f.key_bytes(1, kvheads, head_dim) + f.value_bytes(1, kvheads, head_dim));
            bytes = std::to_string(per_tok / 1024.0).substr(0, 6) + " KB/token";
        } catch (const std::exception &) {}
        out += kv.first + "\n  " + p.summary + "\n  size: " + bytes + "; scratch: " +
               (f.needs_context_scratch() ? "context-sized (conservative)" : "multi-query only") +
               "\n  by " + p.authors + "; source: " + p.source + "; licence: " + p.licence + "; borrowed: " + p.borrowed + "\n";
    }
    return out;
}
}  // namespace tqllm
