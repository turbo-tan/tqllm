#include "registry.hpp"
#include <mutex>
#include <utility>

namespace tqllm {
namespace {
std::vector<EngineEntry> & table() {
    static std::vector<EngineEntry> t;
    return t;
}
}  // namespace

void register_engine(EngineEntry entry) {
    for (auto & e : table())
        if (e.name == entry.name) { e = std::move(entry); return; }
    table().push_back(std::move(entry));
}
const std::vector<EngineEntry> & engines() {
    static std::once_flag once;
    std::call_once(once, register_builtin_engines);
    return table();
}
const EngineEntry * find_engine(const std::filesystem::path & model) {
    for (const auto & e : engines())
        if (e.matches(model)) return &e;
    return nullptr;
}
}  // namespace tqllm
