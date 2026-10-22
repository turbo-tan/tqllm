#include "tqllm/runtime.hpp"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
namespace tqllm {
void Plan::run(const float * a, const float * b, float * out) const {
    if (!invoke || !elements) throw std::logic_error("unresolved plan");
    if (!a || !b || !out) throw std::invalid_argument("null tensor pointer");
    tq_call call{sizeof(tq_call), TQ_ABI_VERSION, elements, a, b, out};
    tq_error error{sizeof(tq_error), 0, {}};
    if (invoke(&call, &error)) throw std::runtime_error(std::string(error.message, strnlen(error.message, sizeof(error.message))));
}
void Runtime::load(const std::string & path) {
    void * raw = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!raw) throw std::runtime_error(dlerror());
    std::shared_ptr<void> owner(raw, [](void * h) { dlclose(h); });
    auto init = reinterpret_cast<tq_plugin_init_fn>(dlsym(raw, "tq_plugin_init_v1"));
    if (!init) throw std::runtime_error("plugin missing tq_plugin_init_v1");
    struct Pending { std::vector<Entry> entries; std::shared_ptr<void> owner; } pending{{}, owner};
    tq_registry registry{sizeof(tq_registry), TQ_ABI_VERSION, &pending,
        [](void * context, const tq_kernel * k, tq_error * error) -> int32_t {
            try {
                if (!k || k->struct_size < sizeof(tq_kernel) || k->abi_version != TQ_ABI_VERSION ||
                    !k->id || !*k->id || !k->invoke || !k->max_elements) throw std::invalid_argument("invalid kernel descriptor");
                auto & p = *static_cast<Pending *>(context);
                p.entries.push_back({k->id, k->max_elements, k->invoke, p.owner});
                return 0;
            } catch (...) {
                if (error && error->struct_size >= sizeof(tq_error)) {
                    error->code = 1;
                    std::snprintf(error->message, sizeof(error->message), "kernel registration rejected");
                }
                return 1;
            }
        }};
    tq_error error{sizeof(tq_error), 0, {}};
    if (init(&registry, &error)) throw std::runtime_error("plugin initialization failed");
    auto merged = entries_;
    for (const auto & e : pending.entries) {
        for (const auto & existing : merged) if (existing.id == e.id) throw std::runtime_error("duplicate kernel: " + e.id);
        merged.push_back(e);
    }
    entries_.swap(merged);
}
Plan Runtime::resolve(const std::string & id, uint64_t elements) const {
    for (const auto & e : entries_) if (e.id == id && elements && elements <= e.max_elements) return {e.invoke, elements, e.owner};
    throw std::runtime_error("unsupported kernel or shape: " + id);
}
std::vector<std::string> Runtime::kernels() const {
    std::vector<std::string> result;
    for (const auto & e : entries_) result.push_back(e.id);
    return result;
}
}
