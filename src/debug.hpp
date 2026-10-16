#pragma once
// TQLLM_DEBUG=key[:value],key2,... - the single diagnostics switch (see config/flags.yaml). Keys: memlog, profile,
// steps, stageprof, matvec, tune, prefill, decode, draft, reqlog, tree_chain, tf_dump:FILE, fakeq_k:N, fakeq_v:N, x3n1x3v.
#include <cstdlib>
#include <map>
#include <string>

namespace tqllm {
// value of key (empty when given without a value; stable for the process); nullptr when absent
inline const std::string * debug_value(const char * key) {
    static const std::map<std::string, std::string> items = [] {
        std::map<std::string, std::string> m;
        const std::string s = std::getenv("TQLLM_DEBUG") ? std::getenv("TQLLM_DEBUG") : "";
        size_t p = 0;
        while (p < s.size()) {
            size_t q = s.find(',', p);
            if (q == std::string::npos) q = s.size();
            const std::string it = s.substr(p, q - p);
            const size_t c = it.find(':');
            if (!it.empty()) m[it.substr(0, c)] = c == std::string::npos ? std::string() : it.substr(c + 1);
            p = q + 1;
        }
        return m;
    }();
    const auto it = items.find(key);
    return it == items.end() ? nullptr : &it->second;
}
inline bool debug_on(const char * key) { return debug_value(key) != nullptr; }
inline int debug_int(const char * key) {
    const auto * v = debug_value(key);
    return v ? std::atoi(v->c_str()) : 0;
}
}  // namespace tqllm
