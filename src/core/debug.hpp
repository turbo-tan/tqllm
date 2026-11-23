#pragma once
// TQLLM_DEBUG=key[:value],key2,... - the single diagnostics switch (see config/flags.yaml). Keys: memlog, profile,
// steps, stageprof, matvec, tune, prefill, decode, draft, reqlog, tree_chain, sample_chain, tf_dump:FILE, fakeq_k:N, fakeq_v:N, x3v, x3sq,
// moe_dump:DIR[@L].
#include <cstdlib>
#include <map>
#include <string>

namespace tqllm {
// TQLLM_DEBUG text -> key -> value (items split on ',', key and value on the first ':')
inline std::map<std::string, std::string> parse_debug(const std::string & s) {
    std::map<std::string, std::string> m;
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
}
// value of key (empty when given without a value; stable for the process); nullptr when absent
inline const std::string * debug_value(const char * key) {
    static const std::map<std::string, std::string> items = parse_debug(std::getenv("TQLLM_DEBUG") ? std::getenv("TQLLM_DEBUG") : "");
    const auto it = items.find(key);
    return it == items.end() ? nullptr : &it->second;
}
// a DIR[@L] value (moe_dump): '@', not ',', since ',' separates TQLLM_DEBUG items; layer -1 = every layer
struct DirLayer { std::string dir; long layer = -1; };
inline DirLayer debug_dir_layer(const std::string & v) {
    const size_t a = v.rfind('@');
    if (a == std::string::npos) return {v, -1};
    return {v.substr(0, a), std::stol(v.substr(a + 1))};
}
inline bool debug_on(const char * key) { return debug_value(key) != nullptr; }
// the value as a C string ("" when given without one); nullptr when the key is absent
inline const char * debug_cstr(const char * key) {
    const auto * v = debug_value(key);
    return v ? v->c_str() : nullptr;
}
inline int debug_int(const char * key) {
    const auto * v = debug_value(key);
    return v ? std::atoi(v->c_str()) : 0;
}
}  // namespace tqllm
