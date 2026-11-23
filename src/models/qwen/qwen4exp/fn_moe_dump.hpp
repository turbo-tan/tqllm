#pragma once
// TQLLM_DEBUG=moe_dump:A-B:DIR (calibration capture for expert quantisers, off by default): for every trunk MoE layer
// in [A, B] and every non-captured forward call, appends the routed experts' input rows and routing to
//   DIR/L<layer>.x16  FP16 [n][hidden]   (the FFN input the routed and shared experts read)
//   DIR/L<layer>.rt   per call: int32 n, then int32 ids [n][topk], float weights [n][topk], float shared weight [n]
// Synchronous host copies: slow, for offline capture runs only.
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "debug.hpp"

namespace tqllm::flashnext {
struct MoeDumpSpec { int a = -1, b = -1; std::string dir; };
inline const MoeDumpSpec & moe_dump_spec() {
    static const MoeDumpSpec s = [] {
        MoeDumpSpec r;
        const char * v = tqllm::debug_cstr("moe_dump");
        if (!v || !*v) return r;
        const std::string t(v);
        const size_t c = t.find(':');
        if (c == std::string::npos) throw std::invalid_argument("TQLLM_DEBUG=moe_dump:A-B:DIR, got \"" + t + "\"");
        const std::string ls = t.substr(0, c);
        r.dir = t.substr(c + 1);
        const size_t d = ls.find('-');
        r.a = std::atoi(ls.substr(0, d).c_str());
        r.b = d == std::string::npos ? r.a : std::atoi(ls.substr(d + 1).c_str());
        if (r.dir.empty() || r.b < r.a) throw std::invalid_argument("TQLLM_DEBUG=moe_dump: bad spec \"" + t + "\"");
        return r;
    }();
    return s;
}
inline bool moe_dump_wants(int layer) { const auto & s = moe_dump_spec(); return s.a >= 0 && layer >= s.a && layer <= s.b; }
inline void moe_dump_write(int layer, int n, int hidden, int topk, const std::vector<float> & x, const std::vector<int> & ids,
                           const std::vector<float> & w, const std::vector<float> & sw) {
    static std::map<int, std::pair<FILE *, FILE *>> files;
    auto & f = files[layer];
    if (!f.first) {
        const std::string base = moe_dump_spec().dir + "/L" + std::to_string(layer);
        f.first = std::fopen((base + ".x16").c_str(), "ab");
        f.second = std::fopen((base + ".rt").c_str(), "ab");
        if (!f.first || !f.second) throw std::runtime_error("moe_dump: cannot open " + base);
    }
    std::vector<__half> xh(size_t(n) * hidden);
    for (size_t i = 0; i < xh.size(); ++i) xh[i] = __float2half_rn(x[i]);
    std::fwrite(xh.data(), 2, xh.size(), f.first);
    const int32_t nn = n;
    std::fwrite(&nn, 4, 1, f.second);
    std::fwrite(ids.data(), 4, size_t(n) * topk, f.second);
    std::fwrite(w.data(), 4, size_t(n) * topk, f.second);
    std::fwrite(sw.data(), 4, size_t(n), f.second);
    std::fflush(f.first);
    std::fflush(f.second);
}
}  // namespace tqllm::flashnext
