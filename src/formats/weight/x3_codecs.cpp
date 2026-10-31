#include "x3_codecs.hpp"

#include <string>

namespace tqllm {

const x3::X3Codec* x3_codec_lookup(uint16_t id, int words) {
    int n = 0;
    const x3::X3Codec* t = x3::x3_codec_table(n);
    for (int i = 0; i < n; ++i)
        if (t[i].id == id && t[i].words == words) return &t[i];
    return nullptr;
}

std::string x3_codec_unknown(uint16_t id, int words) {
    std::string s = "x3: unknown codec id " + std::to_string(id) + " / words " + std::to_string(words) + " (known:";
    int n = 0;
    const x3::X3Codec* t = x3::x3_codec_table(n);
    for (int i = 0; i < n; ++i) s += std::string(" ") + t[i].name;
    return s + ")";
}

}  // namespace tqllm
