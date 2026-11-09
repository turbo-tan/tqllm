#pragma once
// JSON text writers shared by the server's response builders (no CUDA, no engine).
#include <third_party/nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <string>

namespace tqserve {

inline void jstr(std::string & out, const std::string & value) {
    out.push_back('"');
    for (size_t i = 0; i < value.size();) {
        const auto c = static_cast<unsigned char>(value[i++]);
        if (c >= 0x80) {
            // Token decoding can produce invalid or incomplete UTF-8. Keep the JSON valid
            // and consume only the valid prefix of a malformed sequence.
            const size_t start = i - 1;
            const size_t extra = c < 0xC2 ? 0 : c < 0xE0 ? 1 : c < 0xF0 ? 2 : c < 0xF5 ? 3 : 0;
            while (i < value.size() && i < start + 1 + extra) {
                const auto next = static_cast<unsigned char>(value[i]);
                if (next < 0x80 || next > 0xBF) break;
                if (i == start + 1 &&
                    ((c == 0xE0 && next < 0xA0) || (c == 0xED && next > 0x9F) ||
                     (c == 0xF0 && next < 0x90) || (c == 0xF4 && next > 0x8F))) break;
                ++i;
            }
            if (extra && i == start + 1 + extra) out.append(value, start, i - start);
            else out += "\\ufffd";
            continue;
        }
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[7];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    out += b;
                } else out.push_back(static_cast<char>(c));
        }
    }
    out.push_back('"');
}

// Compact JSON with Python json.dumps separators (", " and ": "), as the chat template's tojson renders tools.
inline void dump(std::string & out, const tqllm::Json & v) {
    switch (v.kind) {
        case tqllm::Json::Null: out += "null"; break;
        case tqllm::Json::Bool: out += v.num != 0 ? "true" : "false"; break;
        case tqllm::Json::Number: {
            char b[32];
            if (v.num == std::floor(v.num) && std::fabs(v.num) < 1e15) std::snprintf(b, sizeof(b), "%.0f", v.num);
            else std::snprintf(b, sizeof(b), "%.17g", v.num);
            out += b;
            break;
        }
        case tqllm::Json::String: jstr(out, v.str); break;
        case tqllm::Json::Array:
            out.push_back('[');
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i) out += ", ";
                dump(out, v.arr[i]);
            }
            out.push_back(']');
            break;
        case tqllm::Json::Object:
            out.push_back('{');
            for (size_t i = 0; i < v.obj.size(); ++i) {
                if (i) out += ", ";
                jstr(out, v.obj[i].first);
                out += ": ";
                dump(out, v.obj[i].second);
            }
            out.push_back('}');
            break;
    }
}

inline std::string dump(const tqllm::Json & v) {
    std::string out;
    dump(out, v);
    return out;
}

} // namespace tqserve
