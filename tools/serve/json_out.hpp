#pragma once
// JSON text writers shared by the server's response builders (no CUDA, no engine).
#include <src/json.hpp>

#include <cmath>
#include <cstdio>
#include <string>

namespace tqserve {

inline void jstr(std::string & out, const std::string & value) {
    out.push_back('"');
    for (unsigned char c : value) {
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
