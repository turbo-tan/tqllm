#pragma once
// Minimal UTF-8 helper for the vendored jinja engine: struct and function copied from llama.cpp common/unicode.{h,cpp}
// (MIT licence, (c) the ggml authors).
#include <cstddef>
#include <cstdint>
#include <string_view>

struct utf8_parse_result {
    uint32_t codepoint;
    size_t bytes_consumed;
    enum status { SUCCESS, INCOMPLETE, INVALID } status;
    utf8_parse_result(enum status s, uint32_t cp = 0, size_t bytes = 0) : codepoint(cp), bytes_consumed(bytes), status(s) {}
};

inline utf8_parse_result common_parse_utf8_codepoint(std::string_view input, size_t offset) {
    if (offset >= input.size()) {
        return utf8_parse_result(utf8_parse_result::INCOMPLETE);
    }

    // ASCII fast path
    if (!(input[offset] & 0x80)) {
        return utf8_parse_result(utf8_parse_result::SUCCESS, input[offset], 1);
    }

    // Invalid: continuation byte as first byte
    if (!(input[offset] & 0x40)) {
        return utf8_parse_result(utf8_parse_result::INVALID);
    }

    // 2-byte sequence
    if (!(input[offset] & 0x20)) {
        if (offset + 1 >= input.size()) {
            return utf8_parse_result(utf8_parse_result::INCOMPLETE);
        }
        if ((input[offset + 1] & 0xc0) != 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        auto result = ((input[offset] & 0x1f) << 6) | (input[offset + 1] & 0x3f);
        return utf8_parse_result(utf8_parse_result::SUCCESS, result, 2);
    }

    // 3-byte sequence
    if (!(input[offset] & 0x10)) {
        if (offset + 2 >= input.size()) {
            return utf8_parse_result(utf8_parse_result::INCOMPLETE);
        }
        if ((input[offset + 1] & 0xc0) != 0x80 || (input[offset + 2] & 0xc0) != 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        auto result = ((input[offset] & 0x0f) << 12) | ((input[offset + 1] & 0x3f) << 6) | (input[offset + 2] & 0x3f);
        return utf8_parse_result(utf8_parse_result::SUCCESS, result, 3);
    }

    // 4-byte sequence
    if (!(input[offset] & 0x08)) {
        if (offset + 3 >= input.size()) {
            return utf8_parse_result(utf8_parse_result::INCOMPLETE);
        }
        if ((input[offset + 1] & 0xc0) != 0x80 || (input[offset + 2] & 0xc0) != 0x80 || (input[offset + 3] & 0xc0) != 0x80) {
            return utf8_parse_result(utf8_parse_result::INVALID);
        }
        auto result = ((input[offset] & 0x07) << 18) | ((input[offset + 1] & 0x3f) << 12) | ((input[offset + 2] & 0x3f) << 6) | (input[offset + 3] & 0x3f);
        return utf8_parse_result(utf8_parse_result::SUCCESS, result, 4);
    }

    // Invalid first byte
    return utf8_parse_result(utf8_parse_result::INVALID);
}
