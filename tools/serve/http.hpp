#pragma once
// tqserve: one-shot HTTP/1.1 request splitting for the inference server.
//
// Pure, header-only, no sockets and no tqllm dependency: tests/serve.cpp compiles and runs it
// on a bare host with just g++ (no CUDA, no model) - that is the point of the test. Both
// tools/tqserve.cpp and tests/serve.cpp include this file; it is the single source of truth.
//
// split_request() validates one complete request captured from a TCP byte stream.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace tqserve {

struct Request {
    std::string method;               // "GET" | "POST" | ...
    std::string path;                 // target, no query string
    std::string query;                // empty when absent
    std::string authorization;
    std::string content_type, origin, host;
    uint64_t content_length = 0;      // as advertised
    std::vector<uint8_t> body;        // exactly content_length bytes, empty when 0
    std::string error;                 // set when the request is unusable
    bool ok() const { return error.empty(); }
};

// Splits one full HTTP/1.1 request (headers + body already captured in [in, in+len)) into a
// Request. Minimal-server rules - refuse what we cannot serve exactly:
//  - one request line "METHOD SP TARGET SP HTTP/1.x", then headers until the first empty line;
//  - the header block is capped at max_header bytes (reject, never loop on a hostile client);
//  - at most one Content-Length header, value v with 0 <= v <= body_limit; body is the first v
//    bytes; a Content-Length above body_limit is an error;
//  - any Transfer-Encoding is refused (no chunked); authorization, origin, host and content type are retained; other headers are ignored.
inline Request split_request(const uint8_t * in, size_t len, uint64_t body_limit,
                             uint64_t max_header = 65536) {
    Request req;
    auto fail = [&](const char * what) {
        if (req.error.empty()) req.error = what;
        return req;
    };

    // ---- request line ----------------------------------------------------
    size_t eol = 0;
    while (eol < len && in[eol] != '\n') ++eol;
    if (eol == len) return fail("no end of line");
    std::string line(reinterpret_cast<const char *>(in), eol);
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();

    const size_t sp1 = line.find(' ');
    const size_t sp2 = sp1 == std::string::npos ? sp1 : line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos || sp1 == 0 || sp2 == sp1 + 1)
        return fail("bad request line");
    const auto version = line.substr(sp2 + 1);
    if (version != "HTTP/1.0" && version != "HTTP/1.1") return fail("unsupported HTTP version");
    req.method = line.substr(0, sp1);
    for (auto & c : req.method) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (req.method.empty() || req.method.size() > 10 ||
        !std::all_of(req.method.begin(), req.method.end(), [](unsigned char c) { return std::isalpha(c) != 0; }))
        return fail("bad method");

    size_t tb = sp1 + 1;
    req.path = line.substr(tb, sp2 - tb);
    const size_t q = req.path.find('?');
    if (q != std::string::npos) { req.query = req.path.substr(q + 1); req.path.resize(q); }
    if (req.path.empty() || (req.path[0] != '/' && req.path[0] != '*')) return fail("bad target");

    // ---- headers ---------------------------------------------------------
    size_t header_end = std::string::npos;
    for (size_t i = eol + 1; i < len; ++i) {
        if (i + 3 < len && in[i] == '\r' && in[i + 1] == '\n' && in[i + 2] == '\r' && in[i + 3] == '\n') {
            header_end = i + 4;
            break;
        }
        if (i + 1 < len && in[i] == '\n' && in[i + 1] == '\n') {
            header_end = i + 2;
            break;
        }
    }
    if (header_end == std::string::npos) return fail("incomplete headers");
    if (header_end > max_header) return fail("header block too large");
    size_t pos = eol + 1;
    uint64_t header_bytes = 0;
    bool has_content_length = false;
    while (pos < header_end) {
        size_t n = 0;
        while (n < header_end - pos && in[pos + n] != '\n') ++n;
        header_bytes += n + 1;
        if (header_bytes > max_header) return fail("header block too large");

        std::string hl(reinterpret_cast<const char *>(in + pos), n);
        while (!hl.empty() && (hl.back() == '\r' || hl.back() == '\n')) hl.pop_back();
        pos += n + 1;
        if (hl.empty()) break; // end of headers

        const size_t c = hl.find(':');
        if (c == std::string::npos || c == 0) return fail("bad header");
        std::string name = hl.substr(0, c);
        std::string val = hl.substr(c + 1);
        for (auto & ch : name) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.erase(val.begin());

        if (name == "content-length") {
            if (has_content_length) return fail("duplicate content-length");
            has_content_length = true;
            for (char ch : val)
                if (!std::isdigit(static_cast<unsigned char>(ch))) return fail("bad content-length");
            try { req.content_length = std::stoull(val); }
            catch (...) { return fail("bad content-length"); }
        } else if (name == "transfer-encoding") {
            return fail("transfer-encoding not supported");
        } else if (name == "authorization") {
            req.authorization = val;
        } else if (name == "content-type") {
            req.content_type = val;
        } else if (name == "origin") {
            req.origin = val;
        } else if (name == "host") {
            req.host = val;
        }
    }

    if (req.content_length > body_limit) return fail("content-length above body limit");
    if (req.content_length > len - pos) return fail("body shorter than content-length");
    req.body.assign(in + pos, in + pos + req.content_length);
    return req;
}

inline Request split_request(const std::vector<uint8_t> & data, uint64_t body_limit,
                             uint64_t max_header = 65536) {
    return split_request(data.data(), data.size(), body_limit, max_header);
}

class RequestReader {
public:
    explicit RequestReader(uint64_t body_limit, uint64_t max_header = 65536)
        : body_limit_(body_limit), max_header_(max_header) {}

    bool append(const uint8_t * data, size_t len) {
        if (complete_ || !error_.empty()) return complete_;
        if (len > max_header_ + body_limit_ - std::min<uint64_t>(buffer_.size(), max_header_ + body_limit_)) {
            error_ = "request too large";
            return false;
        }
        buffer_.insert(buffer_.end(), data, data + len);
        size_t header_end = 0;
        if (!find_header_end(header_end)) {
            if (buffer_.size() > max_header_) error_ = "header block too large";
            return false;
        }
        if (header_end > max_header_) {
            error_ = "header block too large";
            return false;
        }
        const Request head = split_request(buffer_.data(), header_end, body_limit_, max_header_);
        if (!head.ok() && head.error != "body shorter than content-length") {
            error_ = head.error;
            return false;
        }
        expected_ = static_cast<uint64_t>(header_end) + head.content_length;
        if (buffer_.size() >= expected_) complete_ = true;
        return complete_;
    }

    bool complete() const { return complete_; }
    bool failed() const { return !error_.empty(); }
    const std::string & error() const { return error_; }

    Request request() const {
        if (!error_.empty()) {
            Request req;
            req.error = error_;
            return req;
        }
        if (!complete_) {
            Request req;
            req.error = "incomplete request";
            return req;
        }
        return split_request(buffer_.data(), static_cast<size_t>(expected_), body_limit_, max_header_);
    }

private:
    bool find_header_end(size_t & end) const {
        for (size_t i = 0; i < buffer_.size(); ++i) {
            if (i + 3 < buffer_.size() && buffer_[i] == '\r' && buffer_[i + 1] == '\n' &&
                buffer_[i + 2] == '\r' && buffer_[i + 3] == '\n') {
                end = i + 4;
                return true;
            }
            if (i + 1 < buffer_.size() && buffer_[i] == '\n' && buffer_[i + 1] == '\n') {
                end = i + 2;
                return true;
            }
        }
        return false;
    }

    uint64_t body_limit_;
    uint64_t max_header_;
    uint64_t expected_ = 0;
    std::vector<uint8_t> buffer_;
    std::string error_;
    bool complete_ = false;
};

} // namespace tqserve
