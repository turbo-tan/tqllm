#pragma once
// PAW X3 virtual-tensor record. Shared by the GGUF adapter (which writes it) and the X3 weight
// format (which parses it). One record precedes one row-group of trellis/suh/svh bytes; the loader
// may concatenate several records (row groups) into a single byte buffer, so a record is
// self-describing and splittable.
//
// Layout after the header: trellis (rows/16 * cols/16 * words * 2 B), suh (cols * 2 B),
// svh (rows * 2 B). The kernel math is y = svh * H128( W_hat * H128( suh * x ) ), W_hat from the
// mul1 codebook (exllamav3 trellis; fractional K uses mask 0xAAAA).
#include <cstdint>
#include <cstring>

namespace tqllm {

struct X3Record {
    char     magic[4];  // "X3R1"
    uint32_t rows;      // output rows (multiples of 128)
    uint32_t cols;      // input cols (multiples of 128)
    uint32_t words;     // uint16 per 16x16 tile: 16*K (56 = K3.5, 48 = K3, ...)
};

static_assert(sizeof(X3Record) == 16, "X3Record must be 16 bytes");

inline constexpr char X3_MAGIC[4] = {'X', '3', 'R', '1'};

inline X3Record x3_record_make(uint32_t rows, uint32_t cols, uint32_t words) {
    X3Record r{};
    std::memcpy(r.magic, X3_MAGIC, 4);
    r.rows = rows;
    r.cols = cols;
    r.words = words;
    return r;
}

inline bool x3_record_valid(const X3Record & r) {
    return std::memcmp(r.magic, X3_MAGIC, 4) == 0 && r.rows && r.cols && r.words;
}

} // namespace tqllm
