#pragma once
// X3 record: the GGUF container of one X3 row group. Written by the PAW GGUF adapter and minimax_convert --exl3,
// parsed by the X3 weight format. One record precedes one row-group of trellis/suh/svh bytes; the loader
// may concatenate several records (row groups) into a single byte buffer, so a record is
// self-describing and splittable.
//
// The codec id names how the trellis words become weights. The engines selects it through the host
// codec table (src/formats/x3_codecs.cpp), so a new codec is a new table row, never an edit here.
//
// Layout after the header: trellis (rows/16 * cols/16 * words * 2 B), suh (cols * 2 B),
// svh (rows * 2 B). The kernel math is y = svh * H128( W_hat * H128( suh * x ) ), W_hat from the
// codec named by `codec` (mul1 = exllamav3 trellis; fractional K uses mask 0xAAAA).
#include <cstdint>
#include <cstring>

namespace tqllm {

// Codec ids stored in X3Record::codec. Keep 0 unused so a zeroed record is invalid.
enum X3CodecId : uint16_t {
    X3_CODEC_MUL1 = 1,  // exllamav3 mul1-v1 trellis
};

struct X3Record {
    char     magic[4];  // "X3R2"
    uint32_t rows;      // output rows (multiples of 128)
    uint32_t cols;      // input cols (multiples of 128)
    uint32_t words;     // uint16 per 16x16 tile: 16*K (56 = K3.5, 48 = K3, ...)
    uint16_t codec;     // X3CodecId
    uint16_t flags;     // reserved, 0
};

static_assert(sizeof(X3Record) == 20, "X3Record must be 20 bytes");

inline constexpr char X3_MAGIC[4] = {'X', '3', 'R', '2'};

inline X3Record x3_record_make(uint32_t rows, uint32_t cols, uint32_t words,
                               uint16_t codec, uint16_t flags = 0) {
    X3Record r{};
    std::memcpy(r.magic, X3_MAGIC, 4);
    r.rows = rows;
    r.cols = cols;
    r.words = words;
    r.codec = codec;
    r.flags = flags;
    return r;
}

inline bool x3_record_valid(const X3Record & r) {
    return std::memcmp(r.magic, X3_MAGIC, 4) == 0 && r.rows && r.cols && r.words && r.codec;
}

} // namespace tqllm
