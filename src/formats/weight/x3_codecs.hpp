#pragma once
// x3_codecs.host lookup over the codec table (defined in kernels/formats/x3/dispatch.cu).
#include "x3.cuh"

#include <cstdint>
#include <string>

namespace tqllm {

// The codec row for (id, words), or null if the file names an unknown codec/K.
const x3::X3Codec* x3_codec_lookup(uint16_t id, int words);

// Human-readable message listing the known codecs, for a clear load-time failure.
std::string x3_codec_unknown(uint16_t id, int words);

}  // namespace tqllm
