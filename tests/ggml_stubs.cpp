// Minimal stubs for the few ggml symbols ggml-quants.c references outside the decoders we
// test (quantizers / TQ3 paths). The golden test only calls dequantize_row_q2_K and
// dequantize_row_iq2_s, neither of which touches these; the stubs exist purely to satisfy
// the linker and abort loudly if ever reached.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

extern "C" {

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    std::fprintf(stderr, "STUB ggml_abort reached (%s:%d) - should not happen in the golden test\n",
                 file, line);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::abort();
}

size_t ggml_row_size(uint32_t type, int64_t ne) {
    std::fprintf(stderr, "STUB ggml_row_size reached (type=%u) - should not happen\n", type);
    std::abort();
}

size_t ggml_type_size(uint32_t type) {
    std::fprintf(stderr, "STUB ggml_type_size reached (type=%u) - should not happen\n", type);
    std::abort();
}

const char * ggml_type_name(uint32_t type) {
    std::fprintf(stderr, "STUB ggml_type_name reached (type=%u) - should not happen\n", type);
    std::abort();
}

}  // extern "C"
