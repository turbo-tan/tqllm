// float_to_half: every finite half round-trips, midpoints round to even, overflow -> inf, NaN stays NaN
#include "dequant.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
using tqllm::float_to_half; using tqllm::half_to_float;
int main() {
    for (uint32_t h = 0; h < 0x10000; ++h) {
        if ((h & 0x7C00) == 0x7C00) continue;
        assert(float_to_half(half_to_float(uint16_t(h))) == h);
        if ((h & 0x7FFF) >= 0x7BFF) continue;
        const uint16_t next = uint16_t(h + 1);  // same sign, one ulp further from zero
        const float mid = 0.5f * (half_to_float(uint16_t(h)) + half_to_float(next));
        assert(float_to_half(mid) == ((h & 1) ? next : h));
    }
    assert(float_to_half(65504.f) == 0x7BFF && float_to_half(65520.f) == 0x7C00 && float_to_half(-1e9f) == 0xFC00);
    assert(float_to_half(std::numeric_limits<float>::infinity()) == 0x7C00);
    const uint16_t nan = float_to_half(std::nanf(""));
    assert((nan & 0x7C00) == 0x7C00 && (nan & 0x3FF));
    assert(float_to_half(std::ldexp(1.f, -25)) == 0 && float_to_half(std::ldexp(1.5f, -25)) == 1);
    std::puts("float_to_half ok");
}
