#pragma once
static uint32_t word(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
static std::vector<uint8_t> fixture(int rows, int k, int bits = 3) {
    const int payload = bits == 3 ? 50 : 66;
    std::vector<uint8_t> raw(size_t(rows) * (k / 128) * payload);
    for (int r = 0; r < rows; ++r) for (int g = 0; g < k / 128; ++g) {
        auto* p = raw.data() + (size_t(r) * (k / 128) + g) * payload;
        uint16_t scale = uint16_t(0x1c00 + (r * 17 + g * 31) % 1024);
        if (r % 3 == 1) scale |= 0x8000;
        p[0] = uint8_t(scale); p[1] = uint8_t(scale >> 8);
        for (int b = 0; b < 4; ++b) {
            uint32_t planes[4] = {};
            for (int j = 0; j < 32; ++j) {
                uint32_t code = uint32_t((r * 3 + g * 5 + b * 7 + j) % (1 << bits));
                if (bits == 3) {
                    planes[j / 16] |= (code & 3u) << (4 * (j % 8) + 2 * ((j % 16) / 8));
                    planes[2] |= (code >> 2) << (4 * (j % 8) + j / 8);
                } else planes[j / 8] |= code << (4 * (j % 8));
            }
            for (int a = 0; a < bits; ++a) for (int byte = 0; byte < 4; ++byte)
                p[2 + b * (bits * 4) + a * 4 + byte] = uint8_t(planes[a] >> (8 * byte));
        }
    }
    return raw;
}
static void check_layout(const Tq3TiledWeights& tiled, const std::vector<uint8_t>& raw) {
    mx::eval(tiled.planes, tiled.scales);
    const auto* codes = tiled.planes.data<uint32_t>();
    auto bits = mx::view(tiled.scales, mx::uint16); mx::eval(bits);
    const auto* scales = bits.data<uint16_t>();
    const int blocks = tiled.columns / 32;
    for (int r = 0; r < ((tiled.rows + 31) / 32) * 32; ++r) {
        for (int b = 0; b < blocks; ++b) for (int a = 0; a < tiled.bits; ++a) {
            uint32_t expected = 0;
            if (r < tiled.rows) {
                const auto* p = raw.data() + (size_t(r) * (blocks / 4) + b / 4) * (tiled.bits == 3 ? 50 : 66);
                expected = word(p + 2 + (b % 4) * (tiled.bits * 4) + a * 4);
            }
            const size_t physical = ((size_t(r / 32) * blocks + b) * tiled.bits + a) * 32 + r % 32;
            if (codes[physical] != expected) throw std::runtime_error("tiled code payload differs from raw GGUF");
        }
        for (int g = 0; g < blocks / 4; ++g) {
            uint16_t expected = 0;
            if (r < tiled.rows) {
                const auto* p = raw.data() + (size_t(r) * (blocks / 4) + g) * (tiled.bits == 3 ? 50 : 66);
                expected = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
            }
            if (scales[(size_t(r / 32) * (blocks / 4) + g) * 32 + r % 32] != expected)
                throw std::runtime_error("tiled scale payload differs from raw GGUF");
        }
    }
}
static std::vector<float> inputs(int m, int k) {
    std::vector<float> values(size_t(m) * k);
    for (size_t i = 0; i < values.size(); ++i) values[i] = .1f * std::sin(float(i) * .037f);
    return values;
}
static double rel_l2(const mx::array& a, const mx::array& b) {
    mx::eval(a, b);
    double error = 0, norm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a.data<float>()[i]) - b.data<float>()[i];
        error += d * d; norm += double(b.data<float>()[i]) * b.data<float>()[i];
    }
    return std::sqrt(error / std::max(norm, 1e-30));
}
