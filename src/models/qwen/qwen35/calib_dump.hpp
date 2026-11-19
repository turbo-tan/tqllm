#pragma once
// TQLLM_DEBUG=calib_dump:LAYERS:DIR - calibration capture for weight quantisers (tools/convert/qwen35/tqv12_quant.py).
// During prefill (multi-token forwards) the FP16 input of every quantised matrix of the listed layers is appended to
// DIR/L<layer>_<role>.x16 (row-major [tokens][K]), and DIR/L<layer>_<role>.json says K, the token count so far and the
// input basis. Roles name the inputs, not the matrices, so matrices that share an input are written once:
//   mix  attention q / k / v or Gated DeltaNet qkv / z (the normed hidden state)
//   gu   ffn gate / up      down   ffn down      out   attention output or Gated DeltaNet ssm_out
// Not written: decode steps (one token), forwards inside CUDA graph capture, and the load-time prefill calibration
// (synthetic inputs). LAYERS: comma-free list of layer numbers and ranges joined by '+', e.g. 0-3 or 4+5+9.
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {

struct CalibDumpSpec {
    std::vector<int> layers;  // ascending, unique
    std::string dir;
};

// "LAYERS:DIR" -> spec; throws std::invalid_argument on a malformed value
inline CalibDumpSpec parse_calib_dump(const std::string & v) {
    const auto c = v.find(':');
    if (c == std::string::npos || c == 0 || c + 1 >= v.size())
        throw std::invalid_argument("TQLLM_DEBUG=calib_dump:LAYERS:DIR, got \"" + v + "\"");
    CalibDumpSpec s;
    s.dir = v.substr(c + 1);
    const std::string ls = v.substr(0, c);
    auto num = [&](const std::string & t) {
        if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("calib_dump: bad layer \"" + t + "\" in \"" + ls + "\"");
        return std::stoi(t);
    };
    size_t p = 0;
    while (p <= ls.size()) {
        size_t q = ls.find('+', p);
        if (q == std::string::npos) q = ls.size();
        const std::string t = ls.substr(p, q - p);
        const auto d = t.find('-');
        const int a = num(t.substr(0, d)), b = d == std::string::npos ? a : num(t.substr(d + 1));
        if (b < a) throw std::invalid_argument("calib_dump: empty range \"" + t + "\"");
        for (int l = a; l <= b; ++l) s.layers.push_back(l);
        p = q + 1;
    }
    std::vector<int> u;
    for (int l : s.layers)
        if (u.empty() || u.back() != l) u.push_back(l);
    s.layers = u;
    for (size_t i = 1; i < s.layers.size(); ++i)
        if (s.layers[i] <= s.layers[i - 1]) throw std::invalid_argument("calib_dump: layers must ascend: \"" + ls + "\"");
    return s;
}

}  // namespace tqllm
