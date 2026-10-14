#include "tq_mlx.hpp"
#include "weights.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
namespace {
int bits_of(uint32_t type) {return (type == GGML_TQMLX_TILED3 || type == GGML_TQMLX_MMA8_3) ? 3 : 4;}
bool tiled_type(uint32_t type) {return type == GGML_TQMLX_TILED3 || type == GGML_TQMLX_TILED4;}

mx::array half_array(const uint8_t* data, mx::Shape shape) {
    return mx::view(mx::array(reinterpret_cast<const uint16_t*>(data), shape, mx::uint16), mx::float16);
}

mx::array tiled_dense(const TqTiledWeights& w) {
    static auto kernel = mx::fast::metal_kernel("tqllm_tqmlx_tiled_dense", {"planes", "scales"}, {"y"}, R"METAL(
uint i = thread_position_in_grid.x;
if (i >= R * K) return;
uint row = i / K, col = i % K, tile = row / 32u, lane = row % 32u, block = col / 32u, j = col % 32u;
ulong origin = (ulong(tile) * (K / 32u) + block) * (BITS * 32u) + lane;
int value;
if (BITS == 3) {
    uint ranks = planes[origin + (j / 16u) * 32u];
    uint rank = (ranks >> (4u * (j % 8u) + 2u * ((j % 16u) / 8u))) & 3u;
    uint sign = (planes[origin + 64u] >> (4u * (j % 8u) + j / 8u)) & 1u;
    value = int((0x7f4f2d0eu >> (rank * 8u)) & 255u);
    if (sign) value = -value;
} else {
    uint code = (planes[origin + (j / 8u) * 32u] >> (4u * (j % 8u))) & 15u;
    value = LEVELS[code];
}
float scale = float(scales[(ulong(tile) * (K / 128u) + col / 128u) * 32u + lane]);
y[i] = bfloat16_t(scale * float(value));
)METAL", "constant int LEVELS[16] = {6,18,31,44,58,75,96,127,-6,-18,-31,-44,-58,-75,-96,-127};");
    return kernel({w.planes, w.scales}, {{w.rows, w.columns}}, {mx::bfloat16}, {w.rows * w.columns, 1, 1}, {256, 1, 1},
                  {{"R", w.rows}, {"K", w.columns}, {"BITS", w.bits}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

// Inverse of the MMA8 fragment order: locate the nibble that holds weight (row, col).
mx::array mma8_dense(const Mma8Weights& w) {
    static auto kernel = mx::fast::metal_kernel("tqllm_tqmlx_mma8_dense", {"W", "S", "lut"}, {"y"}, R"METAL(
uint i = thread_position_in_grid.x;
if (i >= R * K) return;
uint row = i / K, col = i % K;
uint g = row / 16u, t = (row / 8u) % 2u, rr = row % 8u, fn = rr & 6u, e = rr & 1u;
uint kb = col / 128u, kk = (col % 128u) / 8u, fm = col % 8u;
uint wq = kk / 4u, b = kk % 4u;
uint qid = (fm & 4u) | ((fn & 4u) >> 1) | ((fm >> 1) & 1u);
uint lane = qid * 4u + ((fm & 1u) << 1) + ((fn >> 1) & 1u);
ulong base = (ulong(g) * (K / 128u) + kb) * 2u + t;
ulong at = (base * 32u + lane) * 32u + wq * 8u + 2u * b + e;
uint nib = (W[at / 8u] >> (4u * uint(at % 8u))) & 15u;
float level = float(as_type<half2>(lut[nib]).x);
y[i] = bfloat16_t(float(S[base * 8u + rr]) * level);
)METAL");
    return kernel({w.words, w.scales, w.lut}, {{w.rows, w.columns}}, {mx::bfloat16}, {w.rows * w.columns, 1, 1}, {256, 1, 1},
                  {{"R", w.rows}, {"K", w.columns}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

void append(std::vector<uint8_t>& out, const mx::array& a) {
    auto flat = mx::contiguous(a);
    mx::eval(flat);
    const auto* p = flat.data<uint8_t>();
    out.insert(out.end(), p, p + flat.nbytes());
}

struct Planned {
    std::string name;
    GgufTensor src;
    uint32_t type;
    uint64_t bytes, offset;
};

bool tiled_class(const std::string& name) {
    return name.find(".ffn_") != std::string::npos || name.ends_with(".attn_qkv.weight") ||
           name.ends_with(".attn_gate.weight") || name.ends_with(".ssm_out.weight");
}

}
std::vector<uint8_t> tq_mlx_pack_tensor(const std::string& name, const GgufTensor& t, std::span<const uint8_t> payload,
                                        uint32_t& type) {
    std::vector<uint8_t> out;
    auto m = load_matrix(t, payload);
    if (auto* w = std::get_if<Tq3Weights>(&m)) {
        if (tiled_class(name)) {
            auto tiled = tq3_tile(*w);
            type = GGML_TQMLX_TILED3;
            append(out, tiled.planes);
            append(out, tiled.scales);
        } else {
            auto pack = mma8_pack(*w);
            type = GGML_TQMLX_MMA8_3;
            append(out, pack.words);
            append(out, pack.scales);
        }
    } else if (auto* w = std::get_if<Tq4Weights>(&m)) {
        if (tiled_class(name)) {
            auto tiled = tq4_tile(*w);
            type = GGML_TQMLX_TILED4;
            append(out, tiled.planes);
            append(out, tiled.scales);
        } else {
            auto pack = mma8_pack(*w);
            type = GGML_TQMLX_MMA8_4;
            append(out, pack.words);
            append(out, pack.scales);
        }
    }
    return out;
}
namespace {
template <class T> void put(std::ostream& os, T v) {os.write(reinterpret_cast<const char*>(&v), sizeof(v));}
void put_string(std::ostream& os, const std::string& s) {put<uint64_t>(os, s.size()); os.write(s.data(), std::streamsize(s.size()));}
uint64_t align_up(uint64_t v, uint64_t a) {return (v + a - 1) / a * a;}
}

bool is_tq_mlx_type(uint32_t type) {return type >= GGML_TQMLX_TILED3 && type <= GGML_TQMLX_MMA8_4;}
bool is_tq_mlx_file(const GgufFile& file) {return file.kv.count(kTqMlxVersionKey) > 0;}

TqMlxMatrix tq_mlx_load(const GgufTensor& t, std::span<const uint8_t> payload) {
    if (!is_tq_mlx_type(t.type) || t.ne.size() != 2 || payload.size() != t.bytes)
        throw std::invalid_argument("not a tq_mlx matrix");
    const int k = int(t.ne[0]), rows = int(t.ne[1]), bits = bits_of(t.type);
    if (tiled_type(t.type)) {
        const int tiles = (rows + 31) / 32;
        const size_t plane_bytes = size_t(tiles) * k * bits * 4;
        auto planes = mx::array(reinterpret_cast<const uint32_t*>(payload.data()), {tiles, k / 32, bits, 32}, mx::uint32);
        auto scales = half_array(payload.data() + plane_bytes, {tiles, k / 128, 32});
        mx::eval(planes, scales);
        return TqTiledWeights{planes, scales, rows, k, bits, true};
    }
    const int rp = (rows + 63) / 64 * 64;
    const size_t word_bytes = size_t(rp) * k / 2;
    auto words = mx::array(reinterpret_cast<const uint32_t*>(payload.data()), {int(word_bytes / 4)}, mx::uint32);
    auto scales = half_array(payload.data() + word_bytes, {rp * (k / 128)});
    mx::eval(words, scales);
    return mma8_from_parts(words, scales, rows, k, bits);
}

mx::array tq_mlx_dense(const TqMlxMatrix& weights) {
    return std::visit([](const auto& w) {
        if constexpr (std::is_same_v<std::decay_t<decltype(w)>, TqTiledWeights>) return tiled_dense(w);
        else return mma8_dense(w);
    }, weights);
}

mx::array tq_mlx_prefill(const TqMlxMatrix& weights, const mx::array& rotated) {
    auto dense = tq_mlx_dense(weights);
    auto y = mx::astype(mx::matmul(mx::astype(rotated, mx::bfloat16), mx::transpose(dense)), mx::float32);
    mx::eval(y);  // finish one projection before the next allocates its dense scratch
    return y;
}

TqMlxStats tq_mlx_convert(const std::string& source, const std::string& destination) {
    auto file = gguf_open(source);
    if (is_tq_mlx_file(file)) throw std::invalid_argument("source is already tq_mlx");
    std::vector<std::pair<std::string, GgufTensor>> order(file.tensors.begin(), file.tensors.end());
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {return a.second.offset < b.second.offset;});
    TqMlxStats stats;
    std::vector<Planned> plan;
    uint64_t at = 0;
    // The converted type and size depend only on dtype, shape and name, so the layout is planned before any data moves.
    for (const auto& [name, t] : order) {
        Planned p{name, t, t.type, t.bytes, 0};
        if (t.ne.size() == 2 && (t.type == GGML_TQ3 || t.type == GGML_TQ4)) {
            const bool tiled = tiled_class(name);
            p.type = tiled ? (t.type == GGML_TQ3 ? GGML_TQMLX_TILED3 : GGML_TQMLX_TILED4)
                           : (t.type == GGML_TQ3 ? GGML_TQMLX_MMA8_3 : GGML_TQMLX_MMA8_4);
            p.bytes = gguf_tensor_bytes(p.type, t.ne, name);
        }
        at = align_up(at, file.alignment);
        p.offset = at;
        at += p.bytes;
        plan.push_back(std::move(p));
    }
    std::ifstream in(source, std::ios::binary);
    std::ofstream out(destination, std::ios::binary | std::ios::trunc);
    if (!in || !out) throw std::runtime_error("cannot open conversion files");
    std::vector<char> header(file.kv_end);
    in.read(header.data(), std::streamsize(header.size()));
    if (!in) throw std::runtime_error("cannot read source header");
    const uint64_t n_tensors = plan.size(), n_kv = file.n_kv + 1;
    std::memcpy(header.data() + 8, &n_tensors, 8);
    std::memcpy(header.data() + 16, &n_kv, 8);
    out.write(header.data(), std::streamsize(header.size()));
    put_string(out, kTqMlxVersionKey);
    put<uint32_t>(out, 4);
    put<uint32_t>(out, 1);
    for (const auto& p : plan) {
        put_string(out, p.name);
        put<uint32_t>(out, uint32_t(p.src.ne.size()));
        for (auto e : p.src.ne) put<uint64_t>(out, e);
        put<uint32_t>(out, p.type);
        put<uint64_t>(out, p.offset);
    }
    auto pad_to = [&](uint64_t target) {
        static const char zeros[4096] = {};
        auto pos = uint64_t(out.tellp());
        while (pos < target) {
            auto n = std::min<uint64_t>(sizeof(zeros), target - pos);
            out.write(zeros, std::streamsize(n));
            pos += n;
        }
    };
    const uint64_t data_start = align_up(uint64_t(out.tellp()), file.alignment);
    pad_to(data_start);
    for (const auto& p : plan) {
        pad_to(data_start + p.offset);
        auto raw = file.read(p.src);
        stats.bytes_in += raw.size();
        std::span<const uint8_t> payload(reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
        if (p.type == p.src.type) {
            out.write(raw.data(), std::streamsize(raw.size()));
            ++stats.copied;
        } else {
            uint32_t type = p.src.type;
            auto blob = tq_mlx_pack_tensor(p.name, p.src, payload, type);
            if (type != p.type || blob.size() != p.bytes)
                throw std::runtime_error("tq_mlx conversion disagrees with the plan for " + p.name);
            out.write(reinterpret_cast<const char*>(blob.data()), std::streamsize(blob.size()));
            ++(tiled_type(type) ? stats.tiled : stats.mma8);
        }
        stats.bytes_out += p.bytes;
    }
    out.flush();
    if (!out) throw std::runtime_error("failed writing tq_mlx file");
    return stats;
}
}
