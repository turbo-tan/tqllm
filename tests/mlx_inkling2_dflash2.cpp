// Inkling2MlxDFlash2 against the sglang-math reference (tools/inkling2/dflash2_reference.py, fixture
// tests/data/inkling2_dflash2_ref.bin): context taps = the normalised input rows of a fixed token list, one draft block
// after the next token, the final normalised rows, the top-16 candidates and the selector's 7-token path.
// The drafter amplifies rounding: the reference itself moves 10 % (hidden relative RMS) and flips its path after the first
// token between BF16 and FP32 activations, so BF16 projections must stay within that band, keep the candidates and the
// first proposal; the served 8-bit projections are held to a wider band.
// Usage: test-mlx-inkling2-dflash2 DRAFTER_DIR INKLING_CHECKPOINT_DIR FIXTURE
#include "inkling2_mlx_draft.hpp"
#include "hf_folder.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <stdexcept>
#include <vector>
namespace mx = mlx::core;
using tqllm::mlx_backend::Inkling2MlxDFlash2;
namespace {
// TOKENS of tools/inkling2/dflash2_reference.py
const uint32_t kTokens[] = {1719, 220, 46389, 7, 77, 8, 25, 198, 149862, 150003, 149934, 196982, 220, 17, 25, 198, 149860,
                            149862, 150583, 149934, 198, 149862, 150583, 220};
mx::array tensor(const tqllm::HfFolder& f, const std::string& name) {
    const auto& t = f.at(name);
    mx::Shape shape;
    for (auto d : t.shape) shape.push_back(int(d));
    const auto bytes = tqllm::safetensors_read(t);
    mx::array raw(reinterpret_cast<const uint8_t*>(bytes.data()), {int(bytes.size())}, mx::uint8);
    auto v = mx::reshape(mx::view(raw, mx::bfloat16), shape);
    mx::eval(v);
    return v;
}
struct Fixture {
    int m = 0, anchor = 0, vocab = 0, block = 0, D = 0, K = 0;
    std::vector<uint32_t> tokens, ids;
    std::vector<float> logits, hidden;
};
Fixture read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    int32_t h[8];
    if (!in.read(reinterpret_cast<char*>(h), sizeof(h)) || h[0] != 0x32464C44) throw std::runtime_error("fixture header");
    Fixture f{h[1], h[2], h[3], h[4], h[5], h[6]};
    f.tokens.resize(size_t(f.block - 1)); f.ids.resize(size_t(f.block * f.K)); f.logits.resize(f.ids.size());
    f.hidden.resize(size_t(f.block * f.D));
    in.read(reinterpret_cast<char*>(f.tokens.data()), f.tokens.size() * 4);
    in.read(reinterpret_cast<char*>(f.ids.data()), f.ids.size() * 4);
    in.read(reinterpret_cast<char*>(f.logits.data()), f.logits.size() * 4);
    in.read(reinterpret_cast<char*>(f.hidden.data()), f.hidden.size() * 4);
    if (!in) throw std::runtime_error("fixture truncated");
    return f;
}
// one block; returns {hidden relative RMS (rows 1..7), min top-16 overlap, matching path tokens}
struct Result { double hidden_rms; int overlap, path; };
Result run(const Fixture& fx, const std::string& drafter, const mx::array& embed, const mx::array& embed_norm,
           const mx::array& unembed, bool bf16) {
    auto head = [&](const mx::array& x) { return mx::matmul(mx::astype(x, mx::bfloat16), mx::transpose(unembed)); };
    auto emb = [&](const mx::array& ids) {
        return mx::astype(mx::fast::rms_norm(mx::astype(mx::take(embed, ids, 0), mx::float32), mx::astype(embed_norm, mx::float32), 1e-6f), mx::bfloat16);
    };
    Inkling2MlxDFlash2 d(drafter, 4096, fx.vocab, head, emb, bf16);
    // the block runs on a second thread, as tqserve decodes on its own thread while the drafter is built on the main
    // one: every array the drafter keeps must be evaluated at load, else MLX has no stream of the building thread there
    std::vector<float> hidden_rows;
    std::vector<uint32_t> id_rows, path_tokens;
    std::string failure;
    std::thread worker([&] {
        try {
            auto rows = emb(mx::array(kTokens, {fx.m}, mx::uint32));
            d.commit(mx::broadcast_to(mx::expand_dims(rows, 1), {fx.m, 4, fx.D}), 0);
            const uint32_t a = uint32_t(fx.anchor);
            auto anchor = mx::array(&a, {1}, mx::uint32);
            auto path = d.propose(emb(anchor), anchor, fx.m, fx.block - 1);
            auto tr = d.last_trace();
            auto hidden = mx::contiguous(tr.hidden), ids = mx::contiguous(tr.ids);
            mx::eval({path, hidden, ids});
            hidden_rows.assign(hidden.data<float>(), hidden.data<float>() + hidden.size());
            id_rows.assign(ids.data<uint32_t>(), ids.data<uint32_t>() + ids.size());
            path_tokens.assign(path.data<uint32_t>(), path.data<uint32_t>() + path.size());
        } catch (const std::exception& e) { failure = e.what(); }
    });
    worker.join();
    if (!failure.empty()) throw std::runtime_error("drafter on a second thread: " + failure);
    double err = 0, mag = 0;
    const float* hid = hidden_rows.data();
    for (size_t i = size_t(fx.D); i < fx.hidden.size(); ++i) { const double e = hid[i] - fx.hidden[i]; err += e * e; mag += double(fx.hidden[i]) * fx.hidden[i]; }
    int overlap = fx.K;
    const uint32_t* ids = id_rows.data();
    for (int r = 1; r < fx.block; ++r) {
        int same = 0;
        for (int j = 0; j < fx.K; ++j)
            same += std::count(fx.ids.begin() + r * fx.K, fx.ids.begin() + (r + 1) * fx.K, ids[r * fx.K + j]) > 0;
        overlap = std::min(overlap, same);
    }
    int matched = 0;
    const uint32_t* p = path_tokens.data();
    while (matched < fx.block - 1 && p[matched] == fx.tokens[size_t(matched)]) ++matched;
    std::printf("DFLASH2 %s hidden_rel_rms %.5f top16_overlap_min %d/%d path %d/%d tokens", bf16 ? "bf16" : "q8",
                std::sqrt(err / mag), overlap, fx.K, matched, fx.block - 1);
    for (int i = 0; i < fx.block - 1; ++i) std::printf(" %u", p[i]);
    std::printf("\n");
    return {std::sqrt(err / mag), overlap, matched};
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 4) { std::fprintf(stderr, "usage: test-mlx-inkling2-dflash2 DRAFTER_DIR CHECKPOINT_DIR FIXTURE\n"); return 2; }
        mx::set_default_device(mx::Device::gpu);
        const auto fx = read(argv[3]);
        if (fx.m != int(std::size(kTokens))) throw std::runtime_error("fixture context length");
        tqllm::HfFolder ck(argv[2]);
        const auto embed = tensor(ck, "model.llm.embed.weight"), norm = tensor(ck, "model.llm.embed_norm.weight");
        const auto unembed = mx::contiguous(mx::slice(tensor(ck, "model.llm.unembed.weight"), {0, 0}, {fx.vocab, fx.D}));
        mx::eval(unembed);
        std::printf("reference tokens");
        for (auto t : fx.tokens) std::printf(" %u", t);
        std::printf("\n");
        const auto exact = run(fx, argv[1], embed, norm, unembed, true);
        const auto served = run(fx, argv[1], embed, norm, unembed, false);
        const bool pass = exact.hidden_rms < 0.12 && exact.overlap >= 12 && exact.path >= 1 &&
                          served.hidden_rms < 0.25 && served.overlap >= 10;
        std::printf("DFLASH2 %s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
}
