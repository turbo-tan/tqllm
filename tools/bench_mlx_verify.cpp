// Decode / verify matmul bandwidth at 16 input rows (one spec step) on the Qwen3.8-27B shapes: our NAX 16-row kernel
// (nax_affine_matmul, 4-bit affine and TQ4) vs MLX quantized_matmul (affine 4-bit g64) and dequant + matmul. 32 calls per
// eval like a stack of layers; GB/s of packed weight read.
#include "nax_affine.hpp"
#include "tq4.hpp"
#include "draft_head.hpp"
#include "q6.hpp"
#include "weights.hpp"
#include <mlx/random.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <tuple>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
static double ms(const std::function<void()>& f) {
    f(); double b = 1e30;
    for (int i = 0; i < 7; ++i) {auto t = std::chrono::steady_clock::now(); f(); b = std::min(b, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());}
    return b;
}
int main() {
    std::mt19937 rng(5);
    auto bf = [](float v) {uint32_t u; std::memcpy(&u, &v, 4); return uint16_t(u >> 16);};
    const int calls = 32;
    for (auto [name, rows, k] : {std::tuple{"gate|up", 17408, 5120}, {"down", 5120, 17408}, {"gdn qkv", 10240, 5120},
                                 {"gdn z", 6144, 5120}, {"gdn out", 5120, 6144}, {"attn q", 12288, 5120}}) {
        std::vector<uint8_t> codes(size_t(rows) * k);
        for (auto& c : codes) c = uint8_t(rng() & 15u);
        std::vector<uint16_t> sc(size_t(rows) * (k / 64)), bi(sc.size());
        for (size_t i = 0; i < sc.size(); ++i) {sc[i] = bf(0.002f + float(rng() % 100) * 4e-5f); bi[i] = bf(-0.015f + float(rng() % 100) * 3e-4f);}
        auto nax = nax_affine_load(nax_affine_pack(codes, sc, bi, rows, k, 4, 64), rows, k, 4, 64);
        std::vector<uint8_t> packed(size_t(rows) * (k / 2));
        for (auto& c : packed) c = uint8_t(rng());
        std::vector<mx::float16_t> tsc(size_t(rows) * (k / 128));
        for (auto& v : tsc) v = mx::float16_t(1e-4f + float(rng() % 100) * 4e-6f);
        auto tq = nax_from_tq4(Tq4Weights{mx::array(packed.data(), {rows, k / 2}, mx::uint8), mx::array(tsc.data(), {rows, k / 128}, mx::float16), true});
        auto w16 = nax_affine_dequant(nax);
        auto q = mx::quantize(w16, 64, 4);
        auto x = mx::astype(mx::random::normal({16, k}), mx::float16);
        mx::eval(w16, q[0], q[1], q[2], x);
        const double bytes = double(rows) * k / 2 + double(rows) * (k / 64) * 4;
        auto run = [&](auto f) {return ms([&] {std::vector<mx::array> o; for (int c = 0; c < calls; ++c) o.push_back(f()); mx::eval(o);}) / calls;};
        const double t_nax = run([&] {return nax_affine_matmul(nax, x, true);});
        const double t_tq4 = run([&] {return nax_affine_matmul(tq, x, true);});
        const double t_qmm = run([&] {return mx::quantized_matmul(x, q[0], q[1], q[2], true, 64, 4);});
        std::printf("VERIFY16 %-8s %5dx%-5d ours affine %.3f ms (%.0f GB/s)  ours tq4 %.3f ms (%.0f GB/s)  mlx qmm %.3f ms (%.0f GB/s)\n",
                    name, rows, k, t_nax, bytes / t_nax * 1e-6, t_tq4, bytes / t_tq4 * 1e-6, t_qmm, bytes / t_qmm * 1e-6);
    }
    // 1. streaming bandwidth: read 2 x 1 GiB, write 1 GiB
    {
        auto a1 = mx::astype(mx::random::normal({1 << 28}), mx::float32), a2 = mx::astype(mx::random::normal({1 << 28}), mx::float32);
        mx::eval(a1, a2);
        const double t = ms([&] {mx::eval(mx::add(a1, a2));});
        std::printf("STREAM add 3 GiB moved: %.2f ms = %.0f GB/s\n", t, 3.0 * (1 << 30) / t * 1e-6);
    }
    // 2. the gate|up shape over 16 distinct weight copies (1.4 GB working set, no cache reuse), like 16 layers in a row
    {
        const int rows = 17408, k = 5120;
        std::vector<uint8_t> codes(size_t(rows) * k);
        for (auto& c : codes) c = uint8_t(rng() & 15u);
        std::vector<uint16_t> sc(size_t(rows) * (k / 64)), bi(sc.size());
        for (size_t i = 0; i < sc.size(); ++i) {sc[i] = bf(0.002f + float(rng() % 100) * 4e-5f); bi[i] = bf(-0.015f + float(rng() % 100) * 3e-4f);}
        const auto payload = nax_affine_pack(codes, sc, bi, rows, k, 4, 64);
        std::vector<NaxAffine> ws;
        for (int c = 0; c < 16; ++c) {ws.push_back(nax_affine_load(payload, rows, k, 4, 64)); mx::eval(ws.back().words, ws.back().scales, ws.back().biases);}
        auto x = mx::astype(mx::random::normal({16, k}), mx::float16);
        mx::eval(x);
        const double bytes = 16.0 * (double(rows) * k / 2 + double(rows) * (k / 64) * 4);
        const double t = ms([&] {std::vector<mx::array> o; for (auto& w : ws) o.push_back(nax_affine_matmul(w, x, true)); mx::eval(o);});
        std::printf("VERIFY16 distinct gate|up x16: %.2f ms = %.0f GB/s (no weight reuse)\n", t, bytes / t * 1e-6);
        // 3. the same 16 matmuls as a dependent chain (each input is the previous output's first 5120 columns)
        const double t2 = ms([&] {auto y = x; for (auto& w : ws) y = mx::astype(mx::slice(nax_affine_matmul(w, y, true), {0, 0}, {16, k}), mx::float16); mx::eval(y);});
        std::printf("VERIFY16 dependent chain x16: %.2f ms = %.0f GB/s (+%.3f ms per dependent step)\n", t2, bytes / t2 * 1e-6, (t2 - t) / 16);
    }
    // 3c. the verify FFN as the engine runs it: 64 layers of distinct gate / up / down (8.6 GB, nothing cached),
    // fused SwiGLU then down, each layer's input the previous output (dependent chain, no other kernels between).
    // Compared with the in-situ FFN stage time this splits the step's FFN excess into kernel rate vs interleaving.
    {
        const int d = 5120, f = 17408, layers = 64;
        std::vector<uint8_t> codes(size_t(f) * d);
        std::vector<uint16_t> sc(size_t(f) * (d / 64)), bi(sc.size());
        for (auto& c : codes) c = uint8_t(rng() & 15u);
        for (size_t i = 0; i < sc.size(); ++i) {sc[i] = bf(0.002f + float(rng() % 100) * 4e-5f); bi[i] = bf(-0.015f + float(rng() % 100) * 3e-4f);}
        const auto up_payload = nax_affine_pack(codes, sc, bi, f, d, 4, 64);
        std::vector<uint8_t> dcodes(size_t(d) * f);
        std::vector<uint16_t> dsc(size_t(d) * (f / 64)), dbi(dsc.size());
        for (auto& c : dcodes) c = uint8_t(rng() & 15u);
        for (size_t i = 0; i < dsc.size(); ++i) {dsc[i] = bf(0.002f + float(rng() % 100) * 4e-5f); dbi[i] = bf(-0.015f + float(rng() % 100) * 3e-4f);}
        const auto down_payload = nax_affine_pack(dcodes, dsc, dbi, d, f, 4, 64);
        std::vector<NaxAffine> gate, up, down;
        for (int l = 0; l < layers; ++l) {
            gate.push_back(nax_affine_load(up_payload, f, d, 4, 64)); up.push_back(nax_affine_load(up_payload, f, d, 4, 64));
            down.push_back(nax_affine_load(down_payload, d, f, 4, 64));
            // distinct device copies: + 0 forces a fresh buffer per layer
            gate.back().words = mx::add(gate.back().words, mx::array(uint32_t(0), gate.back().words.dtype()));
            up.back().words = mx::add(up.back().words, mx::array(uint32_t(0), up.back().words.dtype()));
            down.back().words = mx::add(down.back().words, mx::array(uint32_t(0), down.back().words.dtype()));
            mx::eval(gate.back().words, gate.back().scales, gate.back().biases, up.back().words, up.back().scales, up.back().biases,
                     down.back().words, down.back().scales, down.back().biases);
        }
        const double bytes = double(layers) * 3 * (double(f) * d / 2 + double(f) * (d / 64) * 4);
        for (int rows : {8, 16}) {
            auto x = mx::astype(mx::random::normal({rows, d}), mx::float16);
            mx::eval(x);
            const double t = ms([&] {auto y = x; for (int l = 0; l < layers; ++l) {auto h = nax_affine_swiglu(gate[l], up[l], y); y = mx::astype(nax_affine_matmul(down[l], h, true), mx::float16);} mx::eval(y);});
            std::printf("FFN64 chain rows=%d: %.2f ms = %.0f GB/s (%.3f ms per layer; engine FFN stage = this + interleaving)\n",
                        rows, t, bytes / t * 1e-6, t / layers);
            const double t_sep = ms([&] {std::vector<mx::array> o; for (int l = 0; l < layers; ++l) o.push_back(nax_affine_matmul(down[l], nax_affine_swiglu(gate[l], up[l], x), true)); mx::eval(o);});
            std::printf("FFN64 independent rows=%d: %.2f ms = %.0f GB/s\n", rows, t_sep, bytes / t_sep * 1e-6);
        }
    }
    // 3d. the LM head shapes (j1038: the head kernels run at ~150 GB/s in the step, the FFN kernels at ~270): 4-bit affine
    // head (mlxc verify head), the 4-bit proposal head the drafter uses, and the Q6 head in NAX Q6 mode (TQ_L-4S), 16 rows
    {
        const int rows = 151936, k = 5120;
        std::vector<uint8_t> codes(size_t(rows) * k);
        for (auto& c : codes) c = uint8_t(rng() & 15u);
        std::vector<uint16_t> sc(size_t(rows) * (k / 64)), bi(sc.size());
        for (size_t i = 0; i < sc.size(); ++i) {sc[i] = bf(0.002f + float(rng() % 100) * 4e-5f); bi[i] = bf(-0.015f + float(rng() % 100) * 3e-4f);}
        auto nax = nax_affine_load(nax_affine_pack(codes, sc, bi, rows, k, 4, 64), rows, k, 4, 64);
        auto w16 = nax_affine_dequant(nax);
        mx::eval(nax.words, nax.scales, nax.biases, w16);
        auto head4 = draft_head4(Matrix{w16});
        std::vector<uint8_t> q6(size_t(rows) * (k / 256) * 210);
        for (auto& c : q6) c = uint8_t(rng());
        auto q6w = load_q6_blocks(q6, rows, k);
        auto q6nax = nax_from_q6(q6w);
        mx::eval(q6w.blocks, q6nax.words, q6nax.scales, q6nax.biases);
        const double b4 = double(rows) * k / 2 + double(rows) * (k / 64) * 4, b6 = double(rows) * (k / 256) * 210.0;
        for (int m : {1, 8, 16}) {
            auto x = mx::astype(mx::random::normal({m, k}), mx::float32);
            mx::eval(x);
            const double t4 = ms([&] {mx::eval(nax_affine_matmul(nax, x, true));});
            const double td = ms([&] {mx::eval(draft_head4_matmul(head4, x));});
            const double t6 = ms([&] {mx::eval(nax_affine_matmul(q6nax, x, true));});
            const double tq = ms([&] {mx::eval(q6_matmul(q6w, x));});
            std::printf("HEAD m=%2d 4-bit affine %.3f ms (%.0f GB/s)  draft_head4 %.3f ms (%.0f GB/s)  q6 nax %.3f ms (%.0f GB/s)  q6 generic %.3f ms (%.0f GB/s)\n",
                        m, t4, b4 / t4 * 1e-6, td, b4 / td * 1e-6, t6, b6 / t6 * 1e-6, tq, b6 / tq * 1e-6);
        }
    }
    // 4. dependent tiny kernels: per-kernel boundary cost
    {
        auto y = mx::astype(mx::random::normal({16, 5120}), mx::float32);
        mx::eval(y);
        for (int n : {500, 2000}) {
            const double t = ms([&] {auto z = y; for (int i = 0; i < n; ++i) z = mx::add(mx::multiply(z, mx::array(0.999f)), mx::array(0.001f)); mx::eval(z);});
            std::printf("CHAIN %d dependent tiny ops (2 kernels each): %.2f ms = %.1f us per kernel\n", n, t, t * 1000 / (2.0 * n));
        }
    }
    std::puts("VERIFY16_DONE");
}
