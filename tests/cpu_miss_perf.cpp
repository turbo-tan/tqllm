// CPU expert-miss pool cost on this host (Flash-Next shapes: hidden 2560, expert ff 640). Where a miss expert's time
// goes: rebuilding GGUF rows from the pinned device layout vs the int8 dots, single thread and in the pool. The pool
// output from the device layout must match the output from the same bytes as GGUF rows (to float rounding). 64 distinct experts (> L3) per run.
#include "fn_cpu_experts.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr int H = 2560, FF = 640, kExperts = 64;
using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

// one matrix in the pinned device layout (untiled) + its GGUF rows rebuilt by the pool's own cpu_gguf_row
struct MatBuf {
    int fmt, type, rows, cols, bpb, aux_rec; size_t row_bytes;
    std::vector<char> dev, aux, gguf;
};
MatBuf make(int fmt, int rows, int cols, std::mt19937 & rng) {
    MatBuf m{fmt, 0, rows, cols, 0, 0, 0, {}, {}, {}};
    const int nb = cols / 32, kb = cols / 256;
    if (fmt == 5) { m.type = 18; m.bpb = 12; m.aux_rec = 4; m.row_bytes = size_t(kb) * 98; }
    if (fmt == 4) { m.type = 21; m.bpb = 12; m.aux_rec = 16; m.row_bytes = size_t(kb) * 110; }
    if (fmt == 10) { m.type = 22; m.bpb = 8; m.aux_rec = 20; m.row_bytes = size_t(kb) * 82; }
    if (fmt == 14) { m.type = 20; m.bpb = 16; m.aux_rec = 0; m.row_bytes = size_t(nb) * 18; }
    m.dev.resize(size_t(rows) * nb * m.bpb);
    for (auto & b : m.dev) b = char(rng());
    const uint16_t one = 0x3c00;  // fp16 1.0 scales: no denormals in the timing
    if (fmt == 14) { m.aux.resize(size_t(rows) * nb * 2); for (size_t i = 0; i < m.aux.size() / 2; ++i) std::memcpy(m.aux.data() + 2 * i, &one, 2); }
    else {
        m.aux.resize(size_t(rows) * kb * m.aux_rec);
        for (auto & b : m.aux) b = char(rng());
        for (size_t i = 0; i < m.aux.size() / m.aux_rec; ++i) std::memcpy(m.aux.data() + i * m.aux_rec, &one, 2);
    }
    m.gguf.resize(size_t(rows) * m.row_bytes);
    for (int r = 0; r < rows; ++r)
        if (!cpu_gguf_row(fmt, m.dev.data(), m.aux.data(), r, cols, false, m.gguf.data() + size_t(r) * m.row_bytes))
            throw std::runtime_error("cpu_gguf_row: format not handled");
    return m;
}
struct Expert { MatBuf g, u, d; };
}  // namespace

int main() {
    std::mt19937 rng(7);
    const struct { int gu, dn; const char * name; } mixes[] = {{5, 14, "IQ3_XXS + IQ4_NL"}, {4, 14, "IQ3_S + IQ4_NL"}, {10, 14, "IQ2_S + IQ4_NL"}};
    std::vector<float> x(size_t(8) * H);
    for (auto & v : x) v = std::uniform_real_distribution<float>(-1.f, 1.f)(rng);
    std::vector<float> xr(x.size(), 0.f), xd(x.size() / 32); std::vector<int8_t> xq(x.size());
    cpu_prepare_x(x.data(), xr.data(), xq.data(), xd.data(), int(x.size()));
    const int threads = std::max(1, std::min(15, int(std::thread::hardware_concurrency()) / 2 - 1));
    CpuMissPool pool(threads, H, FF);
    std::vector<__nv_bfloat16> out_a(size_t(8) * 8 * H), out_b(out_a.size());
    for (const auto & mix : mixes) {
        std::vector<Expert> ex;
        for (int e = 0; e < kExperts; ++e) ex.push_back({make(mix.gu, FF, H, rng), make(mix.gu, FF, H, rng), make(mix.dn, H, FF, rng)});
        // single thread: rebuild vs dot per gate matrix and per down matrix (one token column)
        std::vector<char> row(8192); double t_rb_gu = 0, t_dot_gu = 0, t_rb_dn = 0, t_dot_dn = 0; float sink = 0.f;
        const int8_t * xqc[1] = {xq.data()}; const float * xdc[1] = {xd.data()};
        using DotQ8 = void (*)(const uint8_t *, const int8_t * const *, const float * const *, int, int, float *);
        const DotQ8 dot = mix.gu == 5 ? cpu_dot_iq3xxs_q8 : mix.gu == 4 ? cpu_dot_iq3s_q8 : cpu_dot_iq2s_q8;
        std::vector<int8_t> aq(FF); std::vector<float> ad(FF / 32); cpu_quant_q8(x.data(), FF, aq.data(), ad.data());
        const int8_t * aqc[1] = {aq.data()}; const float * adc[1] = {ad.data()};
        for (auto & e : ex) {
            auto t = clk::now();
            for (int r = 0; r < FF; ++r) cpu_gguf_row(e.g.fmt, e.g.dev.data(), e.g.aux.data(), r, H, false, row.data());
            t_rb_gu += ms_since(t); t = clk::now();
            for (int r = 0; r < FF; ++r) { float o; dot(reinterpret_cast<const uint8_t *>(e.g.gguf.data() + size_t(r) * e.g.row_bytes), xqc, xdc, 1, H / 256, &o); sink += o; }
            t_dot_gu += ms_since(t); t = clk::now();
            for (int r = 0; r < H; ++r) cpu_gguf_row(e.d.fmt, e.d.dev.data(), e.d.aux.data(), r, FF, false, row.data());
            t_rb_dn += ms_since(t); t = clk::now();
            for (int r = 0; r < H; ++r) { float o; cpu_dot_iq4nl_q8_mc(reinterpret_cast<const uint8_t *>(e.d.gguf.data() + size_t(r) * e.d.row_bytes), aqc, adc, 1, FF / 32, &o); sink += o; }
            t_dot_dn += ms_since(t);
        }
        const double n = kExperts;
        std::printf("%-17s 1 thread per expert matrix: gate rebuild %.3f ms, gate dot %.3f ms | down rebuild %.3f ms, down dot %.3f ms (sink %.1f)\n",
                    mix.name, t_rb_gu / n, t_dot_gu / n, t_rb_dn / n, t_dot_dn / n, double(sink != sink));
        // the pool, as a decode layer call: k distinct experts, `cols` token columns each, rebuilt rows vs GGUF rows
        for (const int k : {1, 4, 8}) for (const int cols : {1, 4}) {
            double t_dev = 0, t_gguf = 0; int calls = 0;
            for (int base = 0; base + k <= kExperts; base += k, ++calls) for (const bool gguf : {false, true}) {
                std::vector<CpuMissPool::Job> jobs;
                for (int q = 0; q < k; ++q) {
                    const Expert & e = ex[size_t(base + q)]; CpuMissPool::Job j;
                    const MatBuf * ms[3] = {&e.g, &e.u, &e.d};
                    for (int m = 0; m < 3; ++m) {
                        j.w[m] = ms[m]->gguf.data(); j.type[m] = uint32_t(ms[m]->type); j.row_bytes[m] = ms[m]->row_bytes;
                        if (!gguf) { j.dev[m] = ms[m]->dev.data(); j.aux[m] = ms[m]->aux.data(); j.fmt[m] = ms[m]->fmt; }
                    }
                    for (int c = 0; c < cols; ++c) j.cols_t[j.ncols++] = c;
                    jobs.push_back(j);
                }
                auto & out = gguf ? out_b : out_a;
                const auto t = clk::now(); pool.start(x.data(), cols, std::move(jobs), out.data()); pool.wait();
                (gguf ? t_gguf : t_dev) += ms_since(t);
                if (gguf) {  // device-layout IQ4_NL dots sum the int32 lanes in another order: equal to float rounding
                    float err = 0.f, mag = 0.f;
                    for (size_t q = 0; q < size_t(k) * cols * H; ++q) {
                        const float a = __bfloat162float(out_a[q]), b = __bfloat162float(out_b[q]);
                        err = std::max(err, std::fabs(a - b)); mag = std::max(mag, std::fabs(b));
                    }
                    if (!(err <= 1e-2f * mag)) throw std::runtime_error("pool output from the device layout differs from GGUF rows");
                }
            }
            std::printf("%-17s pool %2d threads, %d experts x %d cols: %.3f ms per call device layout, %.3f ms GGUF rows (%.3f / %.3f ms per expert)\n",
                        mix.name, threads, k, cols, t_dev / calls, t_gguf / calls, t_dev / calls / k, t_gguf / calls / k);
        }
    }
    std::printf("cpu_miss_perf: PASS\n");
    return 0;
}
