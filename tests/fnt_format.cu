// FNT trellis expert format (src/formats/weight/fnt.cu / fnt.cpp, docs/reference/formats.md "FNT"):
//  - dequant_h equals an independent decoder written here from the format text (bit by bit window extraction, no
//    funnel shifts), rounded to FP16: bit-identical;
//  - dequantize_host (the CPU fallback / cold store) equals the same oracle in FP32: bit-identical;
//  - expert_matvec agrees with a double oracle (rotated FP16 inputs) and the plain-basis double product within FP error;
//  - dedup, grouped (prefill) and the plain matvec give the bits of expert_matvec per column; invalid ids give NaN.
#include "weight_format.hpp"
#include "gguf.hpp"
#include "dequant.hpp"
#include "fnt.cuh"
#include "quant_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace tqllm;
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static void require(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
template <class T> struct Dev {
    T * p = nullptr;
    explicit Dev(size_t n) { check(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    void put(const std::vector<T> & x) { check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) { std::vector<T> x(n); check(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x; }
};
static const uint32_t S1[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};
static float sgn(int i) { return (S1[i / 32] >> (i % 32)) & 1u ? -1.f : 1.f; }

// independent oracle: the rotated-basis values of one 640-group (FP32: FP16 code times the FP32 scale)
static void oracle_group(const uint8_t * blk, int kb, float * out) {
    uint16_t hs; std::memcpy(&hs, blk, 2);
    const float d = half_to_float(hs);
    for (int b = 0; b < 5; ++b) {
        const uint8_t * B = blk + 2 + b * kb * 8;
        const int nbits = kb * 64;
        for (int p = 0; p < 64; ++p) {
            uint32_t w = 0;
            for (int i = 15; i >= 0; --i) {  // most significant first, unlike the format's own decoder
                const int bit = (p * kb + i) % nbits;
                w = (w << 1) | ((B[bit / 8] >> (bit % 8)) & 1u);
            }
            const uint32_t y = (w * 0x9E3779B1u + 0x7F4A7C15u) % 65536u;
            const uint32_t z = y * 0x2C1B3C6Du + 0x9E3779B9u, z2 = z * 0x297A2D39u + 0x7F4A7C15u;
            for (int e = 0; e < 2; ++e) {
                const uint16_t a = uint16_t((((z >> (16 * e)) & 0x8FFFu) ^ 0x3B60u)), c = uint16_t((((z2 >> (16 * e)) & 0x8FFFu) ^ 0x3B60u));
                const float v = __half2float(__float2half_rn(half_to_float(a) + half_to_float(c)));
                out[b * 128 + 2 * p + e] = v * d;
            }
        }
    }
}
// plain-basis value: D H y / sqrt(128) per 128, in fnt.cu's butterfly order
static void unrot(float * v) {
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; ++i)
            if (!(i & h)) { const float a = v[i], c = v[i + h]; v[i] = a + c; v[i + h] = a - c; }
    for (int i = 0; i < 128; ++i) v[i] = v[i] * 0.08838834764831845f * sgn(i);
}
static void rot_h(const __half * x, __half * o) {  // forward rotation of one 128-block, rounded to FP16 (fnt.cu stage_rot)
    float v[128];
    for (int i = 0; i < 128; ++i) v[i] = __half2float(x[i]) * sgn(i);
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; ++i)
            if (!(i & h)) { const float a = v[i], c = v[i + h]; v[i] = a + c; v[i + h] = a - c; }
    for (int i = 0; i < 128; ++i) o[i] = __float2half_rn(v[i] * 0.08838834764831845f);
}

static void run(int kb, int rows, int cols, int n, bool per_expert) {
    const int experts = 3, topk = 2;
    const uint32_t type = GGML_FNT4 + uint32_t(kb - 4);
    const auto * fmt = weight_format(type);
    require(fmt && fmt->has_expert_matvec() && fmt->has_expert_dedup() && fmt->basis() == Basis::Plain, "FNT format missing");
    std::mt19937 rng(kb * 7919 + cols + n);
    const size_t gb = 2 + 40 * size_t(kb), ng = size_t(experts) * rows * (cols / 640);
    std::vector<char> raw(ng * gb);
    for (size_t g = 0; g < ng; ++g) {
        const float s = 0.01f * float(1 + rng() % 50) * ((rng() & 7) ? 1.f : -1.f);
        const __half h = __float2half_rn(s);
        std::memcpy(raw.data() + g * gb, &h, 2);
        for (size_t j = 2; j < gb; ++j) raw[g * gb + j] = char(rng() & 255);
    }
    std::vector<float> rotv(ng * 640), plain(ng * 640);
    for (size_t g = 0; g < ng; ++g) {
        oracle_group(reinterpret_cast<const uint8_t *>(raw.data()) + g * gb, kb, rotv.data() + g * 640);
        std::memcpy(plain.data() + g * 640, rotv.data() + g * 640, 640 * 4);
        for (int b = 0; b < 5; ++b) unrot(plain.data() + g * 640 + b * 128);
    }
    // CPU dequant (fallback path) = oracle in FP32
    std::vector<float> host(ng * 640);
    dequantize_host(type, raw.data(), host.data(), int64_t(ng) * 640);
    require(std::memcmp(host.data(), plain.data(), host.size() * 4) == 0, "FNT dequantize_host differs from the oracle");
    std::vector<const void *> owned;
    auto up = [&](const void * h, size_t bytes) -> const void * {
        void * d = nullptr; check(cudaMalloc(&d, bytes)); check(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); owned.push_back(d); return d;
    };
    const Mat m = fmt->load(raw, experts * rows, cols, up, nullptr);
    Dev<__half> dq(ng * 640);
    fmt->dequant_h(m, dq.p, nullptr);
    const auto dqh = dq.get(ng * 640);
    for (size_t i = 0; i < dqh.size(); ++i)
        { const __half want = __float2half_rn(plain[i]); require(std::memcmp(&dqh[i], &want, 2) == 0, "FNT dequant_h differs from the oracle"); }
    // matvec
    const int A = n * topk, nin = per_expert ? A : n;
    std::vector<int> ids(A);
    for (int a = 0; a < A; ++a) ids[a] = int((a * 5 + 1 + rng() % 2) % experts);
    std::vector<__half> x(size_t(nin) * cols);
    for (auto & v : x) v = __float2half_rn(std::normal_distribution<float>(0.f, 1.f)(rng));
    Dev<int> di(A); di.put(ids);
    Dev<__half> dx(x.size()); dx.put(x);
    Dev<__nv_bfloat16> dy(size_t(A) * rows);
    fmt->expert_matvec(m, dx.p, dy.p, rows, di.p, n, topk, per_expert, nullptr);
    const auto y = dy.get(size_t(A) * rows);
    std::vector<__half> xr(x.size());
    for (size_t i = 0; i < x.size(); i += 128) rot_h(x.data() + i, xr.data() + i);
    double worst = 0, worst_plain = 0;
    for (int a = 0; a < A; ++a)
        for (int r = 0; r < rows; ++r) {
            const size_t row = size_t(ids[a]) * rows + r, in = size_t(per_expert ? a : a / topk) * cols;
            double e = 0, ep = 0, mag = 0;
            for (int j = 0; j < cols; ++j) {
                e += double(rotv[row * cols + j]) * __half2float(xr[in + j]);
                ep += double(plain[row * cols + j]) * __half2float(x[in + j]);
                mag += std::abs(double(rotv[row * cols + j]) * __half2float(xr[in + j]));
            }
            const double got = __bfloat162float(y[size_t(a) * rows + r]);
            worst = std::max(worst, std::abs(got - e) / (mag + 1e-9));
            worst_plain = std::max(worst_plain, std::abs(got - ep) / (mag + 1e-9));
        }
    require(worst < 4e-3, "FNT expert_matvec differs from the rotated double oracle");
    require(worst_plain < 2e-2, "FNT expert_matvec differs from the plain-basis double product");
    // dedup: groups of distinct experts with their assignments (<= kDedupCols columns each)
    {
        constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
        std::vector<int> dmap;
        for (int e = 0; e < experts; ++e) {
            std::vector<int> as;
            for (int a = 0; a < A; ++a) if (ids[a] == e) as.push_back(a);
            for (size_t i = 0; i < as.size(); i += DC) {
                std::vector<int> g(DS, 0);
                g[0] = e; g[1] = int(std::min<size_t>(DC, as.size() - i));
                for (int c = 0; c < g[1]; ++c) { g[2 + c] = per_expert ? as[i + c] : as[i + c] / topk; g[2 + DC + c] = as[i + c]; }
                dmap.insert(dmap.end(), g.begin(), g.end());
            }
        }
        const int groups = int(dmap.size() / DS);
        Dev<int> dm(dmap.size()); dm.put(dmap);
        Dev<__nv_bfloat16> dd(size_t(A) * rows);
        fmt->expert_matvec_dedup(m, dx.p, dd.p, rows, dm.p, groups, nullptr);
        const auto yd = dd.get(size_t(A) * rows);
        require(std::memcmp(yd.data(), y.data(), y.size() * 2) == 0, "FNT dedup differs from expert_matvec");
    }
    // grouped (prefill)
    {
        const int capacity = A;
        std::vector<int> counts(experts, 0), groups(size_t(experts) * capacity, 0);
        for (int a = 0; a < A; ++a) groups[size_t(ids[a]) * capacity + counts[ids[a]]++] = a;
        Dev<int> dc(counts.size()), dg(groups.size()); dc.put(counts); dg.put(groups);
        Dev<__nv_bfloat16> yg(size_t(A) * rows);
        require(fmt->expert_gemm(m, dx.p, yg.p, rows, dc.p, dg.p, capacity, experts, topk, per_expert, nullptr), "FNT expert_gemm");
        const auto g = yg.get(size_t(A) * rows);
        require(std::memcmp(g.data(), y.data(), y.size() * 2) == 0, "FNT grouped differs from expert_matvec");
    }
    // plain matvec of expert 0 (F32) = expert_matvec on assignment rows with id 0, per_expert inputs
    if (per_expert) {
        Mat m0 = m; m0.rows = rows;
        Dev<float> yf(size_t(nin) * rows);
        fmt->matvec_f32(m0, dx.p, yf.p, nin, nullptr);
        const auto f = yf.get(size_t(nin) * rows);
        for (int a = 0; a < A; ++a)
            if (ids[a] == 0)
                for (int r = 0; r < rows; ++r)
                {
                    const __nv_bfloat16 b = __float2bfloat16_rn(f[size_t(a) * rows + r]);
                    require(std::memcmp(&b, &y[size_t(a) * rows + r], 2) == 0, "FNT matvec_f32 differs from expert_matvec");
                }
    }
    // the decode split path (fn_layer_forward.cpp, host maps): the hits run first with the misses' ids set to -1, then the
    // misses with the hits' ids set to -1, into the same output. An id of -1 writes nothing (the u4h kernels' rule), so the
    // two launches give expert_matvec's rows bit for bit (a -1 that wrote NaN put NaN into every hit row of a one-row
    // step with both hits and misses).
    for (int mask = 1; mask < (1 << std::min(A, 4)); ++mask) {
        std::vector<int> hit(ids), miss(ids);
        for (int a = 0; a < A; ++a) ((mask >> (a % 4)) & 1 ? miss : hit)[a] = -1;
        Dev<int> dh(A), dm(A); dh.put(hit); dm.put(miss);
        Dev<__nv_bfloat16> ys(size_t(A) * rows);
        const std::vector<__nv_bfloat16> fill(size_t(A) * rows, __float2bfloat16(-123.f));
        ys.put(fill);
        fmt->expert_matvec(m, dx.p, ys.p, rows, dh.p, n, topk, per_expert, nullptr);
        const auto part = ys.get(size_t(A) * rows);
        for (int a = 0; a < A; ++a)
            if (hit[a] < 0)
                for (int r = 0; r < rows; ++r)
                    require(std::memcmp(&part[size_t(a) * rows + r], &fill[0], 2) == 0, "FNT expert_matvec wrote an assignment with id -1");
        fmt->expert_matvec(m, dx.p, ys.p, rows, dm.p, n, topk, per_expert, nullptr);
        require(std::memcmp(ys.get(size_t(A) * rows).data(), y.data(), y.size() * 2) == 0,
                "FNT hit + miss launches (ids -1 for the other part) differ from one expert_matvec");
    }
    // invalid id -> NaN
    ids[0] = experts; di.put(ids);
    fmt->expert_matvec(m, dx.p, dy.p, rows, di.p, n, topk, per_expert, nullptr);
    require(std::isnan(__bfloat162float(dy.get(1)[0])), "FNT invalid expert id did not fail closed");
    check(cudaDeviceSynchronize());
    for (auto * p : owned) cudaFree(const_cast<void *>(p));
    std::printf("FNT%d rows %d cols %d n %d per_expert %d: ok (rel err vs oracle %.2e, vs plain %.2e)\n", kb, rows, cols, n,
                int(per_expert), worst, worst_plain);
}

int main() {
    try {
        for (int kb = 4; kb <= 7; ++kb) {
            run(kb, 70, 2560, 3, false);
            run(kb, 130, 640, 2, true);
        }
        run(5, 640, 2560, 1, false);  // one row (decode without drafts): the split path's hit / miss launches
        run(6, 2560, 640, 1, true);
        run(5, 640, 2560, 9, false);
        run(6, 640, 2560, 60, false);  // > 32 tokens per expert: two prefill GEMM chunks
        run(7, 2560, 640, 40, true);
        std::printf("fnt_format: all passed\n");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fnt_format FAILED: %s\n", e.what());
        return 1;
    }
}
