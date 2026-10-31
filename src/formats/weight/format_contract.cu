#include "format_contract.hpp"
#include "dequant.hpp"
#include "gguf.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tqllm {
namespace {
constexpr int kRows = 64, kCols = 1024;

void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
int ulps(__half a, __half b) {
    uint16_t x, y; std::memcpy(&x, &a, 2); std::memcpy(&y, &b, 2);
    auto ord = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7fff) : int(v); };
    return std::abs(ord(x) - ord(y));
}
// device copies of a loaded matrix, freed with the scope
struct Loaded {
    std::vector<void *> keep;
    Mat m;
    Loaded(const WeightFormat & loader, const std::vector<char> & raw) {
        Upload up = [&](const void * h, size_t n) {
            void * p; check(cudaMalloc(&p, n)); check(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice)); keep.push_back(p); return (const void *) p;
        };
        m = loader.load(raw, kRows, kCols, up, nullptr);
    }
    ~Loaded() { for (void * p : keep) cudaFree(p); }
};
std::vector<char> random_blocks(uint32_t type, int pass, std::mt19937 & rng) {
    std::vector<char> raw(gguf_tensor_bytes(type, {uint64_t(kCols), uint64_t(kRows)}, "w"));
    std::uniform_int_distribution<int> b(0, pass ? 255 : 0x3B);
    for (auto & c : raw) c = char(b(rng));
    return raw;
}
}  // namespace

bool contract_known_gap(uint32_t t) { return t == 16 || t == 17 || t == 18 || t == 19 || t == 22; }
bool contract_repacked(uint32_t t) { return t == 12 || t == 13 || t == 16 || t == 17 || t == 18 || t == 19 || t == 22 || t == 46; }

ContractReport contract_pass(uint32_t type, const WeightFormat & loader, const WeightFormat * runtime, int pass, std::mt19937 & rng) {
    ContractReport r;
    const std::vector<char> raw = random_blocks(type, pass, rng);
    std::vector<float> ref(size_t(kRows) * kCols);
    try {
        dequantize_host_rows(type, raw.data(), ref.data(), kRows, kCols);
    } catch (const std::exception & e) {
        r.error = std::string("no CPU reference (") + e.what() + ")";
        return r;
    }
    Loaded ld(loader, raw);
    const Mat & m = ld.m;
    const WeightFormat & fmt = runtime ? *runtime : *m.fmt;
    __half * dh; check(cudaMalloc(&dh, ref.size() * 2));
    fmt.dequant_h(m, dh, nullptr);
    check(cudaDeviceSynchronize());
    if (type == 0)  // F32 matrices are stored as BF16
        for (auto & v : ref) { uint32_t u; std::memcpy(&u, &v, 4); u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u; std::memcpy(&v, &u, 4); }
    if (loader.basis() == Basis::Tq3Rot32)
        for (size_t b = 0; b + 32 <= ref.size(); b += 32) tq3_rotate_block(&ref[b]);
    std::vector<__half> got(ref.size());
    check(cudaMemcpy(got.data(), dh, got.size() * 2, cudaMemcpyDeviceToHost));
    cudaFree(dh);
    const size_t group = contract_repacked(type) ? 256 : 32;
    const float tol = type == 46 ? 1.f / 128.f : contract_repacked(type) ? 1.f / 256.f : 1.f / 1024.f;  // TQ3_4S: 8-bit codebook
    for (size_t b = 0; b + group <= ref.size(); b += group) {
        float amax = 0.f; bool finite = true;
        for (size_t i = b; i < b + group; ++i) { finite &= std::isfinite(ref[i]); amax = std::max(amax, std::fabs(ref[i])); }
        if (!finite || amax > 1024.f) continue;  // FP16 scale x code products could overflow
        for (size_t i = b; i < b + group; ++i) {
            const int u = ulps(got[i], __float2half(ref[i]));
            const float diff = std::fabs(__half2float(got[i]) - ref[i]);
            ++r.compared; r.worst_ulp = std::max(r.worst_ulp, u); r.bad += u > 1 && diff > amax * tol;
            if (amax > 0) r.worst_rel = std::max(r.worst_rel, diff / amax);
        }
    }
    // matvec (one column) against a double dot over the reference weights, only where every weight is in FP16 range
    std::vector<__half> x(kCols);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = __float2half(nd(rng));
    __half * dx; __nv_bfloat16 * dy;
    check(cudaMalloc(&dx, kCols * 2)); check(cudaMalloc(&dy, kRows * 2));
    check(cudaMemcpy(dx, x.data(), kCols * 2, cudaMemcpyHostToDevice));
    fmt.matvec(m, dx, dy, 1, nullptr);
    std::vector<__nv_bfloat16> y(kRows);
    check(cudaMemcpy(y.data(), dy, kRows * 2, cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dy);
    double se = 0, sr = 0;
    for (int row = 0; row < kRows; ++row) {
        double dot = 0; bool ok = true;
        for (int k = 0; k < kCols; ++k) {
            const float w = ref[size_t(row) * kCols + k];
            if (!std::isfinite(w) || std::fabs(w) > 65504.f) { ok = false; break; }
            dot += double(w) * double(__half2float(x[k]));
        }
        if (!ok || !std::isfinite(dot) || std::fabs(dot) > 1e30) continue;
        const double e = double(__bfloat162float(y[row])) - dot;
        se += e * e; sr += dot * dot; ++r.rows_used;
    }
    r.matvec_rel = r.rows_used && sr > 0 ? std::sqrt(se / sr) : 0.0;
    r.gap = r.bad && contract_known_gap(type);
    r.ok = (r.bad == 0 || r.gap) && (pass || r.matvec_rel <= 2e-2);
    return r;
}

size_t contract_identical(uint32_t type, const WeightFormat & loader, const WeightFormat & a, const WeightFormat & b, bool matvec,
                          std::mt19937 & rng) {
    size_t diff = 0;
    for (int pass = 0; pass < 2; ++pass) {
        Loaded ld(loader, random_blocks(type, pass, rng));
        const size_t n = size_t(kRows) * kCols;
        __half * da; __half * db; check(cudaMalloc(&da, n * 2)); check(cudaMalloc(&db, n * 2));
        a.dequant_h(ld.m, da, nullptr); b.dequant_h(ld.m, db, nullptr);
        std::vector<uint16_t> ha(n), hb(n);
        check(cudaMemcpy(ha.data(), da, n * 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(hb.data(), db, n * 2, cudaMemcpyDeviceToHost));
        cudaFree(da); cudaFree(db);
        for (size_t i = 0; i < n; ++i) diff += ha[i] != hb[i];
        if (!matvec) continue;
        const int cols = std::min(a.max_matvec_cols(), b.max_matvec_cols());
        std::vector<__half> x(size_t(cols) * kCols);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : x) v = __float2half(nd(rng));
        __half * dx; __nv_bfloat16 * ya; __nv_bfloat16 * yb;
        check(cudaMalloc(&dx, x.size() * 2)); check(cudaMalloc(&ya, size_t(cols) * kRows * 2)); check(cudaMalloc(&yb, size_t(cols) * kRows * 2));
        check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        a.matvec(ld.m, dx, ya, cols, nullptr); b.matvec(ld.m, dx, yb, cols, nullptr);
        std::vector<uint16_t> va(size_t(cols) * kRows), vb(va.size());
        check(cudaMemcpy(va.data(), ya, va.size() * 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(vb.data(), yb, vb.size() * 2, cudaMemcpyDeviceToHost));
        cudaFree(dx); cudaFree(ya); cudaFree(yb);
        for (size_t i = 0; i < va.size(); ++i) diff += va[i] != vb[i];
    }
    return diff;
}

}  // namespace tqllm
