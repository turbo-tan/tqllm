// Flash-Next directional ablation (TQFN_ABLATE, fn_ablate.hpp): the projection kernels and the TQAB file parser.
#include "fn_ablate.hpp"
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static void expect(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }
template <typename T> struct Dev {
    T * p = nullptr; size_t n = 0;
    explicit Dev(const std::vector<T> & v) : n(v.size()) { check(cudaMalloc(&p, n * sizeof(T))); check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    explicit Dev(size_t count) : n(count) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const { std::vector<T> v(n); check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
static const int H = 2560;
static std::mt19937 rng(5);
static std::normal_distribution<float> nd(0.f, 1.f);
static std::vector<float> unit(int h) {
    std::vector<float> r(static_cast<size_t>(h)); double s = 0; for (auto & v : r) { v = nd(rng); s += double(v) * v; }
    for (auto & v : r) v = float(v / std::sqrt(s)); return r;
}

// fn_ablate_rows matches x - w (x . r) r computed on the host in double; at weight w the component left along r is
// (1 - w) of the old one (0 at w = 1)
static void rows_match_projection() {
    const auto r = unit(H); Dev<float> dr(r);
    for (int n : {1, 5, 300}) for (float w : {1.f, 0.6f}) {
        std::vector<float> x(size_t(n) * H); for (auto & v : x) v = 2.f * nd(rng);
        for (int t = 0; t < n; ++t) for (int j = 0; j < H; ++j) x[size_t(t) * H + j] += 3.f * r[size_t(j)];  // a large component along r
        Dev<float> dx(x);
        tqllm_cuda::fn_ablate_rows(dx.p, dr.p, w, H, n, nullptr); check(cudaDeviceSynchronize());
        const auto y = dx.get();
        double err = 0, dotmax = 0;
        for (int t = 0; t < n; ++t) {
            double d = 0, d2 = 0; for (int j = 0; j < H; ++j) d += double(x[size_t(t) * H + j]) * r[size_t(j)];
            for (int j = 0; j < H; ++j) {
                const double e = x[size_t(t) * H + j] - w * d * r[size_t(j)];
                err = std::max(err, std::fabs(e - y[size_t(t) * H + j])); d2 += y[size_t(t) * H + j] * double(r[size_t(j)]);
            }
            dotmax = std::max(dotmax, std::fabs(d2 - (1 - w) * d));
        }
        char why[128]; std::snprintf(why, sizeof(why), "n %d w %.1f: max err %.3g, residual dot %.3g", n, w, err, dotmax);
        expect(err < 1e-4 && dotmax < 1e-3, why);
    }
}

// two orthonormal directions (ndirs 2): the rows end orthogonal to both, in fp32 and in BF16 rows
static void two_directions_fp32_bf16() {
    const auto r = unit(H); std::vector<float> r2(size_t(2) * H); double d01 = 0, n1 = 0;
    for (int j = 0; j < H; ++j) { r2[size_t(j)] = r[size_t(j)]; r2[size_t(H + j)] = nd(rng); d01 += double(r2[size_t(H + j)]) * r[size_t(j)]; }
    for (int j = 0; j < H; ++j) { r2[size_t(H + j)] -= float(d01 * r[size_t(j)]); n1 += double(r2[size_t(H + j)]) * r2[size_t(H + j)]; }
    for (int j = 0; j < H; ++j) r2[size_t(H + j)] = float(r2[size_t(H + j)] / std::sqrt(n1));
    const int n = 12; std::vector<float> x(size_t(n) * H); for (auto & v : x) v = 2.f * nd(rng);
    std::vector<__nv_bfloat16> xb(x.size()); for (size_t i = 0; i < x.size(); ++i) xb[i] = __float2bfloat16_rn(x[i]);
    Dev<float> dr2(r2), dx(x); Dev<__nv_bfloat16> dxb(xb);
    tqllm_cuda::fn_ablate_rows(dx.p, dr2.p, 1.f, H, n, nullptr, 2); tqllm_cuda::fn_ablate_rows(dxb.p, dr2.p, 1.f, H, n, nullptr, 2);
    check(cudaDeviceSynchronize());
    const auto y = dx.get(); const auto yb = dxb.get();
    double worst = 0, worst_b = 0;
    for (int t = 0; t < n; ++t) for (int k = 0; k < 2; ++k) {
        double a = 0, b = 0;
        for (int j = 0; j < H; ++j) {
            a += double(y[size_t(t) * H + j]) * r2[size_t(k) * H + j];
            b += double(__bfloat162float(yb[size_t(t) * H + j])) * r2[size_t(k) * H + j];
        }
        worst = std::max(worst, std::fabs(a)); worst_b = std::max(worst_b, std::fabs(b));
    }
    char why[128]; std::snprintf(why, sizeof(why), "residual dot fp32 %.3g, bf16 %.3g", worst, worst_b);
    expect(worst < 1e-3 && worst_b < 0.05, why);
}

// a zero direction leaves the rows bit for bit as they were (x - 0 * 0 = x)
static void zero_direction_is_identity() {
    const int n = 7; std::vector<float> x(size_t(n) * H); for (auto & v : x) v = 3.f * nd(rng);
    Dev<float> dz(std::vector<float>(size_t(H), 0.f)), dx(x);
    tqllm_cuda::fn_ablate_rows(dx.p, dz.p, 1.f, H, n, nullptr); check(cudaDeviceSynchronize());
    const auto y = dx.get();
    expect(std::memcmp(x.data(), y.data(), x.size() * 4) == 0, "rows changed");
}

// layer-major prefill: fn_ablate_moe_coef + fn_moe_hc_combine with the direction match fn_moe_combine, fn_ablate_rows
// and hc_combine; without a direction fn_moe_hc_combine equals the pair bit for bit (the off path is unchanged)
static void fused_combine_matches_pair() {
    const int c = 4, k = 10; const auto r = unit(H); Dev<float> dr(r);
    for (int n : {1, 37, 300}) {
        std::vector<__nv_bfloat16> routed(size_t(n) * k * H); std::vector<float> wts(size_t(n) * k), sh(size_t(n) * H), sw(static_cast<size_t>(n)), inject(size_t(n) * c), wide(size_t(n) * c * H);
        for (auto & v : routed) v = __float2bfloat16_rn(nd(rng));
        for (auto & v : wts) v = std::fabs(nd(rng)) * 0.2f;
        for (auto & v : sh) v = nd(rng);
        for (auto & v : sw) v = std::fabs(nd(rng));
        for (auto & v : inject) v = 2.f * nd(rng);
        for (auto & v : wide) v = 3.f * nd(rng);
        Dev<__nv_bfloat16> dro(routed); Dev<float> dw(wts), dsh(sh), dsw(sw), dinj(inject), wa(wide), wb(wide), wc(wide);
        Dev<float> fo(size_t(n) * H), out(wide.size()), coef(static_cast<size_t>(n));
        tqllm_cuda::fn_moe_combine(dro.p, dw.p, dsh.p, dsw.p, fo.p, H, k, n, nullptr);
        tqllm_cuda::hc_combine(wc.p, fo.p, dinj.p, out.p, H, c, n, nullptr);
        tqllm_cuda::fn_moe_hc_combine(dro.p, dw.p, dsh.p, dsw.p, dinj.p, wa.p, H, k, c, n, nullptr);
        check(cudaDeviceSynchronize());
        const auto plain = wa.get(), ref0 = out.get();
        expect(std::memcmp(plain.data(), ref0.data(), plain.size() * 4) == 0, "n " + std::to_string(n) + ": no direction differs from the pair");
        const float w = 0.8f;
        tqllm_cuda::fn_ablate_rows(fo.p, dr.p, w, H, n, nullptr);
        tqllm_cuda::hc_combine(wc.p, fo.p, dinj.p, out.p, H, c, n, nullptr);
        tqllm_cuda::fn_ablate_moe_coef(dro.p, dw.p, dsh.p, dsw.p, dr.p, w, coef.p, H, k, n, nullptr);
        tqllm_cuda::fn_moe_hc_combine(dro.p, dw.p, dsh.p, dsw.p, dinj.p, wb.p, H, k, c, n, nullptr, dr.p, coef.p);
        check(cudaDeviceSynchronize());
        const auto A = wb.get(), B = out.get();
        double err = 0; for (size_t i = 0; i < A.size(); ++i) err = std::max(err, double(std::fabs(A[i] - B[i])));
        expect(err < 1e-4, "n " + std::to_string(n) + " with a direction: max err " + std::to_string(err));
    }
}

// TQAB writer for the parser cases: version, hidden, entries, k (version 3), per entry the weights then the directions
struct TmpFile {
    std::string path;
    TmpFile() { char t[] = "/tmp/tqab_testXXXXXX"; const int fd = mkstemp(t); expect(fd >= 0, "mkstemp"); close(fd); path = t; }
    ~TmpFile() { std::remove(path.c_str()); }
};
static void write_tqab(const std::string & path, int version, int hidden, int entries, int k, float base, bool short_body = false, bool trailing = false) {
    FILE * f = std::fopen(path.c_str(), "wb"); expect(f != nullptr, "open " + path);
    std::fwrite("TQAB", 1, 4, f); const int32_t hdr[3] = {version, hidden, entries}; std::fwrite(hdr, 4, 3, f);
    if (version == 3) { const int32_t kk = k; std::fwrite(&kk, 4, 1, f); }
    for (int e = 0; e < entries; ++e) {
        const float w[3] = {base + e, base + e + 0.25f, base + e + 0.5f};
        std::fwrite(w, 4, version >= 2 ? 3 : 2, f);
        for (int j = 0; j < hidden; ++j) { const float v = float(e * 1000 + j); std::fwrite(&v, 4, 1, f); }
        if (version == 3) for (int s = 0; s < k; ++s) for (int j = 0; j < hidden; ++j) { const float v = -float(e * 1000 + s * 100 + j); std::fwrite(&v, 4, 1, f); }
        if (short_body && e == entries - 1) break;
    }
    if (short_body) { std::fclose(f); const long keep = 16 + (version == 3 ? 4 : 0) + 8; expect(::truncate(path.c_str(), keep) == 0, "truncate"); return; }
    if (trailing) std::fputc(0, f);
    std::fclose(f);
}
static bool throws(const std::string & path, int hidden, int entries) {
    try { tqllm::flashnext::parse_tqab(path, hidden, entries); } catch (const std::runtime_error &) { return true; }
    return false;
}

// versions 1, 2 and 3 parse to the same layout: weights per entry (w_stream 0 in version 1), the block directions, the
// version 3 stream directions in [entry][4][hidden]; entries after the file's last one have weight 0
static void tqab_versions() {
    const int hd = 8, entries = 5, stored = 3;
    for (int version : {1, 2, 3}) {
        TmpFile t; const int k = version == 3 ? 2 : 0;
        write_tqab(t.path, version, hd, stored, k, 1.f);
        const auto f = tqllm::flashnext::parse_tqab(t.path, hd, entries);
        const std::string v = "v" + std::to_string(version) + ": ";
        expect(f.version == version && f.k == k && f.stored == stored, v + "header");
        expect(f.wa.size() == size_t(entries) && f.dir.size() == size_t(entries) * hd, v + "sizes");
        for (int e = 0; e < entries; ++e) {
            const bool in = e < stored;
            expect(f.wa[size_t(e)] == (in ? 1.f + e : 0.f) && f.wf[size_t(e)] == (in ? 1.25f + e : 0.f), v + "attention / FFN weights");
            expect(f.ws[size_t(e)] == (in && version >= 2 ? 1.5f + e : 0.f), v + "stream weight");
            for (int j = 0; j < hd; ++j) expect(f.dir[size_t(e) * hd + size_t(j)] == (in ? float(e * 1000 + j) : 0.f), v + "direction");
            if (version == 3) for (int s = 0; s < 4; ++s) for (int j = 0; j < hd; ++j)
                expect(f.sdir[(size_t(e) * 4 + size_t(s)) * hd + size_t(j)] == (in && s < k ? -float(e * 1000 + s * 100 + j) : 0.f), v + "stream directions");
        }
    }
}

// bad files are refused: other magic or version, another hidden size, more entries than the model's blocks, a bad
// direction count, a short file, trailing bytes, no file
static void tqab_rejects_bad_files() {
    const int hd = 8, entries = 5;
    { TmpFile t; write_tqab(t.path, 4, hd, 2, 0, 1.f); expect(throws(t.path, hd, entries), "version 4 accepted"); }
    { TmpFile t; write_tqab(t.path, 2, hd, 2, 0, 1.f); expect(throws(t.path, hd + 1, entries), "other hidden size accepted"); }
    { TmpFile t; write_tqab(t.path, 2, hd, 6, 0, 1.f); expect(throws(t.path, hd, entries), "too many entries accepted"); }
    { TmpFile t; write_tqab(t.path, 3, hd, 2, 5, 1.f); expect(throws(t.path, hd, entries), "k 5 accepted"); }
    { TmpFile t; write_tqab(t.path, 3, hd, 2, 0, 1.f); expect(throws(t.path, hd, entries), "k 0 accepted"); }
    { TmpFile t; write_tqab(t.path, 2, hd, 2, 0, 1.f, true); expect(throws(t.path, hd, entries), "short file accepted"); }
    { TmpFile t; write_tqab(t.path, 1, hd, 2, 0, 1.f, false, true); expect(throws(t.path, hd, entries), "trailing bytes accepted"); }
    { TmpFile t; FILE * f = std::fopen(t.path.c_str(), "wb"); std::fwrite("TQAX\1\0\0\0", 1, 8, f); std::fclose(f);
      expect(throws(t.path, hd, entries), "other magic accepted"); }
    expect(throws("/nonexistent/x.tqab", hd, entries), "missing file accepted");
}

int main() {
    const struct { const char * name; void (*run)(); } cases[] = {
        {"rows_match_projection", rows_match_projection},
        {"two_directions_fp32_bf16", two_directions_fp32_bf16},
        {"zero_direction_is_identity", zero_direction_is_identity},
        {"fused_combine_matches_pair", fused_combine_matches_pair},
        {"tqab_versions", tqab_versions},
        {"tqab_rejects_bad_files", tqab_rejects_bad_files},
    };
    int failed = 0;
    for (const auto & c : cases) {
        try { c.run(); std::printf("PASS %s\n", c.name); }
        catch (const std::exception & e) { ++failed; std::printf("FAIL %s: %s\n", c.name, e.what()); cudaGetLastError(); }
    }
    std::printf("%d of %zu cases failed\n", failed, sizeof(cases) / sizeof(cases[0]));
    return failed ? 1 : 0;
}
