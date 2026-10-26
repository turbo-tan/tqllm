// GLM-5.3 DSA k-pool indexer + sparse MLA against a CPU reference of the same rules (glm5next_dsa.cu): 2400 cached
// positions (600 pools of 4, so the top 512 really drop some), queries at the end of the cache and in the middle,
// a query that completes a pool (empty tail) and ones with a 1..3-token tail. Checks the pool keys, the selected cells
// (as sets) and the attention output.
#include "glm5next_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <class T> T * up(const std::vector<T> & v) { T * d; ck(cudaMalloc(&d, v.size() * sizeof(T))); ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> std::vector<T> down(const T * d, size_t n) { std::vector<T> v(n); ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
float h16(float x) { return __half2float(__float2half(x)); }
}  // namespace

int main() try {
    const int D = 64, r = 4, NH = 32, ctx = 2400, pools = 512, R = 64, H = 2, stride = pools * r + r - 1;
    const float eps = 1e-6f;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n) { std::vector<float> v(n); for (auto & x : v) x = nd(rng); return v; };
    const auto k = rnd(size_t(ctx) * D), g = rnd(size_t(ctx) * D), lnw = rnd(D), lnb = rnd(D), ape = rnd(size_t(r) * D);
    auto lat = rnd(size_t(ctx) * R);
    for (auto & x : lat) x = h16(x);  // the engine's FP16 latent cache
    std::vector<__half> lath(lat.size());
    for (size_t i = 0; i < lat.size(); ++i) lath[i] = __float2half(lat[i]);
    // the cache filled in three calls (as prefill chunks would), d_pos-free
    float * dk = up(k), * dg = up(g), * dlnw = up(lnw), * dlnb = up(lnb), * dape = up(ape), * dpool;
    __half * dkg;
    const int ring = 1000 + 2 * r;  // the largest call (1000 tokens) plus a pool, as the engine sizes it
    ck(cudaMalloc(&dkg, size_t(ring) * 2 * D * sizeof(__half)));
    ck(cudaMalloc(&dpool, size_t(ctx / r) * D * sizeof(float)));
    for (int p0 : {0, 1000, 1801}) {
        const int n = (p0 == 0 ? 1000 : p0 == 1000 ? 801 : ctx - 1801);
        tqllm_cuda::glm_dsa_store(dk + size_t(p0) * D, dg + size_t(p0) * D, dlnw, dlnb, dkg, dape, dpool, D, r, eps, n, nullptr, p0, ring, nullptr);
    }
    ck(cudaDeviceSynchronize());
    // CPU: LayerNorm keys rounded to FP16 as stored, gates FP16, pools
    std::vector<float> kn(size_t(ctx) * D), gh(size_t(ctx) * D), pool(size_t(ctx / r) * D);
    for (int t = 0; t < ctx; ++t) {
        double m = 0, v = 0;
        for (int j = 0; j < D; ++j) m += k[size_t(t) * D + j];
        m /= D;
        for (int j = 0; j < D; ++j) v += (k[size_t(t) * D + j] - m) * (k[size_t(t) * D + j] - m);
        v /= D;
        for (int j = 0; j < D; ++j) {
            kn[size_t(t) * D + j] = h16(float((k[size_t(t) * D + j] - m) / std::sqrt(v + eps) * lnw[j] + lnb[j]));
            gh[size_t(t) * D + j] = h16(g[size_t(t) * D + j]);
        }
    }
    for (int b = 0; b < ctx / r; ++b)
        for (int c = 0; c < D; ++c) {
            double lg[4], m = -1e30, z = 0, acc = 0;
            for (int i = 0; i < r; ++i) { lg[i] = gh[size_t(b * r + i) * D + c] + ape[size_t(i) * D + c]; m = std::max(m, lg[i]); }
            for (int i = 0; i < r; ++i) { const double e = std::exp(lg[i] - m); z += e; acc += e * kn[size_t(b * r + i) * D + c]; }
            pool[size_t(b) * D + c] = float(acc / z);
        }
    const auto gpool = down(dpool, pool.size());
    double perr = 0;
    for (size_t i = 0; i < pool.size(); ++i) perr = std::max(perr, double(std::fabs(gpool[i] - pool[i])));
    std::printf("pool keys: max |gpu - cpu| %.2e\n", perr);
    bool ok = perr < 2e-3;

    // queries: a group at the end (q = 2396..2399: tails 1, 2, 3, then a completed pool) and one at 1000 (n_pick 250 < 512)
    int failures = 0;
    for (int pos0 : {2396, 1000}) {
        const int nt = pos0 == 2396 ? 4 : 1;
        const auto q = rnd(size_t(nt) * NH * D), w = rnd(size_t(nt) * NH), qabs = rnd(size_t(nt) * H * R);
        float * dq = up(q), * dw = up(w), * dqa = up(qabs), * dkv;
        int * dcells, * dcount;
        ck(cudaMalloc(&dcells, size_t(nt) * stride * sizeof(int)));
        ck(cudaMalloc(&dcount, size_t(nt) * sizeof(int)));
        ck(cudaMalloc(&dkv, size_t(nt) * H * R * sizeof(float)));
        const size_t sb = tqllm_cuda::glm_dsa_select_bytes(nt, (pos0 + nt) / r);
        void * scratch; ck(cudaMalloc(&scratch, sb));
        tqllm_cuda::glm_dsa_select(dq, dw, dpool, D, r, pools, nt, pos0, scratch, sb, dcells, dcount, stride, nullptr);
        __half * dlat = up(lath);
        const float scale = 1.f / std::sqrt(256.f);
        tqllm_cuda::glm_mla_attn_sparse(dqa, dlat, dkv, H, R, dcells, dcount, stride, scale, nt, nullptr);
        ck(cudaDeviceSynchronize());
        const auto cells = down(dcells, size_t(nt) * stride), count = down(dcount, size_t(nt));
        const auto kv = down(dkv, size_t(nt) * H * R);
        for (int t = 0; t < nt; ++t) {
            const int qp = pos0 + t, pick = (qp + 1) / r;
            std::vector<std::pair<double, int>> sc;
            for (int b = 0; b < pick; ++b) {
                double s = 0;
                for (int h = 0; h < NH; ++h) {
                    double dot = 0;
                    for (int d = 0; d < D; ++d) dot += double(q[(size_t(t) * NH + h) * D + d]) * pool[size_t(b) * D + d];
                    s += w[size_t(t) * NH + h] / std::sqrt(double(D) * NH) * std::max(dot, 0.0);
                }
                sc.push_back({s, b});
            }
            std::stable_sort(sc.begin(), sc.end(), [](auto a, auto b) { return a.first > b.first; });
            std::set<int> want;
            for (int i = 0; i < std::min(pick, pools); ++i) for (int m = 0; m < r; ++m) want.insert(sc[size_t(i)].second * r + m);
            for (int p = pick * r; p <= qp; ++p) want.insert(p);
            const std::set<int> got(cells.begin() + size_t(t) * stride, cells.begin() + size_t(t) * stride + count[size_t(t)]);
            const bool same = got == want && size_t(count[size_t(t)]) == want.size();
            // attention over the CPU's cell set
            double aerr = 0;
            for (int h = 0; h < H; ++h) {
                std::vector<double> p;
                double m = -1e300, z = 0;
                for (int c : want) {
                    double dot = 0;
                    for (int j = 0; j < R; ++j) dot += double(qabs[(size_t(t) * H + h) * R + j]) * lat[size_t(c) * R + j];
                    p.push_back(dot * scale); m = std::max(m, dot * scale);
                }
                for (auto & x : p) z += (x = std::exp(x - m));
                for (int j = 0; j < R; ++j) {
                    double acc = 0; size_t i = 0;
                    for (int c : want) acc += p[i++] * lat[size_t(c) * R + j];
                    aerr = std::max(aerr, std::fabs(acc / z - kv[(size_t(t) * H + h) * R + j]));
                }
            }
            std::printf("query %d: pickable pools %d, cells %d (%s), attention max err %.2e\n", qp, pick, count[size_t(t)],
                        same ? "same set" : "DIFFERENT set", aerr);
            if (!same || aerr > 1e-4) ++failures;
        }
        cudaFree(dq); cudaFree(dw); cudaFree(dqa); cudaFree(dkv); cudaFree(dcells); cudaFree(dcount); cudaFree(scratch); cudaFree(dlat);
    }
    ok = ok && failures == 0;
    std::puts(ok ? "glm_dsa: PASS" : "glm_dsa: FAIL");
    return ok ? 0 : 1;
} catch (const std::exception & e) { std::fprintf(stderr, "glm_dsa: %s\n", e.what()); return 1; }
