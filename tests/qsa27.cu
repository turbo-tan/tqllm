// QSA for the dense 27B verify path (qwen35_qsa.cuh, attention_mq256_qsa):
// - the selection equals a CPU top-k of the same scores (ascending blocks, ties to the lower block, per-row bounds);
// - a row of a verify block equals its one-token step bit for bit (token key, selection, attention output), for sparse
//   blocks and for a block that crosses `from`;
// - with every block below the split point picked, the sparse rows equal dense two-part attention within BF16 rounding.
#include "qwen2_kernels.cuh"
#include "qwen35_qsa.cuh"
#include "qwen35_kernels.cuh"
#include "qwen35_qsa_plan.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <tuple>
#include <stdexcept>
#include <string>
#include <vector>

using bf16 = __nv_bfloat16;
using namespace tqllm_cuda;
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
template <class T> T * dev(size_t n) { T * p = nullptr; check(cudaMalloc(&p, n * sizeof(T))); check(cudaMemset(p, 0, n * sizeof(T))); return p; }
template <class T> T * upload(const std::vector<T> & v) { T * p = dev<T>(v.size()); check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
template <class T> std::vector<T> download(const T * p, size_t n) { std::vector<T> v(n); check(cudaDeviceSynchronize()); check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }

static int failures = 0;
static void expect(bool ok, const char * what) { std::printf("%s: %s\n", what, ok ? "ok" : "FAIL"); failures += !ok; }

// the score kernel's arithmetic: per lane an fma chain over its 4 dims, the xor-shuffle reduction, heads added in order
static uint32_t score_bits(const float * key, const float * q) {
    float total = 0.f;
    for (int h = 0; h < kQsa27Heads; ++h) {
        float v[32];
        for (int l = 0; l < 32; ++l) {
            float d = 0.f;
            for (int j = 0; j < 4; ++j) d = std::fmaf(key[l + 32 * j], q[h * kQsa27Dim + l + 32 * j], d);
            v[l] = d;
        }
        for (int o = 16; o; o >>= 1) {
            float w[32];
            for (int l = 0; l < 32; ++l) w[l] = v[l] + v[l ^ o];
            std::memcpy(v, w, sizeof(v));
        }
        total += std::max(v[0], 0.f);
    }
    if (total == 0.f) total = 0.f;
    uint32_t u;
    std::memcpy(&u, &total, 4);
    return u;
}

static void test_select(std::mt19937 & rng) {
    constexpr int from = 2048, maxb = 6000;
    std::normal_distribution<float> nd;
    for (int variant = 0; variant < 3; ++variant) {
        std::vector<float> blocks(size_t(maxb) * kQsa27Dim);
        for (auto & x : blocks) x = nd(rng);
        if (variant == 1)  // every block the same: all ties, the lowest blocks win
            for (int b = 1; b < maxb; ++b) std::copy_n(blocks.begin(), kQsa27Dim, blocks.begin() + size_t(b) * kQsa27Dim);
        if (variant == 2)  // most scores zero
            for (int b = 0; b < maxb; ++b) if (b % 3) std::fill_n(blocks.begin() + size_t(b) * kQsa27Dim, kQsa27Dim, 0.f);
        const std::vector<int> depth = {0, 1, 1, 2, 3, 3, 4, 40, 5, 6, 7, 8, 8, 9, 10, 11};
        for (const auto & [pos, n, tree] : {std::tuple<int, int, bool>{20000, 16, false}, {2070, 12, true}, {9001, 1, false}, {2050, 16, true}}) {
            std::vector<float> q(size_t(n) * kQsa27Heads * kQsa27Dim);
            for (auto & x : q) x = nd(rng);
            auto * dq = upload(q); auto * db = upload(blocks);
            auto * bits = dev<uint32_t>(size_t(n) * maxb); auto * sel = dev<int32_t>(size_t(n) * kQsa27Pick);
            auto * dpos = upload(std::vector<int>{pos, pos + 1}); auto * ddep = upload(depth);
            check(cudaMemset(sel, 0x5a, size_t(n) * kQsa27Pick * 4));
            qsa27_select(dq, db, bits, sel, dpos, tree ? ddep : nullptr, n, from, maxb, nullptr);
            const auto got = download(sel, size_t(n) * kQsa27Pick);
            bool same = true;
            for (int r = 0; r < n; ++r) {
                const int split = qsa27_split_point(pos + (tree ? depth[size_t(r)] : r)), nb = split >= from ? split / 4 : 0;
                std::vector<std::pair<uint32_t, int>> s;
                for (int b = 0; b < nb; ++b) s.push_back({score_bits(&blocks[size_t(b) * kQsa27Dim], &q[size_t(r) * kQsa27Heads * kQsa27Dim]), b});
                std::sort(s.begin(), s.end(), [](const auto & a, const auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
                std::vector<int> want;
                for (int i = 0; i < std::min(nb, kQsa27Pick); ++i) want.push_back(s[size_t(i)].second);
                std::sort(want.begin(), want.end());
                want.resize(kQsa27Pick, -1);
                same &= std::equal(want.begin(), want.end(), got.begin() + size_t(r) * kQsa27Pick);
            }
            char name[96];
            std::snprintf(name, sizeof(name), "select == CPU top-k (variant %d, pos %d, %d rows%s)", variant, pos, n, tree ? ", depths" : "");
            expect(same, name);
            cudaFree(dq); cudaFree(db); cudaFree(bits); cudaFree(sel); cudaFree(dpos); cudaFree(ddep);
        }
    }
}

struct Fixture {
    static constexpr int H = 512, heads = 24, kvh = 4, D = 256, cap = 8192, nsplit = 20, NV = 16;
    __half * w, * x, * tok;
    float * qn, * kn, * bn, * blocks, * proj, * iq, * part;
    bf16 * q, * kc, * vc, * y;
    uint32_t * bits; int32_t * sel; int * dpos, * par, * dep;
    float scale = 1.f / 16.f;
    explicit Fixture(std::mt19937 & rng) {
        std::normal_distribution<float> nd;
        auto halves = [&](size_t n, float s) { std::vector<__half> v(n); for (auto & e : v) e = __float2half(nd(rng) * s); return v; };
        auto floats = [&](size_t n) { std::vector<float> v(n); for (auto & e : v) e = 1.f + 0.1f * nd(rng); return v; };
        auto bfs = [&](size_t n) { std::vector<bf16> v(n); for (auto & e : v) e = __float2bfloat16_rn(nd(rng)); return v; };
        w = upload(halves(size_t(kQsa27Rows) * H, 0.05f));
        x = upload(halves(size_t(cap) * H, 1.f));    // the indexer input of the row at each position
        tok = upload(halves(size_t(cap) * kQsa27Dim, 1.f));  // committed token keys
        qn = upload(floats(kQsa27Dim)); kn = upload(floats(kQsa27Dim)); bn = upload(floats(kQsa27Dim));
        blocks = dev<float>(size_t(cap / 4) * kQsa27Dim);
        proj = dev<float>(size_t(NV) * kQsa27Rows); iq = dev<float>(size_t(NV) * kQsa27Heads * kQsa27Dim);
        part = dev<float>(size_t(attention_mq_partial_floats(heads, NV, nsplit + 1)));
        q = upload(bfs(size_t(cap) * heads * D));     // the attention query of the row at each position
        kc = upload(bfs(size_t(cap) * kvh * D)); vc = upload(bfs(size_t(cap) * kvh * D));
        y = dev<bf16>(size_t(NV) * heads * D);
        bits = dev<uint32_t>(size_t(NV) * (cap / 4)); sel = dev<int32_t>(size_t(NV) * kQsa27Pick);
        dpos = dev<int>(2);
        std::vector<int> chain(NV), depth(NV);
        for (int i = 0; i < NV; ++i) chain[size_t(i)] = i - 1, depth[size_t(i)] = i;
        par = upload(chain); dep = upload(depth);
    }
    struct Out { std::vector<bf16> y; std::vector<int32_t> sel; std::vector<__half> keys; int mode; };
    // one verify block of n rows at pos, as the engine runs it (qsa_plan, qsa_index, attn_core); dense: QSA off
    Out run(int pos, int n, int from, bool dense = false) {
        tqllm::QsaPlan plan;
        plan.keyed = pos;
        const auto step = dense ? tqllm::QsaPlan::Step{} : plan.write(pos, n, true, true, from);
        qsa27_pool_blocks(tok, bn, blocks, step.pool_first, step.pool_count, 1e7f, 1e-6f, nullptr);
        const int p2[2] = {pos, pos + 1};
        check(cudaMemcpy(dpos, p2, sizeof(p2), cudaMemcpyHostToDevice));
        const __half * xr = x + size_t(pos) * H;
        const bf16 * qr = q + size_t(pos) * heads * D;
        Out o;
        o.mode = step.mode;
        if (!step.mode) {
            qsa27_project(xr, w, proj, kQsa27Heads * kQsa27Dim, kQsa27Dim, n, H, nullptr);
            qsa27_token_keys(proj, kQsa27Dim, 0, kn, tok, dpos, n, 1e-6f, nullptr);
            attention_mq256(qr, kc, vc, y, part, dpos + 1, n, heads, kvh, scale, nsplit, nullptr, 0, par, dep);
        } else {
            qsa27_project(xr, w, proj, 0, kQsa27Rows, n, H, nullptr);
            qsa27_token_keys(proj, kQsa27Rows, kQsa27Heads * kQsa27Dim, kn, tok, dpos, n, 1e-6f, nullptr);
            qsa27_queries(proj, kQsa27Rows, qn, iq, dpos, dep, n, 1e7f, 1e-6f, nullptr);
            qsa27_select(iq, blocks, bits, sel, dpos, dep, n, from, cap / 4, nullptr);
            attention_mq256_qsa(qr, kc, vc, y, part, dpos + 1, n, heads, kvh, scale, nsplit, nullptr, par, dep, sel, from, step.mode == 2);
            o.sel = download(sel, size_t(n) * kQsa27Pick);
        }
        o.y = download(y, size_t(n) * heads * D);
        o.keys = download(tok + size_t(pos) * kQsa27Dim, size_t(n) * kQsa27Dim);
        return o;
    }
};

static void test_row_equals_step(Fixture & f) {
    constexpr int from = 4096;
    for (const auto & [pos, n, mode] : {std::tuple<int, int, int>{6000, 12, 1}, {4105, 12, 2}, {7003, 16, 1}, {4090, 8, 0}, {4111, 1, 1}}) {
        const auto block = f.run(pos, n, from);
        bool same = block.mode == mode;
        int sparse_rows = 0;
        for (int z = 0; z < n; ++z) {
            const auto one = f.run(pos + z, 1, from);
            const size_t yr = size_t(Fixture::heads) * Fixture::D;
            same &= std::memcmp(one.y.data(), block.y.data() + size_t(z) * yr, yr * sizeof(bf16)) == 0;
            same &= std::memcmp(one.keys.data(), block.keys.data() + size_t(z) * kQsa27Dim, kQsa27Dim * sizeof(__half)) == 0;
            const bool sparse = qsa27_split_point(pos + z) >= from;
            same &= sparse == (one.mode == 1);
            if (sparse) same &= std::equal(one.sel.begin(), one.sel.end(), block.sel.begin() + size_t(z) * kQsa27Pick);
            sparse_rows += sparse;
        }
        char name[96];
        std::snprintf(name, sizeof(name), "row == one-token step, bits (pos %d, %d rows, %d sparse)", pos, n, sparse_rows);
        expect(same, name);
    }
}

static void test_pick_all(Fixture & f) {
    // split point 2048 for rows at 2063 .. 2094: 512 blocks, all picked
    const auto sparse = f.run(2063, 8, 2048), dense = f.run(2063, 8, 2048, true);
    double worst = 0, scale = 0;
    for (size_t i = 0; i < dense.y.size(); ++i) {
        const double a = __bfloat162float(sparse.y[i]), b = __bfloat162float(dense.y[i]);
        worst = std::max(worst, std::fabs(a - b)), scale = std::max(scale, std::fabs(b));
    }
    bool all = sparse.mode == 1;
    for (int r = 0; r < 8; ++r)
        for (int i = 0; i < kQsa27Pick; ++i) all &= sparse.sel[size_t(r) * kQsa27Pick + i] == i;
    std::printf("every block picked: max |sparse - dense| %.5f of max |dense| %.3f\n", worst, scale);
    expect(all && worst <= 0.02 * scale, "sparse with every block picked == dense attention (BF16 rounding)");
    // and a real selection differs from dense (the test would pass trivially if the sparse path fell back)
    const auto s2 = f.run(7003, 4, 4096), d2 = f.run(7003, 4, 4096, true);
    expect(std::memcmp(s2.y.data(), d2.y.data(), s2.y.size() * sizeof(bf16)) != 0, "a sparse row differs from dense attention");
}


// The compact KV formats (attention_kv_split): the same two properties per codec pair, on caches written by the
// formats' own writers.
// A staged host KV tail as the engine has it: the first `evicted` bytes of the cache copied to pinned host memory, and a
// device copy that is stale (all bytes 0xFF) below the last 64 KiB "granule" before `evicted`
struct StagedTail {
    void * host = nullptr; uint8_t * stale = nullptr; int64_t evicted = 0;
    void release() const { cudaFreeHost(host); cudaFree(stale); }
};
static StagedTail staged_tail(const void * cache, size_t bytes, int64_t evicted) {
    constexpr int64_t G = 65536;
    StagedTail t; t.evicted = evicted;
    check(cudaHostAlloc(&t.host, size_t(evicted), cudaHostAllocPortable));
    check(cudaMemcpy(t.host, cache, size_t(evicted), cudaMemcpyDeviceToHost));
    check(cudaMalloc(&t.stale, bytes));
    check(cudaMemcpy(t.stale, cache, bytes, cudaMemcpyDeviceToDevice));
    check(cudaMemset(t.stale, 0xFF, size_t(evicted - G)));
    return t;
}

struct CodecCase { const char * name; KvCodec k, v; size_t krow, vrow; };
static void test_codecs(std::mt19937 & rng) {
    constexpr int H = 512, heads = 24, kvh = 4, D = 256, cap = 8192, nsplit = 20, NV = 16, from = 4096;
    const CodecCase cases[] = {{"tq3/tq3", KvCodec::Tq3, KvCodec::Tq3, sizeof(Tq3KV256), sizeof(Tq3KV256)},
                               {"q8/q8", KvCodec::Q8, KvCodec::Q8, sizeof(Q8KV256), sizeof(Q8KV256)},
                               {"q4/q4", KvCodec::Q4, KvCodec::Q4, sizeof(Q4KV256), sizeof(Q4KV256)},
                               {"tcq3/tcq2", KvCodec::Tcq3, KvCodec::Tcq2, sizeof(Tcq3K256), sizeof(Tcq2V256)},
                               {"tq3/tq2s", KvCodec::Tq3, KvCodec::Tq2s, sizeof(Tq3KV256), sizeof(Tq2V256)}};
    std::normal_distribution<float> nd;
    auto halves = [&](size_t n, float s) { std::vector<__half> v(n); for (auto & e : v) e = __float2half(nd(rng) * s); return v; };
    auto bfs = [&](size_t n) { std::vector<bf16> v(n); for (auto & e : v) e = __float2bfloat16_rn(nd(rng)); return v; };
    auto * w = upload(halves(size_t(kQsa27Rows) * H, 0.05f)); auto * x = upload(halves(size_t(cap) * H, 1.f));
    auto * tok = upload(halves(size_t(cap) * kQsa27Dim, 1.f));
    auto * nw = upload(std::vector<float>(kQsa27Dim, 1.f));
    auto * blocks = dev<float>(size_t(cap / 4) * kQsa27Dim);
    auto * proj = dev<float>(size_t(NV) * kQsa27Rows); auto * iq = dev<float>(size_t(NV) * kQsa27Heads * kQsa27Dim);
    auto * part = dev<float>(size_t(attention_mq_partial_floats(heads, NV, nsplit + 1)));
    auto * q = upload(bfs(size_t(cap) * heads * D));
    auto * raw = upload(bfs(size_t(cap) * 2 * kvh * D));  // per position: K rows, then V rows (the writers' input layout)
    auto * y = dev<bf16>(size_t(NV) * heads * D);
    auto * bits = dev<uint32_t>(size_t(NV) * (cap / 4)); auto * sel = dev<int32_t>(size_t(NV) * kQsa27Pick); auto * dpos = dev<int>(2);
    std::vector<int> chain(NV), depth(NV);
    for (int i = 0; i < NV; ++i) chain[size_t(i)] = i - 1, depth[size_t(i)] = i;
    auto * par = upload(chain); auto * dep = upload(depth);
    qsa27_pool_blocks(tok, nw, blocks, 0, cap / 4, 1e7f, 1e-6f, nullptr);
    for (const auto & c : cases) {
        auto * kc = dev<uint8_t>(size_t(cap) * kvh * c.krow); auto * vc = dev<uint8_t>(size_t(cap) * kvh * c.vrow);
        const int ts = 2 * kvh * D;
        for (int p = 0; p < cap; p += 16) {
            const int p2[2] = {p, p + 1};
            check(cudaMemcpy(dpos, p2, sizeof(p2), cudaMemcpyHostToDevice));
            cache_write_kv_side(c.k, raw + size_t(p) * ts, ts, 16, kc, dpos, kvh, D, nullptr);
            cache_write_kv_side(c.v, raw + size_t(p) * ts + kvh * D, ts, 16, vc, dpos, kvh, D, nullptr);
            check(cudaDeviceSynchronize());
        }
        // mode: 0 dense, 1 sparse, 2 crossing; all: every block below the split point picked (ascending)
        auto run = [&](int pos, int n, int mode, int from_, bool all = false) {
            const int p2[2] = {pos, pos + 1};
            check(cudaMemcpy(dpos, p2, sizeof(p2), cudaMemcpyHostToDevice));
            const bf16 * qr = q + size_t(pos) * heads * D;
            if (mode) {
                qsa27_project(x + size_t(pos) * H, w, proj, 0, kQsa27Rows, n, H, nullptr);
                qsa27_queries(proj, kQsa27Rows, nw, iq, dpos, dep, n, 1e7f, 1e-6f, nullptr);
                qsa27_select(iq, blocks, bits, sel, dpos, dep, n, from_, cap / 4, nullptr);
                if (all) {
                    std::vector<int32_t> every(size_t(n) * kQsa27Pick);
                    for (size_t i = 0; i < every.size(); ++i) every[i] = int32_t(i % kQsa27Pick);
                    check(cudaDeviceSynchronize());
                    check(cudaMemcpy(sel, every.data(), every.size() * 4, cudaMemcpyHostToDevice));
                }
            }
            attention_kv_split(c.k, c.v, qr, kc, vc, y, part, dpos + 1, n, heads, kvh, 1.f / 16, nsplit, nullptr, par, dep, nullptr,
                               mode ? sel : nullptr, from_, mode == 2);
            return download(y, size_t(n) * heads * D);
        };
        bool same = true;
        for (const auto & [pos, n] : {std::pair<int, int>{6000, 12}, {4105, 12}, {7003, 16}}) {
            tqllm::QsaPlan plan; plan.keyed = pos;
            const auto block = run(pos, n, plan.write(pos, n, true, true, from).mode, from);
            for (int z = 0; z < n; ++z) {
                tqllm::QsaPlan one; one.keyed = pos + z;
                const auto row = run(pos + z, 1, one.write(pos + z, 1, true, true, from).mode, from);
                same &= std::memcmp(row.data(), block.data() + size_t(z) * heads * D, row.size() * sizeof(bf16)) == 0;
            }
        }
        expect(same, (std::string(c.name) + ": row == one-token step, bits (sparse and crossing blocks)").c_str());
        const auto sparse = run(2063, 8, 1, 2048, true), dense = run(2063, 8, 0, 2048);
        expect(std::memcmp(sparse.data(), dense.data(), dense.size() * sizeof(bf16)) == 0,
               (std::string(c.name) + ": every block picked == dense attention, bits").c_str());
        const auto s2 = run(7003, 4, 1, from), d2 = run(7003, 4, 0, from);
        expect(std::memcmp(s2.data(), d2.data(), s2.size() * sizeof(bf16)) != 0, (std::string(c.name) + ": a sparse row differs from dense").c_str());
        {  // staged host KV tail: the evicted bytes in pinned host memory, the staged cache stale except the last granule;
           // qsa27_stage_selected copies the picked host rows back, then the sparse rows read the cache as usual
            const size_t kr = size_t(kvh) * c.krow, vr = size_t(kvh) * c.vrow;
            const auto kb = staged_tail(kc, cap * kr, int64_t(6000) * kr + 100), vb = staged_tail(vc, cap * vr, int64_t(6000) * vr + 52);
            auto * nbytes = upload(std::vector<int64_t>{kb.evicted, vb.evicted});
            auto * mark = upload(std::vector<int>(cap / 4 + 1, -1));
            check(cudaMemset(mark + cap / 4, 0, sizeof(int)));
            auto sparse_rows = [&](const void * K, const void * V, bool stage) {
                const int p2[2] = {7003, 7004};
                check(cudaMemcpy(dpos, p2, sizeof(p2), cudaMemcpyHostToDevice));
                qsa27_project(x + size_t(7003) * H, w, proj, 0, kQsa27Rows, 8, H, nullptr);
                qsa27_queries(proj, kQsa27Rows, nw, iq, dpos, dep, 8, 1e7f, 1e-6f, nullptr);
                qsa27_select(iq, blocks, bits, sel, dpos, dep, 8, from, cap / 4, nullptr);
                if (stage) qsa27_stage_selected(sel, 8, dpos, dep, from, kb.host, vb.host, const_cast<void *>(K), const_cast<void *>(V), nbytes,
                                                int(kr), int(vr), mark, mark + cap / 4, nullptr);
                attention_kv_split(c.k, c.v, q + size_t(7003) * heads * D, K, V, y, part, dpos + 1, 8, heads, kvh, 1.f / 16, nsplit, nullptr,
                                   par, dep, nullptr, sel, from, false);
                return download(y, size_t(8) * heads * D);
            };
            const auto ref = sparse_rows(kc, vc, false), bad = sparse_rows(kb.stale, vb.stale, false), got = sparse_rows(kb.stale, vb.stale, true);
            expect(std::memcmp(ref.data(), got.data(), ref.size() * sizeof(bf16)) == 0,
                   (std::string(c.name) + ": picked host rows staged back == device-resident cache, bits").c_str());
            expect(std::memcmp(ref.data(), bad.data(), ref.size() * sizeof(bf16)) != 0,
                   (std::string(c.name) + ": (the stale cache alone differs: the test reads the staged rows)").c_str());
            kb.release(); vb.release(); cudaFree(nbytes); cudaFree(mark);
        }
        cudaFree(kc); cudaFree(vc);
    }
}

// Turbo4 K + V (t4kv, one-part chain attention): a row == its one-token step, and with every block picked == dense, bits
// qsa27_stage_selected copies exactly the picked positions whose whole row lies below the evicted bytes, once per call
static void test_stage_selected() {
    constexpr int cap = 4096, row = 64, n = 3, from = 2048;
    std::vector<uint8_t> src(size_t(cap) * row);
    for (size_t i = 0; i < src.size(); ++i) src[i] = uint8_t(i * 131 + 7);
    void * host = nullptr;
    check(cudaHostAlloc(&host, src.size(), cudaHostAllocPortable));
    std::memcpy(host, src.data(), src.size());
    auto * dst = dev<uint8_t>(src.size());
    check(cudaMemset(dst, 0, src.size()));
    std::vector<int32_t> picks(size_t(n) * kQsa27Pick, -1);
    for (int z = 0; z < n; ++z)
        for (int i = 0; i < 300; ++i) picks[size_t(z) * kQsa27Pick + i] = (i * 3 + z) % 700;  // overlapping picks across rows
    auto * sel = upload(picks);
    auto * dpos = upload(std::vector<int>{4000, 4001});
    const int64_t evicted = int64_t(1000) * row + 40;  // positions 0..999 whole, position 1000 straddles
    auto * nbytes = upload(std::vector<int64_t>{evicted, evicted});
    auto * mark = upload(std::vector<int>(cap / 4 + 1, -1));
    check(cudaMemset(mark + cap / 4, 0, sizeof(int)));
    qsa27_stage_selected(sel, n, dpos, nullptr, from, host, host, dst, dst, nbytes, row, row, mark, mark + cap / 4, nullptr);
    const auto got = download(dst, src.size());
    std::vector<char> want(cap, 0);
    for (int32_t b : picks) if (b >= 0) for (int p = b * 4; p < b * 4 + 4; ++p) want[size_t(p)] = int64_t(p + 1) * row <= evicted;
    bool ok = true;
    for (int p = 0; p < cap; ++p)
        for (int i = 0; i < row; ++i) ok &= got[size_t(p) * row + i] == (want[size_t(p)] ? src[size_t(p) * row + i] : 0);
    expect(ok, "stage_selected: exactly the picked whole host rows are copied (overlapping picks, a straddling row)");
    cudaFreeHost(host); cudaFree(dst); cudaFree(sel); cudaFree(dpos); cudaFree(nbytes); cudaFree(mark);
}

static void test_t4kv(std::mt19937 & rng) {
    constexpr int H = 512, heads = 24, kvh = 4, D = 256, cap = 8192, nsplit = 40, NV = 8, from = 4096;
    std::normal_distribution<float> nd;
    auto halves = [&](size_t n, float s) { std::vector<__half> v(n); for (auto & e : v) e = __float2half(nd(rng) * s); return v; };
    auto bfs = [&](size_t n) { std::vector<bf16> v(n); for (auto & e : v) e = __float2bfloat16_rn(nd(rng)); return v; };
    auto * w = upload(halves(size_t(kQsa27Rows) * H, 0.05f)); auto * x = upload(halves(size_t(cap) * H, 1.f));
    auto * tok = upload(halves(size_t(cap) * kQsa27Dim, 1.f)); auto * nw = upload(std::vector<float>(kQsa27Dim, 1.f));
    auto * blocks = dev<float>(size_t(cap / 4) * kQsa27Dim);
    auto * proj = dev<float>(size_t(NV) * kQsa27Rows); auto * iq = dev<float>(size_t(NV) * kQsa27Heads * kQsa27Dim);
    auto * part = dev<float>(size_t(attention_mq_partial_floats(heads, NV, nsplit)));
    auto * q = upload(bfs(size_t(cap) * heads * D)); auto * raw = upload(bfs(size_t(cap) * 2 * kvh * D));
    auto * y = dev<bf16>(size_t(NV) * heads * D);
    auto * bits = dev<uint32_t>(size_t(NV) * (cap / 4)); auto * sel = dev<int32_t>(size_t(NV) * kQsa27Pick); auto * dpos = dev<int>(2);
    auto * kc = dev<Turbo4K256>(size_t(cap) * kvh); auto * vc = dev<Turbo4K256>(size_t(cap) * kvh);
    qsa27_pool_blocks(tok, nw, blocks, 0, cap / 4, 1e7f, 1e-6f, nullptr);
    const int ts = 2 * kvh * D;
    for (int p = 0; p < cap; p += 16) {
        const int p2[2] = {p, p + 1};
        check(cudaMemcpy(dpos, p2, sizeof(p2), cudaMemcpyHostToDevice));
        cache_write_multi_t4kv(raw + size_t(p) * ts, raw + size_t(p) * ts + kvh * D, ts, 16, kc, vc, dpos, kvh, D, nullptr);
        check(cudaDeviceSynchronize());
    }
    auto run = [&](int pos, int n, int mode, int from_, bool all = false) {
        const int p2[2] = {pos, pos + 1};
        check(cudaMemcpy(dpos, p2, sizeof(p2), cudaMemcpyHostToDevice));
        const bf16 * qr = q + size_t(pos) * heads * D;
        if (mode) {
            qsa27_project(x + size_t(pos) * H, w, proj, 0, kQsa27Rows, n, H, nullptr);
            qsa27_queries(proj, kQsa27Rows, nw, iq, dpos, nullptr, n, 1e7f, 1e-6f, nullptr);
            qsa27_select(iq, blocks, bits, sel, dpos, nullptr, n, from_, cap / 4, nullptr);
            if (all) {
                std::vector<int32_t> every(size_t(n) * kQsa27Pick);
                for (size_t i = 0; i < every.size(); ++i) every[i] = int32_t(i % kQsa27Pick);
                check(cudaDeviceSynchronize());
                check(cudaMemcpy(sel, every.data(), every.size() * 4, cudaMemcpyHostToDevice));
            }
            attention_ctv_qsa(2, qr, kc, vc, nullptr, y, part, dpos + 1, n, heads, kvh, 1.f / 16, nsplit, nullptr, sel, from_, mode == 2);
        } else {
            attention_mq256_t4kv(qr, kc, vc, y, part, dpos + 1, n, heads, kvh, 1.f / 16, nsplit, nullptr, 0);
        }
        return download(y, size_t(n) * heads * D);
    };
    bool same = true;
    for (const auto & [pos, n] : {std::pair<int, int>{6000, 8}, {4105, 8}, {7003, 5}}) {
        tqllm::QsaPlan plan; plan.keyed = pos;
        const auto block = run(pos, n, plan.write(pos, n, true, true, from).mode, from);
        for (int z = 0; z < n; ++z) {
            tqllm::QsaPlan one; one.keyed = pos + z;
            const auto row = run(pos + z, 1, one.write(pos + z, 1, true, true, from).mode, from);
            same &= std::memcmp(row.data(), block.data() + size_t(z) * heads * D, row.size() * sizeof(bf16)) == 0;
        }
    }
    expect(same, "t4kv: row == one-token step, bits (sparse and crossing blocks)");
    const auto sparse = run(2063, 8, 1, 2048, true), dense = run(2063, 8, 0, 2048);
    expect(std::memcmp(sparse.data(), dense.data(), dense.size() * sizeof(bf16)) == 0, "t4kv: every block picked == dense attention, bits");
    const auto s2 = run(7003, 4, 1, from), d2 = run(7003, 4, 0, from);
    expect(std::memcmp(s2.data(), d2.data(), s2.size() * sizeof(bf16)) != 0, "t4kv: a sparse row differs from dense");
    {  // staged host KV tail (see staged_tail): the picked host rows staged back, then the cache as usual
        const size_t kr = size_t(kvh) * sizeof(Turbo4K256);
        const auto kb = staged_tail(kc, cap * kr, int64_t(6000) * kr + 100), vb = staged_tail(vc, cap * kr, int64_t(6000) * kr + 52);
        auto * nbytes = upload(std::vector<int64_t>{kb.evicted, vb.evicted});
        auto * mark = upload(std::vector<int>(cap / 4 + 1, -1));
        check(cudaMemset(mark + cap / 4, 0, sizeof(int)));
        auto sparse_rows = [&](const void * K, const void * V, bool stage) {
            const int p2[2] = {7003, 7004};
            check(cudaMemcpy(dpos, p2, sizeof(p2), cudaMemcpyHostToDevice));
            qsa27_project(x + size_t(7003) * H, w, proj, 0, kQsa27Rows, 5, H, nullptr);
            qsa27_queries(proj, kQsa27Rows, nw, iq, dpos, nullptr, 5, 1e7f, 1e-6f, nullptr);
            qsa27_select(iq, blocks, bits, sel, dpos, nullptr, 5, from, cap / 4, nullptr);
            if (stage) qsa27_stage_selected(sel, 5, dpos, nullptr, from, kb.host, vb.host, const_cast<void *>(K), const_cast<void *>(V), nbytes,
                                            int(kr), int(kr), mark, mark + cap / 4, nullptr);
            attention_ctv_qsa(2, q + size_t(7003) * heads * D, K, V, nullptr, y, part, dpos + 1, 5, heads, kvh, 1.f / 16, nsplit, nullptr, sel,
                              from, false);
            return download(y, size_t(5) * heads * D);
        };
        const auto ref = sparse_rows(kc, vc, false), bad = sparse_rows(kb.stale, vb.stale, false), got = sparse_rows(kb.stale, vb.stale, true);
        expect(std::memcmp(ref.data(), got.data(), ref.size() * sizeof(bf16)) == 0, "t4kv: picked host rows staged back == device-resident cache, bits");
        expect(std::memcmp(ref.data(), bad.data(), ref.size() * sizeof(bf16)) != 0, "t4kv: (the stale cache alone differs: the test reads the staged rows)");
        kb.release(); vb.release(); cudaFree(nbytes); cudaFree(mark);
    }
}

// QSA_BENCH=1: kernel times at the 27B's shapes (80K keys, hidden 5120), sparse steps beside the dense two-part attention
static void bench() {
    constexpr int H = 5120, heads = 24, kvh = 4, D = 256, cap = 80000, nsplit = 20, NV = 16, from = 49152, pos = 79000;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    auto halves = [&](size_t n) { std::vector<__half> v(n); for (auto & e : v) e = __float2half(ud(rng)); return v; };
    auto bfs = [&](size_t n) { std::vector<bf16> v(n); for (size_t i = 0; i < n; ++i) v[i] = __float2bfloat16_rn(float((i * 2654435761u >> 8) & 1023) / 512.f - 1.f); return v; };
    auto * w = upload(halves(size_t(kQsa27Rows) * H)); auto * x = upload(halves(size_t(NV) * H));
    auto * tok = upload(halves(size_t(cap) * kQsa27Dim));
    auto * nw = upload(std::vector<float>(kQsa27Dim, 1.f));
    auto * blocks = dev<float>(size_t(cap / 4) * kQsa27Dim);
    auto * proj = dev<float>(size_t(NV) * kQsa27Rows); auto * iq = dev<float>(size_t(NV) * kQsa27Heads * kQsa27Dim);
    auto * part = dev<float>(size_t(attention_mq_partial_floats(heads, NV, nsplit + 1)));
    auto * q = upload(bfs(size_t(NV) * heads * D)); auto * kc = upload(bfs(size_t(cap) * kvh * D)); auto * vc = upload(bfs(size_t(cap) * kvh * D));
    auto * y = dev<bf16>(size_t(NV) * heads * D);
    auto * bits = dev<uint32_t>(size_t(NV) * (cap / 4)); auto * sel = dev<int32_t>(size_t(NV) * kQsa27Pick);
    auto * dpos = upload(std::vector<int>{pos, pos + 1});
    std::vector<int> chain(NV), depth(NV);
    for (int i = 0; i < NV; ++i) chain[size_t(i)] = i - 1, depth[size_t(i)] = i;
    auto * par = upload(chain); auto * dep = upload(depth);
    qsa27_pool_blocks(tok, nw, blocks, 0, cap / 4, 1e7f, 1e-6f, nullptr);
    auto time = [&](const char * name, int n, auto && fn) {
        cudaEvent_t e0, e1; check(cudaEventCreate(&e0)); check(cudaEventCreate(&e1));
        fn(); check(cudaDeviceSynchronize());
        check(cudaEventRecord(e0));
        for (int i = 0; i < 20; ++i) fn();
        check(cudaEventRecord(e1)); check(cudaEventSynchronize(e1));
        float ms = 0; check(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("  %-22s n=%2d  %7.1f us\n", name, n, ms * 1000 / 20);
    };
    for (int n : {1, 4, 8, 12}) {
        time("project 640", n, [&] { qsa27_project(x, w, proj, 0, kQsa27Rows, n, H, nullptr); });
        time("token keys", n, [&] { qsa27_token_keys(proj, kQsa27Rows, 512, nw, tok, dpos, n, 1e-6f, nullptr); });
        time("queries", n, [&] { qsa27_queries(proj, kQsa27Rows, nw, iq, dpos, dep, n, 1e7f, 1e-6f, nullptr); });
        time("select", n, [&] { qsa27_select(iq, blocks, bits, sel, dpos, dep, n, from, cap / 4, nullptr); });
        time("attention sparse", n, [&] { attention_mq256_qsa(q, kc, vc, y, part, dpos + 1, n, heads, kvh, 1.f / 16, nsplit, nullptr, par, dep, sel, from, false); });
        {  // the same contiguous blocks for every row: no gather cost, compute only
            std::vector<int32_t> same(size_t(NV) * kQsa27Pick);
            for (size_t i = 0; i < same.size(); ++i) same[i] = int32_t(i % kQsa27Pick);
            auto * ds = upload(same);
            time("sparse, contiguous", n, [&] { attention_mq256_qsa(q, kc, vc, y, part, dpos + 1, n, heads, kvh, 1.f / 16, nsplit, nullptr, par, dep, ds, from, false); });
            cudaFree(ds);
        }
        time("attention dense", n, [&] { attention_mq256(q, kc, vc, y, part, dpos + 1, n, heads, kvh, 1.f / 16, nsplit, nullptr, 0, par, dep); });
    }
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    if (std::getenv("QSA_BENCH")) { bench(); return 0; }
    bool agree = true;
    for (int p = 0; p < 300000; ++p) agree &= qsa27_split_point(p) == tqllm::qsa_split_point(p);
    expect(agree, "host split point == tq_split_point");
    std::mt19937 rng(2711);
    test_select(rng);
    Fixture f(rng);
    test_row_equals_step(f);
    test_pick_all(f);
    test_codecs(rng);
    test_t4kv(rng);
    test_stage_selected();
    check(cudaDeviceSynchronize());
    std::puts(failures ? "qsa27 FAIL" : "qsa27 PASS");
    return failures ? 1 : 0;
}
