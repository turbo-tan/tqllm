// Bench + bit check: the Flash-Next decode matvecs at their real shapes (one 3090 sequence: T = 1 decode, T = 2-6 verify),
// timed by CUDA-graph replay as decode runs them (no launch gaps), on weights that stream from DRAM (512-slot expert banks
// tiled as the engine tiles them, dense K6 / K5 tiled by tile_dense, rotating copies past the L2). Prints us and GB/s of
// weight bytes per call and an FNV-1a hash of each case's outputs; with OUTDIR every output is written to
// OUTDIR/<case>.bin so two builds can be compared byte for byte (cmp).
// usage: mv-decode-bench [iters=20] [case substring] [OUTDIR]
#include "formats/weight/weight_format.hpp"
#include "formats/weight/quant_kernels.cuh"
#include "core/gguf.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>
using namespace tqllm;

static void ck(cudaError_t e, const char * w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
static uint64_t fnv(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t *>(p)[i]) * 1099511628211ull;
    return h;
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    const char * only = argc > 2 && std::strcmp(argv[2], "all") ? argv[2] : nullptr;
    const char * outdir = argc > 3 ? argv[3] : nullptr;
    constexpr int topk = 10, slots = 512, DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols, H = 2560, FF = 640;
    std::mt19937 rng(11);
    const Upload up = [](const void * h, size_t n) { void * p = nullptr; ck(cudaMalloc(&p, n), "malloc"); ck(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice), "up"); return static_cast<const void *>(p); };
    // every code / index / sign / sub-scale byte over its full range; only the FP16 block scales are set (random in
    // [2^-12, 2^-6), either sign where the format allows it), so outputs stay finite and every decode branch is taken
    auto rawbytes = [&](uint32_t ty, int rows, int cols) {
        std::vector<char> raw(gguf_tensor_bytes(ty, {uint64_t(cols), uint64_t(rows)}, "w"));
        for (auto & c : raw) c = char(rng() & 255);
        struct L { int block; std::vector<int> d; };  // block bytes, FP16 scale offsets in the block
        L l;
        switch (ty) {
            case 22: l = {82, {0}}; break;            // IQ2_S
            case 18: l = {98, {0}}; break;            // IQ3_XXS
            case 21: l = {110, {0}}; break;           // IQ3_S
            case 23: l = {136, {0}}; break;           // IQ4_XS
            case 20: l = {18, {0}}; break;            // IQ4_NL
            case 12: l = {144, {0, 2}}; break;        // Q4_K d, dmin
            case 13: l = {176, {0, 2}}; break;        // Q5_K d, dmin
            case 14: l = {210, {208}}; break;         // Q6_K
            case 8: l = {34, {0}}; break;             // Q8_0
            case GGML_GSQ_Q2_0: l = {18, {0}}; break; // GSQ Q2_0 (18 bytes per 64)
            default: std::fprintf(stderr, "no scale layout for type %u\n", ty); std::exit(1);
        }
        if (raw.size() % size_t(l.block)) { std::fprintf(stderr, "type %u: %zu bytes not whole blocks of %d\n", ty, raw.size(), l.block); std::exit(1); }
        std::uniform_real_distribution<float> e(-12.f, -6.f);
        for (size_t b = 0; b < raw.size(); b += size_t(l.block))
            for (int off : l.d) {
                const __half v = __float2half(std::exp2(e(rng)) * ((rng() & 1) ? -1.f : 1.f));
                std::memcpy(raw.data() + b + off, &v, 2);
            }
        return raw;
    };
    __half * dx; ck(cudaMalloc(&dx, size_t(64) * 12288 * 2), "x");
    {
        std::vector<__half> xh(size_t(64) * 12288);
        std::uniform_real_distribution<float> xd(-1.f, 1.f);
        for (auto & v : xh) v = __float2half(xd(rng));
        cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
    }
    cudaStream_t st; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // calls(i) runs the case's i-th rotation; out/out_bytes: what call 0 writes
    auto run = [&](const std::string & name, double bytes, int rot, const std::function<void(int)> & call, const void * out, size_t out_bytes) {
        ck(cudaMemsetAsync(const_cast<void *>(out), 0, out_bytes, st), "memset");
        call(0); ck(cudaStreamSynchronize(st), name.c_str());
        std::vector<char> host(out_bytes); cudaMemcpy(host.data(), out, out_bytes, cudaMemcpyDeviceToHost);
        if (outdir) { const std::string f = std::string(outdir) + "/" + name + ".bin"; if (FILE * fp = std::fopen(f.c_str(), "wb")) { std::fwrite(host.data(), 1, host.size(), fp); std::fclose(fp); } }
        const int per = 40;
        cudaGraph_t g; cudaGraphExec_t ge;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
        for (int i = 0; i < per; ++i) call(1 + i % std::max(1, rot - 1));
        ck(cudaStreamEndCapture(st, &g), "end capture");
        ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
        for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, st);
        cudaEventRecord(e0, st);
        for (int i = 0; i < iters; ++i) cudaGraphLaunch(ge, st);
        cudaEventRecord(e1, st); ck(cudaEventSynchronize(e1), "sync");
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1e3 * ms / (iters * per);
        std::printf("%-34s %8.2f us %6.0f GB/s  hash %016llx\n", name.c_str(), us, bytes / us / 1e3, (unsigned long long) fnv(host.data(), host.size()));
        std::fflush(stdout);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    };
    auto want = [&](const std::string & n) { return !only || n.find(only) != std::string::npos; };

    // ---- experts: 512-slot banks, tiled as the engine does (resident banks), T tokens x 10 assignments
    struct Ex { uint32_t ty; const char * what; };
    const Ex gu_types[] = {{22, "IQ2_S"}, {18, "IQ3_XXS"}, {21, "IQ3_S"}, {23, "IQ4_XS"}};
    const Ex dn_types[] = {{20, "IQ4_NL"}, {GGML_GSQ_Q2_0, "GSQ_Q2_0"}};
    auto bank = [&](uint32_t ty, int rows, int cols) {
        const auto * f = weight_format(ty);
        Mat m = f->load(rawbytes(ty, rows * slots, cols), rows * slots, cols, up, nullptr);
        const int bpb = m.fmt->u4h_id() >= 0 ? tqllm_cuda::tiled_block_bytes(m.fmt->u4h_id()) : 0;
        if (bpb) tqllm_cuda::tile_expert_rows_device(static_cast<char *>(const_cast<void *>(m.p)), slots, rows, cols, bpb, nullptr);
        ck(cudaDeviceSynchronize(), "bank");
        return m;
    };
    const int R = 24;  // rotations: disjoint-ish expert sets, so the weights come from DRAM
    // ids[r][a]: T tokens x topk distinct slots per token, shared across tokens with p = 0.4 (verify tokens repeat experts)
    auto make_ids = [&](int T) {
        std::vector<int> ids(size_t(R) * T * topk);
        for (int r = 0; r < R; ++r)
            for (int t = 0; t < T; ++t) {
                int * row = ids.data() + (size_t(r) * T + t) * topk;
                for (int j = 0; j < topk; ++j) {
                    int s;
                    do {
                        s = (t > 0 && rng() % 10 < 4) ? ids[(size_t(r) * T + t - 1) * topk + rng() % topk] : int(rng() % slots);
                    } while (std::find(row, row + j, s) != row + j);
                    row[j] = s;
                }
            }
        return ids;
    };
    auto make_dmap = [&](const std::vector<int> & ids, int T, bool per_expert, std::vector<int> & groups) {
        const int A = T * topk;
        std::vector<int> map(size_t(R) * A * DS, -1);
        groups.assign(R, 0);
        for (int r = 0; r < R; ++r) {
            int * out = map.data() + size_t(r) * A * DS; int g = 0;
            std::vector<int> open_at(slots, -1);
            for (int a = 0; a < A; ++a) {
                const int slot = ids[size_t(r) * A + a]; int & at = open_at[slot];
                if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = slot; out[at * DS + 1] = 0; }
                int * e = out + at * DS; const int col = e[1]++;
                e[2 + col] = per_expert ? a : a / topk; e[2 + DC + col] = a;
            }
            groups[r] = g;
        }
        return map;
    };
    auto dev_ints = [&](const std::vector<int> & v) { int * p; ck(cudaMalloc(&p, v.size() * 4), "ints"); cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice); return p; };
    tqllm_cuda::set_tiled_experts(true);
    if (const char * v = std::getenv("MV_DEDUP_MT")) {  // bench only: "fmt,paired,single" row tiles of the dedup launches
        int f = 0, pm = 0, sm = 0;
        if (std::sscanf(v, "%d,%d,%d", &f, &pm, &sm) == 3) tqllm_cuda::set_dedup_mt(f, pm, sm);
    }
    for (const Ex & ex : gu_types) {
        const std::string tag = std::string("gu ") + ex.what;
        bool any = false; for (int T : {1, 2, 4, 6}) any |= want(tag + " T" + std::to_string(T));
        if (!any) continue;
        const Mat mg = bank(ex.ty, FF, H), mu = bank(ex.ty, FF, H);
        const double bytes1 = double(mg.fmt->row_bytes(H) + mg.fmt->aux_row_bytes(H)) * FF;  // one expert, one matrix
        __nv_bfloat16 * yg, * yu; ck(cudaMalloc(&yg, size_t(6) * topk * FF * 2), "y"); ck(cudaMalloc(&yu, size_t(6) * topk * FF * 2), "y");
        for (int T : {1, 2, 4, 6}) {
            const std::string nm = tag + " T" + std::to_string(T);
            if (!want(nm)) continue;
            const int A = T * topk;
            const auto ids = make_ids(T); int * di = dev_ints(ids);
            std::vector<int> groups; const auto map = make_dmap(ids, T, false, groups); int * dm = dev_ints(map);
            double uniq = 0; for (int g : groups) uniq += g; uniq /= R;
            if (T == 1) {
                run(nm + " pair", 2 * bytes1 * topk, R, [&](int r) { mg.fmt->expert_matvec2(mg, mu, dx, yg, yu, FF, di + size_t(r) * A, T, topk, st); }, yg, size_t(A) * FF * 2);
                run(nm + " pair.up", 2 * bytes1 * topk, R, [&](int r) { mg.fmt->expert_matvec2(mg, mu, dx, yg, yu, FF, di + size_t(r) * A, T, topk, st); }, yu, size_t(A) * FF * 2);
                std::vector<int> part = ids;  // ~30 % of assignments -1 (the split path's hit launch with misses)
                for (auto & v : part) if (rng() % 10 < 3) v = -1;
                int * dp = dev_ints(part);
                run(nm + " pair.skip", 2 * bytes1 * topk * 0.7, R, [&](int r) { mg.fmt->expert_matvec2(mg, mu, dx, yg, yu, FF, dp + size_t(r) * A, T, topk, st); }, yu, size_t(A) * FF * 2);
                cudaFree(dp);
                run(nm + " two", 2 * bytes1 * topk, R, [&](int r) { mg.fmt->expert_matvec(mg, dx, yg, FF, di + size_t(r) * A, T, topk, false, st);
                                                                    mu.fmt->expert_matvec(mu, dx, yu, FF, di + size_t(r) * A, T, topk, false, st); }, yg, size_t(A) * FF * 2);
            } else {
                run(nm + " dedup2", 2 * bytes1 * uniq, R, [&](int r) { mg.fmt->expert_matvec_dedup2(mg, mu, dx, yg, yu, FF, dm + size_t(r) * A * DS, groups[r], st); }, yg, size_t(A) * FF * 2);
                run(nm + " dedup2.up", 2 * bytes1 * uniq, R, [&](int r) { mg.fmt->expert_matvec_dedup2(mg, mu, dx, yg, yu, FF, dm + size_t(r) * A * DS, groups[r], st); }, yu, size_t(A) * FF * 2);
            }
            cudaFree(di); cudaFree(dm);
        }
        cudaFree(yg); cudaFree(yu); cudaFree(const_cast<void *>(mg.p)); cudaFree(const_cast<void *>(mu.p));
        if (mg.aux) cudaFree(const_cast<void *>(mg.aux)); if (mu.aux) cudaFree(const_cast<void *>(mu.aux));
    }
    for (const Ex & ex : dn_types) {
        const std::string tag = std::string("dn ") + ex.what;
        bool any = false; for (int T : {1, 2, 4, 6}) any |= want(tag + " T" + std::to_string(T));
        if (!any) continue;
        const Mat md = bank(ex.ty, H, FF);
        const double bytes1 = double(md.fmt->row_bytes(FF) + md.fmt->aux_row_bytes(FF)) * H;
        __nv_bfloat16 * yd; ck(cudaMalloc(&yd, size_t(6) * topk * H * 2), "y");
        for (int T : {1, 2, 4, 6}) {
            const std::string nm = tag + " T" + std::to_string(T);
            if (!want(nm)) continue;
            const int A = T * topk;
            const auto ids = make_ids(T); int * di = dev_ints(ids);
            std::vector<int> groups; const auto map = make_dmap(ids, T, true, groups); int * dm = dev_ints(map);
            double uniq = 0; for (int g : groups) uniq += g; uniq /= R;
            if (T == 1 || !md.fmt->has_expert_dedup())
                run(nm + " grouped", bytes1 * (T == 1 ? topk : uniq), R, [&](int r) { md.fmt->expert_matvec(md, dx, yd, H, di + size_t(r) * A, T, topk, true, st); }, yd, size_t(A) * H * 2);
            else
                run(nm + " dedup", bytes1 * uniq, R, [&](int r) { md.fmt->expert_matvec_dedup(md, dx, yd, H, dm + size_t(r) * A * DS, groups[r], st); }, yd, size_t(A) * H * 2);
            cudaFree(di); cudaFree(dm);
        }
        cudaFree(yd); cudaFree(const_cast<void *>(md.p));
    }
    tqllm_cuda::set_tiled_experts(false);

    // ---- dense: the decode shapes x the formats the file uses for them; K6 / K5 tiled (tile_dense) as resident in the engine
    struct Shape { int rows, cols; const char * what; std::vector<uint32_t> types; };
    const std::vector<Shape> shapes = {
        {10240, 2560, "GDN qkv", {14, 13, 12, 23}}, {6144, 2560, "GDN gate z", {14, 13, 12, 23}}, {2560, 6144, "GDN out", {14, 13, 12}},
        {12288, 2560, "attn q", {14, 13, 23}}, {512, 2560, "attn k/v", {14, 13}}, {2560, 6144, "attn out", {14}},
        {640, 2560, "shared gate/up", {12, 14, 23, 13}}, {2560, 640, "shared down", {20}}};
    for (const auto & sh : shapes)
        for (uint32_t ty : sh.types) {
            const auto * f = weight_format(ty);
            const std::string base = std::string(sh.what) + " " + f->name();
            bool any = false; for (int n : {1, 2, 4, 6}) any |= want(base + " n" + std::to_string(n));
            if (!any) continue;
            const auto raw = rawbytes(ty, sh.rows, sh.cols);
            std::vector<Mat> ms;
            const size_t one = raw.size();
            const int copies = int(std::max<size_t>(4, (size_t(96) << 20) / one));
            bool tiled = false;
            for (int c = 0; c < copies; ++c) { Mat m = f->load(raw, sh.rows, sh.cols, up, nullptr); tiled = tile_dense(m); ms.push_back(m); }
            const double bytes = double(ms[0].fmt->row_bytes(sh.cols) + ms[0].fmt->aux_row_bytes(sh.cols)) * sh.rows;
            const double bytes_raw = double(one);
            __nv_bfloat16 * y; ck(cudaMalloc(&y, size_t(6) * sh.rows * 2), "y");
            for (int n : {1, 2, 4, 6}) {
                const std::string nm = base + " n" + std::to_string(n);
                if (!want(nm)) continue;
                run(nm + (tiled ? " tiled" : ""), bytes > 0 ? bytes : bytes_raw, copies,
                    [&](int r) { const Mat & m = ms[size_t(r) % ms.size()]; m.fmt->matvec(m, dx, y, n, st); }, y, size_t(n) * sh.rows * 2);
            }
            cudaFree(y);
            for (auto & m : ms) { cudaFree(const_cast<void *>(m.p)); if (m.aux) cudaFree(const_cast<void *>(m.aux)); }
        }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
