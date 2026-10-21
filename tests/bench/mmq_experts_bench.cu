// src/experts/mmq_experts.cu (our bank layout, slot groups, activations quantised once per input row, bf16 out) vs stock ggml
// (libggml-cuda's ggml_mul_mat_id on the same data as GGUF blocks, fp32 out), same routing (Zipf, top-10 of 512).
// Gate: bf16(stock) == ours bit for bit. Then time: ours (quantize + GEMM), GEMM only, stock mul_mat_id.
// usage: mmq_experts_bench [T=8192] [fmt=5: 4 IQ3_S, 5 IQ3_XXS, 10 IQ2_S, 13 IQ4_XS, 14 IQ4_NL] [down]
//   gate/up: 640 x 2560 per expert, per-token input; down: 2560 x 640, per-assignment input
// Build (needs a llama.cpp build for libggml): see the NOTICE in third_party/ggml_mmq
#define TQ_MMQ_LINKS_GGML 1
#include "mmq_experts.cu"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int T = argc > 1 ? std::atoi(argv[1]) : 8192, fmt = argc > 2 ? std::atoi(argv[2]) : 5;
    const bool down = argc > 3 && std::string(argv[3]) == "down";
    const int E = 512, TOPK = 10, ROWS = down ? 2560 : 640, K = down ? 640 : 2560, R = E * ROWS, A = T * TOPK, nb32 = K / 32;
    ggml_type type; int bpb, rb;  // our bytes per 32 weights, record bytes per 256 (0: IQ4_NL's fp16 scale array)
    switch (fmt) {
        case 4: type = GGML_TYPE_IQ3_S; bpb = 12; rb = 16; break;
        case 5: type = GGML_TYPE_IQ3_XXS; bpb = 12; rb = 4; break;
        case 10: type = GGML_TYPE_IQ2_S; bpb = 8; rb = 20; break;
        case 13: type = GGML_TYPE_IQ4_XS; bpb = 16; rb = 8; break;
        case 14: type = GGML_TYPE_IQ4_NL; bpb = 16; rb = 0; break;
        default: std::fprintf(stderr, "fmt %d\n", fmt); return 2;
    }
    const size_t ts = ggml_type_size(type); const int qk = int(ggml_blck_size(type)), nbq = K / qk;
    if (K % qk) { std::fprintf(stderr, "K %d not a multiple of %d\n", K, qk); return 2; }
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> w(size_t(R) * nbq * ts);
    for (auto & b : w) b = uint8_t(byte(rng));
    for (size_t b = 0; b < size_t(R) * nbq; ++b) {  // sane fp16 d at the head of every block
        const __half d = __float2half(0.002f + 0.001f * std::fabs(nd(rng))); std::memcpy(&w[b * ts], &d, 2);
    }
    // our layout: tiled blocks of bpb bytes per 32 weights; records per 256 (or fp16 scales per 32 for IQ4_NL)
    std::vector<uint8_t> codes(size_t(R) * nb32 * bpb), rec(rb ? size_t(R) * (K / 256) * rb : size_t(R) * nb32 * 2, 0);
    auto put = [&](int r, int s, const uint8_t * a, int na, const uint8_t * b, int nbb) {
        const size_t to = (((size_t(r >> 4) * (nb32 >> 2) + (s >> 2)) * 16 + (r & 15)) * 4 + (s & 3)) * bpb;
        std::memcpy(&codes[to], a, na); if (nbb) std::memcpy(&codes[to + na], b, nbb);
    };
    auto nat = [](const uint8_t * qs, uint8_t * o) {  // GGUF IQ4 nibbles -> the engine's natural order (byte j = weights 2j, 2j + 1)
        auto g = [&](int e) { return e < 16 ? qs[e] & 0xF : qs[e - 16] >> 4; };
        for (int j = 0; j < 16; ++j) o[j] = uint8_t(g(2 * j) | (g(2 * j + 1) << 4));
    };
    uint8_t nb16[16];
    for (int r = 0; r < R; ++r)
        for (int q = 0; q < nbq; ++q) {
            const uint8_t * B = &w[(size_t(r) * nbq + q) * ts];
            uint8_t * rc = rb ? &rec[(size_t(r) * nbq + q) * rb] : nullptr;
            switch (fmt) {
                case 5: std::memcpy(rc, B, 2); for (int j = 0; j < 8; ++j) put(r, 8 * q + j, B + 2 + 8 * j, 8, B + 66 + 4 * j, 4); break;
                case 10: std::memcpy(rc, B, 2); std::memcpy(rc + 2, B + 66, 8); std::memcpy(rc + 10, B + 74, 8);
                         for (int j = 0; j < 8; ++j) put(r, 8 * q + j, B + 2 + 4 * j, 4, B + 34 + 4 * j, 4); break;
                case 4: std::memcpy(rc, B, 2); std::memcpy(rc + 2, B + 66, 8); std::memcpy(rc + 10, B + 106, 4);
                        for (int j = 0; j < 8; ++j) put(r, 8 * q + j, B + 2 + 8 * j, 8, B + 74 + 4 * j, 4); break;
                case 13: std::memcpy(rc, B, 8); for (int j = 0; j < 8; ++j) { nat(B + 8 + 16 * j, nb16); put(r, 8 * q + j, nb16, 16, nullptr, 0); } break;
                case 14: std::memcpy(&rec[(size_t(r) * nb32 + q) * 2], B, 2); nat(B + 2, nb16); put(r, q, nb16, 16, nullptr, 0); break;
            }
        }
    std::vector<double> pop(E); for (int e = 0; e < E; ++e) pop[e] = 1.0 / std::pow(e + 1.0, 0.6);
    std::shuffle(pop.begin(), pop.end(), rng); std::discrete_distribution<int> pick(pop.begin(), pop.end());
    std::vector<std::vector<int>> per(E);
    for (int t = 0; t < T; ++t) {
        std::vector<int> ch; while (int(ch.size()) < TOPK) { int e = pick(rng); if (std::find(ch.begin(), ch.end(), e) == ch.end()) ch.push_back(e); }
        for (int j = 0; j < TOPK; ++j) per[ch[j]].push_back(t * TOPK + j);
    }
    int cap = 1; for (auto & v : per) cap = std::max(cap, int(v.size()));
    std::vector<int32_t> counts(E), groups(size_t(E) * cap, 0), idv(static_cast<size_t>(A));
    for (int e = 0; e < E; ++e) {
        counts[e] = int(per[e].size());
        for (size_t j = 0; j < per[e].size(); ++j) { groups[size_t(e) * cap + j] = per[e][j]; idv[size_t(per[e][j])] = e; }
    }
    const int NY = down ? A : T;  // input rows: per token (gate/up) or per assignment (down)
    std::vector<float> x(size_t(NY) * K); std::vector<__half> xh(x.size());
    for (size_t i = 0; i < x.size(); ++i) { xh[i] = __float2half(0.3f * nd(rng)); x[i] = __half2float(xh[i]); }

    uint8_t * dc, * dr; __half * dxh; __nv_bfloat16 * dout; int32_t * dcnt, * dgrp; char * dq; int2 * dtiles;
    const int tcap = tqllm_cuda::mmq_max_tiles(cap, E);
    CK(cudaMalloc(&dc, codes.size())); CK(cudaMalloc(&dr, rec.size())); CK(cudaMalloc(&dxh, xh.size() * 2));
    CK(cudaMalloc(&dout, size_t(A) * ROWS * 2)); CK(cudaMalloc(&dcnt, E * 4)); CK(cudaMalloc(&dgrp, groups.size() * 4));
    CK(cudaMalloc(&dq, tqllm_cuda::mmq_act_bytes(NY, K))); CK(cudaMalloc(&dtiles, size_t(tcap) * sizeof(int2)));
    CK(cudaMemcpy(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dr, rec.data(), rec.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxh, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dcnt, counts.data(), E * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dgrp, groups.data(), groups.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t st = cudaStreamPerThread;

    // stock: libggml-cuda's mul_mat_id (its MMQ, routing helpers, quantizer), output [T][TOPK][ROWS] fp32
    ggml_backend_t be = ggml_backend_cuda_init(0);
    ggml_init_params ip{ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    ggml_context * gctx = ggml_init(ip);
    ggml_tensor * gw = ggml_new_tensor_3d(gctx, type, K, ROWS, E);
    ggml_tensor * gx = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, K, down ? TOPK : 1, T);
    ggml_tensor * gid = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, TOPK, T);
    ggml_tensor * gout = ggml_mul_mat_id(gctx, gw, gx, gid);
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, be); (void) gbuf;
    ggml_backend_tensor_set(gw, w.data(), 0, w.size());
    ggml_backend_tensor_set(gx, x.data(), 0, x.size() * 4);
    ggml_backend_tensor_set(gid, idv.data(), 0, idv.size() * 4);
    ggml_cgraph * gf = ggml_new_graph(gctx); ggml_build_forward_expand(gf, gout);
    auto stock = [&]() { ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be); };
    auto ours_gemm = [&]() {
        if (!tqllm_cuda::mmq_expert_gemm(fmt, dc, dr, true, ROWS, K, dcnt, counts.data(), dgrp, cap, E, down ? 1 : TOPK, dq, NY, dout, dtiles, tcap, st))
            { std::fprintf(stderr, "mmq_expert_gemm: unsupported\n"); std::exit(1); }
    };
    auto ours = [&]() {  // down: the engine's per-pass form (the pass's rows only, from its groups)
        if (down) tqllm_cuda::mmq_quantize_act_rows(dxh, NY, K, dgrp, dcnt, cap, E, dq, st);
        else tqllm_cuda::mmq_quantize_act(dxh, NY, K, dq, st);
        ours_gemm();
    };
    stock(); ours(); CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
    std::vector<float> ref(size_t(A) * ROWS); std::vector<__nv_bfloat16> out(ref.size());
    ggml_backend_tensor_get(gout, ref.data(), 0, ref.size() * 4);
    CK(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost));
    size_t diff = 0, first = SIZE_MAX;
    for (size_t i = 0; i < ref.size(); ++i) {
        const __nv_bfloat16 r = __float2bfloat16_rn(ref[i]);
        if (std::memcmp(&r, &out[i], 2)) { ++diff; if (first == SIZE_MAX) first = i; }
    }
    std::printf("%s %s T%d: %zu / %zu bf16 outputs differ", ggml_type_name(type), down ? "down" : "gate/up", T, diff, ref.size());
    if (diff) std::printf(" (first at %zu: ref %g ours %g)", first, ref[first], __bfloat162float(out[first]));
    std::printf("  cap %d\n", cap);
    auto timeit = [&](const char * name, auto && f) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        f(); cudaEventRecord(e0, st); for (int i = 0; i < 20; ++i) f(); cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1); std::printf("  %-28s %7.3f ms\n", name, ms / 20);
    };
    timeit("stock mul_mat_id (all of it)", stock);
    timeit("ours GEMM", ours_gemm);
    timeit("ours quantize + GEMM", ours);
    std::fflush(stdout);
    std::_Exit(0);
}
