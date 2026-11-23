// GB10 microbenchmark: native FP4 gate/up/down against the existing BF16 grouped kernels.
// Build (from the source root):
//   nvcc -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a \
//     -I . -I src -I src/models/inkling2/loop -I third_party -I third_party/ggml_mmq/ggml \
//     -Xcompiler -fPIC -c tests/inkling2_fp4_micro.cu -o /tmp/inkling2_fp4_micro.o
//   nvcc -gencode arch=compute_121a,code=sm_121a /tmp/inkling2_fp4_micro.o /tmp/inkling2_grouped.o \
//     -lcublas -lcudart -o /tmp/inkling2_fp4_micro
#include "inkling2_grouped.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace tqllm_cuda::inkling2;

static void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Mat {
    ExpertMat m;
    uint8_t * w = nullptr;
    uint8_t * a = nullptr;
};

static Mat make_mat(int E, int rows, int cols, uint32_t seed) {
    Mat x;
    const size_t stride = size_t(rows) * cols / 2;
    const size_t aux_stride = size_t(rows) * (cols / 16 + 4);
    x.w = nullptr; x.a = nullptr;
    ck(cudaMalloc(&x.w, stride * E), "malloc w");
    ck(cudaMalloc(&x.a, aux_stride * E), "malloc aux");
    std::vector<uint8_t> hw(stride * E), ha(aux_stride * E);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(0, 255);
    for (auto & v : hw) v = uint8_t(code(rng));
    for (int e = 0; e < E; ++e) {
        for (int r = 0; r < rows; ++r) {
            uint8_t * ar = ha.data() + size_t(e) * aux_stride + size_t(r) * (cols / 16 + 4);
            for (int k = 0; k < cols / 16; ++k) ar[k] = uint8_t(0x34 + (k % 4));  // E4M3 0.25 .. 1.0
            const float g = 0.08f + 0.01f * float((r + e) % 5);
            std::memcpy(ar + cols / 16, &g, 4);
        }
    }
    ck(cudaMemcpy(x.w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "copy w");
    ck(cudaMemcpy(x.a, ha.data(), ha.size(), cudaMemcpyHostToDevice), "copy aux");
    x.m.w = x.w; x.m.aux = x.a; x.m.stride = stride; x.m.aux_stride = aux_stride; x.m.fmt = 2;
    return x;
}

static void fill_const(Mat & x, int E, int rows, int cols, float g0, float g1) {
    std::vector<uint8_t> hw(x.m.stride * E, 0x22), ha(x.m.aux_stride * E, 0);
    for (int e = 0; e < E; ++e)
        for (int r = 0; r < rows; ++r) {
            uint8_t * ar = ha.data() + size_t(e) * x.m.aux_stride + size_t(r) * (cols / 16 + 4);
            std::memset(ar, 0x38, cols / 16);
            const float g = rows > 1 ? g0 + (g1 - g0) * float(r) / float(rows - 1) : g0;
            std::memcpy(ar + cols / 16, &g, 4);
        }
    ck(cudaMemcpy(x.w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "copy const w");
    ck(cudaMemcpy(x.a, ha.data(), ha.size(), cudaMemcpyHostToDevice), "copy const aux");
}

static void fill_random(Mat & x, int E, int rows, int cols, uint32_t seed) {
    std::vector<uint8_t> hw(x.m.stride * E), ha(x.m.aux_stride * E);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(0, 255);
    for (auto & v : hw) v = uint8_t(code(rng));
    for (int e = 0; e < E; ++e)
        for (int r = 0; r < rows; ++r) {
            uint8_t * ar = ha.data() + size_t(e) * x.m.aux_stride + size_t(r) * (cols / 16 + 4);
            for (int k = 0; k < cols / 16; ++k) ar[k] = uint8_t(0x34 + (k % 4));
            const float g = 0.08f + 0.01f * float((r + e) % 5);
            std::memcpy(ar + cols / 16, &g, 4);
        }
    ck(cudaMemcpy(x.w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "copy random w");
    ck(cudaMemcpy(x.a, ha.data(), ha.size(), cudaMemcpyHostToDevice), "copy random aux");
}

static double nmse(const std::vector<__nv_bfloat16> & got, const std::vector<__nv_bfloat16> & ref, bool & finite) {
    double num = 0.0, den = 0.0;
    finite = true;
    for (size_t i = 0; i < got.size(); ++i) {
        const double a = __bfloat162float(got[i]), b = __bfloat162float(ref[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) finite = false;
        num += (a - b) * (a - b);
        den += b * b;
    }
    return den > 0.0 ? num / den : 0.0;
}

int main() {
    const int E = 4, I = 64, D = 256, A = 1024;
    Mat gate = make_mat(E, I, D, 1), up = make_mat(E, I, D, 2), down = make_mat(E, D, I, 3);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");

    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> hx(size_t(A) * D), hact(size_t(A) * I);
    for (auto & v : hx) v = __float2bfloat16(normal(rng));
    for (auto & v : hact) v = __float2bfloat16(normal(rng));
    std::vector<int> hids(A);
    for (int a = 0; a < A; ++a) hids[a] = a % E;

    __nv_bfloat16 * x = nullptr, * act_in = nullptr, * act_ref = nullptr, * act_fp4 = nullptr, * y_ref = nullptr, * y_fp4 = nullptr;
    int * ids = nullptr, * order = nullptr, * counts = nullptr, * offsets = nullptr, * tile_off = nullptr;
    uint8_t * qcodes = nullptr, * qscales = nullptr;
    float * qrow = nullptr;
    ck(cudaMalloc(&x, hx.size() * 2), "x");
    ck(cudaMalloc(&act_in, hact.size() * 2), "act in");
    ck(cudaMalloc(&act_ref, size_t(A) * I * 2), "act ref");
    ck(cudaMalloc(&act_fp4, size_t(A) * I * 2), "act fp4");
    ck(cudaMalloc(&y_ref, size_t(A) * D * 2), "y ref");
    ck(cudaMalloc(&y_fp4, size_t(A) * D * 2), "y fp4");
    ck(cudaMalloc(&ids, size_t(A) * 4), "ids");
    ck(cudaMalloc(&order, size_t(A) * 4), "order");
    ck(cudaMalloc(&counts, size_t(E) * 4), "counts");
    ck(cudaMalloc(&offsets, size_t(E) * 4), "offsets");
    ck(cudaMalloc(&tile_off, size_t(E + 1) * 4), "tile off");
    ck(cudaMalloc(&qcodes, size_t(A) * D / 2), "qcodes");
    ck(cudaMalloc(&qscales, size_t(A) * D / 16), "qscales");
    ck(cudaMalloc(&qrow, size_t(A) * 4), "qrow");
    ck(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice), "copy x");
    ck(cudaMemcpy(act_in, hact.data(), hact.size() * 2, cudaMemcpyHostToDevice), "copy act");
    ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "copy ids");

    const float gc = 4.0f, uc = 4.0f;
    group_assignments(ids, A, E, order, counts, offsets, tile_off, s);
    grouped_gate_up(gate.m, up.m, I, D, x, A, E, 1, order, counts, offsets, tile_off, act_ref, gc, uc, s);
    quantize_act_nvfp4(x, A, D, qcodes, qscales, qrow, s);
    grouped_gate_up_fp4(gate.m, up.m, I, D, qcodes, qscales, qrow, A, E, 1, order, counts, offsets, tile_off, act_fp4, gc, uc, s);
    ck(cudaStreamSynchronize(s), "gate/up");

    std::vector<__nv_bfloat16> h_ref(size_t(A) * I), h_fp4(size_t(A) * I);
    ck(cudaMemcpy(h_ref.data(), act_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost), "read act ref");
    ck(cudaMemcpy(h_fp4.data(), act_fp4, h_fp4.size() * 2, cudaMemcpyDeviceToHost), "read act fp4");
    bool finite = false;
    auto compare_gu = [&](const char * name) {
        grouped_gate_up(gate.m, up.m, I, D, x, A, E, 1, order, counts, offsets, tile_off, act_ref, gc, uc, s);
        quantize_act_nvfp4(x, A, D, qcodes, qscales, qrow, s);
        grouped_gate_up_fp4(gate.m, up.m, I, D, qcodes, qscales, qrow, A, E, 1, order, counts, offsets, tile_off, act_fp4, gc, uc, s);
        ck(cudaStreamSynchronize(s), name);
        ck(cudaMemcpy(h_ref.data(), act_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost), name);
        ck(cudaMemcpy(h_fp4.data(), act_fp4, h_fp4.size() * 2, cudaMemcpyDeviceToHost), name);
        std::printf("%s NMSE %.6g (ref %.5f fp4 %.5f)\n", name, nmse(h_fp4, h_ref, finite),
                    __bfloat162float(h_ref[0]), __bfloat162float(h_fp4[0]));
    };
    const double gu_nmse = nmse(h_fp4, h_ref, finite);
    std::printf("gate/up NMSE %.6g finite %d\n", gu_nmse, int(finite));
    if (!finite || gu_nmse > 5e-2) {
        // controlled case: every weight is E2M1 1.0 (code 2), scale 1.0, global 1.0; x = 1
        std::vector<uint8_t> cw(gate.m.stride * E, 0x22), ca(gate.m.aux_stride * E, 0);
        for (int e = 0; e < E; ++e)
            for (int r = 0; r < I; ++r) {
                uint8_t * ar = ca.data() + size_t(e) * gate.m.aux_stride + size_t(r) * (D / 16 + 4);
                std::memset(ar, 0x38, D / 16);
                const float one = 1.0f;
                std::memcpy(ar + D / 16, &one, 4);
            }
        ck(cudaMemcpy(gate.w, cw.data(), cw.size(), cudaMemcpyHostToDevice), "const gate w");
        ck(cudaMemcpy(up.w, cw.data(), cw.size(), cudaMemcpyHostToDevice), "const up w");
        ck(cudaMemcpy(gate.a, ca.data(), ca.size(), cudaMemcpyHostToDevice), "const gate a");
        ck(cudaMemcpy(up.a, ca.data(), ca.size(), cudaMemcpyHostToDevice), "const up a");
        std::vector<__nv_bfloat16> ones(size_t(A) * D, __float2bfloat16(1.0f));
        ck(cudaMemcpy(x, ones.data(), ones.size() * 2, cudaMemcpyHostToDevice), "const x");
        grouped_gate_up(gate.m, up.m, I, D, x, A, E, 1, order, counts, offsets, tile_off, act_ref, gc, uc, s);
        quantize_act_nvfp4(x, A, D, qcodes, qscales, qrow, s);
        grouped_gate_up_fp4(gate.m, up.m, I, D, qcodes, qscales, qrow, A, E, 1, order, counts, offsets, tile_off, act_fp4, gc, uc, s);
        ck(cudaStreamSynchronize(s), "const gate/up");
        ck(cudaMemcpy(h_ref.data(), act_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost), "const ref");
        ck(cudaMemcpy(h_fp4.data(), act_fp4, h_fp4.size() * 2, cudaMemcpyDeviceToHost), "const fp4");
        std::printf("const gate/up ref %.5f fp4 %.5f (expect 16)\n", __bfloat162float(h_ref[0]), __bfloat162float(h_fp4[0]));
        // random x, constant weights: tests the B-side scale and assignment mapping
        ck(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice), "rand x");
        grouped_gate_up(gate.m, up.m, I, D, x, A, E, 1, order, counts, offsets, tile_off, act_ref, gc, uc, s);
        quantize_act_nvfp4(x, A, D, qcodes, qscales, qrow, s);
        grouped_gate_up_fp4(gate.m, up.m, I, D, qcodes, qscales, qrow, A, E, 1, order, counts, offsets, tile_off, act_fp4, gc, uc, s);
        ck(cudaStreamSynchronize(s), "randx constw");
        ck(cudaMemcpy(h_ref.data(), act_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost), "randx ref");
        ck(cudaMemcpy(h_fp4.data(), act_fp4, h_fp4.size() * 2, cudaMemcpyDeviceToHost), "randx fp4");
        std::printf("rand-x const-w NMSE %.6g\n", nmse(h_fp4, h_ref, finite));
        // random weights, x = 1: tests the A-side scale and row mapping
        fill_random(gate, E, I, D, 1); fill_random(up, E, I, D, 2);
        ck(cudaMemcpy(x, ones.data(), ones.size() * 2, cudaMemcpyHostToDevice), "ones x");
        grouped_gate_up(gate.m, up.m, I, D, x, A, E, 1, order, counts, offsets, tile_off, act_ref, gc, uc, s);
        quantize_act_nvfp4(x, A, D, qcodes, qscales, qrow, s);
        grouped_gate_up_fp4(gate.m, up.m, I, D, qcodes, qscales, qrow, A, E, 1, order, counts, offsets, tile_off, act_fp4, gc, uc, s);
        ck(cudaStreamSynchronize(s), "onesx randw");
        ck(cudaMemcpy(h_ref.data(), act_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost), "onesx ref");
        ck(cudaMemcpy(h_fp4.data(), act_fp4, h_fp4.size() * 2, cudaMemcpyDeviceToHost), "onesx fp4");
        std::printf("ones-x rand-w NMSE %.6g\n", nmse(h_fp4, h_ref, finite));
        // constant weights, row-varying global: tests C row and weight-global mapping
        for (int e = 0; e < E; ++e)
            for (int r = 0; r < I; ++r) {
                uint8_t * ar = ca.data() + size_t(e) * gate.m.aux_stride + size_t(r) * (D / 16 + 4);
                std::memset(ar, 0x38, D / 16);
                const float g = 1.0f + 10.0f * float(r);
                std::memcpy(ar + D / 16, &g, 4);
            }
        ck(cudaMemcpy(gate.w, cw.data(), cw.size(), cudaMemcpyHostToDevice), "row gate w");
        ck(cudaMemcpy(up.w, cw.data(), cw.size(), cudaMemcpyHostToDevice), "row up w");
        ck(cudaMemcpy(gate.a, ca.data(), ca.size(), cudaMemcpyHostToDevice), "row gate a");
        ck(cudaMemcpy(up.a, ca.data(), ca.size(), cudaMemcpyHostToDevice), "row up a");
        ck(cudaMemcpy(x, ones.data(), ones.size() * 2, cudaMemcpyHostToDevice), "row ones x");
        grouped_gate_up(gate.m, up.m, I, D, x, A, E, 1, order, counts, offsets, tile_off, act_ref, gc, uc, s);
        quantize_act_nvfp4(x, A, D, qcodes, qscales, qrow, s);
        grouped_gate_up_fp4(gate.m, up.m, I, D, qcodes, qscales, qrow, A, E, 1, order, counts, offsets, tile_off, act_fp4, gc, uc, s);
        ck(cudaStreamSynchronize(s), "row global");
        ck(cudaMemcpy(h_ref.data(), act_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost), "row ref");
        ck(cudaMemcpy(h_fp4.data(), act_fp4, h_fp4.size() * 2, cudaMemcpyDeviceToHost), "row fp4");
        std::printf("row-global NMSE %.6g\n", nmse(h_fp4, h_ref, finite));
        fill_random(gate, E, I, D, 1); fill_const(up, E, I, D, 1.0f, 1.0f);
        ck(cudaMemcpy(x, ones.data(), ones.size() * 2, cudaMemcpyHostToDevice), "gate rand x");
        compare_gu("gate-random up-one");
        fill_const(gate, E, I, D, 1.0f, 1.0f); fill_random(up, E, I, D, 2);
        compare_gu("gate-one up-random");
        fill_random(gate, E, I, D, 1);
        ck(cudaMemcpy(up.w, gate.w, gate.m.stride * E, cudaMemcpyDeviceToDevice), "same up w");
        ck(cudaMemcpy(up.a, gate.a, gate.m.aux_stride * E, cudaMemcpyDeviceToDevice), "same up a");
        compare_gu("same-up-gate");
        std::printf("samples constx %.5f %.5f, onesx %.5f %.5f\n", __bfloat162float(h_ref[0]), __bfloat162float(h_fp4[0]),
                    __bfloat162float(h_ref[1]), __bfloat162float(h_fp4[1]));
        return 2;
    }

    grouped_down(down.m, D, I, act_ref, A, E, order, counts, offsets, tile_off, y_ref, s);
    quantize_act_nvfp4(act_ref, A, I, qcodes, qscales, qrow, s);
    grouped_down_fp4(down.m, D, I, qcodes, qscales, qrow, A, E, order, counts, offsets, tile_off, y_fp4, s);
    ck(cudaStreamSynchronize(s), "down");

    std::vector<__nv_bfloat16> hy_ref(size_t(A) * D), hy_fp4(size_t(A) * D);
    ck(cudaMemcpy(hy_ref.data(), y_ref, hy_ref.size() * 2, cudaMemcpyDeviceToHost), "read y ref");
    ck(cudaMemcpy(hy_fp4.data(), y_fp4, hy_fp4.size() * 2, cudaMemcpyDeviceToHost), "read y fp4");
    const double d_nmse = nmse(hy_fp4, hy_ref, finite);
    std::printf("down NMSE %.6g finite %d\n", d_nmse, int(finite));
    if (!finite || d_nmse > 5e-2) return 3;

    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event"); ck(cudaEventCreate(&e1), "event");
    const int reps = 20;
    cudaEventRecord(e0, s);
    for (int i = 0; i < reps; ++i) {
        quantize_act_nvfp4(x, A, D, qcodes, qscales, qrow, s);
        grouped_gate_up_fp4(gate.m, up.m, I, D, qcodes, qscales, qrow, A, E, 1, order, counts, offsets, tile_off, act_fp4, gc, uc, s);
    }
    cudaEventRecord(e1, s); ck(cudaEventSynchronize(e1), "time");
    float ms = 0.0f; ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    std::printf("gate/up FP4 %.3f ms per call\n", ms / reps);
    std::printf("PASS\n");
    return 0;
}
