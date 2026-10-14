#include "../src/formats/weight_format.hpp"
#include "../src/gguf.hpp"
#include "formats/gsq_q2.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

using namespace tqllm;
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static void require(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
template<class T> struct Device {
    T * p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & x) { check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) { std::vector<T> x(n); check(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x; }
};
static void run(int cols, int n, bool per_expert) {
    const int experts = 4, rows = 19, topk = 3;
    std::mt19937 rng(cols + n);
    const size_t weights = size_t(experts) * rows * cols;
    std::vector<char> raw(weights / 64 * 18);
    std::vector<float> decoded(weights);
    const float scales[] = {0.f, .5f, -1.f, .0625f};
    for (size_t b = 0; b < weights / 64; ++b) {
        const float d = scales[b % 4]; const auto half = __float2half_rn(d);
        std::memcpy(raw.data() + b * 18, &half, 2);
        for (int j = 0; j < 16; ++j) {
            const unsigned byte = rng() & 255;
            raw[b * 18 + 2 + j] = char(byte);
            for (int t = 0; t < 4; ++t) decoded[b * 64 + j * 4 + t] = d * (int((byte >> (2 * t)) & 3) - 1);
        }
    }
    Device<char> dw(raw.size()); dw.put(raw);
    const auto * format = weight_format(GGML_GSQ_Q2_0);
    require(format && format->has_expert_matvec() && !weight_format(42), "GSQ format not isolated from type42");
    Mat m{dw.p, GGML_GSQ_Q2_0, rows * experts, cols, nullptr, format};
    Device<__half> dense(weights); format->dequant_h(m, dense.p, nullptr);
    auto actual = dense.get(weights);
    for (size_t j = 0; j < weights; ++j) require(__half2float(actual[j]) == decoded[j], "Q2 decode differs from code/scale oracle");
    std::vector<int> ids(n * topk);
    for (int j = 0; j < n * topk; ++j) ids[j] = (j * 3 + 1) % experts;
    Device<int> di(ids.size()); di.put(ids);
    std::vector<__half> x(size_t(cols) * (per_expert ? n * topk : n));
    for (size_t j = 0; j < x.size(); ++j) x[j] = __float2half_rn(.07f * std::sin(float(j * 7)));
    Device<__half> dx(x.size()); dx.put(x);
    Device<__nv_bfloat16> dy(size_t(rows) * ids.size());
    format->expert_matvec(m, dx.p, dy.p, rows, di.p, n, topk, per_expert, nullptr);
    auto y = dy.get(size_t(rows) * ids.size());
    for (size_t a = 0; a < ids.size(); ++a) for (int r = 0; r < rows; ++r) {
        double expect = 0;
        for (int j = 0; j < cols; ++j)
            expect += double(decoded[(size_t(ids[a]) * rows + r) * cols + j]) * __half2float(x[(per_expert ? a : a / topk) * cols + j]);
        const float got = __bfloat162float(y[a * rows + r]);
        require(std::abs(got - expect) <= .005 * std::abs(expect) + 2e-5, "Q2 grouped dot differs from independent double oracle");
    }
    Device<float> logits(size_t(m.rows) * n);
    format->matvec_f32(m, dx.p, logits.p, n, nullptr);
    auto f = logits.get(size_t(m.rows) * n);
    for (int a = 0; a < n; ++a) for (int r = 0; r < m.rows; ++r) {
        double expect = 0;
        for (int j = 0; j < cols; ++j) expect += double(decoded[size_t(r) * cols + j]) * __half2float(x[size_t(a) * cols + j]);
        require(std::abs(f[size_t(a) * m.rows + r] - expect) <= 2e-6 * std::abs(expect) + 2e-6, "Q2 F32 dot differs from oracle");
    }
    cudaGraph_t graph{}; cudaGraphExec_t exec{}; cudaStream_t stream{};
    check(cudaStreamCreate(&stream)); check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    format->expert_matvec(m, dx.p, dy.p, rows, di.p, n, topk, per_expert, stream);
    check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream));
    const auto replay = dy.get(y.size()); require(std::memcmp(y.data(), replay.data(), y.size() * sizeof(y[0])) == 0, "Q2 graph replay differs");
    for (auto & id : ids) id = (id + 1) % experts;
    di.put(ids);
    format->expert_matvec(m, dx.p, dy.p, rows, di.p, n, topk, per_expert, stream);
    check(cudaStreamSynchronize(stream)); const auto changed = dy.get(y.size());
    check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream));
    const auto changed_replay = dy.get(y.size());
    require(std::memcmp(changed.data(), changed_replay.data(), changed.size() * sizeof(y[0])) == 0, "Q2 graph ignored changed device IDs");
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    ids[0] = experts; di.put(ids);
    format->expert_matvec(m, dx.p, dy.p, rows, di.p, n, topk, per_expert, nullptr);
    y = dy.get(y.size()); require(std::isnan(__bfloat162float(y[0])), "invalid device expert ID did not fail closed");
    bool rejected = false;
    try { tqllm_cuda::gsq_q2_matvec_f32(dw.p, dx.p, logits.p, rows, cols, 65536, nullptr); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "Q2 grid overflow accepted");
}
int main() {
    try {
        for (int k : {64, 640, 2560}) for (int n : {1, 8, 12}) for (bool per : {false, true}) run(k, n, per);
        std::puts("GSQ Q2_0: 18 code/scale/oracle/grouped/F32/graph/bounds scenarios PASS"); return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
