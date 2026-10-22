// V codec reconstruction through the real attention kernel: with one cached token the attention output is the decoded V
// row (rotation undone). Compares tq3 / scalar 2-bit / trellis 2-bit V error on Gaussian rows with outliers, and checks
// the TCQ GPU decode against a host decode of the same record.
#include "qwen35_kernels.cuh"
#include "tcq2_tables.inc"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using tqllm_cuda::KvCodec;

static const float CB[256] = TQ_TCQ2_CODEBOOK_V;
static const float S1[128] = TQ_TCQ_SIGNS1;
static const float S2[128] = TQ_TCQ_SIGNS2;

static void host_tcq_decode(const tqllm_cuda::Tcq2V256 & r, float * out) {  // 256 values, original basis
    for (int g = 0; g < 2; ++g) {
        const uint32_t * w = r.w + 9 * g;
        const unsigned short hs = static_cast<unsigned short>(w[8] >> 16);
        __half h;
        std::memcpy(&h, &hs, 2);
        const float norm = __half2float(h);
        float z[128];
        for (int t = 0; t < 128; ++t) {
            const int b = 2 * t;
            const uint64_t win = (uint64_t(w[b / 32 + 1]) << 32) | w[b / 32];
            z[t] = CB[(win >> (b % 32)) & 0xFF] * norm * S2[t];
        }
        for (int len = 1; len < 128; len <<= 1)
            for (int i = 0; i < 128; i += 2 * len)
                for (int j = i; j < i + len; ++j) {
                    const float a = z[j], c = z[j + len];
                    z[j] = a + c, z[j + len] = a - c;
                }
        for (int t = 0; t < 128; ++t) out[128 * g + t] = z[t] * 0.08838834764831845f * S1[t];
    }
}

int main() {
    const int KH = 4, H = 4, D = 256, ROWS = 64;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> kv(2 * KH * D), q(H * D);
    for (auto & x : q) x = __float2bfloat16(nd(rng));
    __nv_bfloat16 * dkv, * dq, * dy;
    void * dkc, * dvc;
    int * dpos;
    cudaMalloc(&dkv, kv.size() * 2), cudaMalloc(&dq, q.size() * 2), cudaMalloc(&dy, H * D * 2);
    cudaMalloc(&dkc, KH * 512), cudaMalloc(&dvc, KH * 512), cudaMalloc(&dpos, 8);
    const int pos[2] = {0, 1};
    cudaMemcpy(dpos, pos, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    const KvCodec vs[3] = {KvCodec::Tq3, KvCodec::Tq2s, KvCodec::Tcq2};
    const char * names[3] = {"tq3", "tq2s", "tcq2"};
    double err[3] = {}, ref = 0, dec_diff = 0;
    std::vector<__nv_bfloat16> y(H * D);
    for (int r = 0; r < ROWS; ++r) {
        for (int i = 0; i < 2 * KH * D; ++i) {
            float x = nd(rng) * (1.f + 0.5f * (i % 64 == 0));  // a few larger channels
            if (i % 997 == 0) x *= 6.f;
            kv[i] = __float2bfloat16(x);
        }
        cudaMemcpy(dkv, kv.data(), kv.size() * 2, cudaMemcpyHostToDevice);
        for (int i = KH * D; i < 2 * KH * D; ++i) ref += double(__bfloat162float(kv[i])) * __bfloat162float(kv[i]);
        for (int c = 0; c < 3; ++c) {
            tqllm_cuda::cache_write_kv_side(KvCodec::Tq3, dkv, 2 * KH * D, 1, dkc, dpos, KH, D, nullptr);
            tqllm_cuda::cache_write_kv_side(vs[c], dkv + KH * D, 2 * KH * D, 1, dvc, dpos, KH, D, nullptr);
            tqllm_cuda::attention_kv_split(KvCodec::Tq3, vs[c], dq, dkc, dvc, dy, nullptr, dpos + 1, 1, H, KH, 0.0625f, 1, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("CUDA error %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
            cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost);
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    const double e = double(__bfloat162float(y[h * D + d])) - __bfloat162float(kv[KH * D + h * D + d]);
                    err[c] += e * e;
                }
            if (vs[c] == KvCodec::Tcq2) {
                std::vector<tqllm_cuda::Tcq2V256> rec(KH);
                cudaMemcpy(rec.data(), dvc, KH * sizeof(tqllm_cuda::Tcq2V256), cudaMemcpyDeviceToHost);
                float host[256];
                for (int h = 0; h < H; ++h) {
                    host_tcq_decode(rec[h], host);
                    for (int d = 0; d < D; ++d) dec_diff = std::fmax(dec_diff, std::fabs(host[d] - __bfloat162float(y[h * D + d])));
                }
            }
        }
    }
    for (int c = 0; c < 3; ++c) std::printf("V %-5s relative MSE %.5f\n", names[c], err[c] / ref);
    std::printf("tcq2 GPU vs host decode max abs diff %.4f\n", dec_diff);
    const bool ok = err[2] < err[1] && err[2] / ref < 0.1 && dec_diff < 0.08;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
