// Weight-format contract: every registered GGUF weight format decodes on the GPU (dequant_h) to the CPU reference
// (dequantize_host_rows; in the format's own basis: TQ3 rotated, F32 stored as BF16) within one FP16 ulp, or within FP16
// precision of its 32-value block's largest value (formats stored as code x scale + min lose ulps near zero), and its matvec agrees with a double dot over the reference weights. Random
// blocks: pass A keeps every byte below 0x3C (all FP16 / BF16 / FP32 scale fields finite and small), pass B uses full-range
// bytes (code bits everywhere; blocks whose values leave the FP16 range are skipped). A registered format with no
// CPU reference is a contract gap and fails.
#include "formats/weight/weight_format.hpp"
#include "formats/weight/dequant.hpp"
#include "gguf.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static int ulps(__half a, __half b) {
    uint16_t x, y; std::memcpy(&x, &a, 2); std::memcpy(&y, &b, 2);
    auto ord = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7fff) : int(v); };
    return std::abs(ord(x) - ord(y));
}

int main() {
    std::mt19937 rng(11);
    const int rows = 64, cols = 1024;
    int tested = 0, fails = 0;
    for (uint32_t type = 0; type < 64; ++type) {
        const auto * f = tqllm::weight_format(type);
        if (!f) continue;
        const uint64_t bytes = tqllm::gguf_tensor_bytes(type, {uint64_t(cols), uint64_t(rows)}, "w");
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<char> raw(bytes);
            std::uniform_int_distribution<int> b(0, pass ? 255 : 0x3B);
            for (auto & c : raw) c = char(b(rng));
            std::vector<float> ref(size_t(rows) * cols);
            try {
                tqllm::dequantize_host_rows(type, raw.data(), ref.data(), rows, cols);
            } catch (const std::exception & e) {
                std::printf("FAIL %-10s type %2u: no CPU reference (%s)\n", f->name(), type, e.what());
                ++fails; break;
            }
            std::vector<void *> keep;
            tqllm::Upload up = [&](const void * h, size_t n) {
                void * p; check(cudaMalloc(&p, n)); check(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice)); keep.push_back(p); return (const void *) p;
            };
            const tqllm::Mat m = f->load(raw, rows, cols, up, nullptr);
            __half * dh; check(cudaMalloc(&dh, ref.size() * 2));
            m.fmt->dequant_h(m, dh, nullptr);
            if (type == 0)  // F32 matrices are stored as BF16
                for (auto & v : ref) { uint32_t u; std::memcpy(&u, &v, 4); u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u; std::memcpy(&v, &u, 4); }
            if (f->basis() == tqllm::Basis::Tq3Rot32)
                for (size_t b = 0; b + 32 <= ref.size(); b += 32) tqllm::tq3_rotate_block(&ref[b]);
            std::vector<__half> got(ref.size());
            check(cudaMemcpy(got.data(), dh, got.size() * 2, cudaMemcpyDeviceToHost));
            size_t compared = 0, bad = 0; int worst = 0;
            for (size_t b = 0; b + 32 <= ref.size(); b += 32) {
                float amax = 0.f; bool finite = true;
                for (size_t i = b; i < b + 32; ++i) { finite &= std::isfinite(ref[i]); amax = std::max(amax, std::fabs(ref[i])); }
                if (!finite || amax > 1024.f) continue;  // FP16 scale x code products could overflow
                for (size_t i = b; i < b + 32; ++i) {
                    const int u = ulps(got[i], __float2half(ref[i]));
                    const float diff = std::fabs(__half2float(got[i]) - ref[i]);
                    ++compared; worst = std::max(worst, u); bad += u > 1 && diff > amax * (1.f / 1024.f);
                }
            }
            // matvec (one column) against a double dot over the reference weights, only where every weight is in FP16 range
            std::vector<__half> x(cols);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto & v : x) v = __float2half(nd(rng));
            __half * dx; __nv_bfloat16 * dy;
            check(cudaMalloc(&dx, cols * 2)); check(cudaMalloc(&dy, rows * 2));
            check(cudaMemcpy(dx, x.data(), cols * 2, cudaMemcpyHostToDevice));
            m.fmt->matvec(m, dx, dy, 1, nullptr);
            std::vector<__nv_bfloat16> y(rows);
            check(cudaMemcpy(y.data(), dy, rows * 2, cudaMemcpyDeviceToHost));
            double se = 0, sr = 0; int rows_used = 0;
            for (int r = 0; r < rows; ++r) {
                double dot = 0; bool ok = true;
                for (int k = 0; k < cols; ++k) {
                    const float w = ref[size_t(r) * cols + k];
                    if (!std::isfinite(w) || std::fabs(w) > 65504.f) { ok = false; break; }
                    dot += double(w) * double(__half2float(x[k]));
                }
                if (!ok || !std::isfinite(dot) || std::fabs(dot) > 1e30) continue;
                const double e = double(__bfloat162float(y[r])) - dot;
                se += e * e; sr += dot * dot; ++rows_used;
            }
            const double rel = rows_used && sr > 0 ? std::sqrt(se / sr) : 0.0;
            const bool ok = bad == 0 && (pass || rel <= 2e-2);  // matvec checked on the finite pass
            std::printf("%s %-10s type %2u pass %c: dequant %zu values, %zu > 1 ulp (worst %d); matvec rel-rms %.2e over %d rows\n",
                        ok ? "ok  " : "FAIL", f->name(), type, pass ? 'B' : 'A', compared, bad, worst, rel, rows_used);
            fails += !ok;
            cudaFree(dh); cudaFree(dx); cudaFree(dy);
            for (void * p : keep) cudaFree(p);
        }
        ++tested;
    }
    std::printf("FORMAT_CONTRACT %s (%d formats)\n", fails ? "FAIL" : "PASS", tested);
    return fails ? 1 : 0;
}
