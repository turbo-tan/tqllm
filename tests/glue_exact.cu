// Decode glue kernels against verbatim copies of their 3bd32bc5 versions (the reference below): the hyper-connection norm
// with its BF16 image (hc_norm_bf16, row kept in registers now) and the segmented router GEMV (fn_gemv_bf16_segments, column
// loop unrolled now) must give the same bits at the Flash-Next decode / verify shapes (1-8 rows).
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
namespace ref {
using tqllm_cuda::GemvSegs;
__global__ void norm_bf16_k(const float * x, const float * gamma, float * out, __nv_bfloat16 * outb, int h, int c, float eps) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const float * src = x + int64_t(row) * h;
    float sum = 0.f;
    for (int j = tid; j < h; j += blockDim.x) sum += src[j] * src[j];
    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / h + eps);
    for (int j = tid; j < h; j += blockDim.x) {
        const float v = __fmul_rn(__fmul_rn(src[j], inv), gamma[(row % c) * h + j]);
        out[int64_t(row) * h + j] = v; outb[int64_t(row) * h + j] = __float2bfloat16_rn(v);
    }
}
template<int R, int T> __global__ void fn_gemv_bf16_seg_rows_k(GemvSegs g, const float * x, int cols) {
    int seg = 0, base = 0;
    for (int s = 0; s < g.n; ++s) {  // g.start has n + 1 entries, so the last segment maps like the others
        const int nb = (g.start[s + 1] - g.start[s] + R - 1) / R;
        if (int(blockIdx.x) < base + nb) { seg = s; break; }
        base += nb;
    }
    const int rows = g.start[seg + 1] - g.start[seg], row0 = (int(blockIdx.x) - base) * R;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    float sum[R][T];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int t = 0; t < T; ++t) sum[r][t] = 0.f;
    for (int v = threadIdx.x; v < cols / 8; v += blockDim.x) {
        float xs[T][8];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xa = reinterpret_cast<const float4 *>(x + int64_t(t) * cols)[2 * v], xb = reinterpret_cast<const float4 *>(x + int64_t(t) * cols)[2 * v + 1];
            xs[t][0] = xa.x; xs[t][1] = xa.y; xs[t][2] = xa.z; xs[t][3] = xa.w; xs[t][4] = xb.x; xs[t][5] = xb.y; xs[t][6] = xb.z; xs[t][7] = xb.w;
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 packed = reinterpret_cast<const uint4 *>(g.w[seg] + int64_t(min(row0 + r, rows - 1)) * cols)[v];
            const __nv_bfloat16 * wv = reinterpret_cast<const __nv_bfloat16 *>(&packed);
            #pragma unroll
            for (int t = 0; t < T; ++t)
                #pragma unroll
                for (int k = 0; k < 8; ++k) sum[r][t] += __bfloat162float(wv[k]) * __bfloat162float(__float2bfloat16_rn(xs[t][k]));
        }
    }
    __shared__ float part[R * T][8];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            float s = sum[r][t];
            for (int d = 16; d; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
            if (!lane) part[r * T + t][warp] = s;
        }
    __syncthreads();
    for (int i = threadIdx.x; i < R * T; i += blockDim.x) {
        const int r = i / T, t = i % T;
        if (row0 + r >= rows) continue;
        float s = 0.f;
        for (int k = 0; k < int(blockDim.x / 32); ++k) s += part[i][k];
        g.y[seg][int64_t(t) * g.ld[seg] + row0 + r] = s;
    }
}
static int rows_per_block(int blocks_at_one_row, int tokens) {  // gemv_rows_per_block of 3bd32bc5
    const int need = tokens <= 2 ? 128 : 24;
    int r = 8;
    while (r > 1 && (blocks_at_one_row + r - 1) / r < need) r /= 2;
    return r;
}
template<int R> static void seg(int tokens, int blocks, int thr, const GemvSegs & g, const float * x, int cols) {
    switch (tokens) {
    case 1: fn_gemv_bf16_seg_rows_k<R, 1><<<blocks, thr>>>(g, x, cols); break; case 2: fn_gemv_bf16_seg_rows_k<R, 2><<<blocks, thr>>>(g, x, cols); break;
    case 3: fn_gemv_bf16_seg_rows_k<R, 3><<<blocks, thr>>>(g, x, cols); break; case 4: fn_gemv_bf16_seg_rows_k<R, 4><<<blocks, thr>>>(g, x, cols); break;
    case 5: fn_gemv_bf16_seg_rows_k<R, 5><<<blocks, thr>>>(g, x, cols); break; case 6: fn_gemv_bf16_seg_rows_k<R, 6><<<blocks, thr>>>(g, x, cols); break;
    case 7: fn_gemv_bf16_seg_rows_k<R, 7><<<blocks, thr>>>(g, x, cols); break; default: fn_gemv_bf16_seg_rows_k<R, 8><<<blocks, thr>>>(g, x, cols); break;
    }
}
}  // namespace ref
static bool equal(const float * a, const float * b, size_t n) {
    std::vector<float> x(n), y(n);
    cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost); cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost);
    return std::memcmp(x.data(), y.data(), n * 4) == 0;
}
int main() {
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 1.f);
    auto upload = [&](size_t n, float scale) { std::vector<float> h(n); for (auto & v : h) v = scale * nd(rng); float * d; cudaMalloc(&d, n * 4); cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice); return d; };
    int bad = 0, cases = 0;
    for (int h : {2560, 1024, 4096, 5000}) for (int n = 1; n <= 8; ++n) {  // hyper-connection norm: 4 streams per row
        const int c = 4; const size_t count = size_t(n) * c * h;
        float * x = upload(count, 3.f), * gamma = upload(size_t(c) * h, 1.f), * a, * b; __nv_bfloat16 * ab, * bb;
        cudaMalloc(&a, count * 4); cudaMalloc(&b, count * 4); cudaMalloc(&ab, count * 2); cudaMalloc(&bb, count * 2);
        ref::norm_bf16_k<<<c * n, 256>>>(x, gamma, a, ab, h, c, 1e-6f);
        tqllm_cuda::hc_norm_bf16(x, gamma, b, bb, h, c, n, 1e-6f, nullptr);
        cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> ha(count), hb(count);
        cudaMemcpy(ha.data(), ab, count * 2, cudaMemcpyDeviceToHost); cudaMemcpy(hb.data(), bb, count * 2, cudaMemcpyDeviceToHost);
        const bool ok = equal(a, b, count) && std::memcmp(ha.data(), hb.data(), count * 2) == 0;
        if (!ok) { ++bad; std::printf("FAIL hc norm h %d rows %d\n", h, n); }
        ++cases; cudaFree(x); cudaFree(gamma); cudaFree(a); cudaFree(b); cudaFree(ab); cudaFree(bb);
    }
    for (int h : {2560, 4096}) for (int n = 1; n <= 8; ++n) {  // hc_combine + hc_norm_bf16 vs the fused launch
        const int c = 4; const size_t count = size_t(n) * c * h;
        float * res = upload(count, 3.f), * blk = upload(size_t(n) * h, 2.f), * inj = upload(size_t(n) * c, 2.f), * gamma = upload(size_t(c) * h, 1.f);
        float * ca, * cb, * na, * nb; __nv_bfloat16 * ba, * bb;
        cudaMalloc(&ca, count * 4); cudaMalloc(&cb, count * 4); cudaMalloc(&na, count * 4); cudaMalloc(&nb, count * 4);
        cudaMalloc(&ba, count * 2); cudaMalloc(&bb, count * 2);
        tqllm_cuda::hc_combine(res, blk, inj, ca, h, c, n, nullptr);
        tqllm_cuda::hc_norm_bf16(ca, gamma, na, ba, h, c, n, 1e-6f, nullptr);
        tqllm_cuda::hc_combine_norm_bf16(res, blk, inj, gamma, cb, nb, bb, h, c, n, 1e-6f, nullptr);
        cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> ha(count), hb(count);
        cudaMemcpy(ha.data(), ba, count * 2, cudaMemcpyDeviceToHost); cudaMemcpy(hb.data(), bb, count * 2, cudaMemcpyDeviceToHost);
        if (!equal(ca, cb, count) || !equal(na, nb, count) || std::memcmp(ha.data(), hb.data(), count * 2)) { ++bad; std::printf("FAIL combine+norm h %d rows %d\n", h, n); }
        ++cases; for (void * p : {(void *) res, (void *) blk, (void *) inj, (void *) gamma, (void *) ca, (void *) cb, (void *) na, (void *) nb, (void *) ba, (void *) bb}) cudaFree(p);
    }
    const int cols = 2560;
    for (int segs : {1, 2}) for (int n = 1; n <= 8; ++n) {  // router (+ next layer's router) with the shared gate row each
        const int E = 512; std::vector<int> rows; for (int s = 0; s < segs; ++s) { rows.push_back(E); rows.push_back(1); }
        tqllm_cuda::GemvSegs g; int total = 0; std::vector<__nv_bfloat16 *> w; std::vector<float *> ya, yb;
        float * x = upload(size_t(n) * cols, 1.f);
        for (size_t i = 0; i < rows.size(); ++i) {
            std::vector<__nv_bfloat16> hw(size_t(rows[i]) * cols); for (auto & v : hw) v = __float2bfloat16(0.05f * nd(rng));
            __nv_bfloat16 * d; cudaMalloc(&d, hw.size() * 2); cudaMemcpy(d, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice); w.push_back(d);
            float * p, * q; cudaMalloc(&p, size_t(n) * rows[i] * 4); cudaMalloc(&q, size_t(n) * rows[i] * 4); ya.push_back(p); yb.push_back(q);
            g.w[i] = d; g.ld[i] = rows[i]; g.start[i] = total; total += rows[i]; g.start[i + 1] = total;
        }
        g.n = int(rows.size());
        tqllm_cuda::GemvSegs gr = g; for (size_t i = 0; i < rows.size(); ++i) gr.y[i] = ya[i];
        tqllm_cuda::GemvSegs gn = g; for (size_t i = 0; i < rows.size(); ++i) gn.y[i] = yb[i];
        const int thr = std::min(256, std::max(32, (cols / 8 / 4 + 31) / 32 * 32)), r = ref::rows_per_block(total, n);
        int blocks = 0; for (int s = 0; s < g.n; ++s) blocks += (g.start[s + 1] - g.start[s] + r - 1) / r;
        if (r == 2) ref::seg<2>(n, blocks, thr, gr, x, cols); else if (r == 4) ref::seg<4>(n, blocks, thr, gr, x, cols);
        else if (r == 8) ref::seg<8>(n, blocks, thr, gr, x, cols); else { std::printf("unexpected rows per block %d\n", r); return 1; }
        tqllm_cuda::fn_gemv_bf16_segments(gn, x, cols, n, nullptr);
        cudaDeviceSynchronize();
        for (size_t i = 0; i < rows.size(); ++i) if (!equal(ya[i], yb[i], size_t(n) * rows[i])) { ++bad; std::printf("FAIL router segs %d rows %d segment %zu\n", segs, n, i); }
        ++cases; cudaFree(x); for (auto * p : w) cudaFree(p); for (auto * p : ya) cudaFree(p); for (auto * p : yb) cudaFree(p);
    }
    if (std::getenv("QSA_BENCH")) {  // router launch (512 + 1 rows x 2560) per rows-per-block choice: us per call
        tqllm_cuda::GemvSegs g; const int rows[2] = {512, 1}; int total = 0; float * x = upload(size_t(8) * cols, 1.f);
        for (int i = 0; i < 2; ++i) {
            std::vector<__nv_bfloat16> hw(size_t(rows[i]) * cols); for (auto & v : hw) v = __float2bfloat16(0.05f * nd(rng));
            __nv_bfloat16 * d; cudaMalloc(&d, hw.size() * 2); cudaMemcpy(d, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
            float * y; cudaMalloc(&y, size_t(8) * rows[i] * 4);
            g.w[i] = d; g.y[i] = y; g.ld[i] = rows[i]; g.start[i] = total; total += rows[i]; g.start[i + 1] = total;
        }
        g.n = 2;
        {   // hyper-connection norm at the decode shape (4 x 2560) and the fused combine + norm
            const int h = 2560, c = 4; float * xr = upload(size_t(c) * h, 3.f), * gm = upload(size_t(c) * h, 1.f), * bl = upload(h, 1.f), * ij = upload(c, 1.f), * o1, * o2;
            __nv_bfloat16 * ob; cudaMalloc(&o1, size_t(c) * h * 4); cudaMalloc(&o2, size_t(c) * h * 4); cudaMalloc(&ob, size_t(c) * h * 2);
            for (int m = 0; m < 2; ++m) {
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
                for (int r = 0; r < 200; ++r) {
                    if (m == 0) tqllm_cuda::hc_norm_bf16(xr, gm, o1, ob, h, c, 1, 1e-6f, nullptr);
                    else tqllm_cuda::hc_combine_norm_bf16(xr, bl, ij, gm, o2, o1, ob, h, c, 1, 1e-6f, nullptr);
                }
                cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                std::printf("%s 4 x 2560: %.2f us per call (back to back)\n", m ? "hc combine + norm (BF16 image)" : "hc norm (BF16 image)", 5.f * ms);
            }
        }
        for (int n : {1, 2, 8}) for (int rpb : {0, 1, 2, 4, 8}) {
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            tqllm_cuda::fn_gemv_bf16_segments(g, x, cols, n, nullptr, rpb); cudaEventRecord(e0);
            for (int r = 0; r < 200; ++r) tqllm_cuda::fn_gemv_bf16_segments(g, x, cols, n, nullptr, rpb);
            cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            std::printf("router rows %d rows-per-block %d (0 = auto): %.1f us\n", n, rpb, 5.f * ms);
        }
    }
    if (cudaGetLastError() != cudaSuccess) { std::puts("glue_exact: CUDA error"); return 1; }
    std::printf("glue_exact: %d cases, %d differ\n%s\n", cases, bad, bad ? "GLUE_EXACT FAIL" : "GLUE_EXACT PASS");
    return bad ? 1 : 0;
}
