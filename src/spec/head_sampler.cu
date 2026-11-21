// Exact sampling through the screened LM head: kernels and rounds (head_sampler.hpp).
#include "head_sampler.hpp"
#include "dev_alloc.cuh"
#include "debug.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tqllm {
namespace {

constexpr int kBins = 512;          // round histogram of (ref - upper bound): 1/16 nat per bin, the last bin open-ended
constexpr float kBinScale = 16.f;
constexpr int kChunk = 512;         // rows per block of the row sweeps
constexpr int kSweep = 256;         // threads per block of the row sweeps
constexpr int kW = HeadSampler::kCap + HeadSampler::kMaxExtra;  // id slots per request

struct Req {
    int col, gumbel, ex_n, bsel;
    float t, inv_t, ref, theta;
    unsigned long long key;
};

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("head sampler: ") + cudaGetErrorString(e));
}

__device__ __forceinline__ float half_width(const __half * sig, const float * xg, uint32_t r, float a) {
    if (!sig) return 0.f;  // exact first pass
    float v = 0.f;
#pragma unroll
    for (int g = 0; g < HeadSampler::kSegs; ++g) {
        const float m = __half2float(sig[int64_t(r) * HeadSampler::kSegs + g]);
        v += m * m * xg[g];
    }
    return HeadSampler::kSigmas * sqrtf(v) + 1e-3f + 1e-4f * fabsf(a);  // + FP16 dequant / accumulation slack
}

__device__ __forceinline__ float gumbel_f(unsigned long long key, uint32_t r) {
    const float u = (float(gumbel_bits(key, r)) + 0.5f) * kGumbelUnit;
    return -logf(-logf(u));
}

// The exact int8 re-score of one row: the arithmetic of i8h_rescore_argmax_k (quant.cu), so both give the same bits.
__device__ __forceinline__ float i8h_dot(const int8_t * w8, const __half * sc, const __half * xr, uint32_t row, int k, int lane) {
    const int8_t * wr = w8 + int64_t(row) * k;
    const __half * sr = sc + int64_t(row) * (k / 16);
    float acc = 0.f;
    for (int i = lane * 8; i < k; i += 256) {
        const uint2 wq = __ldg(reinterpret_cast<const uint2 *>(wr + i));
        const uint4 xq = __ldg(reinterpret_cast<const uint4 *>(xr + i));
        const __half s = sr[i / 16];
        const int8_t * wb = reinterpret_cast<const int8_t *>(&wq);
        const __half * xv = reinterpret_cast<const __half *>(&xq);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc += __half2float(__hmul(__int2half_rn(wb[e]), s)) * __half2float(xv[e]);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

// block reduction of (max, sum, sum) over kSweep threads; result in thread 0
__device__ __forceinline__ void reduce3(float & mx, float & s1, float & s2) {
    __shared__ float r[3][kSweep / 32];
    for (int o = 16; o; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
        s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    }
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (!l) r[0][w] = mx, r[1][w] = s1, r[2][w] = s2;
    __syncthreads();
    if (!threadIdx.x) {
        for (int i = 1; i < kSweep / 32; ++i) mx = fmaxf(mx, r[0][i]), s1 += r[1][i], s2 += r[2][i];
    }
}

// sig[r][s] = (largest 4-bit scale of the segment's 32-weight groups) / sqrt(12), rounded up
__global__ void sig_build_k(const __half2 * sm, int rows, int ng, int per, __half * sig) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    for (int g = 0; g < HeadSampler::kSegs; ++g) {
        float m = 0.f;
        for (int j = g * per; j < min(ng, (g + 1) * per); ++j) m = fmaxf(m, fabsf(__low2float(sm[int64_t(r) * ng + j])));
        sig[int64_t(r) * HeadSampler::kSegs + g] = __float2half_ru(m * 0.28867514f * 1.001f);
    }
}

// k3: sig[r][s] = (largest |d * scale| of the segment's 16-weight sub-blocks) / sqrt(12), rounded up
__global__ void sig_build_k3_k(const uint8_t * kr, int rows, int k, int seg_len, __half * sig) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    for (int g = 0; g < HeadSampler::kSegs; ++g) {
        float m = 0.f;
        for (int t = g * seg_len / 16; t < min(k, (g + 1) * seg_len) / 16; ++t) {
            const uint8_t * rc = kr + (int64_t(r) * (k / 256) + t / 16) * 20;
            m = fmaxf(m, fabsf(__half2float(*reinterpret_cast<const __half *>(rc + 16)) * float(int8_t(rc[t % 16]))));
        }
        sig[int64_t(r) * HeadSampler::kSegs + g] = __float2half_ru(m * 0.28867514f * 1.001f);
    }
}

// xg[c][s] = |x_c|^2 over segment s (one warp per segment)
__global__ void xg_k(const __half * xh, int k, int seg_len, float * xg) {
    const int c = blockIdx.x, g = threadIdx.x / 32, l = threadIdx.x % 32;
    const __half * x = xh + int64_t(c) * k;
    float s = 0.f;
    for (int i = g * seg_len + l; i < min(k, (g + 1) * seg_len); i += 32) {
        const float v = __half2float(x[i]);
        s += v * v;
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (!l) xg[c * HeadSampler::kSegs + g] = s;
}

// Screen statistics of columns c0 .. c0 + n over the rows outside each column's top-16: max upper bound, softmax mass
// bounds relative to the top-16's exact maximum. part [block][n][3].
__global__ void __launch_bounds__(kSweep) stats_k(const float * a, int rows, int c0, int n, const __half * sig, const float * xg,
                                                  const uint32_t * ids16, const float * ex16, const float * inv_t, float * part) {
    __shared__ uint32_t ids[16];
    __shared__ float ref_s;
    const int r0 = blockIdx.x * kChunk;
    for (int j = 0; j < n; ++j) {
        const int c = c0 + j;
        __syncthreads();
        if (threadIdx.x < 16) ids[threadIdx.x] = ids16[c * 16 + threadIdx.x];
        if (!threadIdx.x) {
            float m = ex16[c * 16];
            for (int i = 1; i < 16; ++i) m = fmaxf(m, ex16[c * 16 + i]);
            ref_s = m;
        }
        __syncthreads();
        const float ref = ref_s, it = inv_t[c];
        const float * ac = a + int64_t(c) * rows;
        const float * xc = xg + c * HeadSampler::kSegs;
        float mx = -INFINITY, lo = 0.f, hi = 0.f;
        for (int r = r0 + threadIdx.x; r < min(rows, r0 + kChunk); r += kSweep) {
            bool top = false;
#pragma unroll
            for (int i = 0; i < 16; ++i) top |= ids[i] == uint32_t(r);
            if (top) continue;
            const float av = ac[r], e = half_width(sig, xc, uint32_t(r), av);
            mx = fmaxf(mx, av + e);
            hi += __expf((av + e - ref) * it);
            lo += __expf((av - e - ref) * it);
        }
        reduce3(mx, lo, hi);
        if (!threadIdx.x) {
            float * p = part + (int64_t(blockIdx.x) * n + j) * 3;
            p[0] = mx, p[1] = lo, p[2] = hi;
        }
    }
}

__global__ void __launch_bounds__(kSweep) stats_fin_k(const float * part, int nblk, int c0, int n, const float * ex16, float * stat) {
    const int j = blockIdx.x, c = c0 + j;
    float mx = -INFINITY, lo = 0.f, hi = 0.f;
    for (int b = threadIdx.x; b < nblk; b += kSweep) {
        const float * p = part + (int64_t(b) * n + j) * 3;
        mx = fmaxf(mx, p[0]), lo += p[1], hi += p[2];
    }
    reduce3(mx, lo, hi);
    if (!threadIdx.x) {
        float m = ex16[c * 16];
        for (int i = 1; i < 16; ++i) m = fmaxf(m, ex16[c * 16 + i]);
        stat[c * 4 + 0] = mx, stat[c * 4 + 1] = m, stat[c * 4 + 2] = lo, stat[c * 4 + 3] = hi;
    }
}

__device__ __forceinline__ bool excluded(const uint32_t * ex, int n, uint32_t r) {
    for (int i = 0; i < n; ++i) if (ex[i] == r) return true;
    return false;
}

// Round pass A (grid: row blocks x requests): nucleus requests histogram their upper bounds below ref, Gumbel requests
// take the largest lower Gumbel score. Rows in the request's extra list (exact anyway) are left out.
__global__ void __launch_bounds__(kSweep) round_a_k(const float * a, int rows, const __half * sig, const float * xg, const Req * req,
                                                    const uint32_t * ids, int * hist, float * part) {
    __shared__ int h[kBins];
    __shared__ uint32_t ex[HeadSampler::kMaxExtra];
    const int q = blockIdx.y;
    const Req rq = req[q];
    for (int i = threadIdx.x; i < kBins; i += kSweep) h[i] = 0;
    for (int i = threadIdx.x; i < rq.ex_n; i += kSweep) ex[i] = ids[q * kW + HeadSampler::kCap + i];
    __syncthreads();
    const float * ac = a + int64_t(rq.col) * rows;
    const float * xc = xg + rq.col * HeadSampler::kSegs;
    const int r0 = blockIdx.x * kChunk;
    float mx = -INFINITY, d1 = 0.f, d2 = 0.f;
    for (int r = r0 + threadIdx.x; r < min(rows, r0 + kChunk); r += kSweep) {
        if (excluded(ex, rq.ex_n, uint32_t(r))) continue;
        const float av = ac[r], e = half_width(sig, xc, uint32_t(r), av);
        if (rq.gumbel) {
            mx = fmaxf(mx, av + rq.t * gumbel_f(rq.key, uint32_t(r)) - e);
        } else {
            const float d = (rq.ref - (av + e)) * kBinScale;
            const int b = d <= 0.f ? 0 : d >= float(kBins - 1) ? kBins - 1 : int(d);
            atomicAdd(&h[b], 1);
        }
    }
    if (rq.gumbel) {
        reduce3(mx, d1, d2);
        if (!threadIdx.x) part[(int64_t(blockIdx.x) * HeadSampler::kMaxReq + q) * 3] = mx;
    } else {
        __syncthreads();
        for (int i = threadIdx.x; i < kBins; i += kSweep)
            if (h[i]) atomicAdd(&hist[q * kBins + i], h[i]);
    }
}

// per request: the histogram bins that fit the re-score capacity, or the Gumbel threshold
__global__ void round_pick_k(Req * req, const int * hist, const float * part, int nblk, int cap) {
    const int q = blockIdx.x;
    Req & rq = req[q];
    if (threadIdx.x) return;
    if (rq.gumbel) {
        float mx = -INFINITY;
        for (int b = 0; b < nblk; ++b) mx = fmaxf(mx, part[(int64_t(b) * HeadSampler::kMaxReq + q) * 3]);
        rq.theta = mx - 2e-3f * (1.f + rq.t);  // the device's FP32 Gumbel noise against the host's FP64 draw
    } else {
        int cum = 0, b = -1;
        for (int i = 0; i < kBins; ++i) {
            if (cum + hist[q * kBins + i] > cap) break;
            cum += hist[q * kBins + i];
            b = i;
        }
        rq.bsel = b;
    }
}

// Round pass B: select the rows to re-score; nucleus requests also bound the rows left out (part [block][kMaxReq][3])
__global__ void __launch_bounds__(kSweep) round_b_k(const float * a, int rows, const __half * sig, const float * xg, const Req * req,
                                                    uint32_t * ids, int * cnt, int cap, float * part) {
    __shared__ uint32_t ex[HeadSampler::kMaxExtra];
    const int q = blockIdx.y;
    const Req rq = req[q];
    for (int i = threadIdx.x; i < rq.ex_n; i += kSweep) ex[i] = ids[q * kW + HeadSampler::kCap + i];
    __syncthreads();
    const float * ac = a + int64_t(rq.col) * rows;
    const float * xc = xg + rq.col * HeadSampler::kSegs;
    const int r0 = blockIdx.x * kChunk;
    float mx = -INFINITY, lo = 0.f, hi = 0.f;
    for (int r = r0 + threadIdx.x; r < min(rows, r0 + kChunk); r += kSweep) {
        if (excluded(ex, rq.ex_n, uint32_t(r))) continue;
        const float av = ac[r], e = half_width(sig, xc, uint32_t(r), av);
        bool sel;
        if (rq.gumbel) {
            sel = av + rq.t * gumbel_f(rq.key, uint32_t(r)) + e >= rq.theta;
        } else {
            const float d = (rq.ref - (av + e)) * kBinScale;
            const int b = d <= 0.f ? 0 : d >= float(kBins - 1) ? kBins - 1 : int(d);
            sel = b <= rq.bsel;
        }
        if (sel) {
            const int i = atomicAdd(&cnt[q], 1);
            if (i < cap) ids[q * kW + i] = uint32_t(r);
        } else if (!rq.gumbel) {
            mx = fmaxf(mx, av + e);
            hi += __expf((av + e - rq.ref) * rq.inv_t);
            lo += __expf((av - e - rq.ref) * rq.inv_t);
        }
    }
    reduce3(mx, lo, hi);
    if (!threadIdx.x) {
        float * p = part + (int64_t(blockIdx.x) * HeadSampler::kMaxReq + q) * 3;
        p[0] = mx, p[1] = lo, p[2] = hi;
    }
}

__global__ void __launch_bounds__(kSweep) round_tail_k(const float * part, int nblk, float * tail) {
    const int q = blockIdx.x;
    float mx = -INFINITY, lo = 0.f, hi = 0.f;
    for (int b = threadIdx.x; b < nblk; b += kSweep) {
        const float * p = part + (int64_t(b) * HeadSampler::kMaxReq + q) * 3;
        mx = fmaxf(mx, p[0]), lo += p[1], hi += p[2];
    }
    reduce3(mx, lo, hi);
    if (!threadIdx.x) tail[q * 4 + 0] = mx, tail[q * 4 + 1] = 0.f, tail[q * 4 + 2] = lo, tail[q * 4 + 3] = hi;
}

// exact logits of the selected rows and the extra rows: grid (kW / 16, requests), one warp per row
__global__ void __launch_bounds__(512) round_rescore_k(const float * a, int rows, const int8_t * w8, const __half * sc,
                                                       const __half * xh, int k, const Req * req, const uint32_t * ids,
                                                       const int * cnt, int cap, float * out) {
    const int q = blockIdx.y, idx = blockIdx.x * 16 + threadIdx.x / 32, lane = threadIdx.x % 32;
    const Req & rq = req[q];
    const bool ok = idx < HeadSampler::kCap ? idx < min(cnt[q], cap) : idx - HeadSampler::kCap < rq.ex_n;
    if (!ok) return;
    const uint32_t row = ids[q * kW + idx];
    if (!w8) {
        if (!lane) out[q * kW + idx] = a[int64_t(rq.col) * rows + row];
        return;
    }
    const float v = i8h_dot(w8, sc, xh + int64_t(rq.col) * k, row, k, lane);
    if (!lane) out[q * kW + idx] = v;
}

// every row of one column, exactly (warp per row)
__global__ void __launch_bounds__(512) full_k(const int8_t * w8, const __half * sc, const __half * xr, int rows, int k, float * out) {
    const int r = blockIdx.x * 16 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    const float v = i8h_dot(w8, sc, xr, uint32_t(r), k, lane);
    if (!lane) out[r] = v;
}

// every row of n columns (warp per row, columns in turn: the row stays in L1)
__global__ void __launch_bounds__(512) full_cols_k(const int8_t * w8, const __half * sc, const __half * xh, int rows, int k, int n,
                                                   float * out) {
    const int r = blockIdx.x * 16 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    for (int c = 0; c < n; ++c) {
        const float v = i8h_dot(w8, sc, xh + int64_t(c) * k, uint32_t(r), k, lane);
        if (!lane) out[int64_t(c) * rows + r] = v;
    }
}

}  // namespace

void HeadSampler::rescore_all(const int8_t * w8, const __half * sc, const __half * xh, int n, float * out, cudaStream_t s) const {
    full_cols_k<<<(rows_ + 15) / 16, 512, 0, s>>>(w8, sc, xh, rows_, k_, n, out);
}

HeadSampler::~HeadSampler() {
    for (void * p : allocs_) tqllm_cuda::dev_free(p);
    if (hbuf_) cudaFreeHost(hbuf_);
}

void * HeadSampler::dalloc(size_t bytes) {
    void * p = nullptr;
    ck(tqllm_cuda::dev_malloc(&p, bytes));
    allocs_.push_back(p);
    dev_bytes_ += bytes;
    return p;
}

void HeadSampler::init(int rows, int k, int max_cols, FirstPass fp, const void * scales, cudaStream_t s) {
    if (k % 256) throw std::invalid_argument("head sampler: k must be a multiple of 256");
    for (void * p : allocs_) tqllm_cuda::dev_free(p);
    allocs_.clear();
    dev_bytes_ = 0;
    if (hbuf_) cudaFreeHost(hbuf_), hbuf_ = nullptr;
    rows_ = rows, k_ = k, max_cols_ = max_cols;
    const int ng = k / 32;
    per_ = (ng + kSegs - 1) / kSegs;
    nblk_ = (rows + kChunk - 1) / kChunk;
    sig_ = nullptr;
    if (fp != FirstPass::Exact && scales) {
        sig_ = static_cast<__half *>(dalloc(size_t(rows) * kSegs * 2));
        if (fp == FirstPass::U4h)
            sig_build_k<<<(rows + 255) / 256, 256, 0, s>>>(static_cast<const __half2 *>(scales), rows, ng, per_, sig_);
        else
            sig_build_k3_k<<<(rows + 255) / 256, 256, 0, s>>>(static_cast<const uint8_t *>(scales), rows, k, per_ * 32, sig_);
        ck(cudaGetLastError());
    }
    xh_ = static_cast<__half *>(dalloc(size_t(max_cols) * k * 2));
    xg_ = static_cast<float *>(dalloc(size_t(max_cols) * kSegs * 4));
    temps_ = static_cast<float *>(dalloc(size_t(max_cols) * 4));
    part_ = static_cast<float *>(dalloc(size_t(nblk_) * std::max(max_cols, kMaxReq) * 3 * 4));
    stat_ = static_cast<float *>(dalloc(size_t(max_cols) * 4 * 4));
    req_ = dalloc(sizeof(Req) * kMaxReq);
    hist_ = static_cast<int *>(dalloc(size_t(kMaxReq) * kBins * 4));
    cnt_ = static_cast<int *>(dalloc(size_t(kMaxReq) * 4));
    ids_ = static_cast<uint32_t *>(dalloc(size_t(kMaxReq) * kW * 4));
    ex_ = static_cast<float *>(dalloc(size_t(kMaxReq) * kW * 4));
    tail_ = static_cast<float *>(dalloc(size_t(kMaxReq) * 4 * 4));
    full_ = static_cast<float *>(dalloc(size_t(rows) * 4));
    const size_t hb = size_t(max_cols) * (16 * 4 + 16 * 4 + 4 * 4 + 4) + sizeof(Req) * kMaxReq + size_t(kMaxReq) * kW * 8 +
                      size_t(kMaxReq) * (4 + 16) + size_t(rows) * 4 + 256;
    ck(cudaHostAlloc(&hbuf_, hb, cudaHostAllocPortable));
    char * h = static_cast<char *>(hbuf_);
    auto take = [&](size_t n) { char * p = h; h += (n + 15) / 16 * 16; return p; };
    h_ids16_ = reinterpret_cast<uint32_t *>(take(size_t(max_cols) * 64));
    h_ex16_ = reinterpret_cast<float *>(take(size_t(max_cols) * 64));
    h_stat_ = reinterpret_cast<float *>(take(size_t(max_cols) * 16));
    h_temps_ = reinterpret_cast<float *>(take(size_t(max_cols) * 4));
    h_req_ = take(sizeof(Req) * kMaxReq);
    h_ids_ = reinterpret_cast<uint32_t *>(take(size_t(kMaxReq) * kW * 4));
    h_ex_ = reinterpret_cast<float *>(take(size_t(kMaxReq) * kW * 4));
    h_cnt_ = reinterpret_cast<int *>(take(size_t(kMaxReq) * 4));
    h_tail_ = reinterpret_cast<float *>(take(size_t(kMaxReq) * 16));
    h_full_ = reinterpret_cast<float *>(take(size_t(rows) * 4));
    if (size_t(h - static_cast<char *>(hbuf_)) > hb + 256) throw std::logic_error("head sampler: host buffer layout");
}

void HeadSampler::save_x(const __half * xh, int n, cudaStream_t s) {
    if (!ready() || n > max_cols_) return;
    ck(cudaMemcpyAsync(xh_, xh, size_t(n) * k_ * 2, cudaMemcpyDeviceToDevice, s));
}

void HeadSampler::queue_stats(const float * a, const uint32_t * ids16, const float * ex16, int c0, int n, const float * temps,
                              cudaStream_t s) {
    if (n <= 0) return;
    if (c0 < 0 || c0 + n > max_cols_) throw std::out_of_range("head sampler: columns");
    for (int j = 0; j < n; ++j) h_temps_[c0 + j] = 1.f / temps[j];
    ck(cudaMemcpyAsync(temps_ + c0, h_temps_ + c0, size_t(n) * 4, cudaMemcpyHostToDevice, s));
    xg_k<<<n, 32 * kSegs, 0, s>>>(xh_ + size_t(c0) * k_, k_, per_ * 32, xg_ + c0 * kSegs);
    stats_k<<<nblk_, kSweep, 0, s>>>(a, rows_, c0, n, sig_, xg_, ids16, ex16, temps_, part_);
    stats_fin_k<<<n, kSweep, 0, s>>>(part_, nblk_, c0, n, ex16, stat_);
    ck(cudaGetLastError());
    ck(cudaMemcpyAsync(h_ids16_ + c0 * 16, ids16 + c0 * 16, size_t(n) * 64, cudaMemcpyDeviceToHost, s));
    ck(cudaMemcpyAsync(h_ex16_ + c0 * 16, ex16 + c0 * 16, size_t(n) * 64, cudaMemcpyDeviceToHost, s));
    ck(cudaMemcpyAsync(h_stat_ + c0 * 4, stat_ + c0 * 4, size_t(n) * 16, cudaMemcpyDeviceToHost, s));
}

struct HeadSampler::State {
    HeadSampleCol c;
    ExactRows e;
    std::vector<std::pair<uint32_t, int>> pc;
    std::vector<uint32_t> extra;  // always exact: the top-16 and the penalty window
    uint64_t key = 0;
    float ref = 0;
    bool gumbel = false, done = false, overflow = false;
    uint32_t tok = 0;
};

namespace {
// device FP32 sums of many terms: widen the mass bounds (relative error far below this)
void widen(ExactRows & e) {
    e.z_lo = e.z_lo * (1.0 - 1e-3);
    e.z_hi = e.z_hi * (1.0 + 1e-3) + 1e-30;
}
}  // namespace

void HeadSampler::round(std::vector<State *> & st, int cap, const SamplingParams & sp, const float * a, const int8_t * w8,
                        const __half * sc, cudaStream_t s) {
    for (size_t b0 = 0; b0 < st.size(); b0 += kMaxReq) {
        const int nr = int(std::min<size_t>(kMaxReq, st.size() - b0));
        Req * hr = static_cast<Req *>(h_req_);
        for (int q = 0; q < nr; ++q) {
            State & x = *st[b0 + size_t(q)];
            hr[q] = Req{x.c.col, x.gumbel ? 1 : 0, int(x.extra.size()), -1, sp.temperature, 1.f / sp.temperature, x.ref, 0.f,
                        static_cast<unsigned long long>(x.key)};
            std::copy(x.extra.begin(), x.extra.end(), h_ids_ + size_t(q) * kW + kCap);
        }
        ck(cudaMemcpyAsync(req_, hr, sizeof(Req) * size_t(nr), cudaMemcpyHostToDevice, s));
        ck(cudaMemcpyAsync(ids_, h_ids_, size_t(nr) * kW * 4, cudaMemcpyHostToDevice, s));
        ck(cudaMemsetAsync(hist_, 0, size_t(nr) * kBins * 4, s));
        ck(cudaMemsetAsync(cnt_, 0, size_t(nr) * 4, s));
        const dim3 g{unsigned(nblk_), unsigned(nr)};
        round_a_k<<<g, kSweep, 0, s>>>(a, rows_, sig_, xg_, static_cast<const Req *>(req_), ids_, hist_, part_);
        round_pick_k<<<nr, 32, 0, s>>>(static_cast<Req *>(req_), hist_, part_, nblk_, cap);
        round_b_k<<<g, kSweep, 0, s>>>(a, rows_, sig_, xg_, static_cast<const Req *>(req_), ids_, cnt_, kCap, part_);
        round_tail_k<<<nr, kSweep, 0, s>>>(part_, nblk_, tail_);
        round_rescore_k<<<dim3(unsigned(kW / 16), unsigned(nr)), 512, 0, s>>>(a, rows_, w8, sc, xh_, k_,
                                                                              static_cast<const Req *>(req_), ids_, cnt_, kCap, ex_);
        ck(cudaGetLastError());
        ck(cudaMemcpyAsync(h_req_, req_, sizeof(Req) * size_t(nr), cudaMemcpyDeviceToHost, s));
        ck(cudaMemcpyAsync(h_cnt_, cnt_, size_t(nr) * 4, cudaMemcpyDeviceToHost, s));
        ck(cudaMemcpyAsync(h_ids_, ids_, size_t(nr) * kW * 4, cudaMemcpyDeviceToHost, s));
        ck(cudaMemcpyAsync(h_ex_, ex_, size_t(nr) * kW * 4, cudaMemcpyDeviceToHost, s));
        ck(cudaMemcpyAsync(h_tail_, tail_, size_t(nr) * 16, cudaMemcpyDeviceToHost, s));
        ck(cudaStreamSynchronize(s));
        for (int q = 0; q < nr; ++q) {
            State & x = *st[b0 + size_t(q)];
            const int n = h_cnt_[q];
            if (n > kCap || (!x.gumbel && hr[q].bsel < 0)) {
                x.overflow = true;
                continue;
            }
            ExactRows e;
            e.rows = rows_;
            const uint32_t * id = h_ids_ + size_t(q) * kW;
            const float * ex = h_ex_ + size_t(q) * kW;
            for (int i = 0; i < n; ++i) e.id.push_back(id[i]), e.lg.push_back(ex[i]);
            for (size_t i = 0; i < x.extra.size(); ++i) e.id.push_back(x.extra[i]), e.lg.push_back(ex[kCap + i]);
            if (x.gumbel) {
                e.gumbel_closed = true;
            } else {
                e.m_out = h_tail_[q * 4 + 0];
                e.z_ref = x.ref;
                e.z_lo = h_tail_[q * 4 + 2];
                e.z_hi = h_tail_[q * 4 + 3];
                widen(e);
            }
            x.e = std::move(e);
            x.done = sample_bounded(x.e, sp, x.pc, x.key, x.tok);
        }
    }
}

void HeadSampler::full_rows(State & x, const SamplingParams & sp, const int8_t * w8, const __half * sc, const float * a,
                            cudaStream_t s) {
    if (w8) {
        full_k<<<(rows_ + 15) / 16, 512, 0, s>>>(w8, sc, xh_ + size_t(x.c.col) * k_, rows_, k_, full_);
        ck(cudaGetLastError());
        ck(cudaMemcpyAsync(h_full_, full_, size_t(rows_) * 4, cudaMemcpyDeviceToHost, s));
    } else {
        ck(cudaMemcpyAsync(h_full_, a + size_t(x.c.col) * rows_, size_t(rows_) * 4, cudaMemcpyDeviceToHost, s));
    }
    ck(cudaStreamSynchronize(s));
    x.tok = sample_reference(h_full_, rows_, sp, x.c.recent, x.key);
    x.done = true;
}

std::vector<uint32_t> HeadSampler::sample(const std::vector<HeadSampleCol> & cols, const SamplingParams & sp, const float * a,
                                          const int8_t * w8, const __half * sc, cudaStream_t s) {
    std::vector<State> st(cols.size());
    std::vector<State *> todo;
    for (size_t i = 0; i < cols.size(); ++i) {
        State & x = st[i];
        x.c = cols[i];
        const int c = x.c.col;
        x.key = gumbel_key(sp.seed, x.c.pos);
        x.pc = penalty_counts(sp, x.c.recent);
        x.gumbel = full_vocab_draw(sp, rows_);
        x.ref = h_stat_[c * 4 + 1];
        for (int j = 0; j < 16; ++j) x.extra.push_back(h_ids16_[c * 16 + j]);
        for (const auto & p : x.pc)
            if (p.first < uint32_t(rows_) && std::find(x.extra.begin(), x.extra.end(), p.first) == x.extra.end())
                x.extra.push_back(p.first);
        ++stats.cols;
        if (!x.gumbel && x.pc.empty()) {  // round 1: the top-16 and the screen statistics
            ExactRows & e = x.e;
            e.rows = rows_;
            for (int j = 0; j < 16; ++j) e.id.push_back(h_ids16_[c * 16 + j]), e.lg.push_back(h_ex16_[c * 16 + j]);
            e.m_out = h_stat_[c * 4 + 0];
            e.z_ref = x.ref;
            e.z_lo = h_stat_[c * 4 + 2];
            e.z_hi = h_stat_[c * 4 + 3];
            widen(e);
            x.done = sample_bounded(e, sp, x.pc, x.key, x.tok);
            if (x.done) ++stats.round1;
        }
        if (!x.done) todo.push_back(&x);
    }
    std::vector<State *> full;
    for (int cap : {64, 1024, kCap}) {  // re-score more rows: the 64, 1024, 8192 largest upper bounds (Gumbel: all candidates)
        if (todo.empty()) break;
        for (State * x : todo) ++(x->gumbel ? stats.gumbel : cap == 64 ? stats.rounds64 : cap == 1024 ? stats.rounds1024 : stats.rounds8192);
        round(todo, cap, sp, a, w8, sc, s);
        std::vector<State *> left;
        for (State * x : todo)
            if (!x->done) (x->gumbel || x->overflow ? full : left).push_back(x);
        todo = std::move(left);
    }
    full.insert(full.end(), todo.begin(), todo.end());
    for (State * x : full) {
        ++stats.full;
        full_rows(*x, sp, w8, sc, a, s);
    }
    static const bool chk = debug_on("head_check");
    if (chk && w8)
        for (const State & x : st) check(x, sp, a, w8, sc, s);
    std::vector<uint32_t> out(cols.size());
    for (size_t i = 0; i < st.size(); ++i) out[i] = st[i].tok;
    return out;
}

void HeadSampler::check(const State & x, const SamplingParams & sp, const float * a, const int8_t * w8, const __half * sc,
                        cudaStream_t s) {
    const int c = x.c.col;
    if (h_sig_.empty() && sig_) {
        h_sig_.resize(size_t(rows_) * kSegs);
        ck(cudaMemcpyAsync(h_sig_.data(), sig_, h_sig_.size() * 2, cudaMemcpyDeviceToHost, s));
    }
    std::vector<float> av(static_cast<size_t>(rows_)), xg(kSegs);
    full_k<<<(rows_ + 15) / 16, 512, 0, s>>>(w8, sc, xh_ + size_t(c) * k_, rows_, k_, full_);
    ck(cudaMemcpyAsync(h_full_, full_, size_t(rows_) * 4, cudaMemcpyDeviceToHost, s));
    ck(cudaMemcpyAsync(av.data(), a + size_t(c) * rows_, size_t(rows_) * 4, cudaMemcpyDeviceToHost, s));
    ck(cudaMemcpyAsync(xg.data(), xg_ + c * kSegs, kSegs * 4, cudaMemcpyDeviceToHost, s));
    ck(cudaStreamSynchronize(s));
    ++stats.chk_cols;
    if (sample_reference(h_full_, rows_, sp, x.c.recent, x.key) != x.tok) ++stats.chk_diff;
    if (!sig_) return;
    for (int r = 0; r < rows_; ++r) {
        double v = 0;
        for (int g = 0; g < kSegs; ++g) {
            const double m = __half2float(h_sig_[size_t(r) * kSegs + g]);
            v += m * m * xg[size_t(g)];
        }
        const double sd = std::sqrt(v), d = std::fabs(double(av[size_t(r)]) - h_full_[r]);
        const double ratio = sd > 0 ? d / sd : (d > 0 ? 1e9 : 0);
        ++stats.chk_rows;
        for (int t = 0; t < 4; ++t) stats.chk_over[t] += ratio > 3.0 + t;
        stats.chk_max = std::max(stats.chk_max, ratio);
    }
}

std::vector<uint32_t> HeadSampler::sample_full(const std::vector<HeadSampleCol> & cols, const SamplingParams & sp, const float * a,
                                               cudaStream_t s) {
    std::vector<uint32_t> out;
    for (const auto & c : cols) {
        ck(cudaMemcpyAsync(h_full_, a + size_t(c.col) * rows_, size_t(rows_) * 4, cudaMemcpyDeviceToHost, s));
        ck(cudaStreamSynchronize(s));
        out.push_back(sample_reference(h_full_, rows_, sp, c.recent, gumbel_key(sp.seed, c.pos)));
    }
    return out;
}

}  // namespace tqllm
