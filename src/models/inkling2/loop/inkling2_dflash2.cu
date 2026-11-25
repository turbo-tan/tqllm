// Inkling-2-Mini DFlash2 drafter kernels (see inkling2_dflash2.cuh)
#include "inkling2_dflash2.cuh"

#include <cmath>
#include <stdexcept>

namespace tqllm_cuda::inkling2 {
namespace {
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }

__global__ void conv_k(const float * x, const __nv_bfloat16 * dyn, const float * base, int m, int H, int group, int side, const float * res, float * y) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(m) * H) return;
    const int r = int(i / H), c = int(i % H), G = H / group, c0 = 2 * side, gi = c / group;
    const float w0 = __fadd_rn(__bfloat162float(dyn[size_t(r) * 4 * G + c0 * G + gi]), base[size_t(c0) * H + c]);
    const float w1 = __fadd_rn(__bfloat162float(dyn[size_t(r) * 4 * G + (c0 + 1) * G + gi]), base[size_t(c0 + 1) * H + c]);
    const float x1 = r > 0 ? x[i - H] : 0.0f;
    float acc = __fadd_rn(0.0f, __fmul_rn(w0, x[i]));
    acc = __fadd_rn(acc, __fmul_rn(w1, x1));
    float out = bf(acc);
    if (res) out = bf(__fadd_rn(res[i], out));
    y[i] = out;
}

__global__ void silu_mul_k(const __nv_bfloat16 * gu, __nv_bfloat16 * act, int m, int I) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(m) * I) return;
    const int r = int(i / I), c = int(i % I);
    const float g = __bfloat162float(gu[size_t(r) * 2 * I + c]), u = __bfloat162float(gu[size_t(r) * 2 * I + I + c]);
    act[i] = __float2bfloat16(__fmul_rn(bf(__fmul_rn(g, 1.0f / (1.0f + expf(-g)))), u));
}

// (value, id) order: larger value first, then the lower id
__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }
constexpr int kTopThreads = 128, kTopChunk = 4096;
// each thread keeps its 16 best (sorted) of the candidates it reads; the block then takes the 16 best heads one by one
template <class Load>
__device__ void block_top16(int n, Load load, int * out_ids, float * out_vals) {
    float v[16]; int id[16];
    for (int j = 0; j < 16; ++j) { v[j] = -INFINITY; id[j] = 0x7fffffff; }
    for (int c = threadIdx.x; c < n; c += kTopThreads) {
        float x; int xi;
        load(c, x, xi);
        if (!better(x, xi, v[15], id[15])) continue;
        int j = 15;
        while (j > 0 && better(x, xi, v[j - 1], id[j - 1])) { v[j] = v[j - 1]; id[j] = id[j - 1]; --j; }
        v[j] = x; id[j] = xi;
    }
    __shared__ float sv[kTopThreads / 32];
    __shared__ int si[kTopThreads / 32], st[kTopThreads / 32];
    __shared__ int winner;
    int head = 0;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int k = 0; k < 16; ++k) {
        float bv = head < 16 ? v[head] : -INFINITY;
        int bi = head < 16 ? id[head] : 0x7fffffff, bt = threadIdx.x;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o), ot = __shfl_xor_sync(0xffffffffu, bt, o);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; bt = ot; }
        }
        if (lane == 0) { sv[warp] = bv; si[warp] = bi; st[warp] = bt; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int w = 0;
            for (int j = 1; j < kTopThreads / 32; ++j) if (better(sv[j], si[j], sv[w], si[w])) w = j;
            out_vals[k] = sv[w]; out_ids[k] = si[w]; winner = st[w];
        }
        __syncthreads();
        if (int(threadIdx.x) == winner) ++head;
        __syncthreads();
    }
}
__global__ void __launch_bounds__(kTopThreads) top16_chunks_k(const float * x, int n, int ld, int chunks, float * pv, int * pi) {
    const int row = blockIdx.y, c0 = blockIdx.x * kTopChunk, cn = min(kTopChunk, n - c0);
    const float * xr = x + size_t(row) * ld + c0;
    const size_t o = (size_t(row) * chunks + blockIdx.x) * 16;
    block_top16(cn, [&](int c, float & v, int & i) { v = xr[c]; i = c0 + c; }, pi + o, pv + o);
}
__global__ void __launch_bounds__(kTopThreads) top16_merge_k(const float * pv, const int * pi, int chunks, int * ids, float * vals) {
    const int row = blockIdx.x;
    const size_t o = size_t(row) * chunks * 16;
    block_top16(chunks * 16, [&](int c, float & v, int & i) { v = pv[o + c]; i = pi[o + c]; }, ids + size_t(row) * 16, vals + size_t(row) * 16);
}

__global__ void select_k(const int * ids, const float * unary, const __nv_bfloat16 * gate, const __nv_bfloat16 * succ, const __nv_bfloat16 * pred, const int * anchor,
                         int count, int vocab, int * tokens) {
    const int lane = threadIdx.x;
    int previous = *anchor;
    for (int step = 1; step <= count; ++step) {
        float maximum = -INFINITY;
        int winner = 0;
        for (int cand = 0; cand < 16; ++cand) {
            const int id = ids[step * 16 + cand];
            float sum = 0.0f;
            for (int block = 0; block < 2; ++block)
                for (int j = 0; j < 4; ++j) {
                    const int col = block * 128 + lane * 4 + j;
                    const float weighted = __fmul_rn(__bfloat162float(pred[size_t(previous) * 256 + col]), __bfloat162float(gate[step * 256 + col]));
                    sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(succ[size_t(id) * 256 + col]), weighted));
                }
            for (int o = 16; o; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
            const float score = __fadd_rn(sum, unary[step * 16 + cand]);
            if (score > maximum) { maximum = score; winner = cand; }
        }
        previous = ids[step * 16 + winner];
        if (previous >= vocab) {
            if (lane == 0) for (int rest = step; rest <= count; ++rest) tokens[rest - 1] = vocab;
            return;
        }
        if (lane == 0) tokens[step - 1] = previous;
    }
}
int grid_of(size_t n) { return int((n + 255) / 256); }
}  // namespace

void dflash_conv(const float * x, const __nv_bfloat16 * dyn, const float * base, int m, int H, int group, int side, const float * res, float * y,
                 cudaStream_t s) {
    if (m <= 0) return;
    if (H % group || side < 0 || side > 1) throw std::runtime_error("dflash_conv: shape");
    conv_k<<<grid_of(size_t(m) * H), 256, 0, s>>>(x, dyn, base, m, H, group, side, res, y);
}
void dflash_silu_mul(const __nv_bfloat16 * gu, __nv_bfloat16 * act, int m, int I, cudaStream_t s) {
    if (m > 0) silu_mul_k<<<grid_of(size_t(m) * I), 256, 0, s>>>(gu, act, m, I);
}
int dflash_top16_floats(int rows, int n) { return 2 * rows * ((n + kTopChunk - 1) / kTopChunk) * 16; }
void dflash_top16(const float * x, int rows, int n, int ld, int * ids, float * vals, float * scratch, cudaStream_t s) {
    if (rows <= 0 || n < 16) throw std::runtime_error("dflash_top16: shape");
    const int chunks = (n + kTopChunk - 1) / kTopChunk;
    float * pv = scratch;
    int * pi = reinterpret_cast<int *>(scratch + size_t(rows) * chunks * 16);
    top16_chunks_k<<<dim3(chunks, rows), kTopThreads, 0, s>>>(x, n, ld, chunks, pv, pi);
    top16_merge_k<<<rows, kTopThreads, 0, s>>>(pv, pi, chunks, ids, vals);
}
void dflash_select(const int * ids, const float * unary, const __nv_bfloat16 * gate, const __nv_bfloat16 * succ, const __nv_bfloat16 * pred, const int * anchor,
                   int count, int vocab, int * tokens, cudaStream_t s) {
    if (count < 1 || count > 7) throw std::runtime_error("dflash_select: count");
    select_k<<<1, 32, 0, s>>>(ids, unary, gate, succ, pred, anchor, count, vocab, tokens);
}

}  // namespace tqllm_cuda::inkling2
