#include "qwen35_qsa.cuh"
#include "common.cuh"

#include <cub/block/block_scan.cuh>
#include <math_constants.h>
#include <stdexcept>
#include <string>

namespace tqllm_cuda {
namespace {
using tqllm_dev::tq_split_point;
constexpr int D = kQsa27Dim, NH = kQsa27Heads, BLK = kQsa27Block, PICK = kQsa27Pick, ROT = kQsa27Rope;

void ck(cudaError_t e, int line = __builtin_LINE()) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(cudaGetErrorString(e)) + " (qwen35_qsa.cu:" + std::to_string(line) + ")");
}

// one warp per (row, output): a lane sums its 8 columns of every 256 in order, then a fixed warp reduction
__global__ void project_k(const __half * x, const __half * w, float * y, int r0, int rows, int hidden) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = blockIdx.x * (blockDim.x / 32) + warp, i = blockIdx.y;
    if (r >= rows) return;
    const uint4 * xr = reinterpret_cast<const uint4 *>(x + int64_t(i) * hidden) + lane;
    const uint4 * wr = reinterpret_cast<const uint4 *>(w + int64_t(r0 + r) * hidden) + lane;
    float s = 0.f;
    for (int j = 0; j < hidden / 256; ++j) {
        const uint4 xv = xr[j * 32], wv = wr[j * 32];
        const __half * xh = reinterpret_cast<const __half *>(&xv), * wh = reinterpret_cast<const __half *>(&wv);
#pragma unroll
        for (int e = 0; e < 8; ++e) s = fmaf(__half2float(wh[e]), __half2float(xh[e]), s);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) y[int64_t(i) * rows + r] = s;
}

// sum of squares of a 128-thread block's values, the same order for every caller
__device__ __forceinline__ float sumsq128(float v, float * part) {
    float ss = v * v;
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = ss;
    __syncthreads();
    return ((part[0] + part[1]) + part[2]) + part[3];
}

// NeoX RoPE on the first ROT dims of a normed 128-vector held in y (shared), at position pos
__device__ __forceinline__ float rope128(const float * y, int d, int pos, float theta) {
    if (d >= ROT) return y[d];
    const int i = d % (ROT / 2);
    float s, c;
    sincosf(float(pos) * powf(theta, -float(2 * i) / float(ROT)), &s, &c);
    return d < ROT / 2 ? y[d] * c - y[d + ROT / 2] * s : y[d] * c + y[d - ROT / 2] * s;
}

__global__ void token_keys_k(const float * proj, int stride, int off, const float * w, __half * tok, const int * dpos, float eps) {
    __shared__ float part[4];
    const int i = blockIdx.x, d = threadIdx.x;
    const float v = proj[int64_t(i) * stride + off + d];
    const float total = sumsq128(v, part);
    tok[(int64_t(*dpos) + i) * D + d] = __float2half(v * rsqrtf(total / D + eps) * w[d]);
}

__global__ void pool_blocks_k(const __half * tok, const float * w, float * blocks, int first, float theta, float eps) {
    __shared__ float part[4], y[D];
    const int b = first + blockIdx.x, d = threadIdx.x;
    const __half * t = tok + int64_t(b) * BLK * D + d;
    const float v = (((__half2float(t[0]) + __half2float(t[D])) + __half2float(t[2 * D])) + __half2float(t[3 * D])) * 0.25f;
    const float total = sumsq128(v, part);
    y[d] = v * rsqrtf(total / D + eps) * w[d];
    __syncthreads();
    blocks[int64_t(b) * D + d] = rope128(y, d, b * BLK, theta);
}

__global__ void queries_k(const float * proj, int stride, const float * w, float * q, const int * dpos, const int * dep,
                          float theta, float eps) {
    __shared__ float part[4], y[D];
    const int h = blockIdx.x, i = blockIdx.y, d = threadIdx.x;
    const float v = proj[int64_t(i) * stride + h * D + d];
    const float total = sumsq128(v, part);
    y[d] = v * rsqrtf(total / D + eps) * w[d];
    __syncthreads();
    q[(int64_t(i) * NH + h) * D + d] = rope128(y, d, *dpos + (dep ? dep[i] : i), theta);
}

// blocks a row may pick from: those below its split point when that is >= from, else none (a dense row)
__device__ __forceinline__ int row_blocks(const int * dpos, const int * dep, int row, int from) {
    const int split = tq_split_point(*dpos + (dep ? dep[row] : row));
    return split >= from ? split / BLK : 0;
}

// score bits of every (row, block below the row's bound): sum over the 4 heads of max(q_h . key, 0). A CTA takes 32
// blocks (4 per warp, the keys in registers) for all rows, the queries staged once in shared memory with each lane's 4
// dims as one float4; the 16 dots of a row (4 blocks x 4 heads) reduce in a transposing butterfly, whose sums have the
// bits of a shfl_xor tree per dot.
constexpr int SCORE_ROWS = 16;
__global__ void __launch_bounds__(256) score_k(const float * q, const float * blocks, uint32_t * bits, const int * dpos,
                                               const int * dep, int n, int from, int stride) {
    constexpr int RB = 4, J = D / 32;
    __shared__ float4 qs[SCORE_ROWS][NH][32];
    __shared__ int nb[SCORE_ROWS];
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    if (t < SCORE_ROWS) nb[t] = t < n ? row_blocks(dpos, dep, t, from) : 0;
    __syncthreads();
    int nbmax = 0;
    for (int r = 0; r < n; ++r) nbmax = max(nbmax, nb[r]);
    const int block0 = (blockIdx.x * 8 + warp) * RB;
    if (int(blockIdx.x) * 8 * RB >= nbmax) return;
    for (int i = t; i < n * NH * 32; i += 256) {  // float4 i of the queries, transposed: [lane][j] = dim lane + 32 j
        const int r = i / (NH * 32), h = (i / 32) % NH, m = i % 32;
        const float4 v = reinterpret_cast<const float4 *>(q + (int64_t(r) * NH + h) * D)[m];
        float * dst = reinterpret_cast<float *>(&qs[r][h][0]);
        dst[((4 * m) % 32) * 4 + (4 * m) / 32] = v.x, dst[((4 * m + 1) % 32) * 4 + (4 * m + 1) / 32] = v.y;
        dst[((4 * m + 2) % 32) * 4 + (4 * m + 2) / 32] = v.z, dst[((4 * m + 3) % 32) * 4 + (4 * m + 3) / 32] = v.w;
    }
    __syncthreads();
    if (block0 >= nbmax) return;
    float kv[RB][J];
#pragma unroll
    for (int b = 0; b < RB; ++b) {
        const float * k = blocks + int64_t(min(block0 + b, nbmax - 1)) * D;
#pragma unroll
        for (int j = 0; j < J; ++j) kv[b][j] = k[lane + 32 * j];
    }
    const int ob = block0 + (((lane >> 4) << 1) | ((lane >> 3) & 1));  // the block whose total lands on this lane
#pragma unroll 1
    for (int r = 0; r < n; ++r) {
        if (block0 >= nb[r]) continue;
        float a[RB * NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            const float4 q4 = qs[r][h][lane];
            const float qr[J] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
            for (int b = 0; b < RB; ++b) {
                float dot = 0.f;
#pragma unroll
                for (int j = 0; j < J; ++j) dot = fmaf(kv[b][j], qr[j], dot);
                a[b * NH + h] = dot;
            }
        }
        // halve at offsets 16, 8, 4, 2: lane bits 4..1 then name (block, head) = (bit4 bit3, bit2 bit1)
        tqllm_dev::butterfly_fold<16, RB * NH>(a, lane);
        const float dot = a[0] + __shfl_xor_sync(0xffffffffu, a[0], 1);
        const float d1 = __shfl_down_sync(0xffffffffu, dot, 2), d2 = __shfl_down_sync(0xffffffffu, dot, 4), d3 = __shfl_down_sync(0xffffffffu, dot, 6);
        if ((lane & 7) == 0 && ob < nb[r]) {
            float total = 0.f;
            total += fmaxf(dot, 0.f), total += fmaxf(d1, 0.f), total += fmaxf(d2, 0.f), total += fmaxf(d3, 0.f);
            bits[int64_t(r) * stride + ob] = __float_as_uint(total == 0.f ? 0.f : total);  // scores >= +0: bits order as values
        }
    }
}

// top PICK blocks of one row: three radix passes (11, 11, 10 bits) find the threshold score and how many blocks equal to
// it are taken (the lowest ones); then each thread walks its own contiguous range of blocks, so the picks come out in
// ascending block order from two scans of per-thread counts
__global__ void __launch_bounds__(1024) topk_k(const uint32_t * bits, int32_t * sel, const int * dpos, const int * dep,
                                               int from, int stride) {
    using Scan = cub::BlockScan<int, 1024>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ unsigned hist[2048];
    __shared__ uint32_t prefix_s;
    __shared__ int remain_s;
    const int r = blockIdx.x, tid = threadIdx.x;
    const int nb = row_blocks(dpos, dep, r, from), need = min(nb, PICK);
    sel += int64_t(r) * PICK;
    if (tid < PICK && tid >= need) sel[tid] = -1;
    if (need <= 0) return;
    bits += int64_t(r) * stride;
    uint32_t prefix = 0, mask = 0;
    int remain = need;
    const int shifts[3] = {21, 10, 0}, width[3] = {11, 11, 10};
#pragma unroll 1
    for (int p = 0; p < 3; ++p) {
        const int shift = shifts[p];
        const uint32_t bm = (1u << width[p]) - 1u;
        const int bins = int(bm) + 1;
        for (int i = tid; i < 2048; i += 1024) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < nb; i += 1024) {
            const uint32_t k = bits[i];
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & bm], 1u);
        }
        __syncthreads();
        // bins from the top: thread j holds bins hi = bins - 1 - 2 j and lo = hi - 1
        const int hi = bins - 1 - 2 * tid, lo = hi - 1;
        const int ch = hi >= 0 ? int(hist[hi]) : 0, cl = lo >= 0 ? int(hist[lo]) : 0;
        int before;  // blocks in the bins above hi
        Scan(scan).ExclusiveSum(ch + cl, before);
        const int s_hi = before + ch, s_lo = s_hi + cl;
        if (hi >= 0 && s_hi >= remain && before < remain) prefix_s = prefix | (uint32_t(hi) << shift), remain_s = remain - before;
        else if (lo >= 0 && s_lo >= remain && s_hi < remain) prefix_s = prefix | (uint32_t(lo) << shift), remain_s = remain - s_hi;
        __syncthreads();
        prefix = prefix_s, remain = remain_s, mask |= bm << shift;
        __syncthreads();
    }
    const int per = (nb + 1023) / 1024, i0 = min(nb, tid * per), i1 = min(nb, i0 + per);
    int above = 0, equal = 0;
    for (int i = i0; i < i1; ++i) above += bits[i] > prefix, equal += bits[i] == prefix;
    int ties_before;
    Scan(scan).ExclusiveSum(equal, ties_before);
    __syncthreads();
    int ties = min(equal, max(0, remain - ties_before)), at;  // this range's share of the blocks equal to the threshold
    Scan(scan).ExclusiveSum(above + ties, at);
    for (int i = i0; i < i1; ++i) {
        const uint32_t k = bits[i];
        if (k > prefix || (k == prefix && ties-- > 0)) sel[at++] = i;
    }
}
__global__ void tag_bump_k(int * tag) { ++*tag; }

// one CTA per (selected block, row): the first CTA to claim a block this call copies its 4 positions' host rows
__global__ void __launch_bounds__(128) stage_selected_k(const int32_t * sel, const int * dpos, const int * dep, int from,
                                                        const uint4 * hk, const uint4 * hv, uint4 * dk, uint4 * dv,
                                                        const int64_t * evicted, int krow, int vrow, int * mark, const int * tag) {
    const int z = blockIdx.y, b = sel[int64_t(z) * PICK + blockIdx.x];
    if (b < 0 || row_blocks(dpos, dep, z, from) == 0) return;
    __shared__ int mine;
    if (threadIdx.x == 0) { const int t = *tag; mine = atomicExch(mark + b, t) != t; }
    __syncthreads();
    if (!mine) return;
    for (int p = b * BLK; p < (b + 1) * BLK; ++p) {
        if ((int64_t(p) + 1) * krow <= evicted[0])
            for (int i = threadIdx.x; i < krow / 16; i += blockDim.x) dk[int64_t(p) * (krow / 16) + i] = hk[int64_t(p) * (krow / 16) + i];
        if ((int64_t(p) + 1) * vrow <= evicted[1])
            for (int i = threadIdx.x; i < vrow / 16; i += blockDim.x) dv[int64_t(p) * (vrow / 16) + i] = hv[int64_t(p) * (vrow / 16) + i];
    }
}
}  // namespace

void qsa27_stage_selected(const int32_t * sel, int n, const int * dpos, const int * dep, int from, const void * hk, const void * hv,
                          void * dk, void * dv, const int64_t * evicted, int krow, int vrow, int * mark, int * tag, cudaStream_t s) {
    if (n <= 0 || krow % 16 || vrow % 16) throw std::invalid_argument("QSA stage: rows must be multiples of 16 bytes");
    tag_bump_k<<<1, 1, 0, s>>>(tag);
    stage_selected_k<<<dim3(PICK, unsigned(n)), 128, 0, s>>>(sel, dpos, dep, from, static_cast<const uint4 *>(hk), static_cast<const uint4 *>(hv),
                                                            static_cast<uint4 *>(dk), static_cast<uint4 *>(dv), evicted, krow, vrow, mark, tag);
    ck(cudaPeekAtLastError());
}

int qsa27_split_point(int qpos) { return tq_split_point(qpos); }

void qsa27_project(const __half * x, const __half * w, float * y, int r0, int rows, int n, int hidden, cudaStream_t s) {
    if (!x || !w || !y || rows <= 0 || n <= 0 || hidden <= 0 || hidden % 256)
        throw std::invalid_argument("QSA projection: hidden size must be a multiple of 256");
    project_k<<<dim3(unsigned((rows + 7) / 8), unsigned(n)), 256, 0, s>>>(x, w, y, r0, rows, hidden);
    ck(cudaPeekAtLastError());
}

void qsa27_token_keys(const float * proj, int stride, int off, const float * k_norm, __half * tok, const int * dpos, int n,
                      float eps, cudaStream_t s) {
    token_keys_k<<<n, D, 0, s>>>(proj, stride, off, k_norm, tok, dpos, eps);
    ck(cudaPeekAtLastError());
}

void qsa27_pool_blocks(const __half * tok, const float * block_norm, float * blocks, int first, int count, float theta,
                       float eps, cudaStream_t s) {
    if (count <= 0) return;
    pool_blocks_k<<<count, D, 0, s>>>(tok, block_norm, blocks, first, theta, eps);
    ck(cudaPeekAtLastError());
}

void qsa27_queries(const float * proj, int stride, const float * q_norm, float * q, const int * dpos, const int * dep, int n,
                   float theta, float eps, cudaStream_t s) {
    queries_k<<<dim3(NH, unsigned(n)), D, 0, s>>>(proj, stride, q_norm, q, dpos, dep, theta, eps);
    ck(cudaPeekAtLastError());
}

void qsa27_select(const float * q, const float * blocks, uint32_t * bits, int32_t * sel, const int * dpos, const int * dep,
                  int n, int from, int max_blocks, cudaStream_t s) {
    if (n <= 0 || n > SCORE_ROWS || max_blocks <= 0 || from < PICK * BLK)
        throw std::invalid_argument("QSA select: 1..16 rows, from >= 2048");
    score_k<<<unsigned((max_blocks + 31) / 32), 256, 0, s>>>(q, blocks, bits, dpos, dep, n, from, max_blocks);
    topk_k<<<n, 1024, 0, s>>>(bits, sel, dpos, dep, from, max_blocks);
    ck(cudaPeekAtLastError());
}

}  // namespace tqllm_cuda
