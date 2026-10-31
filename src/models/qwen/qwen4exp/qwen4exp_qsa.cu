#include <string>
#include <cstdlib>
#include <vector>
#include "qwen4exp_kernels.cuh"
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <climits>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tqllm_cuda {
namespace {
void ck(cudaError_t e, int line = __builtin_LINE()) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(cudaGetErrorString(e)) + " (qwen4exp_qsa.cu:" + std::to_string(line) + ")");
}

__global__ void pack_scores(const float * scores, uint64_t * keys, int blocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const float score = scores[b] == 0.f ? 0.f : scores[b];
    keys[b] = (uint64_t(__float_as_uint(score)) << 32) | uint32_t(~uint32_t(b));
}

__global__ void score_blocks(const float * pooled_keys, const float * query, float * scores,
                             int blocks, int heads, int dim) {
    const int block = blockIdx.x * 4 + threadIdx.y;
    if (block >= blocks) return;
    const int lane = threadIdx.x;
    float total = 0.f;
    for (int h = 0; h < heads; ++h) {
        float dot = 0.f;
        for (int d = lane; d < dim; d += 32)
            dot = fmaf(pooled_keys[int64_t(block) * dim + d], query[int64_t(h) * dim + d], dot);
        for (int delta = 16; delta; delta >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, delta);
        if (lane == 0) total += fmaxf(dot, 0.f);
    }
    if (lane == 0) scores[block] = total;
}

__global__ void pool_raw_keys(const float * raw, float * pooled, int first_block,
                              int nblocks, int ratio, int dim) {
    const int b = first_block + blockIdx.x;
    if (blockIdx.x >= nblocks) return;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = raw[int64_t(b) * ratio * dim + d];
        for (int r = 1; r < ratio; ++r)
            sum = __fadd_rn(sum, raw[(int64_t(b) * ratio + r) * dim + d]);
        pooled[int64_t(b) * dim + d] = __fmul_rn(sum, 1.f / ratio);
    }
}

__global__ void norm_rope_blocks(const float * pooled, const float * weight, float * normalized,
                                 int first_block, int ratio, int dim, int rotary,
                                 float theta, float epsilon) {
    const int block = first_block + blockIdx.x;
    const int lane = threadIdx.x;
    const float * src = pooled + int64_t(block) * dim;
    float ss = 0.f;
    for (int d = lane; d < dim; d += 32) ss += src[d] * src[d];
    for (int delta = 16; delta; delta >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, delta);
    const float inv = rsqrtf(ss / dim + epsilon);
    const int half = rotary / 2;
    for (int d = lane; d < dim; d += 32) {
        float value = __fmul_rn(__fmul_rn(src[d], inv), weight[d]);
        if (d < rotary) {
            const int pair = d % half;
            const double angle = double(int64_t(block) * ratio) / pow(double(theta), double(2 * pair) / rotary);
            const float cosine = float(cos(angle)), sine = float(sin(angle));
            const int other = d < half ? d + half : d - half;
            const float paired = __fmul_rn(__fmul_rn(src[other], inv), weight[other]);
            value = d < half ? value * cosine - paired * sine : value * cosine + paired * sine;
        }
        normalized[int64_t(block) * dim + d] = value;
    }
}

__global__ void expand_selection(const uint64_t * sorted, int32_t * positions, uint8_t * valid,
                                 int width, int complete, int tail, int ratio, int query_position) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    if (i < tail) {
        positions[i] = complete * ratio + i;
        valid[i] = 1;
        return;
    }
    const int rank = i - tail;
    if (rank < complete * ratio) {
        const int block = int(~uint32_t(sorted[rank / ratio]));
        positions[i] = block * ratio + rank % ratio;
        valid[i] = 1;
        return;
    }
    positions[i] = query_position + 1 + rank - complete * ratio;
    valid[i] = 0;
}

__global__ void gather_flat_bf16(const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                                 const int32_t * positions, const uint8_t * valid,
                                 __nv_bfloat16 * gathered_keys, __nv_bfloat16 * gathered_values,
                                 int width, int cells, int row) {
    const int i = blockIdx.x;
    if (i >= width) return;
    const int pos = positions[i];
    const bool fetch = valid[i] && pos >= 0 && pos < cells;
    for (int j = threadIdx.x; j < row; j += blockDim.x) {
        const int64_t out = int64_t(i) * row + j;
        gathered_keys[out] = fetch ? keys[int64_t(pos) * row + j] : __float2bfloat16(0.f);
        gathered_values[out] = fetch ? values[int64_t(pos) * row + j] : __float2bfloat16(0.f);
    }
}

__global__ void attention_selected(const float * query, const __nv_bfloat16 * keys,
                                   const __nv_bfloat16 * values, const uint8_t * valid,
                                   float * output, int width, int heads, int kv_heads, int dim) {
    const int head = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kv_head = head / (heads / kv_heads);
    const float * q = query + int64_t(head) * dim;
    extern __shared__ float scores[];
    __shared__ float reduction[128];
    for (int row = warp; row < width; row += 4) {
        float dot = 0.f;
        if (valid[row])
            for (int d = lane; d < dim; d += 32)
                dot = fmaf(q[d], __bfloat162float(keys[(int64_t(row) * kv_heads + kv_head) * dim + d]), dot);
        for (int delta = 16; delta; delta >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, delta);
        if (lane == 0) scores[row] = valid[row] ? dot * rsqrtf(float(dim)) : -INFINITY;
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (int row = tid; row < width; row += 128) maximum = fmaxf(maximum, scores[row]);
    reduction[tid] = maximum;
    __syncthreads();
    for (int delta = 64; delta; delta >>= 1) {
        if (tid < delta) reduction[tid] = fmaxf(reduction[tid], reduction[tid + delta]);
        __syncthreads();
    }
    maximum = reduction[0];
    __syncthreads();
    if (!isfinite(maximum)) {
        for (int d = tid; d < dim; d += 128) output[int64_t(head) * dim + d] = 0.f;
        return;
    }
    float sum = 0.f;
    for (int row = tid; row < width; row += 128) {
        const float weight = valid[row] ? expf(scores[row] - maximum) : 0.f;
        scores[row] = weight;
        sum += weight;
    }
    reduction[tid] = sum;
    __syncthreads();
    for (int delta = 64; delta; delta >>= 1) {
        if (tid < delta) reduction[tid] += reduction[tid + delta];
        __syncthreads();
    }
    const float inv = 1.f / reduction[0];
    __syncthreads();
    for (int d = tid; d < dim; d += 128) {
        float value = 0.f;
        for (int row = 0; row < width; ++row)
            value = fmaf(scores[row], __bfloat162float(values[(int64_t(row) * kv_heads + kv_head) * dim + d]), value);
        output[int64_t(head) * dim + d] = value * inv;
    }
}
__global__ void score_pack_batch(const float * pooled_keys, const float * query, uint64_t * keys,
                                 int blocks, int first_position, int ratio, int heads, int dim) {
    const int block = blockIdx.x * 4 + threadIdx.y, t = blockIdx.y;
    if (block >= blocks) return;
    uint64_t * out = keys + int64_t(t) * blocks;
    if (block >= (first_position + t + 1) / ratio) {  // not complete for this query: sorts below every valid key
        if (threadIdx.x == 0) out[block] = 0;
        return;
    }
    const float * q = query + int64_t(t) * heads * dim;
    const int lane = threadIdx.x;
    float total = 0.f;
    for (int h = 0; h < heads; ++h) {
        float dot = 0.f;
        for (int d = lane; d < dim; d += 32)
            dot = fmaf(pooled_keys[int64_t(block) * dim + d], q[int64_t(h) * dim + d], dot);
        for (int delta = 16; delta; delta >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, delta);
        if (lane == 0) total += fmaxf(dot, 0.f);
    }
    if (lane == 0) {
        const float score = total == 0.f ? 0.f : total;
        out[block] = (uint64_t(__float_as_uint(score)) << 32) | uint32_t(~uint32_t(block));
    }
}

// score_pack_batch for 4 indexer heads of 128: a CTA of W warps scores W * RB blocks for QT consecutive queries, the
// blocks' keys in registers and the queries staged once in shared memory (one key read per QT queries and one query read
// per W * RB blocks instead of one each per (query, block): at 128K the re-reads made this kernel 13 s of a 74 s prefill);
// the RB * 4 dots of a query reduce in a transposing butterfly (the pairs of the shfl_down tree; a + b == b + a) instead
// of 5 shuffles each. Same keys bit for bit.
// One level of the transposing butterfly over a lane's N values: lanes with bit OFF set keep their upper half and add
// the partner's lower half, the others the reverse; N / 2 values remain. Compile-time OFF and N keep the array in
// registers (a runtime loop turned the 15 shuffles into ~500 instructions of register-array emulation).
template<int OFF, int N>
__device__ __forceinline__ void butterfly_fold(float (&a)[16], int lane) {
    const bool upper = lane & OFF;
    #pragma unroll
    for (int j = 0; j < N / 2; ++j) {
        const float mine = upper ? a[j + N / 2] : a[j], send = upper ? a[j] : a[j + N / 2];
        a[j] = mine + __shfl_xor_sync(0xffffffffu, send, OFF);
    }
    if constexpr (OFF > 2) butterfly_fold<OFF / 2, N / 2>(a, lane);
}
template<int RB, int QT, int W>
__global__ void __launch_bounds__(32 * W) score_pack_fixed_k(const float * pooled_keys, const float * query, uint64_t * keys,
                                                            int blocks, int tokens, int first_position, int ratio) {
    constexpr int H = 4, DIM = 128, J = DIM / 32;
    static_assert(RB == 4, "lane mapping below assumes 16 values per lane");
    __shared__ float4 qs[QT][H][32];  // [query][head][lane] = the lane's 4 dims (lane + 32 j), one 16-byte read per head
    const int lane = threadIdx.x, t0 = blockIdx.y * QT, block0 = (blockIdx.x * W + threadIdx.y) * RB;
    const int nq = min(QT, tokens - t0);
    for (int i = threadIdx.y * 32 + lane; i < nq * H * 32; i += 32 * W) {  // float4 i of the group's queries, transposed
        const int qi = i / (H * 32), h = (i / 32) % H, m = i % 32;
        const float4 v = reinterpret_cast<const float4 *>(query + int64_t(t0 + qi) * H * DIM + h * DIM)[m];
        float * dst = reinterpret_cast<float *>(&qs[qi][h][0]);
        dst[((4 * m) % 32) * 4 + (4 * m) / 32] = v.x; dst[((4 * m + 1) % 32) * 4 + (4 * m + 1) / 32] = v.y;
        dst[((4 * m + 2) % 32) * 4 + (4 * m + 2) / 32] = v.z; dst[((4 * m + 3) % 32) * 4 + (4 * m + 3) / 32] = v.w;
    }
    __syncthreads();
    if (block0 >= blocks) return;
    float kv[RB][J];
    #pragma unroll
    for (int b = 0; b < RB; ++b) {
        const float * k = pooled_keys + int64_t(min(block0 + b, blocks - 1)) * DIM;
        #pragma unroll
        for (int j = 0; j < J; ++j) kv[b][j] = k[lane + 32 * j];
    }
    const int ob = block0 + (((lane >> 4) << 1) | ((lane >> 3) & 1));  // the block whose total lands on this lane
    int complete = (first_position + t0 + 1) / ratio, rem = (first_position + t0 + 1) % ratio;  // complete blocks of query t
    #pragma unroll 1
    for (int qi = 0; qi < QT; ++qi, complete += ++rem == ratio ? (rem = 0, 1) : 0) {
        const int t = t0 + qi;
        if (t >= tokens) return;
        uint64_t * out = keys + int64_t(t) * blocks;
        if (block0 >= complete) {  // no block of the group complete for this query: every key 0
            if ((lane & 7) == 0 && ob < blocks) out[ob] = 0;
            continue;
        }
        float a[RB * H];
        #pragma unroll
        for (int h = 0; h < H; ++h) {
            const float4 q4 = qs[qi][h][lane];
            const float qr[J] = {q4.x, q4.y, q4.z, q4.w};
            #pragma unroll
            for (int b = 0; b < RB; ++b) {
                float dot = 0.f;
                #pragma unroll
                for (int j = 0; j < J; ++j) dot = fmaf(kv[b][j], qr[j], dot);
                a[b * H + h] = dot;
            }
        }
        // 16 values: halve at offsets 16, 8, 4, 2; lane bits 4..1 then name (block, head) = (bit4 bit3, bit2 bit1)
        butterfly_fold<16, RB * H>(a, lane);
        const float dot = a[0] + __shfl_xor_sync(0xffffffffu, a[0], 1);
        // heads 1..3 of this lane's block sit 2, 4, 6 lanes up; the total adds them in head order as score_pack_batch does
        const float d1 = __shfl_down_sync(0xffffffffu, dot, 2), d2 = __shfl_down_sync(0xffffffffu, dot, 4), d3 = __shfl_down_sync(0xffffffffu, dot, 6);
        if ((lane & 7) == 0 && ob < blocks) {
            if (ob >= complete) { out[ob] = 0; continue; }
            float total = 0.f;
            total += fmaxf(dot, 0.f); total += fmaxf(d1, 0.f); total += fmaxf(d2, 0.f); total += fmaxf(d3, 0.f);
            const float score = total == 0.f ? 0.f : total;
            out[ob] = (uint64_t(__float_as_uint(score)) << 32) | uint32_t(~uint32_t(ob));
        }
    }
}

__global__ void segment_offsets(int * offsets, int segments, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= segments) offsets[i] = i * size;
}

__global__ void expand_selection_batch(const uint64_t * sorted, int32_t * positions, uint8_t * valid,
                                       int width, int sorted_stride, int first_position, int ratio) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= width) return;
    const int query_position = first_position + t;
    const int complete = (query_position + 1) / ratio, tail = (query_position + 1) % ratio;
    sorted += int64_t(t) * sorted_stride; positions += int64_t(t) * width; valid += int64_t(t) * width;
    if (i < tail) {
        positions[i] = complete * ratio + i;
        valid[i] = 1;
        return;
    }
    const int rank = i - tail;
    if (rank < complete * ratio) {
        const int block = int(~uint32_t(sorted[rank / ratio]));
        positions[i] = block * ratio + rank % ratio;
        valid[i] = 1;
        return;
    }
    positions[i] = query_position + 1 + rank - complete * ratio;
    valid[i] = 0;
}

// Top `need` keys of each query's complete blocks, in descending order: radix select of the need-th largest key (keys are
// unique: score bits | inverted block), compaction, then a bitonic sort in shared memory. Same prefix as a full sort.
constexpr int kTopkCap = 1024;
__global__ void __launch_bounds__(1024) topk_sorted_k(const uint64_t * keys, uint64_t * out, int blocks, int first_position,
                                                     int ratio, int need_cap) {
    const int t = blockIdx.x, tid = threadIdx.x;
    const int complete = (first_position + t + 1) / ratio, need = min(complete, need_cap);
    if (need <= 0) return;
    keys += int64_t(t) * blocks; out += int64_t(t) * kTopkCap;
    __shared__ unsigned hist[256];
    __shared__ uint64_t prefix_s; __shared__ int remain_s;
    __shared__ uint64_t buf[kTopkCap];
    __shared__ int count_s;
    uint64_t prefix = 0, mask = 0; int remain = need;
    for (int shift = 56; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < complete; i += 1024) {
            const uint64_t k = keys[i];
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int r = remain, d = 255;
            for (; d > 0; --d) { if (int(hist[d]) >= r) break; r -= int(hist[d]); }
            prefix_s = prefix | (uint64_t(d) << shift); remain_s = r;
        }
        __syncthreads();
        prefix = prefix_s; remain = remain_s; mask |= uint64_t(255) << shift;
        __syncthreads();
    }
    // prefix is now the need-th largest key
    if (tid == 0) count_s = 0;
    for (int i = tid; i < kTopkCap; i += 1024) buf[i] = 0;
    __syncthreads();
    for (int i = tid; i < complete; i += 1024) {
        const uint64_t k = keys[i];
        if (k >= prefix) buf[atomicAdd(&count_s, 1)] = k;
    }
    __syncthreads();
    for (int size = 2; size <= kTopkCap; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int j = tid ^ stride;
            if (j > tid) {
                const bool desc = (tid & size) == 0;
                const uint64_t a = buf[tid], b = buf[j];
                if (desc ? a < b : a > b) { buf[tid] = b; buf[j] = a; }
            }
            __syncthreads();
        }
    if (tid < need) out[tid] = buf[tid];
}

// score_pack_fixed_k with the score bits only: the key's low word is ~block, i.e. the index
template<int RB, int QT, int W>
__global__ void __launch_bounds__(32 * W) score_bits_k(const float * pooled_keys, const float * query, uint32_t * scores,
                                                     int blocks, int tokens, int first_position, int ratio) {
    constexpr int H = 4, DIM = 128, J = DIM / 32;
    __shared__ float4 qs[QT][H][32];
    const int lane = threadIdx.x, t0 = blockIdx.y * QT, block0 = (blockIdx.x * W + threadIdx.y) * RB;
    const int nq = min(QT, tokens - t0);
    for (int i = threadIdx.y * 32 + lane; i < nq * H * 32; i += 32 * W) {
        const int qi = i / (H * 32), h = (i / 32) % H, m = i % 32;
        const float4 v = reinterpret_cast<const float4 *>(query + int64_t(t0 + qi) * H * DIM + h * DIM)[m];
        float * dst = reinterpret_cast<float *>(&qs[qi][h][0]);
        dst[((4 * m) % 32) * 4 + (4 * m) / 32] = v.x; dst[((4 * m + 1) % 32) * 4 + (4 * m + 1) / 32] = v.y;
        dst[((4 * m + 2) % 32) * 4 + (4 * m + 2) / 32] = v.z; dst[((4 * m + 3) % 32) * 4 + (4 * m + 3) / 32] = v.w;
    }
    __syncthreads();
    if (block0 >= blocks) return;
    float kv[RB][J];
    #pragma unroll
    for (int b = 0; b < RB; ++b) {
        const float * k = pooled_keys + int64_t(min(block0 + b, blocks - 1)) * DIM;
        #pragma unroll
        for (int j = 0; j < J; ++j) kv[b][j] = k[lane + 32 * j];
    }
    const int ob = block0 + (((lane >> 4) << 1) | ((lane >> 3) & 1));
    int complete = (first_position + t0 + 1) / ratio, rem = (first_position + t0 + 1) % ratio;
    #pragma unroll 1
    for (int qi = 0; qi < QT; ++qi, complete += ++rem == ratio ? (rem = 0, 1) : 0) {
        const int t = t0 + qi;
        if (t >= tokens) return;
        uint32_t * out = scores + int64_t(t) * blocks;
        if (block0 >= complete) continue;  // the top-k reads only complete blocks
        float a[RB * H];
        #pragma unroll
        for (int h = 0; h < H; ++h) {
            const float4 q4 = qs[qi][h][lane];
            const float qr[J] = {q4.x, q4.y, q4.z, q4.w};
            #pragma unroll
            for (int b = 0; b < RB; ++b) {
                float dot = 0.f;
                #pragma unroll
                for (int j = 0; j < J; ++j) dot = fmaf(kv[b][j], qr[j], dot);
                a[b * H + h] = dot;
            }
        }
        butterfly_fold<16, RB * H>(a, lane);
        const float dot = a[0] + __shfl_xor_sync(0xffffffffu, a[0], 1);
        const float d1 = __shfl_down_sync(0xffffffffu, dot, 2), d2 = __shfl_down_sync(0xffffffffu, dot, 4), d3 = __shfl_down_sync(0xffffffffu, dot, 6);
        if ((lane & 7) == 0 && ob < blocks && ob < complete) {
            float total = 0.f;
            total += fmaxf(dot, 0.f); total += fmaxf(d1, 0.f); total += fmaxf(d2, 0.f); total += fmaxf(d3, 0.f);
            out[ob] = __float_as_uint(total == 0.f ? 0.f : total);
        }
    }
}
// topk_sorted_k on score bits: the need-th largest key is (T << 32 | ~i) for the threshold score T; three radix passes
// (11, 11, 10 bits) find T and how many keys equal to T it takes (r); ties go by index, smallest first, as the key order does.
__global__ void __launch_bounds__(1024) topk_bits_k(const uint32_t * scores, uint64_t * out, int blocks, int first_position,
                                                    int ratio, int need_cap) {
    using Scan = cub::BlockScan<int, 1024>;
    __shared__ typename Scan::TempStorage scan;
    const int t = blockIdx.x, tid = threadIdx.x;
    const int complete = (first_position + t + 1) / ratio, need = min(complete, need_cap);
    if (need <= 0) return;
    scores += int64_t(t) * blocks; out += int64_t(t) * kTopkCap;
    __shared__ unsigned hist[2048];
    __shared__ uint32_t prefix_s; __shared__ int remain_s, count_s;
    __shared__ uint64_t buf[kTopkCap];
    uint32_t prefix = 0, mask = 0; int remain = need;
    const int shifts[3] = {21, 10, 0}, bits[3] = {11, 11, 10};
    #pragma unroll 1
    for (int p = 0; p < 3; ++p) {
        const int shift = shifts[p]; const uint32_t bm = (1u << bits[p]) - 1u; const int nb = int(bm) + 1;
        for (int i = tid; i < 2048; i += 1024) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < complete; i += 1024) {
            const uint32_t k = scores[i];
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & bm], 1u);
        }
        __syncthreads();
        // bins from the top: thread j holds bins hi = nb-1-2j and lo = hi-1; S(d) = keys in bins >= d
        const int hi = nb - 1 - 2 * tid, lo = hi - 1;
        const int ch = hi >= 0 ? int(hist[hi]) : 0, cl = lo >= 0 ? int(hist[lo]) : 0;
        int before;  // keys in the bins above hi
        Scan(scan).ExclusiveSum(ch + cl, before);
        const int s_hi = before + ch, s_lo = s_hi + cl;
        // chosen bin d: S(d) >= remain and S(d + 1) < remain (bin 0 if no higher one), as the serial walk from the top picks
        if (hi >= 0 && s_hi >= remain && before < remain) { prefix_s = prefix | (uint32_t(hi) << shift); remain_s = remain - before; }
        else if (lo >= 0 && s_lo >= remain && s_hi < remain) { prefix_s = prefix | (uint32_t(lo) << shift); remain_s = remain - s_hi; }
        __syncthreads();
        prefix = prefix_s; remain = remain_s; mask |= bm << shift;
        __syncthreads();
    }
    // prefix = T; keys above T all go, the first `remain` keys equal to T (by index) go
    if (tid == 0) count_s = 0;
    for (int i = tid; i < kTopkCap; i += 1024) buf[i] = 0;
    __syncthreads();
    int ties = 0;
    #pragma unroll 1
    for (int base = 0; base < complete; base += 1024) {
        const int i = base + tid;
        const uint32_t k = i < complete ? scores[i] : 0u;
        const bool gt = i < complete && k > prefix, eq = i < complete && k == prefix;
        int rank, total;
        Scan(scan).ExclusiveSum(int(eq), rank, total);
        if (gt || (eq && ties + rank < remain)) buf[atomicAdd(&count_s, 1)] = (uint64_t(k) << 32) | uint32_t(~uint32_t(i));
        ties += total;
        __syncthreads();
    }
    __syncthreads();
    for (int size = 2; size <= kTopkCap; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int j = tid ^ stride;
            if (j > tid) {
                const bool desc = (tid & size) == 0;
                const uint64_t a = buf[tid], b = buf[j];
                if (desc ? a < b : a > b) { buf[tid] = b; buf[j] = a; }
            }
            __syncthreads();
        }
    if (tid < need) out[tid] = buf[tid];
}

} // namespace

QsaBatchScratch::QsaBatchScratch(int max_queries_, int max_cells, int ratio, int top_k)
    : max_queries(max_queries_), max_blocks(std::max(1, max_cells / ratio)),
      width_cap(int(std::min<int64_t>(max_cells, int64_t(top_k) + ratio - 1))) {
    if (max_queries <= 0 || max_cells <= 0 || ratio <= 0 || top_k <= 0)
        throw std::invalid_argument("QSA batch scratch: invalid dimensions");
    const size_t items = size_t(max_queries) * max_blocks;
    lean = (width_cap - 1) / ratio + 1 <= kTopkCap && max_blocks >= kTopkCap;
    try {
        ck(cudaMalloc(&query, size_t(max_queries) * 4 * 128 * sizeof(float)));
        if (lean) {
            ck(cudaMalloc(&keys, items * sizeof(uint32_t)));
            ck(cudaMalloc(&sorted, size_t(max_queries) * kTopkCap * sizeof(uint64_t)));
        } else {
            ck(cudaMalloc(&keys, items * sizeof(uint64_t)));
            ck(cudaMalloc(&sorted, items * sizeof(uint64_t)));
            ck(cudaMalloc(&offsets, size_t(max_queries + 1) * sizeof(int)));
            ck(cub::DeviceSegmentedRadixSort::SortKeysDescending(nullptr, temp_bytes, keys, sorted, int(items), max_queries,
                                                                 offsets, offsets + 1));
            ck(cudaMalloc(&temp, temp_bytes));
        }
        ck(cudaMalloc(&positions, size_t(max_queries) * width_cap * sizeof(int32_t)));
        ck(cudaMalloc(&valid, size_t(max_queries) * width_cap));
    } catch (...) {
        cudaFree(valid); cudaFree(positions); cudaFree(temp); cudaFree(offsets); cudaFree(sorted); cudaFree(keys); cudaFree(query);
        throw;
    }
}

void QsaBatchScratch::full() {
    if (!lean) return;
    const size_t items = size_t(max_queries) * max_blocks;
    ck(cudaDeviceSynchronize()); cudaFree(keys); cudaFree(sorted); keys = sorted = nullptr;
    ck(cudaMalloc(&keys, items * sizeof(uint64_t)));
    ck(cudaMalloc(&sorted, items * sizeof(uint64_t)));
    ck(cudaMalloc(&offsets, size_t(max_queries + 1) * sizeof(int)));
    ck(cub::DeviceSegmentedRadixSort::SortKeysDescending(nullptr, temp_bytes, keys, sorted, int(items), max_queries, offsets, offsets + 1));
    ck(cudaMalloc(&temp, temp_bytes));
    lean = false;
}
QsaBatchScratch::~QsaBatchScratch() {
    cudaFree(valid); cudaFree(positions); cudaFree(temp); cudaFree(offsets); cudaFree(sorted); cudaFree(keys); cudaFree(query);
}

void qsa_score_pack(const float * pooled_keys, const float * query, uint64_t * keys, int blocks, int first_position, int ratio,
                    int heads, int dim, int tokens, cudaStream_t stream, int fixed) {
    if (fixed < 0) fixed = 1;  // fixed-shape kernels (bitwise equal to the generic ones, test-qsa-pack-exact)
    if (fixed && heads == 4 && dim == 128)
        score_pack_fixed_k<4, 16, 8><<<dim3(unsigned((blocks + 31) / 32), unsigned((tokens + 15) / 16)), dim3(32, 8), 0, stream>>>(pooled_keys, query, keys, blocks, tokens, first_position, ratio);
    else
        score_pack_batch<<<dim3((blocks + 3) / 4, tokens), dim3(32, 4), 0, stream>>>(pooled_keys, query, keys, blocks, first_position, ratio, heads, dim);
    ck(cudaPeekAtLastError());
}

void qsa_score_blocks(const float * pooled_keys, const float * query, float * scores,
                      int blocks, int heads, int dim, cudaStream_t stream) {
    if (!pooled_keys || !query || !scores || blocks <= 0 || heads <= 0 || dim <= 0)
        throw std::invalid_argument("QSA scores: invalid dimensions or pointer");
    score_blocks<<<(blocks + 3) / 4, dim3(32, 4), 0, stream>>>(pooled_keys, query, scores,
                                                                blocks, heads, dim);
    ck(cudaPeekAtLastError());
}

void qsa_norm_rope_blocks(const float * pooled, const float * weight, float * normalized,
                          int first_block, int blocks, int ratio, int dim, int rotary,
                          float theta, float epsilon, cudaStream_t stream) {
    if (!pooled || !weight || !normalized || normalized == pooled || normalized == weight ||
        first_block < 0 || blocks <= 0 || ratio <= 0 || dim <= 0 ||
        rotary < 2 || rotary > dim || rotary % 2 ||
        !std::isfinite(theta) || theta <= 0.f || !std::isfinite(epsilon) || epsilon <= 0.f ||
        (int64_t(first_block) + blocks - 1) * ratio > std::numeric_limits<int>::max())
        throw std::invalid_argument("QSA norm/RoPE: invalid dimensions or pointer");
    norm_rope_blocks<<<blocks, 32, 0, stream>>>(pooled, weight, normalized, first_block,
                                                 ratio, dim, rotary, theta, epsilon);
    ck(cudaPeekAtLastError());
}

QsaRawKeyCache::QsaRawKeyCache(int max_cells, int ratio, int dim)
    : max_cells_(max_cells), ratio_(ratio), dim_(dim) {
    if (max_cells <= 0 || ratio <= 0 || ratio > 64 || dim <= 0 ||
        int64_t(max_cells) * dim > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::invalid_argument("QSA raw-key cache: invalid dimensions");
    try {
        ck(cudaMalloc(&raw_, size_t(max_cells) * dim * sizeof(float)));
        if (max_cells >= ratio)
            ck(cudaMalloc(&pooled_, size_t(max_cells / ratio) * dim * sizeof(float)));
        if (max_cells >= ratio)
            ck(cudaMalloc(&normalized_, size_t(max_cells / ratio) * dim * sizeof(float)));
    } catch (...) {
        cudaFree(normalized_); cudaFree(pooled_); cudaFree(raw_);
        throw;
    }
}

QsaRawKeyCache::~QsaRawKeyCache() {
    cudaFree(normalized_);
    cudaFree(pooled_);
    cudaFree(raw_);
}

void QsaRawKeyCache::normalize_new(const float * weight, int rotary, float theta,
                                   float epsilon, cudaStream_t stream) {
    const int count = complete_blocks() - normalized_blocks_;
    if (!count) return;
    qsa_norm_rope_blocks(pooled_, weight, normalized_, normalized_blocks_, count,
                         ratio_, dim_, rotary, theta, epsilon, stream);
    normalized_blocks_ += count;
}

void QsaRawKeyCache::append(const float * keys, int tokens, cudaStream_t stream) {
    if (!keys || tokens <= 0 || tokens > max_cells_ - size_)
        throw std::invalid_argument("QSA raw-key cache: invalid append");
    const int end = size_ + tokens;
    ck(cudaMemcpyAsync(raw_ + int64_t(size_) * dim_, keys, size_t(tokens) * dim_ * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream));
    const int first = size_ / ratio_;
    const int count = end / ratio_ - first;
    if (count > 0) {
        pool_raw_keys<<<count, 128, 0, stream>>>(raw_, pooled_, first, count, ratio_, dim_);
        ck(cudaPeekAtLastError());
    }
    size_ = end;
}

void QsaRawKeyCache::truncate(int cells) {
    if (cells < 0 || cells > size_) throw std::invalid_argument("QSA raw-key cache: invalid truncate");
    size_ = cells;
    normalized_blocks_ = std::min(normalized_blocks_, cells / ratio_);
}

QsaSelector::QsaSelector(int max_cells, int ratio, int top_k)
    : max_cells_(max_cells), ratio_(ratio), top_k_(top_k) {
    if (max_cells <= 0 || ratio <= 0 || ratio > 64 || top_k <= 0 ||
        int64_t(top_k) + ratio - 1 > std::numeric_limits<int>::max())
        throw std::invalid_argument("QSA selector: invalid dimensions");
    const int blocks = int((int64_t(max_cells) + ratio - 1) / ratio);
    try {
        ck(cudaMalloc(&keys_, size_t(blocks) * sizeof(uint64_t)));
        ck(cudaMalloc(&sorted_, size_t(blocks) * sizeof(uint64_t)));
        ck(cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_bytes_, keys_, sorted_, blocks));
        ck(cudaMalloc(&temp_, temp_bytes_));
    } catch (...) {
        cudaFree(temp_); cudaFree(sorted_); cudaFree(keys_);
        throw;
    }
}

QsaSelector::~QsaSelector() {
    cudaFree(temp_);
    cudaFree(sorted_);
    cudaFree(keys_);
}

int QsaSelector::select(const float * block_scores, int cells, int query_position,
                        int32_t * positions, uint8_t * valid, cudaStream_t stream) {
    if (cells <= 0 || cells > max_cells_ || query_position < 0 || query_position >= cells ||
        !block_scores || !positions || !valid)
        throw std::invalid_argument("QSA selector: invalid query or pointer");
    const int complete = (query_position + 1) / ratio_;
    const int tail = (query_position + 1) % ratio_;
    const int width = std::min(cells, top_k_ + ratio_ - 1);
    if (complete) {
        pack_scores<<<(complete + 255) / 256, 256, 0, stream>>>(block_scores, keys_, complete);
        ck(cudaPeekAtLastError());
        size_t needed = 0;
        ck(cub::DeviceRadixSort::SortKeysDescending(nullptr, needed, keys_, sorted_, complete, 0, 64, stream));
        if (needed > temp_bytes_) throw std::runtime_error("QSA selector: sort scratch is too small");
        size_t available = temp_bytes_;
        ck(cub::DeviceRadixSort::SortKeysDescending(temp_, available, keys_, sorted_, complete, 0, 64, stream));
    }
    expand_selection<<<(width + 255) / 256, 256, 0, stream>>>(sorted_, positions, valid,
                                                               width, complete, tail, ratio_, query_position);
    ck(cudaPeekAtLastError());
    return width;
}

void qsa_gather_flat_bf16(const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                          const QsaDeviceSelection & selected, __nv_bfloat16 * gathered_keys,
                          __nv_bfloat16 * gathered_values, int cells, int heads, int dim, cudaStream_t stream) {
    if (!keys || !values || !selected.positions || !selected.valid || !gathered_keys || !gathered_values ||
        selected.width <= 0 || cells <= 0 || heads <= 0 || dim <= 0 ||
        int64_t(heads) * dim > std::numeric_limits<int>::max())
        throw std::invalid_argument("QSA flat gather: invalid dimensions or pointer");
    gather_flat_bf16<<<selected.width, 256, 0, stream>>>(keys, values, selected.positions, selected.valid,
                                                          gathered_keys, gathered_values,
                                                          selected.width, cells, heads * dim);
    ck(cudaPeekAtLastError());
}

void qsa_attention_selected(const float * query, const __nv_bfloat16 * gathered_keys,
                            const __nv_bfloat16 * gathered_values, const uint8_t * valid,
                            float * output, int width, int heads, int kv_heads, int dim, cudaStream_t stream) {
    if (!query || !gathered_keys || !gathered_values || !valid || !output ||
        width <= 0 || width > 8192 || heads <= 0 || kv_heads <= 0 || heads % kv_heads ||
        dim <= 0 || dim > 256)
        throw std::invalid_argument("QSA selected attention: invalid dimensions or pointer");
    attention_selected<<<heads, 128, size_t(width) * sizeof(float), stream>>>(
        query, gathered_keys, gathered_values, valid, output, width, heads, kv_heads, dim);
    ck(cudaPeekAtLastError());
}

QsaIndexer::QsaIndexer(int max_cells, int ratio, int top_k, int heads, int dim, int rotary,
                       float theta, float epsilon)
    : keys_(max_cells, ratio, dim), selector_(max_cells, ratio, top_k),
      ratio_(ratio), heads_(heads), dim_(dim), rotary_(rotary),
      theta_(theta), epsilon_(epsilon) {
    if (heads <= 0 || dim <= 0 || int64_t(heads) * dim > std::numeric_limits<int>::max() ||
        rotary < 2 || rotary > dim || rotary % 2 || !std::isfinite(theta) || theta <= 0.f ||
        !std::isfinite(epsilon) || epsilon <= 0.f)
        throw std::invalid_argument("QSA indexer: invalid query dimensions");
    const size_t width = std::min<int64_t>(max_cells, int64_t(top_k) + ratio - 1);
    try {
        ck(cudaMalloc(&query_, size_t(heads) * dim * sizeof(float)));
        ck(cudaMalloc(&scores_, size_t(std::max(1, max_cells / ratio)) * sizeof(float)));
        ck(cudaMalloc(&positions_, width * sizeof(int32_t)));
        ck(cudaMalloc(&valid_, width * sizeof(uint8_t)));
    } catch (...) {
        cudaFree(valid_); cudaFree(positions_); cudaFree(scores_); cudaFree(query_);
        throw;
    }
}

QsaIndexer::~QsaIndexer() {
    cudaFree(valid_);
    cudaFree(positions_);
    cudaFree(scores_);
    cudaFree(query_);
}

void QsaIndexer::append(const float * raw_keys, int tokens, const float * key_norm, cudaStream_t stream) {
    if (!key_norm) throw std::invalid_argument("QSA indexer: missing key norm");
    keys_.append(raw_keys, tokens, stream);
    keys_.normalize_new(key_norm, rotary_, theta_, epsilon_, stream);
}

QsaDeviceSelection QsaIndexer::select(const float * raw_query, const float * query_norm,
                                      int query_position, cudaStream_t stream) {
    if (!raw_query || !query_norm || query_position < 0 || query_position >= keys_.size())
        throw std::invalid_argument("QSA indexer: invalid query");
    fn_norm_rope(raw_query, query_norm, query_, heads_, dim_, 1, query_position,
                 rotary_, theta_, epsilon_, false, stream);
    if (const int blocks = (query_position + 1) / ratio_; blocks > 0)
        qsa_score_blocks(keys_.normalized(), query_, scores_, blocks, heads_, dim_, stream);
    const int width = selector_.select(scores_, keys_.size(), query_position, positions_, valid_, stream);
    return {positions_, valid_, width};
}
QsaDeviceSelection QsaIndexer::select_batch(const float * raw_queries, const float * query_norm, int first_position,
                                            int tokens, QsaBatchScratch & scratch, cudaStream_t stream) {
    const int cells = keys_.size();
    if (!raw_queries || !query_norm || tokens <= 0 || tokens > scratch.max_queries || first_position < 0 ||
        first_position + tokens > cells || heads_ * dim_ > 4 * 128)
        throw std::invalid_argument("QSA indexer: invalid query batch");
    fn_norm_rope(raw_queries, query_norm, scratch.query, heads_, dim_, tokens, first_position,
                 rotary_, theta_, epsilon_, false, stream);
    const int blocks = (first_position + tokens) / ratio_;  // complete blocks of the last query
    if (blocks > scratch.max_blocks) throw std::invalid_argument("QSA indexer: batch exceeds its scratch");
    const int width = std::min(cells, scratch.width_cap);
    // blocks a selection can reach: rank < width - tail, so block index <= (width - 1) / ratio
    const int need_cap = (width - 1) / ratio_ + 1;
    constexpr bool topk_on = true;  // top-k select instead of a full sort (TQFN_QSA_CHECK compares them)
    static const bool check = std::getenv("TQFN_QSA_CHECK") != nullptr;
    const bool topk = topk_on && need_cap <= kTopkCap && int64_t(tokens) * kTopkCap <= int64_t(scratch.max_queries) * scratch.max_blocks;
    int stride = blocks;
    if (blocks > 0 && topk) {
        if (heads_ == 4 && dim_ == 128) {  // score bits (4 B) and a 3-pass top-k: the same keys, 2.5x faster top-k at 122K positions
            uint32_t * bits = reinterpret_cast<uint32_t *>(scratch.keys);
            score_bits_k<4, 16, 8><<<dim3(unsigned((blocks + 31) / 32), unsigned((tokens + 15) / 16)), dim3(32, 8), 0, stream>>>(
                keys_.normalized(), scratch.query, bits, blocks, tokens, first_position, ratio_);
            topk_bits_k<<<tokens, 1024, 0, stream>>>(bits, scratch.sorted, blocks, first_position, ratio_, need_cap);
        } else {
            scratch.full();  // 64-bit keys
            qsa_score_pack(keys_.normalized(), scratch.query, scratch.keys, blocks, first_position, ratio_, heads_, dim_, tokens, stream);
            topk_sorted_k<<<tokens, 1024, 0, stream>>>(scratch.keys, scratch.sorted, blocks, first_position, ratio_, need_cap);
        }
        ck(cudaPeekAtLastError());
        stride = kTopkCap;
        if (check) {  // full sort into temp positions, compare the reachable prefix of every query
            std::vector<uint64_t> fast(size_t(tokens) * kTopkCap);
            ck(cudaMemcpyAsync(fast.data(), scratch.sorted, fast.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
            ck(cudaStreamSynchronize(stream)); scratch.full();  // the reference sort needs the full key buffers
            qsa_score_pack(keys_.normalized(), scratch.query, scratch.keys, blocks, first_position, ratio_, heads_, dim_, tokens, stream);  // full keys for the reference sort
            segment_offsets<<<(tokens + 256) / 256, 256, 0, stream>>>(scratch.offsets, tokens, blocks);
            size_t available = scratch.temp_bytes;
            ck(cub::DeviceSegmentedRadixSort::SortKeysDescending(scratch.temp, available, scratch.keys, scratch.sorted, tokens * blocks,
                                                                 tokens, scratch.offsets, scratch.offsets + 1, 0, 64, stream));
            std::vector<uint64_t> full(size_t(tokens) * blocks);
            ck(cudaMemcpyAsync(full.data(), scratch.sorted, full.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
            ck(cudaStreamSynchronize(stream));
            for (int t = 0; t < tokens; ++t) {
                const int need = std::min((first_position + t + 1) / ratio_, need_cap);
                for (int i = 0; i < need; ++i)
                    if (fast[size_t(t) * kTopkCap + i] != full[size_t(t) * blocks + i])
                        throw std::runtime_error("QSA top-k differs from the full sort at query " + std::to_string(first_position + t));
            }
            ck(cudaMemcpyAsync(scratch.sorted, fast.data(), fast.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
            ck(cudaStreamSynchronize(stream));
        }
    } else if (blocks > 0) {
        scratch.full();
        qsa_score_pack(keys_.normalized(), scratch.query, scratch.keys, blocks, first_position, ratio_, heads_, dim_, tokens, stream);
        segment_offsets<<<(tokens + 256) / 256, 256, 0, stream>>>(scratch.offsets, tokens, blocks);
        ck(cudaPeekAtLastError());
        size_t needed = 0;
        ck(cub::DeviceSegmentedRadixSort::SortKeysDescending(nullptr, needed, scratch.keys, scratch.sorted, tokens * blocks, tokens,
                                                             scratch.offsets, scratch.offsets + 1, 0, 64, stream));
        if (needed > scratch.temp_bytes) throw std::runtime_error("QSA indexer: batch sort scratch is too small");
        size_t available = scratch.temp_bytes;
        ck(cub::DeviceSegmentedRadixSort::SortKeysDescending(scratch.temp, available, scratch.keys, scratch.sorted, tokens * blocks,
                                                             tokens, scratch.offsets, scratch.offsets + 1, 0, 64, stream));
    }
    expand_selection_batch<<<dim3((width + 255) / 256, tokens), 256, 0, stream>>>(
        scratch.sorted, scratch.positions, scratch.valid, width, stride, first_position, ratio_);
    ck(cudaPeekAtLastError());
    return {scratch.positions, scratch.valid, width};
}

namespace {
__global__ void gather_mark_k(const int32_t * positions, const uint8_t * valid, int tokens, int width, int first_cells, int * flags) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * width) return;
    const int t = i / width, pos = positions[i];
    if (valid[i] && pos >= 0 && pos < first_cells + t) flags[pos] = 1;
}
// one warp per cell: rows of the marked cells to their compact index (16-byte loads). A host cell (below host_cells) whose
// cache slot holds it is read from the device cache; a host cell that misses claims its slot (lowest cell wins) for the
// fill pass, which runs after every read of this call
__global__ void gather_rows_k(const int * flags, const int * map, int cells, const uint4 * keys, const uint4 * values,
                              uint4 * k_out, uint4 * v_out, int row_vec, int host_cells, int cache_rows, const int * tag,
                              int * claim, const uint4 * ck, const uint4 * cv, unsigned long long * counters) {
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (c >= cells || !flags[c]) return;
    const uint4 * ks = keys + int64_t(c) * row_vec, * vs = values + int64_t(c) * row_vec;
    bool hit = false;
    if (cache_rows > 0 && c < host_cells) {
        const int slot = c % cache_rows;
        hit = tag[slot] == c;
        if (hit) { ks = ck + int64_t(slot) * row_vec; vs = cv + int64_t(slot) * row_vec; }
        else if (lane == 0) atomicMin(claim + slot, c);
    }
    const int64_t dst = int64_t(map[c]) * row_vec;
    for (int j = lane; j < row_vec; j += 32) { k_out[dst + j] = ks[j]; v_out[dst + j] = vs[j]; }
    if (lane == 0 && c < host_cells) { atomicAdd(counters, 1ull); if (hit) atomicAdd(counters + 1, 1ull); }
}
// fill pass: each claimed slot takes its cell's row from the compact copy and its tag
__global__ void gather_fill_k(const int * map, int cells, int cache_rows, int * tag, int * claim, const uint4 * k_out,
                              const uint4 * v_out, uint4 * ck, uint4 * cv, int row_vec) {
    const int slot = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (slot >= cache_rows) return;
    const int c = claim[slot];
    if (c >= cells) return;  // unclaimed (INT_MAX) this call
    const int64_t src = int64_t(map[c]) * row_vec, dst = int64_t(slot) * row_vec;
    for (int j = lane; j < row_vec; j += 32) { ck[dst + j] = k_out[src + j]; cv[dst + j] = v_out[src + j]; }
    __syncwarp();
    if (lane == 0) { tag[slot] = c; claim[slot] = INT_MAX; }
}
__global__ void gather_remap_k(const int32_t * positions, const uint8_t * valid, int tokens, int width, int first_cells,
                               const int * map, int32_t * out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * width) return;
    const int t = i / width, pos = positions[i];
    out[i] = valid[i] && pos >= 0 && pos < first_cells + t ? map[pos] : -1;
}
}  // namespace

QsaGather::QsaGather(int max_cells, size_t row_bytes, int cache_rows, int max_rows) : max_cells_(max_cells), row_bytes_(row_bytes), cache_rows_(std::max(0, cache_rows)) {
    if (max_cells <= 0 || row_bytes == 0 || row_bytes % 16) throw std::invalid_argument("QSA gather: invalid shape");
    ck(cudaMalloc(&counters_, 2 * sizeof(unsigned long long))); ck(cudaMemset(counters_, 0, 2 * sizeof(unsigned long long)));
    if (max_rows > 0) {
        max_rows_ = max_rows;
        ck(cudaMalloc(&k_, size_t(max_rows_) * row_bytes_)); ck(cudaMalloc(&v_, size_t(max_rows_) * row_bytes_));
        ck(cudaMalloc(&pos_, size_t(max_rows_) * sizeof(int32_t)));
    }
    if (cache_rows_) {
        ck(cudaMalloc(&tag_, size_t(cache_rows_) * sizeof(int))); ck(cudaMemset(tag_, 0xff, size_t(cache_rows_) * sizeof(int)));  // -1: empty
        ck(cudaMalloc(&claim_, size_t(cache_rows_) * sizeof(int))); ck(cudaMemset(claim_, 0x7f, size_t(cache_rows_) * sizeof(int)));  // ~INT_MAX
        ck(cudaMalloc(&ck_, size_t(cache_rows_) * row_bytes)); ck(cudaMalloc(&cv_, size_t(cache_rows_) * row_bytes));
    }
    ck(cudaMalloc(&flags_, size_t(max_cells) * sizeof(int)));
    ck(cudaMalloc(&map_, size_t(max_cells) * sizeof(int)));
    ck(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes_, flags_, map_, max_cells));
    ck(cudaMalloc(&temp_, temp_bytes_));
}
QsaGather::~QsaGather() { cudaFree(flags_); cudaFree(map_); cudaFree(temp_); cudaFree(k_); cudaFree(v_); cudaFree(pos_);
                          cudaFree(tag_); cudaFree(claim_); cudaFree(ck_); cudaFree(cv_); cudaFree(counters_); }
void QsaGather::run(const int32_t * positions, const uint8_t * valid, int tokens, int width, int first_cells,
                    const void * keys, const void * values, cudaStream_t stream, int host_cells, int written_from) {
    const int cells = first_cells + tokens - 1;
    if (tokens <= 0 || width <= 0 || cells <= 0) throw std::invalid_argument("QSA gather: invalid call");
    if (cells > max_cells_) {  // the attention state is sized per request: a longer one grows the cell maps
        ck(cudaStreamSynchronize(stream)); cudaFree(flags_); cudaFree(map_); cudaFree(temp_);
        max_cells_ = cells; temp_bytes_ = 0;
        ck(cudaMalloc(&flags_, size_t(max_cells_) * sizeof(int))); ck(cudaMalloc(&map_, size_t(max_cells_) * sizeof(int)));
        ck(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes_, flags_, map_, max_cells_)); ck(cudaMalloc(&temp_, temp_bytes_));
    }
    if (tokens * width > max_rows_) {
        cudaFree(k_); cudaFree(v_); cudaFree(pos_);
        max_rows_ = tokens * width;
        ck(cudaMalloc(&k_, size_t(max_rows_) * row_bytes_)); ck(cudaMalloc(&v_, size_t(max_rows_) * row_bytes_));
        ck(cudaMalloc(&pos_, size_t(max_rows_) * sizeof(int32_t)));
    }
    const int items = tokens * width;
    ck(cudaMemsetAsync(flags_, 0, size_t(cells) * sizeof(int), stream));
    gather_mark_k<<<(items + 255) / 256, 256, 0, stream>>>(positions, valid, tokens, width, first_cells, flags_);
    size_t tb = temp_bytes_; ck(cub::DeviceScan::ExclusiveSum(temp_, tb, flags_, map_, cells, stream));
    const int rv = int(row_bytes_ / 16), cached = cache_rows_ ? std::min(host_cells, cells) : 0;
    // other K/V, or rows rewritten at or below a cell the cache may hold: start empty, never serve a stale row
    if (cache_rows_ && (keys != cached_keys_ || written_from < cached_below_)) {
        ck(cudaMemsetAsync(tag_, 0xff, size_t(cache_rows_) * sizeof(int), stream)); cached_keys_ = keys; cached_below_ = 0;
    }
    cached_below_ = std::max(cached_below_, cached);
    gather_rows_k<<<(int64_t(cells) * 32 + 255) / 256, 256, 0, stream>>>(flags_, map_, cells, static_cast<const uint4 *>(keys),
        static_cast<const uint4 *>(values), static_cast<uint4 *>(k_), static_cast<uint4 *>(v_), rv, std::max(0, host_cells), cached ? cache_rows_ : 0,
        tag_, claim_, static_cast<const uint4 *>(ck_), static_cast<const uint4 *>(cv_), counters_);
    if (cached) gather_fill_k<<<(int64_t(cache_rows_) * 32 + 255) / 256, 256, 0, stream>>>(map_, cells, cache_rows_, tag_, claim_,
        static_cast<const uint4 *>(k_), static_cast<const uint4 *>(v_), static_cast<uint4 *>(ck_), static_cast<uint4 *>(cv_), rv);
    gather_remap_k<<<(items + 255) / 256, 256, 0, stream>>>(positions, valid, tokens, width, first_cells, map_, pos_);
    ck(cudaPeekAtLastError());
}
QsaGather::Stats QsaGather::stats() const {
    unsigned long long c[2] = {0, 0}; ck(cudaMemcpy(c, counters_, sizeof(c), cudaMemcpyDeviceToHost));
    return {c[0], c[1]};
}
} // namespace tqllm_cuda
