#include <string>
#include <cstdlib>
#include <vector>
#include "qwen4exp_kernels.cuh"
#include <cub/device/device_radix_sort.cuh>
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

// score_pack_batch for 4 indexer heads of 128: a warp scores RB blocks at once with q in registers, and the RB * 4 dots
// reduce in a transposing butterfly (the pairs of the shfl_down tree; a + b == b + a) instead of 5 shuffles each.
// Same keys bit for bit.
template<int RB>
__global__ void score_pack_fixed_k(const float * pooled_keys, const float * query, uint64_t * keys, int blocks, int first_position, int ratio) {
    constexpr int H = 4, DIM = 128, J = DIM / 32;
    static_assert(RB == 4, "lane mapping below assumes 16 values per lane");
    const int lane = threadIdx.x, t = blockIdx.y, block0 = (blockIdx.x * blockDim.y + threadIdx.y) * RB;
    if (block0 >= blocks) return;
    uint64_t * out = keys + int64_t(t) * blocks;
    const int complete = (first_position + t + 1) / ratio;
    const float * q = query + int64_t(t) * H * DIM;
    float qr[H][J];
    #pragma unroll
    for (int h = 0; h < H; ++h)
        #pragma unroll
        for (int j = 0; j < J; ++j) qr[h][j] = q[h * DIM + lane + 32 * j];
    float a[RB * H];
    #pragma unroll
    for (int b = 0; b < RB; ++b) {
        const float * k = pooled_keys + int64_t(min(block0 + b, blocks - 1)) * DIM;
        float kv[J];
        #pragma unroll
        for (int j = 0; j < J; ++j) kv[j] = k[lane + 32 * j];
        #pragma unroll
        for (int h = 0; h < H; ++h) {
            float dot = 0.f;
            #pragma unroll
            for (int j = 0; j < J; ++j) dot = fmaf(kv[j], qr[h][j], dot);
            a[b * H + h] = dot;
        }
    }
    // 16 values: halve at offsets 16, 8, 4, 2; lane bits 4..1 then name (block, head) = (bit4 bit3, bit2 bit1)
    #pragma unroll
    for (int off = 16, n = RB * H; off >= 2; off >>= 1, n >>= 1) {
        const bool upper = lane & off; const int half = n / 2;
        #pragma unroll
        for (int j = 0; j < half; ++j) {
            const float mine = upper ? a[j + half] : a[j], send = upper ? a[j] : a[j + half];
            a[j] = mine + __shfl_xor_sync(0xffffffffu, send, off);
        }
    }
    const float dot = a[0] + __shfl_xor_sync(0xffffffffu, a[0], 1);
    // heads 1..3 of this lane's block sit 2, 4, 6 lanes up; the total adds them in head order as score_pack_batch does
    const float d1 = __shfl_down_sync(0xffffffffu, dot, 2), d2 = __shfl_down_sync(0xffffffffu, dot, 4), d3 = __shfl_down_sync(0xffffffffu, dot, 6);
    if ((lane & 7) == 0) {
        const int b = ((lane >> 4) << 1) | ((lane >> 3) & 1), block = block0 + b;
        if (block >= blocks) return;
        if (block >= complete) { out[block] = 0; return; }
        float total = 0.f;
        total += fmaxf(dot, 0.f); total += fmaxf(d1, 0.f); total += fmaxf(d2, 0.f); total += fmaxf(d3, 0.f);
        const float score = total == 0.f ? 0.f : total;
        out[block] = (uint64_t(__float_as_uint(score)) << 32) | uint32_t(~uint32_t(block));
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

__global__ void attention_direct(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                                 const int32_t * positions, const uint8_t * valid, float * output,
                                 int width, int first_cells, int heads, int kv_heads, int dim) {
    const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kv_head = head / (heads / kv_heads), cells = first_cells + t;
    const float * q = query + (int64_t(t) * heads + head) * dim;
    positions += int64_t(t) * width; valid += int64_t(t) * width; output += (int64_t(t) * heads + head) * dim;
    extern __shared__ float scores[];
    int * cell = reinterpret_cast<int *>(scores + width);  // -1: not read (a zero row in the gathered layout)
    __shared__ float reduction[128];
    for (int row = tid; row < width; row += 128) {
        const int pos = positions[row];
        cell[row] = valid[row] && pos >= 0 && pos < cells ? pos : -1;
    }
    __syncthreads();
    for (int row = warp; row < width; row += 4) {
        float dot = 0.f;
        if (cell[row] >= 0)
            for (int d = lane; d < dim; d += 32)
                dot = fmaf(q[d], __bfloat162float(keys[(int64_t(cell[row]) * kv_heads + kv_head) * dim + d]), dot);
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
        for (int d = tid; d < dim; d += 128) output[d] = 0.f;
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
            if (cell[row] >= 0)  // skipped rows add fmaf(w, 0, value) == value in the gathered layout
                value = fmaf(scores[row], __bfloat162float(values[(int64_t(cell[row]) * kv_heads + kv_head) * dim + d]), value);
        output[d] = value * inv;
    }
}
// Grouped selected attention, three passes with the arithmetic of attention_direct (same dot order, same reductions):
// scores per kv head (K rows read once for every query head of the group), softmax per query head, values per kv head.
constexpr int kQsaMaxGroup = 16;
__global__ void grouped_scores_k(const float * query, const __nv_bfloat16 * keys, const int32_t * positions, const uint8_t * valid,
                                 float * scores, int width, int first_cells, int heads, int kv_heads, int dim) {
    const int kv_head = blockIdx.x, t = blockIdx.y, group = heads / kv_heads, lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5, warps = blockDim.x >> 5, cells = first_cells + t;
    __shared__ float q[kQsaMaxGroup * 256];
    for (int i = threadIdx.x; i < group * dim; i += blockDim.x)
        q[i] = query[(int64_t(t) * heads + kv_head * group) * dim + i];
    __syncthreads();
    positions += int64_t(t) * width; valid += int64_t(t) * width;
    for (int row = blockIdx.z * warps + warp; row < width; row += gridDim.z * warps) {
        const int pos = positions[row];
        const bool fetch = valid[row] && pos >= 0 && pos < cells;
        float dot[kQsaMaxGroup];  // registers: every loop over it is unrolled to the cap
        #pragma unroll
        for (int g = 0; g < kQsaMaxGroup; ++g) dot[g] = 0.f;
        if (fetch)
            for (int d = lane; d < dim; d += 32) {
                const float k = __bfloat162float(keys[(int64_t(pos) * kv_heads + kv_head) * dim + d]);
                #pragma unroll
                for (int g = 0; g < kQsaMaxGroup; ++g) if (g < group) dot[g] = fmaf(q[g * dim + d], k, dot[g]);
            }
        #pragma unroll
        for (int g = 0; g < kQsaMaxGroup; ++g) {
            if (g >= group) break;
            float v = dot[g];
            for (int delta = 16; delta; delta >>= 1) v += __shfl_down_sync(0xffffffffu, v, delta);
            if (lane == 0)
                scores[(int64_t(t) * heads + kv_head * group + g) * width + row] = valid[row] ? v * rsqrtf(float(dim)) : -INFINITY;
        }
    }
}

__global__ void grouped_softmax_k(float * scores, const uint8_t * valid, float * inv, int width, int heads) {
    const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    float * s = scores + (int64_t(t) * heads + head) * width;
    valid += int64_t(t) * width;
    __shared__ float reduction[128];
    float maximum = -INFINITY;
    for (int row = tid; row < width; row += 128) maximum = fmaxf(maximum, s[row]);
    reduction[tid] = maximum;
    __syncthreads();
    for (int delta = 64; delta; delta >>= 1) {
        if (tid < delta) reduction[tid] = fmaxf(reduction[tid], reduction[tid + delta]);
        __syncthreads();
    }
    maximum = reduction[0];
    __syncthreads();
    const bool any = isfinite(maximum);
    float sum = 0.f;
    for (int row = tid; row < width; row += 128) {
        const float weight = any && valid[row] ? expf(s[row] - maximum) : 0.f;
        s[row] = weight;
        sum += weight;
    }
    reduction[tid] = sum;
    __syncthreads();
    for (int delta = 64; delta; delta >>= 1) {
        if (tid < delta) reduction[tid] += reduction[tid + delta];
        __syncthreads();
    }
    if (tid == 0) inv[int64_t(t) * heads + head] = any ? 1.f / reduction[0] : 0.f;
}

__global__ void grouped_values_k(const float * weights, const float * inv, const __nv_bfloat16 * values, const int32_t * positions,
                                 const uint8_t * valid, float * output, int width, int first_cells, int heads, int kv_heads, int dim) {
    const int kv_head = blockIdx.x, t = blockIdx.y, group = heads / kv_heads, cells = first_cells + t;
    positions += int64_t(t) * width; valid += int64_t(t) * width;
    const float * w = weights + (int64_t(t) * heads + kv_head * group) * width;
    constexpr int TILE = 128;
    __shared__ float tw[kQsaMaxGroup][TILE];
    __shared__ int tcell[TILE];
    const int d = threadIdx.x;  // blockDim.x == dim
    float acc[kQsaMaxGroup];
    for (int g = 0; g < kQsaMaxGroup; ++g) acc[g] = 0.f;
    for (int base = 0; base < width; base += TILE) {
        const int rows = min(TILE, width - base);
        __syncthreads();
        for (int i = threadIdx.x; i < rows; i += blockDim.x) {
            const int pos = positions[base + i];
            tcell[i] = valid[base + i] && pos >= 0 && pos < cells ? pos : -1;  // -1 adds fmaf(w, 0, acc) == acc
        }
        for (int i = threadIdx.x; i < group * rows; i += blockDim.x) tw[i / rows][i % rows] = w[int64_t(i / rows) * width + base + i % rows];
        __syncthreads();
        if (d < dim)
            for (int r = 0; r < rows; ++r) {
                const int pos = tcell[r];
                if (pos < 0) continue;
                const float v = __bfloat162float(values[(int64_t(pos) * kv_heads + kv_head) * dim + d]);
                #pragma unroll
                for (int g = 0; g < kQsaMaxGroup; ++g) if (g < group) acc[g] = fmaf(tw[g][r], v, acc[g]);
            }
    }
    if (d < dim)
        #pragma unroll
        for (int g = 0; g < kQsaMaxGroup; ++g) {
            if (g >= group) break;
            const int head = kv_head * group + g; const float iv = inv[int64_t(t) * heads + head];
            output[(int64_t(t) * heads + head) * dim + d] = iv == 0.f ? 0.f : acc[g] * iv;
        }
}

// grouped_scores_k / grouped_values_k with the group and head size fixed: same per-row arithmetic (each dot over d =
// lane, lane + 32, ... in order, the same shuffles), but a warp keeps RW rows in flight and reads each q value once for
// all of them; values take 4 rows per step and their weights as float4.
template<int G, int DIM, int RW>
__global__ void __launch_bounds__(256) grouped_scores_fixed_k(const float * query, const __nv_bfloat16 * keys, const int32_t * positions,
                                                              const uint8_t * valid, float * scores, int width, int first_cells, int kv_heads) {
    const int kv_head = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31, heads = G * kv_heads;
    const int warp = threadIdx.x >> 5, warps = blockDim.x >> 5, cells = first_cells + t;
    __shared__ float q[G * DIM];
    for (int i = threadIdx.x; i < G * DIM; i += blockDim.x) q[i] = query[(int64_t(t) * heads + kv_head * G) * DIM + i];
    __syncthreads();
    positions += int64_t(t) * width; valid += int64_t(t) * width;
    const float scale = rsqrtf(float(DIM));
    for (int row0 = (blockIdx.z * warps + warp) * RW; row0 < width; row0 += gridDim.z * warps * RW) {
        const __nv_bfloat16 * kr[RW]; bool fetch[RW]; unsigned okbits = 0;
        #pragma unroll
        for (int i = 0; i < RW; ++i) {
            const int row = row0 + i, pos = row < width ? positions[row] : -1;
            const bool ok = row < width && valid[row];
            okbits |= unsigned(ok) << i;
            fetch[i] = ok && pos >= 0 && pos < cells;
            kr[i] = keys + (int64_t(fetch[i] ? pos : 0) * kv_heads + kv_head) * DIM;
        }
        float dot[RW][G];
        #pragma unroll
        for (int i = 0; i < RW; ++i)
            #pragma unroll
            for (int g = 0; g < G; ++g) dot[i][g] = 0.f;
        #pragma unroll
        for (int d = lane; d < DIM; d += 32) {
            float k[RW];
            #pragma unroll
            for (int i = 0; i < RW; ++i) k[i] = fetch[i] ? __bfloat162float(kr[i][d]) : 0.f;
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                const float qv = q[g * DIM + d];
                #pragma unroll
                for (int i = 0; i < RW; ++i) if (fetch[i]) dot[i][g] = fmaf(qv, k[i], dot[i][g]);
            }
        }
        // Transposing butterfly: each step pairs lanes l and l ^ off (the pairs of the shfl_down tree; a + b == b + a). While
        // a lane's count is even it keeps half and trades half, so RW * G sums take ~RW * G shuffles instead of 5 * RW * G.
        float * a = &dot[0][0];
        int base = 0, plain = 0, n = RW * G;
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            if (n % 2 == 0) {
                const bool upper = lane & off; const int half = n / 2;
                #pragma unroll
                for (int j = 0; j < half; ++j) {
                    const float mine = upper ? a[j + half] : a[j], send = upper ? a[j] : a[j + half];
                    a[j] = mine + __shfl_xor_sync(0xffffffffu, send, off);
                }
                if (upper) base += half;
                n = half;
            } else {
                #pragma unroll
                for (int j = 0; j < n; ++j) a[j] += __shfl_xor_sync(0xffffffffu, a[j], off);
                plain |= off;
            }
        }
        if (!(lane & plain))
            #pragma unroll
            for (int j = 0; j < n; ++j) {
                const int idx = base + j, i = idx / G, g = idx % G;
                if (row0 + i < width) scores[(int64_t(t) * heads + kv_head * G + g) * width + row0 + i] = (okbits >> i) & 1 ? a[j] * scale : -INFINITY;
            }
    }
}

template<int G, int DIM>
__global__ void __launch_bounds__(DIM) grouped_values_fixed_k(const float * weights, const float * inv, const __nv_bfloat16 * values,
                                                              const int32_t * positions, const uint8_t * valid, float * output,
                                                              int width, int first_cells, int kv_heads) {
    static_assert(G % 4 == 0, "weights are read as float4");
    const int kv_head = blockIdx.x, t = blockIdx.y, heads = G * kv_heads, cells = first_cells + t;
    positions += int64_t(t) * width; valid += int64_t(t) * width;
    const float * w = weights + (int64_t(t) * heads + kv_head * G) * width;
    constexpr int TILE = 128;
    __shared__ __align__(16) float tw[TILE][G];
    __shared__ int tcell[TILE];
    const int d = threadIdx.x;  // blockDim.x == DIM
    float acc[G];
    #pragma unroll
    for (int g = 0; g < G; ++g) acc[g] = 0.f;
    const __nv_bfloat16 * vcol = values + int64_t(kv_head) * DIM + d;
    for (int base = 0; base < width; base += TILE) {
        const int rows = min(TILE, width - base);
        __syncthreads();
        for (int i = threadIdx.x; i < TILE; i += DIM) {
            const int pos = i < rows ? positions[base + i] : -1;
            tcell[i] = i < rows && valid[base + i] && pos >= 0 && pos < cells ? pos : -1;  // -1 adds fmaf(w, 0, acc) == acc
        }
        for (int i = threadIdx.x; i < G * TILE; i += DIM) {
            const int g = i / TILE, r = i % TILE;
            tw[r][g] = r < rows ? w[int64_t(g) * width + base + r] : 0.f;
        }
        __syncthreads();
        for (int r0 = 0; r0 < rows; r0 += 4) {
            float v[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int pos = tcell[r0 + j];
                v[j] = pos >= 0 ? __bfloat162float(vcol[int64_t(pos) * kv_heads * DIM]) : 0.f;
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (tcell[r0 + j] < 0) continue;
                #pragma unroll
                for (int g4 = 0; g4 < G; g4 += 4) {
                    const float4 wq = *reinterpret_cast<const float4 *>(&tw[r0 + j][g4]);
                    acc[g4] = fmaf(wq.x, v[j], acc[g4]); acc[g4 + 1] = fmaf(wq.y, v[j], acc[g4 + 1]);
                    acc[g4 + 2] = fmaf(wq.z, v[j], acc[g4 + 2]); acc[g4 + 3] = fmaf(wq.w, v[j], acc[g4 + 3]);
                }
            }
        }
    }
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        const int head = kv_head * G + g; const float iv = inv[int64_t(t) * heads + head];
        output[(int64_t(t) * heads + head) * DIM + d] = iv == 0.f ? 0.f : acc[g] * iv;
    }
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

} // namespace

QsaBatchScratch::QsaBatchScratch(int max_queries_, int max_cells, int ratio, int top_k)
    : max_queries(max_queries_), max_blocks(std::max(1, max_cells / ratio)),
      width_cap(int(std::min<int64_t>(max_cells, int64_t(top_k) + ratio - 1))) {
    if (max_queries <= 0 || max_cells <= 0 || ratio <= 0 || top_k <= 0)
        throw std::invalid_argument("QSA batch scratch: invalid dimensions");
    const size_t items = size_t(max_queries) * max_blocks;
    try {
        ck(cudaMalloc(&query, size_t(max_queries) * 4 * 128 * sizeof(float)));
        ck(cudaMalloc(&keys, items * sizeof(uint64_t)));
        ck(cudaMalloc(&sorted, items * sizeof(uint64_t)));
        ck(cudaMalloc(&offsets, size_t(max_queries + 1) * sizeof(int)));
        ck(cub::DeviceSegmentedRadixSort::SortKeysDescending(nullptr, temp_bytes, keys, sorted, int(items), max_queries,
                                                             offsets, offsets + 1));
        ck(cudaMalloc(&temp, temp_bytes));
        ck(cudaMalloc(&positions, size_t(max_queries) * width_cap * sizeof(int32_t)));
        ck(cudaMalloc(&valid, size_t(max_queries) * width_cap));
    } catch (...) {
        cudaFree(valid); cudaFree(positions); cudaFree(temp); cudaFree(offsets); cudaFree(sorted); cudaFree(keys); cudaFree(query);
        throw;
    }
}

QsaBatchScratch::~QsaBatchScratch() {
    cudaFree(valid); cudaFree(positions); cudaFree(temp); cudaFree(offsets); cudaFree(sorted); cudaFree(keys); cudaFree(query);
}

void qsa_score_pack(const float * pooled_keys, const float * query, uint64_t * keys, int blocks, int first_position, int ratio,
                    int heads, int dim, int tokens, cudaStream_t stream, int fixed) {
    if (fixed < 0) fixed = 1;  // fixed-shape kernels (bitwise equal to the generic ones, test-qsa-pack-exact)
    if (fixed && heads == 4 && dim == 128)
        score_pack_fixed_k<4><<<dim3(unsigned((blocks + 15) / 16), unsigned(tokens)), dim3(32, 4), 0, stream>>>(pooled_keys, query, keys, blocks, first_position, ratio);
    else
        score_pack_batch<<<dim3((blocks + 3) / 4, tokens), dim3(32, 4), 0, stream>>>(pooled_keys, query, keys, blocks, first_position, ratio, heads, dim);
    ck(cudaPeekAtLastError());
}

void qsa_attention_direct(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                          const QsaDeviceSelection & selected, float * output, int tokens, int first_cells,
                          int heads, int kv_heads, int dim, cudaStream_t stream) {
    if (!query || !keys || !values || !selected.positions || !selected.valid || !output || tokens <= 0 || tokens > 65535 ||
        selected.width <= 0 || selected.width > 8192 || heads <= 0 || kv_heads <= 0 || heads % kv_heads || dim <= 0 || dim > 256)
        throw std::invalid_argument("QSA direct attention: invalid dimensions or pointer");
    attention_direct<<<dim3(heads, tokens), 128, size_t(selected.width) * (sizeof(float) + sizeof(int)), stream>>>(
        query, keys, values, selected.positions, selected.valid, output, selected.width, first_cells, heads, kv_heads, dim);
    ck(cudaPeekAtLastError());
}

void qsa_attention_grouped(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                           const QsaDeviceSelection & selected, float * scores, float * inv, float * output, int tokens,
                           int first_cells, int heads, int kv_heads, int dim, cudaStream_t stream) {
    if (!query || !keys || !values || !selected.positions || !selected.valid || !scores || !inv || !output || tokens <= 0 ||
        tokens > 65535 || selected.width <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads ||
        heads / kv_heads > kQsaMaxGroup || dim <= 0 || dim > 256)
        throw std::invalid_argument("QSA grouped attention: invalid dimensions or pointer");
    if (heads / kv_heads == 12 && dim == 256) {  // fixed-shape kernel (bitwise equal to the generic one)
        constexpr int rw_env = 1;  // rows per warp (2 and 4 measured 2-4x slower on the 3090)
        const int RW = rw_env == 4 ? 4 : rw_env == 1 ? 1 : 2;
        const int blocks_z = std::max(1, std::min(8, (selected.width + 8 * RW * 8 - 1) / (8 * RW * 8)));
        const dim3 sg(kv_heads, tokens, blocks_z);
        if (RW == 4) grouped_scores_fixed_k<12, 256, 4><<<sg, 256, 0, stream>>>(query, keys, selected.positions, selected.valid, scores, selected.width, first_cells, kv_heads);
        else if (RW == 2) grouped_scores_fixed_k<12, 256, 2><<<sg, 256, 0, stream>>>(query, keys, selected.positions, selected.valid, scores, selected.width, first_cells, kv_heads);
        else grouped_scores_fixed_k<12, 256, 1><<<sg, 256, 0, stream>>>(query, keys, selected.positions, selected.valid, scores, selected.width, first_cells, kv_heads);
        ck(cudaPeekAtLastError());
        grouped_softmax_k<<<dim3(heads, tokens), 128, 0, stream>>>(scores, selected.valid, inv, selected.width, heads);
        ck(cudaPeekAtLastError());
        grouped_values_fixed_k<12, 256><<<dim3(kv_heads, tokens), 256, 0, stream>>>(
            scores, inv, values, selected.positions, selected.valid, output, selected.width, first_cells, kv_heads);
        ck(cudaPeekAtLastError());
        return;
    }
    const int row_blocks = std::max(1, std::min(32, (selected.width + 63) / 64));
    grouped_scores_k<<<dim3(kv_heads, tokens, row_blocks), 256, 0, stream>>>(query, keys, selected.positions, selected.valid, scores,
                                                                             selected.width, first_cells, heads, kv_heads, dim);
    ck(cudaPeekAtLastError());
    grouped_softmax_k<<<dim3(heads, tokens), 128, 0, stream>>>(scores, selected.valid, inv, selected.width, heads);
    ck(cudaPeekAtLastError());
    grouped_values_k<<<dim3(kv_heads, tokens), dim, 0, stream>>>(scores, inv, values, selected.positions, selected.valid, output,
                                                                 selected.width, first_cells, heads, kv_heads, dim);
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
        qsa_score_pack(keys_.normalized(), scratch.query, scratch.keys, blocks, first_position, ratio_, heads_, dim_, tokens, stream);
        topk_sorted_k<<<tokens, 1024, 0, stream>>>(scratch.keys, scratch.sorted, blocks, first_position, ratio_, need_cap);
        ck(cudaPeekAtLastError());
        stride = kTopkCap;
        if (check) {  // full sort into temp positions, compare the reachable prefix of every query
            std::vector<uint64_t> fast(size_t(tokens) * kTopkCap);
            ck(cudaMemcpyAsync(fast.data(), scratch.sorted, fast.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
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
} // namespace tqllm_cuda
