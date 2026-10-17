// QSA selected attention (Flash-Next's sparse attention over the indexer's chosen positions) on BF16 K/V rows: the
// decode / verify kernels (direct, split + merge) and the prefill kernels (grouped scalar, tensor-core). Moved from the
// model folder so the KV formats own sparse attention (AGENTS.md rule 4); the arithmetic is unchanged.
#include <string>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>
#include "kv_qsa.cuh"

namespace tqllm_cuda {
namespace {
void ck(cudaError_t e, int line = __builtin_LINE()) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(cudaGetErrorString(e)) + " (kv_qsa.cu:" + std::to_string(line) + ")");
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
// TQFN_QSA_SPLIT=S: decode attention split over the selected positions (flash-decoding): block (head, row, slice) scores its
// width / S positions and writes its max, its exp-sum and its weighted value sum; attention_merge combines the S slices.
// heads * rows blocks (72 at verify) leave most SMs idle; S = 8 gives 576. Not bit-identical to attention_direct (the
// softmax and value sums are split), a numeric flag.
__global__ void attention_split(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                                const int32_t * positions, const uint8_t * valid, float * part, int width, int first_cells,
                                int heads, int kv_heads, int dim, int splits) {
    const int head = blockIdx.x, t = blockIdx.y, sp = blockIdx.z, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kv_head = head / (heads / kv_heads), cells = first_cells + t;
    const int per = (width + splits - 1) / splits, r0 = sp * per, r1 = min(width, r0 + per), nr = max(0, r1 - r0);
    const float * q = query + (int64_t(t) * heads + head) * dim;
    positions += int64_t(t) * width; valid += int64_t(t) * width;
    float * out = part + ((int64_t(t) * heads + head) * splits + sp) * (dim + 2);
    extern __shared__ float scores[];
    int * cell = reinterpret_cast<int *>(scores + per);
    __shared__ float reduction[128];
    for (int i = tid; i < nr; i += 128) {
        const int row = r0 + i, pos = positions[row];
        cell[i] = valid[row] && pos >= 0 && pos < cells ? pos : -1;
    }
    __syncthreads();
    float qreg[8];
    const int nd = (dim + 31) / 32;
#pragma unroll
    for (int i = 0; i < 8; ++i) qreg[i] = i < nd && lane + 32 * i < dim ? q[lane + 32 * i] : 0.f;
    constexpr int R = 4;
    for (int i0 = warp * R; i0 < nr; i0 += 4 * R) {
        float dot[R];
#pragma unroll
        for (int j = 0; j < R; ++j) {
            dot[j] = 0.f; const int i = i0 + j;
            if (i < nr && cell[i] >= 0) {
                const __nv_bfloat16 * kr = keys + (int64_t(cell[i]) * kv_heads + kv_head) * dim;
#pragma unroll
                for (int k = 0; k < 8; ++k) if (k < nd && lane + 32 * k < dim) dot[j] = fmaf(qreg[k], __bfloat162float(kr[lane + 32 * k]), dot[j]);
            }
        }
#pragma unroll
        for (int j = 0; j < R; ++j) for (int delta = 16; delta; delta >>= 1) dot[j] += __shfl_down_sync(0xffffffffu, dot[j], delta);
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < R; ++j) { const int i = i0 + j; if (i < nr) scores[i] = valid[r0 + i] ? dot[j] * rsqrtf(float(dim)) : -INFINITY; }
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (int i = tid; i < nr; i += 128) maximum = fmaxf(maximum, scores[i]);
    reduction[tid] = maximum;
    __syncthreads();
    for (int delta = 64; delta; delta >>= 1) { if (tid < delta) reduction[tid] = fmaxf(reduction[tid], reduction[tid + delta]); __syncthreads(); }
    maximum = reduction[0];
    __syncthreads();
    float sum = 0.f;
    if (isfinite(maximum))
        for (int i = tid; i < nr; i += 128) { const float w = valid[r0 + i] ? expf(scores[i] - maximum) : 0.f; scores[i] = w; sum += w; }
    reduction[tid] = sum;
    __syncthreads();
    for (int delta = 64; delta; delta >>= 1) { if (tid < delta) reduction[tid] += reduction[tid + delta]; __syncthreads(); }
    if (tid == 0) { out[dim] = maximum; out[dim + 1] = reduction[0]; }
    for (int d = tid; d < dim; d += 128) {
        float value = 0.f;
        if (isfinite(maximum)) {
            constexpr int U = 8; int i = 0;
            for (; i + U <= nr; i += U) {
                float v[U];
#pragma unroll
                for (int j = 0; j < U; ++j) { const int cr = cell[i + j]; v[j] = cr >= 0 ? __bfloat162float(values[(int64_t(cr) * kv_heads + kv_head) * dim + d]) : 0.f; }
#pragma unroll
                for (int j = 0; j < U; ++j) value = fmaf(scores[i + j], v[j], value);
            }
            for (; i < nr; ++i) if (cell[i] >= 0) value = fmaf(scores[i], __bfloat162float(values[(int64_t(cell[i]) * kv_heads + kv_head) * dim + d]), value);
        }
        out[d] = value;
    }
}
__global__ void attention_merge(const float * part, float * output, int heads, int dim, int splits) {
    const int head = blockIdx.x, t = blockIdx.y;
    const float * p = part + (int64_t(t) * heads + head) * splits * (dim + 2);
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, p[s * (dim + 2) + dim]);
    float * o = output + (int64_t(t) * heads + head) * dim;
    if (!isfinite(m)) { for (int d = threadIdx.x; d < dim; d += blockDim.x) o[d] = 0.f; return; }
    float l = 0.f;
    for (int s = 0; s < splits; ++s) { const float ms = p[s * (dim + 2) + dim]; if (isfinite(ms)) l += p[s * (dim + 2) + dim + 1] * expf(ms - m); }
    const float inv = 1.f / l;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < splits; ++s) { const float ms = p[s * (dim + 2) + dim]; if (isfinite(ms)) v = fmaf(p[s * (dim + 2) + d], expf(ms - m), v); }
        o[d] = v * inv;
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

} // namespace

static int g_qsa_split = -1;  // -1: TQFN_QSA_SPLIT, else qsa_set_split()
constexpr int64_t kQsaSplitPart = int64_t(16) * 32 * 16 * (256 + 2);  // rows x heads x splits x (dim + 2)
void qsa_set_split(int splits) { g_qsa_split = splits; }
// TQFN_QSA_SPLIT (default 16; 0 or 1 = attention_direct): slices per decode attention row
static int qsa_split_count() {
    static const int env = std::getenv("TQFN_QSA_SPLIT") ? std::atoi(std::getenv("TQFN_QSA_SPLIT")) : 16;
    return g_qsa_split >= 0 ? g_qsa_split : env;
}
static float * g_qsa_split_part = nullptr;
static float * qsa_split_scratch(cudaStream_t stream) {
    if (!g_qsa_split_part) {
        cudaStreamCaptureStatus cs; cudaStreamIsCapturing(stream, &cs);
        if (cs != cudaStreamCaptureStatusNone) throw std::runtime_error("QSA split: scratch must be allocated outside capture");
        ck(cudaMalloc(&g_qsa_split_part, size_t(kQsaSplitPart) * sizeof(float)));
    }
    return g_qsa_split_part;
}
// at start-up, before the expert banks take the free memory (allocated mid-request it can fail on a full device)
void qsa_reserve_split_scratch() { if (qsa_split_count() > 1) qsa_split_scratch(nullptr); }
void qsa_attention_direct(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                          const QsaDeviceSelection & selected, float * output, int tokens, int first_cells,
                          int heads, int kv_heads, int dim, cudaStream_t stream) {
    if (!query || !keys || !values || !selected.positions || !selected.valid || !output || tokens <= 0 || tokens > 65535 ||
        selected.width <= 0 || selected.width > 8192 || heads <= 0 || kv_heads <= 0 || heads % kv_heads || dim <= 0 || dim > 256)
        throw std::invalid_argument("QSA direct attention: invalid dimensions or pointer");
    const int splits = qsa_split_count();
    if (splits > 1 && dim <= 256 && int64_t(tokens) * heads * splits * (dim + 2) <= kQsaSplitPart) {
        float * part = qsa_split_scratch(stream);
        const int per = (selected.width + splits - 1) / splits;
        attention_split<<<dim3(heads, tokens, splits), 128, size_t(per) * (sizeof(float) + sizeof(int)), stream>>>(
            query, keys, values, selected.positions, selected.valid, part, selected.width, first_cells, heads, kv_heads, dim, splits);
        attention_merge<<<dim3(heads, tokens), 128, 0, stream>>>(part, output, heads, dim, splits);
        ck(cudaPeekAtLastError());
        return;
    }
    attention_direct<<<dim3(heads, tokens), 128, size_t(selected.width) * (sizeof(float) + sizeof(int)), stream>>>(
        query, keys, values, selected.positions, selected.valid, output, selected.width, first_cells, heads, kv_heads, dim);
    ck(cudaPeekAtLastError());
}

// Tensor-core prefill selected attention at the Flash-Next shape (12 query heads per kv head, dim 256) on the tensor cores.
// Block = (kv head, query token): the 12 heads' queries as one 16 x 256 bf16 tile; the token's selected keys / values are
// gathered 32 rows at a time into shared memory; S = Q K^T and O += P V are mma.m16n8k16 (bf16 in, fp32 out) with an
// online softmax between them (scores never leave the block). Numeric (bf16 Q and P), not bit-identical to the grouped path.
namespace qsatc {
constexpr int G = 12, D = 256, TK = 32, QP = D + 8, KP = D + 8, PP = TK + 8, NT = 128;
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return uint32_t(__bfloat16_as_ushort(lo)) | (uint32_t(__bfloat16_as_ushort(hi)) << 16);
}
constexpr size_t smem_bytes() { return size_t(16 * QP + 2 * TK * KP + 16 * PP) * 2 + size_t(16 * TK + 3 * 16) * 4 + 2 * TK * 4; }
}  // namespace qsatc
__global__ void __launch_bounds__(qsatc::NT) qsa_prefill_tc_k(const float * __restrict__ query, const __nv_bfloat16 * __restrict__ keys,
                                                           const __nv_bfloat16 * __restrict__ values, const int32_t * __restrict__ positions,
                                                           const uint8_t * __restrict__ valid, float * __restrict__ output,
                                                           int width, int first_cells, int kv_heads) {
    using namespace qsatc;
    extern __shared__ __align__(16) unsigned char sm_raw[];
    __nv_bfloat16 * Qs = reinterpret_cast<__nv_bfloat16 *>(sm_raw);
    __nv_bfloat16 * Ks = Qs + 16 * QP, * Vs = Ks + TK * KP, * Ps = Vs + TK * KP;
    float * Ss = reinterpret_cast<float *>(Ps + 16 * PP), * mrow = Ss + 16 * TK, * lrow = mrow + 16, * arow = lrow + 16;
    int * cell = reinterpret_cast<int *>(arow + 16), * vflag = cell + TK;  // vflag: the row is valid (a score, 0 when not read)
    const int kv_head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tq = lane & 3;
    const int heads = G * kv_heads, cells = first_cells + t;
    positions += int64_t(t) * width; valid += int64_t(t) * width;
    const float scale = rsqrtf(float(D));
    for (int i = tid; i < 16 * D; i += NT) {  // rows 12..15 zero
        const int r = i / D, d = i % D;
        Qs[r * QP + d] = __float2bfloat16(r < G ? query[(int64_t(t) * heads + kv_head * G + r) * D + d] * scale : 0.f);
    }
    if (tid < 16) { mrow[tid] = -INFINITY; lrow[tid] = 0.f; }
    float acc[8][4];  // warp: output dims warp * 64 .. + 63 (8 n8 tiles), rows g / g + 8
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    for (int base = 0; base < width; base += TK) {
        __syncthreads();  // previous tile's K / V / P reads are done
        if (tid < TK) {
            const int row = base + tid, pos = row < width ? positions[row] : -1;
            vflag[tid] = row < width && valid[row];
            cell[tid] = vflag[tid] && pos >= 0 && pos < cells ? pos : -1;
        }
        __syncthreads();
        for (int i = tid; i < TK * (D / 8); i += NT) {  // 16-byte pieces of the gathered rows (zero for unread rows)
            const int r = i / (D / 8), c8 = (i % (D / 8)) * 8, pos = cell[r];
            uint4 kk = make_uint4(0, 0, 0, 0), vv = kk;
            if (pos >= 0) {
                kk = __ldg(reinterpret_cast<const uint4 *>(keys + (int64_t(pos) * kv_heads + kv_head) * D + c8));
                vv = __ldg(reinterpret_cast<const uint4 *>(values + (int64_t(pos) * kv_heads + kv_head) * D + c8));
            }
            *reinterpret_cast<uint4 *>(Ks + r * KP + c8) = kk; *reinterpret_cast<uint4 *>(Vs + r * KP + c8) = vv;
        }
        __syncthreads();
        {   // S = Q K^T: warp w -> keys w * 8 .. + 7 (one n8 tile), k over D
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            const int n0 = warp * 8;
#pragma unroll 4
            for (int k0 = 0; k0 < D; k0 += 16) {
                uint32_t a[4];
                a[0] = *reinterpret_cast<const uint32_t *>(Qs + g * QP + k0 + tq * 2);
                a[1] = *reinterpret_cast<const uint32_t *>(Qs + (g + 8) * QP + k0 + tq * 2);
                a[2] = *reinterpret_cast<const uint32_t *>(Qs + g * QP + k0 + tq * 2 + 8);
                a[3] = *reinterpret_cast<const uint32_t *>(Qs + (g + 8) * QP + k0 + tq * 2 + 8);
                const uint32_t b0 = *reinterpret_cast<const uint32_t *>(Ks + (n0 + g) * KP + k0 + tq * 2);
                const uint32_t b1 = *reinterpret_cast<const uint32_t *>(Ks + (n0 + g) * KP + k0 + tq * 2 + 8);
                mma16816(c, a, b0, b1);
            }
            Ss[g * TK + n0 + tq * 2] = c[0]; Ss[g * TK + n0 + tq * 2 + 1] = c[1];
            Ss[(g + 8) * TK + n0 + tq * 2] = c[2]; Ss[(g + 8) * TK + n0 + tq * 2 + 1] = c[3];
        }
        __syncthreads();
        // online softmax: warp w -> rows 4w .. 4w + 3, lane = key
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            const int r = warp * 4 + rr;
            const float sv = !vflag[lane] ? -INFINITY : cell[lane] >= 0 ? Ss[r * TK + lane] : 0.f;  // as grouped_scores: valid, unread = 0
            float mx = sv;
            for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = mrow[r], m_new = fmaxf(m_old, mx);
            const float p = m_new == -INFINITY ? 0.f : __expf(sv - m_new);
            float sum = p;
            for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            Ps[r * PP + lane] = __float2bfloat16(p);
            __syncwarp();
            if (lane == 0) {
                const float alpha = m_old == -INFINITY ? 0.f : __expf(m_old - m_new);
                arow[r] = alpha; lrow[r] = lrow[r] * alpha + sum; mrow[r] = m_new;
            }
        }
        __syncthreads();
        {   // O = alpha O + P V: warp w -> output dims w * 64 .. + 63, k over the tile's keys
            const float al0 = arow[g], al1 = arow[g + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc[j][0] *= al0; acc[j][1] *= al0; acc[j][2] *= al1; acc[j][3] *= al1; }
#pragma unroll
            for (int k0 = 0; k0 < TK; k0 += 16) {
                uint32_t a[4];
                a[0] = *reinterpret_cast<const uint32_t *>(Ps + g * PP + k0 + tq * 2);
                a[1] = *reinterpret_cast<const uint32_t *>(Ps + (g + 8) * PP + k0 + tq * 2);
                a[2] = *reinterpret_cast<const uint32_t *>(Ps + g * PP + k0 + tq * 2 + 8);
                a[3] = *reinterpret_cast<const uint32_t *>(Ps + (g + 8) * PP + k0 + tq * 2 + 8);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int n = warp * 64 + j * 8 + g;  // B[k][n] = V[key k][dim n]
                    const uint32_t b0 = pack2(Vs[(k0 + tq * 2) * KP + n], Vs[(k0 + tq * 2 + 1) * KP + n]);
                    const uint32_t b1 = pack2(Vs[(k0 + tq * 2 + 8) * KP + n], Vs[(k0 + tq * 2 + 9) * KP + n]);
                    mma16816(acc[j], a, b0, b1);
                }
            }
        }
    }
    __syncthreads();
    const float il0 = lrow[g] > 0.f ? 1.f / lrow[g] : 0.f, il1 = lrow[g + 8] > 0.f ? 1.f / lrow[g + 8] : 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + j * 8 + tq * 2;
        if (g < G) {
            float * o = output + (int64_t(t) * heads + kv_head * G + g) * D + d;
            o[0] = acc[j][0] * il0; o[1] = acc[j][1] * il0;
        }
        if (g + 8 < G) {
            float * o = output + (int64_t(t) * heads + kv_head * G + g + 8) * D + d;
            o[0] = acc[j][2] * il1; o[1] = acc[j][3] * il1;
        }
    }
}
static int g_qsa_tc = 1;  // tensor-core prefill attention (qsa_set_tc(0): the scalar kernel, tests)
void qsa_set_tc(int on) { g_qsa_tc = on; }
void qsa_attention_grouped(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                           const QsaDeviceSelection & selected, float * scores, float * inv, float * output, int tokens,
                           int first_cells, int heads, int kv_heads, int dim, cudaStream_t stream) {
    if (!query || !keys || !values || !selected.positions || !selected.valid || !scores || !inv || !output || tokens <= 0 ||
        tokens > 65535 || selected.width <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads ||
        heads / kv_heads > kQsaMaxGroup || dim <= 0 || dim > 256)
        throw std::invalid_argument("QSA grouped attention: invalid dimensions or pointer");
    if (g_qsa_tc != 0 && heads / kv_heads == 12 && dim == 256) {
        static bool attr = false;
        if (!attr) { cudaFuncSetAttribute(qsa_prefill_tc_k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(qsatc::smem_bytes())); attr = true; }
        qsa_prefill_tc_k<<<dim3(kv_heads, tokens), qsatc::NT, qsatc::smem_bytes(), stream>>>(query, keys, values, selected.positions, selected.valid,
                                                                                          output, selected.width, first_cells, kv_heads);
        ck(cudaPeekAtLastError());
        return;
    }
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

} // namespace tqllm_cuda
