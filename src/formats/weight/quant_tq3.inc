// TQ3 matvec / int8 path / fused norm-rotate / argmax kernels. Included by quant.cu inside its anonymous namespace (kernels only; no includes here).
// One warp per 32-element block: lane = element.
__global__ void tq3_rotate_k(const __nv_bfloat16 * x, float * xr, int total_blocks) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= total_blocks) return;
    const int64_t i = int64_t(warp) * 32 + lane;
    xr[i] = wht32(__bfloat162float(x[i]) * TQ3_SIGNS[lane], lane);
}

__global__ void tq3_rotate_f32_k(const float * x, float * xr, int total_blocks) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= total_blocks) return;
    const int64_t i = int64_t(warp) * 32 + lane;
    xr[i] = wht32(x[i] * TQ3_SIGNS[lane], lane);
}

template<bool Repacked>
__global__ void tq3_matvec_f32_raw_k(const uint4 * weights, const float * xr, float * y, int rows, int k) {
    const int row = blockIdx.x, token = blockIdx.y, blocks = k / 32;
    const float * x = xr + int64_t(token) * k;
    float sum = 0.f;
    for (int b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint4 q = weights[int64_t(row) * blocks + b];
        const uint32_t groups[4] = {q.y, (q.y >> 24) | (q.z << 8), (q.z >> 16) | (q.w << 16), q.w >> 8};
        for (int g = 0; g < 4; ++g) {
            float part = 0.f;
            for (int j = 0; j < 8; ++j) {
                uint32_t index;
                if constexpr (Repacked) {
                    const uint32_t words[3] = {q.y, q.z, q.w};
                    index = g < 3 ? (words[g] >> (4 * j)) & 7u :
                        ((q.y >> (4 * j + 3)) & 1u) | (((q.z >> (4 * j + 3)) & 1u) << 1) |
                        (((q.w >> (4 * j + 3)) & 1u) << 2);
                } else index = (groups[g] >> (3 * j)) & 7u;
                part += TQ3_CODEBOOK[index] * x[b * 32 + g * 8 + j];
            }
            sum += e3m5((q.x >> (8 * g)) & 255) * part;
        }
    }
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
    __shared__ float partial[8];
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0.f;
        for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (threadIdx.x == 0) y[int64_t(token) * rows + row] = sum;
    }
}

__global__ void tq3_row_k(const uint4 * w, int row, __nv_bfloat16 * y, int nblk, const uint32_t * dtok = nullptr) {
    const int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (b >= nblk) return;
    if (dtok) {
        row = int(dtok[blockIdx.y]);
        y += int64_t(blockIdx.y) * nblk * 32;
    }
    const uint4 blk = w[int64_t(row) * nblk + b];
    const int g = lane / 8, j = lane % 8;
    const uint32_t scales = blk.x;
    uint32_t bits;
    if (g == 0) bits = blk.y;
    else if (g == 1) bits = (blk.y >> 24) | (blk.z << 8);
    else if (g == 2) bits = (blk.z >> 16) | (blk.w << 16);
    else bits = blk.w >> 8;
    const float v = TQ3_CODEBOOK[(bits >> (3 * j)) & 7] * e3m5((scales >> (8 * g)) & 255);
    y[int64_t(b) * 32 + lane] = __float2bfloat16_rn(wht32(v, lane) * TQ3_SIGNS[lane]);
}

// Block: 8 warps x RPW rows. K is processed in tiles of 32 blocks (1024 weights); the N activation columns of a
// tile sit in shared memory transposed ([n][element-in-block][block]) so lane l (= block l of the tile) reads
// conflict-free. Each lane decodes its weight block once and applies it to all N columns.
template <int N, int RPW>
__global__ void __launch_bounds__(256) tq3_matvec_k(const uint4 * w, const float * xr, __nv_bfloat16 * y, int rows,
                                                    int k) {
    __shared__ float xs[N][32][33];
    __shared__ float cb[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    const int nblk = k / 32, ntile = (nblk + 31) / 32;
    if (threadIdx.x < 8) cb[threadIdx.x] = TQ3_CODEBOOK[threadIdx.x];

    float acc[RPW][N];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int n = 0; n < N; ++n) acc[r][n] = 0.f;

    for (int t = 0; t < ntile; ++t) {
        const int b0 = t * 32, nb = min(32, nblk - b0);
        __syncthreads();
        for (int i = threadIdx.x; i < N * nb * 32; i += blockDim.x) {
            const int n = i / (nb * 32), rem = i % (nb * 32), bl = rem / 32, e = rem % 32;
            xs[n][e][bl] = xr[int64_t(n) * k + int64_t(b0 + bl) * 32 + e];
        }
        __syncthreads();
        if (lane >= nb) continue;
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = row0 + r;
            if (row >= rows) break;
            const uint4 blk = w[int64_t(row) * nblk + b0 + lane];
            const uint32_t g0 = blk.y, g1 = (blk.y >> 24) | (blk.z << 8), g2 = (blk.z >> 16) | (blk.w << 16),
                           g3 = blk.w >> 8;
            const uint32_t gb[4] = {g0, g1, g2, g3};
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const float d = e3m5((blk.x >> (8 * g)) & 255);
                float part[N];
#pragma unroll
                for (int n = 0; n < N; ++n) part[n] = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float c = cb[(gb[g] >> (3 * j)) & 7];
#pragma unroll
                    for (int n = 0; n < N; ++n) part[n] += c * xs[n][g * 8 + j][lane];
                }
#pragma unroll
                for (int n = 0; n < N; ++n) acc[r][n] += d * part[n];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int n = 0; n < N; ++n) {
            float v = acc[r][n];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && row < rows) y[int64_t(n) * rows + row] = __float2bfloat16_rn(v);
        }
    }
}


// Single-column fast path. Lane l of a warp walks blocks l, l+32, ... of its RPW rows. For each block it loads the
// 32 rotated activations into registers once (8 x float4, coalesced) and reuses them for all RPW rows, whose weight
// blocks are fetched up front to keep RPW 16-byte loads in flight per lane. No shared-memory tile, no block sync.
template <int RPW>
__global__ void __launch_bounds__(256) tq3_matvec1_k(const uint4 * w, const float * xr, __nv_bfloat16 * y, int rows,
                                                     int k) {
    __shared__ float cb[8];
    if (threadIdx.x < 8) cb[threadIdx.x] = TQ3_CODEBOOK[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    if (row0 >= rows) return;
    const int nblk = k / 32;
    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

    for (int b = lane; b < nblk; b += 32) {
        uint4 blk[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) blk[r] = row0 + r < rows ? __ldg(&w[int64_t(row0 + r) * nblk + b]) : make_uint4(0, 0, 0, 0);
        float xv[32];
        const float4 * xp = reinterpret_cast<const float4 *>(xr + int64_t(b) * 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 t = __ldg(&xp[i]);
            xv[4 * i] = t.x;
            xv[4 * i + 1] = t.y;
            xv[4 * i + 2] = t.z;
            xv[4 * i + 3] = t.w;
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const uint4 q = blk[r];
            const uint32_t gb[4] = {q.y, (q.y >> 24) | (q.z << 8), (q.z >> 16) | (q.w << 16), q.w >> 8};
            float s = 0.f;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) part += cb[(gb[g] >> (3 * j)) & 7] * xv[g * 8 + j];
                s += e3m5((q.x >> (8 * g)) & 255) * part;
            }
            acc[r] += s;
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && row0 + r < rows) y[row0 + r] = __float2bfloat16_rn(v);
    }
}


// ---- int8 fast path ----
// Repacked block (16 bytes): x = 4 E3M5 scales (unchanged); y,z,w = W0,W1,W2. Elements 0..23: low 3 bits of
// nibble (e % 8) of W[e / 8], i.e. already a valid PRMT byte selector after masking with 0x7777. Elements 24..31:
// bit t of index (24 + m) is bit 3 of nibble m of W[t]. Codebook lookup = PRMT on an int8 codebook, dot = DP4A.
constexpr float TQ3_CB_Q = 63.5f;  // int8 codebook = round(c * 63.5), |c| <= 1.997

__global__ void tq3_repack_k(uint4 * w, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint4 q = w[i];
    const uint32_t gb[4] = {q.y, (q.y >> 24) | (q.z << 8), (q.z >> 16) | (q.w << 16), q.w >> 8};
    uint32_t W[3] = {0, 0, 0};
    for (int e = 0; e < 24; ++e) W[e / 8] |= ((gb[e / 8] >> (3 * (e % 8))) & 7u) << (4 * (e % 8));
    for (int m = 0; m < 8; ++m) {
        const uint32_t idx = (gb[3] >> (3 * m)) & 7u;
        for (int t = 0; t < 3; ++t) W[t] |= ((idx >> t) & 1u) << (4 * m + 3);
    }
    w[i] = make_uint4(q.x, W[0], W[1], W[2]);
}

// One warp per 32-element block: rotate (sign flip + WHT), then max-abs int8 quantisation.
__global__ void tq3_rotate_i8_k(const __nv_bfloat16 * x, int8_t * xq, float * xs, int nblk) {
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (b >= nblk) return;
    const int64_t i = int64_t(b) * 32 + lane;
    const float v = wht32(__bfloat162float(x[i]) * TQ3_SIGNS[lane], lane);
    float m = fabsf(v);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float scale = m / 127.f;
    xq[i] = int8_t(m > 0.f ? __float2int_rn(v / scale) : 0);
    if (lane == 0) xs[b] = scale;
}

__device__ __forceinline__ float e3m5_byte(uint32_t b) {
    return b ? __uint_as_float((b << 18) + 0x3B000000u) : 0.f;
}

template <int RPW>
__global__ void __launch_bounds__(256) tq3_matvec_i8_k(const uint4 * w, const int4 * xq, const float * xs,
                                                       __nv_bfloat16 * y, int rows, int k, uint32_t cblo,
                                                       uint32_t cbhi) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    if (row0 >= rows) return;
    const int nblk = k / 32;
    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;
    for (int b = lane; b < nblk; b += 32) {
        uint4 blk[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r)
            blk[r] = row0 + r < rows ? __ldg(&w[int64_t(row0 + r) * nblk + b]) : make_uint4(0, 0, 0, 0);
        const int4 xa = __ldg(&xq[2 * b]), xb = __ldg(&xq[2 * b + 1]);
        const float sx = __ldg(&xs[b]);
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const uint4 q = blk[r];
            const uint32_t W0 = q.y, W1 = q.z, W2 = q.w;
            int s0 = __dp4a(int(__byte_perm(cblo, cbhi, W0 & 0x7777)), xa.x, 0);
            s0 = __dp4a(int(__byte_perm(cblo, cbhi, (W0 >> 16) & 0x7777)), xa.y, s0);
            int s1 = __dp4a(int(__byte_perm(cblo, cbhi, W1 & 0x7777)), xa.z, 0);
            s1 = __dp4a(int(__byte_perm(cblo, cbhi, (W1 >> 16) & 0x7777)), xa.w, s1);
            int s2 = __dp4a(int(__byte_perm(cblo, cbhi, W2 & 0x7777)), xb.x, 0);
            s2 = __dp4a(int(__byte_perm(cblo, cbhi, (W2 >> 16) & 0x7777)), xb.y, s2);
            const uint32_t sa = ((W0 >> 3) & 0x1111) | ((W1 >> 2) & 0x2222) | ((W2 >> 1) & 0x4444);
            const uint32_t sb = ((W0 >> 19) & 0x1111) | ((W1 >> 18) & 0x2222) | ((W2 >> 17) & 0x4444);
            int s3 = __dp4a(int(__byte_perm(cblo, cbhi, sa)), xb.z, 0);
            s3 = __dp4a(int(__byte_perm(cblo, cbhi, sb)), xb.w, s3);
            acc[r] += sx * (e3m5_byte(q.x & 255) * float(s0) + e3m5_byte((q.x >> 8) & 255) * float(s1) +
                            e3m5_byte((q.x >> 16) & 255) * float(s2) + e3m5_byte(q.x >> 24) * float(s3));
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && row0 + r < rows) y[row0 + r] = __float2bfloat16_rn(v * (1.f / TQ3_CB_Q));
    }
}

struct Tq3Codebook8 {
    uint32_t lo, hi;
};

Tq3Codebook8 tq3_codebook_i8() {
    static const float c[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f,
                               0.230106f,  0.725222f,  1.277503f,  1.988943f};
    Tq3Codebook8 cb{0, 0};
    for (int i = 0; i < 4; ++i) {
        cb.lo |= uint32_t(uint8_t(int8_t(lrintf(c[i] * TQ3_CB_Q)))) << (8 * i);
        cb.hi |= uint32_t(uint8_t(int8_t(lrintf(c[i + 4] * TQ3_CB_Q)))) << (8 * i);
    }
    return cb;
}


// ---- fused residual / norm / rotate / int8 ----
__device__ float block_sum256(float v) {
    __shared__ float sh[8];
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += sh[i];
    return t;
}

__global__ void add_sumsq_k(__nv_bfloat16 * x, const __nv_bfloat16 * y, float * partial, int k) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    float v = 0.f;
    if (i < k) {
        const __nv_bfloat16 r = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(y[i]));
        x[i] = r;
        v = __bfloat162float(r);
    }
    const float t = block_sum256(v * v);
    if (threadIdx.x == 0) partial[blockIdx.x] = t;
}

__global__ void sumsq_k(const __nv_bfloat16 * x, float * partial, int k) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    const float v = i < k ? __bfloat162float(x[i]) : 0.f;
    const float t = block_sum256(v * v);
    if (threadIdx.x == 0) partial[blockIdx.x] = t;
}

// Block of 256 threads = 8 warps = 8 TQ3 blocks of 32 elements.
__global__ void norm_rotate_i8_k(const __nv_bfloat16 * x, const float * w, const float * partial, int nparts, float eps,
                                 int8_t * xq, float * xs, int k) {
    __shared__ float inv_s;
    if (threadIdx.x < 32) {
        float t = 0.f;
        for (int i = threadIdx.x; i < nparts; i += 32) t += partial[i];
        for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) inv_s = rsqrtf(t / k + eps);
    }
    __syncthreads();
    const int i = blockIdx.x * 256 + threadIdx.x, lane = threadIdx.x % 32, b = i / 32;
    if (i >= k) return;
    const float n = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x[i]) * inv_s * w[i]));
    const float v = wht32(n * TQ3_SIGNS[lane], lane);
    float m = fabsf(v);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float scale = m / 127.f;
    xq[i] = int8_t(m > 0.f ? __float2int_rn(v / scale) : 0);
    if (lane == 0) xs[b] = scale;
}

// ---- two-stage argmax ----
__device__ __forceinline__ void amax_merge(float & v, uint32_t & ix, float ov, uint32_t oi) {
    if (ov > v || (ov == v && oi < ix)) {
        v = ov;
        ix = oi;
    }
}

__global__ void argmax1_k(const float * x, int n, float * pv, uint32_t * pi) {
    float v = -INFINITY;
    uint32_t ix = 0xffffffffu;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) amax_merge(v, ix, x[i], i);
    for (int o = 16; o; o >>= 1) amax_merge(v, ix, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, ix, o));
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    if (threadIdx.x % 32 == 0) {
        sv[threadIdx.x / 32] = v;
        si[threadIdx.x / 32] = ix;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; ++w) amax_merge(v, ix, sv[w], si[w]);
        pv[blockIdx.x] = v;
        pi[blockIdx.x] = ix;
    }
}

__global__ void argmax2_k(const float * pv, const uint32_t * pi, int np, uint32_t * out) {
    float v = -INFINITY;
    uint32_t ix = 0xffffffffu;
    for (int i = threadIdx.x; i < np; i += 32) amax_merge(v, ix, pv[i], pi[i]);
    for (int o = 16; o; o >>= 1) amax_merge(v, ix, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, ix, o));
    if (threadIdx.x == 0) *out = ix;
}


// ---- multi-column verify path ----
__global__ void tq3_rotate_h_k(const __nv_bfloat16 * x, __half * xh, int total_blocks) {
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (b >= total_blocks) return;
    const int64_t i = int64_t(b) * 32 + lane;
    xh[i] = __float2half_rn(wht32(__bfloat162float(x[i]) * TQ3_SIGNS[lane], lane));
}

// grid (k / 256, ncols); partial [ncols][k / 256]
__global__ void norm_rotate_h_k(const __nv_bfloat16 * x, const float * w, const float * partial, int nparts, float eps,
                                __half * xh, int k) {
    __shared__ float inv_s;
    const int n = blockIdx.y;
    if (threadIdx.x < 32) {
        float t = 0.f;
        for (int i = threadIdx.x; i < nparts; i += 32) t += partial[n * nparts + i];
        for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) inv_s = rsqrtf(t / k + eps);
    }
    __syncthreads();
    const int i = blockIdx.x * 256 + threadIdx.x, lane = threadIdx.x % 32;
    if (i >= k) return;
    const int64_t o = int64_t(n) * k + i;
    const float v = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x[o]) * inv_s * w[i]));
    xh[o] = __float2half_rn(wht32(v * TQ3_SIGNS[lane], lane));
}

__device__ __forceinline__ void mma_f16(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
#if defined(TQLLM_WITH_HIP)
    tqllm_dev::mma_f16(c, a0, a1, a2, a3, b0, b1);
#else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void mma_f16h(uint32_t * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {  // FP16 accumulate: twice the GA10x tensor rate
#if defined(TQLLM_WITH_HIP)
    tqllm_dev::mma_f16acc(*reinterpret_cast<uint32_t(*)[2]>(c), a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
    uint32_t r;
    #if defined(TQLLM_WITH_HIP)
    return tqllm_dev::hmul2u(a, b);
#else
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
#endif
    return r;
}
__device__ __forceinline__ uint32_t hadd2u(uint32_t a, uint32_t b) {
    uint32_t r;
    #if defined(TQLLM_WITH_HIP)
    return tqllm_dev::hadd2u(a, b);
#else
    asm("add.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
#endif
    return r;
}

__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    #if defined(TQLLM_WITH_HIP)
    return tqllm_dev::hfma2u(a, b, c);
#else
    asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#endif
    return r;
}

// FP16 tensor-core matvec over the repacked TQ3_4S layout. CTA = 8 warps on one 16-row tile, split-K across warps.
// k is permuted identically in A and B so lane t of a k16 half owns the 4 consecutive weights 16h + 4t.. (one scale
// group); weights decode in registers: int8 codebook (+128) -> PRMT into fp16 (1024 + u) -> HFMA2 with (s, -1152 s).
// FP16 tensor-core matvec over the repacked TQ3_4S layout. CTA = 8 warps on one 16-row tile, split-K across warps.
// The 4 lanes of a quad take 4 consecutive blocks of the same rows (64 contiguous bytes per row, no redundant
// loads), and each lane decodes its two blocks (rows g, g+8) exactly once. k is permuted identically in A and B:
// MMA u (0..7) takes, from lane t, elements 4u..4u+3 of lane t's block, so the activation fragment of lane t is
// simply the 32 halves of that block. Decode: int8 codebook (+128) -> PRMT into fp16 (1024 + u) -> HFMA2 with
// (s, -1152 s), exact for E3M5 scales.
template <int NCOL, bool EXPERT = false>
__global__ void __launch_bounds__(256) tq3_matvec_h_k(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows,
                                                      int k, uint32_t cblo_u, uint32_t cbhi_u,
                                                      const int * ids = nullptr, int topk = 1, bool per_expert = false) {
    if constexpr (EXPERT) {
        const int assignment = blockIdx.y;
        w += int64_t(ids[assignment]) * rows * (k / 32);
        xh += int64_t(per_expert ? assignment : assignment / topk) * k;
        y += int64_t(assignment) * rows;
    }
    constexpr int NACC = NCOL > 8 ? 8 : 4;
    __shared__ float red[8][32][NACC];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16, nblk = k / 32;
    const uint32_t NEG1152 = 0xE480E480u;
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    float c[NACC] = {};
    for (int b0 = warp * 4; b0 < nblk; b0 += 32) {
        const int b = b0 + t;
        const bool ok = b < nblk;
        const uint4 qa = ok ? __ldg(&w[int64_t(ra) * nblk + b]) : make_uint4(0, 0, 0, 0);
        const uint4 qb = ok ? __ldg(&w[int64_t(rb) * nblk + b]) : make_uint4(0, 0, 0, 0);
        uint4 x4[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        uint4 x42[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if (g < NCOL && ok) {
            const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g) * k + int64_t(b) * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) x4[i] = __ldg(&xp[i]);
        }
        if constexpr (NCOL > 8) {
            if (g < NCOL - 8 && ok) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g + 8) * k + int64_t(b) * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) x42[i] = __ldg(&xp[i]);
            }
        }
        const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
        const uint32_t * xw2 = reinterpret_cast<const uint32_t *>(x42);
        uint32_t sa[4], oa[4], sb[4], ob[4];
#pragma unroll
        for (int gi = 0; gi < 4; ++gi) {
            const uint32_t ba = (qa.x >> (8 * gi)) & 255u, bb = (qb.x >> (8 * gi)) & 255u;
            const uint32_t ha = ba ? (ba << 5) + 0x1800u : 0u, hb = bb ? (bb << 5) + 0x1800u : 0u;
            sa[gi] = ha | (ha << 16);
            sb[gi] = hb | (hb << 16);
            oa[gi] = hmul2u(sa[gi], NEG1152);
            ob[gi] = hmul2u(sb[gi], NEG1152);
        }
        const uint32_t sla = ((qa.y >> 3) & 0x1111u) | ((qa.z >> 2) & 0x2222u) | ((qa.w >> 1) & 0x4444u);
        const uint32_t slb = ((qb.y >> 3) & 0x1111u) | ((qb.z >> 2) & 0x2222u) | ((qb.w >> 1) & 0x4444u);
        const uint32_t sla2 = ((qa.y >> 19) & 0x1111u) | ((qa.z >> 18) & 0x2222u) | ((qa.w >> 17) & 0x4444u);
        const uint32_t slb2 = ((qb.y >> 19) & 0x1111u) | ((qb.z >> 18) & 0x2222u) | ((qb.w >> 17) & 0x4444u);
        const uint32_t sela[8] = {qa.y, qa.y >> 16, qa.z, qa.z >> 16, qa.w, qa.w >> 16, sla, sla2};
        const uint32_t selb[8] = {qb.y, qb.y >> 16, qb.z, qb.z >> 16, qb.w, qb.w >> 16, slb, slb2};
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
            const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
            const int gi = u / 2;
            mma_f16(c, hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]),
                    hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]),
                    hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]),
                    hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]), xw[2 * u], xw[2 * u + 1]);
            if constexpr (NCOL > 8)
                mma_f16(c + 4, hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]),
                        hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]),
                        hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]),
                        hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]), xw2[2 * u], xw2[2 * u + 1]);
        }
    }
#pragma unroll
    for (int i = 0; i < NACC; ++i) red[warp][lane][i] = c[i];
    __syncthreads();
    if (warp == 0) {
        float v[NACC] = {};
        for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
            for (int i = 0; i < NACC; ++i) v[i] += red[w2][lane][i];
        const int n0 = 2 * t, n1 = 2 * t + 1, r = row0 + g;
        const float inv = 1.f / TQ3_CB_Q;
        if (r < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0] * inv);
            if (n1 < NCOL) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1] * inv);
        }
        if (r + 8 < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2] * inv);
            if (n1 < NCOL) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3] * inv);
        }
        if constexpr (NCOL > 8) {
            if (r < rows) {
                if (n0 + 8 < NCOL) y[int64_t(n0 + 8) * rows + r] = __float2bfloat16_rn(v[4] * inv);
                if (n1 + 8 < NCOL) y[int64_t(n1 + 8) * rows + r] = __float2bfloat16_rn(v[5] * inv);
            }
            if (r + 8 < rows) {
                if (n0 + 8 < NCOL) y[int64_t(n0 + 8) * rows + r + 8] = __float2bfloat16_rn(v[6] * inv);
                if (n1 + 8 < NCOL) y[int64_t(n1 + 8) * rows + r + 8] = __float2bfloat16_rn(v[7] * inv);
            }
        }
    }
}

// Wide variant for batched verify (13..32 columns): NT tiles of 8 columns, MT row tiles of 16 per CTA so each
// activation load serves MT row tiles (at 32 columns the 16-row CTA re-read the activations from L2 ~9x over the
// weight bytes). Per row and column the k split, MMA order and 8-warp reduction are those of tq3_matvec_h_k, so the
// results are bit-identical to narrow calls.
template <int NT, int MT>
__global__ void __launch_bounds__(256) tq3_matvec_hw_k(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows,
                                                       int k, int nc, uint32_t cblo_u, uint32_t cbhi_u) {
    __shared__ float red[8][32][4 * NT];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nblk = k / 32;
    const uint32_t NEG1152 = 0xE480E480u;
    float c[MT][NT][4] = {};
    for (int b0 = warp * 4; b0 < nblk; b0 += 32) {
        const int b = b0 + t;
        const bool ok = b < nblk;
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
            uint4 x4[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
            if (8 * tt + g < nc && ok) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(8 * tt + g) * k + int64_t(b) * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) x4[i] = __ldg(&xp[i]);
            }
            const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
#pragma unroll
            for (int m = 0; m < MT; ++m) {  // one activation load serves MT row tiles (weights re-read from L1)
                const int ra = min(row0 + 16 * m + g, rows - 1), rb = min(row0 + 16 * m + g + 8, rows - 1);
                const uint4 qa = ok ? __ldg(&w[int64_t(ra) * nblk + b]) : make_uint4(0, 0, 0, 0);
                const uint4 qb = ok ? __ldg(&w[int64_t(rb) * nblk + b]) : make_uint4(0, 0, 0, 0);
                uint32_t sa[4], oa[4], sb[4], ob[4];
#pragma unroll
                for (int gi = 0; gi < 4; ++gi) {
                    const uint32_t ba = (qa.x >> (8 * gi)) & 255u, bb = (qb.x >> (8 * gi)) & 255u;
                    const uint32_t ha = ba ? (ba << 5) + 0x1800u : 0u, hb = bb ? (bb << 5) + 0x1800u : 0u;
                    sa[gi] = ha | (ha << 16);
                    sb[gi] = hb | (hb << 16);
                    oa[gi] = hmul2u(sa[gi], NEG1152);
                    ob[gi] = hmul2u(sb[gi], NEG1152);
                }
                const uint32_t sla = ((qa.y >> 3) & 0x1111u) | ((qa.z >> 2) & 0x2222u) | ((qa.w >> 1) & 0x4444u);
                const uint32_t slb = ((qb.y >> 3) & 0x1111u) | ((qb.z >> 2) & 0x2222u) | ((qb.w >> 1) & 0x4444u);
                const uint32_t sla2 = ((qa.y >> 19) & 0x1111u) | ((qa.z >> 18) & 0x2222u) | ((qa.w >> 17) & 0x4444u);
                const uint32_t slb2 = ((qb.y >> 19) & 0x1111u) | ((qb.z >> 18) & 0x2222u) | ((qb.w >> 17) & 0x4444u);
                const uint32_t sela[8] = {qa.y, qa.y >> 16, qa.z, qa.z >> 16, qa.w, qa.w >> 16, sla, sla2};
                const uint32_t selb[8] = {qb.y, qb.y >> 16, qb.z, qb.z >> 16, qb.w, qb.w >> 16, slb, slb2};
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
                    const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
                    const int gi = u / 2;
                    mma_f16(c[m][tt], hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]),
                            hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]),
                            hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]),
                            hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]), xw[2 * u], xw[2 * u + 1]);
                }
            }
        }
    }
    const float inv = 1.f / TQ3_CB_Q;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m) __syncthreads();
#pragma unroll
        for (int tt = 0; tt < NT; ++tt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[warp][lane][4 * tt + i] = c[m][tt][i];
        __syncthreads();
        if (warp < NT) {
            const int tt = warp;
            float v[4] = {};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * tt + i];
            const int n0 = 8 * tt + 2 * t, n1 = n0 + 1, r = row0 + 16 * m + g;
            if (r < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1] * inv);
            }
            if (r + 8 < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3] * inv);
            }
        }
    }
}

// <= 8 columns, MT row tiles per CTA, next iteration's weights and activations loaded while the current one computes
// (two loads in flight per warp). Per-column arithmetic identical to tq3_matvec_h_k.
__device__ __forceinline__ void tq3_tile_mma(float * c, const uint4 & qa, const uint4 & qb, const uint32_t * xw,
                                             uint32_t cblo_u, uint32_t cbhi_u) {
    const uint32_t NEG1152 = 0xE480E480u;
    uint32_t sa[4], oa[4], sb[4], ob[4];
#pragma unroll
    for (int gi = 0; gi < 4; ++gi) {
        const uint32_t ba = (qa.x >> (8 * gi)) & 255u, bb = (qb.x >> (8 * gi)) & 255u;
        const uint32_t ha = ba ? (ba << 5) + 0x1800u : 0u, hb = bb ? (bb << 5) + 0x1800u : 0u;
        sa[gi] = ha | (ha << 16);
        sb[gi] = hb | (hb << 16);
        oa[gi] = hmul2u(sa[gi], NEG1152);
        ob[gi] = hmul2u(sb[gi], NEG1152);
    }
    const uint32_t sla = ((qa.y >> 3) & 0x1111u) | ((qa.z >> 2) & 0x2222u) | ((qa.w >> 1) & 0x4444u);
    const uint32_t slb = ((qb.y >> 3) & 0x1111u) | ((qb.z >> 2) & 0x2222u) | ((qb.w >> 1) & 0x4444u);
    const uint32_t sla2 = ((qa.y >> 19) & 0x1111u) | ((qa.z >> 18) & 0x2222u) | ((qa.w >> 17) & 0x4444u);
    const uint32_t slb2 = ((qb.y >> 19) & 0x1111u) | ((qb.z >> 18) & 0x2222u) | ((qb.w >> 17) & 0x4444u);
    const uint32_t sela[8] = {qa.y, qa.y >> 16, qa.z, qa.z >> 16, qa.w, qa.w >> 16, sla, sla2};
    const uint32_t selb[8] = {qb.y, qb.y >> 16, qb.z, qb.z >> 16, qb.w, qb.w >> 16, slb, slb2};
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
        const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
        const int gi = u / 2;
        mma_f16(c, hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]),
                hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]),
                hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]),
                hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]), xw[2 * u], xw[2 * u + 1]);
    }
}

// > 8 columns: NT tiles of 8 columns, MT row tiles per CTA, next iteration's loads in flight while the current one
// computes (as tq3_matvec_p_k), and each weight block decoded once for all NT tiles (tq3_matvec_hw_k decodes per tile).
// Per accumulator the k split, MMA order and 8-warp reduction are tq3_matvec_h_k's: bit-identical per column.
template <int NT, int MT>
__global__ void __launch_bounds__(256) tq3_matvec_pw_k(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows,
                                                       int k, int nc, uint32_t cblo_u, uint32_t cbhi_u) {
    __shared__ float red[8][32][4 * NT];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nblk = k / 32;
    const uint32_t NEG1152 = 0xE480E480u;
    int ra[MT], rb[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        ra[m] = min(row0 + 16 * m + g, rows - 1);
        rb[m] = min(row0 + 16 * m + g + 8, rows - 1);
    }
    float c[MT][NT][4] = {};
    uint4 qa[MT], qb[MT], x4[NT][4];
    auto load = [&](int b0, uint4 (&A)[MT], uint4 (&B)[MT], uint4 (&X)[NT][4]) {
        const int b = b0 + t;
        const bool ok = b < nblk;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            A[m] = ok ? __ldg(&w[int64_t(ra[m]) * nblk + b]) : make_uint4(0, 0, 0, 0);
            B[m] = ok ? __ldg(&w[int64_t(rb[m]) * nblk + b]) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
            const bool on = 8 * tt + g < nc && ok;
            const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(8 * tt + g) * k + int64_t(b) * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) X[tt][i] = on ? __ldg(&xp[i]) : make_uint4(0, 0, 0, 0);
        }
    };
    int b0 = warp * 4;
    if (b0 < nblk) load(b0, qa, qb, x4);
    for (; b0 < nblk; b0 += 32) {
        uint4 na[MT], nb[MT], nx[NT][4];
        const bool more = b0 + 32 < nblk;
        if (more) load(b0 + 32, na, nb, nx);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            uint32_t sa[4], oa[4], sb[4], ob[4];
#pragma unroll
            for (int gi = 0; gi < 4; ++gi) {
                const uint32_t ba = (qa[m].x >> (8 * gi)) & 255u, bb = (qb[m].x >> (8 * gi)) & 255u;
                const uint32_t ha = ba ? (ba << 5) + 0x1800u : 0u, hb = bb ? (bb << 5) + 0x1800u : 0u;
                sa[gi] = ha | (ha << 16);
                sb[gi] = hb | (hb << 16);
                oa[gi] = hmul2u(sa[gi], NEG1152);
                ob[gi] = hmul2u(sb[gi], NEG1152);
            }
            const uint4 A = qa[m], B = qb[m];
            const uint32_t sla = ((A.y >> 3) & 0x1111u) | ((A.z >> 2) & 0x2222u) | ((A.w >> 1) & 0x4444u);
            const uint32_t slb = ((B.y >> 3) & 0x1111u) | ((B.z >> 2) & 0x2222u) | ((B.w >> 1) & 0x4444u);
            const uint32_t sla2 = ((A.y >> 19) & 0x1111u) | ((A.z >> 18) & 0x2222u) | ((A.w >> 17) & 0x4444u);
            const uint32_t slb2 = ((B.y >> 19) & 0x1111u) | ((B.z >> 18) & 0x2222u) | ((B.w >> 17) & 0x4444u);
            const uint32_t sela[8] = {A.y, A.y >> 16, A.z, A.z >> 16, A.w, A.w >> 16, sla, sla2};
            const uint32_t selb[8] = {B.y, B.y >> 16, B.z, B.z >> 16, B.w, B.w >> 16, slb, slb2};
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
                const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
                const int gi = u / 2;
                const uint32_t a0 = hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]);
                const uint32_t a1 = hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]);
                const uint32_t a2 = hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]);
                const uint32_t a3 = hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]);
#pragma unroll
                for (int tt = 0; tt < NT; ++tt) {
                    const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4[tt]);
                    mma_f16(c[m][tt], a0, a1, a2, a3, xw[2 * u], xw[2 * u + 1]);
                }
            }
        }
        if (more) {
#pragma unroll
            for (int m = 0; m < MT; ++m) { qa[m] = na[m]; qb[m] = nb[m]; }
#pragma unroll
            for (int tt = 0; tt < NT; ++tt)
#pragma unroll
                for (int i = 0; i < 4; ++i) x4[tt][i] = nx[tt][i];
        }
    }
    const float inv = 1.f / TQ3_CB_Q;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m) __syncthreads();
#pragma unroll
        for (int tt = 0; tt < NT; ++tt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[warp][lane][4 * tt + i] = c[m][tt][i];
        __syncthreads();
        if (warp < NT) {
            const int tt = warp;
            float v[4] = {};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * tt + i];
            const int n0 = 8 * tt + 2 * t, n1 = n0 + 1, r = row0 + 16 * m + g;
            if (r < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1] * inv);
            }
            if (r + 8 < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3] * inv);
            }
        }
    }
}

template <int MT>
__global__ void __launch_bounds__(256) tq3_matvec_p_k(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows,
                                                      int k, int nc, uint32_t cblo_u, uint32_t cbhi_u) {
    __shared__ float red[8][32][4];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nblk = k / 32;
    int ra[MT], rb[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        ra[m] = min(row0 + 16 * m + g, rows - 1);
        rb[m] = min(row0 + 16 * m + g + 8, rows - 1);
    }
    float c[MT][4] = {};
    uint4 qa[MT], qb[MT], x4[4];
    auto load = [&](int b0, uint4 (&A)[MT], uint4 (&B)[MT], uint4 (&X)[4]) {
        const int b = b0 + t;
        const bool ok = b < nblk;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            A[m] = ok ? __ldg(&w[int64_t(ra[m]) * nblk + b]) : make_uint4(0, 0, 0, 0);
            B[m] = ok ? __ldg(&w[int64_t(rb[m]) * nblk + b]) : make_uint4(0, 0, 0, 0);
        }
        const bool on = g < nc && ok;
        const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g) * k + int64_t(b) * 32);
#pragma unroll
        for (int i = 0; i < 4; ++i) X[i] = on ? __ldg(&xp[i]) : make_uint4(0, 0, 0, 0);
    };
    int b0 = warp * 4;
    if (b0 < nblk) load(b0, qa, qb, x4);
    for (; b0 < nblk; b0 += 32) {
        uint4 na[MT], nb[MT], nx[4];
        const bool more = b0 + 32 < nblk;
        if (more) load(b0 + 32, na, nb, nx);
        const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
#pragma unroll
        for (int m = 0; m < MT; ++m) tq3_tile_mma(c[m], qa[m], qb[m], xw, cblo_u, cbhi_u);
        if (more) {
#pragma unroll
            for (int m = 0; m < MT; ++m) { qa[m] = na[m]; qb[m] = nb[m]; }
#pragma unroll
            for (int i = 0; i < 4; ++i) x4[i] = nx[i];
        }
    }
    const float inv = 1.f / TQ3_CB_Q;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m) __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][lane][i] = c[m][i];
        __syncthreads();
        if (warp == 0) {
            float v[4] = {};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][i];
            const int n0 = 2 * t, n1 = n0 + 1, r = row0 + 16 * m + g;
            if (r < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1] * inv);
            }
            if (r + 8 < rows) {
                if (n0 < nc) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2] * inv);
                if (n1 < nc) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3] * inv);
            }
        }
    }
}

struct Tq3CodebookU {
    uint32_t lo, hi;
};

Tq3CodebookU tq3_codebook_u8() {  // int8 codebook + 128, as unsigned bytes
    static const float c[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f,
                               0.230106f,  0.725222f,  1.277503f,  1.988943f};
    Tq3CodebookU cb{0, 0};
    for (int i = 0; i < 4; ++i) {
        cb.lo |= uint32_t(uint8_t(lrintf(c[i] * TQ3_CB_Q) + 128)) << (8 * i);
        cb.hi |= uint32_t(uint8_t(lrintf(c[i + 4] * TQ3_CB_Q) + 128)) << (8 * i);
    }
    return cb;
}

__global__ void argmax_cols1_k(const float * x, int len, float * pv, uint32_t * pi, int nb) {
    const int n = blockIdx.y;
    const float * xn = x + int64_t(n) * len;
    float v = -INFINITY;
    uint32_t ix = 0xffffffffu;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) amax_merge(v, ix, xn[i], i);
    for (int o = 16; o; o >>= 1) amax_merge(v, ix, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, ix, o));
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    if (threadIdx.x % 32 == 0) {
        sv[threadIdx.x / 32] = v;
        si[threadIdx.x / 32] = ix;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; ++w) amax_merge(v, ix, sv[w], si[w]);
        pv[n * nb + blockIdx.x] = v;
        pi[n * nb + blockIdx.x] = ix;
    }
}

__global__ void argmax_cols2_k(const float * pv, const uint32_t * pi, int nb, uint32_t * out) {
    const int n = blockIdx.x;
    float v = -INFINITY;
    uint32_t ix = 0xffffffffu;
    for (int i = threadIdx.x; i < nb; i += 32) amax_merge(v, ix, pv[n * nb + i], pi[n * nb + i]);
    for (int o = 16; o; o >>= 1) amax_merge(v, ix, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, ix, o));
    if (threadIdx.x == 0) out[n] = ix;
}
