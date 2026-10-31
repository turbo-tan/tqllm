// TQ3 prefill GEMM, dequant / convert / top-16 kernels and the launch helpers. Included by quant.cu inside its anonymous namespace (kernels only; no includes here).
// Prefill GEMM over the repacked TQ3_4S layout: y[n][r] for up to 64 columns per grid.y slice. CTA = 64 rows (warp w:
// row tile w & 3, k half w >> 2). Weight decode is exactly tq3_matvec_h_k's (same numerics per weight); each lane
// decodes its block once and reuses the A fragments across 8 n-tiles, so the weight stream is read once per 64 columns.
template <bool ACC16, bool EXPERT = false>
__global__ void __launch_bounds__(256) tq3_gemm_h_k(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows,
                                                    int k, int ncols, uint32_t cblo_u, uint32_t cbhi_u,
                                                    const int * counts = nullptr, const int * groups = nullptr,
                                                    int capacity = 0, int topk = 1, bool per_expert = false) {
    if constexpr (EXPERT) {
        ncols = counts[blockIdx.z];
        if (int(blockIdx.y) * 64 >= ncols) return;
        w += int64_t(blockIdx.z) * rows * (k / 32);
        groups += int64_t(blockIdx.z) * capacity;
    }
    constexpr int NT = 8;  // n-tiles of 8 columns
    __shared__ float red[4][32][NT * 4];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int rt = warp & 3, ks = warp >> 2;
    const int row0 = blockIdx.x * 64 + rt * 16, col0 = blockIdx.y * 64, nblk = k / 32;
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    const uint32_t NEG1152 = 0xE480E480u;
    float c[NT][4] = {};
    for (int b0 = ks * 4; b0 < nblk; b0 += 8) {
        const int b = b0 + t;
        const bool ok = b < nblk;
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
        uint32_t af[8][4];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
            const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
            const int gi = u / 2;
            af[u][0] = hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]);
            af[u][1] = hfma2u(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]);
            af[u][2] = hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]);
            af[u][3] = hfma2u(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]);
        }
        uint32_t c16[NT][2] = {};  // ACC16: this iteration's 128-k segment accumulated in FP16 (2x tensor rate)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int col = col0 + nt * 8 + g;
            uint4 x4[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
            if (ok && col < ncols) {
                int input_col = col;
                if constexpr (EXPERT) input_col = per_expert ? groups[col] : groups[col] / topk;
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(input_col) * k + int64_t(b) * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) x4[i] = __ldg(&xp[i]);
            }
            const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                if constexpr (ACC16) mma_f16h(c16[nt], af[u][0], af[u][1], af[u][2], af[u][3], xw[2 * u], xw[2 * u + 1]);
                else mma_f16(c[nt], af[u][0], af[u][1], af[u][2], af[u][3], xw[2 * u], xw[2 * u + 1]);
            }
        }
        if constexpr (ACC16)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&c16[nt][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&c16[nt][1]));
                c[nt][0] += lo.x;
                c[nt][1] += lo.y;
                c[nt][2] += hi.x;
                c[nt][3] += hi.y;
            }
    }
    if (ks == 1)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[rt][lane][nt * 4 + i] = c[nt][i];
    __syncthreads();
    if (ks == 0) {
        const float inv = 1.f / TQ3_CB_Q;
        const int r = row0 + g;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int n0 = col0 + nt * 8 + 2 * t, n1 = n0 + 1;
            int out0 = n0, out1 = n1;
            if constexpr (EXPERT) {
                if (n0 < ncols) out0 = groups[n0];
                if (n1 < ncols) out1 = groups[n1];
            }
            const float v0 = (c[nt][0] + red[rt][lane][nt * 4 + 0]) * inv, v1 = (c[nt][1] + red[rt][lane][nt * 4 + 1]) * inv;
            const float v2 = (c[nt][2] + red[rt][lane][nt * 4 + 2]) * inv, v3 = (c[nt][3] + red[rt][lane][nt * 4 + 3]) * inv;
            if (r < rows) {
                if (n0 < ncols) y[int64_t(out0) * rows + r] = __float2bfloat16_rn(v0);
                if (n1 < ncols) y[int64_t(out1) * rows + r] = __float2bfloat16_rn(v1);
            }
            if (r + 8 < rows) {
                if (n0 < ncols) y[int64_t(out0) * rows + r + 8] = __float2bfloat16_rn(v2);
                if (n1 < ncols) y[int64_t(out1) * rows + r + 8] = __float2bfloat16_rn(v3);
            }
        }
    }
}

// TQ3_4S (repacked) -> FP16 [rows][k], true values in the rotated basis (the tq3_matvec_h_k decode / TQ3_CB_Q):
// feeds cuBLAS for large prefill chunks. One thread per 32-weight block, natural k order.
__global__ void tq3_dequant_h_k(const uint4 * w, __half * out, int64_t nblocks, uint32_t cblo_u, uint32_t cbhi_u) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint4 q = w[i];
    const uint32_t NEG1152 = 0xE480E480u;
    const __half2 inv = __float2half2_rn(1.f / TQ3_CB_Q);
    const uint32_t invu = *reinterpret_cast<const uint32_t *>(&inv);
    uint32_t sg[4], og[4];
#pragma unroll
    for (int gi = 0; gi < 4; ++gi) {
        const uint32_t b = (q.x >> (8 * gi)) & 255u, hh = b ? (b << 5) + 0x1800u : 0u;
        sg[gi] = hh | (hh << 16);
        og[gi] = hmul2u(sg[gi], NEG1152);
    }
    const uint32_t sl = ((q.y >> 3) & 0x1111u) | ((q.z >> 2) & 0x2222u) | ((q.w >> 1) & 0x4444u);
    const uint32_t sl2 = ((q.y >> 19) & 0x1111u) | ((q.z >> 18) & 0x2222u) | ((q.w >> 17) & 0x4444u);
    const uint32_t sel[8] = {q.y, q.y >> 16, q.z, q.z >> 16, q.w, q.w >> 16, sl, sl2};
    uint32_t o[16];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sel[u] & 0x7777u);
        o[2 * u]     = hmul2u(hfma2u(__byte_perm(ua, 0x64646464u, 0x5140), sg[u / 2], og[u / 2]), invu);
        o[2 * u + 1] = hmul2u(hfma2u(__byte_perm(ua, 0x64646464u, 0x5342), sg[u / 2], og[u / 2]), invu);
    }
    uint4 * dst = reinterpret_cast<uint4 *>(out + i * 32);
#pragma unroll
    for (int j = 0; j < 4; ++j) dst[j] = make_uint4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]);
}

__global__ void f32_to_bf16_k(const float * x, __nv_bfloat16 * y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}

__global__ void i8h_quantize_k(const float * x, int8_t * w8, __half * sc, int64_t ngroups) {
    const int64_t gi = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gi >= ngroups) return;
    const float * xg = x + gi * 16;
    float m = 0.f;
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(xg[i]));
    const __half hs = __float2half_rn(m / 127.f);
    const float s = __half2float(hs);
    for (int i = 0; i < 16; ++i) w8[gi * 16 + i] = int8_t(s > 0.f ? max(-127, min(127, __float2int_rn(xg[i] / s))) : 0);
    sc[gi] = hs;
}

// stage 1: grid (chunks, rows), one top-16 per chunk; stage 2: top-16 over chunks*16 candidates per row
__global__ void topk16_chunk_k(const float * x, int len, int chunk, float * cv, uint32_t * ci) {
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    __shared__ float lv_s;
    __shared__ uint32_t li_s;
    const int row = blockIdx.y, c = blockIdx.x;
    const int beg = c * chunk, end = min(len, beg + chunk);
    const float * xr = x + int64_t(row) * len;
    if (threadIdx.x == 0) {
        lv_s = CUDART_INF_F;
        li_s = 0xffffffffu;
    }
    __syncthreads();
    for (int r = 0; r < 16; ++r) {
        float bv = -CUDART_INF_F;
        uint32_t bi = 0xffffffffu;
        const float lv = lv_s;
        const uint32_t li = li_s;
        for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
            const float v = xr[i];
            const bool below = v < lv || (v == lv && uint32_t(i) > li);
            if (below && (v > bv || (v == bv && uint32_t(i) < bi))) {
                bv = v;
                bi = i;
            }
        }
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (threadIdx.x % 32 == 0) {
            sv[threadIdx.x / 32] = bv;
            si[threadIdx.x / 32] = bi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < 8; ++w)
                if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) {
                    bv = sv[w];
                    bi = si[w];
                }
            const int64_t o = (int64_t(row) * gridDim.x + c) * 16 + r;
            cv[o] = bv;
            ci[o] = bi;
            lv_s = bv;
            li_s = bi;
        }
        __syncthreads();
    }
}

__global__ void topk16_merge_k(const float * cv, const uint32_t * ci, int ncand, float * vals, uint32_t * idx) {
    // one warp per row; ncand <= 1024: repeated argmax over (value desc, index asc) strictly after the last pick
    const int row = blockIdx.x, lane = threadIdx.x;
    const float * v = cv + int64_t(row) * ncand;
    const uint32_t * ix = ci + int64_t(row) * ncand;
    float lv = CUDART_INF_F;
    uint32_t li = 0xffffffffu;
    for (int r = 0; r < 16; ++r) {
        float bv = -CUDART_INF_F;
        uint32_t bi = 0xffffffffu;
        for (int i = lane; i < ncand; i += 32) {
            const float a = v[i];
            const uint32_t id = ix[i];
            const bool below = a < lv || (a == lv && id > li);
            if (below && (a > bv || (a == bv && id < bi))) {
                bv = a;
                bi = id;
            }
        }
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            vals[row * 16 + r] = bv;
            idx[row * 16 + r] = bi;
        }
        lv = bv;
        li = bi;
    }
}

__global__ void bf16_to_h_k(const __nv_bfloat16 * x, __half * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(__bfloat162float(x[i]));
}

__global__ void bf16_to_f32_k(const __nv_bfloat16 * x, float * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]);
}

template <int N>
void tq3_launch(const void * w, const float * xr, __nv_bfloat16 * y, int rows, int k, cudaStream_t s) {
    constexpr int RPW = 2;
    const int grid = (rows + 8 * RPW - 1) / (8 * RPW);
    tq3_matvec_k<N, RPW><<<grid, 256, 0, s>>>(static_cast<const uint4 *>(w), xr, y, rows, k);
}

template <int N>
void q6k_launch(const void * w, const float * x, float * y, int rows, int k, cudaStream_t s) {
    q6k_matvec_w_k<N><<<(rows + 7) / 8, 256, 0, s>>>(static_cast<const uint8_t *>(w), x, y, rows, k);
}
