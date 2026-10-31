// Q6_K matvec and the int8 + FP16-scale (i8h) LM head kernels. Included by quant.cu inside its anonymous namespace (kernels only; no includes here).
// Q6_K: one warp per row; lane l handles, in each 128-half of a 256 block, elements l, l+32, l+64, l+96.
template <int N>
__global__ void __launch_bounds__(256) q6k_matvec_k(const uint8_t * w, const float * x, float * y, int rows, int k) {
    const int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const int nblk = k / Q6K_BLOCK;
    const uint8_t * wr = w + int64_t(row) * nblk * Q6K_BLOCK_BYTES;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; ++n) acc[n] = 0.f;
    const int is = lane / 16;
    for (int b = 0; b < nblk; ++b) {
        const uint8_t * blk = wr + b * Q6K_BLOCK_BYTES;
        const uint8_t *ql = blk, *qh = blk + 128;
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
        const float d = __half2float(*reinterpret_cast<const __half *>(blk + 208));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int l0 = ql[h * 64 + lane], l1 = ql[h * 64 + lane + 32], hb = qh[h * 32 + lane];
            const float q1 = float(((l0 & 0xF) | ((hb & 3) << 4)) - 32) * d * sc[h * 8 + is];
            const float q2 = float(((l1 & 0xF) | (((hb >> 2) & 3) << 4)) - 32) * d * sc[h * 8 + is + 2];
            const float q3 = float(((l0 >> 4) | (((hb >> 4) & 3) << 4)) - 32) * d * sc[h * 8 + is + 4];
            const float q4 = float(((l1 >> 4) | (((hb >> 6) & 3) << 4)) - 32) * d * sc[h * 8 + is + 6];
            const int64_t base = int64_t(b) * Q6K_BLOCK + h * 128 + lane;
#pragma unroll
            for (int n = 0; n < N; ++n) {
                const float * xn = x + int64_t(n) * k + base;
                acc[n] += q1 * xn[0] + q2 * xn[32] + q3 * xn[64] + q4 * xn[96];
            }
        }
    }
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float v = acc[n];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) y[int64_t(n) * rows + row] = v;
    }
}


// Q6_K with word loads: 8 lanes per 128-element half block, lane t owns l = 4t..4t+3 of that half (16 weights:
// l, l+32, l+64, l+96 for each of its 4 l values). A warp covers 2 blocks per iteration.
template <int N>
__global__ void __launch_bounds__(256) q6k_matvec_w_k(const uint8_t * w, const float * x, float * y, int rows, int k) {
    const int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const int nblk = k / Q6K_BLOCK;
    const uint8_t * wr = w + int64_t(row) * nblk * Q6K_BLOCK_BYTES;
    const int half = lane / 8 % 2, sub = lane / 16, t = lane % 8, l = 4 * t, is = l / 16;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; ++n) acc[n] = 0.f;
    for (int b = sub; b < nblk; b += 2) {
        const uint8_t * blk = wr + b * Q6K_BLOCK_BYTES;
        const uint32_t ql0 = *reinterpret_cast<const uint16_t *>(blk + half * 64 + l) |
                             (uint32_t(*reinterpret_cast<const uint16_t *>(blk + half * 64 + l + 2)) << 16);
        const uint32_t ql1 = *reinterpret_cast<const uint16_t *>(blk + half * 64 + l + 32) |
                             (uint32_t(*reinterpret_cast<const uint16_t *>(blk + half * 64 + l + 34)) << 16);
        const uint32_t qh = *reinterpret_cast<const uint16_t *>(blk + 128 + half * 32 + l) |
                            (uint32_t(*reinterpret_cast<const uint16_t *>(blk + 128 + half * 32 + l + 2)) << 16);
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192) + half * 8 + is;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk + 208));
        const float d1 = d * sc[0], d2 = d * sc[2], d3 = d * sc[4], d4 = d * sc[6];
        float w1[4], w2[4], w3[4], w4[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int a = (ql0 >> (8 * e)) & 255, c = (ql1 >> (8 * e)) & 255, h = (qh >> (8 * e)) & 255;
            w1[e] = d1 * float(((a & 0xF) | ((h & 3) << 4)) - 32);
            w2[e] = d2 * float(((c & 0xF) | (((h >> 2) & 3) << 4)) - 32);
            w3[e] = d3 * float(((a >> 4) | (((h >> 4) & 3) << 4)) - 32);
            w4[e] = d4 * float(((c >> 4) | (((h >> 6) & 3) << 4)) - 32);
        }
        const int64_t base = int64_t(b) * Q6K_BLOCK + half * 128 + l;
#pragma unroll
        for (int n = 0; n < N; ++n) {
            const float * xn = x + int64_t(n) * k + base;
            const float4 x1 = *reinterpret_cast<const float4 *>(xn), x2 = *reinterpret_cast<const float4 *>(xn + 32);
            const float4 x3 = *reinterpret_cast<const float4 *>(xn + 64), x4 = *reinterpret_cast<const float4 *>(xn + 96);
            acc[n] += w1[0] * x1.x + w1[1] * x1.y + w1[2] * x1.z + w1[3] * x1.w + w2[0] * x2.x + w2[1] * x2.y +
                      w2[2] * x2.z + w2[3] * x2.w + w3[0] * x3.x + w3[1] * x3.y + w3[2] * x3.z + w3[3] * x3.w +
                      w4[0] * x4.x + w4[1] * x4.y + w4[2] * x4.z + w4[3] * x4.w;
        }
    }
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float v = acc[n];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) y[int64_t(n) * rows + row] = v;
    }
}


// One thread per element of one Q6_K row (ggml dequantize_row_q6_K layout).
__global__ void q6k_row_k(const uint8_t * w, int row, __nv_bfloat16 * y, int k, const uint32_t * dtok = nullptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    if (dtok) {  // multi-token: blockIdx.y selects the token, id read on the device
        row = int(dtok[blockIdx.y]);
        y += int64_t(blockIdx.y) * k;
    }
    const int nblk = k / Q6K_BLOCK, b = i / Q6K_BLOCK, e = i % Q6K_BLOCK;
    const uint8_t * blk = w + (int64_t(row) * nblk + b) * Q6K_BLOCK_BYTES;
    const int h = e / 128, q = (e % 128) / 32, l = e % 32, is = l / 16;
    const uint8_t * ql = blk + h * 64;
    const uint8_t hb = blk[128 + h * 32 + l];
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192) + h * 8;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk + 208));
    int v;
    if (q == 0) v = (ql[l] & 0xF) | ((hb & 3) << 4);
    else if (q == 1) v = (ql[l + 32] & 0xF) | (((hb >> 2) & 3) << 4);
    else if (q == 2) v = (ql[l] >> 4) | (((hb >> 4) & 3) << 4);
    else v = (ql[l + 32] >> 4) | (((hb >> 6) & 3) << 4);
    y[i] = __float2bfloat16_rn(d * sc[is + 2 * q] * float(v - 32));
}


// ---- Q6_K -> int8 + FP16 scales (load time) and the FP16 tensor-core LM head ----
__global__ void q6k_to_i8_k(const uint8_t * w, int8_t * w8, __half * sc, int rows, int k) {
    const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // element index
    if (e >= int64_t(rows) * k) return;
    const int row = int(e / k), i = int(e % k), b = i / Q6K_BLOCK, el = i % Q6K_BLOCK;
    const uint8_t * blk = w + (int64_t(row) * (k / Q6K_BLOCK) + b) * Q6K_BLOCK_BYTES;
    const int h = el / 128, q = (el % 128) / 32, l = el % 32, is = l / 16;
    const uint8_t * ql = blk + h * 64;
    const uint8_t hb = blk[128 + h * 32 + l];
    int v;
    if (q == 0) v = (ql[l] & 0xF) | ((hb & 3) << 4);
    else if (q == 1) v = (ql[l + 32] & 0xF) | (((hb >> 2) & 3) << 4);
    else if (q == 2) v = (ql[l] >> 4) | (((hb >> 4) & 3) << 4);
    else v = (ql[l + 32] >> 4) | (((hb >> 6) & 3) << 4);
    w8[e] = int8_t(v - 32);
    if (el % 16 == 0) {  // one scale per 16 consecutive weights (Q6_K sub-block)
        const int8_t * scl = reinterpret_cast<const int8_t *>(blk + 192) + h * 8;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk + 208));
        sc[e / 16] = __float2half_rn(d * float(scl[is + 2 * q]));
    }
}

// CTA 8 warps on a 16-row tile, split-K over warps. The 4 lanes of a quad take 4 consecutive 16-weight scale groups
// of the same rows (64 contiguous bytes per row per quad, no redundant loads); k is permuted identically in A and B
// so MMA u (0..3) takes elements 4u..4u+3 of each lane's group. Per element: (int8 ^ 0x80) -> fp16 (1024 + u),
// + (-1152) = q exactly, * s (the FP16 d * sc of that group).
template <int NCOL, class OUT = float>
__global__ void __launch_bounds__(256) i8h_matvec_k(const int8_t * w8, const __half * sc, const __half * xh, OUT * y,
                                                    int rows, int k, int ldy = 0) {
    const int64_t ld = ldy ? ldy : rows;  // output column stride
    constexpr int NACC = NCOL > 8 ? 8 : 4;
    __shared__ float red[8][32][NACC];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16, nk = k / 16;
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    const uint32_t NEG1152 = 0xE480E480u;
    float c[NACC] = {};
    for (int s0 = warp * 4; s0 < nk; s0 += 32) {
        const int s = s0 + t;
        const bool ok = s < nk;
        uint4 wa = make_uint4(0, 0, 0, 0), wb = make_uint4(0, 0, 0, 0);
        uint32_t sau = 0, sbu = 0;
        uint4 x0 = make_uint4(0, 0, 0, 0), x1 = make_uint4(0, 0, 0, 0);
        uint4 x02 = make_uint4(0, 0, 0, 0), x12 = make_uint4(0, 0, 0, 0);
        if (ok) {
            wa = __ldg(reinterpret_cast<const uint4 *>(w8 + int64_t(ra) * k + int64_t(s) * 16));
            wb = __ldg(reinterpret_cast<const uint4 *>(w8 + int64_t(rb) * k + int64_t(s) * 16));
            const __half sa = sc[int64_t(ra) * nk + s], sb = sc[int64_t(rb) * nk + s];
            const __half2 sa2 = __halves2half2(sa, sa), sb2 = __halves2half2(sb, sb);
            sau = *reinterpret_cast<const uint32_t *>(&sa2);
            sbu = *reinterpret_cast<const uint32_t *>(&sb2);
            if (g < NCOL) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g) * k + int64_t(s) * 16);
                x0 = __ldg(&xp[0]);
                x1 = __ldg(&xp[1]);
            }
            if constexpr (NCOL > 8) {
                if (g < NCOL - 8) {
                    const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g + 8) * k + int64_t(s) * 16);
                    x02 = __ldg(&xp[0]);
                    x12 = __ldg(&xp[1]);
                }
            }
        }
        const uint32_t wa4[4] = {wa.x ^ 0x80808080u, wa.y ^ 0x80808080u, wa.z ^ 0x80808080u, wa.w ^ 0x80808080u};
        const uint32_t wb4[4] = {wb.x ^ 0x80808080u, wb.y ^ 0x80808080u, wb.z ^ 0x80808080u, wb.w ^ 0x80808080u};
        const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        const uint32_t xw2[8] = {x02.x, x02.y, x02.z, x02.w, x12.x, x12.y, x12.z, x12.w};
        auto deq = [&](uint32_t w, uint32_t sel, uint32_t su) { return hmul2u(hadd2u(__byte_perm(w, 0x64646464u, sel), NEG1152), su); };
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            mma_f16(c, deq(wa4[u], 0x5140, sau), deq(wb4[u], 0x5140, sbu), deq(wa4[u], 0x5342, sau),
                    deq(wb4[u], 0x5342, sbu), xw[2 * u], xw[2 * u + 1]);
            if constexpr (NCOL > 8)
                mma_f16(c + 4, deq(wa4[u], 0x5140, sau), deq(wb4[u], 0x5140, sbu), deq(wa4[u], 0x5342, sau),
                        deq(wb4[u], 0x5342, sbu), xw2[2 * u], xw2[2 * u + 1]);
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
        auto put = [&](int64_t i, float v2) {
            if constexpr (std::is_same_v<OUT, float>) y[i] = v2;
            else y[i] = __float2bfloat16_rn(v2);
        };
        if (r < rows) {
            if (n0 < NCOL) put(int64_t(n0) * ld + r, v[0]);
            if (n1 < NCOL) put(int64_t(n1) * ld + r, v[1]);
        }
        if (r + 8 < rows) {
            if (n0 < NCOL) put(int64_t(n0) * ld + r + 8, v[2]);
            if (n1 < NCOL) put(int64_t(n1) * ld + r + 8, v[3]);
        }
        if constexpr (NCOL > 8) {
            if (r < rows) {
                if (n0 + 8 < NCOL) put(int64_t(n0 + 8) * ld + r, v[4]);
                if (n1 + 8 < NCOL) put(int64_t(n1 + 8) * ld + r, v[5]);
            }
            if (r + 8 < rows) {
                if (n0 + 8 < NCOL) put(int64_t(n0 + 8) * ld + r + 8, v[6]);
                if (n1 + 8 < NCOL) put(int64_t(n1 + 8) * ld + r + 8, v[7]);
            }
        }
    }
}
