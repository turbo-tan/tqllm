// ctv: compressed-KV (int8 / trellis, rotated basis) verify attention. Included by attention.cu inside namespace tqllm_cuda (kernels and launchers; no includes here).
// CTV verify attention (int8 K with FP16 scale per 32 + Turbo3 V), DIM 256, rows = nq * group <= 48, 8 warps,
// sized for 2 blocks per SM (~46 KB shared) so one block's barrier phases overlap the other's.
//   S = Q K^T on int8 tensor cores (m16n8k32): Q is quantised per (row, 32 dims) in the prologue; each 32-dim block's
//     int32 product is rescaled by q_scale * k_scale. Warp w -> key n-tile w % 4, m-tiles {0, 1} (w < 4) or {2}.
//   O += P V on FP16 tensor cores (FP16 accumulation per 32-key tile, then FP32): V fragments are decoded from the Turbo3 blocks in registers (rotated basis;
//     the caller applies the inverse WHT). Warp w -> dims 32 w .. 32 w + 31, all m-tiles.
namespace ctv {
constexpr int DIM = 256, TILE = 32, MAXR = 48, RS = DIM + 16;  // int8 row stride 272 B: conflict-free 32-bit loads
constexpr int V3B = 100, SCS = TILE + 4, NBK = DIM / 32;  // V3B = sizeof(Turbo3V256)
constexpr int K4B = 132;  // sizeof(Turbo4K256)
// K region per stage: Q8 K = int8 rows (RS) + FP16 scales per 32; Turbo4 K = 132-byte records
__host__ __device__ constexpr size_t kbytes(bool t4k) { return t4k ? size_t(TILE) * K4B : size_t(TILE) * RS + size_t(TILE) * NBK * 2; }
__host__ __device__ constexpr size_t vbytes(bool t4v) { return size_t(TILE) * (t4v ? K4B : V3B); }
__host__ __device__ constexpr size_t smem(bool t4k, bool t4v = false) {
    return size_t(MAXR) * RS + size_t(MAXR) * NBK * 4 + 2 * kbytes(t4k) + (2 * vbytes(t4v) + 15) / 16 * 16 + 8 * 4 +
           2 * size_t(MAXR) * SCS * 4 + 2 * size_t(MAXR) * 4;
}
__device__ __constant__ uint32_t S1BITS[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};  // turbo3 sign flips (1 = -)
__device__ __constant__ uint32_t S2BITS[4] = {0x16ACEE90u, 0x3F628FDCu, 0xB7A5357Au, 0xBEA56562u};
}

// 4 Turbo4 K nibbles ((negative << 3) | rank) -> 4 int8 levels +-{6,18,31,44,58,75,96,127}
__device__ __forceinline__ uint32_t t4k_expand(uint32_t c) {
    const uint32_t pm = __byte_perm(0x2C1F1206u, 0x7F604B3Au, c & 0x7777u);
    uint32_t sg = (c >> 3) & 0x1111u;
    sg = (sg & 0x0011u) | ((sg & 0x1100u) << 8);         // nibble sign bits -> bits 0, 4, 16, 20
    sg = (sg & 0x00010001u) | ((sg & 0x00100010u) << 4); // -> bits 0, 8, 16, 24 (one per byte)
    return __vadd4(pm ^ (sg * 0xFFu), sg);               // two's-complement negate the flagged bytes
}

__device__ __forceinline__ uint32_t hmul2_u(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint32_t hadd2_u(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ void mma_f16acc(uint32_t (&c)[2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// T4K: Turbo4 K (Q rotated per 128 by the same transform, int8 per (row, 128); scores rescaled twice per head)
// T4V: Turbo4 V records (4-bit, PV layout) instead of Turbo3 V.
template <bool T4K, bool T4V>
__global__ void __launch_bounds__(256, 2) attn_ctv_k(const __nv_bfloat16 * q, const void * kq_, const uint8_t * v3,
                                                     const __half * ksc, float * part, const int * dseq0, int nq,
                                                     int heads, int kvheads, float scale, int nsplit, int query_offset) {
    using namespace ctv;
    constexpr int MT = MAXR / 16;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int8_t * q8 = reinterpret_cast<int8_t *>(smem_raw);                          // [MAXR][RS]
    float * qsc = reinterpret_cast<float *>(q8 + MAXR * RS);                     // [MAXR][NBK] (T4K: 2 used)
    int8_t * k8 = reinterpret_cast<int8_t *>(qsc + MAXR * NBK);                  // Q8: [2][TILE][RS]; T4K: [2][TILE][K4B]
    __half * kss = reinterpret_cast<__half *>(k8 + 2 * TILE * RS);               // Q8: [2][TILE][NBK]
    uint8_t * v3s = reinterpret_cast<uint8_t *>(k8) + 2 * kbytes(T4K);           // [2][TILE][V3B]
    const auto * kq = static_cast<const int8_t *>(kq_);
    const auto * k4g = static_cast<const uint8_t *>(kq_);
    constexpr int VB = T4V ? K4B : V3B;  // V record bytes per key
    float * lut = reinterpret_cast<float *>(v3s + (2 * vbytes(T4V) + 15) / 16 * 16);
    float * scb = lut + 8;                                                       // [2][MAXR][SCS], col 32 = alpha
    float * mrun = scb + 2 * MAXR * SCS, * lrun = mrun + MAXR;                   // running max / sum per row
    const int kh = blockIdx.x, split = blockIdx.y, t = threadIdx.x;
    const int warp = t / 32, lane = t % 32, gid = lane / 4, tig = lane % 4;
    const int g = heads / kvheads, rows = nq * g, mt = (rows + 15) / 16;
    const int seq0 = *dseq0 + query_offset, seq_last = seq0 + nq - 1;
    // tiles interleaved over the splits (tile i -> split i % nsplit): a query's result does not depend on nq
    const int stride = nsplit * TILE, beg = split * TILE, end = seq_last;

    auto stage_k = [&](int buf, int t0) {
        const int n = min(TILE, end - t0);
        if constexpr (T4K) {
            for (int i = t; i < TILE * (K4B / 4); i += blockDim.x) {
                const int p = i / (K4B / 4), w = i % (K4B / 4);
                if (p < n)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(
                                     uint32_t(__cvta_generic_to_shared(k8 + (buf * TILE + p) * K4B + w * 4))),
                                 "l"(k4g + (int64_t(t0 + p) * kvheads + kh) * K4B + w * 4));
            }
        } else {
            for (int i = t; i < TILE * (DIM / 16); i += blockDim.x) {
                const int p = i / (DIM / 16), c = i % (DIM / 16);
                if (p < n) cp_async16(k8 + (buf * TILE + p) * RS + c * 16, kq + (int64_t(t0 + p) * kvheads + kh) * DIM + c * 16);
            }
            if (t < n) cp_async16(kss + (buf * TILE + t) * NBK, ksc + (int64_t(t0 + t) * kvheads + kh) * NBK);
        }
    };
    auto stage_v = [&](int buf, int t0) {
        const int n = min(TILE, end - t0);
        for (int i = t; i < TILE * (VB / 4); i += blockDim.x) {
            const int p = i / (VB / 4), w = i % (VB / 4);
            if (p < n)
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(
                                 uint32_t(__cvta_generic_to_shared(v3s + (buf * TILE + p) * VB + w * 4))),
                             "l"(v3 + (int64_t(t0 + p) * kvheads + kh) * VB + w * 4));
        }
    };
    auto commit = [] { asm volatile("cp.async.commit_group;\n" ::); };
    const int ntile = beg < end ? (end - beg + stride - 1) / stride : 0;
    for (int u = 0; u < 2 && u < ntile; ++u) {
        stage_k(u, beg + u * stride);
        stage_v(u, beg + u * stride);
    }
    commit();

    if constexpr (T4K)  // Q -> rotated (same transform as K) -> int8 per (row, 128): warp per (row, half), 4 values/lane
        for (int pr = warp; pr < MAXR * 2; pr += 8) {
            const int r = pr / 2, h = pr % 2;
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (r < rows) {
                const __nv_bfloat16 * qr = q + (int64_t(r / g) * heads + kh * g + r % g) * DIM + h * 128 + lane * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(qr[j]);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if ((S1BITS[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) v[j] = -v[j];
            float a = v[0], b = v[1];
            v[0] = a + b, v[1] = a - b;
            a = v[2], b = v[3];
            v[2] = a + b, v[3] = a - b;
            a = v[0], b = v[2];
            v[0] = a + b, v[2] = a - b;
            a = v[1], b = v[3];
            v[1] = a + b, v[3] = a - b;
#pragma unroll
            for (int m = 1; m < 32; m <<= 1)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float other = __shfl_xor_sync(0xffffffffu, v[j], m);
                    v[j] = (lane & m) ? other - v[j] : v[j] + other;
                }
            float mx = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] *= 0.08838834764831845f;
                if ((S2BITS[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) v[j] = -v[j];
                mx = fmaxf(mx, fabsf(v[j]));
            }
            for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float sq = mx / 127.f;
            uint32_t pk = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) pk |= (uint32_t(sq > 0.f ? __float2int_rn(v[j] / sq) : 0) & 0xFFu) << (8 * j);
            *reinterpret_cast<uint32_t *>(q8 + r * RS + h * 128 + lane * 4) = pk;
            if (lane == 0) qsc[r * NBK + h] = sq;
        }
    else
    // Q -> int8 per (row, 32 dims): warp handles (row, block) pairs, lane = element
    for (int pr = warp; pr < MAXR * NBK; pr += 8) {
        const int r = pr / NBK, b = pr % NBK;
        float v = 0.f;
        if (r < rows) v = __bfloat162float(q[(int64_t(r / g) * heads + kh * g + r % g) * DIM + b * 32 + lane]);
        float m = fabsf(v);
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float s = m / 127.f;
        q8[r * RS + b * 32 + lane] = int8_t(s > 0.f ? __float2int_rn(v / s) : 0);
        if (lane == 0) qsc[r * NBK + b] = s;
    }
    if (t < 8) lut[t] = turbo3_attn_centroids[t];
    for (int r = t; r < MAXR; r += blockDim.x) {
        mrun[r] = -CUDART_INF_F;
        lrun[r] = 0.f;
    }
    const int nt = warp % 4, m_lo = warp < 4 ? 0 : 2, m_n = warp < 4 ? 2 : 1;  // QK ownership
    const int d0 = warp * 32, vsub = d0 / 128, vj0 = d0 % 128;                 // PV ownership (one Turbo3 block)
    float o[MT][4][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[m][j][0] = o[m][j][1] = o[m][j][2] = o[m][j][3] = 0.f;

    // Software pipeline, 2 barriers per tile: phase A = QK(tile u + 1) and PV(tile u) (independent tensor work);
    // phase B = softmax(tile u + 1). Scores are double-buffered; K(u + 2) is prefetched in phase A into the slot QK(u)
    // freed, V(u + 2) in phase B into the slot PV(u) freed.
    auto qk_tile = [&](int buf, int t0, float * sc) {
        const int n = min(TILE, end - t0);
        if constexpr (T4K) {  // S = Q K^T in the rotated basis, int32 per 128-dim half, 2 rescales
            int ci[2][2][4] = {};
            const uint8_t * krec = reinterpret_cast<const uint8_t *>(k8) + (buf * TILE + nt * 8 + gid) * K4B;
#pragma unroll
            for (int b = 0; b < NBK; ++b) {
                const uint32_t cw = *reinterpret_cast<const uint32_t *>(krec + 4 + (b * 4 + tig) * 4);
                const uint32_t b0 = t4k_expand(cw & 0xFFFFu), b1 = t4k_expand(cw >> 16);
#pragma unroll
                for (int mm = 0; mm < 2; ++mm) {
                    if (mm >= m_n || m_lo + mm >= mt) break;
                    const int8_t * qa = q8 + ((m_lo + mm) * 16 + gid) * RS + b * 32 + 4 * tig;
                    mma_s8(ci[mm][b / 4], *reinterpret_cast<const uint32_t *>(qa),
                           *reinterpret_cast<const uint32_t *>(qa + 8 * RS), *reinterpret_cast<const uint32_t *>(qa + 16),
                           *reinterpret_cast<const uint32_t *>(qa + 8 * RS + 16), b0, b1);
                }
            }
            const __half * sk0 = reinterpret_cast<const __half *>(reinterpret_cast<const uint8_t *>(k8) +
                                                                  (buf * TILE + nt * 8 + 2 * tig) * K4B);
            const __half * sk1 = reinterpret_cast<const __half *>(reinterpret_cast<const uint8_t *>(sk0) + K4B);
#pragma unroll
            for (int mm = 0; mm < 2; ++mm) {
                if (mm >= m_n || m_lo + mm >= mt) break;
                const int r0 = (m_lo + mm) * 16 + gid;
                float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float q0s = qsc[r0 * NBK + h], q1s = qsc[(r0 + 8) * NBK + h];
                    const float k0s = __half2float(sk0[h]), k1s = __half2float(sk1[h]);
                    cf[0] += float(ci[mm][h][0]) * q0s * k0s;
                    cf[1] += float(ci[mm][h][1]) * q0s * k1s;
                    cf[2] += float(ci[mm][h][2]) * q1s * k0s;
                    cf[3] += float(ci[mm][h][3]) * q1s * k1s;
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = r0 + (i >= 2 ? 8 : 0), p = nt * 8 + tig * 2 + (i & 1);
                    sc[r * SCS + p] = r < rows && p < n && t0 + p < seq0 + r / g ? cf[i] * scale : -CUDART_INF_F;
                }
            }
        } else {  // S = Q K^T
            float cf[2][4] = {};
            const int8_t * kb = k8 + (buf * TILE + nt * 8 + gid) * RS;
            const __half * ks0 = kss + (buf * TILE + nt * 8 + 2 * tig) * NBK;
#pragma unroll
            for (int b = 0; b < NBK; ++b) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t *>(kb + b * 32 + 4 * tig);
                const uint32_t b1 = *reinterpret_cast<const uint32_t *>(kb + b * 32 + 16 + 4 * tig);
                const float k0s = __half2float(ks0[b]), k1s = __half2float(ks0[NBK + b]);
#pragma unroll
                for (int mm = 0; mm < 2; ++mm) {
                    if (mm >= m_n || m_lo + mm >= mt) break;
                    const int r0 = (m_lo + mm) * 16 + gid;
                    const int8_t * qa = q8 + r0 * RS + b * 32 + 4 * tig;
                    int ci[4] = {0, 0, 0, 0};
                    mma_s8(ci, *reinterpret_cast<const uint32_t *>(qa), *reinterpret_cast<const uint32_t *>(qa + 8 * RS),
                           *reinterpret_cast<const uint32_t *>(qa + 16), *reinterpret_cast<const uint32_t *>(qa + 8 * RS + 16),
                           b0, b1);
                    const float q0s = qsc[r0 * NBK + b], q1s = qsc[(r0 + 8) * NBK + b];
                    cf[mm][0] += float(ci[0]) * q0s * k0s;
                    cf[mm][1] += float(ci[1]) * q0s * k1s;
                    cf[mm][2] += float(ci[2]) * q1s * k0s;
                    cf[mm][3] += float(ci[3]) * q1s * k1s;
                }
            }
#pragma unroll
            for (int mm = 0; mm < 2; ++mm) {
                if (mm >= m_n || m_lo + mm >= mt) break;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = (m_lo + mm) * 16 + gid + (i >= 2 ? 8 : 0), p = nt * 8 + tig * 2 + (i & 1);
                    sc[r * SCS + p] = r < rows && p < n && t0 + p < seq0 + r / g ? cf[mm][i] * scale : -CUDART_INF_F;
                }
            }
        }
    };
    auto softmax_tile = [&](float * sc) {
        if (t < rows * 4) {  // 4 threads per row; thread q owns key pairs (8i + 2q, 8i + 2q + 1), i = 0..3
            const int r = t / 4, q = t % 4;
            const unsigned qm4 = 0xFu << (lane & 28);
            float * row = sc + r * SCS;
            float v[8], mx = -CUDART_INF_F;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                mx = fmaxf(mx, v[2 * i] = row[8 * i + 2 * q]);
                mx = fmaxf(mx, v[2 * i + 1] = row[8 * i + 2 * q + 1]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(qm4, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(qm4, mx, 2));  // all quad reads of the row are done past this point
            const float mold = mrun[r], mnew = fmaxf(mold, mx);
            float sum = 0.f;
            uint32_t * ph = reinterpret_cast<uint32_t *>(row);  // P as FP16 pairs: word j = keys (2j, 2j + 1)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float e0 = v[2 * i] > -CUDART_INF_F ? __expf(v[2 * i] - mnew) : 0.f;
                const float e1 = v[2 * i + 1] > -CUDART_INF_F ? __expf(v[2 * i + 1] - mnew) : 0.f;
                ph[4 * i + q] = pack_h2(e0, e1);
                sum += e0 + e1;
            }
            sum += __shfl_xor_sync(qm4, sum, 1);
            sum += __shfl_xor_sync(qm4, sum, 2);
            if (q == 0) {
                const float alpha = mold > -CUDART_INF_F ? __expf(mold - mnew) : 0.f;
                row[TILE] = alpha;
                lrun[r] = lrun[r] * alpha + sum;
                mrun[r] = mnew;
            }
        }
    };
    auto pv_tile = [&](int buf, int t0, const float * sc) {
        const int n = min(TILE, end - t0);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= mt) break;
            const float s0 = sc[(m * 16 + gid) * SCS + TILE], s1 = sc[(m * 16 + gid + 8) * SCS + TILE];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o[m][j][0] *= s0;
                o[m][j][1] *= s0;
                o[m][j][2] *= s1;
                o[m][j][3] *= s1;
            }
        }
        uint32_t acc[MT][4][2] = {};  // per-tile FP16 accumulation (2x tensor rate on GA10x), folded into FP32 below
#pragma unroll
        for (int kk = 0; kk < TILE / 16; ++kk) {
            // V fragments: lane needs keys kk*16 + {2tig, 2tig+1, 2tig+8, 2tig+9}, dims d0 + 8j + gid (j = 0..3)
            uint32_t bv[4][2];
            if constexpr (T4V) {
                // Turbo4 V: lane l = gid reads one 16-bit field per key (its 4 dims d0 + 8j + l), expands to int8 levels,
                // makes FP16 (1024 + level + 128) with PRMT, pairs keys per register, then (x - 1152) * scale exactly.
                uint32_t h01[4], h23[4], sh[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int key = kk * 16 + 2 * tig + (u & 1) + (u >> 1) * 8;
                    const uint8_t * rec = v3s + (buf * TILE + key) * VB;
                    const uint32_t f = *reinterpret_cast<const uint16_t *>(rec + 4 + (warp * 4 + gid / 2) * 4 + (gid % 2) * 2);
                    const uint32_t lv = t4k_expand(f) ^ 0x80808080u;
                    h01[u] = __byte_perm(lv, 0x64646464u, 0x5140);
                    h23[u] = __byte_perm(lv, 0x64646464u, 0x5342);
                    sh[u] = key < n ? *reinterpret_cast<const uint16_t *>(rec + 2 * vsub) : 0u;
                }
                const uint32_t s01 = sh[0] | (sh[1] << 16), s23 = sh[2] | (sh[3] << 16), NEG = 0xE480E480u;  // -1152
                auto fin = [&](uint32_t x, uint32_t sc2) { return hmul2_u(hadd2_u(x, NEG), sc2); };
                bv[0][0] = fin(__byte_perm(h01[0], h01[1], 0x5410), s01);
                bv[1][0] = fin(__byte_perm(h01[0], h01[1], 0x7632), s01);
                bv[2][0] = fin(__byte_perm(h23[0], h23[1], 0x5410), s01);
                bv[3][0] = fin(__byte_perm(h23[0], h23[1], 0x7632), s01);
                bv[0][1] = fin(__byte_perm(h01[2], h01[3], 0x5410), s23);
                bv[1][1] = fin(__byte_perm(h01[2], h01[3], 0x7632), s23);
                bv[2][1] = fin(__byte_perm(h23[2], h23[3], 0x5410), s23);
                bv[3][1] = fin(__byte_perm(h23[2], h23[3], 0x7632), s23);
            } else
            {
                // lane l = gid owns dims d0 + 8j + l: one 12-bit field per key (Turbo3V256 layout); |centroid| halves
                // come from a PRMT on an FP16 table, the sign by XOR, the norm by HMUL2 (two keys per register)
                uint32_t fld[4], nh[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int key = kk * 16 + 2 * tig + (u & 1) + (u >> 1) * 8;
                    const uint32_t * rec = reinterpret_cast<const uint32_t *>(v3s + (buf * TILE + key) * VB);
                    const int k0 = (12 * gid) / 32;
                    const uint32_t lo = rec[1 + 3 * warp + k0], hi = k0 < 2 ? rec[2 + 3 * warp + k0] : 0u;
                    fld[u] = uint32_t(((uint64_t(hi) << 32) | lo) >> (12 * gid - 32 * k0)) & 0xFFFu;
                    nh[u] = key < n ? (vsub ? rec[0] >> 16 : rec[0] & 0xFFFFu) : 0u;
                }
                const uint32_t n01 = nh[0] | (nh[1] << 16), n23 = nh[2] | (nh[3] << 16);
                constexpr uint32_t M0 = 0x2C47258Cu, M1 = 0x32162F9Au;  // fp16 pairs {0.021663, 0.066822}, {0.118786, 0.190207}
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    auto pair = [&](uint32_t fa, uint32_t fb, uint32_t nn) {
                        const uint32_t ca = (fa >> (3 * j)) & 7u, cb = (fb >> (3 * j)) & 7u;
                        const uint32_t sel = (0x10u + 0x22u * (ca & 3u)) | ((0x10u + 0x22u * (cb & 3u)) << 8);
                        const uint32_t sgn = ((~ca & 4u) << 13) | ((~cb & 4u) << 29);  // negative -> half sign bit
                        return hmul2_u(__byte_perm(M0, M1, sel) ^ sgn, nn);
                    };
                    bv[j][0] = pair(fld[0], fld[1], n01);
                    bv[j][1] = pair(fld[2], fld[3], n23);
                }
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m >= mt) break;
                const uint32_t * p0 = reinterpret_cast<const uint32_t *>(sc + (m * 16 + gid) * SCS) + kk * 8 + tig;
                const uint32_t * p1 = p0 + 8 * SCS;
                const uint32_t a0 = p0[0], a1 = p1[0], a2 = p0[4], a3 = p1[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_f16acc(acc[m][j], a0, a1, a2, a3, bv[j][0], bv[j][1]);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&acc[m][j][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&acc[m][j][1]));
                o[m][j][0] += lo.x;
                o[m][j][1] += lo.y;
                o[m][j][2] += hi.x;
                o[m][j][3] += hi.y;
            }
    };
    if (ntile > 0) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        qk_tile(0, beg, scb);
        __syncthreads();
        softmax_tile(scb);
        __syncthreads();
    }
    for (int u = 0; u < ntile; ++u) {
        const int cur = u & 1, nxt = cur ^ 1;
        if (u + 2 < ntile) stage_k(cur, beg + (u + 2) * stride);
        commit();
        if (u + 1 < ntile) qk_tile(nxt, beg + (u + 1) * stride, scb + nxt * MAXR * SCS);
        pv_tile(cur, beg + u * stride, scb + cur * MAXR * SCS);
        __syncthreads();
        if (u + 2 < ntile) stage_v(cur, beg + (u + 2) * stride);
        commit();
        if (u + 1 < ntile) softmax_tile(scb + nxt * MAXR * SCS);
        asm volatile("cp.async.wait_group 1;\n" ::);  // K(u + 2) landed; V(u + 2) may still be in flight
        __syncthreads();
    }
    auto out = [&](int r) { return part + ((int64_t(r / g) * heads + kh * g + r % g) * nsplit + split) * (DIM + 2); };
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = m * 16 + gid + (i >= 2 ? 8 : 0);
                if (r < rows) out(r)[d0 + j * 8 + tig * 2 + (i & 1)] = o[m][j][i];
            }
    if (t < rows) {
        out(t)[DIM] = mrun[t];
        out(t)[DIM + 1] = lrun[t];
    }
}
