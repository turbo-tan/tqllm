// fap: prefill attention over the compressed (int8) cache. Included by attention.cu inside namespace tqllm_cuda (kernels and launchers; no includes here).
// Prefill attention over a Turbo4 K+V cache (FA2 style). Block = 8 warps x 16 rows = 128 (query, head) rows of one KV
// head; every 32-key tile is decoded once per block into shared memory (K int8 [key][256] + 2 scales, V FP16 [key][256]
// in the rotated basis), double-buffered, one barrier per tile. Each warp keeps S, P and O for its 16 rows in registers.
namespace fap {
constexpr int NW = 8, ROWS = NW * 16, TILE = 32, DIM = 256, QS = DIM + 16, KS = DIM + 16, VS = DIM + 8;
constexpr size_t smem() {
    return size_t(ROWS) * QS + 2 * size_t(TILE) * KS + 2 * size_t(TILE) * VS * 2 + 2 * 2 * TILE * 4 + ROWS * 2 * 4;
}
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(uint32_t(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(uint32_t(__cvta_generic_to_shared(p))));
}

__global__ void __launch_bounds__(fap::NW * 32, 1) attn_t4kv_prefill_k(const __nv_bfloat16 * q, const uint8_t * kc,
                                                                       const uint8_t * vc, __nv_bfloat16 * y,
                                                                       const int * dseq0, int nq, int heads, int kvheads,
                                                                       float scale) {
    using namespace fap;
    constexpr int RB = ctv::K4B;  // 132-byte Turbo4 record
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int8_t * q8 = reinterpret_cast<int8_t *>(smem_raw);            // [ROWS][QS]
    int8_t * k8 = q8 + ROWS * QS;                                   // [2][TILE][KS]
    __half * vh = reinterpret_cast<__half *>(k8 + 2 * TILE * KS);   // [2][TILE][VS]
    float * ksf = reinterpret_cast<float *>(vh + 2 * TILE * VS);    // [2][2 halves][TILE]
    float * qsc = ksf + 2 * 2 * TILE;                               // [ROWS][2]
    const int kh = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32, gid = lane / 4, tig = lane % 4;
    const int g = heads / kvheads, rows = nq * g;
    const int r0 = (gridDim.y - 1 - blockIdx.y) * ROWS;  // longest (latest) row blocks first
    const int seq0 = *dseq0;                             // query i sees keys < seq0 + i
    const int q_lo = r0 / g, q_hi = min(r0 + ROWS - 1, rows - 1) / g;
    const int key_end = seq0 + q_hi, ntile = (key_end + TILE - 1) / TILE, mask_from = seq0 + q_lo;

    // Q -> rotated (same transform as K) -> int8 per (row, 128); rows past the end repeat the last row
    for (int pr = warp; pr < ROWS * 2; pr += NW) {
        const int r = pr / 2, h = pr % 2, rr = min(r0 + r, rows - 1);
        const __nv_bfloat16 * qr = q + (int64_t(rr / g) * heads + kh * g + rr % g) * DIM + h * 128 + lane * 4;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = __bfloat162float(qr[j]);
            if ((ctv::S1BITS[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) v[j] = -v[j];
        }
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
            if ((ctv::S2BITS[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) v[j] = -v[j];
            mx = fmaxf(mx, fabsf(v[j]));
        }
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float sq = mx / 127.f;
        uint32_t pk = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) pk |= (uint32_t(sq > 0.f ? __float2int_rn(v[j] / sq) : 0) & 0xFFu) << (8 * j);
        *reinterpret_cast<uint32_t *>(q8 + r * QS + h * 128 + lane * 4) = pk;
        if (lane == 0) qsc[r * 2 + h] = sq;
    }

    // tile loader: thread -> key t % 32, 32-dim chunk c = t / 32 (K words 4c..4c+3 = dims 32c..; V words 4c.. = chunk c);
    // key-major lanes keep the 16-byte shared stores conflict-free (row strides are 4 banks apart)
    const int lk = t % 32, lc = t / 32;
    uint32_t kr[4], vr[4], ksw = 0, vsw = 0;
    auto load = [&](int t0) {
        const int p = t0 + lk;
        if (p < key_end) {
            const uint32_t * rk = reinterpret_cast<const uint32_t *>(kc + (int64_t(p) * kvheads + kh) * RB);
            const uint32_t * rv = reinterpret_cast<const uint32_t *>(vc + (int64_t(p) * kvheads + kh) * RB);
#pragma unroll
            for (int i = 0; i < 4; ++i) kr[i] = __ldg(rk + 1 + 4 * lc + i), vr[i] = __ldg(rv + 1 + 4 * lc + i);
            ksw = __ldg(rk), vsw = __ldg(rv);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) kr[i] = vr[i] = 0u;
            ksw = vsw = 0u;
        }
    };
    auto store = [&](int buf) {
        uint4 lo, hi;
        lo.x = t4k_expand(kr[0] & 0xFFFFu), hi.x = t4k_expand(kr[0] >> 16);
        lo.y = t4k_expand(kr[1] & 0xFFFFu), hi.y = t4k_expand(kr[1] >> 16);
        lo.z = t4k_expand(kr[2] & 0xFFFFu), hi.z = t4k_expand(kr[2] >> 16);
        lo.w = t4k_expand(kr[3] & 0xFFFFu), hi.w = t4k_expand(kr[3] >> 16);
        int8_t * kd = k8 + (buf * TILE + lk) * KS + lc * 32;
        *reinterpret_cast<uint4 *>(kd) = lo;
        *reinterpret_cast<uint4 *>(kd + 16) = hi;
        if (lc < 2) {
            const __half2 s2 = *reinterpret_cast<const __half2 *>(&ksw);
            ksf[(buf * 2 + lc) * TILE + lk] = __half2float(lc ? __high2half(s2) : __low2half(s2));
        }
        // V: word i holds fields l = 2i, 2i + 1 (4 nibbles j each, dims 32c + 8j + l) -> half2 (l, l + 1) per j
        const uint32_t sh = (lc < 4 ? vsw : vsw >> 16) & 0xFFFFu, s2 = sh | (sh << 16), NEG = 0xE480E480u;  // -1152
        uint32_t out[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t la = t4k_expand(vr[i] & 0xFFFFu) ^ 0x80808080u, lb = t4k_expand(vr[i] >> 16) ^ 0x80808080u;
            const uint32_t a01 = __byte_perm(la, 0x64646464u, 0x5140), a23 = __byte_perm(la, 0x64646464u, 0x5342);
            const uint32_t b01 = __byte_perm(lb, 0x64646464u, 0x5140), b23 = __byte_perm(lb, 0x64646464u, 0x5342);
            out[0][i] = hmul2_u(hadd2_u(__byte_perm(a01, b01, 0x5410), NEG), s2);
            out[1][i] = hmul2_u(hadd2_u(__byte_perm(a01, b01, 0x7632), NEG), s2);
            out[2][i] = hmul2_u(hadd2_u(__byte_perm(a23, b23, 0x5410), NEG), s2);
            out[3][i] = hmul2_u(hadd2_u(__byte_perm(a23, b23, 0x7632), NEG), s2);
        }
        __half * vd = vh + (buf * TILE + lk) * VS + lc * 32;
#pragma unroll
        for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4 *>(vd + 8 * j) = make_uint4(out[j][0], out[j][1], out[j][2], out[j][3]);
    };

    if (ntile > 0) {
        load(0);
        store(0);
        if (ntile > 1) load(TILE);
    }
    __syncthreads();

    const int ra = warp * 16 + gid, rb = ra + 8;
    const float qa0 = qsc[ra * 2], qa1 = qsc[ra * 2 + 1], qb0 = qsc[rb * 2], qb1 = qsc[rb * 2 + 1];
    const int lim_a = seq0 + min(r0 + ra, rows - 1) / g, lim_b = seq0 + min(r0 + rb, rows - 1) / g;  // key < lim visible
    const float sl2 = scale * 1.4426950408889634f;
    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_a = -CUDART_INF_F, m_b = -CUDART_INF_F, l_a = 0.f, l_b = 0.f;
    const int8_t * qrow = q8 + (warp * 16 + (lane % 8) + ((lane / 8) & 1) * 8) * QS + (lane / 16) * 16;

    for (int u = 0; u < ntile; ++u) {
        const int buf = u & 1, t0 = u * TILE;
        const int8_t * kt = k8 + buf * TILE * KS;
        const __half * vt = vh + buf * TILE * VS;
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int ci[4][4] = {};
#pragma unroll
            for (int bp = 0; bp < 2; ++bp) {
                const int b0 = 4 * h + 2 * bp;
                uint32_t a0[4], a1[4];
                ldsm_x4(a0, qrow + b0 * 32);
                ldsm_x4(a1, qrow + (b0 + 1) * 32);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    uint32_t kb[4];
                    ldsm_x4(kb, kt + (nt * 8 + lane % 8) * KS + b0 * 32 + (lane / 8) * 16);
                    mma_s8(ci[nt], a0[0], a0[1], a0[2], a0[3], kb[0], kb[1]);
                    mma_s8(ci[nt], a1[0], a1[1], a1[2], a1[3], kb[2], kb[3]);
                }
            }
            const float qa = h ? qa1 : qa0, qb = h ? qb1 : qb0;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const float2 kk = *reinterpret_cast<const float2 *>(ksf + (buf * 2 + h) * TILE + nt * 8 + 2 * tig);
                s[nt][0] += float(ci[nt][0]) * qa * kk.x;
                s[nt][1] += float(ci[nt][1]) * qa * kk.y;
                s[nt][2] += float(ci[nt][2]) * qb * kk.x;
                s[nt][3] += float(ci[nt][3]) * qb * kk.y;
            }
        }
        const bool msk = t0 + TILE > mask_from;
        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float x = s[nt][i] * sl2;
                if (msk && t0 + nt * 8 + 2 * tig + (i & 1) >= (i < 2 ? lim_a : lim_b)) x = -CUDART_INF_F;
                s[nt][i] = x;
                if (i < 2) mxa = fmaxf(mxa, x);
                else mxb = fmaxf(mxb, x);
            }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(m_a, mxa), nb = fmaxf(m_b, mxb);  // finite: every row sees key 0 in tile 0
        const float al = exp2f(m_a - na), bl = exp2f(m_b - nb);
        m_a = na, m_b = nb;
        uint32_t pf[2][4];
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p0 = exp2f(s[nt][0] - na), p1 = exp2f(s[nt][1] - na);
            const float p2 = exp2f(s[nt][2] - nb), p3 = exp2f(s[nt][3] - nb);
            sa += p0 + p1, sb += p2 + p3;
            pf[nt / 2][(nt & 1) * 2] = pack_h2(p0, p1);
            pf[nt / 2][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
        }
        l_a = l_a * al + sa, l_b = l_b * bl + sb;
        if (__any_sync(0xffffffffu, al != 1.f || bl != 1.f))
#pragma unroll
            for (int j = 0; j < 32; ++j) o[j][0] *= al, o[j][1] *= al, o[j][2] *= bl, o[j][3] *= bl;
        const __half * vrow = vt + ((lane / 8) & 1) * 8 * VS + (lane % 8) * VS + (lane / 16) * 8;
#pragma unroll
        for (int np = 0; np < 16; ++np) {
            uint32_t acc[2][2] = {};
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                uint32_t vb[4];
                ldsm_x4_t(vb, vrow + kk * 16 * VS + np * 16);
                mma_f16acc(acc[0], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vb[0], vb[1]);
                mma_f16acc(acc[1], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vb[2], vb[3]);
            }
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&acc[n][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&acc[n][1]));
                o[2 * np + n][0] += lo.x, o[2 * np + n][1] += lo.y, o[2 * np + n][2] += hi.x, o[2 * np + n][3] += hi.y;
            }
        }
        if (u + 1 < ntile) store(buf ^ 1);
        if (u + 2 < ntile) load(t0 + 2 * TILE);
        __syncthreads();
    }
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    const float ia = l_a > 0.f ? 1.f / l_a : 0.f, ib = l_b > 0.f ? 1.f / l_b : 0.f;
    const int ga = r0 + ra, gb = r0 + rb;
    __nv_bfloat16 * ya = y + (int64_t(ga / g) * heads + kh * g + ga % g) * DIM + 2 * tig;
    __nv_bfloat16 * yb = y + (int64_t(gb / g) * heads + kh * g + gb % g) * DIM + 2 * tig;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        if (ga < rows) *reinterpret_cast<__nv_bfloat162 *>(ya + 8 * j) = __floats2bfloat162_rn(o[j][0] * ia, o[j][1] * ia);
        if (gb < rows) *reinterpret_cast<__nv_bfloat162 *>(yb + 8 * j) = __floats2bfloat162_rn(o[j][2] * ib, o[j][3] * ib);
    }
}
