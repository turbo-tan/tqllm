// fab: prefill attention over a BF16 cache (FA2 style). Included by attention.cu inside namespace tqllm_cuda (kernels and launchers; no includes here).
// Prefill attention over a BF16 cache (FA2 style). Block = NW warps x 16 rows of one KV head; Q and 32-key K / V tiles
// in shared memory (double-buffered on CUDA; single-buffered on HIP, where the dynamic shared memory limit is 64 KB and
// cp_async16 is a plain synchronous copy), S, P (BF16) and O in registers.
namespace fab {
#if defined(TQLLM_WITH_HIP)
constexpr int NW = 2, ROWS = NW * 16, TILE = 32, DIM = 256, LS = DIM + 8, NB = 1;
#else
constexpr int NW = 4, ROWS = NW * 16, TILE = 32, DIM = 256, LS = DIM + 8, NB = 2;
#endif
constexpr size_t smem() { return (size_t(ROWS) + size_t(NB) * 2 * TILE) * LS * 2; }
}

__global__ void __launch_bounds__(fab::NW * 32, 1) attn_bf16_prefill_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kc,
                                                                       const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                                                                       const int * dseq0, int nq, int heads, int kvheads,
                                                                       float scale) {
    using namespace fab;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16 * qs = reinterpret_cast<__nv_bfloat16 *>(smem_raw);  // [ROWS][LS]
    __nv_bfloat16 * ks = qs + ROWS * LS;                                 // [NB][TILE][LS]
    __nv_bfloat16 * vs = ks + NB * TILE * LS;                            // [NB][TILE][LS]
    const int kh = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32, gid = lane / 4, tig = lane % 4;
    const int g = heads / kvheads, rows = nq * g;
    const int r0 = (gridDim.y - 1 - blockIdx.y) * ROWS;  // longest (latest) row blocks first
    const int seq0 = *dseq0;                             // query i sees keys < seq0 + i
    const int q_lo = r0 / g, q_hi = min(r0 + ROWS - 1, rows - 1) / g;
    const int key_end = seq0 + q_hi, ntile = (key_end + TILE - 1) / TILE, mask_from = seq0 + q_lo;

    auto stage = [&](int buf, int t0) {
        for (int i = t; i < TILE * (DIM / 8); i += NW * 32) {
            const int p = i / (DIM / 8), c = i % (DIM / 8);
            __nv_bfloat16 * dk = ks + (buf * TILE + p) * LS + c * 8, * dv = vs + (buf * TILE + p) * LS + c * 8;
            if (t0 + p < key_end) {
                const int64_t src = (int64_t(t0 + p) * kvheads + kh) * DIM + c * 8;
                cp_async16(dk, kc + src);
                cp_async16(dv, vc + src);
            } else {  // zero tail: 0 * stale NaN in the MMA
                *reinterpret_cast<uint4 *>(dk) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4 *>(dv) = make_uint4(0, 0, 0, 0);
            }
        }
        cp_commit();
    };
#if defined(TQLLM_WITH_HIP)
    if (ntile > 0) stage(0, 0);
#else
    if (ntile > 0) stage(0, 0);
    if (ntile > 1) stage(1, TILE);
#endif
    for (int i = t; i < ROWS * (DIM / 8); i += NW * 32) {  // rows past the end repeat the last row
        const int r = i / (DIM / 8), c = i % (DIM / 8), rr = min(r0 + r, rows - 1);
        *reinterpret_cast<uint4 *>(qs + r * LS + c * 8) =
            *reinterpret_cast<const uint4 *>(q + (int64_t(rr / g) * heads + kh * g + rr % g) * DIM + c * 8);
    }

    const int ra = warp * 16 + gid, rb = ra + 8;
    const int lim_a = seq0 + min(r0 + ra, rows - 1) / g, lim_b = seq0 + min(r0 + rb, rows - 1) / g;  // key < lim visible
    const float sl2 = scale * 1.4426950408889634f;
    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_a = -CUDART_INF_F, m_b = -CUDART_INF_F, l_a = 0.f, l_b = 0.f;
    const __nv_bfloat16 * qrow = qs + (warp * 16 + (lane % 8) + ((lane / 8) & 1) * 8) * LS + (lane / 16) * 8;

    for (int u = 0; u < ntile; ++u) {
#if defined(TQLLM_WITH_HIP)
        const int buf = 0, t0 = u * TILE;
        __syncthreads();  // the staged tile is ready and the previous tile's reads are done
#else
        const int buf = u & 1, t0 = u * TILE;
        if (u + 1 < ntile) cp_wait<1>();
        else cp_wait<0>();
        __syncthreads();
#endif
        const __nv_bfloat16 * kt = ks + buf * TILE * LS;
        const __nv_bfloat16 * vt = vs + buf * TILE * LS;
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < DIM / 16; kk += 2) {
            uint32_t a0[4], a1[4];
            ldsm_x4(a0, qrow + kk * 16);
            ldsm_x4(a1, qrow + (kk + 1) * 16);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                uint32_t kb[4];
                ldsm_x4(kb, kt + (nt * 8 + lane % 8) * LS + kk * 16 + (lane / 8) * 8);
                mma_bf16(s[nt], a0[0], a0[1], a0[2], a0[3], kb[0], kb[1]);
                mma_bf16(s[nt], a1[0], a1[1], a1[2], a1[3], kb[2], kb[3]);
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
            pf[nt / 2][(nt & 1) * 2] = pack_bf16(p0, p1);
            pf[nt / 2][(nt & 1) * 2 + 1] = pack_bf16(p2, p3);
        }
        l_a = l_a * al + sa, l_b = l_b * bl + sb;
        if (__any_sync(0xffffffffu, al != 1.f || bl != 1.f))
#pragma unroll
            for (int j = 0; j < 32; ++j) o[j][0] *= al, o[j][1] *= al, o[j][2] *= bl, o[j][3] *= bl;
        const __nv_bfloat16 * vrow = vt + ((lane / 8) & 1) * 8 * LS + (lane % 8) * LS + (lane / 16) * 8;
#pragma unroll
        for (int np = 0; np < 16; ++np)
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                uint32_t vb[4];
                ldsm_x4_t(vb, vrow + kk * 16 * LS + np * 16);
                mma_bf16(o[2 * np], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vb[0], vb[1]);
                mma_bf16(o[2 * np + 1], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vb[2], vb[3]);
            }
#if defined(TQLLM_WITH_HIP)
        if (u + 1 < ntile) {  // single buffer: refill only after every warp has finished reading this tile
            __syncthreads();
            stage(0, t0 + TILE);
        }
#else
        __syncthreads();  // buffer buf is refilled below
        if (u + 2 < ntile) stage(buf, t0 + 2 * TILE);
#endif
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

// In-block part of two-part verify attention: query z (a tree node, column z) over its ancestors root .. z in depth
// order (keys at cache slots pos + column), FP32 online softmax, merged with the nsplit cache partials (stride pstride;
// the attn_gen_merge_k arithmetic, block partial last) into y.
// par == chain gives a linear block; a node's result depends only on its path, never on the tree's other nodes.
template <int DIM>
__global__ void __launch_bounds__(DIM) attn_blk_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kc,
                                                  const __nv_bfloat16 * vc, float * part, const int * dseq0,
                                                  const int * par, const int * dep, int heads, int kvheads, float scale,
                                                  int nsplit, int pstride, __nv_bfloat16 * y) {
    constexpr int MAXK = 64;  // qpos - split + 1 <= TREE_MAX_DEPTH + 32
    __shared__ float sc[MAXK];
    __shared__ int slot[MAXK];
    const int h = blockIdx.x, z = blockIdx.y, d = threadIdx.x, lane = d % 32, wid = d / 32;
    const int pos = *dseq0 - 1, kh = h / (heads / kvheads);
    int path[16], len = 0;
    for (int a = z; a >= 0; a = par[a]) path[len++] = a;  // z .. root; depth e is path[len - 1 - e]
    const int qpos = pos + len - 1, from = tq_split_point(qpos), nk = qpos - from + 1;
    if (d < nk) {  // canonical positions: committed cache, then this node's ancestors
        const int p = from + d;
        slot[d] = p < pos ? p : pos + path[len - 1 - (p - pos)];
    }
    __syncthreads();
    // scores: one warp per key, 8 dims per lane, fixed-order warp reduction
    const uint4 qv = *reinterpret_cast<const uint4 *>(q + (int64_t(z) * heads + h) * DIM + lane * 8);
    const auto * qh = reinterpret_cast<const __nv_bfloat162 *>(&qv);
    for (int i = wid; i < nk; i += DIM / 32) {
        const uint4 kv = *reinterpret_cast<const uint4 *>(kc + (int64_t(slot[i]) * kvheads + kh) * DIM + lane * 8);
        const auto * kh2 = reinterpret_cast<const __nv_bfloat162 *>(&kv);
        float v = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 a = __bfloat1622float2(qh[e]), b = __bfloat1622float2(kh2[e]);
            v += a.x * b.x;
            v += a.y * b.y;
        }
#pragma unroll
        for (int w = 16; w; w >>= 1) v += __shfl_xor_sync(0xffffffffu, v, w);
        if (lane == 0) sc[i] = v * scale;
    }
    __syncthreads();
    // softmax weights and the value sum in position order
    float m = -CUDART_INF_F;
    for (int i = 0; i < nk; ++i) m = fmaxf(m, sc[i]);
    float l = 0.f, o = 0.f;
    for (int i = 0; i < nk; ++i) {
        const float pr = __expf(sc[i] - m);
        l += pr;
        o += pr * __bfloat162float(vc[(int64_t(slot[i]) * kvheads + kh) * DIM + d]);
    }
    (void) dep;
    const float * base = part + (int64_t(z) * heads + h) * pstride * (DIM + 2);
    float mm = m;
    for (int s2 = 0; s2 < nsplit; ++s2) mm = fmaxf(mm, base[s2 * (DIM + 2) + DIM]);
    float zz = 0.f, acc = 0.f;
    for (int s2 = 0; s2 < nsplit; ++s2) {
        const float * pp = base + s2 * (DIM + 2);
        if (!(pp[DIM + 1] > 0.f)) continue;
        const float w = expf(pp[DIM] - mm);
        zz += pp[DIM + 1] * w;
        acc += pp[d] * w;
    }
    if (l > 0.f) {
        const float w = expf(m - mm);
        zz += l * w;
        acc += o * w;
    }
    y[(int64_t(z) * heads + h) * DIM + d] = __float2bfloat16_rn(acc / zz);
}

template <bool Q8, bool TURBO3V>
void mq_launch(const __nv_bfloat16 * q, const void * kc, const void * vc, const __half * ksc, const __half * vsc,
               __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit,
               cudaStream_t s, int query_offset, const int * par = nullptr, const int * dep = nullptr) {
    if (h <= 0 || kh <= 0 || h % kh || h / kh > MQ_MAXROWS || nq <= 0)
        throw std::invalid_argument("attention_mq256: invalid head or query count");
    constexpr int DIM = 256, KS = DIM + 8;
#if defined(TQLLM_WITH_HIP)
    // one BF16 K and one BF16 V tile (no double buffering; Q8/Q8V records are decoded straight from global)
    const size_t smem = (size_t(MQ_MAXROWS) + 2 * size_t(MQ_TILE)) * KS * 2 +
                        size_t(MQ_MAXROWS) * (MQ_TILE + 4) * 4;
#else
    constexpr bool Q8V = Q8 && !TURBO3V;
    constexpr size_t V3B = (DIM / 128) * 50;
    const size_t smem = (size_t(MQ_MAXROWS) + (Q8 ? 2 : 4) * MQ_TILE) * KS * 2 +
                        (Q8 ? (Q8V ? 4 : 2) * size_t(MQ_TILE) * DIM : 0) +             // int8 K (and V) stages
                        (Q8 ? (Q8V ? 4 : 2) * size_t(MQ_TILE) * (DIM / 32) * 2 : 0) +  // their scales
                        (TURBO3V ? (2 * MQ_TILE * V3B + 15) / 16 * 16 + 8 * 4 : 0) + // Turbo3 V stages + LUT
                        size_t(MQ_MAXROWS) * (MQ_TILE + 4) * 4;
#endif
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_mq_k<DIM, Q8, TURBO3V>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
        configured = true;
    }
    if (par) {  // two-part verify attention: cache keys here, the block's ancestors in attn_blk_k, merged in fixed order
        if (Q8 || TURBO3V || query_offset) throw std::invalid_argument("tree attention: BF16 K/V, one verify block");
        const int qmax = MQ_MAXROWS / (h / kh);  // queries per tiled launch; each query's rows are independent
        for (int z0 = 0; z0 < nq; z0 += qmax) {
            const int nz = std::min(qmax, nq - z0);
#if defined(TQLLM_WITH_HIP)
            const int qbase = z0;  // dep == null is a chain: carry the chunk offset instead of node depths
#else
            if (dep == nullptr && z0) throw std::invalid_argument("tree attention: > 8 queries need node depths");
#endif
            attn_mq_k<DIM, Q8, TURBO3V><<<dim3(kh, nsplit), DIM, smem, s>>>(
                q + int64_t(z0) * h * DIM, kc, vc, ksc, vsc, part + int64_t(z0) * h * (nsplit + 1) * (DIM + 2), dseq0, nz, h,
                kh, scale, nsplit, 0, nsplit + 1, 1, dep ? dep + z0 : nullptr
#if defined(TQLLM_WITH_HIP)
                , qbase
#endif
                );
        }
        attn_blk_k<DIM><<<dim3(h, nq), DIM, 0, s>>>(q, static_cast<const __nv_bfloat16 *>(kc),
                                                    static_cast<const __nv_bfloat16 *>(vc), part, dseq0, par, dep, h, kh,
                                                    scale, nsplit, nsplit + 1, y);
        return;
    }
    const int qmax = MQ_MAXROWS / (h / kh);
    for (int z0 = 0; z0 < nq; z0 += qmax) {
        const int nz = std::min(qmax, nq - z0);
        const int64_t offset = int64_t(z0) * h * DIM;
        attn_mq_k<DIM, Q8, TURBO3V><<<dim3(kh, nsplit), DIM, smem, s>>>(
            q + offset, kc, vc, ksc, vsc, part, dseq0, nz, h, kh, scale, nsplit, query_offset + z0, nsplit, 0);
        attn_gen_merge_k<DIM><<<dim3(h, nz), DIM, 0, s>>>(part, y + offset, nsplit, h);
    }
}

void attention_mq256(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                     float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit, cudaStream_t s,
                     int query_offset, const int * par, const int * dep) {
    mq_launch<false, false>(q, kc, vc, nullptr, nullptr, y, part, dseq0, nq, h, kh, scale, nsplit, s, query_offset, par, dep);
}

void attention_mq256_prefill(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc,
                             __nv_bfloat16 * y, float * part, size_t part_floats, const int * dseq0, int nq, int h, int kh,
                             float scale, int nsplit, cudaStream_t s) {
    constexpr int DIM = 256;
    static bool configured = false;
    if (!configured) {
        if (cudaFuncSetAttribute(attn_bf16_prefill_k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(fab::smem())) !=
            cudaSuccess)
            throw std::runtime_error("attention_mq256_prefill: shared memory per block unavailable");
        configured = true;
    }
    const int blocks = ((h / kh) * nq + fab::ROWS - 1) / fab::ROWS;
    attn_bf16_prefill_k<<<dim3(kh, blocks), fab::NW * 32, fab::smem(), s>>>(q, kc, vc, y, dseq0, nq, h, kh, scale);
}

void attention_mq256_q8(const __nv_bfloat16 * q, const int8_t * kc, const int8_t * vc, const __half * ksc,
                        const __half * vsc, __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh,
                        float scale, int nsplit, cudaStream_t s, int query_offset) {
    mq_launch<true, false>(q, kc, vc, ksc, vsc, y, part, dseq0, nq, h, kh, scale, nsplit, s, query_offset);
}

template <bool T4K, bool T4V = false>
static void ctv_launch(const __nv_bfloat16 * q, const void * kc, const void * vc, const __half * ksc,
                       __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit,
                       cudaStream_t s, int query_offset) {
    if (h % kh || (h / kh) * nq > ctv::MAXR)
        throw std::invalid_argument("CTV attention: nq * group must be <= 48");
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_ctv_k<T4K, T4V>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(ctv::smem(T4K, T4V)));
        configured = true;
    }
    attn_ctv_k<T4K, T4V><<<dim3(kh, nsplit), 256, ctv::smem(T4K, T4V), s>>>(q, kc, static_cast<const uint8_t *>(vc), ksc, part,
                                                                  dseq0, nq, h, kh, scale, nsplit, query_offset);
    attn_gen_merge_k<256><<<dim3(h, nq), 256, 0, s>>>(part, y, nsplit, h);
    turbo3_inverse_multi_k<<<nq * h * (256 / 128), 128, 0, s>>>(y, h, 256);  // V was accumulated in the rotated basis
}

void attention_mq256_q8_turbo3v(const __nv_bfloat16 * q, const int8_t * kc, const Turbo3V256 * vc,
                               const __half * ksc, __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h,
                               int kh, float scale, int nsplit, cudaStream_t s, int query_offset) {
    ctv_launch<false>(q, kc, vc, ksc, y, part, dseq0, nq, h, kh, scale, nsplit, s, query_offset);
}

void attention_mq256_t4k_turbo3v(const __nv_bfloat16 * q, const Turbo4K256 * kc, const Turbo3V256 * vc,
                                 __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh, float scale,
                                 int nsplit, cudaStream_t s, int query_offset) {
    ctv_launch<true>(q, kc, vc, nullptr, y, part, dseq0, nq, h, kh, scale, nsplit, s, query_offset);
}

void attention_mq256_t4kv(const __nv_bfloat16 * q, const Turbo4K256 * kc, const Turbo4K256 * vc, __nv_bfloat16 * y,
                          float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit, cudaStream_t s,
                          int query_offset) {
    ctv_launch<true, true>(q, kc, vc, nullptr, y, part, dseq0, nq, h, kh, scale, nsplit, s, query_offset);
}

void attention_mq256_t4kv_prefill(const __nv_bfloat16 * q, const Turbo4K256 * kc, const Turbo4K256 * vc,
                                  __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh, float scale,
                                  int nsplit, cudaStream_t s) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_t4kv_prefill_k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(fap::smem()));
        configured = true;
    }
    const int blocks = ((h / kh) * nq + fap::ROWS - 1) / fap::ROWS;
    attn_t4kv_prefill_k<<<dim3(kh, blocks), fap::NW * 32, fap::smem(), s>>>(
        q, reinterpret_cast<const uint8_t *>(kc), reinterpret_cast<const uint8_t *>(vc), y, dseq0, nq, h, kh, scale);
    turbo3_inverse_multi_k<<<nq * h * (256 / 128), 128, 0, s>>>(y, h, 256);
}
