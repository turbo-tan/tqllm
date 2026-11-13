// qsa: QSA rows of the two-part verify attention (BF16 cache). Included by attention.cu inside namespace tqllm_cuda
// (kernels and launchers; no includes here).
namespace qsa27 {
constexpr int DIM = 256, PICK = 512, BLK = 4;
}

// Share of a row's dense attention (softmax over keys 0 .. qpos, row 0 of the block) that falls on its selected keys,
// on the other keys below the split point and on the dense tail: out [heads][3]. A check of the indexer wiring, not a
// decode path.
__global__ void __launch_bounds__(256) attn_qsa_mass_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const int32_t * sel,
                                                       const int * dseq0, int heads, int kvheads, float scale, float * out) {
    using namespace qsa27;
    constexpr int WORDS = 2048;  // 65,536 blocks
    __shared__ unsigned bitmap[WORDS];
    __shared__ float red[256], sums[3];
    const int h = blockIdx.x, t = threadIdx.x, kh = h / (heads / kvheads);
    const int qpos = *dseq0 - 1, split = tq_split_point(qpos);
    for (int i = t; i < WORDS; i += 256) bitmap[i] = 0u;
    if (t < 3) sums[t] = 0.f;
    __syncthreads();
    for (int i = t; i < PICK; i += 256)
        if (sel[i] >= 0 && sel[i] < WORDS * 32) atomicOr(&bitmap[sel[i] >> 5], 1u << (sel[i] & 31));
    __syncthreads();
    auto score = [&](int key) {
        const __nv_bfloat16 * qr = q + int64_t(h) * DIM, * kr = kc + (int64_t(key) * kvheads + kh) * DIM;
        float s = 0.f;
        for (int d = 0; d < DIM; ++d) s = fmaf(__bfloat162float(qr[d]), __bfloat162float(kr[d]), s);
        return s * scale;
    };
    float m = -CUDART_INF_F;
    for (int key = t; key <= qpos; key += 256) m = fmaxf(m, score(key));
    red[t] = m;
    __syncthreads();
    for (int o = 128; o; o >>= 1) {
        if (t < o) red[t] = fmaxf(red[t], red[t + o]);
        __syncthreads();
    }
    m = red[0];
    float a[3] = {0.f, 0.f, 0.f};
    for (int key = t; key <= qpos; key += 256) {
        const float e = __expf(score(key) - m);
        const int b = key / BLK;
        a[key >= split ? 2 : (bitmap[b >> 5] >> (b & 31)) & 1u ? 0 : 1] += e;
    }
    for (int i = 0; i < 3; ++i) atomicAdd(&sums[i], a[i]);
    __syncthreads();
    if (t < 3) out[h * 3 + t] = sums[t];
}

void attention_mq256_qsa(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                         float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit, cudaStream_t s,
                         const int * par, const int * dep, const int32_t * sel, int from, bool mixed) {
    using namespace qsa27;
    if (!par || !sel || h <= 0 || kh <= 0 || h % kh || h / kh > 16 || nq <= 0 || nsplit < 1 || (nq > 8 && !dep))
        throw std::invalid_argument("QSA attention: two-part verify block, <= 16 heads per KV head");
    constexpr int KS = DIM + 8;
    const size_t smem = (size_t(MQ_MAXROWS) + 4 * MQ_TILE) * KS * 2 + size_t(MQ_MAXROWS) * (MQ_TILE + 4) * 4;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_mq_k<DIM, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
        configured = true;
    }
    // a block that crosses `from`: the dense cache part first, then the sparse rows' partials replace theirs
    if (mixed) mq_launch<false, false>(q, kc, vc, nullptr, nullptr, y, part, dseq0, nq, h, kh, scale, nsplit, s, 0, par, dep);
    // the cache-part kernel, one row per grid z over that row's selected keys (every split of a sparse row is written)
    attn_mq_k<DIM, false, false, true><<<dim3(kh, nsplit, nq), DIM, smem, s>>>(q, kc, vc, nullptr, nullptr, part, dseq0, nq, h, kh,
                                                                              scale, nsplit, 0, nsplit + 1, 1, dep, sel, from);
    attn_blk_k<DIM><<<dim3(h, nq), DIM, 0, s>>>(q, kc, vc, part, dseq0, par, dep, h, kh, scale, nsplit, nsplit + 1, y);
}

void attention_qsa_mass(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const int32_t * sel, const int * dseq0, int h,
                        int kh, float scale, float * out, cudaStream_t s) {
    attn_qsa_mass_k<<<h, 256, 0, s>>>(q, kc, sel, dseq0, h, kh, scale, out);
}

// QSA rows of a chain verify block on the turbo KV formats (attn_ctv_k; one-part attention). kind: 0 int8 K + Turbo3 V
// (ksc: the K scales), 1 Turbo4 K + Turbo3 V, 2 Turbo4 K + Turbo4 V. mixed: the block crosses `from`: the dense launches
// run first for every row, then the sparse rows are replaced.
template <bool T4K, bool T4V>
static void ctv_qsa_launch(const __nv_bfloat16 * q, const void * kc, const void * vc, const __half * ksc, __nv_bfloat16 * y,
                           float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit, cudaStream_t s,
                           const int32_t * sel, int from, bool mixed) {
    const int g = h / kh, chunk = std::max(1, std::min(8, ctv::MAXR / g));
    if (h % kh || g > ctv::MAXR || nq <= 0 || !sel) throw std::invalid_argument("QSA attention (turbo KV): invalid shape");
    for (int off = 0; mixed && off < nq; off += chunk)
        ctv_launch<T4K, T4V>(q + int64_t(off) * h * 256, kc, vc, ksc, y + int64_t(off) * h * 256, part, dseq0, std::min(chunk, nq - off),
                             h, kh, scale, nsplit, s, off);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_ctv_k<T4K, T4V, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(ctv::smem(T4K, T4V)));
        configured = true;
    }
    attn_ctv_k<T4K, T4V, true><<<dim3(kh, nsplit, nq), 256, ctv::smem(T4K, T4V), s>>>(q, kc, static_cast<const uint8_t *>(vc), ksc, part,
                                                                                    dseq0, nq, h, kh, scale, nsplit, 0, sel, from);
    attn_gen_merge_k<256><<<dim3(h, nq), 256, 0, s>>>(part, y, nsplit, h, dseq0, from);
    turbo3_inverse_multi_k<<<nq * h * (256 / 128), 128, 0, s>>>(y, h, 256, dseq0, from);  // V was accumulated in the rotated basis
}

void attention_ctv_qsa(int kind, const __nv_bfloat16 * q, const void * kc, const void * vc, const __half * ksc, __nv_bfloat16 * y,
                       float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit, cudaStream_t s,
                       const int32_t * sel, int from, bool mixed) {
    if (kind == 0) ctv_qsa_launch<false, false>(q, kc, vc, ksc, y, part, dseq0, nq, h, kh, scale, nsplit, s, sel, from, mixed);
    else if (kind == 1) ctv_qsa_launch<true, false>(q, kc, vc, nullptr, y, part, dseq0, nq, h, kh, scale, nsplit, s, sel, from, mixed);
    else ctv_qsa_launch<true, true>(q, kc, vc, nullptr, y, part, dseq0, nq, h, kh, scale, nsplit, s, sel, from, mixed);
}
