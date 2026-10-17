// TqCache compaction: an old block of tq3 KV rows -> m stand-in rows (Q8 keys in the original basis, Q8 values in the
// TQ3 rotated basis, one log-weight bias per row and KV head) fitted to the layer's recent queries so attention to the
// block keeps its output and its mass. Runs on the GPU while the server is idle (cuBLAS + cuSOLVER, FP32).
// Idea: Fast KV Compaction via Attention Matching (arXiv 2602.16284); this code is written from the published description.
#include "qwen35_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm_cuda {
namespace {
constexpr int DIM = 256;
__constant__ float c_signs[32] = {1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1,
                                  -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1};
__constant__ float c_lev[8] = {-127, -82, -47, -16, 15, 46, 81, 127};
constexpr float LEV_F = 127.f / 1.996684f;

__device__ inline float warp_wht32(float v, int lane) {  // unnormalised 32-point Walsh-Hadamard across a warp
    for (int m = 1; m < 32; m <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, m);
        v = (lane & m) ? o - v : v + o;
    }
    return v;
}

// tq3 records [n][kvh] -> FP32 keys in the original basis and values in the rotated basis, [n][kvh][256]
__global__ void dequant_tq3_block_k(const Tq3KV256 * kc, const Tq3KV256 * vc, int n, int kvh, float * kf, float * vf,
                                    int kcod = 0, int vcod = 0) {
    const int row = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;  // 8 warps = 8 blocks of 32
    if (row >= n * kvh) return;
    for (int side = 0; side < 2; ++side) {
        if (side ? vcod : kcod) continue;  // trellis side: dequant_tcq_block_k
        const Tq3KV256 & r = (side ? vc : kc)[row];
        const int bit = 3 * lane, b = warp;
        const uint64_t lo = uint64_t(r.w[3 * b]) | (uint64_t(r.w[3 * b + 1]) << 32);
        uint32_t idx;
        if (bit + 3 <= 64) idx = uint32_t(lo >> bit) & 7u;
        else if (bit >= 64) idx = (r.w[3 * b + 2] >> (bit - 64)) & 7u;
        else idx = uint32_t((lo >> bit) | (uint64_t(r.w[3 * b + 2]) << (64 - bit))) & 7u;
        const float rot = c_lev[idx] / LEV_F * __half2float(r.rms[b]);
        if (side) {
            vf[int64_t(row) * DIM + b * 32 + lane] = rot;
        } else {  // R^T: signs * WHT / sqrt(32)
            const float o = warp_wht32(rot, lane) * 0.17677669529663687f * c_signs[lane];
            kf[int64_t(row) * DIM + b * 32 + lane] = o;
        }
    }
}

// trellis records (tcq3 K / tcq2 V): two 128-value groups per row, value i of a group = codebook[bits 3i (2i) .. + 9 (8)
// of the group stream] * the group's fp16 norm, in the TCQ rotated basis (signs S1, 128-point WHT / sqrt(128), signs S2).
// Keys go back to the original basis; values stay rotated (the attention kernels rotate V back after the merge).
#include "tcq2_tables.inc"
#include "tcq3_tables.inc"
__device__ const float c_cb2v[256] = TQ_TCQ2_CODEBOOK_V;
__device__ const float c_cb3k[512] = TQ_TCQ3_CODEBOOK_K;
__constant__ float c_s1[128] = TQ_TCQ_SIGNS1;
__constant__ float c_s2[128] = TQ_TCQ_SIGNS2;

__device__ inline float tcq_value(const uint32_t * g, int words, int i, int kb, int lbits, const float * cb) {
    const int bit = kb * i, w0 = bit / 32;
    const uint64_t win = uint64_t(g[w0]) | (w0 + 1 < words ? uint64_t(g[w0 + 1]) << 32 : 0);
    const uint32_t st = uint32_t(win >> (bit % 32)) & ((1u << lbits) - 1);
    const unsigned short hs = static_cast<unsigned short>(g[words - 1] >> 16);
    return cb[st] * __half2float(*reinterpret_cast<const __half *>(&hs));
}

// one block of 256 threads per (row, kv head); side 0 = K, 1 = V; codec per side: 0 = tq3, 1 = trellis (tcq3 K / tcq2 V)
__global__ void dequant_tcq_block_k(const void * kc, const void * vc, int n, int kvh, int kcod, int vcod, float * kf, float * vf) {
    __shared__ float x[DIM];
    const int row = blockIdx.x, t = threadIdx.x, grp = t / 128, i = t % 128;
    if (row >= n * kvh) return;
    for (int side = 0; side < 2; ++side) {
        const bool tcq = side ? vcod == 1 : kcod == 1;
        if (!tcq) continue;  // tq3 sides are decoded by dequant_tq3_block_k
        const auto * r = static_cast<const uint32_t *>(side ? vc : kc) + int64_t(row) * (side ? 18 : 26);
        const float v = side ? tcq_value(r + 9 * grp, 9, i, 2, 8, c_cb2v) : tcq_value(r + 13 * grp, 13, i, 3, 9, c_cb3k);
        if (side) {
            vf[int64_t(row) * DIM + t] = v;
            continue;
        }
        x[t] = v * c_s2[i];  // R^T = S1 * WHT / sqrt(128) * S2, per 128-value group
        __syncthreads();
        for (int h = 1; h < 128; h <<= 1) {
            const float a = x[t], b = x[t ^ h];
            __syncthreads();
            x[t] = (i & h) ? b - a : a + b;
            __syncthreads();
        }
        kf[int64_t(row) * DIM + t] = x[t] * 0.08838834764831845f * c_s1[i];
        __syncthreads();
    }
}

// rows of q ring [R][heads][256] (bf16) for the heads of KV head kh -> FP32 [R * g][256]
// ring_rows > 0: a ring written at position % ring_rows; the r_used newest rows end before row ring_end
__global__ void gather_q_k(const __nv_bfloat16 * ring, int r_used, int heads, int g, int kh, float * qf, int ring_rows, int ring_end) {
    const int i = blockIdx.x, t = threadIdx.x;  // i over r_used * g
    const int hh = kh * g + i % g;
    const int r = ring_rows ? ((ring_end - r_used + i / g) % ring_rows + ring_rows) % ring_rows : i / g;
    qf[int64_t(i) * DIM + t] = __bfloat162float(ring[(int64_t(r) * heads + hh) * DIM + t]);
}

// L [nq][nb] (scaled logits) -> A = exp(L - rowmax) in place, rowsum b[nq], rowmax mrow[nq]
__global__ void rowexp_k(float * L, int nb, float * b, float * mrow) {
    __shared__ float red[32];
    float * row = L + int64_t(blockIdx.x) * nb;
    float m = -1e30f;
    for (int j = threadIdx.x; j < nb; j += blockDim.x) m = fmaxf(m, row[j]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -1e30f;
        for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();
    float s = 0.f;
    for (int j = threadIdx.x; j < nb; j += blockDim.x) {
        const float e = expf(row[j] - m);
        row[j] = e;
        s += e;
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.f;
        for (int w = 0; w < int(blockDim.x / 32); ++w) v += red[w];
        b[blockIdx.x] = v;
        mrow[blockIdx.x] = m;
    }
}

// mass[j] = sum_q A[q][j] / b[q]  (column sums of the softmax)

__global__ void gather_cols_k(const float * A, int nb, const int * sel, int m, float * As) {  // As [nq][m]
    const int q = blockIdx.x;
    for (int j = threadIdx.x; j < m; j += blockDim.x) As[int64_t(q) * m + j] = A[int64_t(q) * nb + sel[j]];
}

__global__ void nnls_step_k(float * w, const float * atb, const float * atw, int m) {  // w <- w * Atb / (AtA w)
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m) w[j] = w[j] * fmaxf(atb[j], 0.f) / fmaxf(atw[j], 1e-30f);
}

__global__ void weight_rows_k(float * As, const float * w, int m) {  // P = As * w, rows normalised
    __shared__ float red[32];
    float * row = As + int64_t(blockIdx.x) * m;
    float s = 0.f;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
        const float v = row[j] * w[j];
        row[j] = v;
        s += v;
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.f;
        for (int w2 = 0; w2 < int(blockDim.x / 32); ++w2) v += red[w2];
        red[0] = v > 0.f ? 1.f / v : 0.f;
    }
    __syncthreads();
    const float inv = red[0];
    for (int j = threadIdx.x; j < m; j += blockDim.x) row[j] *= inv;
}

__global__ void inv_k(const float * b, float * inv, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) inv[i] = 1.f / b[i];
}
__global__ void scale_o_k(float * O, const float * inv) { O[int64_t(blockIdx.x) * blockDim.x + threadIdx.x] *= inv[blockIdx.x]; }

// PtP += lam I; rhs (column-major [m][256], ld m, as cuSOLVER wants) += lam * V[sel] taken straight from the block
__global__ void add_ridge_k(float * PtP, float * rhs, const float * vf, int kvh, int kh, const int * sel, int m, float lam) {
    const int i = blockIdx.x, d = threadIdx.x;
    if (d == 0) PtP[int64_t(i) * m + i] += lam;
    rhs[int64_t(d) * m + i] += lam * vf[(int64_t(sel[i]) * kvh + kh) * DIM + d];
}

// top-m keys of a block on the GPU (no host round trip): score = mass, raised to the mass of the previous `succ` keys;
// order = score descending, then index ascending (deterministic ties); sel = the chosen indices ascending. One block of
// 1024 threads, bitonic sorts of 64-bit keys in shared memory (P = next power of two >= nb).
__global__ void __launch_bounds__(1024) topm_select_k(const float * mass, int nb, int m, int succ, int * sel) {
    extern __shared__ unsigned long long key[];
    int P = 1;
    while (P < nb) P <<= 1;
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        unsigned long long k = 0ull;  // padding sorts last
        if (i < nb) {
            float sc = mass[i];
            for (int d = 1; d <= succ && i - d >= 0; ++d) sc = fmaxf(sc, mass[i - d]);
            const unsigned int b = __float_as_uint(sc), ord = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
            k = (unsigned long long)(ord) << 32 | (0xFFFFFFFFu - unsigned(i));
        }
        key[i] = k;
    }
    __syncthreads();
    for (int k = 2; k <= P; k <<= 1)  // descending
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < P; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool desc = (i & k) == 0;
                    const unsigned long long a = key[i], c = key[ixj];
                    if (desc ? a < c : a > c) key[i] = c, key[ixj] = a;
                }
            }
            __syncthreads();
        }
    int Q = 1;
    while (Q < m) Q <<= 1;
    // the chosen indices into key[0, Q) as ascending values (padding INT_MAX last), then sort ascending
    for (int i = threadIdx.x; i < Q; i += blockDim.x)
        key[i] = i < m ? (unsigned long long)(0xFFFFFFFFu - unsigned(key[i] & 0xFFFFFFFFull)) : 0x7FFFFFFFull;
    __syncthreads();
    for (int k = 2; k <= Q; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < Q; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool asc = (i & k) == 0;
                    const unsigned long long a = key[i], c = key[ixj];
                    if (asc ? a > c : a < c) key[i] = c, key[ixj] = a;
                }
            }
            __syncthreads();
        }
    for (int i = threadIdx.x; i < m; i += blockDim.x) sel[i] = int(key[i]);
}

// rhs (column-major [m][256], ld m) = V[sel] taken from the block
__global__ void gather_vt_k(const float * vf, int kvh, int kh, const int * sel, int m, float * rhs) {
    rhs[int64_t(threadIdx.x) * m + blockIdx.x] = vf[(int64_t(sel[blockIdx.x]) * kvh + kh) * DIM + threadIdx.x];
}
__global__ void fill_k(float * x, int n, float v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}
__global__ void add_diag_k(float * G, int n, float lam) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) G[int64_t(i) * n + i] += lam;
}

// one stand-in row (FP32, 256) -> Q8 record (int8 + fp16 scale per 32)
__device__ void q8_store(const float * x, KvStandIn256 & r, int warp, int lane) {
    const float v = x[warp * 32 + lane];
    float a = fabsf(v);
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float s = a > 0.f ? a / 127.f : 1.f;
    r.q[warp * 32 + lane] = int8_t(__float2int_rn(v / s));
    if (lane == 0) r.s[warp] = __float2half_rn(s);
}

// writes stand-in rows [row0, row0 + m) for KV head kh: keys = selected original-basis keys, values = refit C (rotated),
// bias = log w
__global__ void store_standin_k(const float * kf, int kvh, int kh, const int * sel, const float * C, const float * w,
                                int m, int row0, KvStandIn256 * ck, KvStandIn256 * cv, float * cb) {
    const int i = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (i >= m) return;
    const int64_t dst = int64_t(row0 + i) * kvh + kh;
    // key rows in the TQ3_0 basis (signs, WHT / sqrt(32) per 32): the main attention kernel reads stand-ins as Q8 rows
    __shared__ float kr[DIM];
    kr[warp * 32 + lane] = warp_wht32(kf[(int64_t(sel[i]) * kvh + kh) * DIM + warp * 32 + lane] * c_signs[lane], lane) * 0.17677669529663687f;
    __syncwarp();
    q8_store(kr, ck[dst], warp, lane);
    q8_store(C + int64_t(i) * DIM, cv[dst], warp, lane);
    if (threadIdx.x == 0) cb[dst] = logf(fmaxf(w[i], 1e-30f));
}

void check(cublasStatus_t s, const char * what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("kv compaction: cuBLAS ") + what);
}
void check(cusolverStatus_t s, const char * what) {
    if (s != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string("kv compaction: cuSOLVER ") + what);
}
}  // namespace

__global__ void q_ring_write_k(const __nv_bfloat16 * q, int n, int width, const int * dpos0, int ring_rows, __nv_bfloat16 * ring) {
    const int i = blockIdx.x, pos = *dpos0 + i;
    __nv_bfloat16 * dst = ring + int64_t(pos % ring_rows) * width;
    for (int d = threadIdx.x; d < width; d += blockDim.x) dst[d] = q[int64_t(i) * width + d];
}

void q_ring_write(const __nv_bfloat16 * q, int n, int width, const int * dpos0, int ring_rows, __nv_bfloat16 * ring,
                  cudaStream_t s) {
    if (n > 0) q_ring_write_k<<<n, 256, 0, s>>>(q, n, width, dpos0, ring_rows, ring);
}

void kv_topm_select(const float * mass, int nb, int m, int succ, int * sel, cudaStream_t s) {
    int P = 1;
    while (P < nb) P <<= 1;
    const int bytes = P * 8;
    static int configured = 0;
    if (bytes > configured) {
        cudaFuncSetAttribute(topm_select_k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        configured = bytes;
    }
    topm_select_k<<<1, 1024, bytes, s>>>(mass, nb, m, succ, sel);
}

size_t kv_compact_scratch_bytes(int nb, int nq_rows, int m_max, int kvh) {
    const size_t f = sizeof(float);
    return f * (size_t(nb) * kvh * DIM * 2           // kf, vf
                + size_t(nq_rows) * DIM               // qf
                + size_t(nq_rows) * nb                // L / A
                + size_t(nq_rows) * 2 + nb            // b, mrow, mass
                + size_t(nq_rows) * m_max             // As / P
                + size_t(m_max) * m_max               // AtA / PtP
                + size_t(m_max) * 4                   // w, atb, atw, spare
                + size_t(nq_rows) * DIM               // O
                + size_t(m_max) * DIM * 2)            // rhs, vsel
           + sizeof(int) * (m_max + 1);
}

void kv_compact_block(const void * hk, const void * hv, KvCodec kcodec, KvCodec vcodec, int nb, int kvh, int heads, const __nv_bfloat16 * ring,
                      int r_used, float scale, int m, int row0, KvStandIn256 * ck, KvStandIn256 * cv, float * cb,
                      void * scratch, cublasHandle_t blas, cusolverDnHandle_t sol, cudaStream_t s, int iters, float ridge,
                      int succ, int ring_rows, int ring_end, int solve) {
    const int g = heads / kvh, nq = r_used * g;
    if (m <= 0 || m > nb || nq <= 0) throw std::invalid_argument("kv_compact_block: bad sizes");
    char * p = static_cast<char *>(scratch);
    auto take = [&](size_t n) { float * r = reinterpret_cast<float *>(p); p += n * sizeof(float); return r; };
    float * kf = take(size_t(nb) * kvh * DIM), * vf = take(size_t(nb) * kvh * DIM), * qf = take(size_t(nq) * DIM);
    float * A = take(size_t(nq) * nb), * b = take(nq), * mrow = take(nq), * mass = take(nb);
    float * As = take(size_t(nq) * m), * AtA = take(size_t(m) * m), * w = take(m), * atb = take(m), * atw = take(m);
    take(m);
    float * O = take(size_t(nq) * DIM), * rhs = take(size_t(m) * DIM), * C = take(size_t(m) * DIM);
    int * dsel = reinterpret_cast<int *>(p);
    int * dinfo = dsel + m;
    check(cublasSetStream(blas, s), "stream");
    check(cusolverDnSetStream(sol, s), "stream");
    if ((kcodec != KvCodec::Tq3 && kcodec != KvCodec::Tcq3) || (vcodec != KvCodec::Tq3 && vcodec != KvCodec::Tcq2))
        throw std::invalid_argument("kv_compact_block: K must be tq3 or tcq3, V tq3 or tcq2");
    const int kcod = kcodec == KvCodec::Tcq3, vcod = vcodec == KvCodec::Tcq2;
    if (!kcod || !vcod)
        dequant_tq3_block_k<<<nb * kvh, 256, 0, s>>>(static_cast<const Tq3KV256 *>(hk), static_cast<const Tq3KV256 *>(hv), nb, kvh, kf, vf, kcod, vcod);
    if (kcod || vcod) dequant_tcq_block_k<<<nb * kvh, DIM, 0, s>>>(hk, hv, nb, kvh, kcod, vcod, kf, vf);
    const float one = 1.f, zero = 0.f;
    // Cholesky workspace: grown once per process (a per-call cudaMallocAsync cost ~1.7 ms of host time per KV head)
    static float * work = nullptr;
    static int work_n = 0;
    // values: (PtP + lam I) C = PtO + lam V[sel] (m x m), or with fewer query rows than stand-in rows the same C from the
    // nq x nq system: C = V[sel] + Pt (PPt + lam I)^-1 (O - P V[sel])  (push-through identity)
    const bool dual = solve == 2 || (solve == 0 && nq < m);
    int lwork = 0;
    check(cusolverDnSpotrf_bufferSize(sol, CUBLAS_FILL_MODE_LOWER, dual ? nq : m, AtA, dual ? nq : m, &lwork), "potrf size");
    if (lwork > work_n) {
        cudaStreamSynchronize(s);
        cudaFree(work);
        if (cudaMalloc(&work, size_t(lwork) * 4) != cudaSuccess) throw std::runtime_error("kv compaction: workspace allocation");
        work_n = lwork;
    }
    // TF32 tensor cores for the GEMMs (the handle is the engine's: its mode is restored on return)
    cublasMath_t math_prev;
    check(cublasGetMathMode(blas, &math_prev), "math mode");
    check(cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH), "math mode");
    struct Restore { cublasHandle_t h; cublasMath_t m; ~Restore() { cublasSetMathMode(h, m); } } restore{blas, math_prev};
    for (int kh = 0; kh < kvh; ++kh) {
        gather_q_k<<<nq, DIM, 0, s>>>(ring, r_used, heads, g, kh, qf, ring_rows, ring_end);
        // L[q][j] = scale * q . k_j  (row-major [nq][nb] = column-major [nb][nq]); k rows strided by kvh * 256
        check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nb, nq, DIM, &scale, kf + size_t(kh) * DIM, kvh * DIM, qf, DIM,
                          &zero, A, nb), "logits");
        rowexp_k<<<nq, 256, 0, s>>>(A, nb, b, mrow);
        inv_k<<<(nq + 255) / 256, 256, 0, s>>>(b, mrow, nq);  // mrow now holds 1 / b
        // mass[j] = sum_q A[q][j] / b[q]: A row-major [nq][nb] = column-major nb x nq
        check(cublasSgemv(blas, CUBLAS_OP_N, nb, nq, &one, A, nb, mrow, 1, &zero, mass, 1), "mass");
        kv_topm_select(mass, nb, m, succ, dsel, s);  // on the GPU: the block's work queues without host round trips
        gather_cols_k<<<nq, 256, 0, s>>>(A, nb, dsel, m, As);
        // mass-preserving weights: AtA w ~ At b, w >= 0 (multiplicative non-negative iterations from w = 1)
        // (with fewer query rows than stand-in rows, AtA w = At (A w): two thin gemvs instead of an m x m one and no AtA)
        if (!dual) check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, m, m, nq, &one, As, m, As, m, &zero, AtA, m), "AtA");
        check(cublasSgemv(blas, CUBLAS_OP_N, m, nq, &one, As, m, b, 1, &zero, atb, 1), "Atb");
        fill_k<<<(m + 255) / 256, 256, 0, s>>>(w, m, 1.f);
        for (int it = 0; it < iters; ++it) {
            if (dual) {
                check(cublasSgemv(blas, CUBLAS_OP_T, m, nq, &one, As, m, w, 1, &zero, mass, 1), "Aw");  // mass: free after selection
                check(cublasSgemv(blas, CUBLAS_OP_N, m, nq, &one, As, m, mass, 1, &zero, atw, 1), "AtAw");
            } else {
                check(cublasSgemv(blas, CUBLAS_OP_N, m, m, &one, AtA, m, w, 1, &zero, atw, 1), "AtAw");
            }
            nnls_step_k<<<(m + 255) / 256, 256, 0, s>>>(w, atb, atw, m);
        }
        // values: P = normalised (As * w); O = softmax(L) V; solve (PtP + ridge I) C = PtO + ridge Vsel
        weight_rows_k<<<nq, 256, 0, s>>>(As, w, m);
        check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, DIM, nq, nb, &one, vf + size_t(kh) * DIM, kvh * DIM, A, nb, &zero,
                          O, DIM), "O");
        scale_o_k<<<nq, DIM, 0, s>>>(O, mrow);  // softmax rows: O[q] /= b[q]
        if (!dual) {
            check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, m, m, nq, &one, As, m, As, m, &zero, AtA, m), "PtP");
            // rhs = P^T O as column-major [m][256] (ld m): As is P^T column-major (m x nq), O is O^T column-major (256 x nq)
            check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, m, DIM, nq, &one, As, m, O, DIM, &zero, rhs, m), "PtO");
            add_ridge_k<<<m, DIM, 0, s>>>(AtA, rhs, vf, kvh, kh, dsel, m, ridge);
            check(cusolverDnSpotrf(sol, CUBLAS_FILL_MODE_LOWER, m, AtA, m, work, lwork, dinfo), "potrf");
            check(cusolverDnSpotrs(sol, CUBLAS_FILL_MODE_LOWER, m, DIM, AtA, m, rhs, m, dinfo), "potrs");
        } else {  // rhs <- V[sel]; X (in C, column-major [nq][256]) <- O - P V[sel]; G (in AtA, nq x nq) <- PPt + lam I
            const float minus1 = -1.f;
            gather_vt_k<<<m, DIM, 0, s>>>(vf, kvh, kh, dsel, m, rhs);
            check(cublasSgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, nq, DIM, &one, O, DIM, &zero, C, nq, C, nq), "O^T");
            check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nq, DIM, m, &minus1, As, m, rhs, m, &one, C, nq), "O - PV");
            check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nq, nq, m, &one, As, m, As, m, &zero, AtA, nq), "PPt");
            add_diag_k<<<(nq + 255) / 256, 256, 0, s>>>(AtA, nq, ridge);
            check(cusolverDnSpotrf(sol, CUBLAS_FILL_MODE_LOWER, nq, AtA, nq, work, lwork, dinfo), "potrf");
            check(cusolverDnSpotrs(sol, CUBLAS_FILL_MODE_LOWER, nq, DIM, AtA, nq, C, nq, dinfo), "potrs");
            check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, DIM, nq, &one, As, m, C, nq, &one, rhs, m), "V + PtX");
        }
        // rhs holds C as column-major [m][256]: element (i, d) at rhs[d * m + i]; transpose into rows
        check(cublasSgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, DIM, m, &one, rhs, m, &zero, rhs, DIM, C, DIM), "transpose");
        store_standin_k<<<m, 256, 0, s>>>(kf, kvh, kh, dsel, C, w, m, row0, ck, cv, cb);
    }
}

}  // namespace tqllm_cuda
