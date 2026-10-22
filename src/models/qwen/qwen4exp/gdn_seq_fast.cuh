// GDN sequential state pass for the WY chunked prefill at C = 32, dim = 128 (Flash-Next), written by Omega (the Flash-Next
// self-improvement loop, rung 3, 2026-10-15, candidate 47): 4.15x faster than fn_gdn_seq_k on GB10, same FP32 recurrence
// (max rel err 1.3e-7 vs the reference). Interface as fn_gdn_seq_k; dim != 128 falls back to its generic kernel.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace gdn_seq_fast_ns {

template <int C, int VS, int DIM>
__global__ void __launch_bounds__(512) cand_gdn_seq_fast(
    const float* __restrict__ qk,
    const float* __restrict__ U,
    const float* __restrict__ W,
    const float* __restrict__ Mg,
    const float* __restrict__ lgi,
    const float* __restrict__ state,
    float* __restrict__ next,
    float* __restrict__ out,
    int kh,
    int vh,
    int n,
    int chunks)
{
    extern __shared__ float sm[];

    constexpr int P   = DIM + 1;
    constexpr int PK  = DIM + 4;
    constexpr int VS4 = VS >> 2;
    constexpr int DIM4 = DIM >> 2;
    constexpr int ST  = VS + 4;

    const int head = blockIdx.x;
    const int v0 = blockIdx.y * VS;
    const int tid = threadIdx.x;
    const int key_head = head % kh;

    float* S_T = sm;
    float* Wc  = S_T + DIM * ST;
    float* Qc  = Wc + C * PK;
    float* Kc  = Qc + C * PK;
    float* E   = Kc + C * PK;
    float* M   = E + C * (VS + 4);
    float* eg  = M + C * C;
    float* rg  = eg + C;

    for (int i = tid; i < DIM * VS; i += blockDim.x) {
        const int d = i / VS;
        const int v = i - d * VS;
        S_T[d * ST + v] = __ldg(state + (int64_t(head) * DIM + v0 + v) * DIM + d);
    }

    const int et = tid >> 4;
    const int ev = (tid & 15) << 2;
    const int d4 = tid & 31;
    const int v4 = tid >> 5;

    for (int c = 0; c < chunks; ++c) {
        const int t0 = c * C;
        int cn = C;
        if (n - t0 < cn) cn = n - t0;

        __syncthreads();

        const float* Ug = U + (int64_t(head) * chunks + c) * C * DIM;
        const float* Wg = W + (int64_t(head) * chunks + c) * C * DIM;
        const float* l  = lgi + (int64_t(head) * chunks + c) * C;

        if (tid < C) {
            eg[tid] = expf(l[tid]);
            rg[tid] = (tid < cn) ? expf(l[cn - 1] - l[tid]) : 0.f;
        }
        __syncthreads();

        if (cn == C) {
            for (int i = tid; i < C * DIM4; i += blockDim.x) {
                const int t = i / DIM4;
                const int d4x = i - t * DIM4;

                const float4 wv = __ldg(reinterpret_cast<const float4*>(Wg + t * DIM) + d4x);
                const float4 qv = __ldg(reinterpret_cast<const float4*>(
                    qk + (int64_t(t0 + t) * 2 * kh + key_head) * DIM) + d4x);
                float4 kv = __ldg(reinterpret_cast<const float4*>(
                    qk + (int64_t(t0 + t) * 2 * kh + kh + key_head) * DIM) + d4x);

                const float r = rg[t];
                kv.x *= r;
                kv.y *= r;
                kv.z *= r;
                kv.w *= r;

                reinterpret_cast<float4*>(Wc + t * PK)[d4x] = wv;
                reinterpret_cast<float4*>(Qc + t * PK)[d4x] = qv;
                reinterpret_cast<float4*>(Kc + t * PK)[d4x] = kv;
            }

            for (int i = tid; i < C * VS4; i += blockDim.x) {
                const int t = i / VS4;
                const int v4x = i - t * VS4;
                const int v = v4x << 2;
                const float4 uv = __ldg(reinterpret_cast<const float4*>(Ug + t * DIM + v0) + v4x);

                reinterpret_cast<float4*>(E + t * (VS + 4))[v4x] = uv;
            }

            for (int i = tid; i < (C * C) / 4; i += blockDim.x) {
                reinterpret_cast<float4*>(M)[i] =
                    __ldg(reinterpret_cast<const float4*>(Mg + (int64_t(head) * chunks + c) * C * C) + i);
            }
        } else {
            for (int i = tid; i < C * DIM; i += blockDim.x) {
                const int t = i / DIM;
                const int d = i - t * DIM;
                const bool in = t < cn;
                const int64_t tt = int64_t(t0 + t);

                Wc[t * PK + d] = __ldg(Wg + i);
                Qc[t * PK + d] = in ? __ldg(qk + (tt * 2 * kh + key_head) * DIM + d) : 0.f;
                Kc[t * PK + d] = in ? __ldg(qk + (tt * 2 * kh + kh + key_head) * DIM + d) * rg[t] : 0.f;
            }

            for (int i = tid; i < C * VS; i += blockDim.x) {
                const int t = i / VS;
                const int v = i - t * VS;
                E[t * (VS + 4) + v] = (t < cn) ? __ldg(Ug + t * DIM + v0 + v) : 0.f;
            }

            for (int i = tid; i < C * C; i += blockDim.x) {
                M[i] = __ldg(Mg + (int64_t(head) * chunks + c) * C * C + i);
            }
        }

        __syncthreads();

        float ws[4], qs[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            ws[k] = 0.f;
            qs[k] = 0.f;
        }

        for (int d = 0; d < DIM; ++d) {
            const float w = Wc[et * PK + d];
            const float q = Qc[et * PK + d];
            const float4 s4 = reinterpret_cast<const float4*>(S_T + d * ST)[ev >> 2];

            ws[0] = __fmaf_rn(w, s4.x, ws[0]);
            ws[1] = __fmaf_rn(w, s4.y, ws[1]);
            ws[2] = __fmaf_rn(w, s4.z, ws[2]);
            ws[3] = __fmaf_rn(w, s4.w, ws[3]);

            qs[0] = __fmaf_rn(q, s4.x, qs[0]);
            qs[1] = __fmaf_rn(q, s4.y, qs[1]);
            qs[2] = __fmaf_rn(q, s4.z, qs[2]);
            qs[3] = __fmaf_rn(q, s4.w, qs[3]);
        }

#pragma unroll
        for (int k = 0; k < 4; ++k) {
            E[et * (VS + 4) + ev + k] -= ws[k];
        }

        __syncthreads();

        if (et < cn) {
            float o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                o[k] = eg[et] * qs[k];
            }

            for (int j = 0; j <= et; ++j) {
                const float m = M[et * C + j];
                const float4 e4 = reinterpret_cast<const float4*>(E + j * (VS + 4))[ev >> 2];

                o[0] = __fmaf_rn(m, e4.x, o[0]);
                o[1] = __fmaf_rn(m, e4.y, o[1]);
                o[2] = __fmaf_rn(m, e4.z, o[2]);
                o[3] = __fmaf_rn(m, e4.w, o[3]);
            }

            const int64_t tok = int64_t(t0 + et);
            reinterpret_cast<float4*>(out)[(tok * vh + head) * (DIM / 4) + (v0 + ev) / 4] =
                make_float4(o[0], o[1], o[2], o[3]);
        }

        __syncthreads();

        const float gc = eg[cn - 1];
        float acc[4][4];

#pragma unroll
        for (int dd = 0; dd < 4; ++dd) {
#pragma unroll
            for (int vv = 0; vv < 4; ++vv) {
                acc[dd][vv] = gc * S_T[(d4 * 4 + dd) * ST + v4 * 4 + vv];
            }
        }

        if (cn == C) {
#pragma unroll
            for (int j = 0; j < C; ++j) {
                const float4 e4 = reinterpret_cast<const float4*>(E + j * (VS + 4))[v4];
                const float4 k4 = reinterpret_cast<const float4*>(Kc + j * PK)[d4];

                acc[0][0] = __fmaf_rn(k4.x, e4.x, acc[0][0]);
                acc[0][1] = __fmaf_rn(k4.x, e4.y, acc[0][1]);
                acc[0][2] = __fmaf_rn(k4.x, e4.z, acc[0][2]);
                acc[0][3] = __fmaf_rn(k4.x, e4.w, acc[0][3]);

                acc[1][0] = __fmaf_rn(k4.y, e4.x, acc[1][0]);
                acc[1][1] = __fmaf_rn(k4.y, e4.y, acc[1][1]);
                acc[1][2] = __fmaf_rn(k4.y, e4.z, acc[1][2]);
                acc[1][3] = __fmaf_rn(k4.y, e4.w, acc[1][3]);

                acc[2][0] = __fmaf_rn(k4.z, e4.x, acc[2][0]);
                acc[2][1] = __fmaf_rn(k4.z, e4.y, acc[2][1]);
                acc[2][2] = __fmaf_rn(k4.z, e4.z, acc[2][2]);
                acc[2][3] = __fmaf_rn(k4.z, e4.w, acc[2][3]);

                acc[3][0] = __fmaf_rn(k4.w, e4.x, acc[3][0]);
                acc[3][1] = __fmaf_rn(k4.w, e4.y, acc[3][1]);
                acc[3][2] = __fmaf_rn(k4.w, e4.z, acc[3][2]);
                acc[3][3] = __fmaf_rn(k4.w, e4.w, acc[3][3]);
            }
        } else {
            for (int j = 0; j < cn; ++j) {
                const float4 e4 = reinterpret_cast<const float4*>(E + j * (VS + 4))[v4];
                const float4 k4 = reinterpret_cast<const float4*>(Kc + j * PK)[d4];

                acc[0][0] = __fmaf_rn(k4.x, e4.x, acc[0][0]);
                acc[0][1] = __fmaf_rn(k4.x, e4.y, acc[0][1]);
                acc[0][2] = __fmaf_rn(k4.x, e4.z, acc[0][2]);
                acc[0][3] = __fmaf_rn(k4.x, e4.w, acc[0][3]);

                acc[1][0] = __fmaf_rn(k4.y, e4.x, acc[1][0]);
                acc[1][1] = __fmaf_rn(k4.y, e4.y, acc[1][1]);
                acc[1][2] = __fmaf_rn(k4.y, e4.z, acc[1][2]);
                acc[1][3] = __fmaf_rn(k4.y, e4.w, acc[1][3]);

                acc[2][0] = __fmaf_rn(k4.z, e4.x, acc[2][0]);
                acc[2][1] = __fmaf_rn(k4.z, e4.y, acc[2][1]);
                acc[2][2] = __fmaf_rn(k4.z, e4.z, acc[2][2]);
                acc[2][3] = __fmaf_rn(k4.z, e4.w, acc[2][3]);

                acc[3][0] = __fmaf_rn(k4.w, e4.x, acc[3][0]);
                acc[3][1] = __fmaf_rn(k4.w, e4.y, acc[3][1]);
                acc[3][2] = __fmaf_rn(k4.w, e4.z, acc[3][2]);
                acc[3][3] = __fmaf_rn(k4.w, e4.w, acc[3][3]);
            }
        }

#pragma unroll
        for (int dd = 0; dd < 4; ++dd) {
            reinterpret_cast<float4*>(S_T + (d4 * 4 + dd) * ST)[v4] =
                make_float4(acc[dd][0], acc[dd][1], acc[dd][2], acc[dd][3]);
        }
    }

    __syncthreads();

    for (int i = tid; i < DIM * VS; i += blockDim.x) {
        const int d = i / VS;
        const int v = i - d * VS;
        next[(int64_t(head) * DIM + v0 + v) * DIM + d] = S_T[d * ST + v];
    }
}

template <int C, int VS, int DIM>
__global__ void __launch_bounds__(256) cand_gdn_seq_generic(
    const float* __restrict__ qk,
    const float* __restrict__ U,
    const float* __restrict__ W,
    const float* __restrict__ Mg,
    const float* __restrict__ lgi,
    const float* __restrict__ state,
    float* __restrict__ next,
    float* __restrict__ out,
    int kh,
    int vh,
    int dim,
    int n,
    int chunks)
{
    extern __shared__ float sm[];
    const int P = dim + 1;
    const int head = blockIdx.x;
    const int v0 = blockIdx.y * VS;
    const int tid = threadIdx.x;
    const int key_head = head % kh;

    float* S  = sm;
    float* Wc = S + VS * P;
    float* Qc = Wc + C * P;
    float* Kc = Qc + C * P;
    float* E  = Kc + C * P;
    float* M  = E + C * (VS + 1);
    float* eg = M + C * C;
    float* rg = eg + C;

    for (int i = tid; i < VS * dim; i += blockDim.x) {
        const int r = i / dim;
        const int d = i - r * dim;
        S[r * P + d] = state[(int64_t(head) * dim + v0 + r) * dim + d];
    }

    for (int c = 0; c < chunks; ++c) {
        const int t0 = c * C;
        int cn = C;
        if (n - t0 < cn) cn = n - t0;

        __syncthreads();

        const float* Ug = U + (int64_t(head) * chunks + c) * C * dim;
        const float* Wg = W + (int64_t(head) * chunks + c) * C * dim;
        const float* l  = lgi + (int64_t(head) * chunks + c) * C;

        if (tid < C) {
            eg[tid] = expf(l[tid]);
            rg[tid] = (tid < cn) ? expf(l[cn - 1] - l[tid]) : 0.f;
        }
        __syncthreads();

        for (int i = tid; i < C * dim; i += blockDim.x) {
            const int t = i / dim;
            const int d = i - t * dim;
            const bool in = t < cn;
            const int64_t tt = int64_t(t0 + t);

            Wc[t * P + d] = Wg[i];
            Qc[t * P + d] = in ? qk[(tt * 2 * kh + key_head) * dim + d] : 0.f;
            Kc[t * P + d] = in ? qk[(tt * 2 * kh + kh + key_head) * dim + d] * rg[t] : 0.f;
        }

        for (int i = tid; i < C * VS; i += blockDim.x) {
            const int t = i / VS;
            const int v = i - t * VS;
            E[t * (VS + 1) + v] = (t < cn) ? Ug[t * dim + v0 + v] : 0.f;
        }

        for (int i = tid; i < C * C; i += blockDim.x) {
            M[i] = Mg[(int64_t(head) * chunks + c) * C * C + i];
        }

        __syncthreads();

        float ws[4], qs[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            ws[k] = 0.f;
            qs[k] = 0.f;
        }

        const int et = tid / 8;
        const int ev = tid % 8;

        for (int d = 0; d < dim; ++d) {
            const float w = Wc[et * P + d];
            const float q = Qc[et * P + d];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float s = S[(ev + 8 * k) * P + d];
                ws[k] = __fmaf_rn(w, s, ws[k]);
                qs[k] = __fmaf_rn(q, s, qs[k]);
            }
        }

#pragma unroll
        for (int k = 0; k < 4; ++k) {
            E[et * (VS + 1) + ev + 8 * k] -= ws[k];
        }

        __syncthreads();

        if (et < cn) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int v = ev + 8 * k;
                float o = eg[et] * qs[k];
                for (int j = 0; j <= et; ++j) {
                    o = __fmaf_rn(M[et * C + j], E[j * (VS + 1) + v], o);
                }
                out[(int64_t(t0 + et) * vh + head) * dim + v0 + v] = o;
            }
        }

        __syncthreads();

        const float gc = eg[cn - 1];
        const int sv = tid / 8;
        const int sd = tid % 8;

        for (int k = sd; k < dim; k += 8) {
            float acc = gc * S[sv * P + k];
            for (int j = 0; j < cn; ++j) {
                acc = __fmaf_rn(E[j * (VS + 1) + sv], Kc[j * P + k], acc);
            }
            S[sv * P + k] = acc;
        }
    }

    __syncthreads();

    for (int i = tid; i < VS * dim; i += blockDim.x) {
        const int r = i / dim;
        const int d = i - r * dim;
        next[(int64_t(head) * dim + v0 + r) * dim + d] = S[r * P + d];
    }
}

} // namespace gdn_seq_fast_ns

static void fn_gdn_seq_fast(const float* qk,
                  const float* U,
                  const float* W,
                  const float* Mg,
                  const float* lg,
                  const float* state,
                  float* next,
                  float* out,
                  int kh,
                  int vh,
                  int dim,
                  int n,
                  int chunks,
                  cudaStream_t s)
{
    constexpr int C = 32;
    constexpr int VS_FAST = 64;
    constexpr int DIM_FAST = 128;

    if (dim == DIM_FAST && chunks > 0) {
        const size_t smem =
            (size_t(DIM_FAST) * (VS_FAST + 4) +
             size_t(3 * C) * (DIM_FAST + 4) +
             size_t(C) * (VS_FAST + 4) +
             size_t(C * C) +
             size_t(2 * C)) * sizeof(float);

        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(gdn_seq_fast_ns::cand_gdn_seq_fast<C, VS_FAST, DIM_FAST>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 int(smem));
            configured = true;
        }

        gdn_seq_fast_ns::cand_gdn_seq_fast<C, VS_FAST, DIM_FAST><<<dim3(vh, DIM_FAST / VS_FAST), 512, smem, s>>>(
            qk, U, W, Mg, lg, state, next, out, kh, vh, n, chunks);
        return;
    }

    constexpr int VS_GENERIC = 32;
    const size_t smem =
        (size_t(VS_GENERIC) * (dim + 1) +
         size_t(3 * C) * (dim + 1) +
         size_t(C) * (VS_GENERIC + 1) +
         size_t(C * C) +
         size_t(2 * C)) * sizeof(float);

    static bool configured = false;
    if (smem > 48u * 1024u && !configured) {
        cudaFuncSetAttribute(gdn_seq_fast_ns::cand_gdn_seq_generic<C, VS_GENERIC, 128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             int(smem));
        configured = true;
    }

    gdn_seq_fast_ns::cand_gdn_seq_generic<C, VS_GENERIC, 128><<<dim3(vh, dim / VS_GENERIC), 256, smem, s>>>(
        qk, U, W, Mg, lg, state, next, out, kh, vh, dim, n, chunks);
}
