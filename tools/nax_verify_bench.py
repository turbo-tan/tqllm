#!/usr/bin/env python3
# M5 neural-accelerator (MPP matmul2d 16x32x16) verify matmul: Y[MR,N] = X[MR,K] @ W^T, W 4-bit affine g128.
# CONST=1 keeps the weight tile in registers (no weight loads, no decode): the matrix-unit ceiling.
import json, sys, time
import numpy as np

HEADER = """
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

SRC = """
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + sg;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto ca2 = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto cc2 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (short i = 0; i < 16; i++) { cc[i] = 0.0f; cc2[i] = 0.0f; }
    if (CONST) { for (short i = 0; i < 16; i++) cb[i] = half(0.001f * float(lane + i)); }
    const device uint2* wp = (const device uint2*)W + (size_t)blk * (K / 16) * 32 + lane;
    const device half* sp = S + (size_t)blk * (K / 128) * 32;
    const device half* bp = Bz + (size_t)blk * (K / 128) * 32;
    const device half* xs = X + fm * K + fn;
    for (int g = 0; g < K / 128; g++) {
        half4 s0 = *(const device half4*)(sp + g * 32 + fn), s1 = *(const device half4*)(sp + g * 32 + 16 + fn);
        half4 b0 = *(const device half4*)(bp + g * 32 + fn), b1 = *(const device half4*)(bp + g * 32 + 16 + fn);
        for (short t = 0; t < 8; t++) {
            const int ks = g * 8 + t;
            half4 a0 = *(const device half4*)(xs + ks * 16);
            half4 a1 = MR > 8 ? *(const device half4*)(xs + 8 * K + ks * 16) : half4(0.0h);
            for (short i = 0; i < 4; i++) { ca[i] = a0[i]; ca[4 + i] = a1[i]; }
            if (!CONST) {
                uint2 w = wp[ks * 32];
                for (short p = 0; p < 4; p++) {
                    half2 q0 = as_type<half2>(((w.x >> (4 * p)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
                    half2 q1 = as_type<half2>(((w.y >> (4 * p)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
                    half2 v0 = fma(q0, half2(s0[p]), half2(b0[p]));
                    half2 v1 = fma(q1, half2(s1[p]), half2(b1[p]));
                    cb[p] = v0.x; cb[p + 4] = v0.y; cb[8 + p] = v1.x; cb[12 + p] = v1.y;
                }
            }
            op.run(ca, cb, cc);
            if (MR > 16) {
                half4 a2 = *(const device half4*)(xs + 16 * K + ks * 16);
                half4 a3 = *(const device half4*)(xs + 24 * K + ks * 16);
                for (short i = 0; i < 4; i++) { ca2[i] = a2[i]; ca2[4 + i] = a3[i]; }
                op.run(ca2, cb, cc2);
            }
        }
    }
    const uint col0 = blk * 32 + fn;
    for (short i = 0; i < 16; i++) {
        const short r = fm + ((i & 7) >> 2) * 8;
        const uint c = col0 + (i >> 3) * 16 + (i & 3);
        if (r < MR) Y[r * N + c] = cc[i];
        if (MR > 16) Y[(r + 16) * N + c] = cc2[i];
    }
"""


def lane_coords():
    lane = np.arange(32)
    qid = lane >> 2
    fm = (qid & 4) | ((lane >> 1) & 3)
    fn = ((qid & 2) | (lane & 1)) * 4
    return fm, fn


def pack(q, s, b):
    # q [N,K] in 0..15, s/b [N,K/128] -> words [N/32, K/16, 32, 2] u32, S/B [N/32, K/128, 32] f16
    N, K = q.shape
    fm, fn = lane_coords()
    words = np.zeros((N // 32, K // 16, 32, 2), np.uint32)
    for w in range(2):
        for i in range(8):
            rows = fm + (i >> 2) * 8                    # k within the 16-step, per lane
            cols = w * 16 + fn + (i & 3)                # n within the 32-block, per lane
            qq = q.reshape(N // 32, 32, K // 16, 16)[:, cols, :, rows]   # [32 lanes, N/32, K/16]
            words[..., w] |= np.transpose(qq, (1, 2, 0)).astype(np.uint32) << (4 * i)
    S = s.reshape(N // 32, 32, K // 128).transpose(0, 2, 1).astype(np.float16)
    B = b.reshape(N // 32, 32, K // 128).transpose(0, 2, 1).astype(np.float16)
    return words, S, B


def emulate(words, S, B, x, N, K):
    # numpy replay of the kernel's decode + lane mapping (checks pack(), not the matrix unit)
    fm, fn = lane_coords()
    W = np.zeros((N, K), np.float32)
    for w in range(2):
        for i in range(8):
            qv = (words[..., w] >> (4 * i)) & 15          # [N/32, K/16, 32]
            rows = fm + (i >> 2) * 8
            cols = w * 16 + fn + (i & 3)
            g = np.arange(K // 16) // 8
            sc = S[:, g][:, :, cols].astype(np.float32)   # [N/32, K/16, 32]
            bz = B[:, g][:, :, cols].astype(np.float32)
            v = qv * sc + bz
            for L in range(32):
                n = np.arange(N // 32)[:, None] * 32 + cols[L]
                k = np.arange(K // 16)[None, :] * 16 + rows[L]
                W[n, k] = v[:, :, L]
    return x.astype(np.float32) @ W.T


def make(N, K, MR, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 16, (N, K), dtype=np.uint8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float32)
    b = (rng.standard_normal((N, K // 128)) * 0.02).astype(np.float32)
    x = rng.standard_normal((max(MR, 16) if MR <= 16 else MR, K)).astype(np.float16)
    if MR < 16: x[MR:] = 0
    wd = (q.astype(np.float32).reshape(N, K // 128, 128) * s[:, :, None] + b[:, :, None]).reshape(N, K)
    return q, s, b, x, wd


def selftest():
    N, K, MR = 64, 256, 16
    q, s, b, x, wd = make(N, K, MR)
    words, S, B = pack(q, s, b)
    ref = x.astype(np.float32) @ wd.T
    em = emulate(words, S, B, x, N, K)
    print("pack selftest rel err", float(np.abs(em - ref).max() / np.abs(ref).max()))


def run(N, K, MR, reps=640):
    import mlx.core as mx
    q, s, b, x, wd = make(N, K, MR)
    words, S, B = pack(q, s, b)
    C = COPIES
    Ws = [mx.array(words.reshape(-1)) for _ in range(C)]
    Ss = [mx.array(S.reshape(-1)) for _ in range(C)]
    Bs = [mx.array(B.reshape(-1)) for _ in range(C)]
    X = mx.array(x)
    mx.eval(Ws, Ss, Bs, X)
    ctr = [0]
    def kern(const, j=None):
        if j is None: j = ctr[0] = (ctr[0] + 1) % C
        return KERNEL(inputs=[Ws[j], Ss[j], Bs[j], X], output_shapes=[(MR, N)], output_dtypes=[mx.float32],
                      grid=(N // 128 * 128, 1, 1), threadgroup=(128, 1, 1),
                      template=[("K", K), ("N", N), ("MR", MR), ("CONST", const)])[0]
    y = kern(0, 0); mx.eval(y)
    ref = x[:MR].astype(np.float32) @ wd.T
    err = float(np.abs(np.array(y) - ref).max() / np.abs(ref).max())
    wq, ss, bb = mx.quantize(mx.array(wd), group_size=128, bits=4)
    qs = [(wq + 0, ss + 0, bb + 0) for _ in range(C)]; mx.eval(qs)
    xb = mx.array(x[:MR]).astype(mx.bfloat16)
    sc = [0]
    def stock():
        sc[0] = (sc[0] + 1) % C; a, b2, c = qs[sc[0]]
        return mx.quantized_matmul(xb, a, b2, c, transpose=True, group_size=128, bits=4)
    out = {"N": N, "K": K, "MR": MR, "copies": C, "rel_err": round(err, 5)}
    bytes_ = N * K // 2 + N * K // 128 * 4
    for name, fn in (("nax", lambda: kern(0)), ("nax_const", lambda: kern(1)), ("stock_qmm", stock)):
        Bt = 64
        mx.eval([fn() for _ in range(Bt)])
        t0 = time.perf_counter()
        for _ in range(reps // Bt + 1): mx.eval([fn() for _ in range(Bt)])
        ms = (time.perf_counter() - t0) / ((reps // Bt + 1) * Bt) * 1e3
        out[name + "_ms"] = round(ms, 4)
        out[name + "_GBs"] = round(bytes_ / ms / 1e6, 1)
        out[name + "_TFLOPS16"] = round(2 * max(MR, 16) * N * K / ms / 1e9, 2)
    return out


if __name__ == "__main__":
    selftest()
    if "--selftest" in sys.argv: sys.exit(0)
    import mlx.core as mx
    KERNEL = mx.fast.metal_kernel(name="nax_verify", input_names=["W", "S", "Bz", "X"], output_names=["Y"],
                                  source=SRC, header=HEADER)
    COPIES = 8
    for N, K in ((17408, 5120), (5120, 17408), (10240, 5120), (34816, 5120), (6144, 5120)):
        for MR in (1, 8, 16):
            try:
                print(json.dumps(run(N, K, MR)), flush=True)
            except Exception as e:
                print(json.dumps({"N": N, "K": K, "MR": MR, "error": str(e)[:1500]}), flush=True)
