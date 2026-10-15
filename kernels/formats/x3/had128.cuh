#pragma once
// had128.cuh: the 128-point natural-order Hadamard transform (1/sqrt(128) per side) used on both
// sides of an X3 weight tile. Ported verbatim (source line comments kept) from llama-paw-k25
// ggml/src/ggml-cuda/paw-x3.cu (hadamard_inner.cuh).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tqllm {
namespace x3 {

struct half4 { half2 x, y; };

__device__ inline void shuffle_had_f4x32(float& h0, float& h1, float& h2, float& h3, const int lane_id) {
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        uint32_t i0 = __float_as_uint(h0), i1 = __float_as_uint(h1);
        uint32_t i2 = __float_as_uint(h2), i3 = __float_as_uint(h3);
        uint64_t h01 = (uint64_t) i0 | (((uint64_t) i1) << 32);
        uint64_t h23 = (uint64_t) i2 | (((uint64_t) i3) << 32);
        uint64_t ph01 = __shfl_xor_sync(0xffffffff, h01, i);
        uint64_t ph23 = __shfl_xor_sync(0xffffffff, h23, i);
        float ph0 = __uint_as_float((uint32_t) (ph01 & 0xffffffff));
        float ph1 = __uint_as_float((uint32_t) (ph01 >> 32));
        float ph2 = __uint_as_float((uint32_t) (ph23 & 0xffffffff));
        float ph3 = __uint_as_float((uint32_t) (ph23 >> 32));
        int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
        i0 ^= sfm & 0x80000000; i1 ^= sfm & 0x80000000; i2 ^= sfm & 0x80000000; i3 ^= sfm & 0x80000000;
        h0 = __uint_as_float(i0) + ph0;
        h1 = __uint_as_float(i1) + ph1;
        h2 = __uint_as_float(i2) + ph2;
        h3 = __uint_as_float(i3) + ph3;
    }
}

__device__ inline void shuffle_had_f2x32(float& v, float& w, const int lane_id) {
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        uint64_t vw = ((uint64_t) __float_as_uint(v)) | (((uint64_t) __float_as_uint(w)) << 32);
        uint64_t pvw = __shfl_xor_sync(0xffffffff, vw, i);
        float pv = __uint_as_float((uint32_t) (pvw & 0xffffffff));
        float pw = __uint_as_float((uint32_t) (pvw >> 32));
        uint32_t vi = __float_as_uint(v), wi = __float_as_uint(w);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        vi ^= (sfm & 0x80000000); wi ^= (sfm & 0x80000000);
        v = __uint_as_float(vi) + pv;
        w = __uint_as_float(wi) + pw;
    }
}

__device__ inline half2 shuffle_had_h2x32(half2 v, int lane_id) {
    for (int i = 1; i < 32; i <<= 1) {
        half2 pv = __shfl_xor_sync(0xffffffff, v, i);
        uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80008000);
        v = __hadd2(v, pv);
    }
    return v;
}

// Half vector, half scales (input transform: pre_scale = suh).
template <bool pre_scale, bool post_scale>
inline __device__ void had_hf_r_128_inner(const half* __restrict__ input_ptr, half* __restrict__ output_ptr,
                                          const half* __restrict__ scale, const float r_scale) {
    int t = threadIdx.x & 31;
    half4 v = ((half4*) input_ptr)[t];
    if constexpr (pre_scale) {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }
    float v0 = __half2float(__low2half(v.x)), v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y)), v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
    if constexpr (post_scale) {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }
    ((half4*) output_ptr)[t] = v;
}

// Fused f32-input variant for the gemv input prologue: round-to-nearest to half matches the scalar
// __float2half cast pass, so the separate cast can be skipped.
template <bool pre_scale, bool post_scale>
inline __device__ void had_hf_r_128_inner_f32(const float* __restrict__ input_ptr, half* __restrict__ output_ptr,
                                              const half* __restrict__ scale, const float r_scale) {
    int t = threadIdx.x & 31;
    float4 f = ((float4*) input_ptr)[t];
    half4 v;
    v.x = __floats2half2_rn(f.x, f.y);
    v.y = __floats2half2_rn(f.z, f.w);
    if constexpr (pre_scale) {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }
    float v0 = __half2float(__low2half(v.x)), v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y)), v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
    if constexpr (post_scale) {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }
    ((half4*) output_ptr)[t] = v;
}

// Float vector, half scales, half output (epilogue transform: post_scale = svh).
template <bool pre_scale, bool post_scale>
inline __device__ void had_fh_r_128_inner(const float* __restrict__ input_ptr, half* __restrict__ output_ptr,
                                          const half* __restrict__ scale, const float r_scale) {
    int t = threadIdx.x & 31;
    float4 v = ((float4*) input_ptr)[t];
    if constexpr (pre_scale) {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        v.x *= __low2float(scales.x); v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y); v.w *= __high2float(scales.y);
    }
    float v0 = v.x, v1 = v.y, v2 = v.z, v3 = v.w;
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    v.x = s0 + s1; v.y = d0 + d1; v.z = s0 - s1; v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= r_scale; v.y *= r_scale; v.z *= r_scale; v.w *= r_scale;
    half4 o;
    o.x = __floats2half2_rn(v.x, v.y);
    o.y = __floats2half2_rn(v.z, v.w);
    if constexpr (post_scale) {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) scale)[i];
        o.x = __hmul2(o.x, scales.x);
        o.y = __hmul2(o.y, scales.y);
    }
    ((half4*) output_ptr)[t] = o;
}

inline constexpr float kRScale = 0.08838834764831845f;  // 1/sqrt(128)

}  // namespace x3
}  // namespace tqllm
