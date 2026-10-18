#include "int8_tiles.hpp"
#include "tq4.hpp"
#include <cstdlib>
#include <stdexcept>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;

mx::array verify_tq3_int8(const Tq3Weights& w, const mx::array& input, int planes) {
    if (input.ndim() != 2 || input.dtype() != mx::float32 || input.shape(0) <= 0 ||
        input.shape(1) <= 0 || input.shape(1) % 128 || planes < 2 || planes > 3 ||
        w.planes.ndim() != 3 || w.planes.dtype() != mx::uint32)
        throw std::invalid_argument("invalid integer TQ3 projection");
    const int m = input.shape(0), k = input.shape(1), r = w.planes.shape(0);
    if (r <= 0 || w.scales.shape() != mx::Shape{r, k / 128} ||
        w.scales.dtype() != mx::float16 ||
        w.planes.shape() != (w.interleaved ? mx::Shape{r, k / 32, 4} : mx::Shape{r, 3, k / 32}))
        throw std::invalid_argument("integer TQ3 weight shape mismatch");
    auto option = [](const char* key, int fallback) {
        const char* value = std::getenv(key);
        return value ? std::stoi(value) : fallback;
    };
    const int nt = option("TQLLM_MLX_INT8_N", 64), simds = option("TQLLM_MLX_INT8_SIMDS", 2);
    if ((nt != 32 && nt != 64 && nt != 128) || (simds != 1 && simds != 2 && simds != 4))
        throw std::invalid_argument("invalid integer TQ3 tile");
    static auto quantize = mx::fast::metal_kernel("tqllm_residual_int8", {"x"}, {"codes", "as"}, R"METAL(
uint group=threadgroup_position_in_grid.x,lane=thread_position_in_threadgroup.x;
float4 residual=((const device float4*)x)[group*32u+lane];
for(uint p=0;p<uint(P);++p) {
 float peak=simd_max(max(max(abs(residual.x),abs(residual.y)),max(abs(residual.z),abs(residual.w))));
 float scale=peak/127.0f;
 float4 q=scale>0.0f?clamp(rint(residual/scale),float4(-127.0f),float4(127.0f)):float4(0.0f);
 ((device char4*)codes)[p*M*(K/4u)+group*32u+lane]=char4(q);
 if(lane==0u)as[p*M*(K/128u)+group]=scale;
 residual-=q*scale;
}
)METAL");
    auto a = quantize({tq4_rotate(input)}, {{planes, m, k}, {planes, m, k / 128}},
        {mx::int8, mx::float32}, {m * (k / 128) * 32, 1, 1}, {32, 1, 1},
        {{"P", planes}, {"M", m}, {"K", k}}, std::nullopt, false, mx::Device(mx::Device::gpu));
    static auto multiply = mx::fast::metal_kernel("tqllm_tq3_int8_tiles",
        {"packed", "ws", "codes", "as"}, {"y"}, R"METAL(
uint tid=thread_position_in_threadgroup.x;
uint row0=threadgroup_position_in_grid.y*uint(NT),m0=threadgroup_position_in_grid.z*8u;
threadgroup int8_t A[8*128],B[NT*128];
threadgroup float C[8*NT];
tensor<threadgroup int8_t,extents<int,128,8>,tensor_inline> ta(A,extents<int,128,8>());
tensor<threadgroup int8_t,extents<int,128,NT>,tensor_inline> tb(B,extents<int,128,NT>());
constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(8,NT,128,false,true,false,mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroups<SIMDS>> op;
for(uint i=tid;i<8u*uint(NT);i+=32u*uint(SIMDS))C[i]=0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint base=0;base<K;base+=128u) {
 for(uint i=tid;i<uint(NT)*32u;i+=32u*uint(SIMDS)) {
  uint row=row0+i/32u,slot=i%32u,block=base/32u+slot/8u,j=slot%8u;
  uint dest=(row-row0)*128u+(slot/8u)*32u+j;
  uint a0=0u,a1=0u,signs=0u;
  if(row<R) {
   a0=INTERLEAVED?packed[(row*(K/32u)+block)*4u]:packed[(row*3u)*(K/32u)+block];
   a1=INTERLEAVED?packed[(row*(K/32u)+block)*4u+1u]:packed[(row*3u+1u)*(K/32u)+block];
   signs=INTERLEAVED?packed[(row*(K/32u)+block)*4u+2u]:packed[(row*3u+2u)*(K/32u)+block];
  }
  a0>>=4u*j;a1>>=4u*j;signs>>=4u*j;
  B[dest]=row<R?L3[(a0&3u)|((signs&1u)<<2u)]:int8_t(0);
  B[dest+8u]=row<R?L3[((a0>>2u)&3u)|(((signs>>1u)&1u)<<2u)]:int8_t(0);
  B[dest+16u]=row<R?L3[(a1&3u)|(((signs>>2u)&1u)<<2u)]:int8_t(0);
  B[dest+24u]=row<R?L3[((a1>>2u)&3u)|(((signs>>3u)&1u)<<2u)]:int8_t(0);
 }
 for(uint p=0;p<uint(P);++p) {
  for(uint i=tid;i<8u*128u;i+=32u*uint(SIMDS)) {
   uint m=m0+i/128u;
   A[i]=m<M?codes[(p*M+m)*K+base+i%128u]:int8_t(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto acc=op.template get_destination_cooperative_tensor<decltype(ta),decltype(tb),int>();
  for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0;
  op.run(ta,tb,acc);
  for(uint i=0;i<acc.get_capacity();++i) {
   if(!acc.is_valid_element(i))continue;
   auto coord=acc.get_multidimensional_index(i);
   uint n=uint(coord[0]),m=uint(coord[1]);
   if(row0+n<R && m0+m<M)
    C[m*uint(NT)+n]+=float(acc[i])*float(ws[(row0+n)*(K/128u)+base/128u])*as[(p*M+m0+m)*(K/128u)+base/128u];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}
for(uint i=tid;i<8u*uint(NT);i+=32u*uint(SIMDS)) {
 uint m=m0+i/uint(NT),r=row0+i%uint(NT);
 if(m<M && r<R)y[m*R+r]=C[i];
}
)METAL", R"METAL(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
constant int8_t L3[8]={14,45,79,127,-14,-45,-79,-127};
)METAL");
    return multiply({w.planes, w.scales, a[0], a[1]}, {{m, r}}, {mx::float32},
        {32 * simds, (r + nt - 1) / nt, (m + 7) / 8}, {32 * simds, 1, 1},
        {{"NT", nt}, {"SIMDS", simds}, {"P", planes}, {"M", m}, {"R", r}, {"K", k},
         {"INTERLEAVED", w.interleaved}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}
