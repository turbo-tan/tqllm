#include "q6.hpp"
#include <stdexcept>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
Q6Weights load_q6_blocks(std::span<const uint8_t> blocks, int rows, int columns) {
    if (rows <= 0 || columns <= 0 || columns % 256 ||
        blocks.size() != uint64_t(rows) * uint64_t(columns / 256) * 210)
        throw std::invalid_argument("invalid Q6_K block payload or dimensions");
    return {mx::array(blocks.data(), {rows, columns / 256, 210}, mx::uint8)};
}

mx::array q6_matmul(const Q6Weights& w, const mx::array& x) {
    if (w.blocks.ndim() != 3 || w.blocks.shape(2) != 210 || w.blocks.dtype() != mx::uint8 ||
        x.ndim() != 2 || x.dtype() != mx::float32)
        throw std::invalid_argument("invalid Q6_K matrix types");
    const int rows = w.blocks.shape(0), k = w.blocks.shape(1) * 256, m = x.shape(0);
    if (rows <= 0 || k <= 0 || m <= 0 || x.shape(1) != k)
        throw std::invalid_argument("incompatible Q6_K matrix dimensions");
    static auto kernel = mx::fast::metal_kernel("tqllm_q6_matmul", {"blocks", "x"}, {"y"}, R"METAL(
uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y * threads_per_threadgroup.y + thread_position_in_threadgroup.y;
uint col = threadgroup_position_in_grid.z * MCOLS;
if(row>=R) return;
float acc[MCOLS];
for(uint m=0;m<MCOLS;++m)acc[m]=0.0f;
for(uint b=lane;b<K/256;b+=32) {
 const device uchar* p=blocks+(row*(K/256)+b)*210;
 float d=float(as_type<half>(ushort(uint(p[208])|(uint(p[209])<<8))));
 float part[MCOLS];
 for(uint m=0;m<MCOLS;++m)part[m]=0.0f;
 for(uint j=0;j<256;++j) {
  uint h=j/128,l=j%32,c=(j%128)/32;
  uint lo=p[h*64+(c&1u)*32+l],hi=p[128+h*32+l];
  int q=int(((c<2?lo&15u:lo>>4)|(((hi>>(2*c))&3u)<<4)))-32;
  float s=d*float(as_type<char>(p[192+h*8+c*2+l/16]));
  float value=float(q)*s;
  for(uint m=0;m<MCOLS;++m) {
   if(col+m>=M)break;
   part[m]+=value*x[(col+m)*K+b*256+j];
  }
 }
 for(uint m=0;m<MCOLS;++m)acc[m]+=part[m];
}
for(uint m=0;m<MCOLS;++m) {
 if(col+m>=M) break;
 float value=simd_sum(acc[m]);
 if(lane==0) y[(col+m)*R+row]=value;
}
)METAL");
    const int cols=m<=1?1:m<=2?2:4;
    return kernel({w.blocks, x}, {{m, rows}}, {mx::float32},
                  {32, (rows + 1) / 2 * 2, (m + cols - 1) / cols}, {32, 2, 1},
                  {{"R", rows}, {"K", k}, {"M", m}, {"MCOLS", cols}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

}
