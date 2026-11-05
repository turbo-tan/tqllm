#include "q8.hpp"
#include <stdexcept>
namespace tqllm::mlx_backend {
namespace mx = mlx::core;
Q8Weights load_q8_blocks(std::span<const uint8_t> blocks, int rows, int columns) {
    if (rows <= 0 || columns <= 0 || columns % 32 ||
        blocks.size() != uint64_t(rows) * uint64_t(columns / 32) * 34)
        throw std::invalid_argument("invalid Q8_0 block payload or dimensions");
    return {mx::array(blocks.data(), {rows, columns / 32, 34}, mx::uint8)};
}
mx::array q8_matmul(const Q8Weights& w, const mx::array& x) {
    if (w.blocks.ndim() != 3 || w.blocks.shape(2) != 34 || w.blocks.dtype() != mx::uint8 ||
        x.ndim() != 2 || x.dtype() != mx::float32)
        throw std::invalid_argument("invalid Q8_0 matrix types");
    const int rows = w.blocks.shape(0), k = w.blocks.shape(1) * 32, m = x.shape(0);
    if (rows <= 0 || k <= 0 || m <= 0 || x.shape(1) != k)
        throw std::invalid_argument("incompatible Q8_0 matrix dimensions");
    static auto kernel = mx::fast::metal_kernel("tqllm_q8_matmul", {"blocks", "x"}, {"y"}, R"METAL(
uint lane=thread_position_in_threadgroup.x;
uint row=threadgroup_position_in_grid.y*threads_per_threadgroup.y+thread_position_in_threadgroup.y;
uint col=threadgroup_position_in_grid.z*4;
if(row>=R) return;
float acc[4]={0.0f,0.0f,0.0f,0.0f};
for(uint b=lane;b<K/32;b+=32) {
 const device uchar* p=blocks+(row*(K/32)+b)*34;
 float d=float(as_type<half>(ushort(uint(p[0])|(uint(p[1])<<8))));
 for(uint m=0;m<4;++m) {
  if(col+m>=M) break;
  float partial=0.0f;
  for(uint j=0;j<32;++j) partial+=(float(as_type<char>(p[2+j]))*d)*x[(col+m)*K+b*32+j];
  acc[m]+=partial;
 }
}
for(uint m=0;m<4;++m) {
 if(col+m>=M) break;
 float value=simd_sum(acc[m]);
 if(lane==0) y[(col+m)*R+row]=value;
}
)METAL");
    return kernel({w.blocks,x},{{m,rows}},{mx::float32},
                  {32,(rows+1)/2*2,(m+3)/4},{32,2,1},
                  {{"R",rows},{"K",k},{"M",m}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
}
