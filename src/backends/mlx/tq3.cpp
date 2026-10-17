#include "tq3.hpp"
#include "tuning.hpp"
#include <stdexcept>
#include <vector>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
Tq3Weights load_tq3_blocks(std::span<const uint8_t> blocks, int rows, int columns) {
    if (rows <= 0 || columns <= 0 || columns % 128)
        throw std::invalid_argument("invalid TQ3 block dimensions");
    const uint64_t groups = uint64_t(rows) * uint64_t(columns / 128);
    if (blocks.size() != groups * 50)
        throw std::invalid_argument("TQ3 block payload size mismatch");
    const size_t nb = size_t(columns) / 32;
    std::vector<uint32_t> planes(size_t(rows) * nb * 3);
    std::vector<uint16_t> scales(static_cast<size_t>(groups));
    for (uint64_t g = 0; g < groups; ++g) {
        scales[g] = uint16_t(blocks[g * 50]) | (uint16_t(blocks[g * 50 + 1]) << 8);
        const size_t row = g / (columns / 128), at = g % (columns / 128) * 4;
        for (size_t b = 0; b < 4; ++b)
            for (size_t p = 0; p < 3; ++p) {
                const uint8_t* v = blocks.data() + g * 50 + 2 + (b * 3 + p) * 4;
                planes[(row * 3 + p) * nb + at + b] = uint32_t(v[0]) | (uint32_t(v[1]) << 8) |
                    (uint32_t(v[2]) << 16) | (uint32_t(v[3]) << 24);
            }
    }
    return {mx::array(planes.data(), {rows, 3, columns / 32}, mx::uint32),
            mx::view(mx::array(scales.data(), {rows, columns / 128}, mx::uint16), mx::float16)};
}

Tq3Weights tq3_interleave(const Tq3Weights& w) {
    if(w.interleaved)return w;
    if(w.planes.ndim()!=3 || w.planes.shape(1)!=3 || w.planes.dtype()!=mx::uint32)
        throw std::invalid_argument("invalid TQ3 planar layout");
    auto packed=mx::contiguous(mx::concatenate({mx::transpose(w.planes,{0,2,1}),mx::zeros({w.planes.shape(0),w.planes.shape(2),1},mx::uint32)},2));
    mx::eval(packed);
    return {packed,w.scales,true};
}

mx::array tq3_matmul(const Tq3Weights& w, const mx::array& x) {
    if (w.planes.ndim() != 3 || (w.interleaved?w.planes.shape(2)!=4:w.planes.shape(1)!=3) || w.planes.dtype() != mx::uint32 ||
        w.scales.ndim() != 2 || w.scales.dtype() != mx::float16 || x.ndim() != 2 || x.dtype() != mx::float32)
        throw std::invalid_argument("invalid TQ3 matrix types");
    const int rows = w.planes.shape(0), k = (w.interleaved?w.planes.shape(1):w.planes.shape(2)) * 32, m = x.shape(0);
    if (rows <= 0 || k <= 0 || k % 128 || m <= 0 || x.shape(1) != k ||
        w.scales.shape(0) != rows || w.scales.shape(1) != k / 128)
        throw std::invalid_argument("incompatible TQ3 matrix shapes");
    static auto kernel = mx::fast::metal_kernel("tqllm_tq3_matmul", {"planes", "scales", "x"}, {"y"}, R"METAL(
uint lane = thread_position_in_threadgroup.x;
uint klane = lane % KLANES;
uint rows_per_simd = 32u / KLANES;
uint row = (threadgroup_position_in_grid.y * threads_per_threadgroup.y + thread_position_in_threadgroup.y) * rows_per_simd + lane / KLANES;
uint col = threadgroup_position_in_grid.z * MCOLS;
if (row >= R) return;
float acc[MCOLS];
for(uint m=0;m<MCOLS;++m)acc[m]=0.0f;
for (uint b = klane; b < K/32; b += KLANES) {
 uint a0,a1,sg;
 if(INTERLEAVED) {
  uint4 record=((const device uint4*)planes)[row*(K/32)+b];
  a0=record.x;a1=record.y;sg=record.z;
 } else {
  a0=planes[(row*3)*(K/32)+b];
  a1=planes[(row*3+1)*(K/32)+b];
  sg=planes[(row*3+2)*(K/32)+b];
 }
 float s = float(scales[row*(K/128)+b/4]);
 if(STREAMING) {
  float part[MCOLS];
  for(uint m=0;m<MCOLS;++m)part[m]=0.0f;
  #pragma clang loop unroll(disable)
  for(uint j=0;j<8;++j) {
   uint side=j/2,shift=(j%2)*16;
   uint ranks=(side<2?a0:a1)>>(shift+2*(side%2));
   uint signs=sg>>(shift+side);
   float4 weight=float4(tq3_level(ranks&3u,signs&1u),
     tq3_level((ranks>>4)&3u,(signs>>4)&1u),
     tq3_level((ranks>>8)&3u,(signs>>8)&1u),
     tq3_level((ranks>>12)&3u,(signs>>12)&1u))*s;
   for(uint m=0;m<MCOLS;++m) {
    if(col+m>=M)break;
    const device float4* input=(const device float4*)(x+(col+m)*K+b*32);
    part[m]+=dot(weight,input[j]);
   }
  }
  for(uint m=0;m<MCOLS;++m)acc[m]+=part[m];
 } else {
 float4 weights[8];
 for (uint side=0;side<4;++side) {
  for (uint group=0;group<2;++group) {
   uint shift=group*16;
   uint ranks=(side<2?a0:a1)>>(shift+2*(side%2));
   uint signs=sg>>(shift+side);
   weights[side*2+group]=float4(
    tq3_level(ranks&3u,signs&1u),
    tq3_level((ranks>>4)&3u,(signs>>4)&1u),
    tq3_level((ranks>>8)&3u,(signs>>8)&1u),
    tq3_level((ranks>>12)&3u,(signs>>12)&1u))*s;
  }
 }
 for (uint m=0;m<MCOLS;++m) {
  if (col+m>=M) break;
  const device float4* input=(const device float4*)(x+(col+m)*K+b*32);
  float partial=0.0f;
  for (uint j=0;j<8;++j) partial+=dot(weights[j],input[j]);
  acc[m]+=partial;
 }
 }
}
for (uint m = 0; m < MCOLS; ++m) {
 if (col+m >= M) break;
 float value = acc[m];
 if (KLANES == 32) value = simd_sum(value);
 else {
  value += simd_shuffle_down(value, 4);
  value += simd_shuffle_down(value, 2);
  value += simd_shuffle_down(value, 1);
 }
 if (klane == 0) y[(col+m)*R+row] = value;
}
)METAL", "inline float tq3_level(uint rank,uint sign) { int value=int((0x7f4f2d0eu>>(rank*8u))&255u); return float(sign?-value:value); }");
    const int cols=matmul_columns(m);
    const int simds=matmul_simds();
    int klanes=32;
    if (const char* option=std::getenv("TQLLM_MLX_TQ3_SUBGROUP")) {
        if (std::string_view(option)!="8")
            throw std::invalid_argument("TQLLM_MLX_TQ3_SUBGROUP must be 8");
        klanes=8;
    }
    const int rows_per_group=simds*32/klanes;
    return kernel({w.planes, w.scales, x}, {{m, rows}}, {mx::float32},
                  {32, (rows + rows_per_group - 1) / rows_per_group * simds, (m + cols - 1) / cols}, {32, simds, 1},
                  {{"R", rows}, {"K", k}, {"M", m}, {"MCOLS", cols}, {"KLANES", klanes}, {"INTERLEAVED",w.interleaved},
                   {"STREAMING",std::getenv("TQLLM_MLX_TQ3_STREAM")!=nullptr}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}
