#include "q6.hpp"
#include <stdexcept>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
Q6AffineSplit q6_affine_split(const Q6Weights& w) {
    if (w.blocks.ndim() != 3 || w.blocks.shape(2) != 210 || w.blocks.dtype() != mx::uint8 ||
        w.blocks.shape(0) <= 0 || w.blocks.shape(1) <= 0)
        throw std::invalid_argument("invalid Q6 affine expansion");
    const int r = w.blocks.shape(0), k = w.blocks.shape(1) * 256;
    static auto expand = mx::fast::metal_kernel("tqllm_q6_affine_split", {"blocks"}, {"codes", "scales", "biases"}, R"METAL(
uint i=thread_position_in_grid.x;
if(i>=R*K)return;
uint row=i/K,col=i%K,g=col/32u,j=col%32u;
uint src=g*32u+uint(PART)*16u+j%16u;
const device uchar* p=blocks+(row*(K/256u)+src/256u)*210u;
uint within=src%256u,h=within/128u,l=within%32u,c=(within%128u)/32u;
int q=0;
if(j<16u) {
 uint lo=p[h*64u+(c&1u)*32u+l],hi=p[128u+h*32u+l];
 q=int(((c<2u?lo&15u:lo>>4u)|(((hi>>(2u*c))&3u)<<4u)))-32;
}
codes[i]=uchar(q+128);
if(j==0u) {
 float d=float(as_type<half>(ushort(uint(p[208])|(uint(p[209])<<8u))));
 float s=d*float(as_type<char>(p[192u+h*8u+c*2u+l/16u]));
 scales[row*(K/32u)+g]=s;biases[row*(K/32u)+g]=-128.0f*s;
}
)METAL");
    Q6AffineSplit result;
    for (int part = 0; part < 2; ++part) {
        auto a = expand({w.blocks}, {{r, k}, {r, k / 32}, {r, k / 32}},
            {mx::uint8, mx::float32, mx::float32}, {r * k, 1, 1}, {256, 1, 1},
            {{"R", r}, {"K", k}, {"PART", part}}, std::nullopt, false, mx::Device(mx::Device::gpu));
        result.codes.push_back(mx::reshape(mx::view(a[0], mx::uint32), {r, k / 4}));
        result.scales.push_back(a[1]); result.biases.push_back(a[2]);
        mx::eval(result.codes.back(), result.scales.back(), result.biases.back());
    }
    return result;
}

mx::array q6_affine_matmul(const Q6AffineSplit& w, const mx::array& x) {
    if (w.codes.size() != 2 || w.scales.size() != 2 || w.biases.size() != 2 ||
        x.ndim() != 2 || x.dtype() != mx::float32 || x.shape(0) <= 0 ||
        x.shape(1) <= 0 || x.shape(1) % 256 || w.codes[0].ndim() != 2)
        throw std::invalid_argument("invalid Q6 affine projection");
    const int m = x.shape(0), k = x.shape(1), r = w.codes[0].shape(0);
    auto pairs = mx::reshape(x, {m, k / 32, 2, 16});
    std::vector<mx::array> products;
    for (int part = 0; part < 2; ++part) {
        if (w.codes[part].shape() != mx::Shape{r, k / 4} || w.codes[part].dtype() != mx::uint32 ||
            w.scales[part].shape() != mx::Shape{r, k / 32} || w.scales[part].dtype() != mx::float32 ||
            w.biases[part].shape() != mx::Shape{r, k / 32} || w.biases[part].dtype() != mx::float32)
            throw std::invalid_argument("Q6 affine weight shape mismatch");
        auto columns = mx::reshape(mx::slice(pairs, {0, 0, part, 0}, {m, k / 32, part + 1, 16}), {m, k / 32, 16});
        auto padded = mx::reshape(mx::concatenate({columns, mx::zeros({m, k / 32, 16}, mx::float32)}, 2), {m, k});
        products.push_back(mx::quantized_matmul(padded, w.codes[part], w.scales[part], w.biases[part], true, 32, 8, "affine"));
    }
    return mx::add(products[0], products[1]);
}

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
