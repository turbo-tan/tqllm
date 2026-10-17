#pragma once
#include <mlx/mlx.h>
#include <limits>
#include <cstdint>
#include <stdexcept>

namespace tqllm::mlx_backend {
inline mlx::core::array dense_row_views(const mlx::core::array& input) {
    namespace mx=mlx::core;
    if(input.ndim()!=2 || input.dtype()!=mx::float32 || input.shape(0)<1 || input.shape(0)>8 ||
       input.shape(1)<1 || input.shape(1)>std::numeric_limits<int>::max()-32)
        throw std::invalid_argument("invalid singleton dense batch input");
    const int m=input.shape(0),k=input.shape(1);
    auto padded=mx::concatenate({input,mx::zeros({m,32},mx::float32)},1);
    // Keep each matrix row contiguous, but prevent MLX from folding batches into M.
    return mx::as_strided(padded,{m,1,k},{int64_t(k+32),int64_t(k),1},0);
}
inline mlx::core::array dense_decode_rows(const mlx::core::array& weights,
        const mlx::core::array& views) {
    namespace mx=mlx::core;
    if(weights.ndim()!=2 || views.ndim()!=3 || views.shape(1)!=1 ||
       views.shape(2)!=weights.shape(1) || views.dtype()!=mx::float32)
        throw std::invalid_argument("incompatible singleton dense batch weights");
    auto matrix=mx::expand_dims(mx::transpose(mx::astype(weights,mx::float32)),0);
    return mx::reshape(mx::matmul(views,matrix),
        {views.shape(0),weights.shape(0)});
}
}
