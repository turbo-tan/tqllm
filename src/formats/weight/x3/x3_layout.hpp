#pragma once
// x3_layout.hpp: the ONE definition of the X3 trellis tile order, shared by reconstruct, gemv_x3v,
// gemv_sq and the host reference. Canonical order (llama-paw tests/test-paw-x3-frac.cpp decode_what):
// tile (kb, nb) lives at (kb*(m/16) + nb)*tw, i.e. k-tile major, n/out-tile minor. No kernel does
// its own tile index math; they all call x3_tile_offset.
#include <cstdint>

namespace tqllm {
namespace x3 {

// Linear tile index of tile (k_tile, n_tile) in a grid of tiles_n n-tiles (k-major, n-minor).
__host__ __device__ constexpr int x3_tile_offset(int k_tile, int n_tile, int tiles_n) {
    return k_tile * tiles_n + n_tile;
}

}  // namespace x3
}  // namespace tqllm
