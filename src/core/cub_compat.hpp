#pragma once
// CUB (NVIDIA) and hipCUB (AMD) behind one name, so a model file does not carry the backend choice (AGENTS.md rule 2).
// hipCUB mirrors the CUB API; the call sites keep the CUDA spelling.
#if defined(TQLLM_WITH_HIP)
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#endif