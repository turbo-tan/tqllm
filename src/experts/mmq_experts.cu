// Prefill expert GEMMs on llama.cpp's MMQ kernel (vendored in third_party/ggml_mmq, MIT, see NOTICE there), reading
// tqLLM's expert banks in place: the tile loaders take our bank layout (codes + records, tiled or not), the columns are
// the layer-major pass's slot groups (counts / groups / cap), the activations are quantised once per input row (token or
// assignment) and gathered per tile, and the output is bf16 in the pass's [assignment][rows] buffers.
#define TQ_MMQ 1
#include "quant_kernels.cuh"
#include "ggml_mmq/common.cuh"
#include "ggml_mmq/mmq.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// ggml's GGML_ASSERT / GGML_ABORT end here (tqLLM does not link ggml); the bench links libggml and brings its own
#ifndef TQ_MMQ_LINKS_GGML
#include <cstdarg>
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    std::fprintf(stderr, "%s:%d: ", file, line);
    va_list ap; va_start(ap, fmt); std::vfprintf(stderr, fmt, ap); va_end(ap);
    std::fprintf(stderr, "\n"); std::abort();
}
#endif

namespace tqllm_cuda {

// half rows [rows][k] -> block_q8_1_mmq [k / 128][rows], one fp32 scale per 32 (D4: the layout of every i-quant).
// Same arithmetic as ggml's quantize_mmq_q8_1 on the fp32 values of the halves.
// Columns k .. kpad - 1 (kpad: k rounded up to MMQ's 256-wide K step) are written as zeros.
// rows_of != null: blockIdx.z is a slot, blockIdx.x its column (< counts[z]), the row = rows_of[z cap + column]
__device__ __forceinline__ void mmq_quant_row(const __half * __restrict__ x, block_q8_1_mmq * __restrict__ y, int rows, int k,
                                              int row, int i0) {
    const uint2 raw = i0 < k ? *reinterpret_cast<const uint2 *>(x + int64_t(row) * k + i0) : make_uint2(0, 0);
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&raw.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&raw.y));
    float amax = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o, 32));
    const float d_inv = amax > 0.0f || i0 < k ? 127.0f / amax : 0.0f;
    char4 q;
    q.x = roundf(a.x * d_inv); q.y = roundf(a.y * d_inv); q.z = roundf(b.x * d_inv); q.w = roundf(b.y * d_inv);
    block_q8_1_mmq & blk = y[int64_t(i0 / QK8_1_MMQ) * rows + row];
    const int iqs = i0 % QK8_1_MMQ;
    reinterpret_cast<char4 *>(blk.qs)[iqs / 4] = q;
    if (iqs % 32 == 0) blk.d4[iqs / 32] = i0 < k ? 1.0f / d_inv : 0.0f;
}
__global__ void mmq_quant_d4_k(const __half * __restrict__ x, block_q8_1_mmq * __restrict__ y, int rows, int k, int kpad) {
    const int row = blockIdx.x;
    const int i0 = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (i0 >= kpad) return;
    mmq_quant_row(x, y, rows, k, row, i0);
}
// the rows of a grouped pass: blockIdx.x strides over the slot's rows (a cap x slots grid would launch mostly empty blocks)
__global__ void mmq_quant_rows_k(const __half * __restrict__ x, block_q8_1_mmq * __restrict__ y, int rows, int k, int kpad,
                                      const int * __restrict__ rows_of, const int * __restrict__ counts, int cap) {
    const int i0 = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (i0 >= kpad) return;
    const int n = counts[blockIdx.z];
    for (int c = blockIdx.x; c < n; c += gridDim.x) mmq_quant_row(x, y, rows, k, rows_of[int64_t(blockIdx.z) * cap + c], i0);
}

static int mmq_kpad(int k) { return (k + MMQ_ITER_K - 1) / MMQ_ITER_K * MMQ_ITER_K; }
size_t mmq_act_bytes(int rows, int k) { return size_t(mmq_kpad(k) / QK8_1_MMQ) * size_t(rows) * sizeof(block_q8_1_mmq); }

void mmq_quantize_act(const __half * x, int rows, int k, void * yq, cudaStream_t s) {
    if (rows <= 0) return;
    if (k % QK8_1_MMQ) throw std::invalid_argument("mmq_quantize_act: k must be a multiple of 128");
    const int kpad = mmq_kpad(k);
    mmq_quant_d4_k<<<dim3(unsigned(rows), unsigned((kpad / 4 + 127) / 128)), 128, 0, s>>>(x, static_cast<block_q8_1_mmq *>(yq), rows, k, kpad);
}

// the rows of a layer-major pass only (rows_of / counts / cap / slots: the pass's groups); rows = the buffer's row count
void mmq_quantize_act_rows(const __half * x, int rows, int k, const int * rows_of, const int * counts, int cap, int slots, void * yq, cudaStream_t s) {
    if (cap <= 0 || slots <= 0) return;
    if (k % QK8_1_MMQ) throw std::invalid_argument("mmq_quantize_act_rows: k must be a multiple of 128");
    const int kpad = mmq_kpad(k);
    mmq_quant_rows_k<<<dim3(unsigned(std::min(cap, 32)), unsigned((kpad / 4 + 127) / 128), unsigned(slots)), 128, 0, s>>>(
        x, static_cast<block_q8_1_mmq *>(yq), rows, k, kpad, rows_of, counts, cap);
}

namespace {
struct DevInfo { int cc = 0, nsm = 0; size_t smpbo = 0; };
const DevInfo & dev_info() {
    static thread_local DevInfo d[16];
    int id = 0; cudaGetDevice(&id);
    if (!d[id].cc) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, id);
        d[id].cc = 100 * p.major + 10 * p.minor; d[id].nsm = p.multiProcessorCount; d[id].smpbo = p.sharedMemPerBlockOptin;
    }
    return d[id];
}

template <ggml_type type, int J, bool fallback>
void launch(const mmq_args & a, cudaStream_t s) {
    const DevInfo & di = dev_info();
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, di.cc);
    const int nbytes_shared = int(mmq_get_nbytes_shared(config, di.cc));
    static bool raised = false;
    if (!raised) { cudaFuncSetAttribute(mul_mat_q<type, J, fallback>, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared); raised = true; }
    const dim3 block_dims(32, config.nthreads / 32, 1);
    const int nty = int((a.nrows_x + config.I - 1) / config.I);
    const uint3 bpr = init_fastdiv_values(uint32_t(a.ncols_x / ggml_cuda_type_traits<type>::qk));
    const uint3 one = init_fastdiv_values(1), ntx_fd = one, nch = init_fastdiv_values(uint32_t(a.nchannels_y));
    // one block per (row tile, listed column tile): only tiles that hold columns are launched, no stream-k fixup
    const dim3 grid(unsigned(nty), unsigned(a.ncols_max), 1);
    mul_mat_q<type, J, fallback><<<grid, block_dims, nbytes_shared, s>>>(
        a.x, a.y, a.ids_dst, a.expert_bounds, a.dst, nullptr, nullptr, bpr, int(a.nrows_x), int(a.ncols_dst), int(a.stride_row_x),
        int(a.ncols_y), int(a.nrows_dst), one, nch, int(a.stride_channel_x), int(a.stride_channel_y), int(a.stride_channel_dst),
        one, one, 0, 0, 0, ntx_fd);
}

struct Plan { const int * counts_host; int slots; int2 * tiles_dev; int tiles_cap; };

template <ggml_type type, bool fallback>
void switch_j(mmq_args a, const Plan & pl, cudaStream_t s) {
    const DevInfo & di = dev_info();
    int jb = 0; int64_t best = INT64_MAX;
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, di.cc);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, di.cc) > di.smpbo) continue;
        int64_t cost = 0;  // padded columns, plus a tile's fixed cost (x tile loads, launch) as 32 columns
        for (int z = 0; z < pl.slots; ++z) cost += int64_t((pl.counts_host[z] + J - 1) / J) * (J + 32);
        if (cost < best) { jb = J; best = cost; }
    }
    static const int force_j = std::getenv("TQ_MMQ_J") ? std::atoi(std::getenv("TQ_MMQ_J")) : 0;
    if (force_j) jb = force_j;
    if (std::getenv("TQ_MMQ_VERBOSE")) std::fprintf(stderr, "mmq: J %d\n", jb);
    static thread_local std::vector<int2> tl;
    tl.clear();
    for (int z = 0; z < pl.slots; ++z) for (int j = 0; j * jb < pl.counts_host[z]; ++j) tl.push_back(make_int2(z, j));
    if (tl.empty()) return;
    if (int(tl.size()) > pl.tiles_cap) throw std::runtime_error("mmq: tile list larger than its buffer");
    cudaMemcpyAsync(pl.tiles_dev, tl.data(), tl.size() * sizeof(int2), cudaMemcpyHostToDevice, s);
    a.ncols_max = int64_t(tl.size());
    switch (jb) {
#define TQ_J(j) case j: launch<type, j, fallback>(a, s); break;
        TQ_J(8) TQ_J(16) TQ_J(24) TQ_J(32) TQ_J(40) TQ_J(48) TQ_J(56) TQ_J(64)
        TQ_J(72) TQ_J(80) TQ_J(88) TQ_J(96) TQ_J(104) TQ_J(112) TQ_J(120) TQ_J(128)
#undef TQ_J
        default: throw std::runtime_error("mmq: no tile size");
    }
}

template <ggml_type type>
void run(const mmq_args & a, const Plan & pl, cudaStream_t s) {
    if (a.nrows_x % 128 == 0) switch_j<type, false>(a, pl, s);
    else switch_j<type, true>(a, pl, s);
}
}  // namespace

// Loads every MMQ kernel this file can launch (code + registers) now: lazy loading would otherwise do it at the first
// prefill, after the expert cache took the free memory. Returns the device bytes it cost.
namespace {
template <ggml_type type, int J> void warm_one() {
    if constexpr (J <= 128) {
        cudaFuncAttributes a; cudaFuncGetAttributes(&a, mul_mat_q<type, J, false>);
        const DevInfo & di = dev_info();
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, di.cc);
        if (config.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(config, di.cc) <= di.smpbo)
            cudaFuncSetAttribute(mul_mat_q<type, J, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(mmq_get_nbytes_shared(config, di.cc)));
        warm_one<type, J + 8>();
    }
}
}  // namespace
size_t mmq_warm() {
    size_t f0 = 0, f1 = 0, tot = 0; cudaMemGetInfo(&f0, &tot);
    warm_one<GGML_TYPE_IQ3_S, 8>(); warm_one<GGML_TYPE_IQ3_XXS, 8>(); warm_one<GGML_TYPE_IQ2_S, 8>();
    warm_one<GGML_TYPE_IQ4_XS, 8>(); warm_one<GGML_TYPE_IQ4_NL, 8>();
    mmq_quant_d4_k<<<1, 128>>>(nullptr, nullptr, 0, 0, 0);  // k = 0: returns at once, loads the quantizer
    cudaDeviceSynchronize(); cudaGetLastError(); cudaMemGetInfo(&f1, &tot);
    return f0 > f1 ? f0 - f1 : 0;
}

// fmt: u4h_matvec_k's FMT (4 IQ3_S, 5 IQ3_XXS, 10 IQ2_S, 13 IQ4_XS, 14 IQ4_NL). rows / k per slot, slot z's rows at z * rows of the bank. yq: mmq_quantize_act
// of ny input rows; column j of slot z reads row groups[z cap + j] / div and writes y[groups[z cap + j] * rows + r].
// counts_host: the slots' column counts (the launch lists only the tiles that hold columns); tiles: a device buffer of
// at least mmq_max_tiles(cap, slots) int2.
int mmq_max_tiles(int cap, int slots) { return slots * ((cap + 7) / 8); }
bool mmq_supports(int fmt, int k) {
    const int qk = fmt == 14 ? 32 : 256;
    return (fmt == 4 || fmt == 5 || fmt == 10 || fmt == 13 || fmt == 14) && k % qk == 0 && k % QK8_1_MMQ == 0;
}
bool mmq_expert_gemm(int fmt, const uint8_t * codes, const void * rec, bool tiled, int rows, int k, const int * counts,
                     const int * counts_host, const int * groups, int cap, int slots, int div, const void * yq, int ny,
                     __nv_bfloat16 * y, int2 * tiles, int tiles_cap, cudaStream_t s, const __nv_bfloat16 * gate, __half * act) {
    ggml_type type; int qk;
    switch (fmt) {
        case 4: type = GGML_TYPE_IQ3_S; qk = ggml_cuda_type_traits<GGML_TYPE_IQ3_S>::qk; break;
        case 5: type = GGML_TYPE_IQ3_XXS; qk = ggml_cuda_type_traits<GGML_TYPE_IQ3_XXS>::qk; break;
        case 10: type = GGML_TYPE_IQ2_S; qk = ggml_cuda_type_traits<GGML_TYPE_IQ2_S>::qk; break;
        case 13: type = GGML_TYPE_IQ4_XS; qk = ggml_cuda_type_traits<GGML_TYPE_IQ4_XS>::qk; break;
        case 14: type = GGML_TYPE_IQ4_NL; qk = ggml_cuda_type_traits<GGML_TYPE_IQ4_NL>::qk; break;
        default: return false;
    }
    if (k % qk || k % QK8_1_MMQ || cap <= 0 || slots <= 0) return false;
    const tq_u4h_desc du = {static_cast<const uint8_t *>(rec), k / qk, k / 32, tiled ? 1 : 0};
    const tq_grp_desc dg = {counts, groups, cap, div, tiles, gate, act, y};
    cudaMemcpyToSymbolAsync(tq_u4h, &du, sizeof du, 0, cudaMemcpyHostToDevice, s);
    cudaMemcpyToSymbolAsync(tq_grp, &dg, sizeof dg, 0, cudaMemcpyHostToDevice, s);
    const mmq_args a = {reinterpret_cast<const char *>(codes), type, static_cast<const int *>(yq), groups, counts,
                        reinterpret_cast<float *>(y), nullptr,
                        k, rows, cap, k / qk, ny, rows,
                        slots, slots, int64_t(rows) * (k / qk), 0, 0,
                        1, 1, 0, 0, 0,
                        cap, cap};
    switch (type) {
        case GGML_TYPE_IQ3_S: run<GGML_TYPE_IQ3_S>(a, Plan{counts_host, slots, tiles, tiles_cap}, s); break;
        case GGML_TYPE_IQ3_XXS: run<GGML_TYPE_IQ3_XXS>(a, Plan{counts_host, slots, tiles, tiles_cap}, s); break;
        case GGML_TYPE_IQ2_S: run<GGML_TYPE_IQ2_S>(a, Plan{counts_host, slots, tiles, tiles_cap}, s); break;
        case GGML_TYPE_IQ4_XS: run<GGML_TYPE_IQ4_XS>(a, Plan{counts_host, slots, tiles, tiles_cap}, s); break;
        case GGML_TYPE_IQ4_NL: run<GGML_TYPE_IQ4_NL>(a, Plan{counts_host, slots, tiles, tiles_cap}, s); break;
        case GGML_TYPE_Q8_0: run<GGML_TYPE_Q8_0>(a, Plan{counts_host, slots, tiles, tiles_cap}, s); break;
        default: return false;
    }
    return true;
}

}  // namespace tqllm_cuda
