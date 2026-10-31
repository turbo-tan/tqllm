#ifndef TQLLM_PLUGIN_H
#define TQLLM_PLUGIN_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental ABI 1: host F32 vector-add proof only; not a quantization ABI. */
#define TQ_ABI_VERSION 1u
typedef struct tq_error { uint32_t struct_size; int32_t code; char message[256]; } tq_error;
typedef struct tq_call {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t elements;
    const float * a;
    const float * b;
    float * out;
} tq_call;
typedef int32_t (*tq_invoke_fn)(const tq_call *, tq_error *);
typedef struct tq_kernel {
    uint32_t struct_size;
    uint32_t abi_version;
    const char * id;
    uint64_t max_elements;
    tq_invoke_fn invoke;
} tq_kernel;
typedef struct tq_registry {
    uint32_t struct_size;
    uint32_t abi_version;
    void * context;
    int32_t (*add)(void *, const tq_kernel *, tq_error *);
} tq_registry;
typedef int32_t (*tq_plugin_init_fn)(const tq_registry *, tq_error *);
/* IDs and callbacks must survive until the runtime closes the shared library. */
int32_t tq_plugin_init_v1(const tq_registry *, tq_error *);

/* ABI 2: weight formats and kernel patches for the CUDA backend (docs/guides/plugins.md).
   A plugin exports tq_plugin_init_v2 and registers formats through the host table. Device pointers and streams are
   void * (no CUDA headers here): data in device memory, stream = cudaStream_t. Every callback returns 0 on success.
   - New format: gguf_type in [TQ_PLUGIN_TYPE_MIN, TQ_PLUGIN_TYPE_MAX]; the host uploads the tensor's GGUF blocks
     unchanged, so `blocks` below is the GGUF block stream of a [rows][cols] matrix (row-major).
   - Patch: gguf_type is a patchable run-time format (TQ_PATCH_I8H); `blocks` / `aux` are that format's device layout.
     Callbacks left null keep the built-in kernel. A patch with exact = 1 must give byte-identical results to the
     built-in; the host checks that at load and refuses the plugin otherwise.
   Every format passes the format contract at load (GPU dequant vs cpu_dequant, matvec vs a double dot). */
#define TQ_ABI_VERSION_2 2u
#define TQ_PLUGIN_TYPE_MIN 2000u
#define TQ_PLUGIN_TYPE_MAX 2999u
/* I8H: int8 weights [rows][cols] row-major in blocks; one FP16 scale per 16 consecutive values in aux */
#define TQ_PATCH_I8H 1000u
typedef struct tq_format_v2 {
    uint32_t struct_size;
    uint32_t abi_version;          /* TQ_ABI_VERSION_2 */
    const char * name;
    uint32_t gguf_type;
    uint32_t block_values;         /* new formats: values per GGUF block (cols % block_values == 0) */
    uint32_t block_bytes;          /* new formats: bytes per GGUF block */
    uint32_t sm_min, sm_max;       /* compute capability x 10 the kernels run on, e.g. 80..120; 0, 0: any */
    uint32_t exact;                /* patches: 1 = byte-identical to the built-in (checked at load) */
    /* host reference: n values from GGUF blocks (new formats; patches: null, the built-in reference is used) */
    int32_t (*cpu_dequant)(const void * blocks, float * out, uint64_t n);
    /* device: the [rows][cols] matrix as FP16 (out: rows * cols halves) */
    int32_t (*gpu_dequant_f16)(const void * blocks, const void * aux, int32_t rows, int32_t cols, void * out, void * stream);
    /* device: y[j][r] = sum_c W[r][c] * x[j][c] for j < ncols; x FP16 [ncols][cols], y BF16 [ncols][rows] */
    int32_t (*gpu_matvec)(const void * blocks, const void * aux, int32_t rows, int32_t cols, const void * x, void * y,
                          int32_t ncols, void * stream);
} tq_format_v2;
typedef struct tq_host_v2 {
    uint32_t struct_size;
    uint32_t abi_version;          /* TQ_ABI_VERSION_2 */
    void * context;
    int32_t (*register_format)(void * context, const tq_format_v2 *, tq_error *);
} tq_host_v2;
typedef int32_t (*tq_plugin_init_v2_fn)(const tq_host_v2 *, tq_error *);
/* Descriptors, names and callbacks must survive until the runtime closes the shared library. */
int32_t tq_plugin_init_v2(const tq_host_v2 *, tq_error *);
#ifdef __cplusplus
}
#endif
#endif
