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
#ifdef __cplusplus
}
#endif
#endif
